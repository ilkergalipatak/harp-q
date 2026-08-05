"""Deterministic offline allocation for progressive packet prefixes.

Each logical tile is a ``group_id``.  Incremental packets in that group declare
their predecessors.  The allocator enumerates the valid dependency-closed prefix
choices for each (small) group and then solves the resulting multiple-choice
knapsack exactly with a sparse Pareto dynamic program.

The exponential part is confined to one tile's packet ladder (four packets for
HARP-Q's 2+2+2+2 representation), not to the whole model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


class AllocationError(ValueError):
    """Base class for an invalid or infeasible allocation problem."""


class AllocationDependencyError(AllocationError):
    """Raised when incremental packet dependencies are invalid."""


class InfeasibleBudgetError(AllocationError):
    """Raised when mandatory packet prefixes do not fit the budget."""


@dataclass(frozen=True, slots=True)
class AllocationItem:
    """One incremental packet considered by the offline allocator."""

    packet_id: str
    group_id: str
    cost_bytes: int
    distortion_gain: float
    dependencies: tuple[str, ...] = ()
    mandatory: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("packet_id must be a non-empty string")
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        if isinstance(self.cost_bytes, bool) or not isinstance(self.cost_bytes, int):
            raise TypeError("cost_bytes must be an integer")
        if self.cost_bytes < 0:
            raise ValueError("cost_bytes must be non-negative")
        if not math.isfinite(self.distortion_gain):
            raise ValueError("distortion_gain must be finite")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("dependencies must contain non-empty packet IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies cannot contain duplicates")
        if self.packet_id in dependencies:
            raise AllocationDependencyError(f"packet {self.packet_id!r} cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)


@dataclass(frozen=True, slots=True)
class AllocationOption:
    """One dependency-closed precision choice for a group."""

    group_id: str
    packet_ids: tuple[str, ...]
    cost_bytes: int
    distortion_gain: float


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """The optimal deterministic packet set for a byte budget."""

    selected_packet_ids: tuple[str, ...]
    selected_by_group: Mapping[str, tuple[str, ...]]
    total_cost_bytes: int
    total_distortion_gain: float
    budget_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "selected_by_group": {
                group_id: list(packet_ids)
                for group_id, packet_ids in self.selected_by_group.items()
            },
            "selected_packet_ids": list(self.selected_packet_ids),
            "total_cost_bytes": self.total_cost_bytes,
            "total_distortion_gain": self.total_distortion_gain,
        }


@dataclass(frozen=True, slots=True)
class _State:
    cost_bytes: int
    distortion_gain: float
    choices: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def packet_ids(self) -> tuple[str, ...]:
        return tuple(sorted(packet_id for _, packet_ids in self.choices for packet_id in packet_ids))


def _validate_items(items: Sequence[AllocationItem]) -> dict[str, tuple[AllocationItem, ...]]:
    by_id: dict[str, AllocationItem] = {}
    by_group: dict[str, list[AllocationItem]] = {}
    for item in items:
        if item.packet_id in by_id:
            raise AllocationDependencyError(f"duplicate packet ID {item.packet_id!r}")
        by_id[item.packet_id] = item
        by_group.setdefault(item.group_id, []).append(item)
    for item in items:
        for dependency_id in item.dependencies:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                raise AllocationDependencyError(
                    f"packet {item.packet_id!r} depends on missing packet {dependency_id!r}"
                )
            if dependency.group_id != item.group_id:
                raise AllocationDependencyError(
                    "cross-group dependencies are not valid precision-prefix choices: "
                    f"{item.packet_id!r} -> {dependency_id!r}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(packet_id: str) -> None:
        if packet_id in visiting:
            raise AllocationDependencyError(f"dependency cycle includes packet {packet_id!r}")
        if packet_id in visited:
            return
        visiting.add(packet_id)
        for dependency_id in by_id[packet_id].dependencies:
            visit(dependency_id)
        visiting.remove(packet_id)
        visited.add(packet_id)

    for packet_id in sorted(by_id):
        visit(packet_id)
    return {
        group_id: tuple(sorted(group_items, key=lambda item: item.packet_id))
        for group_id, group_items in by_group.items()
    }


def _topological_group(items: Sequence[AllocationItem]) -> tuple[AllocationItem, ...]:
    by_id = {item.packet_id: item for item in items}
    ordered: list[AllocationItem] = []
    visited: set[str] = set()

    def visit(item: AllocationItem) -> None:
        if item.packet_id in visited:
            return
        for dependency_id in sorted(item.dependencies):
            visit(by_id[dependency_id])
        visited.add(item.packet_id)
        ordered.append(item)

    for item in sorted(items, key=lambda candidate: candidate.packet_id):
        visit(item)
    return tuple(ordered)


def enumerate_group_options(
    items: Iterable[AllocationItem],
    *,
    max_group_items: int = 20,
) -> tuple[AllocationOption, ...]:
    """Enumerate non-dominated dependency-closed choices for one group."""

    item_list = tuple(items)
    if not item_list:
        return ()
    grouped = _validate_items(item_list)
    if len(grouped) != 1:
        raise AllocationError("enumerate_group_options accepts exactly one group")
    group_id, group_items = next(iter(grouped.items()))
    if max_group_items <= 0:
        raise ValueError("max_group_items must be positive")
    if len(group_items) > max_group_items:
        raise AllocationError(
            f"group {group_id!r} has {len(group_items)} items; reference limit is "
            f"{max_group_items}"
        )

    ordered = _topological_group(group_items)
    options: list[AllocationOption] = []

    def enumerate_at(
        index: int,
        selected: frozenset[str],
        cost_bytes: int,
        distortion_gain: float,
    ) -> None:
        if index == len(ordered):
            options.append(
                AllocationOption(
                    group_id=group_id,
                    packet_ids=tuple(sorted(selected)),
                    cost_bytes=cost_bytes,
                    distortion_gain=distortion_gain,
                )
            )
            return
        item = ordered[index]
        if not item.mandatory:
            enumerate_at(index + 1, selected, cost_bytes, distortion_gain)
        if all(dependency in selected for dependency in item.dependencies):
            enumerate_at(
                index + 1,
                selected | {item.packet_id},
                cost_bytes + item.cost_bytes,
                distortion_gain + item.distortion_gain,
            )

    enumerate_at(0, frozenset(), 0, 0.0)
    if not options:
        raise AllocationDependencyError(f"mandatory choices in group {group_id!r} are infeasible")
    return _prune_options(options)


def _prune_options(options: Iterable[AllocationOption]) -> tuple[AllocationOption, ...]:
    # Ascending cost means the running best gain is enough to remove every
    # cost/gain dominated precision prefix.  Lexicographic IDs settle exact ties.
    ordered = sorted(
        options,
        key=lambda option: (option.cost_bytes, -option.distortion_gain, option.packet_ids),
    )
    result: list[AllocationOption] = []
    best_gain = -math.inf
    for option in ordered:
        if option.distortion_gain > best_gain:
            result.append(option)
            best_gain = option.distortion_gain
    return tuple(result)


def _state_is_better(candidate: _State, incumbent: _State) -> bool:
    if candidate.distortion_gain != incumbent.distortion_gain:
        return candidate.distortion_gain > incumbent.distortion_gain
    return candidate.packet_ids < incumbent.packet_ids


def _prune_states(states: Mapping[int, _State]) -> dict[int, _State]:
    result: dict[int, _State] = {}
    best_gain = -math.inf
    for cost_bytes in sorted(states):
        state = states[cost_bytes]
        if state.distortion_gain > best_gain:
            result[cost_bytes] = state
            best_gain = state.distortion_gain
    return result


def allocate_packets(
    items: Iterable[AllocationItem],
    budget_bytes: int,
    *,
    max_group_items: int = 20,
) -> AllocationResult:
    """Solve the precedence-aware multiple-choice knapsack exactly.

    Gains are additive.  For each logical group the selected packet set is one
    valid dependency-closed prefix; mandatory packets must be present.  Equal-gain
    results prefer lower cost and then lexicographically smaller packet IDs.
    """

    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise TypeError("budget_bytes must be an integer")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    item_list = tuple(items)
    grouped = _validate_items(item_list)
    group_options = {
        group_id: enumerate_group_options(group_items, max_group_items=max_group_items)
        for group_id, group_items in grouped.items()
    }

    frontier: dict[int, _State] = {0: _State(0, 0.0, ())}
    for group_id in sorted(group_options):
        next_frontier: dict[int, _State] = {}
        for state in frontier.values():
            for option in group_options[group_id]:
                cost_bytes = state.cost_bytes + option.cost_bytes
                if cost_bytes > budget_bytes:
                    continue
                candidate = _State(
                    cost_bytes=cost_bytes,
                    distortion_gain=state.distortion_gain + option.distortion_gain,
                    choices=state.choices + ((group_id, option.packet_ids),),
                )
                incumbent = next_frontier.get(cost_bytes)
                if incumbent is None or _state_is_better(candidate, incumbent):
                    next_frontier[cost_bytes] = candidate
        if not next_frontier:
            raise InfeasibleBudgetError(
                f"mandatory prefixes do not fit budget while allocating group {group_id!r}"
            )
        frontier = _prune_states(next_frontier)

    if not frontier:
        # This is reachable only for an implementation error; the empty problem
        # always retains the zero-cost state.
        raise InfeasibleBudgetError("no feasible allocation")
    best = min(
        frontier.values(),
        key=lambda state: (-state.distortion_gain, state.cost_bytes, state.packet_ids),
    )
    by_group = MappingProxyType(dict(best.choices))
    return AllocationResult(
        selected_packet_ids=best.packet_ids,
        selected_by_group=by_group,
        total_cost_bytes=best.cost_bytes,
        total_distortion_gain=best.distortion_gain,
        budget_bytes=budget_bytes,
    )


# Short, CLI-friendly spelling without hiding the more descriptive public name.
allocate = allocate_packets


__all__ = [
    "AllocationDependencyError",
    "AllocationError",
    "AllocationItem",
    "AllocationOption",
    "AllocationResult",
    "InfeasibleBudgetError",
    "allocate",
    "allocate_packets",
    "enumerate_group_options",
]
