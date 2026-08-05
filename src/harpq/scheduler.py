"""Deterministic cache/deadline scheduler and replayable runtime trace.

The reference scheduler models the refinement cache, not the always-executable
base model.  It admits dependency-closed packet prefixes, rejects residuals that
would miss their deadline, and prevents cache thrash with minimum residency and a
score hysteresis threshold.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


class SchedulerError(ValueError):
    """Base class for scheduler input or state failures."""


class SchedulerDependencyError(SchedulerError):
    """Raised when a request dependency is missing or cyclic."""


class UnschedulableRequestError(SchedulerError):
    """Raised when a required request cannot meet its contract."""


class ReplayMismatchError(SchedulerError):
    """Raised when replay diverges from a recorded decision."""


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Policy and transfer model for one refinement cache."""

    capacity_bytes: int
    bandwidth_bytes_per_us: float = 7000.0
    fixed_read_latency_us: float = 0.0
    min_residency_tokens: int = 8
    hysteresis: float = 0.15

    def __post_init__(self) -> None:
        if isinstance(self.capacity_bytes, bool) or not isinstance(self.capacity_bytes, int):
            raise TypeError("capacity_bytes must be an integer")
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if not math.isfinite(self.bandwidth_bytes_per_us) or self.bandwidth_bytes_per_us <= 0:
            raise ValueError("bandwidth_bytes_per_us must be finite and positive")
        if not math.isfinite(self.fixed_read_latency_us) or self.fixed_read_latency_us < 0:
            raise ValueError("fixed_read_latency_us must be finite and non-negative")
        if (
            isinstance(self.min_residency_tokens, bool)
            or not isinstance(self.min_residency_tokens, int)
            or self.min_residency_tokens < 0
        ):
            raise ValueError("min_residency_tokens must be a non-negative integer")
        if not math.isfinite(self.hysteresis) or self.hysteresis < 0:
            raise ValueError("hysteresis must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerConfig":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PacketRequest:
    """A packet that may improve the current precision plan.

    ``deadline_us`` is relative to the start of this scheduling step.
    ``transfer_us`` can carry a measured cost; when absent, the scheduler derives
    it from packet size and :class:`SchedulerConfig`.
    """

    packet_id: str
    size_bytes: int
    distortion_gain: float
    deadline_us: float
    dependencies: tuple[str, ...] = ()
    reuse: float = 1.0
    importance: float = 1.0
    kernel_cost_us: float = 0.0
    io_bytes: int | None = None
    transfer_us: float | None = None
    required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("packet_id must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.io_bytes is not None and (
            isinstance(self.io_bytes, bool)
            or not isinstance(self.io_bytes, int)
            or self.io_bytes < self.size_bytes
        ):
            raise ValueError("io_bytes must be an integer greater than or equal to size_bytes")
        for name in ("distortion_gain", "deadline_us", "reuse", "importance", "kernel_cost_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.transfer_us is not None and (
            not math.isfinite(self.transfer_us) or self.transfer_us <= 0
        ):
            raise ValueError("transfer_us must be finite and positive when supplied")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("dependencies must contain non-empty packet IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies cannot contain duplicates")
        if self.packet_id in dependencies:
            raise SchedulerDependencyError(f"packet {self.packet_id!r} cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)

    def transfer_duration_us(self, config: SchedulerConfig) -> float:
        if self.transfer_us is not None:
            return self.transfer_us
        transfer_bytes = self.io_bytes if self.io_bytes is not None else self.size_bytes
        return config.fixed_read_latency_us + transfer_bytes / config.bandwidth_bytes_per_us

    def priority_score(self, config: SchedulerConfig) -> float:
        cost = self.transfer_duration_us(config) + self.kernel_cost_us
        if cost == 0:
            return math.inf if self.distortion_gain * self.reuse * self.importance > 0 else 0.0
        return self.distortion_gain * self.reuse * self.importance / cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadline_us": self.deadline_us,
            "dependencies": list(self.dependencies),
            "distortion_gain": self.distortion_gain,
            "importance": self.importance,
            "io_bytes": self.io_bytes,
            "kernel_cost_us": self.kernel_cost_us,
            "packet_id": self.packet_id,
            "required": self.required,
            "reuse": self.reuse,
            "size_bytes": self.size_bytes,
            "transfer_us": self.transfer_us,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PacketRequest":
        data = dict(value)
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Serializable scheduler cache state."""

    packet_id: str
    size_bytes: int
    score: float
    loaded_token: int
    last_access_token: int
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("cache packet_id must be a non-empty string")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("cache size_bytes must be a positive integer")
        if not math.isfinite(self.score) or self.score < 0:
            raise ValueError("cache score must be finite and non-negative")
        if isinstance(self.loaded_token, bool) or not isinstance(self.loaded_token, int):
            raise TypeError("loaded_token must be an integer")
        if isinstance(self.last_access_token, bool) or not isinstance(self.last_access_token, int):
            raise TypeError("last_access_token must be an integer")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("cache dependencies must contain non-empty packet IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("cache dependencies cannot contain duplicates")
        object.__setattr__(self, "dependencies", dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependencies": list(self.dependencies),
            "last_access_token": self.last_access_token,
            "loaded_token": self.loaded_token,
            "packet_id": self.packet_id,
            "score": self.score,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheEntry":
        data = dict(value)
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    """One complete, deterministic scheduling outcome."""

    token_index: int
    selected: tuple[str, ...]
    loaded: tuple[str, ...]
    evicted: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    scores: tuple[tuple[str, float], ...]
    copy_time_us: float
    cache_bytes: int
    cache_packet_ids: tuple[str, ...]
    logical_load_bytes: int = 0
    io_load_bytes: int = 0
    read_amplification: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_bytes": self.cache_bytes,
            "cache_packet_ids": list(self.cache_packet_ids),
            "copy_time_us": self.copy_time_us,
            "evicted": list(self.evicted),
            "loaded": list(self.loaded),
            "logical_load_bytes": self.logical_load_bytes,
            "io_load_bytes": self.io_load_bytes,
            "read_amplification": self.read_amplification,
            "scores": [[packet_id, score] for packet_id, score in self.scores],
            "selected": list(self.selected),
            "skipped": [[packet_id, reason] for packet_id, reason in self.skipped],
            "token_index": self.token_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerDecision":
        return cls(
            token_index=value["token_index"],
            selected=tuple(value["selected"]),
            loaded=tuple(value["loaded"]),
            evicted=tuple(value["evicted"]),
            skipped=tuple((item[0], item[1]) for item in value["skipped"]),
            scores=tuple((item[0], item[1]) for item in value["scores"]),
            copy_time_us=value["copy_time_us"],
            cache_bytes=value["cache_bytes"],
            cache_packet_ids=tuple(value["cache_packet_ids"]),
            logical_load_bytes=value.get("logical_load_bytes", 0),
            io_load_bytes=value.get("io_load_bytes", 0),
            read_amplification=value.get("read_amplification", 0.0),
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    token_index: int
    now_us: float
    requests: tuple[PacketRequest, ...]
    decision: SchedulerDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "now_us": self.now_us,
            "requests": [request.to_dict() for request in self.requests],
            "token_index": self.token_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        return cls(
            token_index=value["token_index"],
            now_us=value["now_us"],
            requests=tuple(PacketRequest.from_dict(item) for item in value["requests"]),
            decision=SchedulerDecision.from_dict(value["decision"]),
        )


@dataclass(frozen=True, slots=True)
class SchedulerTrace:
    """Self-contained inputs and decisions needed for deterministic replay."""

    config: SchedulerConfig
    initial_cache: tuple[CacheEntry, ...]
    events: tuple[TraceEvent, ...]
    format_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "format": "harpq-scheduler-trace",
            "format_version": self.format_version,
            "initial_cache": [entry.to_dict() for entry in self.initial_cache],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent, allow_nan=False, ensure_ascii=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerTrace":
        if value.get("format") != "harpq-scheduler-trace" or value.get("format_version") != 1:
            raise SchedulerError("unsupported scheduler trace format")
        return cls(
            config=SchedulerConfig.from_dict(value["config"]),
            initial_cache=tuple(CacheEntry.from_dict(item) for item in value["initial_cache"]),
            events=tuple(TraceEvent.from_dict(item) for item in value["events"]),
            format_version=1,
        )

    @classmethod
    def from_json(cls, encoded: str | bytes | bytearray) -> "SchedulerTrace":
        try:
            value = json.loads(encoded)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SchedulerError("scheduler trace is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise SchedulerError("scheduler trace root must be an object")
        return cls.from_dict(value)


class CacheDeadlineScheduler:
    """Stateful deterministic scheduler for optional refinement packets."""

    def __init__(
        self,
        config: SchedulerConfig,
        *,
        initial_cache: Iterable[CacheEntry] = (),
        record_trace: bool = True,
    ) -> None:
        self.config = config
        entries = tuple(sorted(initial_cache, key=lambda entry: entry.packet_id))
        self._validate_initial_cache(entries)
        self._initial_cache = entries
        self._cache = {entry.packet_id: entry for entry in entries}
        self._events: list[TraceEvent] = []
        self._record_trace = record_trace
        self._last_token: int | None = None

    def _validate_initial_cache(self, entries: Sequence[CacheEntry]) -> None:
        by_id = {entry.packet_id: entry for entry in entries}
        if len(by_id) != len(entries):
            raise SchedulerDependencyError("initial cache packet IDs must be unique")
        if sum(entry.size_bytes for entry in entries) > self.config.capacity_bytes:
            raise SchedulerError("initial cache exceeds configured capacity")
        for entry in entries:
            for dependency_id in entry.dependencies:
                if dependency_id not in by_id:
                    raise SchedulerDependencyError(
                        f"cached packet {entry.packet_id!r} misses dependency {dependency_id!r}"
                    )
        _validate_dependency_map({entry.packet_id: entry.dependencies for entry in entries})

    @property
    def cache_entries(self) -> tuple[CacheEntry, ...]:
        return tuple(self._cache[packet_id] for packet_id in sorted(self._cache))

    @property
    def cache_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self._cache.values())

    @property
    def trace(self) -> SchedulerTrace:
        return SchedulerTrace(
            config=self.config,
            initial_cache=self._initial_cache,
            events=tuple(self._events),
        )

    def export_trace(self) -> SchedulerTrace:
        """Method spelling for callers that prefer an explicit snapshot operation."""

        return self.trace

    def schedule(
        self,
        token_index: int,
        requests: Iterable[PacketRequest],
        *,
        now_us: float = 0.0,
    ) -> SchedulerDecision:
        """Schedule one epoch and atomically update the refinement cache."""

        if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 0:
            raise ValueError("token_index must be a non-negative integer")
        if self._last_token is not None and token_index < self._last_token:
            raise SchedulerError("token_index cannot move backwards")
        if not math.isfinite(now_us) or now_us < 0:
            raise ValueError("now_us must be finite and non-negative")
        request_tuple = tuple(sorted(requests, key=lambda request: request.packet_id))
        request_map = self._validate_requests(request_tuple)
        snapshot = dict(self._cache)
        last_token = self._last_token
        try:
            decision = self._schedule_impl(token_index, request_tuple, request_map)
        except Exception:
            self._cache = snapshot
            self._last_token = last_token
            raise
        self._last_token = token_index
        if self._record_trace:
            self._events.append(
                TraceEvent(
                    token_index=token_index,
                    now_us=now_us,
                    requests=request_tuple,
                    decision=decision,
                )
            )
        return decision

    def _validate_requests(
        self, requests: Sequence[PacketRequest]
    ) -> dict[str, PacketRequest]:
        request_map = {request.packet_id: request for request in requests}
        if len(request_map) != len(requests):
            raise SchedulerDependencyError("request packet IDs must be unique")
        dependency_map: dict[str, tuple[str, ...]] = {
            entry.packet_id: entry.dependencies for entry in self._cache.values()
        }
        for request in requests:
            cached = self._cache.get(request.packet_id)
            if cached is not None and (
                cached.size_bytes != request.size_bytes
                or cached.dependencies != request.dependencies
            ):
                raise SchedulerError(
                    f"request metadata changed for cached packet {request.packet_id!r}"
                )
            for dependency_id in request.dependencies:
                if dependency_id not in request_map and dependency_id not in self._cache:
                    raise SchedulerDependencyError(
                        f"packet {request.packet_id!r} depends on unavailable packet "
                        f"{dependency_id!r}"
                    )
            dependency_map[request.packet_id] = request.dependencies
        _validate_dependency_map(dependency_map)
        return request_map

    def _schedule_impl(
        self,
        token_index: int,
        requests: Sequence[PacketRequest],
        request_map: Mapping[str, PacketRequest],
    ) -> SchedulerDecision:
        scores = {
            request.packet_id: request.priority_score(self.config) for request in requests
        }
        order = sorted(
            requests,
            key=lambda request: (
                not request.required,
                -scores[request.packet_id],
                request.deadline_us,
                request.packet_id,
            ),
        )
        protected: set[str] = set()
        loaded: list[str] = []
        evicted: list[str] = []
        skipped: dict[str, str] = {}
        copy_time_us = 0.0
        logical_load_bytes = 0
        io_load_bytes = 0

        for root in order:
            closure = self._dependency_closure(root.packet_id, request_map)
            missing = [packet_id for packet_id in closure if packet_id not in self._cache]
            if not missing:
                protected.update(closure)
                self._touch(closure, request_map, scores, token_index)
                continue

            finish = copy_time_us
            missed_deadline = False
            for packet_id in missing:
                request = request_map[packet_id]
                finish += request.transfer_duration_us(self.config)
                if finish > request.deadline_us:
                    missed_deadline = True
                    break
            if finish > root.deadline_us:
                missed_deadline = True
            if missed_deadline:
                if root.required:
                    raise UnschedulableRequestError(
                        f"required packet {root.packet_id!r} cannot meet its deadline"
                    )
                skipped[root.packet_id] = "deadline"
                continue

            additional_bytes = sum(request_map[item].size_bytes for item in missing)
            needed_bytes = max(0, self.cache_bytes + additional_bytes - self.config.capacity_bytes)
            victims, reason = self._choose_evictions(
                needed_bytes=needed_bytes,
                protected=protected | set(closure),
                admission_score=scores[root.packet_id],
                token_index=token_index,
                required=root.required,
            )
            if victims is None:
                if root.required:
                    raise UnschedulableRequestError(
                        f"required packet {root.packet_id!r} cannot fit cache: {reason}"
                    )
                skipped[root.packet_id] = reason
                continue

            for victim_id in victims:
                if victim_id in self._cache:
                    del self._cache[victim_id]
                    evicted.append(victim_id)
            for packet_id in missing:
                request = request_map[packet_id]
                copy_time_us += request.transfer_duration_us(self.config)
                logical_load_bytes += request.size_bytes
                io_load_bytes += (
                    request.io_bytes if request.io_bytes is not None else request.size_bytes
                )
                self._cache[packet_id] = CacheEntry(
                    packet_id=packet_id,
                    size_bytes=request.size_bytes,
                    score=scores[packet_id],
                    loaded_token=token_index,
                    last_access_token=token_index,
                    dependencies=request.dependencies,
                )
                loaded.append(packet_id)
            protected.update(closure)
            self._touch(closure, request_map, scores, token_index)

        selected = tuple(sorted(packet_id for packet_id in request_map if packet_id in protected))
        for packet_id in selected:
            skipped.pop(packet_id, None)
        self._assert_cache_dependencies()
        return SchedulerDecision(
            token_index=token_index,
            selected=selected,
            loaded=tuple(loaded),
            evicted=tuple(evicted),
            skipped=tuple(sorted(skipped.items())),
            scores=tuple(sorted(scores.items())),
            copy_time_us=copy_time_us,
            cache_bytes=self.cache_bytes,
            cache_packet_ids=tuple(sorted(self._cache)),
            logical_load_bytes=logical_load_bytes,
            io_load_bytes=io_load_bytes,
            read_amplification=(
                io_load_bytes / logical_load_bytes if logical_load_bytes else 0.0
            ),
        )

    def _dependency_closure(
        self, packet_id: str, request_map: Mapping[str, PacketRequest]
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in visited:
                return
            visited.add(current_id)
            if current_id in request_map:
                dependencies = request_map[current_id].dependencies
            else:
                dependencies = self._cache[current_id].dependencies
            for dependency_id in sorted(dependencies):
                visit(dependency_id)
            ordered.append(current_id)

        visit(packet_id)
        return tuple(ordered)

    def _touch(
        self,
        packet_ids: Iterable[str],
        request_map: Mapping[str, PacketRequest],
        scores: Mapping[str, float],
        token_index: int,
    ) -> None:
        for packet_id in packet_ids:
            entry = self._cache[packet_id]
            self._cache[packet_id] = CacheEntry(
                packet_id=entry.packet_id,
                size_bytes=entry.size_bytes,
                score=scores.get(packet_id, entry.score),
                loaded_token=entry.loaded_token,
                last_access_token=token_index,
                dependencies=(
                    request_map[packet_id].dependencies
                    if packet_id in request_map
                    else entry.dependencies
                ),
            )

    def _choose_evictions(
        self,
        *,
        needed_bytes: int,
        protected: set[str],
        admission_score: float,
        token_index: int,
        required: bool,
    ) -> tuple[tuple[str, ...] | None, str]:
        if needed_bytes <= 0:
            return (), ""
        candidates = sorted(
            self._cache.values(),
            key=lambda entry: (entry.score, entry.last_access_token, entry.packet_id),
        )
        planned: set[str] = set()
        freed = 0
        saw_min_residency = False
        saw_hysteresis = False
        for candidate in candidates:
            if candidate.packet_id in planned:
                continue
            bundle = self._dependent_bundle(candidate.packet_id) - planned
            if bundle & protected:
                continue
            entries = [self._cache[packet_id] for packet_id in sorted(bundle)]
            if not required and any(
                token_index - entry.loaded_token < self.config.min_residency_tokens
                for entry in entries
            ):
                saw_min_residency = True
                continue
            if not required and any(
                admission_score < entry.score * (1.0 + self.config.hysteresis)
                for entry in entries
            ):
                saw_hysteresis = True
                continue
            planned.update(bundle)
            freed += sum(entry.size_bytes for entry in entries)
            if freed >= needed_bytes:
                return tuple(sorted(planned)), ""
        if saw_min_residency:
            return None, "min_residency"
        if saw_hysteresis:
            return None, "hysteresis"
        return None, "capacity"

    def _dependent_bundle(self, packet_id: str) -> set[str]:
        reverse: dict[str, set[str]] = {item: set() for item in self._cache}
        for entry in self._cache.values():
            for dependency_id in entry.dependencies:
                reverse.setdefault(dependency_id, set()).add(entry.packet_id)
        bundle: set[str] = set()
        stack = [packet_id]
        while stack:
            current = stack.pop()
            if current in bundle:
                continue
            bundle.add(current)
            stack.extend(sorted(reverse.get(current, ()), reverse=True))
        return bundle

    def _assert_cache_dependencies(self) -> None:
        for entry in self._cache.values():
            missing = set(entry.dependencies) - set(self._cache)
            if missing:
                raise SchedulerDependencyError(
                    f"cache invariant failed for {entry.packet_id!r}: missing {sorted(missing)}"
                )

    @classmethod
    def from_trace(cls, trace: SchedulerTrace, *, verify: bool = True) -> "CacheDeadlineScheduler":
        """Replay *trace* and return the reconstructed scheduler state."""

        scheduler = cls(trace.config, initial_cache=trace.initial_cache, record_trace=True)
        for index, event in enumerate(trace.events):
            decision = scheduler.schedule(
                event.token_index,
                event.requests,
                now_us=event.now_us,
            )
            if verify and decision != event.decision:
                raise ReplayMismatchError(
                    f"scheduler replay diverged at event {index}: "
                    f"expected {event.decision}, got {decision}"
                )
        return scheduler


def _validate_dependency_map(dependencies: Mapping[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(packet_id: str) -> None:
        if packet_id in visiting:
            raise SchedulerDependencyError(f"dependency cycle includes packet {packet_id!r}")
        if packet_id in visited:
            return
        visiting.add(packet_id)
        for dependency_id in dependencies.get(packet_id, ()):
            if dependency_id not in dependencies:
                raise SchedulerDependencyError(
                    f"packet {packet_id!r} depends on missing packet {dependency_id!r}"
                )
            visit(dependency_id)
        visiting.remove(packet_id)
        visited.add(packet_id)

    for packet_id in sorted(dependencies):
        visit(packet_id)


def replay_trace(
    trace: SchedulerTrace | str | bytes | bytearray,
    *,
    verify: bool = True,
) -> tuple[SchedulerDecision, ...]:
    """Replay a trace and return its regenerated decisions."""

    parsed = SchedulerTrace.from_json(trace) if isinstance(trace, (str, bytes, bytearray)) else trace
    scheduler = CacheDeadlineScheduler.from_trace(parsed, verify=verify)
    return tuple(event.decision for event in scheduler.trace.events)


__all__ = [
    "CacheDeadlineScheduler",
    "CacheEntry",
    "PacketRequest",
    "ReplayMismatchError",
    "SchedulerConfig",
    "SchedulerDecision",
    "SchedulerDependencyError",
    "SchedulerError",
    "SchedulerTrace",
    "TraceEvent",
    "UnschedulableRequestError",
    "replay_trace",
]
