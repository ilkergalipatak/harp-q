from __future__ import annotations

import unittest

from harpq.allocator import (
    AllocationDependencyError,
    AllocationItem,
    InfeasibleBudgetError,
    allocate_packets,
    enumerate_group_options,
)


class AllocatorTests(unittest.TestCase):
    def test_exact_knapsack_selects_best_dependency_closed_prefix(self) -> None:
        items = (
            AllocationItem("a.base", "a", 4, 10.0, mandatory=True),
            AllocationItem("a.r1", "a", 4, 6.0, dependencies=("a.base",)),
            AllocationItem("b.base", "b", 4, 8.0, mandatory=True),
            AllocationItem("b.r1", "b", 4, 20.0, dependencies=("b.base",)),
        )
        result = allocate_packets(items, 12)
        self.assertEqual(result.total_cost_bytes, 12)
        self.assertEqual(result.total_distortion_gain, 38.0)
        self.assertEqual(result.selected_packet_ids, ("a.base", "b.base", "b.r1"))
        self.assertEqual(result.selected_by_group["a"], ("a.base",))
        self.assertEqual(result.selected_by_group["b"], ("b.base", "b.r1"))

    def test_group_options_never_expose_a_residual_without_its_base(self) -> None:
        options = enumerate_group_options(
            (
                AllocationItem("base", "tile", 4, 10.0, mandatory=True),
                AllocationItem("r1", "tile", 2, 4.0, dependencies=("base",)),
                AllocationItem("r2", "tile", 2, 2.0, dependencies=("r1",)),
            )
        )
        self.assertEqual(
            [option.packet_ids for option in options],
            [("base",), ("base", "r1"), ("base", "r1", "r2")],
        )

    def test_ties_are_deterministic_by_cost_then_packet_id(self) -> None:
        items = (
            AllocationItem("b", "group-b", 1, 1.0),
            AllocationItem("a", "group-a", 1, 1.0),
        )
        forward = allocate_packets(items, 1)
        reverse = allocate_packets(reversed(items), 1)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.selected_packet_ids, ("a",))

    def test_mandatory_prefix_over_budget_is_infeasible(self) -> None:
        items = (AllocationItem("base", "tile", 10, 1.0, mandatory=True),)
        with self.assertRaises(InfeasibleBudgetError):
            allocate_packets(items, 9)

    def test_missing_cycle_and_cross_group_dependencies_are_rejected(self) -> None:
        with self.assertRaises(AllocationDependencyError):
            allocate_packets((AllocationItem("r1", "tile", 1, 1.0, ("base",)),), 10)
        cycle = (
            AllocationItem("a", "tile", 1, 1.0, ("b",)),
            AllocationItem("b", "tile", 1, 1.0, ("a",)),
        )
        with self.assertRaises(AllocationDependencyError):
            allocate_packets(cycle, 10)
        cross_group = (
            AllocationItem("a", "one", 1, 1.0),
            AllocationItem("b", "two", 1, 1.0, ("a",)),
        )
        with self.assertRaisesRegex(AllocationDependencyError, "cross-group"):
            allocate_packets(cross_group, 10)

    def test_empty_problem_has_zero_cost_solution(self) -> None:
        result = allocate_packets((), 0)
        self.assertEqual(result.selected_packet_ids, ())
        self.assertEqual(result.total_cost_bytes, 0)
        self.assertEqual(result.total_distortion_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
