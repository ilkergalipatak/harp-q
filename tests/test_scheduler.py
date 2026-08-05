from __future__ import annotations

import json
import unittest

from harpq.scheduler import (
    CacheDeadlineScheduler,
    PacketRequest,
    ReplayMismatchError,
    SchedulerConfig,
    SchedulerDependencyError,
    SchedulerTrace,
    replay_trace,
)


def request(
    packet_id: str,
    *,
    gain: float,
    size: int = 100,
    transfer_us: float = 100.0,
    deadline_us: float = 1000.0,
    dependencies: tuple[str, ...] = (),
) -> PacketRequest:
    return PacketRequest(
        packet_id=packet_id,
        size_bytes=size,
        distortion_gain=gain,
        deadline_us=deadline_us,
        dependencies=dependencies,
        transfer_us=transfer_us,
    )


class SchedulerTests(unittest.TestCase):
    def test_aligned_io_bytes_drive_transfer_without_inflating_cache(self) -> None:
        config = SchedulerConfig(
            capacity_bytes=100,
            bandwidth_bytes_per_us=1024.0,
            fixed_read_latency_us=2.0,
        )
        packet = PacketRequest(
            packet_id="small",
            size_bytes=68,
            io_bytes=4096,
            distortion_gain=1.0,
            deadline_us=10.0,
        )
        self.assertEqual(packet.transfer_duration_us(config), 6.0)
        decision = CacheDeadlineScheduler(config).schedule(0, (packet,))
        self.assertEqual(decision.copy_time_us, 6.0)
        self.assertEqual(decision.cache_bytes, 68)
        self.assertEqual(decision.logical_load_bytes, 68)
        self.assertEqual(decision.io_load_bytes, 4096)
        self.assertAlmostEqual(decision.read_amplification, 4096 / 68)
        with self.assertRaises(ValueError):
            PacketRequest("bad", 100, 1.0, 10.0, io_bytes=99)

    def test_dependency_chain_loads_in_topological_order(self) -> None:
        scheduler = CacheDeadlineScheduler(SchedulerConfig(capacity_bytes=300))
        base = request("base", gain=10, size=100, transfer_us=10)
        r1 = request("r1", gain=8, size=100, transfer_us=10, dependencies=("base",))
        decision = scheduler.schedule(0, (r1, base))
        self.assertEqual(decision.loaded, ("base", "r1"))
        self.assertEqual(decision.selected, ("base", "r1"))
        self.assertEqual(decision.cache_packet_ids, ("base", "r1"))
        self.assertEqual(decision.copy_time_us, 20.0)

    def test_late_optional_packet_is_skipped_without_partial_load(self) -> None:
        scheduler = CacheDeadlineScheduler(SchedulerConfig(capacity_bytes=100))
        decision = scheduler.schedule(
            0,
            (request("late", gain=100, transfer_us=100, deadline_us=50),),
        )
        self.assertEqual(decision.selected, ())
        self.assertEqual(decision.loaded, ())
        self.assertEqual(decision.skipped, (("late", "deadline"),))
        self.assertEqual(decision.cache_bytes, 0)

    def test_min_residency_then_hysteresis_prevent_thrashing(self) -> None:
        scheduler = CacheDeadlineScheduler(
            SchedulerConfig(capacity_bytes=100, min_residency_tokens=8, hysteresis=0.15)
        )
        first = scheduler.schedule(0, (request("a", gain=100),))
        self.assertEqual(first.loaded, ("a",))

        too_early = scheduler.schedule(1, (request("b", gain=130),))
        self.assertEqual(too_early.skipped, (("b", "min_residency"),))
        self.assertEqual(too_early.cache_packet_ids, ("a",))

        too_small = scheduler.schedule(8, (request("c", gain=110),))
        self.assertEqual(too_small.skipped, (("c", "hysteresis"),))
        self.assertEqual(too_small.cache_packet_ids, ("a",))

        admitted = scheduler.schedule(8, (request("b", gain=120),))
        self.assertEqual(admitted.evicted, ("a",))
        self.assertEqual(admitted.loaded, ("b",))
        self.assertEqual(admitted.cache_packet_ids, ("b",))

    def test_eviction_preserves_dependency_closure(self) -> None:
        scheduler = CacheDeadlineScheduler(
            SchedulerConfig(capacity_bytes=200, min_residency_tokens=0, hysteresis=0)
        )
        base = request("base", gain=10, size=100, transfer_us=10)
        residual = request(
            "residual", gain=10, size=100, transfer_us=10, dependencies=("base",)
        )
        scheduler.schedule(0, (base, residual))
        replacement = scheduler.schedule(
            1,
            (request("replacement", gain=1000, size=200, transfer_us=10),),
        )
        self.assertEqual(set(replacement.evicted), {"base", "residual"})
        self.assertEqual(replacement.cache_packet_ids, ("replacement",))

    def test_trace_json_roundtrip_and_replay_are_exact(self) -> None:
        scheduler = CacheDeadlineScheduler(
            SchedulerConfig(capacity_bytes=100, min_residency_tokens=0, hysteresis=0.15)
        )
        scheduler.schedule(0, (request("a", gain=100),), now_us=10)
        scheduler.schedule(1, (request("b", gain=120),), now_us=20)
        encoded = scheduler.trace.to_json()
        parsed = SchedulerTrace.from_json(encoded)
        self.assertEqual(parsed, scheduler.trace)
        self.assertEqual(replay_trace(parsed), tuple(event.decision for event in parsed.events))

        raw = json.loads(encoded)
        raw["events"][0]["decision"]["selected"] = []
        tampered = SchedulerTrace.from_dict(raw)
        with self.assertRaises(ReplayMismatchError):
            replay_trace(tampered)

    def test_missing_and_cyclic_dependencies_are_rejected(self) -> None:
        scheduler = CacheDeadlineScheduler(SchedulerConfig(capacity_bytes=300))
        missing = request("r1", gain=1, dependencies=("base",))
        with self.assertRaises(SchedulerDependencyError):
            scheduler.schedule(0, (missing,))
        a = request("a", gain=1, dependencies=("b",))
        b = request("b", gain=1, dependencies=("a",))
        with self.assertRaises(SchedulerDependencyError):
            scheduler.schedule(0, (a, b))


if __name__ == "__main__":
    unittest.main()
