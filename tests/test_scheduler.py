from __future__ import annotations

import json
import unittest

from harpq.scheduler import (
    CacheDeadlineScheduler,
    PacketRequest,
    ReplayMismatchError,
    SchedulerConfig,
    SchedulerDependencyError,
    SchedulerError,
    SchedulerTrace,
    ServiceObservation,
    UnschedulableRequestError,
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
    @staticmethod
    def observation(
        sequence_id: int,
        critical_path_us: float,
        *,
        packet_id: str | None = None,
        logical_bytes: int = 100,
        physical_io_bytes: int = 4096,
        service_class: str = "direct-h2d",
    ) -> ServiceObservation:
        return ServiceObservation(
            sequence_id=sequence_id,
            packet_id=packet_id or f"observed-{sequence_id}",
            logical_bytes=logical_bytes,
            physical_io_bytes=physical_io_bytes,
            critical_path_us=critical_path_us,
            service_class=service_class,
            io_service_us=critical_path_us * 0.75,
            h2d_service_us=critical_path_us * 0.25,
        )

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

    def test_resident_hit_keeps_hypothetical_reload_estimate_but_charges_no_copy(self) -> None:
        scheduler = CacheDeadlineScheduler(SchedulerConfig(capacity_bytes=100))
        packet = request("resident", gain=10, transfer_us=17.0)
        self.assertEqual(scheduler.schedule(0, (packet,)).loaded, ("resident",))

        hit = scheduler.schedule(1, (packet,))
        self.assertEqual(hit.selected, ("resident",))
        self.assertEqual(hit.loaded, ())
        self.assertEqual(hit.copy_time_us, 0.0)
        self.assertEqual(hit.logical_load_bytes, 0)
        self.assertEqual(hit.io_load_bytes, 0)
        self.assertEqual(hit.transfer_estimates[0].source, "request")
        self.assertEqual(hit.transfer_estimates[0].transfer_us, 17.0)

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

    def test_feedback_uses_exact_class_and_byte_bucket_after_minimum_samples(self) -> None:
        config = SchedulerConfig(
            capacity_bytes=100,
            bandwidth_bytes_per_us=4096.0,
            feedback_window_size=4,
            feedback_min_samples=3,
            feedback_quantile=0.95,
        )
        packet = PacketRequest(
            packet_id="target",
            size_bytes=100,
            io_bytes=4096,
            distortion_gain=1.0,
            deadline_us=10.0,
            service_class="direct-h2d",
        )
        scheduler = CacheDeadlineScheduler(config)
        decision = scheduler.schedule(
            0,
            (packet,),
            observations=(
                self.observation(0, 7.0),
                ServiceObservation(
                    1,
                    "overlapped",
                    100,
                    4096,
                    11.0,
                    "direct-h2d",
                    io_service_us=9.0,
                    h2d_service_us=8.0,
                ),
                self.observation(2, 9.0),
            ),
        )
        self.assertEqual(decision.skipped, (("target", "deadline"),))
        self.assertEqual(decision.transfer_estimates[0].source, "feedback")
        self.assertEqual(decision.transfer_estimates[0].sample_count, 3)
        self.assertEqual(decision.transfer_estimates[0].transfer_us, 11.0)

        different_class = PacketRequest(
            packet_id="buffered",
            size_bytes=100,
            io_bytes=4096,
            distortion_gain=1.0,
            deadline_us=10.0,
            service_class="buffered-h2d",
        )
        fallback = scheduler.schedule(1, (different_class,))
        self.assertEqual(fallback.loaded, ("buffered",))
        self.assertEqual(fallback.transfer_estimates[0].source, "config")
        self.assertEqual(fallback.transfer_estimates[0].transfer_us, 1.0)

    def test_explicit_request_cost_overrides_feedback(self) -> None:
        scheduler = CacheDeadlineScheduler(
            SchedulerConfig(capacity_bytes=100, feedback_min_samples=1)
        )
        packet = PacketRequest(
            packet_id="explicit",
            size_bytes=100,
            io_bytes=4096,
            distortion_gain=1.0,
            deadline_us=5.0,
            transfer_us=3.0,
            service_class="direct-h2d",
        )
        decision = scheduler.schedule(
            0,
            (packet,),
            observations=(self.observation(0, 100.0),),
        )
        self.assertEqual(decision.loaded, ("explicit",))
        self.assertEqual(decision.transfer_estimates[0].source, "request")
        self.assertEqual(decision.transfer_estimates[0].transfer_us, 3.0)

    def test_feedback_trace_is_replayable_and_does_not_use_now_us(self) -> None:
        config = SchedulerConfig(
            capacity_bytes=0,
            feedback_window_size=3,
            feedback_min_samples=2,
            feedback_quantile=0.5,
        )
        observations = (self.observation(0, 9.0), self.observation(1, 5.0))
        packet = PacketRequest(
            packet_id="optional",
            size_bytes=100,
            io_bytes=4096,
            distortion_gain=1.0,
            deadline_us=100.0,
            service_class="direct-h2d",
        )
        first = CacheDeadlineScheduler(config)
        decision = first.schedule(0, (packet,), now_us=1.0, observations=observations)
        second = CacheDeadlineScheduler(config)
        self.assertEqual(
            decision,
            second.schedule(0, (packet,), now_us=999999.0, observations=observations),
        )
        self.assertEqual(decision.transfer_estimates[0].transfer_us, 5.0)
        encoded = first.trace.to_json()
        parsed = SchedulerTrace.from_json(encoded)
        self.assertEqual(parsed.format_version, 2)
        self.assertEqual(parsed, first.trace)
        self.assertEqual(replay_trace(parsed), tuple(event.decision for event in parsed.events))

        raw = json.loads(encoded)
        raw["events"][0]["observations"][1]["critical_path_us"] = 500.0
        tampered = SchedulerTrace.from_dict(raw)
        with self.assertRaises(ReplayMismatchError):
            replay_trace(tampered)

    def test_feedback_sequence_and_observation_validation_are_strict(self) -> None:
        scheduler = CacheDeadlineScheduler(
            SchedulerConfig(capacity_bytes=0, feedback_min_samples=1)
        )
        scheduler.schedule(0, (), observations=(self.observation(4, 5.0),))
        with self.assertRaises(SchedulerError):
            scheduler.schedule(1, (), observations=(self.observation(4, 6.0),))
        required = PacketRequest(
            "required",
            100,
            1.0,
            1.0,
            io_bytes=4096,
            required=True,
            service_class="direct-h2d",
        )
        with self.assertRaises(UnschedulableRequestError):
            scheduler.schedule(2, (required,), observations=(self.observation(5, 5.0),))
        scheduler.schedule(2, (), observations=(self.observation(5, 5.0),))
        with self.assertRaises(ValueError):
            self.observation(5, 5.0, logical_bytes=4097, physical_io_bytes=4096)
        with self.assertRaises(ValueError):
            ServiceObservation(5, "bad", 100, 4096, 5.0, io_service_us=6.0)

    def test_v1_trace_remains_readable_and_replayable(self) -> None:
        encoded = """{
          "config": {
            "bandwidth_bytes_per_us": 7000.0,
            "capacity_bytes": 100,
            "fixed_read_latency_us": 0.0,
            "hysteresis": 0.15,
            "min_residency_tokens": 8
          },
          "events": [{
            "decision": {
              "cache_bytes": 100,
              "cache_packet_ids": ["legacy"],
              "copy_time_us": 100.0,
              "evicted": [],
              "io_load_bytes": 100,
              "loaded": ["legacy"],
              "logical_load_bytes": 100,
              "read_amplification": 1.0,
              "scores": [["legacy", 0.01]],
              "selected": ["legacy"],
              "skipped": [],
              "token_index": 0
            },
            "now_us": 3.0,
            "requests": [{
              "deadline_us": 1000.0,
              "dependencies": [],
              "distortion_gain": 1.0,
              "importance": 1.0,
              "io_bytes": null,
              "kernel_cost_us": 0.0,
              "packet_id": "legacy",
              "required": false,
              "reuse": 1.0,
              "size_bytes": 100,
              "transfer_us": 100.0
            }],
            "token_index": 0
          }],
          "format": "harpq-scheduler-trace",
          "format_version": 1,
          "initial_cache": []
        }"""
        parsed = SchedulerTrace.from_json(encoded)
        self.assertEqual(parsed.format_version, 1)
        self.assertNotIn("observations", parsed.to_dict()["events"][0])
        self.assertEqual(replay_trace(parsed), tuple(event.decision for event in parsed.events))

        injected_config = json.loads(encoded)
        injected_config["config"]["feedback_window_size"] = 32
        injected_request = json.loads(encoded)
        injected_request["events"][0]["requests"][0]["service_class"] = "default"
        injected_event = json.loads(encoded)
        injected_event["events"][0]["observations"] = []
        injected_decision = json.loads(encoded)
        injected_decision["events"][0]["decision"]["transfer_estimates"] = []
        for label, invalid in (
            ("feedback config", injected_config),
            ("request service class", injected_request),
            ("event observations", injected_event),
            ("decision estimates", injected_decision),
        ):
            with self.subTest(v1_injected_field=label):
                with self.assertRaises(SchedulerError):
                    SchedulerTrace.from_dict(invalid)

    def test_v2_trace_requires_every_feedback_schema_field(self) -> None:
        scheduler = CacheDeadlineScheduler(SchedulerConfig(capacity_bytes=100))
        scheduler.schedule(0, (request("v2", gain=1.0),))
        encoded = scheduler.trace.to_json()

        missing_config = json.loads(encoded)
        missing_config["config"].pop("feedback_window_size")
        missing_request = json.loads(encoded)
        missing_request["events"][0]["requests"][0].pop("service_class")
        missing_event = json.loads(encoded)
        missing_event["events"][0].pop("observations")
        missing_decision = json.loads(encoded)
        missing_decision["events"][0]["decision"].pop("transfer_estimates")
        for label, invalid in (
            ("feedback config", missing_config),
            ("request service class", missing_request),
            ("event observations", missing_event),
            ("decision estimates", missing_decision),
        ):
            with self.subTest(v2_missing_field=label):
                with self.assertRaises(SchedulerError):
                    SchedulerTrace.from_dict(invalid)

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
