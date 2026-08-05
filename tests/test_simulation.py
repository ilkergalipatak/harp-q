from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from harpq.container import open_container
from harpq.pipeline import calibrate_matrix, write_calibrated_container
from harpq.scheduler import SchedulerConfig, replay_trace
from harpq.simulation import (
    load_scheduler_trace,
    refinement_requests_from_container,
    simulate_container,
    write_scheduler_trace,
)


class SchedulerSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(53)
        self.calibrated = calibrate_matrix(
            "layers.0.output",
            rng.standard_normal((8, 8), dtype=np.float32),
            rng.standard_normal((16, 8), dtype=np.float32),
            tile_shape=(8, 8),
            group_size=4,
        )

    def test_container_adapter_excludes_always_resident_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory) / "model.harpq"
            write_calibrated_container(container, [self.calibrated])
            with open_container(container) as reader:
                requests = refinement_requests_from_container(reader, deadline_us=1000.0)
            self.assertEqual(len(requests), 3)
            by_stage = sorted(requests, key=lambda request: request.packet_id)
            self.assertEqual(by_stage[0].dependencies, ())
            self.assertEqual(by_stage[1].dependencies, (by_stage[0].packet_id,))
            self.assertLess(by_stage[0].size_bytes, 4096)
            self.assertEqual(by_stage[0].io_bytes, 4096)

    def test_trace_write_load_and_replay_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            container = root / "model.harpq"
            write_calibrated_container(container, [self.calibrated])
            capacity = sum(stage.payload_bytes + 20 for stage in self.calibrated.quantized.stages[1:])
            trace = simulate_container(
                container,
                SchedulerConfig(
                    capacity_bytes=capacity,
                    bandwidth_bytes_per_us=1000.0,
                    min_residency_tokens=1,
                ),
                token_count=3,
                deadline_us=1000.0,
            )
            path = write_scheduler_trace(root / "trace.json", trace)
            loaded = load_scheduler_trace(path)
            self.assertEqual(replay_trace(loaded), tuple(event.decision for event in trace.events))
            self.assertEqual(len(loaded.events), 3)
            self.assertGreater(loaded.events[0].decision.read_amplification, 1.0)
            self.assertEqual(loaded.events[1].decision.io_load_bytes, 0)


if __name__ == "__main__":
    unittest.main()
