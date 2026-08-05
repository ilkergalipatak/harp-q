from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from harpq.pipeline import (
    allocate_calibrated_matrices,
    allocation_items_from_profile,
    calibrate_matrix,
    load_matrix_prefix,
    write_calibrated_container,
)


class OfflinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(31)
        weights = rng.standard_normal((12, 10), dtype=np.float32)
        activations = rng.standard_normal((24, 10), dtype=np.float32)
        cls.calibrated = calibrate_matrix(
            "layers.0.mlp.up",
            weights,
            activations,
            tile_shape=(8, 16),
            group_size=4,
        )

    def test_profile_becomes_dependency_closed_allocation_items(self) -> None:
        items = allocation_items_from_profile(self.calibrated)
        self.assertTrue(items[0].mandatory)
        self.assertEqual(items[0].dependencies, ())
        for index in range(1, len(items)):
            self.assertFalse(items[index].mandatory)
            self.assertEqual(items[index].dependencies, (items[index - 1].packet_id,))
            self.assertGreaterEqual(items[index].distortion_gain, 0.0)

        base_budget = items[0].cost_bytes
        allocation = allocate_calibrated_matrices([self.calibrated], base_budget)
        self.assertEqual(allocation.selected_packet_ids, (items[0].packet_id,))

    def test_calibrate_pack_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tiny.harpq"
            write_calibrated_container(
                destination,
                [self.calibrated],
                metadata={"fixture": True},
            )
            loaded = load_matrix_prefix(destination, self.calibrated.matrix_id)
            np.testing.assert_array_equal(
                loaded.dequantize(),
                self.calibrated.quantized.dequantize(),
            )

    def test_manifest_stage_count_and_reserved_metadata_are_truthful(self) -> None:
        rng = np.random.default_rng(37)
        two_stage = calibrate_matrix(
            "two-stage",
            rng.standard_normal((4, 4), dtype=np.float32),
            rng.standard_normal((8, 4), dtype=np.float32),
            tile_shape=(4, 4),
            group_size=2,
            stage_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_calibrated_container(root / "two.harpq", [two_stage])
            self.assertEqual(manifest.metadata["stage_bits"], [2, 2])
            with self.assertRaises(ValueError):
                write_calibrated_container(
                    root / "invalid.harpq",
                    [two_stage],
                    metadata={"stage_bits": [8]},
                )


if __name__ == "__main__":
    unittest.main()
