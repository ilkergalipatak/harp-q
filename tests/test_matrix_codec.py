from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from harpq.container import MissingPackError, Packet, open_container, write_container
from harpq.matrix_codec import (
    MatrixCodecError,
    matrix_to_packets,
    packet_id_for_stage,
    packets_to_matrix,
)
from harpq.quantization import dequantize_prefix, quantize_progressive


class ProgressiveMatrixCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        weights = np.random.default_rng(23).standard_normal((11, 13), dtype=np.float32)
        self.quantized = quantize_progressive(
            weights,
            tile_shape=(8, 16),
            group_size=4,
        )
        self.packets = matrix_to_packets(
            "layers.0.attention.query",
            self.quantized,
            metadata={"role": "query"},
        )

    def test_packet_roundtrip_preserves_every_prefix(self) -> None:
        decoded = packets_to_matrix(self.packets)
        self.assertEqual(decoded.original_shape, self.quantized.original_shape)
        self.assertEqual(decoded.tile_shape, self.quantized.tile_shape)
        for prefix in range(1, self.quantized.stage_count + 1):
            np.testing.assert_array_equal(
                dequantize_prefix(decoded, prefix),
                dequantize_prefix(self.quantized, prefix),
            )

    def test_container_roundtrip_and_optional_residual_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            container_path = Path(directory) / "model.harpq"
            manifest = write_container(
                container_path,
                self.packets,
                metadata={"model": "synthetic"},
            )
            self.assertEqual(manifest.metadata["model"], "synthetic")
            reader = open_container(container_path)
            chain = reader.read_chain(packet_id_for_stage("layers.0.attention.query", 3))
            decoded = packets_to_matrix(chain)
            np.testing.assert_array_equal(decoded.dequantize(), self.quantized.dequantize())
            for record in reader:
                self.assertEqual(record.offset % 4096, 0)

            (container_path / "refine-3.pack").unlink()
            fallback_reader = open_container(container_path)
            fallback_chain = fallback_reader.read_chain(
                packet_id_for_stage("layers.0.attention.query", 3)
            )
            fallback = packets_to_matrix(fallback_chain)
            self.assertEqual(fallback.stage_count, 3)
            np.testing.assert_array_equal(
                fallback.dequantize(),
                dequantize_prefix(self.quantized, 3),
            )

    def test_missing_base_pack_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            container_path = Path(directory) / "model.harpq"
            write_container(container_path, self.packets)
            (container_path / "base.pack").unlink()
            with self.assertRaises(MissingPackError):
                open_container(container_path)

    def test_codec_rejects_inverted_optionality_and_scale_dtype(self) -> None:
        base = self.packets[0]
        optional_base = Packet(
            base.packet_id,
            base.stage,
            base.payload,
            base.dependencies,
            True,
            base.metadata,
        )
        with self.assertRaises(MatrixCodecError):
            packets_to_matrix((optional_base, *self.packets[1:]))

        residual = self.packets[1]
        required_residual = Packet(
            residual.packet_id,
            residual.stage,
            residual.payload,
            residual.dependencies,
            False,
            residual.metadata,
        )
        with self.assertRaises(MatrixCodecError):
            packets_to_matrix((base, required_residual))

        invalid_metadata = dict(base.metadata)
        invalid_metadata["scale_dtype"] = "native"
        invalid_dtype = Packet(
            base.packet_id,
            base.stage,
            base.payload,
            base.dependencies,
            base.optional,
            invalid_metadata,
        )
        with self.assertRaises(MatrixCodecError):
            packets_to_matrix((invalid_dtype,))


if __name__ == "__main__":
    unittest.main()
