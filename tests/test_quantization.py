from __future__ import annotations

import unittest

import numpy as np

from harpq.quantization import (
    ProgressiveQuantizedMatrix,
    dequantize_prefix,
    dequantize_stage,
    pack_2bit,
    progressive_matmul,
    quantize_progressive,
    unpack_2bit,
)


class TwoBitPackingTests(unittest.TestCase):
    def test_known_twos_complement_layout_and_padding(self) -> None:
        codes = np.array([-2, -1, 0, 1, 1], dtype=np.int8)
        packed = pack_2bit(codes)

        # 10, 11, 00, 01 in least-to-most-significant pair order.
        np.testing.assert_array_equal(packed, np.array([0x4E, 0x01], dtype=np.uint8))
        np.testing.assert_array_equal(unpack_2bit(packed, codes.size), codes)

    def test_arbitrary_shape_round_trip(self) -> None:
        codes = np.array(
            [[-2, -1, 0], [1, 0, -2], [-1, 1, 1]], dtype=np.int8
        )
        decoded = unpack_2bit(pack_2bit(codes), codes.size).reshape(codes.shape)
        np.testing.assert_array_equal(decoded, codes)

    def test_pack_and_unpack_reject_invalid_data(self) -> None:
        with self.assertRaises(TypeError):
            pack_2bit(np.array([0.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            pack_2bit(np.array([2], dtype=np.int8))
        with self.assertRaises(TypeError):
            unpack_2bit(np.array([0], dtype=np.int8), 1)
        with self.assertRaises(ValueError):
            unpack_2bit(np.array([], dtype=np.uint8), 1)


class ProgressiveQuantizationTests(unittest.TestCase):
    def test_default_geometry_is_tile_aligned_and_deterministic(self) -> None:
        rng = np.random.default_rng(19)
        weights = rng.normal(0.0, 0.25, size=(137, 259)).astype(np.float32)

        first = quantize_progressive(weights)
        second = quantize_progressive(weights)

        self.assertIsInstance(first, ProgressiveQuantizedMatrix)
        self.assertEqual(first.original_shape, (137, 259))
        self.assertEqual(first.padded_shape, (256, 384))
        self.assertEqual(first.tile_shape, (128, 128))
        self.assertEqual(first.group_size, 128)
        self.assertEqual(first.stage_count, 4)
        for first_stage, second_stage in zip(first.stages, second.stages, strict=True):
            self.assertEqual(first_stage.packed_codes.dtype, np.uint8)
            self.assertEqual(first_stage.scales.dtype, np.float16)
            self.assertEqual(first_stage.scales.shape, (2, 3, 128, 1))
            np.testing.assert_array_equal(
                first_stage.packed_codes, second_stage.packed_codes
            )
            np.testing.assert_array_equal(first_stage.scales, second_stage.scales)

    def test_prefixes_are_finite_and_weight_mse_is_monotone(self) -> None:
        rng = np.random.default_rng(23)
        weights = rng.standard_normal((41, 145), dtype=np.float32)
        quantized = quantize_progressive(weights)

        errors: list[float] = []
        for prefix in range(quantized.stage_count + 1):
            reconstructed = dequantize_prefix(quantized, prefix)
            self.assertEqual(reconstructed.shape, weights.shape)
            self.assertEqual(reconstructed.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(reconstructed)))
            errors.append(float(np.mean((weights - reconstructed) ** 2)))

        self.assertTrue(
            all(current <= previous + 1e-12 for previous, current in zip(errors, errors[1:])),
            msg=f"expected non-increasing error, observed {errors}",
        )
        self.assertLess(errors[-1], errors[1])
        np.testing.assert_array_equal(
            dequantize_prefix(quantized, 0), np.zeros_like(weights)
        )

    def test_dequantized_prefix_is_sum_of_residual_stages(self) -> None:
        rng = np.random.default_rng(29)
        weights = rng.normal(size=(11, 21)).astype(np.float32)
        quantized = quantize_progressive(
            weights, tile_shape=(16, 32), group_size=8
        )

        stage_sum = np.zeros_like(weights)
        for index in range(quantized.stage_count):
            stage_sum += dequantize_stage(quantized, index)
            np.testing.assert_array_equal(
                stage_sum, dequantize_prefix(quantized, index + 1)
            )

    def test_progressive_matmul_matches_materialized_prefix(self) -> None:
        rng = np.random.default_rng(31)
        weights = rng.normal(0.0, 0.4, size=(13, 19)).astype(np.float32)
        inputs = rng.normal(size=(2, 3, 19)).astype(np.float32)
        bias = rng.normal(size=(13,)).astype(np.float32)
        quantized = quantize_progressive(
            weights, tile_shape=(16, 32), group_size=8
        )

        for prefix in range(quantized.stage_count + 1):
            expected = inputs @ dequantize_prefix(quantized, prefix).T + bias
            actual = progressive_matmul(
                inputs, quantized, prefix=prefix, bias=bias
            )
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_zero_matrix_has_zero_scales_and_reconstruction(self) -> None:
        weights = np.zeros((7, 9), dtype=np.float32)
        quantized = quantize_progressive(
            weights, tile_shape=(8, 16), group_size=8
        )
        for stage in quantized.stages:
            self.assertTrue(np.all(stage.scales == 0))
        np.testing.assert_array_equal(quantized.dequantize(), weights)

    def test_geometry_and_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            quantize_progressive(np.zeros((2, 2, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            quantize_progressive(np.array([[np.nan]], dtype=np.float32))
        with self.assertRaises(ValueError):
            quantize_progressive(
                np.ones((2, 2), dtype=np.float32),
                tile_shape=(8, 10),
                group_size=4,
            )
        with self.assertRaises(ValueError):
            quantize_progressive(np.ones((2, 2), dtype=np.float32), stage_count=5)
        with self.assertRaises(TypeError):
            quantize_progressive(np.ones((2, 2), dtype=np.complex64) * (1 + 1j))

        quantized = quantize_progressive(
            np.ones((3, 5), dtype=np.float32), tile_shape=(4, 8), group_size=4
        )
        with self.assertRaises(ValueError):
            dequantize_prefix(quantized, 5)
        with self.assertRaises(ValueError):
            progressive_matmul(np.ones((2, 4), dtype=np.float32), quantized)
        with self.assertRaises(TypeError):
            progressive_matmul(np.ones((2, 5), dtype=np.complex64), quantized)
        with self.assertRaises(TypeError):
            progressive_matmul(
                np.ones((2, 5), dtype=np.float32),
                quantized,
                bias=np.ones(3, dtype=np.complex64),
            )


if __name__ == "__main__":
    unittest.main()
