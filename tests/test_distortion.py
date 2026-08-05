from __future__ import annotations

import unittest

import numpy as np

from harpq.distortion import (
    DistortionPoint,
    DistortionProfile,
    activation_weighted_mse,
    build_distortion_profile,
    hessian_proxy,
    hessian_weighted_error,
    profile_progressive_quantization,
    stable_kl_divergence,
)
from harpq.quantization import quantize_progressive


class DistortionMetricTests(unittest.TestCase):
    def test_activation_mse_matches_normalized_hessian_objective(self) -> None:
        reference = np.array(
            [[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]], dtype=np.float32
        )
        approximation = np.array(
            [[0.8, -1.7, 0.4], [0.2, 0.5, -0.8]], dtype=np.float32
        )
        activations = np.array(
            [[1.0, 0.0, 2.0], [-1.0, 3.0, 0.5], [0.25, -0.5, 1.0]],
            dtype=np.float32,
        )

        direct = activation_weighted_mse(reference, approximation, activations)
        proxy = hessian_proxy(activations, damping=0.0)
        via_hessian = hessian_weighted_error(
            reference, approximation, proxy, reduction="mean"
        )

        self.assertAlmostEqual(direct, via_hessian, places=12)
        np.testing.assert_allclose(proxy, proxy.T, rtol=0.0, atol=0.0)

    def test_hessian_damping_is_added_after_normalization(self) -> None:
        activations = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = activations.T.astype(np.float64) @ activations.astype(np.float64) / 2
        expected += np.eye(2) * 0.25
        np.testing.assert_allclose(
            hessian_proxy(activations, damping=0.25), expected, rtol=0, atol=1e-14
        )

    def test_stable_kl_handles_large_logits(self) -> None:
        reference = np.array(
            [[10_000.0, 9_999.0, -10_000.0], [-12_000.0, -11_999.0, 12_000.0]]
        )
        approximation = reference + np.array([[0.0, 0.25, 1.0], [2.0, -0.5, 0.0]])

        same = stable_kl_divergence(reference, reference)
        divergence = stable_kl_divergence(
            reference, approximation, reduction="none"
        )
        self.assertEqual(same, 0.0)
        self.assertEqual(divergence.shape, (2,))
        self.assertTrue(np.all(np.isfinite(divergence)))
        self.assertTrue(np.all(divergence >= 0))

    def test_kl_is_invariant_to_per_row_constant_shift(self) -> None:
        logits = np.array([[0.0, 1.0, 2.0], [-2.0, 0.0, 4.0]])
        shifted = logits + np.array([[100.0], [-250.0]])
        self.assertAlmostEqual(
            float(stable_kl_divergence(logits, shifted)), 0.0, places=14
        )

    def test_metric_validation(self) -> None:
        with self.assertRaises(ValueError):
            stable_kl_divergence(np.zeros((2, 3)), np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            stable_kl_divergence(np.zeros((2, 3)), np.zeros((2, 3)), temperature=0)
        with self.assertRaises(ValueError):
            hessian_proxy(np.array([[np.inf]], dtype=np.float32))
        with self.assertRaises(ValueError):
            activation_weighted_mse(
                np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((4, 2))
            )
        with self.assertRaises(TypeError):
            stable_kl_divergence(
                np.ones((2, 3), dtype=np.complex64),
                np.ones((2, 3), dtype=np.complex64),
            )
        with self.assertRaises(TypeError):
            activation_weighted_mse(
                np.ones((2, 2), dtype=np.complex64),
                np.zeros((2, 2)),
                np.ones((2, 2)),
            )
        with self.assertRaises(ValueError):
            hessian_weighted_error(
                np.ones((1, 1)),
                np.zeros((1, 1)),
                np.array([[-1.0]]),
            )
        with self.assertRaises(ValueError):
            hessian_weighted_error(
                np.array([[1.0, 0.0]]),
                np.zeros((1, 2)),
                np.diag([1.0, -1.0]),
            )
        with self.assertRaises(ValueError):
            stable_kl_divergence(np.empty((0, 3)), np.empty((0, 3)))

        scalar_kl = stable_kl_divergence(
            np.array([0.0, 1.0]),
            np.array([0.5, 1.0]),
            reduction="none",
        )
        self.assertEqual(scalar_kl.shape, ())


class DistortionProfileTests(unittest.TestCase):
    def test_build_profile_records_metrics_and_optional_kl(self) -> None:
        reference = np.eye(3, dtype=np.float32)
        approximations = (np.zeros((3, 3), dtype=np.float32), reference.copy())
        activations = np.array(
            [[1.0, 0.0, -1.0], [0.25, 2.0, 0.5]], dtype=np.float32
        )
        reference_logits = activations @ reference.T
        candidate_logits = tuple(activations @ candidate.T for candidate in approximations)

        profile = build_distortion_profile(
            reference,
            approximations,
            activations,
            prefixes=(2, 4),
            reference_logits=reference_logits,
            approximate_logits=candidate_logits,
        )

        self.assertEqual(profile.prefixes, (2, 4))
        self.assertGreater(profile.points[0].weight_mse, 0)
        self.assertGreater(profile.points[0].activation_mse, 0)
        self.assertGreater(profile.points[0].logit_kl or 0.0, 0)
        self.assertEqual(profile.points[1].weight_mse, 0.0)
        self.assertEqual(profile.points[1].activation_mse, 0.0)
        self.assertEqual(profile.points[1].hessian_error, 0.0)
        self.assertEqual(profile.points[1].logit_kl, 0.0)
        self.assertEqual(profile.metric("weight_mse"), (1.0 / 3.0, 0.0))
        self.assertEqual(profile.as_dict()["prefix"], [2, 4])

    def test_progressive_profile_covers_every_prefix(self) -> None:
        rng = np.random.default_rng(41)
        reference = rng.normal(size=(7, 11)).astype(np.float32)
        activations = rng.normal(size=(5, 11)).astype(np.float32)
        quantized = quantize_progressive(
            reference, tile_shape=(8, 16), group_size=4
        )

        profile = profile_progressive_quantization(
            reference,
            quantized,
            activations,
            include_zero_prefix=True,
            include_output_kl=True,
        )

        self.assertEqual(profile.prefixes, (0, 1, 2, 3, 4))
        self.assertTrue(all(point.logit_kl is not None for point in profile.points))
        self.assertTrue(
            all(
                current <= previous + 1e-12
                for previous, current in zip(
                    profile.metric("weight_mse"), profile.metric("weight_mse")[1:]
                )
            )
        )
        for point in profile.points:
            self.assertAlmostEqual(
                point.activation_mse, point.hessian_error, places=11
            )

    def test_profile_dataclasses_validate_order_and_values(self) -> None:
        first = DistortionPoint(1, 1.0, 2.0, 2.0)
        second = DistortionPoint(2, 0.5, 1.0, 1.0)
        self.assertEqual(DistortionProfile((first, second)).prefixes, (1, 2))
        with self.assertRaises(ValueError):
            DistortionProfile((second, first))
        with self.assertRaises(ValueError):
            DistortionPoint(1, -1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            build_distortion_profile(
                np.eye(2),
                (np.eye(2),),
                np.eye(2),
                reference_logits=np.eye(2),
            )


if __name__ == "__main__":
    unittest.main()
