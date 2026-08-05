from __future__ import annotations

import unittest

import numpy as np

from harpq.synthetic import SyntheticConfig, SyntheticTransformer, prefix_error_curve


class SyntheticTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dense = SyntheticTransformer.random(
            SyntheticConfig(
                vocab_size=16,
                model_width=8,
                head_count=2,
                mlp_width=12,
                layer_count=1,
                max_sequence=8,
                seed=19,
            )
        )
        cls.quantized = cls.dense.quantize(tile_shape=(8, 8), group_size=4)
        cls.tokens = np.array([1, 4, 2, 7, 3], dtype=np.int64)

    def test_all_prefixes_are_finite_and_deterministic(self) -> None:
        for prefix in range(1, self.quantized.stage_count + 1):
            first = self.quantized.forward(self.tokens, prefix=prefix)
            second = self.quantized.forward(self.tokens, prefix=prefix)
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.all(np.isfinite(first)))

    def test_last_prefix_improves_over_base_prefix(self) -> None:
        errors = prefix_error_curve(self.dense, self.quantized, self.tokens)
        self.assertLess(errors[-1], errors[0])

    def test_invalid_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.quantized.forward(np.array([999], dtype=np.int64), prefix=1)
        with self.assertRaises(TypeError):
            self.quantized.forward(np.array([1.9], dtype=np.float32), prefix=1)

    def test_config_validates_integer_geometry_before_division(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticConfig(head_count=0)
        with self.assertRaises(TypeError):
            SyntheticConfig(model_width=8.0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
