from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harpq.cli import main


class CliTests(unittest.TestCase):
    def test_doctor_json(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["doctor", "--json"])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(report["reference_tests_supported"])
        self.assertFalse(report["nvidia_performance_tests_supported"])
        self.assertIn(
            "correctness-first fused GEMV", report["nvidia_performance_tests_reason"]
        )
        self.assertIn("static-W4", report["nvidia_performance_tests_reason"])

    def test_demo_writes_functional_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["demo", "--output", directory, "--seed", "11"])
            summary = json.loads(
                (Path(directory) / "demo-summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result, 0)
            self.assertTrue(summary["all_prefixes_finite"])
            self.assertEqual(summary["stage_bits"], [2, 2, 2, 2])
            self.assertIn("not an NVIDIA performance result", summary["purpose"])

    def test_quantize_and_inspect_npy_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(41)
            weights_path = root / "weights.npy"
            activations_path = root / "activations.npy"
            np.save(weights_path, rng.standard_normal((9, 7), dtype=np.float32))
            np.save(activations_path, rng.standard_normal((12, 7), dtype=np.float32))
            container = root / "linear.harpq"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        "quantize",
                        "--weights",
                        str(weights_path),
                        "--activations",
                        str(activations_path),
                        "--output",
                        str(container),
                        "--tile-rows",
                        "8",
                        "--tile-columns",
                        "8",
                        "--group-size",
                        "4",
                    ]
                )
            report = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(report["manifest"]["packet_count"], 4)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["inspect", str(container), "--verify-payloads"])
            inspection = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(inspection["payload_crc32c"], "verified")

            trace = root / "trace.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        "simulate",
                        str(container),
                        "--capacity-bytes",
                        "100000",
                        "--tokens",
                        "2",
                        "--output",
                        str(trace),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue(trace.exists())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main(["replay", str(trace)])
            replay = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertTrue(replay["verified"])
            self.assertEqual(replay["events"], 2)


if __name__ == "__main__":
    unittest.main()
