from __future__ import annotations

import runpy
import unittest
from pathlib import Path


class TransferBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                       "scripts/benchmark_transfers.py"))

    def test_fio_is_read_only_bounded_and_has_explicit_depth(self) -> None:
        command = self.module["fio_command"](Path("/scratch/test.pack"), 65536, 8, 1048576)
        for flag in ("--readonly", "--rw=read", "--direct=1", "--ioengine=io_uring",
                     "--allow_file_create=0", "--iodepth=8", "--io_size=1048576"):
            self.assertIn(flag, command)
        self.assertNotIn("--time_based", command)

    def test_fio_metrics_reject_errors_wrong_volume_and_depth(self) -> None:
        job = {"error": 0, "read": {"io_bytes": 1048576, "bw_bytes": 1000000},
               "write": {"io_bytes": 0}, "iodepth_level": {"1": 100.0}}
        parse = self.module["fio_rate"]
        self.assertEqual(parse({"jobs": [job]}, 1048576, 1), 1000000)
        for change in ({"error": 5}, {"read": {"io_bytes": 1, "bw_bytes": 1}},
                       {"write": {"io_bytes": 1}}, {"iodepth_level": {"1": 0.0}}):
            with self.assertRaises(ValueError):
                parse({"jobs": [{**job, **change}]}, 1048576, 1)
        with self.assertRaises(ValueError):
            parse({"jobs": [job, job]}, 1048576, 1)
        self.assertEqual(parse({"jobs": [job]}, 1048576, 8, check_depth=False), 1000000)

    def test_summary_reports_median_range_not_best_run(self) -> None:
        summary = self.module["summarize"]([1e9, 3e9, 2e9])
        self.assertEqual(summary, {"median_gb_s": 2.0, "min_gb_s": 1.0,
                                   "max_gb_s": 3.0, "repeats": 3})
        for invalid in ([], [float("nan")], [0.0], [-1.0]):
            with self.assertRaises(ValueError):
                self.module["summarize"](invalid)

    def test_fixture_pattern_and_size_are_bounded(self) -> None:
        pattern = self.module["pattern_block"]()
        self.assertEqual(len(pattern), 1048576)
        self.assertEqual(pattern[:256], bytes(range(256)))
        self.assertEqual(pattern[-256:], bytes(range(256)))


if __name__ == "__main__":
    unittest.main()
