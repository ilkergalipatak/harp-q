from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harpq.hardware import (
    _parse_fio_output,
    basic_profile,
    collect_active_linux_report,
    collect_hardware_report,
    measure_ram_copy_bandwidth,
    write_hardware_report,
)


class HardwareProfileTests(unittest.TestCase):
    def test_basic_profile_has_nonnegative_capabilities(self) -> None:
        profile = basic_profile()
        self.assertTrue(profile.platform)
        self.assertTrue(profile.machine)
        self.assertTrue(profile.python)
        self.assertTrue(profile.cpu_count is None or profile.cpu_count > 0)

    def test_report_is_json_serializable_and_does_not_claim_measurements(self) -> None:
        report = collect_hardware_report()
        encoded = json.dumps(report, sort_keys=True)
        self.assertIn("profile", encoded)
        self.assertEqual(report["measurement_status"]["ram_bandwidth"], "not_run")
        self.assertEqual(report["measurement_status"]["nvme_bandwidth"], "not_run")

    def test_report_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_hardware_report(Path(directory) / "profile.json")
            decoded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(decoded["schema_version"], 1)

    def test_bounded_ram_measurement_is_finite(self) -> None:
        result = measure_ram_copy_bandwidth(buffer_bytes=1024 * 1024, repetitions=1)
        self.assertGreater(result["payload_gib_s"], 0.0)
        self.assertEqual(result["buffer_bytes"], 1024 * 1024)

    def test_active_linux_profile_refuses_non_linux_host(self) -> None:
        import platform

        if platform.system() != "Linux":
            with self.assertRaises(RuntimeError):
                collect_active_linux_report("/tmp")

    def test_fio_json_requires_two_successful_nonempty_jobs(self) -> None:
        self.assertEqual(_parse_fio_output([])["status"], "failed")
        self.assertEqual(_parse_fio_output({"jobs": []})["status"], "failed")
        valid = {
            "jobs": [
                {
                    "jobname": "sequential-write",
                    "error": 0,
                    "write": {"io_bytes": 256 * 1024 * 1024, "bw_bytes": 512},
                },
                {
                    "jobname": "sequential-read",
                    "error": 0,
                    "read": {"io_bytes": 256 * 1024 * 1024, "bw_bytes": 768},
                },
            ]
        }
        self.assertEqual(_parse_fio_output(valid)["status"], "ok")
        valid["jobs"][1]["error"] = 5
        self.assertEqual(_parse_fio_output(valid)["status"], "failed")

        valid["jobs"][1]["error"] = 0
        valid["jobs"][1]["read"]["bw_bytes"] = float("nan")
        self.assertEqual(_parse_fio_output(valid)["status"], "failed")
        valid["jobs"][1]["read"]["bw_bytes"] = float("inf")
        self.assertEqual(_parse_fio_output(valid)["status"], "failed")
        valid["jobs"][1]["read"]["io_bytes"] = True
        self.assertEqual(_parse_fio_output(valid)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
