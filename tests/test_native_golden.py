from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class NativeGoldenDriftTests(unittest.TestCase):
    def test_checked_cpp_header_matches_python_reference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "generate_native_golden.py"), "--check"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
