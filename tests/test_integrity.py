from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from harpq.integrity import (
    ChecksumMismatchError,
    crc32c,
    sha256_bytes,
    sha256_file,
    verify_crc32c,
    verify_sha256_file,
)


class IntegrityTests(unittest.TestCase):
    def test_crc32c_standard_vector_and_incremental_update(self) -> None:
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        prefix = crc32c(b"1234")
        self.assertEqual(crc32c(b"56789", prefix), crc32c(b"123456789"))

    def test_sha256_bytes_and_file_match_hashlib(self) -> None:
        payload = (b"harpq\x00" * 1000) + b"tail"
        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(sha256_bytes(payload), expected)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.pack"
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path, chunk_size=31), expected)
            verify_sha256_file(path, expected)

    def test_verifiers_report_corruption(self) -> None:
        payload = b"correct packet"
        with self.assertRaisesRegex(ChecksumMismatchError, "CRC32C mismatch"):
            verify_crc32c(payload + b"!", crc32c(payload), label="p0")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.pack"
            path.write_bytes(payload)
            with self.assertRaisesRegex(ChecksumMismatchError, "SHA-256 mismatch"):
                verify_sha256_file(path, "0" * 64, label="pack")

    def test_invalid_checksum_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            crc32c(b"x", -1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload"
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                verify_sha256_file(path, "not-a-digest")


if __name__ == "__main__":
    unittest.main()
