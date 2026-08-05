"""Integrity primitives used by the portable HARP-Q container format.

The reference implementation deliberately keeps CRC32C in pure Python.  This makes
the on-disk format usable on a clean Mac or Linux Python installation and gives the
future C++ runtime an unambiguous test oracle.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import BinaryIO


class IntegrityError(ValueError):
    """Base class for an integrity verification failure."""


class ChecksumMismatchError(IntegrityError):
    """Raised when bytes do not match their recorded checksum."""


_CRC32C_POLYNOMIAL = 0x82F63B78


def _make_crc32c_table() -> tuple[int, ...]:
    table: list[int] = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (_CRC32C_POLYNOMIAL if value & 1 else 0)
        table.append(value & 0xFFFFFFFF)
    return tuple(table)


_CRC32C_TABLE = _make_crc32c_table()


def crc32c(data: bytes | bytearray | memoryview, crc: int = 0) -> int:
    """Return the standard Castagnoli CRC-32C of *data*.

    ``crc`` accepts the result of an earlier call, so callers may checksum a
    stream incrementally.  The result is an unsigned 32-bit integer and matches
    the conventional ``CRC32C("123456789") == 0xe3069283`` test vector.
    """

    if not 0 <= crc <= 0xFFFFFFFF:
        raise ValueError("crc must be an unsigned 32-bit integer")
    value = crc ^ 0xFFFFFFFF
    for octet in memoryview(data).cast("B"):
        value = _CRC32C_TABLE[(value ^ octet) & 0xFF] ^ (value >> 8)
    return (value ^ 0xFFFFFFFF) & 0xFFFFFFFF


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return a lowercase hexadecimal SHA-256 digest."""

    return hashlib.sha256(data).hexdigest()


def _sha256_stream(stream: BinaryIO, chunk_size: int) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    with Path(path).open("rb") as stream:
        return _sha256_stream(stream, chunk_size)


def verify_crc32c(
    data: bytes | bytearray | memoryview,
    expected: int,
    *,
    label: str = "payload",
) -> None:
    """Raise :class:`ChecksumMismatchError` if *data* has the wrong CRC32C."""

    actual = crc32c(data)
    if actual != expected:
        raise ChecksumMismatchError(
            f"CRC32C mismatch for {label}: expected {expected:08x}, got {actual:08x}"
        )


def verify_sha256_file(path: str | Path, expected: str, *, label: str | None = None) -> None:
    """Raise :class:`ChecksumMismatchError` if a file has the wrong SHA-256."""

    if len(expected) != 64:
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
    try:
        bytes.fromhex(expected)
    except ValueError as error:
        raise ValueError("expected SHA-256 is not hexadecimal") from error

    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected.lower()):
        name = label if label is not None else str(path)
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {name}: expected {expected.lower()}, got {actual}"
        )


__all__ = [
    "ChecksumMismatchError",
    "IntegrityError",
    "crc32c",
    "sha256_bytes",
    "sha256_file",
    "verify_crc32c",
    "verify_sha256_file",
]
