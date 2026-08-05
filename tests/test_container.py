from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harpq.container import (
    ContainerFormatError,
    DependencyError,
    MissingPackError,
    PACKET_ALIGNMENT,
    Packet,
    PacketUnavailableError,
    open_container,
    write_container,
)
from harpq.integrity import ChecksumMismatchError


def progressive_packets() -> tuple[Packet, ...]:
    return (
        Packet(
            "tile.0.base",
            0,
            b"base-weights",
            metadata={"shape": [128, 128], "tensor": "layers.0.w", "unicode": "ölçek"},
        ),
        Packet(
            "tile.0.r1",
            1,
            b"first-residual",
            dependencies=("tile.0.base",),
            optional=True,
            metadata={"gain": 0.25, "stage_scale": [0.5]},
        ),
        Packet(
            "tile.0.r2",
            2,
            b"second-residual",
            dependencies=("tile.0.r1",),
            optional=True,
            metadata={"gain": 0.125},
        ),
        Packet("tile.1.base", 0, b"another-base", metadata={"tile": [0, 1]}),
    )


class ContainerTests(unittest.TestCase):
    def test_roundtrip_alignment_metadata_and_pack_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            manifest = write_container(
                destination,
                progressive_packets(),
                metadata={"model": "synthetic", "nested": {"revision": 1}},
            )
            self.assertTrue((destination / "manifest.json").is_file())
            self.assertTrue((destination / "index.bin").is_file())
            self.assertTrue((destination / "boot.pack").is_file())
            self.assertTrue((destination / "base.pack").is_file())
            self.assertTrue((destination / "refine-1.pack").is_file())
            self.assertTrue((destination / "refine-2.pack").is_file())
            self.assertEqual(manifest.packet_count, 4)

            reader = open_container(destination)
            self.assertEqual(
                dict(reader.manifest.metadata),
                {"model": "synthetic", "nested": {"revision": 1}},
            )
            for record in reader.records:
                self.assertEqual(record.offset % PACKET_ALIGNMENT, 0)
            base_offsets = [
                record.offset for record in reader.records if record.pack_name == "base.pack"
            ]
            self.assertEqual(base_offsets, [0, PACKET_ALIGNMENT])
            r1 = reader.read_packet("tile.0.r1")
            self.assertIsNotNone(r1)
            assert r1 is not None
            self.assertEqual(r1.payload, b"first-residual")
            self.assertEqual(dict(r1.metadata), {"gain": 0.25, "stage_scale": [0.5]})
            self.assertEqual(
                [packet.packet_id for packet in reader.read_chain("tile.0.r2")],
                ["tile.0.base", "tile.0.r1", "tile.0.r2"],
            )
            reader.verify_all_payloads()

            serialized_manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(serialized_manifest, manifest.to_dict())

    def test_missing_optional_pack_falls_back_to_available_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            write_container(destination, progressive_packets())
            (destination / "refine-1.pack").unlink()
            reader = open_container(destination)
            self.assertEqual(
                reader.unavailable_packet_ids,
                ("tile.0.r1", "tile.0.r2"),
            )
            fallback = reader.read_packet("tile.0.r2")
            self.assertIsNotNone(fallback)
            assert fallback is not None
            self.assertEqual(fallback.packet_id, "tile.0.base")
            with self.assertRaises(PacketUnavailableError):
                reader.read_packet("tile.0.r1", fallback=False)
            self.assertEqual(
                [packet.packet_id for packet in reader.read_chain("tile.0.r2")],
                ["tile.0.base"],
            )

    def test_missing_required_pack_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            write_container(destination, progressive_packets())
            (destination / "base.pack").unlink()
            with self.assertRaises(MissingPackError):
                open_container(destination)

    def test_pack_sha_and_payload_crc_detect_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            write_container(destination, progressive_packets())
            pack_path = destination / "base.pack"
            with pack_path.open("r+b") as stream:
                stream.seek(1)
                original = stream.read(1)
                stream.seek(1)
                stream.write(bytes([original[0] ^ 0xFF]))
            with self.assertRaises(ChecksumMismatchError):
                open_container(destination)
            reader = open_container(destination, verify=False)
            with self.assertRaises(ChecksumMismatchError):
                reader.read_packet("tile.0.base")

    def test_index_corruption_is_detected_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            write_container(destination, progressive_packets())
            index_path = destination / "index.bin"
            data = bytearray(index_path.read_bytes())
            data[-1] ^= 1
            index_path.write_bytes(data)
            with self.assertRaises(ChecksumMismatchError):
                open_container(destination)
            with self.assertRaises(ContainerFormatError):
                open_container(destination, verify=False)

    def test_invalid_dependencies_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Packet("bad-optional", 0, b"payload", optional=1)
        missing = Packet("r1", 1, b"r", dependencies=("no-base",), optional=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(DependencyError):
                write_container(Path(temporary) / "missing", (missing,))

        base = Packet("base", 0, b"b")
        r1 = Packet("r1", 1, b"r1", dependencies=("base",), optional=True)
        incomplete_r2 = Packet("r2", 2, b"r2", dependencies=("base",), optional=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(DependencyError, "misses stages"):
                write_container(Path(temporary) / "incomplete", (base, r1, incomplete_r2))


if __name__ == "__main__":
    unittest.main()
