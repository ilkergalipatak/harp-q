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
from harpq.integrity import ChecksumMismatchError, sha256_file


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

    def test_pack_tails_are_zero_padded_and_every_direct_extent_fits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            write_container(destination, progressive_packets())
            reader = open_container(destination)
            records_by_pack = {
                pack.name: sorted(
                    (record for record in reader.records if record.pack_name == pack.name),
                    key=lambda record: record.offset,
                )
                for pack in reader.manifest.packs
            }

            for pack in reader.manifest.packs:
                pack_bytes = (destination / pack.name).read_bytes()
                self.assertEqual(len(pack_bytes), pack.size)
                self.assertEqual(pack.size % PACKET_ALIGNMENT, 0)
                records = records_by_pack[pack.name]
                if not records:
                    self.assertEqual(pack.name, "boot.pack")
                    self.assertEqual(pack_bytes, b"")
                    continue

                for record in records:
                    physical_length = (
                        (record.length + PACKET_ALIGNMENT - 1)
                        // PACKET_ALIGNMENT
                        * PACKET_ALIGNMENT
                    )
                    self.assertLessEqual(record.offset + physical_length, pack.size)

                last = records[-1]
                tail_start = last.offset + last.length
                self.assertEqual(pack_bytes[tail_start:], b"\x00" * (pack.size - tail_start))

    def test_padding_is_in_pack_sha_but_not_packet_crc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.harpq"
            write_container(destination, (Packet("base", 0, b"payload"),))
            reader = open_container(destination)
            record = reader.record("base")
            pack_path = destination / record.pack_name
            padding_offset = record.offset + record.length
            self.assertLess(padding_offset, pack_path.stat().st_size)

            with pack_path.open("r+b") as stream:
                stream.seek(padding_offset)
                stream.write(b"\x01")

            with self.assertRaises(ChecksumMismatchError):
                open_container(destination)
            unchecked_reader = open_container(destination, verify=False)
            packet = unchecked_reader.read_packet("base")
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.payload, b"payload")

    def test_legacy_unpadded_v1_pack_remains_buffered_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "legacy.harpq"
            write_container(destination, (Packet("base", 0, b"legacy-payload"),))
            reader = open_container(destination)
            record = reader.record("base")
            pack_path = destination / record.pack_name
            logical_end = record.offset + record.length
            with pack_path.open("r+b") as stream:
                stream.truncate(logical_end)

            manifest_path = destination / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for pack in manifest["packs"]:
                if pack["name"] == record.pack_name:
                    pack["size"] = logical_end
                    pack["sha256"] = sha256_file(pack_path)
                    break
            else:  # pragma: no cover - writer always declares the canonical pack
                self.fail(f"manifest is missing {record.pack_name}")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self.assertNotEqual(logical_end % PACKET_ALIGNMENT, 0)
            legacy_reader = open_container(destination)
            packet = legacy_reader.read_packet("base")
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.payload, b"legacy-payload")

    def test_container_output_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.harpq"
            second = root / "second.harpq"
            metadata = {"model": "synthetic", "revision": 7}
            write_container(first, progressive_packets(), metadata=metadata)
            write_container(second, progressive_packets(), metadata=metadata)

            first_files = {
                path.relative_to(first): path.read_bytes()
                for path in first.iterdir()
                if path.is_file()
            }
            second_files = {
                path.relative_to(second): path.read_bytes()
                for path in second.iterdir()
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

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
