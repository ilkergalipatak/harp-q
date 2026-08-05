"""Versioned, integrity-checked HARP-Q packet container.

The format is intentionally small and language-neutral:

* ``manifest.json`` describes immutable pack files and hashes;
* ``index.bin`` is a length-prefixed binary index;
* ``boot.pack`` always exists;
* stage ``0`` is stored in ``base.pack`` and later stages in
  ``refine-<stage>.pack``;
* every packet payload begins at a 4096-byte aligned offset.

``Packet`` is the neutral hand-off boundary to the quantizer.  A payload can be
packed quantized weights today and a future CUDA-specific representation later;
the container does not interpret it.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .integrity import crc32c, sha256_file, verify_crc32c, verify_sha256_file


FORMAT_NAME = "harpq-container"
FORMAT_VERSION = 1
INDEX_VERSION = 1
PACKET_ALIGNMENT = 4096

_INDEX_MAGIC = b"HARPQIDX"
_INDEX_HEADER = struct.Struct("<8sHHI")
_INDEX_LENGTH = struct.Struct("<I")
_MAX_INDEX_RECORD_BYTES = 16 * 1024 * 1024


class ContainerError(ValueError):
    """Base class for malformed or unusable HARP-Q containers."""


class ContainerFormatError(ContainerError):
    """Raised when serialized metadata violates the format contract."""


class DependencyError(ContainerError):
    """Raised when packet dependencies are missing, cyclic, or out of order."""


class MissingPackError(ContainerError):
    """Raised when a required pack file is absent."""


class PacketUnavailableError(ContainerError):
    """Raised when an optional packet is unavailable and fallback is disabled."""


def _normalise_json_object(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        # ``MappingProxyType`` is used on parsed records; convert the outer
        # mapping back to a plain JSON object before canonical validation.
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain only finite JSON values") from error
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by Mapping above
        raise TypeError(f"{label} must encode a JSON object")
    return decoded


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _align_up(value: int, alignment: int = PACKET_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def pack_name_for_stage(stage: int) -> str:
    """Return the canonical pack filename for a logical packet stage."""

    if isinstance(stage, bool) or not isinstance(stage, int) or stage < -1:
        raise ValueError("stage must be an integer greater than or equal to -1")
    if stage == -1:
        return "boot.pack"
    if stage == 0:
        return "base.pack"
    return f"refine-{stage}.pack"


@dataclass(frozen=True, slots=True)
class Packet:
    """Opaque packet payload plus scheduling/decoding metadata.

    Stages use ``-1`` for boot data, ``0`` for executable base weights, and
    positive integers for progressive residuals.  Dependencies are packet IDs;
    residual dependencies must point strictly to earlier stages.
    """

    packet_id: str
    stage: int
    payload: bytes
    dependencies: tuple[str, ...] = ()
    optional: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("packet_id must be a non-empty string")
        if "\x00" in self.packet_id:
            raise ValueError("packet_id cannot contain a NUL character")
        pack_name_for_stage(self.stage)
        payload = bytes(self.payload)
        if not payload:
            raise ValueError("packet payload cannot be empty")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("dependencies must contain non-empty packet IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies cannot contain duplicates")
        if self.packet_id in dependencies:
            raise DependencyError(f"packet {self.packet_id!r} cannot depend on itself")
        if not isinstance(self.optional, bool):
            raise TypeError("optional must be a boolean")
        metadata = _normalise_json_object(self.metadata, label="packet metadata")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True, slots=True)
class PacketRecord:
    """A payload's immutable location and checksum from ``index.bin``."""

    packet_id: str
    stage: int
    pack_name: str
    offset: int
    length: int
    crc32c: int
    dependencies: tuple[str, ...]
    optional: bool
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "crc32c": self.crc32c,
            "dependencies": list(self.dependencies),
            "id": self.packet_id,
            "length": self.length,
            "metadata": dict(self.metadata),
            "offset": self.offset,
            "optional": self.optional,
            "pack": self.pack_name,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PacketRecord":
        expected = {
            "crc32c",
            "dependencies",
            "id",
            "length",
            "metadata",
            "offset",
            "optional",
            "pack",
            "stage",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ContainerFormatError(f"invalid index fields; missing={missing}, extra={extra}")
        try:
            packet_id = value["id"]
            stage = value["stage"]
            pack_name = value["pack"]
            offset = value["offset"]
            length = value["length"]
            checksum = value["crc32c"]
            dependencies = tuple(value["dependencies"])
            optional = value["optional"]
            metadata = _normalise_json_object(value["metadata"], label="packet metadata")
        except (KeyError, TypeError, ValueError) as error:
            raise ContainerFormatError("invalid packet index record") from error

        if not isinstance(packet_id, str) or not packet_id:
            raise ContainerFormatError("index packet ID must be a non-empty string")
        if isinstance(stage, bool) or not isinstance(stage, int) or stage < -1:
            raise ContainerFormatError(f"invalid stage for packet {packet_id!r}")
        if not isinstance(pack_name, str) or pack_name != pack_name_for_stage(stage):
            raise ContainerFormatError(f"non-canonical pack name for packet {packet_id!r}")
        for name, number in (("offset", offset), ("length", length), ("crc32c", checksum)):
            if isinstance(number, bool) or not isinstance(number, int):
                raise ContainerFormatError(f"{name} for packet {packet_id!r} must be an integer")
        if offset < 0 or length <= 0 or not 0 <= checksum <= 0xFFFFFFFF:
            raise ContainerFormatError(f"invalid extent or checksum for packet {packet_id!r}")
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ContainerFormatError(f"invalid dependency for packet {packet_id!r}")
        if len(set(dependencies)) != len(dependencies):
            raise ContainerFormatError(f"duplicate dependency for packet {packet_id!r}")
        if not isinstance(optional, bool):
            raise ContainerFormatError(f"optional for packet {packet_id!r} must be boolean")
        return cls(
            packet_id=packet_id,
            stage=stage,
            pack_name=pack_name,
            offset=offset,
            length=length,
            crc32c=checksum,
            dependencies=dependencies,
            optional=optional,
            metadata=MappingProxyType(metadata),
        )


@dataclass(frozen=True, slots=True)
class PackInfo:
    name: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class ContainerManifest:
    """Parsed contents of ``manifest.json``."""

    format_version: int
    alignment: int
    index_size: int
    index_sha256: str
    packet_count: int
    packs: tuple[PackInfo, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment,
            "format": FORMAT_NAME,
            "format_version": self.format_version,
            "index": {
                "name": "index.bin",
                "sha256": self.index_sha256,
                "size": self.index_size,
            },
            "metadata": dict(self.metadata),
            "packet_count": self.packet_count,
            "packs": [pack.to_dict() for pack in self.packs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContainerManifest":
        try:
            if value["format"] != FORMAT_NAME:
                raise ContainerFormatError(f"unsupported container format {value['format']!r}")
            version = value["format_version"]
            alignment = value["alignment"]
            packet_count = value["packet_count"]
            index = value["index"]
            packs_data = value["packs"]
            metadata = _normalise_json_object(value["metadata"], label="manifest metadata")
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ContainerFormatError):
                raise
            raise ContainerFormatError("manifest is missing a required field") from error
        if version != FORMAT_VERSION:
            raise ContainerFormatError(f"unsupported container version {version!r}")
        if alignment != PACKET_ALIGNMENT:
            raise ContainerFormatError(f"unsupported packet alignment {alignment!r}")
        if isinstance(packet_count, bool) or not isinstance(packet_count, int) or packet_count < 0:
            raise ContainerFormatError("packet_count must be a non-negative integer")
        if not isinstance(index, Mapping) or index.get("name") != "index.bin":
            raise ContainerFormatError("manifest index entry is invalid")
        index_size = index.get("size")
        index_sha256 = index.get("sha256")
        if isinstance(index_size, bool) or not isinstance(index_size, int) or index_size < 0:
            raise ContainerFormatError("manifest index size is invalid")
        if not _valid_sha256(index_sha256):
            raise ContainerFormatError("manifest index SHA-256 is invalid")
        if not isinstance(packs_data, list):
            raise ContainerFormatError("manifest packs must be a list")
        packs: list[PackInfo] = []
        names: set[str] = set()
        for raw_pack in packs_data:
            if not isinstance(raw_pack, Mapping):
                raise ContainerFormatError("manifest pack entry is invalid")
            name = raw_pack.get("name")
            size = raw_pack.get("size")
            digest = raw_pack.get("sha256")
            if not isinstance(name, str) or name in names:
                raise ContainerFormatError("manifest pack name is invalid or duplicated")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ContainerFormatError(f"manifest size for {name!r} is invalid")
            if not _valid_sha256(digest):
                raise ContainerFormatError(f"manifest SHA-256 for {name!r} is invalid")
            names.add(name)
            packs.append(PackInfo(name=name, size=size, sha256=digest.lower()))
        return cls(
            format_version=version,
            alignment=alignment,
            index_size=index_size,
            index_sha256=index_sha256.lower(),
            packet_count=packet_count,
            packs=tuple(packs),
            metadata=MappingProxyType(metadata),
        )


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _validate_dependency_graph(records: Sequence[PacketRecord]) -> None:
    by_id = {record.packet_id: record for record in records}
    if len(by_id) != len(records):
        raise DependencyError("packet IDs must be unique")
    for record in records:
        if record.stage == -1 and record.dependencies:
            raise DependencyError(f"boot packet {record.packet_id!r} cannot have dependencies")
        if record.stage > 0 and not record.dependencies:
            raise DependencyError(f"residual packet {record.packet_id!r} has no base dependency")
        for dependency_id in record.dependencies:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                raise DependencyError(
                    f"packet {record.packet_id!r} depends on missing packet {dependency_id!r}"
                )
            if dependency.stage >= record.stage:
                raise DependencyError(
                    f"packet {record.packet_id!r} stage {record.stage} depends on non-earlier "
                    f"packet {dependency_id!r} stage {dependency.stage}"
                )

    # Earlier-stage ordering already precludes cycles, but explicit DFS keeps the
    # failure understandable if a future format relaxes that rule.
    visiting: set[str] = set()
    closure_cache: dict[str, frozenset[int]] = {}

    def visit(packet_id: str) -> set[int]:
        if packet_id in visiting:
            raise DependencyError(f"dependency cycle includes packet {packet_id!r}")
        if packet_id in closure_cache:
            return set(closure_cache[packet_id])
        visiting.add(packet_id)
        record = by_id[packet_id]
        closure_stages: set[int] = set()
        for dependency_id in record.dependencies:
            dependency = by_id[dependency_id]
            closure_stages.add(dependency.stage)
            closure_stages.update(visit(dependency_id))
        visiting.remove(packet_id)
        if record.stage > 0:
            missing_stages = set(range(record.stage)) - closure_stages
            if missing_stages:
                raise DependencyError(
                    f"packet {packet_id!r} dependency closure misses stages "
                    f"{sorted(missing_stages)}"
                )
        closure_cache[packet_id] = frozenset(closure_stages)
        return closure_stages

    for packet_id in sorted(by_id):
        visit(packet_id)


def _encode_index(records: Sequence[PacketRecord]) -> bytes:
    pieces = [_INDEX_HEADER.pack(_INDEX_MAGIC, INDEX_VERSION, 0, len(records))]
    for record in records:
        encoded = _canonical_json_bytes(record.to_dict())
        if len(encoded) > _MAX_INDEX_RECORD_BYTES:
            raise ContainerFormatError(f"index record {record.packet_id!r} is too large")
        pieces.extend((_INDEX_LENGTH.pack(len(encoded)), encoded))
    return b"".join(pieces)


def _decode_index(data: bytes) -> tuple[PacketRecord, ...]:
    if len(data) < _INDEX_HEADER.size:
        raise ContainerFormatError("index is truncated")
    magic, version, flags, count = _INDEX_HEADER.unpack_from(data)
    if magic != _INDEX_MAGIC:
        raise ContainerFormatError("index magic is invalid")
    if version != INDEX_VERSION or flags != 0:
        raise ContainerFormatError(f"unsupported index version/flags {version}/{flags}")
    records: list[PacketRecord] = []
    cursor = _INDEX_HEADER.size
    for _ in range(count):
        if cursor + _INDEX_LENGTH.size > len(data):
            raise ContainerFormatError("index record length is truncated")
        (length,) = _INDEX_LENGTH.unpack_from(data, cursor)
        cursor += _INDEX_LENGTH.size
        if length > _MAX_INDEX_RECORD_BYTES or cursor + length > len(data):
            raise ContainerFormatError("index record is truncated or too large")
        try:
            raw = json.loads(data[cursor : cursor + length].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContainerFormatError("index record is not valid UTF-8 JSON") from error
        if not isinstance(raw, Mapping):
            raise ContainerFormatError("index record must be an object")
        records.append(PacketRecord.from_dict(raw))
        cursor += length
    if cursor != len(data):
        raise ContainerFormatError("index contains trailing bytes")
    return tuple(records)


def _records_from_packets(packets: Sequence[Packet]) -> list[PacketRecord]:
    packet_ids: set[str] = set()
    for packet in packets:
        if packet.packet_id in packet_ids:
            raise DependencyError(f"duplicate packet ID {packet.packet_id!r}")
        packet_ids.add(packet.packet_id)
    # Extents are filled while writing; placeholder values keep validation shared.
    return [
        PacketRecord(
            packet_id=packet.packet_id,
            stage=packet.stage,
            pack_name=pack_name_for_stage(packet.stage),
            offset=0,
            length=len(packet.payload),
            crc32c=crc32c(packet.payload),
            dependencies=packet.dependencies,
            optional=packet.optional,
            metadata=packet.metadata,
        )
        for packet in packets
    ]


def write_container(
    path: str | Path,
    packets: Iterable[Packet],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ContainerManifest:
    """Create a new immutable container directory and return its manifest.

    The destination must not already exist.  Files are first written to a sibling
    temporary directory and then renamed, so a failed conversion cannot look like
    a valid model.
    """

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"container destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    packet_list = sorted(tuple(packets), key=lambda item: (item.stage, item.packet_id))
    manifest_metadata = _normalise_json_object(metadata or {}, label="manifest metadata")
    provisional = _records_from_packets(packet_list)
    _validate_dependency_graph(provisional)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    completed = False
    try:
        records: list[PacketRecord] = []
        by_pack: dict[str, list[Packet]] = {"boot.pack": []}
        for packet in packet_list:
            by_pack.setdefault(pack_name_for_stage(packet.stage), []).append(packet)

        pack_infos: list[PackInfo] = []
        for pack_name in sorted(by_pack, key=_pack_sort_key):
            pack_path = temporary / pack_name
            with pack_path.open("wb") as stream:
                for packet in by_pack[pack_name]:
                    offset = _align_up(stream.tell())
                    if offset > stream.tell():
                        stream.write(b"\x00" * (offset - stream.tell()))
                    stream.write(packet.payload)
                    records.append(
                        PacketRecord(
                            packet_id=packet.packet_id,
                            stage=packet.stage,
                            pack_name=pack_name,
                            offset=offset,
                            length=len(packet.payload),
                            crc32c=crc32c(packet.payload),
                            dependencies=packet.dependencies,
                            optional=packet.optional,
                            metadata=packet.metadata,
                        )
                    )
            pack_infos.append(
                PackInfo(name=pack_name, size=pack_path.stat().st_size, sha256=sha256_file(pack_path))
            )

        records.sort(key=lambda item: (item.stage, item.packet_id))
        index_data = _encode_index(records)
        index_path = temporary / "index.bin"
        index_path.write_bytes(index_data)
        manifest = ContainerManifest(
            format_version=FORMAT_VERSION,
            alignment=PACKET_ALIGNMENT,
            index_size=len(index_data),
            index_sha256=sha256_file(index_path),
            packet_count=len(records),
            packs=tuple(pack_infos),
            metadata=MappingProxyType(manifest_metadata),
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        completed = True
        return manifest
    finally:
        if not completed and temporary.exists():
            shutil.rmtree(temporary)


def _pack_sort_key(name: str) -> tuple[int, str]:
    if name == "boot.pack":
        return (-1, name)
    if name == "base.pack":
        return (0, name)
    if name.startswith("refine-") and name.endswith(".pack"):
        try:
            return (int(name[7:-5]), name)
        except ValueError:
            pass
    return (2**31 - 1, name)


class ContainerReader:
    """Read-only view over an immutable HARP-Q container directory."""

    def __init__(self, path: str | Path, *, verify: bool = True) -> None:
        self.path = Path(path)
        self.manifest = self._read_manifest()
        index_path = self.path / "index.bin"
        try:
            index_size = index_path.stat().st_size
        except FileNotFoundError as error:
            raise ContainerFormatError("container is missing index.bin") from error
        if index_size != self.manifest.index_size:
            raise ContainerFormatError(
                f"index size mismatch: expected {self.manifest.index_size}, got {index_size}"
            )
        if verify:
            verify_sha256_file(index_path, self.manifest.index_sha256, label="index.bin")
        self.records = _decode_index(index_path.read_bytes())
        if len(self.records) != self.manifest.packet_count:
            raise ContainerFormatError("manifest packet count does not match index")
        _validate_dependency_graph(self.records)
        self._records_by_id = {record.packet_id: record for record in self.records}
        self._pack_info = {pack.name: pack for pack in self.manifest.packs}
        self._validate_extents_and_packs(verify=verify)
        self._unavailable = self._find_unavailable_packets()

    def _read_manifest(self) -> ContainerManifest:
        manifest_path = self.path / "manifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ContainerFormatError("container is missing manifest.json") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContainerFormatError("manifest.json is not valid UTF-8 JSON") from error
        if not isinstance(raw, Mapping):
            raise ContainerFormatError("manifest root must be an object")
        return ContainerManifest.from_dict(raw)

    def _validate_extents_and_packs(self, *, verify: bool) -> None:
        expected_pack_names = {"boot.pack"} | {record.pack_name for record in self.records}
        if set(self._pack_info) != expected_pack_names:
            raise ContainerFormatError(
                "manifest pack set does not match index: "
                f"manifest={sorted(self._pack_info)}, index={sorted(expected_pack_names)}"
            )
        records_by_pack: dict[str, list[PacketRecord]] = {
            pack_name: [] for pack_name in expected_pack_names
        }
        for record in self.records:
            if record.offset % self.manifest.alignment:
                raise ContainerFormatError(f"packet {record.packet_id!r} is not 4KiB aligned")
            records_by_pack[record.pack_name].append(record)

        for pack_name, pack_info in self._pack_info.items():
            pack_path = self.path / pack_name
            pack_records = sorted(records_by_pack[pack_name], key=lambda item: item.offset)
            if not pack_path.exists():
                if pack_name != "boot.pack" and pack_records and all(
                    record.optional for record in pack_records
                ):
                    continue
                raise MissingPackError(f"required pack is missing: {pack_name}")
            actual_size = pack_path.stat().st_size
            if actual_size != pack_info.size:
                raise ContainerFormatError(
                    f"pack size mismatch for {pack_name}: expected {pack_info.size}, got {actual_size}"
                )
            previous_end = 0
            for record in pack_records:
                if record.offset < previous_end or record.offset + record.length > actual_size:
                    raise ContainerFormatError(
                        f"packet {record.packet_id!r} has an overlapping or out-of-bounds extent"
                    )
                previous_end = record.offset + record.length
            if verify:
                verify_sha256_file(pack_path, pack_info.sha256, label=pack_name)

    def _find_unavailable_packets(self) -> frozenset[str]:
        unavailable = {
            record.packet_id
            for record in self.records
            if not (self.path / record.pack_name).exists()
        }
        changed = True
        while changed:
            changed = False
            for record in self.records:
                if record.packet_id in unavailable or not any(
                    dependency in unavailable for dependency in record.dependencies
                ):
                    continue
                if not record.optional:
                    raise MissingPackError(
                        f"required packet {record.packet_id!r} depends on an unavailable residual"
                    )
                unavailable.add(record.packet_id)
                changed = True
        return frozenset(unavailable)

    @property
    def packet_ids(self) -> tuple[str, ...]:
        return tuple(record.packet_id for record in self.records)

    @property
    def available_packet_ids(self) -> tuple[str, ...]:
        return tuple(
            record.packet_id for record in self.records if record.packet_id not in self._unavailable
        )

    @property
    def unavailable_packet_ids(self) -> tuple[str, ...]:
        return tuple(
            record.packet_id for record in self.records if record.packet_id in self._unavailable
        )

    def record(self, packet_id: str) -> PacketRecord:
        try:
            return self._records_by_id[packet_id]
        except KeyError as error:
            raise KeyError(f"unknown packet ID {packet_id!r}") from error

    def resolve_packet_id(self, packet_id: str) -> str | None:
        """Resolve an unavailable optional packet to its nearest available ancestor."""

        record = self.record(packet_id)
        if record.packet_id not in self._unavailable:
            return record.packet_id
        candidates: list[PacketRecord] = []
        visited: set[str] = set()

        def collect(current: PacketRecord) -> None:
            if current.packet_id in visited:
                return
            visited.add(current.packet_id)
            for dependency_id in current.dependencies:
                dependency = self._records_by_id[dependency_id]
                if dependency.packet_id not in self._unavailable:
                    candidates.append(dependency)
                else:
                    collect(dependency)

        collect(record)
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.stage, item.packet_id)).packet_id

    def read_packet(self, packet_id: str, *, fallback: bool = True) -> Packet | None:
        """Read and CRC-check one packet.

        If an optional stage pack is missing and ``fallback`` is true, the nearest
        available dependency is returned.  ``None`` is returned only when an
        unavailable optional packet has no available ancestor.
        """

        record = self.record(packet_id)
        if record.packet_id in self._unavailable:
            if not fallback:
                raise PacketUnavailableError(f"optional packet is unavailable: {packet_id}")
            resolved = self.resolve_packet_id(packet_id)
            if resolved is None:
                return None
            record = self._records_by_id[resolved]
        pack_path = self.path / record.pack_name
        with pack_path.open("rb") as stream:
            stream.seek(record.offset)
            payload = stream.read(record.length)
        if len(payload) != record.length:
            raise ContainerFormatError(f"payload for packet {record.packet_id!r} is truncated")
        verify_crc32c(payload, record.crc32c, label=f"packet {record.packet_id}")
        return Packet(
            packet_id=record.packet_id,
            stage=record.stage,
            payload=payload,
            dependencies=record.dependencies,
            optional=record.optional,
            metadata=record.metadata,
        )

    def read_chain(self, packet_id: str) -> tuple[Packet, ...]:
        """Read every available dependency followed by the requested prefix."""

        root = self.record(packet_id)
        output: list[Packet] = []
        visited: set[str] = set()

        def append_record(record: PacketRecord) -> None:
            if record.packet_id in visited:
                return
            visited.add(record.packet_id)
            for dependency_id in sorted(
                record.dependencies,
                key=lambda item: (self._records_by_id[item].stage, item),
            ):
                append_record(self._records_by_id[dependency_id])
            if record.packet_id not in self._unavailable:
                packet = self.read_packet(record.packet_id, fallback=False)
                if packet is not None:  # pragma: no branch - fallback=False and available
                    output.append(packet)

        append_record(root)
        return tuple(output)

    def verify_all_payloads(self) -> None:
        """Read every available packet and verify its per-payload CRC32C."""

        for packet_id in self.available_packet_ids:
            self.read_packet(packet_id, fallback=False)

    def __enter__(self) -> "ContainerReader":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self) -> Iterator[PacketRecord]:
        return iter(self.records)


def open_container(path: str | Path, *, verify: bool = True) -> ContainerReader:
    """Open a container using the read-only reference implementation."""

    return ContainerReader(path, verify=verify)


__all__ = [
    "ContainerError",
    "ContainerFormatError",
    "ContainerManifest",
    "ContainerReader",
    "DependencyError",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MissingPackError",
    "PACKET_ALIGNMENT",
    "PackInfo",
    "Packet",
    "PacketRecord",
    "PacketUnavailableError",
    "open_container",
    "pack_name_for_stage",
    "write_container",
]
