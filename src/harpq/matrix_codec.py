"""Bridge progressive NumPy matrices to opaque HARP-Q container packets."""

from __future__ import annotations

import struct
from typing import Iterable, Mapping, Sequence

import numpy as np

from .container import Packet
from .quantization import ProgressiveQuantizedMatrix, ProgressiveStage


CODEC_NAME = "harpq-progressive-stage"
CODEC_VERSION = 1

_PAYLOAD_MAGIC = b"HQSTAGE\x00"
_PAYLOAD_HEADER = struct.Struct("<8sHHII")


class MatrixCodecError(ValueError):
    """Raised when progressive stage packets cannot form one valid matrix."""


def packet_id_for_stage(matrix_id: str, stage_index: int) -> str:
    if not isinstance(matrix_id, str) or not matrix_id:
        raise ValueError("matrix_id must be a non-empty string")
    if not isinstance(stage_index, int) or isinstance(stage_index, bool) or stage_index < 0:
        raise ValueError("stage_index must be a non-negative integer")
    return f"{matrix_id}:stage:{stage_index}"


def _stage_metadata(
    matrix_id: str,
    quantized: ProgressiveQuantizedMatrix,
    stage_index: int,
    user_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    stage = quantized.stages[stage_index]
    return {
        "codec": CODEC_NAME,
        "codec_version": CODEC_VERSION,
        "matrix_id": matrix_id,
        "stage_index": stage_index,
        "total_stage_count": quantized.stage_count,
        "original_shape": list(quantized.original_shape),
        "padded_shape": list(quantized.padded_shape),
        "tile_shape": list(quantized.tile_shape),
        "group_size": quantized.group_size,
        "scale_shape": list(stage.scales.shape),
        "scale_dtype": "float16-le",
        "code_bytes": int(stage.packed_codes.size),
        "user": dict(user_metadata or {}),
    }


def encode_stage_payload(stage: ProgressiveStage, stage_index: int) -> bytes:
    """Encode one stage with an endian-stable header and FP16 scale bytes."""

    if not isinstance(stage, ProgressiveStage):
        raise TypeError("stage must be a ProgressiveStage")
    if not isinstance(stage_index, int) or isinstance(stage_index, bool) or stage_index < 0:
        raise ValueError("stage_index must be a non-negative integer")
    codes = np.ascontiguousarray(stage.packed_codes, dtype=np.uint8).tobytes(order="C")
    scales = np.ascontiguousarray(stage.scales, dtype="<f2").tobytes(order="C")
    header = _PAYLOAD_HEADER.pack(
        _PAYLOAD_MAGIC,
        CODEC_VERSION,
        stage_index,
        len(codes),
        stage.scales.size,
    )
    return header + codes + scales


def matrix_to_packets(
    matrix_id: str,
    quantized: ProgressiveQuantizedMatrix,
    *,
    metadata: Mapping[str, object] | None = None,
) -> tuple[Packet, ...]:
    """Convert a progressive matrix into an executable base plus optional residuals."""

    if not isinstance(quantized, ProgressiveQuantizedMatrix):
        raise TypeError("quantized must be a ProgressiveQuantizedMatrix")
    packets: list[Packet] = []
    previous_id: str | None = None
    for stage_index, stage in enumerate(quantized.stages):
        packet_id = packet_id_for_stage(matrix_id, stage_index)
        packets.append(
            Packet(
                packet_id=packet_id,
                stage=stage_index,
                payload=encode_stage_payload(stage, stage_index),
                dependencies=() if previous_id is None else (previous_id,),
                optional=stage_index > 0,
                metadata=_stage_metadata(
                    matrix_id,
                    quantized,
                    stage_index,
                    metadata,
                ),
            )
        )
        previous_id = packet_id
    return tuple(packets)


def _integer_sequence(value: object, label: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MatrixCodecError(f"{label} must be a sequence")
    parsed = tuple(value)
    if len(parsed) != length or any(
        isinstance(item, bool) or not isinstance(item, int) for item in parsed
    ):
        raise MatrixCodecError(f"{label} must contain {length} integers")
    return parsed


def decode_stage_payload(
    payload: bytes,
    *,
    expected_stage: int,
    scale_shape: Sequence[int],
) -> ProgressiveStage:
    """Decode and validate one matrix-stage packet payload."""

    encoded = bytes(payload)
    if len(encoded) < _PAYLOAD_HEADER.size:
        raise MatrixCodecError("stage payload is truncated")
    magic, version, stage_index, code_bytes, scale_count = _PAYLOAD_HEADER.unpack_from(encoded)
    if magic != _PAYLOAD_MAGIC:
        raise MatrixCodecError("stage payload magic is invalid")
    if version != CODEC_VERSION:
        raise MatrixCodecError(f"unsupported stage codec version {version}")
    if stage_index != expected_stage:
        raise MatrixCodecError(
            f"payload stage {stage_index} does not match metadata stage {expected_stage}"
        )
    parsed_scale_shape = _integer_sequence(scale_shape, "scale_shape", 4)
    expected_scale_count = int(np.prod(parsed_scale_shape, dtype=np.int64))
    if scale_count != expected_scale_count:
        raise MatrixCodecError("scale count does not match metadata")
    expected_length = _PAYLOAD_HEADER.size + code_bytes + scale_count * 2
    if len(encoded) != expected_length:
        raise MatrixCodecError("stage payload length does not match its header")
    codes_start = _PAYLOAD_HEADER.size
    codes = np.frombuffer(encoded[codes_start : codes_start + code_bytes], dtype=np.uint8).copy()
    scales = np.frombuffer(encoded[codes_start + code_bytes :], dtype="<f2").copy()
    scales = scales.astype(np.float16, copy=False).reshape(parsed_scale_shape)
    return ProgressiveStage(packed_codes=codes, scales=scales)


def packets_to_matrix(packets: Iterable[Packet]) -> ProgressiveQuantizedMatrix:
    """Reconstruct the longest contiguous prefix represented by *packets*."""

    packet_list = sorted(tuple(packets), key=lambda packet: packet.stage)
    if not packet_list:
        raise MatrixCodecError("at least the base packet is required")
    if any(packet.stage < 0 for packet in packet_list):
        raise MatrixCodecError("boot packets cannot be decoded as matrix stages")
    stages_seen = tuple(packet.stage for packet in packet_list)
    if stages_seen != tuple(range(len(packet_list))):
        raise MatrixCodecError(f"packets are not a contiguous prefix: {stages_seen}")

    base = packet_list[0].metadata
    if base.get("codec") != CODEC_NAME or base.get("codec_version") != CODEC_VERSION:
        raise MatrixCodecError("packet does not use the supported progressive matrix codec")
    if base.get("scale_dtype") != "float16-le":
        raise MatrixCodecError("packet scale_dtype must be 'float16-le'")
    matrix_id = base.get("matrix_id")
    if not isinstance(matrix_id, str) or not matrix_id:
        raise MatrixCodecError("matrix_id metadata is invalid")
    original_shape = _integer_sequence(base.get("original_shape"), "original_shape", 2)
    padded_shape = _integer_sequence(base.get("padded_shape"), "padded_shape", 2)
    tile_shape = _integer_sequence(base.get("tile_shape"), "tile_shape", 2)
    group_size = base.get("group_size")
    total_stage_count = base.get("total_stage_count")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise MatrixCodecError("group_size metadata is invalid")
    if (
        isinstance(total_stage_count, bool)
        or not isinstance(total_stage_count, int)
        or not len(packet_list) <= total_stage_count <= 4
    ):
        raise MatrixCodecError("total_stage_count metadata is invalid")

    geometry_keys = (
        "codec",
        "codec_version",
        "matrix_id",
        "total_stage_count",
        "original_shape",
        "padded_shape",
        "tile_shape",
        "group_size",
        "scale_dtype",
        "code_bytes",
        "user",
    )
    decoded_stages: list[ProgressiveStage] = []
    for stage_index, packet in enumerate(packet_list):
        metadata = packet.metadata
        if packet.optional != (stage_index > 0):
            expected = "optional" if stage_index > 0 else "required"
            raise MatrixCodecError(f"stage {stage_index} must be {expected}")
        if metadata.get("stage_index") != stage_index or packet.packet_id != packet_id_for_stage(
            matrix_id, stage_index
        ):
            raise MatrixCodecError(f"stage {stage_index} identity metadata is invalid")
        for key in geometry_keys:
            if metadata.get(key) != base.get(key):
                raise MatrixCodecError(f"stage {stage_index} disagrees on metadata field {key}")
        expected_dependencies = () if stage_index == 0 else (
            packet_id_for_stage(matrix_id, stage_index - 1),
        )
        if packet.dependencies != expected_dependencies:
            raise MatrixCodecError(f"stage {stage_index} dependency chain is invalid")
        scale_shape = _integer_sequence(metadata.get("scale_shape"), "scale_shape", 4)
        decoded = decode_stage_payload(
            packet.payload,
            expected_stage=stage_index,
            scale_shape=scale_shape,
        )
        if decoded.packed_codes.size != metadata.get("code_bytes"):
            raise MatrixCodecError("decoded code bytes do not match metadata")
        decoded_stages.append(decoded)

    return ProgressiveQuantizedMatrix(
        original_shape=(original_shape[0], original_shape[1]),
        padded_shape=(padded_shape[0], padded_shape[1]),
        tile_shape=(tile_shape[0], tile_shape[1]),
        group_size=group_size,
        stages=tuple(decoded_stages),
    )


__all__ = [
    "CODEC_NAME",
    "CODEC_VERSION",
    "MatrixCodecError",
    "decode_stage_payload",
    "encode_stage_payload",
    "matrix_to_packets",
    "packet_id_for_stage",
    "packets_to_matrix",
]
