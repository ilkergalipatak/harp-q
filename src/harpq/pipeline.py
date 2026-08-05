"""Composable offline calibration, packing, and allocation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike

from .allocator import AllocationItem, AllocationResult, allocate_packets
from .container import ContainerManifest, Packet, open_container, write_container
from .distortion import DistortionProfile, profile_progressive_quantization
from .matrix_codec import matrix_to_packets, packet_id_for_stage, packets_to_matrix
from .quantization import ProgressiveQuantizedMatrix, quantize_progressive


@dataclass(frozen=True, slots=True)
class CalibratedMatrix:
    """Offline result retained before a model is written to packs."""

    matrix_id: str
    quantized: ProgressiveQuantizedMatrix
    distortion: DistortionProfile


def calibrate_matrix(
    matrix_id: str,
    weights: ArrayLike,
    activations: ArrayLike,
    *,
    tile_shape: tuple[int, int] = (128, 128),
    group_size: int = 128,
    stage_count: int = 4,
    include_output_kl: bool = False,
) -> CalibratedMatrix:
    """Quantize one linear projection and measure every executable prefix."""

    if not isinstance(matrix_id, str) or not matrix_id:
        raise ValueError("matrix_id must be a non-empty string")
    quantized = quantize_progressive(
        weights,
        tile_shape=tile_shape,
        group_size=group_size,
        stage_count=stage_count,
    )
    distortion = profile_progressive_quantization(
        weights,
        quantized,
        activations,
        include_output_kl=include_output_kl,
    )
    return CalibratedMatrix(matrix_id, quantized, distortion)


def calibrated_to_packets(calibrated: CalibratedMatrix) -> tuple[Packet, ...]:
    return matrix_to_packets(
        calibrated.matrix_id,
        calibrated.quantized,
        metadata={"distortion": calibrated.distortion.as_dict()},
    )


def write_calibrated_container(
    path: str | Path,
    matrices: Iterable[CalibratedMatrix],
    *,
    metadata: Mapping[str, object] | None = None,
) -> ContainerManifest:
    """Write calibrated matrices into shared boot/base/refinement pack files."""

    matrix_list = tuple(matrices)
    if not matrix_list:
        raise ValueError("at least one calibrated matrix is required")
    matrix_ids = tuple(matrix.matrix_id for matrix in matrix_list)
    if len(set(matrix_ids)) != len(matrix_ids):
        raise ValueError("matrix IDs must be unique")
    stage_counts = {matrix.quantized.stage_count for matrix in matrix_list}
    if len(stage_counts) != 1:
        raise ValueError("all matrices in one container must use the same stage count")
    stage_count = next(iter(stage_counts))
    packets = tuple(packet for matrix in matrix_list for packet in calibrated_to_packets(matrix))
    caller_metadata = dict(metadata or {})
    reserved_keys = {"matrix_ids", "stage_bits"}
    collisions = reserved_keys & set(caller_metadata)
    if collisions:
        raise ValueError(f"metadata cannot replace reserved keys: {sorted(collisions)}")
    manifest_metadata = {
        **caller_metadata,
        "matrix_ids": list(sorted(matrix_ids)),
        "stage_bits": [2] * stage_count,
    }
    return write_container(path, packets, metadata=manifest_metadata)


def load_matrix_prefix(
    path: str | Path,
    matrix_id: str,
    *,
    prefix: int | None = None,
) -> ProgressiveQuantizedMatrix:
    """Load one available matrix prefix, falling back if an optional pack is absent."""

    with open_container(path) as reader:
        if prefix is None:
            matching = [
                record.stage
                for record in reader
                if record.metadata.get("matrix_id") == matrix_id and record.stage >= 0
            ]
            if not matching:
                raise KeyError(f"unknown matrix ID {matrix_id!r}")
            requested_stage = max(matching)
        else:
            if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix <= 0:
                raise ValueError("prefix must be a positive integer or None")
            requested_stage = prefix - 1
        chain = reader.read_chain(packet_id_for_stage(matrix_id, requested_stage))
    return packets_to_matrix(chain)


def allocation_items_from_profile(
    calibrated: CalibratedMatrix,
    *,
    metric: str = "activation_mse",
) -> tuple[AllocationItem, ...]:
    """Translate a measured distortion curve into precedence-aware knapsack items.

    The executable base is mandatory. A residual's gain is the non-negative reduction in the
    selected distortion metric from the preceding prefix.
    """

    values = calibrated.distortion.metric(metric)
    if len(values) != calibrated.quantized.stage_count:
        raise ValueError("distortion profile does not cover every quantization stage")
    if any(value is None for value in values):
        raise ValueError(f"distortion metric {metric!r} is unavailable")
    numeric = tuple(float(value) for value in values)
    packets = calibrated_to_packets(calibrated)
    items: list[AllocationItem] = []
    for stage_index, packet in enumerate(packets):
        gain = 0.0 if stage_index == 0 else max(0.0, numeric[stage_index - 1] - numeric[stage_index])
        items.append(
            AllocationItem(
                packet_id=packet.packet_id,
                group_id=calibrated.matrix_id,
                cost_bytes=len(packet.payload),
                distortion_gain=gain,
                dependencies=packet.dependencies,
                mandatory=stage_index == 0,
            )
        )
    return tuple(items)


def allocate_calibrated_matrices(
    matrices: Iterable[CalibratedMatrix],
    budget_bytes: int,
    *,
    metric: str = "activation_mse",
) -> AllocationResult:
    items = tuple(
        item
        for calibrated in matrices
        for item in allocation_items_from_profile(calibrated, metric=metric)
    )
    return allocate_packets(items, budget_bytes)


__all__ = [
    "CalibratedMatrix",
    "allocate_calibrated_matrices",
    "allocation_items_from_profile",
    "calibrate_matrix",
    "calibrated_to_packets",
    "load_matrix_prefix",
    "write_calibrated_container",
]
