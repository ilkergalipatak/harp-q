"""NumPy reference implementation of HARP-Q progressive quantization.

The implementation in this module intentionally favours an explicit and
testable representation over speed.  A weight matrix is padded into tiles,
quantized group-wise along the input-feature axis, and represented as four
successive signed-int2 residual stages.  Every stage stores FP16 scales and
packed two-bit codes; no floating-point weight copy is retained.

Weights follow the usual linear-layer convention ``(out_features,
in_features)``.  Consequently :func:`progressive_matmul` computes
``inputs @ weights.T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


INT2_MIN = -2
INT2_MAX = 1
BITS_PER_STAGE = 2
DEFAULT_TILE_SHAPE = (128, 128)
DEFAULT_GROUP_SIZE = 128
DEFAULT_STAGE_COUNT = 4


def pack_2bit(codes: ArrayLike) -> NDArray[np.uint8]:
    """Pack signed two-bit values into bytes.

    Values use the two's-complement int2 alphabet ``[-2, -1, 0, 1]``.  Four
    consecutive values are stored in each byte; the first value occupies the
    least-significant two bits.  The final byte is zero-padded when necessary.
    """

    values = np.asarray(codes)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("2-bit codes must have an integer dtype")
    flat = values.reshape(-1)
    if np.any((flat < INT2_MIN) | (flat > INT2_MAX)):
        raise ValueError("2-bit codes must be in the inclusive range [-2, 1]")

    encoded = np.bitwise_and(flat.astype(np.int8, copy=False), 0x03).astype(
        np.uint8, copy=False
    )
    padded_count = (-encoded.size) % 4
    if padded_count:
        encoded = np.pad(encoded, (0, padded_count), mode="constant")
    groups = encoded.reshape(-1, 4)
    packed = (
        groups[:, 0]
        | (groups[:, 1] << np.uint8(2))
        | (groups[:, 2] << np.uint8(4))
        | (groups[:, 3] << np.uint8(6))
    )
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_2bit(packed: ArrayLike, count: int) -> NDArray[np.int8]:
    """Unpack *count* signed two-bit values from *packed*.

    The returned array is one-dimensional.  Extra padding bits in the final
    byte are ignored.
    """

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count cannot be negative")
    payload = np.asarray(packed)
    if payload.dtype != np.uint8:
        raise TypeError("packed payload must have dtype uint8")
    payload = payload.reshape(-1)
    required_bytes = ceil(count / 4)
    if payload.size < required_bytes:
        raise ValueError(
            f"packed payload contains {payload.size} bytes; {required_bytes} required"
        )
    if count == 0:
        return np.empty((0,), dtype=np.int8)

    shifts = np.array((0, 2, 4, 6), dtype=np.uint8)
    raw = np.bitwise_and(payload[:required_bytes, None] >> shifts[None, :], 0x03)
    raw = raw.reshape(-1)[:count].astype(np.int8, copy=False)
    # Sign-extend the two-bit two's-complement representation.
    signed = np.where(raw >= 2, raw - 4, raw).astype(np.int8, copy=False)
    return np.ascontiguousarray(signed)


@dataclass(frozen=True)
class ProgressiveStage:
    """One residual precision stage in a progressive weight representation."""

    packed_codes: NDArray[np.uint8]
    scales: NDArray[np.float16]

    def __post_init__(self) -> None:
        packed = np.ascontiguousarray(self.packed_codes, dtype=np.uint8)
        scales = np.ascontiguousarray(self.scales, dtype=np.float16)
        if packed.ndim != 1:
            raise ValueError("packed_codes must be one-dimensional")
        if scales.ndim != 4:
            raise ValueError(
                "scales must have shape (tile_rows, tile_cols, rows_per_tile, groups)"
            )
        if not np.all(np.isfinite(scales)):
            raise ValueError("stage scales must be finite")
        if np.any(scales < 0):
            raise ValueError("stage scales cannot be negative")
        packed.setflags(write=False)
        scales.setflags(write=False)
        object.__setattr__(self, "packed_codes", packed)
        object.__setattr__(self, "scales", scales)

    @property
    def payload_bytes(self) -> int:
        """Number of bytes occupied by codes and scales for this stage."""

        return int(self.packed_codes.nbytes + self.scales.nbytes)


@dataclass(frozen=True)
class ProgressiveQuantizedMatrix:
    """Tile-aligned progressive representation of a two-dimensional matrix."""

    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    group_size: int
    stages: tuple[ProgressiveStage, ...]

    def __post_init__(self) -> None:
        original_shape = _shape_pair(self.original_shape, "original_shape")
        padded_shape = _shape_pair(self.padded_shape, "padded_shape")
        tile_shape = _shape_pair(self.tile_shape, "tile_shape")
        if any(dimension <= 0 for dimension in original_shape):
            raise ValueError("original_shape dimensions must be positive")
        if any(dimension <= 0 for dimension in tile_shape):
            raise ValueError("tile_shape dimensions must be positive")
        if isinstance(self.group_size, bool) or not isinstance(
            self.group_size, (int, np.integer)
        ):
            raise TypeError("group_size must be an integer")
        group_size = int(self.group_size)
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if tile_shape[1] % group_size != 0:
            raise ValueError("tile width must be divisible by group_size")

        expected_padded = tuple(
            ceil(size / tile) * tile
            for size, tile in zip(original_shape, tile_shape, strict=True)
        )
        if padded_shape != expected_padded:
            raise ValueError(
                f"padded_shape must be {expected_padded} for the supplied geometry"
            )
        stages = tuple(self.stages)
        if not 1 <= len(stages) <= DEFAULT_STAGE_COUNT:
            raise ValueError("a progressive matrix must contain between one and four stages")

        tile_grid = (
            padded_shape[0] // tile_shape[0],
            padded_shape[1] // tile_shape[1],
        )
        expected_scales = (
            tile_grid[0],
            tile_grid[1],
            tile_shape[0],
            tile_shape[1] // group_size,
        )
        expected_code_bytes = ceil((padded_shape[0] * padded_shape[1]) / 4)
        for index, stage in enumerate(stages):
            if not isinstance(stage, ProgressiveStage):
                raise TypeError(f"stage {index} is not a ProgressiveStage")
            if stage.scales.shape != expected_scales:
                raise ValueError(
                    f"stage {index} scale shape is {stage.scales.shape}; "
                    f"expected {expected_scales}"
                )
            if stage.packed_codes.size != expected_code_bytes:
                raise ValueError(
                    f"stage {index} code payload has {stage.packed_codes.size} bytes; "
                    f"expected {expected_code_bytes}"
                )

        object.__setattr__(self, "original_shape", original_shape)
        object.__setattr__(self, "padded_shape", padded_shape)
        object.__setattr__(self, "tile_shape", tile_shape)
        object.__setattr__(self, "group_size", group_size)
        object.__setattr__(self, "stages", stages)

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def logical_weight_count(self) -> int:
        return int(self.original_shape[0] * self.original_shape[1])

    @property
    def payload_bytes(self) -> int:
        """Bytes used by packed codes and FP16 scales, excluding metadata."""

        return sum(stage.payload_bytes for stage in self.stages)

    def bits_for_prefix(self, prefix: int | None = None) -> int:
        """Return the nominal code bits per padded weight for *prefix*."""

        return BITS_PER_STAGE * _resolve_prefix(self, prefix)

    def dequantize(self, prefix: int | None = None) -> NDArray[np.float32]:
        return dequantize_prefix(self, prefix)

    def matmul(
        self,
        inputs: ArrayLike,
        prefix: int | None = None,
        bias: ArrayLike | None = None,
    ) -> NDArray[np.float32]:
        return progressive_matmul(inputs, self, prefix=prefix, bias=bias)


def quantize_progressive(
    weights: ArrayLike,
    *,
    tile_shape: Sequence[int] = DEFAULT_TILE_SHAPE,
    group_size: int = DEFAULT_GROUP_SIZE,
    stage_count: int = DEFAULT_STAGE_COUNT,
) -> ProgressiveQuantizedMatrix:
    """Quantize a matrix into signed-int2 residual stages.

    Scale fitting is deterministic least-squares refinement on the zero-point
    free signed-int2 grid.  Crucially, each residual is formed from the actual
    FP16 scale that a decoder will observe.  Prefix reconstructions therefore
    match the encoder exactly and their unweighted squared error cannot grow.
    """

    matrix = np.asarray(weights)
    if matrix.ndim != 2:
        raise ValueError("weights must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("weights cannot have an empty dimension")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("weights must have a numeric dtype")
    if np.iscomplexobj(matrix):
        raise TypeError("weights must contain real values")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weights must contain only finite values")

    parsed_tile_shape = _shape_pair(tile_shape, "tile_shape")
    if any(dimension <= 0 for dimension in parsed_tile_shape):
        raise ValueError("tile_shape dimensions must be positive")
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)):
        raise TypeError("group_size must be an integer")
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if parsed_tile_shape[1] % group_size != 0:
        raise ValueError("tile width must be divisible by group_size")
    if isinstance(stage_count, bool) or not isinstance(
        stage_count, (int, np.integer)
    ):
        raise TypeError("stage_count must be an integer")
    stage_count = int(stage_count)
    if not 1 <= stage_count <= DEFAULT_STAGE_COUNT:
        raise ValueError("stage_count must be between one and four")

    padded_shape = tuple(
        ceil(size / tile) * tile
        for size, tile in zip(matrix.shape, parsed_tile_shape, strict=True)
    )
    residual = np.zeros(padded_shape, dtype=np.float32)
    residual[: matrix.shape[0], : matrix.shape[1]] = matrix

    stages: list[ProgressiveStage] = []
    for _ in range(stage_count):
        grouped = _matrix_to_group_blocks(
            residual, parsed_tile_shape, group_size
        )
        codes, scales = _quantize_group_blocks(grouped)
        reconstructed_groups = codes.astype(np.float32) * scales[..., None].astype(
            np.float32
        )
        reconstructed = _group_blocks_to_matrix(
            reconstructed_groups, padded_shape, parsed_tile_shape
        )
        residual -= reconstructed

        code_matrix = _group_blocks_to_matrix(codes, padded_shape, parsed_tile_shape)
        stages.append(
            ProgressiveStage(
                packed_codes=pack_2bit(code_matrix),
                scales=scales,
            )
        )

    return ProgressiveQuantizedMatrix(
        original_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        padded_shape=(int(padded_shape[0]), int(padded_shape[1])),
        tile_shape=parsed_tile_shape,
        group_size=group_size,
        stages=tuple(stages),
    )


def dequantize_stage(
    quantized: ProgressiveQuantizedMatrix, stage_index: int
) -> NDArray[np.float32]:
    """Decode one zero-based residual stage to the original matrix shape."""

    if not isinstance(quantized, ProgressiveQuantizedMatrix):
        raise TypeError("quantized must be a ProgressiveQuantizedMatrix")
    if isinstance(stage_index, bool) or not isinstance(
        stage_index, (int, np.integer)
    ):
        raise TypeError("stage_index must be an integer")
    stage_index = int(stage_index)
    if not 0 <= stage_index < quantized.stage_count:
        raise ValueError(
            f"stage_index must be in [0, {quantized.stage_count - 1}]"
        )
    padded = _dequantize_stage_padded(quantized, stage_index)
    rows, columns = quantized.original_shape
    return np.ascontiguousarray(padded[:rows, :columns], dtype=np.float32)


def dequantize_prefix(
    quantized: ProgressiveQuantizedMatrix, prefix: int | None = None
) -> NDArray[np.float32]:
    """Decode the first *prefix* stages, or every stage when it is ``None``.

    A prefix of zero is valid and returns an all-zero matrix.  Prefixes 1..4
    correspond to nominal 2/4/6/8-bit representations.
    """

    if not isinstance(quantized, ProgressiveQuantizedMatrix):
        raise TypeError("quantized must be a ProgressiveQuantizedMatrix")
    prefix_count = _resolve_prefix(quantized, prefix)
    output = np.zeros(quantized.padded_shape, dtype=np.float32)
    for stage_index in range(prefix_count):
        output += _dequantize_stage_padded(quantized, stage_index)
    rows, columns = quantized.original_shape
    return np.ascontiguousarray(output[:rows, :columns], dtype=np.float32)


def progressive_matmul(
    inputs: ArrayLike,
    quantized: ProgressiveQuantizedMatrix,
    *,
    prefix: int | None = None,
    bias: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """Reference linear operation that accumulates available residual stages.

    Unlike a materialize-then-matmul implementation, this function multiplies
    each decoded residual stage independently and accumulates its output.  It
    therefore mirrors the arithmetic contract expected from a future fused
    progressive CUDA kernel.
    """

    if not isinstance(quantized, ProgressiveQuantizedMatrix):
        raise TypeError("quantized must be a ProgressiveQuantizedMatrix")
    prefix_count = _resolve_prefix(quantized, prefix)
    values = np.asarray(inputs)
    if values.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if values.shape[-1] != quantized.original_shape[1]:
        raise ValueError(
            f"input feature dimension is {values.shape[-1]}; "
            f"expected {quantized.original_shape[1]}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("inputs must have a numeric dtype")
    if np.iscomplexobj(values):
        raise TypeError("inputs must contain real values")
    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("inputs must contain only finite values")

    leading_shape = values.shape[:-1]
    flat_inputs = values.reshape(-1, values.shape[-1])
    output_features = quantized.original_shape[0]
    result = np.zeros((flat_inputs.shape[0], output_features), dtype=np.float32)
    for stage_index in range(prefix_count):
        stage_weights = dequantize_stage(quantized, stage_index)
        result += flat_inputs @ stage_weights.T

    if bias is not None:
        bias_array = np.asarray(bias)
        if not np.issubdtype(bias_array.dtype, np.number) or np.iscomplexobj(bias_array):
            raise TypeError("bias must contain real numeric values")
        bias_array = np.asarray(bias_array, dtype=np.float32)
        if bias_array.shape != (output_features,):
            raise ValueError(f"bias must have shape ({output_features},)")
        if not np.all(np.isfinite(bias_array)):
            raise ValueError("bias must contain only finite values")
        result += bias_array
    return np.ascontiguousarray(
        result.reshape(leading_shape + (output_features,)), dtype=np.float32
    )


def _shape_pair(value: Sequence[int], name: str) -> tuple[int, int]:
    try:
        parsed = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of two integers") from error
    if len(parsed) != 2:
        raise ValueError(f"{name} must contain exactly two dimensions")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in parsed):
        raise TypeError(f"{name} dimensions must be integers")
    return (int(parsed[0]), int(parsed[1]))


def _resolve_prefix(
    quantized: ProgressiveQuantizedMatrix, prefix: int | None
) -> int:
    if prefix is None:
        return quantized.stage_count
    if isinstance(prefix, bool) or not isinstance(prefix, (int, np.integer)):
        raise TypeError("prefix must be an integer or None")
    prefix = int(prefix)
    if not 0 <= prefix <= quantized.stage_count:
        raise ValueError(f"prefix must be between zero and {quantized.stage_count}")
    return prefix


def _matrix_to_group_blocks(
    matrix: NDArray[np.generic],
    tile_shape: tuple[int, int],
    group_size: int,
) -> NDArray[np.generic]:
    tile_rows, tile_columns = tile_shape
    grid_rows = matrix.shape[0] // tile_rows
    grid_columns = matrix.shape[1] // tile_columns
    tiled = matrix.reshape(grid_rows, tile_rows, grid_columns, tile_columns)
    tiled = tiled.transpose(0, 2, 1, 3)
    return tiled.reshape(
        grid_rows,
        grid_columns,
        tile_rows,
        tile_columns // group_size,
        group_size,
    )


def _group_blocks_to_matrix(
    groups: NDArray[np.generic],
    padded_shape: tuple[int, int],
    tile_shape: tuple[int, int],
) -> NDArray[np.generic]:
    tile_rows, tile_columns = tile_shape
    grid_rows = padded_shape[0] // tile_rows
    grid_columns = padded_shape[1] // tile_columns
    tiled = groups.reshape(grid_rows, grid_columns, tile_rows, tile_columns)
    matrix = tiled.transpose(0, 2, 1, 3).reshape(padded_shape)
    return np.ascontiguousarray(matrix)


def _quantize_group_blocks(
    groups: NDArray[np.float32],
) -> tuple[NDArray[np.int8], NDArray[np.float16]]:
    original_shape = groups.shape
    flattened = np.ascontiguousarray(groups.reshape(-1, original_shape[-1]), dtype=np.float32)
    max_abs = np.max(np.abs(flattened), axis=1)

    # 1.5 is the midpoint of the magnitudes represented by the two's-complement
    # int2 endpoints (-2 and +1).  Least-squares iterations below adapt this
    # deterministic starting point to the observed group.
    scales = np.where(max_abs > 0, max_abs / np.float32(1.5), 0).astype(np.float32)
    for _ in range(4):
        normalized = np.zeros_like(flattened)
        np.divide(
            flattened,
            scales[:, None],
            out=normalized,
            where=scales[:, None] != 0,
        )
        codes = np.clip(np.rint(normalized), INT2_MIN, INT2_MAX).astype(np.int8)
        numerator = np.sum(
            flattened.astype(np.float64) * codes.astype(np.float64), axis=1
        )
        denominator = np.sum(codes.astype(np.float64) ** 2, axis=1)
        fitted = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0,
        )
        scales = np.where(denominator != 0, fitted, scales).astype(np.float32)
        scales = np.maximum(scales, np.float32(0.0))

    fp16_max = np.finfo(np.float16).max
    if np.any(scales > fp16_max):
        raise OverflowError("a fitted group scale cannot be represented as FP16")
    stored_scales = scales.astype(np.float16)
    if not np.all(np.isfinite(stored_scales)):
        raise OverflowError("a fitted group scale cannot be represented as FP16")

    # Select codes again against the exact stored scale.  Since zero belongs to
    # the codebook and every element selects a nearest code, each residual stage
    # is guaranteed not to increase element-wise squared error.
    exact_scales = stored_scales.astype(np.float32)
    normalized = np.zeros_like(flattened)
    np.divide(
        flattened,
        exact_scales[:, None],
        out=normalized,
        where=exact_scales[:, None] != 0,
    )
    codes = np.clip(np.rint(normalized), INT2_MIN, INT2_MAX).astype(np.int8)
    return (
        np.ascontiguousarray(codes.reshape(original_shape), dtype=np.int8),
        np.ascontiguousarray(stored_scales.reshape(original_shape[:-1]), dtype=np.float16),
    )


def _dequantize_stage_padded(
    quantized: ProgressiveQuantizedMatrix, stage_index: int
) -> NDArray[np.float32]:
    stage = quantized.stages[stage_index]
    value_count = quantized.padded_shape[0] * quantized.padded_shape[1]
    code_matrix = unpack_2bit(stage.packed_codes, value_count).reshape(
        quantized.padded_shape
    )
    code_groups = _matrix_to_group_blocks(
        code_matrix, quantized.tile_shape, quantized.group_size
    )
    reconstructed_groups = code_groups.astype(np.float32) * stage.scales[
        ..., None
    ].astype(np.float32)
    return _group_blocks_to_matrix(
        reconstructed_groups, quantized.padded_shape, quantized.tile_shape
    ).astype(np.float32, copy=False)


__all__ = [
    "BITS_PER_STAGE",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_STAGE_COUNT",
    "DEFAULT_TILE_SHAPE",
    "INT2_MAX",
    "INT2_MIN",
    "ProgressiveQuantizedMatrix",
    "ProgressiveStage",
    "dequantize_prefix",
    "dequantize_stage",
    "pack_2bit",
    "progressive_matmul",
    "quantize_progressive",
    "unpack_2bit",
]
