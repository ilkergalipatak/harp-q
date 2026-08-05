"""A tiny deterministic Transformer used for prefix-correctness tests.

This is intentionally a correctness fixture, not a language model benchmark. It keeps embeddings
and normalization in floating point while replacing every dense projection with HARP-Q matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .quantization import ProgressiveQuantizedMatrix, progressive_matmul, quantize_progressive

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    vocab_size: int = 32
    model_width: int = 16
    head_count: int = 4
    mlp_width: int = 32
    layer_count: int = 1
    max_sequence: int = 16
    seed: int = 7

    def __post_init__(self) -> None:
        for field_name in (
            "vocab_size",
            "model_width",
            "head_count",
            "mlp_width",
            "layer_count",
            "max_sequence",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.model_width % self.head_count != 0:
            raise ValueError("model_width must be divisible by head_count")


@dataclass(frozen=True, slots=True)
class DenseBlock:
    query: FloatArray
    key: FloatArray
    value: FloatArray
    output: FloatArray
    up: FloatArray
    gate: FloatArray
    down: FloatArray


@dataclass(frozen=True, slots=True)
class QuantizedBlock:
    query: ProgressiveQuantizedMatrix
    key: ProgressiveQuantizedMatrix
    value: ProgressiveQuantizedMatrix
    output: ProgressiveQuantizedMatrix
    up: ProgressiveQuantizedMatrix
    gate: ProgressiveQuantizedMatrix
    down: ProgressiveQuantizedMatrix


@dataclass(frozen=True, slots=True)
class SyntheticTransformer:
    config: SyntheticConfig
    token_embedding: FloatArray
    position_embedding: FloatArray
    blocks: tuple[DenseBlock, ...]
    lm_head: FloatArray

    @classmethod
    def random(cls, config: SyntheticConfig | None = None) -> "SyntheticTransformer":
        config = config or SyntheticConfig()
        rng = np.random.default_rng(config.seed)
        width = config.model_width

        def matrix(rows: int, columns: int) -> FloatArray:
            scale = np.float32(1.0 / np.sqrt(columns))
            return (rng.standard_normal((rows, columns), dtype=np.float32) * scale).astype(
                np.float32
            )

        blocks = tuple(
            DenseBlock(
                query=matrix(width, width),
                key=matrix(width, width),
                value=matrix(width, width),
                output=matrix(width, width),
                up=matrix(config.mlp_width, width),
                gate=matrix(config.mlp_width, width),
                down=matrix(width, config.mlp_width),
            )
            for _ in range(config.layer_count)
        )
        return cls(
            config=config,
            token_embedding=matrix(config.vocab_size, width),
            position_embedding=matrix(config.max_sequence, width),
            blocks=blocks,
            lm_head=matrix(config.vocab_size, width),
        )

    def quantize(
        self,
        *,
        tile_shape: tuple[int, int] = (128, 128),
        group_size: int = 128,
        stage_count: int = 4,
    ) -> "QuantizedSyntheticTransformer":
        def quantize(weight: FloatArray) -> ProgressiveQuantizedMatrix:
            return quantize_progressive(
                weight,
                tile_shape=tile_shape,
                group_size=group_size,
                stage_count=stage_count,
            )

        return QuantizedSyntheticTransformer(
            config=self.config,
            token_embedding=self.token_embedding.copy(),
            position_embedding=self.position_embedding.copy(),
            blocks=tuple(
                QuantizedBlock(
                    query=quantize(block.query),
                    key=quantize(block.key),
                    value=quantize(block.value),
                    output=quantize(block.output),
                    up=quantize(block.up),
                    gate=quantize(block.gate),
                    down=quantize(block.down),
                )
                for block in self.blocks
            ),
            lm_head=self.lm_head.copy(),
        )

    def forward(self, token_ids: IntArray) -> FloatArray:
        return _forward(
            self.config,
            self.token_embedding,
            self.position_embedding,
            self.blocks,
            self.lm_head,
            token_ids,
            lambda inputs, weight: np.matmul(inputs, weight.T, dtype=np.float32),
        )


@dataclass(frozen=True, slots=True)
class QuantizedSyntheticTransformer:
    config: SyntheticConfig
    token_embedding: FloatArray
    position_embedding: FloatArray
    blocks: tuple[QuantizedBlock, ...]
    lm_head: FloatArray

    @property
    def stage_count(self) -> int:
        return self.blocks[0].query.stage_count if self.blocks else 0

    @property
    def payload_bytes(self) -> int:
        return sum(
            matrix.payload_bytes
            for block in self.blocks
            for matrix in (
                block.query,
                block.key,
                block.value,
                block.output,
                block.up,
                block.gate,
                block.down,
            )
        )

    def forward(self, token_ids: IntArray, *, prefix: int | None = None) -> FloatArray:
        return _forward(
            self.config,
            self.token_embedding,
            self.position_embedding,
            self.blocks,
            self.lm_head,
            token_ids,
            lambda inputs, weight: progressive_matmul(inputs, weight, prefix=prefix),
        )


def _layer_norm(inputs: FloatArray, epsilon: float = 1e-5) -> FloatArray:
    mean = np.mean(inputs, axis=-1, keepdims=True, dtype=np.float32)
    variance = np.mean(np.square(inputs - mean), axis=-1, keepdims=True, dtype=np.float32)
    return ((inputs - mean) / np.sqrt(variance + epsilon)).astype(np.float32)


def _silu(inputs: FloatArray) -> FloatArray:
    clipped = np.clip(inputs, -30.0, 30.0)
    return (inputs / (np.float32(1.0) + np.exp(-clipped))).astype(np.float32)


def _softmax(inputs: FloatArray) -> FloatArray:
    shifted = inputs - np.max(inputs, axis=-1, keepdims=True)
    exponentials = np.exp(shifted).astype(np.float32)
    return (exponentials / np.sum(exponentials, axis=-1, keepdims=True)).astype(np.float32)


def _validate_token_ids(config: SyntheticConfig, token_ids: IntArray) -> IntArray:
    raw_tokens = np.asarray(token_ids)
    if not np.issubdtype(raw_tokens.dtype, np.integer) or np.issubdtype(
        raw_tokens.dtype, np.bool_
    ):
        raise TypeError("token_ids must contain integers")
    tokens = raw_tokens.astype(np.int64, copy=False)
    if tokens.ndim != 1:
        raise ValueError("token_ids must be a one-dimensional sequence")
    if not 0 < tokens.size <= config.max_sequence:
        raise ValueError("sequence length is outside the configured range")
    if np.any(tokens < 0) or np.any(tokens >= config.vocab_size):
        raise ValueError("token id is outside the vocabulary")
    return tokens


def _forward(
    config: SyntheticConfig,
    token_embedding: FloatArray,
    position_embedding: FloatArray,
    blocks: tuple[DenseBlock, ...] | tuple[QuantizedBlock, ...],
    lm_head: FloatArray,
    token_ids: IntArray,
    linear: Callable[[FloatArray, object], FloatArray],
) -> FloatArray:
    tokens = _validate_token_ids(config, token_ids)
    sequence = tokens.size
    hidden = (token_embedding[tokens] + position_embedding[:sequence]).astype(np.float32)
    head_width = config.model_width // config.head_count
    causal_mask = np.triu(np.full((sequence, sequence), -1e9, dtype=np.float32), k=1)

    for block in blocks:
        normalized = _layer_norm(hidden)
        query = linear(normalized, block.query)
        key = linear(normalized, block.key)
        value = linear(normalized, block.value)
        query = query.reshape(sequence, config.head_count, head_width).transpose(1, 0, 2)
        key = key.reshape(sequence, config.head_count, head_width).transpose(1, 0, 2)
        value = value.reshape(sequence, config.head_count, head_width).transpose(1, 0, 2)
        scores = np.matmul(query, key.transpose(0, 2, 1), dtype=np.float32)
        scores = scores / np.float32(np.sqrt(head_width)) + causal_mask
        attended = np.matmul(_softmax(scores), value, dtype=np.float32)
        attended = attended.transpose(1, 0, 2).reshape(sequence, config.model_width)
        hidden = hidden + linear(attended, block.output)

        normalized = _layer_norm(hidden)
        gated = _silu(linear(normalized, block.gate)) * linear(normalized, block.up)
        hidden = hidden + linear(gated.astype(np.float32), block.down)

    return np.matmul(_layer_norm(hidden), lm_head.T, dtype=np.float32).astype(np.float32)


def prefix_error_curve(
    dense: SyntheticTransformer,
    quantized: QuantizedSyntheticTransformer,
    token_ids: IntArray,
) -> tuple[float, ...]:
    """Return logits MSE for every executable progressive prefix."""

    reference = dense.forward(token_ids)
    return tuple(
        float(np.mean(np.square(quantized.forward(token_ids, prefix=prefix) - reference)))
        for prefix in range(1, quantized.stage_count + 1)
    )
