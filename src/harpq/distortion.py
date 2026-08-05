"""Distortion measurements for progressive HARP-Q experiments.

All reductions use float64 accumulation.  The functions are deliberately
framework-independent so calibration and format tests can run on macOS before
the CUDA runtime exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def activation_weighted_mse(
    reference_weights: ArrayLike,
    approximate_weights: ArrayLike,
    activations: ArrayLike,
) -> float:
    """Return the output MSE induced by a weight approximation.

    For weights shaped ``(out_features, in_features)`` and calibration
    activations ``X``, this computes ``mean((X @ (W - W_hat).T) ** 2)``.  It
    includes the full empirical input covariance rather than only per-channel
    activation magnitudes.
    """

    reference, approximation = _weight_pair(reference_weights, approximate_weights)
    activation_matrix = _activation_matrix(activations, reference.shape[1])
    error = reference.astype(np.float64) - approximation.astype(np.float64)
    projected_error = activation_matrix @ error.T
    return float(np.mean(np.square(projected_error), dtype=np.float64))


def hessian_proxy(
    activations: ArrayLike,
    *,
    damping: float = 1e-6,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Build the empirical second-moment Hessian proxy ``X.T @ X``.

    With ``normalize=True`` (the default), the Gram matrix is divided by the
    number of activation samples.  ``damping * I`` is then added to make small
    calibration problems numerically well behaved.
    """

    values = np.asarray(activations)
    if values.ndim < 1:
        raise ValueError("activations must have at least one dimension")
    if values.shape[-1] == 0:
        raise ValueError("activations cannot have an empty feature dimension")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("activations must have a numeric dtype")
    if np.iscomplexobj(values):
        raise TypeError("activations must contain real values")
    matrix = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    if matrix.shape[0] == 0:
        raise ValueError("activations must contain at least one sample")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("activations must contain only finite values")
    damping_value = _finite_nonnegative_scalar(damping, "damping")

    result = matrix.T @ matrix
    if normalize:
        result /= matrix.shape[0]
    result = (result + result.T) * 0.5
    if damping_value:
        result.flat[:: result.shape[0] + 1] += damping_value
    return np.ascontiguousarray(result, dtype=np.float64)


def hessian_weighted_error(
    reference_weights: ArrayLike,
    approximate_weights: ArrayLike,
    hessian: ArrayLike,
    *,
    reduction: str = "mean",
    validate_psd: bool = True,
) -> float:
    """Measure ``trace((W-W_hat) H (W-W_hat).T)``.

    ``reduction='mean'`` divides by the number of output rows and therefore
    matches :func:`activation_weighted_mse` when ``H`` is the normalized,
    undamped Hessian proxy from the same activations. ``validate_psd=True`` performs an
    eigenvalue check; the profile builder validates an externally supplied Hessian once and skips
    repeated checks for its remaining prefix points.
    """

    reference, approximation = _weight_pair(reference_weights, approximate_weights)
    if not isinstance(validate_psd, bool):
        raise TypeError("validate_psd must be a boolean")
    hessian_matrix = _validated_hessian(
        hessian,
        reference.shape[1],
        validate_psd=validate_psd,
    )
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    error = reference.astype(np.float64) - approximation.astype(np.float64)
    terms = (error @ hessian_matrix) * error
    value = float(np.sum(terms, dtype=np.float64))
    negative_tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.sum(np.abs(terms), dtype=np.float64))
    ) * 64.0
    if value < -negative_tolerance:
        raise ValueError("hessian produced a negative quadratic form")
    if value < 0:
        value = 0.0
    if reduction == "mean":
        value /= reference.shape[0]
    return value


def stable_kl_divergence(
    reference_logits: ArrayLike,
    approximate_logits: ArrayLike,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> float | NDArray[np.float64]:
    """Compute stable ``KL(reference || approximate)`` over the last axis.

    Log-softmax is evaluated after max subtraction in float64, so very large
    finite logits remain safe.  For temperature scaling, the returned KL is
    the distributional divergence itself and is not multiplied by ``T**2``.
    """

    reference = np.asarray(reference_logits)
    approximation = np.asarray(approximate_logits)
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximate logits must have the same shape")
    if reference.ndim < 1 or reference.shape[-1] == 0:
        raise ValueError("logits must have a non-empty class dimension")
    if not np.issubdtype(reference.dtype, np.number) or not np.issubdtype(
        approximation.dtype, np.number
    ):
        raise TypeError("logits must have numeric dtypes")
    if np.iscomplexobj(reference) or np.iscomplexobj(approximation):
        raise TypeError("logits must contain real values")
    if reference.size == 0:
        raise ValueError("logits must contain at least one distribution")
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(approximation)):
        raise ValueError("logits must contain only finite values")
    temperature_value = _finite_positive_scalar(temperature, "temperature")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    reference_log_prob = _log_softmax(reference / temperature_value)
    approximate_log_prob = _log_softmax(approximation / temperature_value)
    reference_prob = np.exp(reference_log_prob)
    divergence = np.sum(
        reference_prob * (reference_log_prob - approximate_log_prob), axis=-1
    )
    # Floating-point cancellation can produce values a few ulps below zero.
    divergence = np.maximum(divergence, 0.0)
    if reduction == "none":
        return np.asarray(divergence, dtype=np.float64)
    if reduction == "sum":
        return float(np.sum(divergence, dtype=np.float64))
    return float(np.mean(divergence, dtype=np.float64))


@dataclass(frozen=True)
class DistortionPoint:
    """Distortion metrics for one progressive precision prefix."""

    prefix: int
    weight_mse: float
    activation_mse: float
    hessian_error: float
    logit_kl: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.prefix, bool) or not isinstance(
            self.prefix, (int, np.integer)
        ):
            raise TypeError("prefix must be an integer")
        prefix = int(self.prefix)
        if prefix < 0:
            raise ValueError("prefix cannot be negative")
        object.__setattr__(self, "prefix", prefix)
        for field_name in ("weight_mse", "activation_mse", "hessian_error"):
            value = _finite_nonnegative_scalar(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        if self.logit_kl is not None:
            object.__setattr__(
                self,
                "logit_kl",
                _finite_nonnegative_scalar(self.logit_kl, "logit_kl"),
            )


@dataclass(frozen=True)
class DistortionProfile:
    """Ordered collection of prefix distortion measurements."""

    points: tuple[DistortionPoint, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not points:
            raise ValueError("a distortion profile must contain at least one point")
        if not all(isinstance(point, DistortionPoint) for point in points):
            raise TypeError("all profile entries must be DistortionPoint instances")
        prefixes = tuple(point.prefix for point in points)
        if tuple(sorted(prefixes)) != prefixes or len(set(prefixes)) != len(prefixes):
            raise ValueError("profile prefixes must be unique and increasing")
        object.__setattr__(self, "points", points)

    @property
    def prefixes(self) -> tuple[int, ...]:
        return tuple(point.prefix for point in self.points)

    def metric(self, name: str) -> tuple[float | None, ...]:
        """Return one metric series by its dataclass field name."""

        if name not in {"weight_mse", "activation_mse", "hessian_error", "logit_kl"}:
            raise ValueError(f"unknown distortion metric: {name}")
        return tuple(getattr(point, name) for point in self.points)

    def as_dict(self) -> dict[str, list[float | int | None]]:
        """Return a JSON-friendly column-oriented representation."""

        return {
            "prefix": [point.prefix for point in self.points],
            "weight_mse": [point.weight_mse for point in self.points],
            "activation_mse": [point.activation_mse for point in self.points],
            "hessian_error": [point.hessian_error for point in self.points],
            "logit_kl": [point.logit_kl for point in self.points],
        }


def build_distortion_profile(
    reference_weights: ArrayLike,
    approximations: Sequence[ArrayLike],
    activations: ArrayLike,
    *,
    prefixes: Sequence[int] | None = None,
    hessian: ArrayLike | None = None,
    reference_logits: ArrayLike | None = None,
    approximate_logits: Sequence[ArrayLike] | None = None,
    temperature: float = 1.0,
) -> DistortionProfile:
    """Measure a sequence of weight approximations using common calibration data.

    Logit KL is optional.  If either logit argument is supplied, both must be
    supplied and ``approximate_logits`` must contain one array per approximation.
    """

    reference = _finite_weight_matrix(reference_weights, "reference_weights")
    candidates = tuple(approximations)
    if not candidates:
        raise ValueError("approximations must contain at least one matrix")
    candidate_matrices = tuple(
        _finite_weight_matrix(candidate, f"approximations[{index}]")
        for index, candidate in enumerate(candidates)
    )
    for index, candidate in enumerate(candidate_matrices):
        if candidate.shape != reference.shape:
            raise ValueError(
                f"approximations[{index}] has shape {candidate.shape}; "
                f"expected {reference.shape}"
            )
    activation_matrix = _activation_matrix(activations, reference.shape[1])

    if prefixes is None:
        parsed_prefixes = tuple(range(1, len(candidate_matrices) + 1))
    else:
        parsed_prefixes = tuple(prefixes)
        if len(parsed_prefixes) != len(candidate_matrices):
            raise ValueError("prefixes must contain one value per approximation")
        for prefix in parsed_prefixes:
            if isinstance(prefix, bool) or not isinstance(prefix, (int, np.integer)):
                raise TypeError("prefix values must be integers")
        parsed_prefixes = tuple(int(prefix) for prefix in parsed_prefixes)

    if hessian is None:
        hessian_matrix = hessian_proxy(
            activation_matrix, damping=0.0, normalize=True
        )
    else:
        hessian_matrix = _validated_hessian(
            hessian,
            reference.shape[1],
            validate_psd=True,
        )

    has_reference_logits = reference_logits is not None
    has_approximate_logits = approximate_logits is not None
    if has_reference_logits != has_approximate_logits:
        raise ValueError(
            "reference_logits and approximate_logits must either both be supplied or both omitted"
        )
    candidate_logits: tuple[ArrayLike, ...] | None = None
    if approximate_logits is not None:
        candidate_logits = tuple(approximate_logits)
        if len(candidate_logits) != len(candidate_matrices):
            raise ValueError("approximate_logits must contain one array per approximation")

    points: list[DistortionPoint] = []
    for index, (prefix, candidate) in enumerate(
        zip(parsed_prefixes, candidate_matrices, strict=True)
    ):
        error = reference.astype(np.float64) - candidate.astype(np.float64)
        logit_kl: float | None = None
        if candidate_logits is not None:
            logit_kl = float(
                stable_kl_divergence(
                    reference_logits,
                    candidate_logits[index],
                    temperature=temperature,
                    reduction="mean",
                )
            )
        points.append(
            DistortionPoint(
                prefix=prefix,
                weight_mse=float(np.mean(np.square(error), dtype=np.float64)),
                activation_mse=activation_weighted_mse(
                    reference, candidate, activation_matrix
                ),
                hessian_error=hessian_weighted_error(
                    reference,
                    candidate,
                    hessian_matrix,
                    reduction="mean",
                    validate_psd=False,
                ),
                logit_kl=logit_kl,
            )
        )
    return DistortionProfile(tuple(points))


def profile_progressive_quantization(
    reference_weights: ArrayLike,
    quantized: object,
    activations: ArrayLike,
    *,
    include_zero_prefix: bool = False,
    include_output_kl: bool = False,
    temperature: float = 1.0,
) -> DistortionProfile:
    """Convenience profile builder for a progressive quantized matrix.

    ``include_output_kl`` treats the linear-layer outputs as logits.  Callers
    should enable it only for an LM head or another genuine classifier output.
    The local import keeps the primitive distortion helpers independent of the
    quantization representation.
    """

    from .quantization import (  # Local import avoids an unnecessary cycle.
        ProgressiveQuantizedMatrix,
        dequantize_prefix,
    )

    if not isinstance(quantized, ProgressiveQuantizedMatrix):
        raise TypeError("quantized must be a ProgressiveQuantizedMatrix")
    reference = _finite_weight_matrix(reference_weights, "reference_weights")
    if reference.shape != quantized.original_shape:
        raise ValueError(
            f"reference_weights has shape {reference.shape}; "
            f"expected {quantized.original_shape}"
        )
    start = 0 if include_zero_prefix else 1
    prefixes = tuple(range(start, quantized.stage_count + 1))
    approximations = tuple(
        dequantize_prefix(quantized, prefix) for prefix in prefixes
    )

    reference_logits: NDArray[np.float64] | None = None
    candidate_logits: tuple[NDArray[np.float64], ...] | None = None
    if include_output_kl:
        activation_matrix = _activation_matrix(activations, reference.shape[1])
        reference_logits = activation_matrix @ reference.astype(np.float64).T
        candidate_logits = tuple(
            activation_matrix @ candidate.astype(np.float64).T
            for candidate in approximations
        )
    return build_distortion_profile(
        reference,
        approximations,
        activations,
        prefixes=prefixes,
        reference_logits=reference_logits,
        approximate_logits=candidate_logits,
        temperature=temperature,
    )


def _log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _finite_weight_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    if np.iscomplexobj(matrix):
        raise TypeError(f"{name} must contain real values")
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _weight_pair(
    reference_weights: ArrayLike, approximate_weights: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference = _finite_weight_matrix(reference_weights, "reference_weights")
    approximation = _finite_weight_matrix(approximate_weights, "approximate_weights")
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximate weights must have the same shape")
    return reference, approximation


def _activation_matrix(
    activations: ArrayLike, input_features: int
) -> NDArray[np.float64]:
    values = np.asarray(activations)
    if values.ndim < 1:
        raise ValueError("activations must have at least one dimension")
    if values.shape[-1] != input_features:
        raise ValueError(
            f"activation feature dimension is {values.shape[-1]}; expected {input_features}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("activations must have a numeric dtype")
    if np.iscomplexobj(values):
        raise TypeError("activations must contain real values")
    matrix = np.ascontiguousarray(values, dtype=np.float64).reshape(-1, input_features)
    if matrix.shape[0] == 0:
        raise ValueError("activations must contain at least one sample")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("activations must contain only finite values")
    return matrix


def _finite_nonnegative_scalar(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed


def _validated_hessian(
    value: ArrayLike,
    input_features: int,
    *,
    validate_psd: bool,
) -> NDArray[np.float64]:
    matrix = np.asarray(value)
    expected_shape = (input_features, input_features)
    if matrix.shape != expected_shape:
        raise ValueError(f"hessian must have shape {expected_shape}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("hessian must have a numeric dtype")
    if np.iscomplexobj(matrix):
        raise TypeError("hessian must contain real values")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hessian must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("hessian must be symmetric")
    matrix = np.ascontiguousarray((matrix + matrix.T) * 0.5, dtype=np.float64)
    if validate_psd:
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
        tolerance = np.finfo(np.float64).eps * input_features * scale * 64.0
        if float(eigenvalues[0]) < -tolerance:
            raise ValueError("hessian must be positive semidefinite")
    return matrix


def _finite_positive_scalar(value: float, name: str) -> float:
    parsed = _finite_nonnegative_scalar(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = [
    "DistortionPoint",
    "DistortionProfile",
    "activation_weighted_mse",
    "build_distortion_profile",
    "hessian_proxy",
    "hessian_weighted_error",
    "profile_progressive_quantization",
    "stable_kl_divergence",
]
