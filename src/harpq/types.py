"""Shared, serialization-friendly runtime types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class MemoryTier(str, Enum):
    """Storage tiers ordered by increasing distance from compute."""

    GPU = "gpu"
    PINNED_HOST = "pinned_host"
    HOST = "host"
    SSD = "ssd"


class ExecutionMode(str, Enum):
    """Runtime policies exposed by the reference scheduler."""

    DETERMINISTIC = "deterministic"
    ADAPTIVE_SLO = "adaptive_slo"
    THROUGHPUT = "throughput"


@dataclass(frozen=True, slots=True)
class RuntimeBudget:
    """Resource limits used by offline allocation and runtime scheduling."""

    gpu_bytes: int
    host_bytes: int
    ssd_bytes_per_step: int
    deadline_us: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Portable subset of conservatively verified host capabilities.

    CUDA and io_uring booleans require successful driver/syscall probes. O_DIRECT remains false in
    a passive report because support is target-filesystem-specific; the active fio probe may set
    it true in the serialized report.
    """

    platform: str
    machine: str
    python: str
    cpu_count: int | None
    cuda_available: bool
    io_uring_available: bool
    odirect_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
