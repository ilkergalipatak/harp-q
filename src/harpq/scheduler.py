"""Deterministic cache/deadline scheduler and replayable runtime trace.

The reference scheduler models the refinement cache, not the always-executable
base model.  It admits dependency-closed packet prefixes, rejects residuals that
would miss their deadline, and prevents cache thrash with minimum residency and a
score hysteresis threshold.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


_TRACE_FORMAT_VERSION = 2


class SchedulerError(ValueError):
    """Base class for scheduler input or state failures."""


class SchedulerDependencyError(SchedulerError):
    """Raised when a request dependency is missing or cyclic."""


class UnschedulableRequestError(SchedulerError):
    """Raised when a required request cannot meet its contract."""


class ReplayMismatchError(SchedulerError):
    """Raised when replay diverges from a recorded decision."""


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Policy and transfer model for one refinement cache."""

    capacity_bytes: int
    bandwidth_bytes_per_us: float = 7000.0
    fixed_read_latency_us: float = 0.0
    min_residency_tokens: int = 8
    hysteresis: float = 0.15
    feedback_window_size: int = 32
    feedback_min_samples: int = 3
    feedback_quantile: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.capacity_bytes, bool) or not isinstance(self.capacity_bytes, int):
            raise TypeError("capacity_bytes must be an integer")
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if not math.isfinite(self.bandwidth_bytes_per_us) or self.bandwidth_bytes_per_us <= 0:
            raise ValueError("bandwidth_bytes_per_us must be finite and positive")
        if not math.isfinite(self.fixed_read_latency_us) or self.fixed_read_latency_us < 0:
            raise ValueError("fixed_read_latency_us must be finite and non-negative")
        if (
            isinstance(self.min_residency_tokens, bool)
            or not isinstance(self.min_residency_tokens, int)
            or self.min_residency_tokens < 0
        ):
            raise ValueError("min_residency_tokens must be a non-negative integer")
        if not math.isfinite(self.hysteresis) or self.hysteresis < 0:
            raise ValueError("hysteresis must be finite and non-negative")
        if (
            isinstance(self.feedback_window_size, bool)
            or not isinstance(self.feedback_window_size, int)
            or self.feedback_window_size <= 0
        ):
            raise ValueError("feedback_window_size must be a positive integer")
        if (
            isinstance(self.feedback_min_samples, bool)
            or not isinstance(self.feedback_min_samples, int)
            or self.feedback_min_samples <= 0
            or self.feedback_min_samples > self.feedback_window_size
        ):
            raise ValueError(
                "feedback_min_samples must be a positive integer no larger than "
                "feedback_window_size"
            )
        if (
            isinstance(self.feedback_quantile, bool)
            or not math.isfinite(self.feedback_quantile)
            or self.feedback_quantile <= 0
            or self.feedback_quantile > 1
        ):
            raise ValueError("feedback_quantile must be finite and in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerConfig":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PacketRequest:
    """A packet that may improve the current precision plan.

    ``deadline_us`` is relative to the start of this scheduling step.
    ``transfer_us`` can carry a measured cache-miss/reload cost; when absent, the scheduler
    derives one from feedback or packet size and :class:`SchedulerConfig`. The hypothetical cost
    still participates in priority and eviction scoring when this packet is already resident; it
    is added to decision copy time only when a load actually occurs.
    """

    packet_id: str
    size_bytes: int
    distortion_gain: float
    deadline_us: float
    dependencies: tuple[str, ...] = ()
    reuse: float = 1.0
    importance: float = 1.0
    kernel_cost_us: float = 0.0
    io_bytes: int | None = None
    transfer_us: float | None = None
    required: bool = False
    service_class: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("packet_id must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.io_bytes is not None and (
            isinstance(self.io_bytes, bool)
            or not isinstance(self.io_bytes, int)
            or self.io_bytes < self.size_bytes
        ):
            raise ValueError("io_bytes must be an integer greater than or equal to size_bytes")
        for name in ("distortion_gain", "deadline_us", "reuse", "importance", "kernel_cost_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.transfer_us is not None and (
            not math.isfinite(self.transfer_us) or self.transfer_us <= 0
        ):
            raise ValueError("transfer_us must be finite and positive when supplied")
        if not isinstance(self.service_class, str) or not self.service_class:
            raise ValueError("service_class must be a non-empty string")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("dependencies must contain non-empty packet IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies cannot contain duplicates")
        if self.packet_id in dependencies:
            raise SchedulerDependencyError(f"packet {self.packet_id!r} cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)

    def transfer_duration_us(self, config: SchedulerConfig) -> float:
        if self.transfer_us is not None:
            return self.transfer_us
        transfer_bytes = self.io_bytes if self.io_bytes is not None else self.size_bytes
        return config.fixed_read_latency_us + transfer_bytes / config.bandwidth_bytes_per_us

    def priority_score(self, config: SchedulerConfig) -> float:
        cost = self.transfer_duration_us(config) + self.kernel_cost_us
        if cost == 0:
            return math.inf if self.distortion_gain * self.reuse * self.importance > 0 else 0.0
        return self.distortion_gain * self.reuse * self.importance / cost

    def to_dict(self, *, format_version: int = _TRACE_FORMAT_VERSION) -> dict[str, Any]:
        result = {
            "deadline_us": self.deadline_us,
            "dependencies": list(self.dependencies),
            "distortion_gain": self.distortion_gain,
            "importance": self.importance,
            "io_bytes": self.io_bytes,
            "kernel_cost_us": self.kernel_cost_us,
            "packet_id": self.packet_id,
            "required": self.required,
            "reuse": self.reuse,
            "size_bytes": self.size_bytes,
            "transfer_us": self.transfer_us,
        }
        if format_version >= 2:
            result["service_class"] = self.service_class
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PacketRequest":
        data = dict(value)
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ServiceObservation:
    """A completed transfer measurement supplied by the runtime.

    The scheduler never reads a clock.  A caller measures a completed cache-miss path and supplies
    the result at the next scheduling step. ``critical_path_us`` is the observed time from storage
    submission until the packet became device-ready. The component times are audit metadata only:
    they are not added together, so the reference scheduler does not invent an overlap ratio.
    """

    sequence_id: int
    packet_id: str
    logical_bytes: int
    physical_io_bytes: int
    critical_path_us: float
    service_class: str = "default"
    io_service_us: float | None = None
    h2d_service_us: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence_id, bool)
            or not isinstance(self.sequence_id, int)
            or self.sequence_id < 0
        ):
            raise ValueError("sequence_id must be a non-negative integer")
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("packet_id must be a non-empty string")
        if not isinstance(self.service_class, str) or not self.service_class:
            raise ValueError("service_class must be a non-empty string")
        if (
            isinstance(self.logical_bytes, bool)
            or not isinstance(self.logical_bytes, int)
            or self.logical_bytes <= 0
        ):
            raise ValueError("logical_bytes must be a positive integer")
        if (
            isinstance(self.physical_io_bytes, bool)
            or not isinstance(self.physical_io_bytes, int)
            or self.physical_io_bytes < self.logical_bytes
        ):
            raise ValueError(
                "physical_io_bytes must be an integer greater than or equal to logical_bytes"
            )
        if (
            isinstance(self.critical_path_us, bool)
            or not math.isfinite(self.critical_path_us)
            or self.critical_path_us <= 0
        ):
            raise ValueError("critical_path_us must be finite and positive")
        for name in ("io_service_us", "h2d_service_us"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative when supplied")
            if value is not None and value > self.critical_path_us:
                raise ValueError(f"{name} cannot exceed critical_path_us")

    @property
    def service_key(self) -> tuple[str, int, int]:
        return (self.service_class, self.logical_bytes, self.physical_io_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_path_us": self.critical_path_us,
            "h2d_service_us": self.h2d_service_us,
            "io_service_us": self.io_service_us,
            "logical_bytes": self.logical_bytes,
            "packet_id": self.packet_id,
            "physical_io_bytes": self.physical_io_bytes,
            "sequence_id": self.sequence_id,
            "service_class": self.service_class,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ServiceObservation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class TransferEstimate:
    """Auditable cache-miss/reload estimate used for one request decision.

    An estimate is recorded for every request because the policy uses hypothetical reload cost in
    priority and eviction scoring. Its presence does not mean a transfer occurred: only packet IDs
    in :attr:`SchedulerDecision.loaded` contribute to copy time and byte counters.
    """

    packet_id: str
    transfer_us: float
    source: str
    sample_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("packet_id must be a non-empty string")
        if (
            isinstance(self.transfer_us, bool)
            or not math.isfinite(self.transfer_us)
            or self.transfer_us <= 0
        ):
            raise ValueError("transfer_us must be finite and positive")
        if self.source not in {"request", "feedback", "config"}:
            raise ValueError("source must be request, feedback, or config")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be a non-negative integer")
        if self.source == "feedback" and self.sample_count == 0:
            raise ValueError("feedback estimates require at least one sample")
        if self.source != "feedback" and self.sample_count != 0:
            raise ValueError("non-feedback estimates cannot report feedback samples")

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "sample_count": self.sample_count,
            "source": self.source,
            "transfer_us": self.transfer_us,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransferEstimate":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Serializable scheduler cache state."""

    packet_id: str
    size_bytes: int
    score: float
    loaded_token: int
    last_access_token: int
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.packet_id, str) or not self.packet_id:
            raise ValueError("cache packet_id must be a non-empty string")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("cache size_bytes must be a positive integer")
        if not math.isfinite(self.score) or self.score < 0:
            raise ValueError("cache score must be finite and non-negative")
        if isinstance(self.loaded_token, bool) or not isinstance(self.loaded_token, int):
            raise TypeError("loaded_token must be an integer")
        if isinstance(self.last_access_token, bool) or not isinstance(self.last_access_token, int):
            raise TypeError("last_access_token must be an integer")
        dependencies = tuple(self.dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ValueError("cache dependencies must contain non-empty packet IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("cache dependencies cannot contain duplicates")
        object.__setattr__(self, "dependencies", dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependencies": list(self.dependencies),
            "last_access_token": self.last_access_token,
            "loaded_token": self.loaded_token,
            "packet_id": self.packet_id,
            "score": self.score,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheEntry":
        data = dict(value)
        data["dependencies"] = tuple(data.get("dependencies", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    """One complete, deterministic scheduling outcome.

    ``transfer_estimates`` are hypothetical cache-miss/reload costs for policy scoring. Actual
    transfers are identified by ``loaded`` and charged to ``copy_time_us``,
    ``logical_load_bytes``, and ``io_load_bytes``.
    """

    token_index: int
    selected: tuple[str, ...]
    loaded: tuple[str, ...]
    evicted: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    scores: tuple[tuple[str, float], ...]
    copy_time_us: float
    cache_bytes: int
    cache_packet_ids: tuple[str, ...]
    logical_load_bytes: int = 0
    io_load_bytes: int = 0
    read_amplification: float = 0.0
    transfer_estimates: tuple[TransferEstimate, ...] = ()

    def to_dict(self, *, format_version: int = _TRACE_FORMAT_VERSION) -> dict[str, Any]:
        result = {
            "cache_bytes": self.cache_bytes,
            "cache_packet_ids": list(self.cache_packet_ids),
            "copy_time_us": self.copy_time_us,
            "evicted": list(self.evicted),
            "loaded": list(self.loaded),
            "logical_load_bytes": self.logical_load_bytes,
            "io_load_bytes": self.io_load_bytes,
            "read_amplification": self.read_amplification,
            "scores": [[packet_id, score] for packet_id, score in self.scores],
            "selected": list(self.selected),
            "skipped": [[packet_id, reason] for packet_id, reason in self.skipped],
            "token_index": self.token_index,
        }
        if format_version >= 2:
            result["transfer_estimates"] = [item.to_dict() for item in self.transfer_estimates]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerDecision":
        return cls(
            token_index=value["token_index"],
            selected=tuple(value["selected"]),
            loaded=tuple(value["loaded"]),
            evicted=tuple(value["evicted"]),
            skipped=tuple((item[0], item[1]) for item in value["skipped"]),
            scores=tuple((item[0], item[1]) for item in value["scores"]),
            copy_time_us=value["copy_time_us"],
            cache_bytes=value["cache_bytes"],
            cache_packet_ids=tuple(value["cache_packet_ids"]),
            logical_load_bytes=value.get("logical_load_bytes", 0),
            io_load_bytes=value.get("io_load_bytes", 0),
            read_amplification=value.get("read_amplification", 0.0),
            transfer_estimates=tuple(
                TransferEstimate.from_dict(item) for item in value.get("transfer_estimates", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    token_index: int
    now_us: float
    requests: tuple[PacketRequest, ...]
    decision: SchedulerDecision
    observations: tuple[ServiceObservation, ...] = ()

    def to_dict(self, *, format_version: int = _TRACE_FORMAT_VERSION) -> dict[str, Any]:
        result = {
            "decision": self.decision.to_dict(format_version=format_version),
            "now_us": self.now_us,
            "requests": [
                request.to_dict(format_version=format_version) for request in self.requests
            ],
            "token_index": self.token_index,
        }
        if format_version >= 2:
            result["observations"] = [item.to_dict() for item in self.observations]
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        return cls(
            token_index=value["token_index"],
            now_us=value["now_us"],
            requests=tuple(PacketRequest.from_dict(item) for item in value["requests"]),
            decision=SchedulerDecision.from_dict(value["decision"]),
            observations=tuple(
                ServiceObservation.from_dict(item) for item in value.get("observations", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class SchedulerTrace:
    """Self-contained inputs and decisions needed for deterministic replay."""

    config: SchedulerConfig
    initial_cache: tuple[CacheEntry, ...]
    events: tuple[TraceEvent, ...]
    format_version: int = _TRACE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version not in {1, _TRACE_FORMAT_VERSION}
        ):
            raise SchedulerError("unsupported scheduler trace format")
        if self.format_version == 1 and any(
            event.observations
            or event.decision.transfer_estimates
            or any(request.service_class != "default" for request in event.requests)
            for event in self.events
        ):
            raise SchedulerError("scheduler trace v1 cannot contain service feedback fields")

    def to_dict(self) -> dict[str, Any]:
        config = self.config.to_dict()
        if self.format_version == 1:
            config.pop("feedback_window_size")
            config.pop("feedback_min_samples")
            config.pop("feedback_quantile")
        return {
            "config": config,
            "events": [
                event.to_dict(format_version=self.format_version) for event in self.events
            ],
            "format": "harpq-scheduler-trace",
            "format_version": self.format_version,
            "initial_cache": [entry.to_dict() for entry in self.initial_cache],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent, allow_nan=False, ensure_ascii=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SchedulerTrace":
        if not isinstance(value, Mapping):
            raise SchedulerError("scheduler trace root must be an object")
        version = value.get("format_version")
        if (
            value.get("format") != "harpq-scheduler-trace"
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version not in {1, _TRACE_FORMAT_VERSION}
        ):
            raise SchedulerError("unsupported scheduler trace format")

        def require_fields(
            item: Any, expected: frozenset[str], label: str
        ) -> Mapping[str, Any]:
            if not isinstance(item, Mapping):
                raise SchedulerError(f"{label} must be an object")
            actual = frozenset(item)
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(
                    (actual - expected), key=lambda field: str(field)
                )
                details: list[str] = []
                if missing:
                    details.append(f"missing={missing!r}")
                if unexpected:
                    details.append(f"unexpected={unexpected!r}")
                raise SchedulerError(
                    f"{label} has invalid fields for trace v{version}: "
                    + ", ".join(details)
                )
            return item

        def require_array(item: Any, label: str) -> Sequence[Any]:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes, bytearray))
            ):
                raise SchedulerError(f"{label} must be an array")
            return item

        root_fields = frozenset(
            {"config", "events", "format", "format_version", "initial_cache"}
        )
        config_fields = frozenset(
            {
                "bandwidth_bytes_per_us",
                "capacity_bytes",
                "fixed_read_latency_us",
                "hysteresis",
                "min_residency_tokens",
            }
        )
        request_fields = frozenset(
            {
                "deadline_us",
                "dependencies",
                "distortion_gain",
                "importance",
                "io_bytes",
                "kernel_cost_us",
                "packet_id",
                "required",
                "reuse",
                "size_bytes",
                "transfer_us",
            }
        )
        decision_fields = frozenset(
            {
                "cache_bytes",
                "cache_packet_ids",
                "copy_time_us",
                "evicted",
                "io_load_bytes",
                "loaded",
                "logical_load_bytes",
                "read_amplification",
                "scores",
                "selected",
                "skipped",
                "token_index",
            }
        )
        event_fields = frozenset({"decision", "now_us", "requests", "token_index"})
        if version >= 2:
            config_fields |= frozenset(
                {"feedback_min_samples", "feedback_quantile", "feedback_window_size"}
            )
            request_fields |= frozenset({"service_class"})
            decision_fields |= frozenset({"transfer_estimates"})
            event_fields |= frozenset({"observations"})

        require_fields(value, root_fields, "scheduler trace")
        config_value = require_fields(value["config"], config_fields, "scheduler config")
        initial_cache_value = require_array(value["initial_cache"], "initial_cache")
        cache_fields = frozenset(
            {
                "dependencies",
                "last_access_token",
                "loaded_token",
                "packet_id",
                "score",
                "size_bytes",
            }
        )
        for index, item in enumerate(initial_cache_value):
            require_fields(item, cache_fields, f"initial_cache[{index}]")

        events_value = require_array(value["events"], "events")
        observation_fields = frozenset(
            {
                "critical_path_us",
                "h2d_service_us",
                "io_service_us",
                "logical_bytes",
                "packet_id",
                "physical_io_bytes",
                "sequence_id",
                "service_class",
            }
        )
        estimate_fields = frozenset(
            {"packet_id", "sample_count", "source", "transfer_us"}
        )
        for event_index, event_item in enumerate(events_value):
            event = require_fields(
                event_item, event_fields, f"events[{event_index}]"
            )
            requests = require_array(
                event["requests"], f"events[{event_index}].requests"
            )
            for request_index, request_item in enumerate(requests):
                require_fields(
                    request_item,
                    request_fields,
                    f"events[{event_index}].requests[{request_index}]",
                )
            decision = require_fields(
                event["decision"],
                decision_fields,
                f"events[{event_index}].decision",
            )
            if version >= 2:
                observations = require_array(
                    event["observations"], f"events[{event_index}].observations"
                )
                for observation_index, observation_item in enumerate(observations):
                    require_fields(
                        observation_item,
                        observation_fields,
                        f"events[{event_index}].observations[{observation_index}]",
                    )
                estimates = require_array(
                    decision["transfer_estimates"],
                    f"events[{event_index}].decision.transfer_estimates",
                )
                for estimate_index, estimate_item in enumerate(estimates):
                    require_fields(
                        estimate_item,
                        estimate_fields,
                        "events["
                        f"{event_index}].decision.transfer_estimates[{estimate_index}]",
                    )
        return cls(
            config=SchedulerConfig.from_dict(config_value),
            initial_cache=tuple(CacheEntry.from_dict(item) for item in initial_cache_value),
            events=tuple(TraceEvent.from_dict(item) for item in events_value),
            format_version=version,
        )

    @classmethod
    def from_json(cls, encoded: str | bytes | bytearray) -> "SchedulerTrace":
        try:
            value = json.loads(encoded)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SchedulerError("scheduler trace is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise SchedulerError("scheduler trace root must be an object")
        return cls.from_dict(value)


class CacheDeadlineScheduler:
    """Stateful deterministic scheduler for optional refinement packets."""

    def __init__(
        self,
        config: SchedulerConfig,
        *,
        initial_cache: Iterable[CacheEntry] = (),
        record_trace: bool = True,
        _trace_format_version: int = _TRACE_FORMAT_VERSION,
    ) -> None:
        if _trace_format_version not in {1, _TRACE_FORMAT_VERSION}:
            raise SchedulerError("unsupported scheduler trace format")
        self.config = config
        entries = tuple(sorted(initial_cache, key=lambda entry: entry.packet_id))
        self._validate_initial_cache(entries)
        self._initial_cache = entries
        self._cache = {entry.packet_id: entry for entry in entries}
        self._events: list[TraceEvent] = []
        self._record_trace = record_trace
        self._last_token: int | None = None
        self._trace_format_version = _trace_format_version
        self._feedback_samples: dict[tuple[str, int, int], list[float]] = {}
        self._last_observation_sequence: int | None = None

    def _validate_initial_cache(self, entries: Sequence[CacheEntry]) -> None:
        by_id = {entry.packet_id: entry for entry in entries}
        if len(by_id) != len(entries):
            raise SchedulerDependencyError("initial cache packet IDs must be unique")
        if sum(entry.size_bytes for entry in entries) > self.config.capacity_bytes:
            raise SchedulerError("initial cache exceeds configured capacity")
        for entry in entries:
            for dependency_id in entry.dependencies:
                if dependency_id not in by_id:
                    raise SchedulerDependencyError(
                        f"cached packet {entry.packet_id!r} misses dependency {dependency_id!r}"
                    )
        _validate_dependency_map({entry.packet_id: entry.dependencies for entry in entries})

    @property
    def cache_entries(self) -> tuple[CacheEntry, ...]:
        return tuple(self._cache[packet_id] for packet_id in sorted(self._cache))

    @property
    def cache_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self._cache.values())

    @property
    def trace(self) -> SchedulerTrace:
        return SchedulerTrace(
            config=self.config,
            initial_cache=self._initial_cache,
            events=tuple(self._events),
            format_version=self._trace_format_version,
        )

    def export_trace(self) -> SchedulerTrace:
        """Method spelling for callers that prefer an explicit snapshot operation."""

        return self.trace

    def schedule(
        self,
        token_index: int,
        requests: Iterable[PacketRequest],
        *,
        now_us: float = 0.0,
        observations: Iterable[ServiceObservation] = (),
    ) -> SchedulerDecision:
        """Schedule one epoch and atomically update cache and feedback state.

        Observations must have been measured before this call. They update a deterministic service
        model before the current requests are scored; this function does not sample wall-clock
        time. A failed required request rolls the observations back so a retry can resubmit them.
        """

        if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 0:
            raise ValueError("token_index must be a non-negative integer")
        if self._last_token is not None and token_index < self._last_token:
            raise SchedulerError("token_index cannot move backwards")
        if not math.isfinite(now_us) or now_us < 0:
            raise ValueError("now_us must be finite and non-negative")
        request_tuple = tuple(sorted(requests, key=lambda request: request.packet_id))
        observation_tuple = tuple(observations)
        if self._trace_format_version == 1 and observation_tuple:
            raise SchedulerError("scheduler trace v1 cannot record service observations")
        request_map = self._validate_requests(request_tuple)
        snapshot = dict(self._cache)
        feedback_snapshot = {
            key: list(samples) for key, samples in self._feedback_samples.items()
        }
        last_token = self._last_token
        last_observation_sequence = self._last_observation_sequence
        try:
            self._apply_observations(observation_tuple)
            decision = self._schedule_impl(token_index, request_tuple, request_map)
        except Exception:
            self._cache = snapshot
            self._feedback_samples = feedback_snapshot
            self._last_token = last_token
            self._last_observation_sequence = last_observation_sequence
            raise
        self._last_token = token_index
        if self._record_trace:
            self._events.append(
                TraceEvent(
                    token_index=token_index,
                    now_us=now_us,
                    requests=request_tuple,
                    decision=decision,
                    observations=observation_tuple,
                )
            )
        return decision

    def _apply_observations(self, observations: Sequence[ServiceObservation]) -> None:
        last_sequence = self._last_observation_sequence
        for observation in observations:
            if not isinstance(observation, ServiceObservation):
                raise TypeError("observations must contain ServiceObservation instances")
            if last_sequence is not None and observation.sequence_id <= last_sequence:
                raise SchedulerError("service observation sequence IDs must increase strictly")
            samples = self._feedback_samples.setdefault(observation.service_key, [])
            samples.append(observation.critical_path_us)
            if len(samples) > self.config.feedback_window_size:
                del samples[: len(samples) - self.config.feedback_window_size]
            last_sequence = observation.sequence_id
        self._last_observation_sequence = last_sequence

    def _estimate_transfer(self, request: PacketRequest) -> TransferEstimate:
        if request.transfer_us is not None:
            return TransferEstimate(
                packet_id=request.packet_id,
                transfer_us=request.transfer_us,
                source="request",
            )
        physical_io_bytes = request.io_bytes if request.io_bytes is not None else request.size_bytes
        samples = self._feedback_samples.get(
            (request.service_class, request.size_bytes, physical_io_bytes), ()
        )
        if len(samples) >= self.config.feedback_min_samples:
            ordered = sorted(samples)
            rank = math.ceil(self.config.feedback_quantile * len(ordered)) - 1
            return TransferEstimate(
                packet_id=request.packet_id,
                transfer_us=ordered[rank],
                source="feedback",
                sample_count=len(ordered),
            )
        return TransferEstimate(
            packet_id=request.packet_id,
            transfer_us=request.transfer_duration_us(self.config),
            source="config",
        )

    def _validate_requests(
        self, requests: Sequence[PacketRequest]
    ) -> dict[str, PacketRequest]:
        request_map = {request.packet_id: request for request in requests}
        if len(request_map) != len(requests):
            raise SchedulerDependencyError("request packet IDs must be unique")
        dependency_map: dict[str, tuple[str, ...]] = {
            entry.packet_id: entry.dependencies for entry in self._cache.values()
        }
        for request in requests:
            cached = self._cache.get(request.packet_id)
            if cached is not None and (
                cached.size_bytes != request.size_bytes
                or cached.dependencies != request.dependencies
            ):
                raise SchedulerError(
                    f"request metadata changed for cached packet {request.packet_id!r}"
                )
            for dependency_id in request.dependencies:
                if dependency_id not in request_map and dependency_id not in self._cache:
                    raise SchedulerDependencyError(
                        f"packet {request.packet_id!r} depends on unavailable packet "
                        f"{dependency_id!r}"
                    )
            dependency_map[request.packet_id] = request.dependencies
        _validate_dependency_map(dependency_map)
        return request_map

    def _schedule_impl(
        self,
        token_index: int,
        requests: Sequence[PacketRequest],
        request_map: Mapping[str, PacketRequest],
    ) -> SchedulerDecision:
        estimates = {
            request.packet_id: self._estimate_transfer(request) for request in requests
        }
        scores = {
            request.packet_id: self._priority_score(
                request, estimates[request.packet_id].transfer_us
            )
            for request in requests
        }
        order = sorted(
            requests,
            key=lambda request: (
                not request.required,
                -scores[request.packet_id],
                request.deadline_us,
                request.packet_id,
            ),
        )
        protected: set[str] = set()
        loaded: list[str] = []
        evicted: list[str] = []
        skipped: dict[str, str] = {}
        copy_time_us = 0.0
        logical_load_bytes = 0
        io_load_bytes = 0

        for root in order:
            closure = self._dependency_closure(root.packet_id, request_map)
            missing = [packet_id for packet_id in closure if packet_id not in self._cache]
            if not missing:
                protected.update(closure)
                self._touch(closure, request_map, scores, token_index)
                continue

            finish = copy_time_us
            missed_deadline = False
            for packet_id in missing:
                request = request_map[packet_id]
                finish += estimates[packet_id].transfer_us
                if finish > request.deadline_us:
                    missed_deadline = True
                    break
            if finish > root.deadline_us:
                missed_deadline = True
            if missed_deadline:
                if root.required:
                    raise UnschedulableRequestError(
                        f"required packet {root.packet_id!r} cannot meet its deadline"
                    )
                skipped[root.packet_id] = "deadline"
                continue

            additional_bytes = sum(request_map[item].size_bytes for item in missing)
            needed_bytes = max(0, self.cache_bytes + additional_bytes - self.config.capacity_bytes)
            victims, reason = self._choose_evictions(
                needed_bytes=needed_bytes,
                protected=protected | set(closure),
                admission_score=scores[root.packet_id],
                token_index=token_index,
                required=root.required,
            )
            if victims is None:
                if root.required:
                    raise UnschedulableRequestError(
                        f"required packet {root.packet_id!r} cannot fit cache: {reason}"
                    )
                skipped[root.packet_id] = reason
                continue

            for victim_id in victims:
                if victim_id in self._cache:
                    del self._cache[victim_id]
                    evicted.append(victim_id)
            for packet_id in missing:
                request = request_map[packet_id]
                copy_time_us += estimates[packet_id].transfer_us
                logical_load_bytes += request.size_bytes
                io_load_bytes += (
                    request.io_bytes if request.io_bytes is not None else request.size_bytes
                )
                self._cache[packet_id] = CacheEntry(
                    packet_id=packet_id,
                    size_bytes=request.size_bytes,
                    score=scores[packet_id],
                    loaded_token=token_index,
                    last_access_token=token_index,
                    dependencies=request.dependencies,
                )
                loaded.append(packet_id)
            protected.update(closure)
            self._touch(closure, request_map, scores, token_index)

        selected = tuple(sorted(packet_id for packet_id in request_map if packet_id in protected))
        for packet_id in selected:
            skipped.pop(packet_id, None)
        self._assert_cache_dependencies()
        return SchedulerDecision(
            token_index=token_index,
            selected=selected,
            loaded=tuple(loaded),
            evicted=tuple(evicted),
            skipped=tuple(sorted(skipped.items())),
            scores=tuple(sorted(scores.items())),
            copy_time_us=copy_time_us,
            cache_bytes=self.cache_bytes,
            cache_packet_ids=tuple(sorted(self._cache)),
            logical_load_bytes=logical_load_bytes,
            io_load_bytes=io_load_bytes,
            read_amplification=(
                io_load_bytes / logical_load_bytes if logical_load_bytes else 0.0
            ),
            transfer_estimates=(
                tuple(estimates[packet_id] for packet_id in sorted(estimates))
                if self._trace_format_version >= 2
                else ()
            ),
        )

    @staticmethod
    def _priority_score(request: PacketRequest, transfer_us: float) -> float:
        cost = transfer_us + request.kernel_cost_us
        benefit = request.distortion_gain * request.reuse * request.importance
        if cost == 0:
            return math.inf if benefit > 0 else 0.0
        return benefit / cost

    def _dependency_closure(
        self, packet_id: str, request_map: Mapping[str, PacketRequest]
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in visited:
                return
            visited.add(current_id)
            if current_id in request_map:
                dependencies = request_map[current_id].dependencies
            else:
                dependencies = self._cache[current_id].dependencies
            for dependency_id in sorted(dependencies):
                visit(dependency_id)
            ordered.append(current_id)

        visit(packet_id)
        return tuple(ordered)

    def _touch(
        self,
        packet_ids: Iterable[str],
        request_map: Mapping[str, PacketRequest],
        scores: Mapping[str, float],
        token_index: int,
    ) -> None:
        for packet_id in packet_ids:
            entry = self._cache[packet_id]
            self._cache[packet_id] = CacheEntry(
                packet_id=entry.packet_id,
                size_bytes=entry.size_bytes,
                score=scores.get(packet_id, entry.score),
                loaded_token=entry.loaded_token,
                last_access_token=token_index,
                dependencies=(
                    request_map[packet_id].dependencies
                    if packet_id in request_map
                    else entry.dependencies
                ),
            )

    def _choose_evictions(
        self,
        *,
        needed_bytes: int,
        protected: set[str],
        admission_score: float,
        token_index: int,
        required: bool,
    ) -> tuple[tuple[str, ...] | None, str]:
        if needed_bytes <= 0:
            return (), ""
        candidates = sorted(
            self._cache.values(),
            key=lambda entry: (entry.score, entry.last_access_token, entry.packet_id),
        )
        planned: set[str] = set()
        freed = 0
        saw_min_residency = False
        saw_hysteresis = False
        for candidate in candidates:
            if candidate.packet_id in planned:
                continue
            bundle = self._dependent_bundle(candidate.packet_id) - planned
            if bundle & protected:
                continue
            entries = [self._cache[packet_id] for packet_id in sorted(bundle)]
            if not required and any(
                token_index - entry.loaded_token < self.config.min_residency_tokens
                for entry in entries
            ):
                saw_min_residency = True
                continue
            if not required and any(
                admission_score < entry.score * (1.0 + self.config.hysteresis)
                for entry in entries
            ):
                saw_hysteresis = True
                continue
            planned.update(bundle)
            freed += sum(entry.size_bytes for entry in entries)
            if freed >= needed_bytes:
                return tuple(sorted(planned)), ""
        if saw_min_residency:
            return None, "min_residency"
        if saw_hysteresis:
            return None, "hysteresis"
        return None, "capacity"

    def _dependent_bundle(self, packet_id: str) -> set[str]:
        reverse: dict[str, set[str]] = {item: set() for item in self._cache}
        for entry in self._cache.values():
            for dependency_id in entry.dependencies:
                reverse.setdefault(dependency_id, set()).add(entry.packet_id)
        bundle: set[str] = set()
        stack = [packet_id]
        while stack:
            current = stack.pop()
            if current in bundle:
                continue
            bundle.add(current)
            stack.extend(sorted(reverse.get(current, ()), reverse=True))
        return bundle

    def _assert_cache_dependencies(self) -> None:
        for entry in self._cache.values():
            missing = set(entry.dependencies) - set(self._cache)
            if missing:
                raise SchedulerDependencyError(
                    f"cache invariant failed for {entry.packet_id!r}: missing {sorted(missing)}"
                )

    @classmethod
    def from_trace(cls, trace: SchedulerTrace, *, verify: bool = True) -> "CacheDeadlineScheduler":
        """Replay *trace* and return the reconstructed scheduler state."""

        scheduler = cls(
            trace.config,
            initial_cache=trace.initial_cache,
            record_trace=True,
            _trace_format_version=trace.format_version,
        )
        for index, event in enumerate(trace.events):
            decision = scheduler.schedule(
                event.token_index,
                event.requests,
                now_us=event.now_us,
                observations=event.observations,
            )
            if verify and decision != event.decision:
                raise ReplayMismatchError(
                    f"scheduler replay diverged at event {index}: "
                    f"expected {event.decision}, got {decision}"
                )
        return scheduler


def _validate_dependency_map(dependencies: Mapping[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(packet_id: str) -> None:
        if packet_id in visiting:
            raise SchedulerDependencyError(f"dependency cycle includes packet {packet_id!r}")
        if packet_id in visited:
            return
        visiting.add(packet_id)
        for dependency_id in dependencies.get(packet_id, ()):
            if dependency_id not in dependencies:
                raise SchedulerDependencyError(
                    f"packet {packet_id!r} depends on missing packet {dependency_id!r}"
                )
            visit(dependency_id)
        visiting.remove(packet_id)
        visited.add(packet_id)

    for packet_id in sorted(dependencies):
        visit(packet_id)


def replay_trace(
    trace: SchedulerTrace | str | bytes | bytearray,
    *,
    verify: bool = True,
) -> tuple[SchedulerDecision, ...]:
    """Replay a trace and return its regenerated decisions."""

    parsed = (
        SchedulerTrace.from_json(trace)
        if isinstance(trace, (str, bytes, bytearray))
        else trace
    )
    scheduler = CacheDeadlineScheduler.from_trace(parsed, verify=verify)
    return tuple(event.decision for event in scheduler.trace.events)


__all__ = [
    "CacheDeadlineScheduler",
    "CacheEntry",
    "PacketRequest",
    "ReplayMismatchError",
    "SchedulerConfig",
    "SchedulerDecision",
    "SchedulerDependencyError",
    "SchedulerError",
    "SchedulerTrace",
    "ServiceObservation",
    "TraceEvent",
    "TransferEstimate",
    "UnschedulableRequestError",
    "replay_trace",
]
