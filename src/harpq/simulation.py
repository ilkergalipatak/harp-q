"""Adapters that drive the reference scheduler from a packed HARP-Q model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .container import ContainerReader, open_container
from .scheduler import (
    CacheDeadlineScheduler,
    PacketRequest,
    SchedulerConfig,
    SchedulerTrace,
    replay_trace,
)


def refinement_requests_from_container(
    reader: ContainerReader,
    *,
    deadline_us: float,
    metric: str = "activation_mse",
    reuse: float = 1.0,
    importance: float = 1.0,
) -> tuple[PacketRequest, ...]:
    """Build refinement-cache requests from container records and calibration metadata.

    Stage zero is the always-executable base and is deliberately excluded from this cache. The
    first residual therefore has no refinement-cache dependency; later residuals retain their
    chain.
    """

    requests: list[PacketRequest] = []
    for record in reader:
        if record.stage <= 0 or record.packet_id in reader.unavailable_packet_ids:
            continue
        user = record.metadata.get("user", {})
        distortion = user.get("distortion", {}) if isinstance(user, dict) else {}
        series = distortion.get(metric) if isinstance(distortion, dict) else None
        if not isinstance(series, list) or len(series) <= record.stage:
            raise ValueError(
                f"packet {record.packet_id!r} has no complete {metric!r} distortion profile"
            )
        previous = series[record.stage - 1]
        current = series[record.stage]
        if previous is None or current is None:
            raise ValueError(f"packet {record.packet_id!r} has unavailable {metric!r} values")
        gain = max(0.0, float(previous) - float(current))
        refinement_dependencies = tuple(
            dependency_id
            for dependency_id in record.dependencies
            if reader.record(dependency_id).stage > 0
        )
        requests.append(
            PacketRequest(
                packet_id=record.packet_id,
                size_bytes=record.length,
                io_bytes=(
                    (record.length + reader.manifest.alignment - 1)
                    // reader.manifest.alignment
                    * reader.manifest.alignment
                ),
                distortion_gain=gain,
                deadline_us=deadline_us,
                dependencies=refinement_dependencies,
                reuse=reuse,
                importance=importance,
            )
        )
    return tuple(sorted(requests, key=lambda request: request.packet_id))


def simulate_container(
    container: str | Path,
    config: SchedulerConfig,
    *,
    token_count: int,
    deadline_us: float,
    metric: str = "activation_mse",
) -> SchedulerTrace:
    """Run a deterministic repeated-token scheduling simulation."""

    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError("token_count must be a positive integer")
    with open_container(container) as reader:
        requests = refinement_requests_from_container(
            reader,
            deadline_us=deadline_us,
            metric=metric,
            reuse=float(token_count),
        )
    scheduler = CacheDeadlineScheduler(config)
    for token_index in range(token_count):
        scheduler.schedule(token_index, requests)
    trace = scheduler.export_trace()
    replay_trace(trace, verify=True)
    return trace


def write_scheduler_trace(path: str | Path, trace: SchedulerTrace) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(trace.to_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_scheduler_trace(path: str | Path) -> SchedulerTrace:
    return SchedulerTrace.from_json(Path(path).read_bytes())


__all__ = [
    "load_scheduler_trace",
    "refinement_requests_from_container",
    "simulate_container",
    "write_scheduler_trace",
]
