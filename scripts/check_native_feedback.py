"""Check the CUDA direct-I/O test's nine real samples against scheduler replay.

Run with PYTHONPATH=src and the JSON emitted by harpq_cuda_direct_packet_io_test
--json. This is a correctness check, not a throughput or calibration benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from harpq.scheduler import (
    CacheDeadlineScheduler,
    PacketRequest,
    SchedulerConfig,
    SchedulerTrace,
    ServiceObservation,
    replay_trace,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_native_feedback.py OBSERVATIONS_JSON")
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    observations = tuple(ServiceObservation.from_dict(item) for item in data["observations"])
    assert len(observations) == 9, "expected three samples at each of three sizes"
    assert tuple(item.sequence_id for item in observations) == tuple(range(9))
    scheduler = CacheDeadlineScheduler(SchedulerConfig(capacity_bytes=2 * 1048576))
    for index, physical in enumerate((65536, 262144, 1048576)):
        samples = observations[index * 3 : (index + 1) * 3]
        for sample in samples:
            assert sample.service_class == "direct-h2d/host-observed-v1"
            assert sample.logical_bytes == physical - 29 and sample.physical_io_bytes == physical
            assert sample.critical_path_us > 0
        packet = PacketRequest(
            packet_id=f"target-{physical}",
            size_bytes=physical - 29,
            io_bytes=physical,
            service_class=samples[0].service_class,
            distortion_gain=1.0,
            deadline_us=max(sample.critical_path_us for sample in samples) + 1.0,
        )
        decision = scheduler.schedule(index, (packet,), observations=samples)
        estimate = decision.transfer_estimates[0]
        assert estimate.source == "feedback" and estimate.sample_count == 3
        assert estimate.transfer_us == max(sample.critical_path_us for sample in samples)
        assert decision.loaded == (packet.packet_id,)
    parsed = SchedulerTrace.from_json(scheduler.trace.to_json())
    assert parsed == scheduler.trace
    assert replay_trace(parsed) == tuple(event.decision for event in parsed.events)
    print("native feedback: 9 real samples, 3 exact buckets, trace round-trip/replay passed")


if __name__ == "__main__":
    main()
