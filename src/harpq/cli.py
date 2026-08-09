"""Command-line entry points for the portable HARP-Q reference workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__
from .container import open_container
from .hardware import (
    basic_profile,
    collect_active_linux_report,
    collect_hardware_report,
    write_hardware_report,
)
from .pipeline import calibrate_matrix, write_calibrated_container
from .scheduler import SchedulerConfig, replay_trace
from .simulation import load_scheduler_trace, simulate_container, write_scheduler_trace
from .synthetic import SyntheticConfig, SyntheticTransformer, prefix_error_curve


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _doctor(arguments: argparse.Namespace) -> int:
    profile = basic_profile()
    report = {
        "harpq_version": __version__,
        "profile": profile.to_dict(),
        "numpy": np.__version__,
        "reference_tests_supported": True,
        "cuda_driver_device_available": profile.cuda_available,
        "nvidia_performance_tests_supported": False,
        "nvidia_performance_tests_reason": (
            "a correctness-first fused GEMV and registered pread-to-H2D ring are implemented; "
            "optimized throughput/static-W4 validation, the io_uring/O_DIRECT runtime, and "
            "model integration are pending"
        ),
    }
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        profile = report["profile"]
        print(f"HARP-Q {report['harpq_version']}")
        print(f"Host: {profile['platform']} ({profile['machine']})")
        print(f"Python/NumPy: {profile['python']} / {report['numpy']}")
        print(f"CUDA visible: {profile['cuda_available']}")
        print(f"io_uring visible: {profile['io_uring_available']}")
        if not report["nvidia_performance_tests_supported"]:
            print(
                "NVIDIA performance tests: unavailable "
                f"({report['nvidia_performance_tests_reason']})"
            )
    return 0


def _profile(arguments: argparse.Namespace) -> int:
    if arguments.active and not arguments.scratch:
        raise ValueError("--active requires an explicit --scratch directory")
    if arguments.output:
        destination = write_hardware_report(
            arguments.output,
            active_scratch=arguments.scratch if arguments.active else None,
        )
        print(destination)
    else:
        report = (
            collect_active_linux_report(arguments.scratch)
            if arguments.active
            else collect_hardware_report()
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _demo(arguments: argparse.Namespace) -> int:
    output = Path(arguments.output)
    config = SyntheticConfig(seed=arguments.seed)
    dense = SyntheticTransformer.random(config)
    quantized = dense.quantize()
    tokens = np.array([1, 5, 2, 9, 4, 3], dtype=np.int64)
    errors = prefix_error_curve(dense, quantized, tokens)
    for prefix in range(1, quantized.stage_count + 1):
        logits = quantized.forward(tokens, prefix=prefix)
        if not np.all(np.isfinite(logits)):
            raise RuntimeError(f"non-finite output at prefix {prefix}")

    summary = {
        "schema_version": 1,
        "purpose": "functional reference only; not an NVIDIA performance result",
        "seed": arguments.seed,
        "tile_shape": [128, 128],
        "group_size": 128,
        "stage_bits": [2, 2, 2, 2],
        "payload_bytes": quantized.payload_bytes,
        "prefix_logits_mse": list(errors),
        "all_prefixes_finite": True,
    }
    _write_json(output / "demo-summary.json", summary)
    _write_json(output / "hardware.json", collect_hardware_report())
    print(output / "demo-summary.json")
    return 0


def _quantize(arguments: argparse.Namespace) -> int:
    weights = np.load(arguments.weights, allow_pickle=False)
    activations = np.load(arguments.activations, allow_pickle=False)
    calibrated = calibrate_matrix(
        arguments.matrix_id,
        weights,
        activations,
        tile_shape=(arguments.tile_rows, arguments.tile_columns),
        group_size=arguments.group_size,
        stage_count=arguments.stages,
        include_output_kl=arguments.output_kl,
    )
    manifest = write_calibrated_container(
        arguments.output,
        [calibrated],
        metadata={"source": "harpq quantize"},
    )
    result = {
        "container": str(Path(arguments.output)),
        "manifest": manifest.to_dict(),
        "distortion": calibrated.distortion.as_dict(),
        "payload_bytes": calibrated.quantized.payload_bytes,
    }
    if arguments.report:
        _write_json(Path(arguments.report), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _inspect(arguments: argparse.Namespace) -> int:
    with open_container(arguments.container, verify=not arguments.skip_sha256) as reader:
        result = {
            "manifest": reader.manifest.to_dict(),
            "available_packet_ids": list(reader.available_packet_ids),
            "unavailable_packet_ids": list(reader.unavailable_packet_ids),
            "records": [record.to_dict() for record in reader],
        }
        if arguments.verify_payloads:
            reader.verify_all_payloads()
            result["payload_crc32c"] = "verified"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _simulate(arguments: argparse.Namespace) -> int:
    config = SchedulerConfig(
        capacity_bytes=arguments.capacity_bytes,
        bandwidth_bytes_per_us=arguments.bandwidth_bytes_per_us,
        fixed_read_latency_us=arguments.fixed_read_latency_us,
        min_residency_tokens=arguments.min_residency_tokens,
        hysteresis=arguments.hysteresis,
    )
    trace = simulate_container(
        arguments.container,
        config,
        token_count=arguments.tokens,
        deadline_us=arguments.deadline_us,
        metric=arguments.metric,
    )
    if arguments.output:
        print(write_scheduler_trace(arguments.output, trace))
    else:
        print(trace.to_json(indent=2))
    return 0


def _replay(arguments: argparse.Namespace) -> int:
    trace = load_scheduler_trace(arguments.trace)
    decisions = replay_trace(trace, verify=True)
    result = {
        "verified": True,
        "events": len(decisions),
        "final_decision": decisions[-1].to_dict() if decisions else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harpq",
        description="HARP-Q deterministic CPU reference tools",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="show host capabilities")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable output")
    doctor.set_defaults(handler=_doctor)

    profile = subparsers.add_parser("profile-host", help="capture a handoff hardware profile")
    profile.add_argument("--output", help="write JSON to this path")
    profile.add_argument(
        "--active",
        action="store_true",
        help="run bounded RAM, fio/O_DIRECT, and CUDA sample probes (Linux only)",
    )
    profile.add_argument(
        "--scratch",
        help="existing filesystem directory for the temporary 256 MiB fio probe",
    )
    profile.set_defaults(handler=_profile)

    demo = subparsers.add_parser("demo", help="run a small progressive-prefix correctness demo")
    demo.add_argument("--output", default="artifacts/demo", help="artifact directory")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(handler=_demo)

    quantize = subparsers.add_parser(
        "quantize",
        help="quantize one .npy linear weight matrix and write a HARP-Q container",
    )
    quantize.add_argument("--weights", required=True, help="2-D .npy weight matrix")
    quantize.add_argument("--activations", required=True, help="calibration .npy array")
    quantize.add_argument("--output", required=True, help="new container directory")
    quantize.add_argument("--matrix-id", default="linear.weight")
    quantize.add_argument("--tile-rows", type=int, default=128)
    quantize.add_argument("--tile-columns", type=int, default=128)
    quantize.add_argument("--group-size", type=int, default=128)
    quantize.add_argument("--stages", type=int, default=4)
    quantize.add_argument("--output-kl", action="store_true")
    quantize.add_argument("--report", help="optional JSON report path")
    quantize.set_defaults(handler=_quantize)

    inspect = subparsers.add_parser("inspect", help="inspect and verify a HARP-Q container")
    inspect.add_argument("container")
    inspect.add_argument("--skip-sha256", action="store_true")
    inspect.add_argument("--verify-payloads", action="store_true")
    inspect.set_defaults(handler=_inspect)

    simulate = subparsers.add_parser(
        "simulate",
        help="run cache/deadline scheduling against a calibrated container",
    )
    simulate.add_argument("container")
    simulate.add_argument("--capacity-bytes", type=int, required=True)
    simulate.add_argument("--tokens", type=int, default=8)
    simulate.add_argument("--deadline-us", type=float, default=5000.0)
    simulate.add_argument("--bandwidth-bytes-per-us", type=float, default=7000.0)
    simulate.add_argument("--fixed-read-latency-us", type=float, default=0.0)
    simulate.add_argument("--min-residency-tokens", type=int, default=8)
    simulate.add_argument("--hysteresis", type=float, default=0.15)
    simulate.add_argument("--metric", default="activation_mse")
    simulate.add_argument("--output", help="write the deterministic trace JSON")
    simulate.set_defaults(handler=_simulate)

    replay = subparsers.add_parser("replay", help="verify a deterministic scheduler trace")
    replay.add_argument("trace")
    replay.set_defaults(handler=_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
