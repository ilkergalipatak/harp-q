#!/usr/bin/env python3
"""Bounded Linux SSD/H2D sweep; raw records are evidence, not model throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


def pattern_block() -> bytes:
    return bytes(range(256)) * 4096


def fio_command(pack: Path, block: int, depth: int, volume: int) -> list[str]:
    return ["fio", "--readonly", "--name=harpq-read", f"--filename={pack}",
            "--rw=read", "--ioengine=io_uring", "--direct=1", "--allow_file_create=0",
            f"--bs={block}", f"--iodepth={depth}", "--numjobs=1",
            f"--size={volume}", f"--io_size={volume}", "--offset=0",
            "--iodepth_batch_submit=1", "--iodepth_batch_complete_min=1",
            "--iodepth_batch_complete_max=1", "--output-format=json"]


def fio_rate(data: dict, volume: int, depth: int, *, check_depth: bool = True) -> float:
    jobs = data.get("jobs", [])
    if len(jobs) != 1:
        raise ValueError("expected exactly one fio job")
    job = jobs[0]
    read = job.get("read", {})
    rate = read.get("bw_bytes", 0)
    if (job.get("error") != 0 or read.get("io_bytes") != volume or
            job.get("write", {}).get("io_bytes") != 0 or
            not math.isfinite(rate) or rate <= 0 or
            (check_depth and job.get("iodepth_level", {}).get(str(depth), 0) < 90)):
        raise ValueError("fio error, volume/rate mismatch, write, or unachieved queue depth")
    return rate


def summarize(rates: list[float]) -> dict:
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("rates must be finite and positive")
    return {"median_gb_s": statistics.median(rates) / 1e9,
            "min_gb_s": min(rates) / 1e9, "max_gb_s": max(rates) / 1e9,
            "repeats": len(rates)}


def record(output: Path, label: str, command: list[str], *, decode: bool = True):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    data = {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}
    (output / f"{label}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    result.check_returncode()
    return json.loads(result.stdout) if decode else result.stdout


def native_negative_checks(output: Path, executable: Path, scratch: Path, pack: Path) -> None:
    cases = [("invalid-size", ["direct", str(pack), "65535", "1", "1"]),
             ("invalid-volume", ["direct", str(pack), "1048576", "1", "4096"]),
             ("invalid-mode", ["unknown", str(pack), "65536", "1", "1"])]
    bad = scratch / "corrupted.pack"
    data = bytearray(pattern_block()[:65536])
    data[0] ^= 1
    with bad.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    cases.extend((f"corrupt-{mode}", [mode, str(bad), "65536", "1", "1"])
                 for mode in ("direct", "bridge"))
    for label, arguments in cases:
        try:
            record(output, label, [str(executable), *arguments], decode=False)
        except subprocess.CalledProcessError as error:
            if error.returncode != 1 or "transfer benchmark failed:" not in error.stderr:
                raise
            if label.startswith("corrupt-") and "CRC32C mismatch" not in error.stderr:
                raise ValueError("corruption was rejected for the wrong reason") from error
        else:
            raise ValueError(f"native benchmark accepted {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True, help="existing target filesystem directory")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory; never overwrite")
    parser.add_argument("--build", type=Path, required=True, help="explicit CUDA-ON/io_uring-ON Release build")
    parser.add_argument("--smoke", action="store_true", help="one small functional round, no performance claim")
    args = parser.parse_args()
    if sys.platform != "linux" or not args.scratch.is_dir():
        parser.error("requires Linux and an existing scratch directory")
    binaries = args.build.resolve() / "native/cuda"
    native = binaries / "harpq_transfer_benchmark"
    h2d = binaries / "harpq_h2d_benchmark"
    if not native.is_file() or not h2d.is_file():
        parser.error("build harpq_transfer_benchmark and harpq_h2d_benchmark first")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    snapshot = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,"
                "utilization.gpu,temperature.gpu,pstate,clocks.sm,clocks.mem,"
                "pcie.link.gen.current,pcie.link.width.current", "--format=csv"]
    record(output, "gpu-before", snapshot, decode=False)
    record(output, "source", ["git", "rev-parse", "HEAD"], decode=False)
    record(output, "source-status", ["git", "status", "--porcelain"], decode=False)
    record(output, "filesystem", ["findmnt", "-T", str(args.scratch.resolve()),
                                   "-o", "SOURCE,FSTYPE,OPTIONS", "--json"])
    record(output, "fio-version", ["fio", "--version"], decode=False)
    (output / "binary-sha256.json").write_text(json.dumps({
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (native, h2d)
    }, indent=2) + "\n", encoding="utf-8")
    volume = 128 * 1024 if args.smoke else 256 * 1024 * 1024
    warmup_volume = 65536 if args.smoke else 16 * 1024 * 1024
    sizes = (65536,) if args.smoke else (65536, 262144, 1048576)
    repeats = 1 if args.smoke else 5
    rates: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory(prefix="harpq-transfer-", dir=args.scratch.resolve()) as scratch:
        pack = Path(scratch) / "fixture.pack"
        block = pattern_block()
        with pack.open("xb") as stream:
            for _ in range(max(1, volume // len(block))):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if args.smoke:
            native_negative_checks(output, native, Path(scratch), pack)
        for repeat in range(repeats):
            for size in sizes if repeat % 2 == 0 else reversed(sizes):
                modes = ["fio1", "fio8", "direct", "bridge", "h2d"]
                if args.smoke:
                    modes.remove("fio8")
                if repeat % 2:
                    modes.reverse()
                for mode in modes:
                    label = f"{mode}-{size}-r{repeat}"
                    if mode.startswith("fio"):
                        depth = int(mode[3:])
                        warm = record(output, label + "-warmup",
                                      fio_command(pack, size, depth, warmup_volume))
                        fio_rate(warm, warmup_volume, depth, check_depth=False)
                        data = record(output, label, fio_command(pack, size, depth, volume))
                        rate = fio_rate(data, volume, depth)
                    elif mode == "h2d":
                        data = record(output, label, [str(h2d), "--bytes", str(size),
                                      "--warmup", str(warmup_volume // size),
                                      "--iterations", str(volume // size), "--json"])
                        if (data.get("status") != "ok" or data.get("payload_bytes") != size or
                                data.get("iterations") != volume // size or
                                data.get("verification_bytes", 0) <= 0):
                            raise ValueError("H2D scope/volume/verification mismatch")
                        rate = data["bandwidth_bytes_per_second"]
                    else:
                        data = record(output, label, [str(native), mode, str(pack), str(size),
                                      str(warmup_volume // size), str(volume // size)])
                        if (data.get("status") != "ok" or data.get("physical_io_bytes") != volume or
                                data.get("logical_bytes") != volume or data.get("verified") is not True):
                            raise ValueError("native volume/verification mismatch")
                        rate = data["bandwidth_bytes_per_second"]
                    summarize([rate])
                    rates.setdefault(f"{mode}-{size}", []).append(rate)
                    print(f"{label}: {rate / 1e9:.4f} GB/s", flush=True)
            record(output, f"gpu-after-r{repeat}", snapshot, decode=False)
    summary = {"scope": "bounded synthetic transfer baseline; no model or overlap claim",
               "smoke": args.smoke, "acceptance_evidence": False,
               "timed_bytes_per_trial": volume, "warmup_bytes_per_trial": warmup_volume,
               "results": {key: summarize(values) for key, values in rates.items()},
               "note": "fio1 matches native QD1; fio8 and queued-event H2D are denominators only; "
                       "native rates include CRC, dispatch and, for bridge, allocation/free"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.json"))}
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence: {output}; temporary synthetic pack removed", flush=True)


if __name__ == "__main__":
    main()
