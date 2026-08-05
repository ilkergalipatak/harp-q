"""Host discovery plus explicitly activated, bounded handoff microbenchmarks."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .types import HardwareProfile


@dataclass(frozen=True, slots=True)
class CommandProbe:
    """Captured output from a bounded, read-only system probe."""

    command: tuple[str, ...]
    available: bool
    returncode: int | None
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_probe(command: Sequence[str], timeout_s: float = 5.0) -> CommandProbe:
    executable = shutil.which(command[0])
    normalized = tuple(command)
    if executable is None:
        return CommandProbe(normalized, False, None, "", f"{command[0]} not found")
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandProbe(normalized, True, None, "", str(exc))
    return CommandProbe(
        normalized,
        True,
        result.returncode,
        result.stdout.strip(),
        result.stderr.strip(),
    )


def _executable_probe(name: str) -> CommandProbe:
    """Report whether a helper exists without executing it."""

    executable = shutil.which(name)
    return CommandProbe(
        command=(name,),
        available=executable is not None,
        returncode=None,
        stdout=executable or "",
        stderr="" if executable else f"{name} not found",
    )


def _io_uring_available() -> bool:
    if platform.system() != "Linux":
        return False
    syscall_number = {
        "x86_64": 425,
        "amd64": 425,
        "aarch64": 425,
        "arm64": 425,
    }.get(platform.machine().lower())
    if syscall_number is None:
        return False

    class SqOffsets(ctypes.Structure):
        _fields_ = [
            ("head", ctypes.c_uint32),
            ("tail", ctypes.c_uint32),
            ("ring_mask", ctypes.c_uint32),
            ("ring_entries", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("dropped", ctypes.c_uint32),
            ("array", ctypes.c_uint32),
            ("resv1", ctypes.c_uint32),
            ("user_addr", ctypes.c_uint64),
        ]

    class CqOffsets(ctypes.Structure):
        _fields_ = [
            ("head", ctypes.c_uint32),
            ("tail", ctypes.c_uint32),
            ("ring_mask", ctypes.c_uint32),
            ("ring_entries", ctypes.c_uint32),
            ("overflow", ctypes.c_uint32),
            ("cqes", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("resv1", ctypes.c_uint32),
            ("user_addr", ctypes.c_uint64),
        ]

    class IoUringParams(ctypes.Structure):
        _fields_ = [
            ("sq_entries", ctypes.c_uint32),
            ("cq_entries", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("sq_thread_cpu", ctypes.c_uint32),
            ("sq_thread_idle", ctypes.c_uint32),
            ("features", ctypes.c_uint32),
            ("wq_fd", ctypes.c_uint32),
            ("resv", ctypes.c_uint32 * 3),
            ("sq_off", SqOffsets),
            ("cq_off", CqOffsets),
        ]

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        parameters = IoUringParams()
        file_descriptor = int(libc.syscall(syscall_number, 1, ctypes.byref(parameters)))
        if file_descriptor < 0:
            return False
        os.close(file_descriptor)
        return True
    except (OSError, AttributeError):
        return False


def _cuda_driver_available() -> bool:
    """Verify that the current user can initialize the CUDA driver and see a device."""

    if platform.system() != "Linux":
        return False
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        return driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False


def basic_profile() -> HardwareProfile:
    """Return the cross-platform capability subset required by every run."""

    return HardwareProfile(
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        python=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        cuda_available=_cuda_driver_available(),
        io_uring_available=_io_uring_available(),
        # O_DIRECT support is filesystem-specific and is only asserted after the active fio probe.
        odirect_available=False,
    )


def collect_hardware_report() -> dict[str, Any]:
    """Collect a deterministic, read-only report without running bandwidth benchmarks.

    Bandwidth and latency microbenchmarks belong to the Linux benchmark phase. This function only
    records capabilities and device metadata, so it is safe to run during Mac development.
    """

    profile = basic_profile()
    probes = {
        "nvidia_smi": _run_probe(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,pcie.link.gen.current,"
                "pcie.link.width.current",
                "--format=csv,noheader,nounits",
            )
        ),
        "cuda_compiler": _run_probe(("nvcc", "--version")),
        "cuda_device_query": _executable_probe("deviceQuery"),
        "cuda_bandwidth_test": _executable_probe("bandwidthTest"),
        "block_devices": _run_probe(
            ("lsblk", "--json", "--output", "NAME,TYPE,SIZE,ROTA,MODEL,TRAN,MOUNTPOINTS")
        ),
        "filesystems": _run_probe(("findmnt", "--json", "--real")),
        "memory": _run_probe(("cat", "/proc/meminfo"))
        if platform.system() == "Linux"
        else _run_probe(("sysctl", "-n", "hw.memsize")),
    }
    return {
        "schema_version": 1,
        "profile": profile.to_dict(),
        "probes": {name: probe.to_dict() for name, probe in probes.items()},
        "measurement_status": {
            "ram_bandwidth": "not_run",
            "nvme_bandwidth": "not_run",
            "h2d_bandwidth": "not_run",
            "cuda_copy_engines": "requires CUDA deviceQuery on Linux",
        },
        "capability_evidence": {
            "cuda_available": "successful cuInit + cuDeviceGetCount",
            "io_uring_available": "successful one-entry io_uring_setup syscall",
            "odirect_available": "unverified until active target-filesystem fio probe",
        },
    }


def measure_ram_copy_bandwidth(
    *,
    buffer_bytes: int = 64 * 1024 * 1024,
    repetitions: int = 3,
) -> dict[str, float | int]:
    """Run a bounded NumPy copy microbenchmark.

    Both payload bandwidth and estimated memory traffic (one read plus one write) are reported so
    benchmark reports cannot silently mix the two conventions.
    """

    if isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes <= 0:
        raise ValueError("buffer_bytes must be a positive integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    element_count = max(1, buffer_bytes // np.dtype(np.float32).itemsize)
    source = np.arange(element_count, dtype=np.float32)
    destination = np.empty_like(source)
    np.copyto(destination, source)
    started = time.perf_counter_ns()
    for _ in range(repetitions):
        np.copyto(destination, source)
    elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000.0
    transferred = source.nbytes * repetitions
    gib = float(1024**3)
    return {
        "buffer_bytes": int(source.nbytes),
        "repetitions": repetitions,
        "elapsed_s": elapsed_s,
        "payload_gib_s": transferred / elapsed_s / gib,
        "read_plus_write_gib_s": 2 * transferred / elapsed_s / gib,
    }


def _fio_direct_report(scratch_directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Run bounded sequential O_DIRECT write/read jobs in a dedicated temporary directory."""

    if platform.system() != "Linux":
        return {"status": "unsupported", "reason": "fio/O_DIRECT probe requires Linux"}
    scratch = Path(scratch_directory).resolve()
    if not scratch.is_dir():
        raise ValueError(f"scratch directory does not exist: {scratch}")
    filesystem = _run_probe(
        (
            "findmnt",
            "--json",
            "--target",
            str(scratch),
            "--output",
            "SOURCE,TARGET,FSTYPE,OPTIONS",
        )
    ).to_dict()
    fio = shutil.which("fio")
    safe_command = [
        "fio",
        "--output-format=json",
        "--direct=1",
        "--ioengine=io_uring",
        "--iodepth=32",
        "--bs=1M",
        "--size=256M",
        "--filename=probe.bin",
        "--name=sequential-write",
        "--rw=write",
        "--end_fsync=1",
        "--name=sequential-read",
        "--stonewall",
        "--rw=read",
    ]
    if fio is None:
        return {
            "status": "unavailable",
            "reason": "fio not found",
            "command": safe_command,
            "scratch_path": str(scratch),
            "scratch_filesystem": filesystem,
        }
    fio_version = _run_probe((fio, "--version")).to_dict()
    evidence = {
        "command": safe_command,
        "scratch_path": str(scratch),
        "scratch_filesystem": filesystem,
        "fio_version": fio_version,
    }
    temporary = Path(tempfile.mkdtemp(prefix="harpq-fio-", dir=scratch))
    command = (fio, *safe_command[1:])
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=temporary,
        )
        if result.returncode != 0:
            return {
                "status": "failed",
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
                **evidence,
            }
        decoded = json.loads(result.stdout)
        parsed = _parse_fio_output(decoded)
        parsed.update(evidence)
        return parsed
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": str(exc), **evidence}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _parse_fio_output(decoded: object) -> dict[str, Any]:
    """Validate just enough fio JSON before it can become capability evidence."""

    if not isinstance(decoded, Mapping):
        return {"status": "failed", "reason": "fio JSON root must be an object"}
    jobs = decoded.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2 or not all(
        isinstance(job, Mapping) for job in jobs
    ):
        return {"status": "failed", "reason": "fio must return exactly two job objects"}
    expected_names = {"sequential-write", "sequential-read"}
    names = [job.get("jobname") for job in jobs]
    if not all(isinstance(name, str) for name in names):
        return {"status": "failed", "reason": "fio job names must be strings"}
    jobs_by_name = {name: job for name, job in zip(names, jobs, strict=True)}
    if set(jobs_by_name) != expected_names:
        return {"status": "failed", "reason": "fio returned unexpected or duplicate job names"}

    summaries: list[dict[str, Any]] = []
    for name, direction in (("sequential-write", "write"), ("sequential-read", "read")):
        job = jobs_by_name[name]
        if job.get("error") != 0:
            return {"status": "failed", "reason": f"fio job {name} reported an error"}
        metrics = job.get(direction)
        if not isinstance(metrics, Mapping):
            return {"status": "failed", "reason": f"fio job {name} lacks {direction} metrics"}
        transferred = metrics.get("io_bytes")
        kilobytes = metrics.get("io_kbytes")
        if transferred is None and isinstance(kilobytes, int) and not isinstance(kilobytes, bool):
            transferred = kilobytes * 1024
        bandwidth = metrics.get("bw_bytes")
        if (
            isinstance(transferred, bool)
            or not isinstance(transferred, int)
            or transferred != 256 * 1024 * 1024
        ):
            return {
                "status": "failed",
                "reason": f"fio job {name} did not transfer the expected 256 MiB",
            }
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, (int, float))
            or not np.isfinite(float(bandwidth))
            or bandwidth <= 0
        ):
            return {"status": "failed", "reason": f"fio job {name} reported no bandwidth"}
        summaries.append(
            {
                "name": name,
                "direction": direction,
                "bytes": int(transferred),
                "bytes_per_second": float(bandwidth),
            }
        )
    return {"status": "ok", "jobs": summaries}


def collect_active_linux_report(
    scratch_directory: str | os.PathLike[str],
    *,
    ram_buffer_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Collect the Linux handoff report plus bounded RAM, NVMe, and CUDA probes."""

    if platform.system() != "Linux":
        raise RuntimeError("active Linux profiling must run on the NVIDIA/Linux host")
    report = collect_hardware_report()
    report["active_measurements"] = {
        "ram_copy": measure_ram_copy_bandwidth(buffer_bytes=ram_buffer_bytes),
        "nvme_direct": _fio_direct_report(scratch_directory),
        "cuda_device_query": _run_probe(("deviceQuery",), timeout_s=30.0).to_dict(),
        "cuda_pinned_h2d": _run_probe(
            ("bandwidthTest", "--memory=pinned", "--mode=quick"), timeout_s=60.0
        ).to_dict(),
    }
    report["profile"]["odirect_available"] = (
        report["active_measurements"]["nvme_direct"]["status"] == "ok"
    )
    nvme_status = report["active_measurements"]["nvme_direct"]["status"]
    report["capability_evidence"]["odirect_available"] = (
        "verified by serialized fio direct I/O on the recorded scratch filesystem"
        if nvme_status == "ok"
        else f"not verified; active fio status is {nvme_status!r}"
    )
    report["measurement_status"] = {
        "ram_bandwidth": "run",
        "nvme_bandwidth": report["active_measurements"]["nvme_direct"]["status"],
        "h2d_bandwidth": "see cuda_pinned_h2d probe",
        "cuda_copy_engines": "see cuda_device_query asyncEngineCount",
    }
    return report


def write_hardware_report(
    path: str | os.PathLike[str],
    *,
    active_scratch: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the report atomically enough for a single-process handoff workflow."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    report = (
        collect_active_linux_report(active_scratch)
        if active_scratch is not None
        else collect_hardware_report()
    )
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
