# Native Ubuntu host baseline

Last verified: 2026-08-09. This is a small reproducibility summary; the raw passive and active
JSON reports remain ignored artifacts on the measured host.

## Host

- OS: Ubuntu 24.04.4 LTS, native x86-64
- Kernel: `6.11.0-28-generic`
- CPU: AMD Ryzen 9 7950X, 16 cores / 32 threads
- RAM: 61 GiB visible, 3.3 GiB swap
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB VRAM, compute capability 8.9
- NVIDIA driver: `580.173.02`
- CUDA toolkit compiler: `nvcc 12.0.140`
- Host compiler candidates: GCC/G++ 13.3 default; GCC/G++ 12.4 installed for CUDA 12.0
- Build tools: CMake 3.28.3, Ninja 1.11.1, Git 2.43.0
- Direct-I/O tool: fio 3.36; liburing 2.5

The GPU also drives the desktop. The first inventory observed roughly 291 MiB of VRAM in use by
Xorg, GNOME, Firefox, and Nautilus. Performance runs must record or eliminate this background load.

## Target storage

The initial scratch path resolves to ext4 on `/dev/nvme0n1p4`, backed by a Samsung SSD 980 PRO
2 TB. The mounted root filesystem exposes 1.5 TB, with approximately 1.3 TB free during the run.
Other installed NVMe devices and unmounted partitions were not modified.

## Verified reference gates

- The repository was clean at commit `0641d1e9c722e3eba287065b3fa8c964d490960d`.
- All 62 Python CPU/reference tests passed on Python 3.12.3 and NumPy 2.3.5.
- The CUDA driver probe completed with a positive device count.
- A one-entry `io_uring_setup` syscall succeeded.
- A bounded active fio run verified `io_uring + O_DIRECT` on the target ext4 filesystem.
- A standalone CUDA 12.0 build targeted `sm_89` and matched the Python-generated multi-tile fixture
  for every prefix from zero through four. Maximum absolute error was zero for this fixture.

## Bounded smoke measurements

The active profiler used two 64 MiB NumPy buffers and serialized 256 MiB direct fio jobs with a
1 MiB block size and queue depth 32:

| Probe | Result |
|---|---:|
| RAM copy payload | 25.60 GiB/s |
| RAM read + write accounting | 51.21 GiB/s |
| Direct sequential write | 4.07 GB/s |
| Direct sequential read | 5.59 GB/s |

These values verify capability and provide an initial denominator only. They are not HARP-Q
runtime throughput, cold-cache evidence, tail-latency evidence, or a publication result. The CUDA
SDK sample executables `deviceQuery` and `bandwidthTest` were not installed, so pinned-H2D and copy
engine measurements remain pending. HARP-Q will add native, size-matched microbenchmarks instead of
depending on those samples.

## Toolchain note

The installed driver advertises CUDA 13.0 compatibility while the Ubuntu toolkit is CUDA 12.0.
The standalone oracle build uses GCC/G++ 12 explicitly and targets `sm_89`. Before the
PyTorch custom-op phase, select and lock one mutually compatible PyTorch, CUDA toolkit, compiler,
and C++ ABI set; do not treat driver compatibility alone as a complete build-toolchain contract.
