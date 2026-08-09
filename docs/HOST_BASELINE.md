# Native Ubuntu host baseline

Last verified: 2026-08-09. This is a small reproducibility summary; raw passive, active, and
microbenchmark outputs remain ignored artifacts on the measured host.

CUDA correctness and microbenchmark evidence below was rebuilt from clean source commit
`2a6c481a077fd11e46fc5e68a6da10b78389a9d5` on branch `main`.

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

- The repository was clean at commit `2a6c481a077fd11e46fc5e68a6da10b78389a9d5`.
- All 63 Python CPU/reference tests passed on Python 3.12.3 and NumPy 2.3.5.
- A clean CUDA-OFF CMake run passed the native CPU test and recorded four explicit CUDA skips.
- A clean CUDA-ON `sm_89` CMake run passed all five tests: stage codec, materialize-plus-cuBLAS
  oracle, fused progressive GEMV, H2D smoke, and exploratory GEMV benchmark smoke.
- The CUDA driver probe completed with a positive device count.
- A one-entry `io_uring_setup` syscall succeeded.
- A bounded active fio run verified `io_uring + O_DIRECT` on the target ext4 filesystem.
- A standalone CUDA 12.0 build targeted `sm_89` and matched the Python-generated multi-tile fixture
  for every prefix from zero through four. Maximum absolute error was zero for this fixture.
- The batch-one fused direct-decode GEMV matched both the checked Python vectors and the cuBLAS
  oracle at every prefix, including one-hot coverage of every logical weight, base-only operation,
  bias, two-dimensional edge padding, `K=257/513/4096`, and raw FP16 subnormal/max-finite scales.

## Bounded smoke measurements

The active profiler used two 64 MiB NumPy buffers and serialized 256 MiB direct fio jobs with a
1 MiB block size and queue depth 32:

| Probe | Result |
|---|---:|
| RAM copy payload | 25.60 GiB/s |
| RAM read + write accounting | 51.21 GiB/s |
| Direct sequential write | 4.07 GB/s |
| Direct sequential read | 5.59 GB/s |
| Registered pinned H2D | 26.8710 GB/s (25.0256 GiB/s) |

These values verify capability and provide an initial denominator only. They are not HARP-Q
runtime throughput, cold-cache evidence, tail-latency evidence, or a publication result. The CUDA
SDK sample executables `deviceQuery` and `bandwidthTest` were not installed, so HARP-Q used its own
bounded native H2D microbenchmark. That run used a 64 MiB page-aligned allocation registered with
`cudaHostRegister`, 10 warm-up copies, 100 timed copies, CUDA events on a nonblocking stream, and a
12 KiB sampled D2H checksum. Its elapsed time was 249.7444763 ms for the timed volume. A future
size-matched runtime path must reach at least 21.4968 GB/s to satisfy the predeclared 80% H2D gate;
the denominator must be rerun when payload size or background conditions change.

A single-run payload-size sweep used the same registration and event method, with enough iterations
to amortize event setup:

| H2D payload | Timed iterations | Bandwidth |
|---:|---:|---:|
| 4 KiB | 10,000 | 3.2347 GB/s |
| 8 KiB | 10,000 | 5.9374 GB/s |
| 64 KiB | 4,096 | 18.5033 GB/s |
| 256 KiB | 1,024 | 24.2314 GB/s |
| 1 MiB | 256 | 25.9842 GB/s |
| 4 MiB | 64 | 26.6177 GB/s |
| 64 MiB | 100 | 26.8619 GB/s |

This preliminary curve argues against one H2D operation per tiny tile packet: 256 KiB already
reaches roughly 90% of the 64 MiB result, while 4–8 KiB is dominated by per-copy overhead. It does
not settle the container chunk size by itself; the direct-I/O latency curve, cache waste, deadlines,
and copy/compute overlap must be measured jointly.

## Exploratory fused GEMV measurement

The source-linked run used shape `4096x4096`, tile/group `128/128`, prefix two (four stored code
bits), 10 warm-up launches per path, 100 launches per timed repeat, and seven alternating repeats.
Allocation and transfers were outside CUDA-event timing. Median fused progressive latency was
82.872324 us; the matched-naive static-W4 median was 67.3894405 us, for a throughput ratio of
0.813171844. The static-W4 output passed its deterministic CPU reduction-tree oracle with maximum
absolute error zero; the tolerance at that element was `2.07820124e-05`.

This is explicitly `acceptance_evidence=false`: the comparator is a benchmark-local naive W4
kernel, not an optimized production or library-parity baseline. Therefore the ratio neither passes
nor fails the formal 85% gate. The GPU was also driving the desktop; snapshots moved from P8/210
MHz before the run to P2/2745 MHz after it, so clocks and background load were uncontrolled.

## Exact commands and artifact checksums

The CUDA build and tests used:

```sh
cmake -S . -B build/evidence-2a6c481-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/evidence-2a6c481-cuda --parallel
ctest --test-dir build/evidence-2a6c481-cuda --output-on-failure

build/evidence-2a6c481-cuda/native/cuda/harpq_h2d_benchmark \
  --bytes 67108864 --warmup 10 --iterations 100 --json
build/evidence-2a6c481-cuda/native/cuda/harpq_gemv_benchmark \
  --rows 4096 --columns 4096 --tile 128 --group 128 --prefix 2 \
  --warmup 10 --iterations 100 --repeats 7 --json
```

Raw outputs are ignored under `benchmarks/results/` on the measured host:

| Artifact | SHA-256 |
|---|---|
| `2a6c481-h2d-64m.json` | `192e4a93e53562c67fbdf80e06f598da2e20c866251ed41c42fdb799a272ee61` |
| `2a6c481-gemv-4096-p2.json` | `3771aedb915182c6e992ff1af6d99da970f2c2d01a8f16cb986aa1f4e68a2a5f` |
| `2a6c481-h2d-size-sweep.jsonl` | `1ca5067410ef2c7001cef14eaf7e32ccc6adef8ace604fd695a475cc8cff5fe5` |
| `2a6c481-gpu-snapshot-before.txt` | `0db7066e115d2ca8a794abc8f771ac2efdb42832f35f1422bd3c51640966dbf6` |
| `2a6c481-gpu-snapshot-after.txt` | `c9bf9147097ad08eefbf9222d50f6013f150c4a5397a4bcc926b88122c00e04d` |

## Toolchain note

The installed driver advertises CUDA 13.0 compatibility while the Ubuntu toolkit is CUDA 12.0.
The standalone oracle build uses GCC/G++ 12 explicitly and targets `sm_89`. Before the
PyTorch custom-op phase, select and lock one mutually compatible PyTorch, CUDA toolkit, compiler,
and C++ ABI set; do not treat driver compatibility alone as a complete build-toolchain contract.
