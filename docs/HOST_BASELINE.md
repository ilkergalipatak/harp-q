# Native Ubuntu host baseline

Latest correctness verification and bounded SSD/H2D sweep: 2026-10-30 Europe/Istanbul
(2026-10-29 UTC). Historical August measurements remain attributed separately below.
This is a small reproducibility summary; raw passive, active, and microbenchmark outputs remain
ignored artifacts on the measured host.

The historical August performance measurements were rebuilt from clean source commit
`2a6c481a077fd11e46fc5e68a6da10b78389a9d5`. The newer packet-ring and `pread -> H2D`
correctness evidence was rebuilt separately from clean source commit
`9cc7dc593ab8c650dd7d21a41dd31e4b2492e246`. Both commits are on branch `main`; no old
performance number is relabeled as a result from the newer runtime.

On 2026-10-29, clean commit `44eac3e6e81ac824ca8929ca65dd89d8934d6fee` passed all 74 Python tests,
four CPU/direct-I/O tests with CUDA disabled, and all nine native tests with CUDA and io_uring
explicitly enabled. The target remained ext4 on `/dev/nvme0n1p4`, with RTX 4090, driver 580.173.02,
CUDA 12.0.140, G++ 12.4.0, and liburing 2.5. Exact commands and CTest log hashes are in
`HANDOFF.md`. This correctness run did not repeat the performance measurements below.

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

## Repeated SSD/H2D baseline — 2026-10-30

Source: clean `decdf707f4963b0aefae83d9252b304ef8728cc5`, CMake Release (`-O3 -DNDEBUG`), G++ 12.4,
CUDA 12.0.140, architecture 89, CUDA/io_uring explicitly ON. Runtime algorithms are unchanged from
the fifth slice; this commit adds measurement tools. Before measurement, the byte-identical
verification source passed 78 Python tests, all ten native tests, and the new smoke runner's
invalid-size/volume/mode and corrupt-CRC checks. Both Mac and Linux passed the Python suite.

The scratch directory was on ext4 `/dev/nvme0n1p4` (`rw,relatime`), Samsung SSD 980 PRO 2 TB.
The runner created/fsynced one 256 MiB synthetic pattern file and removed it afterward. No
existing pack, raw device, clock policy or user process was modified. No model, model batch,
context or KV cache was involved; the fixture is deterministic (`byte[index] = index % 256`).
Each case uses 16 MiB warm-up and 256 MiB timed volume, five repetitions, and alternating method/
size order. Native direct/bridge use one slot and QD1; fio uses QD1 or QD8; isolated H2D queues
copies on one stream with CUDA-event timing. See `BENCHMARK_PROTOCOL.md` for endpoint differences.

All rates are decimal **GB/s**, median **[min–max]** of five trials:

| Payload | fio QD1 | fio QD8 | Native direct + CRC, QD1 | Native SSD→GPU, QD1 | Isolated pinned H2D |
|---|---|---|---|---|---|
| 64 KiB | 2.049 [1.688–2.065] | 4.067 [3.948–4.474] | 0.503 [0.353–0.515] | 0.306 [0.304–0.307] | 18.469 [18.326–18.490] |
| 256 KiB | 2.632 [2.632–2.658] | 5.478 [5.369–5.711] | 0.524 [0.519–0.562] | 0.470 [0.469–0.471] | 24.134 [24.104–24.181] |
| 1 MiB | 3.532 [3.532–3.532] | 5.478 [5.478–5.592] | 0.582 [0.581–0.583] | 0.542 [0.539–0.544] | 25.925 [25.919–25.989] |

fio reports elapsed read time at millisecond resolution; the five 1 MiB/QD1 trials all rounded to
76 ms. Identical displayed rates are not proof of zero variance. Larger repetitions/volumes and
controlled background load are required for fine-grained performance claims. The 64 KiB native
direct case varied substantially; keep its full range instead of discarding the slow first run.

Bridge latency audit, median of each trial's statistic (microseconds):

| Payload | Critical p50 | Critical p95 | Mean read/CRC/dispatch | Mean host-observed H2D | Mean enqueue call |
|---|---:|---:|---:|---:|---:|
| 64 KiB | 183.837 | 197.526 | 138.438 | 8.722 | 42.367 |
| 256 KiB | 525.290 | 546.550 | 471.779 | 16.157 | 44.120 |
| 1 MiB | 1898.377 | 1916.536 | 1805.812 | 52.531 | 45.857 |

The enqueue call includes allocation and asynchronous submission; it overlaps the H2D observation
interval. Do not add these columns or interpret them as CUDA-event bandwidth. Whole-loop
throughput includes free/cleanup; per-packet critical-path latency ends before that cleanup.

Inference: the current read/CRC/dispatch stage is the leading optimization target, not proof that
PCIe is saturated. The portable byte-at-a-time CRC implementation is a candidate that must be
profiled separately before attributing the entire gap to it. Native-direct/fio-QD1 median ratios
are about 24.5%, 19.9% and 16.5%: well below the 75% target in this diagnostic workload. No
throughput acceptance gate is closed. Queued H2D is not a valid 80% denominator for the differently
timed serial runtime component. All synthetic reads had logical=physical bytes, one SQE/CQE and
read amplification 1; this does not establish amplification for real model packet layouts.
STATX_DIOALIGN reported memory alignment 4 and offset alignment 512; ring alignment was 4096.

GPU snapshots: about 637 MiB occupied by the desktop, 0–7% utilization at snapshot times, 40–45°C,
SM clocks 210–2565 MHz, PCIe generation 1/2/4 at x16. Clocks were not locked; snapshot values are
not a continuous timeline. No overlap, compute kernel, cold-cache or model-throughput claim is made.

Reproduce from the repository root:

```sh
cmake -S . -B build/benchmark-decdf70 -DHARPQ_ENABLE_CUDA=ON -DHARPQ_ENABLE_IO_URING=ON \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/benchmark-decdf70 --target harpq_transfer_benchmark harpq_h2d_benchmark --parallel 4
PYTHONPATH=src .venv/bin/python -m harpq profile-host --output artifacts/host-transfer-decdf70.json
.venv/bin/python scripts/benchmark_transfers.py --build build/benchmark-decdf70 \
  --scratch "$PWD/build/benchmark-decdf70" --output benchmarks/results/transfer-decdf70-20261030
```

Choose a new output directory for reruns; the tool refuses to overwrite it. Raw records remain on
Ubuntu in that ignored result directory: 117 JSON artifacts were checksum-verified. `sha256.json`
lists each command/stdout/stderr record, binary hash record, snapshot and summary. Main digests:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `1c4323062b4fdcd524be3712e915b170fb64c023817a6a7d19d9652b34442e8c` |
| `sha256.json` | `ff808c0c178fb305aa72f1b5d9223da1cc1b898905885defdb4381aa1c18fdcf` |
| `artifacts/host-transfer-decdf70.json` | `4bff2b2c964dc99aebdfaf36a36d939eb9a7e4400ceb5021165ce3193fd3bba4` |
| `harpq_transfer_benchmark` binary | `af7ba6d8ce7a806d4449466ba48512494568aca3895c404edbf1cd2831dbf446` |
| `harpq_h2d_benchmark` binary | `f2450018d5d2fdc61095a431820d74c6b186075bb07efa6ee5a5af6d5f645b8c` |

## Verified reference gates — 2026-08-09

- The runtime repository for that run was clean at commit
  `9cc7dc593ab8c650dd7d21a41dd31e4b2492e246`.
- All 67 Python CPU/reference tests passed on Python 3.12.3 and NumPy 2.3.5.
- A clean CUDA-OFF CMake run passed stage codec, packet-ring, and buffered-`pread` tests and
  recorded five explicit CUDA skips.
- A clean CUDA-ON `sm_89` CMake run passed all eight tests: the three CPU tests,
  materialize-plus-cuBLAS oracle, fused progressive GEMV, registered CUDA packet ring, H2D smoke,
  and exploratory GEMV benchmark smoke.
- `native.cuda_packet_ring` verified byte equality along the canonical padded path from buffered
  pread, through a registered host slot and async H2D, to a compute-stream event wait. It also
  covered pending-slot non-reuse, facade/lease teardown order, and exact-or-fail registration.
  Injected event/stream/device/unregister failures exercised the fail-closed policy; unprovable DMA
  safety and failed unregister terminated with `SIGABRT` rather than recycle or free the slot.
- CUDA `AUTO` with the GPU hidden passed the CPU tests and explicitly skipped all five CUDA tests;
  CUDA `ON` with the GPU hidden failed as required.
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

The later registered-ring correctness gate used the clean `9cc7dc5` clone:

```sh
make check PYTHON=.venv/bin/python \
  NATIVE_BUILD_DIR=build/evidence-9cc7dc5-cpu NATIVE_CUDA=OFF

cmake -S . -B build/evidence-9cc7dc5-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/evidence-9cc7dc5-cuda --parallel
ctest --test-dir build/evidence-9cc7dc5-cuda --output-on-failure
```

This was a functional gate only; it produced no new performance claim or benchmark artifact. The
same commit's [CPU reference workflow](https://github.com/ilkergalipatak/harp-q/actions/runs/31330702569)
completed successfully.

## Toolchain note

The installed driver advertises CUDA 13.0 compatibility while the Ubuntu toolkit is CUDA 12.0.
The standalone oracle build uses GCC/G++ 12 explicitly and targets `sm_89`. Before the
PyTorch custom-op phase, select and lock one mutually compatible PyTorch, CUDA toolkit, compiler,
and C++ ABI set; do not treat driver compatibility alone as a complete build-toolchain contract.
