# HARP-Q handoff

Latest Linux verification: 2026-10-30 Europe/Istanbul (2026-10-29 UTC) on the native Ubuntu 24.04
RTX 4090 host. Tested source matches commit `decdf70` byte-for-byte. All 78 Python tests and ten
CPU/CUDA tests passed. A five-repeat SSD/H2D/native-path baseline is recorded in `HOST_BASELINE.md`;
the earlier direct-H2D bridge also passed UBSan and measured-sample replay.
Earlier Mac and performance evidence is dated separately below. This file is the recovery source if the
Codex task cannot be handed off directly.

## Repository

- Remote: `https://github.com/ilkergalipatak/harp-q.git`
- Branch: `main`
- Latest validated source / SSD-H2D baseline: `decdf707f4963b0aefae83d9252b304ef8728cc5`
- Historical GEMV/performance baseline: `2a6c481a077fd11e46fc5e68a6da10b78389a9d5`
- Python package: `harpq`, Python 3.11+
- Native floor: CMake 3.24+, C++17; CUDA toolkit 11.8+ when enabled
- Validated Mac runtime: Python 3.12.14, NumPy 2.3.5
- Validated Ubuntu runtime: Python 3.12.3, NumPy 2.3.5
- Dependency locks: `requirements/runtime.lock`, `requirements/build.lock`

The repository-relative layout is portable. SSH keys, host addresses, model weights, generated
packs, raw profiles, and benchmark artifacts are intentionally absent from Git.

## Implemented state

Python remains the semantic and offline oracle:

- `quantization.py`: deterministic tile-capable signed-int2 `2+2+2+2` residual quantization,
  packing, prefix decode, and progressive matmul.
- `distortion.py`: activation-weighted MSE, empirical Hessian proxy/error, stable KL, and prefix
  profiles.
- `integrity.py`, `container.py`, `matrix_codec.py`: CRC32C/SHA-256, versioned manifest/binary
  index, 4 KiB pack alignment, stage codec, dependency checks, and optional fallback.
- `allocator.py`: exact precedence-aware multiple-choice knapsack reference.
- `scheduler.py`, `simulation.py`: deadline/cache policy, dependency-safe eviction, physical versus
  logical byte accounting, deterministic trace serialization, and exact replay.
- `synthetic.py`: small deterministic Transformer for prefix functional accuracy.
- `hardware.py`: passive capability report and bounded opt-in Linux RAM/fio/CUDA probes with target
  filesystem and fio provenance.
- `cli.py`: `doctor`, `profile-host`, `demo`, `quantize`, `inspect`, `simulate`, and `replay`.

The first three native correctness/performance-bring-up slices are validated:

- `native/src/stage_codec.cpp`: dependency-free little-endian stage payload decoder, manual FP16
  conversion, signed-int2 prefix dequantization, CPU progressive-linear oracle, geometry checks,
  and incremental CRC32C.
- `scripts/generate_native_golden.py`: deterministic Python-to-C++ golden generator. Its checked
  fixture uses original shape `5x9`, padded shape `8x16`, tiles `4x8`, group size 4, a `2x2` tile
  grid, edge padding, four stages, and prefixes zero through four.
- `native/src/cuda_oracle.cu`: correctness-first CUDA stage materialization plus pedantic cuBLAS
  SGEMM. It remains the differential oracle, not a throughput implementation.
- `native/src/progressive_gemv.cu`: batch-one asynchronous fused GEMV. It decodes a contiguous
  signed-int2/FP16 residual prefix directly from device views, accumulates in FP32, and never
  materializes an FP32 weight matrix. Pointer/count contracts and buffer lifetimes are explicit.
- `native/benchmarks/h2d_benchmark.cu`: bounded, page-aligned `cudaHostRegister` H2D denominator
  with CUDA-event timing and sampled D2H checksum verification.
- `native/benchmarks/gemv_benchmark.cu`: exploratory direct-decode versus matched-naive W4 timing.
  The W4 path is CPU-oracle checked and the JSON explicitly says `acceptance_evidence=false`.
- `native/src/packet_ring.cpp`: contiguous page-aligned slots with distinct format-I/O and host
  allocation alignment, generation-tagged move-only leases, and deterministic
  `free -> filling -> ready -> H2D-in-flight -> free` transitions.
- `native/src/packet_io_posix.cpp`: synchronous buffered `pread` correctness backend with exact
  extent validation, `EINTR`/partial/EOF handling, cancellation, logical CRC32C, required-versus-
  optional failure policy, backpressure, and explicit legacy-unpadded-tail telemetry.
- `native/src/cuda_packet_ring.cpp`: exact-or-fail whole-ring `cudaHostRegister`, one owned
  nonblocking copy stream, per-slot events, a pread-to-H2D compute-stream wait, and shared CUDA
  ownership across filling/ready/in-flight leases. Unprovable DMA safety or failed unregister is
  fail-closed with `SIGABRT`; memory is never silently recycled or freed.
- The canonical v1 Python writer now zero-pads each non-empty pack tail to 4096 bytes while readers
  retain buffered compatibility with old unpadded terminal tails.
- Root/native CMake: `HARPQ_ENABLE_CUDA=AUTO|ON|OFF`; explicit `ON` fails without CUDA, while a
  CPU-only test run records all six CUDA-dependent tests as explained CTest skips. CUDA architectures
  supplied through CMake or `CUDAARCHS` are preserved; otherwise the project defaults to `sm_89`.
- CPU CI runs the Python suite, CLI smoke, generated-fixture drift check, and native CPU CTest on
  macOS and Ubuntu. The Linux job installs liburing and builds the direct-I/O adapter in `AUTO`
  mode; macOS records a reasoned direct-I/O skip. Both jobs record explicit CUDA skips.

The fourth slice passed its Linux correctness gate on 2026-10-29:

- `native/src/direct_packet_io_linux.cpp`: Linux CPU-only `io_uring + O_DIRECT` adapter using strict
  `STATX_DIOALIGN`, bounded ordinary reads into `PacketRing` slots, cooperative cancellation,
  logical CRC, and a distinct legacy `buffered-only` outcome. It has no fixed-buffer registration,
  fork support, or throughput claim. The fifth slice composes it with CUDA ownership below.
- `scheduler.py` trace v2: caller-supplied ordered `ServiceObservation` values, exact
  `(service_class, logical_bytes, physical_io_bytes)` rolling-p95 buckets, auditable estimate
  sources, and unchanged v1 replay. I/O/H2D component durations are audit-only; the observed
  critical path drives deadlines.

The fifth slice adds a lifetime-safe storage-to-device path:

- `CudaDirectPackFile` retains a private CUDA ring facade and drains outstanding original CQEs
  before releasing its registration owner. Ready packets remain usable after both caller facades
  are destroyed. Required/optional/backpressure/cancellation semantics remain those of the reader.
- `CudaPacketTransfer` owns the destination allocation and H2D lease; destruction proves DMA
  completion before releasing either. Consumers must retain it through downstream compute.
- Host-observed storage-submission-to-device-ready durations use service class
  `direct-h2d/host-observed-v1`; failed or abandoned transfers emit no successful sample. The caller
  supplies global sequence IDs when exporting. CPU polling, CRC, dispatch and per-upload allocation
  remain part of the critical path; these are not isolated device-event timings.
- `scripts/check_native_feedback.py` checks the native fixture's nine observations against exact
  byte/class buckets, default p95 selection and deterministic trace round-trip/replay.

The measurement slice adds `harpq_transfer_benchmark` and `scripts/benchmark_transfers.py` without
changing runtime algorithms: bounded QD1 direct/bridge timing, fio QD1/QD8, and isolated pinned H2D.
It records five-repeat medians/ranges, raw commands, snapshots and hashes; no acceptance/model or
overlap claim. Full settings and current results are in `HOST_BASELINE.md`.

Architecture decisions are in `DECISIONS.md`; file/layout contracts are in `FORMAT.md`; benchmark
gates are in `BENCHMARK_PROTOCOL.md`; the measured host summary is in `HOST_BASELINE.md`.

## Verified commands and results

### 2026-08-09 — previous baseline

On the original Mac, using the bundled interpreter:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/ilkergalipatak/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest discover -s tests -v

mkdir -p build/native-direct
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -I native/include -I native/tests \
  native/src/stage_codec.cpp native/tests/stage_codec_test.cpp \
  -o build/native-direct/stage_codec_test
build/native-direct/stage_codec_test

c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -pthread -I native/include \
  native/src/stage_codec.cpp native/src/packet_ring.cpp \
  native/src/packet_io_posix.cpp native/tests/packet_ring_test.cpp \
  -o build/native-direct/packet_ring_test
build/native-direct/packet_ring_test

c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -pthread -I native/include \
  native/src/stage_codec.cpp native/src/packet_ring.cpp \
  native/src/packet_io_posix.cpp native/tests/packet_io_test.cpp \
  -o build/native-direct/packet_io_test
build/native-direct/packet_io_test
```

Result: **67 Python tests pass**, the generated header drift check passes, and stage codec,
packet-ring, and packet-I/O tests pass with strict Apple Clang. Packet-ring and packet-I/O also
pass ASan+UBSan on this Mac. CMake is not installed locally, so the CMake-driven CPU test was
verified on Ubuntu and remains covered by macOS GitHub Actions. The runtime source commit's
[CPU reference workflow](https://github.com/ilkergalipatak/harp-q/actions/runs/31330702569)
completed successfully on its first attempt.

On the Ubuntu RTX 4090 host, the clean clone at
`9cc7dc593ab8c650dd7d21a41dd31e4b2492e246` first passed the full CPU gate:

```sh
make check PYTHON=.venv/bin/python \
  NATIVE_BUILD_DIR=build/evidence-9cc7dc5-cpu NATIVE_CUDA=OFF
```

Result: 67 Python tests, CLI doctor, golden drift, stage codec, packet ring, and buffered `pread`
pass; the five CUDA-only CTest entries are skipped with an explicit reason. CUDA was then
configured explicitly:

```sh
cmake -S . -B build/evidence-9cc7dc5-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/evidence-9cc7dc5-cuda --parallel
ctest --test-dir build/evidence-9cc7dc5-cuda --output-on-failure
```

Result: all eight tests pass: the three CPU tests plus `native.cuda_oracle`,
`native.progressive_gemv`, `native.cuda_packet_ring`, `native.h2d_smoke`, and
`native.gemv_benchmark_smoke`. In addition to prior prefix/kernel coverage, the ring test covers
registered-host pointer identity, byte-exact H2D, pending-slot non-reuse, generation reuse,
facade-before-lease destruction, synchronous `pread` integration, compute-stream event ordering,
recoverable CUDA failures, and fail-closed `SIGABRT` subprocesses. `AUTO` with a hidden GPU marks
all five CUDA tests skipped; `ON` with a hidden GPU fails. The host reports RTX 4090 / compute
capability 8.9, two async engines, CUDA runtime 12.0, and driver API compatibility 13.0.

The source-linked 64 MiB registered-H2D run measured 26.8710 GB/s. The exploratory `4096x4096`,
prefix-two GEMV run measured 82.872324 us versus 67.3894405 us for its matched-naive W4 helper, a
ratio of 0.813171844. The W4 helper matched its deterministic CPU oracle with maximum error zero.
This is not the formal 85% gate: the denominator is deliberately marked non-production and GPU
clocks were uncontrolled. Exact commands, raw artifact SHA-256 values, and the size sweep are in
`HOST_BASELINE.md`.

A separate CUDA-off CMake build passes all three CPU tests and marks all five CUDA tests skipped
with an explicit reason. Apple Clang ASan+UBSan passes packet ring and packet I/O; CUDA-ring UBSan
passes on Ubuntu. CUDA-ring ASan fails during CUDA runtime device discovery with
`cudaErrorMemoryAllocation`, so it is not valid sanitizer evidence. `compute-sanitizer` 12.0 is
installed, but its packaged launcher could not instrument the earlier executable
(`target application terminated before first instrumented API call`) even with the packaged
injection path; do not claim a CUDA memcheck pass until that toolchain issue is fixed.

The bounded host profile verified `io_uring + O_DIRECT` on the selected ext4/NVMe scratch
filesystem. The raw JSON reports remain ignored on that host. See `HOST_BASELINE.md` for the exact
hardware and smoke measurements.

### 2026-10-29 — fourth slice from committed source

The Ubuntu clone was clean before and after testing at
`44eac3e6e81ac824ca8929ca65dd89d8934d6fee`. This closes the previously blocked Linux rerun of the
terminal-operation ownership fix, liburing API check, strict trace-v1/v2 schema validation, and
restricted AUTO skip policy. The earlier Mac worktree passed 74 Python tests; Mac tests were not
rerun during this Linux verification.

Environment: Ubuntu 24.04.4, kernel `6.11.0-28-generic`, Python 3.12.3, NumPy 2.3.5, CMake 3.28.3,
G++ 12.4.0, CUDA toolkit 12.0.140, liburing 2.5, RTX 4090 (24,564 MiB), NVIDIA driver 580.173.02.
Both direct-I/O test directories resolved to ext4 on `/dev/nvme0n1p4`, mounted `rw,relatime`.

Run from the repository root:

```sh
CXX=/usr/bin/g++-12 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  HARPQ_DIRECT_IO_TEST_DIR="$PWD/build/evidence-44eac3e-20261029-cpu" \
  make check PYTHON=.venv/bin/python \
    NATIVE_BUILD_DIR=build/evidence-44eac3e-20261029-cpu \
    NATIVE_CUDA=OFF NATIVE_IO_URING=ON

cmake -S . -B build/evidence-44eac3e-20261029-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DHARPQ_ENABLE_IO_URING=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/evidence-44eac3e-20261029-cuda --parallel 4
HARPQ_DIRECT_IO_TEST_DIR="$PWD/build/evidence-44eac3e-20261029-cuda" \
  ctest --test-dir build/evidence-44eac3e-20261029-cuda --output-on-failure
```

Results: all 74 Python tests, CLI doctor, and golden drift passed. The CUDA-OFF build passed four
CPU/direct-I/O tests and explicitly skipped the five CUDA tests. With CUDA and io_uring both `ON`,
all nine CTest entries passed with no skips, including direct reads, the CUDA oracle, fused GEMV,
registered-ring lifetime/fault tests, and H2D/GEMV smoke tests. The liburing compile/link API check
passed in both builds.

CTest logs remain in the ignored build directories on Ubuntu:

| Log | SHA-256 |
|---|---|
| `build/evidence-44eac3e-20261029-cpu/Testing/Temporary/LastTest.log` | `0cae9380ec680d9d086791f26be8f7869d62590215eb0d6235d92b89fbb3c1c2` |
| `build/evidence-44eac3e-20261029-cuda/Testing/Temporary/LastTest.log` | `1c61d87aa6c2c96537712d44fcb39524ef777d1af0f20d066b7540a507b4ede6` |

This run adds correctness evidence only. Sanitizers, hidden-device/AUTO cases, full fio/H2D
benchmarks, and model-quality/throughput evaluation were not rerun. The earlier performance
measurements remain tied to `2a6c481`; the bridge was implemented in the next slice below.

### 2026-10-30 — owning direct-H2D bridge

Tests ran in an isolated Ubuntu directory, then SHA-256 comparison of every tracked file under
`native`, `src`, `tests`, `scripts`, plus root CMake/Makefile/pyproject, verified exact equality with
the clean pulled commit `913211e42fc9e7778c995eda240b8b5fa40d3643`. No model weights or large
artifacts were committed. The target was the same RTX 4090, driver 580.173.02, CUDA 12.0.140,
G++ 12.4.0, Python 3.12.3/NumPy 2.3.5, liburing 2.5 and ext4 `/dev/nvme0n1p4` (`rw,relatime`).

Reproduction commands (run from the repository root; test directories must exist):

```sh
make test smoke golden-check PYTHON=.venv/bin/python
cmake -S . -B build/bridge-cpu -DHARPQ_ENABLE_CUDA=OFF \
  -DHARPQ_ENABLE_IO_URING=ON -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
cmake --build build/bridge-cpu --parallel 4
HARPQ_DIRECT_IO_TEST_DIR="$PWD/build/bridge-cpu" \
  ctest --test-dir build/bridge-cpu --output-on-failure

cmake -S . -B build/bridge-cuda -DHARPQ_ENABLE_CUDA=ON -DHARPQ_ENABLE_IO_URING=ON \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/bridge-cuda --parallel 4
HARPQ_DIRECT_IO_TEST_DIR="$PWD/build/bridge-cuda" \
  ctest --test-dir build/bridge-cuda --output-on-failure
HARPQ_DIRECT_IO_TEST_DIR="$PWD/build/bridge-cuda" \
  build/bridge-cuda/native/cuda/harpq_cuda_direct_packet_io_test --json \
  > build/native-service-observations.json
PYTHONPATH=src .venv/bin/python scripts/check_native_feedback.py build/native-service-observations.json
```

Results: 74 Python tests, doctor and golden drift pass on both Mac and Linux. CUDA-OFF/io_uring-ON
has four passes and six reasoned skips. CUDA-ON/io_uring-ON has ten passes, no skips. The bridge
test covers byte equality against pread and D2H; facade and abandoned-operation lifetime; CRC,
required/optional, cancellation/backpressure and legacy tails; event-record retry/event-query
failure; compute-stream ordering; concurrent read polling and observation validity. Nine real
fixture transfers (three at each 64 KiB/256 KiB/1 MiB physical size) round-trip and replay exactly
through the Python scheduler. They are not model or throughput measurements.

A separate CUDA-ON/io_uring-ON build with
`-DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"` built and
passed `harpq_cuda_direct_packet_io_test` (C++ host-path UBSan, not CUDA device memcheck). Hiding
the GPU with `CUDA_VISIBLE_DEVICES=""` made the explicit-ON bridge executable exit 1, not skip.
Temporarily inverting the observation-completion condition made the test fail as expected; the
condition was restored and the full CUDA suite passed afterward. AUTO-hidden-GPU, ASan,
compute-sanitizer, matched fio/H2D throughput, overlap and real models were not rerun. Mac native
CMake tests were not rerun because CMake is absent from the current Mac PATH.

Copied evidence is retained under the Ubuntu clone's ignored
`build/evidence-913211e-20261030/` directory:

| Artifact | SHA-256 |
|---|---|
| `cpu.log` | `c737300909822fe0f04a4b5fe966c1207f8e4550f16b5d091d736321c69108be` |
| `cuda.log` | `4b0f247a66cea833b58d7bb87839626c1ff9cba1790c38930730c2bd7e74a1c5` |
| `ubsan.log` | `14568b45e58458c46b574ce0da63691cade6a3655098cb4a994f5c71801cc2d7` |
| `native-service-observations.json` | `1929ede78a3465fda6ab418c785501bfa402e95bb0c6ee3e519d3b5d37a3012d` |

The source-list digest below was identical in the tested directory and clean clone:
`5b9e177897ff758cfb0a798016c47786a87887c3298fbb5dfe0bc5e676e0aa10`.

```sh
git ls-files -z CMakeLists.txt native src tests scripts Makefile pyproject.toml \
  | xargs -0 sha256sum | sha256sum
```

### 2026-10-30 — repeated transfer measurement

Clean `decdf707f4963b0aefae83d9252b304ef8728cc5` was rebuilt in Release on Ubuntu for the actual
measurement. The separate verification directory's source digest matched it exactly:
`2f858b6806db6381b31995e58cf034c7c1ad6b4a928eb6b24754043af3b21edb` using the source-list command
above. All 78 Python tests, doctor and golden drift passed on Mac/Linux; all ten CUDA-ON/io_uring-ON
native tests passed. The benchmark smoke round additionally rejected bad size/volume/mode and
CRC-corrupt files in both direct/bridge modes, and checked byte totals/D2H. Inverting the fio
zero-write guard caused its unit test to fail; after restoring it all 78 Mac tests passed again.

The actual sweep used three sizes, five repetitions, 16 MiB warm-up and 256 MiB timed volume per
case. At 1 MiB, median fio QD1/QD8 was 3.532/5.478 GB/s; isolated H2D was 25.925 GB/s; the unchanged
native direct+CRC/bridge paths reached only 0.582/0.542 GB/s. These results establish a measured
software-path bottleneck; they do not justify disabling CRC or claiming throughput acceptance.
The native I/O stage includes CRC and dispatch, so isolate/profile those before attributing cost.

Raw results, GPU snapshots, method differences, uncertainty and checksums are in `HOST_BASELINE.md`.
All 117 result JSON hashes were verified and the temporary 256 MiB synthetic pack was removed.
Test logs remain under the Ubuntu clone's ignored `build/evidence-decdf70-20261030/`:

| Artifact | SHA-256 |
|---|---|
| `cuda.log` | `c145b1ccf98413503defd6ae698ddcd197bf05e6dca4857d779ca24b2d630a7f` |
| `smoke-sha256.json` | `c95d6c643d1ccad9626011e6d0758b87920a84ce89d085e40ceeda85009071f3` |

No new CPU-only native configure, sanitizer, model, optimized-W4, cold-cache or overlap run was
performed in this measurement slice. No CUDA/runtime algorithm, quantizer, format or dependency
was changed; the user's unrelated local `AGENTS.md` edits were preserved.

## Known limits — do not overclaim

- The direct-decode CUDA path is batch-one GEMV only. It has no tensor-core/GEMM path and is not yet
  optimized against a production or library-parity static-W4 implementation. Its current naive-W4
  ratio is exploratory and cannot satisfy the ≥85% acceptance gate.
- The cuBLAS path still materializes FP32 weights by design because it is the correctness oracle.
- There is no native full-container/index reader, native scheduler or device cache. The owning
  io_uring-to-CUDA bridge is a strict functional path against buffered `pread`/D2H, with per-packet
  `cudaMalloc/cudaFree`. It proves neither fixed io_uring buffer use, asynchronous GPU overlap,
  nor NVMe throughput. Downstream compute lifetime remains the consumer's responsibility.
- The CUDA ring registers its complete fixed allocation and has one copy stream plus per-slot
  events. It has no bounded registration-capacity probe, multi-ring concurrency policy, or measured
  runtime queue depth yet. Its catastrophic CUDA-safety policy is deliberately fail-stop rather
  than quarantine/recovery.
- The current v1 container makes one packet per whole matrix stage, not per tile/chunk. Fine-grained
  adaptive SSD/cache placement requires a measured coalescing/chunking decision and likely a
  versioned format extension.
- No PyTorch custom op, Mistral/Mixtral adapter, real checkpoint, or model-quality evaluation exists.
- The repeated SSD/H2D sweep is a bounded warm synthetic baseline, not production acceptance.
  Native direct performance is far below its QD1 fio denominator; runtime H2D, overlap, real-packet
  read amplification, TTFT, inter-token latency and Pareto gates remain open.
- Native service observations now populate trace v2 in a functional integration check; their
  small synthetic sample set is not production calibration. No live native scheduler is connected.
- Direct-I/O operation IDs are monotonic only within one `DirectPackFile`; multi-pack tracing must
  key them by source as well. Catastrophic submit ambiguity/`IORING_CQE_F_MORE` and the less-used
  `Poll` terminal branches do not yet have the deterministic fault-injection coverage used by the
  CUDA ring.
- The encoder's fitting step is residual least-squares round-to-nearest, not yet a multi-prefix
  GPTQ-style optimizer.
- Active KV cache remains in GPU/RAM and must not be paged to SSD in this prototype.

## Exact next implementation action

Profile the now-measured read/CRC/dispatch bottleneck before optimizing. Add a bounded standalone
CRC32C cost measurement and, when available, CPU sampling of the native direct loop; distinguish
CPU checksum work, submission/completion overhead and storage wait. The portable CRC loop is a
candidate, not yet a measured attribution. If it dominates, introduce a feature-detected accelerated
path with the current portable implementation as fallback, preserving incremental CRC and every
corruption/optional/base safety test. Rerun exactly the committed sweep and retain an optimization
only if it beats variance without dropping validation.

Then address per-packet GPU allocation and queueing using measured evidence, bounded reusable
storage and ownership through downstream compute. Preserve cancellation, backpressure and the
fail-closed rule; measure overlap with an event timeline instead of inferring it from components.

Production scheduler feedback must be calibrated for its actual service class and exact byte
bucket. The 80% H2D, 75% fio, read-amplification and optimized-W4 gates remain open. Do not begin
Mistral/Mixtral integration yet; native container/chunk integration and bounded cache are still needed.

## Recovery on either machine

Clone or pull the `main` branch, install the locked Python dependencies, and run:

```sh
make check PYTHON=.venv/bin/python NATIVE_CUDA=OFF
```

On the NVIDIA host, follow with the explicit CUDA CMake command above. If direct task handoff fails,
start a new Codex task in the clone and instruct it:

> Read AGENTS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/FORMAT.md,
> docs/BENCHMARK_PROTOCOL.md, docs/RESEARCH_PLAN.md, docs/PRIOR_ART.md, and
> docs/HOST_BASELINE.md completely. Run the CPU suite and CUDA CTest. Continue from “Exact next
> implementation action” without changing v1 observable semantics unless a new decision is
> recorded.

Before stopping future work, update this file with the branch, verified commands, environment,
unresolved failures, and one exact next action.
