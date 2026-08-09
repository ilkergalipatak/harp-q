# HARP-Q handoff

Last verified baseline: 2026-08-09 on the original Apple-silicon Mac and the native Ubuntu 24.04
RTX 4090 host. Fourth-slice working-tree changes are explicitly marked pending below and are not
covered by that baseline. This file is the recovery source if the Codex task cannot be handed off
directly.

## Repository

- Remote: `https://github.com/ilkergalipatak/harp-q.git`
- Branch: `main`
- Latest validated runtime source commit: `9cc7dc593ab8c650dd7d21a41dd31e4b2492e246`
- Performance-baseline source commit: `2a6c481a077fd11e46fc5e68a6da10b78389a9d5`
- Python package: `harpq`, Python 3.11+
- Native floor: CMake 3.24+, C++17; CUDA toolkit 11.8+ when enabled
- Validated Mac runtime: Python 3.12.13, NumPy 2.3.5
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
  CPU-only test run records all five CUDA-only tests as explained CTest skips. CUDA architectures
  supplied through CMake or `CUDAARCHS` are preserved; otherwise the project defaults to `sm_89`.
- CPU CI runs the Python suite, CLI smoke, generated-fixture drift check, and native CPU CTest on
  macOS and Ubuntu. The Linux job installs liburing and builds the direct-I/O adapter in `AUTO`
  mode; macOS records a reasoned direct-I/O skip. Both jobs record explicit CUDA skips.

The fourth slice is being integrated; its exact-source Linux results and source commit must be
recorded before it is called validated:

- `native/src/direct_packet_io_linux.cpp`: Linux CPU-only `io_uring + O_DIRECT` adapter using strict
  `STATX_DIOALIGN`, bounded ordinary reads into `PacketRing` slots, cooperative cancellation,
  logical CRC, and a distinct legacy `buffered-only` outcome. It has no fixed-buffer registration,
  CUDA ownership bridge, fork support, or throughput claim.
- `scheduler.py` trace v2: caller-supplied ordered `ServiceObservation` values, exact
  `(service_class, logical_bytes, physical_io_bytes)` rolling-p95 buckets, auditable estimate
  sources, and unchanged v1 replay. I/O/H2D component durations are audit-only; the observed
  critical path drives deadlines.

Architecture decisions are in `DECISIONS.md`; file/layout contracts are in `FORMAT.md`; benchmark
gates are in `BENCHMARK_PROTOCOL.md`; the measured host summary is in `HOST_BASELINE.md`.

## Verified commands and results

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

The fourth-slice working tree still requires an exact-source Mac/Linux verification record here.
The older counts and commit above apply only to the last validated source; they are not evidence for
the new direct-I/O or measured-feedback paths.

Before the final exception-lifetime and liburing configure-probe hardening, the fourth-slice tree
passed 73 Python tests on Mac and Ubuntu, strict GCC 12 plus ASan/UBSan direct-I/O tests, four
CPU/direct tests with five CUDA skips, and all nine CPU/CUDA tests on the RTX 4090. Subsequent
hardening moves terminal-operation ownership before failure-message allocation, strengthens the
configure-time API check, requires exact versioned trace schemas, and prevents generic `EINVAL`
from becoming an AUTO-mode test skip. The final Mac worktree passes 74 Python tests, but its exact
source was not rerun on Ubuntu because the remote Codex execution allowance was exhausted. Treat
the preceding Linux results as near-source diagnostic evidence, not as an exact-source validation
record; rerun the commands above or rely on a green CI run before promoting this slice to the
validated baseline.

## Known limits — do not overclaim

- The direct-decode CUDA path is batch-one GEMV only. It has no tensor-core/GEMM path and is not yet
  optimized against a production or library-parity static-W4 implementation. Its current naive-W4
  ratio is exploratory and cannot satisfy the ≥85% acceptance gate.
- The cuBLAS path still materializes FP32 weights by design because it is the correctness oracle.
- There is no native full-container/index reader, native scheduler, device cache, or lifetime-safe
  io_uring-to-CUDA bridge yet. The Linux CPU direct adapter uses `io_uring + O_DIRECT`, but only as a
  strict functional path against the synchronous buffered `pread` oracle; it proves neither
  fixed-buffer use, asynchronous GPU overlap, nor NVMe throughput.
- The CUDA ring registers its complete fixed allocation and has one copy stream plus per-slot
  events. It has no bounded registration-capacity probe, multi-ring concurrency policy, or measured
  runtime queue depth yet. Its catastrophic CUDA-safety policy is deliberately fail-stop rather
  than quarantine/recovery.
- The current v1 container makes one packet per whole matrix stage, not per tile/chunk. Fine-grained
  adaptive SSD/cache placement requires a measured coalescing/chunking decision and likely a
  versioned format extension.
- No PyTorch custom op, Mistral/Mixtral adapter, real checkpoint, or model-quality evaluation exists.
- The host profiler's fio and H2D values are bounded capability denominators only. Runtime H2D,
  overlap, read amplification, TTFT, inter-token latency, and Pareto gates remain open.
- Scheduler trace v2 can consume completed service observations, but no native runtime currently
  emits the end-to-end storage-submission-to-device-ready measurement needed to populate them.
- Direct-I/O operation IDs are monotonic only within one `DirectPackFile`; multi-pack tracing must
  key them by source as well. Catastrophic submit ambiguity/`IORING_CQE_F_MORE` and the less-used
  `Poll` terminal branches do not yet have the deterministic fault-injection coverage used by the
  CUDA ring.
- The encoder's fitting step is residual least-squares round-to-nearest, not yet a multi-prefix
  GPTQ-style optimizer.
- Active KV cache remains in GPU/RAM and must not be paged to SSD in this prototype.

## Exact next implementation action

Build an owning, lifetime-safe bridge from a completed `DirectPackFile` read into
`CudaPacketRing`'s ready/H2D/in-flight lease chain. The bridge must keep the registered backing,
io_uring operation, CUDA copy stream/event, and destination alive until both kernel I/O ownership
and DMA completion are proven; it must not reach through either facade's private storage or make
destruction order a caller convention. Preserve required/optional/backpressure/cancellation
semantics and the fail-closed CUDA safety rule.

Instrument that bridge from storage submission to device-ready completion and emit ordered
`ServiceObservation` records with exact service class, logical bytes, physical bytes, and critical
path. Keep I/O and H2D components audit-only, measure overlap separately, and compare 64 KiB,
256 KiB, and 1 MiB runtime reads against size/queue-matched fio plus the registered-H2D denominator.
Only after functional differential, lifetime/fault tests, and matched timing may these values feed
the scheduler as runtime evidence. Do not begin Mistral/Mixtral integration yet.

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
