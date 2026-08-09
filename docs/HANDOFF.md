# HARP-Q handoff

Last verified: 2026-08-09 on the original Apple-silicon Mac and the native Ubuntu 24.04 RTX 4090
host. This file is the recovery source if the Codex task cannot be handed off directly.

## Repository

- Remote: `https://github.com/ilkergalipatak/harp-q.git`
- Branch: `main`
- Source-linked CUDA evidence commit: `2a6c481a077fd11e46fc5e68a6da10b78389a9d5`
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

The first two native correctness/performance-bring-up slices are implemented:

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
- Root/native CMake: `HARPQ_ENABLE_CUDA=AUTO|ON|OFF`; explicit `ON` fails without CUDA, while a
  CPU-only test run records all four CUDA-only tests as explained CTest skips. CUDA architectures
  supplied through CMake or `CUDAARCHS` are preserved; otherwise the project defaults to `sm_89`.
- CPU CI now runs the Python suite, CLI smoke, generated-fixture drift check, native CPU CTest, and
  explicit CUDA skips on macOS and Ubuntu.

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
```

Result: **63 Python tests pass**, the generated header drift check passes, and the dependency-free
native CPU test passes with Apple Clang. CMake is not installed on this Mac, so the CMake-driven
CPU test was instead verified on Ubuntu and remains covered by macOS GitHub Actions.
The source commit's [CPU reference workflow](https://github.com/ilkergalipatak/harp-q/actions/runs/31328324940)
completed successfully on its first attempt.

On the Ubuntu RTX 4090 host, the clean clone at
`2a6c481a077fd11e46fc5e68a6da10b78389a9d5` first passed the full CPU gate:

```sh
make check PYTHON=.venv/bin/python \
  NATIVE_BUILD_DIR=build/evidence-2a6c481-cpu NATIVE_CUDA=OFF
```

Result: 63 Python tests, CLI doctor, golden drift, and the native CPU test pass; the four CUDA-only
CTest entries are skipped with an explicit reason. CUDA was then configured explicitly:

```sh
cmake -S . -B build/evidence-2a6c481-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/evidence-2a6c481-cuda --parallel
ctest --test-dir build/evidence-2a6c481-cuda --output-on-failure
```

Result: all five tests pass: `native.stage_codec`, `native.cuda_oracle`,
`native.progressive_gemv`, `native.h2d_smoke`, and `native.gemv_benchmark_smoke`. The tests cover
prefixes zero through four, `K=257/513/4096`, base-only operation, bias, padding, raw FP16 edge bit
patterns, and malformed device views. The host reports RTX 4090 / compute capability 8.9, two async
engines, CUDA runtime 12.0, and driver API compatibility 13.0.

The source-linked 64 MiB registered-H2D run measured 26.8710 GB/s. The exploratory `4096x4096`,
prefix-two GEMV run measured 82.872324 us versus 67.3894405 us for its matched-naive W4 helper, a
ratio of 0.813171844. The W4 helper matched its deterministic CPU oracle with maximum error zero.
This is not the formal 85% gate: the denominator is deliberately marked non-production and GPU
clocks were uncontrolled. Exact commands, raw artifact SHA-256 values, and the size sweep are in
`HOST_BASELINE.md`.

A separate CUDA-off CMake build passes the CPU test and marks all CUDA tests skipped with an
explicit reason. GCC 12 ASan+UBSan also passes that CPU build. `compute-sanitizer` 12.0 is installed,
but its packaged launcher could not instrument the executable (`target application terminated
before first instrumented API call`) even with the packaged injection path; do not claim a CUDA
memcheck pass until that toolchain issue is fixed.

The bounded host profile verified `io_uring + O_DIRECT` on the selected ext4/NVMe scratch
filesystem. The raw JSON reports remain ignored on that host. See `HOST_BASELINE.md` for the exact
hardware and smoke measurements.

## Known limits — do not overclaim

- The direct-decode CUDA path is batch-one GEMV only. It has no tensor-core/GEMM path and is not yet
  optimized against a production or library-parity static-W4 implementation. Its current naive-W4
  ratio is exploratory and cannot satisfy the ≥85% acceptance gate.
- The cuBLAS path still materializes FP32 weights by design because it is the correctness oracle.
- There is no native full-container reader, zero-copy payload view, native scheduler, device cache,
  registered pinned-memory ring, SSD-to-ring backend, asynchronous copy/compute event DAG, or
  `io_uring` runtime backend yet. The H2D benchmark registers one isolated buffer; it is not a ring.
- The current v1 container makes one packet per whole matrix stage, not per tile/chunk. Fine-grained
  adaptive SSD/cache placement requires a measured coalescing/chunking decision and likely a
  versioned format extension.
- No PyTorch custom op, Mistral/Mixtral adapter, real checkpoint, or model-quality evaluation exists.
- The host profiler's fio and H2D values are bounded capability denominators only. Runtime H2D,
  overlap, read amplification, TTFT, inter-token latency, and Pareto gates remain open.
- The encoder's fitting step is residual least-squares round-to-nearest, not yet a multi-prefix
  GPTQ-style optimizer.
- Active KV cache remains in GPU/RAM and must not be paged to SSD in this prototype.

## Exact next implementation action

Implement the registered/aligned RAM ring and a synchronous `pread` correctness backend before
adding `io_uring`. Use fixed, page-aligned slots registered through `cudaHostRegister`; model each
slot as `free -> filling -> ready -> H2D-in-flight -> free`, and retain its lease until a CUDA event
proves the copy complete. Validate aligned offsets/physical lengths, short reads, EOF, cancellation,
base-packet failure, optional-residual fallback, CRC32C corruption, and safe teardown with in-flight
work. Keep logical resident bytes separate from aligned physical I/O bytes.

Once that contract passes deterministic native tests, replace the fill operation with
`io_uring + O_DIRECT`, add bounded queue depth and copy/compute streams, then feed measured service
times into the scheduler/cache. Benchmark at 64 KiB, 256 KiB, and 1 MiB first; the H2D curve favors
coalescing, but final chunk size must also satisfy NVMe latency, read-amplification, cache-waste, and
deadline constraints.

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
