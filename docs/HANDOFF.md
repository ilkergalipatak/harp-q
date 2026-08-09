# HARP-Q handoff

Last verified: 2026-08-09 on the original Apple-silicon Mac and the native Ubuntu 24.04 RTX 4090
host. This file is the recovery source if the Codex task cannot be handed off directly.

## Repository

- Remote: `https://github.com/ilkergalipatak/harp-q.git`
- Branch: `main`
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

The first native correctness slice is also implemented:

- `native/src/stage_codec.cpp`: dependency-free little-endian stage payload decoder, manual FP16
  conversion, signed-int2 prefix dequantization, CPU progressive-linear oracle, geometry checks,
  and incremental CRC32C.
- `scripts/generate_native_golden.py`: deterministic Python-to-C++ golden generator. Its checked
  fixture uses original shape `5x9`, padded shape `8x16`, tiles `4x8`, group size 4, a `2x2` tile
  grid, edge padding, four stages, and prefixes zero through four.
- `native/src/cuda_oracle.cu`: correctness-first CUDA stage materialization plus pedantic cuBLAS
  SGEMM. It is deliberately not a fused or throughput implementation.
- Root/native CMake: `HARPQ_ENABLE_CUDA=AUTO|ON|OFF`; explicit `ON` fails without CUDA, while a
  CPU-only test run records the CUDA oracle as an explained CTest skip.
- CPU CI now runs the Python suite, CLI smoke, generated-fixture drift check, native CPU CTest, and
  explicit CUDA skip on macOS and Ubuntu.

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

On the Ubuntu RTX 4090 host, CUDA was configured explicitly:

```sh
cmake -S . -B build/native-cuda \
  -DHARPQ_ENABLE_CUDA=ON -DBUILD_TESTING=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/native-cuda --parallel
ctest --test-dir build/native-cuda --verbose
```

Result: `native.stage_codec` and `native.cuda_oracle` both pass. The test reports an RTX 4090,
compute capability 8.9, two async engines, CUDA runtime 12.0, and driver API compatibility 13.0.
All prefix outputs from zero through four match the checked Python fixture with maximum absolute
error zero. This is correctness evidence, not a kernel-throughput result.

A separate CUDA-off CMake build passes the CPU test and marks `native.cuda_oracle` skipped with an
explicit reason. GCC 12 ASan+UBSan also passes that CPU build. `compute-sanitizer` 12.0 is installed,
but its packaged launcher could not instrument the executable (`target application terminated
before first instrumented API call`) even with the packaged injection path; do not claim a CUDA
memcheck pass until that toolchain issue is fixed.

The bounded host profile verified `io_uring + O_DIRECT` on the selected ext4/NVMe scratch
filesystem. The raw JSON reports remain ignored on that host. See `HOST_BASELINE.md` for the exact
hardware and smoke measurements.

## Known limits — do not overclaim

- The CUDA path materializes FP32 weights and calls cuBLAS. There is no fused progressive GEMV/GEMM
  yet, so the ≥85% static-W4 throughput gate has not been tested.
- There is no native full-container reader, zero-copy payload view, native scheduler, device cache,
  registered pinned-memory ring, asynchronous H2D event DAG, or `io_uring` runtime backend yet.
- The current v1 container makes one packet per whole matrix stage, not per tile/chunk. Fine-grained
  adaptive SSD/cache placement requires a measured coalescing/chunking decision and likely a
  versioned format extension.
- No PyTorch custom op, Mistral/Mixtral adapter, real checkpoint, or model-quality evaluation exists.
- The host profiler's fio values are bounded capability baselines only. H2D bandwidth, overlap,
  read amplification under the runtime, TTFT, inter-token latency, and Pareto gates remain open.
- The encoder's fitting step is residual least-squares round-to-nearest, not yet a multi-prefix
  GPTQ-style optimizer.
- Active KV cache remains in GPU/RAM and must not be paged to SSD in this prototype.

## Exact next implementation action

Implement a batch-1 fused progressive GEMV for `sm_89` whose public input is device-resident packed
int2 codes plus FP16 scales. Compare prefixes one through four against both the CUDA cuBLAS oracle
and the Python golden output, including bias-once and edge-padding cases. Add a matched static-W4
baseline and a native pinned-H2D microbenchmark before making any throughput claim. Do not couple
the fused kernel to Python objects or to the on-disk 20-byte header alignment.

After the kernel contract is stable, implement the registered/aligned RAM ring and `pread`
correctness backend, then `io_uring + O_DIRECT`, CUDA copy/compute streams, event-safe device cache,
and scheduler cost feedback. Determine chunk size from the target NVMe/H2D curve rather than
hard-coding tile-per-packet.

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
