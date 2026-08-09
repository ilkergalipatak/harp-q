# Architectural decisions

This log records decisions that affect the research contract or on-disk compatibility. “Accepted”
means the Mac reference follows the decision; it does not imply the Linux implementation or paper
claim has been validated.

## D-001 — Research identity

Status: accepted.

Use the name **HARP-Q: Hierarchy-Aware Residual Progressive Quantization**. Frame the project as a
joint quantization/format/scheduler/kernel study. Do not claim novelty for generic SSD offload,
progressive precision, mixed precision, or residual decomposition.

## D-002 — Mac is the semantic oracle

Status: accepted.

The NumPy implementation defines deterministic stage bytes, prefix reconstruction, progressive
linear output, distortion metrics, container validation, allocation, and scheduler replay. Mac CPU
or MPS timings are never NVIDIA evidence. Every CPU test must pass unchanged after Linux handoff.

## D-003 — Default quantization geometry

Status: accepted for format v1; subject to ablation.

Use `(out_features, in_features)` weights, `128 x 128` tiles, groups of 128 along input features,
four ordered signed-int2 stages, and FP16 scales. Edge matrices are zero-padded. The code alphabet
is two's-complement `[-2, -1, 0, 1]`, packed four values per byte.

Rationale: fixed tile/group geometry provides a clear kernel and packet contract. Risk: signed
int2 is asymmetric around zero and may be too weak for the base; compare alternate alphabets,
rotations, and group sizes without silently changing v1 files.

## D-004 — Residual is formed from decoded bytes

Status: accepted.

After each stage, compute the next residual from codes multiplied by the exact stored FP16 scales,
not higher-precision encoder-only scales. This guarantees encoder/decoder prefix agreement and
prevents hidden state from making pack replay non-reproducible.

## D-005 — Base is executable; refinements are optional

Status: accepted.

Stage zero is required, has no dependency, and must allow execution. Stage `n > 0` is optional and
depends on stage `n-1` for the same matrix. Missing residual packs fall back to the longest
available prefix. Missing base data, malformed dependency closure, or checksum failure is explicit.

## D-006 — Directory container with 4 KiB packet alignment

Status: accepted as format v1.

Use `manifest.json`, `index.bin`, `boot.pack`, `base.pack`, and `refine-n.pack`. Payload offsets are
4096-byte aligned. The binary index is header plus length-prefixed canonical JSON records. Matrix
stages use a versioned little-endian binary payload. See `FORMAT.md`.

Rationale: it is simple to inspect on Mac and maps cleanly to later direct I/O. A fixed-record index
or coalescing table may be justified by Linux data; incompatible changes require a version bump.

## D-007 — Two integrity layers

Status: accepted.

Use SHA-256 for immutable index/pack identity and CRC32C for each packet after reading. Alignment
padding is excluded from packet CRC. Do not treat checksum success as authorization or authenticity;
signed release manifests are a separate future concern.

## D-008 — Offline allocation is exact at reference scale

Status: accepted.

Model each matrix/tile ladder as dependency-closed prefix choices, then solve the multiple-choice
knapsack with a sparse Pareto dynamic program. Base packets are mandatory. Residual gain is the
non-negative reduction in a selected measured distortion metric. The reference solver favors
determinism over model-scale speed; approximation/hierarchical solvers require comparison to it.

## D-009 — Runtime scheduler models refinement cache

Status: accepted.

Assume the executable base is handled by the placement plan. Schedule residuals using gain, reuse,
importance, measured or modeled transfer cost, kernel cost, deadline, cache capacity, minimum
residency, and hysteresis. Eviction preserves dependency closure. Required requests fail loudly;
late optional requests are skipped atomically.

Logical resident bytes and physical I/O bytes are separate. A standalone direct read rounds to
alignment until coalescing metadata exists. Decisions record useful bytes, physical bytes, read
amplification, copy time, cache state, loads, evictions, skips, and scores.

## D-010 — Trace is a reproducibility artifact

Status: accepted.

Serialize scheduler config, initial cache, complete input requests, timing inputs, and decisions to
versioned JSON. Replay regenerates each decision and rejects divergence. Small deterministic traces
may be committed; large benchmark traces remain artifacts with checksums/summaries only.

## D-011 — Linux implementation target

Status: accepted.

Target one consumer NVIDIA GPU on Linux with CUDA C++/PyTorch custom ops, registered pinned memory,
`io_uring + O_DIRECT`, CUDA copy/compute streams, and event dependencies. Begin with Mistral-7B,
then Mixtral-8x7B. Multi-GPU, Windows, Apple production kernels, and a production serving control
plane are out of scope for the first six months.

## D-012 — Active KV cache does not go to SSD

Status: accepted.

The first prototype streams model weights/refinements only. Active KV cache remains in GPU/RAM.
This prevents weight-format research from being confounded by a separate latency-sensitive KV
paging problem.

## D-013 — Hardware capabilities require evidence

Status: accepted.

Passive profiling may assert CUDA only after `cuInit` and a positive device count, and io_uring only
after a successful setup syscall. O_DIRECT is filesystem-specific and remains unverified until the
bounded active fio test succeeds on the explicit scratch mount. NVIDIA performance support stays
false until CUDA code and acceptance tests exist.

## D-014 — Benchmark gates are fixed before optimization

Status: accepted.

Use the gates in `BENCHMARK_PROTOCOL.md`: ≥85% of matched static-W4 kernel throughput, ≥80% of
isolated pinned-H2D bandwidth, ≥75% of matched fio bandwidth, read amplification ≤1.2, reference
numeric agreement, and dense/MoE Pareto evidence at two memory budgets.

## D-015 — Repository is the durable context

Status: accepted.

The authoritative remote is `https://github.com/ilkergalipatak/harp-q.git`, branch `main`. Keep
requirements, rules, research state, decisions, and handoff notes in Git. Model weights, secrets,
pack files, and large benchmark outputs are ignored. If Codex task handoff fails, `HANDOFF.md` is
the bootstrap source for a new task.

## D-016 — Python remains the oracle; the runtime is native

Status: accepted.

Keep the deterministic NumPy quantizer, distortion experiments, offline allocator, pack writer,
trace simulator, model adaptation, and CLI in Python. Implement the latency-critical reader,
scheduler, cache, asynchronous I/O, and buffer/event lifetime management in C++; implement GPU
decode and linear kernels in CUDA C++. The Python and native implementations must be compared with
generated byte-level and numeric golden vectors at every supported prefix.

Use C++17 as the initial compiler floor because it is supported by the installed CUDA 12.0/GCC 12
toolchain and is compatible with the intended PyTorch extension boundary. Rust is deferred: CUDA
and PyTorch already require C++ interfaces, so adding a third native-language FFI would increase
prototype risk without accelerating the measured data path. This decision does not prevent a later
Rust serving or network control plane.

## D-017 — Native Ubuntu is the primary NVIDIA evidence host

Status: accepted.

Use the bare-metal Ubuntu 24.04 installation on the RTX 4090 workstation as the primary CUDA and
storage-development host. WSL2 remains a fallback development environment, not the source of final
NVMe, pinned-memory, cold-cache, tail-latency, or Pareto evidence. Preserve the exact host, driver,
toolkit, compiler, filesystem, and benchmark metadata with every measured claim.

The first verified host baseline is recorded in `HANDOFF.md`. GPUDirect Storage is not an initial
dependency; the portable path remains direct SSD reads into a registered pinned-RAM ring followed
by asynchronous H2D copies.

## D-018 — Native numeric agreement is declared before tuning

Status: accepted.

Require exact agreement for serialized payload bytes and materialized FP32 prefix weights. For the
correctness-first FP32 CUDA materialization plus pedantic cuBLAS SGEMM path, require every output to
be finite and satisfy
`abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)` at prefixes zero through four. The absolute
term protects values near zero and the relative term permits harmless FP32 accumulation-order
differences. The current small golden fixture happens to match exactly; zero observed error does
not tighten the declared contract.

Fused kernels that use FP16/BF16 output, tensor-core accumulation, or a different reduction order
must receive a separate documented tolerance before performance tuning. Do not relax a tolerance
after observing a failing benchmark without recording a new decision and justification.

## Open decisions for Linux evidence

- final tile aggregation and coalescing metadata;
- base placement split between VRAM and pinned/ordinary RAM;
- exact multi-prefix Hessian objective and compensation order;
- GEMV/GEMM code layout and accumulation dtype;
- queue depth, ring count, registered-buffer size, and cancellation policy;
- deadline source and router-prediction model;
- production model checkpoint revisions, calibration corpus, and evaluation tasks.
