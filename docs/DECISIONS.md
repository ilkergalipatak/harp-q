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

## D-019 — The first fused decode kernel is a device-view GEMV

Status: accepted for the correctness-first CUDA slice.

For batch-one decode, expose an asynchronous CUDA launch that consumes device-resident signed-int2
code arrays and raw FP16 scale bits for a contiguous prefix of up to four stages. Accumulate input,
decoded residual weights, and output in FP32; add bias exactly once. Do not materialize an FP32
weight matrix and do not expose Python objects at the kernel boundary. The caller owns device
buffer lifetime, stream ordering, and completion; a separate owning host wrapper exists only for
differential tests and bring-up.

Because the fused block reduction and FMA order differ from the scalar/SGEMM oracle, require each
finite FP32 output to satisfy
`abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected)` at prefixes zero through four. This
tolerance was fixed before throughput tuning. It does not apply to a future FP16/BF16 output path.

## D-020 — Pinned-H2D baseline uses future-ring-compatible memory

Status: accepted.

Measure the isolated H2D denominator with ordinary page-aligned host memory registered through
`cudaHostRegister`, not a pageable fallback and not `cudaHostAlloc`. This mirrors the ownership and
alignment strategy intended for the future `O_DIRECT`/`io_uring` ring. Use a nonblocking CUDA
stream, CUDA events around timed H2D copies only, bounded warm-up/iteration volumes, and a sampled
D2H checksum after the run. Registration failure is explicit.

Treat this tool as a link-level microbenchmark, never as model throughput. Runtime H2D acceptance
remains at least 80% of a size-matched run captured with the same device, driver/runtime, payload,
registration strategy, and background-load conditions.

## D-021 — The first static-W4 comparison is exploratory and oracle-checked

Status: accepted for kernel bring-up, not for the throughput acceptance gate.

The first fused-GEMV timing tool compares against a benchmark-local signed-int4 kernel with the
same output-row mapping and block-reduction geometry. Before reporting any ratio, validate that
static kernel element by element against a deterministic CPU implementation of its packing, scale
indexing, FMA lanes, warp tree, and block tree. The FP32 validation bound is fixed at
`abs(actual - expected) <= 2e-5 + 2e-6 * abs(expected)`.

Always label this denominator `exploratory_matched_naive_baseline` and emit
`acceptance_evidence=false`. It is a regression and tuning aid only; the project's 85% gate still
requires an optimized production or library-parity static-W4 implementation under matched shape,
warm-up, clocks, and measurement settings.

## D-022 — Canonical v1 packs have a zero-padded physical tail

Status: accepted; backward-compatible writer clarification.

After writing the final payload, the canonical v1 writer pads every non-empty pack with zero bytes
to the manifest's 4096-byte alignment. Packet offsets, logical lengths, index records, and packet
CRC32C values do not change. Pack size and SHA-256 include all inter-packet and terminal padding.
An empty `boot.pack` remains zero bytes.

This does not increment the format version: existing readers already accept unreferenced trailing
bytes, and new buffered readers must continue to accept legacy unpadded v1 packs. A packet is safe
for a full aligned direct read only when
`offset + align_up(logical_length, alignment) <= declared_pack_size`. A legacy terminal short tail
may use an explicit buffered compatibility path; it must never be silently presented as successful
`O_DIRECT` evidence.

## D-023 — The packet ring separates I/O geometry from allocation geometry

Status: accepted for the synchronous correctness backend.

The CPU-portable packet ring owns one contiguous allocation and move-only, generation-tagged
leases. Its only normal state path is
`free -> filling -> ready -> H2D-in-flight -> free`; cancellation may return filling/ready slots to
free, but an in-flight slot is not recycled until a completion object proves it safe. Leases share
storage ownership so destroying the facade cannot invalidate a live slot.

Format I/O alignment and host allocation alignment are distinct. The v1 I/O alignment is 4096,
while allocation alignment is at least `max(io_alignment, host_page_size)` and slot stride is
rounded to that allocation alignment. This matters on the Apple-silicon reference host, whose page
size is 16384 bytes, even though the Linux target currently uses 4096 for both.

The first `pread` backend is buffered and synchronous. It validates extents before the syscall,
retries `EINTR`/partial reads, checks CRC32C over logical bytes only, and records logical bytes,
planned physical bytes, actual file bytes, and syscall count separately. Required packet failure is
fatal; an optional residual can become explicitly unavailable, while ring-full and cancellation
remain distinct non-fallback outcomes. These results are functional evidence only, never NVMe or
overlap performance evidence.

## D-024 — CUDA registration ownership follows every slot lease

Status: accepted for the first registered-host-memory adapter.

The CUDA facade registers the ring's complete contiguous backing allocation once with
`cudaHostRegisterDefault`. Registration is exact-or-fail: there is no pageable fallback and no
silent reduction in slot count. CUDA-specific filling, ready, and in-flight leases all retain one
shared control object, so destroying the facade cannot destroy the copy stream, completion events,
host registration, or backing allocation while any slot is still observable.

The adapter owns one nonblocking H2D stream and one disable-timing completion event per slot. It
copies logical packet bytes, records the event, and exposes `MakeStreamWait` so a compute stream can
depend on that exact copy without exposing the raw copy stream. The synchronous `pread` adapter can
load directly through the registered ring and wraps the core ready lease before it leaves the CUDA
facade. Validation or enqueue failure retains the caller's ready lease; a failure after work may
have been accepted first establishes stream or device completion before returning.

Host-memory safety takes precedence over graceful recovery. If event synchronization, owned-stream
synchronization, and the captured-device synchronization fallback all fail to prove that DMA has
stopped, the process terminates rather than recycle a slot or unregister/free its backing memory.
Likewise, teardown does not free an allocation after a failed `cudaHostUnregister`. A later faulted
ring/quarantine mode may replace this fail-stop policy, but silent reuse is forbidden.

## Open decisions for Linux evidence

- final tile aggregation and coalescing metadata;
- base placement split between VRAM and pinned/ordinary RAM;
- exact multi-prefix Hessian objective and compensation order;
- GEMV/GEMM code layout and accumulation dtype;
- queue depth, ring count, registered-buffer size, and cancellation policy;
- deadline source and router-prediction model;
- production model checkpoint revisions, calibration corpus, and evaluation tasks.
