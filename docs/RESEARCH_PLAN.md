# HARP-Q research plan

Last updated: 2026-08-09

## Research question

Can a single post-training-quantized LLM checkpoint be represented as an always-executable base
plus independently stored residual precision packets, then scheduled across GPU, pinned RAM, RAM,
and SSD according to cache state and per-token deadlines, without giving back the quality gained
from higher precision or stalling on every layer?

HARP-Q expands to **Hierarchy-Aware Residual Progressive Quantization**. The intended contribution
is not “put model weights on SSD”; that problem already has substantial prior art. The research
target is the joint contract between:

1. a successive-refinement PTQ representation;
2. a dependency-aware, direct-I/O-friendly packet format;
3. an offline quality/byte allocator;
4. a runtime policy that turns cache, routing, and deadlines into a realized precision prefix;
5. fused kernels that consume those prefixes without reconstructing a second full model.

## Falsifiable hypotheses

- **H1 — quality:** activation/Hessian-aware residual fitting yields a better prefix quality curve
  than naive residual round-to-nearest at equal serialized bytes.
- **H2 — systems:** physical-I/O-aware packet grouping and prefetch keep read amplification at or
  below 1.2 while overlapping enough SSD→RAM→GPU traffic to improve latency over naive offload.
- **H3 — kernels:** a fused two-stage (nominal 4-bit) progressive kernel reaches at least 85% of a
  matched static W4 kernel on consumer NVIDIA hardware.
- **H4 — adaptation:** under at least two constrained memory budgets, deadline/cache-aware precision
  choices produce a quality–latency Pareto point that static W2/W4/W8 and fixed offload do not.
- **H5 — MoE:** router-aware residual/expert prefetch reduces cold-expert latency without excessive
  wrong-prefetch bytes on Mixtral-8x7B.

Failure of any hypothesis is a valid research result. The implementation must report failed gates,
not redefine baselines after measurement.

## Current executable reference

The Mac reference implementation is deliberately simple and auditable:

- weights use `(out_features, in_features)` layout;
- default tiles are `128 x 128`, grouped along the input-feature dimension by 128;
- four signed-int2 stages approximate `W`, then the exact FP16-decoded residual three times;
- every prefix from zero through four stages can be decoded; executable prefixes 1–4 correspond to
  nominal 2/4/6/8 code bits per padded weight;
- activation MSE, empirical Hessian proxy/error, and optional logit KL measure distortion;
- the container stores a required base and optional chained residuals at 4 KiB-aligned offsets;
- an exact precedence-aware multiple-choice knapsack allocates packet prefixes offline;
- the reference scheduler uses distortion gain, reuse, importance, cache state, transfer time,
  kernel cost, capacity, deadlines, minimum residency, and hysteresis;
- completed runtime service can be supplied as ordered observations; exact service/byte buckets use
  a bounded rolling p95 only after the configured minimum sample count;
- every scheduling event, observation, estimate source, and decision is serialized and replayed
  exactly, while legacy trace v1 remains replayable.

This is a correctness baseline, not the final quantizer. In particular, the current encoder measures
activation/Hessian-aware distortion but does not yet optimize codes with a GPTQ-style second-order
solver. That algorithmic upgrade is part of the research, and no claim should imply it already
exists.

## Work packages

### WP0 — reference and format (Mac, implemented)

- deterministic NumPy quantize/dequantize/progressive matmul;
- FP16-scale and signed-int2 stage codec;
- manifest, binary index, boot/base/refinement packs, CRC32C and SHA-256;
- missing residual fallback and dependency validation;
- distortion profiler, exact allocator, scheduler trace/replay;
- synthetic Transformer prefix tests and CLI workflows.

Exit gate: all CPU tests pass on Mac and later pass unchanged on Linux.

### WP1 — calibration algorithm (Mac/Linux CPU first)

Compare three encoders under one format:

1. current least-squares residual round-to-nearest;
2. activation-weighted tile/group scale and code search;
3. Hessian-aware sequential compensation with a multi-prefix objective.

The objective must score every useful prefix, not only the 8-bit parent. Ablate tile size, group
size, scale precision, code alphabet, prefix weights, calibration corpus size, and layerwise versus
global allocation. Measure perplexity/task quality and serialized bytes, including scales/padding.

### WP2 — packet planning and I/O model (Linux)

- profile target NVMe, filesystem, RAM, PCIe, H2D, and CUDA copy engines;
- decide tile aggregation from measured latency/bandwidth and 4 KiB alignment;
- introduce a coalescing plan so physical bytes and useful bytes remain explicit;
- registered pinned-RAM ring buffers: correctness slice implemented with fixed aligned slots,
  move-only leases, whole-allocation registration, and fail-closed teardown; capacity/throughput
  tuning remains;
- CPU-only `io_uring + O_DIRECT` correctness adapter: ordinary reads target aligned `PacketRing`
  slots under strict `STATX_DIOALIGN`, bounded queue depth, cooperative cancellation, and explicit
  legacy buffered-only policy; fixed buffers, CUDA ownership composition, and throughput remain;
- validate CRC after reads and preserve base-prefix fallback on late/corrupt optional packets
  (the synchronous buffered `pread` path remains the differential oracle for direct I/O).

### WP3 — CUDA execution (Linux)

- dequantize-plus-cuBLAS oracle: implemented;
- fused progressive GEMV for batch-one decode: correctness path implemented, optimization pending;
- implement GEMM path for prefill/small batches;
- add PyTorch C++/CUDA custom op; the registered H2D-to-compute stream/event edge is implemented,
  while the full model execution DAG remains;
- compare each prefix against NumPy and measure static W4, H2D, and overlap gates.

Arithmetic tolerances are fixed before performance tuning: the FP32 materialize-plus-cuBLAS oracle
uses `atol=rtol=1e-4`, while the FP32 direct-decode block-reduction GEMV uses
`atol=rtol=2e-4`. Any lower-precision or differently accumulated path requires a new declared
contract.

### WP4 — dense integration (Linux)

- adapt Mistral-7B linear projections and model traversal;
- keep a guaranteed executable base placement plan;
- drive residual requests from layer order and measured deadlines;
- evaluate cold start, warm cache, TTFT, inter-token latency, SSD bytes/token, and quality.

### WP5 — MoE integration (Linux)

- adapt Mixtral-8x7B experts and router outputs;
- separate base-expert availability from residual-expert quality;
- predict/prefetch likely next experts and residual prefixes;
- measure router prediction, expert hit rate, wasted bytes, and cold-expert latency.

### WP6 — evaluation and release

- reproduce at least two GPU/RAM budgets for dense and MoE;
- compare static W2/W4/W8, the strongest practical offload baseline, and relevant progressive work;
- publish configs, checksums, small summaries, and scripts without weights or secrets;
- write the paper around supported claims only.

## Six-month sequence

| Period | Deliverable |
|---|---|
| Weeks 1–4 | Reference quantizer, metrics, tests, container and claim audit |
| Weeks 5–8 | Multi-prefix calibration objectives, allocator ablations, format freeze candidate |
| Weeks 9–12 | Linux profile, async I/O backend, pinned buffers, physical-byte model |
| Weeks 13–16 | CUDA oracle, fused GEMV/GEMM, PyTorch op, kernel gates |
| Weeks 17–20 | Mistral dense integration and end-to-end baselines |
| Weeks 21–24 | Mixtral router-aware integration, ablations, paper/release |

If the NVIDIA host arrives early, WP2 may start while Mac calibration work continues. Multi-GPU,
Windows, Apple production kernels, server-grade orchestration, and SSD-resident active KV cache are
outside the first six months.

## Evaluation contract

Use `BENCHMARK_PROTOCOL.md`. At minimum report full hardware/software metadata, cold and warm TTFT,
median/p95/p99 inter-token latency, tokens/s, physical and useful SSD bytes/token, H2D bytes/token,
cache hit rate, deadline misses, overlap, VRAM/RAM use, and quality. A “Pareto improvement” requires
matched hardware, model, workload, and memory budget.

Mac CPU/MPS results establish determinism and correctness only. The native Linux host now has
bounded CUDA correctness plus NVMe/`io_uring` capability evidence. The CPU direct-I/O adapter and
measured-feedback trace contract do not by themselves establish fixed-buffer use, SSD-to-CUDA
lifetime safety, runtime bandwidth, overlap, tail latency, model throughput, quality, or Pareto
evidence.

## Principal risks and mitigations

- **Novelty collision:** progressive precision and SSD caching already exist. Maintain the claim
  matrix and frame only the experimentally demonstrated joint mechanism.
- **I/O granularity:** per-tile packets can have catastrophic padding/read amplification. Coalesce
  from measured target-device data and keep physical-byte accounting in traces.
- **Kernel overhead:** four residual launches can erase bandwidth savings. Fuse decoding and
  accumulation; compare against optimized static W4 with identical shapes.
- **Quality at the base:** signed int2 may be too weak for some layers. Permit heterogeneous prefix
  allocation and test alternative alphabets/rotations without breaking the format contract.
- **Scheduler instability:** precision/cache choices can thrash. Minimum residency, hysteresis, and
  deterministic replay make failures reproducible.
- **MoE misprediction:** prefetch can increase bytes and latency. Track wrong-prefetch cost and keep
  the base expert executable.
- **Evaluation leakage:** calibration/test overlap can inflate quality. Freeze dataset splits and
  checksums before final experiments.
