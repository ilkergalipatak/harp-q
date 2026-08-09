# Linux/NVIDIA benchmark protocol

No benchmark in this document is considered executed until its raw JSON metadata and summarized
result are produced on the target NVIDIA/Linux host. Mac CPU/MPS timings are functional evidence
only.

## Environment record

Run the passive profile first, then the active profile on the filesystem that will hold model
packs:

```sh
PYTHONPATH=src python -m harpq profile-host --output artifacts/host.json
PYTHONPATH=src python -m harpq profile-host --active \
  --scratch /target/nvme/harpq-scratch --output artifacts/host-active.json
```

The active command uses a dedicated temporary subdirectory and removes it after the bounded fio
run. Preserve the reports outside Git if they are large; commit only small summaries.

Every published run records GPU name and VRAM, driver, CUDA toolkit/runtime, compiler, CPU, RAM,
PCIe generation/width, NVMe device and filesystem/mount options, model checksum, HARP-Q commit,
batch size, prompt/decode lengths, cache budgets, warm-up count, and random seed.

## Microbenchmarks and gates

1. Measure pinned-host H2D with `harpq_h2d_benchmark` at the runtime's actual transfer size.
2. Measure target-filesystem sequential direct I/O with fio using the same block sizes and queue
   depth as the runtime.
3. Measure static W4 GEMV/GEMM baselines with identical shapes, batches, clocks, and warm-up.
4. Measure each progressive prefix and the fused multi-stage path against the NumPy reference.
5. Measure copy/compute overlap on separate CUDA streams with events, not host wall-time alone.

Acceptance gates from the project plan:

- progressive 4-bit kernel throughput is at least 85% of the static W4 baseline;
- pinned H2D reaches at least 80% of its isolated microbenchmark;
- `O_DIRECT` runtime reads reach at least 75% of matched fio bandwidth;
- read amplification, physical bytes divided by useful packet bytes, is at most 1.2;
- every FP32 CUDA-oracle prefix satisfies
  `abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)` for each finite element. A lower-precision
  path requires its own predeclared dtype/order-specific tolerance. The FP32 fused progressive GEMV
  uses the separately declared `2e-4 + 2e-4 * abs(expected)` bound.

If a gate fails, report the value and reason; never silently relax the denominator or compare
different shapes.

`harpq_gemv_benchmark` currently uses a benchmark-local naive static-W4 kernel with matched launch
geometry. It is useful for regression and tuning direction only: its output explicitly sets
`acceptance_evidence=false`, and it cannot satisfy the 85% gate until an optimized production or
library-parity W4 denominator is selected and documented. Before emitting a timing ratio, that
naive kernel must pass its CPU reduction-tree oracle at
`abs(actual - expected) <= 2e-5 + 2e-6 * abs(expected)`; see D-021.

## End-to-end model protocol

Start with Mistral-7B dense, then Mixtral-8x7B MoE. For each model use at least two constrained
GPU/host-cache budgets and compare against static quantization and an offload baseline. Report:

- time to first token (TTFT), including cold and warm cache;
- median, p95, and p99 inter-token latency;
- tokens/second after warm-up;
- physical SSD bytes/token and useful SSD bytes/token;
- H2D bytes/token and achieved bandwidth;
- copy/compute overlap ratio and deadline misses;
- VRAM, pinned RAM, ordinary RAM, and cache hit rates;
- perplexity or task metric for each realized precision policy.

For Mixtral, include router prediction accuracy, expert-prefetch hit rate, incorrect prefetch bytes,
and expert cold-start latency. Active KV cache stays off SSD. A Pareto claim requires at least one
HARP-Q point that is no worse in quality and latency and strictly better in one dimension under the
same hardware and memory budget.
