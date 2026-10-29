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

Before any runtime storage timing, the CPU direct reader must pass a functional differential
against the buffered `pread` oracle for the same canonical packets and logical CRCs. Record
`STATX_DIOALIGN`, actual offset/length/buffer alignment, queue depth, submission/completion counts,
logical bytes, physical requested bytes, and file bytes returned. Ordinary `IORING_OP_READ` into a
`PacketRing` slot is not io_uring fixed-buffer evidence. A legacy unpadded terminal tail is a
buffered-only compatibility case and must be excluded from the `O_DIRECT` denominator rather than
silently routed through buffered I/O.

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

For scheduler feedback, measure `critical_path_us` from storage submission until the packet is
device-ready under the lifetime-safe SSD-to-CUDA path. Label every observation with its exact
`service_class`, logical packet bytes, and physical I/O bytes; do not reuse a sample across another
class or byte bucket. Record I/O and H2D component times when their clock domains and endpoints are
well defined, but treat them as audit metadata only. Do not add the components or infer overlap
from them: report overlap separately using the event timeline required above.

### Direct-H2D correctness observations (not acceptance evidence)

`CudaDirectPackFile` now retains the registered ring through original read CQEs and H2D completion.
The native test emits `direct-h2d/host-observed-v1` samples from immediately before the first SQE
preparation through successful host observation of the completion event. Durations include CPU
polling/dispatch, CRC and destination allocation in the critical path. I/O ends after CRC and ready
wrapping; the H2D component starts after allocation, so components intentionally omit intervening
host work. They must not be interpreted as isolated device-transfer rates.

Use an existing scratch directory on the target filesystem:

```sh
HARPQ_DIRECT_IO_TEST_DIR=/target/nvme/harpq-scratch \
  build/native-cuda/native/cuda/harpq_cuda_direct_packet_io_test --json \
  > build/native-service-observations.json
PYTHONPATH=src python scripts/check_native_feedback.py build/native-service-observations.json
```

The assert-based checker must run without Python `-O`. Three completed serial reads at each of
64 KiB, 256 KiB and 1 MiB use two host slots, configured queue depth 8 (effective depth 1), 29 bytes of padding per
packet, and a fresh GPU allocation per upload. The fixture has no calibration warm-up, computes no
model, and verifies every logical byte through D2H. Its samples prove schema ingestion, exact-byte
bucket selection and deterministic replay, not percentile stability or a tuned runtime policy.
Matched fio/H2D runs at representative queue depth, preallocated device storage, and an event
timeline for overlap remain prerequisites for throughput acceptance and production feedback.

`harpq_gemv_benchmark` currently uses a benchmark-local naive static-W4 kernel with matched launch
geometry. It is useful for regression and tuning direction only: its output explicitly sets
`acceptance_evidence=false`, and it cannot satisfy the 85% gate until an optimized production or
library-parity W4 denominator is selected and documented. Before emitting a timing ratio, that
naive kernel must pass its CPU reduction-tree oracle at
`abs(actual - expected) <= 2e-5 + 2e-6 * abs(expected)`; see D-021.

## Bounded SSD/H2D sweep

The opt-in runner creates a new 256 MiB deterministic pack in a unique temporary directory on the
requested filesystem, fsyncs it, and removes only its own fixture at exit. fio receives
`--readonly --allow_file_create=0 --direct=1 --ioengine=io_uring`. No raw device, existing pack,
clock setting, cache-dropping sysctl or other user's workload is modified. fio option semantics
follow the [official fio documentation](https://fio.readthedocs.io/en/latest/fio_doc.html).

From the repository root, after the explicit CUDA-ON/io_uring-ON **Release** build and correctness
suite, run (the output directory must not exist):

```sh
.venv/bin/python scripts/benchmark_transfers.py \
  --build build/native-cuda --scratch /target/nvme/harpq-scratch \
  --output benchmarks/results/transfer-sweep
```

Add `--smoke` for one 128 KiB functional round at 64 KiB, including invalid argument/volume and
corrupt-CRC rejection checks. Smoke rates are not performance evidence. The full sweep uses
64 KiB, 256 KiB and 1 MiB payloads, 16 MiB warm-up, 256 MiB timed volume and five repetitions per
case, reversing size/method order on alternating repetitions. Report the median and full observed
range, not the best run. Methods:

- fio sequential reads at QD1 and QD8; ordinary buffers, submit/completion batches of one. Validate
  successful completion, exact read volume, zero writes and at least 90% of timed requests in the
  target depth bucket. Short warm-up runs retain the depth distribution without that threshold.
- `harpq_transfer_benchmark direct PACK BYTES WARMUP ITERATIONS`: the existing CPU direct reader,
  one ordinary aligned ring slot and QD1, with CRC and dispatch included in host wall time.
- The same executable's `bridge` mode: registered ring, QD1, per-packet GPU allocation/free and
  completion wait included. Pre/post full-payload D2H checks are outside timing; every timed read
  checks CRC. Both modes require logical=physical bytes and record SQE/CQE totals, statx alignment,
  critical-path p50/p95 and host-observed component means. Each native case has two additional
  untimed verification packets; timings include fixed harness checks and sample collection.
- Existing `harpq_h2d_benchmark`: equal payload/warm-up/timed volume, one stream with queued copies
  bracketed by CUDA events, allocation outside timing and sampled D2H verification afterward.
  It is not queue-semantics-matched to serial per-packet runtime waits.

The native QD1/fio-QD1 ratio diagnoses runtime overhead, not a raw NVMe limit. fio QD8 and queued
H2D are separate capability denominators; never divide a host-observed runtime component into the
CUDA-event denominator to claim the 80% gate. `acceptance_evidence=false` remains explicit because
this is a short, warm, synthetic baseline without model/compute overlap, cold-cache control,
production queueing or locked clocks. Read amplification of 1 only describes these fully aligned
fixtures. Preserve all command/stdout/stderr records, binary hashes, GPU snapshots and JSON hashes.

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
