# HARP-Q handoff

Last verified: 2026-08-04 on MacBook Pro arm64, macOS/Darwin. This file is the recovery source if
the Codex task cannot be handed off directly.

## Repository

- Remote: `https://github.com/ilkergalipatak/harp-q.git`
- Branch: `main`
- Package: `harpq`
- Python: 3.11+
- Validated Mac runtime: Python 3.12.13, NumPy 2.3.5
- Dependency locks: `requirements/runtime.lock`, `requirements/build.lock`

The repository-relative layout is portable; model weights and generated pack/benchmark artifacts
are intentionally absent from Git.

## Implemented state

- `quantization.py`: deterministic `128 x 128` tile-capable signed-int2 `2+2+2+2` residual
  quantization, packing, prefix decode, and progressive matmul.
- `distortion.py`: activation-weighted MSE, empirical Hessian proxy/error, stable KL, prefix profiles.
- `integrity.py`, `container.py`, `matrix_codec.py`: CRC32C/SHA-256, versioned manifest/binary index,
  4 KiB pack alignment, matrix-stage codec, dependency checks, and optional fallback.
- `allocator.py`: exact precedence-aware multiple-choice knapsack reference.
- `scheduler.py`, `simulation.py`: deadline/cache policy, dependency-safe eviction, physical versus
  logical byte accounting, trace serialization, and exact replay.
- `synthetic.py`: small deterministic Transformer for prefix functional accuracy.
- `hardware.py`: passive capability report and explicitly activated bounded Linux RAM/fio/CUDA
  probes with scratch filesystem and fio provenance.
- `cli.py`: `doctor`, `profile-host`, `demo`, `quantize`, `inspect`, `simulate`, and `replay`.
- `FORMAT.md` and `BENCHMARK_PROTOCOL.md`: executable format and Linux measurement contracts.

## Verified commands

On the original Mac, without installing dependencies:

```sh
make check PYTHON=/Users/ilkergalipatak/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
make demo PYTHON=/Users/ilkergalipatak/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Current result: **62 unit tests pass**, followed by a successful CLI doctor smoke test. The wheel
also builds and installs with:

```sh
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /tmp/harpq-wheel
```

The deterministic functional demo reported all finite prefixes and logits MSE versus the dense
fixture of approximately `0.425010`, `0.0525473`, `0.0145978`, and `0.00241263` for nominal
2/4/6/8-bit prefixes. This is functional evidence only, not an NVIDIA performance result.

## Known limits — do not overclaim

- No CUDA source, custom op, fused kernel, pinned-memory runtime, or async I/O backend exists yet.
- No real Mistral/Mixtral checkpoint adapter or evaluation has run.
- Active Linux hardware profiling has not run on this Mac.
- The current encoder measures activation/Hessian distortion but its code fitting is residual
  least-squares round-to-nearest, not yet a multi-prefix GPTQ-style optimizer.
- Packet coalescing is not implemented. The scheduler conservatively rounds standalone physical
  reads to 4 KiB; tiny fixtures therefore have deliberately high read amplification.
- No NVIDIA throughput, H2D, NVMe, latency, quality–latency Pareto, or novelty claim is validated.

## Transfer this task to NVIDIA/Linux

1. On Linux, install the NVIDIA driver, matching CUDA toolkit, C++ compiler, Git, Python 3.11+, fio,
   and Codex CLI/host prerequisites.
2. Clone this remote. Preserve the repository-relative directory structure; a conventional target
   is `$HOME/projects/harp-q`.
3. Install locked CPU dependencies and run all reference tests before adding CUDA code:

   ```sh
   python3 -m venv .venv
   .venv/bin/pip install -r requirements/runtime.lock -r requirements/build.lock
   make check PYTHON=.venv/bin/python
   ```

4. Verify SSH access from the Mac to the Linux host.
5. In Codex Desktop, open Settings → Connections, add the SSH host as a remote project, then use the
   task footer’s working-location control to hand this task to that host.
6. Confirm `pwd`, `git remote -v`, `git status --short --branch`, and the test output before changing
   files.
7. Run passive and active hardware profiles on the actual model-pack filesystem:

   ```sh
   PYTHONPATH=src .venv/bin/python -m harpq profile-host \
     --output artifacts/linux-host.json
   PYTHONPATH=src .venv/bin/python -m harpq profile-host --active \
     --scratch /path/on/target/nvme \
     --output artifacts/linux-host-active.json
   ```

The active profile allocates two 64 MiB RAM buffers, creates a dedicated temporary fio directory in
the supplied scratch mount, performs serialized 256 MiB direct write/read jobs, attempts installed
CUDA `deviceQuery`/`bandwidthTest`, records mount/fio evidence, and removes only that temporary
directory.

## First Linux implementation sequence

1. Commit a small hardware-profile summary and record driver/toolkit/filesystem versions.
2. Add a CUDA dequantize-plus-cuBLAS oracle and compare every prefix with NumPy.
3. Add fused progressive GEMV, then GEMM, with static W4 microbenchmarks.
4. Implement a registered pinned-RAM ring and measured H2D path.
5. Implement `io_uring + O_DIRECT` reads into the ring, keeping CRC and fallback semantics.
6. Add copy/compute streams and CUDA events; feed measured transfer/kernel costs to scheduler traces.
7. Determine packet coalescing from the target NVMe curve and enforce read amplification ≤1.2.
8. Add Mistral-7B dense integration, then Mixtral-8x7B router-aware expert/refinement prefetch.

Use `BENCHMARK_PROTOCOL.md` for gates. Do not move active KV cache to SSD.

## If direct task handoff fails

Start a new Codex task in the cloned repository and give it this instruction:

> Read AGENTS.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/FORMAT.md,
> docs/BENCHMARK_PROTOCOL.md, docs/RESEARCH_PLAN.md, and docs/PRIOR_ART.md completely. Run the CPU
> suite. Continue from “First Linux implementation sequence” without changing the v1 observable
> semantics unless a new decision is recorded.

Before stopping any future work session, update this file with the branch, verified commands,
measured environment, unresolved failures, and one exact next action.
