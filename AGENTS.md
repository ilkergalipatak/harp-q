# HARP-Q contributor instructions

## Scope

HARP-Q is a research prototype for hierarchy-aware, residual progressive post-training
quantization. The Mac implementation is the deterministic reference implementation. Linux/CUDA
code must preserve its observable semantics.

## Invariants

- A weight tile is `128 x 128` unless a test explicitly exercises edge padding.
- Precision is progressive: four ordered 2-bit residual stages (`2+2+2+2`). Stage `n` depends on
  every earlier stage and cannot be decoded or scheduled alone.
- The base stage must be sufficient to execute. Optional residual packets may improve quality but
  their absence must never make the runtime crash.
- Serialized packets are 4096-byte aligned and integrity checked. Metadata round-trips exactly.
- Scheduling and trace replay are deterministic for a fixed seed, manifest, budget, and trace.
- Do not report CPU or Apple MPS timings as evidence for NVIDIA throughput.
- Active KV cache is never paged to SSD in the first prototype.
- Never commit model weights, secrets, pack artifacts, or large benchmark results.

## Commands

Create an environment with Python 3.11+ and install `requirements/runtime.lock`, then run:

```sh
make test
make smoke
make demo
```

To reproduce the validated packaging toolchain, install `requirements/build.lock` as well. Build
without dependency resolution after those locks are installed:

```sh
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

Inside Codex Desktop on the original Mac, the bundled interpreter can be selected explicitly:

```sh
make test PYTHON=/Users/ilkergalipatak/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Do not silently skip a test because CUDA, `io_uring`, or `O_DIRECT` is unavailable. Platform-only
tests must report an explicit skip reason. CPU reference tests must pass on both Mac and Linux.

## Change protocol

1. Add or update a deterministic unit test with each behavior change.
2. Keep the on-disk format versioned and backward-compatible, or record an explicit breaking
   decision in `docs/DECISIONS.md`.
3. Compare optimized dequantize/GEMM paths against the NumPy reference at every supported prefix.
4. Update `docs/HANDOFF.md` before moving the task between machines.
5. Record performance claims with hardware, driver, CUDA, filesystem, model, batch, context,
   warm-up, and measurement settings.

## Linux phase gate

Before CUDA implementation, capture a hardware profile and rerun all CPU tests. CUDA kernels,
registered pinned-memory buffers, `io_uring + O_DIRECT`, and real-model benchmarks belong on the
NVIDIA/Linux host; their interfaces may be stubbed on Mac but their performance must not be
inferred there.
