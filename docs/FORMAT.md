# HARP-Q container and stage codec v1

Status: executable reference specification. Integer fields are little-endian. Any incompatible
change increments the relevant version and is recorded in `DECISIONS.md`.

## Directory layout

```text
model.harpq/
├── manifest.json
├── index.bin
├── boot.pack
├── base.pack
├── refine-1.pack
├── refine-2.pack
└── refine-3.pack
```

`boot.pack` always exists, including when empty. Stage zero packets live in `base.pack`; stage
`n > 0` packets live in `refine-n.pack`. A stage-zero packet is required and executable. Higher
stages are optional and form an ordered dependency chain.

## Manifest

`manifest.json` is UTF-8 JSON serialized with stable key ordering. It declares:

- format name `harpq-container` and `format_version = 1`;
- packet alignment, currently 4096 bytes;
- `index.bin` size and SHA-256;
- packet count;
- exact size and SHA-256 of every pack;
- user/model metadata containing only finite JSON values.

SHA-256 protects immutable whole-file identity and distribution. It is not a substitute for the
per-packet checksum used after an asynchronous read.

## Binary index

The index starts with `<8sHHI`:

| Field | Value |
|---|---|
| magic | ASCII `HARPQIDX` |
| version | `1` |
| flags | `0` |
| record count | unsigned 32-bit |

Each record is a 32-bit byte length followed by canonical UTF-8 JSON. A record identifies the
packet, logical stage, canonical pack, aligned offset, payload length, CRC32C, dependencies,
optionality, and codec metadata. Length-prefixing makes unknown/truncated records unambiguous while
keeping the reference reader language-neutral.

Every payload offset is a multiple of 4096. Padding bytes are not part of a packet checksum or
packet length. A production packer may later coalesce adjacent tile payloads to reduce read
amplification; such coalescing must remain explicit in the index.

The reference scheduler keeps logical resident bytes separate from physical I/O bytes. Until a
coalescing plan exists, one direct packet read is rounded up to the manifest alignment. Traces
record both counters and their ratio; cache capacity is charged only for logical resident bytes.

## Progressive matrix stage payload

The packet metadata identifies:

- codec `harpq-progressive-stage`, codec version `1`;
- matrix ID and zero-based stage index;
- original, padded, and tile shapes;
- group size and FP16 scale shape;
- packed-code byte count and total available stages;
- caller metadata, including the prefix distortion curve when calibrated.

The payload header is `<8sHHII`:

| Field | Value |
|---|---|
| magic | `HQSTAGE\0` |
| codec version | `1` |
| stage index | unsigned 16-bit |
| packed code bytes | unsigned 32-bit |
| FP16 scale count | unsigned 32-bit |

The header is followed by packed signed-int2 codes and little-endian FP16 scales. Codes use the
two's-complement alphabet `[-2, -1, 0, 1]`; four consecutive values occupy one byte, starting at
the least-significant two bits.

## Dependency and fallback rules

- Stage zero has no dependency and is never optional.
- Stage `n` depends on stage `n-1` of the same matrix.
- A valid request is a contiguous prefix; an isolated residual is invalid.
- Missing `base.pack` is fatal.
- A missing pack is tolerated only when all of its indexed packets are optional.
- Requesting an unavailable optional stage resolves to the nearest available dependency.
- A missing residual also makes its dependent descendants unavailable.

The reader validates metadata, index SHA-256, dependency closure, extent non-overlap, pack
SHA-256, then packet CRC32C. `--skip-sha256` may shorten local inspection but payload CRC32C remains
available through explicit verification.

## Quantization contract

Weights use `(out_features, in_features)` layout and are padded to `128 x 128` tiles by default.
Each of four stages approximates the previous FP32 residual using signed int2 codes and FP16
group scales. The encoder forms the next residual from the exact FP16-decoded stage, so decoder and
encoder prefixes agree. Prefixes 1–4 correspond to nominal 2/4/6/8 code bits per padded weight.

The CPU reference operation accumulates `X @ residual_stage.T` independently. Serialized bytes and
materialized FP32 prefix weights must match exactly. For the current FP32 CUDA cuBLAS oracle, every
finite output element must satisfy
`abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)` for every prefix. A fused kernel may change
arithmetic ordering or output dtype only after a separate tolerance is recorded before tuning; see
D-018 in `DECISIONS.md`.
