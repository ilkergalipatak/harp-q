# Prior art and claim matrix

Survey snapshot: 2026-08-04. This is an engineering literature screen, not a patent freedom-to-
operate opinion. A publication claim still requires a fresh scholarly and patent search before
submission.

## Closest systems and methods

| Work | Relevant mechanism | Overlap with HARP-Q | Boundary HARP-Q must demonstrate |
|---|---|---|---|
| [GPTQ](https://arxiv.org/abs/2210.17323) | One-shot, approximate second-order weight PTQ | Hessian-aware calibration objective | One parent with several runtime prefixes plus hierarchy/deadline execution |
| [AWQ](https://arxiv.org/abs/2306.00978) | Activation-aware salient-channel scaling and efficient W4 serving | Activation statistics and hardware-aware packing | Successive residual prefix objective, not only one W4 representation |
| [SqueezeLLM](https://arxiv.org/abs/2306.07629) | Sensitivity-aware non-uniform PTQ and dense/sparse outliers | Second-order sensitivity and memory-bound decode | Ordered residual packets and adaptive physical placement |
| [AQLM](https://arxiv.org/abs/2401.06118) | Additive/multi-codebook quantization with activation-aware reconstruction | Additive residual representation and low-bit kernels | Prefix dependency, tier-aware packetization, deadlines, fallback |
| [Any-Precision LLM](https://arxiv.org/abs/2402.10517) | Multiple precisions overlaid in one model footprint with a serving engine | Single checkpoint supporting several bit widths | SSD/RAM/GPU packet flow and per-step deadline/cache policy |
| [BitStack](https://arxiv.org/abs/2410.23918) | Training-free residual decomposition into small, ordered blocks loaded for current memory | Extremely close to “stack residuals and load what fits” | Do not claim this concept; test hierarchy/deadline/I/O/kernel co-design beyond memory-size adaptation |
| [MatGPTQ](https://arxiv.org/abs/2602.03537) | PTQ Matryoshka bit slicing, cross-bit compensation, budget search, mixed-precision kernels | Extremely close to one-shot multi-prefix PTQ and heterogeneous bit allocation | Residual packet semantics plus measured GPU–RAM–SSD runtime behavior must be the differentiator |
| [FlexGen](https://arxiv.org/abs/2303.06865) | LP-planned GPU/CPU/disk tensor placement and 4-bit compression for throughput | Three-tier model execution and offline placement optimization | Interactive/decode deadlines, progressive quality packets, fused prefix execution |
| [LLM in a Flash](https://arxiv.org/abs/2312.11514) | On-demand flash weights, windowing, and large contiguous reads | SSD demand loading and flash-aware transfer granularity | Progressive precision representation and NVIDIA tier/cache scheduler |
| [M2Cache](https://arxiv.org/abs/2410.14740) | Mixed precision plus HBM/DRAM/SSD caches at neuron/layer granularity | Direct overlap: three tiers, importance, dynamic precision, SSD full model | Ordered residual prefix contract, dependency-safe fallback, physical-byte/deadline optimization |
| [HOBBIT](https://arxiv.org/abs/2411.01433) | Token/layer/sequence mixed-precision MoE expert load, prefetch, and cache | Direct overlap for Mixtral expert offload and adaptive precision | Router-aware residual prefixes and unified dense/MoE format/runtime evidence |

The matrix shows that none of the following phrases are defensible as broad novelty claims:

- “the first system to run an LLM from SSD”;
- “the first GPU–RAM–SSD LLM cache”;
- “the first dynamically sized or progressive quantized LLM”;
- “the first residual-block model representation”;
- “the first mixed-precision MoE expert offloader”;
- “the first activation- or Hessian-aware PTQ method.”

## Claim-threat matrix

| Candidate claim | Strongest threat | Evidence required before using the claim |
|---|---|---|
| Deadline-aware selection of nested residual precision packets across GPU/RAM/SSD | M2Cache, BitStack, FlexGen | Algorithmic distinction plus latency/quality ablation against memory-only selection |
| Direct-I/O packet format with executable base and corruption/late-residual fallback | LLM in a Flash, storage/offload systems | Format semantics, physical read amplification, failure tests, and end-to-end benefit |
| Fused progressive kernels consuming cached residual prefixes without parent reconstruction | MatGPTQ, Any-Precision, AQLM kernels | Kernel design comparison and ≥85% matched static-W4 gate |
| Joint offline quality/byte placement and online deadline/cache refinement | MatGPTQ budget search, FlexGen LP, M2Cache policy | Formal objective, solver/policy ablations, two-budget Pareto evidence |
| Router-aware precision-prefix prefetch for MoE experts | HOBBIT | Residual-specific mechanism and measured useful versus wrong-prefetch bytes |

The likely paper contribution is therefore a **co-designed storage-to-kernel execution contract**,
not a standalone quantizer or a standalone offloader. If experiments show only parity with the
closest work, the honest result is an implementation study rather than a new-method claim.

## Design implications from prior art

1. BitStack makes ordered residual loading an explicit prior-art risk. HARP-Q must compare against
   memory-only residual stacking and quantify why deadlines, cache state, and physical I/O change
   the chosen prefixes.
2. MatGPTQ makes multi-prefix PTQ, cross-bit compensation, budget search, and kernels a direct
   algorithmic baseline. A naive `2+2+2+2` residual encoder is a reference, not publishable novelty.
3. M2Cache already joins mixed precision with HBM/DRAM/SSD. HARP-Q needs stronger prefix
   dependency/fallback semantics and matched measurements, not just different naming.
4. LLM in a Flash shows that fewer, larger contiguous reads can dominate clever fine-grained
   selection. Packet coalescing and physical bytes must be first-class.
5. HOBBIT sets the MoE baseline: routing, prefetch, and cache policy must be compared at token,
   layer, and sequence timescales.
6. GPTQ/AWQ/SqueezeLLM/AQLM show that low-bit quality needs calibration-aware treatment. The
   current symmetric residual RTN path is useful as an oracle but insufficient as the final method.

## Additional baseline families for experiments

- Static weight-only W2/W3/W4/W8 using a strong maintained implementation.
- Dequantize-then-GEMM oracle and optimized static W4 kernels.
- Uniform versus heterogeneous per-layer prefixes.
- No-cache, LRU-only, memory-capacity-only, deadline-only, and full HARP-Q scheduling.
- Buffered I/O versus `io_uring`, direct versus cached I/O, and coalesced versus per-packet reads.
- Dense Mistral-7B and MoE Mixtral-8x7B under the same two or more memory budgets.

## Source list

- GPTQ, arXiv:2210.17323 — https://arxiv.org/abs/2210.17323
- AWQ, arXiv:2306.00978 — https://arxiv.org/abs/2306.00978
- SqueezeLLM, arXiv:2306.07629 — https://arxiv.org/abs/2306.07629
- FlexGen, arXiv:2303.06865 — https://arxiv.org/abs/2303.06865
- LLM in a Flash, arXiv:2312.11514 — https://arxiv.org/abs/2312.11514
- AQLM, arXiv:2401.06118 — https://arxiv.org/abs/2401.06118
- Any-Precision LLM, arXiv:2402.10517 — https://arxiv.org/abs/2402.10517
- M2Cache, arXiv:2410.14740 — https://arxiv.org/abs/2410.14740
- BitStack, arXiv:2410.23918 — https://arxiv.org/abs/2410.23918
- HOBBIT, arXiv:2411.01433 — https://arxiv.org/abs/2411.01433
- MatGPTQ, arXiv:2602.03537 — https://arxiv.org/abs/2602.03537
