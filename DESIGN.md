# volta: batched DeepSeek-V4.1-Flash on 8x V100

Goal: the most aggregate tokens per second for many concurrent agent sessions on one 8x V100-SXM2-32GB box,
with quality kept above a fixed limit. Speed comes first. A weight format that runs faster on Volta wins even if
it stores fewer bits per byte than another one, as long as quality stays above the limit.

This is the original design, with dated amendments at the end where measurements changed the plan.

## The box (measured 2026-10-21)

| | Measured |
|---|---|
| GPUs | 8x V100-SXM2-32GB, 274.9 GB HBM total, 1530 MHz |
| HBM read | 868 GB/s per card |
| FP16 tensor cores (cuBLAS, FP32 accumulate) | 100 TFLOPS at M=8192, 82 at 512, 40 at 128, 15 at 32 |
| NVLink copies | 48.5 GB/s on double-link pairs, 24.3 on single-link pairs, 10 GB/s between cards with no NVLink. ~8 µs per small copy |
| NCCL 2.25 collectives, 64 KB | all-gather 38 µs, all-reduce 56 µs, reduce-scatter 59 µs (latency floor) |
| Kernel launch | 1.8 µs plain, 1.24 µs inside a CUDA graph |
| H2D pinned | 12.4 GB/s per card |
| CPU / RAM | 2x Xeon Gold 6248 (AVX-512 with VNNI), 80 threads, 754 GB over 2 NUMA nodes |
| 4-bit GEMV, CUDA cores, magic-number dequant | 746 GB/s at batch 1, 638 at 2, 465 at 4, 305 at 8 |
| 2-bit GEMV, same kernel | 627 GB/s at batch 1, 383 at 2, 212 at 4, 124 at 8 |

The double-link pairs are (0,3), (1,2), (4,7) and (5,6).

Baseline: llama.cpp with PR #30817 and UD-IQ1_M decodes at 26.7 tok/s for one session and stops scaling at
72-97 tok/s aggregate. vLLM, SGLang and DeepInfer cannot run here: the official checkpoint is ~320 GB before
Engram, and Volta has no FP8 or FP4 math (`bench/v100-engines-2026-10-21.md`).

## Where the time goes

At 128-256 sessions, a decode step touches nearly every expert (330-377 of the 384 per layer), so **a step reads
the whole expert set once**. Step time is therefore roughly expert bytes per GPU divided by achieved bandwidth, and:

    aggregate tok/s  ≈  tokens per step / (expert bytes per GPU / bandwidth + fixed costs)

Three levers follow from that:

1. **Fewer expert bytes** give a shorter step, and the relationship is linear. 2.5 bpw steps ~40% faster than 4.25 bpw.
2. **More tokens per step** cost almost nothing until compute or KV memory runs out. Use more sessions, DSpark
   draft tokens, and chunked prefill mixed into decode steps.
3. **Fixed costs** must stay small next to a ~25 ms step: launches, collectives, attention, and the dense half.

Prefill is limited by compute rather than bandwidth. With bounded replay (layers 20-39 run only on the prompt's
last ~140 tokens), it costs ~16 GFLOP per token. At ~50 TFLOPS effective per card, the 8 cards can prefill
~10-20K tok/s. Agent loops add more input than output, so mixing prefill into decode steps matters as much as decode speed.

## Memory plan per GPU (34.4 GB)

| Item | GB per GPU | How |
|---|---|---|
| Routed experts (EP8: 48 experts per layer per GPU) | ~25 | the quant budget, ~2.9 bpw average |
| Attention + shared expert + head + Engram projections | ~3 | attention TP2 inside a double-link pair, the rest split 8 ways |
| KV cache, compressed, 890 B/token in the native FP4 layout | ~3.5 | ~16M tokens across the 4 attention groups |
| Per-session window state (~2.8 MB each) | ~0.7 | 256 sessions |
| Activations, workspaces, CUDA graphs | ~1.5 | |
| DSpark draft (optional, phase 5) | ~0.7 | 3 layers at ~3 bpw, split 8 ways |

Engram's 196 GB stays in pinned host RAM. GPU kernels read its rows directly over UVA, the way vLLM does.
Lookups depend only on token ids, so they can be issued as soon as a token is sampled.

Idle sessions park their KV in host RAM, which has room for ~800M tokens. A session waiting on a tool call costs
no GPU memory, and when it comes back its KV is copied back at 12 GB/s per card instead of being prefilled again.

## Parallelism

- **Attention: 4 groups of TP2**, one per double-link pair. Each group holds a quarter of the sessions and their
  KV. TP2 halves the attention weights per card while copying the KV only twice instead of eight times.
- **MoE: EP8.** After attention, each group's tokens are all-gathered to every card (one collapsed 5120-wide
  vector per token, about 2.6 MB per layer at 256 tokens). Each card runs its local experts, and a reduce-scatter
  returns the results.
- **Our own collectives instead of NCCL.** NCCL's 40-60 µs floor at small sizes costs ≥4 ms per step at 2
  collectives per layer. We'd use one-shot peer-to-peer stores into peer memory, with ready flags, routed over
  the NVLink mesh and taking two hops for the pairs with no link. The target is ≤10 µs per collective.
- **One CUDA graph per batch-size bucket**, and a persistent kernel per layer for the routed experts. The router
  writes a device-side list of (expert, token group) pairs, as `route_t` in `amd/kernels/prefill.h` does, so the
  CPU never syncs inside a step.

## Weight formats, chosen for speed

The format decides the kernel's instruction count per weight, and on Volta that, not HBM, is the limit as soon
as an expert sees more than one token. The candidates below are ranked by measured or expected cost on sm_70.

| Format | bpw | Dequant on Volta | Quality source |
|---|---|---|---|
| MXFP4 (the checkpoint's own experts, exact) | 4.25 | e2m1 → fp16 bit trick, or a 16-entry `prmt` lookup | exact |
| cb3: 8-level codebook + e4m3 scale per 32 | 3.25 | `prmt` lookup from 4 registers, 2 weights per ~2 instructions | rel. error 0.021 vs FP4 |
| cb2: 4-level codebook | 2.25 | same, cheaper | rel. error 0.10 |
| int4 / int2 with fp16 scales | 4.25 / 2.25 | magic-number trick (measured above) | GPTQ + Hadamard tooling exists |
| int8 activations with dp4a | — | 4 MACs per instruction vs 2 for HFMA2, so it doubles small-batch math | needs per-token activation scales |
| Expert pruning (e.g. a REAP-style 272-of-384) | — | removes bytes outright; the rest could stay exact MXFP4 | must measure routing quality |

Rules for the kernel-friendly choice:
- **Group size matches the tile** (32 along K) so scales load once per fragment.
- **Weights are pre-swizzled at load** into the order the `mma.m8n8k4` FP16 fragments (or the GEMV lanes) consume.
- **One format per expert, a few formats per layer.** The persistent MoE kernel dispatches on format per work
  item, and mixing formats inside an expert is not allowed.
- **Uniform bytes per card.** EP placement balances bytes and expected picks across the 8 cards, because the
  slowest card sets the step time.

The allocator (`quant/v41/v41_alloc.py`, error per byte with a global budget) gets a second cost term: **measured
kernel time per expert** for each format at the expected batch per expert, instead of bytes alone. The objective
becomes the most tok/s subject to KL against FP8 ≤ the limit.

Proposed quality limit: KL ≤ 0.12 on the Lucebox eval, about Unsloth UD-Q2_K_XL's quality (0.114). Our
MXFP4+cb3 build measured 0.049 at 270 GB, and dsx 2-bit 0.373 at 148 GB, so ~200 GB should land in between. That
is unmeasured until phase 3.

## Kernels to write (sm_70)

| Kernel | Notes |
|---|---|
| Expert GEMV, 1-2 tokens per expert | CUDA cores, register dequant, 2-4 rows per warp. 746 GB/s at 4-bit batch 1 already measured |
| Expert grouped GEMM, ≥3 tokens per expert | register dequant into `mma.m8n8k4` FP16 tiles. The FP16 matrix units do the multiplies so dequant is paid once per weight |
| Prefill expert GEMM | dequant to shared memory, then a tensor-core GEMM (cuBLAS-class, 80-100 TFLOPS at large M) |
| Sparse attention + indexer, batched | compute-bound on Volta (~390 FLOP/byte). Load each KV row once for all 64 heads and all verify tokens; FP16 HMMA for QK and PV; double-buffered loads in place of `cp.async` |
| Dense GEMVs (attention, shared expert, head) | FP16 or int8 weights, TP2/TP8 splits |
| Hyper-connections, norms, Sinkhorn, router | FP32. Port `nvidia/src/kernels/dsx.cu` from bf16 to fp16/fp32 |
| P2P collectives | all-gather and reduce-scatter over NVLink with flags |
| Engram gather over UVA | from pinned host memory, deduplicated per prefill chunk |

Numerics: there is no bf16 on Volta, so the residual stream, hyper-connection mixing, norms and router stay in
FP32. GEMM inputs are FP16 with per-token scales, and the SwiGLU clamp at 10 keeps expert inputs in range.
Correctness is checked against the CPU reference in `core/src/model.rs` and the Lucebox eval.

## Reuse

- **pocket-engine** `src/sched.rs`: continuous batching, admission with whole-request KV reservation, a chained-hash
  prefix cache over 64-token units, ngram lookup. `src/engine.rs` / `src/gpu.rs`: CUDA graph per padded batch size.
  Sampling kernels in `qwen35.cu`.
- **dsx** `core/`: model config, GGUF and format readers, the CPU reference, and the OpenAI/Anthropic server with
  tools. `nvidia/src/kernels/dsx.cu`: GEMVs, hyper-connections, indexer, compressor, top-k, Engram gate (bf16 → fp16 port).
  `nvidia/src/gpumodel/prefill.rs`: chunked prefill with bounded replay.
- **dsx-amd** `amd/kernels/verify.h` `vgroup` and `prefill.h` `route_t`: device-side grouping by expert. `amd/src/load.rs`:
  MXFP4 and cb3/cb2 packing.
- **DeepInfer**: the async scheduler (plan step N+1 while step N runs), the confidence-scheduled DSpark verify,
  the peak-minimising expert placement (`eplb/policy.py`).

## Phases and gates

| Phase | Work | Gate |
|---|---|---|
| 1. Correct and distributed | sm_70 port of the dsx decode path; EP8 + TP2 with P2P collectives; existing v5 2-bit build (already on the box) | logits match dsx's GPU build for the same weights (same KL); batch-1 tok/s recorded |
| 2. Batched decode | paged compressed KV; pocket-engine scheduler; persistent MoE kernel with GEMV and mma paths; CUDA graphs | aggregate tok/s at 32/128/256 sessions ≥ 50% of the roofline from the measured bandwidth |
| 3. Box-tuned quant | kernel time per format as a cost term in the allocator; MXFP4 / cb3 / cb2 / int; optional pruning | fastest build with KL ≤ the limit; tok/s vs KL curve published |
| 4. Agentic prefill | bounded replay, chunked prefill mixed into decode, prefix cache, KV parking in host RAM | replay a recorded agent trace; measure the input:output ratio and sustained tok/s |
| 5. DSpark | draft layers, confidence-scheduled verify | tok/s per session at 64 sessions, against phase 4 |
| 6. Serve | `core/src/server` on top | OpenAI/Anthropic API under load, parity with dsx's tool tests |

Each phase ends with its numbers in the amendments below.

## Risks

- **FP16 range** in places the reference rounds to bf16: hyper-connection outputs, router logits, attention.
  Near-tie expert flips change outputs. Phase 1 checks this first.
- **Small-batch arithmetic.** 2-bit at batch 4-8 per expert is limited by math on CUDA cores (124-212 GB/s
  measured), so the mma path must work before low-bit formats pay off.
- **Hybrid-cube collectives.** Cards with no NVLink between them need two-hop routing, and the slowest path sets
  the collective time.
- **Quality of a ~2.9 bpw mix** is unmeasured.

## Amendments

### 2026-10-21: feasibility tests 1 and 2

**Expert kernel (`bench/v100-moe-m884.cu`).** WMMA was limited by shared memory: weights unpacked to shared memory and
read back cost ~6 bytes of shared traffic per weight, which capped it at 160-330 GB/s whatever the token count. The
kernel that works uses raw `mma.m8n8k4` with operands in registers:
- A = weights: each thread unpacks its own row (shift, lop3 into fp16 1024.0, exact hsub2, hmul2 by the scale).
  Folding the bias into one hfma2 rounds `(1024+off)·scale` in fp16 and costs up to half a quant step, so it is not used.
- B = activations from shared memory, stored with k permuted to match the unpack order.
- **Weights are stored interleaved per warp**: `[32-row group][16-byte unit][lane]`, so one warp load is one
  contiguous 512-byte read. Row-major storage ran at 70-115 GB/s because each warp load touched 32 cache lines.
- The output layout of `m8n8k4` is in `bench/v100-m884-layout.cu`: lane L's accumulator e holds row
  `8q + (L&1) + 2((e>>1)&1) + 4h` and token `2((L>>1)&1) + (e&1) + 4(e>>2)`, with q = (L>>2)&3 and h = L>>4.

| Tokens per expert | 4-bit GB/s | 2-bit GB/s |
|---|---|---|
| 1 | 750 | 585 |
| 4 | 712 | 547 |
| 8 | 662 | 496 |
| 16 | 440 | 323 |
| 32 | 200 | 122 |

Results match the CPU reference (relative error 2e-4 at 4-bit). At ≤8 tokens per expert, which covers most experts at
256 tokens per step, the kernel reads at 76-86% of HBM. At 16-32 tokens it is limited by how the tensor cores are fed
(one k-step of 4 per mma, little operand reuse); experts that hot get a shared-memory-staged kernel like prefill's.
2-bit reads at ~78% of 4-bit's bandwidth with half the bytes, so its expert step takes ~64% as long as 4-bit's.

**Collectives (`bench/v100-cube-allgather.cu`).** Pushing to all 7 peers was slow because 3 of them go over PCIe
(2 ms for a 2.2 MB gather). The two-stage cube route (stage 1: own chunk to the 3 quad-mates and twin g^4; stage 2:
forward the twin's chunk to the quad-mates) uses NVLink only:

| Per-GPU chunk | Cube all-gather | NCCL all-gather |
|---|---|---|
| 8 KB | 15 µs | 38 µs (64 KB) |
| 128 KB | 25 µs | — |
| 320 KB (256 tokens of 5120 fp16, split 8 ways) | 40 µs | — |

Best with 8 blocks, one load feeding all destinations, and one system fence per stage. At 2 collectives per layer that
is ~1-3 ms per ~25 ms step. Two easy cuts remain: send each token only to the GPUs that hold its experts (~4.4 of 8 on
average), and send int8 activations with a per-token scale. Phase 1 uses the cube route as written.

**Process change (lxe):** accuracy gets spot checks during development and a full eval at the end of each phase; there
is no logit-matching gate. DSpark (phase 5) moves up as soon as batched decode works, because the step time is flat up
to ~256 tokens per step, which leaves room for draft tokens.

### 2026-10-21: test 3, the old dsx on Volta, and the phase 1 layout

**FP16 numerics (CPU reference, 95-token spot check, `/root/dq-fp16` on the box).** Nothing overflows FP16: the largest
GEMM input is ~150 (SwiGLU output before the down projection), and the residual (up to 8,779) and router stay FP32 as
planned. KL against FP8 went from 0.059 to 0.071 with FP16 at the GEMM inputs (excluding position 0, which is noisy).
That is small next to what a lossy quant costs. If it matters later, bf16 rounding of the residual costs three integer
ops in FP32 registers.

**The old dsx on Volta.** The kernels now compile for sm_70 (branch `volta`): the five Ampere-only kernels are guarded to
sm_80, attention falls back to the scalar `sparse_attn`, and `DSX_OLD_MOE=1` selects the plain expert kernels. A run
placed all ~16,000 experts on 6 GPUs, then hung while loading the main GPU (two CPU threads spinning, ptrace blocked in
the container). Not worth debugging: the new engine reuses the kernels, not the loader.

**Phase 1 layout: data-parallel attention + expert-parallel MoE, dense weights replicated** (DeepInfer's decode layout).
The v5 build's experts are ~130 GB, so each GPU fits a full copy of the dense weights (~8 GB at Q8) plus 1/8 of the
experts (~16 GB) and its sessions' KV. Each GPU runs attention, the router and the shared expert for its own sessions
only. Per layer, the MoE all-gathers the routed tokens to every GPU (cube route), each GPU runs its local experts, and
the partial outputs go back to each token's home GPU. There are no tensor-parallel collectives. The cost is that every
GPU reads the whole dense half each step (~9 ms at Q8); 4-bit dense weights or TP2 can cut that later.

### 2026-10-21: phase 1 result

The engine (`nvidia/src/volta.rs`, binary `volta`) runs the v5 2-bit build across all 8 GPUs with the phase 1 layout.

| | Result |
|---|---|
| Quality, Lucebox eval (8 x 1024, all 8 sequences decoded together, Q8 dense) | KL 0.3746, top-1 74.4%, ppl 10.60; dsx's own figures for this model family are 0.373 / 74.4% / 10.68 |
| Throughput, 64 sessions (8 per GPU), 6-bit dense | 181 ms/step, 354 tok/s aggregate, 5.5 tok/s per session |
| Throughput, 8 sessions (1 per GPU) | 92 ms/step, 87 tok/s |
| Load time | ~200 s (Q8 dense) to ~330 s (6-bit dense requant) |

Things found on the way, all fixed:
- v5's experts are stored Hadamard-rotated (`expert_rot`): the routed experts take FWHT(x), and the SwiGLU output gets
  the FWHT before the down projection.
- Q3_K experts (2,263 of 15,360 in v5) need a scale per 16 to convert exactly; the vmoe format now carries two
  scales per 16-byte unit (per 16 weights at 4-bit, per 32 at 2-bit).
- Round-robin expert placement left two GPUs short of memory; experts are now assigned per layer to balance bytes.
- Memory is tight with replicated Q8 dense weights (0.1 GB free with 2 slots). 6-bit dense planes give ~1.7 GB of
  headroom; their quality cost is not measured yet.

Accuracy spot checks against the CPU reference are noisy: FP8 fake quantization of KV and attention outputs, and near-tie
router scores, amplify FP16-level differences (layer 2's 6th and 7th expert scores differ by 0.002 at position 0). The
eval against the FP8 teacher is the reliable check.

### 2026-10-21: what runs now (phases 2-3)

The design changed in four places, all from measurements:

1. **Weights come from the official checkpoint, not a GGUF.** Each expert is re-packed on the GPU at load from
   its own MXFP4 into one of three lookup formats: MXFP4 exact (4.25 bpw), cb3 (3.25 bpw) and cb2 (2.25 bpw). A
   tier per expert comes from routing counts: the coldest experts drop to cb3, then to cb2, until the experts fit
   the byte budget. The kernel (`vlut_*`) decodes two weights per shared-memory lookup into `mma.m8n8k4` fragments.
   The dense half is 4-bit with an FP16 scale per 16 weights.
2. **Attention is data-parallel, with no TP2.** Each GPU owns its sessions' KV and runs their attention and dense
   layers. Every token's x is all-gathered to every GPU once per layer, and partial sums come back by all-to-all.
   Both collectives are our own NVLink kernels with in-kernel flags (`vx_allgather`, `vx_alltoall`), at ~20-30 µs
   each.
3. **Every routed expert is split 8 ways** (`VOLTA_EXPERT_TP`): each GPU holds 1/8 of every expert's
   intermediate dim (gate/up rows and down columns), as well as 1/8 of the shared expert. With whole experts
   placed 1/8 per GPU, the GPU that drew the most active experts in a layer set the pace, and the others waited
   ~180 µs per layer at the exchange. Split experts give every GPU the same work.
4. **One CUDA graph per step**, keyed by tokens per GPU and a context bucket. Routing is built on the device
   (count, scan, fill), and lists with more than 16 tokens are compacted before their launch.

Quality and speed with 27.5 GB of experts per GPU (220 GB total: 15,172 experts in cb3 and 188 in cb2, none in
MXFP4):

| | Result |
|---|---|
| Lucebox eval (8 x 1024 tokens, teacher top-256) | KL 0.130, top-1 84.6%, ppl 8.97 |
| 64 sessions (8 per GPU), real text | 58.7 ms/step, 1,091 tok/s aggregate, 17.0 tok/s per session |

One 64-session step on GPU0, in ms:

| Part | ms | Floor at 868 GB/s |
|---|---|---|
| Routed experts (`vlut_cb3*`, all token buckets) | 25.0 | ~17 |
| Dense 4-bit matmuls (attention, shared slice, head) | 9.2 | ~3.7 |
| Attention (`vattn_tile5`) | 2.2 | |
| Collectives (all-gather + all-to-all) | 2.2 | |
| Hyper-connections, norms, router, small kernels | ~8 | |

Kernel facts measured on the way (`tests/split_test.cu`, `tests/dense_test.cu`):
- The lookup kernel is limited by load latency at its occupancy. One row group per warp (64 registers) reaches
  ~680 GB/s; four per warp (170 registers) reach ~550. A plain read of the same word layout reaches ~850 GB/s at
  full occupancy. Capping registers to raise occupancy spills and runs slower, and so does prefetching two units
  ahead.
- Bank conflicts in the 64-entry cb3 table cost nothing measurable; a per-bank copy of the table did not help.
- Dense matmuls at 8 tokens run best with one row group per warp and the longest K chunk that still gives at
  least 3 blocks per SM: 145 µs per layer, against 250 µs before and a 91 µs floor.

The 33 ms step that 30 tok/s per session needs at ~64 sessions is below what kernel fixes alone reach (~43 ms
with every kernel at its floor). The DSpark draft layers (3 layers in the checkpoint, 5-token blocks) are the next
lever, because verifying several draft tokens per session in one step reads each expert's weights once.

### 2026-10-22: multi-token steps and DSpark

A step now takes several tokens per session (up to 64 per GPU). Each token gets its own position; the window ring
keeps 64 spare rows and the compressor state a whole group per 64 tokens, so a step never overwrites rows its own
earlier tokens still read. Feeding 4, 32 or 64 tokens per sequence per step gives the same KL as token by token
(0.128). This is also the prefill path: 8 sequences prefill at ~680 tok/s with 64 tokens per GPU per step.

The DSpark draft runs on the GPUs as 3 extra layers after the main ones, so routing, the NVLink exchange and the
8-way expert split serve it unchanged (its 384 experts in cb3, ~0.7 GB per GPU). Its attention reads a window of
the main model's hidden-state means at layers 37-39, projected and stored per slot as main tokens are accepted.
A pass drafts k tokens per session (~12 ms, not yet in a CUDA graph), verifies them in one main step and rewinds
the rejected ones.

Verification is not free at 64 sessions: a step with 6 tokens per session costs ~157 ms against ~55 ms for 1, so k
has to follow acceptance. Greedy decoding, 24 passes after a 256-token prompt:

| Text | Sessions | k | Tokens per pass | tok/s per session |
|---|---|---|---|---|
| prose (eval text) | 64 | 0 / 1 / 3 / 5 | 1.00 / 1.57 / 1.92 / 2.02 | 17.5 / 17.9 / 15.3 / 12.0 |
| code (this repo's Rust) | 64 | 0 / 1 / 3 / 5 | 1.00 / 1.88 / 3.23 / 4.21 | 18.3 / 21.7 / 26.0 / 25.0 |
| code | 8 | 0 / 3 / 5 | 1.00 / 3.40 / 4.28 | 28.9 / 69.8 / 82.6 |

Next: the draft step in a CUDA graph (12 ms -> a few), k chosen per session from the confidence head or recent
acceptance, and sampling (only greedy verification so far).

### 2026-10-22: draft policy and where a verify step goes

The draft step runs as a CUDA graph (observed rows padded to 64). `spec --adaptive` picks each session's drafts
from its recent per-draft acceptance p: the most drafts j <= k with p^j >= 0.35. Every GPU then verifies the same
number of tokens, the busiest GPU's rounded up to a multiple of 8. The step waits for that GPU anyway, so the
spare slots go to the sessions likeliest to accept one more draft, and the few shapes stay in CUDA graphs.
Without the equal counts and rounding, the shapes vary step to step, run without graphs, and cost ~25-30% more.
The draft's experts are cb2: they accept as many drafts as cb3, and ~0.2 GB per GPU is needed for graphs.

64 sessions, k <= 5, 48 passes: code 25.2 tok/s per session (best fixed k: 26.0), prose 17.6 (best fixed: 17.9;
no drafts: 17.5). One setting serves both.

One pass at 64 sessions, k = 3 (GPU0, nsys): the draft step spans 10.5 ms, 5.3 of it in dense 4-bit matmuls at
40 tokens per GPU (32-token tiles read each weight, the head included, twice). The verify step spans 107 ms at 32
tokens per GPU: routed experts 49 ms, dense matmuls 18.5, attention 15.8, collectives 6. Attention grows with tokens
(2.2 ms at 8 per GPU): it runs on CUDA cores, and with one KV head shared by all 64 query heads each token's
attention is a [64 x 512] x [512 x rows] product, tensor-core work. The 32-token expert and dense kernels are
limited by their inner loop (register dequant + mma) more than by bandwidth at this size.

### 2026-10-22: attention on the tensor cores

`vattn_mma` replaces the CUDA-core attention. With one KV row per position shared by all 64 query heads (and
used as both key and value), each 32-row tile is S = R Q^T for 16 heads, then O^T += R^T P^T, both with
`mma.m8n8k4` in fp16 and fp32 accumulation; the online softmax, the 160-row splits and the partial layout are
unchanged. Only the row tile is staged in shared memory (two blocks per SM). The grid puts a token's four head
groups next to each other, so they share its rows through L2. In `tests/attn_test.cu` the kernel matches the
CUDA-core one to 1e-3 (outputs up to 0.16) and runs 2.7x faster at 32 tokens per GPU (292 against 797 us per
layer). In the engine: KL 0.127; 64 sessions 50.9 ms per step; adaptive drafts at 64 sessions, code 27.2 tok/s
per session (25.2 before), prose 18.2 (17.6).

### 2026-10-22: other hardware and an official container (plan, not built)

What ties the engine to the DGX-1 V100 box today, and what each piece would need:

| Piece | Today | To run elsewhere |
|---|---|---|
| Tensor-core kernels (experts, dense, attention) | `mma.m8n8k4`, Volta's only fp16 MMA | sm_80+: `mma.m16n8k16` with `ldmatrix` / `cp.async` variants of the same kernels, same packed formats; the formats (cb3/cb2 lookups, 4-bit dense) are not Volta-specific |
| Collectives | cube kernels for the DGX-1 hybrid mesh (two quads + twins), 8 GPUs | a topology table from `nvidia-smi topo`: NVSwitch boxes (A100/H100 HGX) do the all-gather and all-to-all in one hop; PCIe boxes (8x RTX 3090/4090) need a host-staged or NCCL path |
| Memory plan | 27.5 GB of experts per 32 GB GPU, experts split 8 ways | the tier plan is already a function of GPUs x expert bytes per GPU (`plans/`); a pack per (GPU count, memory) pair, e.g. 4 x 80 GB, 8 x 40 GB, 8 x 24 GB (cb2-heavy) |
| Engram | 196 GB of tables in host RAM | unchanged; boxes with under ~400 GB of RAM need the tables quantized (4-bit ~100 GB, quality to be measured) |

Machines that mimic this one closely enough to keep the same optimizations: 8x V100 32 GB (SXM2 or SXM3), 8x
A100 40 GB (NVSwitch; same memory pressure, more bandwidth), 4x A100 / H100 80 GB (fewer, larger GPUs: experts
split 4 ways). 8x V100 16 GB does not fit: even all-cb2 experts are ~153 GB.

Container: an `nvidia/cuda:12.9.x-devel` image building `volta` for the target architectures, whose entrypoint
reads the GPU count, memory and topology, picks the matching pack from the Hugging Face repo (`base/` plus
`pack-<GB>gb-<N>x<gpu>/`), downloads what is missing, and starts. It becomes worth building with the API server
(`core/src/server` on top of the engine), which the engine does not have yet.

### 2026-10-22: the server

`volta-server` puts dsx's HTTP layer (OpenAI and Anthropic routes, chat template, thinking levels, tools, web UI)
in front of a batching scheduler (`engine/src/server/volta.rs`) instead of dsx's one-request-at-a-time engine.
Every request takes a session slot on the least loaded GPU. Each iteration: the draft step (observe the main
tokens the last step kept, draft for the decoding sessions), then one main step that mixes their verification tokens
with prompt chunks of the sessions still prefilling, then emit, stop and rewind. Sampled requests verify drafts by
speculative sampling: with a one-hot draft, d is kept with probability p(d) under the request's temperature and
top-p, and a miss samples from p without d, which is exactly the target distribution. A finished session's slot
keeps its tokens; a new prompt sharing at least 64 tokens with one is rewound to that point instead of prefilled
(agents resend the whole conversation). Step shapes vary more than in the benches, so at most 4 are kept as CUDA
graphs, and a graph that fails to instantiate (out of memory) turns capturing off instead of failing the step.

Measured through the API, temperature 1, 300-token answers: 1 stream 38 tok/s; 8 streams 29 tok/s each (169
in total); 32 streams 20 each (500 in total; 32 = 4 slots x 8 GPUs at an 8,192-token context). A repeated
1,655-token prompt reused 1,654 cached tokens and returned its first token in 39 ms.

### 2026-10-22: long context

Only 4 layers store compressed KV (the other compressed layers read theirs), so a token costs 3.2 KB per session
in bf16. The reference rounds those rows to FP4 with an E4M3 scale per 16, and the indexer keys to FP4 with an E8M0
scale per 32, so they are now stored packed and exactly (`vstore_fp4_rows`, KL unchanged: 0.1287 against 0.1273):
~0.9 KB per token, 0.12 GB for 128K and 0.23 GB for 256K per session. Attention per token does not grow with the
context (window + top-512 compressed rows), but the indexer scans every compressed key: at 40K context it took 99
of a 294 ms prefill step (64 tokens per GPU). On the tensor cores (`vindexer_mma`) it is 5.7x faster.

8 sessions, one per GPU, code, drafts on: 64K-token prompts prefill at 2,490 tok/s in total and decode at 71 tok/s
per session; 128K-token prompts (before the indexer change) at 1,430 and 46. A single long prompt prefills on its
own GPU's attention, 64 tokens per step, so one 128K prompt alone takes minutes; larger prefill steps need
memory the step buffers do not have yet. Next on that path: prefill steps that keep only the last row's logits,
and the 32-token expert and dense kernels, which now take most of a prefill step.

### 2026-10-22: one session at a time

A single session's step was 29.4 ms: almost all fixed cost, since one token reads only ~20 MB of weights per GPU
per layer (~25 us). Each change below was predicted from a per-kernel floor and checked end to end, by the
steady-state median step (`volta spec` now reports it; the mean counted each context size's first steps, which run
without a CUDA graph and capture one).

Fixed costs measured on the box (bench/v100-fixed-costs.cu, bench/v100-stream.cu; exchanges in bench/v100-ll-exchange.cu):

| | |
|---|---|
| graph node, empty 1-block kernel | 0.9 us |
| empty kernel of 640 / 5,000 blocks | 1.8 / 7.9 us |
| dependent load, L2 / DRAM | 145 / 285 ns |
| NVLink flag, one way (quad-mates and twin / the far quad, routed) | 1.0 / 1.26 us |
| streaming read of 7.2 MB, 90-1,440 blocks | 10.4 us (690 GB/s) |

What changed, one session:

| Change | Why it was slow | Step |
|---|---|---|
| attention rows per block 32-160 by token count; expert lists bucketed once per layer | 16 attention blocks; ~5,700 mostly idle expert blocks | 29.4 -> 23.0 ms |
| MoE rows bounded by the real token counts | fp16 conversion and zeroing of all 512 exchange rows | 23.0 -> 21.2 ms |
| head-parallel mode | dense attention weights read whole by one GPU while 7 wait | 20.9 -> 18.1 ms |
| ~13 fewer launches per layer (fused hc, router top-k, routing chain, ropes) | | 18.1 -> 17.2 ms |
| hc step over 20 blocks; loads before stores | one block per token is issue-bound on one SM (probes: 3 us loads, 5 us math); stores between loads serialized them (SASS) | 16.6 -> 16.0 ms |
| LL-word exchanges for small head-parallel steps | flags and fences: 13.9 us for 20 KB against 7.2 in LL words (lab) | 15.9 -> 15.0 ms |
| router weights in bf16, top-k scores across the block | the router read runs at the streaming rate; 12 transcendental scores per lane in sequence | 15.0 -> 14.4 ms |

Tried and dropped (measured no better): a deeper weight prefetch ring in the expert kernel and smaller K chunks (the
1-token expert pass is bound by 4-byte loads at low parallelism: a plain stream of the same bytes takes 14.4 us at
90 blocks with 4-byte loads, 10.5 with 16-byte ones); dense matmuls with more K chunks; folding vprep into the
dense matmuls (every row tile converted x again in its prologue, which sits on each block's critical path: +0.3 ms
head-parallel, +0.9 ms DP).

Where a head-parallel layer goes now (~360 us, GPU0 trace), against its floor:

| Part | us | Floor | Note |
|---|---|---|---|
| graph-node gaps (~48 kernels) | ~45 | - | fewer kernels or a persistent kernel |
| routed experts (gate/up, down) | 41 | ~16 | 16-byte loads (staged through shared memory) and a shorter per-block prologue |
| dense matmuls + their vprep | 52 | ~26 | same load pattern; vprep cannot move into the tiles (above) |
| hc mixes (dots + Sinkhorn) x 2 | 35 | ~12 | the Sinkhorn is one warp's 20-step chain |
| attention + merge | 28 | ~6 | one 32-row sub-tile per block is a ~12 us chain |
| exchanges (LL) | ~13 | ~5 | |

### 2026-10-22: megakernel plan

The one-session step is ~14.4 ms of which weight reads are ~1.2 ms: the rest is ~48 kernel boundaries per layer,
each kernel's launch ramp and tail, and serial chains that leave most SMs idle. Prior work: Hazy Research's Llama-1B
megakernel (on-SM instruction interpreter, counters for dependencies, the next op's weights loaded while the current
one runs; 1.5-2.5x over vLLM/SGLang at batch 1) and Mirage MPK (1.0-1.7x). Volta has no cp.async, TMA or
programmatic dependent launch, so loads go through registers and dependencies through counters in global memory.

Primitives measured on the box (bench/v100-persistent.cu):

| | |
|---|---|
| grid barrier, 80 resident blocks (1 per SM) | 0.66 us |
| grid barrier, 160 blocks | 1.0 us |
| counter handoff between two blocks, one way | 0.42 us |
| one kernel boundary in a CUDA graph (empty 80-block kernels) | 1.16 us |

Design:

- **One persistent kernel per GPU per step segment**, 80 blocks of 256 threads (1 per SM), executing a static op
  table built on the host per step shape (like the CUDA graphs now; the kernel itself is one graph node). An op is
  an existing kernel body as a `__device__` function over virtual blocks: physical block b runs virtual blocks
  b, b + 80, ...; bodies read `vb.x / vb.y / vb.z` and `vg` instead of blockIdx / gridDim.
- **Dependencies**: first a grid barrier between dependent ops (0.66 us instead of 1.16 plus ramp and tail);
  independent neighbours share a phase. Then counters per op (or per output tile) so a consumer's blocks start as
  soon as the tiles they read are done.
- **Idle SMs prefetch**: while an op runs on few blocks (Sinkhorn, top-k, routing, the norm), the other blocks touch
  the next dense ops' weights into L2 (6 MB): those matmuls then read at L2 latency (145 ns, not 285) and bandwidth.
- **Bodies at 256 threads**: kernels written for 1024 (rmsnorm, kv store, routing) get 256-thread versions; bodies
  must not return early before a `__syncthreads` (they become functions inside a loop); dynamic shared memory sized
  for the largest op (attention 43 KB).

Stages, each measured against the graph path and checked for the same outputs:

1. Executor + the attention segment of a head-parallel layer for t <= 8 (vhc_pp2 through o_b and its LL exchange,
   ~18 kernels). Prediction: ~20 us per layer, ~0.8 ms per step (barriers alone ~9 us; the rest kernel ramps).
2. The MoE segment (router, routing, shared slice, experts, LL exchange, hc).
3. Whole layers in one kernel, then several layers; L2 prefetch of the next ops' dense weights by idle blocks.
4. Counters instead of barriers; DP mode and larger t.


The executor works (`VOLTA_MEGA=1`, outputs identical) but is slower than the CUDA graph, so it stays opt-in. In
a megakernel each op waits on the one before it through counters, so the ops still run one after another, and a
1-token matmul as an op takes longer than the same matmul as its own kernel: qkv_a 18.0 us against 11.7, q_b/8
13.5 against 8.1 (tests/mm1_test.cu). The time went instead into fewer, shorter and overlapped kernels in the
graph (below).

### 2026-10-23: one session, continued

Steady-state median step, one session, greedy, head-parallel (`VOLTA_HEADS=1 volta spec --k 0`):

| Change | Why it was slow | Step |
|---|---|---|
| shared expert slice on the 1-token dense kernel | the bucketed expert path for one row | 14.4 -> 13.2 ms |
| attention and merge from Volta Scout's lab (vattn2, vmerge2) | | 13.2 -> 13.1 ms |
| routing chain: picks loaded once, lists bucketed in one pass | a 1-block chain re-reading the picks | 13.1 -> 12.9 ms |
| the producers of three matmuls' inputs write their fp16 copy (vprep folded upstream) | a vprep node per matmul | 12.9 -> 12.7 ms |
| hc mixes on the side stream | their Sinkhorn is one warp's serial chain, on the critical path | 12.7 -> 12.1 ms |
| output head split by vocabulary (cross-GPU argmax) | one GPU read the whole head | 12.1 -> 11.6 ms |
| indexer and compressor work on the side stream | a ~55 us serial chain before attention in 8 layers | 11.6 -> 11.3 ms |

The draft's head and Markov chain are split by vocabulary the same way in head-parallel mode, and its argmax runs
over many blocks per session (vmarkov_argmax2: 54 -> 5 us per Markov step at one session, lab).

The side stream carries work that only a later step reads: the attention's and the FFN's hc mixes, the indexer's
keys, weights and (in layers that are not kv sources) queries, the compressor chain and its cache store. The main
stream waits for it only where its output is read (`Gpu::set_side` sends a branch's launches there; its matmuls
prep into their own fp16 buffer). In DP at 64 sessions this is neutral (40.6 -> 40.5 ms).

Where a head-parallel layer goes now, against what is reachable (labs in tests/ and Volta Scout's reports):

| Part | us per layer | Reachable | Note |
|---|---|---|---|
| routed experts, gate/up + down | ~45 | ~29 | vlut2 (4-warp blocks, decode reordered) is 18.6 us for gate/up against 31.2 in the lab, bit-identical; it lost in the engine while the shared slice ran beside it on the side stream |
| dense matmuls | ~42 | ~37 | within ~1.5 us of bytes / 690 GB/s plus a ~2 us launch ramp each (mm1_test) |
| attention chain (kv store, q rope, attention, merge, FP8 round) | ~34 + 5 nodes | ~10 | five serial nodes for one token's 8 heads |
| exchanges (LL), attention + MoE | ~14 | ~9 | the last GPU to arrive spends 8.2 / 6 us; GPUs arrive within 2-3 us of each other |

One trace showed five GPUs 7-13% slower on every kernel, which no other trace and no isolated probe reproduced (all
eight: 1535 MHz effective, 302 ns dependent DRAM load). Check that per-GPU kernel times agree before reading skew
from a trace.
