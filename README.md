# volta

Batched inference for DeepSeek-V4.1-Flash on one 8x V100-SXM2-32GB machine (the DGX-1 NVLink mesh). Volta has
no FP8, FP4 or bf16 math, and the official checkpoint does not fit in its 256 GB of HBM, so no existing serving
engine runs this model there. The 8 cards together still read 6.9 TB/s of HBM. This engine turns that bandwidth
into throughput for many concurrent agent sessions with its own low-bit formats and kernels, written for sm_70.

## Results

DeepSeek-V4.1-Flash-1017 from the official checkpoint, with its experts re-packed to 220 GB (27.5 GB per GPU):

| | |
|---|---|
| Quality, Lucebox eval (8 x 1024 tokens against the FP8 teacher's top-256) | KL 0.126-0.130 across runs, top-1 agreement 84%, ppl 8.95 |
| 64 concurrent sessions, real text | 50.9 ms per step: 1,258 tok/s aggregate, 19.7 tok/s per session |
| 128 sessions | 69.1 ms per step: 1,853 tok/s aggregate, 14.5 tok/s per session |
| 256 sessions | 103.6 ms per step: **2,472 tok/s** aggregate, 9.7 tok/s per session |
| 512 sessions | 166.8 ms per step: 3,070 tok/s aggregate, 6.0 tok/s per session |
| With the DSpark draft, code, 64 sessions, 3 drafts per pass | 3.23 tokens per pass: **26.0 tok/s per session**, 1,667 tok/s aggregate |
| With the DSpark draft, code, 8 sessions, 5 drafts per pass | 4.28 tokens per pass: **82.6 tok/s per session** |
| With the DSpark draft, prose, 64 sessions, 1 draft per pass | 1.57 tokens per pass: 17.9 tok/s per session (17.5 without) |
| With the DSpark draft, `--adaptive` (drafts per session from its acceptance), 64 sessions | code **27.2**, prose 18.2 tok/s per session |
| llama.cpp (PR #30817, UD-IQ1_M) on the same machine, for comparison | 26.7 tok/s for one session; 72-97 tok/s aggregate at most |

 [DESIGN.md](DESIGN.md) has the design, the
hardware measurements, the time per step by kernel, and dated amendments where the plan changed.

## How it works

- **Data-parallel attention.** Each GPU owns a pool of sessions with their compressed KV cache, and runs their
  attention and dense layers itself.
- **Every routed expert split 8 ways.** Each GPU holds 1/8 of the intermediate dim of every expert (and of the
  shared expert), so expert work per GPU is the same whichever experts the router picks.
- **NVLink-only collectives.** Per layer, an all-gather sends every token's hidden state to every GPU and an
  all-to-all brings the partial sums home. Both are single kernels that store into peer memory and signal with
  flags (~20-30 µs each, against NCCL's 40-60 µs floor).
- **Lookup-table expert formats.** At load, each expert's MXFP4 weights are kept as is (4.25 bpw) or re-quantized
  on the GPU to cb3 (3.25 bpw) or cb2 (2.25 bpw), by routing frequency, until they fit the byte budget. The kernel
  decodes two weights per shared-memory lookup straight into Volta's `mma.m8n8k4` tensor-core fragments.
- **Device-side routing and one CUDA graph per step.** The router's picks become per-expert token lists on the
  GPU, so the host never waits inside a step.
- **Engram** tables stay in host RAM, and their rows are gathered for each step.
- **Several tokens per session in a step** (up to 64 per GPU): prefill chunks and draft verification use the
  same step.
- **DSpark**, the checkpoint's 3-layer draft, runs on the GPUs through the same MoE path. A pass drafts k tokens
  per session, verifies them in one main step and keeps the agreed prefix. Greedy only for now.

## Layout

| Path | What |
|---|---|
| `engine/src/volta.rs` | the engine: placement, expert packing, step planning, attention, MoE, collectives, CUDA graphs |
| `engine/src/kernels/volta.cuh` | the sm_70 kernels: `vlut_*` and `vmoe_*` matmuls, routing, attention, collectives |
| `engine/src/bin/volta.rs` | the CLI: `check`, `eval`, `bench`, `experts` |
| `core/`, the rest of `engine/` | model loader, CPU reference and shared CUDA kernels, from [dsx](https://github.com/lxe/dsx) |
| `tests/` | standalone kernel tests and sweeps (`vmoe_test`, `vlut_test`, `dense_test`, `split_test`) |
| `bench/` | hardware probes and the existing-engine benchmarks |
| `plans/` | the published pack's tier plan and placement, and the routing counts that choose tiers |

## Build

Needs 8x V100 with NVLink, CUDA 12.9 (the last toolkit that compiles for sm_70), Rust stable, and ~380 GB of
`/dev/shm` or RAM for the packed-expert cache.

```bash
CUDA_ARCHS=70 cargo build --release -p dsx --features cuda --bin volta
```

The kernel tests build with nvcc directly, from `tests/`:

```bash
nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels vlut_test.cu -o vlut_test
```

## Run

The ready-made weights are on Hugging Face at
[lxe/DeepSeek-V4.1-Flash-1017-Volta](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-Volta): `base/` is the
official checkpoint without its routed experts (dense weights and Engram, unchanged), and `pack-27.5gb-8xv100/` holds
the packed experts for each of the 8 GPUs. `/dev/shm` or a fast disk for the pack keeps the load at ~2 minutes.

```bash
VOLTA_PACK=pack-27.5gb-8xv100 ./target/release/volta bench --model base --sessions 8 --steps 32 --text heldout.i32
```

From the official checkpoint instead, the engine packs the experts on the GPUs at load (~3 minutes), using the
routing counts in `plans/rank.json`, and caches the result in `/dev/shm/vcache`:

```bash
VOLTA_EXPERT_GB=27.5 ./target/release/volta bench --model DeepSeek-V4.1-Flash-1017 --sessions 8 --steps 32
```

To rebuild exactly the published pack, start a pack directory from its plan: `mkdir pack && cp
plans/v41-1017-27.5gb-8xv100.json pack/manifest.json`, then run once with `VOLTA_PACK=pack` and the official
checkpoint as `--model`. The plan holds each expert's tier (15,172 in cb3, 188 in cb2) and its placement.

`--sessions` is per GPU. `--text` teacher-forces real tokens (i32 LE) so routing looks like real traffic.
`volta eval --model ... --eval <dir>` runs the Lucebox eval (`--chunk k` feeds k tokens per sequence per step), and
`volta check` compares logits with the CPU reference. `VOLTA_SPARK=<checkpoint dir> volta spec --model ... --text
<tokens> --k 3` prefills each session and decodes greedily with k drafts per pass (`--k 0` is the no-draft
baseline; `--k 5 --adaptive` picks each session's drafts from its acceptance).

| Variable | Default | Effect |
|---|---|---|
| `VOLTA_PACK` | | load the experts from this pack directory, or make the pack there if it has no `gpu*.bin` yet |
| `VOLTA_EXPERT_GB` | 25 | expert bytes per GPU; sets how many experts drop from MXFP4 to cb3 and cb2 |
| `VOLTA_RANK` | `plans/rank.json` | routing counts per (layer, expert) as `[layer, expert, picks]`, which choose the tiers |
| `VOLTA_EXPERT_TP` | 8 | GPUs that split each expert (1 places whole experts) |
| `VOLTA_CACHE` | on | `0` skips the packed-expert cache in `/dev/shm/vcache` |
| `VOLTA_TRACE_STEP` | | wraps that step in a CUDA profiler range, for `nsys --capture-range=cudaProfilerApi` |
| `VOLTA_PROF` | | prints where each step's time went, with a sync at each boundary |
| `VOLTA_SPARK` | | the checkpoint directory to load the DSpark draft from (its experts are packed at load) |
| `VOLTA_ATTN` | | `tile5` runs attention on CUDA cores instead of the tensor cores |
| `VOLTA_OVERLAP` | on | `0` runs everything on one stream (no side-stream branches in the step graph) |

## License

MIT
