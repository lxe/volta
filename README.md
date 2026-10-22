# volta

Batched inference for DeepSeek-V4.1-Flash on one 8x V100-SXM2-32GB machine (the DGX-1 NVLink mesh). Volta has
no FP8, FP4 or bf16 math, and the official checkpoint does not fit in its 256 GB of HBM, so no existing serving
engine runs this model there. The 8 cards together still read 6.9 TB/s of HBM. This engine turns that bandwidth
into throughput for many concurrent agent sessions with its own low-bit formats and kernels, written for sm_70.

## Results

DeepSeek-V4.1-Flash-1017 from the official checkpoint, re-packed at load to 220 GB of experts (27.5 GB per GPU):

| | |
|---|---|
| Quality, Lucebox eval (8 x 1024 tokens against the FP8 teacher's top-256) | KL 0.130, top-1 agreement 84.6%, ppl 8.97 |
| 64 concurrent sessions, real text | 58.7 ms per step: **1,091 tok/s** aggregate, 17.0 tok/s per session |
| llama.cpp (PR #30817, UD-IQ1_M) on the same machine, for comparison | 26.7 tok/s for one session; 72-97 tok/s aggregate at most |

The target is ~2,000 tok/s aggregate at ~30 tok/s per session. [DESIGN.md](DESIGN.md) has the design, the
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

## Layout

| Path | What |
|---|---|
| `engine/src/volta.rs` | the engine: placement, expert packing, step planning, attention, MoE, collectives, CUDA graphs |
| `engine/src/kernels/volta.cuh` | the sm_70 kernels: `vlut_*` and `vmoe_*` matmuls, routing, attention, collectives |
| `engine/src/bin/volta.rs` | the CLI: `check`, `eval`, `bench`, `experts` |
| `core/`, the rest of `engine/` | model loader, CPU reference and shared CUDA kernels, from [dsx](https://github.com/lxe/dsx) |
| `tests/` | standalone kernel tests and sweeps (`vmoe_test`, `vlut_test`, `dense_test`, `split_test`) |
| `bench/` | hardware probes and the existing-engine benchmarks |

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

`--model` is the checkpoint directory (`config.json`, the safetensors shards, Engram included).

```bash
VOLTA_EXPERT_GB=27.5 ./target/release/volta bench --model DeepSeek-V4.1-Flash-1017 --sessions 8 --steps 32 --text heldout.i32
```

`--sessions` is per GPU. `--text` teacher-forces real tokens (i32 LE) so routing looks like real traffic.
`volta eval --model ... --eval <dir>` runs the Lucebox eval, and `volta check` compares logits with the CPU
reference.

| Variable | Default | Effect |
|---|---|---|
| `VOLTA_EXPERT_GB` | 25 | expert bytes per GPU; sets how many experts drop from MXFP4 to cb3 and cb2 |
| `VOLTA_RANK` | `/root/dsx/stage/rank.json` | routing counts per (layer, expert) as `[layer, expert, picks]`, which choose the tiers |
| `VOLTA_EXPERT_TP` | 8 | GPUs that split each expert (1 places whole experts) |
| `VOLTA_CACHE` | on | `0` skips the packed-expert cache in `/dev/shm/vcache` |
| `VOLTA_TRACE_STEP` | | wraps that step in a CUDA profiler range, for `nsys --capture-range=cudaProfilerApi` |
| `VOLTA_PROF` | | prints where each step's time went, with a sync at each boundary |

## License

MIT
