# 1-token routed-expert kernel: vlut2 (lab: /root/scout/expert_lab.cu)

V100-SXM2-32GB, GPU 7. Graph-timed per launch, weights rotated over 2 copies so every launch reads DRAM. One token,
6 experts on one GPU's 8-way slice, cb3. Reference = vlut_cb3r1_n1 / _n2 from a snapshot of volta.cuh (/root/scout/ref).

## Result

Use `vlut2_cb3r1_n1_p2_a1_x1_w4` for gate/up. Keep `vlut_cb3r1_n1` for down: vlut2 is slower there.

| | reference | vlut2 (W4) |
|---|---|---|
| gate/up, 1 token, grid y 43 | 31.2 us | **18.6 us** |
| gate/up, 1 token, grid y 6 | 25.9 us | 19.2 us |
| gate/up, 8 tokens, y 43 | 31.9 us | 21.0 us |
| gate/up, 12 / 16 tokens (n2), y 43 | 40.1 / 40.5 us | 28.8 / 29.8 us |
| down, 1 token, y 32 | 10.4 us | 12.3 us |

Targets (gate/up 13 us, down 6 us) are not met. Output is bit-identical to the reference for 1, 3, 8, 9 and 16
tokens: 132 comparisons, each of the 5 K chunks checked on its own, plus down (`./expert_lab` prints them).

## What changed, and why it works

1. **4 warps per block instead of 8.** The engine's gate/up grid is 90 blocks of 8 warps on 80 SMs, so 10 SMs run 2
   blocks and set the span: ~26 us against ~16 us for a single-block SM (per-block globaltimer probes). The 18 row
   groups of a matrix also leave 6 of the 24 warps in the third block idle. With 4-warp blocks, 150 blocks spread at
   most 2 per SM, all 4 warps busy.
2. **Tasks in block order.** Each block walks tasks (x, entry, K chunk) over the real `*nwork` entries. With y = 43,
   the real tasks are the first blocks dispatched, and the empty blocks exit at once. Before this, y 43 cost ~6 us
   more than y 6; now the two are equal.
3. **A unit's 16 lookups and 8 x fragments come first, then its 8 mma.** The compiler had serialised lookup -> scale ->
   mma for each pair, reusing the same registers, so every mma waited on a shared-memory load. Decode-only time for 32
   units went from 17 to 11.7 us.
4. **16-byte weight loads, 2 stages of 4 units ahead, staged through shared memory**, and cnt/off loaded together.
   This alone measured no faster. It reduces load instructions 4x and keeps the layout.

The arithmetic is vlut_mm's: the same lookups, scales and mma in the same order per warp, and the same atomics. The
x tile now holds only the real token rows; the B fragments of the other rows are zeros from registers instead of
from shared memory.

Integration: 128 threads per block; grid.x = ceil(N / 32 / 4) * tiles, with `rtiles` = ceil(N / 32 / 4); dynamic
shared memory = `vlut2_smem<FMT>(NT, KC, 4)` (vlut_mm's + 1.66 KB per warp of staging). The kernel signature is
unchanged.

## Measurements behind it (hypotheses tested)

- **Scoreboard / 4-byte loads:** not the limiter. 16-byte loads plus 2-4 stages in flight changed nothing by
  themselves, and 3-6 stages were slower. Registers are capped at 128, so the deeper rings spill.
- **mma.m8n8k4 f32 (mma_lat.cu):** a dependent chain costs 15 cycles; throughput is 1 mma per ~3.1 cycles per SM.
  The whole gate/up pass is ~3.5 us of tensor work, so tensor cores are not the limit.
- **The binding cost is per-warp decode.** With no weight loads at all, the decode takes ~0.33 us per unit per warp
  at the ~7.5 warps/SM that 1 token gives: 32 units = ~10.5 us. Bank-private tables (conflict-free) gave nothing.
  Replacing the table loads with register math saved only ~12%. The loop is ~120 SASS instructions per unit (32
  HMMA, 16 HMUL2, 16 table LDS + 32 index ops, 8 x LDS), and per-warp IPC is ~0.26.
- **Prologue:** the dependent chain nwork -> work -> cnt/off -> in_rows -> x takes ~2.2 us alone and ~5 us under
  the weight stream. Issuing the weights after x lands (x2/x3 variants) only moves the wait into the units phase:
  the totals are equal.
- **Finer K chunks** (KC 512 / 256) are slower here, because each extra block repeats the prologue.

## Why 13 us is out of reach at bit identity

At KC 1024, each output's partial sum is one chain of 256 mma in one warp (32 units). At the measured ~0.33 us per
unit that is ~10.5 us, plus ~2.2 us before x is in shared memory and ~1-2 us of launch/tail, so ~14-15 us per
launch. Going lower means fewer decode instructions per warp, or more warps per output. More warps per output
re-associates the K sum: the `A` accumulator variants (A2/A4) do exactly that, with max |diff| ~2e-6 relative, but
they measured no faster at 1 token. So I could not find a bit-identical path to 13 us. Down's floor is similar:
~2.5-3 us of x chain, plus 9 units of decode, plus streaming 3.6 MB.

Lab: `./expert_lab` (checks + timing), `./expert_lab probe` (per-block phases), `mma_lat.cu`. Build:
`nvcc -O3 -arch=sm_70 -std=c++17 -I/root/scout/ref expert_lab.cu -o expert_lab`. Run with
`CUDA_MODULE_LOADING=LAZY CUDA_VISIBLE_DEVICES=7 flock /root/gpu.lock ./expert_lab`.
