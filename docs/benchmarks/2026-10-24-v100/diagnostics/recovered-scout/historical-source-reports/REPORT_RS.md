# Router || shared-expert slice (lab: rs_lab.cu)

## Budget and prediction (written before the measurement)

At head-parallel t = 1, per layer:

| piece | bytes | floor at 690 GB/s | launches | measured before |
|---|---|---|---|---|
| router (vrouter_topk, bf16 gate 384 x 5120) | 3.9 MB | 5.7 us | 1 (240 blocks), + ~5 us serial tail in one block | 13.9-15.6 us (rt_lab) |
| shared slice: zero, gate/up 576 x 5120, swiglu, down 5120 x 288 (4-bit + fp16 scales) | 1.84 + 1.84 MB | 5.3 us | 4 | not yet |

- **Serial:** router + shared, about 14 + (2 x ~8 + 2 x ~1) ≈ 32 us.
- **Forked:** bounded below by the larger branch (~18 us) and by DRAM for both together (7.6 MB, ~11 us). The
  router's ~5 us one-block tail leaves the SMs idle, so the shared slice can use them.
- **Prediction:** forked saves ~10-14 us per layer. That is the whole router, not just its tail: the router body and
  the shared slice are both far from bandwidth-bound at t = 1, so they can share DRAM and SMs.
- **Confirming measurement:** per-layer graph time, serial vs forked, 100 layers per graph, weights rotated over 8 copies.
- **Kill criterion:** forked saves < 3 us per layer (less than the tail alone), or the branches slow each other
  enough that forked >= serial - 3 us.

## Measured (2026-10-23 03:40 PT, GPU 7, two runs agree within 0.1 us)

| per layer | us |
|---|---|
| router alone | 15.3 |
| shared slice alone (zero, gate/up, swiglu, down) | 18.4 |
| router then shared (serial, today's main path) | 35.7 |
| shared then router (serial) | 35.6 |
| forked: router || shared, joined with events | **31.3** |

Forked saves **4.3 us per layer**, below the 10-14 us predicted (it passes the 3 us kill line). The prediction
assumed the two branches would co-run on the SMs; they mostly don't. Forked is 31.3 us against an ideal of
max(18.4, 15.3) = 18.4 us. Suspect: vrouter_topk declares 32 KB of static shared memory per block (sized for 32
tokens), so its 240 blocks fill the SMs' shared memory and the shared slice's blocks wait.

## Follow-up: the router's shared memory sized by T (vrouter_topk_d)

Budget: at t = 1 the router needs 1 KB for its x tile and 3 KB for the tail's scores and keys, not 32 KB, so up to
8 router blocks fit per SM instead of 3. Prediction: forked drops to ~20 us. Kill: forked > 28 us.

vrouter_topk_d is vrouter_topk with dynamic shared memory of max(T * 256, 2 * T * E) floats (3 KB at t = 1) and the
keys at xs + T * E. It has the same experts, weights and logits as vrouter_topk over 8 weight copies.

Measured (2026-10-23 ~04:00 PT, two runs within 0.1 us; this run's absolute numbers sit ~2 us below the first run's):

| per layer | vrouter_topk (32 KB static) | vrouter_topk_d (3 KB) |
|---|---|---|
| router alone | 13.6 us | **12.65** |
| shared slice alone | 16.2 | 16.2 |
| serial (router then shared) | 31.8 | 30.7 |
| forked, joined | 28.05 | **25.8** |

- **Together** (smaller router plus fork): 31.8 -> 25.8 = **6.0 us per layer**, ~0.26 ms per step over 43 layers.
- **The fork alone** with today's router: 3.7 us.
- **The smaller router alone**, serial: 1.1 us.

The prediction (~20 us forked) was too optimistic. The two branches still don't fully overlap: forked is 25.8 us
against max(12.65, 16.2) = 16.2 us. Both branches are latency chains of 1 and 4 launches, and the measurement doesn't
show where the remaining serialization comes from (graph branch scheduling, or SM residency of the shared slice's
own blocks).

## Engine result (Volta Main, 2026-10-23)

The fork made the engine **0.15 ms per step slower** (11.33 -> 11.48 ms, HP one session). vrouter_topk_d alone is kept;
DP gained 0.1 ms from it.

**Why the lab misled:** its baseline ran router then shared slice in series, but in the engine the shared slice
already overlapped the expert pass on the side stream. The fork only moved that overlap, so the lab's "saving"
didn't exist in the engine. Lesson (LESSONS 6): before measuring a reordering, model the engine's actual stream
schedule, not the main path alone.
