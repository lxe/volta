# Eight-V100 runtime measurements

These October 24, 2026 measurements use the complete single-file GGUF on eight V100-SXM2-32GB GPUs with all 48 NVLinks active, 754 GiB of host RAM and an 800 GB disk. Direct disk reads measured 6.3 GB/s. The model was hydrated from R2 and its full-file SHA-256 matched before performance measurement.

The file contains 441,574,130,240 bytes, with SHA-256 `9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c`. The frozen engine is `cc54d97568adb3d9438a8692c3415cb72ac18957`; its server binary hash is recorded in configuration.json. The DP sampler candidate changes prefill eligibility while preserving the exact CUDA fatbin, quantization and weights.

## Quality and context

The teacher evaluation covers 8,184 positions from eight 1,024-token sequences. Raw outputs and comparison bounds are in [quality/](quality/).

| Mode | Perplexity | KL top-256 | Top-1 agreement |
|---|---:|---:|---:|
| DP | 8.9423 | 0.126695 | 84.3231% |
| HP | 8.9417 | 0.129511 | 84.0787% |

The actual context probes returned the exact deployment key placed at the beginning, with zero cache reuse. The original DP/HP probes used one active request and naturally stopped after 13 output tokens. The later four-session check used distinct keys and 17-token natural replies; it ran with diagnostic synchronization and lacked continuous memory observations because its guard failed startup, as recorded in its receipt. The four-session check establishes simultaneous retrieval for those four DP 128K requests. These probes do not establish general long-context quality or behavior in unmeasured configurations.

| Mode and capacity | Actual prompt tokens | Time to first token | Receipts |
|---|---:|---:|---|
| DP, 131072 | 126041 | 226.50s | [context/](context/) |
| HP, 262144 | 250042 | 468.57s | [hp-256k/](hp-256k/) |
| Four concurrent DP requests, 131072 | 4 × 126045 | 296.89–297.47s | [context/four-concurrent-128k/](context/four-concurrent-128k/) |

## Matched DP sampler comparison

Four fresh servers ran in A/B/B/A order with one DP slot per GPU, context 131072, chunk 64, spec-k 5, theta 0.35, temperature 0.7, top-p 1 and graph budget 4. Every matched trace completed all 2048 output tokens with zero cache reuse. Within each pair the prompts, seeds and requested work matched; sampled output strings differed. Full configurations, raw outputs and comparison receipts are in [sampler/](sampler/).

The sampler candidate improved aggregate whole-request throughput by 15.8–17.7% in the mixed four-client workload and 9.6–12.3% in the fixed four-client workload. These are the controlled gain measurements.

Separate cold 4K-input/1024-output profiles measured 165.2–169.3 aggregate whole-request tokens/s for four DP clients and 267.2–273.2 for eight. One-client DP decode rate was 80.6–84.4 tokens/s. The earlier HP 256K profile used CPU sampling and was slower for four clients, at 75.5–76.0 whole-request tokens/s. Eight submitted HP requests queued behind four active slots. HP single-client results varied substantially; see [hp-256k/](hp-256k/) for the complete scope. DP remains faster in these measured four-client short profiles.

## Opt-in HP GPU sampling

The opt-in HP path collects the eight packed vocabulary slices into a separate GPU 0 buffer and runs the existing GPU sampler there. Default HP serving retains CPU sampling. The model, quantization, head arithmetic and CUDA fatbin remain unchanged. The earlier CPU-sampling HP profile above is separate from this candidate.

The strict v3 audit passed 598 real-row collection/replay checks covering 9,032,535,040 bytes, with 735 valid dispatch records, all required prefill/speculative/mixed/fallback coverage and zero log errors. It compares gathered bits with independent CPU concatenation, then replays the same sampler with identical supplied tables and uniforms. This establishes the observed collection and addressing checks, not an independent statistical proof of the sampler. The failed v1 constructor and v2 interleaved-log attempts remain archived alongside [the passed v3 audit](hp-gpu-sampling/audit-v3-passed/).

Four fresh HP timing servers ran in A/B/B/A order with four slots, context 262144, graph budget 4, chunk 64, spec-k 5 and theta 0.35. Each fixed/mixed phase used identical request JSON across arms, completed 2,048 output tokens and recorded zero cache reuse. Audit and profiling were off.

| Arm | Fixed whole-request tokens/s | Mixed whole-request tokens/s |
|---|---:|---:|
| CPU A1 | 48.983879 | 41.999733 |
| GPU B1 | 75.397173 | 62.132762 |
| GPU B2 | 75.100050 | 62.291473 |
| CPU A2 | 49.478208 | 42.139802 |

The swapped fixed gains were 53.92% and 51.78%; mixed gains were 47.94% and 47.82%. Candidate timing guards observed at least 1,707 MiB free per GPU. [Matched arms and attribution](hp-gpu-sampling/matched-abba/) retain all raw work and explain the earlier prediction's incomplete transfer/merge budget. CPU/GPU RNG consumption and reductions differ, so equal seeds do not imply identical output trajectories. These HP gains do not establish that HP exceeds DP. The DP concurrency profiles above used 1,024 outputs per client, while this HP comparison used 512; their cross-mode rates are not a matched-work comparison.

Three v3 teacher evaluations, including the final disabled/enabled flag pair, passed the original bounds over 8,184 positions: perplexity 8.9402–8.9605, KL top-256 0.129370–0.131653 and top-1 agreement 83.8710–84.2131%. The unchanged bounds were perplexity ≤ 9.0423, KL ≤ 0.131695 and top-1 agreement ≥ 83.8231%. [Final quality receipts](hp-gpu-sampling/quality-final/) bind the tested source and flags. The teacher CLI evaluates model quality separately and does not exercise the server's StepSampleHp path; the real-row server audit supplies that coverage.

Four concurrent HP requests each exercised 250,046 prompt tokens with zero cache reuse and returned their distinct keys exactly. Four active slots and no queue were observed; prefills were serialized in one 64-row group. Wall time was 1,872.11s, with a sampled minimum of 1,753 MiB free per GPU. This establishes that retrieval workload, not general long-context quality. See [four-request HP receipts](context/four-concurrent-250k-hp/).

The subsequent cold-request profile measured 70.0–72.2 aggregate whole-request tokens/s for one client, 112.3–113.4 for four and 119.3–120.1 for eight submissions. Each request generated 1,024 outputs; eight submissions used four active slots and four queued requests. Single-client decode was 108.6–109.1 tokens/s. The server retained prior long-context history, so this is a descriptive profile. See [all six cases](hp-gpu-sampling/concurrency-profile/).

The fresh four-Pi HP gauntlet completed 49 requests and 20,653 outputs, with 175.26 aggregate tokens/s during the sampled multi-active interval. The original torus discarded lighting; a separately recorded Pi correction added per-dot depth/intensity and visible dithering (five requests, 2,317 outputs). All four programs ran with distinct live frames, but cube clipping and approximate pyramid depth remain documented. Original and repair work are distinct; variable work prevents a controlled speedup claim. See [original evidence](hp-gpu-sampling/gauntlet-original/).

## Actual coding-agent gauntlet

Four concurrent Pi agents completed cube, pyramid, torus and sphere programs. All programs used Python's standard library, parsed successfully, ran in their assigned local kitty/tmux panes and produced three distinct captured frames. The original baseline used 47 requests and 19805 output tokens; the sampler repeat used 117 requests and 44901 output tokens. Their variable task work prevents a direct speed comparison.

The baseline's mean aggregate rate while at least two sessions were active was 167.326 tokens/s. Its original 246.99s controller interval includes 41.97s of recorder repair before task submission and about 60s of idle closeout. The same prepared prompt buffers were submitted once after repair. Original and clarified timing intervals are retained in baseline.json. The closed cast hash is in cast-receipt.json. The sampler programs and their separate animation/recording checks are in [optimized-gauntlet/](optimized-gauntlet/).

## Memory and publication

DP128K initially had 491MiB free per GPU; short controlled traces reached a 273MiB low watermark. A proposed graph-budget 16 trial was therefore rejected by its 1GiB admission rule. HP4/256K initially had 1879MiB free, and its process-specific 1Hz guard observed at least 1743MiB throughout the profile and long retrieval, above the 192MiB reserve.

The [Hugging Face model card](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-Volta) includes the measured sampler and context results. runtime-publication.json records readback verification of the two metadata files and confirms the GGUF bytes and hash remained unchanged.
