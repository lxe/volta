# Eight-V100 runtime measurements

These October 24, 2026 measurements use the complete single-file GGUF on eight V100-SXM2-32GB GPUs with all 48 NVLinks active, 754GiB of host RAM and an 800GB disk. Direct disk reads measured 6.3GB/s. The model was hydrated from R2 and its full-file SHA-256 matched before performance measurement.

The file contains 441574130240 bytes, with SHA-256 `9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c`. The frozen engine is `cc54d97568adb3d9438a8692c3415cb72ac18957`; its server binary hash is recorded in configuration.json. The sampler candidate changes prefill eligibility while preserving the exact CUDA fatbin, quantization and weights.

## Quality and context

The teacher evaluation covers 8184 positions from eight 1024-token sequences. Raw outputs and comparison bounds are in [quality/](quality/).

| Mode | Perplexity | KL top-256 | Top-1 agreement |
|---|---:|---:|---:|
| DP | 8.9423 | 0.126695 | 84.3231% |
| HP | 8.9417 | 0.129511 | 84.0787% |

The actual context probes returned the exact deployment key placed at the beginning, with zero cache reuse. The original DP/HP probes used one active request and naturally stopped after 13 output tokens. The later four-session check used distinct keys and 17-token natural replies; it ran with diagnostic synchronization and lacked continuous memory observations because its guard failed startup, as recorded in its receipt. They establish these controlled retrieval workloads, not general long-context quality or simultaneous long requests.

| Mode and capacity | Actual prompt tokens | Time to first token | Receipts |
|---|---:|---:|---|
| DP, 131072 | 126041 | 226.50s | [context/](context/) |
| HP, 262144 | 250042 | 468.57s | [hp-256k/](hp-256k/) |
| Four concurrent DP requests, 131072 | 4 × 126045 | 296.89–297.47s | [context/four-concurrent-128k/](context/four-concurrent-128k/) |

## Matched sampler comparison

Four fresh servers ran in A/B/B/A order with one DP slot per GPU, context 131072, chunk 64, spec-k 5, theta 0.35, temperature 0.7, top-p 1 and graph budget 4. Every matched trace completed all 2048 output tokens with zero cache reuse. Within each pair the prompts, seeds and requested work matched; sampled output strings differed. Full configurations, raw outputs and comparison receipts are in [sampler/](sampler/).

The sampler candidate improved aggregate whole-request throughput by 15.8–17.7% in the mixed four-client workload and 9.6–12.3% in the fixed four-client workload. These are the controlled gain measurements.

Separate cold 4K-input/1024-output profiles measured 165.2–169.3 aggregate whole-request tokens/s for four DP clients and 267.2–273.2 for eight. One-client DP decode rate was 80.6–84.4 tokens/s. The HP256K profile was slower for four clients, at 75.5–76.0 whole-request tokens/s. Eight submitted HP requests queued behind four active slots. HP single-client results varied substantially; see [hp-256k/](hp-256k/) for the complete scope.

## Actual coding-agent gauntlet

Four concurrent Pi agents completed cube, pyramid, torus and sphere programs. All programs used Python's standard library, parsed successfully, ran in their assigned local kitty/tmux panes and produced three distinct captured frames. The original baseline used 47 requests and 19805 output tokens; the sampler repeat used 117 requests and 44901 output tokens. Their variable task work prevents a direct speed comparison.

The baseline's mean aggregate rate while at least two sessions were active was 167.326 tokens/s. Its original 246.99s controller interval includes 41.97s of recorder repair before task submission and about 60s of idle closeout. The same prepared prompt buffers were submitted once after repair. Original and clarified timing intervals are retained in baseline.json. The closed cast hash is in cast-receipt.json. The sampler programs and their separate animation/recording checks are in [optimized-gauntlet/](optimized-gauntlet/).

## Memory and publication

DP128K initially had 491MiB free per GPU; short controlled traces reached a 273MiB low watermark. A proposed graph-budget 16 trial was therefore rejected by its 1GiB admission rule. HP4/256K initially had 1879MiB free, and its process-specific 1Hz guard observed at least 1743MiB throughout the profile and long retrieval, above the 192MiB reserve.

The [Hugging Face model card](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-Volta) includes the measured sampler and context results. runtime-publication.json records readback verification of the two metadata files and confirms the GGUF bytes and hash remained unchanged.
