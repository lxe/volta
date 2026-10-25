# DP and HP teacher quality

The complete Volta GGUF passed the same8184-position teacher evaluation in data-parallel (DP) and head-parallel (HP) modes on eight V100-SXM2-32GB GPUs. HP's KL increased0.002816 and its top1 agreement fell0.2444percentage points relative to DP; both remain within the recorded regression gate. This is a short-context numerical comparison, not a serving throughput or256K quality result.

| Measure | DP | HP | HP minus DP |
|---|---:|---:|---:|
| Evaluated positions |8184|8184|0|
| Perplexity |8.9423|8.9417|-0.0006|
| KL on teacher top256 |0.126695|0.129511|+0.002816|
| Top1 agreement |84.3231%|84.0787%|-0.2444percentage points|
| Evaluation wall time |47.8s|48.0s|+0.2s|

Both runs use eight1024-token sequences, chunk1, the frozen cc54d975 engine's volta eval command, and the same complete441574130240-byte model. Evaluated positions exclude each sequence's first token:8*(1024-1)=8184. HP starts with VOLTA_HEADS=1 and VOLTA_GRAPHS=4; its archived configuration explicitly records heads=true, and its archived runner records the environment setup. The HP run began October24,2026 at19:36:11Pacific. Its load log reports133seconds to load and3.9GB free per GPU before evaluation; that model-loading interval is separate from the48.0s evaluation receipt.

The tight HP gate compares against DP: perplexity at most DP+0.1, KL_top256 at most DP+0.005, and top1 agreement at least DP-0.005 (a0.5percentage-point allowance). The archived metrics receipt records that gate passing, and comparison.json recomputes the same inequalities from both raw stdout receipts. The HP result is close to DP but is not bit-identical. Partitioned HP execution is evaluated here as a distinct numerical path.

The model SHA256 is9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c. The evaluated volta CLI binary SHA256 ise216e64da3524d36b657f0cdc0cf3bf4a4099f421c5282970b908ead5817351c. Its hash is distinct from the HTTP server binaries. The sampler guard affects serving eligibility; these teacher evaluations do not test speculative DSpark acceptance, API sampling, tool behavior, agent throughput or effective long context.

Raw stdout and load/progress logs are under dp/ and hp/. HP configuration/metrics and exact historical runner source are retained; the runner copies have .txt suffixes and are evidence, not automatic launch instructions. comparison.json gives the common command, model identity, metrics and computed differences. sha256.json binds the evidence files. No new evaluation was run to assemble this archive, and no performance claim is derived from the0.2s wall difference.
