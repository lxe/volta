# Head-parallel GPU sampling matched timing

The opt-in GPU sampler passed the matched HP timing gate on the eight-V100 rental. In A/B then B/A comparisons, fixed-work throughput improved by 53.92% and 51.78%; mixed-work throughput improved by 47.94% and 47.82%. Each phase completed four requests with exactly 2,048 output tokens and zero cache reuse. These are whole-request throughput measurements including prefill, not isolated decode rates or a claim that HP is faster than the primary DP configuration.

| Arm, in execution order | Fixed tokens/s | Mixed tokens/s | Guard minimum free memory |
|---|---:|---:|---:|
| CPU A1 | 48.983879 | 41.999733 | 1,745 MiB |
| GPU B1 | 75.397173 | 62.132762 | 1,707 MiB |
| GPU B2 | 75.100050 | 62.291473 | 1,711 MiB |
| CPU A2 | 49.478208 | 42.139802 | 1,741 MiB |

All four fixed request files have identical bytes and SHA256 `b1abc333c3b07a09c8f79ad9bec52f831fbb792ac7a53213cff6206a9c527b44`. All four mixed request files have identical bytes and SHA256 `629ed38e37ac8a83975dbee79df0dcb258b78e92fe368c206d12f2a07f7c4961`. The actual prompt totals are 16,588 fixed and 20,685 mixed in every arm. Output counts were recomputed from individual API usage and matched to server counter deltas; throughput was recomputed from earliest client start to latest completion. Fixed starts all four clients together; mixed admits the longer fourth prefiller after the three decoders emit their first content.

The common settings were HP four slots, context 262144, graph budget 4, chunk 64, spec_k 5, theta 0.35, temperature 0.7 and top_p 1. Audit, step profiling, loop profiling and graph tracing were off. CPU binary SHA256 was `b9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4`; GPU binary SHA256 was `69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363`, built from source `2e0e16166709d11f4e98802eb74343cdba77a605`. Both use the same frozen CUDA fatbin SHA256 `4387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5`. The normalized command and environment match apart from the intended HP GPU-sampling enablement and executable identity.

The strict analyzer checked guard admission and exact server PIDs, all eight GPU readings, no stop flag, at least 192 MiB free, and monotonically sampled observations with maximum gap below two seconds. Guard receipts bracket each complete fixed/mixed phase using recorded server start epoch plus telemetry uptime. This is sampled coverage, not a claim about unsampled microsecond intervals. A1/B1/B2/A2 have 99/69/68/99 guard observations respectively, with actual maximum gaps about 1.14 seconds.

## Attribution and limits

The candidate replaces full-vocabulary downloads, host concatenation and CPU distribution work with GPU vocabulary collection and the existing GPU sampler, followed by small result downloads. It preserves the model, quantization, head arithmetic and CUDA module. The successful real-row v3 audit is a separate prerequisite; timing here contains no audit overhead.

The earlier prediction called 27–28% a zero-new-cost ceiling based on the observed 21% CPU-distribution fraction. That was a CPU-distribution-only Amdahl calculation, not a valid ceiling for the whole candidate. The old profile explicitly excluded Engine::step's GPU-to-host downloads and host vocabulary merge from cpu_sample_ms, and the candidate also replaces those costs. The measured 48–54% net improvement is therefore not bounded by that one component's fraction. No matched per-component profile was collected for these timing arms, so the result cannot be partitioned precisely among transfer, merge, sampling, changed draft acceptance or scheduling.

CPU A2/A1 drift was +1.009% fixed and +0.334% mixed; GPU B2/B1 drift was -0.394% fixed and +0.255% mixed. The two swapped gains exceed those observed drifts substantially, but two pairs on one host do not provide a confidence interval or universal hardware/workload guarantee. A1 ran earlier than the remaining three fresh servers, with the same request bytes and frozen CPU binary. Its earlier timestamp is preserved rather than presented as an adjacent launch.

CPU and GPU sampling consume RNG values differently and use different reduction precision. Equal request seeds therefore do not imply equal generated text, acceptance histories or internal verifier work. The comparison holds external prompt/output budgets fixed; it does not claim identical model trajectories. Teacher quality, long-context capacity, the real agent gauntlet and final promotion remain separate checks. The timing marker authorizes only the parent's next gate and is not a promotion marker.
