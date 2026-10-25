# Graph admission and vocabulary-head diagnostics

Actual eight-V100 traces demonstrate repeated graph-admission misses in four-client DP serving, but both proposed four-slot replacement rules failed their CPU screening gates. A separate synchronized finish profile also rejected the narrow last-row vocabulary-head candidate: eliminating all eligible finish work would save at most1.52% of fixed and2.58% of mixed request wall time, below the5% target. No graph replacement or head selection was implemented or run.

## Runs and evidence boundary

The parent ran diagnostic v2 on the same eight V100-SXM2-32GB GPUs and complete441574130240-byte GGUF used by the baseline. Model SHA256 is9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c. Diagnostic source is frozen cc54d975 plus the measured sampler guard and optional bounded CPU tracing; its binary SHA256 is fd6c71b58683a113b2b4746526c32681dd43933af5fcd5e37eb81500e7f593e1. The build receipt proves its CUDA fatbin matches frozen exactly.

Both actual runs use DPslots1, context131072, chunk64, spec-k5, temperature0.7, top-p1 and nominal graphs4. Each fresh process receives the same smoke, two warm-ups, fixed four-client and mixed four-client protocol. Fixed/mixed generate2048tokens each with zero cached tokens. Actual prompt counts39/563/4145/8242 have final prefill chunks39/51/49/50, all above9. This verifies that t1..6 are generated verification rows in these particular controlled traces; the classification is not a generic engine rule.

Graph files are diag-v2-001, with primary fixed/mixed activity at20:05:43-20:06:27Pacific on October24,2026. Profile files are the separate SAME-process v2+VOLTA_STEPT run diag-v2-stept-001, with primary activity at20:10:07-20:10:51Pacific. The analyzer uses each run's own server UTC admission/done timestamps and its own trace wall epoch plus monotonic elapsed. It never applies the earlier run's sequence numbers to the profile.

Every GPU has768detailed observations and matching completed summaries. Raw eight-GPU JSONL, original server stderr, configuration and client summaries are retained in graph/ and profile/. Source/code/protocol copies under source/ explain the analysis and are evidence, not automatic launch instructions. The archived CPU scripts expect the original task directory layout. Synthetic test receipts are explicitly labeled and separate from actual trace reports. sha256.json binds all archived files.

## Admission problem and rejected replacement policies

The current nominal-four main cache can exceed its limit because previously warmed keys bypass the absent-key budget check: actual resident maxima are6onGPU0 and8onGPUs1-3. Early smoke/warm/prefill graphs remain cached when primary generated shapes at bucket8192 arrive. During fixed and visible mixed intervals, all four active GPUs run100% of t1..6 observations eagerly due to budget, with repeatedly missed keys and persistently resident graphs never replayed in that generated interval. Idle GPUs4-7 mostly replay t0 and provide no active-decoder policy gate.

Original-eight simulation results remain unmodified in graph/original-eight-analysis.json. That trial quota binds on each active GPU, and later qualifying observations are blocked. Its failed result was not relabeled as passing after selecting better phase boundaries. Phase-local statistics establish that stale admission is real; they do not establish a successful replacement policy.

A separately written budget then allowed a CPU-only sixteen-replacement hypothesis with the SAME four-slot policy:64-step frequency window, minimum8candidate observations, frequency advantage4, minimum64steps between replacements, and strict four-resident cache. It replays the whole smoke/warm prefix before evaluating phases. Active GPUs need10/11/11/11replacements, so16no longer binds, but GPU1mixed coverage becomes worse than under8. The revised hypothesis fails its prewritten gates.

| GPU | Fixed hit gain vs actual, pp (8/16) | Mixed gain8, pp | Mixed gain16, pp |16 minus8 mixed, pp |
|---|---:|---:|---:|---:|
|0|36.69|54.55|68.60|+14.05|
|1|29.55|24.24|11.85|-12.40|
|2|14.61|7.16|39.39|+32.23|
|3|46.43|8.82|17.91|+9.09|

GPU1mixed andGPU2fixed are below the15percentage-point coverage threshold; the additional GPU1mixed regression is decisive evidence of churn. Sixteen replacements is a trial safety quota, not an eventual production lifetime contract. A rolling production rate bound would let adaptation continue after later requests/context changes, but it would not resolve this four-slot replacement failure. Do not infer a production policy from this CPU simulation.

Actual successful active capture host durations span12.78-70.12ms; first idle/t0 captures reached92.09ms. A100ms conditional per-replacement-step ceiling charges1.9s fixed and2.0s mixed (11.44%/7.41% of client wall). A proxy based on same-GPU/same-t measured capture medians charges0.572s/0.601s (3.44%/2.23%). Neither proxy measures destruction/reclaim synchronization or proves saved serving time. The policy fails coverage gates independently of those costs. Host enqueue medians also show backpressure, not isolated GPU execution or causal gains.

The final graph snapshot omits approximately1.28s of mixed completion tail. All graph-policy mixed percentages concern the visible prefix; no exact full-mixed policy result is claimed. The same selected-interval and full-prefix inputs, boundaries and replacement events are preserved for review.

## Last-row head finish gate

The existing finish operation computes all prefill vocabulary rows although only the final row is sampled. Source review found a narrow DPslots1 opportunity using an input-pointer offset and sampler-row remap, but the measurement gate rejected it before implementation. Generic chunk1 teacher evaluation would not exercise that proposed StepSample dispatch; any future version would require a real hidden-row/full-head comparison before sampling.

Existing VOLTA_STEPT text omits t. The profile parser therefore joins only the profile run's own ordered positive-t v2 observations to its own stderr lines, skipping idle t0 records. It checks the configuration/hash and complete-prefix provenance. These logs synchronize after begin/layers; finish contains output normalization, head projection, GPU sampler and result download. Finish is an upper bound on head savings, not isolated head GPU time or primary throughput timing.

| Profile interval | Own sequence range | API wall | Eligible finish upper bound | Wall fraction |
|---|---|---:|---:|---:|
| Fixed |100..393|16.2360s|247.15ms|1.52%|
| Mixed |394..768|27.2165s|702.48ms|2.58%|

The budget sums the per-global-step maximum eligible finish across GPUs instead of adding concurrent GPU times. Eligible coverage is complete: fixed has4145uncached prompt tokens in65chunks on each active GPU; mixed has4145/65 onGPUs0-2 and8242/129 onGPU3. The last eligible mixed prefill is sequence608, before the768snapshot. Its missing final~1.33s contains only decode work outside the proposed t>=32 prefill scope. No eligible prefill budget is omitted by the truncated trace tail.

Typical64-row finish is3.53-3.75ms. Even eliminating the entire eligible finish, including work the proposed head optimization would retain, cannot meet5% on either primary trace. The head candidate is rejected. No numerical/logit comparison or head performance improvement is claimed because no candidate was built.
