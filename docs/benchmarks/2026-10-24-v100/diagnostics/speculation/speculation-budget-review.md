# Speculative decoding budget review

A single paired test of `theta=0.2` against the current `theta=0.35` is justified as a throughput hypothesis on the primary DP, 128K-context, four-client configuration. The existing traces show relatively cheap growth in verifier rows, and the draft network already generates five guesses at either threshold. The evidence does not yet establish the sign or size of an end-to-end gain. In particular, the summary acceptance of 0.57167 does not mean that the scheduler usually verifies one draft.

This review used existing local receipts and source only. It ran no GPU or remote commands and made no engine changes.

## Scheduler and acceptance definitions

`engine/src/server/volta.rs`, `Seq::k`, computes `p=(acc/tried).clamp(0,0.99)` and chooses the largest `j<=spec_k` for which `p^j>=theta`, with at least one draft if drafts are available. Every new sequence starts with `acc=0.6,tried=1`. After verifying `k` drafts and accepting a prefix of length `a`, the scheduler updates `acc=0.8*acc+0.2*a` and `tried=0.8*tried+0.2*min(a+1,k)`. This is an exponentially weighted estimate over actually attempted prefix positions, not a measurement of each position's independent acceptance probability.

The telemetry snapshot's top-level `acceptance` instead divides total kept drafts by total proposed drafts over the last ten seconds. Proposed drafts after the first failure remain in its denominator. The summary's median of those overlapping snapshots can include previous warmups and prefill periods. These three quantities—the summary median, rolling kept/proposed ratio and per-sequence `p`—must not be substituted for one another.

For the actual sampler-c1 fixed trace, the 18 snapshots with decoding active and no prefill contain 66 live sequence observations. Their median sequence `p` is 0.699899. Evaluating the scheduler at those observed values gives the following decisions; observations are time-sampled and correlated, rather than a complete distribution of passes:

| Threshold | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---:|---:|---:|---:|---:|
| Current 0.35 | 16 | 18 | 17 | 9 | 6 |
| Proposed 0.2, evaluated at the same p values | 1 | 15 | 11 | 13 | 26 |

The mixed trace has 34 such sequence observations with median `p=0.667999`: current counts are `{1:8,2:12,3:3,4:6,5:5}`, while the proposed threshold would give `{1:0,2:8,3:9,4:5,5:12}`. Once the threshold changes, accepted-prefix histories and future p values also change; these tables are a static counterfactual, not a prediction of the new stationary distribution.

At p=0.57 the specific proposed change is k=1→2. At p=0.70 it is k=2→4. At the initial p=0.60 it is k=2→3. Thus theta0.2 also exercises longer verification and higher expert-batch sizes.

## Per-pass resources and measured timing

For each unfinished decoding session, a pass verifies `T_local=1+k` rows and emits `1+a` new tokens, except at output/stop boundaries. With four active DP sessions on separate GPUs, `T_total=sum(1+k_i)`; expert exchanges depend on the collective workload, so a local T=2 versus T=3 comparison is not an isolated kernel A/B.

The `spark` call occurs before selecting `k`, and `engine/src/volta/spark.rs::spark_step` sets its draft rows to `draft_sessions*block`, with block5 here. Lowering theta therefore adds no extra draft-network generation pass. It adds main-model verification rows, their head/sampler work, potentially longer accepted-token observations on the next spark call, and possibly more rewind work. Existing GPU sampling still supports these row counts: at most five drafts is below the eight-draft restriction.

On longer primary sampler-concurrency four-client runs, snapshots more than ten seconds after all four sessions entered decode show the following rolling averages. They retain only snapshots with four active sessions and no prefill, so their ten-second timing windows contain decoding rather than the initial prompt work:

| Run | Qualifying snapshots | step_ms median | draft_ms median | kept/proposed median | aggregate rolling output tokens/s median |
|---|---:|---:|---:|---:|---:|
| clients-4-a | 9 | 31.0634 | 3.2836 | 0.51189 | 256.9477 |
| clients-4-b | 8 | 31.1031 | 3.2837 | 0.51881 | 260.6883 |

Here `step_ms` is the server timer after draft generation and includes the main verification, token decision and rewind path. `draft_ms` is separate. These are overlapping rolling windows, not independent per-pass samples or shape-specific draft timings. Adding the two median values suggests an approximate 34.35ms decode-pass cost for resource budgeting, with about 9.6% in the draft call. The measured whole-request throughput, which includes prefill, was 165.15 and 169.30 tokens/s for 4×1024 outputs.

The original 512-output fixed sampler-c1 trace completed at 123.888 tokens/s and mixed at 73.655 tokens/s. Their short runs cannot supply a clean ten-second steady-state fixed-decode window. Their unfiltered rolling step medians include prompt work and should not be used as verifier cost.

The separate DP diagnostic `diag-v2-stept-001` records synchronized begin/layers/finish phases and same-process V2 token-count traces. Source/binary provenance is recorded in `server-runs/diag-v2-stept-001/configuration.json`; binary SHA256 is `fd6c71b58683a113b2b4746526c32681dd43933af5fcd5e37eb81500e7f593e1`. This is a diagnostic instrumented build, not the primary sampler binary (`b9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4`). Synchronization perturbs timing.

The CPU analysis joins positive-T trace rows to each GPU's STEPT lines by ordinal, excludes t=0 records, and selects the previously established fixed sequence interval100..393 or mixed394..768. The rows below require exactly four GPUs with positive T and every local T in1..6. Under this controlled 4145/8242-token prompt protocol with slots1 and prompt tails49/50, these are four-client pure-decode passes. Timings are sums of each GPU's observed begin/layers/finish phases, in milliseconds; GPU samples within a pass are correlated.

| Local T | Fixed GPU samples | Fixed phase-sum median | Mixed GPU samples | Mixed phase-sum median |
|---:|---:|---:|---:|---:|
| 2 | 136 | 31.310 | 20 | 31.260 |
| 3 | 223 | 31.800 | 46 | 31.325 |
| 4 | 168 | 31.995 | 55 | 32.640 |
| 5 | 75 | 32.720 | 23 | 32.710 |
| 6 | 142 | 32.860 | 28 | 33.605 |

Across fixed pure-decode passes, the maximum per-GPU phase sum has medians 30.120ms at collective T_total12 (11 passes), 32.540ms at T_total16 (34), 33.890ms at T_total20 (5), and 35.550ms at T_total24 (one). These are observed associations across changing acceptance, routes and positions, not isolated shape-cost perturbations. They support budgeting several milliseconds for extra verifier rows; they do not prove the incremental slope. Do not sum the GPU durations as serial time. No shape-specific draft-call timing is present in these receipts; the approximately3.28ms measured draft cost is the primary four-session aggregate baseline, not a measured draft cost for each T.

## Output model and break-even conditions

Define `q_j=P(the first j proposed drafts are all accepted)` for the actual workload and pass state. Expected output for k drafts is

`E_k = 1 + sum(q_j, j=1..k)`.

This expression uses prefix probabilities and requires no independence assumption. For a change k→m, the new expected output is `E_k + sum(q_j,j=k+1..m)`. It pays when

`sum(extra prefix probabilities) / E_k > (C_m-C_k)/C_k`,

where C includes draft, main verification, observation, sampling and rewind costs. The present receipts contain no position-conditioned q_j estimates or complete accepted-prefix/pass records, so neither the global acceptance ratio nor a median sequence p identifies the missing terms.

For an illustrative p=0.70 session changing k2→4, adding the observed approximate3.28ms draft baseline to the diagnostic T_total12→20 phase-sum medians gives 33.40→37.17ms, or 11.3% higher pass cost if all four sessions made that transition. Break-even requires `(q3+q4)>0.113*E2`. Since E2<=3, the conservative sufficient condition is q3+q4>0.339. The diagnostic workload is not a paired perturbation, so this is a planning threshold rather than a measured threshold.

For scale only, a *hypothetical stationary independent-position model* with q_j=0.70^j gives E2=2.19 and E4=2.7731. Under the above approximate costs its conditional decode gain is about14%. This calculation is not evidence that those q_j values hold. Correlated failures or degraded later guesses can erase the gain; added GPU verification without accepted output slows the run. The p=0.57 illustration likewise has E1=1.57 and E2=1.8949 only under that explicit hypothetical model, not from the summary acceptance number.

## Proposed bounded experiment and kill criterion

Test only theta0.2, keeping sampler source/binary, weights, quantization, DP slots1, ctx131072, chunk64, spec_k5, temperature0.7, top_p1 and graph settings identical. Use the existing fixed/mixed workloads and a four-client1024-output case with fresh nonces and zero cache hits. Repeat in reversed A/B order against theta0.35; keep exact completion counts, requested outputs, prompt sizes, TTFT, total wall throughput and rolling pass timing. If the first candidate arm is invalid or clearly slower, stop without a parameter sweep.

The hypothesis is a positive decode gain of roughly5–15%, with a smaller whole-request gain because prefill is unchanged. For the current four-client1024-output workload, approximately one third of wall time is before first token, so a conditional5–15% decode improvement would translate to roughly3–9% whole-request improvement if the amount of prompt work and overlap stays comparable. This is a falsifiable working prediction, not a booked gain. The mixed trace has interacting prompt/decoder critical paths, so no precise percentage is justified there; it must not regress materially.

Kill on any invalid sample/result, incorrect token/output count, context or tool-format regression, resource reserve violation, or unexplained behavior. Reject promotion if matched four-client whole-request gain is below5%, if reversed-order results do not agree in direction beyond observed baseline drift, or if mixed throughput/TTFT worsens by more than3%. A lower kept/proposed acceptance is expected when more unused tail drafts are proposed and is not by itself a correctness failure. Measure actual outputs per pass and verifier-row histograms if a reviewed observer is already available; do not mistake the currently archived rolling counters for that instrumentation. End-to-end throughput determines promotion.

## Evidence paths

- `screens/sampler-c1/{fixed,mixed}/{stats.jsonl,summary.json}`
- `screens/sampler-concurrency/clients-4-{a,b}/stats.jsonl`
- `receipts/sampler-concurrency-profile.json`
- `graph-traces/diag-v2-stept-001/{gpu0..7.jsonl,server.log}`
- `receipts/diag-v2-stept-001-{fixed,mixed}-analysis.json`
- `parse-volta-stept.py` and `analyze-graph-trace.py` for the established ordinal-join/provenance rules
- `speculation-cpu-analysis.py` and `speculation-timing-analysis.json` for this CPU-only per-T regrouping
