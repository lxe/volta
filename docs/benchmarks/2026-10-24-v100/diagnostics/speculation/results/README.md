# Theta0.2 A/B/B/A results

Retain theta0.35. The single proposed theta0.2 change completed all four fresh-server arms but failed the precommitted requirement of at least5% higher primary four-client whole-request throughput in both swapped comparisons. Primary1024-output gains were3.8832% and2.4533%. No additional threshold was tested in this experiment.

| Arm | Theta | Fixed512-out TPS | Mixed512-out TPS | Primary1024-out TPS | Minimum free MiB |
|---|---:|---:|---:|---:|---:|
| A1 |0.35|123.4308|73.7528|167.9582|319|
| B1 |0.20|129.6879|74.8199|174.4804|297|
| B2 |0.20|130.4228|74.7475|168.4230|297|
| A2 |0.35|125.5614|76.4673|164.3899|319|

B1/A1 improved fixed5.0693%, mixed1.4468%, and primary3.8832%. B2/A2 improved fixed3.8718% and primary2.4533%, while mixed fell2.2491%. Mixed worst TTFT increased0.3777% and1.1413%, within the3% limit, and mixed throughput remained within its3% regression limit. These secondary passes do not satisfy the failed primary gain gate.

Baseline A2/A1 drift was+1.7261% fixed,+3.6805% mixed and-2.1245% primary. Candidate B2/B1 primary throughput also fell3.4717%. These changes limit interpretation of the small gains; two same-direction primary comparisons establish neither a5% gain nor a global threshold optimum. Sampled text, expert routing and acceptance can differ despite matched requests. The rolling acceptance summary does not expose per-position prefix probabilities or the scheduler's exponentially weighted state.

All arms used the same sampler binaryb9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4, frozen4387f0e3 CUDA image, model and quantization. Serving remained DP, eight GPUs, slots1, context131072, chunk64, graph budget4 and spec-k5, with profiling/audit off. The complete request JSON objects, including prompts and seeds, match across all four arms for each phase. Every fixed/mixed phase completed2048output tokens and every primary phase completed4096; all four clients reached their full requested output budgets with zero cache reuse. Prompt-token totals and actual SSE wall times are retained in each summary and archive-validation.json.

Every arm's guard configuration proves a first valid observation before workload admission. All copied observations bind the correct server PID, contain no stop, and remain above the192MiB reserve. Counts were75/72/73/74 for A1/B1/B2/A2; maximum gaps were approximately1.10–1.14s. Guard minima were319MiB for both A arms and297MiB for both B arms. These are sampled observations, not proof of the instantaneous allocation minimum between polls.

This archive retains full screens, server configurations/readiness/switch logs, server logs, guard observations, exact experiment/analyzer scripts, the original hypothesis and supporting CPU timing analysis, sampler source patch/build proofs, and the completion/controller receipts. Controller/harness source files are snapshots copied after the experiment; per-arm configurations bind actual commands, environment, PID and binary. The initial build receipt's activated=false field describes that historical build stage. No code change, parameter sweep or model/GPU action was performed by this archival step.

manifest.json records byte counts and SHA256 for every other archived file. archive-validation.json independently checks prompt equality, full output budgets, zero cache, configuration/binary identity, first guard observations and actual minima. The original promotion_gate_passed=false analysis is preserved unchanged.
