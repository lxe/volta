# Strict8 joint collective-step criterion

This is a NEW criterion, written before the joint simulation analysis. The existing strict8per-GPU screen failure remains in graph-policy-strict8-hypothesis.md and receipts/strict8-graph-policy-cpu-simulation.json, including GPU1fixed's3.74ppgain on the profile prefix. No value from that failed criterion is changed or relabeled. The new analysis can at most admit a separately authorized bounded actual trial; no code, build, GPU workload or serving gain is authorized by this note.

## Why the unit changes to a synchronized step

Each main engine call enqueues one local shard step on all eight GPUs. Even a shard with t0 participates in routing, experts and collective exchange work. The host waits for all worker results before the next main call; a peer with eager host launch overhead can delay the other GPUs at collective dependencies. Individual key hit percentages average over this shared step and can hide one peer's persistent eager work. A GPU with nine legacy cached shapes can also have relatively little room for improvement while other peers still determine the step. Thus the primary model unit should be the synchronized global main-step sequence, not an arbitrary mean of per-GPU gains.

For each selected global sequence, define a joint hit only when ALL EIGHT GPUs actually replay an existing graph. Count t0 workers because their MoE/exchange contribution is required. A capture/initial-fill is not a replay hit even though it executes the step. The strict8model uses the same definition on its simulated decisions, with no idle-peer exemption. This joint measure is stricter than averaging hit fractions and does not assume that every avoided eager launch is fully exposed on the critical path.

Joint coverage is still a mechanism proxy, not GPU elapsed time. An eager enqueue can overlap useful compute, and an all-hit step can remain limited by kernels, memory or links. Therefore a joint CPU pass cannot establish a throughput gain. Actual paired wall time, capture pause and resource measurements remain mandatory. The model's relevance is that it removes launch-mode disagreement across a collective step; it does not replace actual performance timing with cache statistics.

## Predeclared test and gates

Use BOTH independently observed actual prefixes: normal diag-v2-001 and synchronized diag-v2-stept-001. Each simulation starts at its own first smoke observation and preserves all warm-up state. Select fixed and mixed intervals from that same run's UTC admission/done log and trace epochs. Require matching sequence numbers and eight records for every selected step; reject any cross-GPU disagreement in phase bounds rather than choosing a favorable peer's subset. Compare strict8/16trial to the actual legacy decisions on that identical key sequence. Do not compare it only with the already failed strict4models.

For each of the four run/phase combinations, require at least15percentage-point improvement in ALL-EIGHT joint replay coverage. This replaces no historical per-GPU result: report original per-GPU values beside it and explain any difference. It is a new, explicitly harder-to-satisfy global opportunity gate tied to the collective dependency mechanism. Both fixed and mixed, on both prefixes, must pass; one favorable phase or an aggregate average is insufficient.

Also require the preexisting measured-capture-median proxy (sameGPU/same-t observed successful capture where available, otherwise that GPU's active-capture median; maximum per global capture step) to charge at most5% of each corresponding API wall duration. Include simulated initial fills within the selected interval as well as replacements. Keep full-prefix counts so early capture work remains visible. Explicitly report the conservative100ms-per-global-capture-step stress charge as risk; it may exceed5% and is not discarded. No actual trial can proceed if measured cleanup/transient pauses exceed its accepted budget.

The192MiBreserve,45MiBpossible GPU0steady growth,46MiBtransient allowance, strict8resident bound,16replacement trial quota and minimum64step replacement spacing remain as previously budgeted. A lifetime16quota is only a trial stop; a production design would need the rolling rate/resource contract and evidence of repayment. No quota/gate is tuned further during this analysis. Failure of any joint/capture gate concludes this strict8hypothesis for the available evidence, without another automatic change.

## Coverage and interpretation limits

All completed snapshots contain768records/GPU. Fixed coverage is complete; mixed snapshots omit approximately1.28s normal and1.33s profile of final decoder tail. Report joint mixed coverage for the visible prefix only and retain that limitation. A pass would therefore be provisional evidence for a bounded test, not an exact full-mixed claim. Include all idle t0 observations through the last completed snapshot. The already archived head result remains rejected and is unrelated to this cache analysis.

## CPU result: joint hypothesis rejected

The predeclared all-peer criterion fails three of the four required run/phase combinations. Actual versus simulated ALL-EIGHT replay counts are8versus51of308normal fixed (+13.96pp),8versus40of363normal mixed (+8.82pp),8versus10of294profile fixed (+0.68pp), and8versus95of375profile mixed (+23.20pp). Only the last reaches15pp. The measured-capture-median cost proxy is below2.65% in all phases, but that cannot rescue a failed coverage gate; the100msstress charge remains6.25-9.24% of primary wall as previously reported risk.

This rejects strict8under the new global criterion as well as preserving the prior per-GPU robustness failure. No further threshold change or policy implementation follows automatically. The result supports the narrower conclusion that the existing admission problem is real while this frequency/rate/capacity combination is not a demonstrated four-agent remedy. Strict8-joint-step-simulation.json contains each actual/simulated joint-hit sequence, all criteria and the hash of the preanalysis criterion. There is no booked serving gain, implementation, compiler or GPU trial.
