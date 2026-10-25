# Strict eight-resident graph hypothesis

A strict eight-main-graph cache is a separate CPU-only hypothesis from the rejected four-resident replacement rules. Actual legacy nominal-four behavior already retained eight graphs on GPUs1-3 in diag-v2-001, and nine onGPU1 in the separate profile, so increasing a strict resident bound to8can be budgeted against observed allocations. This note precedes the CPU simulation. It authorizes no implementation, build or GPU trial. Both failed four-slot8/16replacement reports remain preserved.

## Memory and capture admission budget

Actual final normal-diag free memory was349MiB onGPU0 (six main graphs),319MiB onGPUs1-3 (eight), and481MiB on idleGPUs4-7 (one). The profile had the same graph counts exceptGPU1 retained nine, with297MiB free. The22MiB difference for that extra graph is direct measured evidence for one shape, not a universal graph size. Use approximately22.5MiB/graph as a screening allowance and46MiB transient capture/instantiate allowance. CUDA driver reclamation and graph-shape byte variation still require a future measured check.

At strict8, GPU0's six-to-eight increment budgets45MiB plus46MiB transient:349-45-46=258MiB,66MiB above the192MiB reserve. GPUs1-3 already held8or9, so conservatively subtract only46MiB from the lowest observed297MiB:251MiB,59MiB above reserve, without crediting any ninth-graph reclamation. IdleGPU t0 stays hot and should not admit unused shapes; if different traffic activates them, their481MiB baseline and per-graph growth need a new workload receipt.

This is a NEW resident-count budget, not a silent relaxation of the previous same-four-slot32MiB-delta rule. A future trial must keep at least192MiB measured free at every capture/replacement and stop on any observed allocation/cleanup error or transient larger than46MiB. Before proceeding, require a fresh pretrial snapshot and add actual admitted-shape bytes to the ledger. Eight graph handles are not a byte-exact cap. Graph16resident remains rejected: from an eight-graph319MiB baseline, another8*22.5+46 leaves93MiB; fromGPU0's six-graph349MiB, another10*22.5+46 leaves78MiB, both below192MiB.

All weights, KV, draft graphs and fixed tensors remain unchanged. CPU metadata stays bounded; no extra model data, pinned RAM, transfer or new CUDA kernel is required for CPU replay. Eventual source work would need strict admission on every capture, explicit checked destruction of a completed victim before instantiate, and cleanup of temporary graphs on all instantiate outcomes. No such code is implemented here.

Use the existing observed active capture12.78-70.12ms range to price CPU simulation events, with same-GPU/same-t medians where available and a100ms replacement-pause ceiling as a conservative synchronization/destruction/instantiate hypothesis. Neither measures reclamation. Initial fills are separate from replacements and must be charged too: compared with a strict four-slot model, up to four extra initial fills/GPU could add0.4s/GPU at that ceiling, though many correspond to graphs the actual legacy baseline already held. Preserve smoke/warm cost and report primary-interval events separately rather than hiding all capture work outside timing.

## Policy, prediction and failure conditions

Change only the resident capacity in the CPU model from4to8. Keep second-occurrence initial fill,64-step frequency window, minimum8candidate observations, frequency advantage4, least-frequency/oldest-replay victim, minimum64steps between replacements and trial safety quota16replacements/GPU. This tests whether fitting the observed t2..6context8192working set reduces churn. It introduces no hard-coded token-count preference. Replaying the full observed prefix is required; evaluating a primary interval must not reset admission state.

A lifetime16quota is a trial safety stop. An eventual sustained server contract would enforce the rolling rate bound (at most16replacements/GPU in any half-open1024-step window), memory reserve and replacement-cost repayment instead of permanently disabling adaptation after16changes. The rolling bound alone does not prove good policy; four independently replacing GPUs can still create aggregate pauses. Keep the trial quota until actual captures/correctness/timing justify any production design.

Prediction before CPU replay: retaining eight graphs should reduce the four-slot model's churn among five common generated token counts plus idle/prefill shapes. Target at least15percentage-point primary coverage gain on every active GPU, preferably20ppor more in aggregate, without renewed cap pressure. Actual serving improvement remains a conditional5-10% hypothesis only if measured new capture pauses and memory growth preserve a net5% primary budget; unmeasured gain is zero.

Kill this CPU hypothesis if any active primary interval remains below15pp coverage gain versus actual, if it materially worsens coverage relative to the failed four-slot models, if the16trial cap still binds, or if measured-capture-median cost proxies exceed5% of primary wall before any cleanup charge. Retain the100ms ceiling as a worst-case admission/kill budget, and report when it is too pessimistic to settle net benefit. A CPU pass supports only a separately approved correctness/resource test, not implementation or promotion. Actual promotion still requires teacher/API/tool correctness and swapped-order matched four-agent>=5% throughput/makespan improvement with<=5% p95 inter-token regression.

## CPU results and decision

The actual nonprofile diag-v2-001 prefix passes the strict8cache-coverage screen, with8/9/10/9replacements on active GPUs and no quota-blocked observations. All primary intervals gain at least15pp versus actual. The separate synchronized profile prefix supplies a useful robustness check with a different adaptive sequence: GPU1fixed gains only3.74pp, below the prewritten15pp gate. That prefix also loses3.40pp compared with the already failed four-slot16model. Thus this hypothesis is not sufficiently stable across the available actual prefixes to justify automatic implementation. The profile is instrumented and changes scheduling; its low model gain is not a measured policy slowdown, but it prevents a broad robustness claim.

| Input prefix | GPU | Fixed gain vs actual, pp | Mixed gain vs actual, pp | Replacements |
|---|---:|---:|---:|---:|
| Normal v2 |0|39.61|83.47|8|
| Normal v2 |1|34.09|35.81|9|
| Normal v2 |2|19.48|39.94|10|
| Normal v2 |3|51.30|24.24|9|
| V2+STEPT |0|26.87|84.00|9|
| V2+STEPT |1|3.74|51.73|9|
| V2+STEPT |2|22.11|49.87|8|
| V2+STEPT |3|43.54|23.47|9|

The normal-prefix measured-capture-median proxies charge0.439s fixed (2.64%wall) and0.567s mixed (2.10%); profile proxies charge0.410s (2.52%) and0.445s (1.63%). Conservative100ms-per-global-capture-step ceilings are1.5/1.8s normal (9.03%/6.67%) and1.5/1.7s profile (9.24%/6.25%). The actual transient/reclaim pauses remain unmeasured; these ranges do not establish net5% serving benefit. Primary intervals have no initial fills in this simulation because initial admission occurred in smoke/warm-up; the full-prefix report still retains initial-fill counts so cold-start cost is not erased.

The new memory ledger is plausible for a bounded future test, and the normal-only CPU coverage result is promising, but no implementation, build or GPU trial occurred. The immediate result is further evidence, not promotion. A future proposal would need explicit handling of the stability failure and measured capture-cost repayment; changing the gate after observing this profile would be a new hypothesis. Sixteen remains a trial safety quota, while minimum64step spacing defines the potential sustained rolling rate. No production policy is claimed.

Receipts: strict8-graph-policy-cpu-simulation.json (both actual prefixes, complete event lists, phase boundaries and cost proxies) and strict8-graph-policy-synthetic-test.json (CPU implementation checks only). Failed four-slot reports and their committed raw evidence remain unchanged.

A subsequent NEW collective-step criterion was written before additional analysis (graph-policy-strict8-joint-criterion.md). It preserves the per-GPU failure and requires>=15pp ALL-EIGHT joint replay coverage gain plus<=5% measured-capture proxy on BOTH phases of BOTH prefixes. It also fails: normal fixed/mixed gains13.96/8.82pp, profile fixed/mixed0.68/23.20pp. Three of four fail; no implementation or trial is justified. This separate rejection is not a reinterpretation of the earlier per-GPU gate.
