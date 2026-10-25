# Revised sixteen-replacement graph hypothesis

The original eight-replacement policy failed its own screening gate on the actual diag-v2-001 trace. That result remains in receipts/diag-v2-001-graph-analysis.json and must not be relabeled as passing. A separate sixteen-replacement CPU hypothesis is justified because the bounded fixed/mixed phases show sustained admission misses and stale graphs, while the observed capture costs are small enough to estimate a finite replacement budget. This note precedes the revised CPU simulation. It authorizes no implementation, compiler, model or GPU launch.

## Actual phase evidence and coverage

The same-process server UTC log identifies fixed request admissions20:05:43.351Pacific through final completion20:05:59.669, followed by mixed admissions20:06:00.495 through final completion20:06:27.160. Client wall durations are16.6056s fixed and26.9778s mixed, each2048output tokens with zero cache. Matching trace header epoch plus monotonic elapsed selects sequences98..405 for fixed and406..768 for the visible mixed prefix. Timestamp alignment is approximate host-record alignment; it is not GPU completion timing.

All four active GPUs show100% budget misses among t1..6 rows within both primary intervals, with repeatedly missed keys and graphs that remain resident throughout that interval without being replayed during those rows. Thus the phase-local stale/admission evidence gate passes for all four active GPUs. The failed whole-prefix persistent-stale check was caused by earlier smoke observations before those graphs were admitted; phase selection repairs the observation interval, not the policy or replacement-cap gate. Simulation must still replay the entire smoke/warm/fixed/mixed prefix without resetting its cache at sequence98 or406.

The source log confirms actual smoke39, warm563, fixed4145 and mixed8242 prompt-token counts, final chunks39/51/49/50, all above9. Under this DPslots1/spec5 protocol, t1..6 rows are generated verification rows. Four idle GPUs have t0 and no generation evidence; they should not be required to pass an active-decoder gate.

Every GPU has768detailed records and a completed matching summary; no full/failure was observed. The final snapshot is approximately20:06:25.879-25.884, leaving1.276-1.281s (about4.7%) of mixed wall time unobserved. Fixed coverage is complete. Do not extrapolate an observed-prefix win into an exact full mixed outcome. receipts/diag-v2-001-primary-interval-analysis.json preserves boundaries, input hashes, phase decisions and the original-eight simulation evaluated over those intervals.

## Measured costs and why a new CPU screen is reasonable

Actual successful active-GPU capture-path host durations range12.78-70.12ms (GPU0 mean20.42ms; GPUs1-3 means42.79-44.62ms including their first t0 capture). First idle/t0 captures were80.11-92.09ms. There were no captures during the selected fixed/mixed intervals: the current cache had already filled. These are capture/enqueue/instantiate host times, not GPU elapsed times or measurements of replacement destruction/synchronization.

Primary budget-miss enqueue medians were6.20-6.27ms on the fixed trace. Mixed medians were21.26-21.72ms on GPUs0-2 and6.29ms onGPU3; corresponding means were11.18-17.81ms. These include host backpressure and are not removable-time estimates. The trace does not supply a same-key hit/miss A/B at the same context, so no throughput gain can be booked from subtracting these numbers from graph-launch calls.

The original eight-replacement simulation reached its cap on every active GPU and left63/16/148/122 qualifying observations blocked. This is specific evidence that the cap binds in a768-step controlled run. The sixty-four-step replacement spacing itself bounds this prefix to at most twelve replacements/GPU. A limit16 therefore permits testing the spacing/frequency policy over this actual prefix without changing its four-resident memory budget or relaxing its admission thresholds. It also makes explicit whether a remaining problem is four-slot churn rather than the original small trial cap.

## Revised resource and capture budget

Keep four retained main executable graphs per GPU, no changed draft policy,64-observation frequency window, minimum8candidate observations, frequency advantage4, and at least64steps between replacements. Increase only the per-GPU replacement limit from8 to16; total worst-case replacements are128 across8GPUs. In this controlled prefix only four GPUs are active; idle t0 remains hot and should not replace anything. The frequency/spacing logic and strict cap are otherwise identical. No token-count priority is introduced.

CPU simulation needs only the existing768-key sequence/GPU and small counters; it adds no GPU/host weight storage, networking or disk bulk. A future actual candidate would add no persistent tensor allocation, must destroy a completed victim before instantiating another, and must preserve all previously stated CUDA ownership/temporary-graph cleanup rules. Four handles do not guarantee equal byte usage for different graph shapes. The existing same-budget proposed headroom gate still needs parent acceptance before any actual policy trial: no more than32MiB lower free memory than its matched baseline and never under192MiB. The rejected graph16-resident experiment remains rejected.

Use100ms as a conditional replacement pause ceiling including synchronization, destruction and instantiation; this exceeds the measured70.12ms active capture maximum but is not a measurement of cleanup. With16replacements on four active GPUs, a fully serial host ceiling is6.4s, or14.7% of the43.58s combined primary client wall. For this768-step trace the spacing ceiling is12/GPU, giving4.8s; the eight-to-twelve increment is at most1.6s. Concurrent GPU captures can overlap, so sum the maximum observed replacement pause per global step for the actual critical-path charge, not all GPU pauses. The deliberately conservative serial bound shows why cache-hit coverage alone is insufficient.

Before a future actual trial, use revised simulation event counts with measured per-shape capture estimates to price the likely cost. Require actual total replacement critical-path pauses under5% of matched primary wall and every replacement under100ms; kill/restart on an allocation/cleanup error, reserve failure, pause excess or remaining cap pressure. No clock, precision, quant tier, context or kernel change accompanies this hypothesis.

## Prediction and decision criteria

The revised CPU screen predicts fewer missed generated-step keys than the original-eight model, with the largest improvement expected after the mixed interval begins. It may instead reveal four-slot churn and weak gains on GPUs2/3; such a result is a reason to stop, not to increase limits again automatically. The existing fifteen-percentage-point cache-hit coverage threshold remains a screening threshold, and the original failed-eight record remains unchanged.

A plausible actual serving improvement is5-10% only if increased hot-key coverage saves enough host launch/backpressure time after the measured replacement charge. That range is a hypothesis; unmeasured gain is zero. CPU simulation cannot settle GPU resource reclamation, capture pause, numerical correctness or makespan. A passing revised cache screen supports a separately approved implementation/measurement cycle, not promotion.

Kill the revised hypothesis if additional replacements produce under5percentage-points extra selected-interval coverage versus eight, if any active primary interval remains under the original15percentage-point gain threshold, if the16cap still binds, or if a conservative capture-cost charge removes the plausible5% timing budget. A useful but workload-dependent GPU0-only cache benefit is not a four-agent winner. Actual promotion still requires the original teacher/API/tool correctness gates and swapped-order paired four-agent runs with at least5% median throughput/makespan improvement and no more than5% p95 inter-token regression. No policy implementation or GPU run has been approved by this note.

## CPU result: revised hypothesis rejected

The revised simulation completed after the preceding budget was written, without engine or GPU activity. It replays all768observations from smoke onward, then evaluates fixed/mixed UTC-selected intervals. Sixteen allows10/11/11/11replacements on active GPUs0/1/2/3, with zero observations blocked by the cap. The remaining failure is therefore not the16trial quota.

| GPU | Fixed hit gain vs actual, pp (8 and16) | Mixed gain,8 pp | Mixed gain,16 pp |16 minus8 mixed pp |
|---|---:|---:|---:|---:|
|0|36.69|54.55|68.60|+14.05|
|1|29.55|24.24|11.85|-12.40|
|2|14.61|7.16|39.39|+32.23|
|3|46.43|8.82|17.91|+9.09|

GPU1mixed is below the prewritten15pp threshold and worsens12.40pp versus the original-eight simulation. GPU2fixed remains below15pp. Extra replacements thus displace useful resident coverage on at least one active GPU instead of producing a stable four-agent improvement. This fails the revised hypothesis's prewritten kill criteria. Do not implement this exact policy or silently raise its limit again. The actual primary admission problem is demonstrated, but this replacement rule is not justified as its remedy.

The observed replacement schedules have19distinct global replacement steps during fixed and20during the visible mixed prefix. Charging the proposed100ms ceiling per such step produces1.9s (11.44% of fixed wall) and2.0s (7.41% of mixed wall). These are conservative pause-budget proxies, not observations that actual replacement would take that long. A second proxy uses the same GPU's measured capture median for the same t, falling back to its active-capture median where that t was not captured, and charges the per-global-step maximum. receipts/diag-v2-001-revised16-simulation.json records both estimates and their exact methods. Neither includes measured synchronization/destruction/reclamation time. The policy fails the coverage/churn gates independently of which capture proxy is used.

## Trial quota versus production rate

A lifetime16replacement limit is a bounded experiment safety quota. It is unsuitable as an eventual long-running server contract because it permanently stops adapting after later requests and context buckets change. The existing minimum64steps between replacements already implies at most16replacements per GPU in any half-open1024-step window. An eventual production policy would retain that rolling rate bound and would separately enforce memory/latency/correctness limits, while a bounded trial additionally stops at16total replacements/GPU so an unsuccessful experiment cannot consume the rental.

A rolling bound alone does not solve the actual four-slot churn seen here. Four GPUs can replace at different step offsets, producing up to64distinct global replacement pauses per1024steps even though each GPU respects its local rate. At a hypothetical100ms pause ceiling that is6.4s across the box; the true charge depends on measured overlap and steady step duration. A production design would need evidence that each replacement's future saved launch cost repays its capture/reclaim pause, and a way to bound the aggregate critical-path charge. None of that is proved by raising a lifetime quota. This discussion separates contracts for a possible later design; it is not authorization to implement another policy now.
