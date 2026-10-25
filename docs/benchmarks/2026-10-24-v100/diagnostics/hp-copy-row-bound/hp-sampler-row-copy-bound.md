# Bound for copying only HP sampler-needed rows

Defer this candidate. The available evidence does not establish a measured opportunity for at least 5% further throughput improvement, and no new trace, build or GPU experiment is justified in the remaining window. A deliberately optimistic calculation using the old HP trajectory gives only 3.33–3.41% higher tokens/s even if every copy vanished. This is a conditional screening result, not proof that a different production trajectory can never benefit by 5%.

This review used existing local source, logs, audit tables and timing receipts only. It changed no engine source, server, shared repository or GPU state. Exact calculations are in `hp-sampler-row-copy-bound.json`.

## Conditional old-trace calculation

The source-bound `hp4-loop-prof-001` receipt records 485 fixed-phase and 574 mixed-phase loop passes. Its configuration enables STEPT and per-pass loop profiling but has graph_trace=false. The STEPT text records begin/layers/finish times without T; therefore these logs do **not** supply per-pass row counts. Assigning actual row shapes from a different run would be invalid.

The standalone collection diagnostic measured a t64 median of 1.842944 ms and maximum of 1.848576 ms over 20 repeats. Charge that observed maximum to **every** old pass, regardless of its actual row count, then assume all copy cost can be removed without replacement work:

| Phase | Old pass count | Conditional removable time | New measured GPU wall | Conditional tokens/s gain |
|---|---:|---:|---:|---:|
| Fixed | 485 | 0.896559 s | 27.162822–27.270288 s | 3.40–3.41% |
| Mixed | 574 | 1.061083 s | 32.877694–32.961676 s | 3.33–3.34% |

For wall time T and saved time c, the tokens/s gain is T/(T-c)-1. A 5% gain requires c > T/21: 1.293468–1.298585 seconds fixed and 1.565604–1.569604 seconds mixed. At the standalone observed t64 maximum, that requires at least 700–703 fixed copy passes or 847–850 mixed copy passes before accounting for any new cost. The old trajectory has only 485/574 passes.

The old counts come from an instrumented CPU-sampling run with different stochastic trajectories and slightly different prompt totals from the matched GPU arms. The uninstrumented B1/B2 logs and final API counters contain no iteration or row-count counter. Thus the old counts are not actual production B counts, and combining them with new walls is explicitly a conditional comparison. The standalone observed maximum is also not a universal upper bound on model-context copy latency. No interpolation between t1/t3/t12/t64 was used or presented as a measurement.

## Actual row-selection opportunity in the v3 audit

For each of the 598 successful real-row audit records, the existing sampler may read every logical row in rb..rb+nd, inclusive. The required set is the union of those ranges across table entries; it cannot be limited to the eventually accepted prefix because acceptance has not been determined yet.

The actual audit contains 17,467 full rows and 5,020 unique potentially needed rows. Row compaction could omit 12,447 rows, or 6,436,592,640 bytes across these audit cases. There are 232 records with unused rows and 366 where every row is needed. Of 228 full64 records, 24 need only one row; the remainder need between 2 and 18. There are also 571 records with speculative tables, including 205 mixed records that still contain unused prefill rows.

These figures describe the six audit workloads, not fixed/mixed B1/B2 timing phases. They demonstrate that prefill has unused rows, but do not convert byte savings into measured latency or a throughput gain. The source log SHA256 is `aa18432ffe1e2e4afe922695dd328ca89c08ea5f600f891fc766a28cd4368a8c`. The row unions are checked against the original matrix bounds and retain all speculative rows. No trace is fabricated for the production arms.

## Costs and invariants of a future design

The head still computes its existing full matrix; selecting rows for collection alone does not remove head arithmetic. Pure decode already needs all rb..rb+nd rows, including rejected drafts, so it has little or no row-copy saving. Prefill needs its final row; mixed steps need that row plus every decoder's speculative interval.

A host-side mapping would need to preserve row order, assign compact destination bases and rewrite every sampler rb while retaining nd, draft IDs, temperatures and supplied uniforms. Multiple disjoint intervals can require several pitched copies per GPU instead of one: up to four HP slot intervals could expand eight copy submissions to as many as 32 before merging adjacent ranges. More driver calls, host mapping/table uploads and launch synchronization can consume the saved transfer time. A source-side gather kernel could reduce copy submissions but adds GPU work, a new kernel/fatbin and its own correctness/resource scope. Neither option has a measured budget here.

All eight workers must still complete their head outputs before collection, and the caller must still wait for GPU0's sampling/download before issuing a command that overwrites sources. The readiness barrier and source-lifetime discipline cannot be removed by reducing row count. Audit must compare the selected original bits and deterministic sampler output against an independent reference with the remapped table; rb63, mixed greedy/sample, multi-slot, tail, speculative accept/reject and fallback coverage remain necessary. DP and disabled-flag behavior must remain unchanged.

A future confirming observation would need actual production per-pass T, table-required ranges, copy interval and pass count, bound to the same source/workload as its wall-time denominator. Copy timing itself can perturb execution and must stay separate from the primary score. Without such an already available observation, the conditional result below 5% and added compaction costs support deferral. This is not a universal impossibility claim, and no positive gain is booked.
