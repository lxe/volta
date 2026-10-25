# Independent HP GPU sampling source review

Reviewed the isolated `/home/lxe/volta-hp-sampling` working diff against b93eebf, limited to engine/src/gpu.rs, engine/src/volta.rs and engine/src/server/volta.rs, alongside hp-sampling-prototype-plan.md and hp-sampling-validation-protocol.md. This was a CPU-only read of source; no source was edited, no build was launched by this reviewer and no hardware validation was performed.

No blocking correctness bug was identified in the reviewed diff. This is source-review clearance for the parent's bounded validation protocol, not evidence that the candidate runs correctly or faster on hardware.

## Reviewed ownership and sequencing

- `engine/src/volta.rs:2678-2695`: every HP worker computes its existing packed head output, synchronizes its own main stream, then reaches the common barrier. GPU0 therefore reads completed source rows. Context tokens and pointers originate in each live Shard, whose allocations and contexts remain owned by worker threads throughout the command. The existing peer-access setup applies to these primary contexts.
- `engine/src/volta.rs:3225-3248`: the caller validates its single caller-visible HP plan, broadcasts an explicit sampled command to all eight workers and drains all eight results before returning. Nonzero workers can return early after the barrier, but cannot overwrite logits with another command until the caller issues one after GPU0 completes. A second audit barrier is unnecessary under that single-command ownership discipline.
- `engine/src/volta.rs:3237-3240`: returned worker errors preserve the existing immediate process-exit contract. An error before the readiness barrier therefore does not cause the caller to join peers stuck in that barrier. As in the existing collective engine, this reasoning covers returned errors rather than arbitrary worker panics, hangs, driver stalls or hardware failure; the independent hard deadline/watchdog remains necessary.

## Reviewed bounds, layout and audit

- The existing `w.logits` allocation is full `TMAX*vocab` FP32 capacity, even though the HP head writes packed `t*(vocab/8)` rows. The new separate GPU0 gather is also full TMAX*vocab capacity. The copy helper uses source pitch vs*4, destination pitch vocab*4, slice offset gi*vs*4 and height t; the union of the eight slices exactly fills each full vocabulary row without source/destination aliasing.
- `engine/src/volta.rs:2848-2858` rejects zero/oversized T and table counts, mismatched ten-word tables, negative/out-of-range row bases and draft counts, rb+nd beyond available rows, invalid draft token IDs, nonfinite temperature and uniforms outside [0,1). The validator covers every table entry read by the unchanged kernel, including its final sampling uniform at tf[1+nd]. Finite nonpositive temperature is deliberately allowed for mixed greedy/sample tables.
- `engine/src/volta.rs:2683-2685,2702-2726`: audit slices are downloaded from the original eight local packed heads before the barrier, and CPU concatenation uses an independent host-side indexing operation. It compares every gathered u32 bit. Production gather and small result download finish before GPU0 uploads that independent concatenation into its old full-sized logits workspace for replay. The production result is already retained on the CPU. Other GPU source allocations are untouched, and subsequent normal head/draft operations rewrite logits rather than consuming these old logits as hidden state.
- The audit's maximum logits host storage is approximately three full matrices: original eight local slices, the gathered matrix and the independent reference, about99.29MB at64rows. `drop(actual)` occurs before replay upload. Per-command Arc ownership keeps the host slices alive until the whole command has returned, then releases them. This is bounded by one synchronous command, not accumulated across audited steps.
- Replay deliberately uses the same unchanged sampler kernel and tables on independently concatenated real logits. It validates collection and addressing, not an independent statistical implementation of sampling. The validation protocol states this limit correctly.

## Defaults, fallbacks and RNG semantics

`hp_sampling_enabled` at engine/src/volta.rs:2842-2844 requires actual eight-GPU HP, divisible vocabulary/head shape and explicit VOLTA_HP_GPU_SAMPLE=1. Allocation and dispatch use this same guard; default DP and HP-off execution retain their old branches. The server additionally preserves VOLTA_GPU_SAMPLE=0, top-p<1, hesitation and oversized-decode fallback exclusions. Entirely greedy batches retain StepGreedy; mixed greedy/sample tables may use the existing GPU sampler with temperature0 for greedy rows. Generic teacher evaluation does not exercise this sampled command, so the planned real-row audit remains required.

The GPU path consumes nd+1 random numbers up front per table row, including greedy rows in a mixed batch, whereas the CPU path draws only the values reached by its acceptance/residual branches. Floating-point reductions and f64-to-f32 uniform rounding also differ. This behavior already exists in DP GPU sampling and is now exposed to opted-in HP. Consequently, equal seeded text across CPU and GPU arms is not a valid acceptance condition; use equal supplied-table/uniform replay for deterministic audit, quality gates for model behavior, and exact output budgets for throughput. The candidate must not claim CPU/GPU seeded trajectory equivalence.

The dispatch and sample-audit counters are separately bounded. Dispatch counts all HP scheduler steps, while sample audits count actual StepSampleHp commands. Their audit IDs therefore need not line up one-to-one. The protocol correctly places fallback checks before the long staggered phase and requires explicit coverage records. Check actual rb63, speculative accept/reject, multiple table rows, mixed greedy/sample, tails and top-p/hesitation fallback evidence rather than inferring coverage from successful responses.

## Reviewed source fingerprints

- engine/src/gpu.rs SHA256 b84deb7f5a1117027798c763674ea81acea53c59e83721a32be03637387c0b2d
- engine/src/volta.rs SHA256 9926e1c5cb86846c25a0dd7b6eb6669b8581e121ba1e320d1d94d01bea2082c9
- engine/src/server/volta.rs SHA256 2f871c5da82894bb573b95713a66e89473459cc1c10191a01b55fb44e0100051

These fingerprints identify the reviewed working files rather than a committed candidate or a remotely executed binary. Bind any launch to its final source/binary/kernel hashes, and review changes beyond these fingerprints before treating this report as applying to them.
