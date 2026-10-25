# Isolated HP GPU sampling prototype

Base is local repack b93eebf: the sampler guard is present, V3 loop/graph diagnostic patches are archived but not active. Create a new worktree/branch and a distinct remote source/target. Parent retains the live server, controller and all GPU scheduling. VOLTA_HP_GPU_SAMPLE=1 explicitly enables this path only for an actual split vocabulary head; default HP remains CPU. Audit is separately opt-in and off during matched timing.

The reviewed standalone benchmark verified827,392,000bytes over80collections; copy medians t1/3/12/64 are0.076384/0.123120/0.384304/1.842944ms. The actual V3 profile attributes21.80/21.27% fixed/mixed wall to CPU distribution. These measurements justify this prototype, but no engine gain is established.

## Changes and numerical invariants

In engine/src/gpu.rs add a narrow context-token accessor and checked pitched-copy helper using CUDA12.9 driver APIs. In engine/src/volta.rs extend PeerBufs with each worker's logits pointer/context; keep an all8 head-source list separately from the existing other7 collective peers. Allocate one33,095,680byte full-vocabulary gather buffer only onGPU0 when the flag and split head are valid. Existing packed head outputs remain unchanged.

Add explicit Cmd::StepSampleHp rather than inferring participation from sample table rows. Engine::step_sample validates HP support, single caller plan, row/table/draft bounds, spreads the single plan, sends the same caller tables to all workers and collects onlyGPU0's result. Server opts into GPU sampling only when existing eligibility and actual split-head support both hold. All existing CPU, DP, greedy, top-p<1, hesitation and disabled-flag fallbacks remain available.

finish() computes the existing local HP head, synchronizes its own source, and participates in one common host barrier for the explicit HP sample command. GPU0 copies7peer slices and1local slice on its own stream into the separate full buffer, launches the existing vspec_sample with unchanged tables/uniforms, then downloads its small result. Nonzero workers return empty; Engine collects all8before returning, preventing next-command overwrite. Error handling must preserve the existing immediate-fatal worker-result contract so a barrier cannot strand the server. Layer graphs/kernels/quant values/head arithmetic remain unchanged.

## Real-row audit

VOLTA_HP_SAMPLE_AUDIT=N (maximum4096, planned validation1024) performs a bounded number of HP sampled-step audits. All8workers download their actual packed head FP32 bits before the readiness barrier into a shared per-step host slot. GPU0 collects logits through the production pitched-copy path and downloads the full matrix as u32 bits. Independently concatenate the original8CPU slices and compare every bit of every logical row; report row count and checked scalar count. The maximum debug host payload is3full matrices (99.29MB decimal) plus small tables/results. It is freed each step. A separate audit barrier or equivalent ownership discipline prevents source reuse while this check runs.

For deterministic sampler validation, preserve production output, upload independently CPU-concatenated FP32 bits into GPU0's existing full-sized logits workspace after all production source copies/downloads are complete, and rerun the exact frozen vspec_sample with the same supplied tables/uniforms. Require identical(accepted count,token) pairs. This checks gathered data and sampler-table addressing on real hidden rows without introducing a second sampler implementation or new kernel. The next step rewrites logits. Audit logs must report sample row bases/nd, checked counts and supplied uniform/table fingerprints, with at mostNentries. Generic teacher step alone does not exercise this path.

Before server experiments, parent must exercise actual prefills with last row63, generated speculative rows, multi-slot mixed prefill/decode, mixed greedy/sample tables, and fallback top-p/hesitation. Candidate teacher gate remains required, but does not replace actual StepSample audits. No audit overhead belongs in A/B/B/A timing.

## Resource, gain and kill budget

GPU payload increase is31.56MiB onGPU0, small pointer/context metadata onCPU, existing sampler tables/result space reused. No new CUDA image. Audit host payload stays below128MiB plus existing process state; timing mode needs no full CPU logits. Current HP profile minimum1745MiB free leaves calculated1713MiB after gather, above192MiB reserve, but candidate admission and continuous guard remain mandatory. All per-step buffers are bounded by64rows and129280vocabulary.

CPU build allowance is2jobs and180s using a distinct target; verify every kernel source hash and every resulting fatbin equals4387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5. No GPU/model/server launch is authorized to this agent. Aim compile/review readiness04:30UTC; rental release06:18:26UTC remains parent-owned.

Screening prediction is10–20% better matched HP fixed/mixed throughput, with a zero-new-cost ceiling27–28% from observed CPU share. This does not imply HP exceeds DP four-agent throughput. Kill on any real collected-bit or deterministic sampler mismatch, invalid topology/table, worker/barrier failure, unexpected kernel hash change, quality-gate failure, reserve breach, or less than5% repeatable matched fixed/mixed improvement. Parent owns four fresh A/B/B/A HP4/256K/graphs4 servers with identical prompts/budgets/no profiling and the subsequent real256K Pi gauntlet if justified. No policy/kernel/dense-head optimization is bundled.

Audit-only server dispatch records are independently capped byN and include StepSampleHp/Step/StepGreedy, row/table counts and per-row top-p/hesitation/prefill/greedy flags. All-greedy steps retain StepGreedy and do not run the sampled-command parity audit. Parent must include sampled sessions for rb63 parity coverage. Neither dispatch nor full-logit audit records are emitted in timing mode withN=0.
