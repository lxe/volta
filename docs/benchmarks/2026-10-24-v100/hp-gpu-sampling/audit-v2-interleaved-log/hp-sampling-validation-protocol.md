# HP GPU sampling validation

This protocol applies to the isolated, opt-in HP GPU sampling candidate. The parent owns model and server launches. Audit and synchronizing profile modes stay off during primary throughput timing.

## Correctness workload

Use four HP slots, context 262144, chunk 64, graph budget 4, spec-k 5 and theta 0.35. Enable VOLTA_HP_GPU_SAMPLE=1 and VOLTA_HP_SAMPLE_AUDIT=1024. Verify binary and kernel hashes, then start a memory guard tied to the exact live server PID before submitting requests. Allow at most 12 minutes for validation, subject to the absolute inference cutoff at 06:16:26 UTC. Stop on an audit mismatch, worker failure, memory reserve breach or request error.

First submit one sampled 512-token input with a 128-token output budget. This exercises full 64-row prefill chunks, whose final sampling row is 63, and ordinary decode. Next submit four distinct 512-token inputs with 256-token output budgets and temperatures 0, 0.7, 0 and 0.7. These requests exercise multiple HP slots and mixed greedy/sample tables. After the fallback checks below, submit three sampled 512-token decoders with 512-token output budgets. After all three emit a token, admit one sampled 8192-token prefiller. Retain actual requests, streamed outputs, token counts, active-session statistics and phase timestamps.

Audit records must demonstrate actual row base 63, at least one speculative draft, accepted and rejected drafts, multiple table entries, mixed greedy/sample tables and partial tail rows. Record a missing coverage category explicitly rather than inferring coverage from a request. An entirely greedy batch uses the preserved StepGreedy path and does not exercise the sampled audit.

Before the longer staggered phase can exhaust the bounded dispatch log, submit one top-p 0.9 request and one reasoning request with a positive hesitation_penalty. Confirm CPU fallback through source-bound selection and logs or an explicit counter. A successful response alone does not prove which path ran. Check pure greedy StepGreedy behavior separately. A fresh audit server is justified only by a specific missing coverage case; do not retry automatically.

Each audit compares every gathered FP32 bit against an independent concatenation of the original CPU-downloaded local slices. It then compares sampled (accepted count, token) pairs by running the unchanged GPU kernel with the same uniforms and tables on that independently collected matrix. These checks validate collection and table addressing on real model rows. They do not independently establish the sampler's statistical quality.

Run the established 8184-position teacher gate with a candidate binary built from the same engine code. Relative to frozen DP, require perplexity <= 9.0423, KL top-256 <= 0.131695 and top-1 agreement >= 0.838231. These are the existing tight gate tolerances: +0.1 perplexity, +0.005 KL and -0.005 top-1 agreement. Preserve actual metrics and configuration. Teacher evaluation alone does not exercise StepSampleHp.

## Matched performance

Start four fresh HP servers in A/B/B/A order, each with four slots, 256K context and graph budget 4. A uses CPU HP sampling; B enables HP GPU sampling. Model and kernel hashes must match. Profiling and audit remain off.

Use identical pair nonces, one smoke, two warmups, the fixed four-client 4096-to-512 trace, and the mixed trace with three decoders plus an 8192-token prefiller from run-arm.py. Require 2048 output tokens and zero cache reuse in every fixed and mixed trace. Before replacing a server, copy its configuration, log, GPU clocks and memory, and guard observations. Retain every arm, including failures.

The screening prediction is 10–20% higher HP throughput. Promote only if both swapped comparisons exceed 5% in each phase, correctness and teacher gates pass, and continuously observed free memory remains at least 192 MiB per GPU. Compare HP separately against the existing DP128K winner; an HP improvement does not establish that HP is the fastest four-agent configuration.

If the candidate is useful and correct, run the actual four-Pi geometric shape gauntlet with the private 262144-context configuration. Archive a closed recording and verify all four live animations. Task work varies, so gauntlet wall time does not support a controlled speedup claim.
