# HP GPU sampling audit v2 with an interleaved log

The repaired v2 candidate completed all six API validation phases and614valid numerical parity audits, covering8,796,728,320bytes of collected logits. Every required audit/dispatch coverage category was observed in other valid records. Nevertheless, the **strict audit-log gate failed** because one HP_SAMPLE_DISPATCH JSON record at line140 was interrupted by concurrent API tracing. This archive preserves that failure; it is not an overall correctness-gate pass or a candidate promotion.

The numerical records compare gathered FP32 bits against independent CPU concatenation of the original eight local slices, then compare sampled accepted-count/token pairs with the same kernel/tables/uniforms applied to that independently collected matrix. The614records cover2,199,182,080u32 values with614distinct logit fingerprints and matrix sizes2..64rows. They establish observed numerical parity on those records, not independent sampler statistical quality or a teacher quality result. This audit-mode run establishes no performance gain.

## Strict failure and API coverage

The parser found762dispatch records,761valid; all614parity audit records were valid. The malformed dispatch at line140 was parsed as an unterminated JSON string. The raw log visibly contains API tracing inserted inside the JSON string, followed by the remaining JSON on subsequent lines. No record was stitched, repaired, dropped from the original log or accepted through a weakened parser. `hp-gpu-audit2-analysis.json` remains statusFAIL, audit_log_gate_passed=false, errors_total1 and missing_required_coverage[]. Observed coverage elsewhere does not erase a malformed audit-channel record.

The completed phases were sampled prefill/decode, mixed greedy/sample tables, top-p fallback, positive-hesitation fallback, pure greedy, and staggered sampled prefill. Requests/results/stats and summaries are preserved in `screens/audit/`; `api-phase-summary.json` simply collects the existing phase summaries. The client exit code was0 and CLIENT_DONE exists. The strict analyzer exit code was1, so there is no HP_GPU_AUDIT2_PASSED marker. Required rb63, drafts/accept/reject, mixed/multiple tables, tails and fallback categories are supported by actual valid records rather than inferred from successful responses.

## Source, repair and launch binding

This attempt ran source commit48660962e1a80847e27b82a8dceca245564515af, server binary SHA256 f3939f58a70217130a11da15284083d9da22c23df03a8719c4c910f904b88227, PID24433. Model settings were HP four slots,262144context, graph budget4, chunk64, spec_k5, theta0.35, temperature0.7 and top_p1, with explicit HP GPU sampling and1024audit budget. The v2 repair replaces the invalid Rust zero-initialized enum-containing local/peer copy structs with valid complete literals and includes a CPU constructor regression. Its exact patch, local tests, source/build proofs and parent review are retained.

The failed-log SHA256 is6518f148686069310ea2342426b9eaf02af670471f24a98887c20390b9a0100a. Parser-source proof and the separate launch-binding receipt retain source/binary provenance. The analyzer itself reports binary_identity_verified_here=false because log parsing cannot verify a remote executable. The copied v2 controller and API client are the local versions available when archived; their launch-time hashes were not separately recorded. `copied-parser-source-binding.json` reports which copied parser/protocol files still exactly match the v2 proof, without silently relabeling a later copy as an earlier hash.

The planned v3 change repairs logging only and is a separate candidate/run. Nothing in this archive proves that v3 builds, passes audit, meets a quality gate or improves throughput. No numerical result was recomputed from a repaired v2 log.

## Resource and stop receipts

The PID-bound guard recorded54observations, all stop=false, with minimum1663MiB free across all eight GPUs. This exceeds the192MiB reserve but covers sampled observations only. The copied API idle snapshot records active=queued=0 before the parent stopped exactly PID24433. `screens/stop-postcheck.json` then records all eight GPUs at32495MiB free and0MiB used. This proves the recorded process cleanup snapshot, not rental release or later machine state. Evidence-copy receipts show successful log/guard retrieval.

This archive task used local files only and changed no engine source, live server, STATE or remote/GPU resource. `manifest.json` records SHA256 and byte sizes for every other archived file.
