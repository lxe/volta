# Four concurrent 250K HP retrieval requests

Four concurrent greedy requests each consumed 250,046 prompt tokens and recovered their own distinct verification key exactly on eight Tesla V100 32 GB GPUs. The strict analyzer passed every original gate: 1,000,184 total prompt tokens, 68 output tokens, zero cached tokens, four observed active sessions, zero observed queued sessions, and continuous sampled memory coverage. Each response stopped naturally after 17 output tokens within its 64-token allowance. The client wall time was 1,872.105775 seconds.

The server admitted all four sessions between 22:06:44 and 22:06:45 PDT on October 24, 2026 (05:06:44–05:06:45 UTC on October 25). The final server completion occurred at 22:37:54 PDT. The raw server events retain their UTC timestamps and request IDs in the analysis receipt. Small nonzero admission queue_ms values are preserved; the original no-queue gate concerns the sampled queued-session count, whose maximum was zero.

| Client input index | Prompt tokens | Output tokens | Client time to first token, seconds |
| --- | ---: | ---: | ---: |
| 0 | 250,046 | 17 | 1,871.937610 |
| 1 | 250,046 | 17 | 936.411809 |
| 2 | 250,046 | 17 | 1,403.895677 |
| 3 | 250,046 | 17 | 468.858532 |

The client returns futures in input order, and each result contains the expected distinct key. API results do not store server request IDs, so the analyzer validates raw server admissions and completions separately. It does not assign those IDs to client indices or infer a server-to-client mapping from completion order.

## Configuration and source binding

The run used PID 27988 with four HP slots, a 262,144-token context per slot, a shared 64-row chunk, four graphs, speculative maximum 5, and theta 0.35. Profiling, graph tracing, and sampling audit were disabled. The HP GPU sampling flag was enabled, but these temperature-zero requests use the unchanged StepGreedy path. This capacity result therefore supplies no measurement of the sampled GPU path's speedup.

The tested source commit was `2e0e16166709d11f4e98802eb74343cdba77a605`. The server binary SHA256 was `69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363`. The three tested engine source files are preserved under `tested-source/`; their hashes match both the build proof and the integrated shared-source binding. The complete GGUF contains 441,574,130,240 bytes and has SHA256 `9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c`. That model identity comes from the earlier verified full-file hydration receipt and the configured path, rather than a new full-file rehash for this probe.

## Memory coverage and interpretation

The refreshed raw guard file contains 1,749 observations. The analyzer selected rows 3 through 1,723 inclusive, starting with the last observation at or before first admission and ending with the first observation at or after final completion. Across those 1,721 observations, all eight GPU identities and PID 27988 were valid, no stop was recorded, minimum free memory was 1,753 MiB, and the maximum observation gap was 1.137309 seconds. The original reserve was 192 MiB and maximum allowed gap was 2 seconds. The entire refreshed file remains preserved and hashed; observations outside the selected capacity interval do not change its gate outcome. The original guard snapshot is also preserved without replacement.

Four active sessions establish admission and capacity for this controlled retrieval workload. The scheduler serializes prefill through one shared 64-row HP group, so four active slots do not imply four independent parallel 64-row prefills. Sampled guard readings cannot establish memory behavior between observations. Distinct-key retrieval does not establish general long-context quality, and naturally stopped short responses do not constitute a fixed-output throughput benchmark.

## Evidence layout

`screens/four-hp-250k/` preserves all actual requests, streamed-result records, statistics, summary, server log, configuration, and both guard snapshots. `server-runs/` and `memory-guards/` preserve launch and admission receipts. `receipts/four-hp-250k-analysis.json` contains the strict PASS, individual checks, raw admission/completion events, source bindings, and input hashes. The client, controller, server-control helper, guard helpers, analyzer, original conditional budget, and build/source proofs are included at their original relative paths. The budget describes pre-run predictions and remains unchanged; actual results are reported here and in the analysis receipt.

The archived analyzer is read-only with respect to servers and GPUs. To recompute locally, run `python analyze-four-hp-capacity.py --root . --out /tmp/four-hp-capacity-recheck.json` from this directory. Its original analysis input hashes can be checked independently of machine-specific absolute paths. `manifest.json` records every archived file's byte length and SHA256, excluding the manifest itself.
