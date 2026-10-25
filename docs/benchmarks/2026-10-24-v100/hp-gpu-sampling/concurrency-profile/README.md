# HP GPU sampling concurrency profile

This profile measured one, four, eight, eight, four and one submitted clients on an eight-GPU V100 32 GB HP server with four active slots. All 26 requests completed their full 1,024-token output budgets from 4,145 actual prompt tokens each, with zero cache reuse. The strict descriptive analyzer passed all six cases and the memory guard. The ranges below contain both actual repeats of each client count.

| Submitted clients | Maximum observed active sessions | Maximum observed queued sessions | Whole-request aggregate tokens/s | Median per-client decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 69.995–72.205 | 108.648–109.118 |
| 4 | 4 | 0 | 112.303–113.434 | 49.101–50.306 |
| 8 | 4 | 4 | 119.266–120.137 | 42.667–42.871 |

Whole-request aggregate throughput divides total output tokens by the wall interval from the earliest client start to the final client completion. It includes prefill and any queued waiting. Per-client decode throughput uses `(1,024 - 1) / (client wall time - time to first token)`; the final column reports the range of case medians. Eight submissions used four active slots and four queued requests, so these measurements do not establish eight simultaneously active sessions.

The profile ran at approximately 22:38–22:42 PDT on October 24, 2026 (05:38–05:42 UTC on October 25). It used the same PID 27988 that had just completed four 250K retrieval requests. Distinct case/client nonces and zero client/server cache counters establish cold requests, while the process retained its prior KV, graph and runtime history. This is a descriptive profile rather than a controlled comparison against DP or the earlier CPU sampler. Sampled outputs can also follow different RNG and speculative trajectories.

## Configuration and identity

The server used four HP slots, a 262,144-token context per slot, a 64-row shared chunk, four graphs, speculative maximum 5, theta 0.35, temperature 0.7 and top-p 1. Profiling, graph tracing and sampling audit were disabled, and HP GPU sampling was enabled. The requests used the same 4K fixture content with distinct nonces and a full 1,024-token output allowance. The analyzer checks the exact request structure, seeds, fixture identity beyond intended nonce/index changes, all output hashes, raw server events and continuous API counter changes.

The source commit was `2e0e16166709d11f4e98802eb74343cdba77a605`, and the server binary SHA256 was `69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363`. The three tested engine source files under `tested-source/` match the build receipt and integrated shared-source hashes. The model contains 441,574,130,240 bytes with SHA256 `9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c`. Model identity uses the earlier verified full-file hydration receipt and configured path; the profile did not rehash the full model again.

## Memory and timing limits

The refreshed raw guard contains 269 observations. The analyzer selected rows 1 through 224, from the last observation at or before the first profile admission through the first observation at or after final completion. All 224 selected observations matched PID 27988 and all eight GPU identities, recorded no stop, and retained at least 1,699 MiB free. The maximum gap was 1.134854 seconds, within the original 2-second limit and 192 MiB reserve rule. Both the original and refreshed guard snapshots remain preserved, and the whole refreshed input is hashed. Sampled observations cannot establish memory behavior between readings.

Client results remain in submission order. API records do not retain server request IDs, so the analyzer checks each case's raw admission/completion ID sets separately without inventing a client-to-ID mapping. Per-client time to first token and SSE content event timestamps are preserved in the analysis and raw results. SSE gaps describe client-observed content chunks, which may contain multiple model tokens; they do not measure model inter-token latency. This profile supplies no general quality claim or controlled cross-mode speedup.

## Evidence and reproduction

`screens/hp-gpu-concurrency/` contains all six cases' requests, results, initial/final statistics, sampled session statistics, summaries and client logs, plus the complete server log, batch configuration, completion marker, prior-work note and both guard snapshots. The source harness, controller, helpers, original pre-run budget, source/build/model receipts and strict analyzer are preserved at their original relative paths. `screens/four-hp-250k/final-stats.json` supplies the prior-work counter boundary used by the analyzer. The original budget remains a historical planning document; actual results are reported in `receipts/hp-gpu-concurrency-analysis.json`.

From this directory, `python analyze-hp-concurrency.py --root . --out /tmp/hp-concurrency-recheck.json` recomputes the CPU-only checks without contacting a server or writing a promotion marker. The archived analyzer reproduced PASS before commit. `manifest.json` records all other archive files' byte lengths and SHA256 hashes.
