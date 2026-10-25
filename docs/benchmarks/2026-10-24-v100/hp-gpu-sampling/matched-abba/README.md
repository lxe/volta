# Matched head-parallel GPU sampling throughput

The opt-in HP GPU sampler passed two swapped timing comparisons with all four phase gains above the 5% threshold. Fixed-work throughput improved by 53.92% in A1/B1 and 51.78% in B2/A2. Mixed-work throughput improved by 47.94% and 47.82%. Each phase completed four requests and exactly 2,048 output tokens, with identical request JSON across arms and zero cache reuse.

| Arm, in execution order | Fixed tokens/s | Mixed tokens/s | Minimum observed free GPU memory |
|---|---:|---:|---:|
| CPU A1 | 48.983879 | 41.999733 | 1,745 MiB |
| GPU B1 | 75.397173 | 62.132762 | 1,707 MiB |
| GPU B2 | 75.100050 | 62.291473 | 1,711 MiB |
| CPU A2 | 49.478208 | 42.139802 | 1,741 MiB |

These are whole-request rates including prefill for the recorded workload. The common configuration was HP four slots, context 262144, graph budget 4, chunk 64, spec_k 5, theta 0.35, temperature 0.7 and top_p 1. Audit and profiling were disabled. The CPU and GPU binaries share the exact frozen CUDA fatbin; source/build and launch identities are preserved. A1 was an earlier fresh baseline server, followed later by fresh B1, B2 and A2 servers in that order. The archive retains the actual timestamps.

The strict CPU-only analyzer verifies byte-identical fixed and mixed request files, recomputed client output and wall-time totals, actual prompt counts, server counter agreement, idle boundaries, cache reuse, normalized settings and environment, binary/build bindings, and PID-bound guard admission. It requires all eight GPU readings, no guard stop or reserve breach, monotonic observations with gaps at most two seconds, and guard observations bracketing each complete phase. The recorded maximum gaps were about 1.14 seconds. This is sampled guard coverage, not a guarantee about unsampled intervals.

`hp-gpu-abba-attribution.md` explains the comparison and uncertainty. The measured gains exceed the earlier CPU-distribution-only prediction because the candidate also replaces GPU-to-host downloads and host vocabulary concatenation; the earlier 27–28% figure was not a valid ceiling for the whole change. No matched component profile isolates each contribution. CPU and GPU random-number consumption and floating-point reductions differ, so equal seeds do not imply identical text, draft histories or internal work. The comparison fixes external prompts and output budgets, rather than claiming identical stochastic trajectories.

The actual v3 collection/replay audit is archived separately. This directory's marker certifies only the matched timing gate. It is not a promotion marker, a teacher-quality outcome, a long-context capacity result or a real agent-gauntlet speedup. Later quality and gauntlet results must be reported separately. In particular, an HP improvement does not prove that HP exceeds the primary DP configuration.

`screens/`, `server-runs/` and `memory-guards/` retain every arm, including smoke and warmups, raw requests/results/stats, full server logs, configurations and bounded guard snapshots. Controller sources and logs preserve the procedure; helper scripts are historical evidence and must not be run against a live machine without current admission checks. The timing analyzer performs local reads only and can be rerun against this directory. It invalidates its own timing marker on failure, and writes no promotion marker. The copied original controller/client helpers were not all separately hashed at launch; the manifest binds their archived versions rather than inventing launch-time provenance.

The archive task issued no GPU, remote, server-control or rental operation and changed no engine source or STATE. `manifest.json` records SHA256 and byte size for every other archived file.
