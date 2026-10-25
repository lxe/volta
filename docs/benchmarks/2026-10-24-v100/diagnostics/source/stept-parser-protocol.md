# VOLTA_STEPT finish-cost screen

parse-volta-stept.py reads the current engine's existing stderr format: `GPU0 step at 12.34: begin 0.10 ms, layers 40.00, finish 2.00`. It reports each GPU's begin/layers/finish sums, means, medians, p95 and finish share. It does not start a server, open CUDA or alter any source/input file. No actual profile log was supplied when the parser was prepared; its receipt is synthetic CPU evidence.

The existing line contains no t or step sequence. A standalone log therefore produces unknown_t statistics; token count must not be inferred from latency. The parser can classify t1..6, t32 and t64 only by joining a trace from the SAME fresh v2 process that enabled both VOLTA_STEPT and VOLTA_GRAPH_TRACE. An earlier normal v2 trace cannot label a later profiling run, even with nominally matching prompts. The current planned diag-v2-001 run remains unchanged.

For a later parent-approved profile, the existing controller already accepts both --graph-trace and --profile on engine graph-diag; no rebuild is needed. Run at most one fixed/mixed controlled four-client diagnostic after current GPU work ends. Copy its entire server stderr, configuration.json, and the eight trace files. Retain the original configured stderr filename, or archive it as server.log inside the matching run trace directory. Keep the local trace directory named exactly as the configured run name. The parser checks the v2 binary SHA, enabled flags, trace-output provenance, header launch window, complete per-GPU trace prefix and summary/caps before labeling.

The join filters t0 trace observations because run_step returns before STEPT logging on idle GPUs. Each GPU's positive-t trace records map in order to its own STEPT lines. This requires a full stderr prefix copied from process start; a manually tail-truncated log is invalid. Buffered trace tails/records beyond4096 remain unclassified and are reported. The parser verifies nondecreasing STEPT offsets and requires enough stderr lines for every visible positive-t trace record. The first offsets can be0 because the existing global origin initializes after the first step; do not treat that value as exact first-step epoch.

Example after an actual combined profile, with fresh output filename:

```sh
python3 /home/lxe/volta-sixhour-20261024/parse-volta-stept.py /home/lxe/volta-sixhour-20261024/receipts/server-dp-finish-profile-001.log --trace-dir /home/lxe/volta-sixhour-20261024/graph-traces/dp-finish-profile-001 --configuration /home/lxe/volta-sixhour-20261024/server-runs/dp-finish-profile-001/configuration.json --out /home/lxe/volta-sixhour-20261024/receipts/dp-finish-profile-001-analysis.json
```

Use --start-seq/--end-seq to select one controlled trace interval when matching wall-time receipts. Simulation/admission analysis still retains the entire prefix in its separate analyzer. Optional --wall-seconds must refer to this exact selected profile interval, not model load, a separate nonsynchronized benchmark or an unrelated request duration. The parser divides a conservative finish budget by that caller-supplied denominator and labels the limitation.

For each global local-step sequence it takes the maximum finish duration among candidate GPUs with t32..64, then sums those maxima. This includes final prompt chunks49/50/51 as well as full32/64 chunks. It avoids adding concurrent GPU finish durations as serial time. Under verified DPslots1/spec5, these sizes are prefill candidates; they are only hints in an arbitrary workload. The result is an upper bound on savings from the proposed narrow t>=32 head selection, with unknown/omitted steps excluded. If coverage is incomplete or eligible final-tail sizes were omitted, it cannot support a whole-trace rejection by itself.

VOLTA_STEPT synchronizes after begin and after layers, and finish's result download waits for its GPU work. Finish therefore includes HC pre/RMS, vocabulary head, sampler/argmax and download. It is an upper bound on removable head time; it is not isolated head timing. The v2 unsynchronized enqueue counter ends before finish and cannot substitute for this boundary. Summing per-GPU phase totals double-counts overlap, and profile synchronization changes serving behavior, so none of these totals is a primary TPS claim.

If even the complete eligible finish budget is below5% of the matching trace wall time, the proposed head-only change cannot meet the primary5% gate and should be dropped. If the finish budget is large enough, measure the head separately before implementation: retain the baseline64-row final logits, project the same real normalized final row with t1 into output row0, and compare both before sampling. The generic chunk1 teacher does not exercise this narrow StepSample dispatch. Keep graph policy, row selection, numerical comparison and primary timing as distinct gates.

Synthetic validation covers raw format parsing, per-GPU fractions, refusing token classification without provenance, same-process joining with idle t0 omissions, t64 versus small-t grouping, and the per-step-max finish upper bound. receipts/stept-parser-synthetic-test.json records this as SYNTHETIC_CPU_FIXTURE_ONLY. No engine implementation, compilation or GPU/model launch occurred for this parser.

The parent scheduled diag-v2-stept-001 as a separate combined profile after diag-v2-001 snapshots are copied. Its planned same-process log is graph-traces/diag-v2-stept-001/server.log; the parser accepts that standard archive name only when it is beside the matching trace files. Later manual analysis is:

```sh
python3 /home/lxe/volta-sixhour-20261024/parse-volta-stept.py /home/lxe/volta-sixhour-20261024/graph-traces/diag-v2-stept-001/server.log --trace-dir /home/lxe/volta-sixhour-20261024/graph-traces/diag-v2-stept-001 --configuration /home/lxe/volta-sixhour-20261024/server-runs/diag-v2-stept-001/configuration.json --out /home/lxe/volta-sixhour-20261024/receipts/diag-v2-stept-001-analysis.json
```

This parser command has not been run on actual data yet. It does not launch GPU work.
