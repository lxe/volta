# Graph trace analysis protocol

analyze-graph-trace.py validates the eight per-GPU JSONL files from the diagnostic binary and replays their observed key sequences through the proposed four-slot frequency policy. It uses Python's standard library, accesses no GPU or network, and changes neither engine source nor trace files. No actual graph trace was available when this analyzer was prepared; its current validation receipt is synthetic CPU evidence only.

Copy the completed diagnostic gpu0.jsonl through gpu7.jsonl into one local directory. Use a fresh output filename; the analyzer refuses to overwrite a report. For the controlled no-cache DPslots1 fixed/mixed prompts with actual API counts4145/8242, the final49/50-token prefill chunk exceeds9, so t1..6 identify generated steps in those specific traces. Enable this interpretation only with the explicit flag after checking actual prompt counts/configuration; generic traces retain the hint-only classification.

Example, after actual trace collection:

```sh
python3 /home/lxe/volta-sixhour-20261024/analyze-graph-trace.py /home/lxe/volta-sixhour-20261024/graph-traces/ACTUAL_RUN --controlled-4k8k-slots1 --out /home/lxe/volta-sixhour-20261024/receipts/ACTUAL_RUN-graph-analysis.json
```

Default coverage requires128 detailed observations per GPU. Validate exactly eight headers, version/label order,4096-step/4MiB caps, contiguous sequence numbers, completed summaries, counter totals, resident list/count consistency and valid t/bucket values. The four-slot configuration is required; observed resident counts above4 are reported because the current admission bug can exceed its nominal budget. Summary counters must match the visible prefix exactly until the4096 record cap and may only add observations afterward. A validation error exits unsuccessfully rather than reporting partial success.

The report gives per-GPU key counts, selected decisions, resident ranges, capture-path host time, repeated t1..6 budget-miss keys with at least20misses, and resident keys never replayed during that small-t interval. The stronger stale list includes only keys present at every small-t observation. Admission evidence requires the controlled prompt interpretation, at least25% small-t calls missing for budget, a key with at least20misses and at least one persistently stale resident key. Idle t0 GPU workers are reported separately and need not pass an active-decoder gate. Use API timestamps to verify workload phase; independent GPU step sequence numbers are not wall-clock synchronization points.

Optional --start-seq and --end-seq restrict observed statistics to a selected local sequence interval. Simulation always starts at the beginning of the available trace, preserving its actual warm-up/cache history; selected-interval simulated hit coverage is then calculated from that replay. Do not restart a simulator cold at the start of an already warm generated phase. Whole-prefix and selected-interval coverage gains are separately reported.

The simulator admits the first four keys on second encounter, then keeps a64-observation frequency window. A missing key needs frequency>=8 and at least4more observations than the least-frequent resident; victim ties use oldest last replay then key order. Replacements are separated by at least64steps and capped at8/GPU. Captures execute a step but are counted separately from graph replays. The simulator tracks opportunities blocked by the replacement cap. It includes idle t0 and gives no inherent priority to token count.

Simulation is a CPU cache-coverage model only. It assumes captured execution remains numerically valid, cannot predict CUDA memory/instantiation failures, and does not model capture latency, peer synchronization or throughput. Any observed forced/full/failure decision invalidates that GPU's direct policy comparison, though its raw statistics remain available. The screening gate needs at least15percentage-point whole-prefix hit-coverage gain, at most8replacements and no opportunities blocked by that cap. A passing result authorizes no automatic source change or GPU trial. The proposal's resource, correctness, paired timing and deadline gates still apply.

Detailed simulation is limited to the recorded first4096steps. A reached cap or summaries extending beyond detail is explicitly reported. Default SIGTERM may lose up to63pending tail observations; the final completed summary bounds what is known. A short/truncated interval cannot establish absence of later misses. Preserve source/binary hashes, trace directory and actual workload receipts with every actual report.

Synthetic validation:

```sh
python3 /home/lxe/volta-sixhour-20261024/analyze-graph-trace.py --self-test
```

The fixture creates eight valid256-step streams with four cold prefill graphs and two sustained missed small-t keys. It verifies both admission/simulation gates and two hysteresis replacements, then checks duplicate sequence and missing-GPU rejection. Its receipt receipts/graph-analyzer-synthetic-test.json says SYNTHETIC_CPU_FIXTURE_ONLY and actual_trace_analyzed=false. These generated traces establish analyzer behavior, not a measured graph-cache problem or speedup on this rental.

## Optional version2 host timing

The analyzer accepts version1 and version2 traces. Version2 requires the header wall epoch and nonnegative per-step elapsed/enqueue fields; elapsed times must be monotonic and capture-path duration cannot exceed its enclosing host enqueue duration. Version1 files continue to work with timing reported absent.

When present, the report includes selected-interval first/last host record epoch estimates and host_enqueue_backpressure statistics (count, mean, median, p95, minimum, maximum in milliseconds), both overall and by graph decision, plus a t1..6 subset. These are unsynchronized host enqueue/backpressure measurements. They provide context for comparing repeated eager misses against graph hits, but differences also depend on shape, context, capture and actual GPU backpressure. They do not measure GPU kernel execution or establish a causal speedup. Use matching keys and independent API wall-clock receipts for attribution.

The updated synthetic analyzer fixture mixes version1 and version2 GPU files, verifies a known0.123ms enqueue median and epoch alignment, and retains all cache-policy/cap tests. Its receipt is receipts/graph-analyzer-synthetic-test-v2.json; no actual graph trace has been analyzed. The separate Rust-generated CPU version2 writer fixture also parses all4096detailed records and checks epoch presence, elapsed monotonicity and enqueue fields. Version2 CUDA-feature release compilation completed successfully02:42:36UTC in the approved window; no diagnostic GPU/model launch occurred.

The controller/protocol now consistently select the v2 binary under /root/volta-graph-diag-v2/target-v2. Prior actual smoke/warm prompt counts were39/563 (final chunks39/51), and fixed/mixed counts4145/8242 (final chunks49/50). Thus all those recorded final prefill chunks exceeded9; confirm the later diagnostic's actual usage before classifying its entire controlled prefix. The prepared invocation is in graph-diag-protocol.md and is documentation only, with no automatic launch.
