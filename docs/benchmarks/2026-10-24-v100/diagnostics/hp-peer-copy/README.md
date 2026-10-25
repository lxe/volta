# HP vocabulary peer-copy diagnostic

On the Minnesota Vast.ai machine 27801 (instance 57584845), the standalone eight-GPU pitched-copy diagnostic verified all 827,392,000 destination bytes across 80 collections. It collected eight FP32 vocabulary slices into GPU0 using seven cross-context `cuMemcpy3DPeerAsync` calls and one local `cuMemcpy3DAsync` per collection. No CUDA kernels, model pointers, engine edits, inference requests, or device resets were used.

## Measured timing distributions

Each shape has 20 repeats with fresh GPU/row/column/repeat-specific bytes. Source uploads, destination poisoning and the byte-for-byte host verification are outside the event interval. CUDA events bracket the eight collection copies. Host completion includes enqueue and waiting for the stop event; it is not full-process wall time. All timing values below are milliseconds.

| Rows | Collected bytes/repeat | Event minimum | Event median | Event p95 | Event maximum | Host enqueue median | Host completion median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 517,120 | 0.071776 | 0.076384 | 0.084413 | 0.091648 | 0.030478 | 0.086043 |
| 3 | 1,551,360 | 0.121152 | 0.123120 | 0.140909 | 0.405632 | 0.030235 | 0.133795 |
| 12 | 6,205,440 | 0.376128 | 0.384304 | 0.387082 | 0.392736 | 0.031487 | 0.395557 |
| 64 | 33,095,680 | 1.835808 | 1.842944 | 1.848576 | 1.848576 | 0.053323 | 1.858189 |

`distribution-summary.json` contains every raw timing value and min, quartiles, median, p95, max, mean and sample standard deviation for event, host enqueue and host completion timing. `run/stdout.jsonl` preserves the original samples, memory-admission observations and success record. Percentiles use linear interpolation at index `(n-1)*p`. The t=3 maximum of 0.405632ms is retained; no outliers were removed. The wrapper reports 5.611165 seconds of wall time covering its benchmark launch and subsequent checks.

The artifact was collected on October 24, 2026 Pacific time; the raw stdout receipt's saved mtime is October 25, 2026 03:47:49 UTC (October 24, 20:47:49 PDT). This is a receipt-file timestamp, not a separately instrumented device start timestamp.

## Resource and process checks

The reviewed budget allocates 66,191,360 bytes of GPU source/destination payload, with CUDA-context overhead additional, requires at least 1GiB free on all eight GPUs before context creation, and checks a 512MiB reserve after each resource-admission step. The program has a 55-second cooperative deadline and the wrapper applies an external 60-second hard timeout. Owned resources use RAII cleanup; the raw completion record states release on return.

The parent wrapper checked `/proc/19770/exe` against `/root/volta-loop-prof-v3/target-v3/release/volta-server` before and after the test, and both shell checks succeeded. The model remained idle, with API request count 17, generated tokens 4,640 and prompt tokens 41,800 unchanged in the saved before/after stats. All eight GPUs had 1,745MiB free before launch and recovered to exactly 1,745MiB afterward. The postcheck also found no remaining `hp_pitched_copy` process. These are the recorded executable/PID and resource checks; a process-start-time identity was not separately archived.

## Scope and reproduction evidence

This is a measured transfer feasibility result. It does not establish a model throughput gain, sampler correctness, graph compatibility, engine integration, or performance while inference is active. The budget's comparison with an observed approximately 17ms CPU distribution pass is motivation for the diagnostic, not a paired end-to-end measurement. No tokens-per-second gain is booked.

The executed binary SHA256 is `5f30685eba565b2a33657731066c88dac7d52e8b42026f1d1fe38a0e3d0a9c08`; the reviewed C++ source SHA256 is `e1d427e3b48c754274e9a5907702c3b72883e843cb3f6610e3475ba815a491fa`. `hp-copy-compile-proof.json` records the original CPU-only build stage and flags; its `runtime_tested:false` describes that earlier stage. The later GPU execution is evidenced by `run/result.json` and the raw samples. The binary is identified by SHA rather than committed as a generated executable.

`run-hp-copy.py` is the original parent launch wrapper with its historical paths and PID. It is archived as evidence, not as a script to rerun blindly. The source and budget were reread during this CPU-only archive task. No remote command, GPU operation or additional benchmark was run to prepare the archive. `manifest.json` records SHA256 and sizes of every other archived file.
