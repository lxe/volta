# HP GPU sampling final quality evidence

This directory preserves the initial V3 teacher evaluation and the predeclared flag 0 → flag 1 pair. All three evaluations passed the original DP-relative bounds: PPL ≤ 9.0423, top-256 KL ≤ 0.131695, and top-1 agreement ≥ 0.838231. Each evaluated 8,184 teacher positions using eight sequences of length 1,024 and chunk size 1.

| V3 evaluation | HP GPU sampling flag | PPL | Top-256 KL | Top-1 agreement | Evaluator wall seconds |
| --- | --- | --- | --- | --- | --- |
| Initial | 1 | 8.9425 | 0.131653 | 0.838710 | 50.2 |
| Paired first arm | 0 | 8.9605 | 0.130730 | 0.839198 | 49.5 |
| Paired second arm | 1 | 8.9402 | 0.129370 | 0.842131 | 51.3 |

The initial result remains part of the decision. The pair ran once, in its declared order, without retries or selection of the best result. Its complete process duration was 431.504 seconds, exceeding the 420-second plan by 11.504 seconds. The worker recorded BUDGET_EXCEEDED and continued the already owned process. Evaluator wall time in the table excludes model loading and process setup.

## Files and provenance

`teacher-runs/hp-gpu-quality3/` contains the original initial-run configuration, metrics, raw teacher log, launch stderr and gate marker. `teacher-runs/hp-v3-flag-pair/` contains the paired controller records and the complete copied remote directory, including the actual executed worker, specification, raw stdout/stderr, per-arm configuration and results, memory observations, final state and budget alert. The top-level runner and worker are the reviewed local sources; the actual remote worker has the same SHA-256 as the reviewed worker.

`archive-validation.json` records CPU checks of both finite metric sets, the original bounds, equal input hashes and model metadata, the sole VOLTA environment difference, process ownership, successful exits and released GPU allocations. Both arms observed all eight GPUs throughout polling. Their minimum observed free memory was 3,601 MiB and 3,569 MiB, respectively, with maximum observation gaps of 2.139 and 2.198 seconds. These observations do not establish an instantaneous memory minimum.

The pair hashed heldout.i32, teacher_idx.i32, teacher_lp.f32 and teacher_kl.json before each arm and obtained identical hashes. It checked the cached full-model hash receipt and stable file metadata rather than repeating the approximately 17-minute full-model hash. Raw results retain those hashes and metadata. Both arms used the same V3 CLI SHA-256, bcfe792f228ba7216237c13e5d582dd7e4a34e0872ddd4c293c77d10a237cd67, and the unchanged CUDA fatbin bound by `hp-sampling-v3-build-proof.json`.

`receipts/hp-shared-source-binding.json` binds the three integrated engine files to the measured V3 source bytes. The integrated final source commit is 78d2dc9; the binding was recorded at shared HEAD 5d6734c678b3880301d4d70715e2415befe3045b. Historical raw proofs identify the original isolated source commit 2e0e16166709d11f4e98802eb74343cdba77a605. File hashes establish equality across those different Git histories.

The copied variation review, source review and gain-attribution review are historical originals. Their predictions and pending conditions describe the time they were written. The preserved promotion marker binds the strict real-row audit, swapped timing, initial teacher and paired teacher gates; its scope explicitly leaves fresh Pi and four-long capacity validation pending. This archive does not update those later gates.

## Interpretation

The generic teacher evaluator uses Engine::step, so these quality evaluations do not directly exercise StepSampleHp. The separate real-row audit compares collected logits bit for bit and replays the existing sampler with identical supplied tables and uniforms. Neither check proves a quality distribution for all future prompts. Floating-point atomic reductions provide a possible source of repeat variation, but the available results do not establish its cause or prove an allocation effect. All three measured quality results are retained.

SHA256SUMS.json covers every archived file except itself. Archive creation and validation performed CPU and filesystem work only; they did not build or launch a GPU process.
