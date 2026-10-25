# Passed HP GPU sampling audit v3

The v3 real-model audit passed the strict log gate with598valid numerical parity records,735valid dispatch records,9,032,535,040checked bytes and zero parser errors. All required coverage categories were observed. Six API phases completed12requests,3392output tokens and zero cache hits. This establishes the recorded real-row collection/replay audit; it does not establish a teacher quality pass, a statistical proof of the sampler, a throughput gain or a long-context capacity result.

The audit compares every gathered FP32 bit against CPU concatenation of the eight original local vocabulary slices, then replays the existing GPU sampler on that independently collected matrix with the same supplied tables and uniforms. It checks collection and addressing on actual hidden rows. Reusing the existing sampler is not an independent sampling implementation. Logs retain checks/fingerprints rather than original full tensors/uniforms, so this archive cannot independently recompute the numerical comparison from tensor data.

## Independent archive checks

An independent CPU-only pass parsed every target JSON line without stitching or repairs and rejected duplicate keys. It verified598contiguous audit IDs0..597,598distinct logit fingerprints, exact checked-scalar totals, and every logged rb/nd/draft/accepted/token bound. Each record has positive bounded matrix/table counts and true collection/replay checks. It independently found rb63, positive speculative count, acceptance, rejection, all drafts accepted, multiple tables, mixed greedy/sample tables, smaller matrices, top-p CPU fallback, positive-hesitation CPU fallback and pure greedy dispatch. The dispatch totals include598StepSampleHp,105Step and32StepGreedy. Source-side validation bounds uniforms before kernel launch; the logged hashes alone cannot revalidate the underlying uniform values.

The guard snapshot has52observations tied to exact PID25169, all stop=false, with minimum1643MiB free across all eight GPUs. Each of the six API phase results contains no recorded request error and zero cached-token delta. `independent-archive-verification.json` records these checks and their limits; `api-phase-summary.json` collects the raw existing phase summaries. The archived parser/protocol, original analysis, launch receipt, controller/client and raw requests/results/stats/log remain available for inspection.

## Source and repaired behavior

The executed source commit is2e0e16166709d11f4e98802eb74343cdba77a605 and server binary SHA25669d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363. The model settings are HP slots4,context262144,graphs4,chunk64,spec_k5,theta0.35,temperature0.7,top_p1, explicit HP GPU sampling and audit budget1024. Source, build and launch receipts agree on the binary identity; the parser's binary_identity_verified_here=false correctly distinguishes log parsing from that separate executable binding. The seven kernel sources/fatbin remain unchanged, fatbin SHA2564387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5.

The archive includes exact three-file source snapshots read from the recorded commit and checked against the source receipt, plus the v3 logging-only diff, v2 valid-struct initializer repair, CPU constructor tests and parent reviews. V1 had aborted on invalid zero-initialization of Rust enum-containing copy structs. V2 completed numerical checks but failed its strict gate on one JSON record interleaved with tracing. V3 preformats complete records and writes them under the stderr lock; no copy, sampler or numeric kernel change is bundled into that logging repair. The earlier failed attempts remain archived separately.

The saved v3 controller verifies source/binary/kernel binding before launching and retains audit evidence even on failures. Client/controller copies preserve the locally available code at archive time; their launch-time file hashes were not separately recorded. Model output streams may differ from CPU sampling because supplied random values and reductions differ; the deterministic claim here is parity for identical supplied tables/uniforms on identical real logits.

## Handoff and scope

After the audit, the parent stopped the idle serverPID25169 before starting a separately scheduled CLI teacher-quality process. `next-quality-launch-handoff.json` records stopped_server_pid25169 and the new quality launch; it is a launch/handoff receipt, not a quality outcome. This archive makes no claim about that job's later results or a final rental release. Audit-mode timings include debug overhead and must not be compared with matched performance runs.

The archive task used local files only and changed no engine source, STATE, server, rental or GPU resource. `manifest.json` lists exact SHA256 and byte size of every other file.
