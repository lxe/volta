# Failed HP GPU sampling audit v1

The first hardware validation attempt failed on October24,2026 at21:18:23PDT (October25,04:18:23UTC), during the first64-row prefill. The server aborted before emitting output or completing any parity audit. This attempt establishes no collected-logit correctness, sampler correctness, quality-gate pass or throughput gain. All required parity/fallback coverage remains unobserved.

The server log records one StepSampleHp dispatch followed by a Rust non-unwinding panic: `attempted to zero-initialize type cudarc::driver::sys::CUDA_MEMCPY3D_st, which is invalid`. The Rust binding's CUmemorytype enum has no valid zero discriminant. Constructing the whole struct with std::mem::zeroed() is invalid even if the code intends to assign valid enum fields immediately afterward. The panic happens before that assignment or a useful copy/audit can complete. The analogous peer-copy struct used the same construction pattern and also requires a valid initializer. The proposed repair must use complete valid struct literals and a CPU constructor regression check; a repaired version is a separate unvalidated candidate.

Both source reviews missed this Rust constructor-validity bug. The earlier C++ standalone peer-copy benchmark also did not catch it: zero-initialized C++ CUDA structs do not exercise Rust's enum validity checks. The archived v1 build console contains no invalid_value warning. Compilation, unchanged kernel hashes, the source reviews and the C++ byte-copy success were therefore insufficient proof at this boundary. The candid correction applies to the copied independent-review report's earlier “no blocking bug identified” conclusion; that report is retained as historical evidence and must not be treated as current clearance for v1.

## Artifact and failure binding

The v1 source commit recorded by the parent is acf798786b1962f5ea45ecbebc6cf114f2126b81; server binary SHA256 is63f597b080dee291d0f146d88d6cef70c075f546d82dcfec192880c13c7d6820. Model context was262144, HP slots4, graphs4, spec_k5, theta0.35, temperature0.7, top_p1, with explicit HP GPU sampling enabled and audit budget1024. Source, archive, patch, binary and unchanged fatbin hashes are retained in the source/build receipts. The failed log's SHA256 isfcaaecefc4052114ec13719c93cb6f9df618fac009de165d3db9621260cf1c43.

`hp-gpu-audit1-failed-analysis.json` reports one valid dispatch, zero parity audits, zero checked scalar/byte count and missing coverage for rb63, speculative drafts/accept/reject, multiple/mixed tables, tails, both CPU fallbacks and pure greedy dispatch. Its binary_identity_verified_here=false correctly limits the parser: binary provenance comes from the separate source/build/launch receipts rather than from parsing the log itself. Do not infer audit coverage from the submitted556-token request or from the dispatch's64rows.

The controller failed with RemoteDisconnected after the process aborted. The failed attempt has no ARM_DONE, completed result, normal final stats or performance summary; those absences are preserved. `screens/` contains the submitted request/initial stats, full server log, process postcheck and exact saved client/controller sources. `server-run/` contains launch/readiness receipts. `memory-guard/` contains the bounded guard snapshot through failure. Review, plan and validation protocol files identify what had been intended, rather than completed gates.

## Cleanup evidence

The postfailure snapshot records no known live servers and all eight GPUs at32495MiB free with0MiB used, showing that this aborted process's GPU allocations were reclaimed. This is a postfailure snapshot, not proof of later rental release. The model files remain on disk for the parent's separately scheduled work. This archive task issued no GPU, remote, model-control or rental command and did not change live source or STATE.

`manifest.json` gives exact SHA256 and sizes of every other archived file. The failed artifacts remain separate from any future v2 repair and audit result.
