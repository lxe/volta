# GPU sampling during prefill

The server can sample the final logits row of a prefill chunk directly on the GPU. Previously the eligibility condition excluded chunks longer than nine tokens, forcing all participating sessions through the CPU logits-download path whenever one session was prefilling. The change allows longer prefill chunks while preserving the existing limit of eight speculative drafts for decoding. It retains the top-p, hesitation and head-parallel exclusions. Weights, CUDA kernels and sampling parameters are unchanged.

Four fresh-server arms ran in A/B/B/A order on the same eight V100-SXM2-32GB rental. A is frozen cc54d975; B changes only the Rust eligibility condition. Both binaries embed the exact same sm70 fatbin, SHA2564387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5. Configuration and executable hashes accompany every arm. Each arm received an identical32-token API smoke, two four-client warmups, then fixed and mixed traces. Pair1 and pair2 use separate prompt nonces; each A/B pair uses identical prompts, seeds and sampling parameters.

| Order pair | Fixed: A → B, aggregate tokens/s | Mixed: A → B, aggregate tokens/s |
|---|---|---|
| A1 then B1 |112.24 →126.03 (+12.29%)|63.28 →74.48 (+17.70%)|
| B2 then A2 |116.90 →128.13 (+9.60%)|63.35 →73.36 (+15.80%)|

Each trace generated exactly2048tokens (512 per client), with no cached tokens. Fixed traces used16580prompt tokens total; mixed traces used20677. The mixed trace admitted an8K prefiller after three4K requests had each emitted a token. Aggregate rate divides actual generated tokens by first request start through last request completion; it includes prefill and excludes monitor polling overhead. The pure agentic baseline remains a separate variable-work result.

Sampled texts differ across arms and were coherent in inspected excerpts. CPU and GPU floating-point reductions differ; speculative GPU sampling also preallocates random uniforms differently from the CPU fallback when requests mix prefill and decoding. Equal seeds do not imply identical text across these paths. Equal token work supports the timing comparison, while the independent sampler boundary check and real serving checks support correctness. This experiment is not a new broad statistical sampling-quality evaluation.

Before the patch, the exact existing vspec_sample kernel passed64 controlled table cases including last-row indices31,32and63 with zero drafts and mixed greedy/sampled entries. The full GGUF's frozen teacher evaluation covered8184positions: KL0.126695, top1agreement84.3231%, perplexity8.9423. A real candidate four-agent tool/animation run and longer-context/concurrency screens remain pending at this checkpoint; no CUDA graph-cache policy change is included.

The candidate server SHA256 is b9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4; frozen SHA2569c82a8e15e96855badeb4128a7af7e01ab28d015fa575c717a36fc156513d1a9. Both use context131072,DPslots1,chunk64,spec-k5,temperature0.7,top-p1,graphbudget4. The2.23/hour rental retains its independent six-hour release deadline.
