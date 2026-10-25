# Speculative decoding hypothesis and timing budget

These files record a CPU-only review that proposed one bounded comparison of theta0.2 against theta0.35 on DP128K/four-client serving. The review's projected5–15%decode improvement and3–9%whole-request improvement are conditional predictions, not measured gains. No parameter experiment or new GPU workload was run by the review author.

The measured inputs are the existing sampler-c1/four-client telemetry and the separately instrumented DP graph/STEPT diagnostic. The review distinguishes the scheduler's per-session exponentially weighted acceptance from the summary's rolling kept/proposed ratio. The missing position-conditioned prefix probabilities prevent a guaranteed throughput prediction.

The JSON contains per-local-T and collective-T regrouping of the existing diagnostic timings, with sample counts, minima, medians, means and maxima. These observations are correlated across GPUs and are not isolated shape perturbations. The analyzer retains its original absolute task-workspace paths and depends on that original graph-trace/server-log input; it is archived as the exact calculation source. The original diagnostic receipts and classification rules are described in the review and archived elsewhere under diagnostics/graph. Later actual theta results, if any, must be reported separately and must not relabel this prediction as a measured gain.

`manifest.json` gives exact SHA256 and byte size of every other file here.
