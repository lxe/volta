# Head-parallel 256K screening

Eight V100-SXM2-32GB GPUs ran the same full GGUF and sampler binary as the earlier DP measurements, using head parallelism, four total active slots, context262144, graph budget4 and chunk64. See configuration.json for all flags and binary SHA. The teacher-quality check is archived separately in ../quality/.

The fixed four-client workload completed16580 prompt tokens and2048 output tokens with zero cache reuse at48.76 aggregate tokens/s. The mixed workload completed20677 prompt tokens and2048 output tokens at42.12 tokens/s. These are configuration screens; context reservation and parallel execution mode both differ from DP128K.

The subsequent cold4K-input/1024-output profile used unique nonces for each client count and repeat, and every client filled its output budget. Four clients measured75.49–76.02 aggregate whole-request tokens/s. Eight submitted clients measured78.71–79.21, with at most four active and four queued. One-client results varied42.76–62.88 whole-request tokens/s and58.39–97.52 decode tokens/s, so the best individual result does not establish a repeatable speedup.

Short-request results do not establish effective256K retrieval. The real250K retrieval probe was still running when these short-profile receipts were committed. Its separate actual result must be added before claiming that context is usable. Client-visible SSE content-event gaps can contain multiple tokens and are not pure per-token latency.
