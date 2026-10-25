# Four concurrent 128K retrievals

Four DP sessions each read 126045 actual prompt tokens with zero cache reuse and returned their own distinct initial key exactly. All four naturally stopped after 17 output tokens. The server observed four active sessions and no queue. First-token times were 296.89–297.47s; complete request wall time was 297.69s.

This used the existing V2 diagnostic server with context 131072, one DP slot per GPU, graph budget 4, STEPT synchronization and bounded CPU graph recording. It was a capacity and session-isolation check, not a throughput comparison or general long-context quality evaluation. Earlier short requests had already warmed the server, but these four inputs had no cached prefix.

The memory guard exited before its first observation because this diagnostic binary was absent from the guard's executable allowlist. Its failed startup log is retained. These successful requests establish retrieval and concurrency; they do not establish a continuously measured free-memory reserve. The guard allowlist and startup verification were repaired afterward, requiring a live process and first valid observation before future GPU work.

Raw requests, outputs, initial/final counters, sampled active-session stats, server log, configuration and exact client source are included. manifest.json binds every archived file.
