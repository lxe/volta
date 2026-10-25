# Original HP four-agent gauntlet

This archive preserves the original fresh HP4/256K Pi coding gauntlet before any repair. Four agents wrote cube, pyramid, torus and sphere programs. All four original programs ran and produced three distinct observed Braille frames each. The original torus did not display its computed surface-normal shading, so the gauntlet does not establish that all four programs met every requested visual requirement.

The parent validated the closed recording and running animations separately from the source review. The full static geometry review is retained in original/hp-gauntlet-source-geometry-review.md and its JSON companion. It identifies absent torus shading, per-character torus depth artifacts, possible cube clipping and pyramid depth approximations. Any later repair is a separate attempt and must not replace these files or retroactively change the original outcome.

## Configuration and measured work

The fresh server used eight V100 GPUs, head parallelism, four total active slots, capacity 262,144, prefill chunk 64, five drafts, theta 0.35 and graph budget 4. GPU sampling was explicitly enabled; audit, STEPT, loop and graph profiling were disabled. Server PID 32809 and binary SHA 69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363 are bound by original/configuration.json. The server used the unchanged complete GGUF. No API workload preceded this gauntlet on that fresh server.

The controller ended with agents_idle after 193.085 seconds. Its raw summary records 49 requests, 20,653 generated tokens, a prompt-token counter delta of 44,745 and 448,485 cached tokens. Whole-controller generated tokens divided by wall time is 106.963 tokens/s. The summary's 175.263 tokens/s multi-session mean and 182.477 median use telemetry samples with at least two active sessions, with the mean measured between the first and last such sample; they are not whole-run throughput or per-agent rates. Variable agent work, cache reuse and different task trajectories prevent a controlled speed comparison with another gauntlet or the fixed/mixed A/B/B/A traces.

The retained memory guard has 182 PID-bound observations, a minimum observed free memory of 1747 MiB, and no stop decision. Polling does not establish an instantaneous memory minimum. Its raw records, statistics, initial state and complete server log are preserved for independent interpretation.

## Original evidence and integrity

original/ contains unchanged renderer sources and assigned tasks, complete agent session logs, controller source, configuration, raw summary/statistics, guard observations, captured panes and live frame samples, validator source and results, geometry review and the closed recording. Python bytecode caches were excluded. The original recording is stored as gauntlet.cast.gz; decompression restores exactly 74,514,267 bytes with SHA-256 45f8e858f0902104f89531fe39511c4bf4404614622ffd3e1c8b8dc9bbae5f1e. archive-validation.json records the CPU compression round trip and exact source-hash checks. The original source-file and validator hashes are retained in original/original-SHA256SUMS.json; its gauntlet.cast entry describes the decompressed recording. SHA256SUMS.json describes the public archive's actual files, excluding itself.

The local immutable snapshot was created at /home/lxe/volta-sixhour-20261024/hp-256k-gauntlet/original/ before repair. Archive preparation did not modify running renderers, execute them, access the remote rental or launch GPU work. Source inspection alone does not measure FPS or establish visible pixel quality.
