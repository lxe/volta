# dsx-core

What both runners share:

| | |
|---|---|
| `src/model.rs` | config (`config.json` or GGUF metadata), weight handles, the CPU reference forward with model.py's numerics |
| `src/dspark.rs` | the DSpark draft stack, CPU reference |
| `src/gguf.rs`, `src/safetensors.rs` | mmap readers for GGUF splits and sharded HF checkpoints |
| `src/formats.rs`, `src/numerics.rs` | block decoders (Q8_0, Q2_0, PTQ1_0, Q3_K, Q6_K, FP8 32x32, FP4), fake quant, RoPE + YaRN |
| `src/server/` | the OpenAI / Anthropic compatible server over any `server::Engine` (chat template, tool calls, streaming, sampling), the web UI |
| `webui/` | llama.cpp's web UI, vendored with its license |

`cargo test --release -p dsx-core` runs the server tests against a mock engine.
