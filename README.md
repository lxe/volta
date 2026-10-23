<h1 align="center">volta</h1>

<p align="center"><b>DeepSeek-V4.1-Flash-1017 (552 B + 196 B Engram) on one used 8x V100 server</b><br>
2,400-3,000 tokens/s across hundreds of sessions · DSpark speculative decoding · OpenAI- and Anthropic-compatible
API · thinking levels, streaming, prefix reuse</p>

<p align="center">
<a href="https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-Volta"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20model-Volta-yellow" alt="Model: Volta"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
<a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/Rust-stable-orange?logo=rust" alt="Rust"></a>
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.9-76B900?logo=nvidia" alt="CUDA 12.9"></a>
</p>

An 8x V100 machine (the DGX-1 NVLink layout, 256 GB of HBM) is old and cheap, and no serving engine runs
**[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (DeepSeek, MIT license) on it:
Volta has no FP8, FP4 or bf16 math, and the official checkpoint is larger than its memory. Together the eight cards
still read 6.9 TB/s. volta is an engine in Rust and CUDA, written for this one model on this one class of machine,
that turns that bandwidth into throughput for many sessions at once.

- **Fast:** 2,400 tokens/s across 256 sessions and 3,070 across 512; 24 tokens/s per session with 64 at once. A
  single session decodes 53 tokens/s, and 76 with `--head-parallel` (~230 on code with the draft).
- **Close to the original:** the same next token as the FP8 model 84% of the time on the Lucebox eval (KL 0.127).
- **Faster with the draft model:** the checkpoint's DSpark draft runs on the GPUs. With 64 sessions writing code, each
  gets 27 tokens/s instead of 20, and sampled outputs keep the model's own distribution (speculative sampling).
- **Plugs into your tools:** the OpenAI- and Anthropic-compatible API of [dsx](https://github.com/lxe/dsx), with
  streaming, tools, thinking levels and the chat in the browser.
- **Long context:** 128K tokens per session for 16 sessions at once (256K with one per GPU). The compressed KV
  cache is stored as FP4, ~0.9 KB per token.
- **Picks up where you left off:** a finished session's KV cache stays on the GPU, so an agent that resends the whole
  conversation only pays for the new text.

## The model

https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-Volta

Two folders. `base/` (221 GB) is the official checkpoint without its routed experts: the dense weights, the draft
layers and the Engram tables, unchanged. `pack-27.5gb-8xv100/` (220 GB) holds the routed experts, re-quantized for
this engine and already split across the 8 GPUs: 15,172 experts in cb3 (3.25 bits per weight) and 188 cold ones in
cb2 (2.25), chosen by how often the router picks them.

Against the FP8 model on the [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) eval
(8 x 1,024 tokens), it picks the same next token 84.1% of the time, with KL 0.127 and perplexity 8.88.

## How fast is it?

All on one 8x V100-SXM2-32GB machine. Engine benchmark (`volta bench`, real text, one token per session per step):

| Sessions | Step | Total |
|---|---|---|
| 64 | 41.7 ms | 1,536 tok/s |
| 128 | 69 ms | 1,850 tok/s |
| 256 | 105 ms | 2,440 tok/s |
| 512 | 167 ms | 3,070 tok/s |

With the DSpark draft (`volta spec --adaptive`, greedy, 64 sessions): 28.4 tok/s per session on code and 18.2 on
prose. With 8 sessions on code: 81 tok/s each.

One session (`volta spec`, greedy, steady-state step):

| Mode | Step | Per session | With the draft, on code |
|---|---|---|---|
| default (attention on the session's GPU) | 17.9 ms | 56 tok/s | 200 tok/s |
| `--head-parallel` (every GPU, 8 heads each) | 11.3 ms | 89 tok/s | 267 tok/s |

With `--head-parallel`, 4 sessions with the draft on code get 110 tok/s each.

Through the server (`volta-server`, sampling at temperature 1 with drafts, 300-token answers):

| Concurrent streams | Per stream | Total | First token |
|---|---|---|---|
| 1 | 38 tok/s | 38 tok/s | 0.27 s |
| 8 | 29 tok/s | 169 tok/s | 0.22 s |
| 32 | 20 tok/s | 500 tok/s | 0.4 s |

Long prompts (`volta spec`, 8 sessions, one per GPU, code, with drafts): with 64K-token prompts, prefill runs at
2,490 tok/s in total and decoding at 71 tok/s per session; with 128K-token prompts, at 1,430 and 46. The compressed
KV cache is stored as FP4, ~0.9 KB per token per session, so `--slots 2 --ctx 131072` gives 16 sessions of 128K.

For comparison, llama.cpp (PR #30817, UD-IQ1_M) on the same machine decodes 26.7 tok/s for one stream and 72-97
tok/s in total at most.

## Quick start

Download the weights (base and pack, ~441 GB):

```bash
hf download lxe/DeepSeek-V4.1-Flash-1017-Volta --local-dir volta-weights
```

Build and start the server:

```bash
CUDA_ARCHS=70 cargo build --release -p dsx --features cuda --bin volta-server

VOLTA_PACK=volta-weights/pack-27.5gb-8xv100 ./target/release/volta-server \
  --model volta-weights/base \
  --host 0.0.0.0 --port 8090
```

That serves 32 sessions of 8,192 tokens. For agents with long conversations, trade sessions for context:
`--slots 2 --ctx 131072` serves 16 sessions of 128K, and `--slots 1 --ctx 262144` 8 sessions of 256K.

For the fastest single sessions, add `--head-parallel`: every GPU runs every session with 8 of the 64 attention
heads, so the machine holds `--slots` sessions in total instead of `--slots` per GPU, each at up to ~89 tok/s
without the draft. Every GPU keeps a copy of each session's KV cache, so `--slots` and `--ctx` share memory as
they do without the flag.

Loading takes ~2 minutes (the first start also maps the 196 GB Engram tables into RAM). Open `http://localhost:8090/`
for the web UI, or point any OpenAI client at `http://localhost:8090/v1` (Anthropic clients at
`http://localhost:8090`):

```bash
curl http://localhost:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "stream": true,
    "reasoning_effort": "none",
    "messages": [
      {"role": "user", "content": "Write a haiku about MoE routing."}
    ]
  }'
```

An admin dashboard runs on `http://localhost:8091/`: output and prompt throughput (live and over the last 5 minutes),
sessions in use, speed per session, drafts kept, step time, and the active and recently finished requests. The same
numbers are JSON at `/stats`.

From the official checkpoint instead of the Hugging Face build, drop `VOLTA_PACK` and pass the checkpoint as
`--model`: the engine packs the experts on the GPUs at load (~3 minutes) and caches them in `/dev/shm/vcache`.

### Agents

For the [pi](https://github.com/badlogic/pi-mono) coding agent, add this to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "volta": {
      "baseUrl": "http://localhost:8090/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "authHeader": false,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "deepseek-v4.1-flash",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 16384,
          "thinkingLevelMap": { "off": "none", "high": "high" }
        }
      ]
    }
  }
}
```

```bash
pi --provider volta --model deepseek-v4.1-flash:off
```

`contextWindow` must match the server's `--ctx` (here `--slots 2 --ctx 131072 --max-tokens 16384`).

pi compacts long conversations by itself (it summarizes older messages when the context is nearly full). To start
compaction with room for a full answer and keep more recent history, add this to `~/.pi/agent/settings.json`:

```json
{
  "compaction": {
    "enabled": true,
    "modelOverrides": {
      "volta/deepseek-v4.1-flash": { "reserveTokens": 20480, "keepRecentTokens": 32768 }
    }
  }
}
```

The server's prefix cache reuses everything before the summary, so the first request after a compaction only
reads the summary and the messages kept after it.

With `--api-key`: drop `"authHeader": false`, set `"apiKey"` to the key and point `baseUrl` at the server's URL.

## Requirements

| | |
|---|---|
| GPUs | 8x V100 32 GB with NVLink in the DGX-1 layout (SXM2 or SXM3) |
| CUDA | 12.9 (the last toolkit that builds for sm_70) |
| RAM | ~400 GB (the Engram tables, 196 GB, are read from RAM) |
| Disk | ~441 GB for the Hugging Face build; `/dev/shm` or fast disk for the 220 GB pack keeps loading short |

Other machines need other kernels and collectives; [DESIGN.md](DESIGN.md) lists what each would take.

## Parameters

`volta-server` needs `--model`; every other option has a default.

**Model files**

| | |
|---|---|
| `--model` | the official checkpoint directory, or the Hugging Face build's `base/` with `VOLTA_PACK` set |
| `VOLTA_PACK` | the pack directory (`pack-27.5gb-8xv100/`): the routed experts, already packed for the 8 GPUs |
| `--tokenizer` | `tokenizer.json`; default: the model directory's |

**Server**

| | |
|---|---|
| `--host`, `--port` | listen address (default `127.0.0.1:8090`) |
| `--api-key` | require this key (`Authorization: Bearer <key>` or `x-api-key`; env `DSX_API_KEY`) |
| `--model-name` | model id reported by `/v1/models` (default `deepseek-v4.1-flash`) |
| `--slots` | concurrent sessions per GPU (default 4); with `--head-parallel`, in total |
| `--head-parallel` | every GPU runs every session with 8 of the 64 attention heads and a copy of its KV cache: the fastest single sessions, for fewer of them |
| `--ctx` | context window per session, prompt + generated tokens (default 8192); slots and context share the GPUs' spare memory: 2 x 131072 fits |
| `--max-tokens` | generation cap when a request sets none (default 4096) |
| `--chunk` | prompt tokens per session per step while reading a prompt (default 32) |
| `--admin-port` | the admin dashboard and `/stats` (default 8091, on `--host`; 0 = off). It has no API key: keep it on loopback or a private network |

**Sampling defaults** (a request's own fields override them)

| | |
|---|---|
| `--temperature`, `--top-p` | default 1.0 and 1.0, the model's recommended sampling |
| `--hesitation-penalty` | logit penalty on hesitation tokens ("Wait", "But", ...) while reasoning; 0 = off (default) |

**Draft**

| | |
|---|---|
| `--spec-k` | most draft tokens verified per pass (default 5; 0 turns the draft off) |
| `--theta` | a session verifies j drafts while p^j >= theta, p = its recent per-draft acceptance (default 0.35) |

## Thinking

Thinking is on by default, at high effort. A request turns it off or sets the effort:

| | OpenAI API (`/v1/chat/completions`) | Anthropic API (`/v1/messages`) |
|---|---|---|
| Thinking off | `"reasoning_effort": "none"` or `"thinking": {"type": "disabled"}` | `"thinking": {"type": "disabled"}` |
| Effort | `"reasoning_effort"`: `low` (or `minimal`), `high` (or `medium`), `xhigh`, `max` | `"output_config": {"effort": ...}`: `low`, `high` (or `medium`), `xhigh`, `max` |

## How it works

The model has 15,360 routed experts (384 in each of its 40 layers); each token uses 6 per layer, plus a shared expert,
attention, and two Engram lookups. Every step runs all sessions' tokens together.

- **Every expert split 8 ways:** each GPU holds 1/8 of every expert's weights (and of the shared expert), so every
  GPU does the same work in every layer, whichever experts the router picks.
- **Formats made for the hardware:** each expert's MXFP4 weights become cb3 or cb2 codes. The kernel turns two
  weights into fp16 with one shared-memory lookup and feeds Volta's `mma.m8n8k4` tensor cores directly. The dense
  weights are 4-bit with an fp16 scale per 16.
- **Attention per GPU, experts across all of them:** each GPU keeps its own sessions' KV cache and runs their attention
  and dense layers. Per layer, one NVLink all-gather sends every token's hidden state to every GPU, and one
  all-to-all brings the partial sums home. Both are single kernels that write into peer memory and signal with
  flags (~13-17 µs each).
- **Head-parallel for few sessions:** with `--head-parallel` each GPU runs 8 of the 64 heads (one of the model's 8
  output groups) of every session, so a token's dense attention weights are read once across the GPUs instead of
  whole on one. The attention output and the MoE partials are summed on every GPU in a fixed order, so all eight
  hold the same bits and route to the same experts; small exchanges use NCCL-style LL words (data and sequence
  number in one 8-byte store), 5-8 µs.
- **Attention on the tensor cores:** one KV row serves all 64 query heads, so a token's attention is a matrix product,
  run with `mma.m8n8k4` in fp16. The indexer that picks each token's 512 compressed rows scores them on the tensor
  cores too.
- **KV cache in FP4:** the model rounds its compressed KV rows and indexer keys to FP4, so they are stored packed,
  exactly: ~0.9 KB per token per session, 0.23 GB for a 256K-token context.
- **One CUDA graph per step:** routing is built on the GPU, small kernels are merged, and independent work (the
  shared expert, the larger expert batches) runs on a second stream inside the graph.
- **Several tokens per session per step:** prompts are read in chunks, and the draft's tokens are verified, in the same
  step as everyone else's decoding.
- **DSpark on the GPUs:** the checkpoint's 3-layer draft runs through the same MoE path. Each session verifies as many
  drafts as its recent acceptance justifies; every GPU verifies the same number of tokens per step, so the step stays
  in a CUDA graph.

[DESIGN.md](DESIGN.md) has the design, the measurements behind each choice, and the time per step by kernel.

## Command-line tools

`volta` (`--bin volta`) runs the engine without the server:

| | |
|---|---|
| `volta bench --model ... --sessions 8 --steps 32 --text tokens.i32` | throughput with `--sessions` per GPU; `--text` feeds real tokens (i32 LE) |
| `volta eval --model ... --eval <dir>` | the Lucebox eval; `--chunk k` feeds k tokens per sequence per step |
| `volta spec --model ... --text tokens.i32 --k 5 --adaptive` | greedy decoding with the draft (`VOLTA_SPARK=<checkpoint dir>`); `--total N` for N sessions in all |
| `volta check`, `volta experts` | logits and expert kernels against the CPU reference |

| Variable | Effect |
|---|---|
| `VOLTA_EXPERT_GB` | expert bytes per GPU when packing from the official checkpoint (default 25; the published pack is 27.5) |
| `VOLTA_GRAPHS` | most step shapes kept as CUDA graphs (the server's default is 4) |
| `VOLTA_TRACE_STEP` | wraps that step in a profiler range, for `nsys --capture-range=cudaProfilerApi` |
| `VOLTA_PROF` | prints where each step's time went |
| `VOLTA_OVERLAP=0`, `VOLTA_ATTN=tile5`, `VOLTA_SPARK_CB2=0` | one stream, CUDA-core attention, cb3 draft experts |
| `VOLTA_KV=bf16` | store the compressed KV cache in bf16 instead of FP4 (3.5x larger) |

## Repo

| Path | |
|---|---|
| [`engine/src/volta.rs`](engine/src/volta.rs), [`engine/src/volta/spark.rs`](engine/src/volta/spark.rs) | the engine: packing, placement, steps, attention, MoE, collectives, CUDA graphs; the draft |
| [`engine/src/kernels/volta.cuh`](engine/src/kernels/volta.cuh) | the sm_70 kernels |
| [`engine/src/server/volta.rs`](engine/src/server/volta.rs), [`engine/src/bin/volta_server.rs`](engine/src/bin/volta_server.rs) | the batching scheduler and the server |
| [`engine/src/server/telemetry.rs`](engine/src/server/telemetry.rs), [`engine/src/server/dashboard.html`](engine/src/server/dashboard.html) | the admin dashboard's numbers and page |
| [`core/`](core/), the rest of [`engine/`](engine/) | model loader, CPU reference, the HTTP server and web UI, from [dsx](https://github.com/lxe/dsx) |
| [`plans/`](plans/) | the published pack's tier plan and placement, and the routing counts that choose tiers |
| [`tests/`](tests/) | standalone kernel tests against reference kernels: expert formats (`vmoe_test`, `vlut_test`), dense and expert launch shapes (`dense_test`, `split_test`), attention (`attn_test`), indexer (`indexer_test`) |
| [`bench/`](bench/) | hardware probes and the existing-engine benchmarks |

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) · [dsx](https://github.com/lxe/dsx) for the model
loader, CPU reference, server and web UI · [Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF)
for the eval set · the web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s, by Georgi Gerganov and the ggml
authors (MIT), vendored in [`core/webui`](core/webui) with its license.

MIT license.
