---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - deepseek
  - moe
  - v100
  - volta
  - gguf
  - quantized
---

# DeepSeek-V4.1-Flash-1017-Volta

DeepSeek-V4.1-Flash-1017 for [Volta](https://github.com/lxe/volta), an inference engine for eight V100-SXM2-32GB GPUs with NVLink in the DGX-1 layout. Its main routed experts use cb3 at 3.25 bits per weight, with cb2 at 2.25 bits for the coldest experts, chosen by routing frequency. The expert weights occupy about 220 GB, or 27.5 GB per GPU.

The entire model is distributed as **one `DeepSeek-V4.1-Flash-1017-Volta.gguf` file**, about 442 GB. It contains the original base tensor bytes, Engram tables, draft and vision weights, tokenizer and model configuration, plus the quantized main routed experts. Volta reads these tensors directly from the mapped file; no separate base directory, tokenizer file or extraction cache is required.

The main expert tensors contain row-major packed codes and scale bytes. Volta reconstructs its kernel layout when loading them. **This file requires Volta's custom cb3/cb2 decoder; it is not a llama.cpp-compatible quantization type.** GGUF tensors and Volta metadata preserve the original tensor dtypes, source slice boundaries and multiplier bits.

| Path | Contents |
|---|---|
| `DeepSeek-V4.1-Flash-1017-Volta.gguf` | The complete model, tokenizer and runtime metadata in one file. About 442 GB. |
| `quantization/` | Format specification, checksums and exhaustive lossless-conversion receipts. |
| `LICENSE`, `README-original.md` | The original model's license and README, also preserved inside the model file. |

## Repack verification

The October 24, 2026 repack replaces both the former `base/` directory and `pack-27.5gb-8xv100/gpu*.bin` files with one GGUF. It changes storage layout without requantizing the model. Every source code byte, scale byte, tier, owner, TP slice and float32 multiplier bit was checked against the reconstructed native representation. The source is Hugging Face revision `d0a725d14de03275eea9efe8e4e6329ebe07665f`; its per-file SHA-256 values and the new GGUF hash are recorded in `quantization/manifest.json`.

The exhaustive comparison passed for 245,760 expert matrices, 3,925 base tensors and 20 embedded assets. The complete-file CPU load check constructed the main model, DSpark draft, Engram views and tokenizer using only this GGUF. See `quantization/verify-full.json` and `quantization/smoke.json` for the exact scope and counts.

The original eight-way source slices and their multiplier domains are retained, so this GGUF still requires the original eight-GPU topology. Every base tensor and embedded asset was compared byte for byte against the published source files. The layout-only publication initially included CPU and byte-equivalence checks; subsequent full-GGUF GPU verification is recorded below.

## Results measured with the complete GGUF

On October 24, 2026, the complete GGUF loaded and served requests with frozen Volta commit `cc54d97568ad` on eight V100-SXM2-32GB GPUs with active NVLink. The rental had 754 GiB host RAM and 800 GB of storage. A full-file SHA-256 check matched the published model hash before performance measurement.

| Measurement | Result and scope |
|---|---|
| Teacher evaluation, 8 × 1,024 tokens | 8,184 evaluated positions; KL top-256 **0.126695**, top-1 agreement **84.3231%**, perplexity **8.9423** |
| Four concurrent Pi coding tasks | Cube, pyramid, torus and sphere programs completed; all four animations ran and changed frames in their assigned tmux panes |
| Agent workload | 47 model requests, 19,805 generated tokens; **167.3 tokens/s mean aggregate** during the interval with at least two active sessions |
| Expert reconstruction and GPU upload | About **169–180 seconds** in these observed loads; Engram mapping added about 8–13 seconds |

The agent run used one DP slot per GPU, context capacity 131,072, prefill chunk 64, five speculative drafts, temperature 0.7, top-p 1 and graph budget 4. A separate controlled retrieval test subsequently exercised 126,041 actual prompt tokens with zero cache reuse and returned the exact key placed at the beginning; time to first token was 226.50 seconds. This establishes that one retrieval workload, rather than general long-context quality. The throughput figure excludes the separate API smoke and is specific to the generated task work. It is neither a per-agent rate nor a controlled speed comparison against the historical native-pack runs. The server retained only about 0.5 GB free per GPU after loading, so larger contexts or session configurations require their own memory checks.

Exact metrics, model/binary hashes, configuration, recording hash and animation checks are in [runtime/2026-10-24-v100-baseline.json](runtime/2026-10-24-v100-baseline.json). The original DP sampler eligibility change keeps prefill sampling on the GPU while retaining its decode, top-p, hesitation and head-parallel exclusions. A separate opt-in HP path is described below. Four fresh servers in A/B/B/A order, with matched prompts, full output budgets and zero cache hits, measured 15.8–17.7% higher aggregate throughput for the mixed four-client workload and 9.6–12.3% for the fixed four-client workload. The compared work was 2,048 generated tokens in each trace. Sampled strings differed, so these are matched-work comparisons rather than claims of identical seeded output.

With the sampler change, separate cold 4K-input/1,024-output screens measured 165.2–169.3 aggregate whole-request tokens/s for four clients and 267.2–273.2 for eight clients. One-client decode rate was 80.6–84.4 tokens/s. The actual four-Pi tool gauntlet also completed all four shape programs and live animations; its variable task work is not a controlled speed comparison. The sampler source and receipts are in [Volta PR #1](https://github.com/lxe/volta/pull/1).

Four simultaneous DP requests also each exercised 126,045 actual prompt tokens with zero cache reuse and returned their own distinct initial keys exactly. The server observed four active sessions with no queue; all four naturally stopped after 17 output tokens, and total wall time was 297.69 seconds. This instrumented run establishes retrieval and session separation for that workload. Its memory guard failed before its first observation, so it does not establish a continuously measured free-memory reserve. The guard startup check was repaired afterward. The [four-request evidence](https://github.com/lxe/volta/tree/268cf362c2dd3eb701e1060ed3a765f5837e6402/docs/benchmarks/2026-10-24-v100/context/four-concurrent-128k) retains that limitation.

A separate head-parallel configuration with four active slots and capacity 262,144 passed the same 8 × 1,024-token teacher gate: perplexity 8.9417, KL top-256 0.129511 and top-1 agreement 84.0787%. It also completed a real 250,042-token prompt with zero cache reuse and returned the exact initial deployment key. Time to first token was 468.57 seconds. That earlier result used one active long request; the later four-slot check is reported below. Neither retrieval workload establishes general long-context quality. The 1Hz memory guard observed at least 1,743MiB free on every GPU throughout this screen.

An opt-in head-parallel GPU sampler was also evaluated with four active slots at capacity 262,144. The fixed trace sends four 4K-input requests; the mixed trace starts three 4K-input decoders and then admits an 8K prefiller after they begin generating. Both request 512 output tokens per client, totaling 2,048 outputs per trace. In fresh-server A/B/B/A comparisons, fixed four-client throughput increased from 49.0–49.5 to 75.1–75.4 aggregate tokens/s, a 51.8–53.9% gain. Mixed prefill/decode throughput increased from 42.0–42.1 to 62.1–62.3 tokens/s, a 47.8–47.9% gain. Each trace generated 2,048 tokens with identical supplied requests and zero cache hits. Audit and profiling were disabled during timing, and PID-bound memory observations bracketed each trace with at least 192 MiB free per GPU. The candidate collects the eight vocabulary slices on GPU0 and uses the existing GPU sampler, avoiding host downloads and full-vocabulary assembly as well as CPU distribution work. CPU and GPU sampling consume random numbers differently, so these are matched-work measurements with different sampled trajectories. The component timings do not isolate each source of the gain.

The final HP source passed the real-row collection audit: 598 exact checks covering 9,032,535,040 bytes, followed by sampler replay using the same supplied uniforms on independently CPU-concatenated logits. This validates collection and table addressing with the existing sampler, rather than independently proving its sampling distribution. The initial candidate teacher evaluation and a predeclared fresh flag-off/flag-on pair all passed the original DP-relative thresholds on 8,184 positions. Across those three results, perplexity was 8.9402–8.9605, KL top-256 was 0.129370–0.131653 and top-1 agreement was 83.8710–84.2131%. All three are retained; the initial result was close to the KL limit. The teacher path itself does not invoke the new sampler, and the source review identifies floating-point atomic ordering as a possible source of run variation without establishing its cause.

A fresh four-slot HP server also completed four concurrently submitted 250,046-token prompts, totaling 1,000,184 prompt tokens. Every response returned its own distinct initial key exactly after trimming outer whitespace and naturally stopped after 17 output tokens. There were zero prefix-cache hits, four observed active HP slots and no observed queue. Total wall time was 1,872.11 seconds; client-visible times to first token ranged from 468.86 to 1,871.94 seconds. The scheduler admits all four slots but serializes their prefills within one 64-row HP group. This establishes the measured retrieval and session-isolation workload, rather than four parallel prefills, general long-context quality or a sampling speedup; all four requests were greedy and used the unchanged StepGreedy path.

The PID-bound memory guard bracketed first admission through last completion with 1,721 observations. It observed at least 1,753 MiB free on every GPU, with a maximum sampling gap of 1.137 seconds. These sampled observations do not establish the instantaneous memory minimum. Raw results and the strict analysis are in [the four-request HP evidence](https://github.com/lxe/volta/tree/268cf362c2dd3eb701e1060ed3a765f5837e6402/docs/benchmarks/2026-10-24-v100/context/four-concurrent-250k-hp).

A descriptive six-case HP GPU-sampler profile then used one, four, eight, eight, four and one submitted clients on that same four-slot server. Each cold request supplied 4K input tokens and generated the full 1,024-token output budget with zero cache reuse. Across the two repeats, whole-request aggregate throughput was 70.0–72.2 tokens/s for one client, 112.3–113.4 for four and 119.3–120.1 for eight submissions. The eight-client cases observed four active sessions and four queued requests. Single-client decode, excluding time to first token, was 108.6–109.1 tokens/s. The server retained its prior long-context cache/graph history; this is a descriptive profile, not a controlled cross-mode gain. The guard observed at least 1,699 MiB free per GPU. [Full concurrency receipts](https://github.com/lxe/volta/tree/268cf362c2dd3eb701e1060ed3a765f5837e6402/docs/benchmarks/2026-10-24-v100/hp-gpu-sampling/concurrency-profile) include per-client timing and the exact scope.

A fresh HP server at 262,144-token capacity completed the four-Pi shape gauntlet: 49 API requests and 20,653 generated tokens in a 193.09-second controller interval. Mean aggregate throughput during the sampled multi-active interval was 175.26 tokens/s. All four standard-library programs ran and changed Braille frames in local kitty/tmux panes, and the recording closed. Source review found that the original torus discarded its normal-derived lighting; a separately recorded, parent-requested Pi correction used five additional requests and 2,317 output tokens to implement per-dot depth, intensity and ordered dithering. The corrected source and live frames were checked separately. The cube can clip at some rotation extrema and the pyramid uses approximate depth handling, so this is not a clean pass of every visual requirement. Original work and repair receipts are preserved independently in [the gauntlet archive](https://github.com/lxe/volta/tree/268cf362c2dd3eb701e1060ed3a765f5837e6402/docs/benchmarks/2026-10-24-v100/hp-gpu-sampling/gauntlet-original/). The [separate repair archive](https://github.com/lxe/volta/tree/268cf362c2dd3eb701e1060ed3a765f5837e6402/docs/benchmarks/2026-10-24-v100/hp-gpu-sampling/gauntlet-torus-repair/) contains its prompt, source and recording. Variable generated task work prevents a controlled speed comparison against the earlier gauntlets.

The earlier CPU-sampled head-parallel configuration was slower for four short requests: 75.5–76.0 aggregate whole-request tokens/s in the cold 4K-input/1,024-output profile. This output length differs from the 512-output HP GPU-sampler A/B/B/A comparison above. Eight submitted requests queued behind four active slots. One-client results in that earlier CPU-sampled HP profile varied substantially, so its best decode rate is not a repeatable speedup claim. The measured data-parallel 128K configuration remains the preferred throughput setting for four clients in these cold 4K-input/1,024-output profiles.

The measured DP prefill-guard source is [commit `4ce1545`](https://github.com/lxe/volta/commit/4ce1545); the matching short/long-context receipts are linked from [Volta PR #1](https://github.com/lxe/volta/pull/1). The integrated HP sampler source is [commit `78d2dc9`](https://github.com/lxe/volta/commit/78d2dc90afb4c9782c2893caf8106bf2e754e827), with exact source-byte binding to the tested build. The complete public runtime summary is [runtime/2026-10-24-v100-optimized.json](runtime/2026-10-24-v100-optimized.json). The model file and quantization remain unchanged throughout these runtime experiments.

## Results measured with the native pack

| Measurement | Result |
|---|---|
| Quality, Lucebox eval: 8 × 1,024 tokens against the FP8 model's top-256 | KL 0.127, top-1 agreement 84.1%, perplexity 8.88 |
| 64 concurrent sessions, one token per step | 41.7 ms per step: 1,536 tok/s aggregate |
| 256 / 512 sessions | 2,440 / 3,070 tok/s aggregate |
| DSpark draft, greedy, 64 sessions writing code | 28.4 tok/s per session |
| API server, temperature 1 | One stream 93 tok/s decode; 32 streams 39 tok/s each, 967 tok/s aggregate throughput |
| Eight sessions with 64K-token prompts | Prefill 2,490 tok/s total; decode 71 tok/s per session |

These measurements are recorded in the engine README at commit `362dded4c85095bf56d5ada6ff95d54415f960e8` and come from different configurations; they are not simultaneous guarantees. The API figures use 300-token sampled answers and distinguish request decode rate from aggregate throughput. See the [engine README](https://github.com/lxe/volta) for configurations and subsequent serving measurements.

## Run

Use [Volta commit `78d2dc90afb4`](https://github.com/lxe/volta/commit/78d2dc90afb4c9782c2893caf8106bf2e754e827), which includes the measured DP prefill guard and opt-in HP sampler. Its three engine source files exactly match the tested V3 source hashes; the build receipt retains the original isolated source commit. The machine needs eight 32 GB V100 GPUs with active NVLink in the DGX-1 layout. The measured host had 754 GiB of RAM and 800 GB of storage; this run does not establish a minimum host-RAM requirement. The model alone occupies 441.6 GB, so reserve additional storage for the toolchain and evidence. CUDA 12.9 is the last toolkit that builds for sm_70; set `NVCC` to its compiler if another CUDA toolkit is also installed.

```bash
git clone https://github.com/lxe/volta.git
cd volta
git checkout 78d2dc90afb4c9782c2893caf8106bf2e754e827
hf download lxe/DeepSeek-V4.1-Flash-1017-Volta --local-dir volta-weights
NVCC=/usr/local/cuda-12.9/bin/nvcc CUDA_ARCHS=70 cargo build --release -p dsx --features cuda --bin volta-server

VOLTA_GPU_SAMPLE=1 VOLTA_GRAPHS=4 VOLTA_MEGA=0 VOLTA_KV=fp4 \
VOLTA_OVERLAP=1 VOLTA_SHARED_TP=1 VOLTA_SHARED_EARLY=0 \
./target/release/volta-server \
  --model volta-weights/DeepSeek-V4.1-Flash-1017-Volta.gguf \
  --gpus 8 --slots 1 --ctx 131072 --chunk 64 --spec-k 5 --theta 0.35 \
  --temperature 0.7 --top-p 1 --max-tokens 16384 --host 0.0.0.0 --port 8090
```

The command above allocates eight DP slots at 128K capacity; the four-client measurements used four active sessions. For the tested HP mode, set `VOLTA_HP_GPU_SAMPLE=1` and replace `--slots 1 --ctx 131072` with `--head-parallel --slots 4 --ctx 262144`. This allocates four total active HP slots across all eight GPUs. Eight submitted HP clients queue behind those four slots. Higher slot counts require a separate memory check; eight HP slots at 256K previously failed allocation. Loading the GGUF reconstructs the expert kernel layout before uploading it to the GPUs; it does not run the quantizer. Observed expert reconstruction and GPU upload took about 169–180 seconds in the full-GGUF verification above; these loads are not a controlled cold-cache startup benchmark. The two Engram tables occupy 196 GB of host RAM, and reading them from storage adds loading time.

The server provides OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs, streaming, tool calls and thinking levels. It batches requests, verifies DSpark drafts by speculative sampling, and reuses cached prefixes. A browser chat is available at `/` and an admin dashboard at port 8091.

## Format and reconstruction

Read [quantization/FORMAT.md](quantization/FORMAT.md) for tensor names, codebooks, scale conventions, embedded metadata and source slice domains. Pass the complete GGUF as `--model`; the engine selects its expert tensors and embedded tokenizer automatically. Previously downloaded base directories and native expert packs remain supported through the old `--model BASE_DIR` and `VOLTA_PACK=PACK_DIR` combination. The original quantization plan is in [Volta's plans directory](https://github.com/lxe/volta/tree/main/plans).

## License and credit

MIT, the same as the original model. [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) is by DeepSeek. Its original README and license are retained as `README-original.md` and `LICENSE`, and embedded in the GGUF.
