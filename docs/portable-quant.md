# Complete Volta model GGUF

The complete Volta GGUF contains the main quantized experts, original base tensors, tokenizer, model configuration, Engram constants, draft weights, vision weights and sidecar assets in one file. Volta loads that file directly without a separate base directory or an extracted weight cache. Export changes storage layout without running a quantizer: expert codes, scales, tier/ownership plan and source multiplier bits remain exact, and every base tensor and asset retains its original bytes.

This is a standard GGUF v3 container with Volta encoding metadata. The main CB3/CB2 expert encodings are custom; they are not registered ggml quant types and are not llama.cpp-compatible. The original eight-way expert slices and their multiplier domains remain intact, so the published plan still requires its original GPU topology.

## Published-input verification

The October 24, 2026 conversion of Hugging Face revision `d0a725d14de03275eea9efe8e4e6329ebe07665f` produced one file of 441,574,130,240 bytes. Its SHA-256 is `9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c`. All eight original expert binaries and all original base files were checked against the source repository's hashes before conversion.

Exhaustive verification passed for 245,760 expert matrices: 203,011,522,560 code bytes and 16,986,931,200 scale bytes, along with every tier, ownership entry, TP slice and multiplier bit. It also compared 3,925 base tensors (221,508,192,600 bytes) and 20 embedded assets (7,624,809 bytes). Source and reconstructed expert payload SHA-256 values match; source and embedded base/asset payload SHA-256 values match. The exact receipt is `portable-complete-published-verify.json`.

The actual complete-file CPU load check passed with both original source directories temporarily hidden. It constructed the 40-layer main model, all 3 draft layers, both Engram views and the 129,280-entry tokenizer from the GGUF alone. The file also contains 266 vision tensors. The independence receipt is `portable-complete-published-independence.json`.

| Local operation | Wall time | Peak process RSS |
|---|---:|---:|
| Complete export | 808.67 s | 10.27 GiB |
| Exhaustive verification | 1,033.88 s | 9.66 GiB |
| CPU load with source directories hidden | 1.37 s | 1.49 GiB |

These measurements were made on the local i7-13700. The load check ran after the file had been read and does not populate the full 196 GB Engram mappings, upload GPU weights or create a CUDA context. Its timing is not a serving-engine startup measurement. No new GPU inference, quality evaluation or decode benchmark was run. Historical model results remain identified as native-pack measurements.

## Complete file contents

`general.architecture` is `volta-experts`. `volta.index` describes the main routed experts. A complete file additionally contains `volta.base`, a JSON index with `version: 1`, a `tensors` dictionary and an `assets` dictionary.

Each entry in `tensors` maps an original HF tensor name to its GGUF `tensor` name, original `dtype` and original `shape`. The GGUF name is `base.` followed by the original name. F32, F16, BF16, I8 and I32 payloads use the corresponding standard GGUF element type and reversed dimension order. Other original dtypes, including FP8 E4M3 and E8M0 scale bytes, use I8 byte tensors. The original dtype/shape metadata tells the HF-compatible loader how to interpret those bytes. Scalar tensors retain their original scalar shape in the index.

Each entry in `assets` maps a source-relative filename to `asset.` followed by that filename. Assets are I8 byte tensors with exactly the original file bytes, including empty assets. Required assets are `config.json`, `dsx-engram.json` and `tokenizer.json`. Other non-hidden sidecar files, including tokenizer configuration, encoding files, licenses, the original README and safetensors index JSON, are retained. Files ending in `.safetensors` are read as individual tensors and are not embedded as opaque archive files. Source download caches and other hidden files are excluded. Each asset is limited to 64 MiB and their combined size to 256 MiB.

The GGUF-backed SafeTensors view restores original tensor names, dtypes, shapes and byte ranges. Main model, dense matrices, Engram and DSpark therefore use the existing HF loading and decoding logic. Vision tensors are retained and verified; this conversion does not add vision serving support to Volta.

## Export, verify and load

Build the CPU-only exporter/verifier:

```sh
cargo build --release -p dsx-core --bin volta-quant
```

Export the existing native pack and published base directory into one complete file. The output path must not exist, and the source files must remain unchanged during export and verification:

```sh
./target/release/volta-quant export-full /path/to/pack /path/to/base /path/to/DeepSeek-V4.1-Flash-1017-Volta.gguf 5120 2304
./target/release/volta-quant verify-full /path/to/pack /path/to/base /path/to/DeepSeek-V4.1-Flash-1017-Volta.gguf 5120 2304 > verify-full.json
./target/release/volta-quant smoke /path/to/DeepSeek-V4.1-Flash-1017-Volta.gguf > smoke.json
```

`verify-full` exhaustively compares every reconstructed native expert code/scale byte, exact source multiplier bit pattern and complete tier/ownership/topology plan. It also compares every base tensor byte, original dtype/shape descriptor and asset byte. Success requires exit status 0 and JSON `status: "exact"`. Progress goes to stderr. The receipt includes the expert verification result, base/asset counts, compared byte counts, SHA-256 hashes and timings. Both source and output are opened read-only.

Expert payload hashes bind each sorted matrix descriptor and its reconstructed native code/scale bytes. Base payload hashes bind sorted tensor names/descriptors and original bytes, followed by sorted asset names, lengths and bytes. These logical payload hashes are distinct from whole-file SHA-256 checksums. A publication manifest should record the whole-file checksums separately.

`smoke` constructs the actual main model through the embedded HF tensor views, constructs DSpark from the embedded configuration and draft weights, opens the embedded Engram tensor views, parses the embedded tokenizer and tokenizes a small string. It does not create a CUDA context, instantiate the engine, populate the full Engram mappings, or dequantize the full weight matrices. The normal HF loader still materializes its small parameter vectors and router arrays. The receipt reports model/draft dimensions, Engram byte ranges, tokenizer results, vision tensor count, elapsed time and peak RSS. `gpu_inference_verified` remains false.

Start the GPU server using only the complete file:

```sh
./target/release/volta-server --model /path/to/DeepSeek-V4.1-Flash-1017-Volta.gguf
```

The server automatically selects the same file for experts, tokenizer and draft. The Volta CLI accepts it through `--model` too; `volta spec` selects its embedded draft. Existing checkpoint-directory and native-pack paths continue to work.

## Resource budget and acceptance

Known input sizes are approximately 220 GB of native experts and 221 GB of base data. Full export reads approximately 441 GB and writes approximately 441 GB, plus small directory/index overhead. Exhaustive verification reads both inputs and the output, approximately 882 GB in total. No full checkpoint requantization or second extracted weight cache is required.

Copy/compare chunks are 8 MiB. Expert layout conversion allocates one matrix at a time; the actual source slice shapes are GU `576 x 5120` and down `5120 x 288`. At four bits, the larger temporary is approximately 1.6 MB including scales; the published main experts use three/two bits. File mappings may span hundreds of GB of virtual address space, while heap use remains directory-sized plus these bounded buffers. Consumed source/output payload mappings are released with `MADV_DONTNEED` on Unix. File-cache pages may remain reclaimable in the OS cache; no source bytes are changed.

The production-shaped sparse expert-index screen covers 245,760 matrices and 491,520 tensors, approximately 220 GB of logical offsets, and a 59 MB written header. It measured approximately 369 MiB peak RSS and 0.5 seconds directory-open time locally. It reads only selected zero matrices and is index/reader evidence, not a full payload verification. Complete models add their base tensor and sidecar descriptors. Actual full-file export/verification RSS and elapsed time should be recorded by the publishing run.

Normal GPU serving still needs the existing host RAM budget for Engram; a CPU mapping smoke test with limited RAM does not establish that the full engine can serve within that RAM. GPU layout, allocated GPU weight bytes and decode work are unchanged. Full-model GGUF startup throughput and GPU inference remain unmeasured until an actual engine run.

## CPU verification evidence

Local tests cover exact 2/3/4-bit CUDA-layout reconstruction, mixed-tier native pack export/verification, source multiplier preservation, legacy split-GGUF reading, typed/scalar/FP8/FP4 base tensor views, asset and empty-asset preservation, and rejection of corrupted expert/base bytes or changed multipliers. The full-file fixture also compares FP8 and FP4 HF matrix decoding before and after export.

A synthetic release-CLI fixture exported and verified 61 base tensors, then loaded its main model, DSpark, Engram views, vision tensor and tokenizer with the original source base/pack paths hidden. Receipts are `portable-complete-cli-fixture-verify.json`, `portable-complete-cli-fixture-smoke.json` and `portable-complete-cli-fixture-independence.json`. These establish file independence on that fixture; they are not full-model or GPU measurements. The full published-input verification and actual model CPU load are recorded above and in the published-input receipts.

## Legacy expert-only GGUF

The earlier `export PACK OUTPUT DIM INTER` and `verify PACK OUTPUT DIM INTER` commands remain available for expert-only files. Those files require the original base directory and can be loaded with `VOLTA_PACK=experts.gguf`. They also support the existing DSX split-GGUF reader. New complete exports produce one self-contained model file.

## Tensor and metadata definitions

`general.architecture` is `volta-experts`. `volta.index` is a JSON string with `version: 1`, `dim`, `inter`, the original `pack` manifest, and `matrices`. Each matrix entry contains `layer`, `expert`, `part`, `role`, `rows`, `cols`, `bits`, and `mul_bits`. `part` identifies the source TP slice. `gu` concatenates that slice's gate and up rows; `down` contains that slice's intermediate columns. Rows and columns must both be positive multiples of 32. The matrix dimensions must agree with `dim`, `inter`, the tier plan and `etp`.

Every matrix has two standard GGUF I8 tensors named `layers.L.experts.E.part.P.ROLE.codes` and `.scales`. Both tensors are one-dimensional byte arrays. Codes are packed in row order, low bit first, with exactly `bits` per code and no padding. Scales are row-major with one byte for each consecutive 32 codes. Every tensor starts at a 32-byte boundary.

Tier 0 uses four-bit E2M1 indices and the original E8M0 scale bytes. Tier 1 uses three-bit indices into `[-8, -4.5, -2.5, -0.625, 1.25, 2.75, 4.5, 8]`; tier 2 uses two-bit indices into `[-8, -1.875, 2.75, 8]`. The CB tiers use the original positive E4M3 scale bytes and the matrix multiplier stored as the exact u32 bit representation in `mul_bits`. The decoded CB weight is `codebook[code] * E4M3(scale_byte) * f32::from_bits(mul_bits)`. A source slice's multiplier applies only to that slice; exporters must not merge or normalize multipliers. Format version 1 fixes these codebooks and scale conventions.

## CPU conversion screening plan

The original scalar implementation processes each code one bit at a time. A follow-up CPU mechanism changes the same permutation to whole-word transposes for two/four-bit groups and sixteen six-bit pair moves for CB3. Each 32-weight group still reads and writes `bits * 4 + 1` bytes; it removes up to 96 individual bit extraction/insertion steps per group without adding a buffer or changing the output. Predicted direction is lower instruction cost and higher conversion throughput. Confirm with the same release-built single-thread benchmark on GU `576 x 5120` and down `5120 x 288` (dim 5120, inter 2304, TP 8), before/after; kill if any exact-layout test fails or either CB3 direction slows. This is a local cached fixture screening measurement, excluding model I/O, GGUF directory parsing, concurrent loaders and GPU upload.

## CPU screening results

On the local Intel Core i7-13700, release-built with rustc 1.99.0 on October 24, 2026, the final single-thread fixture measured the following payload rates. Each rate counts code and scale bytes once; it does not count both memory reads and writes. Both source slice shapes passed exact roundtrip checks before timing.

| Tier | GU load / export MB/s | Down load / export MB/s |
|---|---:|---:|
| CB2 | 1716 / 1837 | 1638 / 1758 |
| CB3 | 849 / 726 | 852 / 723 |
| MXFP4 | 2475 / 2667 | 2518 / 2657 |

The initial bit-at-a-time CB3 loop measured approximately 116 MB/s loading and 111–118 MB/s exporting. The word/pair implementation therefore reduces the CPU cost materially. For an approximate 220 GB mostly-CB3 payload, the measured rates correspond to about 4.3 single-core minutes of layout reconstruction, or about 5.1 single-core minutes of export conversion. Those are CPU work estimates from cached synthetic matrices. They exclude full-pack I/O, large metadata parsing, GPU upload and scheduling; eight concurrent loader threads have not been measured and linear scaling is not claimed. Maximum child RSS across the two screening processes was 18,184 KiB; this includes process overhead, while individual output matrices were at most 1,566,720 bytes including scales.

Raw receipts are `portable-quant-cpu-before-gu.txt`, `portable-quant-cpu-before-down.txt`, `portable-quant-cpu-after-gu.txt`, `portable-quant-cpu-after-down.txt` and `portable-quant-cpu-resource.txt` in this directory. Reproduce the final screening with:

```sh
cargo run --release -p dsx-core --example volta_quant_throughput -- 576 5120 1000
cargo run --release -p dsx-core --example volta_quant_throughput -- 5120 288 1000
```

The final local core run discovered 23 tests: 22 passed and one production sparse-index screen was intentionally ignored in the ordinary run. That screen passed when invoked explicitly. Both `volta-server` and the Volta CLI passed release builds for the actual sm_70 target using CUDA12.9.86, GCC14 and locally isolated Ubuntu24.04 C headers. The compiled fatbin contains `dsx.sm_70.cubin`; the build receipt is `portable-complete-v100-build.txt`. Complete published-input verification and load smoke passed as recorded above. Compilation and CPU load checks do not establish GPU inference correctness or throughput.
