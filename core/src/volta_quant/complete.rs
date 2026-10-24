//! Complete single-file model: original HF tensor bytes and small sidecar assets.
use super::*;
use crate::safetensors::{SafeTensors, StTensor};
use std::collections::{BTreeMap, HashMap};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct BaseTensor {
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct CompleteIndex {
    pub version: u32,
    pub tensors: BTreeMap<String, BaseTensor>,
    pub assets: BTreeMap<String, String>,
}
fn element_bytes(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "BF16" | "F16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        _ => anyhow::bail!("unsupported original tensor dtype {dtype}"),
    })
}
fn byte_len(t: &BaseTensor) -> Result<usize> {
    t.shape.iter().try_fold(element_bytes(&t.dtype)?, |n, &v| {
        n.checked_mul(v).context("base tensor size overflow")
    })
}
fn ggml_layout(t: &BaseTensor) -> Result<(u32, Vec<u64>)> {
    let ty = match t.dtype.as_str() {
        "F32" => 0,
        "F16" => 1,
        "BF16" => 30,
        "I8" => 24,
        "I32" => 26,
        _ => 24,
    };
    let dims = if matches!(t.dtype.as_str(), "F32" | "F16" | "BF16" | "I8" | "I32") {
        if t.shape.is_empty() {
            vec![1]
        } else {
            t.shape.iter().rev().map(|&d| d as u64).collect()
        }
    } else {
        vec![byte_len(t)? as u64]
    };
    Ok((ty, dims))
}
pub(crate) fn index(g: &Gguf) -> Result<Option<CompleteIndex>> {
    let Some(raw) = g.kv.get("volta.base") else {
        return Ok(None);
    };
    let base: CompleteIndex =
        serde_json::from_str(raw.as_str().context("volta.base must be JSON string")?)?;
    ensure!(
        base.version == 1,
        "unsupported complete model index version"
    );
    let mut seen = std::collections::HashSet::new();
    for (name, t) in &base.tensors {
        ensure!(
            !name.is_empty() && t.tensor == format!("base.{name}") && seen.insert(&t.tensor),
            "invalid base tensor identity"
        );
        let gg = g.get(&t.tensor)?;
        let (ty, dims) = ggml_layout(t)?;
        ensure!(
            gg.ty as u32 == ty && gg.dims == dims && gg.bytes == byte_len(t)? as u64,
            "invalid base tensor {name}"
        );
    }
    let mut asset_bytes = 0u64;
    for (name, tensor) in &base.assets {
        ensure!(
            !name.is_empty()
                && !name.starts_with('/')
                && !name.split('/').any(|s| s == "..")
                && tensor == &format!("asset.{name}")
                && seen.insert(tensor),
            "invalid asset path"
        );
        let t = g.get(tensor)?;
        ensure!(
            t.ty == GType::I8 && t.dims == [t.bytes] && t.bytes <= 64 << 20,
            "invalid or oversized asset {name}"
        );
        asset_bytes += t.bytes;
    }
    ensure!(
        asset_bytes <= 256 << 20,
        "embedded small assets exceed256MiB"
    );
    for required in ["config.json", "dsx-engram.json", "tokenizer.json"] {
        ensure!(
            base.assets.contains_key(required),
            "complete model missing {required}"
        );
    }
    Ok(Some(base))
}
pub(crate) fn view(path: &Path, wanted: &[&str]) -> Result<SafeTensors> {
    let g = Gguf::open(path)?;
    let base = index(&g)?.context("GGUF has no complete model base tensors")?;
    let mut tensors = HashMap::new();
    let mut assets = HashMap::new();
    for (name, t) in base.tensors {
        if !wanted.is_empty() && !wanted.contains(&name.as_str()) {
            continue;
        }
        let gt = g.get(&t.tensor)?;
        tensors.insert(
            name,
            StTensor {
                dtype: t.dtype,
                shape: t.shape,
                file: gt.shard,
                start: gt.offset as usize,
                end: (gt.offset + gt.bytes) as usize,
            },
        );
    }
    for (name, tensor) in base.assets {
        let t = g.get(&tensor)?;
        assets.insert(
            name,
            StTensor {
                dtype: "I8".into(),
                shape: vec![t.bytes as usize],
                file: t.shard,
                start: t.offset as usize,
                end: (t.offset + t.bytes) as usize,
            },
        );
    }
    Ok(SafeTensors {
        files: g.paths,
        maps: g.maps.into_iter().map(Some).collect(),
        tensors,
        assets,
    })
}
pub fn read_asset(path: &Path, name: &str) -> Result<Vec<u8>> {
    if path.is_dir() {
        return Ok(std::fs::read(path.join(name))?);
    }
    let g = Gguf::open(path)?;
    let base = index(&g)?.context("not a complete Volta GGUF")?;
    Ok(
        g.bytes(g.get(base.assets.get(name).context("embedded asset missing")?)?)
            .to_vec(),
    )
}

pub(super) struct BaseSource {
    pub(super) st: SafeTensors,
    pub(super) index: CompleteIndex,
    pub(super) assets: BTreeMap<String, std::path::PathBuf>,
}
impl BaseSource {
    fn open(dir: &Path) -> Result<Self> {
        let st = SafeTensors::open(dir, &[])?;
        let mut tensors = BTreeMap::new();
        for (n, t) in &st.tensors {
            ensure!(
                !(n.starts_with("layers.") && n.contains(".ffn.experts.")),
                "base must omit main routed experts: {n}"
            );
            let d = BaseTensor {
                tensor: format!("base.{n}"),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
            };
            ensure!(
                byte_len(&d)? == t.end - t.start,
                "unexpected original tensor byte size {n}"
            );
            tensors.insert(n.clone(), d);
        }
        fn walk(
            root: &Path,
            dir: &Path,
            out: &mut BTreeMap<String, std::path::PathBuf>,
        ) -> Result<()> {
            for e in std::fs::read_dir(dir)? {
                let e = e?;
                let p = e.path();
                let name = e.file_name();
                let name = name.to_string_lossy();
                if name.starts_with('.') {
                    continue;
                }
                if e.file_type()?.is_dir() {
                    walk(root, &p, out)?;
                } else if std::fs::metadata(&p)?.is_file() && !name.ends_with(".safetensors") {
                    ensure!(
                        std::fs::metadata(&p)?.len() <= 64 << 20,
                        "base sidecar too large: {}",
                        p.display()
                    );
                    out.insert(
                        p.strip_prefix(root)?.to_string_lossy().replace('\\', "/"),
                        p,
                    );
                }
            }
            Ok(())
        }
        let mut assets = BTreeMap::new();
        walk(dir, dir, &mut assets)?;
        ensure!(
            assets
                .values()
                .map(|p| std::fs::metadata(p).map(|m| m.len()))
                .collect::<std::io::Result<Vec<_>>>()?
                .iter()
                .sum::<u64>()
                <= 256 << 20,
            "base sidecars exceed256MiB"
        );
        for required in ["config.json", "dsx-engram.json", "tokenizer.json"] {
            ensure!(assets.contains_key(required), "base missing {required}");
        }
        let index = CompleteIndex {
            version: 1,
            tensors,
            assets: assets
                .keys()
                .map(|n| (n.clone(), format!("asset.{n}")))
                .collect(),
        };
        Ok(Self { st, index, assets })
    }
}

/// One complete GGUF. Expert conversion and base byte copying both stream bounded chunks.
pub fn export(pack: &Path, base: &Path, out: &Path, dim: usize, inter: usize) -> Result<()> {
    let source = BaseSource::open(base)?;
    let plan: Pack = serde_json::from_slice(&std::fs::read(pack.join("manifest.json"))?)?;
    plan.validate()?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(base.join("config.json"))?)?;
    for (name, expected) in [
        ("hidden_size", dim),
        ("moe_intermediate_size", inter),
        ("num_hidden_layers", plan.tiers.len()),
        ("n_routed_experts", plan.tiers[0].len()),
    ] {
        ensure!(
            config["text_config"][name].as_u64() == Some(expected as u64),
            "base config {name} differs from expert pack/export dimensions"
        );
    }
    super::export_impl(pack, out, dim, inter, Some(source))
}
pub(super) fn write_table(
    b: &BaseSource,
    w: &mut (impl Write + Seek),
    offset: &mut u64,
) -> Result<()> {
    for t in b.index.tensors.values() {
        let (ty, dims) = ggml_layout(t)?;
        string(w, &t.tensor)?;
        w.write_all(&(dims.len() as u32).to_le_bytes())?;
        for d in dims {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&ty.to_le_bytes())?;
        w.write_all(&offset.to_le_bytes())?;
        *offset = (*offset + byte_len(t)? as u64 + 31) & !31;
    }
    for (n, tensor) in &b.index.assets {
        let len = std::fs::metadata(&b.assets[n])?.len();
        string(w, tensor)?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&len.to_le_bytes())?;
        w.write_all(&24u32.to_le_bytes())?;
        w.write_all(&offset.to_le_bytes())?;
        *offset = (*offset + len + 31) & !31;
    }
    Ok(())
}
pub(super) fn write_payload(b: &BaseSource, w: &mut (impl Write + Seek)) -> Result<()> {
    for name in b.index.tensors.keys() {
        let t = b.st.get(name)?;
        let bytes = b.st.bytes(t);
        for (i, chunk) in bytes.chunks(8 << 20).enumerate() {
            w.write_all(chunk)?;
            release_pages(
                b.st.maps[t.file].as_ref().unwrap(),
                t.start + i * (8 << 20),
                chunk.len(),
            )?;
        }
        pad(w)?;
    }
    let mut buffer = vec![0u8; 8 << 20];
    use std::io::Read;
    for path in b.assets.values() {
        let mut f = File::open(path)?;
        loop {
            let n = f.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            w.write_all(&buffer[..n])?;
        }
        pad(w)?;
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteVerificationReceipt {
    pub schema: String,
    pub status: String,
    pub expert_verification: VerificationReceipt,
    pub base_tensors: usize,
    pub assets: usize,
    pub base_tensor_bytes_compared: u64,
    pub asset_bytes_compared: u64,
    pub source_base_payload_sha256: String,
    pub embedded_base_payload_sha256: String,
    pub base_metadata_sha256: String,
    pub io_chunk_bytes: usize,
    pub total_seconds: f64,
    pub gpu_inference_verified: bool,
}
pub fn verify(
    pack: &Path,
    base: &Path,
    out: &Path,
    dim: usize,
    inter: usize,
) -> Result<CompleteVerificationReceipt> {
    let start = std::time::Instant::now();
    let expert = super::verify(pack, out, dim, inter)?;
    let source = BaseSource::open(base)?;
    let g = Gguf::open(out)?;
    ensure!(g.paths.len() == 1, "complete model must be one GGUF file");
    let embedded = index(&g)?.context("complete GGUF base index missing")?;
    ensure!(
        serde_json::to_value(&source.index)? == serde_json::to_value(&embedded)?,
        "embedded base tensor/asset metadata differs from source"
    );
    let (mut source_hash, mut embedded_hash) = (Sha256::new(), Sha256::new());
    let (mut tensor_bytes, mut asset_bytes) = (0u64, 0u64);
    let mut last = std::time::Instant::now();
    let bind = |a: &mut Sha256, b: &mut Sha256, tag: &[u8], d: &[u8]| {
        for h in [a, b] {
            h.update(tag);
            h.update((d.len() as u64).to_le_bytes());
            h.update(d);
        }
    };
    for (name, desc) in &source.index.tensors {
        let st = source.st.get(name)?;
        let gt = g.get(&desc.tensor)?;
        let original = source.st.bytes(st);
        let actual = g.bytes(gt);
        bind(
            &mut source_hash,
            &mut embedded_hash,
            b"TENSOR",
            &serde_json::to_vec(&(name, desc))?,
        );
        ensure!(
            original.len() == actual.len(),
            "base tensor byte length differs {name}"
        );
        for offset in (0..original.len()).step_by(8 << 20) {
            let end = (offset + (8 << 20)).min(original.len());
            let (a, b) = (&original[offset..end], &actual[offset..end]);
            ensure!(a == b, "base tensor bytes differ: {name}, offset {offset}");
            source_hash.update(a);
            embedded_hash.update(b);
            tensor_bytes += a.len() as u64;
            release_pages(
                source.st.maps[st.file].as_ref().unwrap(),
                st.start + offset,
                a.len(),
            )?;
            release_pages(&g.maps[gt.shard], gt.offset as usize + offset, b.len())?;
            if last.elapsed().as_secs() >= 10 {
                eprintln!(
                    "verify-full: {:.3} GB base tensor bytes compared, {:.1}s",
                    tensor_bytes as f64 / 1e9,
                    start.elapsed().as_secs_f64()
                );
                last = std::time::Instant::now();
            }
        }
    }
    use std::io::Read;
    let mut buffer = vec![0u8; 8 << 20];
    for (name, path) in &source.assets {
        let gt = g.get(&embedded.assets[name])?;
        let actual = g.bytes(gt);
        bind(
            &mut source_hash,
            &mut embedded_hash,
            b"ASSET",
            &serde_json::to_vec(&(name, actual.len()))?,
        );
        let mut f = File::open(path)?;
        let mut offset = 0;
        loop {
            let n = f.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            ensure!(
                offset + n <= actual.len() && buffer[..n] == actual[offset..offset + n],
                "asset bytes differ {name}"
            );
            source_hash.update(&buffer[..n]);
            embedded_hash.update(&actual[offset..offset + n]);
            asset_bytes += n as u64;
            release_pages(&g.maps[gt.shard], gt.offset as usize + offset, n)?;
            offset += n;
        }
        ensure!(offset == actual.len(), "asset length differs {name}");
    }
    let source_digest = hex(&source_hash.finalize());
    let embedded_digest = hex(&embedded_hash.finalize());
    ensure!(
        source_digest == embedded_digest,
        "base payload hash mismatch"
    );
    Ok(CompleteVerificationReceipt {
        schema: "volta-complete-verify-v1".into(),
        status: "exact".into(),
        expert_verification: expert,
        base_tensors: embedded.tensors.len(),
        assets: embedded.assets.len(),
        base_tensor_bytes_compared: tensor_bytes,
        asset_bytes_compared: asset_bytes,
        source_base_payload_sha256: source_digest,
        embedded_base_payload_sha256: embedded_digest,
        base_metadata_sha256: hex(&Sha256::digest(serde_json::to_vec(&embedded)?)),
        io_chunk_bytes: 8 << 20,
        total_seconds: start.elapsed().as_secs_f64(),
        gpu_inference_verified: false,
    })
}

/// Load actual HF model/draft/tables through the embedded mmap views. No GPU or
/// full tensor dequantization/extraction is performed; existing small model vectors
/// (norms, routing matrices, hyper-connection parameters) use the normal HF loader.
pub fn smoke(path: &Path) -> Result<serde_json::Value> {
    let start = std::time::Instant::now();
    let q = Portable::open(path)?;
    ensure!(
        q.gguf.paths.len() == 1,
        "complete model must be one GGUF file"
    );
    let m = crate::model::Model::load_hf(path)?;
    q.validate_model(
        m.cfg.dim,
        m.cfg.inter,
        m.cfg.n_layers,
        m.cfg.n_experts,
        q.pack().ngpu,
    )?;
    let crate::model::Source::Hf(st) = &m.src else {
        anyhow::bail!("expected HF-compatible embedded source")
    };
    let draft = crate::dspark::Dspark::load(&m, st, path)?;
    let tok = tokenizers::Tokenizer::from_bytes(st.asset("tokenizer.json")?)
        .map_err(|e| anyhow::anyhow!("embedded tokenizer: {e}"))?;
    let encoding = tok
        .encode("Hello", false)
        .map_err(|e| anyhow::anyhow!("tokenizer smoke: {e}"))?;
    let mut engram = vec![];
    for li in 0..m.cfg.engram_layers.len() {
        match m.engram.as_ref().context("missing embedded Engram")?.layer(li,m.cfg.engram_dim)? {
            crate::model::EngramLayer::Fp8{w,s,dim}=>engram.push(serde_json::json!({"layer":m.cfg.engram_layers[li],"dim":dim,"weight_bytes":w.len(),"scale_bytes":s.len()})),
            _=>anyhow::bail!("expected original FP8 Engram tensors"),
        }
    }
    let base = index(&q.gguf)?.context("missing complete index")?;
    let peak = std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .unwrap_or("unavailable")
        .to_string();
    Ok(
        serde_json::json!({"schema":"volta-complete-load-smoke-v1","status":"loaded","dim":m.cfg.dim,"inter":m.cfg.inter,"layers":m.layers.len(),"experts_per_layer":m.cfg.n_experts,
        "base_tensors":st.tensors.len(),"assets":base.assets.len(),"vision_tensors":st.tensors.keys().filter(|n|n.starts_with("vision.")||n.starts_with("aligner.")||n.starts_with("image_")).count(),
        "draft_layers":draft.layers.len(),"draft_block":draft.block,"engram":engram,"tokenizer_vocab":tok.get_vocab_size(true),"tokenizer_sample_ids":encoding.get_ids(),
        "embedding_shape":[m.embed.rows,m.embed.cols],"head_shape":[m.head.rows,m.head.cols],"seconds":start.elapsed().as_secs_f64(),"peak_rss_proc_status":peak,"gpu_inference_verified":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_preserves_typed_tensors_assets_and_hf_views() -> Result<()> {
        let (tmp, _, _) = super::super::tests::fixture()?;
        let base = tmp.0.join("base");
        std::fs::create_dir(&base)?;
        let tensors = vec![
            ("embed.weight", "BF16", vec![32, 32], vec![0x3fu8; 2048]),
            (
                "layers.0.attn.wq_a.weight",
                "F8_E4M3",
                vec![32, 32],
                (0..1024).map(|i| (i % 119) as u8).collect(),
            ),
            ("layers.0.attn.wq_a.scale", "F8_E8M0", vec![1, 1], vec![127]),
            (
                "mtp.0.example.weight",
                "I8",
                vec![32, 16],
                vec![0x73u8; 512],
            ),
            ("mtp.0.example.scale", "F8_E8M0", vec![32, 1], vec![126; 32]),
            ("vision.patch.weight", "BF16", vec![2, 3, 4], vec![0x12; 48]),
            (
                "example.scalar",
                "F32",
                vec![],
                1.25f32.to_le_bytes().to_vec(),
            ),
        ];
        let mut header = serde_json::Map::new();
        let mut payload = vec![];
        let mut wm = serde_json::Map::new();
        for (n, dtype, shape, bytes) in &tensors {
            let start = payload.len();
            payload.extend(bytes);
            header.insert(n.to_string(),serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}));
            wm.insert(n.to_string(), serde_json::json!("model.safetensors"));
        }
        let mut f = File::create(base.join("model.safetensors"))?;
        let json = serde_json::to_vec(&header)?;
        f.write_all(&(json.len() as u64).to_le_bytes())?;
        f.write_all(&json)?;
        f.write_all(&payload)?;
        std::fs::write(
            base.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":wm}))?,
        )?;
        for (n,bytes) in [("config.json",b"{\"text_config\":{\"hidden_size\":96,\"moe_intermediate_size\":128,\"num_hidden_layers\":2,\"n_routed_experts\":3}}".as_slice()),("dsx-engram.json",b"{}"),("tokenizer.json",b"{}"),("empty.txt",b"")] {std::fs::write(base.join(n),bytes)?;}
        let out = tmp.0.join("complete.gguf");
        export(&tmp.0, &base, &out, 96, 128)?;
        let q = Portable::open(&out)?;
        assert_eq!(q.matrices().len(), 24);
        let original = SafeTensors::open(&base, &[])?;
        let embedded = SafeTensors::open(&out, &[])?;
        assert_eq!(original.tensors.len(), embedded.tensors.len());
        for (n, _, shape, bytes) in &tensors {
            let t = embedded.get(n)?;
            assert_eq!(&t.shape, shape);
            assert_eq!(embedded.bytes(t), bytes);
        }
        assert_eq!(
            read_asset(&out, "config.json")?,
            std::fs::read(base.join("config.json"))?
        );
        assert_eq!(read_asset(&out, "empty.txt")?, Vec::<u8>::new());
        for n in ["layers.0.attn.wq_a.weight", "mtp.0.example.weight"] {
            let a = crate::model::hf_mat_pub(&original, n)?;
            let b = crate::model::hf_mat_pub(&embedded, n)?;
            assert_eq!((a.ty, a.rows, a.cols), (b.ty, b.rows, b.cols));
            let mut x = vec![0f32; a.cols];
            let mut y = x.clone();
            a.dequant(0, &mut x);
            b.dequant(0, &mut y);
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let receipt = verify(&tmp.0, &base, &out, 96, 128)?;
        assert_eq!(receipt.status, "exact");
        assert_eq!(receipt.base_tensors, tensors.len());
        assert_eq!(
            receipt.source_base_payload_sha256,
            receipt.embedded_base_payload_sha256
        );
        // Base payload corruption must be rejected independently of the expert verifier.
        let g = Gguf::open(&out)?;
        let offset = g.get("base.embed.weight")?.offset;
        drop(g);
        use std::io::{Read, SeekFrom};
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&out)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut b = [0u8];
        f.read_exact(&mut b)?;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(&[b[0] ^ 1])?;
        f.sync_all()?;
        assert!(
            verify(&tmp.0, &base, &out, 96, 128)
                .unwrap_err()
                .to_string()
                .contains("base tensor bytes differ")
        );
        Ok(())
    }
}
