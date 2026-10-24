//! Lossless interchange for Volta format-4 LUT experts. GGUF I8 tensors are byte containers,
//! not registered ggml CB3/CB2 quantization types. No floating-point conversion occurs here.
pub mod complete;
use crate::gguf::{GType, Gguf};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Seek, Write},
    path::Path,
};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Pack {
    pub format: u32,
    pub ngpu: usize,
    pub etp: usize,
    pub expert_gb: f64,
    pub tiers: Vec<Vec<u8>>,
    pub owner: Vec<Vec<u8>>,
}
impl Pack {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.format == 4
                && self.etp > 0
                && self.ngpu > 0
                && self.ngpu <= 256
                && self.ngpu % self.etp == 0,
            "invalid format-4 topology"
        );
        ensure!(
            !self.tiers.is_empty() && self.tiers.len() == self.owner.len(),
            "invalid layer plan"
        );
        let ne = self.tiers[0].len();
        ensure!(ne > 0, "empty expert plan");
        for (t, o) in self.tiers.iter().zip(&self.owner) {
            ensure!(
                t.len() == ne
                    && o.len() == ne
                    && t.iter().all(|&v| v <= 2)
                    && o.iter().all(|&v| (v as usize) < self.ngpu / self.etp),
                "invalid expert plan"
            );
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    pub layer: usize,
    pub expert: usize,
    pub part: usize,
    /// gu = concatenated gate/up rows; down = intermediate columns of down.
    pub role: String,
    pub rows: usize,
    pub cols: usize,
    pub bits: usize,
    /// Exact source multiplier, including signed zero / subnormal bit patterns.
    pub mul_bits: u32,
}
impl Matrix {
    pub fn name(&self) -> String {
        format!(
            "layers.{}.experts.{}.part.{}.{}",
            self.layer, self.expert, self.part, self.role
        )
    }
    pub fn sizes(&self) -> Result<(usize, usize)> {
        ensure!(
            matches!(self.bits, 2..=4)
                && self.rows > 0
                && self.cols > 0
                && self.rows % 32 == 0
                && self.cols % 32 == 0,
            "invalid matrix dimensions or code width"
        );
        let n = self
            .rows
            .checked_mul(self.cols)
            .context("matrix size overflow")?;
        Ok((
            n.checked_mul(self.bits).context("matrix byte overflow")? / 8,
            n / 32,
        ))
    }
}
#[derive(Serialize, Deserialize)]
struct Index {
    version: u32,
    dim: usize,
    inter: usize,
    pack: Pack,
    matrices: Vec<Matrix>,
}

fn lane(r: usize) -> usize {
    4 * (r >> 3) + (r & 3) + 16 * ((r >> 2) & 1)
}

/// Convert between kernel word/lane order and ordinary little-endian row-major code bits.
/// The final CB3 pair lives in the top two bits of each of the three native words.
pub fn transform(
    m: &Matrix,
    words: &[u8],
    scales: &[u8],
    to_native: bool,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let (nw, ns) = m.sizes()?;
    ensure!(
        words.len() == nw && scales.len() == ns,
        "wrong matrix payload length"
    );
    let (mut ow, mut os) = (vec![0u8; nw], vec![0u8; ns]);
    let units = m.cols / 32;
    for r in 0..m.rows {
        for u in 0..units {
            let group = (r / 32) * units + u;
            let l = lane(r % 32);
            let row_group = r * units + u;
            let mut q = [0u32; 4];
            let linear_start = row_group * m.bits * 4;
            if to_native {
                if m.bits == 3 {
                    let mut raw = [0u8; 16];
                    raw[..12].copy_from_slice(&words[linear_start..linear_start + 12]);
                    let packed = u128::from_le_bytes(raw);
                    for p in 0..15 {
                        q[p / 5] |= (((packed >> (6 * p)) & 63) as u32) << (6 * (p % 5));
                    }
                    let last = (packed >> 90) as u32;
                    for z in 0..3 {
                        q[z] |= ((last >> (2 * z)) & 3) << 30;
                    }
                } else {
                    for w in 0..m.bits {
                        let o = linear_start + w * 4;
                        q[w] = u32::from_le_bytes(words[o..o + 4].try_into()?);
                    }
                }
                for w in 0..m.bits {
                    let o = ((group * m.bits + w) * 32 + l) * 4;
                    ow[o..o + 4].copy_from_slice(&q[w].to_le_bytes());
                }
            } else {
                for w in 0..m.bits {
                    let o = ((group * m.bits + w) * 32 + l) * 4;
                    q[w] = u32::from_le_bytes(words[o..o + 4].try_into()?);
                }
                if m.bits == 3 {
                    let mut packed = 0u128;
                    for p in 0..15 {
                        let pair = (q[p / 5] >> (6 * (p % 5))) & 63;
                        packed |= (pair as u128) << (6 * p);
                    }
                    let last = (q[0] >> 30) | ((q[1] >> 30) << 2) | ((q[2] >> 30) << 4);
                    packed |= (last as u128) << 90;
                    ow[linear_start..linear_start + 12]
                        .copy_from_slice(&packed.to_le_bytes()[..12]);
                } else {
                    for w in 0..m.bits {
                        let o = linear_start + w * 4;
                        ow[o..o + 4].copy_from_slice(&q[w].to_le_bytes());
                    }
                }
            }
            if to_native {
                os[group * 32 + l] = scales[row_group];
            } else {
                os[row_group] = scales[group * 32 + l];
            }
        }
    }
    Ok((ow, os))
}

pub struct Portable {
    gguf: Gguf,
    index: Index,
    lookup: std::collections::HashMap<(usize, usize, usize, String), usize>,
}
impl Portable {
    pub fn open(path: &Path) -> Result<Self> {
        let mut gguf = Gguf::open(path)?;
        ensure!(
            gguf.kv.get("general.architecture").and_then(|v| v.as_str()) == Some("volta-experts"),
            "GGUF is not a volta-experts container"
        );
        let raw = gguf
            .kv
            .get("volta.index")
            .and_then(|x| x.as_str())
            .context("missing volta.index")?;
        let index: Index = serde_json::from_str(raw)?;
        // Keep the parsed directory, not a second copy of the large JSON index.
        gguf.kv.remove("volta.index");
        ensure!(index.version == 1, "unsupported Volta quant version");
        index.pack.validate()?;
        ensure!(
            index.inter > 0 && index.inter % index.pack.etp == 0,
            "invalid intermediate dimension"
        );
        let expected = index.pack.tiers.iter().map(Vec::len).sum::<usize>() * index.pack.etp * 2;
        let base = complete::index(&gguf)?;
        let extras = base
            .as_ref()
            .map(|b| b.tensors.len() + b.assets.len())
            .unwrap_or(0);
        ensure!(index.matrices.len() == expected, "missing matrices");
        ensure!(
            gguf.order.len() == expected * 2 + extras
                && gguf.tensors.len() == expected * 2 + extras,
            "extra or duplicate portable tensors"
        );
        let mut names = std::collections::HashSet::new();
        for m in &index.matrices {
            ensure!(
                m.layer < index.pack.tiers.len()
                    && m.expert < index.pack.tiers[m.layer].len()
                    && m.part < index.pack.etp,
                "matrix outside plan"
            );
            let bits = [4, 3, 2][index.pack.tiers[m.layer][m.expert] as usize];
            let inter = index.inter / index.pack.etp;
            let (rows, cols) = match m.role.as_str() {
                "gu" => (2 * inter, index.dim),
                "down" => (index.dim, inter),
                _ => anyhow::bail!("unknown matrix role"),
            };
            ensure!(
                m.bits == bits && m.rows == rows && m.cols == cols && names.insert(m.name()),
                "inconsistent or duplicate matrix"
            );
            let (nw, ns) = m.sizes()?;
            for (suffix, len) in [("codes", nw), ("scales", ns)] {
                let t = gguf.get(&format!("{}.{}", m.name(), suffix))?;
                ensure!(
                    t.ty == GType::I8
                        && t.dims == [len as u64]
                        && t.offset % 32 == 0
                        && gguf.bytes(t).len() == len,
                    "invalid byte tensor"
                );
            }
        }
        let mut extents: Vec<_> = gguf
            .tensors
            .values()
            .map(|t| (t.shard, t.offset, t.offset + t.bytes))
            .collect();
        extents.sort_unstable();
        for pair in extents.windows(2) {
            ensure!(
                pair[0].0 != pair[1].0 || pair[0].2 <= pair[1].1,
                "overlapping portable tensors"
            );
        }
        drop(extents);
        let lookup = index
            .matrices
            .iter()
            .enumerate()
            .map(|(i, m)| ((m.layer, m.expert, m.part, m.role.clone()), i))
            .collect();
        Ok(Self {
            gguf,
            index,
            lookup,
        })
    }
    pub fn validate_model(
        &self,
        dim: usize,
        inter: usize,
        layers: usize,
        experts: usize,
        ngpu: usize,
    ) -> Result<()> {
        ensure!(
            self.dimensions() == (dim, inter),
            "portable expert dimensions {:?}, model expects ({dim}, {inter})",
            self.dimensions()
        );
        ensure!(
            self.index.pack.ngpu == ngpu,
            "portable experts require {} GPUs, engine has {ngpu}",
            self.index.pack.ngpu
        );
        ensure!(
            self.index.pack.tiers.len() == layers
                && self.index.pack.tiers.iter().all(|t| t.len() == experts),
            "portable expert layer/count plan differs from model ({layers} layers, {experts} experts per layer)"
        );
        Ok(())
    }
    pub fn pack(&self) -> &Pack {
        &self.index.pack
    }
    pub fn dimensions(&self) -> (usize, usize) {
        (self.index.dim, self.index.inter)
    }
    pub fn matrix(&self, layer: usize, expert: usize, part: usize, role: &str) -> Result<&Matrix> {
        let i = self
            .lookup
            .get(&(layer, expert, part, role.into()))
            .context("missing portable matrix")?;
        Ok(&self.index.matrices[*i])
    }
    pub fn matrices(&self) -> &[Matrix] {
        &self.index.matrices
    }
    pub fn native(&self, m: &Matrix) -> Result<(Vec<u8>, Vec<u8>)> {
        let w = self
            .gguf
            .bytes(self.gguf.get(&format!("{}.codes", m.name()))?);
        let s = self
            .gguf
            .bytes(self.gguf.get(&format!("{}.scales", m.name()))?);
        transform(m, w, s, true)
    }
    /// Discard read-only mapped payload pages after consumption; the file stays unchanged.
    pub fn release_payload(&self, m: &Matrix) -> Result<()> {
        for suffix in ["codes", "scales"] {
            let t = self.gguf.get(&format!("{}.{}", m.name(), suffix))?;
            release_pages(
                &self.gguf.maps[t.shard],
                t.offset as usize,
                t.bytes as usize,
            )?;
        }
        Ok(())
    }
}

fn u64_at(b: &[u8], pos: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        b.get(pos..pos + 8)
            .context("truncated native header")?
            .try_into()?,
    ))
}
struct Source {
    m: Matrix,
    gpu: usize,
    wo: usize,
    so: usize,
}
struct NativePack {
    pack: Pack,
    maps: Vec<memmap2::Mmap>,
    src: Vec<Source>,
}
impl NativePack {
    fn open(dir: &Path, dim: usize, inter: usize) -> Result<Self> {
        let pack: Pack = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
        pack.validate()?;
        ensure!(
            inter > 0 && inter % pack.etp == 0,
            "invalid intermediate dimension"
        );
        let ip = inter / pack.etp;
        let mut maps = vec![];
        let mut src = vec![];
        for gpu in 0..pack.ngpu {
            let f = File::open(dir.join(format!("gpu{gpu}.bin")))?;
            // SAFETY: only reads an existing pack; caller must not mutate it during conversion.
            let b = unsafe { memmap2::Mmap::map(&f)? };
            let n = usize::try_from(u64_at(&b, 0)?)?;
            let hdr = n
                .checked_mul(72)
                .and_then(|n| n.checked_add(8))
                .context("header overflow")?;
            ensure!(hdr <= b.len(), "truncated native entries");
            let mut entries = vec![];
            let (mut nw, mut ns) = (0usize, 0usize);
            let mut seen = std::collections::HashSet::new();
            for i in 0..n {
                let mut v = [0u64; 9];
                for j in 0..9 {
                    v[j] = u64_at(&b, 8 + i * 72 + j * 8)?;
                }
                let (l, e) = (usize::try_from(v[0])?, usize::try_from(v[1])?);
                ensure!(
                    l < pack.tiers.len() && e < pack.tiers[l].len() && seen.insert((l, e)),
                    "invalid native expert index"
                );
                ensure!(
                    pack.owner[l][e] as usize == gpu / pack.etp,
                    "native ownership mismatch"
                );
                let bits = [4, 3, 2][pack.tiers[l][e] as usize];
                let kind = [100, 101, 102][pack.tiers[l][e] as usize];
                ensure!(
                    v[2] == kind | (kind << 8),
                    "native kind differs from tier plan"
                );
                for (role, rows, cols, wi, si, mi) in
                    [("gu", 2 * ip, dim, 3, 4, 7), ("down", dim, ip, 5, 6, 8)]
                {
                    ensure!(v[mi] <= u32::MAX as u64, "invalid multiplier bits");
                    let m = Matrix {
                        layer: l,
                        expert: e,
                        part: gpu % pack.etp,
                        role: role.into(),
                        rows,
                        cols,
                        bits,
                        mul_bits: v[mi] as u32,
                    };
                    let (w, s) = m.sizes()?;
                    nw = nw.checked_add(w).context("arena overflow")?;
                    ns = ns.checked_add(s).context("arena overflow")?;
                    entries.push(Source {
                        m,
                        gpu,
                        wo: usize::try_from(v[wi])?
                            .checked_mul(128)
                            .context("offset overflow")?,
                        so: usize::try_from(v[si])?,
                    });
                }
            }
            let expected = pack
                .owner
                .iter()
                .map(|row| {
                    row.iter()
                        .filter(|&&o| o as usize == gpu / pack.etp)
                        .count()
                })
                .sum::<usize>();
            ensure!(seen.len() == expected, "native pack missing experts");
            ensure!(
                hdr.checked_add(nw).and_then(|v| v.checked_add(ns)) == Some(b.len()),
                "wrong native arena size"
            );
            // Validate that every arena byte belongs to exactly one matrix, including unordered headers.
            for scales in [false, true] {
                let mut ranges: Vec<_> = entries
                    .iter()
                    .map(|s| {
                        let z = s.m.sizes().unwrap();
                        let start = if scales { s.so } else { s.wo };
                        (start, if scales { z.1 } else { z.0 })
                    })
                    .collect();
                ranges.sort_unstable();
                let mut end = 0;
                for (start, len) in ranges {
                    ensure!(start == end, "overlapping or gapped native arena");
                    end += len;
                }
            }
            for mut s in entries {
                s.wo += hdr;
                s.so += hdr + nw;
                src.push(s);
            }
            maps.push(b);
        }
        src.sort_by_key(|s| (s.m.layer, s.m.expert, s.m.part, s.m.role.clone()));
        Ok(Self { pack, maps, src })
    }
}
/// Export already quantized format-4 GPU files. `out` must not exist; failed writes never replace it.
pub fn export(dir: &Path, out: &Path, dim: usize, inter: usize) -> Result<()> {
    export_impl(dir, out, dim, inter, None)
}
fn export_impl(
    dir: &Path,
    out: &Path,
    dim: usize,
    inter: usize,
    base: Option<complete::BaseSource>,
) -> Result<()> {
    let NativePack { pack, maps, src } = NativePack::open(dir, dim, inter)?;
    let index = Index {
        version: 1,
        dim,
        inter,
        pack,
        matrices: src.iter().map(|s| s.m.clone()).collect(),
    };
    let json = serde_json::to_string(&index)?;
    let mut w = BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out)?,
    );
    let result = (|| -> Result<()> {
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        let extra = base
            .as_ref()
            .map(|b| b.index.tensors.len() + b.index.assets.len())
            .unwrap_or(0);
        w.write_all(&(src.len() as u64 * 2 + extra as u64).to_le_bytes())?;
        w.write_all(&(3u64 + base.is_some() as u64).to_le_bytes())?;
        for (k, v) in [
            ("general.architecture", "volta-experts"),
            ("volta.index", json.as_str()),
        ] {
            string(&mut w, k)?;
            w.write_all(&8u32.to_le_bytes())?;
            string(&mut w, v)?;
        }
        if let Some(b) = &base {
            string(&mut w, "volta.base")?;
            w.write_all(&8u32.to_le_bytes())?;
            string(&mut w, &serde_json::to_string(&b.index)?)?;
        }
        string(&mut w, "general.alignment")?;
        w.write_all(&4u32.to_le_bytes())?;
        w.write_all(&32u32.to_le_bytes())?;
        let mut offset = 0u64;
        for s in &src {
            let (nw, ns) = s.m.sizes()?;
            for (suffix, len) in [("codes", nw), ("scales", ns)] {
                string(&mut w, &format!("{}.{}", s.m.name(), suffix))?;
                w.write_all(&1u32.to_le_bytes())?;
                w.write_all(&(len as u64).to_le_bytes())?;
                w.write_all(&24u32.to_le_bytes())?;
                w.write_all(&offset.to_le_bytes())?;
                offset = (offset + len as u64 + 31) & !31;
            }
        }
        if let Some(b) = &base {
            complete::write_table(b, &mut w, &mut offset)?;
        }
        pad(&mut w)?;
        for s in &src {
            let (nw, ns) = s.m.sizes()?;
            let b = &maps[s.gpu];
            let (codes, scales) = transform(&s.m, &b[s.wo..s.wo + nw], &b[s.so..s.so + ns], false)?;
            w.write_all(&codes)?;
            pad(&mut w)?;
            w.write_all(&scales)?;
            pad(&mut w)?;
            release_pages(b, s.wo, nw)?;
            release_pages(b, s.so, ns)?;
        }
        if let Some(b) = &base {
            complete::write_payload(b, &mut w)?;
        }
        w.flush()?;
        w.get_ref().sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        drop(w);
        let _ = std::fs::remove_file(out);
    }
    result
}
// memmap2 rounds the address down to a page boundary. MADV_DONTNEED drops only
// this process's file-backed PTEs; these read-only mappings have no dirty anonymous
// data, and later reads fault the original file bytes back in. No active mutable alias exists.
pub(crate) fn release_pages(map: &memmap2::Mmap, offset: usize, len: usize) -> Result<()> {
    #[cfg(unix)]
    unsafe {
        map.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, offset, len)?;
    }
    #[cfg(not(unix))]
    let _ = (map, offset, len);
    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerificationReceipt {
    pub schema: String,
    pub status: String,
    pub source_pack: String,
    pub gguf_shards: Vec<String>,
    pub dim: usize,
    pub inter: usize,
    pub ngpu: usize,
    pub etp: usize,
    pub layers: usize,
    pub experts_per_layer: usize,
    pub matrices: usize,
    pub matrices_by_tier: [usize; 3],
    pub code_bytes_compared: u64,
    pub scale_bytes_compared: u64,
    pub source_file_bytes: u64,
    pub gguf_file_bytes: u64,
    pub max_matrix_temporary_bytes: usize,
    pub source_manifest_sha256: String,
    pub metadata_sha256: String,
    pub source_native_payload_sha256: String,
    pub reconstructed_native_payload_sha256: String,
    pub open_seconds: f64,
    pub compare_seconds: f64,
    pub total_seconds: f64,
    pub gpu_inference_verified: bool,
}
/// Exhaustively compare a GGUF reconstruction with the unchanged source native pack.
/// Hashes use sorted matrix identities followed by native code and scale bytes; they
/// are logical-payload hashes rather than hashes of the containing GPU/GGUF files.
pub fn verify(dir: &Path, gguf: &Path, dim: usize, inter: usize) -> Result<VerificationReceipt> {
    let start = std::time::Instant::now();
    let source = NativePack::open(dir, dim, inter).context("validate source native pack")?;
    let q = Portable::open(gguf).context("validate portable GGUF")?;
    q.validate_model(
        dim,
        inter,
        source.pack.tiers.len(),
        source.pack.tiers[0].len(),
        source.pack.ngpu,
    )?;
    ensure!(
        &source.pack == q.pack(),
        "portable manifest differs from source topology, tier, ownership or budget metadata"
    );
    ensure!(
        source.src.len() == q.matrices().len(),
        "source/portable matrix counts differ"
    );
    let open_seconds = start.elapsed().as_secs_f64();
    let mut src_hash = Sha256::new();
    let mut dst_hash = Sha256::new();
    let mut meta_hash = Sha256::new();
    meta_hash.update(serde_json::to_vec(&source.pack)?);
    let (mut code_bytes, mut scale_bytes, mut max_temp) = (0u64, 0u64, 0usize);
    let mut tiers = [0usize; 3];
    let mut last_progress = std::time::Instant::now();
    for (i, s) in source.src.iter().enumerate() {
        let m = q.matrix(s.m.layer, s.m.expert, s.m.part, &s.m.role)?;
        ensure!(
            m == &s.m,
            "portable matrix metadata differs from source: {}",
            s.m.name()
        );
        let descriptor = serde_json::to_vec(&s.m)?;
        meta_hash.update((descriptor.len() as u64).to_le_bytes());
        meta_hash.update(&descriptor);
        // Include the descriptor in both payload digests to bind identical bytes to
        // their layer/expert/slice/role, code width, dimensions and multiplier bits.
        for h in [&mut src_hash, &mut dst_hash] {
            h.update((descriptor.len() as u64).to_le_bytes());
            h.update(&descriptor);
        }
        let (nw, ns) = m.sizes()?;
        let b = &source.maps[s.gpu];
        let expected_w = &b[s.wo..s.wo + nw];
        let expected_s = &b[s.so..s.so + ns];
        let (words, scales) = q.native(m)?;
        ensure!(
            words == expected_w,
            "native code bytes differ: {}",
            m.name()
        );
        ensure!(
            scales == expected_s,
            "native scale bytes differ: {}",
            m.name()
        );
        src_hash.update(expected_w);
        src_hash.update(expected_s);
        dst_hash.update(&words);
        dst_hash.update(&scales);
        code_bytes += nw as u64;
        scale_bytes += ns as u64;
        max_temp = max_temp.max(nw + ns);
        tiers[source.pack.tiers[m.layer][m.expert] as usize] += 1;
        q.release_payload(m)?;
        release_pages(b, s.wo, nw)?;
        release_pages(b, s.so, ns)?;
        if last_progress.elapsed().as_secs() >= 10 {
            eprintln!(
                "verify: {}/{} matrices, {:.3} GB compared, {:.1} s",
                i + 1,
                source.src.len(),
                (code_bytes + scale_bytes) as f64 / 1e9,
                start.elapsed().as_secs_f64()
            );
            last_progress = std::time::Instant::now();
        }
    }
    let source_digest = hex(&src_hash.finalize());
    let reconstructed_digest = hex(&dst_hash.finalize());
    ensure!(
        source_digest == reconstructed_digest,
        "reconstructed payload checksum differs"
    );
    let elapsed = start.elapsed().as_secs_f64();
    Ok(VerificationReceipt {
        schema: "volta-quant-verify-v1".into(),
        status: "exact".into(),
        source_pack: dir.display().to_string(),
        gguf_shards: q
            .gguf
            .paths
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        dim,
        inter,
        ngpu: source.pack.ngpu,
        etp: source.pack.etp,
        layers: source.pack.tiers.len(),
        experts_per_layer: source.pack.tiers[0].len(),
        matrices: source.src.len(),
        matrices_by_tier: tiers,
        code_bytes_compared: code_bytes,
        scale_bytes_compared: scale_bytes,
        source_file_bytes: source.maps.iter().map(|m| m.len() as u64).sum(),
        gguf_file_bytes: q.gguf.maps.iter().map(|m| m.len() as u64).sum(),
        max_matrix_temporary_bytes: max_temp,
        source_manifest_sha256: hex(&Sha256::digest(std::fs::read(dir.join("manifest.json"))?)),
        metadata_sha256: hex(&meta_hash.finalize()),
        source_native_payload_sha256: source_digest,
        reconstructed_native_payload_sha256: reconstructed_digest,
        open_seconds,
        compare_seconds: elapsed - open_seconds,
        total_seconds: elapsed,
        gpu_inference_verified: false,
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn string(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
fn pad(w: &mut (impl Write + Seek)) -> Result<()> {
    let n = (32 - w.stream_position()? % 32) % 32;
    w.write_all(&[0u8; 32][..n as usize])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn matrix(bits: usize) -> Matrix {
        Matrix {
            layer: 0,
            expert: 0,
            part: 0,
            role: "gu".into(),
            rows: 64,
            cols: 96,
            bits,
            mul_bits: 0x3b123456,
        }
    }
    // Direct transcription of vlut_pack's pair algorithm, independent of transform's code-bit path.
    fn reference(m: &Matrix, codes: &[u8], scales: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let (nw, ns) = m.sizes().unwrap();
        let mut words = vec![0u8; nw];
        let mut out_sc = vec![0u8; ns];
        for r in 0..m.rows {
            for u in 0..m.cols / 32 {
                let c = &codes[r * m.cols + u * 32..][..32];
                let mut q = [0u32; 4];
                if m.bits == 4 {
                    for i in 0..32 {
                        q[i / 8] |= (c[i] as u32) << (4 * (i % 8));
                    }
                } else {
                    for p in 0..16 {
                        let pc = (c[2 * p] as u32) | ((c[2 * p + 1] as u32) << m.bits);
                        if m.bits == 2 {
                            q[p >> 3] |= pc << (4 * (p & 7));
                        } else if p < 15 {
                            q[p / 5] |= pc << (6 * (p % 5));
                        } else {
                            q[0] |= (pc & 3) << 30;
                            q[1] |= ((pc >> 2) & 3) << 30;
                            q[2] |= ((pc >> 4) & 3) << 30;
                        }
                    }
                }
                let base = (r / 32) * (m.cols / 32) + u;
                for z in 0..m.bits {
                    let o = ((base * m.bits + z) * 32 + lane(r % 32)) * 4;
                    words[o..o + 4].copy_from_slice(&q[z].to_le_bytes());
                }
                out_sc[base * 32 + lane(r % 32)] = scales[r * (m.cols / 32) + u];
            }
        }
        (words, out_sc)
    }
    fn linear_codes(m: &Matrix) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (nw, ns) = m.sizes().unwrap();
        let mut packed = vec![0u8; nw];
        let codes: Vec<u8> = (0..m.rows * m.cols)
            .map(|i| ((i * 13 + i / 7) & ((1 << m.bits) - 1)) as u8)
            .collect();
        for (i, &c) in codes.iter().enumerate() {
            for b in 0..m.bits {
                let bit = i * m.bits + b;
                packed[bit / 8] |= ((c >> b) & 1) << (bit % 8);
            }
        }
        let scales = (0..ns).map(|i| (i * 37) as u8).collect();
        (codes, packed, scales)
    }
    #[test]
    fn all_tiers_match_cuda_layout_and_roundtrip() -> Result<()> {
        for bits in [2, 3, 4] {
            let m = matrix(bits);
            let (codes, packed, scales) = linear_codes(&m);
            let expected = reference(&m, &codes, &scales);
            assert_eq!(transform(&m, &packed, &scales, true)?, expected);
            assert_eq!(
                transform(&m, &expected.0, &expected.1, false)?,
                (packed, scales)
            );
        }
        Ok(())
    }
    pub(super) struct Temp(pub(super) std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    pub(super) fn fixture() -> Result<(Temp, Pack, Vec<(Matrix, Vec<u8>, Vec<u8>)>)> {
        let dir = std::env::temp_dir().join(format!(
            "volta-quant-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&dir)?;
        let tmp = Temp(dir);
        let pack = Pack {
            format: 4,
            ngpu: 2,
            etp: 2,
            expert_gb: 0.01,
            tiers: vec![vec![0, 1, 2], vec![2, 1, 0]],
            owner: vec![vec![0; 3]; 2],
        };
        std::fs::write(tmp.0.join("manifest.json"), serde_json::to_vec(&pack)?)?;
        let mut expected = vec![];
        for gpu in 0..2 {
            let (mut words, mut scales, mut entries) = (vec![], vec![], vec![]);
            for l in 0..2 {
                for e in (0..3).rev() {
                    let kind = [100u64, 101, 102][pack.tiers[l][e] as usize];
                    let mut ent = vec![l as u64, e as u64, kind | (kind << 8)];
                    let mut muls = vec![];
                    for role in ["gu", "down"] {
                        let mut m = matrix([4, 3, 2][pack.tiers[l][e] as usize]);
                        m.layer = l;
                        m.expert = e;
                        m.part = gpu;
                        m.role = role.into();
                        m.rows = if role == "gu" { 128 } else { 96 };
                        m.cols = if role == "gu" { 96 } else { 64 };
                        m.mul_bits = 0x3b123456
                            + (gpu * 1000 + l * 100 + e * 10 + if role == "gu" { 0 } else { 1 })
                                as u32;
                        let (codes, _, ss) = linear_codes(&m);
                        let (ww, ss) = reference(&m, &codes, &ss);
                        ent.extend([words.len() as u64 / 128, scales.len() as u64]);
                        muls.push(m.mul_bits as u64);
                        words.extend(&ww);
                        scales.extend(&ss);
                        expected.push((m, ww, ss));
                    }
                    ent.extend(muls);
                    entries.push(ent);
                }
            }
            let mut f = File::create(tmp.0.join(format!("gpu{gpu}.bin")))?;
            f.write_all(&(entries.len() as u64).to_le_bytes())?;
            for ent in entries {
                for v in ent {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            f.write_all(&words)?;
            f.write_all(&scales)?;
        }
        Ok((tmp, pack, expected))
    }
    #[test]
    fn pack_export_load_preserves_all_bytes_and_slice_multipliers() -> Result<()> {
        let (tmp, pack, expected) = fixture()?;
        let out = tmp.0.join("experts.gguf");
        export(&tmp.0, &out, 96, 128)?;
        let q = Portable::open(&out)?;
        assert_eq!(q.pack().tiers, pack.tiers);
        assert_eq!(q.dimensions(), (96, 128));
        q.validate_model(96, 128, 2, 3, 2)?;
        assert!(q.validate_model(128, 128, 2, 3, 2).is_err());
        assert!(q.validate_model(96, 128, 2, 3, 4).is_err());
        assert!(q.validate_model(96, 128, 3, 3, 2).is_err());
        for (m, ww, ss) in expected {
            let got = q.matrix(m.layer, m.expert, m.part, &m.role)?;
            assert_eq!(got.mul_bits, m.mul_bits);
            assert_eq!(q.native(got)?, (ww, ss));
        }
        let shards = crate::gguf::split(&out, tmp.0.join("split").to_str().unwrap(), 2, None, &[])?;
        let split = Portable::open(&shards[0])?;
        for m in q.matrices() {
            assert_eq!(split.native(m)?, q.native(m)?);
        }
        let receipt = verify(&tmp.0, &out, 96, 128)?;
        assert_eq!(receipt.status, "exact");
        assert_eq!(receipt.matrices, 24);
        assert_eq!(
            receipt.source_native_payload_sha256,
            receipt.reconstructed_native_payload_sha256
        );
        let split_receipt = verify(&tmp.0, &shards[0], 96, 128)?;
        assert_eq!(
            receipt.source_native_payload_sha256,
            split_receipt.source_native_payload_sha256
        );
        assert_eq!(receipt.metadata_sha256, split_receipt.metadata_sha256);
        // Existing output must never be truncated or replaced.
        assert!(export(&tmp.0, &out, 96, 128).is_err());
        assert!(Portable::open(&out).is_ok());
        Ok(())
    }
    #[test]
    fn reject_truncated_or_wrong_topology_pack() -> Result<()> {
        let (tmp, mut pack, _) = fixture()?;
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(tmp.0.join("gpu0.bin"))?;
        f.set_len(23)?;
        assert!(export(&tmp.0, &tmp.0.join("bad.gguf"), 96, 128).is_err());
        assert!(!tmp.0.join("bad.gguf").exists());
        pack.etp = 3;
        assert!(pack.validate().is_err());
        let mut m = matrix(3);
        m.cols = 31;
        assert!(m.sizes().is_err());
        Ok(())
    }
    #[test]
    fn verify_rejects_changed_payload_and_multiplier() -> Result<()> {
        use std::io::{Read, SeekFrom};
        let (tmp, _, _) = fixture()?;
        let out = tmp.0.join("experts.gguf");
        export(&tmp.0, &out, 96, 128)?;
        let g = Gguf::open(&out)?;
        let offset = g
            .tensors
            .values()
            .find(|t| t.name.ends_with(".codes"))
            .unwrap()
            .offset;
        drop(g);
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&out)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut byte = [0u8; 1];
        f.read_exact(&mut byte)?;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(&[byte[0] ^ 1])?;
        f.sync_all()?;
        assert!(
            verify(&tmp.0, &out, 96, 128)
                .unwrap_err()
                .to_string()
                .contains("code bytes differ")
        );
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(&byte)?;
        f.sync_all()?;
        let mut source = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(tmp.0.join("gpu0.bin"))?;
        source.seek(SeekFrom::Start(8 + 7 * 8))?;
        source.read_exact(&mut byte)?;
        source.seek(SeekFrom::Start(8 + 7 * 8))?;
        source.write_all(&[byte[0] ^ 1])?;
        source.sync_all()?;
        assert!(
            verify(&tmp.0, &out, 96, 128)
                .unwrap_err()
                .to_string()
                .contains("metadata differs")
        );
        Ok(())
    }

    /// Full production-sized index, sparse zero payload: metadata/RSS evidence only.
    /// It creates ~234 GB of logical file offsets but writes only the header/index.
    #[test]
    #[ignore = "explicit bounded-memory large-index screening"]
    fn production_index_sparse_open() -> Result<()> {
        let start = std::time::Instant::now();
        let (tmp, _, _) = fixture()?;
        let pack = Pack {
            format: 4,
            ngpu: 8,
            etp: 8,
            expert_gb: 27.5,
            tiers: (0..40)
                .map(|l| {
                    (0..384)
                        .map(|e| if l * 384 + e < 188 { 2 } else { 1 })
                        .collect()
                })
                .collect(),
            owner: vec![vec![0; 384]; 40],
        };
        let mut matrices = Vec::with_capacity(40 * 384 * 8 * 2);
        for l in 0..40 {
            for e in 0..384 {
                for part in 0..8 {
                    for role in ["gu", "down"] {
                        matrices.push(Matrix {
                            layer: l,
                            expert: e,
                            part,
                            role: role.into(),
                            rows: if role == "gu" { 576 } else { 5120 },
                            cols: if role == "gu" { 5120 } else { 288 },
                            bits: if pack.tiers[l][e] == 1 { 3 } else { 2 },
                            mul_bits: 0x3b123456 + part as u32,
                        });
                    }
                }
            }
        }
        let index = Index {
            version: 1,
            dim: 5120,
            inter: 2304,
            pack,
            matrices,
        };
        let json = serde_json::to_string(&index)?;
        let path = tmp.0.join("full-index.gguf");
        let file = File::create(&path)?;
        let mut w = BufWriter::new(&file);
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(index.matrices.len() as u64 * 2).to_le_bytes())?;
        w.write_all(&2u64.to_le_bytes())?;
        string(&mut w, "general.architecture")?;
        w.write_all(&8u32.to_le_bytes())?;
        string(&mut w, "volta-experts")?;
        string(&mut w, "volta.index")?;
        w.write_all(&8u32.to_le_bytes())?;
        string(&mut w, &json)?;
        let mut off = 0u64;
        for m in &index.matrices {
            let (nw, ns) = m.sizes()?;
            for (suffix, len) in [("codes", nw), ("scales", ns)] {
                string(&mut w, &format!("{}.{}", m.name(), suffix))?;
                w.write_all(&1u32.to_le_bytes())?;
                w.write_all(&(len as u64).to_le_bytes())?;
                w.write_all(&24u32.to_le_bytes())?;
                w.write_all(&off.to_le_bytes())?;
                off = (off + len as u64 + 31) & !31;
            }
        }
        pad(&mut w)?;
        let header_bytes = w.stream_position()?;
        w.flush()?;
        file.set_len(header_bytes + off)?;
        drop(w);
        drop(file);
        drop(json);
        drop(index);
        let open_start = std::time::Instant::now();
        let q = Portable::open(&path)?;
        q.validate_model(5120, 2304, 40, 384, 8)?;
        let open_seconds = open_start.elapsed().as_secs_f64();
        for m in [q.matrix(0, 0, 0, "gu")?, q.matrix(39, 383, 7, "down")?] {
            let (w, s) = q.native(m)?;
            assert!(w.iter().chain(&s).all(|&b| b == 0));
            q.release_payload(m)?;
        }
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let peak = status
            .lines()
            .find(|l| l.starts_with("VmHWM:"))
            .unwrap_or("unavailable");
        println!(
            "{}",
            serde_json::json!({"schema":"volta-production-index-screen-v1","matrices":q.matrices().len(),"tensors":q.gguf.tensors.len(),"logical_file_bytes":std::fs::metadata(&path)?.len(),"written_header_bytes":header_bytes,"open_seconds":open_seconds,"total_seconds":start.elapsed().as_secs_f64(),"peak_rss_proc_status":peak,"full_payload_verified":false})
        );
        Ok(())
    }
}
