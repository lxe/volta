//! Batched decode on 8x V100 (Volta) — phase 1 of volta/DESIGN.md.
//!
//! Layout: data-parallel attention + expert-parallel MoE, dense weights replicated. Each GPU holds a full copy of the
//! dense half and owns a set of session slots (their KV in per-GPU pools); routed expert e of every layer lives on
//! GPU e % ngpu. Per layer, every GPU routes its own tokens, pushes their fp16 activations and picks to every peer,
//! runs its local experts over all tokens that picked them (vmoe_* kernels), and pushes the partial outputs back to
//! each token's home GPU. Phase 1 synchronizes the exchanges with a host barrier; later phases replace that with
//! device flags and the cube collectives (bench/v100-cube-allgather.cu).
//!
//! The math is the CPU reference's (`model.rs` ref_step), token by token; GEMV inputs stay f32 (dsx's kernels),
//! routed experts take fp16 activations.

use crate::gguf::GType;
use crate::gpu::{Arg, Buf, Gpu, Kernel};
use crate::gpumodel::{DMat, f32buf, upload_dmat};
use crate::model::{Mat, Model};
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex};

mod spark;

/// most tokens one GPU decodes per step (dense projections run as vmoe tiles of up to 32 tokens)
pub const TMAX: usize = 64;
/// chunks of the hyper-connection mix dots (vhc_dots)
const HC_S: usize = 32;
/// most tokens one session may have in a step (prefill chunks, draft verification)
const KX: usize = TMAX;

/// window ring rows per slot: the window plus room for a step's new tokens, so writing them never overwrites a row
/// an earlier token of the same step still reads
fn wring(win: usize) -> usize {
    win + KX
}

/// compressor state rows per slot at ratio r: enough whole groups that the groups a step closes never share rows
fn state_ring(r: usize) -> usize {
    r * (KX / r + 1)
}
/// bytes per exchange row: x fp16 [dim] | picks i32 [top_k] | weights f32 [top_k], padded to 64
fn xrow_bytes(c: &crate::model::Config) -> usize {
    (c.dim * 2 + c.top_k * 8).div_ceil(64) * 64
}

/// blocks per cube-collective launch (VX_BLOCKS in volta.cuh)
const VX_BLOCKS: usize = 8;

/// most 32-row tiles of the small-T attention (window 128 + top-512 = 20)
const ATTN_SPLITS: usize = 4; // 160 rows per split: window + idx_topk = 640 rows

#[derive(Clone, Debug)]
pub struct Opts {
    pub ngpu: usize,
    /// session slots per GPU
    pub slots: usize,
    pub max_ctx: usize,
}

const KERNELS: &[&str] = &[
    "gemv_q8_0", "gemv_q8r", "gemv_q8r_sk", "gemv_f16", "gemv_f16_sk", "gemv_bf16", "gemv_bf16_sk", "gemv_f32", "gemv_f32_sk",
    "gemv_fp8b32", "gemv_q2_0", "gemv_ptq1_0", "gemv_q3_k", "gemv_q8g", "gemv_q6p", "gemv_q6g4", "gemv_q6s8", "gemv_q5p", "gemv_q5p4",
    "rmsnorm", "rope_pos", "fake_fp8", "fake_fp4", "round_bf16", "scale_bf16", "hc_dots", "hc_finish_w", "hc_pre", "hc_post", "vhc_pre", "vhc_post", "vhc_dots", "vhc_finish", "vprep", "vrouter", "vsrc_rows", "vmean_hc", "vgather_rows", "vmarkov_embed", "vmarkov_argmax", "vargmax_rows", "vq16", "vattn_mma",
    "router_topk", "engram_gate_t", "sparse_attn_tile", "sparse_attn_merge", "fake_fp8_copy",
    "vkv_store", "vrows_scatter", "vcompress", "vstore_bf16_rows", "vindexer_score", "vidx_offset", "vswiglu_f32", "vsum_parts",
    "vswiglu", "vto_f16", "fwht_rows_f16", "vto_f16_rows", "vzero_rows", "vpack_x", "vx_allgather", "vx_alltoall", "vattn_tile5", "vcompact", "vlut_pack", "vcb_quant", "fake_fp8_copy",
    "vlut_fp4r1_n1", "vlut_fp4r1_n2", "vlut_cb3r1_n1", "vlut_cb3r1_n2", "vlut_cb2r1_n1", "vlut_cb2r1_n2", "vlut_fp4_n1", "vlut_fp4_n2", "vlut_fp4_n4", "vlut_cb3_n1", "vlut_cb3_n2", "vlut_cb3_n4", "vlut_cb2_n1", "vlut_cb2_n2", "vlut_cb2_n4", "vsum_parts_h", "vmoe_b4_n1_r1", "vmoe_b4_n2_r1", "vroute_count", "vroute_scan", "vroute_fill", "vswiglu_c", "vfwht_c", "vzero_c", "vmoe_b2_n1", "vmoe_b2_n2", "vmoe_b2_n4", "vmoe_b4_n1", "vmoe_b4_n2", "vmoe_b4_n4",
];

// ---------------------------------------------------------------- routed experts in the vmoe layout

/// one expert's matrices on its GPU: gate+up [2*inter][dim] and down [dim][inter]. `bits` = kind of gate/up | kind of
/// down << 8, a kind being 2 / 4 (offset-binary ints: offsets in uint4 / halves) or K_FP4 / K_CB3 / K_CB2 (lookup
/// formats: word offsets in 32-word units, scale offsets in bytes, `mul` the codebook scale factor)
#[derive(Clone, Copy, Debug, Default)]
struct ExpSlot {
    bits: u32,
    gu_w: u64,
    gu_s: u64,
    dn_w: u64,
    dn_s: u64,
    gu_mul: f32,
    dn_mul: f32,
}

const K_FP4: u32 = 100;
const K_CB3: u32 = 101;
const K_CB2: u32 = 102;
/// codebooks of the 3- and 2-bit tiers (dsx-amd decode.rs CB3 / CB2), in units of the group scale
const CB3: [f32; 8] = [-8.0, -4.5, -2.5, -0.625, 1.25, 2.75, 4.5, 8.0];
const CB2: [f32; 4] = [-8.0, -1.875, 2.75, 8.0];
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// tier t (0 MXFP4, 1 cb3, 2 cb2) -> (lookup kind, bits per weight)
fn tier_kind(t: u8) -> (u32, usize) {
    match t { 0 => (K_FP4, 4), 1 => (K_CB3, 3), _ => (K_CB2, 2) }
}

/// device tables of the lookup formats: value pairs fp4 [256] | cb3 [64] | cb2 [16] (half2), E4M3 -> fp16 [256],
/// then the cb3 / cb2 levels (f32) for the quantizer. Offsets: 0, 1024, 1280, 1344, 1856, 1888.
fn lut_tables() -> Vec<u8> {
    let h = |x: f32| half::f16::from_f32(x).to_le_bytes();
    let mut b = vec![];
    for i in 0..256 { b.extend(h(E2M1[i & 15])); b.extend(h(E2M1[i >> 4])); }
    for i in 0..64 { b.extend(h(CB3[i & 7])); b.extend(h(CB3[i >> 3])); }
    for i in 0..16 { b.extend(h(CB2[i & 3])); b.extend(h(CB2[i >> 2])); }
    for i in 0..256u32 { b.extend(h(crate::model::fp8_e4m3_to_f32((i & 127) as u8))); }
    for v in CB3.iter().chain(CB2.iter()) { b.extend(v.to_le_bytes()); }
    b
}

/// codes (offset binary, BITS wide) and one scale per 32 weights for a row-major matrix
struct Coded {
    bits: u32,
    codes: Vec<u8>,
    scales: Vec<f32>,
}

fn f16_at(b: &[u8], o: usize) -> f32 {
    half::f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

/// Convert one GGUF matrix to vmoe codes (one scale per GS = 64 / bits weights). PTQ1_0 (ternary) and Q2_0 go to
/// 2-bit exactly; everything else to 4-bit with a scale per 16, recovered exactly when the group's values are
/// integer multiples of one step (Q3_K: (q - 4) x a 6-bit sub-scale per 16), else absmax.
fn code_mat(m: &Mat) -> Coded {
    let (rows, k) = (m.rows, m.cols);
    let mut row = vec![0f32; k];
    match m.ty {
        GType::Ptq1_0 | GType::Q2_0 => {
            let (blk_w, blk_b, d_off, flip) = if m.ty == GType::Ptq1_0 { (128, 28, 26, false) } else { (64, 18, 0, true) };
            let mut codes = vec![0u8; rows * k];
            let mut scales = vec![0f32; rows * k / 32];
            for r in 0..rows {
                m.dequant(r, &mut row);
                let raw = m.row(r);
                for b in 0..k / blk_w {
                    let d = f16_at(raw, b * blk_b + d_off);
                    for i in 0..blk_w {
                        let kk = b * blk_w + i;
                        let q = if d == 0.0 { 0 } else { (row[kk] / d).round() as i32 };
                        // ternary: q in -1..1 -> code q + 2; Q2_0: q in -1..2 -> code 2 - q with the scale negated
                        codes[r * k + kk] = if flip { (2 - q).clamp(0, 3) as u8 } else { (q + 2).clamp(0, 3) as u8 };
                    }
                    for g in 0..blk_w / 32 {
                        scales[r * (k / 32) + b * (blk_w / 32) + g] = if flip { -d } else { d };
                    }
                }
            }
            Coded { bits: 2, codes, scales }
        }
        _ => {
            let mut codes = vec![0u8; rows * k];
            let mut scales = vec![0f32; rows * k / 16];
            for r in 0..rows {
                m.dequant(r, &mut row);
                for g in 0..k / 16 {
                    let v = &row[g * 16..g * 16 + 16];
                    let (s, c) = step_of(v);
                    scales[r * (k / 16) + g] = s;
                    codes[r * k + g * 16..r * k + g * 16 + 16].copy_from_slice(&c);
                }
            }
            Coded { bits: 4, codes, scales }
        }
    }
}

/// a scale s and 4-bit codes (offset 8) for 16 values: exact when every value is an integer multiple (in -8..7) of
/// max|v| / n for some n in 8..1, else absmax rounding
fn step_of(v: &[f32]) -> (f32, [u8; 16]) {
    let mut c = [8u8; 16];
    let (mi, mx) = v.iter().fold((0f32, 0f32), |(a, b), &x| (a.min(x), b.max(x)));
    let big = if -mi > mx { mi } else { mx };
    if big == 0.0 {
        return (0.0, c);
    }
    for n in (1..=8).rev() {
        let s = big.abs() / n as f32;
        let ok = v.iter().all(|&x| {
            let q = x / s;
            (q - q.round()).abs() < 1e-3 && (-8.0..=7.0).contains(&q.round())
        });
        if ok {
            for (i, &x) in v.iter().enumerate() {
                c[i] = ((x / s).round() as i32 + 8) as u8;
            }
            return (s, c);
        }
    }
    let s = (-mi / 8.0).max(mx / 7.0);
    for (i, &x) in v.iter().enumerate() {
        c[i] = ((x / s).round() as i32 + 8).clamp(0, 15) as u8;
    }
    (s, c)
}

/// pack codes into the swizzled vmoe layout (volta.cuh): words [group][unit][lane][4], scales [group][k/32][lane]
fn pack(c: &Coded, rows: usize, k: usize, w: &mut Vec<u32>, s: &mut Vec<half::f16>) {
    let bits = c.bits as usize;
    let (wpu, f) = (128 / bits, 32 / bits);
    let units = k / wpu;
    let gs = 64 / bits;
    let (w0, s0) = (w.len(), s.len());
    w.resize(w0 + rows * k * bits / 32, 0);
    s.resize(s0 + rows * k / gs, half::f16::ZERO);
    for r in 0..rows {
        let (rg, rs) = (r / 32, r % 32);
        let lane = 4 * (rs >> 3) + (rs & 3) + 16 * ((rs >> 2) & 1);
        for kk in 0..k {
            let (u, wd, fi) = (kk / wpu, (kk % wpu) / f, kk % f);
            w[w0 + ((rg * units + u) * 32 + lane) * 4 + wd] |= (c.codes[r * k + kk] as u32) << (bits * fi);
        }
        for g in 0..k / gs {
            s[s0 + (rg * (k / gs) + g) * 32 + lane] = half::f16::from_f32(c.scales[r * (k / gs) + g]);
        }
    }
}

struct Experts {
    w: Buf,
    s: Buf,
    tab: HashMap<(usize, usize), ExpSlot>,
}

/// bytes of one expert in the vmoe layout: a lookup tier (bits + a scale byte per 32), else from the GGUF types
/// (2-bit + scale per 32 for PTQ1_0 / Q2_0, 4-bit + scale per 16 otherwise)
fn expert_bytes(m: &Model, l: usize, e: usize, tiers: Option<&[Vec<u8>]>) -> usize {
    let c = &m.cfg;
    if let Some(t) = tiers {
        let bits = tier_kind(t[l][e]).1;
        return 3 * c.inter * c.dim * (bits * 4 + 1) / 32;
    }
    let ex = &m.layers[l].experts[e];
    let b = |t: GType| if matches!(t, GType::Ptq1_0 | GType::Q2_0) { 2usize } else { 4 };
    let gu = b(ex.gate.ty).max(b(ex.up.ty));
    let dn = b(ex.down.ty);
    (2 * c.inter * c.dim) * (gu * 8 + 16) / 64 + (c.dim * c.inter) * (dn * 8 + 16) / 64
}

/// owner[l][e]: the GPU holding expert e of layer l. Each expert's expected cost per step is its bytes x P(picked
/// at least once) plus a term for its expected tokens, from the routing counts in VOLTA_RANK (dsx's rank.json:
/// [layer, expert, picks], default plans/rank.json) at VOLTA_PLACE_T tokens per step (default 128); without
/// counts every expert weighs its bytes. Per layer, largest cost first onto the GPU with the least cost so far.
/// routing counts per (layer, expert) from VOLTA_RANK (dsx's rank.json: [layer, expert, picks]), if present
fn routing_counts(m: &Model) -> Option<Vec<Vec<f64>>> {
    let c = &m.cfg;
    let path = std::env::var("VOLTA_RANK").unwrap_or_else(|_| "plans/rank.json".into());
    let v = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Vec<[u64; 3]>>(&b).ok())?;
    let mut picks = vec![vec![0f64; c.n_experts]; c.n_layers];
    for [l, e, n] in v {
        if (l as usize) < c.n_layers && (e as usize) < c.n_experts {
            picks[l as usize][e as usize] = n as f64;
        }
    }
    Some(picks)
}

/// Tier per expert for an MXFP4 source under `budget` bytes of experts (dsx-amd's rule): every expert starts in its
/// own MXFP4; the coldest (fewest routing picks) are re-quantized to cb3, then the coldest cb3 to cb2, until it fits.
pub fn plan_tiers(m: &Model, budget: f64) -> Vec<Vec<u8>> {
    let c = &m.cfg;
    let picks = routing_counts(m).unwrap_or_else(|| vec![vec![0.0; c.n_experts]; c.n_layers]);
    let mut order: Vec<(f64, usize, usize)> = (0..c.n_layers).flat_map(|l| (0..c.n_experts).map(move |e| (0.0, l, e))).collect();
    for x in order.iter_mut() {
        x.0 = picks[x.1][x.2];
    }
    order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // coldest first
    let mut tiers = vec![vec![0u8; c.n_experts]; c.n_layers];
    let bytes = |t: u8| (3 * c.inter * c.dim * (tier_kind(t).1 * 4 + 1) / 32) as f64;
    let mut total = order.len() as f64 * bytes(0);
    for to in [1u8, 2] {
        for &(_, l, e) in &order {
            if total <= budget {
                break;
            }
            if tiers[l][e] == to - 1 {
                total -= bytes(to - 1) - bytes(to);
                tiers[l][e] = to;
            }
        }
    }
    let n: Vec<usize> = (0..3u8).map(|t| tiers.iter().flatten().filter(|&&x| x == t).count()).collect();
    eprintln!("experts: {} MXFP4, {} cb3, {} cb2: {:.1} GB (budget {:.1} GB)", n[0], n[1], n[2], total / 1e9, budget / 1e9);
    tiers
}

pub fn assign_experts(m: &Model, ngpu: usize, tiers: Option<&[Vec<u8>]>) -> Vec<Vec<u8>> {
    let c = &m.cfg;
    let counts = routing_counts(m);
    let have = counts.is_some();
    let picks = counts.unwrap_or_else(|| vec![vec![0f64; c.n_experts]; c.n_layers]);
    let t = std::env::var("VOLTA_PLACE_T").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(128.0);
    let mut total = vec![0f64; ngpu];
    (0..c.n_layers)
        .map(|l| {
            let sum: f64 = picks[l].iter().sum::<f64>().max(1.0);
            let mut es: Vec<(f64, usize)> = (0..c.n_experts)
                .map(|e| {
                    let b = expert_bytes(m, l, e, tiers) as f64;
                    if !have {
                        return (b, e);
                    }
                    let lam = c.top_k as f64 * t * picks[l][e] / sum; // expected tokens per step
                    (b * ((1.0 - (-lam).exp()) + 0.03 * lam), e)
                })
                .collect();
            es.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut layer = vec![0f64; ngpu];
            let mut own = vec![0u8; c.n_experts];
            for (cost, e) in es {
                let g = (0..ngpu).min_by(|&a, &b| (layer[a], total[a]).partial_cmp(&(layer[b], total[b])).unwrap()).unwrap();
                own[e] = g as u8;
                layer[g] += cost;
                total[g] += cost;
            }
            own
        })
        .collect()
}

/// GPUs that split each routed expert (VOLTA_EXPERT_TP, default every GPU for the MXFP4 source, else 1): expert
/// groups of that many GPUs, each holding a 1/tp slice of the intermediate dim (gate/up rows, down columns) of its
/// group's experts. At tp = ngpu every GPU reads 1/ngpu of every picked expert, so no GPU waits on another's routing
/// luck at the exchange.
pub fn expert_tp(m: &Model, ngpu: usize) -> usize {
    if let Some(p) = Pack::open() {
        return p.etp;
    }
    let fp4 = fp4_source(m);
    let tp = std::env::var("VOLTA_EXPERT_TP").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(if fp4 { ngpu } else { 1 });
    if tp > 1 && fp4 && ngpu % tp == 0 && m.cfg.inter % (32 * tp) == 0 && !m.cfg.expert_rot.iter().any(|&r| r) { tp } else { 1 }
}

/// bump when the packed expert layout changes (invalidates /dev/shm/vcache and packs)
const FORMAT: u32 = 4;

/// experts from DeepSeek's MXFP4 checkpoint (also when they come from a pack, which only that source makes)
fn fp4_source(m: &Model) -> bool {
    m.layers[0].experts.first().map_or(true, |e| e.gate.ty == GType::Fp4)
}

/// A pack (VOLTA_PACK=<dir>): the packed expert store of every GPU (`gpu<i>.bin`, the /dev/shm/vcache format) and
/// `manifest.json` with the plan that made it, so a machine like this one loads the experts without the
/// checkpoint's expert tensors or the GPU re-quantization. If the directory has no manifest yet, the run makes the
/// pack from the full checkpoint (export).
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Pack {
    pub format: u32,
    pub ngpu: usize,
    pub etp: usize,
    pub expert_gb: f64,
    /// [layer][expert]: 0 MXFP4, 1 cb3, 2 cb2
    pub tiers: Vec<Vec<u8>>,
    /// [layer][expert]: the expert group (of etp GPUs) that holds it
    pub owner: Vec<Vec<u8>>,
}

impl Pack {
    pub fn dir() -> Option<std::path::PathBuf> {
        std::env::var_os("VOLTA_PACK").map(std::path::PathBuf::from)
    }
    pub fn open() -> Option<Pack> {
        let b = std::fs::read(Self::dir()?.join("manifest.json")).ok()?;
        serde_json::from_slice(&b).ok()
    }
}

impl Experts {
    /// convert and upload the experts this GPU owns (layer-major), streaming one layer at a time; the packed store is
    /// cached in /dev/shm/vcache (VOLTA_CACHE=0 disables), keyed by the model file, format and expert placement
    fn build(g: &Gpu, m: &Model, gi: usize, etp: usize, owner: &[Vec<u8>], tiers: Option<&[Vec<u8>]>, luts: u64) -> Result<Experts> {
        if let Some(t) = tiers {
            return Self::build_lut(g, m, gi, etp, owner, t, luts);
        }
        ensure!(etp == 1, "split experts need the MXFP4 source");
        use rayon::prelude::*;
        use std::io::{Read, Write};
        let c = &m.cfg;
        let (d, inter) = (c.dim, c.inter);
        let mine: Vec<(usize, usize)> = (0..c.n_layers).flat_map(|l| (0..c.n_experts).filter(move |&e| owner[l][e] as usize == gi).map(move |e| (l, e))).collect();
        // sizes from the source types (2-bit + scale per 32 for PTQ1_0 / Q2_0, 4-bit + scale per 16 otherwise)
        let bits_of = |mt: &Mat| if matches!(mt.ty, GType::Ptq1_0 | GType::Q2_0) { 2usize } else { 4 };
        let mut wtot = 0usize;
        let mut stot = 0usize;
        for &(l, e) in &mine {
            let ex = &m.layers[l].experts[e];
            wtot += 2 * inter * d * bits_of(&ex.gate).max(bits_of(&ex.up)) / 32 + d * inter * bits_of(&ex.down) / 32;
            stot += 2 * inter * d * bits_of(&ex.gate).max(bits_of(&ex.up)) / 64 + d * inter * bits_of(&ex.down) / 64;
        }
        let wbuf = g.alloc(wtot * 4)?;
        let sbuf = g.alloc(stot * 2)?;
        let mut tab = HashMap::new();
        // cache file: header [n][l, e, bits, gu_w, gu_s, dn_w, dn_s as u64], then the words, then the scales
        let cache = if std::env::var("VOLTA_CACHE").is_ok_and(|v| v == "0") {
            None
        } else {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (FORMAT, wtot, stot, owner, c.n_layers, c.n_experts).hash(&mut h);
            // the model's content: a sample of every layer's first and last expert matrices
            for l in &m.layers {
                for ex in [&l.experts[0], &l.experts[c.n_experts - 1]] {
                    for mt in [&ex.gate, &ex.up, &ex.down] {
                        let b = mt.bytes();
                        (b.len(), &b[..b.len().min(512)], &b[b.len().saturating_sub(512)..]).hash(&mut h);
                    }
                }
            }
            Some(std::path::PathBuf::from(format!("/dev/shm/vcache/{:016x}/gpu{gi}.bin", h.finish())))
        };
        if let Some(path) = cache.as_ref().filter(|p| p.exists()) {
            let mut f = std::io::BufReader::with_capacity(64 << 20, std::fs::File::open(path)?);
            let mut hdr = vec![0u8; 8];
            f.read_exact(&mut hdr)?;
            let n = u64::from_le_bytes(hdr[..8].try_into().unwrap()) as usize;
            let mut ent = vec![0u8; n * 7 * 8];
            f.read_exact(&mut ent)?;
            for r in ent.chunks(56) {
                let v: Vec<u64> = r.chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
                tab.insert((v[0] as usize, v[1] as usize), ExpSlot { bits: v[2] as u32, gu_w: v[3], gu_s: v[4], dn_w: v[5], dn_s: v[6], ..Default::default() });
            }
            for (buf, bytes) in [(&wbuf, wtot * 4), (&sbuf, stot * 2)] {
                let mut done = 0usize;
                let mut chunk = vec![0u8; 256 << 20];
                while done < bytes {
                    let n = chunk.len().min(bytes - done);
                    f.read_exact(&mut chunk[..n])?;
                    g.htod_sync(buf.ptr + done as u64, &chunk[..n])?;
                    done += n;
                }
            }
            return Ok(Experts { w: wbuf, s: sbuf, tab });
        }
        let mut words: Vec<u8> = vec![];
        let mut scales: Vec<u8> = vec![];
        let (mut wo, mut so) = (0usize, 0usize);
        for l in 0..c.n_layers {
            let es: Vec<usize> = mine.iter().filter(|x| x.0 == l).map(|x| x.1).collect();
            let packed: Vec<(usize, u32, Vec<u32>, Vec<half::f16>, usize, usize)> = es
                .par_iter()
                .map(|&e| {
                    let ex = &m.layers[l].experts[e];
                    let (cg, cu, cd) = (code_mat(&ex.gate), code_mat(&ex.up), code_mat(&ex.down));
                    // gate and up must share a width; widen the narrower one
                    let (cg, cu) = if cg.bits != cu.bits { (widen(cg), widen(cu)) } else { (cg, cu) };
                    let mut gu = Coded { bits: cg.bits, codes: cg.codes, scales: cg.scales };
                    gu.codes.extend_from_slice(&cu.codes);
                    gu.scales.extend_from_slice(&cu.scales);
                    let (mut w, mut s) = (vec![], vec![]);
                    pack(&gu, 2 * inter, d, &mut w, &mut s);
                    let (gw, gs) = (w.len(), s.len());
                    pack(&cd, d, inter, &mut w, &mut s);
                    let bits = gu.bits | (cd.bits << 8);
                    (e, bits, w, s, gw, gs)
                })
                .collect();
            for (e, bits, w, s, gw, gs) in packed {
                ensure!((wo + w.len()) <= wtot && so + s.len() <= stot, "expert store overflow");
                g.htod_sync(wbuf.ptr + (wo * 4) as u64, &w)?;
                g.htod_sync(sbuf.ptr + (so * 2) as u64, &s)?;
                tab.insert((l, e), ExpSlot { bits, gu_w: (wo / 4) as u64, gu_s: so as u64, dn_w: ((wo + gw) / 4) as u64, dn_s: (so + gs) as u64, ..Default::default() });
                if cache.is_some() {
                    words.extend_from_slice(unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) });
                    scales.extend_from_slice(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 2) });
                }
                wo += w.len();
                so += s.len();
            }
        }
        if let Some(path) = cache {
            if words.len() == wtot * 4 && scales.len() == stot * 2 {
                std::fs::create_dir_all(path.parent().unwrap())?;
                let tmp = path.with_extension("tmp");
                let mut f = std::io::BufWriter::with_capacity(64 << 20, std::fs::File::create(&tmp)?);
                f.write_all(&(tab.len() as u64).to_le_bytes())?;
                for (&(l, e), x) in &tab {
                    for v in [l as u64, e as u64, x.bits as u64, x.gu_w, x.gu_s, x.dn_w, x.dn_s] {
                        f.write_all(&v.to_le_bytes())?;
                    }
                }
                f.write_all(&words)?;
                f.write_all(&scales)?;
                f.flush()?;
                drop(f);
                std::fs::rename(&tmp, &path)?;
            }
        }
        Ok(Experts { w: wbuf, s: sbuf, tab })
    }

    /// MXFP4 source (the HF checkpoint): each expert in its tier (`tiers`), packed on the GPU: MXFP4 as is, cb3 / cb2
    /// re-quantized by vcb_quant. Cached in /dev/shm/vcache like the integer store.
    fn build_lut(g: &Gpu, m: &Model, gi: usize, etp: usize, owner: &[Vec<u8>], tiers: &[Vec<u8>], luts: u64) -> Result<Experts> {
        use std::io::{Read, Write};
        let c = &m.cfg;
        // inter = this GPU's slice of the intermediate dim (all of it when experts are whole)
        let (d, inter, grp, part) = (c.dim, c.inter / etp, gi / etp, gi % etp);
        let mine: Vec<(usize, usize)> = (0..c.n_layers).flat_map(|l| (0..c.n_experts).filter(move |&e| owner[l][e] as usize == grp).map(move |e| (l, e))).collect();
        let (mut wtot, mut stot) = (0usize, 0usize); // u32 words, scale bytes
        for &(l, e) in &mine {
            let bits = tier_kind(tiers[l][e]).1;
            wtot += 3 * inter * d * bits / 32;
            stot += 3 * inter * d / 32;
        }
        let wbuf = g.alloc(wtot * 4)?;
        let sbuf = g.alloc(stot)?;
        let mut tab = HashMap::new();
        let cache = if let Some(dir) = Pack::dir() {
            Some(dir.join(format!("gpu{gi}.bin")))
        } else if std::env::var("VOLTA_CACHE").is_ok_and(|v| v == "0") {
            None
        } else {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (FORMAT, "lut", wtot, stot, owner, tiers, c.n_layers, c.n_experts, etp, part).hash(&mut h);
            for l in &m.layers {
                for ex in [&l.experts[0], &l.experts[c.n_experts - 1]] {
                    for mt in [&ex.gate, &ex.up, &ex.down] {
                        let b = mt.bytes();
                        (b.len(), &b[..b.len().min(512)], &b[b.len().saturating_sub(512)..]).hash(&mut h);
                    }
                }
            }
            Some(std::path::PathBuf::from(format!("/dev/shm/vcache/{:016x}/gpu{gi}.bin", h.finish())))
        };
        if let Some(path) = cache.as_ref().filter(|p| p.exists()) {
            let mut f = std::io::BufReader::with_capacity(64 << 20, std::fs::File::open(path)?);
            let mut hdr = [0u8; 8];
            f.read_exact(&mut hdr)?;
            let n = u64::from_le_bytes(hdr) as usize;
            let mut ent = vec![0u8; n * 9 * 8];
            f.read_exact(&mut ent)?;
            for r in ent.chunks(72) {
                let v: Vec<u64> = r.chunks(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
                tab.insert((v[0] as usize, v[1] as usize), ExpSlot { bits: v[2] as u32, gu_w: v[3], gu_s: v[4], dn_w: v[5], dn_s: v[6], gu_mul: f32::from_bits(v[7] as u32), dn_mul: f32::from_bits(v[8] as u32) });
            }
            for (buf, bytes) in [(&wbuf, wtot * 4), (&sbuf, stot)] {
                let mut done = 0usize;
                let mut chunk = vec![0u8; 256 << 20];
                while done < bytes {
                    let n = chunk.len().min(bytes - done);
                    f.read_exact(&mut chunk[..n])?;
                    g.htod_sync(buf.ptr + done as u64, &chunk[..n])?;
                    done += n;
                }
            }
            return Ok(Experts { w: wbuf, s: sbuf, tab });
        }
        let items: Vec<((usize, usize), &crate::model::Expert, u8)> = mine.iter().map(|&(l, e)| ((l, e), &m.layers[l].experts[e], tiers[l][e])).collect();
        let (wo, so) = pack_lut(g, &items, d, c.inter, etp, part, luts, &wbuf, &sbuf, &mut tab)?;
        g.sync()?;
        ensure!(wo == wtot && so == stot, "expert store size mismatch");
        if let Some(path) = cache {
            std::fs::create_dir_all(path.parent().unwrap())?;
            let tmp = path.with_extension("tmp");
            let mut f = std::io::BufWriter::with_capacity(64 << 20, std::fs::File::create(&tmp)?);
            f.write_all(&(tab.len() as u64).to_le_bytes())?;
            for (&(l, e), x) in &tab {
                for v in [l as u64, e as u64, x.bits as u64, x.gu_w, x.gu_s, x.dn_w, x.dn_s, x.gu_mul.to_bits() as u64, x.dn_mul.to_bits() as u64] {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            for (buf, bytes) in [(&wbuf, wtot * 4), (&sbuf, stot)] {
                let mut done = 0usize;
                while done < bytes {
                    let n = (256usize << 20).min(bytes - done);
                    let v: Vec<u8> = g.download(buf.ptr + done as u64, n)?;
                    f.write_all(&v)?;
                    done += n;
                }
            }
            f.flush()?;
            drop(f);
            std::fs::rename(&tmp, &path)?;
        }
        Ok(Experts { w: wbuf, s: sbuf, tab })
    }
}

/// pack routed experts from MXFP4 into the lookup formats on the GPU, each at its tier, into wbuf / sbuf (sized by the
/// caller); with etp > 1 only slice `part` of each expert's intermediate dim (gate / up rows, down columns)
#[allow(clippy::too_many_arguments)]
fn pack_lut(g: &Gpu, items: &[((usize, usize), &crate::model::Expert, u8)], d: usize, full_inter: usize, etp: usize, part: usize, luts: u64, wbuf: &Buf, sbuf: &Buf, tab: &mut HashMap<(usize, usize), ExpSlot>) -> Result<(usize, usize)> {
    let inter = full_inter / etp;
    // staging: raw nibbles, E8M0 scales, codebook codes and scale bytes of the largest matrix (gate+up)
    let mx = 2 * inter * d;
    let (raw, rsc, codes, sbs) = (g.alloc(mx / 2)?, g.alloc(mx / 32)?, g.alloc(mx)?, g.alloc(mx / 32)?);
    let (kpack, kquant) = (g.kernel("vlut_pack")?, g.kernel("vcb_quant")?);
    let (mut wo, mut so) = (0usize, 0usize);
    for &((l, e), ex, tier) in items {
        ensure!(ex.gate.ty == GType::Fp4 && ex.up.ty == GType::Fp4 && ex.down.ty == GType::Fp4, "layer {l} expert {e}: not MXFP4");
        let (kind, bits) = tier_kind(tier);
        let mut slot = ExpSlot { bits: kind | (kind << 8), ..Default::default() };
        for (mats, rows, cols) in [(vec![&ex.gate, &ex.up], 2 * inter, d), (vec![&ex.down], d, inter)] {
            let (mut ob, mut os) = (0usize, 0usize);
            // this GPU's slice: rows part*inter.. of gate and up, columns part*inter.. of every down row
            let mut slices = vec![];
            for mt in &mats {
                let (b, sb) = (mt.bytes(), mt.scale_bytes());
                let (wb, ws) = if etp == 1 {
                    (b.to_vec(), sb.to_vec())
                } else if rows == 2 * inter {
                    let (rb, rs) = (d / 2, d / 32);
                    (b[part * inter * rb..(part + 1) * inter * rb].to_vec(), sb[part * inter * rs..(part + 1) * inter * rs].to_vec())
                } else {
                    let (rb, rs, full) = (inter / 2, inter / 32, full_inter);
                    ((0..d).flat_map(|r| b[r * full / 2 + part * rb..][..rb].iter().copied()).collect(), (0..d).flat_map(|r| sb[r * full / 32 + part * rs..][..rs].iter().copied()).collect())
                };
                g.htod_sync(raw.ptr + ob as u64, &wb)?;
                g.htod_sync(rsc.ptr + os as u64, &ws)?;
                ob += wb.len();
                os += ws.len();
                slices.push(ws);
            }
            ensure!(ob == rows * cols / 2 && os == rows * cols / 32, "layer {l} expert {e}: unexpected MXFP4 sizes");
            let groups = rows * cols / 32;
            let (ow, osc) = (wbuf.ptr + (wo * 4) as u64, sbuf.ptr + so as u64);
            let mul = if kind == K_FP4 {
                g.launch(kpack, ((groups as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::I(0), Arg::P(raw.ptr), Arg::P(rsc.ptr), Arg::I(rows as i32), Arg::I(cols as i32), Arg::P(ow), Arg::P(osc)])?;
                1.0f32
            } else {
                // per-matrix factor: the largest group scale lands near 256 of E4M3's 448
                let smax = slices.iter().flatten().map(|&b| f32::from_bits((b as u32) << 23)).fold(0f32, f32::max);
                let mul = smax * 6.0 / 8.0 * 1.25 / 256.0;
                let (lv, nl, fmt) = if kind == K_CB3 { (luts + 1856, 8, 1) } else { (luts + 1888, 4, 2) };
                g.launch(kquant, ((groups as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(raw.ptr), Arg::P(rsc.ptr), Arg::I(rows as i32), Arg::I(cols as i32), Arg::P(lv), Arg::I(nl), Arg::F(mul), Arg::P(codes.ptr), Arg::P(sbs.ptr)])?;
                g.launch(kpack, ((groups as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::I(fmt), Arg::P(codes.ptr), Arg::P(sbs.ptr), Arg::I(rows as i32), Arg::I(cols as i32), Arg::P(ow), Arg::P(osc)])?;
                mul
            };
            if rows == 2 * inter {
                (slot.gu_w, slot.gu_s, slot.gu_mul) = ((wo / 32) as u64, so as u64, mul);
            } else {
                (slot.dn_w, slot.dn_s, slot.dn_mul) = ((wo / 32) as u64, so as u64, mul);
            }
            wo += rows * cols * bits / 32;
            so += groups;
        }
        tab.insert((l, e), slot);
    }
    g.sync()?;
    Ok((wo, so))
}

/// a layer's expert lists on this GPU: lidx [n_exp] (expert -> local index, -1 elsewhere) and one VWork list per
/// (gate/up or down, format), pointing into `exps`
fn local_layer(g: &Gpu, exps: &Experts, l: usize, es: &[usize], n_exp: usize) -> Result<LocalLayer> {
    let mut lidx = vec![-1i32; n_exp];
    for (j, &e) in es.iter().enumerate() {
        lidx[e] = j as i32;
    }
    let mut lists = vec![];
    for gu in [true, false] {
        for kind in [2u32, 4, K_FP4, K_CB3, K_CB2] {
            let lut = kind >= K_FP4;
            let v: Vec<u32> = es
                .iter()
                .enumerate()
                .filter(|&(_, &e)| { let b = exps.tab[&(l, e)].bits; (if gu { b & 0xff } else { b >> 8 }) == kind })
                .flat_map(|(j, &e)| {
                    let sl = exps.tab[&(l, e)];
                    let (wo, so, mul) = if gu { (sl.gu_w, sl.gu_s, sl.gu_mul) } else { (sl.dn_w, sl.dn_s, sl.dn_mul) };
                    // VWork (int kinds) or VWork2 (lookup kinds: + mul, padded to 32 bytes)
                    let mut x = vec![wo as u32, so as u32, 0, j as u32];
                    if lut {
                        x.extend([mul.to_bits(), 0, 0, 0]);
                    }
                    x
                })
                .collect();
            if !v.is_empty() {
                let n = v.len() / if lut { 8 } else { 4 };
                lists.push((gu, kind, g.upload(&v)?, n));
            }
        }
    }
    Ok(LocalLayer { wbase: exps.w.ptr, sbase: exps.s.ptr, lidx: g.upload(&lidx)?, lists, n_local: es.len() })
}

/// 2-bit codes as 4-bit (same values): code c (offset 2) -> c + 6 (offset 8)
fn widen(c: Coded) -> Coded {
    if c.bits == 4 {
        return c;
    }
    Coded { bits: 4, codes: c.codes.into_iter().map(|x| x + 6).collect(), scales: c.scales.iter().flat_map(|&x| [x, x]).collect() }
}

// ---------------------------------------------------------------- dense weights + session pools per GPU

/// a dense matrix in the vmoe 4-bit layout (one VWork entry at offset 0), for up to 32 tokens per weight pass
pub(crate) struct VMat {
    w: Buf,
    s: Buf,
    /// VWork entries [part][nt - 1]: row partition `part` (rows part * rows / parts ..), nt tokens
    work: Buf,
    prow: Buf,
    parts: usize,
    rows: usize,
    cols: usize,
    /// the source's inputs are FP8 in the reference (HF checkpoint): round x to FP8 first
    act_fp8: bool,
}

/// a projection: dsx's GEMV layouts (t <= 8 per pass) or vmoe 4-bit
pub(crate) enum Lin {
    D(DMat),
    V(VMat),
}

impl Lin {
    fn rows(&self) -> usize {
        match self { Lin::D(m) => m.rows, Lin::V(m) => m.rows }
    }
    fn cols(&self) -> usize {
        match self { Lin::D(m) => m.cols, Lin::V(m) => m.cols }
    }
}

/// 4-bit codes for 16 values: the scale (absmax-based, searched over a few shrink factors) with the least error
fn quant4_16(v: &[f32], c: &mut [u8]) -> f32 {
    let (mn, mx) = v.iter().fold((0f32, 0f32), |(a, b), &x| (a.min(x), b.max(x)));
    let s0 = (-mn / 8.0).max(mx / 7.0);
    if s0 == 0.0 {
        c.fill(8);
        return 0.0;
    }
    let (mut best, mut bs) = (f32::INFINITY, s0);
    for f in [1.0f32, 0.95, 0.9, 0.85, 0.8, 0.75] {
        let s = s0 * f;
        let e: f32 = v.iter().map(|&x| { let q = (x / s).round().clamp(-8.0, 7.0); (q * s - x).powi(2) }).sum();
        if e < best {
            best = e;
            bs = s;
        }
    }
    for (i, &x) in v.iter().enumerate() {
        c[i] = ((x / bs).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
    }
    bs
}

type Packed4 = (Vec<u32>, Vec<half::f16>);
static PACKED4: std::sync::OnceLock<Mutex<HashMap<usize, Arc<std::sync::OnceLock<Packed4>>>>> = std::sync::OnceLock::new();

/// requantize a dense matrix to vmoe 4-bit (shared between the GPU threads) and upload it; `parts` row partitions
/// get their own work entries (the grouped output projection runs one partition per group)
fn packed4(m: &Mat) -> Arc<std::sync::OnceLock<Packed4>> {
    let key = m.bytes().as_ptr() as usize;
    let cell = PACKED4.get_or_init(Default::default).lock().unwrap().entry(key).or_default().clone();
    cell.get_or_init(|| {
        use rayon::prelude::*;
        let (rows, k) = (m.rows, m.cols);
        let mut codes = vec![0u8; rows * k];
        let mut scales = vec![0f32; rows * k / 16];
        codes.par_chunks_mut(k).zip(scales.par_chunks_mut(k / 16)).enumerate().for_each(|(r, (c, sc))| {
            let mut x = vec![0f32; k];
            m.dequant(r, &mut x);
            for gi in 0..k / 16 {
                sc[gi] = quant4_16(&x[gi * 16..gi * 16 + 16], &mut c[gi * 16..gi * 16 + 16]);
            }
        });
        let (mut w, mut s) = (vec![], vec![]);
        pack(&Coded { bits: 4, codes, scales }, rows, k, &mut w, &mut s);
        (w, s)
    });
    cell
}

/// several matrices with the same columns as one (their row groups appended): one launch for inputs they share
fn upload_v4_cat(g: &Gpu, ms: &[&Mat]) -> Result<VMat> {
    let (mut w, mut s) = (vec![], vec![]);
    for m in ms {
        ensure!(m.rows % 32 == 0 && m.cols == ms[0].cols, "upload_v4_cat shapes");
        let c = packed4(m);
        let (pw, ps) = c.get().unwrap();
        w.extend_from_slice(pw);
        s.extend_from_slice(ps);
    }
    let rows: usize = ms.iter().map(|m| m.rows).sum();
    let work: Vec<u32> = (1..=TMAX as u32).flat_map(|nt| [0, 0, 0, nt]).collect();
    let prow: Vec<i32> = (0..TMAX as i32).collect();
    Ok(VMat { w: g.upload(&w)?, s: g.upload(&s)?, work: g.upload(&work)?, prow: g.upload(&prow)?, parts: 1, rows, cols: ms[0].cols, act_fp8: ms[0].act_fp8 })
}

fn upload_v4(g: &Gpu, m: &Mat, parts: usize) -> Result<VMat> {
    let cell = packed4(m);
    let (w, s) = cell.get().unwrap();
    let (units, prows) = (m.cols / 32, m.rows / parts);
    // work entries [nt - 1][part]: part p's row groups, nt tokens, rows from t0 = p * TMAX of the part row table
    let mut work = vec![];
    for nt in 1..=TMAX as u32 {
        for p in 0..parts {
            let rg = p * prows / 32;
            work.extend_from_slice(&[(rg * units * 32) as u32, (rg * (m.cols / 16) * 32) as u32, (p * TMAX) as u32, nt]);
        }
    }
    // part row table [part][t] = t * parts + p: x viewed as [t][parts][cols] rows, y as [t][parts][prows] rows
    let prow: Vec<i32> = (0..parts).flat_map(|p| (0..TMAX).map(move |t| (t * parts + p) as i32)).collect();
    Ok(VMat { w: g.upload(w)?, s: g.upload(s)?, work: g.upload(&work)?, prow: g.upload(&prow)?, parts, rows: m.rows, cols: m.cols, act_fp8: m.act_fp8 })
}

/// K chunk for a split-K launch of `blocks` row tiles: the longest chunk (a multiple of 128, at most 1024) that
/// divides K evenly and still gives at least `want` blocks; uneven chunks leave the short chunk's blocks idle
fn even_kc(k: usize, blocks: usize, want: usize) -> usize {
    let ok: Vec<usize> = (1..=k / 128).filter(|z| k % z == 0 && (k / z) % 128 == 0 && k / z <= 1024).map(|z| k / z).collect();
    ok.iter().copied().find(|&kc| blocks * (k / kc) >= want).or(ok.last().copied()).unwrap_or(k.min(1024))
}

/// a row-major f32 matrix in the vmoe 4-bit layout (one partition), for slices that are not a GGUF / HF tensor
fn upload_v4_f32(g: &Gpu, rows: usize, k: usize, x: &[f32], act_fp8: bool) -> Result<VMat> {
    use rayon::prelude::*;
    let mut codes = vec![0u8; rows * k];
    let mut scales = vec![0f32; rows * k / 16];
    codes.par_chunks_mut(k).zip(scales.par_chunks_mut(k / 16)).enumerate().for_each(|(r, (c, sc))| {
        for gi in 0..k / 16 {
            sc[gi] = quant4_16(&x[r * k + gi * 16..r * k + gi * 16 + 16], &mut c[gi * 16..gi * 16 + 16]);
        }
    });
    let (mut w, mut s) = (vec![], vec![]);
    pack(&Coded { bits: 4, codes, scales }, rows, k, &mut w, &mut s);
    let units = k / 32;
    let mut work = vec![];
    for nt in 1..=TMAX as u32 {
        work.extend_from_slice(&[0u32, 0, 0, nt]);
    }
    let _ = units;
    let prow: Vec<i32> = (0..TMAX as i32).collect();
    Ok(VMat { w: g.upload(&w)?, s: g.upload(&s)?, work: g.upload(&work)?, prow: g.upload(&prow)?, parts: 1, rows, cols: k, act_fp8 })
}

/// vmoe-able shape: rows a multiple of 32 x 4 (NT 1 uses 4 row groups per warp) and k a multiple of 32
fn v4_ok(m: &Mat) -> bool {
    m.rows % 128 == 0 && m.cols % 32 == 0
}

/// Q8_0 matrices requantized to 6-bit planes (q6.rs) once for all GPUs: (lo, hi, sc, d) per matrix
type Planes = (Vec<u8>, Vec<u8>, Vec<i8>, Vec<u16>);
static PLANES: std::sync::OnceLock<Mutex<HashMap<usize, Arc<std::sync::OnceLock<Planes>>>>> = std::sync::OnceLock::new();

/// upload_dmat, with the Q8_0 -> 6-bit conversion shared between the GPU threads
fn upload_dense(g: &Gpu, w: &Mat, q6: bool) -> Result<DMat> {
    if !(q6 && w.ty == GType::Q8_0 && w.cols % 256 == 0) {
        return upload_dmat(g, w, false);
    }
    let key = w.bytes().as_ptr() as usize;
    let cell = PLANES.get_or_init(Default::default).lock().unwrap().entry(key).or_default().clone();
    let (lo, hi, sc, d) = cell.get_or_init(|| {
        use rayon::prelude::*;
        let (rows, k) = (w.rows, w.cols);
        let (mut lo, mut hi, mut sc, mut d) = (vec![0u8; rows * k / 2], vec![0u8; rows * k / 4], vec![0i8; rows * k / 16], vec![0u16; rows * k / 256]);
        lo.par_chunks_mut(k / 2).zip(hi.par_chunks_mut(k / 4)).zip(sc.par_chunks_mut(k / 16)).zip(d.par_chunks_mut(k / 256)).enumerate().for_each(|(r, (((lo, hi), sc), d))| {
            let mut x = vec![0f32; k];
            w.dequant(r, &mut x);
            crate::q6::quantize_row(&x, lo, hi, sc, d);
        });
        (lo, hi, sc, d)
    });
    Ok(DMat { buf: g.upload(lo)?, scale: None, ty: w.ty, rows: w.rows, cols: w.cols, act_fp8: w.act_fp8, q8r: false, q6: Some((g.upload(hi)?, g.upload(sc)?, g.upload(d)?)) })
}

struct VLayer {
    /// q_a and kv as one matrix (same input): rows [q_lora | head_dim], when both are vmoe 4-bit (q_a, kv: None)
    qkv_a: Option<VMat>,
    q_a: Option<Lin>,
    q_a_norm: Buf,
    q_b: Lin,
    kv: Option<Lin>,
    kv_norm: Buf,
    o_a: Lin,
    o_b: Lin,
    sinks: Buf,
    attn_norm: Buf,
    ffn_norm: Buf,
    comp_kv: Option<Lin>,
    comp_gate: Option<Lin>,
    comp_norm: Option<Buf>,
    idx_q_b: Option<Lin>,
    idx_proj: Option<DMat>,
    idx_k: Option<Lin>,
    idx_k_norm: Option<Buf>,
    hc_attn: (Buf, Buf, Buf),
    hc_ffn: (Buf, Buf, Buf),
    gate_w: Buf, // f32 [E][d]: Volta has no bf16 GEMM; f32 is exact for the bf16 router weights
    gate_bias: Buf,
    /// routed experts and picks per token (the draft layers have their own)
    n_exp: usize,
    top_k: usize,
    sh_gate: Option<Lin>,
    sh_up: Option<Lin>,
    sh_down: Option<Lin>,
    /// the shared expert split 8 ways (this GPU's 1/ngpu of its intermediate rows): gate+up [2 * inter / ngpu][dim]
    /// and down's matching columns [dim][inter / ngpu], run over every token of the exchange (VOLTA_SHARED_TP=0: local)
    sh_slice: Option<(VMat, VMat)>,
    engram_kv: Option<Lin>,
    engram_qk: Option<Buf>,
    // pools: window bf16 [slots][W][hd]; comp / idx caches bf16 [slots][cap][..]; comp state f32 [slots][ratio][hd]
    window: Buf,
    comp_cache: Option<Buf>,
    idx_cache: Option<Buf>,
    comp_state: Option<(Buf, Buf)>,
}

/// host-side state of one session slot
#[derive(Clone, Default)]
struct Sess {
    pos: usize,
    engram_hist: Vec<i64>,
    n_comp: Vec<usize>,
    n_idx: Vec<usize>,
}

/// step work buffers (T = TMAX tokens on this GPU; TG = ngpu * TMAX tokens in the exchange)
struct Work {
    h: Buf,
    x: Buf,
    a: Buf,
    mix_a: Buf,
    mix_f: Buf,
    qr: Buf,
    qkv: Buf,
    /// fp16(q * scale) for vattn_mma
    q16: Buf,
    q: Buf,
    kvv: Buf,
    o: Buf,
    r: Buf,
    lat: Buf,
    t1: Buf,
    t2: Buf,
    iq: Buf,
    iw: Buf,
    score: Buf,
    keep: Buf,
    comp_idx: Buf,
    attn_part: Buf,
    logits: Buf,
    pos: Buf,
    win_idx: Buf,
    ints: Buf,
    tabs: Buf, // every layer's per-token tables of the step (Shard::plan)
    xh: Buf,   // fp16 activations for the vmoe dense projections [TMAX][max(heads * head_dim, 8192)]
    xq: Buf,   // FP8-rounded f32 copy of x for FP8-input projections
    shgu: Buf, // shared-expert slice: gate+up rows [TG][2 * inter / ngpu] f32
    shh: Buf,  // its SwiGLU output [TG][inter / ngpu] fp16
    tile32: Buf, // VWork entries t0 = 32 i, nt = 32 (static 32-row tiles over the exchange)
    srcwork: Buf, // VWork entries, one per source GPU (nt = its index into ntok / srcoff)
    /// vsrc_rows' list: [count, 0, rows...]
    srcl: Buf,
    srcoff: Buf,  // first exchange row of each source GPU (src * TMAX)
    router: Buf,
    sel: Buf,
    wts: Buf,
    topk_scratch: Buf,
    eng: Buf,
    // exchange (peers write here): xg [TG] rows of xrow_bytes (x16 | picks | weights), parts [ngpu][TMAX][d] f32
    xg: Buf,
    x16: Buf, // this GPU's fp16 x staging [TMAX][d] (rotated for rotated layers)
    selg: Buf,
    wtg: Buf,
    parts: Buf,
    // local experts (device-routed): per local list j, tokens tok[j*cap..], weights wt[j*cap..], count cnt[j];
    // pair rows p = j*cap + i: gu [P][2*inter] f32, hm [P][inter] f16; ypart [TG][d] f32 (this GPU's sums)
    gu: Buf,
    hm: Buf,
    ypart: Buf,
    ypart16: Buf, // ypart as fp16, what the all-to-all sends
    /// vhc_dots partials [TMAX][25][HC_S]
    hcp: Buf,
    cnt: Buf,
    /// the >16-token entries of one list (vcompact) and their count, so the big-tile launch only covers them
    hot: Buf,
    nhot: Buf,
    tok: Buf,
    wt: Buf,
    ntok: Buf,
    /// exchange flags written by the peers: [phase 0 = x / picks, 1 = partials][src GPU] = layer sequence number
    flags: Buf,
    /// cube collectives (vx_*): flags [3 stages][8 src][VX_BLOCKS] for the all-gather and the all-to-all, the
    /// all-to-all's relay [4 far][8 src][slot], device peer tables, per-step byte counts [ag 8 | a2a 8]
    vx_fa: Buf,
    vx_fb: Buf,
    relay: Buf,
    vx_pa: Buf,
    vx_pb: Buf,
    vx_bytes: Buf,
    seqd: Buf, // the step's exchange sequence base (the layer adds li)
}

/// static per-layer launch tables for the local experts: lidx [E] (expert -> local list), and VWork lists for
/// gate/up and down split by width
struct LocalLayer {
    /// the expert store these lists point into (words, scales)
    wbase: u64,
    sbase: u64,
    lidx: Buf,
    lists: Vec<(bool, u32, Buf, usize)>, // (gate/up?, bits, VWork entries, count)
    n_local: usize,
}

struct Shard<'m> {
    gi: usize,
    ngpu: usize,
    /// this GPU's width of the routed experts' intermediate dim (expert_tp)
    sl: usize,
    /// run the shared slice and the > 8-token expert lists on the side stream (VOLTA_OVERLAP, default on)
    overlap: bool,
    /// the DSpark draft (VOLTA_SPARK=<checkpoint dir>)
    spark: Option<spark::Spark>,
    /// this step returns greedy ids instead of logits (Engine::step_greedy)
    greedy: bool,
    spark_ids: Buf,
    /// fork / join events for the side stream
    evs: [crate::gpu::Event; 4],
    m: &'m Model,
    g: Gpu,
    k: HashMap<&'static str, Kernel>,
    layers: Vec<VLayer>,
    head: Lin,
    out_norm: Buf,
    freqs: (Buf, Buf),
    topk: crate::topk::TopK,
    exps: Experts,
    w: Work,
    sess: Vec<Sess>,
    slots: usize,
    cap: Vec<usize>, // per layer: compressed rows per slot (max_ctx / ratio)
    peers: Vec<PeerBufs>,
    owner: Arc<Vec<Vec<u8>>>,
    local: Vec<LocalLayer>,
    /// lookup-format tables (lut_tables)
    luts: Buf,
    lt: Vec<LTab>,
    /// exchange sequence base: + n_layers per step, monotonic
    seq: u32,
    /// the step's context bucket (power of two >= the longest session + 1, at least 64)
    bucket: usize,
    /// host staging for the step's Engram rows
    eng_host: Vec<f32>,
    /// this step's Engram hashes [token][engram layer] (begin)
    eng_ids: Vec<Vec<u64>>,
    /// captured steps by (tokens, bucket); None = seen once (the next one captures)
    graphs: HashMap<(usize, usize), Option<crate::gpu::GraphExec>>,
}

/// device addresses of one GPU's exchange buffers (written by every peer)
#[derive(Clone, Copy, Debug, Default)]
struct PeerBufs {
    xg: u64,
    selg: u64,
    wtg: u64,
    parts: u64,
    flags: u64,
    relay: u64,
    vx_fa: u64,
    vx_fb: u64,
}

impl<'m> Shard<'m> {
    fn new(m: &'m Model, gi: usize, o: &Opts, owner: Arc<Vec<Vec<u8>>>, tiers: Option<Arc<Vec<Vec<u8>>>>, ds: Option<&crate::dspark::Dspark>) -> Result<Shard<'m>> {
        let g = Gpu::new(gi as i32)?;
        // owner[] names expert groups of etp GPUs; this GPU holds slice gi % etp of group gi / etp's experts
        let etp = expert_tp(m, o.ngpu);
        let grp = gi / etp;
        let mut k = HashMap::new();
        for &n in KERNELS {
            k.insert(n, g.kernel(n).with_context(|| format!("kernel {n}"))?);
        }
        let c = &m.cfg;
        // dense Q8_0 -> 6-bit planes (q6.rs): ~2 GB per GPU of headroom for experts and sessions
        let dense_q6 = std::env::var("VOLTA_DENSE_Q8").is_err();
        let up = |w: &Mat| upload_dense(&g, w, dense_q6);
        // VOLTA_DENSE=dsx keeps dsx's layouts (t <= 8 per pass); default: vmoe 4-bit where the shape allows
        let v4 = std::env::var("VOLTA_DENSE").map_or(true, |v| v != "dsx");
        let f32dm = |w: &Mat| -> Result<DMat> {
            let mut v = vec![0f32; w.rows * w.cols];
            for r in 0..w.rows {
                w.dequant(r, &mut v[r * w.cols..(r + 1) * w.cols]);
            }
            Ok(DMat { buf: g.upload(&v)?, scale: None, ty: GType::F32, rows: w.rows, cols: w.cols, act_fp8: false, q8r: false, q6: None })
        };
        let lin = |w: &Mat| -> Result<Lin> {
            if v4 && v4_ok(w) {
                Ok(Lin::V(upload_v4(&g, w, 1)?))
            } else if w.act_fp8 || w.ty == GType::Fp8B32 {
                Ok(Lin::D(f32dm(w)?))
            } else {
                Ok(Lin::D(up(w)?))
            }
        };
        let fuse_qkv = |l: &crate::model::Layer| v4 && v4_ok(&l.q_a) && v4_ok(&l.kv) && l.q_a.cols == l.kv.cols && l.q_a.act_fp8 == l.kv.act_fp8;
        let lino = |w: &Option<Mat>| -> Result<Option<Lin>> { w.as_ref().map(lin).transpose() };
        // the shared expert runs split over the GPUs (inside the MoE exchange) unless the layer is rotated (the
        // exchanged x is R x there) or VOLTA_SHARED_TP=0
        let shared_tp = std::env::var("VOLTA_SHARED_TP").map_or(true, |v| v != "0") && c.inter % (32 * o.ngpu) == 0;
        let split_sh = |li: usize| shared_tp && !c.expert_rot[li];
        let upo = |w: &Option<Mat>| -> Result<Option<DMat>> { w.as_ref().map(|w| if w.act_fp8 || w.ty == GType::Fp8B32 { f32dm(w) } else { up(w) }).transpose() };
        let upv = |v: &Option<Vec<f32>>| -> Result<Option<Buf>> { v.as_ref().map(|v| f32buf(&g, v)).transpose() };
        let mut layers = vec![];
        let mut cap = vec![];
        for l in &m.layers {
            let ratio = l.ratio.max(1);
            let is_src = c.kv_sources.contains(&l.id);
            let cp = o.max_ctx / ratio;
            cap.push(cp);
            layers.push(VLayer {
                qkv_a: if fuse_qkv(l) { Some(upload_v4_cat(&g, &[&l.q_a, &l.kv])?) } else { None },
                q_a: if fuse_qkv(l) { None } else { Some(lin(&l.q_a)?) },
                q_a_norm: f32buf(&g, &l.q_a_norm)?,
                q_b: lin(&l.q_b)?,
                kv: if fuse_qkv(l) { None } else { Some(lin(&l.kv)?) },
                kv_norm: f32buf(&g, &l.kv_norm)?,
                o_a: if v4 && l.o_a.cols % 32 == 0 { Lin::V(upload_v4(&g, &l.o_a, c.o_groups)?) } else { Lin::D(up(&l.o_a)?) },
                o_b: lin(&l.o_b)?,
                sinks: f32buf(&g, &l.sinks)?,
                attn_norm: f32buf(&g, &l.attn_norm)?,
                ffn_norm: f32buf(&g, &l.ffn_norm)?,
                comp_kv: lino(&l.comp_kv)?,
                comp_gate: lino(&l.comp_gate)?,
                comp_norm: upv(&l.comp_norm)?,
                idx_q_b: lino(&l.idx_q_b)?,
                idx_proj: upo(&l.idx_proj)?,
                idx_k: lino(&l.idx_k)?,
                idx_k_norm: upv(&l.idx_k_norm)?,
                hc_attn: (f32buf(&g, &l.hc_attn_fn)?, f32buf(&g, &l.hc_attn_scale)?, f32buf(&g, &l.hc_attn_base)?),
                hc_ffn: (f32buf(&g, &l.hc_ffn_fn)?, f32buf(&g, &l.hc_ffn_scale)?, f32buf(&g, &l.hc_ffn_base)?),
                gate_w: f32buf(&g, &l.gate_w)?,
                gate_bias: f32buf(&g, &l.gate_bias)?,
                n_exp: c.n_experts,
                top_k: c.top_k,
                sh_gate: if split_sh(l.id) { None } else { Some(lin(&l.sh_gate)?) },
                sh_up: if split_sh(l.id) { None } else { Some(lin(&l.sh_up)?) },
                sh_down: if split_sh(l.id) { None } else { Some(lin(&l.sh_down)?) },
                sh_slice: if split_sh(l.id) {
                    let (inter, d, ng) = (c.inter, c.dim, o.ngpu);
                    let sl = inter / ng;
                    let r0 = gi * sl;
                    let mut gu = vec![0f32; 2 * sl * d];
                    let mut row = vec![0f32; d];
                    for (mi, mt) in [&l.sh_gate, &l.sh_up].into_iter().enumerate() {
                        for r in 0..sl {
                            mt.dequant(r0 + r, &mut row);
                            gu[(mi * sl + r) * d..(mi * sl + r + 1) * d].copy_from_slice(&row);
                        }
                    }
                    let mut dn = vec![0f32; d * sl];
                    let mut drow = vec![0f32; inter];
                    for r in 0..d {
                        l.sh_down.dequant(r, &mut drow);
                        dn[r * sl..(r + 1) * sl].copy_from_slice(&drow[r0..r0 + sl]);
                    }
                    Some((upload_v4_f32(&g, 2 * sl, d, &gu, false)?, upload_v4_f32(&g, d, sl, &dn, false)?))
                } else {
                    None
                },
                engram_kv: lino(&l.engram_kv)?,
                engram_qk: upv(&l.engram_qk)?,
                window: g.alloc_zeroed(o.slots * wring(c.window) * c.head_dim * 2)?,
                comp_cache: if is_src { Some(g.alloc_zeroed(o.slots * cp * c.head_dim * 2)?) } else { None },
                idx_cache: if is_src { Some(g.alloc_zeroed(o.slots * cp * c.idx_dim * 2)?) } else { None },
                comp_state: if is_src && ratio > 1 {
                    Some((g.alloc_zeroed(o.slots * state_ring(ratio) * c.head_dim * 4)?, g.alloc_zeroed(o.slots * state_ring(ratio) * c.head_dim * 4)?))
                } else {
                    None
                },
            });
        }
        let (d, t, tg) = (c.dim, TMAX, o.ngpu * TMAX);
        let fz = |n: usize| g.alloc_zeroed(n * 4);
        let _ = c.top_k;
        // most local experts in any layer (the byte balancing can give a GPU more than n_experts / ngpu)
        let lmax = (0..c.n_layers).map(|l| owner[l].iter().filter(|&&x| x as usize == grp).count()).max().unwrap_or(1).max(1);
        let w = Work {
            h: fz(t * c.hc * d)?,
            x: fz(t * d)?,
            a: fz(t * d)?,
            mix_a: fz(t * 24)?,
            mix_f: fz(t * 24)?,
            qr: fz(t * c.q_lora)?,
            qkv: fz(t * (c.q_lora + c.head_dim))?,
            q16: g.alloc_zeroed(t * c.n_heads * c.head_dim * 2)?,
            q: fz(t * c.n_heads * c.head_dim)?,
            kvv: fz(t * c.head_dim)?,
            o: fz(t * c.n_heads * c.head_dim)?,
            r: fz(t * c.o_groups * c.o_lora)?,
            lat: fz(t * c.head_dim)?,
            t1: fz(t * (2 * c.inter).max(5 * d).max(c.head_dim))?,
            t2: fz(t * (2 * c.inter).max(c.head_dim))?,
            iq: fz(t * c.idx_heads * c.idx_dim)?,
            iw: fz(t * c.idx_heads)?,
            score: fz(t * o.max_ctx)?,
            keep: g.alloc_zeroed(t * (o.max_ctx / c.cand_block + 64))?,
            comp_idx: g.alloc_zeroed(t * c.idx_topk * 4)?,
            attn_part: fz(t * ATTN_SPLITS * c.n_heads * (c.head_dim + 2))?,
            logits: fz(t * c.vocab)?,
            pos: g.alloc_zeroed(t * 4 * 4)?,
            win_idx: g.alloc_zeroed(t * c.window * 4)?,
            ints: g.alloc_zeroed(64 * t * 4)?,
            tabs: g.alloc_zeroed(c.n_layers * 12 * TMAX * 4 * 2 + 4096)?,
            xh: g.alloc_zeroed(TMAX * (c.n_heads * c.head_dim).max(8192) * 2)?,
            xq: fz(TMAX * (c.n_heads * c.head_dim).max(8192))?,
            shgu: fz(tg * 2 * c.inter / o.ngpu)?,
            shh: g.alloc_zeroed(tg * c.inter / o.ngpu * 2)?,
            tile32: g.upload(&(0..64u32).flat_map(|i| [0u32, 0, 32 * i, 32]).collect::<Vec<u32>>())?,
            srcl: g.alloc_zeroed((2 + o.ngpu * TMAX) * 4)?,
            srcwork: g.upload(&(0..o.ngpu as u32).flat_map(|i| [0u32, 0, 0, i]).collect::<Vec<u32>>())?,
            srcoff: g.upload(&(0..o.ngpu as i32).map(|i| i * TMAX as i32).collect::<Vec<i32>>())?,
            router: fz(t * c.n_experts)?,
            sel: g.alloc_zeroed(t * c.top_k * 4)?,
            wts: fz(t * c.top_k)?,
            topk_scratch: g.alloc_zeroed(
                crate::topk::TopK::topk_scratch(o.max_ctx, c.idx_topk, t).max(crate::topk::TopK::blocks_scratch(o.max_ctx, c.cand_block, c.cand_blocks, t)).max(64),
            )?,
            eng: fz(c.engram_layers.len() * t * (c.engram_ngram - 1) * c.engram_heads * c.engram_dim)?,
            xg: g.alloc_zeroed(tg * xrow_bytes(c))?,
            selg: g.alloc_zeroed(4)?,
            wtg: fz(1)?,
            x16: g.alloc_zeroed(t * d * 2)?,
            parts: g.alloc_zeroed(o.ngpu * t * d * 2)?, // fp16 partials
            gu: fz(tg * c.top_k * 2 * (c.inter / etp))?,
            hm: g.alloc_zeroed(tg * c.top_k * (c.inter / etp) * 2)?,
            ypart: fz(tg * d)?,
            ypart16: g.alloc_zeroed(tg * d * 2)?,
            hcp: fz(TMAX * 25 * HC_S)?,
            cnt: g.alloc_zeroed((3 * lmax + 1) * 4)?, // cnt [lmax] | off [lmax + 1] | cur [lmax]
            hot: g.alloc_zeroed(lmax * 32)?,
            nhot: g.alloc_zeroed(4)?,
            tok: g.alloc_zeroed(tg * c.top_k * 4)?,
            wt: fz(tg * c.top_k)?,
            ntok: g.alloc_zeroed(o.ngpu * 4)?,
            flags: g.alloc_zeroed(2 * o.ngpu * 4)?,
            vx_fa: g.alloc_zeroed(3 * 8 * VX_BLOCKS * 4)?,
            vx_fb: g.alloc_zeroed(3 * 8 * VX_BLOCKS * 4)?,
            relay: g.alloc_zeroed(32 * t * d * 2)?,
            vx_pa: g.alloc_zeroed(24 * 8)?,
            vx_pb: g.alloc_zeroed(24 * 8)?,
            vx_bytes: g.alloc_zeroed(16 * 4)?,
            seqd: g.alloc_zeroed(4)?,
        };
        let luts = g.upload(&lut_tables())?;
        let exps = Experts::build(&g, m, gi, etp, &owner, tiers.as_deref().map(|v| v.as_slice()), luts.ptr)?;
        // per-layer local lists, in expert order
        let mut local = vec![];
        for l in 0..c.n_layers {
            let es: Vec<usize> = (0..c.n_experts).filter(|&e| owner[l][e] as usize == grp).collect();
            local.push(local_layer(&g, &exps, l, &es, c.n_experts)?);
        }
        // the draft layers go after the main ones (VOLTA_SPARK)
        let spark = match ds {
            Some(ds) => {
                let (dl, dlocal, sp) = spark::build(&g, m, ds, gi, o.ngpu, etp, o.slots, luts.ptr)?;
                layers.extend(dl);
                local.extend(dlocal);
                Some(sp)
            }
            None => None,
        };
        let nl = c.n_layers;
        Ok(Shard {
            gi,
            ngpu: o.ngpu,
            sl: c.inter / etp,
            overlap: std::env::var("VOLTA_OVERLAP").map_or(true, |v| v != "0"),
            spark,
            greedy: false,
            spark_ids: g.alloc_zeroed(TMAX * 4)?,
            evs: [g.event()?, g.event()?, g.event()?, g.event()?],
            m,
            // VOLTA_HEAD=q6 keeps the head in dsx's 6-bit planes (two GEMV passes per 16 tokens)
            head: if std::env::var("VOLTA_HEAD").is_ok_and(|v| v == "q6") || !v4_ok(&m.head) { Lin::D(up(&m.head)?) } else { Lin::V(upload_v4(&g, &m.head, 1)?) },
            out_norm: f32buf(&g, &m.out_norm)?,
            freqs: (f32buf(&g, &m.ropes.0.freqs)?, f32buf(&g, &m.ropes.1.freqs)?),
            topk: crate::topk::TopK::new(&g)?,
            k,
            layers,
            exps,
            w,
            sess: vec![Sess { n_comp: vec![0; nl], n_idx: vec![0; nl], ..Default::default() }; o.slots],
            slots: o.slots,
            cap,
            peers: vec![],
            owner,
            local,
            luts,
            lt: vec![],
            seq: 0,
            bucket: 64,
            eng_host: vec![],
            eng_ids: vec![],
            graphs: HashMap::new(),
            g,
        })
    }

    fn peer_bufs(&self) -> PeerBufs {
        let w = &self.w;
        PeerBufs { xg: w.xg.ptr, selg: w.selg.ptr, wtg: w.wtg.ptr, parts: w.parts.ptr, flags: w.flags.ptr, relay: w.relay.ptr, vx_fa: w.vx_fa.ptr, vx_fb: w.vx_fb.ptr }
    }

    // ------------------------------------------------------------ kernel helpers (dsx's GEMV dispatch, t <= 8)

    /// dsx GEMV over t tokens, in passes of 8
    fn gemv(&self, w: &DMat, x: u64, y: u64, t: usize) -> Result<()> {
        for c0 in (0..t).step_by(8) {
            let n = 8.min(t - c0);
            self.gemv_rows(w, 0, w.rows, x + (c0 * w.cols * 4) as u64, w.cols, y + (c0 * w.rows * 4) as u64, w.rows, n)?;
        }
        Ok(())
    }

    /// y[t][rows] = W x[t][cols] (contiguous rows)
    fn mm(&self, w: &Lin, x: u64, y: u64, t: usize) -> Result<()> {
        match w {
            Lin::D(d) => self.gemv(d, x, y, t),
            Lin::V(v) => self.vmm(v, x, v.cols, y, t),
        }
    }

    /// vmoe 4-bit, every partition in one launch: x [t][parts * cols] f32 (row stride ldx), y [t][rows] f32 where
    /// partition p of token t reads x[t][p*cols..] and writes y[t][p*prows..]
    fn vmm(&self, v: &VMat, x: u64, ldx: usize, y: u64, t: usize) -> Result<()> {
        ensure!(t >= 1 && t <= TMAX, "vmm over {t} tokens");
        let (n, k, parts) = (v.rows / v.parts, v.cols, v.parts);
        let w = &self.w;
        let kx = k * parts;
        ensure!(t * kx * 2 <= w.xh.bytes, "vmm: {t} x {kx} activations overflow the fp16 scratch");
        // FP8 inputs: the reference rounds them to FP8 (block 32) first
        let ny = n * parts;
        self.g.launch(self.k["vprep"], ((kx.max(ny) as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(x), Arg::I(ldx as i32), Arg::P(w.xh.ptr), Arg::I(kx as i32), Arg::I(v.act_fp8 as i32), Arg::P(y), Arg::I(ny as i32)])?;
        let nb = if t <= 8 { 1usize } else if t <= 16 { 2 } else { 4 };
        // one row group per warp, and the longest K chunk that still gives >= 3 blocks per SM (tests/dense_test.cu:
        // 145 us per layer at 8 tokens, against 250 with four row groups per warp and ~2 blocks per SM)
        let rtiles = n.div_ceil(8 * 32);
        let kc = even_kc(k, rtiles * parts, 3 * self.g.sms);
        let kern = self.k[match nb { 1 => "vmoe_b4_n1_r1", 2 => "vmoe_b4_n2_r1", _ => "vmoe_b4_n4" }];
        let smem = nb * 8 * (kc + 8) * 2;
        if smem > 48 * 1024 {
            self.g.set_smem(kern, smem)?;
        }
        let work = v.work.ptr + ((t - 1) * parts * 16) as u64;
        let (rows_in, rows_out) = if parts > 1 { (v.prow.ptr, v.prow.ptr) } else { (0, 0) };
        self.g.launch(
            kern,
            (rtiles as u32, parts as u32, (k as u32).div_ceil(kc as u32)),
            256,
            smem as u32,
            &[Arg::P(v.w.ptr), Arg::P(v.s.ptr), Arg::P(work), Arg::P(rows_in), Arg::P(rows_out), Arg::P(0), Arg::P(w.xh.ptr), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(kc as i32), Arg::I(n as i32), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::I(k as i32), Arg::P(0)],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_rows(&self, w: &DMat, r0: usize, n: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize) -> Result<()> {
        ensure!(t <= TMAX, "gemv over {t} tokens");
        if w.q6.is_some() {
            ensure!(ldx == w.cols && ldy == n, "strided q6 GEMV");
            let ([lo, hi, sc, d], kn, _) = w.planes(r0);
            let k = w.cols;
            let r4 = if kn == "gemv_q6p" { n >= 512 } else { n >= 2048 };
            let (kn, rpb) = if kn == "gemv_q6p" && n >= 4096 && k <= 4096 && (k / 32) % 8 == 0 {
                ("gemv_q6s8", 64)
            } else if r4 {
                (if kn == "gemv_q6p" { "gemv_q6g4" } else { "gemv_q5p4" }, 16)
            } else {
                (kn, 4)
            };
            return self.g.launch(
                self.k[kn],
                ((n as u32).div_ceil(rpb), 1, 1),
                128,
                0,
                &[Arg::P(lo), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(t as i32), Arg::I(k as i32), Arg::I(n as i32), Arg::P(hi), Arg::P(sc), Arg::P(d), Arg::I(n as i32), Arg::I(0)],
            );
        }
        let kname = match w.ty {
            GType::Q8_0 if w.q8r => "gemv_q8r",
            GType::Q8_0 => "gemv_q8_0",
            GType::F16 => "gemv_f16",
            GType::BF16 => "gemv_bf16",
            GType::F32 => "gemv_f32",
            GType::Fp8B32 => "gemv_fp8b32",
            GType::Q2_0 => "gemv_q2_0",
            GType::Ptq1_0 => "gemv_ptq1_0",
            GType::Q3K => "gemv_q3_k",
            ty => bail!("no GPU gemv for {ty:?}"),
        };
        ensure!(!w.act_fp8, "FP8-activation weights (HF checkpoint) are not wired; use a GGUF export");
        let (wp, sp, _) = w.ptrs(r0);
        let sk = n < 2048 && matches!(kname, "gemv_q8r" | "gemv_f16" | "gemv_bf16" | "gemv_f32");
        let kname = if sk { match kname { "gemv_q8r" => "gemv_q8r_sk", "gemv_f16" => "gemv_f16_sk", "gemv_bf16" => "gemv_bf16_sk", _ => "gemv_f32_sk" } } else { kname };
        let per_block = if sk { 1 } else { 4 };
        self.g.launch(
            self.k[kname],
            ((n as u32).div_ceil(per_block), 1, 1),
            128,
            0,
            &[Arg::P(wp), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(w.cols as i32), Arg::I(t as i32), Arg::I(ldx as i32), Arg::I(ldy as i32), Arg::P(sp)],
        )
    }

    fn rms_rows(&self, x: u64, w: &Buf, out: u64, n: usize, rows: usize, round: bool) -> Result<()> {
        self.g.launch(self.k["rmsnorm"], (rows as u32, 1, 1), 1024, 0, &[Arg::P(x), Arg::P(w.ptr), Arg::P(out), Arg::I(n as i32), Arg::I(n as i32), Arg::I(n as i32), Arg::F(self.m.cfg.eps), Arg::I(round as i32)])
    }

    fn rope_rows(&self, x: u64, heads: usize, hd: usize, compressed: bool, pos: u64, rows: usize, inverse: bool) -> Result<()> {
        let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
        self.g.launch(self.k["rope_pos"], (rows as u32, 1, 1), 256, 0, &[Arg::P(x), Arg::P(f.ptr), Arg::I(heads as i32), Arg::I(hd as i32), Arg::I(self.m.cfg.rope_dim as i32), Arg::I((heads * hd) as i32), Arg::P(pos), Arg::I(inverse as i32)])
    }

    fn fake_fp4(&self, x: u64, n: usize, block: usize, e4m3s: bool) -> Result<()> {
        self.g.launch(self.k["fake_fp4"], ((n as u32 / 32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(x), Arg::I(n as i32), Arg::I(block as i32), Arg::I(e4m3s as i32)])
    }

    fn hc_mixes_rows(&self, fnw: &(Buf, Buf, Buf), h: u64, out: u64, n: usize) -> Result<()> {
        let c = &self.m.cfg;
        // each fn row is read once for every token, in HC_S chunks; the finish sums the chunks
        let (hcd, p) = (c.hc * c.dim, self.w.hcp.ptr);
        ensure!(hcd % (128 * HC_S) == 0 && hcd / HC_S <= 1024, "hc dots chunking");
        self.g.launch(self.k["vhc_dots"], (HC_S as u32, 6, 1), 256, 0, &[Arg::P(fnw.0.ptr), Arg::P(h), Arg::P(p), Arg::I(hcd as i32), Arg::I(n as i32), Arg::I(HC_S as i32)])?;
        self.g.launch(self.k["vhc_finish"], ((n as u32).div_ceil(4), 1, 1), 128, 0, &[Arg::P(p), Arg::P(out), Arg::I(hcd as i32), Arg::I(HC_S as i32), Arg::F(c.eps), Arg::P(fnw.1.ptr), Arg::P(fnw.2.ptr), Arg::F(c.hc_eps), Arg::I(c.sinkhorn_iters as i32), Arg::I(n as i32)])
    }

    // ------------------------------------------------------------ one step's halves

    /// Begin a step: tokens (slot, token) on this GPU. Embeddings, positions, window rows, and every layer's
    /// session tables (one upload).
    fn begin(&mut self, toks: &[(usize, u32)]) -> Result<()> {
        let t0 = std::time::Instant::now();
        let m = self.m;
        let c = &m.cfg;
        let (d, hc, win) = (c.dim, c.hc, c.window);
        self.g.bind();
        let t = toks.len();
        let mut hbuf = vec![0f32; t * hc * d];
        let mut e = vec![0f32; d];
        let mut widx = vec![-1i32; t * win];
        let pos = self.tok_pos(toks);
        let (wr, ne) = (wring(win), c.engram_layers.len());
        self.eng_ids.clear();
        for (i, &(s, tk)) in toks.iter().enumerate() {
            ensure!(s < self.slots, "slot {s}");
            let p = pos[i];
            ensure!(p < self.cap[0], "slot {s}: context full");
            // a session's tokens arrive in order: each token's Engram n-grams end at itself
            self.sess[s].engram_hist.push(c.engram_token_map[tk as usize] as i64);
            if m.engram.is_some() {
                for eidx in 0..ne {
                    let h = m.engram_hashes_for(&self.sess[s].engram_hist, eidx);
                    self.eng_ids.push(h);
                }
            }
            m.embed.dequant(tk as usize, &mut e);
            for k in 0..hc {
                hbuf[(i * hc + k) * d..(i * hc + k + 1) * d].copy_from_slice(&e);
            }
            // the window's positions p - win + 1 ..= p, oldest first, as pool rows of this slot; -1 = before the start
            for j in 0..win {
                let q = (p + j + 1) as isize - win as isize;
                widx[i * win + j] = if q < 0 { -1 } else { (s * wr + q as usize % wr) as i32 };
            }
        }
        self.g.htod_sync(self.w.h.ptr, &hbuf)?;
        self.g.htod_sync(self.w.win_idx.ptr, &widx)?;
        let mut pre = vec![0f32; t * 24];
        for i in 0..t {
            pre[i * 24] = 1.0;
        }
        self.g.htod_sync(self.w.mix_f.ptr, &pre)?;
        let t1 = std::time::Instant::now();
        self.gather_engram(toks)?;
        let t2 = std::time::Instant::now();
        self.plan(toks)?;
        if self.gi == 0 && std::env::var_os("VOLTA_PROF_HOST").is_some() {
            eprintln!("GPU0 begin: embed+tables {:.2} ms, engram {:.2} ms, plan {:.2} ms", (t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// each token's position: its session's position plus the session's earlier tokens in this step
    fn tok_pos(&self, toks: &[(usize, u32)]) -> Vec<usize> {
        let mut seen: HashMap<usize, usize> = HashMap::new();
        toks.iter()
            .map(|&(s, _)| {
                let k = seen.entry(s).or_insert(0);
                *k += 1;
                self.sess[s].pos + *k - 1
            })
            .collect()
    }

    /// every layer's per-token tables for this step, uploaded in one copy (pointers into w.tabs)
    fn plan(&mut self, toks: &[(usize, u32)]) -> Result<()> {
        let c = &self.m.cfg;
        let (win, t) = (c.window, toks.len());
        let mut host: Vec<i32> = vec![];
        let base = self.w.tabs.ptr;
        // every table is t long and every layer pushes the same ones, so the pointers depend on t only (graphs)
        let mut push = |v: Vec<i32>| -> u64 {
            let off = host.len();
            host.extend(v);
            host.resize(host.len().div_ceil(32) * 32, 0);
            base + (off * 4) as u64
        };
        // each token's own position (a session may have several tokens, in order)
        let pos = self.tok_pos(toks);
        let wr = wring(win);
        let pos_p = push(pos.iter().map(|&p| p as i32).collect());
        let wrow_p = push(toks.iter().zip(&pos).map(|(&(s, _), &p)| (s * wr + p % wr) as i32).collect());
        let bucket = self.bucket;
        let mut lt = vec![];
        let (mut comp_src, mut idxk_src) = (usize::MAX, usize::MAX);
        let mut kcomp = 0usize;
        for li in 0..c.n_layers {
            let ratio = self.m.layers[li].ratio;
            let mut x = LTab { pos: pos_p, wrow: wrow_p, ..Default::default() };
            if ratio > 0 {
                let is_src = c.kv_sources.contains(&li);
                if is_src {
                    comp_src = li;
                    idxk_src = li;
                }
                let r = ratio.max(1);
                let sr = state_ring(r);
                // tokens closing a compression group (all of them at ratio 1); rows counted in token order
                let (mut lat_base, mut comp_rows, mut idx_rows) = (vec![], vec![], vec![]);
                for (&(s, _), &p) in toks.iter().zip(&pos) {
                    let closes = is_src && (p + 1) % ratio == 0;
                    lat_base.push(if closes { (s * sr + (p + 1 - r) % sr) as i32 } else { -1 });
                    comp_rows.push(if closes { (s * self.cap[li] + self.sess[s].n_comp[li]) as i32 } else { -1 });
                    idx_rows.push(if closes { (s * self.cap[li] + self.sess[s].n_idx[li]) as i32 } else { -1 });
                    if closes {
                        self.sess[s].n_comp[li] += 1;
                        self.sess[s].n_idx[li] += 1;
                    }
                }
                x.srow = push(toks.iter().zip(&pos).map(|(&(s, _), &p)| (s * sr + p % sr) as i32).collect());
                x.lat_base = push(lat_base);
                x.lat_pos = push(pos.iter().map(|&p| (p + 1).saturating_sub(r) as i32).collect());
                x.comp_rows = push(comp_rows);
                x.idx_rows = push(idx_rows);
                x.comp_src = comp_src;
                x.idxk_src = idxk_src;
                if c.index_sources.contains(&li) {
                    x.ends = push(pos.iter().map(|&p| ((p + 1) / ratio) as i32).collect());
                    x.koff = push(toks.iter().map(|&(s, _)| (s * self.cap[idxk_src]) as i32).collect());
                    x.coff = push(toks.iter().map(|&(s, _)| (s * self.cap[comp_src]) as i32).collect());
                    // the bucketed width (power of two >= every token's visible rows), so the launch shape is stable
                    x.n_max = (bucket / ratio).min(self.cap[idxk_src]).max(1);
                    kcomp = c.idx_topk;
                }
                x.kcomp = kcomp;
            }
            lt.push(x);
        }
        let _ = t;
        ensure!(host.len() * 4 <= self.w.tabs.bytes, "step tables overflow");
        self.g.htod_sync(self.w.tabs.ptr, &host)?;
        self.lt = lt;
        Ok(())
    }

    /// every engram layer's rows for this step's tokens (they depend on token ids only), gathered in parallel from
    /// the tables (page cache) and uploaded once: eng [layer][TMAX][rows][dim]
    fn gather_engram(&mut self, toks: &[(usize, u32)]) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let Some(tabs) = m.engram.as_ref() else { return Ok(()) };
        let (hd, per, t) = (c.engram_dim, (c.engram_ngram - 1) * c.engram_heads, toks.len());
        let ne = c.engram_layers.len();
        // hashes from begin (eng_ids [token][engram layer]), each from the history up to its token
        let mut ids = vec![0u64; ne * t * per];
        for eidx in 0..ne {
            for i in 0..t {
                ids[(eidx * t + i) * per..(eidx * t + i + 1) * per].copy_from_slice(&self.eng_ids[i * ne + eidx]);
            }
        }
        let layers: Vec<_> = (0..ne).map(|e| tabs.layer(e, hd)).collect::<Result<_>>()?;
        // serial on this GPU's thread: the 8 GPU threads gathering at once through rayon's shared pool were slower
        // than ~1 us per in-RAM row
        let _ = rayon::current_num_threads;
        self.eng_host.resize(ne * TMAX * per * hd, 0.0);
        // FP8 E4M3 x E8M0 through lookup tables (the reference's per-value powi was ~15 us per row)
        static LUT: std::sync::OnceLock<([f32; 256], [f32; 256])> = std::sync::OnceLock::new();
        let (f8, e8) = LUT.get_or_init(|| {
            let mut a = [0f32; 256];
            let mut b = [0f32; 256];
            for i in 0..256 {
                a[i] = crate::model::fp8_e4m3_to_f32(i as u8);
                b[i] = crate::model::e8m0_to_f32(i as u8);
            }
            (a, b)
        });
        for eidx in 0..ne {
            for i in 0..t {
                for j in 0..per {
                    let o = ((eidx * TMAX + i) * per + j) * hd;
                    let r = ids[(eidx * t + i) * per + j] as usize;
                    let out = &mut self.eng_host[o..o + hd];
                    match &layers[eidx] {
                        crate::model::EngramLayer::Fp8 { w, s, dim } => {
                            let (wr, sr) = (&w[r * dim..r * dim + dim], &s[r * (dim / 32)..r * (dim / 32) + dim / 32]);
                            for k in 0..*dim {
                                out[k] = crate::numerics::round_bf16(f8[wr[k] as usize] * e8[sr[k / 32] as usize]);
                            }
                        }
                        other => other.row(r, out),
                    }
                }
            }
        }
        // upload only the rows in use (each engram layer's first t tokens)
        for eidx in 0..ne {
            let o = eidx * TMAX * per * hd;
            self.g.htod_sync(self.w.eng.ptr + (o * 4) as u64, &self.eng_host[o..o + t * per * hd])?;
        }
        Ok(())
    }

    fn engram(&mut self, toks: &[(usize, u32)], li: usize, eidx: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let tabs = m.engram.as_ref().context("engram tables not loaded (--engram DIR)")?;
        let tab = tabs.layer(eidx, c.engram_dim)?;
        let hd = c.engram_dim;
        let per = (c.engram_ngram - 1) * c.engram_heads;
        let t = toks.len();
        let _ = (tab, toks);
        // rows were gathered for every engram layer at step start (gather_engram)
        let src = self.w.eng.ptr + (eidx * TMAX * per * hd * 4) as u64;
        let l = &self.layers[li];
        self.mm(l.engram_kv.as_ref().unwrap(), src, self.w.t1.ptr, t)?;
        self.g.launch(self.k["engram_gate_t"], (c.hc as u32, t as u32, 1), 1024, 0, &[Arg::P(self.w.h.ptr), Arg::P(self.w.t1.ptr), Arg::P(l.engram_qk.as_ref().unwrap().ptr), Arg::I(c.dim as i32), Arg::F(c.eps)])
    }

    /// attention half of layer li for this GPU's tokens; leaves the output in w.a
    fn attention(&mut self, toks: &[(usize, u32)], li: usize, st: &mut StepState) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let t = toks.len();
        let (hd, nh, win) = (c.head_dim, c.n_heads, c.window);
        let ratio = m.layers[li].ratio;
        let compressed = ratio > 0;
        let x = std::mem::take(&mut self.lt[li]);
        let w = &self.w;
        let l = &self.layers[li];
        // q latent and kv from one launch when fused: rows [t][q_lora + hd], the q latent normalized out to qr
        let (kvp, kvld) = if let Some(f) = &l.qkv_a {
            let ld = c.q_lora + hd;
            self.vmm(f, w.x.ptr, f.cols, w.qkv.ptr, t)?;
            self.g.launch(self.k["rmsnorm"], (t as u32, 1, 1), 1024, 0, &[Arg::P(w.qkv.ptr), Arg::P(l.q_a_norm.ptr), Arg::P(w.qr.ptr), Arg::I(c.q_lora as i32), Arg::I(ld as i32), Arg::I(c.q_lora as i32), Arg::F(c.eps), Arg::I(1)])?;
            (w.qkv.ptr + (c.q_lora * 4) as u64, ld)
        } else {
            self.mm(l.q_a.as_ref().unwrap(), w.x.ptr, w.qr.ptr, t)?;
            self.rms_rows(w.qr.ptr, &l.q_a_norm, w.qr.ptr, c.q_lora, t, true)?;
            self.mm(l.kv.as_ref().unwrap(), w.x.ptr, w.kvv.ptr, t)?;
            (w.kvv.ptr, hd)
        };
        self.mm(&l.q_b, w.qr.ptr, w.q.ptr, t)?;
        self.rope_rows(w.q.ptr, nh, hd, compressed, x.pos, t, false)?;
        let f = if compressed { &self.freqs.1 } else { &self.freqs.0 };
        self.g.launch(self.k["vkv_store"], (t as u32, 1, 1), 1024, 0, &[Arg::P(kvp), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(f.ptr), Arg::I(c.rope_dim as i32), Arg::P(x.pos), Arg::P(l.window.ptr), Arg::P(x.wrow), Arg::I(kvld as i32)])?;
        if compressed {
            let is_src = c.kv_sources.contains(&li);
            if is_src {
                if ratio == 1 {
                    self.mm(l.comp_kv.as_ref().unwrap(), w.x.ptr, w.lat.ptr, t)?;
                    self.rms_rows(w.lat.ptr, l.comp_norm.as_ref().unwrap(), w.lat.ptr, hd, t, false)?;
                } else {
                    let (ks, ss) = l.comp_state.as_ref().unwrap();
                    self.mm(l.comp_kv.as_ref().unwrap(), w.x.ptr, w.t1.ptr, t)?;
                    self.mm(l.comp_gate.as_ref().unwrap(), w.x.ptr, w.t2.ptr, t)?;
                    self.g.launch(self.k["vrows_scatter"], (t as u32, 1, 1), 256, 0, &[Arg::P(w.t1.ptr), Arg::P(ks.ptr), Arg::P(x.srow), Arg::I(hd as i32)])?;
                    self.g.launch(self.k["vrows_scatter"], (t as u32, 1, 1), 256, 0, &[Arg::P(w.t2.ptr), Arg::P(ss.ptr), Arg::P(x.srow), Arg::I(hd as i32)])?;
                    // every token, masked: those closing no group skip (lat_base -1) and store nothing below
                    self.g.launch(self.k["vcompress"], (t as u32, 1, 1), 512, 0, &[Arg::P(ks.ptr), Arg::P(ss.ptr), Arg::P(x.lat_base), Arg::P(w.lat.ptr), Arg::I(ratio as i32), Arg::I(hd as i32)])?;
                    self.rms_rows(w.lat.ptr, l.comp_norm.as_ref().unwrap(), w.lat.ptr, hd, t, false)?;
                }
            }
            if c.index_sources.contains(&li) {
                self.indexer(li, t, &x, st)?;
            }
            if is_src {
                let (w, l) = (&self.w, &self.layers[li]);
                self.rope_rows(w.lat.ptr, 1, hd, true, x.lat_pos, t, false)?;
                self.fake_fp4(w.lat.ptr, t * hd, 16, true)?;
                self.g.launch(self.k["vstore_bf16_rows"], (t as u32, 1, 1), 256, 0, &[Arg::P(w.lat.ptr), Arg::P(l.comp_cache.as_ref().unwrap().ptr), Arg::P(x.comp_rows), Arg::I(hd as i32)])?;
            }
        }
        let kcomp = x.kcomp;
        let w = &self.w;
        let l = &self.layers[li];
        let comp_ptr = if kcomp > 0 { self.layers[x.comp_src].comp_cache.as_ref().unwrap().ptr } else { 0 };
        let scale = 1.0 / (hd as f32).sqrt();
        // 160 rows per block (5 sub-tiles, online softmax): 4 partials per head for 640 rows
        let s = (win + kcomp).div_ceil(160);
        ensure!(s <= ATTN_SPLITS, "attention over {} rows", win + kcomp);
        self.attn_tiles(t, l.window.ptr, w.win_idx.ptr, win, comp_ptr, w.comp_idx.ptr, kcomp, s, scale)?;
        self.g.launch(self.k["sparse_attn_merge"], (t as u32, nh as u32, 1), 256, 0, &[Arg::P(w.attn_part.ptr), Arg::I(s as i32), Arg::P(l.sinks.ptr), Arg::P(w.o.ptr), Arg::I(nh as i32), Arg::I(hd as i32)])?;
        self.rope_rows(w.o.ptr, nh, hd, compressed, x.pos, t, true)?;
        self.g.launch(self.k["fake_fp8"], (((t * nh * hd) as u32 / 32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(w.o.ptr), Arg::I((t * nh * hd) as i32)])?;
        // grouped low-rank output projection: group gi of token i reads o[i][gi*gsz..]
        let gsz = nh * hd / c.o_groups;
        let rows = c.o_groups * c.o_lora;
        match &l.o_a {
            Lin::V(v) => self.vmm(v, w.o.ptr, nh * hd, w.r.ptr, t)?,
            Lin::D(oa) if oa.q6.is_some() && oa.ty != GType::Q5K && c.o_lora % 16 == 0 => {
                // 6-bit planes: every group and every token in one launch (group gi of token i reads o[i][gi*gsz..])
                let ([lo, hi, sc, d], _, _) = oa.planes(0);
                let (kn, rpb) = if rows >= 4096 && gsz <= 4096 && (gsz / 32) % 8 == 0 { ("gemv_q6s8", 64) } else { ("gemv_q6g4", 16) };
                for c0 in (0..t).step_by(8) {
                    let tn = 8.min(t - c0);
                    self.g.launch(self.k[kn], ((rows as u32).div_ceil(rpb), 1, 1), 128, 0, &[Arg::P(lo), Arg::P(w.o.ptr + (c0 * nh * hd * 4) as u64), Arg::P(w.r.ptr + (c0 * rows * 4) as u64), Arg::I(rows as i32), Arg::I(gsz as i32), Arg::I(tn as i32), Arg::I((nh * hd) as i32), Arg::I(rows as i32), Arg::P(hi), Arg::P(sc), Arg::P(d), Arg::I(c.o_lora as i32), Arg::I(gsz as i32)])?;
                }
            }
            Lin::D(oa) => {
                for c0 in (0..t).step_by(8) {
                    let tn = 8.min(t - c0);
                    for gi in 0..c.o_groups {
                        self.gemv_rows(oa, gi * c.o_lora, c.o_lora, w.o.ptr + ((c0 * nh * hd + gi * gsz) * 4) as u64, nh * hd, w.r.ptr + ((c0 * rows + gi * c.o_lora) * 4) as u64, rows, tn)?;
                    }
                }
            }
        }
        self.g.launch(self.k["round_bf16"], (((t * c.o_groups * c.o_lora) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.r.ptr), Arg::I((t * c.o_groups * c.o_lora) as i32)])?;
        self.mm(&l.o_b, w.r.ptr, w.a.ptr, t)
    }

    fn indexer(&mut self, li: usize, t: usize, x: &LTab, st: &mut StepState) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (id, nh) = (c.idx_dim, c.idx_heads);
        let w = &self.w;
        if c.kv_sources.contains(&li) {
            // every token, masked by idx_rows (-1 = no latent this step)
            let l = &self.layers[li];
            self.mm(l.idx_k.as_ref().unwrap(), w.lat.ptr, w.t1.ptr, t)?;
            self.rms_rows(w.t1.ptr, l.idx_k_norm.as_ref().unwrap(), w.t1.ptr, id, t, false)?;
            self.rope_rows(w.t1.ptr, 1, id, true, x.lat_pos, t, false)?;
            self.fake_fp4(w.t1.ptr, t * id, 32, false)?;
            self.g.launch(self.k["vstore_bf16_rows"], (t as u32, 1, 1), 256, 0, &[Arg::P(w.t1.ptr), Arg::P(l.idx_cache.as_ref().unwrap().ptr), Arg::P(x.idx_rows), Arg::I(id as i32)])?;
        }
        // scores over the bucketed width: a token's rows past its visible count are -inf (ends)
        let n_max = x.n_max;
        let l = &self.layers[li];
        self.mm(l.idx_q_b.as_ref().unwrap(), w.qr.ptr, w.iq.ptr, t)?;
        self.rope_rows(w.iq.ptr, nh, id, true, x.pos, t, false)?;
        self.fake_fp4(w.iq.ptr, t * nh * id, 32, false)?;
        self.gemv(l.idx_proj.as_ref().unwrap(), w.x.ptr, w.iw.ptr, t)?;
        let wscale = (id as f32).powf(-0.5) * (nh as f32).powf(-0.5);
        self.g.launch(self.k["scale_bf16"], (((t * nh) as u32).div_ceil(32), 1, 1), 32, 0, &[Arg::P(w.iw.ptr), Arg::I((t * nh) as i32), Arg::F(wscale)])?;
        let kc = self.layers[x.idxk_src].idx_cache.as_ref().unwrap().ptr;
        self.g.launch(self.k["vindexer_score"], ((n_max as u32).div_ceil(8), t as u32, 1), 256, 0, &[Arg::P(w.iq.ptr), Arg::P(kc), Arg::P(x.koff), Arg::P(w.iw.ptr), Arg::P(w.score.ptr), Arg::P(x.ends), Arg::I(n_max as i32), Arg::I(nh as i32), Arg::I(id as i32)])?;
        let rows = crate::topk::Rows { scores: w.score.ptr, lds: n_max, rows: t, ns: x.ends, n_max };
        let ldk = n_max.div_ceil(c.cand_block);
        if li == c.cand_layer {
            self.topk.blocks(&self.g, &rows, c.cand_block, c.cand_blocks, w.keep.ptr, ldk, w.topk_scratch.ptr)?;
            st.keep_ld = ldk;
        }
        let mask = if li > c.cand_layer { w.keep.ptr } else { 0 };
        let ldm = if li > c.cand_layer { st.keep_ld } else { ldk };
        self.topk.topk(&self.g, &rows, c.idx_topk, mask, ldm, c.cand_block, w.comp_idx.ptr, c.idx_topk, w.topk_scratch.ptr)?;
        // positions -> rows of the comp cache pool (the kv source's slot block)
        self.g.launch(self.k["vidx_offset"], (t as u32, 1, 1), 256, 0, &[Arg::P(w.comp_idx.ptr), Arg::P(x.coff), Arg::I(c.idx_topk as i32)])?;
        st.topk_k = c.idx_topk;
        Ok(())
    }

    /// router + shared expert for this GPU's tokens, then publish x / picks to every peer's exchange slot
    fn moe_dispatch(&mut self, t: usize, li: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (d, inter) = (c.dim, c.inter);
        let w = &self.w;
        let l = &self.layers[li];
        let (ne, kk) = (l.n_exp, l.top_k);
        ensure!(ne % 32 == 0 && d % 256 == 0, "router shape");
        self.g.memset(w.router.ptr, t * ne * 4)?;
        self.g.launch(self.k["vrouter"], ((ne / 32) as u32, (d / 256) as u32, 1), 256, 0, &[Arg::P(l.gate_w.ptr), Arg::P(w.x.ptr), Arg::P(w.router.ptr), Arg::I(ne as i32), Arg::I(d as i32), Arg::I(t as i32)])?;
        self.g.launch(self.k["router_topk"], (t as u32, 1, 1), 32, 0, &[Arg::P(w.router.ptr), Arg::P(l.gate_bias.ptr), Arg::P(w.sel.ptr), Arg::P(w.wts.ptr), Arg::I(ne as i32), Arg::I(kk as i32), Arg::F(c.route_scale), Arg::P(0)])?;
        if let (Some(sg), Some(su), Some(sd)) = (&l.sh_gate, &l.sh_up, &l.sh_down) {
            // shared expert -> a (local)
            let (gp, up, hp) = (w.t1.ptr, w.t1.ptr + (t * inter * 4) as u64, w.t2.ptr);
            self.mm(sg, w.x.ptr, gp, t)?;
            self.mm(su, w.x.ptr, up, t)?;
            self.g.launch(self.k["vswiglu_f32"], ((inter as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(gp), Arg::P(up), Arg::P(hp), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
            self.mm(sd, hp, w.a.ptr, t)?;
        } else {
            // the shared expert arrives with the partial sums (split over the GPUs)
            self.g.memset(w.a.ptr, t * d * 4)?;
        }
        // fp16 x (rotated for rotated layers) + picks + weights into our rows of the exchange, then one copy per peer
        let me = self.gi;
        let rb = xrow_bytes(c);
        let xo = (me * TMAX * rb) as u64;
        self.g.launch(self.k["vto_f16"], (((t * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.x.ptr), Arg::P(w.x16.ptr), Arg::I((t * d) as i32)])?;
        if c.expert_rot.get(li).copied().unwrap_or(false) {
            // rotated layers store W R: the routed experts take R x (the shared expert takes x)
            self.g.launch(self.k["fwht_rows_f16"], ((d / 128) as u32, t as u32, 1), 128, 0, &[Arg::P(w.x16.ptr), Arg::I(d as i32)])?;
        }
        self.g.launch(self.k["vpack_x"], (t as u32, 1, 1), 256, 0, &[Arg::P(w.x16.ptr), Arg::P(w.sel.ptr), Arg::P(w.wts.ptr), Arg::P(w.xg.ptr + xo), Arg::I(d as i32), Arg::I(kk as i32), Arg::I(rb as i32)])
    }

    /// cube all-gather of every GPU's exchange rows (vx_allgather)
    fn vx_gather(&self, li: usize) -> Result<()> {
        let rb = xrow_bytes(&self.m.cfg);
        self.g.launch(self.k["vx_allgather"], (VX_BLOCKS as u32, 1, 1), 256, 0, &[Arg::P(self.w.vx_pa.ptr), Arg::I(self.gi as i32), Arg::P((TMAX * rb) as u64), Arg::P(self.w.vx_bytes.ptr), Arg::P(self.w.seqd.ptr), Arg::I(li as i32)])
    }

    /// cube all-to-all of the partial sums: ypart rows of GPU d's tokens -> d's parts[me] (vx_alltoall)
    fn vx_scatter(&self, li: usize) -> Result<()> {
        let d = self.m.cfg.dim;
        let tg = self.ngpu * TMAX;
        let w = &self.w;
        self.g.launch(self.k["vto_f16"], (((tg * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.ypart.ptr), Arg::P(w.ypart16.ptr), Arg::I((tg * d) as i32)])?;
        let slot = (TMAX * d * 2) as u64;
        self.g.launch(self.k["vx_alltoall"], (VX_BLOCKS as u32, 1, 1), 256, 0, &[Arg::P(w.vx_pb.ptr), Arg::I(self.gi as i32), Arg::P(slot), Arg::P(w.ypart16.ptr), Arg::P(slot), Arg::P(w.vx_bytes.ptr + 32), Arg::P(w.seqd.ptr), Arg::I(li as i32)])
    }

    /// tell every peer that phase `ph` of the current layer has landed (after this stream's copies)
    fn publish(&self, ph: usize) -> Result<()> {
        for p in &self.peers {
            self.g.write_value(p.flags + ((ph * self.ngpu + self.gi) * 4) as u64, self.seq)?;
        }
        Ok(())
    }

    /// this stream waits until every peer has published phase `ph` of the current layer
    fn await_peers(&self, ph: usize) -> Result<()> {
        for src in 0..self.ngpu {
            if src != self.gi {
                self.g.wait_value(self.w.flags.ptr + ((ph * self.ngpu + src) * 4) as u64, self.seq)?;
            }
        }
        Ok(())
    }

    /// run this GPU's experts over every token of the step that picked them (routing built on the device);
    /// partial sums go home to each token's GPU
    fn moe_experts(&mut self, counts: &[usize], li: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (d, inter, kk) = (c.dim, self.sl, self.layers[li].top_k);
        let w = &self.w;
        let tg = self.ngpu * TMAX;
        let ll = &self.local[li];
        let nl = ll.n_local;
        let pmax = (tg * kk).min(nl * tg); // pair rows: at most every pick of the step
        let (cnt, off, cur) = (w.cnt.ptr, w.cnt.ptr + (nl * 4) as u64, w.cnt.ptr + ((2 * nl + 1) * 4) as u64);
        let total = off + (nl * 4) as u64;
        self.g.memset(cnt, nl * 4)?;
        self.g.memset(w.ypart.ptr, tg * d * 4)?;
        let gr = ((tg * kk) as u32).div_ceil(256);
        let rb = xrow_bytes(c);
        let (selp, wtp, lds) = (w.xg.ptr + (d * 2) as u64, w.xg.ptr + (d * 2 + kk * 4) as u64, rb / 4);
        self.g.launch(self.k["vroute_count"], (gr, 1, 1), 256, 0, &[Arg::P(selp), Arg::I(lds as i32), Arg::P(w.ntok.ptr), Arg::I(TMAX as i32), Arg::I(self.ngpu as i32), Arg::I(kk as i32), Arg::P(ll.lidx.ptr), Arg::P(cnt)])?;
        self.g.launch(self.k["vroute_scan"], (1, 1, 1), 1024, 0, &[Arg::P(cnt), Arg::P(off), Arg::P(cur), Arg::I(nl as i32)])?;
        self.g.launch(self.k["vroute_fill"], (gr, 1, 1), 256, 0, &[Arg::P(selp), Arg::P(wtp), Arg::I(lds as i32), Arg::P(w.ntok.ptr), Arg::I(TMAX as i32), Arg::I(self.ngpu as i32), Arg::I(kk as i32), Arg::P(ll.lidx.ptr), Arg::P(off), Arg::P(cur), Arg::P(w.tok.ptr), Arg::P(w.wt.ptr)])?;
        let prow = (pmax as u32).min(64); // grid rows: the kernels loop over the actual pair count
        self.g.launch(self.k["vzero_c"], (((2 * inter) as u32).div_ceil(256), prow, 1), 256, 0, &[Arg::P(w.gu.ptr), Arg::I((2 * inter) as i32), Arg::P(total)])?;
        // the side stream runs the shared-expert slice and the > 8-token lists alongside the main stream's
        // <= 8-token lists (graph branches)
        let (ov, ev) = (self.overlap, self.evs);
        if ov {
            self.g.record(ev[0], false)?;
            self.g.wait(ev[0], true)?;
        }
        if let Some((sgu, sdn)) = &self.layers[li].sh_slice {
            // this GPU's slice of the shared expert over every exchange row (rows past a GPU's count are unused)
            let sl = sdn.cols;
            self.lx(ov, self.k["vzero_rows"], (((2 * sl) as u32).div_ceil(256), tg as u32, 1), 256, 0, &[Arg::P(w.shgu.ptr), Arg::I((2 * sl) as i32), Arg::I((2 * sl) as i32)])?;
            self.lx(ov, self.k["vsrc_rows"], (1, 1, 1), 32, 0, &[Arg::P(w.ntok.ptr), Arg::I(self.ngpu as i32), Arg::I(TMAX as i32), Arg::P(w.srcl.ptr)])?;
            self.vrows_src(sgu, w.xg.ptr, rb / 2, w.shgu.ptr, 2 * sl, ov)?;
            self.lx(ov, self.k["vswiglu"], (((tg * sl) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.shgu.ptr), Arg::P(0), Arg::P(w.shh.ptr), Arg::I(tg as i32), Arg::I(sl as i32), Arg::F(c.swiglu_limit)])?;
            self.vrows_src(sdn, w.shh.ptr, sl, w.ypart.ptr, d, ov)?;
        }
        // gate/up: X = xg rows (tokens of every GPU), Y = gu rows (pairs)
        self.vmoe_lists(li, true, w.tok.ptr, 0, w.xg.ptr, w.gu.ptr, 2 * inter, d, tg, off, rb / 2, false)?;
        self.vmoe_lists(li, true, w.tok.ptr, 0, w.xg.ptr, w.gu.ptr, 2 * inter, d, tg, off, rb / 2, true)?;
        if ov {
            self.g.record(ev[1], true)?;
            self.g.wait(ev[1], false)?;
        }
        self.g.launch(self.k["vswiglu_c"], (4 * self.g.sms as u32, 1, 1), 256, 0, &[Arg::P(w.gu.ptr), Arg::P(w.wt.ptr), Arg::P(w.hm.ptr), Arg::P(total), Arg::I(pmax as i32), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
        if c.expert_rot.get(li).copied().unwrap_or(false) {
            self.g.launch(self.k["vfwht_c"], ((inter / 128) as u32, prow, 1), 128, 0, &[Arg::P(w.hm.ptr), Arg::I(inter as i32), Arg::P(total)])?;
        }
        // down: X = hm rows (pairs), Y = ypart rows (tokens)
        if ov {
            self.g.record(ev[2], false)?;
            self.g.wait(ev[2], true)?;
        }
        self.vmoe_lists(li, false, 0, w.tok.ptr, w.hm.ptr, w.ypart.ptr, d, inter, tg, off, inter, false)?;
        self.vmoe_lists(li, false, 0, w.tok.ptr, w.hm.ptr, w.ypart.ptr, d, inter, tg, off, inter, true)?;
        if ov {
            self.g.record(ev[3], true)?;
            self.g.wait(ev[3], false)?;
        }
        let _ = counts;
        Ok(())
    }

    /// y[r][n] += W x[r] over the exchange rows that hold tokens: one work entry per source GPU (its count from ntok,
    /// rows from src * TMAX), launched per count bucket like the expert lists; one row group per warp (any N % 32)
    fn vrows_src(&self, v: &VMat, x: u64, ldx: usize, y: u64, ldy: usize, side: bool) -> Result<()> {
        // every row with a token in one list (vsrc_rows, built once per layer), in 32-row tiles: the slice's
        // weights are read once per tile instead of once per source GPU
        let (n, k) = (v.rows, v.cols);
        let rtiles = n.div_ceil(256);
        let ttiles = (self.ngpu * TMAX).div_ceil(32);
        let kc = if k <= 1152 { k } else { 256 };
        let kern = self.k["vmoe_b4_n4"];
        let smem = 4 * 8 * (kc + 8) * 2;
        if smem > 48 * 1024 {
            self.g.set_smem(kern, smem)?;
        }
        let l = self.w.srcl.ptr;
        self.lx(
            side,
            kern,
            ((rtiles * ttiles) as u32, 1, (k as u32).div_ceil(kc as u32)),
            256,
            smem as u32,
            &[Arg::P(v.w.ptr), Arg::P(v.s.ptr), Arg::P(self.w.srcwork.ptr), Arg::P(l + 8), Arg::P(l + 8), Arg::P(0), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(kc as i32), Arg::I(ldy as i32), Arg::P(l), Arg::P(l + 4), Arg::I(0), Arg::I(1 << 30), Arg::I(rtiles as i32), Arg::I(ldx as i32), Arg::P(0)],
        )
    }

    /// y[r][n] += W x[r] for every exchange row r < rows (fp16 x, row stride ldx halves), in 32-row tiles
    fn vrows32(&self, v: &VMat, x: u64, ldx: usize, y: u64, ldy: usize, rows: usize) -> Result<()> {
        let (n, k) = (v.rows, v.cols);
        let kc = 1024usize.min(k);
        let kern = self.k["vmoe_b4_n4"];
        let smem = 4 * 8 * (kc + 8) * 2;
        if smem > 48 * 1024 {
            self.g.set_smem(kern, smem)?;
        }
        ensure!(rows % 32 == 0 && rows / 32 <= 64, "vrows32 over {rows} rows");
        self.g.launch(
            kern,
            ((n as u32).div_ceil(256), (rows / 32) as u32, (k as u32).div_ceil(kc as u32)),
            256,
            smem as u32,
            &[Arg::P(v.w.ptr), Arg::P(v.s.ptr), Arg::P(self.w.tile32.ptr), Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(kc as i32), Arg::I(ldy as i32), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::I(ldx as i32), Arg::P(0)],
        )
    }

    /// the layer's static lists, three launches each: lists with <= 8, 9..16 and more tokens (tiles of 32)
    #[allow(clippy::too_many_arguments)]
    fn vmoe_lists(&self, li: usize, gu: bool, in_rows: u64, out_rows: u64, x: u64, y: u64, n: usize, k: usize, cap: usize, off: u64, ldx: usize, side: bool) -> Result<()> {
        let kc = if k % 1024 == 0 { 1024 } else if k % 1152 == 0 { 1152 } else { k };
        ensure!(kc <= 1152 && kc % 32 == 0, "expert K {k}");
        let ll = &self.local[li];
        for (lg, kind, work, cnt) in &ll.lists {
            if *lg != gu {
                continue;
            }
            for (nb, lo, hi) in [(1usize, 0usize, 8usize), (2, 8, 16), (4, 16, cap)] {
                // with overlap the <= 8-token bucket runs on the main stream and the others on the side stream
                if self.overlap && (side != (nb > 1)) || !self.overlap && side {
                    continue;
                }
                let r = if *kind >= K_FP4 { 1 } else { match nb { 1 => 4, 2 => 2, _ => 1 } };
                let rtiles = n.div_ceil(8 * 32 * r);
                let ttiles = if nb == 4 { cap.div_ceil(32) } else { 1 };
                // lists over 16 tokens are rare: compact them first so the big-tile kernel (one block per SM) gets no idle blocks
                let (wptr, ny, nwork) = if nb == 4 {
                    let stride = if *kind >= K_FP4 { 8 } else { 4 };
                    self.lx(side, self.k["vcompact"], (1, 1, 1), 256, 0, &[Arg::P(work.ptr), Arg::I(*cnt as i32), Arg::I(stride), Arg::P(self.w.cnt.ptr), Arg::I(lo as i32), Arg::P(self.w.hot.ptr), Arg::P(self.w.nhot.ptr)])?;
                    (self.w.hot.ptr, (*cnt).min(8), self.w.nhot.ptr)
                } else {
                    (work.ptr, *cnt, 0)
                };
                let grid = ((rtiles * ttiles) as u32, ny as u32, (k as u32).div_ceil(kc as u32));
                let common = [Arg::P(in_rows), Arg::P(out_rows), Arg::P(0), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(kc as i32), Arg::I(n as i32), Arg::P(self.w.cnt.ptr), Arg::P(off), Arg::I(lo as i32), Arg::I(hi as i32), Arg::I(rtiles as i32), Arg::I(ldx as i32), Arg::P(nwork)];
                if *kind >= K_FP4 {
                    let (name, ntab, tab) = match *kind { K_FP4 => ("fp4", 256usize, 0u64), K_CB3 => ("cb3", 64, 1024), _ => ("cb2", 16, 1280) };
                    let kern = self.k[format!("vlut_{name}{}_n{nb}", if nb < 4 { "r1" } else { "" }).as_str()];
                    let smem = ntab * 4 + 512 + nb * 8 * (kc + 8) * 2;
                    if smem > 48 * 1024 {
                        self.g.set_smem(kern, smem)?;
                    }
                    let mut args = vec![Arg::P(ll.wbase), Arg::P(ll.sbase), Arg::P(wptr), Arg::P(self.luts.ptr + tab), Arg::P(self.luts.ptr + 1344)];
                    args.extend(common);
                    self.lx(side, kern, grid, 256, smem as u32, &args)?;
                } else {
                    let kern = self.k[match (*kind, nb) { (2, 1) => "vmoe_b2_n1", (2, 2) => "vmoe_b2_n2", (2, _) => "vmoe_b2_n4", (_, 1) => "vmoe_b4_n1", (_, 2) => "vmoe_b4_n2", _ => "vmoe_b4_n4" }];
                    let smem = nb * 8 * (kc + 8) * 2;
                    if smem > 48 * 1024 {
                        self.g.set_smem(kern, smem)?;
                    }
                    let mut args = vec![Arg::P(ll.wbase), Arg::P(ll.sbase), Arg::P(wptr)];
                    args.extend(common);
                    self.lx(side, kern, grid, 256, smem as u32, &args)?;
                }
            }
        }
        Ok(())
    }

    /// attention partials over each token's rows (window rows by win_idx, then compressed rows by comp_idx), 160 rows
    /// per split: tensor cores (vattn_mma) unless VOLTA_ATTN=tile5
    #[allow(clippy::too_many_arguments)]
    fn attn_tiles(&self, t: usize, win: u64, win_idx: u64, nw: usize, comp: u64, comp_idx: u64, nk: usize, s: usize, scale: f32) -> Result<()> {
        let c = &self.m.cfg;
        let (hd, nh, w) = (c.head_dim, c.n_heads, &self.w);
        if std::env::var("VOLTA_ATTN").is_ok_and(|v| v == "tile5") {
            return self.g.launch(self.k["vattn_tile5"], (t as u32, (nh / 16) as u32, s as u32), 256, 0, &[Arg::P(w.q.ptr), Arg::P(win), Arg::P(win_idx), Arg::I(nw as i32), Arg::P(comp), Arg::P(comp_idx), Arg::I(nk as i32), Arg::P(w.attn_part.ptr), Arg::I(nh as i32), Arg::I(hd as i32), Arg::F(scale)]);
        }
        ensure!(hd == 512 && nh % 16 == 0, "vattn_mma: 512-dim heads in groups of 16");
        let n = t * nh * hd;
        self.g.launch(self.k["vq16"], ((n as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.q.ptr), Arg::P(w.q16.ptr), Arg::F(scale), Arg::I(n as i32)])?;
        // VAM_SMEM in volta.cuh
        let smem = 32 * 520 * 2 + 4 * 32 * 16 * 4 + 16 * 40 * 2 + 4 * 16 * 4 + 32 * 4 + 16;
        self.g.set_smem(self.k["vattn_mma"], smem)?;
        self.g.launch(self.k["vattn_mma"], ((nh / 16) as u32, t as u32, s as u32), 256, smem as u32, &[Arg::P(w.q16.ptr), Arg::P(win), Arg::P(win_idx), Arg::I(nw as i32), Arg::P(comp), Arg::P(comp_idx), Arg::I(nk as i32), Arg::P(w.attn_part.ptr), Arg::I(nh as i32), Arg::I(hd as i32)])
    }

    /// launch on the side stream (side) or the main one
    fn lx(&self, side: bool, k: Kernel, grid: (u32, u32, u32), block: u32, smem: u32, args: &[Arg]) -> Result<()> {
        if side { self.g.launch_side(k, grid, block, smem, args) } else { self.g.launch(k, grid, block, smem, args) }
    }

    /// a = bf16(shared + sum of every GPU's partials) for this GPU's tokens
    fn moe_combine(&mut self, t: usize) -> Result<()> {
        let d = self.m.cfg.dim;
        let w = &self.w;
        self.g.launch(self.k["vsum_parts_h"], (((t * d) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.a.ptr), Arg::P(w.parts.ptr), Arg::I(self.ngpu as i32), Arg::I((TMAX * d) as i32), Arg::I((t * d) as i32)])
    }

    fn finish(&mut self, toks: &[(usize, u32)]) -> Result<Vec<f32>> {
        let c = &self.m.cfg;
        let (d, hc) = (c.dim, c.hc);
        let t = toks.len();
        let w = &self.w;
        self.g.launch(self.k["vhc_pre"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.mix_f.ptr), Arg::I(0), Arg::P(w.x.ptr), Arg::I(d as i32)])?;
        self.rms_rows(w.x.ptr, &self.out_norm, w.x.ptr, d, t, false)?;
        self.mm(&self.head, w.x.ptr, w.logits.ptr, t)?;
        for &(s, _) in toks {
            self.sess[s].pos += 1;
        }
        let _ = hc;
        if self.greedy {
            // greedy ids only (as f32 bit patterns on the result channel), not t x vocab logits
            self.g.launch(self.k["vargmax_rows"], (t as u32, 1, 1), 1024, 0, &[Arg::P(w.logits.ptr), Arg::I(c.vocab as i32), Arg::P(self.spark_ids.ptr)])?;
            let ids: Vec<u32> = self.g.download(self.spark_ids.ptr, t)?;
            return Ok(ids.into_iter().map(f32::from_bits).collect());
        }
        self.g.download(w.logits.ptr, t * c.vocab)
    }
}

/// one layer's per-token tables for the current step (device pointers into Work::tabs; 0 = not used)
#[derive(Default)]
struct LTab {
    pos: u64,
    wrow: u64,
    srow: u64,
    lat_base: u64,
    lat_pos: u64,
    comp_rows: u64,
    idx_rows: u64,
    ends: u64,
    koff: u64,
    coff: u64,
    n_max: usize,
    /// compressed rows attention reads (idx_topk once an indexer layer has run, else 0)
    kcomp: usize,
    comp_src: usize,
    idxk_src: usize,
}

/// per-step state shared across the layers of one step (same for every token: depends on the layer only)
#[derive(Default)]
struct StepState {
    comp_src: usize,
    idxk_src: usize,
    topk_k: usize,
    keep_ld: usize,
}

// ---------------------------------------------------------------- the engine: one worker thread per GPU

enum Cmd {
    Step(Arc<Vec<Vec<(usize, u32)>>>),
    Reset(usize),
    /// a main step that returns each token's greedy next id
    StepGreedy(Arc<Vec<Vec<(usize, u32)>>>),
    /// a draft step: per GPU (observed rows (slot, row, pos), drafts (slot, pos, next))
    Spark(Arc<SparkPlan>),
    /// drop slots' last tokens: (slot, n) per GPU
    Rewind(Arc<Vec<Vec<(usize, usize)>>>),
    Stop,
}

/// a draft step's work, per GPU: main tokens to observe (slot, row of the last main step, position) and sessions
/// to draft for (slot, position of the newest main token, next token)
pub struct SparkPlan {
    pub obs: Vec<Vec<(usize, usize, usize)>>,
    pub drafts: Vec<Vec<(usize, usize, u32)>>,
}

pub struct Engine {
    spark_block: Option<usize>,
    tx: Vec<mpsc::Sender<Cmd>>,
    rx: mpsc::Receiver<(usize, Result<Vec<f32>>)>,
    pub opts: Opts,
    pub vocab: usize,
    joins: Vec<std::thread::JoinHandle<()>>,
}

impl Engine {
    /// load the dense half onto every GPU and each GPU's experts (in parallel), connect the peers
    pub fn new(m: &'static Model, o: Opts) -> Result<Engine> {
        // Engram rows are hashed: every step touches ~800 random pages of the 190 GB tables. Map them all up front
        // (MADV_POPULATE_READ) so a lookup is a memory read, not a page fault; runs alongside the GPU loads
        let populate = std::thread::spawn(move || populate_engram(m));
        let n = o.ngpu;
        let (rtx, rx) = mpsc::channel();
        let barrier = Arc::new(Barrier::new(n));
        let peers: Arc<Mutex<Vec<Option<PeerBufs>>>> = Arc::new(Mutex::new(vec![None; n]));
        let gpus_ready = Arc::new(Barrier::new(n));
        // MXFP4 source (the HF checkpoint): tiers under VOLTA_EXPERT_GB per GPU (default 25), shrunk until the
        // placement fits every GPU
        let fp4 = fp4_source(m);
        let pack = Pack::open();
        if let Some(p) = &pack {
            ensure!(p.format == FORMAT && p.ngpu == o.ngpu, "pack: format {} for {} GPUs, this engine is format {FORMAT} on {}", p.format, p.ngpu, o.ngpu);
        } else if Pack::dir().is_some() {
            ensure!(m.layers[0].experts.len() == m.cfg.n_experts, "VOLTA_PACK has no manifest.json, and making one needs the checkpoint's experts");
        }
        let per_gpu = std::env::var("VOLTA_EXPERT_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(25.0) * 1e9;
        // experts are placed over groups of etp GPUs; a GPU holds 1/etp of its group's bytes
        let etp = expert_tp(m, n);
        let ng = n / etp;
        let (mut tiers, mut owner_v) = (None, if pack.is_some() { vec![] } else { assign_experts(m, ng, None) });
        if let Some(p) = &pack {
            (tiers, owner_v) = (Some(p.tiers.clone()), p.owner.clone());
            eprintln!("experts from the pack: split {etp} ways, {} GB per GPU", p.expert_gb);
        } else if fp4 {
            let mut budget = per_gpu * n as f64;
            for _ in 0..6 {
                let t = plan_tiers(m, budget);
                let own = assign_experts(m, ng, Some(&t));
                let mut per = vec![0f64; ng];
                for l in 0..m.cfg.n_layers {
                    for e in 0..m.cfg.n_experts {
                        per[own[l][e] as usize] += expert_bytes(m, l, e, Some(&t)) as f64 / etp as f64;
                    }
                }
                let worst = per.iter().cloned().fold(0f64, f64::max);
                tiers = Some(t);
                owner_v = own;
                if worst <= per_gpu {
                    break;
                }
                budget -= (worst - per_gpu) * n as f64 + 0.05e9 * n as f64;
            }
            eprintln!("experts split {etp} ways over {ng} group(s)");
            if let (Some(dir), Some(t)) = (Pack::dir(), &tiers) {
                std::fs::create_dir_all(&dir)?;
                let p = Pack { format: FORMAT, ngpu: n, etp, expert_gb: per_gpu / 1e9, tiers: t.clone(), owner: owner_v.clone() };
                std::fs::write(dir.join("manifest.json"), serde_json::to_vec(&p)?)?;
            }
        }
        let owner = Arc::new(owner_v);
        let tiers = tiers.map(Arc::new);
        // the DSpark draft from the checkpoint directory in VOLTA_SPARK (its MXFP4 experts are packed at load)
        let ds: Option<&'static crate::dspark::Dspark> = match std::env::var_os("VOLTA_SPARK") {
            Some(dir) => {
                let crate::model::Source::Hf(st) = &m.src else { bail!("VOLTA_SPARK needs the HF checkpoint") };
                let ds = crate::dspark::Dspark::load(m, st, std::path::Path::new(&dir))?;
                eprintln!("draft: {} layers, block {}", ds.layers.len(), ds.block);
                Some(Box::leak(Box::new(ds)))
            }
            None => None,
        };
        let mut tx = vec![];
        let mut joins = vec![];
        for gi in 0..n {
            let (ctx, crx) = mpsc::channel::<Cmd>();
            tx.push(ctx);
            let (rtx, barrier, peers, ready, o, owner, tiers) = (rtx.clone(), barrier.clone(), peers.clone(), gpus_ready.clone(), o.clone(), owner.clone(), tiers.clone());
            joins.push(std::thread::spawn(move || {
                let r = (|| -> Result<()> {
                    let t0 = std::time::Instant::now();
                    let mut sh = Shard::new(m, gi, &o, owner, tiers, ds)?;
                    let (free, total) = sh.g.mem_info()?;
                    eprintln!("GPU{gi}: loaded in {:.0} s, {} experts, {:.1} of {:.1} GB free", t0.elapsed().as_secs_f64(), sh.exps.tab.len(), free as f64 / 1e9, total as f64 / 1e9);
                    peers.lock().unwrap()[gi] = Some(sh.peer_bufs());
                    ready.wait();
                    // peer access to every other GPU's exchange buffers
                    let all: Vec<PeerBufs> = peers.lock().unwrap().iter().map(|p| p.unwrap()).collect();
                    for (j, p) in all.iter().enumerate() {
                        if j != gi {
                            sh.peers.push(*p);
                        }
                    }
                    peer_access(&sh.g, gi, n)?;
                    ensure!(n == 8, "the cube collectives assume the DGX-1 8-GPU mesh");
                    let ta: Vec<u64> = all.iter().map(|p| p.xg).chain(all.iter().map(|p| p.relay)).chain(all.iter().map(|p| p.vx_fa)).collect();
                    let tb: Vec<u64> = all.iter().map(|p| p.parts).chain(all.iter().map(|p| p.relay)).chain(all.iter().map(|p| p.vx_fb)).collect();
                    sh.g.htod_sync(sh.w.vx_pa.ptr, &ta)?;
                    sh.g.htod_sync(sh.w.vx_pb.ptr, &tb)?;
                    ready.wait();
                    rtx.send((gi, Ok(vec![]))).ok();
                    while let Ok(cmd) = crx.recv() {
                        match cmd {
                            Cmd::Stop => break,
                            Cmd::Reset(s) => {
                                let nl = m.cfg.n_layers;
                                sh.sess[s] = Sess { n_comp: vec![0; nl], n_idx: vec![0; nl], ..Default::default() };
                                rtx.send((gi, Ok(vec![]))).ok();
                            }
                            Cmd::Step(plan) => {
                                let r = run_step(&mut sh, &plan, &barrier);
                                rtx.send((gi, r)).ok();
                            }
                            Cmd::Spark(p) => {
                                // draft tokens travel as f32 bit patterns on the shared result channel
                                let counts: Vec<usize> = p.drafts.iter().map(|d| d.len() * sh.spark.as_ref().map_or(0, |s| s.block)).collect();
                                let r = sh.spark_step(&p.obs[gi], &p.drafts[gi], &counts).map(|v| v.into_iter().map(f32::from_bits).collect());
                                rtx.send((gi, r)).ok();
                            }
                            Cmd::Rewind(v) => {
                                let r = v[gi].iter().try_for_each(|&(s, n)| sh.rewind(s, n)).map(|_| vec![]);
                                rtx.send((gi, r)).ok();
                            }
                            Cmd::StepGreedy(plan) => {
                                sh.greedy = true;
                                let r = run_step(&mut sh, &plan, &barrier);
                                sh.greedy = false;
                                rtx.send((gi, r)).ok();
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = r {
                    eprintln!("GPU{gi}: {e:#}");
                    rtx.send((gi, Err(e))).ok();
                }
            }));
        }
        let e = Engine { spark_block: ds.map(|d| d.block), tx, rx, opts: o, vocab: m.cfg.vocab, joins };
        let _ = populate.join();
        for _ in 0..n {
            let (gi, r) = e.rx.recv()?;
            if let Err(err) = r {
                // the other workers wait at the setup barrier for this one: joining them would hang
                eprintln!("GPU{gi} setup failed: {err:#}");
                std::process::exit(1);
            }
        }
        Ok(e)
    }

    /// One decode step: plan[g] = (slot, token) pairs on GPU g (each slot at most once). Returns logits per GPU,
    /// [tokens][vocab] flattened.
    pub fn step(&self, plan: Vec<Vec<(usize, u32)>>) -> Result<Vec<Vec<f32>>> {
        ensure!(plan.len() == self.opts.ngpu && plan.iter().all(|p| p.len() <= TMAX), "plan: one list per GPU, <= {TMAX} tokens each");
        let plan = Arc::new(plan);
        for t in &self.tx {
            t.send(Cmd::Step(plan.clone()))?;
        }
        let mut out = vec![vec![]; self.opts.ngpu];
        for _ in 0..self.opts.ngpu {
            let (gi, r) = self.rx.recv()?;
            if let Err(e) = &r {
                // the other GPUs wait on this one in the exchange and never answer: report before failing
                eprintln!("GPU{gi} step failed: {e:#}");
                std::process::exit(1);
            }
            out[gi] = r.with_context(|| format!("GPU{gi} step"))?;
        }
        Ok(out)
    }

    /// one draft step on every GPU; returns each GPU's drafts, `block` per entry of `p.drafts[gpu]`
    pub fn spark(&self, p: SparkPlan) -> Result<Vec<Vec<u32>>> {
        ensure!(p.obs.len() == self.opts.ngpu && p.drafts.len() == self.opts.ngpu, "spark plan: one list per GPU");
        let p = Arc::new(p);
        for t in &self.tx {
            t.send(Cmd::Spark(p.clone()))?;
        }
        let mut out = vec![vec![]; self.opts.ngpu];
        for _ in 0..self.opts.ngpu {
            let (gi, r) = self.rx.recv()?;
            out[gi] = r.with_context(|| format!("GPU{gi} draft step"))?.into_iter().map(f32::to_bits).collect();
        }
        Ok(out)
    }

    /// drop slots' last tokens (rejected drafts): (slot, n) per GPU
    pub fn rewind(&self, v: Vec<Vec<(usize, usize)>>) -> Result<()> {
        let v = Arc::new(v);
        for t in &self.tx {
            t.send(Cmd::Rewind(v.clone()))?;
        }
        for _ in 0..self.opts.ngpu {
            let (gi, r) = self.rx.recv()?;
            r.with_context(|| format!("GPU{gi} rewind"))?;
        }
        Ok(())
    }

    /// a step that returns every token's greedy next id instead of its logits
    pub fn step_greedy(&self, plan: Vec<Vec<(usize, u32)>>) -> Result<Vec<Vec<u32>>> {
        ensure!(plan.len() == self.opts.ngpu && plan.iter().all(|p| p.len() <= TMAX), "plan: one list per GPU, <= {TMAX} tokens each");
        let plan = Arc::new(plan);
        for t in &self.tx {
            t.send(Cmd::StepGreedy(plan.clone()))?;
        }
        let mut out = vec![vec![]; self.opts.ngpu];
        for _ in 0..self.opts.ngpu {
            let (gi, r) = self.rx.recv()?;
            if let Err(e) = &r {
                // the other GPUs wait on this one in the exchange and never answer: report before failing
                eprintln!("GPU{gi} step failed: {e:#}");
                std::process::exit(1);
            }
            out[gi] = r.with_context(|| format!("GPU{gi} step"))?.into_iter().map(f32::to_bits).collect();
        }
        Ok(out)
    }

    /// the draft block size, when a draft is loaded
    pub fn spark_block(&self) -> Option<usize> {
        self.spark_block
    }

    pub fn reset(&self, gpu: usize, slot: usize) -> Result<()> {
        self.tx[gpu].send(Cmd::Reset(slot))?;
        let (gi, r) = self.rx.recv()?;
        r.with_context(|| format!("GPU{gi} reset"))?;
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        for t in &self.tx {
            t.send(Cmd::Stop).ok();
        }
        for j in self.joins.drain(..) {
            j.join().ok();
        }
    }
}

/// enable peer access from GPU gi's context to every other device (contexts are per thread; open temporary ones)
fn peer_access(g: &Gpu, gi: usize, n: usize) -> Result<()> {
    use cudarc::driver::sys as cu;
    g.bind();
    for j in 0..n {
        if j == gi {
            continue;
        }
        let mut dev = 0;
        let mut ctx = std::ptr::null_mut();
        unsafe {
            crate::gpu::check(cu::cuDeviceGet(&mut dev, j as i32), "cuDeviceGet")?;
            crate::gpu::check(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        }
        g.bind();
        let r = unsafe { cu::cuCtxEnablePeerAccess(ctx, 0) };
        if r != cu::CUresult::CUDA_SUCCESS && r != cu::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            crate::gpu::check(r, "cuCtxEnablePeerAccess")?;
        }
    }
    Ok(())
}

fn run_step(sh: &mut Shard, plan: &[Vec<(usize, u32)>], _barrier: &Barrier) -> Result<Vec<f32>> {
    let c = &sh.m.cfg;
    let toks = &plan[sh.gi];
    let counts: Vec<usize> = plan.iter().map(|p| p.len()).collect();
    let t = toks.len();
    sh.g.bind();
    let nt32: Vec<i32> = counts.iter().map(|&n| n as i32).collect();
    sh.g.htod_sync(sh.w.ntok.ptr, &nt32)?;
    let rb = xrow_bytes(c) as i32;
    let vb: Vec<i32> = counts.iter().map(|&n| n as i32 * rb).chain(counts.iter().map(|&n| (n * c.dim * 2) as i32)).collect();
    sh.g.htod_sync(sh.w.vx_bytes.ptr, &vb)?;
    // exchange sequence base for this step (layer li uses base + li); flags start at 0, so base starts at 1
    let base = sh.seq + 1;
    sh.seq += c.n_layers as u32;
    sh.g.htod_sync(sh.w.seqd.ptr, &[base])?;
    // context bucket: power of two >= the longest of this GPU's sessions after this token, at least 64
    let longest = sh.tok_pos(toks).iter().map(|&p| p + 1).max().unwrap_or(1);
    sh.bucket = longest.next_power_of_two().max(64).min(sh.cap[0]);
    if t > 0 {
        sh.begin(toks)?;
    }
    let prof = sh.gi == 0 && std::env::var_os("VOLTA_PROF").is_some();
    let eager = prof || std::env::var_os("DSX_DUMP").is_some() || std::env::var("VOLTA_GRAPH").is_ok_and(|v| v == "0");
    let key = (t, sh.bucket);
    // graphs cost device memory: past VOLTA_GRAPHS shapes (default 6), new shapes run without one
    let limit = std::env::var("VOLTA_GRAPHS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(6);
    let eager = eager || (!sh.graphs.contains_key(&key) && sh.graphs.values().filter(|g| g.is_some()).count() >= limit);
    if eager {
        layers(sh, toks, &counts, prof)?;
    } else if let Some(Some(g)) = sh.graphs.get(&key).map(|g| g.as_ref()) {
        sh.g.launch_graph(g)?;
    } else if sh.graphs.contains_key(&key) {
        // second time this shape comes up: capture it (the first run warmed cuBLAS and the kernel attributes)
        sh.g.begin_capture()?;
        layers(sh, toks, &counts, false)?;
        let g = sh.g.end_capture()?;
        sh.g.launch_graph(&g)?;
        sh.graphs.insert(key, Some(g));
    } else {
        layers(sh, toks, &counts, false)?;
        sh.graphs.insert(key, None);
    }
    if t == 0 {
        return Ok(vec![]);
    }
    sh.finish(toks)
}

/// enqueue every layer of the step (captured into a graph unless `prof`)
fn layers(sh: &mut Shard, toks: &[(usize, u32)], counts: &[usize], prof: bool) -> Result<()> {
    let c = &sh.m.cfg;
    let t = toks.len();
    let (d, hc) = (c.dim, c.hc);
    let mut st = StepState { comp_src: usize::MAX, idxk_src: usize::MAX, ..Default::default() };
    // VOLTA_PROF: GPU0 syncs at each boundary and prints where the step went
    let mut pt = [0f64; 8];
    let mut tick = std::time::Instant::now();
    let mut mark = |sh: &Shard, i: usize| -> Result<()> {
        if prof {
            sh.g.sync()?;
            pt[i] += tick.elapsed().as_secs_f64();
            tick = std::time::Instant::now();
        }
        Ok(())
    };
    for li in 0..c.n_layers {
        if t > 0 {
            if let Some(eidx) = c.engram_layers.iter().position(|&x| x == li) {
                sh.engram(toks, li, eidx)?;
            }
            sh.spark_capture(li, t)?;
            mark(sh, 0)?;
            let w = &sh.w;
            let l = &sh.layers[li];
            sh.g.launch(sh.k["vhc_pre"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.mix_f.ptr), Arg::I(0), Arg::P(w.x.ptr), Arg::I(d as i32)])?;
            sh.hc_mixes_rows(&l.hc_attn, w.h.ptr, w.mix_a.ptr, t)?;
            sh.rms_rows(w.x.ptr, &l.attn_norm, w.x.ptr, d, t, true)?;
            sh.attention(toks, li, &mut st)?;
            mark(sh, 1)?;
            let w = &sh.w;
            sh.g.launch(sh.k["vhc_post"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.a.ptr), Arg::P(w.mix_a.ptr), Arg::I(d as i32)])?;
            if sh.gi == 0 && std::env::var_os("DSX_DUMP").is_some() {
                let h: Vec<f32> = sh.g.download(w.h.ptr, hc * d)?;
                crate::model::dump("vgpuA", sh.sess[toks[0].0].pos, li, &h);
            }
            let w = &sh.w;
            let l = &sh.layers[li];
            sh.g.launch(sh.k["vhc_pre"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.mix_a.ptr), Arg::I(0), Arg::P(w.x.ptr), Arg::I(d as i32)])?;
            sh.hc_mixes_rows(&l.hc_ffn, w.h.ptr, w.mix_f.ptr, t)?;
            sh.rms_rows(w.x.ptr, &l.ffn_norm, w.x.ptr, d, t, true)?;
            sh.moe_dispatch(t, li)?;
            mark(sh, 2)?;
        }
        // every GPU's x / picks land everywhere, then every partial lands at its home GPU (cube collectives)
        sh.vx_gather(li)?;
        mark(sh, 3)?;
        sh.moe_experts(counts, li)?;
        mark(sh, 4)?;
        sh.vx_scatter(li)?;
        mark(sh, 5)?;
        if t > 0 {
            sh.moe_combine(t)?;
            let w = &sh.w;
            sh.g.launch(sh.k["vhc_post"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.a.ptr), Arg::P(w.mix_f.ptr), Arg::I(d as i32)])?;
            if sh.gi == 0 && std::env::var_os("DSX_DUMP").is_some() {
                // the residual stream of GPU0's first token, in the CPU reference's dump format (pos already advanced in finish)
                let h: Vec<f32> = sh.g.download(sh.w.h.ptr, hc * d)?;
                crate::model::dump("vgpu", sh.sess[toks[0].0].pos, li, &h);
            }
        }
    }
    if prof {
        let names = ["hc+engram+norms", "attention", "router+shared+push", "wait x", "experts", "wait partials"];
        let tot: f64 = pt.iter().sum();
        let parts: Vec<String> = names.iter().zip(pt.iter()).map(|(n, v)| format!("{n} {:.1}", v * 1e3)).collect();
        eprintln!("GPU0 layers {:.1} ms: {}", tot * 1e3, parts.join(", "));
    }
    Ok(())
}

/// Self-test of the expert path for one (layer, expert): code_mat + pack + vmoe (gate/up, swiglu, down) on GPU
/// `gpu`, against the CPU reference matvecs of the original GGUF matrices. Returns (rel err of gate/up, of output).
pub fn expert_selftest(m: &Model, gpu: i32, li: usize, e: usize) -> Result<(f64, f64)> {
    use crate::model::matvec;
    let c = &m.cfg;
    let (d, inter) = (c.dim, c.inter);
    let g = Gpu::new(gpu)?;
    let ex = &m.layers[li].experts[e];
    let (cg, cu, cd) = (code_mat(&ex.gate), code_mat(&ex.up), code_mat(&ex.down));
    let (cg, cu) = if cg.bits != cu.bits { (widen(cg), widen(cu)) } else { (cg, cu) };
    let mut gu = Coded { bits: cg.bits, codes: cg.codes, scales: cg.scales };
    gu.codes.extend_from_slice(&cu.codes);
    gu.scales.extend_from_slice(&cu.scales);
    let (mut w, mut s) = (vec![], vec![]);
    pack(&gu, 2 * inter, d, &mut w, &mut s);
    let (gw, gs) = (w.len(), s.len());
    pack(&cd, d, inter, &mut w, &mut s);
    let (wb, sb) = (g.upload(&w)?, g.upload(&s)?);
    // x: a bf16-like activation
    let x: Vec<f32> = (0..d).map(|i| (((i * 7919) % 2001) as f32 - 1000.0) * 1e-3).collect();
    let x16: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
    let xb = g.upload(&x16)?;
    let gub = g.alloc_zeroed(2 * inter * 4)?;
    let hb = g.alloc_zeroed(inter * 2)?;
    let yb = g.alloc_zeroed(d * 4)?;
    let rows = g.upload(&[0i32])?;
    let run = |bits: u32, wo: usize, so: usize, x: u64, y: u64, n: usize, k: usize| -> Result<()> {
        let work: [u32; 4] = [(wo / 4) as u32, so as u32, 0, 1];
        let wk = g.upload(&work)?;
        let kc = if k % 1024 == 0 { 1024 } else { 1152 };
        let kern = g.kernel(if bits == 2 { "vmoe_b2_n1" } else { "vmoe_b4_n1" })?;
        g.launch(kern, ((n as u32).div_ceil(1024), 1, (k as u32).div_ceil(kc as u32)), 256, ((kc + 8) * 16) as u32,
            &[Arg::P(wb.ptr), Arg::P(sb.ptr), Arg::P(wk.ptr), Arg::P(rows.ptr), Arg::P(rows.ptr), Arg::P(0), Arg::P(x), Arg::P(y), Arg::I(n as i32), Arg::I(k as i32), Arg::I(kc as i32), Arg::I(n as i32), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::I(0), Arg::I(k as i32), Arg::P(0)])?;
        g.sync()
    };
    run(gu.bits, 0, 0, xb.ptr, gub.ptr, 2 * inter, d)?;
    let gpu_gu: Vec<f32> = g.download(gub.ptr, 2 * inter)?;
    let (mut rg, mut ru) = (vec![0f32; inter], vec![0f32; inter]);
    let xr: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
    matvec(&ex.gate, &xr, &mut rg);
    matvec(&ex.up, &xr, &mut ru);
    let refgu: Vec<f32> = rg.iter().chain(ru.iter()).cloned().collect();
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        (num / den.max(1e-30)).sqrt()
    };
    let e_gu = rel(&gpu_gu, &refgu);
    let one = g.upload(&[1f32])?;
    g.launch(g.kernel("vswiglu")?, ((inter as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(gub.ptr), Arg::P(one.ptr), Arg::P(hb.ptr), Arg::I(1), Arg::I(inter as i32), Arg::F(c.swiglu_limit)])?;
    run(cd.bits, gw, gs, hb.ptr, yb.ptr, d, inter)?;
    let gpu_y: Vec<f32> = g.download(yb.ptr, d)?;
    let lim = c.swiglu_limit;
    let h: Vec<f32> = rg.iter().zip(&ru).map(|(&a, &b)| crate::numerics::round_bf16(crate::numerics::silu(a.min(lim)) * b.clamp(-lim, lim))).collect();
    let mut ry = vec![0f32; d];
    matvec(&ex.down, &h, &mut ry);
    Ok((e_gu, rel(&gpu_y, &ry)))
}

/// start / stop the CUDA profiler's capture range (nsys --capture-range=cudaProfilerApi); a context must be current,
/// so this binds GPU 0's primary context on the calling thread
pub fn profiler(on: bool) -> Result<()> {
    use cudarc::driver::sys as cu;
    unsafe {
        let mut dev = 0;
        let mut ctx = std::ptr::null_mut();
        crate::gpu::check(cu::cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
        crate::gpu::check(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        crate::gpu::check(cu::cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
        crate::gpu::check(if on { cu::cuProfilerStart() } else { cu::cuProfilerStop() }, "cuProfiler")
    }
}

/// populate the page tables of the Engram tables' mapping (Linux 5.14+ MADV_POPULATE_READ; a no-op elsewhere)
fn populate_engram(m: &Model) {
    let Some(tabs) = m.engram.as_ref() else { return };
    let t0 = std::time::Instant::now();
    let mut bytes = 0usize;
    for e in 0..m.cfg.engram_layers.len() {
        if let Ok(crate::model::EngramLayer::Fp8 { w, s, .. }) = tabs.layer(e, m.cfg.engram_dim) {
            for sl in [w, s] {
                let page = 4096usize;
                let p = sl.as_ptr() as usize;
                let (a, b) = (p / page * page, (p + sl.len()).div_ceil(page) * page);
                const MADV_POPULATE_READ: i32 = 22;
                let r = unsafe { libc::madvise(a as *mut libc::c_void, b - a, MADV_POPULATE_READ) };
                if r != 0 {
                    eprintln!("engram: madvise(POPULATE_READ) failed ({}); lookups will page-fault", std::io::Error::last_os_error());
                    return;
                }
                bytes += b - a;
            }
        }
    }
    eprintln!("engram: {:.0} GB mapped in {:.0} s", bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
}
