//! Prefill attention on tensor cores: launch helper for `sparse_attn_tc` / `sparse_attn_tc_fast` (dsx.cu).
//! `sparse_attn` stays for decode (T = 1) and as the reference. T = 4096 on the 5090: 22.2 -> 4.2 ms (fast: 2.6 ms).
use crate::gpu::{Arg, Gpu, Kernel};
use anyhow::{Result, ensure};

/// rows per smem tile (TC_NT in dsx.cu)
const NT: usize = 32;
/// opt-in dynamic smem limit per block on sm_86 and sm_120
pub const MAX_SMEM: usize = 99 * 1024;

/// dynamic smem for `rows` = W + K candidate rows per token: 2 row tiles, partial scores, P (hi, lo), corr/l, scan, row list
pub fn smem_bytes(rows: usize) -> usize {
    2 * NT * 512 * 2 + 4 * 32 * (NT + 4) * 4 + 2 * 32 * (NT + 8) * 2 + 64 * 4 + 8 * 4 + rows * 4
}

pub struct SparseAttnTc {
    k: Kernel,
    kf: Kernel,
    /// sm_70 (Volta) has no bf16 mma / ldmatrix / cp.async: `sparse_attn` (same arguments) runs instead
    scalar: Option<Kernel>,
    /// use sparse_attn_tc_fast: q and p as single bf16 (2x fewer mma, ~bf16-level error on peaky softmax)
    pub fast: bool,
}

impl SparseAttnTc {
    /// looks the kernels up and raises their dynamic smem limit (once per Gpu)
    pub fn new(g: &Gpu) -> Result<Self> {
        let (k, kf) = (g.kernel("sparse_attn_tc")?, g.kernel("sparse_attn_tc_fast")?);
        if g.arch < 80 {
            return Ok(SparseAttnTc { k, kf, scalar: Some(g.kernel("sparse_attn")?), fast: false });
        }
        g.set_smem(k, MAX_SMEM)?;
        g.set_smem(kf, MAX_SMEM)?;
        Ok(SparseAttnTc { k, kf, scalar: None, fast: false })
    }

    /// o[T][H][512] = sparse MQA attention of q[T][H][512] (f32) over win[win_idx[t][..W]] ++ comp[comp_idx[t][..K]]
    /// (bf16 rows, -1 = skip; comp may be 0 when K = 0), sinks[H]; same arguments as `sparse_attn`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(&self, g: &Gpu, q: u64, win: u64, win_idx: u64, w: usize, comp: u64, comp_idx: u64, k: usize, sinks: u64, o: u64, t: usize, nh: usize, hd: usize) -> Result<()> {
        let args = [Arg::P(q), Arg::P(win), Arg::P(win_idx), Arg::I(w as i32), Arg::P(comp), Arg::P(comp_idx), Arg::I(k as i32), Arg::P(sinks), Arg::P(o), Arg::I(nh as i32), Arg::I(hd as i32), Arg::F(1.0 / (hd as f32).sqrt())];
        if let Some(ks) = self.scalar {
            return g.launch(ks, (t as u32, (nh / 8) as u32, 1), 256, 0, &args);
        }
        ensure!(hd == 512 && nh % 32 == 0, "sparse_attn_tc: hd must be 512 and heads a multiple of 32");
        let smem = smem_bytes(w + k);
        ensure!(smem <= MAX_SMEM, "sparse_attn_tc: {} rows per token do not fit in smem", w + k);
        g.launch(
            if self.fast { self.kf } else { self.k },
            (t as u32, (nh / 32) as u32, 1),
            256,
            smem as u32,
            &args,
        )
    }
}
