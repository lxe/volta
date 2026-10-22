//! Routed experts on the CPU: out[t] += down(silu(min(gate x, L)) * clamp(up x, -L, L) * weight), L = swiglu_limit.
//! The routing weight scales the SwiGLU output before down (model.py Expert.forward); h is re-quantized
//! to 256-blocks for the down projection.

use crate::numerics::fwht_blocks;
use super::pool::{Ctx, Pool};
use super::qdot::{QAct, QActRaw, QK, dot_rows, dot_rows_sel};
use crate::gguf::GType;
use crate::numerics::silu;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};

pub const SWIGLU_LIMIT: f32 = 10.0;
/// tokens per call (verify batch)
pub const MAX_TOKENS: usize = 16;
const RU: usize = 64; // gate/up rows per work unit (4 units per 256-block of h)
const RD: usize = 64; // down rows per work unit

/// One expert's weights: gate/up [inter][dim] in `gu`, down [dim][inter] in `dn`, rows contiguous.
#[derive(Clone, Copy)]
pub struct ExpertW<'a> {
    pub gu: GType,
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub dn: GType,
    pub down: &'a [u8],
}

impl ExpertW<'_> {
    pub fn inter(&self, dim: usize) -> usize {
        self.gate.len() / self.gu.row_bytes(dim)
    }
    pub fn bytes(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }
}

#[inline]
pub fn swiglu(g: f32, u: f32, w: f32) -> f32 {
    silu(g.min(SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT) * w
}

/// Single-threaded: x [t][dim], w [t], out [t][dim] +=.
pub fn expert_forward(e: &ExpertW, dim: usize, x: &[f32], t: usize, w: &[f32], out: &mut [f32]) {
    let inter = e.inter(dim);
    let xa = QAct::new(x, t, dim, &[e.gu]);
    let (mut g, mut u) = (vec![0f32; t * inter], vec![0f32; t * inter]);
    dot_rows(e.gu, e.gate, dim, &xa, &mut g);
    dot_rows(e.gu, e.up, dim, &xa, &mut u);
    let h: Vec<f32> = (0..t * inter).map(|i| swiglu(g[i], u[i], w[i / inter])).collect();
    let ha = QAct::new(&h, t, inter, &[e.dn]);
    let mut y = vec![0f32; t * dim];
    dot_rows(e.dn, e.down, inter, &ha, &mut y);
    for (o, y) in out.iter_mut().zip(&y) {
        *o += y;
    }
}

/// One layer's CPU experts on a pool: x quantized across workers, barrier, gate+up rows split across workers
/// (h quantized per 256-block by whoever finishes its last rows), barrier, then down rows (each worker owns
/// its output rows). The caller spins until done, so it should not sit on one of the pool's CPUs.
pub struct CpuMoe {
    pub pool: Pool,
    xa: QAct,
    ha: Vec<QAct>,
    h: Vec<Vec<f32>>,
    toks: Vec<Vec<usize>>,
    cnt: Vec<AtomicU32>,
    next: [Padded; 2],
    /// the experts of this call are stored in the Hadamard-rotated basis: rotate x and h before quantizing them
    pub rot: bool,
}

#[repr(align(128))]
#[derive(Default)]
struct Padded(AtomicUsize);

struct Job<'a> {
    ex: &'a [ExpertW<'a>],
    dim: usize,
    inter: usize,
    x: &'a [f32],
    xa: *const QAct,
    xa_raw: QActRaw,
    ha: *const QAct,
    ha_raw: Vec<QActRaw>,
    h: Vec<*mut f32>,
    toks: &'a [Vec<usize>],
    w: &'a [f32],
    t: usize,
    cnt: &'a [AtomicU32],
    next: &'a [Padded; 2],
    out: *mut f32,
    rot: bool,
}
unsafe impl Sync for Job<'_> {}

impl CpuMoe {
    pub fn new(pool: Pool) -> CpuMoe {
        CpuMoe {
            pool,
            xa: QAct::default(),
            ha: vec![],
            h: vec![],
            toks: vec![],
            cnt: vec![],
            next: Default::default(),
            rot: false,
        }
    }

    /// x [t][dim]; w [expert][t] routing weights (0 = token not routed there); out [t][dim] +=.
    pub fn forward(&mut self, ex: &[ExpertW], dim: usize, x: &[f32], t: usize, w: &[f32], out: &mut [f32]) {
        assert!(t <= MAX_TOKENS && x.len() == t * dim && w.len() == ex.len() * t && out.len() == t * dim);
        if ex.is_empty() {
            return;
        }
        let inter = ex[0].inter(dim);
        assert!(inter % QK == 0 && dim % RD == 0);
        let ne = ex.len();
        let gu: Vec<GType> = ex.iter().map(|e| e.gu).collect();
        self.xa.reset(t, dim, &gu);
        self.toks.resize_with(ne, Vec::new);
        self.ha.resize_with(ne, QAct::default);
        self.h.resize_with(ne, Vec::new);
        for (e, xe) in ex.iter().enumerate() {
            assert_eq!(xe.inter(dim), inter);
            let tk = &mut self.toks[e];
            tk.clear();
            tk.extend((0..t).filter(|&j| w[e * t + j] != 0.0));
            self.ha[e].reset(tk.len(), inter, &[xe.dn]);
            if self.h[e].len() < tk.len() * inter {
                self.h[e].resize(tk.len() * inter, 0.0);
            }
        }
        let nb = inter / QK;
        self.cnt.resize_with(ne * nb, || AtomicU32::new(0));
        for c in &self.cnt[..ne * nb] {
            c.store(0, Relaxed);
        }
        for n in &self.next {
            n.0.store(0, Relaxed);
        }
        let job = Job {
            ex,
            dim,
            inter,
            x,
            xa_raw: self.xa.raw(),
            xa: &self.xa,
            ha: self.ha.as_ptr(),
            ha_raw: self.ha.iter_mut().map(|a| a.raw()).collect(),
            h: self.h.iter_mut().map(|v| v.as_mut_ptr()).collect(),
            toks: &self.toks,
            w,
            t,
            cnt: &self.cnt,
            next: &self.next,
            out: out.as_mut_ptr(),
            rot: self.rot,
        };
        self.pool.run(&|c: &Ctx| job.work(c));
    }
}

impl Job<'_> {
    fn work(&self, c: &Ctx) {
        let (dim, inter) = (self.dim, self.inter);
        let upe = inter / RU; // units per expert
        let ne = self.ex.len();
        let mut g = [0f32; MAX_TOKENS * RU];
        let mut u = [0f32; MAX_TOKENS * RU];
        // x quantized in parallel, block i by worker i % n
        let xb = dim / QK;
        let mut rb = [0f32; QK];
        for i in (c.id..self.t * xb).step_by(c.n) {
            let blk = &self.x[i * QK..(i + 1) * QK];
            let src: &[f32] = if self.rot {
                rb.copy_from_slice(blk);
                fwht_blocks(&mut rb);
                &rb
            } else {
                blk
            };
            unsafe { self.xa_raw.quantize_block(i / xb, i % xb, src) };
        }
        c.barrier();
        let xa = unsafe { &*self.xa };
        loop {
            let i = self.next[0].0.fetch_add(1, Relaxed);
            if i >= ne * upe {
                break;
            }
            let (e, r0) = (i / upe, i % upe * RU);
            let (ex, tk) = (&self.ex[e], &self.toks[e]);
            if tk.is_empty() {
                continue;
            }
            let rb = ex.gu.row_bytes(dim);
            let rows = r0 * rb..(r0 + RU) * rb;
            dot_rows_sel(ex.gu, &ex.gate[rows.clone()], dim, xa, tk, &mut g, RU);
            dot_rows_sel(ex.gu, &ex.up[rows], dim, xa, tk, &mut u, RU);
            let h = self.h[e];
            for (j, &tok) in tk.iter().enumerate() {
                let wt = self.w[e * self.t + tok];
                for k in 0..RU {
                    unsafe { *h.add(j * inter + r0 + k) = swiglu(g[j * RU + k], u[j * RU + k], wt) };
                }
            }
            let b = r0 / QK;
            if self.cnt[e * (inter / QK) + b].fetch_add(1, AcqRel) as usize == QK / RU - 1 {
                for j in 0..tk.len() {
                    unsafe {
                        let hb = std::slice::from_raw_parts_mut(h.add(j * inter + b * QK), QK);
                        if self.rot {
                            fwht_blocks(hb);
                        }
                        self.ha_raw[e].quantize_block(j, b, hb);
                    }
                }
            }
        }
        c.barrier();
        let ha = unsafe { std::slice::from_raw_parts(self.ha, ne) };
        let y = &mut g;
        loop {
            let i = self.next[1].0.fetch_add(1, Relaxed);
            if i >= dim / RD {
                break;
            }
            let r0 = i * RD;
            for (e, ex) in self.ex.iter().enumerate() {
                let tk = &self.toks[e];
                if tk.is_empty() {
                    continue;
                }
                let rb = ex.dn.row_bytes(inter);
                let all: [usize; MAX_TOKENS] = std::array::from_fn(|j| j);
                dot_rows_sel(
                    ex.dn,
                    &ex.down[r0 * rb..(r0 + RD) * rb],
                    inter,
                    &ha[e],
                    &all[..tk.len()],
                    y,
                    RD,
                );
                for (j, &tok) in tk.iter().enumerate() {
                    for k in 0..RD {
                        unsafe { *self.out.add(tok * dim + r0 + k) += y[j * RD + k] };
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::pool::Pool;
    use super::super::qdot::tests::{Rng, rand_rows};
    use super::*;
    use crate::formats::dequant_row;
    use std::time::Duration;

    fn f32_matvec(ty: GType, rows: &[u8], cols: usize, x: &[f32]) -> Vec<f32> {
        let mut w = vec![0f32; cols];
        rows.chunks_exact(ty.row_bytes(cols))
            .map(|r| {
                dequant_row(ty, r, &mut w);
                w.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
            })
            .collect()
    }

    fn f32_expert(e: &ExpertW, dim: usize, x: &[f32], w: f32) -> Vec<f32> {
        let g = f32_matvec(e.gu, e.gate, dim, x);
        let u = f32_matvec(e.gu, e.up, dim, x);
        let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g, u, w)).collect();
        f32_matvec(e.dn, e.down, e.inter(dim), &h)
    }

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
        (num / den).sqrt()
    }

    struct Ex {
        gu: GType,
        dn: GType,
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
    }
    impl Ex {
        fn w(&self) -> ExpertW<'_> {
            ExpertW {
                gu: self.gu,
                gate: &self.gate,
                up: &self.up,
                dn: self.dn,
                down: &self.down,
            }
        }
    }

    // reduced inter keeps the f32 reference quick; dim is the model's
    const DIM: usize = 5120;
    const INTER: usize = 768;

    fn experts(rng: &mut Rng) -> Vec<Ex> {
        use GType::*;
        [
            (Q2_0, Q3K),
            (Ptq1_0, Q2_0),
            (Q3K, Ptq1_0),
            (Q2_0, Q2_0),
            (Ptq1_0, Ptq1_0),
            (Q8_0, Q8_0),
            (Iq1S, Iq2Xxs),
            (Iq2Xxs, Iq2Xs),
            (Iq2S, Iq1S),
            (Iq2Xs, Iq2S),
        ]
        .iter()
        .map(|&(gu, dn)| Ex {
            gu,
            dn,
            gate: rand_rows(gu, INTER, DIM, rng),
            up: rand_rows(gu, INTER, DIM, rng),
            down: rand_rows(dn, DIM, INTER, rng),
        })
        .collect()
    }

    /// f32 weights, activations rounded through the same int8 quantization at both stages
    fn mirrored_expert(e: &ExpertW, dim: usize, x: &[f32], w: f32, qx: bool, qh: bool) -> Vec<f32> {
        let xq = if qx {
            QAct::new(x, 1, dim, &[e.gu]).dequant(0)
        } else {
            x.to_vec()
        };
        let g = f32_matvec(e.gu, e.gate, dim, &xq);
        let u = f32_matvec(e.gu, e.up, dim, &xq);
        let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g, u, w)).collect();
        let hq = if qh {
            QAct::new(&h, 1, h.len(), &[e.dn]).dequant(0)
        } else {
            h
        };
        f32_matvec(e.dn, e.down, e.inter(dim), &hq)
    }

    #[test]
    fn expert_vs_f32() {
        let mut rng = Rng(42);
        let exs = experts(&mut rng);
        let t = 3;
        // |gate x| ~ 1-2 with these random weights, like the model's
        let x: Vec<f32> = (0..t * DIM).map(|_| 0.3 * rng.normal()).collect();
        let w = [0.7f32, 1.3, 0.2];
        for ex in &exs {
            let e = ex.w();
            let mut out = vec![0f32; t * DIM];
            expert_forward(&e, DIM, &x, t, &w, &mut out);
            for j in 0..t {
                let xj = &x[j * DIM..(j + 1) * DIM];
                let o = &out[j * DIM..(j + 1) * DIM];
                let r = f32_expert(&e, DIM, xj, w[j]);
                let (em, ef) = (rel(o, &mirrored_expert(&e, DIM, xj, w[j], true, true)), rel(o, &r));
                if j == 0 {
                    let ex = rel(&mirrored_expert(&e, DIM, xj, w[j], true, false), &r);
                    let eh = rel(&mirrored_expert(&e, DIM, xj, w[j], false, true), &r);
                    println!(
                        "expert {:?}/{:?}: vs int8-mirrored f32 {em:.2e}, vs pure f32 {ef:.2e} (x quant alone {ex:.2e}, h quant alone {eh:.2e})",
                        e.gu, e.dn
                    );
                }
                assert!(em < 2e-3, "{:?}/{:?} token {j}: mirrored {em}", e.gu, e.dn);
                assert!(ef < 5e-2, "{:?}/{:?} token {j}: f32 {ef}", e.gu, e.dn);
            }
        }
    }

    #[test]
    fn pooled_matches_single() {
        let mut rng = Rng(5);
        let exs = experts(&mut rng);
        let ew: Vec<ExpertW> = exs.iter().map(|e| e.w()).collect();
        let mut moe = CpuMoe::new(Pool::new(&[0, 1, 2, 3, 4, 5], Duration::from_millis(1)));
        for t in [1usize, 4, 11] {
            let x: Vec<f32> = (0..t * DIM).map(|_| rng.normal()).collect();
            // sparse routing: some tokens skip some experts
            let w: Vec<f32> = (0..ew.len() * t)
                .map(|i| if i % 3 == 1 { 0.0 } else { 0.1 + rng.f32() })
                .collect();
            let mut want = vec![0.5f32; t * DIM];
            for (e, ex) in ew.iter().enumerate() {
                for j in 0..t {
                    let wt = w[e * t + j];
                    if wt != 0.0 {
                        expert_forward(
                            ex,
                            DIM,
                            &x[j * DIM..(j + 1) * DIM],
                            1,
                            &[wt],
                            &mut want[j * DIM..(j + 1) * DIM],
                        );
                    }
                }
            }
            for _ in 0..3 {
                let mut got = vec![0.5f32; t * DIM];
                moe.forward(&ew, DIM, &x, t, &w, &mut got);
                let err = rel(&got, &want);
                assert!(err < 1e-5, "T {t}: pooled vs single {err}");
            }
        }
    }
}
