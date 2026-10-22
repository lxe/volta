//! Routed experts resident in a GPU's memory (the main GPU's spare VRAM, or an aux GPU as a pure expert store), computed there
//! for one token: out = sum_e w_e * down(silu(min(gate x, L)) * clamp(up x, -L, L)).
//! By default the weights are repacked at upload (same bytes) for the fast kernels (expert_fast.cuh: int8 tensor-core
//! mma, int8 activations, up to 8 tokens per weight read); DSX_OLD_MOE=1 keeps the GGUF blocks and moe_gu / moe_down.

use crate::gguf::GType;
use crate::gpu::{Arg, Buf, Event, Gpu, Kernel};
use crate::model::Model;
use anyhow::{Result, bail};
use std::collections::HashMap;

struct Slot {
    gate: u64,
    up: u64,
    down: u64,
    gu: GType,
    dn: GType,
}

/// One expert's GGUF matrices: gate/up [inter][dim] in type `gu`, down [dim][inter] in `dn`.
pub struct RawExpert<'a> {
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
    pub gu: GType,
    pub dn: GType,
}

#[derive(Clone, Copy)]
pub struct ExpCfg {
    pub n_layers: usize,
    pub n_experts: usize,
    pub inter: usize,
    pub dim: usize,
    pub lim: f32,
    /// bit l: layer l's experts are stored in the Hadamard-rotated input basis (quant_act rotates x and h)
    pub rot: u64,
}

/// most tokens (T) and picks per token (K) of one run_device_fast call
pub const MAXT: usize = 8;
pub const MAXK: usize = 8;
/// warps per block of the fast kernels (KS in expert_fast.cuh): they split a 16-row tile's x-blocks
/// 16-row tiles per block of the fast kernels (ROWT in expert_fast.cuh; build with DSX_ROWT=n)
pub const ROWT: u32 = match option_env!("DSX_ROWT") {
    Some(s) => (s.as_bytes()[0] - b'0') as u32,
    None => 1,
};
pub const KS: u32 = match option_env!("DSX_KS") {
    Some(s) => (s.as_bytes()[0] - b'0') as u32,
    None => 2,
};

/// scratch of the fast path: x and h quantized (quant_act layout), h in f32, the call's plan, and moe_down_fast's
/// per-entry partial sums and row-tile counters
struct Fast {
    xq: Buf,
    xd: Buf,
    xs: Buf,
    h: Buf,
    hq: Buf,
    hd: Buf,
    hs: Buf,
    plan: Buf,
    part: Buf,
    cnt: Buf,
}

pub struct GpuExperts {
    buf: Buf,
    /// device residency table [layers][E] of ExpSlot (offsets into `buf`, ~0 = not here), for moe_gu / moe_down
    tab: Buf,
    pub sel: Buf,
    pub wts: Buf,
    h: Buf,
    n_experts: usize,
    slots: HashMap<(usize, usize), Slot>,
    /// device buffers for one token: x in, out (sum), and scratch
    pub x: Buf,
    pub out: Buf,
    tmp_g: Buf,
    tmp_u: Buf,
    tmp_h: Buf,
    tmp_o: Buf,
    k: HashMap<&'static str, Kernel>,
    inter: usize,
    dim: usize,
    lim: f32,
    rot: u64,
    pub bytes: usize,
    /// dynamic shared memory of the fast kernels: the largest codebook table among the store's types
    smem: u32,
    /// weights repacked for the fast kernels (expert_fast.cuh); None = raw GGUF blocks (DSX_OLD_MOE=1)
    fast: Option<Fast>,
    /// adaptive residency (enable_swaps): in-flight replacements
    swap: Option<Swapper>,
}

/// An expert replacing a same-type victim in place: its raw matrices go to a staging buffer and are repacked into the
/// victim's slot on the side stream; it becomes resident (table entry, `slots`) only at commit, between tokens.
struct Pending {
    l: usize,
    e: usize,
    slot: Slot,
    stage: usize,
}

struct Swapper {
    /// one staging buffer (gate | up | down raw) and one event per in-flight replacement
    stage: Vec<(Buf, Event)>,
    /// pinned ring of 32-byte table entries for async writes (at most a few dozen per token; the main stream is
    /// synchronized every token, so a slot is free again long before the ring wraps)
    ring: crate::gpu::HostBuf,
    ring_i: usize,
    free: Vec<usize>,
    pending: Vec<Pending>,
}

fn gemv_name(t: GType) -> Result<&'static str> {
    Ok(match t {
        GType::Q2_0 => "gemv_q2_0",
        GType::Ptq1_0 => "gemv_ptq1_0",
        GType::Q3K => "gemv_q3_k",
        GType::Q8_0 => "gemv_q8_0",
        t => bail!("no GPU expert kernel for {t:?}"),
    })
}

/// Dynamic shared memory the fast kernels need for `t`'s codebook tables (expert_fast.cuh iq_tab), bytes.
pub fn tile_smem(t: GType) -> u32 {
    4 * match t {
        GType::Iq1S => 0, // its grid is read through L1
        GType::Iq2S => 2048 + 256,
        GType::Iq2Xs => 1024 + 256,
        GType::Iq2Xxs => 512 + 256,
        _ => 0,
    }
}

/// Type id of `t` repacked for the fast kernels (expert_fast.cuh), the same bytes per weight.
pub fn repacked_id(t: GType) -> Result<i32> {
    if !matches!(t, GType::Iq2Xxs | GType::Iq1S | GType::Iq2Xs | GType::Iq2S) {
        gemv_name(t)?; // IQ2_XXS has only the fast (repacked) kernels
    }
    Ok(2000 + t as u32 as i32)
}

/// Enqueue: dst (a matrix of rows x k, same size as src) = src (raw GGUF blocks of type t) repacked.
pub fn repack(g: &Gpu, t: GType, src: u64, dst: u64, rows: usize, k: usize) -> Result<()> {
    repack_on(g, false, t, src, dst, rows, k)
}

fn repack_on(g: &Gpu, side: bool, t: GType, src: u64, dst: u64, rows: usize, k: usize) -> Result<()> {
    repacked_id(t)?;
    let blocks = rows * k / t.block().0;
    let args = [Arg::I(t as u32 as i32), Arg::P(src), Arg::P(dst), Arg::I(rows as i32), Arg::I(k as i32)];
    let grid = ((blocks as u32).div_ceil(256), 1, 1);
    if side { g.launch_side(g.kernel("repack_expert")?, grid, 256, 0, &args) } else { g.launch(g.kernel("repack_expert")?, grid, 256, 0, &args) }
}

impl GpuExperts {
    /// Upload `which` (layer, expert) pairs into one device buffer on `g`, repacked for the fast kernels unless
    /// DSX_OLD_MOE is set.
    pub fn new(g: &Gpu, m: &Model, which: &[(usize, usize)]) -> Result<GpuExperts> {
        let c = &m.cfg;
        let cfg = ExpCfg { n_layers: c.n_layers, n_experts: c.n_experts, inter: c.inter, dim: c.dim, lim: c.swiglu_limit, rot: c.expert_rot.iter().enumerate().fold(0u64, |m, (l, &r)| m | ((r as u64) << l)) };
        let items: Vec<((usize, usize), RawExpert)> = which
            .iter()
            .map(|&(l, e)| {
                let x = &m.layers[l].experts[e];
                ((l, e), RawExpert { gate: x.gate.bytes(), up: x.up.bytes(), down: x.down.bytes(), gu: x.gate.ty, dn: x.down.ty })
            })
            .collect();
        Self::build(g, cfg, &items, std::env::var_os("DSX_OLD_MOE").is_none())
    }

    /// Upload experts given as raw GGUF bytes; `fast` = repacked for the fast kernels (else raw, for moe_gu / moe_down).
    pub fn build(g: &Gpu, c: ExpCfg, items: &[((usize, usize), RawExpert)], fast: bool) -> Result<GpuExperts> {
        if fast && (c.dim % 256 != 0 || c.inter % 256 != 0) {
            bail!("fast expert kernels need dim and inter multiples of 256");
        }
        // matrices 256-byte aligned (the fast kernels load 16-byte vectors)
        let al = |n: usize| n.next_multiple_of(256);
        let total: usize = items.iter().map(|(_, x)| al(x.gate.len()) + al(x.up.len()) + al(x.down.len())).sum();
        let buf = g.alloc(total.max(1))?;
        let stage = if fast { Some(g.alloc(items.iter().map(|(_, x)| x.gate.len().max(x.down.len())).max().unwrap_or(1))?) } else { None };
        let mut slots = HashMap::new();
        let mut off = 0u64;
        for ((l, e), x) in items {
            let mut put = |b: &[u8], t: GType, rows: usize, k: usize| -> Result<u64> {
                let p = buf.ptr + off;
                match &stage {
                    Some(s) => {
                        g.htod_sync(s.ptr, b)?;
                        repack(g, t, s.ptr, p, rows, k)?;
                    }
                    None => g.htod_sync(p, b)?,
                }
                off += al(b.len()) as u64;
                Ok(p)
            };
            for t in [x.gu, x.dn] {
                // a kernel exists for the layout; raw IQ stores (DSX_OLD_MOE) only have prefill's dequant_f16 so far
                if fast {
                    repacked_id(t)?;
                } else if !matches!(t, GType::Iq1S | GType::Iq2Xs | GType::Iq2S | GType::Iq2Xxs) {
                    gemv_name(t)?;
                }
            }
            let (gp, up, dp) = (put(x.gate, x.gu, c.inter, c.dim)?, put(x.up, x.gu, c.inter, c.dim)?, put(x.down, x.dn, c.dim, c.inter)?);
            slots.insert((*l, *e), Slot { gate: gp, up, down: dp, gu: x.gu, dn: x.dn });
        }
        g.sync()?;
        drop(stage);
        let id = |t: GType| if fast { 2000 + t as u32 } else { t as u32 };
        // ExpSlot {u64 gate, up, down; i32 gu, dn} = 32 bytes, offsets relative to the store base
        let mut tab = vec![0u8; c.n_layers * c.n_experts * 32];
        for l in 0..c.n_layers {
            for e in 0..c.n_experts {
                let o = (l * c.n_experts + e) * 32;
                match slots.get(&(l, e)) {
                    Some(sl) => {
                        tab[o..o + 8].copy_from_slice(&(sl.gate - buf.ptr).to_le_bytes());
                        tab[o + 8..o + 16].copy_from_slice(&(sl.up - buf.ptr).to_le_bytes());
                        tab[o + 16..o + 24].copy_from_slice(&(sl.down - buf.ptr).to_le_bytes());
                        tab[o + 24..o + 28].copy_from_slice(&id(sl.gu).to_le_bytes());
                        tab[o + 28..o + 32].copy_from_slice(&id(sl.dn).to_le_bytes());
                    }
                    None => tab[o..o + 8].copy_from_slice(&u64::MAX.to_le_bytes()),
                }
            }
        }
        let tab = g.upload(&tab)?;
        let mut k = HashMap::new();
        for n in ["gemv_q2_0", "gemv_ptq1_0", "gemv_q3_k", "gemv_q8_0", "swiglu_s", "add_inplace", "moe_gu", "moe_down", "quant_act", "moe_gu_fast", "moe_down_fast"] {
            k.insert(n, g.kernel(n)?);
        }
        let f = |n: usize| g.alloc_zeroed(n * 4);
        let fast = if fast {
            let (nx, nh) = (MAXT * c.dim, MAXT * MAXK * c.inter);
            Some(Fast {
                xq: g.alloc_zeroed(nx)?,
                xd: f(nx / 256)?,
                xs: g.alloc_zeroed(nx / 64 * 2)?,
                h: f(nh)?,
                hq: g.alloc_zeroed(nh)?,
                hd: f(nh / 256)?,
                hs: g.alloc_zeroed(nh / 64 * 2)?,
                plan: f(1 + MAXT * MAXK * 10)?,
                part: f(MAXT * MAXK * 8 * c.dim)?,
                cnt: f(c.dim.div_ceil(16))?,
            })
        } else {
            None
        };
        let smem = slots.values().flat_map(|s| [tile_smem(s.gu), tile_smem(s.dn)]).max().unwrap_or(0);
        Ok(GpuExperts {
            buf,
            tab,
            sel: g.alloc_zeroed(4 * MAXT * MAXK)?,
            wts: g.alloc_zeroed(4 * MAXT * MAXK)?,
            h: f(16 * c.inter)?,
            n_experts: c.n_experts,
            slots,
            x: f(MAXT * c.dim)?,
            out: f(MAXT * c.dim)?,
            tmp_g: f(c.inter)?,
            tmp_u: f(c.inter)?,
            tmp_h: f(c.inter)?,
            tmp_o: f(c.dim)?,
            k,
            inter: c.inter,
            dim: c.dim,
            lim: c.lim,
            rot: c.rot,
            bytes: total,
            smem,
            fast,
            swap: None,
        })
    }

    /// Allow up to `n` in-flight replacements (fast layout only), each staging at most `max_raw` bytes.
    pub fn enable_swaps(&mut self, g: &Gpu, n: usize, max_raw: usize) -> Result<()> {
        if self.fast.is_none() || n == 0 {
            return Ok(());
        }
        let stage = (0..n).map(|_| Ok((g.alloc(max_raw)?, g.event()?))).collect::<Result<Vec<_>>>()?;
        self.swap = Some(Swapper { stage, ring: crate::gpu::HostBuf::new(32 * 1024)?, ring_i: 0, free: (0..n).rev().collect(), pending: vec![] });
        Ok(())
    }

    /// (gate/up type, down type) of a resident expert
    pub fn types(&self, l: usize, e: usize) -> Option<(GType, GType)> {
        self.slots.get(&(l, e)).map(|s| (s.gu, s.dn))
    }

    pub fn residents(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.slots.keys().copied()
    }

    /// replacements started and not committed yet
    pub fn pending(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.swap.iter().flat_map(|s| s.pending.iter().map(|p| (p.l, p.e)))
    }

    /// free staging buffers (how many replacements can start now)
    pub fn swap_room(&self) -> usize {
        self.swap.as_ref().map_or(0, |s| s.free.len())
    }

    /// ExpSlot entry of `sl` (None = not resident) written to the device table, ordered on the main stream
    fn write_entry(&mut self, g: &Gpu, l: usize, e: usize, sl: Option<&Slot>) -> Result<()> {
        let mut b = [0u8; 32];
        match sl {
            Some(sl) => {
                let id = |t: GType| if self.fast.is_some() { 2000 + t as u32 } else { t as u32 };
                b[0..8].copy_from_slice(&(sl.gate - self.buf.ptr).to_le_bytes());
                b[8..16].copy_from_slice(&(sl.up - self.buf.ptr).to_le_bytes());
                b[16..24].copy_from_slice(&(sl.down - self.buf.ptr).to_le_bytes());
                b[24..28].copy_from_slice(&id(sl.gu).to_le_bytes());
                b[28..32].copy_from_slice(&id(sl.dn).to_le_bytes());
            }
            None => b[0..8].copy_from_slice(&u64::MAX.to_le_bytes()),
        }
        let dst = self.tab.ptr + ((l * self.n_experts + e) * 32) as u64;
        match self.swap.as_mut() {
            Some(sw) => {
                let i = sw.ring_i % 1024;
                sw.ring_i += 1;
                let r = &mut sw.ring.as_mut_slice::<u8>()[i * 32..i * 32 + 32];
                r.copy_from_slice(&b);
                g.htod(dst, r)
            }
            None => g.htod_sync(dst, &b),
        }
    }

    /// Between tokens (both streams idle on this store's experts): drop `victim` now and start loading `new`
    /// (raw GGUF matrices `raw`, same types as the victim) into its slot. `new` is resident after commit_ready.
    pub fn begin_swap(&mut self, g: &Gpu, victim: (usize, usize), new: (usize, usize), raw: [&[u8]; 3]) -> Result<bool> {
        let (Some(sw), Some(vs)) = (self.swap.as_mut(), self.slots.get(&victim)) else { return Ok(false) };
        let Some(si) = sw.free.pop() else { return Ok(false) };
        let (stage, ev) = (sw.stage[si].0.ptr, sw.stage[si].1);
        let (gu, dn, inter, dim) = (vs.gu, vs.dn, self.inter, self.dim);
        let slot = Slot { gate: vs.gate, up: vs.up, down: vs.down, gu, dn };
        let _ = sw;
        self.slots.remove(&victim);
        self.write_entry(g, victim.0, victim.1, None)?;
        let (og, ou) = (raw[0].len() as u64, (raw[0].len() + raw[1].len()) as u64);
        g.htod_side(stage, raw[0])?;
        g.htod_side(stage + og, raw[1])?;
        g.htod_side(stage + ou, raw[2])?;
        repack_on(g, true, gu, stage, slot.gate, inter, dim)?;
        repack_on(g, true, gu, stage + og, slot.up, inter, dim)?;
        repack_on(g, true, dn, stage + ou, slot.down, dim, inter)?;
        g.record(ev, true)?;
        self.swap.as_mut().unwrap().pending.push(Pending { l: new.0, e: new.1, slot, stage: si });
        Ok(true)
    }

    /// Make finished replacements resident (`all`: wait for every one). Returns the experts committed.
    pub fn commit_ready(&mut self, g: &Gpu, all: bool) -> Result<Vec<(usize, usize)>> {
        let Some(sw) = self.swap.as_mut() else { return Ok(vec![]) };
        if all && !sw.pending.is_empty() {
            g.side_sync()?;
        }
        let mut done = vec![];
        let mut i = 0;
        while i < sw.pending.len() {
            if g.done(sw.stage[sw.pending[i].stage].1)? {
                done.push(sw.pending.swap_remove(i));
            } else {
                i += 1;
            }
        }
        let mut out = vec![];
        for p in done {
            self.write_entry(g, p.l, p.e, Some(&p.slot))?;
            self.slots.insert((p.l, p.e), p.slot);
            self.swap.as_mut().unwrap().free.push(p.stage);
            out.push((p.l, p.e));
        }
        Ok(out)
    }

    /// device pointers (gate, up, down) of a resident expert and the type ids of gate/up and down as dequant_f16
    /// takes them (ggml id, or 2000 + id when repacked)
    pub fn slot_ptrs(&self, l: usize, e: usize) -> Option<(u64, u64, u64, i32, i32)> {
        let off = if self.fast.is_some() { 2000 } else { 0 };
        self.slots.get(&(l, e)).map(|s| (s.gate, s.up, s.down, off + s.gu as u32 as i32, off + s.dn as u32 as i32))
    }

    pub fn has(&self, l: usize, e: usize) -> bool {
        self.slots.contains_key(&(l, e))
    }

    /// true when the weights are in the fast-kernel layout
    pub fn is_fast(&self) -> bool {
        self.fast.is_some()
    }

    fn gemv(&self, g: &Gpu, ty: GType, w: u64, x: u64, y: u64, rows: usize, cols: usize) -> Result<()> {
        g.launch(
            self.k[gemv_name(ty)?],
            ((rows as u32).div_ceil(4), 1, 1),
            128,
            0,
            &[Arg::P(w), Arg::P(x), Arg::P(y), Arg::I(rows as i32), Arg::I(cols as i32), Arg::I(1), Arg::I(cols as i32), Arg::I(rows as i32)],
        )
    }

    /// Enqueue on g's stream: out (if `accumulate` is false it is zeroed first) += experts of layer `l`, input `x`
    /// (a device pointer on g).
    pub fn run(&self, g: &Gpu, l: usize, list: &[(usize, f32)], x: u64, out: u64) -> Result<()> {
        if self.fast.is_some() || std::env::var_os("DSX_MOE_DEVICE").is_some() {
            let sel: Vec<i32> = list.iter().map(|&(e, _)| e as i32).collect();
            let w: Vec<f32> = list.iter().map(|&(_, w)| w).collect();
            g.htod_sync(self.sel.ptr, &sel)?;
            g.htod_sync(self.wts.ptr, &w)?;
            return self.run_device(g, l, self.sel.ptr, self.wts.ptr, list.len(), x, out);
        }
        g.memset(out, self.dim * 4)?;
        for &(e, w) in list {
            let s = &self.slots[&(l, e)];
            self.gemv(g, s.gu, s.gate, x, self.tmp_g.ptr, self.inter, self.dim)?;
            self.gemv(g, s.gu, s.up, x, self.tmp_u.ptr, self.inter, self.dim)?;
            g.launch(
                self.k["swiglu_s"],
                ((self.inter as u32).div_ceil(256), 1, 1),
                256,
                0,
                &[Arg::P(self.tmp_g.ptr), Arg::P(self.tmp_u.ptr), Arg::P(self.tmp_h.ptr), Arg::I(self.inter as i32), Arg::F(self.lim), Arg::F(w)],
            )?;
            self.gemv(g, s.dn, s.down, self.tmp_h.ptr, self.tmp_o.ptr, self.dim, self.inter)?;
            g.launch(self.k["add_inplace"], ((self.dim as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(out), Arg::P(self.tmp_o.ptr), Arg::I(self.dim as i32)])?;
        }
        Ok(())
    }
}

impl GpuExperts {
    /// Enqueue the resident picks of layer `l` (sel/wts device pointers, K picks) for one token: out = their
    /// weighted sum. Fast kernels when the store is repacked, else moe_gu / moe_down.
    pub fn run_device(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, x: u64, out: u64) -> Result<()> {
        anyhow::ensure!(self.fast.is_some() || (self.rot >> l) & 1 == 0, "rotated experts need the repacked store (unset DSX_OLD_MOE)");
        if self.fast.is_some() {
            return self.run_device_fast(g, l, sel, wts, k, 1, x, out);
        }
        g.launch(
            self.k["moe_gu"],
            ((self.inter as u32).div_ceil(4), k as u32, 1),
            128,
            0,
            &[Arg::P(self.tab.ptr), Arg::I(l as i32), Arg::I(self.n_experts as i32), Arg::P(sel), Arg::P(wts), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(x), Arg::P(self.h.ptr), Arg::I(self.inter as i32), Arg::I(self.dim as i32), Arg::F(self.lim)],
        )?;
        g.launch(
            self.k["moe_down"],
            ((self.dim as u32).div_ceil(4), 1, 1),
            128,
            0,
            &[Arg::P(self.tab.ptr), Arg::I(l as i32), Arg::I(self.n_experts as i32), Arg::P(sel), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(self.h.ptr), Arg::P(out), Arg::I(self.inter as i32), Arg::I(self.dim as i32)],
        )
    }

    /// Enqueue T tokens' resident picks of layer `l` on the repacked store: sel/wts [T][K] and x [T][dim] device
    /// pointers, out [T][dim] = per token the weighted sum of its resident picks (0 if none). An expert picked by
    /// several tokens is read once. Four launches: quantize x (+ plan), gate/up, quantize h, down.
    #[allow(clippy::too_many_arguments)]
    pub fn run_device_fast(&self, g: &Gpu, l: usize, sel: u64, wts: u64, k: usize, t: usize, x: u64, out: u64) -> Result<()> {
        let Some(f) = self.fast.as_ref() else { bail!("expert store is not repacked (DSX_OLD_MOE)") };
        if t == 0 || t > MAXT || k == 0 || k > MAXK {
            bail!("run_device_fast: T = {t}, K = {k} (max {MAXT}, {MAXK})");
        }
        let (dim, inter) = (self.dim, self.inter);
        let (tab, e) = (Arg::P(self.tab.ptr), Arg::I(self.n_experts as i32));
        let rot = ((self.rot >> l) & 1) as i32;
        g.launch(
            self.k["quant_act"],
            ((dim / 256) as u32, t as u32, 1),
            256,
            0,
            &[Arg::P(x), Arg::I(dim as i32), Arg::P(f.xq.ptr), Arg::P(f.xd.ptr), Arg::P(f.xs.ptr), Arg::P(sel), tab, Arg::I(l as i32), e, Arg::I(k as i32), Arg::I(t as i32), Arg::P(f.plan.ptr), Arg::I(rot)],
        )?;
        g.launch(
            self.k["moe_gu_fast"],
            ((inter as u32).div_ceil(16 * ROWT), (t * k) as u32, 1),
            32 * KS,
            self.smem,
            &[tab, Arg::I(l as i32), e, Arg::P(f.plan.ptr), Arg::P(wts), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(f.xq.ptr), Arg::P(f.xd.ptr), Arg::P(f.xs.ptr), Arg::P(f.h.ptr), Arg::I(inter as i32), Arg::I(dim as i32), Arg::F(self.lim)],
        )?;
        g.launch(
            self.k["quant_act"],
            ((inter / 256) as u32, (t * k) as u32, 1),
            256,
            0,
            &[Arg::P(f.h.ptr), Arg::I(inter as i32), Arg::P(f.hq.ptr), Arg::P(f.hd.ptr), Arg::P(f.hs.ptr), Arg::P(0), tab, Arg::I(0), e, Arg::I(0), Arg::I(0), Arg::P(0), Arg::I(rot)],
        )?;
        g.launch(
            self.k["moe_down_fast"],
            ((dim as u32).div_ceil(16 * ROWT), (t * k) as u32, 1),
            32 * KS,
            self.smem,
            &[tab, Arg::I(l as i32), e, Arg::P(f.plan.ptr), Arg::I(k as i32), Arg::P(self.buf.ptr), Arg::P(f.hq.ptr), Arg::P(f.hd.ptr), Arg::P(f.hs.ptr), Arg::P(out), Arg::I(inter as i32), Arg::I(dim as i32), Arg::I(t as i32), Arg::P(f.part.ptr), Arg::P(f.cnt.ptr)],
        )
    }
}

/// Spread experts over the stores (`caps[i]` bytes each), hottest first (a ranking from routing statistics when
/// given, else layer-major round robin from expert 0). Stores with a lower `group` fill first (static stores before
/// adaptive ones, which reshuffle at run time anyway); within a group each expert goes to the least-filled store
/// that has room, so the hot experts, and the decode work, spread evenly over GPUs that run in parallel.
pub fn place(m: &Model, rank: Option<&[(usize, usize)]>, caps: &[usize], group: &[u8]) -> Vec<Vec<(usize, usize)>> {
    let c = &m.cfg;
    let order: Vec<(usize, usize)> = match rank {
        Some(r) => r.to_vec(),
        None => (0..c.n_experts).flat_map(|e| (0..c.n_layers).map(move |l| (l, e))).collect(),
    };
    let size = |l: usize, e: usize| {
        let x = &m.layers[l].experts[e];
        x.gate.bytes().len() + x.up.bytes().len() + x.down.bytes().len()
    };
    let mut out = vec![vec![]; caps.len()];
    for ((l, e), st) in order.iter().zip(assign(order.iter().map(|&(l, e)| size(l, e)), caps, group)) {
        if let Some(i) = st {
            out[i].push((*l, *e));
        }
    }
    out
}

/// place's fill rule on sizes: the store each item goes to (None = no room anywhere)
fn assign(sizes: impl Iterator<Item = usize>, caps: &[usize], group: &[u8]) -> Vec<Option<usize>> {
    let mut used = vec![0usize; caps.len()];
    sizes
        .map(|s| {
            let fill = |i: usize| used[i] as f64 / caps[i] as f64;
            let pick = (0..caps.len()).filter(|&i| used[i] + s <= caps[i]).min_by(|&i, &j| group[i].cmp(&group[j]).then(fill(i).total_cmp(&fill(j))));
            if let Some(i) = pick {
                used[i] += s;
            }
            pick
        })
        .collect()
}

#[cfg(test)]
mod place_tests {
    #[test]
    fn static_first_then_balanced() {
        // two static aux stores of 2 and one adaptive main store of 3; 8 unit experts, the last one has no room
        let o = super::assign([1usize; 8].into_iter(), &[2, 2, 3], &[0, 0, 1]);
        assert_eq!(o, vec![Some(0), Some(1), Some(0), Some(1), Some(2), Some(2), Some(2), None]);
    }

    #[test]
    fn two_store_order_unchanged() {
        // the old two-store rule: GPU1 (static) first, GPU0 takes what does not fit, smaller ones still go to GPU1
        let o = super::assign([3usize, 3, 1].into_iter(), &[4, 10], &[0, 1]);
        assert_eq!(o, vec![Some(0), Some(1), Some(0)]);
    }
}

/// Experts copied out of the mmapped file into one hugepage-backed, CUDA-registered host buffer: the CPU reads them
/// at RAM speed and prefill DMAs them to GPU0 at full PCIe speed (24.7 GB/s on a 2x RTX 3090 host vs 13 from 4 KB pages).
pub struct Arena {
    pub buf: crate::gpu::HostBuf,
    /// (layer, expert) -> (gate, up, down) byte ranges in buf
    pub slots: HashMap<(usize, usize), [(usize, usize); 3]>,
    pub bytes: usize,
}

impl Arena {
    /// Copy `which` experts (in that order) until `cap` bytes; returns the arena and the experts that did not fit.
    pub fn new(m: &Model, which: &[(usize, usize)], cap: usize) -> Result<(Arena, Vec<(usize, usize)>)> {
        let size = |l: usize, e: usize| {
            let x = &m.layers[l].experts[e];
            x.gate.bytes().len() + x.up.bytes().len() + x.down.bytes().len()
        };
        let mut take = vec![];
        let mut rest = vec![];
        let mut tot = 0usize;
        for &(l, e) in which {
            let s = size(l, e);
            if tot + s <= cap {
                take.push((l, e, tot));
                tot += s;
            } else {
                rest.push((l, e));
            }
        }
        let buf = crate::gpu::HostBuf::new(tot.max(1))?;
        let base = buf.host_ptr() as usize;
        let mut slots = HashMap::new();
        for &(l, e, off) in &take {
            let x = &m.layers[l].experts[e];
            let (a, b, c) = (x.gate.bytes().len(), x.up.bytes().len(), x.down.bytes().len());
            slots.insert((l, e), [(off, a), (off + a, b), (off + a + b, c)]);
        }
        // parallel copy (page faults on the source dominate; 16 threads keep the NVMe queue deep)
        use rayon::prelude::*;
        take.par_iter().for_each(|&(l, e, off)| {
            let x = &m.layers[l].experts[e];
            let mut o = off;
            for mm in [&x.gate, &x.up, &x.down] {
                let src = mm.bytes();
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), (base + o) as *mut u8, src.len()) };
                o += src.len();
            }
        });
        Ok((Arena { buf, slots, bytes: tot }, rest))
    }

    pub fn get(&self, l: usize, e: usize) -> Option<[&[u8]; 3]> {
        let s = self.slots.get(&(l, e))?;
        let b = self.buf.as_slice::<u8>();
        Some([&b[s[0].0..s[0].0 + s[0].1], &b[s[1].0..s[1].0 + s[1].1], &b[s[2].0..s[2].0 + s[2].1]])
    }
}
