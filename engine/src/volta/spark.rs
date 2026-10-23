//! DSpark on the GPUs: the checkpoint's draft stack (`mtp.*`; core/src/dspark.rs is the CPU reference). Its 3
//! layers sit after the main layers in `Shard::layers` / `Shard::local` (index `base + j`), so the MoE path (routing,
//! the NVLink exchange, experts split 8 ways) serves them unchanged. Attention is their own: each draft layer keeps a
//! window of the main model's projected hidden states per slot (`observe`), and a block's tokens attend to that
//! window and to each other (bidirectional inside the block). The head is the main model's, plus a Markov bias from
//! each position's previous token, decided one position at a time.

use super::*;
use crate::dspark::Dspark;

pub(super) struct Spark {
    /// index of the first draft layer in Shard::layers / local
    pub base: usize,
    pub n: usize,
    pub block: usize,
    noise: u32,
    main_proj: Lin,
    main_norm: Buf,
    norm: Buf,
    /// fp16 [vocab][rank]
    markov_embed: Buf,
    markov_head: Lin,
    /// head-parallel: this GPU's rows of the Markov head (its slice of the vocabulary, as Shard::head_hp)
    markov_head_hp: Option<super::VMat>,
    rank: usize,
    /// the main model's hidden-state means at the target layers, for the last main step's tokens: [TMAX][nt * d]
    pub mh: Buf,
    pub targets: Vec<usize>,
    /// the draft experts (this GPU's slice of each)
    _exps: Experts,
    /// per step: observed rows (gathered, then projected), window indices, small tables, Markov scratch
    obs: Buf,
    widx: Buf,
    tab: Buf,
    prev: Buf,
    out: Buf,
    /// VOLTA_DRAFT_VERIFY: the whole-vocabulary chain's prev / out next to the split one's
    prev2: Buf,
    out2: Buf,
    /// vmarkov_argmax2's per-block (value, index) pairs [TMAX * 160 each] and per-session tickets [TMAX]
    am: Buf,
    emb: Buf,
    bias: Buf,
    /// one CUDA graph per draft-token count (observed rows are always padded to TMAX)
    graphs: HashMap<usize, Option<crate::gpu::GraphExec>>,
}

/// window pool rows per slot of a draft layer: the main model's window, then the block's own rows
fn spark_rows(win: usize, block: usize) -> usize {
    win + block
}

/// this GPU's 1/ngpu slice of a shared expert (gate+up rows, down columns) in the vmoe 4-bit layout
fn shared_slice(g: &Gpu, gate: &Mat, up: &Mat, down: &Mat, inter: usize, d: usize, ng: usize, gi: usize) -> Result<(VMat, VMat)> {
    let sl = inter / ng;
    let r0 = gi * sl;
    let mut gu = vec![0f32; 2 * sl * d];
    let mut row = vec![0f32; d];
    for (mi, mt) in [gate, up].into_iter().enumerate() {
        for r in 0..sl {
            mt.dequant(r0 + r, &mut row);
            gu[(mi * sl + r) * d..(mi * sl + r + 1) * d].copy_from_slice(&row);
        }
    }
    let mut dn = vec![0f32; d * sl];
    let mut drow = vec![0f32; inter];
    for r in 0..d {
        down.dequant(r, &mut drow);
        dn[r * sl..(r + 1) * sl].copy_from_slice(&drow[r0..r0 + sl]);
    }
    Ok((upload_v4_f32(g, 2 * sl, d, &gu, false)?, upload_v4_f32(g, d, sl, &dn, false)?))
}

/// the draft layers (appended to the main ones), their expert lists, and the rest of the draft
#[allow(clippy::too_many_arguments)]
/// heads: head-parallel (the Markov head split by vocabulary, as the main head)
#[allow(clippy::too_many_arguments)]
pub(super) fn build(g: &Gpu, m: &Model, ds: &Dspark, gi: usize, ngpu: usize, etp: usize, slots: usize, luts: u64, heads: bool) -> Result<(Vec<VLayer>, Vec<LocalLayer>, Spark)> {
    let c = &m.cfg;
    ensure!(etp == ngpu, "the draft's experts are split over every GPU: VOLTA_EXPERT_TP must be {ngpu}");
    ensure!(ds.layers.iter().all(|l| l.experts.len() % 32 == 0), "draft experts per layer must be a multiple of 32");
    let (d, base, nd) = (c.dim, c.n_layers, ds.layers.len());
    let lin = |w: &Mat| -> Result<Lin> {
        ensure!(v4_ok(w), "draft matrix {}x{} is not vmoe 4-bit friendly", w.rows, w.cols);
        Ok(Lin::V(upload_v4(g, w, 1)?))
    };
    let rows = spark_rows(c.window, ds.block);
    let mut layers = vec![];
    for dl in &ds.layers {
        layers.push(VLayer {
            qkv_a: None,
            q_a: Some(lin(&dl.q_a)?),
            q_a_norm: f32buf(g, &dl.q_a_norm)?,
            q_b: lin(&dl.q_b)?,
            kv: Some(lin(&dl.kv)?),
            kv_norm: f32buf(g, &dl.kv_norm)?,
            o_a: Lin::V(upload_v4(g, &dl.o_a, c.o_groups)?),
            o_b: lin(&dl.o_b)?,
            sinks: f32buf(g, &dl.sinks)?,
            attn_norm: f32buf(g, &dl.attn_norm)?,
            ffn_norm: f32buf(g, &dl.ffn_norm)?,
            comp_kv: None,
            comp_gate: None,
            comp_norm: None,
            idx_q_b: None,
            idx_proj: None,
            idx_k: None,
            idx_k_norm: None,
            hc_attn: (f32buf(g, &dl.hc_attn.0)?, f32buf(g, &dl.hc_attn.1)?, f32buf(g, &dl.hc_attn.2)?),
            hc_ffn: (f32buf(g, &dl.hc_ffn.0)?, f32buf(g, &dl.hc_ffn.1)?, f32buf(g, &dl.hc_ffn.2)?),
            gate_w: super::router_buf(g, &dl.gate_w)?.0,
            gate_bf: super::router_buf_bf(&dl.gate_w),
            gate_bias: f32buf(g, &dl.gate_bias)?,
            n_exp: dl.experts.len(),
            top_k: ds.top_k,
            sh_gate: None,
            sh_up: None,
            sh_down: None,
            sh_slice: Some(shared_slice(g, &dl.sh.0, &dl.sh.1, &dl.sh.2, c.inter, d, ngpu, gi)?),
            engram_kv: None,
            engram_qk: None,
            window: g.alloc_zeroed(slots * rows * c.head_dim * 2)?,
            comp_cache: None,
            idx_cache: None,
            comp_state: None,
        });
    }
    // experts: cb2 (the draft only has to guess: cb2 accepts as many drafts as cb3 and is ~0.2 GB smaller per GPU;
    // VOLTA_SPARK_CB2=0 for cb3), this GPU's slice of each
    let tier = if std::env::var("VOLTA_SPARK_CB2").is_ok_and(|v| v == "0") { 1u8 } else { 2u8 };
    let items: Vec<((usize, usize), &crate::model::Expert, u8)> =
        ds.layers.iter().enumerate().flat_map(|(j, dl)| dl.experts.iter().enumerate().map(move |(e, ex)| ((base + j, e), ex, tier))).collect();
    let bits = tier_kind(tier).1;
    let sl = c.inter / etp;
    let (wtot, stot) = (items.len() * 3 * sl * d * bits / 32, items.len() * 3 * sl * d / 32);
    let (wbuf, sbuf) = (g.alloc(wtot * 4)?, g.alloc(stot)?);
    let mut tab = HashMap::new();
    let (wo, so) = pack_lut(g, &items, d, c.inter, etp, gi % etp, luts, &wbuf, &sbuf, &mut tab)?;
    ensure!(wo == wtot && so == stot, "draft expert store size mismatch");
    let exps = Experts { w: wbuf, s: sbuf, tab };
    let mut local = vec![];
    for (j, dl) in ds.layers.iter().enumerate() {
        let es: Vec<usize> = (0..dl.experts.len()).collect();
        local.push(local_layer(g, &exps, base + j, &es, dl.experts.len())?);
    }
    // the Markov head: embedding rows as fp16, the projection as vmoe 4-bit
    let rank = ds.markov_embed.cols;
    let mut emb = vec![half::f16::ZERO; ds.markov_embed.rows * rank];
    let mut row = vec![0f32; rank];
    for r in 0..ds.markov_embed.rows {
        ds.markov_embed.dequant(r, &mut row);
        for (o, &v) in emb[r * rank..(r + 1) * rank].iter_mut().zip(&row) {
            *o = half::f16::from_f32(v);
        }
    }
    let nt = ds.targets.len();
    let spark = Spark {
        base,
        n: nd,
        block: ds.block,
        noise: ds.noise,
        main_proj: lin(&ds.main_proj)?,
        main_norm: f32buf(g, &ds.main_norm)?,
        norm: f32buf(g, &ds.norm)?,
        markov_embed: g.upload(&emb)?,
        markov_head: lin(&ds.markov_head)?,
        markov_head_hp: if heads && m.cfg.vocab % (32 * ngpu) == 0 && super::v4_ok(&ds.markov_head) {
            let vs = m.cfg.vocab / ngpu;
            Some(super::upload_v4_sub(g, &ds.markov_head, gi * vs, vs, 0, ds.markov_head.cols)?)
        } else {
            None
        },
        rank,
        mh: g.alloc_zeroed(TMAX * nt * d * 4)?,
        targets: ds.targets.clone(),
        _exps: exps,
        obs: g.alloc_zeroed(TMAX * nt * d * 4)?,
        widx: g.alloc_zeroed(TMAX * rows * 4)?,
        tab: g.alloc_zeroed(6 * TMAX * 4)?,
        prev: g.alloc_zeroed(TMAX * 4)?,
        out: g.alloc_zeroed(TMAX * 4)?,
        prev2: g.alloc_zeroed(TMAX * 4)?,
        am: g.alloc_zeroed((2 * TMAX * 160 + TMAX) * 4)?,
        out2: g.alloc_zeroed(TMAX * 4)?,
        emb: g.alloc_zeroed(TMAX * rank * 4)?,
        bias: g.alloc_zeroed((TMAX / ds.block + 1) * c.vocab * 4)?, // one row per drafting session
        graphs: HashMap::new(),
    };
    Ok((layers, local, spark))
}

impl Shard<'_> {
    /// record the main model's hidden-state mean at target layer `li` (start of the layer) for the draft
    pub(super) fn spark_capture(&self, li: usize, t: usize) -> Result<()> {
        let Some(sp) = &self.spark else { return Ok(()) };
        let Some(j) = sp.targets.iter().position(|&x| x == li) else { return Ok(()) };
        let c = &self.m.cfg;
        self.g.launch(self.k["vmean_hc"], ((c.dim as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(self.w.h.ptr), Arg::P(sp.mh.ptr), Arg::I(c.dim as i32), Arg::I(c.hc as i32), Arg::I(j as i32), Arg::I(sp.targets.len() as i32)])
    }

    /// One draft step on this GPU (every GPU runs it together: the draft's MoE exchanges like the main model's).
    /// `obs`: (slot, row of the last main step, position) of the main tokens to record in the draft windows;
    /// `drafts`: (slot, position of the newest main token, the next token) per session to draft for. Returns the
    /// block's guesses, `block` per draft (the tokens after `next`).
    pub(super) fn spark_step(&mut self, obs: &[(usize, usize, usize)], drafts: &[(usize, usize, u32)], counts: &[usize]) -> Result<Vec<u32>> {
        let m = self.m;
        let c = &m.cfg;
        let (d, hc, win) = (c.dim, c.hc, c.window);
        self.g.bind();
        let (b, nd, noise) = {
            let sp = self.spark.as_ref().context("no draft loaded (VOLTA_SPARK)")?;
            (sp.block, sp.n, sp.noise)
        };
        let rows = spark_rows(win, b);
        let t = drafts.len() * b;
        ensure!(t <= TMAX && obs.len() <= TMAX, "draft step: {t} draft tokens, {} observed rows (at most {TMAX})", obs.len());
        // exchange counts and sequence numbers, as a main step does
        self.exchange_counts(counts)?;
        let sbase = self.seq + 1;
        // the draft layers' exchanges, then (head-parallel) one per Markov position
        self.seq += (nd + b) as u32;
        self.g.htod_sync(self.w.seqd.ptr, &[sbase])?;
        let sp = self.spark.as_ref().unwrap();
        let (tab, widx_p) = (sp.tab.ptr, sp.widx.ptr);
        // tables: [0] draft positions, [1] draft kv rows, [2] observed rows' source rows, [3] their positions, [4] their kv rows
        let mut tabs = vec![0i32; 5 * TMAX];
        let mut widx = vec![-1i32; t * rows];
        for (si, &(s, p, _)) in drafts.iter().enumerate() {
            for i in 0..b {
                let r = si * b + i;
                tabs[r] = (p + 1 + i) as i32;
                tabs[TMAX + r] = (s * rows + win + i) as i32;
                // the window's main positions p - win + 1 ..= p (those that exist), then the block's own rows
                for j in 0..win {
                    let q = (p + j + 1) as isize - win as isize;
                    widx[r * rows + j] = if q < 0 { -1 } else { (s * rows + q as usize % win) as i32 };
                }
                for k in 0..b {
                    widx[r * rows + win + k] = (s * rows + win + k) as i32;
                }
            }
        }
        // observed rows, padded to TMAX (row 0, no store) so every step has the same shape
        for i in 0..TMAX {
            tabs[4 * TMAX + i] = -1;
        }
        for (i, &(s, row, p)) in obs.iter().enumerate() {
            tabs[2 * TMAX + i] = row as i32;
            tabs[3 * TMAX + i] = p as i32;
            tabs[4 * TMAX + i] = (s * rows + p % win) as i32;
        }
        self.g.htod_sync(tab, &tabs)?;
        if t > 0 {
            self.g.htod_sync(widx_p, &widx)?;
        }
        // the block: `next`, then the noise token, at positions p + 1 ..
        if t > 0 {
            let mut hbuf = vec![0f32; t * hc * d];
            let mut e = vec![0f32; d];
            for (si, &(_, _, next)) in drafts.iter().enumerate() {
                for i in 0..b {
                    m.embed.dequant(if i == 0 { next } else { noise } as usize, &mut e);
                    for k in 0..hc {
                        hbuf[((si * b + i) * hc + k) * d..((si * b + i) * hc + k + 1) * d].copy_from_slice(&e);
                    }
                }
            }
            self.g.htod_sync(self.w.h.ptr, &hbuf)?;
            let mut pre = vec![0f32; t * 24];
            for i in 0..t {
                pre[i * 24] = 1.0;
            }
            self.g.htod_sync(self.w.mix_f.ptr, &pre)?;
        }
        if t > 0 {
            let sp = self.spark.as_ref().unwrap();
            let prev: Vec<i32> = drafts.iter().map(|&(_, _, n)| n as i32).collect();
            self.g.htod_sync(sp.prev.ptr, &prev)?;
        }
        // the device part: eager the first time a shape comes up, captured the second, replayed after
        let mut graphs = std::mem::take(&mut self.spark.as_mut().unwrap().graphs);
        let eager = std::env::var("VOLTA_GRAPH").is_ok_and(|v| v == "0");
        let r = match graphs.get(&t) {
            Some(Some(gx)) if !eager => self.g.launch_graph(gx),
            Some(None) if !eager && !self.graph_full => (|| {
                self.g.begin_capture()?;
                self.spark_device(t, drafts.len(), tab, widx_p, rows)?;
                match self.g.end_capture() {
                    Ok(gx) => {
                        self.g.launch_graph(&gx)?;
                        graphs.insert(t, Some(gx));
                    }
                    Err(e) => {
                        eprintln!("GPU{}: no more draft graphs ({e:#})", self.gi);
                        self.graph_full = true;
                        self.spark_device(t, drafts.len(), tab, widx_p, rows)?;
                    }
                }
                Ok(())
            })(),
            _ => {
                let r = self.spark_device(t, drafts.len(), tab, widx_p, rows);
                graphs.entry(t).or_insert(None);
                r
            }
        };
        self.spark.as_mut().unwrap().graphs = graphs;
        r?;
        if t == 0 {
            self.g.sync()?;
            return Ok(vec![]);
        }
        let sp = self.spark.as_ref().unwrap();
        let out: Vec<i32> = self.g.download(sp.out.ptr, t)?;
        if std::env::var_os("VOLTA_DRAFT_CHECK").is_some() {
            // debugging: every GPU's drafts (head-parallel: they must agree)
            eprintln!("GPU{} drafts {:?}", self.gi, &out);
        }
        if std::env::var_os("VOLTA_DRAFT_VERIFY").is_some() && self.gi == 0 {
            let out2: Vec<i32> = self.g.download(sp.out2.ptr, t)?;
            eprintln!("draft verify split {:?} whole {:?}", &out, &out2);
        }
        Ok(out.into_iter().map(|x| x as u32).collect())
    }

    /// the draft step's device work (graph-captured): observe TMAX rows, the draft layers, head and Markov chain
    fn spark_device(&mut self, t: usize, ns: usize, tab: u64, widx_p: u64, rows: usize) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let (d, hd) = (c.dim, c.head_dim);
        let (b, base, nd, rank) = {
            let sp = self.spark.as_ref().unwrap();
            (sp.block, sp.base, sp.n, sp.rank)
        };
        let ofs = |k: usize| tab + (k * TMAX * 4) as u64;
        let f = self.freqs.0.ptr;
        // observe: main hidden rows -> main_proj -> norm, then each draft layer's kv of them into its window
        {
            let sp = self.spark.as_ref().unwrap();
            let (no, k3) = (TMAX, sp.targets.len() * d);
            let w = &self.w;
            self.g.launch(self.k["vgather_rows"], ((k3 as u32).div_ceil(256), no as u32, 1), 256, 0, &[Arg::P(sp.mh.ptr), Arg::P(ofs(2)), Arg::P(sp.obs.ptr), Arg::I(k3 as i32)])?;
            self.mm(&sp.main_proj, sp.obs.ptr, w.a.ptr, no)?;
            self.rms_rows(w.a.ptr, &sp.main_norm, w.a.ptr, d, no, true)?;
            for j in 0..nd {
                let l = &self.layers[base + j];
                self.mm(l.kv.as_ref().unwrap(), w.a.ptr, w.kvv.ptr, no)?;
                self.g.launch(self.k["vkv_store"], (no as u32, 1, 1), 1024, 0, &[Arg::P(w.kvv.ptr), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(f), Arg::I(c.rope_dim as i32), Arg::P(ofs(3)), Arg::P(l.window.ptr), Arg::P(ofs(4)), Arg::I(hd as i32)])?;
            }
        }
        for j in 0..nd {
            let li = base + j;
            if t > 0 {
                let w = &self.w;
                let l = &self.layers[li];
                self.g.launch(self.k["vhc_pre"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.mix_f.ptr), Arg::I(0), Arg::P(w.x.ptr), Arg::I(d as i32)])?;
                self.hc_mixes_rows(&l.hc_attn, w.h.ptr, w.mix_a.ptr, t)?;
                self.rms_rows(w.x.ptr, &l.attn_norm, w.x.ptr, d, t, true)?;
                self.spark_attention(li, t, ofs(0), ofs(1), widx_p, rows)?;
                let w = &self.w;
                let l = &self.layers[li];
                self.g.launch(self.k["vhc_post"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.a.ptr), Arg::P(w.mix_a.ptr), Arg::I(d as i32)])?;
                self.g.launch(self.k["vhc_pre"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.mix_a.ptr), Arg::I(0), Arg::P(w.x.ptr), Arg::I(d as i32)])?;
                self.hc_mixes_rows(&l.hc_ffn, w.h.ptr, w.mix_f.ptr, t)?;
                self.rms_rows(w.x.ptr, &l.ffn_norm, w.x.ptr, d, t, true)?;
                self.moe_dispatch(t, li)?;
            }
            self.vx_gather(j)?;
            // every GPU's draft token count (head-parallel: this one's, which picks the 1-token shared-slice kernel)
            let counts = vec![t; self.ngpu];
            self.moe_experts(&counts, li)?;
            self.vx_scatter(j, t)?;
            if t > 0 {
                self.moe_combine(t)?;
                let w = &self.w;
                self.g.launch(self.k["vhc_post"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.a.ptr), Arg::P(w.mix_f.ptr), Arg::I(d as i32)])?;
            }
        }
        if t == 0 {
            return Ok(());
        }
        // head over every block position, then the Markov chain one position at a time
        let sp = self.spark.as_ref().unwrap();
        let w = &self.w;
        self.g.launch(self.k["vhc_pre"], ((d as u32).div_ceil(256), t as u32, 1), 256, 0, &[Arg::P(w.h.ptr), Arg::P(w.mix_f.ptr), Arg::I(0), Arg::P(w.x.ptr), Arg::I(d as i32)])?;
        self.rms_rows(w.x.ptr, &sp.norm, w.x.ptr, d, t, false)?;
        let split = std::env::var("VOLTA_DRAFT_SPLIT").map_or(true, |v| v != "0");
        if let (true, Some(hh), Some(mh)) = (split, &self.head_hp, &sp.markov_head_hp) {
            // head-parallel: this GPU's slice of the vocabulary for the head and the Markov bias; each position's
            // best per GPU, then the best of the 8 on every GPU (vx_llargmax), which the next position embeds
            let (vs, pair) = (hh.rows, self.spark_ids.ptr);
            if std::env::var_os("VOLTA_DRAFT_VERIFY").is_some() {
                // debugging (eager): the whole-vocabulary chain from the same prev into prev2 / out2 first
                self.g.dtod(sp.prev2.ptr, sp.prev.ptr, ns * 4)?;
                self.mm(&self.head, w.x.ptr, w.logits.ptr, t)?;
                for i in 0..b {
                    self.g.launch(self.k["vmarkov_embed"], (ns as u32, 1, 1), 256, 0, &[Arg::P(sp.markov_embed.ptr), Arg::P(sp.prev2.ptr), Arg::P(sp.emb.ptr), Arg::I(rank as i32)])?;
                    self.mm(&sp.markov_head, sp.emb.ptr, sp.bias.ptr, ns)?;
                    self.g.launch(self.k["vmarkov_argmax"], (ns as u32, 1, 1), 1024, 0, &[Arg::P(w.logits.ptr), Arg::P(sp.bias.ptr), Arg::I(c.vocab as i32), Arg::I(b as i32), Arg::I(i as i32), Arg::P(sp.prev2.ptr), Arg::P(sp.out2.ptr)])?;
                }
            }
            self.vmm(hh, w.x.ptr, d, w.logits.ptr, t)?;
            for i in 0..b {
                self.g.launch(self.k["vmarkov_embed"], (ns as u32, 1, 1), 256, 0, &[Arg::P(sp.markov_embed.ptr), Arg::P(sp.prev.ptr), Arg::P(sp.emb.ptr), Arg::I(rank as i32)])?;
                self.vmm(mh, sp.emb.ptr, rank, sp.bias.ptr, ns)?;
                self.markov_argmax(vs, b, i, ns, self.gi * vs, pair)?;
                self.g.launch(self.k["vx_llargmax"], (1, 1, 1), 256, 0, &[Arg::P(w.ll_t.ptr + 128), Arg::I(self.gi as i32), Arg::P(pair), Arg::I((2 * ns) as i32), Arg::I((2 * TMAX) as i32), Arg::P(sp.prev.ptr), Arg::P(sp.out.ptr), Arg::I(b as i32), Arg::I(i as i32), Arg::P(w.seqd.ptr), Arg::I((nd + i) as i32)])?;
            }
            return Ok(());
        }
        self.mm(&self.head, w.x.ptr, w.logits.ptr, t)?;
        if std::env::var_os("VOLTA_DRAFT_VERIFY").is_some() {
            // debugging: the single-block argmax's chain from the same prev into prev2 / out2 first
            self.g.dtod(sp.prev2.ptr, sp.prev.ptr, ns * 4)?;
            for i in 0..b {
                self.g.launch(self.k["vmarkov_embed"], (ns as u32, 1, 1), 256, 0, &[Arg::P(sp.markov_embed.ptr), Arg::P(sp.prev2.ptr), Arg::P(sp.emb.ptr), Arg::I(rank as i32)])?;
                self.mm(&sp.markov_head, sp.emb.ptr, sp.bias.ptr, ns)?;
                self.g.launch(self.k["vmarkov_argmax"], (ns as u32, 1, 1), 1024, 0, &[Arg::P(w.logits.ptr), Arg::P(sp.bias.ptr), Arg::I(c.vocab as i32), Arg::I(b as i32), Arg::I(i as i32), Arg::P(sp.prev2.ptr), Arg::P(sp.out2.ptr)])?;
            }
        }
        for i in 0..b {
            self.g.launch(self.k["vmarkov_embed"], (ns as u32, 1, 1), 256, 0, &[Arg::P(sp.markov_embed.ptr), Arg::P(sp.prev.ptr), Arg::P(sp.emb.ptr), Arg::I(rank as i32)])?;
            self.mm(&sp.markov_head, sp.emb.ptr, sp.bias.ptr, ns)?;
            self.markov_argmax(c.vocab, b, i, ns, 0, 0)?;
        }
        Ok(())
    }

    /// the Markov step's argmax for position i of ns sessions over `vocab` logits + bias (vmarkov_argmax2): into
    /// prev / out, or (pair != 0) as (value, id + base) pairs for vx_llargmax
    fn markov_argmax(&self, vocab: usize, b: usize, i: usize, ns: usize, base: usize, pair: u64) -> Result<()> {
        let (sp, w) = (self.spark.as_ref().unwrap(), &self.w);
        // ~160 blocks over all the sessions, at least 2K values each (Scout's lab: S = 1 5.1 us, S = 8 13.8 us, cold)
        let nb = (vocab / 2048).clamp(1, 160usize.div_ceil(ns.max(1)));
        let (pv, pi, tick) = (sp.am.ptr, sp.am.ptr + (TMAX * 160 * 4) as u64, sp.am.ptr + (2 * TMAX * 160 * 4) as u64);
        self.g.launch(self.k["vmarkov_argmax2"], (nb as u32, ns as u32, 1), 256, 0, &[Arg::P(w.logits.ptr), Arg::P(sp.bias.ptr), Arg::I(vocab as i32), Arg::I(b as i32), Arg::I(i as i32), Arg::P(sp.prev.ptr), Arg::P(sp.out.ptr), Arg::P(pv), Arg::P(pi), Arg::P(tick), Arg::I(base as i32), Arg::P(pair)])
    }

    /// the draft layer's attention for the block's tokens (positions `pos`, kv rows `kvrow`, `rows` window indices
    /// per token in `widx`); leaves the output in w.a
    fn spark_attention(&self, li: usize, t: usize, pos: u64, kvrow: u64, widx: u64, rows: usize) -> Result<()> {
        let c = &self.m.cfg;
        let (hd, nh) = (c.head_dim, c.n_heads);
        let w = &self.w;
        let l = &self.layers[li];
        self.mm(l.q_a.as_ref().unwrap(), w.x.ptr, w.qr.ptr, t)?;
        self.rms_rows(w.qr.ptr, &l.q_a_norm, w.qr.ptr, c.q_lora, t, true)?;
        self.mm(&l.q_b, w.qr.ptr, w.q.ptr, t)?;
        self.rope_rows(w.q.ptr, nh, hd, false, pos, t, false)?;
        self.mm(l.kv.as_ref().unwrap(), w.x.ptr, w.kvv.ptr, t)?;
        self.g.launch(self.k["vkv_store"], (t as u32, 1, 1), 1024, 0, &[Arg::P(w.kvv.ptr), Arg::P(l.kv_norm.ptr), Arg::I(hd as i32), Arg::F(c.eps), Arg::P(self.freqs.0.ptr), Arg::I(c.rope_dim as i32), Arg::P(pos), Arg::P(l.window.ptr), Arg::P(kvrow), Arg::I(hd as i32)])?;
        let scale = 1.0 / (hd as f32).sqrt();
        let s = self.attn_tiles(t, nh, l.window.ptr, widx, rows, 0, 0, 0, scale, None, None)?;
        self.g.launch(self.k["sparse_attn_merge"], (t as u32, nh as u32, 1), 256, 0, &[Arg::P(w.attn_part.ptr), Arg::I(s as i32), Arg::P(l.sinks.ptr), Arg::P(w.o.ptr), Arg::I(nh as i32), Arg::I(hd as i32)])?;
        self.rope_rows(w.o.ptr, nh, hd, false, pos, t, true)?;
        self.g.launch(self.k["fake_fp8"], (((t * nh * hd) as u32 / 32).div_ceil(8), 1, 1), 256, 0, &[Arg::P(w.o.ptr), Arg::I((t * nh * hd) as i32)])?;
        let Lin::V(oa) = &l.o_a else { bail!("draft o_a is vmoe 4-bit") };
        self.vmm(oa, w.o.ptr, nh * hd, w.r.ptr, t)?;
        self.g.launch(self.k["round_bf16"], (((t * c.o_groups * c.o_lora) as u32).div_ceil(256), 1, 1), 256, 0, &[Arg::P(w.r.ptr), Arg::I((t * c.o_groups * c.o_lora) as i32)])?;
        self.mm(&l.o_b, w.r.ptr, w.a.ptr, t)
    }

    /// drop a slot's last `n` tokens (drafts the main model rejected): their rows stay in the pools but are
    /// overwritten before anything reads them again
    pub(super) fn rewind(&mut self, s: usize, n: usize) -> Result<()> {
        let c = &self.m.cfg;
        let ss = &mut self.sess[s];
        ensure!(n <= ss.pos && n <= ss.engram_hist.len(), "rewind {n} of slot {s} at {}", ss.pos);
        ss.pos -= n;
        let keep = ss.engram_hist.len() - n;
        ss.engram_hist.truncate(keep);
        for li in 0..c.n_layers {
            let r = self.m.layers[li].ratio;
            if r > 0 && c.kv_sources.contains(&li) {
                ss.n_comp[li] = ss.pos / r;
                ss.n_idx[li] = ss.pos / r;
            }
        }
        Ok(())
    }
}
