//! dsx-volta: batched decode on 8x V100 (volta/DESIGN.md, phase 1).
//!   check: a prompt through the batched engine (several copies at once) next to the CPU reference
//!   bench: S sessions per GPU decoding together, aggregate tok/s

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dsx::model::{Model, RefState};
use dsx::volta::{Engine, Opts, SparkPlan, TMAX};
use std::path::PathBuf;
use std::time::Instant;

/// VOLTA_HEADS=1: run eval / spec / bench head-parallel (every GPU runs every session; one logical GPU)
fn heads() -> bool {
    std::env::var("VOLTA_HEADS").is_ok_and(|v| v == "1")
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Feed `ids` to one session on every GPU (copies with different lengths of a filler prefix, so the batch mixes
    /// positions), then compare the first copy's logits with the CPU reference at each position.
    Check {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<u32>,
        #[arg(long, default_value_t = 8)]
        gpus: usize,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
        /// skip the CPU reference (GPU only)
        #[arg(long)]
        no_ref: bool,
    },
    /// Lucebox's 1017 eval (8 x 1024 tokens, teacher top-256): sequence i on GPU i % gpus, all decoded together
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        #[arg(long)]
        eval: PathBuf,
        #[arg(long, default_value_t = 8)]
        seqs: usize,
        #[arg(long, default_value_t = 1024)]
        len: usize,
        #[arg(long, default_value_t = 256)]
        topk: usize,
        #[arg(long, default_value_t = 8)]
        gpus: usize,
        /// tokens per sequence per step (a chunk of the sequence at once, as prefill does); 1 = token by token
        #[arg(long, default_value_t = 1)]
        chunk: usize,
    },
    /// greedy decoding with the DSpark draft (VOLTA_SPARK=<checkpoint dir>): `sessions` per GPU prefill `prompt`
    /// tokens of `text` each, then generate `steps` passes of draft + verify
    Spec {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 8)]
        gpus: usize,
        #[arg(long, default_value_t = 8)]
        sessions: usize,
        /// total sessions, spread round-robin over the GPUs (instead of `sessions` per GPU)
        #[arg(long)]
        total: Option<usize>,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        prompt: usize,
        #[arg(long)]
        text: PathBuf,
        /// draft tokens verified per pass (at most the draft's block)
        #[arg(long, default_value_t = 5)]
        k: usize,
        /// per session, verify the most drafts j (<= k) whose chance of all being accepted, p^j for the session's
        /// recent per-draft acceptance p, is at least VOLTA_SPEC_THETA (default 0.35)
        #[arg(long)]
        adaptive: bool,
    },
    /// expert path self-test: convert + pack + GPU kernels vs the CPU reference, for (layer:expert) pairs
    Experts {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        pairs: Vec<String>,
    },
    /// `sessions` per GPU decode `steps` tokens together (token ids are arbitrary)
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        engram: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        gpus: usize,
        #[arg(long, default_value_t = 8)]
        sessions: usize,
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
        /// feed real text (teacher forcing) from this token file (i32 LE, e.g. the eval's heldout.i32), session i
        /// starting at offset i * 97, so routing is as varied as real traffic; otherwise greedy from arbitrary tokens
        #[arg(long)]
        text: Option<PathBuf>,
    },
}

fn load(model: &PathBuf, engram: Option<&std::path::Path>) -> Result<&'static Model> {
    let t0 = Instant::now();
    // a directory is DeepSeek's HF checkpoint (FP8 dense, MXFP4 experts, Engram inside)
    let m = if model.is_dir() { Model::load_hf(model)? } else { Model::load(model, engram)? };
    eprintln!("model mapped in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(Box::leak(Box::new(m)))
}

fn lsm(l: &[f32]) -> Vec<f64> {
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = mx + l.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
    l.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(l: &[f32]) -> usize {
    l.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Check { model, engram, ids, gpus, ctx, no_ref } => {
            let m = load(&model, engram.as_deref())?;
            let t0 = Instant::now();
            let e = Engine::new(m, Opts { ngpu: gpus, slots: 2, max_ctx: ctx, heads: false })?;
            eprintln!("engine up in {:.0} s", t0.elapsed().as_secs_f64());
            // GPU g's slot 0 gets the prompt after g filler tokens, slot 1 the prompt itself (two sessions per GPU)
            let filler = 1000u32;
            let n = ids.len();
            let mut gpu_logits: Vec<Vec<f32>> = vec![];
            let steps = n + gpus - 1;
            let ts = Instant::now();
            for s in 0..steps {
                let mut plan = vec![vec![]; gpus];
                for (g, p) in plan.iter_mut().enumerate() {
                    if s < g {
                        p.push((0, filler));
                    } else if s - g < n {
                        p.push((0, ids[s - g]));
                    }
                    if s < n {
                        p.push((1, ids[s]));
                    }
                }
                let out = e.step(plan.clone())?;
                // slot 1 on GPU0 is the reference copy (no filler): its row is the last token of GPU0's plan
                if s < n {
                    let row = plan[0].len() - 1;
                    gpu_logits.push(out[0][row * e.vocab..(row + 1) * e.vocab].to_vec());
                }
                // copies agree with each other: GPU g slot 0 at prompt position s - g vs GPU0 slot 1 at the same position
                for g in 1..gpus {
                    if s >= g && s - g < n {
                        let lg = &out[g][0..e.vocab];
                        let r = &gpu_logits[s - g];
                        let (a, b) = (argmax(lg), argmax(r));
                        if a != b {
                            eprintln!("note: GPU{g} copy at pos {} picks {a}, GPU0 picks {b}", s - g);
                        }
                    }
                }
            }
            let el = ts.elapsed().as_secs_f64();
            eprintln!("{steps} steps in {el:.1} s ({:.0} ms/step)", el * 1e3 / steps as f64);
            if no_ref {
                for (p, l) in gpu_logits.iter().enumerate() {
                    println!("pos {p}: top1 {}", argmax(l));
                }
                return Ok(());
            }
            let mut st = RefState::new(m);
            let (mut kl_sum, mut agree) = (0.0, 0);
            for (p, &tk) in ids.iter().enumerate() {
                let tr = Instant::now();
                let rl = m.ref_step(&mut st, tk)?;
                let (lr, lg) = (lsm(&rl), lsm(&gpu_logits[p]));
                // KL(ref || gpu) over the ref's top-256
                let mut idx: Vec<usize> = (0..lr.len()).collect();
                idx.sort_by(|&a, &b| lr[b].partial_cmp(&lr[a]).unwrap());
                let kl: f64 = idx[..256].iter().map(|&i| lr[i].exp() * (lr[i] - lg[i])).sum();
                let (a, b) = (argmax(&rl), argmax(&gpu_logits[p]));
                agree += (a == b) as usize;
                kl_sum += kl;
                println!("pos {p}: ref top1 {a} gpu top1 {b} KL {kl:.4} (ref {:.1} s)", tr.elapsed().as_secs_f64());
            }
            println!("mean KL {:.4}, top-1 agreement {agree}/{n}", kl_sum / n as f64);
        }
        Cmd::Eval { model, engram, eval, seqs, len, topk, gpus, chunk } => {
            let rd = |f: &str| std::fs::read(eval.join(f));
            let toks: Vec<u32> = rd("heldout.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
            let tidx: Vec<i32> = rd("teacher_idx.i32")?.chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
            let tlp: Vec<f32> = rd("teacher_lp.f32")?.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let m = load(&model, engram.as_deref())?;
            let (pg, gpus) = (gpus, if heads() { 1 } else { gpus });
            let slots = seqs.div_ceil(gpus);
            let e = Engine::new(m, Opts { ngpu: pg, slots, max_ctx: len + 64, heads: heads() })?;
            let (mut nll, mut kl, mut agree, mut cnt) = (0f64, 0f64, 0usize, 0usize);
            let t0 = Instant::now();
            // full eval length (1024) uses the teacher rows as laid out: row = seq * 1023 + pos
            let full = 1024;
            for p0 in (0..len - 1).step_by(chunk) {
                let p1 = (p0 + chunk).min(len - 1);
                let mut plan = vec![vec![]; gpus];
                for sq in 0..seqs {
                    for p in p0..p1 {
                        plan[sq % gpus].push((sq / gpus, toks[sq * full + p]));
                    }
                }
                let out = e.step(plan.clone())?;
                for (sq, p) in (0..seqs).flat_map(|sq| (p0..p1).map(move |p| (sq, p))) {
                    // row of (sequence, position) in its GPU's plan: that GPU's earlier sequences, then the offset
                    let g = sq % gpus;
                    let row_in = (0..sq).filter(|&o| o % gpus == g).count() * (p1 - p0) + (p - p0);
                    let l = &out[g][row_in * e.vocab..(row_in + 1) * e.vocab];
                    let lp = lsm(l);
                    nll -= lp[toks[sq * full + p + 1] as usize];
                    let row = sq * (full - 1) + p;
                    let (ti, tl) = (&tidx[row * topk..(row + 1) * topk], &tlp[row * topk..(row + 1) * topk]);
                    kl += ti.iter().zip(tl).map(|(&i, &lt)| (lt as f64).exp() * (lt as f64 - lp[i as usize])).sum::<f64>();
                    agree += (argmax(l) as i32 == ti[0]) as usize;
                    cnt += 1;
                }
                if p1 % 128 < chunk {
                    let p = p1 - 1;
                    eprintln!("pos {}: running ppl {:.3} kl {:.4} top1 {:.4} ({:.0} s)", p + 1, (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
                }
            }
            println!("{{\"tokens\": {cnt}, \"ppl\": {:.4}, \"kl_top{topk}\": {:.6}, \"top1_agree\": {:.6}, \"wall_s\": {:.1}}}", (nll / cnt as f64).exp(), kl / cnt as f64, agree as f64 / cnt as f64, t0.elapsed().as_secs_f64());
        }
        Cmd::Spec { model, gpus, sessions, total, steps, prompt, text, k, adaptive } => {
            let m = load(&model, None)?;
            let (pg, gpus) = (gpus, if heads() { 1 } else { gpus });
            let corpus: Vec<u32> = std::fs::read(&text)?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
            // sessions per GPU: `sessions` each, or `total` spread round-robin (some GPUs may have none)
            let ns = total.unwrap_or(gpus * sessions);
            let nper: Vec<usize> = (0..gpus).map(|g| ns / gpus + usize::from(g < ns % gpus)).collect();
            let gid = |g: usize, s: usize| s * gpus + g; // global session index
            let smax = *nper.iter().max().unwrap();
            let e = Engine::new(m, Opts { ngpu: pg, slots: smax.max(1), max_ctx: prompt + steps * (k + 1) + 64, heads: heads() })?;
            let block = e.spark_block().context("VOLTA_SPARK is not set")?;
            anyhow::ensure!(k <= block && smax * (k + 1) <= TMAX, "k {k}: at most the block ({block}), and sessions per GPU * (k + 1) <= {TMAX}");
            // prefill: chunks of TMAX / sessions tokens per session; the draft observes every prompt token
            let chunk = (TMAX / smax).max(1);
            let mut next = vec![0u32; ns];
            let tp = Instant::now();
            let trace_pf = std::env::var("VOLTA_TRACE_PREFILL").ok().and_then(|v| v.parse::<usize>().ok());
            for p0 in (0..prompt).step_by(chunk) {
                let p1 = (p0 + chunk).min(prompt);
                let w = p1 - p0;
                // VOLTA_TRACE_PREFILL=pos: profile the prefill step that starts at or after pos
                let tr = trace_pf.is_some_and(|t| p0 >= t && p0 < t + chunk);
                if tr {
                    dsx::volta::profiler(true)?;
                }
                let cv = &corpus;
                let plan: Vec<Vec<(usize, u32)>> = (0..gpus).map(|g| (0..nper[g]).flat_map(|s| (p0..p1).map(move |p| (s, cv[(gid(g, s) * 997 + p) % cv.len()]))).collect()).collect();
                let out = e.step_greedy(plan)?;
                let obs = (0..gpus).map(|g| (0..nper[g]).flat_map(|s| (0..w).map(move |j| (s, s * w + j, p0 + j))).collect()).collect();
                e.spark(SparkPlan { obs, drafts: vec![vec![]; gpus] })?;
                if tr {
                    dsx::volta::profiler(false)?;
                }
                for g in 0..gpus {
                    for s in 0..nper[g] {
                        next[gid(g, s)] = out[g][s * w + w - 1];
                    }
                }
            }
            eprintln!("prefill: {ns} x {prompt} tokens in {:.1} s ({:.0} tok/s)", tp.elapsed().as_secs_f64(), (ns * prompt) as f64 / tp.elapsed().as_secs_f64());
            // generation: draft k tokens after `next`, verify [next, drafts] in one main step, keep the agreed prefix
            let mut pos = vec![prompt - 1; ns]; // position of each session's newest main token
            let mut obs: Vec<Vec<(usize, usize, usize)>> = vec![vec![]; gpus];
            let (mut made, mut hist) = (0usize, vec![0usize; k + 1]);
            // per session, the recent per-draft acceptance (exponential averages of accepted and tried drafts)
            let theta = std::env::var("VOLTA_SPEC_THETA").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.35);
            let (mut acc, mut tried) = (vec![0.6f64; ns], vec![1.0f64; ns]);
            let ks = |acc: &[f64], tried: &[f64], i: usize| {
                if !adaptive {
                    return k;
                }
                let p = (acc[i] / tried[i]).clamp(0.0, 0.99);
                (1..=k).take_while(|&j| p.powi(j as i32) >= theta).last().unwrap_or(1)
            };
            let mut verified = 0usize;
            let (mut t_draft, mut t_main) = (0f64, 0f64);
            let t0 = Instant::now();
            let trace_at = std::env::var("VOLTA_TRACE_STEP").ok().and_then(|v| v.parse::<usize>().ok());
            for it in 0..steps {
                // VOLTA_TRACE_STEP=n: profile pass n (draft + verify) for nsys --capture-range=cudaProfilerApi
                if trace_at == Some(it) {
                    dsx::volta::profiler(true)?;
                }
                let td = Instant::now();
                let drafts: Vec<Vec<(usize, usize, u32)>> = (0..gpus).map(|g| (0..nper[g]).map(|s| (s, pos[gid(g, s)], next[gid(g, s)])).collect()).collect();
                // k = 0: plain greedy decoding (the baseline), no draft step
                let dr = if k == 0 { (0..gpus).map(|g| vec![0u32; nper[g] * block]).collect() } else { e.spark(SparkPlan { obs: std::mem::take(&mut obs), drafts })? };
                t_draft += td.elapsed().as_secs_f64();
                let tm = Instant::now();
                // this pass's drafts per session, and where each session's rows start in its GPU's plan
                let mut kk: Vec<Vec<usize>> = (0..gpus).map(|g| (0..nper[g]).map(|s| ks(&acc, &tried, gid(g, s))).collect()).collect();
                if adaptive {
                    // every GPU verifies the same number of tokens, the largest GPU's rounded up to a multiple of 8:
                    // the step waits for its busiest GPU anyway, and few shapes keep it in CUDA graphs. The spare
                    // slots go to the sessions most likely to have one more draft accepted
                    let top = (0..gpus).map(|g| kk[g].iter().map(|x| x + 1).sum::<usize>()).max().unwrap_or(0);
                    let target = (top.div_ceil(8) * 8).min(TMAX);
                    for g in 0..gpus {
                        let tot: usize = kk[g].iter().map(|x| x + 1).sum();
                        let mut spare = target.saturating_sub(tot);
                        let mut order: Vec<usize> = (0..nper[g]).collect();
                        order.sort_by(|&a, &b| (acc[gid(g, b)] / tried[gid(g, b)]).partial_cmp(&(acc[gid(g, a)] / tried[gid(g, a)])).unwrap());
                        while spare > 0 && order.iter().any(|&s| kk[g][s] < k) {
                            for &s in &order {
                                if spare > 0 && kk[g][s] < k {
                                    kk[g][s] += 1;
                                    spare -= 1;
                                }
                            }
                        }
                    }
                }
                let r0: Vec<Vec<usize>> = kk.iter().map(|v| v.iter().scan(0, |a, &x| { let r = *a; *a += x + 1; Some(r) }).collect()).collect();
                let plan: Vec<Vec<(usize, u32)>> = (0..gpus).map(|g| (0..nper[g]).flat_map(|s| std::iter::once(next[gid(g, s)]).chain(dr[g][s * block..s * block + kk[g][s]].iter().copied()).map(move |t| (s, t))).collect()).collect();
                verified += plan.iter().map(|p| p.len()).sum::<usize>();
                let out = e.step_greedy(plan)?;
                t_main += tm.elapsed().as_secs_f64();
                obs = vec![vec![]; gpus];
                let mut rw = vec![vec![]; gpus];
                for g in 0..gpus {
                    for s in 0..nper[g] {
                        let (i, ksi, rs) = (gid(g, s), kk[g][s], r0[g][s]);
                        let row = |j: usize| out[g][rs + j];
                        let mut a = 0;
                        while a < ksi && row(a) == dr[g][s * block + a] {
                            a += 1;
                        }
                        hist[a] += 1;
                        made += a + 1;
                        // drafts tried until the first miss (all of them if every one held)
                        acc[i] = 0.8 * acc[i] + 0.2 * a as f64;
                        tried[i] = 0.8 * tried[i] + 0.2 * (a + 1).min(ksi) as f64;
                        for j in 0..=a {
                            obs[g].push((s, rs + j, pos[i] + 1 + j));
                        }
                        if a < ksi {
                            rw[g].push((s, ksi - a));
                        }
                        next[i] = row(a);
                        pos[i] += a + 1;
                    }
                }
                e.rewind(rw)?;
                if trace_at == Some(it) {
                    dsx::volta::profiler(false)?;
                }
                if it == 0 {
                    eprintln!("first pass {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let wall = t0.elapsed().as_secs_f64();
            let per = made as f64 / ns as f64;
            println!("{ns} sessions, k {k}{}: {:.2} tokens per pass ({:?}), {:.2} verified per session-pass, draft {:.1} ms + verify {:.1} ms per pass, {:.1} tok/s per session, {:.0} tok/s aggregate",
                if adaptive { " adaptive" } else { "" }, per / steps as f64, hist, verified as f64 / (ns * steps) as f64, t_draft / steps as f64 * 1e3, t_main / steps as f64 * 1e3, per / wall, made as f64 / wall);
        }
        Cmd::Experts { model, pairs } => {
            let m = load(&model, None)?;
            for p in pairs {
                let (l, e) = p.split_once(':').unwrap();
                let (l, e): (usize, usize) = (l.parse()?, e.parse()?);
                let ex = &m.layers[l].experts[e];
                let (a, b) = dsx::volta::expert_selftest(m, 0, l, e)?;
                println!("layer {l} expert {e} ({:?}/{:?}/{:?}): gate/up rel err {a:.2e}, output rel err {b:.2e}", ex.gate.ty, ex.up.ty, ex.down.ty);
            }
        }
        Cmd::Bench { model, engram, gpus, sessions, steps, ctx, text } => {
            anyhow::ensure!(sessions <= TMAX, "at most {TMAX} sessions per GPU in phase 1");
            let m = load(&model, engram.as_deref())?;
            let t0 = Instant::now();
            let (pg, gpus) = (gpus, if heads() { 1 } else { gpus });
            let e = Engine::new(m, Opts { ngpu: pg, slots: sessions, max_ctx: ctx, heads: heads() })?;
            eprintln!("engine up in {:.0} s", t0.elapsed().as_secs_f64());
            let corpus: Option<Vec<u32>> = text.map(|p| std::fs::read(p).map(|b| b.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect())).transpose()?;
            let mut toks: Vec<Vec<u32>> = (0..gpus)
                .map(|g| (0..sessions).map(|s| match &corpus { Some(cv) => cv[((g * sessions + s) * 97) % cv.len()], None => (1000 + 37 * (g * sessions + s)) as u32 }).collect())
                .collect();
            let mut times = vec![];
            for step in 0..steps {
                let plan: Vec<Vec<(usize, u32)>> = toks.iter().map(|v| v.iter().enumerate().map(|(s, &t)| (s, t)).collect()).collect();
                // VOLTA_TRACE_STEP=n: profile exactly step n (nsys --capture-range=cudaProfilerApi)
                let trace = std::env::var("VOLTA_TRACE_STEP").ok().and_then(|v| v.parse::<usize>().ok()) == Some(step);
                if trace {
                    dsx::volta::profiler(true)?;
                }
                let t = Instant::now();
                let out = e.step(plan)?;
                times.push(t.elapsed().as_secs_f64());
                if trace {
                    dsx::volta::profiler(false)?;
                }
                for (g, o) in out.iter().enumerate() {
                    for s in 0..sessions {
                        toks[g][s] = match &corpus {
                            Some(cv) => cv[((g * sessions + s) * 97 + step + 1) % cv.len()],
                            None => argmax(&o[s * e.vocab..(s + 1) * e.vocab]) as u32,
                        };
                    }
                }
                if step == 0 {
                    eprintln!("first step {:.0} ms", times[0] * 1e3);
                }
            }
            let warm = &times[times.len().min(2)..];
            let mean = warm.iter().sum::<f64>() / warm.len().max(1) as f64;
            let n = gpus * sessions;
            println!("{gpus} GPUs x {sessions} sessions: {:.1} ms/step, {:.0} tok/s aggregate, {:.1} tok/s per session", mean * 1e3, n as f64 / mean, 1.0 / mean);
        }
    }
    Ok(())
}
