//! dsx-volta: batched decode on 8x V100 (volta/DESIGN.md, phase 1).
//!   check: a prompt through the batched engine (several copies at once) next to the CPU reference
//!   bench: S sessions per GPU decoding together, aggregate tok/s

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dsx::model::{Model, RefState};
use dsx::volta::{Engine, Opts, SparkPlan, TMAX};
use std::path::PathBuf;
use std::time::Instant;

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
        #[arg(long, default_value_t = 32)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        prompt: usize,
        #[arg(long)]
        text: PathBuf,
        /// draft tokens verified per pass (at most the draft's block)
        #[arg(long, default_value_t = 5)]
        k: usize,
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
            let e = Engine::new(m, Opts { ngpu: gpus, slots: 2, max_ctx: ctx })?;
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
            let slots = seqs.div_ceil(gpus);
            let e = Engine::new(m, Opts { ngpu: gpus, slots, max_ctx: len + 64 })?;
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
        Cmd::Spec { model, gpus, sessions, steps, prompt, text, k } => {
            let m = load(&model, None)?;
            let corpus: Vec<u32> = std::fs::read(&text)?.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect();
            let e = Engine::new(m, Opts { ngpu: gpus, slots: sessions, max_ctx: prompt + steps * (k + 1) + 64 })?;
            let block = e.spark_block().context("VOLTA_SPARK is not set")?;
            anyhow::ensure!(k <= block && sessions * (k + 1) <= TMAX, "k {k}: at most the block ({block}), and sessions * (k + 1) <= {TMAX}");
            let ns = gpus * sessions;
            let am = |l: &[f32]| argmax(l) as u32;
            // prefill: chunks of TMAX / sessions tokens per session; the draft observes every prompt token
            let chunk = (TMAX / sessions).max(1);
            let mut last: Vec<Vec<f32>> = vec![vec![]; ns];
            for p0 in (0..prompt).step_by(chunk) {
                let p1 = (p0 + chunk).min(prompt);
                let plan: Vec<Vec<(usize, u32)>> = (0..gpus).map(|g| (0..sessions).flat_map(|s| (p0..p1).map(move |p| (s, (g * sessions + s, p)))).map(|(s, (i, p))| (s, corpus[(i * 997 + p) % corpus.len()])).collect()).collect();
                let out = e.step(plan)?;
                let w = p1 - p0;
                let obs = (0..gpus).map(|_| (0..sessions).flat_map(|s| (0..w).map(move |j| (s, s * w + j, p0 + j))).collect()).collect();
                e.spark(SparkPlan { obs, drafts: vec![vec![]; gpus] })?;
                for g in 0..gpus {
                    for s in 0..sessions {
                        let r = s * w + w - 1;
                        last[g * sessions + s] = out[g][r * e.vocab..(r + 1) * e.vocab].to_vec();
                    }
                }
            }
            // generation: draft k tokens after `next`, verify [next, drafts] in one main step, keep the agreed prefix
            let mut next: Vec<u32> = last.iter().map(|l| am(l)).collect();
            let mut pos = vec![prompt - 1; ns]; // position of each session's newest main token
            let mut obs: Vec<Vec<(usize, usize, usize)>> = vec![vec![]; gpus];
            let (mut made, mut hist) = (0usize, vec![0usize; k + 1]);
            let (mut t_draft, mut t_main) = (0f64, 0f64);
            let t0 = Instant::now();
            for it in 0..steps {
                let td = Instant::now();
                let drafts: Vec<Vec<(usize, usize, u32)>> = (0..gpus).map(|g| (0..sessions).map(|s| (s, pos[g * sessions + s], next[g * sessions + s])).collect()).collect();
                // k = 0: plain greedy decoding (the baseline), no draft step
                let dr = if k == 0 { vec![vec![0u32; sessions * block]; gpus] } else { e.spark(SparkPlan { obs: std::mem::take(&mut obs), drafts })? };
                t_draft += td.elapsed().as_secs_f64();
                let tm = Instant::now();
                let plan: Vec<Vec<(usize, u32)>> = (0..gpus).map(|g| (0..sessions).flat_map(|s| std::iter::once(next[g * sessions + s]).chain(dr[g][s * block..s * block + k].iter().copied()).map(move |t| (s, t))).collect()).collect();
                let out = e.step_greedy(plan)?;
                t_main += tm.elapsed().as_secs_f64();
                obs = vec![vec![]; gpus];
                let mut rw = vec![vec![]; gpus];
                for g in 0..gpus {
                    for s in 0..sessions {
                        let i = g * sessions + s;
                        let row = |j: usize| out[g][s * (k + 1) + j];
                        let mut a = 0;
                        while a < k && row(a) == dr[g][s * block + a] {
                            a += 1;
                        }
                        hist[a] += 1;
                        made += a + 1;
                        for j in 0..=a {
                            obs[g].push((s, s * (k + 1) + j, pos[i] + 1 + j));
                        }
                        if a < k {
                            rw[g].push((s, k - a));
                        }
                        next[i] = row(a);
                        pos[i] += a + 1;
                    }
                }
                e.rewind(rw)?;
                if it == 0 {
                    eprintln!("first pass {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            let wall = t0.elapsed().as_secs_f64();
            let per = made as f64 / ns as f64;
            println!("{ns} sessions, k {k}: {:.2} tokens per pass ({:?}), draft {:.1} ms + verify {:.1} ms per pass, {:.1} tok/s per session, {:.0} tok/s aggregate",
                per / steps as f64, hist, t_draft / steps as f64 * 1e3, t_main / steps as f64 * 1e3, per / wall, made as f64 / wall);
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
            let e = Engine::new(m, Opts { ngpu: gpus, slots: sessions, max_ctx: ctx })?;
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
