//! The batched engine behind `volta-server`: every request gets a session slot on one of the GPUs, and one loop
//! runs them all together. Each iteration: admit new requests into free slots; a draft step (the draft observes
//! the main tokens the last step kept, then drafts for the greedy sessions that are decoding); one main step that
//! mixes those sessions' verification tokens with prompt chunks of the sessions still prefilling; then emit tokens,
//! apply stop / length limits and rewind rejected drafts. Every session verifies as many drafts as its recent
//! acceptance justifies: greedy ones keep the drafts that match the argmax, sampled ones run speculative sampling
//! (a draft d is kept with probability p(d); on a miss the token comes from p without d), so sampled outputs follow
//! the target distribution exactly.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
use rayon::prelude::*;

use super::engine::{BatchEngine, BatchJob, Event, Finish, Hesitation};
use super::sample::{Rng, argmax};
use crate::model::Model;
use super::telemetry::{SessionInfo, Telemetry};
use crate::volta::{Engine, Opts, SparkPlan, TMAX};
use std::sync::Arc;

pub struct VoltaArgs {
    pub gpus: usize,
    /// sessions per GPU (head-parallel: in total)
    pub slots: usize,
    /// head-parallel (Opts::heads): the engine looks like one GPU
    pub heads: bool,
    /// context per session (prompt + generated)
    pub ctx: usize,
    /// prompt tokens per session per step while prefilling
    pub chunk: usize,
    /// most drafts verified per pass (0 = no drafting)
    pub spec_k: usize,
    /// verify j drafts while p^j >= theta, p = the session's recent per-draft acceptance
    pub theta: f64,
    /// live numbers for the admin dashboard
    pub tel: Option<Arc<Telemetry>>,
}

struct Seq {
    job: BatchJob,
    g: usize,
    s: usize,
    /// prompt tokens served from a kept KV cache
    cached: usize,
    /// tokens in the KV cache (the next position), and the tokens themselves
    fed: usize,
    hist: Vec<u32>,
    /// the newest generated token, not fed yet
    next: Option<u32>,
    drafts: Vec<u32>,
    generated: usize,
    rng: Rng,
    /// greedy without a hesitation penalty: the GPU's argmax is enough
    greedy: bool,
    /// the hesitation penalty while still reasoning (cleared at its close token)
    hes: Option<super::engine::Hesitation>,
    acc: f64,
    tried: f64,
    t0: Instant,
    first: Option<Instant>,
    done: Option<Finish>,
}

impl Seq {
    /// send one generated token; false once the sequence is finished (stop token, length, client gone)
    fn emit(&mut self, tok: u32) -> bool {
        if self.done.is_some() {
            return false;
        }
        self.first.get_or_insert_with(Instant::now);
        if self.job.params.stop_token_ids.contains(&tok) {
            self.done = Some(Finish::Stop);
            return false;
        }
        if self.job.tx.send(Event::Token(tok)).is_err() {
            self.done = Some(Finish::Cancelled);
            return false;
        }
        self.generated += 1;
        if self.generated >= self.job.params.max_new {
            self.done = Some(Finish::Length);
            return false;
        }
        true
    }

    /// drafts to verify this pass
    fn k(&self, kmax: usize, theta: f64) -> usize {
        // drafts for greedy and sampled sessions alike (speculative sampling), not while a hesitation penalty is on
        if self.hes.is_some() || kmax == 0 {
            return 0;
        }
        let p = (self.acc / self.tried).clamp(0.0, 0.99);
        (1..=kmax).take_while(|&j| p.powi(j as i32) >= theta).last().unwrap_or(1).min(self.drafts.len())
    }
}

pub struct Volta {
    e: Engine,
    a: VoltaArgs,
    block: usize,
    /// sampled steps choose their tokens on the GPU (vspec_sample) where they can: not head-parallel, and not with
    /// VOLTA_GPU_SAMPLE=0
    gpu_sample: bool,
}

// GPU state lives on the engine's own threads; this struct only holds their handles.
unsafe impl Send for Volta {}

/// Runs on the server's engine thread: load the model and start the GPUs.
pub fn volta_engine(m: &'static Model, a: VoltaArgs) -> Result<Box<dyn BatchEngine>> {
    anyhow::ensure!(a.slots >= 1 && a.slots <= TMAX, "slots per GPU: 1..={TMAX}");
    let e = Engine::new(m, Opts { ngpu: a.gpus, slots: a.slots, max_ctx: a.ctx, heads: a.heads })?;
    let block = e.spark_block().unwrap_or(0);
    // from here on `gpus` is what the scheduler places sessions on
    let a = VoltaArgs { spec_k: if block == 0 { 0 } else { a.spec_k.min(block) }, gpus: e.gpus(), ..a };
    tracing::info!(gpus = a.gpus, slots = a.slots, heads = a.heads, ctx = a.ctx, draft = block > 0, spec_k = a.spec_k, "volta engine up");
    let gpu_sample = !e.head_parallel() && std::env::var("VOLTA_GPU_SAMPLE").map_or(true, |v| v != "0");
    tracing::info!(gpu_sample, "sampling");
    Ok(Box::new(Volta { e, a, block, gpu_sample }))
}

impl BatchEngine for Volta {
    fn serve(&mut self, rx: mpsc::Receiver<BatchJob>) {
        if let Err(e) = self.run(rx) {
            tracing::error!("volta engine stopped: {e:#}");
        }
    }
}

impl Volta {
    fn run(&mut self, rx: mpsc::Receiver<BatchJob>) -> Result<()> {
        let (ng, ns) = (self.a.gpus, self.a.slots);
        let mut queue: VecDeque<BatchJob> = VecDeque::new();
        let mut seqs: Vec<Seq> = vec![];
        let mut busy = vec![vec![false; ns]; ng];
        // what each free slot's KV cache still holds (the tokens of its last session): a new request whose prompt
        // starts the same way reuses it (agents resend the whole conversation every turn)
        let mut kept: Vec<Vec<Vec<u32>>> = vec![vec![vec![]; ns]; ng];
        // main tokens of the last step for the draft to observe: (slot, row, position) per GPU
        let mut obs: Vec<Vec<(usize, usize, usize)>> = vec![vec![]; ng];
        // VOLTA_LOOP_PROF=n: every n passes, where an iteration's time went (draft call, main step call, rewind call,
        // the rest of the loop)
        let prof_n = std::env::var("VOLTA_LOOP_PROF").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let (mut pf, mut pf_i, mut t_iter) = ([0f64; 4], 0usize, Instant::now());
        loop {
            // admit
            if seqs.is_empty() && queue.is_empty() {
                match rx.recv() {
                    Ok(j) => queue.push_back(j),
                    Err(_) => return Ok(()),
                }
            }
            while let Ok(j) = rx.try_recv() {
                queue.push_back(j);
            }
            while !queue.is_empty() {
                let load: Vec<usize> = (0..ng).map(|g| busy[g].iter().filter(|&&b| b).count()).collect();
                let Some(g0) = (0..ng).filter(|&g| load[g] < ns).min_by_key(|&g| load[g]) else { break };
                let j = queue.pop_front().unwrap();
                let bad = if j.prompt.is_empty() {
                    Some("empty prompt".to_string())
                } else if j.prompt.len() + 8 > self.a.ctx {
                    Some(format!("prompt of {} tokens does not fit the {}-token context", j.prompt.len(), self.a.ctx))
                } else if !j.params.images.is_empty() {
                    Some("image input is not supported by this engine".to_string())
                } else {
                    None
                };
                if let Some(msg) = bad {
                    let _ = j.tx.send(Event::Error(msg));
                    continue;
                }
                // the free slot holding the longest prefix of the prompt (at least 64 tokens, and at least one
                // prompt token left to feed), else a fresh slot on the least loaded GPU
                let lcp = |h: &[u32]| h.iter().zip(&j.prompt).take_while(|(a, b)| a == b).count().min(j.prompt.len() - 1);
                let best = (0..ng)
                    .flat_map(|g| (0..ns).map(move |s| (g, s)))
                    .filter(|&(g, s)| !busy[g][s])
                    .map(|(g, s)| (lcp(&kept[g][s]), g, s))
                    .max_by_key(|&(n, g, _)| (n, std::cmp::Reverse(load[g])));
                let (reuse, g, s) = match best {
                    Some((n, g, s)) if n >= 64 => (n, g, s),
                    _ => (0, g0, busy[g0].iter().position(|&b| !b).unwrap()),
                };
                busy[g][s] = true;
                if reuse > 0 {
                    let back = kept[g][s].len() - reuse;
                    self.e.rewind((0..ng).map(|x| if x == g && back > 0 { vec![(s, back)] } else { vec![] }).collect())?;
                } else {
                    self.e.reset(g, s)?;
                }
                kept[g][s].clear();
                tracing::info!(id = %j.id, gpu = g, slot = s, prompt_tokens = j.prompt.len(), cached_tokens = reuse, queue_ms = j.queued.elapsed().as_millis() as u64, "admitted");
                let hes = j.params.hesitation.clone();
                let greedy = j.params.temperature <= 0.0 && hes.is_none();
                let seed = j.params.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) ^ (g * 131 + s) as u64);
                let hist = j.prompt[..reuse].to_vec();
                if let Some(t) = &self.a.tel {
                    t.admitted(reuse);
                }
                seqs.push(Seq { job: j, g, s, cached: reuse, fed: reuse, hist, next: None, drafts: vec![], generated: 0, rng: Rng::new(seed), greedy, hes, acc: 0.6, tried: 1.0, t0: Instant::now(), first: None, done: None });
            }
            for q in seqs.iter_mut() {
                if q.done.is_none() && q.job.tx.is_closed() {
                    q.done = Some(Finish::Cancelled);
                }
            }
            self.retire(&mut seqs, &mut busy, &mut kept);
            if seqs.is_empty() {
                continue;
            }
            // draft step: observe the last step's kept tokens, draft for greedy decoding sessions
            let td = Instant::now();
            let gen0: usize = seqs.iter().map(|q| q.generated).sum();
            if self.block > 0 {
                let want: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].hes.is_none() && seqs[i].next.is_some() && self.a.spec_k > 0).collect();
                if !want.is_empty() || obs.iter().any(|o| !o.is_empty()) {
                    let mut drafts: Vec<Vec<(usize, usize, u32)>> = vec![vec![]; ng];
                    let mut at = vec![];
                    for &i in &want {
                        let q = &seqs[i];
                        at.push((i, q.g, drafts[q.g].len()));
                        drafts[q.g].push((q.s, q.fed - 1, q.next.unwrap()));
                    }
                    let tc = Instant::now();
                    let dr = self.e.spark(SparkPlan { obs: std::mem::replace(&mut obs, vec![vec![]; ng]), drafts })?;
                    pf[0] += tc.elapsed().as_secs_f64();
                    for (i, g, j) in at {
                        seqs[i].drafts = dr[g][j * self.block..(j + 1) * self.block].to_vec();
                    }
                }
            }
            let draft_ms = td.elapsed().as_secs_f64() * 1e3;
            // main step: decoding sessions first ([next, drafts]), then prompt chunks
            let mut plan: Vec<Vec<(usize, u32)>> = vec![vec![]; ng];
            let mut rows: Vec<(usize, usize, usize, usize)> = vec![]; // (seq, gpu, first row, tokens)
            for (i, q) in seqs.iter().enumerate() {
                if let Some(n) = q.next {
                    let k = q.k(self.a.spec_k, self.a.theta).min(self.a.ctx.saturating_sub(q.fed + 2));
                    rows.push((i, q.g, plan[q.g].len(), 1 + k));
                    plan[q.g].push((q.s, n));
                    plan[q.g].extend(q.drafts[..k].iter().map(|&d| (q.s, d)));
                }
            }
            for (i, q) in seqs.iter().enumerate() {
                if q.next.is_none() {
                    let left = TMAX - plan[q.g].len();
                    let n = (q.job.prompt.len() - q.fed).min(self.a.chunk).min(left);
                    if n > 0 {
                        rows.push((i, q.g, plan[q.g].len(), n));
                        plan[q.g].extend(q.job.prompt[q.fed..q.fed + n].iter().map(|&t| (q.s, t)));
                    }
                }
            }
            if rows.is_empty() {
                continue;
            }
            let tm = Instant::now();
            let (mut fed_prompt, mut drafted, mut kept_d) = (0usize, 0usize, 0usize);
            // what each row set decides: (drafts kept, the next token). Greedy sessions compare the GPU's argmax with
            // their drafts; sampled ones run speculative sampling on the logits (exact for the target distribution)
            let sampled = rows.iter().any(|&(i, ..)| !seqs[i].greedy);
            // the tokens chosen on the GPU (vspec_sample) when every row allows it: no top-p, no hesitation penalty,
            // at most 8 drafts, not head-parallel; VOLTA_GPU_SAMPLE=0 keeps the CPU path
            let on_gpu = sampled
                && self.gpu_sample
                && rows.iter().all(|&(i, _, _, n)| {
                    let q = &seqs[i];
                    q.hes.is_none() && (q.greedy || q.job.params.top_p >= 1.0) && n <= 9
                });
            let dec: Vec<(usize, u32)> = if on_gpu {
                let mut tabs: Vec<(Vec<i32>, Vec<f32>)> = vec![(vec![], vec![]); ng];
                let mut at = Vec::with_capacity(rows.len()); // (gpu, row in its table)
                for &(i, g, r0, n) in &rows {
                    let q = &mut seqs[i];
                    let (rb, nd) = if q.next.is_none() { (r0 + n - 1, 0) } else { (r0, n - 1) };
                    let (ti, tf) = &mut tabs[g];
                    at.push((g, ti.len() / 10));
                    ti.extend([rb as i32, nd as i32]);
                    ti.extend((0..8).map(|j| if j < nd { q.drafts[j] as i32 } else { -1 }));
                    tf.push(if q.greedy { 0.0 } else { q.job.params.temperature });
                    // uniforms in [0, 1): an f64 just under 1 rounds to 1.0 as f32
                    tf.extend((0..9).map(|j| if j <= nd { (q.rng.next_f64() as f32).min(0.999_999_94) } else { 0.0 }));
                }
                let tc = Instant::now();
                let r = self.e.step_sample(plan, tabs)?;
                pf[1] += tc.elapsed().as_secs_f64();
                at.into_iter().map(|(g, k)| r[g][k]).collect()
            } else if sampled {
                let lg = self.e.step(plan)?;
                let v = self.e.vocab;
                let mut work: Vec<(usize, Rng, Option<Hesitation>)> = rows.iter().map(|&(i, ..)| (i, seqs[i].rng.clone(), seqs[i].hes.clone())).collect();
                let out: Vec<(usize, u32)> = work
                    .par_iter_mut()
                    .zip(rows.par_iter())
                    .map(|((i, rng, hes), &(_, g, r0, n))| {
                        let q = &seqs[*i];
                        let row = |j: usize| &lg[g][(r0 + j) * v..(r0 + j + 1) * v];
                        let prefill = q.next.is_none();
                        if q.greedy {
                            if prefill {
                                return (0, argmax(row(n - 1)));
                            }
                            let mut a = 0;
                            while a < n - 1 && argmax(row(a)) == q.drafts[a] {
                                a += 1;
                            }
                            return (a, argmax(row(a)));
                        }
                        let (temp, top_p) = (q.job.params.temperature, q.job.params.top_p);
                        let pen = |j: usize, hes: &Option<Hesitation>| -> Vec<f32> {
                            let mut l = row(j).to_vec();
                            if let Some(h) = hes {
                                for &id in h.ids.iter() {
                                    if let Some(x) = l.get_mut(id as usize) {
                                        *x -= h.lambda;
                                    }
                                }
                            }
                            l
                        };
                        if prefill {
                            return (0, dist_sample(&pen(n - 1, hes), temp, top_p, None, rng));
                        }
                        let mut a = 0;
                        while a < n - 1 {
                            let l = pen(a, hes);
                            let d = q.drafts[a];
                            if rng.next_f64() < dist_prob(&l, temp, top_p, d) {
                                a += 1;
                            } else {
                                // the draft is one-hot, so the residual is the target with d removed
                                return (a, dist_sample(&l, temp, top_p, Some(d), rng));
                            }
                        }
                        (a, dist_sample(&pen(a, hes), temp, top_p, None, rng))
                    })
                    .collect();
                for ((i, rng, _), &(_, tok)) in work.into_iter().zip(&out) {
                    let q = &mut seqs[i];
                    q.rng = rng;
                    if q.hes.as_ref().is_some_and(|h| tok == h.close) {
                        q.hes = None;
                    }
                }
                out
            } else {
                let tc = Instant::now();
                let ids = self.e.step_greedy(plan)?;
                pf[1] += tc.elapsed().as_secs_f64();
                rows.iter()
                    .map(|&(i, g, r0, n)| {
                        let q = &seqs[i];
                        if q.next.is_none() {
                            return (0, ids[g][r0 + n - 1]);
                        }
                        let mut a = 0;
                        while a < n - 1 && ids[g][r0 + a] == q.drafts[a] {
                            a += 1;
                        }
                        (a, ids[g][r0 + a])
                    })
                    .collect()
            };
            // keep, emit, rewind
            let mut rw: Vec<Vec<(usize, usize)>> = vec![vec![]; ng];
            for (&(i, g, r0, n), &(a, tok)) in rows.iter().zip(&dec) {
                let q = &mut seqs[i];
                if q.next.is_none() {
                    // a prompt chunk: the draft observes every prompt token
                    for j in 0..n {
                        obs[g].push((q.s, r0 + j, q.fed + j));
                    }
                    q.hist.extend_from_slice(&q.job.prompt[q.fed..q.fed + n]);
                    fed_prompt += n;
                    q.fed += n;
                    if q.fed == q.job.prompt.len() {
                        q.next = Some(tok);
                        q.emit(tok);
                    }
                    continue;
                }
                let k = n - 1;
                drafted += k;
                kept_d += a;
                if k > 0 {
                    q.acc = 0.8 * q.acc + 0.2 * a as f64;
                    q.tried = 0.8 * q.tried + 0.2 * (a + 1).min(k) as f64;
                }
                for j in 0..=a {
                    obs[g].push((q.s, r0 + j, q.fed + j));
                }
                if a < k {
                    rw[g].push((q.s, k - a));
                }
                q.fed += a + 1;
                q.hist.push(q.next.unwrap());
                q.hist.extend_from_slice(&q.drafts[..a]);
                for j in 0..a {
                    let d = q.drafts[j];
                    if !q.emit(d) {
                        break;
                    }
                }
                q.next = Some(tok);
                if q.done.is_none() {
                    q.emit(tok);
                }
                if q.done.is_none() && q.fed + 2 >= self.a.ctx {
                    q.done = Some(Finish::Length);
                }
            }
            if rw.iter().any(|r| !r.is_empty()) {
                let tc = Instant::now();
                self.e.rewind(rw)?;
                pf[2] += tc.elapsed().as_secs_f64();
            }
            if prof_n > 0 {
                pf_i += 1;
                if pf_i == prof_n {
                    let all = t_iter.elapsed().as_secs_f64();
                    let ms = |x: f64| x * 1e3 / prof_n as f64;
                    tracing::info!(iter_ms = format!("{:.2}", ms(all)), spark_ms = format!("{:.2}", ms(pf[0])), step_ms = format!("{:.2}", ms(pf[1])), rewind_ms = format!("{:.2}", ms(pf[2])), rest_ms = format!("{:.2}", ms(all - pf[0] - pf[1] - pf[2])), "loop profile");
                    (pf, pf_i, t_iter) = ([0f64; 4], 0, Instant::now());
                }
            }
            if let Some(t) = &self.a.tel {
                let now = Instant::now();
                let gen1: usize = seqs.iter().map(|q| q.generated).sum();
                let info = seqs
                    .iter()
                    .filter(|q| q.done.is_none())
                    .map(|q| SessionInfo {
                        id: q.job.id.clone(),
                        gpu: q.g,
                        slot: q.s,
                        state: if q.next.is_none() { "prefill" } else { "decode" },
                        prompt_tokens: q.job.prompt.len(),
                        cached_tokens: q.cached,
                        generated: q.generated,
                        tok_s: q.first.map(|f| if q.generated > 1 { (q.generated - 1) as f64 / (now - f).as_secs_f64().max(1e-3) } else { 0.0 }).unwrap_or(0.0),
                        acceptance: q.acc / q.tried,
                        age_s: (now - q.t0).as_secs_f64(),
                        sampled: !q.greedy,
                    })
                    .collect();
                t.step(gen1 - gen0, fed_prompt, tm.elapsed().as_secs_f64() * 1e3, draft_ms, drafted, kept_d, info, queue.len());
            }
            // a finished session's rows are not observed (its slot is about to be reset)
            for q in seqs.iter().filter(|q| q.done.is_some()) {
                obs[q.g].retain(|&(s, ..)| s != q.s);
            }
            self.retire(&mut seqs, &mut busy, &mut kept);
        }
    }

    /// send Done for finished sequences and free their slots
    fn retire(&self, seqs: &mut Vec<Seq>, busy: &mut [Vec<bool>], kept: &mut [Vec<Vec<u32>>]) {
        seqs.retain_mut(|q| {
            let Some(f) = q.done else { return true };
            let _ = q.job.tx.send(Event::Done { finish: f });
            busy[q.g][q.s] = false;
            kept[q.g][q.s] = std::mem::take(&mut q.hist);
            let total = q.t0.elapsed().as_secs_f64();
            let ttft = q.first.map(|f| (f - q.t0).as_secs_f64()).unwrap_or(total);
            let tps = if q.generated > 1 { (q.generated - 1) as f64 / (total - ttft).max(1e-9) } else { 0.0 };
            tracing::info!(id = %q.job.id, prompt_tokens = q.job.prompt.len(), generated = q.generated, ttft_ms = (ttft * 1e3) as u64, decode_tps = format!("{tps:.1}"), finish = ?f, "done");
            if let Some(t) = &self.a.tel {
                t.finished(&q.job.id, q.generated, tps);
            }
            false
        });
    }
}

/// sampling weights exp((l - max) / t) of a logits row and, under top-p < 1, the nucleus: the smallest set of the
/// highest tokens holding top_p of the mass (sorted, with its mass), as the server's Sampler draws from
fn nucleus(l: &[f32], temp: f32, top_p: f32) -> (Vec<u32>, f64, f64, f32) {
    let mx = l.iter().copied().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, f32::max);
    let inv = 1.0 / temp as f64;
    let w = |x: f32| if x.is_nan() { 0.0 } else { ((x - mx) as f64 * inv).exp() };
    let sum: f64 = l.iter().map(|&x| w(x)).sum();
    if !(top_p < 1.0) {
        return (vec![], sum, sum, mx);
    }
    let target = top_p.max(0.0) as f64 * sum;
    let mut idx: Vec<u32> = (0..l.len() as u32).collect();
    let key = |i: &u32| { let x = l[*i as usize]; if x.is_nan() { f32::NEG_INFINITY } else { x } };
    let mut k = 64.min(idx.len());
    loop {
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |a, b| key(b).total_cmp(&key(a)));
        }
        idx[..k].sort_unstable_by(|a, b| key(b).total_cmp(&key(a)));
        let mut acc = 0.0;
        for (j, &i) in idx[..k].iter().enumerate() {
            acc += w(l[i as usize]);
            if acc >= target {
                idx.truncate(j + 1);
                return (idx, acc, sum, mx);
            }
        }
        if k == idx.len() {
            return (idx, acc, sum, mx);
        }
        k = (k * 4).min(idx.len());
    }
}

/// probability of `tok` under temperature `temp` and top-p (greedy when temp <= 0)
fn dist_prob(l: &[f32], temp: f32, top_p: f32, tok: u32) -> f64 {
    if temp <= 0.0 {
        return if argmax(l) == tok { 1.0 } else { 0.0 };
    }
    let (nuc, mass, _, mx) = nucleus(l, temp, top_p);
    let wt = ((l[tok as usize] - mx) as f64 / temp as f64).exp();
    if nuc.is_empty() {
        return wt / mass;
    }
    if nuc.contains(&tok) { wt / mass } else { 0.0 }
}

/// sample from temperature + top-p, optionally with one token removed (renormalized): the residual after a rejected
/// one-hot draft
fn dist_sample(l: &[f32], temp: f32, top_p: f32, without: Option<u32>, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        let mut c = l.to_vec();
        if let Some(d) = without {
            c[d as usize] = f32::NEG_INFINITY;
        }
        return argmax(&c);
    }
    let (nuc, _, _, mx) = nucleus(l, temp, top_p);
    let w = |i: u32| if Some(i) == without || l[i as usize].is_nan() { 0.0 } else { ((l[i as usize] - mx) as f64 / temp as f64).exp() };
    let pick = |it: &mut dyn Iterator<Item = u32>, total: f64, rng: &mut Rng| -> Option<u32> {
        let mut r = rng.next_f64() * total;
        let mut last = None;
        for i in it {
            let p = w(i);
            if p > 0.0 {
                last = Some(i);
                r -= p;
                if r < 0.0 {
                    return Some(i);
                }
            }
        }
        last
    };
    if nuc.is_empty() {
        let total: f64 = (0..l.len() as u32).map(w).sum();
        return pick(&mut (0..l.len() as u32), total, rng).unwrap_or_else(|| argmax(l));
    }
    let total: f64 = nuc.iter().map(|&i| w(i)).sum();
    if total <= 0.0 {
        // the nucleus was only the rejected token: fall back to everything else
        let total: f64 = (0..l.len() as u32).map(w).sum();
        return pick(&mut (0..l.len() as u32), total, rng).unwrap_or_else(|| argmax(l));
    }
    pick(&mut nuc.iter().copied(), total, rng).unwrap_or(nuc[0])
}

