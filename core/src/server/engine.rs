//! The engine abstraction and the single OS thread that runs it (GPU contexts are thread-bound).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_new: usize,
    /// 0 = greedy
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    /// EOS and friends. The worker enforces these and `max_new` too, so an engine may ignore them.
    pub stop_token_ids: Vec<u32>,
    /// logit penalty on hesitation tokens while the model is reasoning (None = off)
    pub hesitation: Option<Hesitation>,
    /// images in the prompt, in order (their spans already expanded in the prompt ids)
    pub images: std::sync::Arc<Vec<crate::vision::ImageInput>>,
}

/// Subtract `lambda` from the logits of `ids` until `close` (`</think>`) is generated.
#[derive(Debug, Clone)]
pub struct Hesitation {
    pub lambda: f32,
    pub ids: std::sync::Arc<Vec<u32>>,
    pub close: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

/// What an engine knows about its last `generate` call. Unset timings fall back to the worker's wall clock
/// (prefill = until the first token, decode = the rest).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenStats {
    pub prefill: Option<Duration>,
    pub decode: Option<Duration>,
    /// prompt tokens served from a KV/prefix cache
    pub cached_prompt_tokens: usize,
}

pub trait Engine: Send {
    /// Generate after `prompt`. Call `emit(token)` per generated token; stop early when it returns false.
    fn generate(&mut self, prompt: &[u32], p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish>;
    /// Timings/cache stats of the last `generate` call.
    fn last_stats(&self) -> GenStats {
        GenStats::default()
    }
}

/// Engine thread -> request handler.
#[derive(Debug)]
pub enum Event {
    Token(u32),
    Done { finish: Finish },
    Error(String),
}

struct Job {
    id: String,
    prompt: Vec<u32>,
    params: GenParams,
    tx: mpsc::UnboundedSender<Event>,
    queued: Instant,
}

/// A job as a batching engine receives it (`EngineHandle::spawn_batched`): it sends `Event`s on `tx` and drops the
/// job when it is finished (which updates the queue count).
pub struct BatchJob {
    pub id: String,
    pub prompt: Vec<u32>,
    pub params: GenParams,
    pub tx: mpsc::UnboundedSender<Event>,
    pub queued: Instant,
    _pending: PendingGuard,
}

struct PendingGuard(Arc<AtomicUsize>);

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// An engine that runs many generations at once: it takes every job as it arrives and schedules them itself.
pub trait BatchEngine: Send {
    /// Serve jobs until `rx` closes.
    fn serve(&mut self, rx: std_mpsc::Receiver<BatchJob>);
}

/// Cloneable handle to the engine thread: a FIFO of jobs, one at a time (or all at once for a batching engine).
#[derive(Clone)]
pub struct EngineHandle {
    tx: std_mpsc::Sender<Job>,
    pending: Arc<AtomicUsize>,
}

impl EngineHandle {
    /// Construct the engine on a dedicated thread (`make` runs there) and start serving jobs.
    pub fn spawn(make: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pend = pending.clone();
        std::thread::Builder::new()
            .name("dsx-engine".into())
            .spawn(move || {
                let mut engine = match make() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    run_job(engine.as_mut(), job);
                    pend.fetch_sub(1, Ordering::SeqCst);
                }
            })?;
        ready_rx.recv().context("engine thread died during construction")??;
        Ok(Self { tx, pending })
    }

    /// Construct a batching engine on a dedicated thread (`make` runs there): every submitted job goes to it at once.
    pub fn spawn_batched(make: impl FnOnce() -> Result<Box<dyn BatchEngine>> + Send + 'static) -> Result<Self> {
        let (tx, rx) = std_mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<()>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let pend = pending.clone();
        std::thread::Builder::new()
            .name("dsx-engine".into())
            .spawn(move || {
                let mut engine = match make() {
                    Ok(e) => {
                        let _ = ready_tx.send(Ok(()));
                        e
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // forward the plain jobs as batch jobs (carrying the queue-count guard)
                let (btx, brx) = std_mpsc::channel::<BatchJob>();
                std::thread::spawn(move || {
                    while let Ok(j) = rx.recv() {
                        let b = BatchJob { id: j.id, prompt: j.prompt, params: j.params, tx: j.tx, queued: j.queued, _pending: PendingGuard(pend.clone()) };
                        if btx.send(b).is_err() {
                            break;
                        }
                    }
                });
                engine.serve(brx);
            })?;
        ready_rx.recv().context("engine thread died during construction")??;
        Ok(Self { tx, pending })
    }

    /// Queue a generation; tokens and the final event arrive on the returned channel.
    /// Dropping the receiver cancels the job (queued or running).
    pub fn submit(&self, id: String, prompt: Vec<u32>, params: GenParams) -> Result<mpsc::UnboundedReceiver<Event>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let ahead = self.pending.fetch_add(1, Ordering::SeqCst);
        tracing::info!(%id, prompt_tokens = prompt.len(), max_new = params.max_new, queue_position = ahead, "queued");
        let job = Job { id, prompt, params, tx, queued: Instant::now() };
        if self.tx.send(job).is_err() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("engine thread is gone");
        }
        Ok(rx)
    }

    /// Jobs queued or running.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }
}

fn run_job(engine: &mut dyn Engine, job: Job) {
    let Job { id, prompt, params, tx, queued } = job;
    if tx.is_closed() {
        tracing::info!(%id, "cancelled while queued");
        return;
    }
    let wait = queued.elapsed();
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    let mut generated = 0usize;
    let (mut hit_stop, mut hit_len, mut gone) = (false, false, false);
    let r = {
        let mut emit = |tok: u32| -> bool {
            if hit_stop || hit_len || gone {
                return false;
            }
            first.get_or_insert_with(Instant::now);
            if params.stop_token_ids.contains(&tok) {
                hit_stop = true;
                return false;
            }
            generated += 1;
            if tx.send(Event::Token(tok)).is_err() {
                gone = true;
                return false;
            }
            if generated >= params.max_new {
                hit_len = true;
                return false;
            }
            true
        };
        if params.max_new == 0 { Ok(Finish::Length) } else { engine.generate(&prompt, &params, &mut emit) }
    };
    let total = t0.elapsed();
    let st = engine.last_stats();
    let ttft = first.map(|f| f - t0).unwrap_or(total);
    let prefill = st.prefill.unwrap_or(ttft);
    let decode = st.decode.unwrap_or(total.saturating_sub(ttft));
    let finish = match r {
        Ok(f) => {
            // the worker's own stop/length/disconnect checks win over what the engine says
            let f = if hit_stop { Finish::Stop } else if hit_len { Finish::Length } else if gone { Finish::Cancelled } else { f };
            let _ = tx.send(Event::Done { finish: f });
            format!("{f:?}")
        }
        Err(e) => {
            tracing::error!(%id, "engine error: {e:#}");
            let _ = tx.send(Event::Error(format!("{e:#}")));
            "Error".into()
        }
    };
    let decode_tps = if generated > 1 { (generated - 1) as f64 / decode.as_secs_f64().max(1e-9) } else { 0.0 };
    tracing::info!(
        %id,
        prompt_tokens = prompt.len(),
        cached_prompt_tokens = st.cached_prompt_tokens,
        generated,
        queue_ms = wait.as_millis() as u64,
        prefill_ms = prefill.as_millis() as u64,
        decode_ms = decode.as_millis() as u64,
        prefill_tps = format!("{:.1}", (prompt.len() - st.cached_prompt_tokens.min(prompt.len())) as f64 / prefill.as_secs_f64().max(1e-9)),
        decode_tps = format!("{decode_tps:.2}"),
        finish,
        "done"
    );
}

static REQ: AtomicU64 = AtomicU64::new(0);

/// Unique-enough request id: prefix + unix seconds + counter.
pub fn request_id(prefix: &str) -> String {
    let n = REQ.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{prefix}{t:x}{n:04x}")
}
