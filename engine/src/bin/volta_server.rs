//! volta-server: the OpenAI / Anthropic compatible API of dsx-server (the same routes, chat template, thinking
//! levels and sampling parameters) on the batched 8x V100 engine. Every request gets a session slot; the GPUs run
//! all of them together, with DSpark drafts verified by speculative sampling.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::Parser;
use dsx::server::telemetry::Telemetry;
use dsx::server::{AppState, EngineHandle, ServerConfig};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(about = "OpenAI / Anthropic compatible API server for DeepSeek-V4.1-Flash on 8x V100 (volta)")]
struct Cli {
    /// Complete Volta GGUF file, or checkpoint directory with optional VOLTA_PACK
    #[arg(long)]
    model: PathBuf,
    /// Optional tokenizer.json override (default: embedded tokenizer, or model directory's)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8090)]
    port: u16,
    /// model id reported by /v1/models
    #[arg(long, default_value = "deepseek-v4.1-flash")]
    model_name: String,
    #[arg(long, default_value_t = 8)]
    gpus: usize,
    /// concurrent sessions per GPU (with --head-parallel: in total)
    #[arg(long, default_value_t = 4)]
    slots: usize,
    /// head-parallel: every GPU runs every session with 8 of the 64 attention heads, for the fastest single
    /// sessions; holds --slots sessions in total instead of --slots per GPU
    #[arg(long)]
    head_parallel: bool,
    /// context window per session (prompt + generated tokens)
    #[arg(long, default_value_t = 8192)]
    ctx: usize,
    /// prompt tokens per session per step while prefilling
    #[arg(long, default_value_t = 32)]
    chunk: usize,
    /// generation cap when a request sets no max_tokens
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    /// default sampling when a request sets none (0 = greedy)
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// logit penalty on hesitation tokens while reasoning; 0 = off. A request's `hesitation_penalty` overrides it
    #[arg(long, default_value_t = 0.0)]
    hesitation_penalty: f32,
    /// most DSpark drafts verified per pass (0 = no draft); greedy requests compare argmaxes, sampled ones run
    /// speculative sampling, so outputs follow the same distribution as without drafts
    #[arg(long, default_value_t = 5)]
    spec_k: usize,
    /// verify j drafts while p^j >= theta (p = the session's recent per-draft acceptance)
    #[arg(long, default_value_t = 0.35)]
    theta: f64,
    /// require this key on every API request (`Authorization: Bearer <key>` or `x-api-key: <key>`)
    #[arg(long, env = "DSX_API_KEY")]
    api_key: Option<String>,
    /// admin dashboard and `/stats` (JSON) on this port, on --host; 0 = off. Not behind --api-key: keep it on
    /// loopback or a private network
    #[arg(long, default_value_t = 8091)]
    admin_port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let cli = Cli::parse();
    let tok = Arc::new(if let Some(path)=&cli.tokenizer {
        Tokenizer::from_file(path).map_err(|e|anyhow!("{}: {e}",path.display()))?
    } else if cli.model.is_file() {
        Tokenizer::from_bytes(dsx_core::volta_quant::complete::read_asset(&cli.model,"tokenizer.json")?).map_err(|e|anyhow!("embedded tokenizer: {e}"))?
    } else {Tokenizer::from_file(cli.model.join("tokenizer.json")).map_err(|e|anyhow!("tokenizer: {e}"))?});
    if cli.model.is_file() {unsafe {std::env::set_var("VOLTA_PACK",&cli.model);}}
    // graphs cost memory, and served traffic comes in many step shapes: keep a few (VOLTA_GRAPHS overrides)
    if std::env::var_os("VOLTA_GRAPHS").is_none() {
        unsafe { std::env::set_var("VOLTA_GRAPHS", "4") };
    }
    // the draft lives in the same checkpoint (mtp.*)
    if cli.spec_k > 0 && std::env::var_os("VOLTA_SPARK").is_none() {
        unsafe { std::env::set_var("VOLTA_SPARK", &cli.model) };
    }
    let model = cli.model.clone();
    let capacity = if cli.head_parallel { cli.slots } else { cli.gpus * cli.slots };
    let tel = Arc::new(Telemetry::new(cli.gpus, capacity, cli.ctx, cli.head_parallel));
    let args = dsx::server::volta::VoltaArgs { gpus: cli.gpus, slots: cli.slots, heads: cli.head_parallel, ctx: cli.ctx, chunk: cli.chunk, spec_k: cli.spec_k, theta: cli.theta, tel: Some(tel.clone()) };
    let engine = EngineHandle::spawn_batched(move || {
        let m: &'static dsx::model::Model = Box::leak(Box::new(dsx::model::Model::load_hf(&model)?));
        dsx::server::volta::volta_engine(m, args)
    })?;
    let cfg = ServerConfig {
        model_name: cli.model_name.clone(),
        ctx: cli.ctx,
        default_max_tokens: cli.max_tokens,
        default_temperature: cli.temperature,
        default_top_p: cli.top_p,
        hesitation_penalty: cli.hesitation_penalty,
        vision: false,
        api_key: cli.api_key.clone(),
        ..ServerConfig::default()
    };
    let st = AppState::new(engine, tok, cfg)?;
    if cli.admin_port != 0 {
        let admin = axum::Router::new()
            .route("/", axum::routing::get(|| async { axum::response::Html(include_str!("../server/dashboard.html")) }))
            .route("/stats", axum::routing::get(move || { let t = tel.clone(); async move { axum::Json(t.snapshot()) } }));
        let al = tokio::net::TcpListener::bind((cli.host.as_str(), cli.admin_port)).await?;
        tracing::info!("admin dashboard on http://{}", al.local_addr()?);
        tokio::spawn(async move { axum::serve(al, admin).await });
    }
    let listener = tokio::net::TcpListener::bind((cli.host.as_str(), cli.port)).await?;
    tracing::info!("listening on http://{} (volta, {} GPUs, {} sessions{})", listener.local_addr()?, cli.gpus, capacity, if cli.head_parallel { ", head-parallel" } else { "" });
    dsx::server::serve(listener, st).await?;
    Ok(())
}
