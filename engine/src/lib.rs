//! dsx: DeepSeek-V4.1-Flash(-1017) on one desktop (2x RTX 3090 or one RTX 5090, 128 GB RAM): dense half on the GPU,
//! routed experts spread over VRAM, a RAM arena and the file. The shared parts (config, readers, formats, CPU
//! reference, server) are in `dsx-core` and re-exported here under their old paths.

pub use dsx_core::{dspark, formats, gguf, iqgrid, model, numerics, safetensors};

#[cfg(feature = "cuda")]
pub mod attn;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod experts;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpumodel;
#[cfg(feature = "cuda")]
pub mod handoff;
pub mod q6;
pub mod spec;
#[cfg(feature = "cuda")]
pub mod topk;
#[cfg(feature = "cuda")]
pub mod volta;

/// The server (from dsx-core) and the GPU engine behind it.
pub mod server {
    pub use dsx_core::server::*;
    #[cfg(feature = "cuda")]
    pub mod real;
    #[cfg(feature = "cuda")]
    pub mod volta;
}
