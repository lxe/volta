//! The CUDA boundary: one context + stream per device, device buffers, host-registered memory,
//! kernel launches, events and CUDA graphs. Driven from the thread that created it.
#![allow(clippy::missing_safety_doc)]

use anyhow::{Result, bail};
use cudarc::cublas::sys as cb;
use cudarc::driver::sys as cu;
use std::cell::Cell;
use std::ffi::{CStr, CString, c_void};

pub fn check(r: cu::CUresult, what: &str) -> Result<()> {
    if r == cu::CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut msg: *const std::ffi::c_char = std::ptr::null();
    unsafe { cu::cuGetErrorString(r, &mut msg) };
    let text = if msg.is_null() { format!("{r:?}") } else { unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned() };
    bail!("{what}: {text}")
}

const KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dsx.fatbin"));

/// Device memory, freed on drop.
pub struct Buf {
    pub ptr: u64,
    pub bytes: usize,
    ctx: cu::CUcontext,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                cu::cuCtxSetCurrent(self.ctx);
                cu::cuMemFree_v2(self.ptr);
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Kernel(cu::CUfunction);
unsafe impl Send for Kernel {}

#[derive(Clone, Copy, Debug)]
pub enum Arg {
    P(u64),
    I(i32),
    F(f32),
}

pub struct Gpu {
    pub ordinal: i32,
    pub name: String,
    pub sms: usize,
    pub arch: u32,
    ctx: cu::CUcontext,
    pub stream: cu::CUstream,
    /// second stream for copies that overlap the main stream (prefill expert streaming)
    pub side: cu::CUstream,
    module: cu::CUmodule,
    blas: Cell<cb::cublasHandle_t>,
}

#[derive(Clone, Copy)]
pub struct Event(cu::CUevent);

/// An instantiated CUDA graph (one captured step).
pub struct GraphExec(cu::CUgraphExec);
unsafe impl Send for GraphExec {}

unsafe impl Send for Gpu {}

impl Gpu {
    pub fn new(ordinal: i32) -> Result<Gpu> {
        if KERNEL_IMAGE.is_empty() {
            bail!("built without the `cuda` feature: no kernels");
        }
        unsafe {
            check(cu::cuInit(0), "cuInit")?;
            let mut dev = 0;
            check(cu::cuDeviceGet(&mut dev, ordinal), "cuDeviceGet")?;
            let mut ctx = std::ptr::null_mut();
            check(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            check(cu::cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut name = [0 as std::ffi::c_char; 256];
            check(cu::cuDeviceGetName(name.as_mut_ptr(), 256, dev), "cuDeviceGetName")?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let attr = |a| -> Result<i32> {
                let mut v = 0;
                check(cu::cuDeviceGetAttribute(&mut v, a, dev), "cuDeviceGetAttribute")?;
                Ok(v)
            };
            use cu::CUdevice_attribute::*;
            let major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            let sms = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
            let mut stream = std::ptr::null_mut();
            check(cu::cuStreamCreate(&mut stream, cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32), "cuStreamCreate")?;
            let mut side = std::ptr::null_mut();
            check(cu::cuStreamCreate(&mut side, cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32), "cuStreamCreate")?;
            let image = {
                let mut v = vec![0u64; KERNEL_IMAGE.len().div_ceil(8)];
                std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, KERNEL_IMAGE.len()).copy_from_slice(KERNEL_IMAGE);
                v
            };
            let mut module = std::ptr::null_mut();
            check(cu::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void), "cuModuleLoadData")?;
            Ok(Gpu { ordinal, name, sms, arch: (major * 10 + minor) as u32, ctx, stream, side, module, blas: Cell::new(std::ptr::null_mut()) })
        }
    }

    /// visible CUDA devices
    pub fn count() -> Result<i32> {
        let mut n = 0;
        unsafe {
            check(cu::cuInit(0), "cuInit")?;
            check(cu::cuDeviceGetCount(&mut n), "cuDeviceGetCount")?;
        }
        Ok(n)
    }

    /// host -> device bandwidth from pinned memory in GB/s (best of 3 copies of `stage`), which tells the link
    /// width apart (x16 vs x4)
    pub fn h2d_gbs(&self, stage: &HostBuf) -> Result<f64> {
        let d = self.alloc(stage.bytes)?;
        let mut best = 0f64;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            self.htod(d.ptr, stage.as_slice::<u8>())?;
            self.sync()?;
            best = best.max(stage.bytes as f64 / t.elapsed().as_secs_f64() / 1e9);
        }
        Ok(best)
    }

    pub fn bind(&self) {
        unsafe { cu::cuCtxSetCurrent(self.ctx) };
    }

    /// Start capturing this stream's work into a graph (relaxed mode: other threads may keep using CUDA).
    pub fn begin_capture(&self) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuStreamBeginCapture_v2(self.stream, cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED), "cuStreamBeginCapture") }
    }

    /// End the capture and instantiate it.
    pub fn end_capture(&self) -> Result<GraphExec> {
        self.bind();
        unsafe {
            let mut g = std::ptr::null_mut();
            check(cu::cuStreamEndCapture(self.stream, &mut g), "cuStreamEndCapture")?;
            let mut e = std::ptr::null_mut();
            check(cu::cuGraphInstantiateWithFlags(&mut e, g, 0), "cuGraphInstantiate")?;
            cu::cuGraphDestroy(g);
            Ok(GraphExec(e))
        }
    }

    pub fn launch_graph(&self, g: &GraphExec) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuGraphLaunch(g.0, self.stream), "cuGraphLaunch") }
    }

    /// Let this context's kernels and copies reach `other`'s memory (NVLink / PCIe peer access).
    pub fn enable_peer(&self, other: &Gpu) -> Result<()> {
        self.bind();
        let r = unsafe { cu::cuCtxEnablePeerAccess(other.ctx, 0) };
        if r == cu::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
            return Ok(());
        }
        check(r, "cuCtxEnablePeerAccess")
    }

    fn blas(&self) -> Result<cb::cublasHandle_t> {
        if self.blas.get().is_null() {
            self.bind();
            let mut h = std::ptr::null_mut();
            unsafe {
                let st = cb::cublasCreate_v2(&mut h);
                if st != cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    bail!("cublasCreate: {st:?}");
                }
                cb::cublasSetStream_v2(h, self.stream.cast());
            }
            self.blas.set(h);
        }
        Ok(self.blas.get())
    }

    /// y[rows][ldy] (f32) = x[rows][inp] (f16) . w[out][inp]^T (f16), f32 accumulation (tensor cores)
    pub fn gemm_f16(&self, x: u64, w: u64, y: u64, rows: usize, inp: usize, out: usize, ldy: usize) -> Result<()> {
        let h = self.blas()?;
        self.bind();
        let (alpha, beta) = (1f32, 0f32);
        // column-major view: y^T (out x rows) = W (out x inp) . x^T (inp x rows)
        let st = unsafe {
            cb::cublasGemmEx(
                h,
                cb::cublasOperation_t::CUBLAS_OP_T,
                cb::cublasOperation_t::CUBLAS_OP_N,
                out as i32,
                rows as i32,
                inp as i32,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                cb::cudaDataType_t::CUDA_R_16F,
                inp as i32,
                x as *const c_void,
                cb::cudaDataType_t::CUDA_R_16F,
                inp as i32,
                (&beta as *const f32).cast(),
                y as *mut c_void,
                cb::cudaDataType_t::CUDA_R_32F,
                ldy as i32,
                cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if st != cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!("cublasGemmEx: {st:?}");
        }
        Ok(())
    }

    /// y[rows][ldy] (f32) = x[rows][inp] (bf16) . w[out][inp]^T (bf16), f32 accumulation
    pub fn gemm_bf16(&self, x: u64, w: u64, y: u64, rows: usize, inp: usize, out: usize, ldy: usize) -> Result<()> {
        let h = self.blas()?;
        self.bind();
        let (alpha, beta) = (1f32, 0f32);
        // column-major view: y^T (out x rows) = W (out x inp) . x^T (inp x rows)
        let st = unsafe {
            cb::cublasGemmEx(
                h,
                cb::cublasOperation_t::CUBLAS_OP_T,
                cb::cublasOperation_t::CUBLAS_OP_N,
                out as i32,
                rows as i32,
                inp as i32,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                cb::cudaDataType_t::CUDA_R_16BF,
                inp as i32,
                x as *const c_void,
                cb::cudaDataType_t::CUDA_R_16BF,
                inp as i32,
                (&beta as *const f32).cast(),
                y as *mut c_void,
                cb::cudaDataType_t::CUDA_R_32F,
                ldy as i32,
                cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if st != cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!("cublasGemmEx: {st:?}");
        }
        Ok(())
    }

    /// Batched bf16 GEMM, f32 out, `batch` problems at fixed strides (elements):
    /// y[rows][ldy] = x[rows][ldx] . op(w), with op(w) = w[out][ldw]^T (`w_t`) or w[inp][ldw] (not `w_t`)
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_sb(&self, w_t: bool, x: u64, w: u64, y: u64, rows: usize, inp: usize, out: usize, ldx: usize, ldw: usize, ldy: usize,
                        sx: usize, sw: usize, sy: usize, batch: usize) -> Result<()> {
        let h = self.blas()?;
        self.bind();
        let (alpha, beta) = (1f32, 0f32);
        // column-major view: y^T (out x rows) = op(W) (out x inp) . x^T (inp x rows)
        let st = unsafe {
            cb::cublasGemmStridedBatchedEx(
                h,
                if w_t { cb::cublasOperation_t::CUBLAS_OP_T } else { cb::cublasOperation_t::CUBLAS_OP_N },
                cb::cublasOperation_t::CUBLAS_OP_N,
                out as i32,
                rows as i32,
                inp as i32,
                (&alpha as *const f32).cast(),
                w as *const c_void,
                cb::cudaDataType_t::CUDA_R_16BF,
                ldw as i32,
                sw as i64,
                x as *const c_void,
                cb::cudaDataType_t::CUDA_R_16BF,
                ldx as i32,
                sx as i64,
                (&beta as *const f32).cast(),
                y as *mut c_void,
                cb::cudaDataType_t::CUDA_R_32F,
                ldy as i32,
                sy as i64,
                batch as i32,
                cb::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cb::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if st != cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!("cublasGemmStridedBatchedEx: {st:?}");
        }
        Ok(())
    }

    /// y[rows][ldy] (f32) = x[rows][inp] (f32) . w[out][inp]^T (f32), exact f32 (no TF32)
    pub fn gemm_f32(&self, x: u64, w: u64, y: u64, rows: usize, inp: usize, out: usize, ldy: usize) -> Result<()> {
        let h = self.blas()?;
        self.bind();
        let (alpha, beta) = (1f32, 0f32);
        let st = unsafe {
            cb::cublasSgemm_v2(
                h,
                cb::cublasOperation_t::CUBLAS_OP_T,
                cb::cublasOperation_t::CUBLAS_OP_N,
                out as i32,
                rows as i32,
                inp as i32,
                &alpha,
                w as *const f32,
                inp as i32,
                x as *const f32,
                inp as i32,
                &beta,
                y as *mut f32,
                ldy as i32,
            )
        };
        if st != cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            bail!("cublasSgemm: {st:?}");
        }
        Ok(())
    }

    pub fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream), "cuMemcpyDtoDAsync") }
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel> {
        let c = CString::new(name)?;
        let mut f = std::ptr::null_mut();
        unsafe { check(cu::cuModuleGetFunction(&mut f, self.module, c.as_ptr()), name)? };
        Ok(Kernel(f))
    }

    pub fn set_smem(&self, k: Kernel, bytes: usize) -> Result<()> {
        unsafe {
            check(
                cu::cuFuncSetAttribute(
                    k.0,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                ),
                "cuFuncSetAttribute",
            )
        }
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        self.bind();
        let mut p = 0;
        unsafe { check(cu::cuMemAlloc_v2(&mut p, bytes.max(1)), "cuMemAlloc")? };
        Ok(Buf { ptr: p, bytes, ctx: self.ctx })
    }

    pub fn alloc_zeroed(&self, bytes: usize) -> Result<Buf> {
        let b = self.alloc(bytes)?;
        self.memset(b.ptr, bytes)?;
        Ok(b)
    }

    pub fn memset(&self, ptr: u64, bytes: usize) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuMemsetD8Async(ptr, 0, bytes, self.stream), "cuMemsetD8Async") }
    }

    /// New buffer holding `data`, in stream order (a NULL-stream cuMemcpyHtoD from pageable memory can return before
    /// its DMA lands, and our non-blocking stream does not wait for it: kernels read stale tails).
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buf> {
        let b = self.alloc(std::mem::size_of_val(data))?;
        self.htod_sync(b.ptr, data)?;
        Ok(b)
    }

    /// synchronous copy host -> device (any host memory)
    /// Copy in stream order (after everything already queued) and wait: `src` may be reused on return.
    /// (A plain cuMemcpyHtoD runs on the NULL stream, which does not order against our non-blocking stream.)
    pub fn htod_sync<T: Copy>(&self, dst: u64, src: &[T]) -> Result<()> {
        self.bind();
        unsafe {
            check(cu::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const c_void, std::mem::size_of_val(src), self.stream), "cuMemcpyHtoDAsync")?;
            check(cu::cuStreamSynchronize(self.stream), "cuStreamSynchronize")
        }
    }

    /// async copy on the stream; `src` must be page-locked (registered) for real asynchrony
    pub fn htod<T: Copy>(&self, dst: u64, src: &[T]) -> Result<()> {
        self.bind();
        unsafe {
            check(
                cu::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const c_void, std::mem::size_of_val(src), self.stream),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    pub fn dtoh<T: Copy>(&self, dst: &mut [T], src: u64) -> Result<()> {
        self.bind();
        unsafe {
            check(
                cu::cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src, std::mem::size_of_val(dst), self.stream),
                "cuMemcpyDtoHAsync",
            )
        }
    }

    pub fn download<T: Copy + Default>(&self, src: u64, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        self.sync()?;
        unsafe { check(cu::cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, src, n * std::mem::size_of::<T>()), "cuMemcpyDtoH")? };
        Ok(v)
    }

    pub fn sync(&self) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuStreamSynchronize(self.stream), "cuStreamSynchronize") }
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.bind();
        let (mut free, mut total) = (0, 0);
        unsafe { check(cu::cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo")? };
        Ok((free, total))
    }

    pub fn event(&self) -> Result<Event> {
        self.bind();
        let mut e = std::ptr::null_mut();
        unsafe { check(cu::cuEventCreate(&mut e, cu::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32), "cuEventCreate")? };
        Ok(Event(e))
    }

    /// an event with timing enabled (for `elapsed_ms`)
    pub fn timed_event(&self) -> Result<Event> {
        self.bind();
        let mut e = std::ptr::null_mut();
        unsafe { check(cu::cuEventCreate(&mut e, 0), "cuEventCreate")? };
        Ok(Event(e))
    }

    /// milliseconds between two completed timed events
    pub fn elapsed_ms(&self, a: Event, b: Event) -> Result<f32> {
        let mut ms = 0f32;
        unsafe { check(cu::cuEventElapsedTime_v2(&mut ms, a.0, b.0), "cuEventElapsedTime")? };
        Ok(ms)
    }

    /// record `e` on the main (side = false) or side stream
    pub fn record(&self, e: Event, side: bool) -> Result<()> {
        unsafe { check(cu::cuEventRecord(e.0, if side { self.side } else { self.stream }), "cuEventRecord") }
    }

    /// make the main (side = false) or side stream wait for `e`
    pub fn wait(&self, e: Event, side: bool) -> Result<()> {
        unsafe { check(cu::cuStreamWaitEvent(if side { self.side } else { self.stream }, e.0, 0), "cuStreamWaitEvent") }
    }

    /// async host->device copy on the side stream (`src` should be page-locked to overlap)
    pub fn htod_side<T: Copy>(&self, dst: u64, src: &[T]) -> Result<()> {
        self.bind();
        unsafe {
            check(
                cu::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const c_void, std::mem::size_of_val(src), self.side),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    /// Stream waits until the u32 at `addr` (device-visible, e.g. mapped host memory) is >= `value`.
    pub fn wait_value(&self, addr: u64, value: u32) -> Result<()> {
        unsafe {
            check(
                cu::cuStreamWaitValue32_v2(self.stream, addr, value, cu::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_GEQ as u32),
                "cuStreamWaitValue32",
            )
        }
    }

    /// Stream writes `value` to the u32 at `addr` once prior work on the stream is done.
    pub fn write_value(&self, addr: u64, value: u32) -> Result<()> {
        unsafe { check(cu::cuStreamWriteValue32_v2(self.stream, addr, value, 0), "cuStreamWriteValue32") }
    }

    pub fn launch(&self, k: Kernel, grid: (u32, u32, u32), block: u32, smem: u32, args: &[Arg]) -> Result<()> {
        self.launch_on(self.stream, k, grid, block, smem, args)
    }

    /// launch on the side stream (background work that must not wait behind the main stream)
    pub fn launch_side(&self, k: Kernel, grid: (u32, u32, u32), block: u32, smem: u32, args: &[Arg]) -> Result<()> {
        self.launch_on(self.side, k, grid, block, smem, args)
    }

    /// true once all work recorded before `e` has finished (non-blocking)
    pub fn done(&self, e: Event) -> Result<bool> {
        self.bind();
        match unsafe { cu::cuEventQuery(e.0) } {
            cu::CUresult::CUDA_SUCCESS => Ok(true),
            cu::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            r => check(r, "cuEventQuery").map(|_| false),
        }
    }

    /// host waits for the work recorded before `e` (and nothing queued after it)
    pub fn event_sync(&self, e: Event) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuEventSynchronize(e.0), "cuEventSynchronize") }
    }

    pub fn side_sync(&self) -> Result<()> {
        self.bind();
        unsafe { check(cu::cuStreamSynchronize(self.side), "cuStreamSynchronize") }
    }

    fn launch_on(&self, stream: cu::CUstream, k: Kernel, grid: (u32, u32, u32), block: u32, smem: u32, args: &[Arg]) -> Result<()> {
        let mut vals = [0u64; 32];
        let mut ptrs = [std::ptr::null_mut::<c_void>(); 32];
        for (i, a) in args.iter().enumerate() {
            vals[i] = match *a {
                Arg::P(p) => p,
                Arg::I(v) => v as u32 as u64,
                Arg::F(v) => v.to_bits() as u64,
            };
        }
        for i in 0..args.len() {
            ptrs[i] = (&mut vals[i] as *mut u64).cast();
        }
        self.bind();
        unsafe {
            check(
                cu::cuLaunchKernel(k.0, grid.0, grid.1, grid.2, block, 1, 1, smem, stream, ptrs.as_mut_ptr(), std::ptr::null_mut()),
                "cuLaunchKernel",
            )
        }
    }
}

/// Host memory registered with CUDA (page-locked, DMA-able). Allocated 2 MB-aligned with transparent huge pages:
/// on a 2x RTX 3090 host the IOMMU halves DMA bandwidth from 4 KB-page memory (13 vs 24.7 GB/s measured).
pub struct HostBuf {
    ptr: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

impl HostBuf {
    pub fn new(bytes: usize) -> Result<HostBuf> {
        let align = 2 << 20;
        let size = bytes.div_ceil(align) * align;
        unsafe {
            let layout = std::alloc::Layout::from_size_align(size, align)?;
            let ptr = std::alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                bail!("host alloc of {size} bytes failed");
            }
            libc_madvise(ptr, size);
            // PORTABLE | DEVICEMAP: usable from every context, and addressable by kernels (zero-copy)
            check(cu::cuMemHostRegister_v2(ptr as *mut c_void, size, 1 | 2), "cuMemHostRegister")?;
            Ok(HostBuf { ptr, bytes: size })
        }
    }
    pub fn host_ptr(&self) -> *mut u8 {
        self.ptr
    }
    /// The device address of this memory in `gpu`'s context (zero-copy access from kernels).
    pub fn dev_ptr(&self, gpu: &Gpu) -> Result<u64> {
        gpu.bind();
        let mut d = 0;
        unsafe { check(cu::cuMemHostGetDevicePointer_v2(&mut d, self.ptr as *mut c_void, 0), "cuMemHostGetDevicePointer")? };
        Ok(d)
    }
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.bytes / std::mem::size_of::<T>()) }
    }
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe {
            cu::cuMemHostUnregister(self.ptr as *mut c_void);
            std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.bytes, 2 << 20).unwrap());
        }
    }
}

unsafe extern "C" {
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

unsafe fn libc_madvise(p: *mut u8, len: usize) {
    const MADV_HUGEPAGE: i32 = 14;
    unsafe { madvise(p as *mut c_void, len, MADV_HUGEPAGE) };
}
