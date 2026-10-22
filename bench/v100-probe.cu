// Hardware probes for the 8x V100 box: HBM read BW, P2P/H2D copies, launch overhead,
// cuBLAS FP16 GEMM at MoE shapes, and a naive 2-bit / 4-bit GEMV (weights unpacked to half2 in registers).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

__global__ void read_bw(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = {0,0,0,0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldg(p + i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345) out[0] = 1;
}
__global__ void empty_k() {}

// y[r] = sum_k W[r,k] * x[k]; W packed: bits per weight b in {2,4}, groups of 32 weights share a half scale.
// One warp per row; each lane reads 16 bytes (a uint4) per step.
template <int BITS, int BATCH>
__global__ void gemv_lowbit(const uint4* __restrict__ W, const half* __restrict__ scales, const half* __restrict__ x,
                            float* __restrict__ y, int rows, int K) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= rows) return;
  constexpr int WPU = 128 / BITS;            // weights per uint4
  const int units = K / WPU;
  const uint4* wr = W + (size_t)warp * units;
  const half* sr = scales + (size_t)warp * (K / 32);
  float acc[BATCH] = {0};
  for (int u = lane; u < units; u += 32) {
    uint4 q = __ldg(wr + u);
    uint32_t qs[4] = {q.x, q.y, q.z, q.w};
    int k0 = u * WPU;
    half s = sr[k0 / 32];
    half2 s2 = __half2half2(s);
#pragma unroll
    for (int b = 0; b < BATCH; b++) {
      const half2* xb = reinterpret_cast<const half2*>(x + (size_t)b * K + k0);
      half2 h = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 4; w++) {
        uint32_t v = qs[w];
#pragma unroll
        for (int j = 0; j < 32 / BITS; j += 2) {
          int a = (v >> (j * BITS)) & ((1 << BITS) - 1), c = (v >> ((j + 1) * BITS)) & ((1 << BITS) - 1);
          half2 wv = __halves2half2(__int2half_rn(a - (1 << (BITS - 1))), __int2half_rn(c - (1 << (BITS - 1))));
          h = __hfma2(wv, xb[(w * (32 / BITS) + j) / 2], h);
        }
      }
      h = __hmul2(h, s2);
      acc[b] += __low2float(h) + __high2float(h);
    }
  }
#pragma unroll
  for (int b = 0; b < BATCH; b++) {
    float v = acc[b];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (lane == 0) y[(size_t)b * rows + warp] = v;
  }
}

template <int BITS, int BATCH> void run_gemv(int rows, int K) {
  size_t wbytes = (size_t)rows * K * BITS / 8;
  uint4* W; half *s, *x; float* y;
  CK(cudaMalloc(&W, wbytes)); CK(cudaMalloc(&s, (size_t)rows * K / 32 * 2));
  CK(cudaMalloc(&x, (size_t)BATCH * K * 2)); CK(cudaMalloc(&y, (size_t)BATCH * rows * 4));
  CK(cudaMemset(W, 0x5a, wbytes)); CK(cudaMemset(s, 0x10, (size_t)rows * K / 32 * 2)); CK(cudaMemset(x, 0x10, (size_t)BATCH * K * 2));
  int threads = 256, blocks = (rows * 32 + threads - 1) / threads;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 3; i++) gemv_lowbit<BITS, BATCH><<<blocks, threads>>>(W, s, x, y, rows, K);
  cudaEventRecord(a);
  int it = 20;
  for (int i = 0; i < it; i++) gemv_lowbit<BITS, BATCH><<<blocks, threads>>>(W, s, x, y, rows, K);
  cudaEventRecord(b); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b); ms /= it;
  double bytes = wbytes + (size_t)rows * K / 32 * 2;
  printf("gemv %d-bit batch %d  rows %d K %d: %.3f ms, %.0f GB/s effective weight read\n", BITS, BATCH, rows, K, ms, bytes / ms / 1e6);
  cudaFree(W); cudaFree(s); cudaFree(x); cudaFree(y);
}

int main() {
  int ng; CK(cudaGetDeviceCount(&ng));
  // 1. HBM read bandwidth on GPU 0
  {
    CK(cudaSetDevice(0));
    size_t bytes = 8ull << 30, n = bytes / 16; int4* p; int* o;
    CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 4)); CK(cudaMemset(p, 1, bytes));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    read_bw<<<80 * 8, 512>>>(p, n, o); cudaEventRecord(a);
    for (int i = 0; i < 5; i++) read_bw<<<80 * 8, 512>>>(p, n, o);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b);
    printf("HBM read GPU0: %.0f GB/s\n", 5.0 * bytes / ms / 1e6);
    cudaFree(p); cudaFree(o);
  }
  // 2. launch overhead: plain vs CUDA graph of 1000 empty kernels
  {
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 100; i++) empty_k<<<1, 32, 0, st>>>();
    cudaEventRecord(a, st); for (int i = 0; i < 1000; i++) empty_k<<<1, 32, 0, st>>>(); cudaEventRecord(b, st);
    CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b);
    printf("1000 empty kernels, stream: %.2f ms\n", ms);
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 1000; i++) empty_k<<<1, 32, 0, st>>>();
    cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
    cudaGraphLaunch(ge, st); cudaStreamSynchronize(st);
    cudaEventRecord(a, st); cudaGraphLaunch(ge, st); cudaEventRecord(b, st);
    CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
    printf("1000 empty kernels, CUDA graph: %.2f ms\n", ms);
  }
  // 3. P2P copy bandwidth/latency from GPU0 to each peer, and H2D pinned
  {
    size_t big = 1ull << 30, small = 16 << 10;
    for (int d = 1; d < ng; d++) {
      int can; cudaDeviceCanAccessPeer(&can, 0, d);
      CK(cudaSetDevice(0)); if (can) cudaDeviceEnablePeerAccess(d, 0);
      void *s0, *dd; CK(cudaMalloc(&s0, big)); CK(cudaSetDevice(d)); CK(cudaMalloc(&dd, big)); CK(cudaSetDevice(0));
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
      cudaMemcpyPeer(dd, d, s0, 0, big); cudaEventRecord(a);
      for (int i = 0; i < 3; i++) cudaMemcpyPeer(dd, d, s0, 0, big);
      cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b);
      float ms2; cudaEventRecord(a); for (int i = 0; i < 200; i++) cudaMemcpyPeerAsync(dd, d, s0, 0, small); cudaEventRecord(b);
      CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms2, a, b);
      printf("P2P 0->%d (peer=%d): %.1f GB/s, 16 KB copy %.1f us\n", d, can, 3.0 * big / ms / 1e6, ms2 * 1000 / 200);
      cudaFree(s0); CK(cudaSetDevice(d)); cudaFree(dd); CK(cudaSetDevice(0));
    }
    void* h; void* dv; CK(cudaMallocHost(&h, big)); CK(cudaMalloc(&dv, big));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaMemcpy(dv, h, big, cudaMemcpyHostToDevice); cudaEventRecord(a);
    for (int i = 0; i < 3; i++) cudaMemcpy(dv, h, big, cudaMemcpyHostToDevice);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b);
    printf("H2D pinned GPU0: %.1f GB/s\n", 3.0 * big / ms / 1e6);
    cudaFreeHost(h); cudaFree(dv);
  }
  // 4. cuBLAS FP16 GEMM (FP32 accumulate) at MoE-expert shapes: [M x 5120] x [5120 x 2304]
  {
    CK(cudaSetDevice(0)); cublasHandle_t hb; cublasCreate(&hb);
    int N = 2304, K = 5120;
    for (int M : {8, 32, 128, 512, 2048, 8192}) {
      half *A, *B, *C; CK(cudaMalloc(&A, (size_t)M * K * 2)); CK(cudaMalloc(&B, (size_t)K * N * 2)); CK(cudaMalloc(&C, (size_t)M * N * 2));
      float al = 1, be = 0;
      auto go = [&] { cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &be, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
      go(); cudaDeviceSynchronize();
      cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
      for (int i = 0; i < 20; i++) go();
      cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
      printf("FP16 GEMM M=%d: %.3f ms, %.1f TFLOPS\n", M, ms, 2.0 * M * N * K / ms / 1e9);
      cudaFree(A); cudaFree(B); cudaFree(C);
    }
  }
  // 5. naive low-bit GEMV (expert-sized and dense-sized rows)
  run_gemv<2, 1>(16384, 5120); run_gemv<2, 4>(16384, 5120); run_gemv<2, 8>(16384, 5120);
  run_gemv<4, 1>(16384, 5120); run_gemv<4, 4>(16384, 5120); run_gemv<4, 8>(16384, 5120);
  return 0;
}
