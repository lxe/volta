// Low-bit GEMV with magic-number dequant (OR the field into the mantissa of fp16 1024.0, subtract 1024+offset):
// one shift+lop3 and one hsub2 per 2 weights. A warp owns ROWS rows and walks K, so each x load serves ROWS rows.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

template <int BITS> __device__ __forceinline__ void unpack(uint32_t q, half2* w) {
  constexpr int N = 16 / BITS;                       // half2 outputs per uint32
  constexpr uint32_t MASK = ((1u << BITS) - 1) * 0x00010001u;
  const half2 off = __float2half2_rn(1024.f + (1 << (BITS - 1)));
#pragma unroll
  for (int i = 0; i < N; i++) {
    uint32_t t = ((q >> (BITS * i)) & MASK) | 0x64006400u;
    w[i] = __hsub2(*reinterpret_cast<half2*>(&t), off);
  }
}

// W layout: [rows/ROWS][K/(128/BITS)][ROWS] uint4 (row-interleaved so a lane's ROWS loads are contiguous).
template <int BITS, int BATCH, int ROWS>
__global__ void gemv(const uint4* __restrict__ W, const half* __restrict__ sc, const half* __restrict__ x,
                     float* __restrict__ y, int rows, int K) {
  constexpr int WPU = 128 / BITS;                    // weights per uint4
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp * ROWS >= rows) return;
  const int units = K / WPU;
  float acc[ROWS][BATCH];
#pragma unroll
  for (int r = 0; r < ROWS; r++)
#pragma unroll
    for (int b = 0; b < BATCH; b++) acc[r][b] = 0.f;
  const uint4* wbase = W + (size_t)warp * units * ROWS;
  for (int u = lane; u < units; u += 32) {
    uint4 q[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) q[r] = __ldg(wbase + (size_t)u * ROWS + r);
    const int k0 = u * WPU;
#pragma unroll
    for (int b = 0; b < BATCH; b++) {
      // x for these WPU weights, as half2 pairs (layout matches unpack's lane order by construction of the packer)
      half2 xv[WPU / 2];
      const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)b * K + k0);
#pragma unroll
      for (int i = 0; i < WPU / 8; i++) { uint4 t = __ldg(xp + i); memcpy(&xv[i * 4], &t, 16); }
#pragma unroll
      for (int r = 0; r < ROWS; r++) {
        uint32_t qs[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
        half2 h = __float2half2_rn(0.f);
#pragma unroll
        for (int w = 0; w < 4; w++) {
          half2 wv[16 / BITS];
          unpack<BITS>(qs[w], wv);
#pragma unroll
          for (int i = 0; i < 16 / BITS; i++) h = __hfma2(wv[i], xv[w * (16 / BITS) + i], h);
        }
        // one scale per 32 weights; WPU = 32 (4-bit) or 64 (2-bit) -> 1 or 2 groups, approximate with one here
        half s = sc[(size_t)(warp * ROWS + r) * (K / 32) + k0 / 32];
        h = __hmul2(h, __half2half2(s));
        acc[r][b] += __low2float(h) + __high2float(h);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; r++)
#pragma unroll
    for (int b = 0; b < BATCH; b++) {
      float v = acc[r][b];
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
      if (lane == 0) y[(size_t)b * rows + warp * ROWS + r] = v;
    }
}

template <int BITS, int BATCH, int ROWS> void run(int rows, int K) {
  size_t wbytes = (size_t)rows * K * BITS / 8, sbytes = (size_t)rows * K / 32 * 2;
  uint4* W; half *s, *x; float* y;
  CK(cudaMalloc(&W, wbytes)); CK(cudaMalloc(&s, sbytes)); CK(cudaMalloc(&x, (size_t)BATCH * K * 2)); CK(cudaMalloc(&y, (size_t)BATCH * rows * 4));
  CK(cudaMemset(W, 0x5a, wbytes)); CK(cudaMemset(s, 0x10, sbytes)); CK(cudaMemset(x, 0x10, (size_t)BATCH * K * 2));
  int threads = 256, warps = rows / ROWS, blocks = (warps * 32 + threads - 1) / threads;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 3; i++) gemv<BITS, BATCH, ROWS><<<blocks, threads>>>(W, s, x, y, rows, K);
  CK(cudaGetLastError());
  cudaEventRecord(a); for (int i = 0; i < 20; i++) gemv<BITS, BATCH, ROWS><<<blocks, threads>>>(W, s, x, y, rows, K);
  cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
  printf("%d-bit batch %d rows/warp %d: %.3f ms, %4.0f GB/s (weights+scales)\n", BITS, BATCH, ROWS, ms, (wbytes + sbytes) / ms / 1e6);
  cudaFree(W); cudaFree(s); cudaFree(x); cudaFree(y);
}

int main() {
  const int R = 65536, K = 5120;   // 335M weights: 84 MB at 2-bit, 168 MB at 4-bit (well past L2)
  run<4, 1, 4>(R, K); run<4, 2, 4>(R, K); run<4, 4, 4>(R, K); run<4, 8, 4>(R, K);
  run<4, 1, 2>(R, K); run<4, 4, 2>(R, K);
  run<2, 1, 4>(R, K); run<2, 2, 4>(R, K); run<2, 4, 4>(R, K); run<2, 8, 4>(R, K);
  run<2, 1, 2>(R, K); run<2, 4, 2>(R, K);
  return 0;
}
