// Expert GEMM for 1-32 tokens per expert with Volta mma.m8n8k4 and weights unpacked straight into registers.
// y[t, row] = sum_k x[t, k] * W[row, k]; W 4-bit or 2-bit, one fp16 scale per 32 weights, offset-binary codes.
// A operand = weights (each thread holds one weight row's 4 k values), B operand = x (one token's 4 k values).
// Dequant per half2: shift, lop3 (and/or into fp16 1024.0), hfma2 with (scale, -(1024+off)*scale).
// x sits in shared memory with k permuted to match the order the unpack produces, so B loads are 8-byte vectors.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

constexpr int WARPS = 8;
#ifndef KCV
#define KCV 1024
#endif
constexpr int KC = KCV;

__device__ __forceinline__ void mma884(float* d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
template <int BITS> __device__ __forceinline__ uint32_t deq(uint32_t v, int i, half2 s2, half2 nb2) {
  constexpr uint32_t MASK = ((1u << BITS) - 1) * 0x00010001u;
  uint32_t t = ((v >> (BITS * i)) & MASK) | 0x64006400u;
#ifdef FMA_DEQ
  half2 h = __hfma2(*reinterpret_cast<half2*>(&t), s2, nb2);
#else
  half2 h = __hmul2(__hsub2(*reinterpret_cast<half2*>(&t), nb2), s2);
#endif
  return *reinterpret_cast<uint32_t*>(&h);
}
// k position (within a uint32 word) of the j-th element of the unpacked stream: pairs (f_i, f_{i+F/2})
template <int BITS> __host__ __device__ inline int perm(int j) { constexpr int F = 32 / BITS; int i = j >> 1; return (j & 1) ? i + F / 2 : i; }

template <int BITS, int NT, int R>
__global__ void __launch_bounds__(WARPS * 32) moe(const uint4* __restrict__ W, const half* __restrict__ sc,
                                                  const half* __restrict__ x, float* __restrict__ y, int rows, int K, int ntok) {
  constexpr int WPU = 128 / BITS, F = 32 / BITS, LDX = KC + 8, UNITS = KC / WPU;
  extern __shared__ __align__(16) half xs[];       // [NT*8][LDX], k permuted per 32-bit word group
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = (lane >> 2) & 3, h = lane >> 4;
  const int k0 = blockIdx.y * KC;
  const int rowslot = 8 * q + (lane & 3) + 4 * h, tok = (lane & 3) + 4 * h;
  const int rowbase = (blockIdx.x * WARPS + warp) * 32 * R;
  const int units_all = K / WPU, u0 = k0 / WPU;
  const uint4* wp[R]; const half* sp[R];
#pragma unroll
  for (int j = 0; j < R; j++) { int row = rowbase + j * 32 + rowslot;
    (void)row; size_t rg = (size_t)(rowbase / 32 + j);
    wp[j] = W + (rg * units_all + u0) * 32 + lane; sp[j] = sc + (rg * (K / 32) + k0 / 32) * 32 + lane; }
  uint4 qn[R];
#pragma unroll
  for (int j = 0; j < R; j++) qn[j] = __ldg(wp[j]);
  for (int i = threadIdx.x; i < NT * 8 * KC; i += WARPS * 32) {
    int t = i / KC, k = i % KC, g = k / F, j = k % F;             // destination slot j of word-group g
    xs[t * LDX + k] = t < ntok ? x[(size_t)t * K + k0 + g * F + perm<BITS>(j)] : __float2half(0.f);
  }
  __syncthreads();
  float acc[R][NT][8];
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][n][e] = 0.f;
  for (int u = 0; u < UNITS; u++) {
    uint4 qc[R];
#pragma unroll
    for (int j = 0; j < R; j++) { qc[j] = qn[j]; if (u + 1 < UNITS) qn[j] = __ldg(wp[j] + (size_t)(u + 1) * 32); }
    half2 s2[R][WPU / 32], nb[R][WPU / 32];
#pragma unroll
    for (int j = 0; j < R; j++)
#pragma unroll
      for (int g = 0; g < WPU / 32; g++) { half s = sp[j][(size_t)(u * (WPU / 32) + g) * 32]; s2[j][g] = __half2half2(s);
        
#ifdef FMA_DEQ
        nb[j][g] = __half2half2(__hmul(s, __float2half(-(1024.f + (1 << (BITS - 1))))));
#else
        nb[j][g] = __float2half2_rn(1024.f + (1 << (BITS - 1)));
#endif
      }
#pragma unroll
    for (int w = 0; w < 4; w++) {                 // 32-bit word w of the uint4
      const int g = w / (4 / (WPU / 32));
#pragma unroll
      for (int m = 0; m < F / 4; m++) {           // each word feeds F/4 mma k-steps of 4
        const int kpos = u * WPU + w * F + m * 4;
        uint32_t b[NT][2];
#pragma unroll
        for (int n = 0; n < NT; n++) { uint2 v = *reinterpret_cast<const uint2*>(&xs[(n * 8 + tok) * LDX + kpos]); b[n][0] = v.x; b[n][1] = v.y; }
#pragma unroll
        for (int j = 0; j < R; j++) {
          uint32_t v = (w == 0) ? qc[j].x : (w == 1) ? qc[j].y : (w == 2) ? qc[j].z : qc[j].w;
          uint32_t a0 = deq<BITS>(v, 2 * m, s2[j][g], nb[j][g]), a1 = deq<BITS>(v, 2 * m + 1, s2[j][g], nb[j][g]);
#pragma unroll
          for (int n = 0; n < NT; n++) mma884(acc[j][n], a0, a1, b[n][0], b[n][1]);
        }
      }
    }
  }
  // accumulator e of this lane = (row slot 8q + (lane&1) + 2*((e>>1)&1) + 4h, token 2*((lane>>1)&1) + (e&1) + 4*(e>>2))
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        int rs = 8 * q + (lane & 1) + 2 * ((e >> 1) & 1) + 4 * h, t = n * 8 + 2 * ((lane >> 1) & 1) + (e & 1) + 4 * (e >> 2);
        if (t < ntok) atomicAdd(&y[(size_t)t * rows + rowbase + j * 32 + rs], acc[j][n][e]);
      }
}

template <int BITS, int NT, int R> void run(int rows, int K, int ntok, bool verify) {
  size_t wbytes = (size_t)rows * K * BITS / 8, nsc = (size_t)rows * K / 32;
  std::vector<uint8_t> hw(wbytes); std::vector<half> hs(nsc), hx((size_t)ntok * K);
  srand(1); for (auto& b : hw) b = rand() & 255;
  for (auto& s : hs) s = __float2half(0.01f + (rand() % 100) * 1e-4f);
  for (auto& v : hx) v = __float2half(((rand() % 2001) - 1000) * 1e-3f);
  uint4* W; half *s, *x; float* y;
  CK(cudaMalloc(&W, wbytes)); CK(cudaMalloc(&s, nsc * 2)); CK(cudaMalloc(&x, hx.size() * 2)); CK(cudaMalloc(&y, (size_t)ntok * rows * 4));
  CK(cudaMemcpy(W, hw.data(), wbytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(s, hs.data(), nsc * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  dim3 grid(rows / (WARPS * 32 * R), K / KC); size_t sm = (size_t)NT * 8 * (KC + 8) * 2;
  CK(cudaFuncSetAttribute(moe<BITS, NT, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  CK(cudaMemset(y, 0, (size_t)ntok * rows * 4));
  moe<BITS, NT, R><<<grid, WARPS * 32, sm>>>(W, s, x, y, rows, K, ntok); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  double maxerr = 0, maxref = 0;
  if (verify) {
    std::vector<float> hy((size_t)ntok * rows); CK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
    const int off = 1 << (BITS - 1);
    for (int r = 0; r < rows; r += 37) for (int t = 0; t < ntok; t++) {
      double ref = 0;
      int rs = r % 32, ln = 4 * (rs >> 3) + (rs & 3) + 16 * ((rs >> 2) & 1); size_t rg = r / 32;
      const int WPU = 128 / BITS, F = 32 / BITS, units = K / WPU;
      for (int k = 0; k < K; k++) {
        int u = k / WPU, w = (k % WPU) / F, f = k % F;
        uint32_t word; memcpy(&word, &hw[((rg * units + u) * 32 + ln) * 16 + w * 4], 4);
        int code = (word >> (BITS * f)) & ((1 << BITS) - 1);
        float scl = __half2float(hs[(rg * (K / 32) + k / 32) * 32 + ln]);
        ref += (double)(code - off) * scl * __half2float(hx[(size_t)t * K + k]); }
      maxerr = fmax(maxerr, fabs(ref - hy[(size_t)t * rows + r])); maxref = fmax(maxref, fabs(ref)); }
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 3; i++) moe<BITS, NT, R><<<grid, WARPS * 32, sm>>>(W, s, x, y, rows, K, ntok);
  cudaEventRecord(a); for (int i = 0; i < 20; i++) moe<BITS, NT, R><<<grid, WARPS * 32, sm>>>(W, s, x, y, rows, K, ntok);
  cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
  printf("m884 %d-bit tok %2d (NT %d, R %d): %.3f ms, %4.0f GB/s%s", BITS, ntok, NT, R, ms, (wbytes + nsc * 2) / ms / 1e6, verify ? "" : "\n");
  if (verify) printf(", max err %.2e of %.2e\n", maxerr, maxref);
  cudaFree(W); cudaFree(s); cudaFree(x); cudaFree(y);
}

int main() {
  const int R_ = 65536, K = 5120;
  run<4, 1, 4>(R_, K, 8, false);
  run<4, 2, 2>(R_, K, 16, false); run<4, 2, 4>(R_, K, 16, false); run<4, 4, 1>(R_, K, 32, false); run<4, 4, 2>(R_, K, 32, false);
  run<2, 2, 2>(R_, K, 16, false); run<2, 2, 4>(R_, K, 16, false); run<2, 4, 1>(R_, K, 32, false); run<2, 4, 2>(R_, K, 32, false);
  return 0;
}
