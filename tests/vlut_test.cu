// Test for the lookup-table expert formats (engine/src/kernels/volta.cuh: vlut_*, vcb_quant, vlut_pack):
// random MXFP4 matrices -> (as is | cb3 | cb2) -> vlut kernels, against a CPU decode of the packed data, plus each
// codebook's error relative to the FP4 source; then the time of one decode step's experts at each format.
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels vlut_test.cu -o vlut_test
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_fp16.h>
#include "dsx.cu"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

static const float E2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
static const float CB3[8] = {-8.0f, -4.5f, -2.5f, -0.625f, 1.25f, 2.75f, 4.5f, 8.0f};
static const float CB2[4] = {-8.0f, -1.875f, 2.75f, 8.0f};
static float e4m3_dec(uint32_t c) { int E = (c >> 3) & 15, M = c & 7; return E == 0 ? M / 512.f : ldexpf(1.f + M / 8.f, E - 7); }

struct Fp4 { int n, k; std::vector<uint8_t> w, s; };
static Fp4 rand_fp4(int n, int k, unsigned seed) {
  Fp4 m{n, k}; srand(seed); m.w.resize((size_t)n * k / 2); m.s.resize((size_t)n * k / 32);
  for (auto& b : m.w) b = rand() & 255;
  for (auto& b : m.s) b = 118 + rand() % 6;  // 2^-9 .. 2^-4
  return m;
}
static float fp4_val(const Fp4& m, int r, int c) { uint8_t b = m.w[((size_t)r * m.k + c) / 2]; return E2M1[(c & 1) ? b >> 4 : b & 15] * ldexpf(1.f, (int)m.s[(size_t)r * (m.k / 32) + c / 32] - 127); }

int main() {
  const int N = 4608, K = 5120, T = 8;
  // device tables
  std::vector<half2> t4(256), t3(64), t2(16);
  for (int i = 0; i < 256; i++) t4[i] = __halves2half2(__float2half(E2M1[i & 15]), __float2half(E2M1[i >> 4]));
  for (int i = 0; i < 64; i++) t3[i] = __halves2half2(__float2half(CB3[i & 7]), __float2half(CB3[i >> 3]));
  for (int i = 0; i < 16; i++) t2[i] = __halves2half2(__float2half(CB2[i & 3]), __float2half(CB2[i >> 2]));
  std::vector<half> e43(256); for (int i = 0; i < 256; i++) e43[i] = __float2half(e4m3_dec(i & 127));
  half2 *dt4, *dt3, *dt2; half* de43; float *dcb3, *dcb2;
  CK(cudaMalloc(&dt4, 1024)); CK(cudaMalloc(&dt3, 256)); CK(cudaMalloc(&dt2, 64)); CK(cudaMalloc(&de43, 512)); CK(cudaMalloc(&dcb3, 32)); CK(cudaMalloc(&dcb2, 16));
  cudaMemcpy(dt4, t4.data(), 1024, cudaMemcpyHostToDevice); cudaMemcpy(dt3, t3.data(), 256, cudaMemcpyHostToDevice); cudaMemcpy(dt2, t2.data(), 64, cudaMemcpyHostToDevice);
  cudaMemcpy(de43, e43.data(), 512, cudaMemcpyHostToDevice); cudaMemcpy(dcb3, CB3, 32, cudaMemcpyHostToDevice); cudaMemcpy(dcb2, CB2, 16, cudaMemcpyHostToDevice);
  Fp4 m = rand_fp4(N, K, 7);
  uint8_t *dw, *ds, *dcodes, *dsb, *outs; uint32_t* outw;
  CK(cudaMalloc(&dw, m.w.size())); CK(cudaMalloc(&ds, m.s.size())); CK(cudaMalloc(&dcodes, (size_t)N * K)); CK(cudaMalloc(&dsb, (size_t)N * K / 32));
  CK(cudaMalloc(&outw, (size_t)N * K / 2)); CK(cudaMalloc(&outs, (size_t)N * K / 32));
  cudaMemcpy(dw, m.w.data(), m.w.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, m.s.data(), m.s.size(), cudaMemcpyHostToDevice);
  std::vector<half> X((size_t)T * K); for (auto& v : X) v = __float2half(((rand() % 2001) - 1000) * 1e-3f);
  half* dX; float* dY; CK(cudaMalloc(&dX, X.size() * 2)); CK(cudaMalloc(&dY, (size_t)T * N * 4));
  cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice);
  int* rows; CK(cudaMalloc(&rows, T * 4)); std::vector<int> ri(T); for (int i = 0; i < T; i++) ri[i] = i; cudaMemcpy(rows, ri.data(), T * 4, cudaMemcpyHostToDevice);
  for (int fmt = 0; fmt < 3; fmt++) {
    const int units = K / 32, groups = N * units;
    float mul = 1.f;
    if (fmt == VF_FP4) {
      vlut_pack<<<(groups + 255) / 256, 256>>>(VF_FP4, dw, ds, N, K, outw, outs);
    } else {
      float smax = 0; for (auto b : m.s) smax = fmaxf(smax, ldexpf(1.f, (int)b - 127));
      mul = smax * 6.f / 8.f * 1.25f / 256.f;
      vcb_quant<<<(groups + 255) / 256, 256>>>(dw, ds, N, K, fmt == VF_CB3 ? dcb3 : dcb2, fmt == VF_CB3 ? 8 : 4, mul, dcodes, dsb);
      vlut_pack<<<(groups + 255) / 256, 256>>>(fmt, dcodes, dsb, N, K, outw, outs);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    // host copies of what was packed (codes + scales) for the reference
    std::vector<uint8_t> codes((size_t)N * K), sb((size_t)N * K / 32);
    if (fmt != VF_FP4) { cudaMemcpy(codes.data(), dcodes, codes.size(), cudaMemcpyDeviceToHost); cudaMemcpy(sb.data(), dsb, sb.size(), cudaMemcpyDeviceToHost); }
    // 48 work entries over the same matrix (a layer's worth of experts on one GPU) so the launch fills the GPU
    const int E = 48;
    std::vector<VWork2> wks(E, VWork2{0, 0, 0, (unsigned)T, mul, {0, 0, 0}}); VWork2* dwk; CK(cudaMalloc(&dwk, E * sizeof(VWork2))); cudaMemcpy(dwk, wks.data(), E * sizeof(VWork2), cudaMemcpyHostToDevice);
    const int KC = 1024; size_t sm = 1024 + 512 + (size_t)8 * (KC + 8) * 2;
    dim3 g((N + 1023) / 1024, 1, K / KC), gE((N + 1023) / 1024, E, K / KC);
    const half2* tab = fmt == VF_FP4 ? dt4 : fmt == VF_CB3 ? dt3 : dt2;
    auto go = [&] {
      cudaMemset(dY, 0, (size_t)T * N * 4);
      if (fmt == VF_FP4) vlut_fp4_n1<<<g, 256, sm>>>(outw, outs, dwk, tab, de43, nullptr, nullptr, nullptr, dX, dY, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr);
      if (fmt == VF_CB3) vlut_cb3_n1<<<g, 256, sm>>>(outw, outs, dwk, tab, de43, nullptr, nullptr, nullptr, dX, dY, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr);
      if (fmt == VF_CB2) vlut_cb2_n1<<<g, 256, sm>>>(outw, outs, dwk, tab, de43, nullptr, nullptr, nullptr, dX, dY, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr);
    };
    go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)T * N); cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
    double me = 0, mr = 0, qe = 0, qn = 0;
    for (int r = 0; r < N; r += 61) {
      for (int t = 0; t < T; t++) {
        double ref = 0;
        for (int c = 0; c < K; c++) {
          float wv;
          if (fmt == VF_FP4) wv = fp4_val(m, r, c);
          else wv = (fmt == VF_CB3 ? CB3 : CB2)[codes[(size_t)r * K + c]] * e4m3_dec(sb[(size_t)r * (K / 32) + c / 32]) * mul;
          ref += (double)wv * __half2float(X[(size_t)t * K + c]);
        }
        me = fmax(me, fabs(ref - Y[(size_t)t * N + r])); mr = fmax(mr, fabs(ref));
      }
      if (fmt != VF_FP4) for (int c = 0; c < K; c++) {
        float wq = (fmt == VF_CB3 ? CB3 : CB2)[codes[(size_t)r * K + c]] * e4m3_dec(sb[(size_t)r * (K / 32) + c / 32]) * mul, w0 = fp4_val(m, r, c);
        qe += (wq - w0) * (wq - w0); qn += w0 * w0;
      }
    }
    auto goE = [&] {
      if (fmt == VF_FP4) vlut_fp4_n1<<<gE, 256, sm>>>(outw, outs, dwk, tab, de43, nullptr, nullptr, nullptr, dX, dY, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr);
      if (fmt == VF_CB3) vlut_cb3_n1<<<gE, 256, sm>>>(outw, outs, dwk, tab, de43, nullptr, nullptr, nullptr, dX, dY, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr);
      if (fmt == VF_CB2) vlut_cb2_n1<<<gE, 256, sm>>>(outw, outs, dwk, tab, de43, nullptr, nullptr, nullptr, dX, dY, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr);
    };
    goE();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int i = 0; i < 20; i++) goE();
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
    double bytes = E * ((double)N * K * (fmt == VF_FP4 ? 0.5 : fmt == VF_CB3 ? 0.375 : 0.25) + (double)N * K / 32);
    printf("%s: kernel max err %.2e of %.2e%s, %d tok %.3f ms = %.0f GB/s\n", fmt == 0 ? "MXFP4" : fmt == 1 ? "cb3" : "cb2", me, mr,
           fmt ? (std::string(", quant rel err ") + std::to_string(sqrt(qe / qn))).c_str() : "", T, ms, bytes / ms / 1e6);
  }
  return 0;
}
