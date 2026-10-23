// Standalone test for vmoe_* (nvidia/src/kernels/volta.cuh): correctness against a CPU reference on a few experts,
// then the time of one decode step's routed experts on one GPU (gate/up 4608x5120, swiglu, down 5120x2304).
// nvcc -O3 -arch=sm_70 -I../engine/src/kernels vmoe_test.cu -o vmoe_test
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include "dsx.cu"  // dsx.cu includes volta.cuh and the helpers it uses
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

// pack a row-major code matrix [N][K] (codes 0..2^BITS-1) and per-32 scales into the swizzled layout
static void pack(int BITS, const std::vector<uint8_t>& codes, const std::vector<float>& sc, int N, int K, std::vector<uint32_t>& W, std::vector<half>& S) {
  int WPU = 128 / BITS, F = 32 / BITS, units = K / WPU;
  int GS = 64 / BITS;
  W.assign((size_t)N * K * BITS / 32, 0); S.assign((size_t)N * K / GS, __float2half(0.f));
  for (int r = 0; r < N; r++) {
    int rg = r / 32, rs = r % 32, lane = 4 * (rs >> 3) + (rs & 3) + 16 * ((rs >> 2) & 1);
    for (int k = 0; k < K; k++) {
      int u = k / WPU, w = (k % WPU) / F, f = k % F;
      W[(((size_t)rg * units + u) * 32 + lane) * 4 + w] |= (uint32_t)codes[(size_t)r * K + k] << (BITS * f);
    }
    for (int g = 0; g < K / GS; g++) S[((size_t)rg * (K / GS) + g) * 32 + lane] = __float2half(sc[(size_t)r * (K / GS) + g]);
  }
}

struct Mat { int bits, N, K; std::vector<uint8_t> codes; std::vector<float> sc; };
static Mat randmat(int bits, int N, int K, unsigned seed) {
  Mat m{bits, N, K}; srand(seed); m.codes.resize((size_t)N * K); m.sc.resize((size_t)N * K / (64 / bits));
  for (auto& c : m.codes) c = rand() & ((1 << bits) - 1);
  for (auto& s : m.sc) s = 0.002f + (rand() % 100) * 2e-5f;
  return m;
}

template <int BITS, int NT> void launch(dim3 g, size_t sm, const uint4* W, const half* S, const VWork* wk, const int* ir, const int* orr, const float* tw, const half* X, float* Y, int N, int K, int KC) {
  if (BITS == 2 && NT == 1) vmoe_b2_n1<<<g, 256, sm>>>(W, S, wk, ir, orr, tw, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr, 0);
  if (BITS == 2 && NT == 2) vmoe_b2_n2<<<g, 256, sm>>>(W, S, wk, ir, orr, tw, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr, 0);
  if (BITS == 2 && NT == 4) vmoe_b2_n4<<<g, 256, sm>>>(W, S, wk, ir, orr, tw, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr, 0);
  if (BITS == 4 && NT == 1) vmoe_b4_n1<<<g, 256, sm>>>(W, S, wk, ir, orr, tw, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr, 0);
  if (BITS == 4 && NT == 2) vmoe_b4_n2<<<g, 256, sm>>>(W, S, wk, ir, orr, tw, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr, 0);
  if (BITS == 4 && NT == 4) vmoe_b4_n4<<<g, 256, sm>>>(W, S, wk, ir, orr, tw, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr, 0);
}
static int rows_per_block(int NT) { int R = NT == 1 ? 4 : NT == 2 ? 2 : 1; return 8 * 32 * R; }

// E experts, each with tokens t_e (list of input rows); Y[out_row] += tw * W_e x
template <int BITS, int NT> double run(int E, int N, int K, int KC, const std::vector<int>& ntok, int T, bool check, int iters) {
  std::vector<uint32_t> Wall; std::vector<half> Sall; std::vector<VWork> work; std::vector<int> ir, orr; std::vector<float> tw;
  std::vector<Mat> mats;
  for (int e = 0; e < E; e++) {
    Mat m = randmat(BITS, N, K, 100 + e); std::vector<uint32_t> W; std::vector<half> S; pack(BITS, m.codes, m.sc, N, K, W, S);
    VWork w{(unsigned)(Wall.size() / 4), (unsigned)Sall.size(), (unsigned)ir.size(), (unsigned)ntok[e]};
    Wall.insert(Wall.end(), W.begin(), W.end()); Sall.insert(Sall.end(), S.begin(), S.end());
    for (int t = 0; t < ntok[e]; t++) { int r = (e * 7 + t * 13) % T; ir.push_back(r); orr.push_back(r); tw.push_back(0.5f + 0.1f * (t % 5)); }
    work.push_back(w); if (check) mats.push_back(std::move(m));
  }
  std::vector<half> X((size_t)T * K); for (auto& v : X) v = __float2half(((rand() % 2001) - 1000) * 1e-3f);
  uint4* dW; half *dS, *dX; VWork* dwk; int *dir, *dor; float *dtw, *dY;
  CK(cudaMalloc(&dW, Wall.size() * 4)); CK(cudaMalloc(&dS, Sall.size() * 2)); CK(cudaMalloc(&dX, X.size() * 2)); CK(cudaMalloc(&dwk, work.size() * sizeof(VWork)));
  CK(cudaMalloc(&dir, ir.size() * 4)); CK(cudaMalloc(&dor, orr.size() * 4)); CK(cudaMalloc(&dtw, tw.size() * 4)); CK(cudaMalloc(&dY, (size_t)T * N * 4));
  CK(cudaMemcpy(dW, Wall.data(), Wall.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dS, Sall.data(), Sall.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dwk, work.data(), work.size() * sizeof(VWork), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dir, ir.data(), ir.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dor, orr.data(), orr.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dtw, tw.data(), tw.size() * 4, cudaMemcpyHostToDevice));
  size_t sm = (size_t)NT * 8 * (KC + 8) * 2;
  if (sm > 48 * 1024) {
    if (BITS == 2 && NT == 4) CK(cudaFuncSetAttribute(vmoe_b2_n4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    if (BITS == 4 && NT == 4) CK(cudaFuncSetAttribute(vmoe_b4_n4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    if (BITS == 2 && NT == 2) CK(cudaFuncSetAttribute(vmoe_b2_n2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    if (BITS == 4 && NT == 2) CK(cudaFuncSetAttribute(vmoe_b4_n2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  }
  dim3 g((N + rows_per_block(NT) - 1) / rows_per_block(NT), E, (K + KC - 1) / KC);
  CK(cudaMemset(dY, 0, (size_t)T * N * 4));
  launch<BITS, NT>(g, sm, dW, dS, dwk, dir, dor, dtw, dX, dY, N, K, KC); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  if (check) {
    std::vector<float> Y((size_t)T * N), ref((size_t)T * N, 0.f); CK(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost));
    int off = 1 << (BITS - 1);
    for (int e = 0; e < E; e++) for (int t = 0; t < ntok[e]; t++) {
      int ti = work[e].t0 + t, r = ir[ti];
      for (int n = 0; n < N; n += 7) { double a = 0;
        int GS = 64 / BITS;
        for (int k = 0; k < K; k++) a += (double)((int)mats[e].codes[(size_t)n * K + k] - off) * mats[e].sc[(size_t)n * (K / GS) + k / GS] * __half2float(X[(size_t)r * K + k]);
        ref[(size_t)orr[ti] * N + n] += a * tw[ti]; }
    }
    double me = 0, mr = 0; for (size_t i = 0; i < Y.size(); i += 1) if ((i % N) % 7 == 0) { me = fmax(me, fabs(Y[i] - ref[i])); mr = fmax(mr, fabs(ref[i])); }
    printf("check %d-bit NT %d N %d K %d: max err %.2e of %.2e\n", BITS, NT, N, K, me, mr);
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a); for (int i = 0; i < iters; i++) launch<BITS, NT>(g, sm, dW, dS, dwk, dir, dor, dtw, dX, dY, N, K, KC);
  cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b);
  double per = ms / iters; double bytes = Wall.size() * 4.0 + Sall.size() * 2.0;
  if (!check) printf("%d-bit NT %d N %d K %d, %d experts x %d tok: %.3f ms, %.0f GB/s\n", BITS, NT, N, K, E, ntok[0], per, bytes / per / 1e6);
  cudaFree(dW); cudaFree(dS); cudaFree(dX); cudaFree(dwk); cudaFree(dir); cudaFree(dor); cudaFree(dtw); cudaFree(dY);
  return per;
}

int main() {
  // correctness: 3 experts with 1, 5 and 8 tokens (NT 1); 13 and 16 (NT 2); 27 and 32 (NT 4)
  run<2, 1>(3, 4608, 5120, 1024, {1, 5, 8}, 40, true, 1);
  run<4, 1>(3, 5120, 2304, 1152, {1, 5, 8}, 40, true, 1);
  run<2, 2>(2, 4608, 5120, 1024, {13, 16}, 40, true, 1);
  run<2, 4>(2, 5120, 2304, 1152, {27, 32}, 40, true, 1);
  run<4, 4>(2, 4608, 5120, 1024, {27, 32}, 40, true, 1);
  // timing: one GPU's 48 experts of a layer at 4 tokens each (256 tokens/step), gate/up then down
  std::vector<int> four(48, 4), eight(48, 8);
  double gu = run<2, 1>(48, 4608, 5120, 1024, four, 256, false, 20), dn = run<2, 1>(48, 5120, 2304, 1152, four, 256, false, 20);
  printf("2-bit layer, 48 experts x 4 tokens: %.3f ms -> 40 layers %.1f ms\n", gu + dn, 40 * (gu + dn));
  gu = run<2, 1>(48, 4608, 5120, 1024, eight, 256, false, 20); dn = run<2, 1>(48, 5120, 2304, 1152, eight, 256, false, 20);
  printf("2-bit layer, 48 experts x 8 tokens: %.3f ms -> 40 layers %.1f ms\n", gu + dn, 40 * (gu + dn));
  return 0;
}
