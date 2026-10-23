// Prefill-shape lab for the dense 4-bit matmuls (DP: one GPU holds every attention matrix): each per-layer shape at
// 8 / 16 / 32 / 64 tokens with the kernel and K chunk the engine picks (vmm: n1_r1, n2_r1, n4), graph-free timing
// over 8 weight copies (each launch reads from DRAM), against bytes / 690 GB/s.
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels pf_test.cu -o pf_test
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dsx.cu"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static int even_kc(int k, int blocks, int want) {  // volta.rs even_kc
  std::vector<int> ok;
  for (int z = 1; z <= k / 128; z++) if (k % z == 0 && (k / z) % 128 == 0 && k / z <= 1024) ok.push_back(k / z);
  for (int kc : ok) if (blocks * (k / kc) >= want) return kc;
  return ok.empty() ? (k < 1024 ? k : 1024) : ok.back();
}
int main() {
  struct S { const char *name; int parts, n, k; } shapes[] = {
      {"qkv_a", 1, 1536, 5120}, {"q_b", 1, 32768, 1024}, {"o_a", 8, 1024, 4096}, {"o_b", 1, 5120, 8192}, {"sh_gu", 1, 4096, 5120}, {"sh_dn", 1, 5120, 2048}};
  const size_t WB = 160u << 20;  // bytes per weight copy (q_b: 32768 x 1024 / 2 = 16 MB + scales)
  std::vector<uint4 *> Wc(8);
  for (auto &w : Wc) { CK(cudaMalloc(&w, WB)); CK(cudaMemset(w, 0x37, WB)); }
  half *Sc, *X; float *Y;
  CK(cudaMalloc(&Sc, 64 << 20)); CK(cudaMalloc(&X, 64 * 8192 * 2)); CK(cudaMalloc(&Y, 64 * 40000 * 4));
  CK(cudaMemset(Sc, 0, 64 << 20)); CK(cudaMemset(X, 0, 64 * 8192 * 2));
  VWork *wk; CK(cudaMalloc(&wk, 64 * 64 * sizeof(VWork)));
  CK(cudaFuncSetAttribute(vmoe_b4_n4, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  double tot[4] = {0, 0, 0, 0}, flo = 0;
  const int TS[4] = {8, 16, 32, 64};
  for (auto &s : shapes) {
    const double bytes = (double)s.parts * s.n * s.k * (0.5 + 2.0 / 16);
    flo += bytes / 690e3;
    printf("%-6s %6.1f MB, floor %5.1f us:", s.name, bytes / 1e6, bytes / 690e3);
    for (int ti = 0; ti < 4; ti++) {
      const int T = TS[ti], nb = T <= 8 ? 1 : T <= 16 ? 2 : 4;
      // work entries [nt - 1][part] as the engine's prow table: nt = T tokens, part p's rows
      std::vector<VWork> w;
      const size_t wsz = (size_t)s.n * s.k / 2 / 16;
      for (int p = 0; p < s.parts; p++) w.push_back(VWork{(unsigned)(p * wsz), (unsigned)(p * s.n * s.k / 16), 0, (unsigned)T});
      CK(cudaMemcpy(wk, w.data(), w.size() * sizeof(VWork), cudaMemcpyHostToDevice));
      const int rtiles = (s.n + 255) / 256, kc = even_kc(s.k, rtiles * s.parts, 3 * 80);
      dim3 g(rtiles, s.parts, s.k / kc);
      const size_t sm = nb * 8 * (kc + 8) * 2;
      int r = 0;
      auto go = [&] {
        const uint4 *W = Wc[r++ % 8];
        if (nb == 1) vmoe_b4_n1_r1<<<g, 256, sm>>>(W, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, kc, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr, 0);
        else if (nb == 2) vmoe_b4_n2_r1<<<g, 256, sm>>>(W, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, kc, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr, 0);
        else vmoe_b4_n4<<<g, 256, sm>>>(W, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, kc, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr, 0);
      };
      go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      cudaEventRecord(a); for (int i = 0; i < 40; i++) go(); cudaEventRecord(b); CK(cudaEventSynchronize(b));
      float ms; cudaEventElapsedTime(&ms, a, b);
      const double us = ms * 1000 / 40;
      tot[ti] += us;
      printf("  t%-2d %6.1f us (KC%d)", T, us, kc);
    }
    printf("\n");
  }
  printf("layer total: t8 %.0f, t16 %.0f, t32 %.0f, t64 %.0f us; floor %.0f us | x40 layers: t64 %.1f ms, floor %.1f ms\n", tot[0], tot[1], tot[2], tot[3], flo, tot[3] * 40 / 1e3, flo * 40 / 1e3);
}
