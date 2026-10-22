// Expert slicing at equal bytes: one GPU's layer of cb3 experts (8 tokens each) as 30 whole experts, or 30 * tp
// slices of 1/tp of the intermediate dim (gate/up rows 2 * 2304 / tp x 5120, down 5120 x 2304 / tp).
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels split_test.cu -o split_test
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cuda_fp16.h>
#include "dsx.cu"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

int main() {
  const int D = 5120, I = 2304, T = 8, E0 = 30;
  uint32_t* W; uint8_t* S; half *X, *e43; half2* tab; float* Y;
  size_t wb = (size_t)E0 * 3 * I * D * 3 / 8, sb = (size_t)E0 * 3 * I * D / 32;
  CK(cudaMalloc(&W, wb)); CK(cudaMalloc(&S, sb)); CK(cudaMalloc(&X, (size_t)T * D * 2)); CK(cudaMalloc(&Y, (size_t)T * D * 4 * 2));
  CK(cudaMalloc(&e43, 512)); CK(cudaMalloc(&tab, 256)); CK(cudaMemset(W, 0x5a, wb)); CK(cudaMemset(S, 0x30, sb)); CK(cudaMemset(X, 0, T * D * 2));
  CK(cudaMemset(e43, 0, 512)); CK(cudaMemset(tab, 0, 256));
  VWork2* wk; CK(cudaMalloc(&wk, 4096 * sizeof(VWork2)));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int tp : {1, 2, 4, 8}) {
    const int E = E0 * tp, sl = I / tp;
    double tot = 0;
    for (int gu = 1; gu >= 0; gu--) {
      const int N = gu ? 2 * sl : D, K = gu ? D : sl;
      const size_t mw = (size_t)N * K * 3 / 8, ms = (size_t)N * K / 32;
      std::vector<VWork2> v;
      for (int e = 0; e < E; e++) v.push_back(VWork2{(unsigned)(e * mw / 128), (unsigned)(e * ms), 0, (unsigned)T, 1.f, {0, 0, 0}});
      CK(cudaMemcpy(wk, v.data(), v.size() * sizeof(VWork2), cudaMemcpyHostToDevice));
      const int KC0 = getenv("KC") ? atoi(getenv("KC")) : 1024; const int KC = K % KC0 == 0 ? KC0 : K % 1152 == 0 && KC0 >= 1024 ? 1152 : K % 576 == 0 && KC0 >= 512 ? 576 : K <= KC0 ? K : 288;
      size_t smem = 64 * 4 + 512 + 8 * (KC + 8) * 2;
      for (int R : {4, 1}) {
      dim3 g((N + 256 * R - 1) / (256 * R), E, (K + KC - 1) / KC);
      auto go = [&] { auto f = R == 4 ? vlut_cb3_n1 : vlut_cb3r1_n1;
                      f<<<g, 256, smem>>>(W, S, wk, tab, e43, nullptr, nullptr, nullptr, X, Y, N, K, KC, N, nullptr, nullptr, 0, 0, 0, K, nullptr); };
      go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      cudaEventRecord(a); for (int i = 0; i < 20; i++) go(); cudaEventRecord(b); cudaEventSynchronize(b);
      float ms_; cudaEventElapsedTime(&ms_, a, b); double us = ms_ * 1000 / 20;
      double bytes = (double)E * (mw + ms);
      printf("tp %d %s R%d: %d x [%d x %d] KC %d: %.1f us, %.0f GB/s\n", tp, gu ? "gate/up" : "down", R, E, N, K, KC, us, bytes / us / 1e3);
      if (R == 1) tot += us;
      }
    }
    printf("tp %d layer: %.1f us\n", tp, tot);
  }
}
