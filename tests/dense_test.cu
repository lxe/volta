// Launch-shape sweep for the dense 4-bit matmuls at 8 tokens (vmoe_b4_n1, R = 4 or 1 row groups per warp):
// each of the layer's dense shapes at every (R, KC), against its HBM floor.
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels dense_test.cu -o dense_test
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include "dsx.cu"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

int main() {
  struct S { const char* name; int parts, n, k; } shapes[] = {{"q_a", 1, 1280, 5120}, {"kv", 1, 512, 5120}, {"q_b", 1, 32768, 1280}, {"o_a", 8, 1024, 4096}, {"o_b", 1, 5120, 8192}};
  const int T = 8;
  uint4* W; half *Sc, *X; float* Y; CK(cudaMalloc(&W, 64 << 20)); CK(cudaMalloc(&Sc, 8 << 20)); CK(cudaMalloc(&X, 1 << 20)); CK(cudaMalloc(&Y, 8 << 20));
  CK(cudaMemset(W, 0x37, 64 << 20)); CK(cudaMemset(Sc, 0, 8 << 20)); CK(cudaMemset(X, 0, 1 << 20));
  VWork* wk; CK(cudaMalloc(&wk, 64 * sizeof(VWork)));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  double tot_best = 0, tot_floor = 0;
  for (auto& s : shapes) {
    std::vector<VWork> w;
    size_t wsz = (size_t)s.n * s.k / 2 / 16;  // uint4 per part
    for (int p = 0; p < s.parts; p++) w.push_back(VWork{(unsigned)(p * wsz), (unsigned)(p * s.n * s.k / 16), 0, (unsigned)T});
    CK(cudaMemcpy(wk, w.data(), w.size() * sizeof(VWork), cudaMemcpyHostToDevice));
    double bytes = (double)s.parts * s.n * s.k * (0.5 + 2.0 / 16), floor_us = bytes / 868e3;
    double best = 1e9; int bR = 0, bKC = 0;
    for (int R : {4, 1}) for (int KC : {128, 256, 320, 512, 640, 1024, 1280}) {
      if (s.k % KC) continue;
      if (KC > s.k) continue;
      dim3 g((s.n + 256 * R - 1) / (256 * R), s.parts, (s.k + KC - 1) / KC);
      size_t sm = 8 * (KC + 8) * 2;
      auto go = [&] { if (R == 4) vmoe_b4_n1<<<g, 256, sm>>>(W, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, KC, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr);
                      else vmoe_b4_n1_r1<<<g, 256, sm>>>(W, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, KC, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr); };
      go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      cudaEventRecord(a); for (int i = 0; i < 50; i++) go(); cudaEventRecord(b); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms, a, b); double us = ms * 1000 / 50;
      printf("  %-4s R%d KC%4d grid %4d x%d x%2d: %6.1f us (%3.0f GB/s)\n", s.name, R, KC, g.x, g.y, g.z, us, bytes / us / 1e3);
      if (us < best) { best = us; bR = R; bKC = KC; }
    }
    printf("%-4s best R%d KC%d %.1f us, floor %.1f us\n", s.name, bR, bKC, best, floor_us);
    tot_best += best; tot_floor += floor_us;
  }
  printf("layer: best %.1f us, floor %.1f us -> 40 layers %.2f / %.2f ms\n", tot_best, tot_floor, tot_best * 40 / 1000, tot_floor * 40 / 1000);
}
