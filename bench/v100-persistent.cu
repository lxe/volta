// Persistent-kernel primitives on V100 (the megakernel's building blocks), one GPU:
//  1. grid barrier: B resident blocks bump one counter and spin until it reaches the next multiple (generation)
//  2. counter handoff: block 0 bumps a counter, block B-1 spins on it and answers (round trip / 2)
//  3. op sequence: N empty "ops" separated by barriers, against N empty kernels in a CUDA graph
//  4. a dependent chain through L2 with __threadfence (the release a producer pays before its counter bump)
// nvcc -O3 -arch=sm_70 v100-persistent.cu -o persistent
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ unsigned ld_acq(const unsigned *p) { unsigned v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

__device__ __forceinline__ void grid_barrier(unsigned *ctr, unsigned &gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    gen += gridDim.x;
    __threadfence();
    atomicAdd(ctr, 1);
    while (ld_acq(ctr) < gen) {}
  }
  __syncthreads();
}

__global__ void barriers(unsigned *ctr, int n, long long *clk) {
  unsigned gen = 0;
  long long t0 = clock64();
  for (int i = 0; i < n; i++) grid_barrier(ctr, gen);
  if (blockIdx.x == 0 && threadIdx.x == 0) clk[0] = clock64() - t0;
}

__global__ void handoff(unsigned *a, unsigned *b, int n, long long *clk) {
  if (threadIdx.x) return;
  long long t0 = clock64();
  if (blockIdx.x == 0) {
    for (unsigned i = 1; i <= (unsigned)n; i++) { __threadfence(); atomicAdd(a, 1); while (ld_acq(b) < i) {} }
    clk[0] = clock64() - t0;
  } else if (blockIdx.x == gridDim.x - 1) {
    for (unsigned i = 1; i <= (unsigned)n; i++) { while (ld_acq(a) < i) {} __threadfence(); atomicAdd(b, 1); }
  }
}

__global__ void empty() {}

int main() {
  int clk_khz; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  const double mhz = clk_khz / 1e3;
  unsigned *c; long long *clk, h;
  CK(cudaMalloc(&c, 64)); CK(cudaMalloc(&clk, 8));
  const int n = 10000;
  for (int B : {80, 160}) {
    for (int T : {256, 512}) {
      if (B == 160 && T == 512) continue;
      CK(cudaMemset(c, 0, 64));
      barriers<<<B, T>>>(c, n, clk); CK(cudaDeviceSynchronize());
      cudaMemcpy(&h, clk, 8, cudaMemcpyDeviceToHost);
      printf("grid barrier, %3d blocks x %d: %.2f us\n", B, T, h / (double)n / mhz);
    }
  }
  CK(cudaMemset(c, 0, 64));
  handoff<<<80, 32>>>(c, c + 8, n, clk); CK(cudaDeviceSynchronize());
  cudaMemcpy(&h, clk, 8, cudaMemcpyDeviceToHost);
  printf("counter handoff block 0 -> block 79 (one way): %.2f us\n", h / (double)n / 2 / mhz);
  // graph of empty kernels for comparison
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 1000; i++) empty<<<80, 256, 0, st>>>();
  CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st); CK(cudaGraphLaunch(ge, st)); cudaEventRecord(b, st); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b);
  printf("graph of 1000 empty 80x256 kernels: %.2f us per kernel\n", ms);
}
