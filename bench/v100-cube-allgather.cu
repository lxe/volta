// One-shot all-gather over the DGX-1 hybrid cube with peer stores and per-block flags.
//   direct: every GPU stores its chunk into all 7 peers (3 of them are PCIe-only from any GPU).
//   cube:   stage 1 pushes to the 4 NVLink neighbours (3 quad-mates + twin g^4);
//           stage 2 forwards the twin's chunk to the 3 quad-mates. NVLink only.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <thread>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)
constexpr int NG = 8, THREADS = 256;
#ifndef BLOCKS
#define BLOCKS 32
#endif

struct Args {
  int4* recv[NG];                 // recv[d] on GPU d: NG chunks laid out by source
  volatile uint32_t* flag[NG];    // flag[d] on GPU d: [stage][src][block]
  int me;
  size_t chunk16;                 // chunk size in int4
  uint32_t epoch;
  int mode;                       // 0 direct, 1 cube
};

__device__ __forceinline__ void copy_slice(int4* dst, const int4* src, size_t n16) {
  size_t per = (n16 + BLOCKS - 1) / BLOCKS, lo = blockIdx.x * per, hi = min(n16, lo + per);
  for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) dst[i] = src[i];
}
__device__ __forceinline__ void signal(volatile uint32_t* f, uint32_t epoch) {
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence_system(); *f = epoch; }
}
template <int N> __device__ __forceinline__ void copy_multi(int4* const* dst, const int4* src, size_t n16) {
  size_t per = (n16 + BLOCKS - 1) / BLOCKS, lo = blockIdx.x * per, hi = min(n16, lo + per);
  for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { int4 v = src[i];
#pragma unroll
    for (int k = 0; k < N; k++) dst[k][i] = v; }
}
template <int N> __device__ __forceinline__ void signal_multi(volatile uint32_t* const* f, uint32_t epoch) {
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence_system();
#pragma unroll
    for (int k = 0; k < N; k++) *f[k] = epoch; }
}
__device__ __forceinline__ void wait(volatile uint32_t* f, uint32_t epoch) {
  if (threadIdx.x == 0) while (*f < epoch) {}
  __syncthreads();
}
#define FLAG(d, stage, src) (a.flag[d] + ((stage) * NG + (src)) * BLOCKS + blockIdx.x)

__global__ void allgather(Args a) {
  const int me = a.me;
  const int4* mine = a.recv[me] + (size_t)me * a.chunk16;   // own chunk already sits in its own slot
  if (a.mode == 0) {
    for (int d = 0; d < NG; d++) if (d != me) {
      copy_slice(a.recv[d] + (size_t)me * a.chunk16, mine, a.chunk16);
      signal(FLAG(d, 0, me), a.epoch);
    }
    for (int s = 0; s < NG; s++) if (s != me) wait(FLAG(me, 0, s), a.epoch);
    return;
  }
  const int q = me & ~3, twin = me ^ 4;
  int4* d1[4]; volatile uint32_t* f1[4]; int n = 0;
  for (int i = 0; i < 4; i++) { int d = q + i; if (d == me) continue; d1[n] = a.recv[d] + (size_t)me * a.chunk16; f1[n++] = FLAG(d, 0, me); }
  d1[3] = a.recv[twin] + (size_t)me * a.chunk16; f1[3] = FLAG(twin, 0, me);
  copy_multi<4>(d1, mine, a.chunk16); signal_multi<4>(f1, a.epoch);
  wait(FLAG(me, 0, twin), a.epoch);
  const int4* tw = a.recv[me] + (size_t)twin * a.chunk16;
  int4* d2[3]; volatile uint32_t* f2[3]; n = 0;
  for (int i = 0; i < 4; i++) { int d = q + i; if (d == me) continue; d2[n] = a.recv[d] + (size_t)twin * a.chunk16; f2[n++] = FLAG(d, 1, twin); }
  copy_multi<3>(d2, tw, a.chunk16); signal_multi<3>(f2, a.epoch);
  for (int i = 0; i < 4; i++) { int s = q + i; if (s == me) continue; wait(FLAG(me, 0, s), a.epoch); }
  for (int i = 0; i < 4; i++) { int s = (q ^ 4) + i; if (s == twin) continue; wait(FLAG(me, 1, s), a.epoch); }
}

__global__ void empty_k() {}

__global__ void fill(int4* p, size_t n16, int src, uint32_t tag) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_int4(src, (int)i, (int)tag, src ^ (int)i);
}
__global__ void check(const int4* p, size_t n16, uint32_t tag, int* bad) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)NG * n16; i += (size_t)gridDim.x * blockDim.x) {
    int s = i / n16; size_t j = i % n16; int4 v = p[i];
    if (v.x != s || v.y != (int)j || v.z != (int)tag) atomicAdd(bad, 1);
  }
}

int main() {
  for (int i = 0; i < NG; i++) { CK(cudaSetDevice(i)); for (int j = 0; j < NG; j++) if (i != j) cudaDeviceEnablePeerAccess(j, 0); }
  cudaGetLastError();
  size_t maxchunk = 1 << 20;
  int4* recv[NG]; uint32_t* flag[NG]; cudaStream_t st[NG]; int* bad[NG];
  for (int d = 0; d < NG; d++) { CK(cudaSetDevice(d)); CK(cudaMalloc(&recv[d], NG * maxchunk)); CK(cudaMalloc(&flag[d], 2 * NG * BLOCKS * 4));
    CK(cudaMemset(flag[d], 0, 2 * NG * BLOCKS * 4)); CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking)); CK(cudaMalloc(&bad[d], 4)); }
  uint32_t epoch = 0;
  for (size_t chunk : {8u << 10, 32u << 10, 128u << 10, 320u << 10, 640u << 10}) {
    size_t c16 = chunk / 16;
    for (int mode = 1; mode < 2; mode++) {
      Args a; for (int d = 0; d < NG; d++) { a.recv[d] = recv[d]; a.flag[d] = flag[d]; } a.chunk16 = c16; a.mode = mode;
      // correctness pass
      ++epoch;
      for (int d = 0; d < NG; d++) { CK(cudaSetDevice(d)); fill<<<80, 256, 0, st[d]>>>(recv[d] + d * c16, c16, d, epoch); }
      for (int d = 0; d < NG; d++) CK(cudaStreamSynchronize(st[d]));
      for (int d = 0; d < NG; d++) { CK(cudaSetDevice(d)); a.me = d; a.epoch = epoch; allgather<<<BLOCKS, THREADS, 0, st[d]>>>(a); }
      int nbad = 0;
      for (int d = 0; d < NG; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d])); cudaMemset(bad[d], 0, 4);
        check<<<80, 256>>>(recv[d], c16, epoch, bad[d]); int b; CK(cudaMemcpy(&b, bad[d], 4, cudaMemcpyDeviceToHost)); nbad += b; }
      // timing: ITER back-to-back launches per GPU, no host sync in between
      const int ITER = 300;
      std::vector<cudaEvent_t> e0(NG), e1(NG);
      for (int d = 0; d < NG; d++) { CK(cudaSetDevice(d)); cudaEventCreate(&e0[d]); cudaEventCreate(&e1[d]); }
      uint32_t base = epoch;
      std::vector<std::thread> th;
      for (int d = 0; d < NG; d++) th.emplace_back([&, d] {
        cudaSetDevice(d); Args b = a; b.me = d; cudaEventRecord(e0[d], st[d]);
        for (int it = 1; it <= ITER; it++) { b.epoch = base + it; allgather<<<BLOCKS, THREADS, 0, st[d]>>>(b); }
        cudaEventRecord(e1[d], st[d]); });
      for (auto& t : th) t.join();
      epoch = base + ITER;
      float worst = 0;
      for (int d = 0; d < NG; d++) { CK(cudaSetDevice(d)); CK(cudaEventSynchronize(e1[d])); float ms; cudaEventElapsedTime(&ms, e0[d], e1[d]); worst = fmaxf(worst, ms); }
      double us = worst * 1000 / ITER;
      printf("%-6s chunk %4zu KB (gather %5zu KB/GPU): %6.1f us per all-gather, %5.1f GB/s received per GPU, errors %d\n",
             mode ? "cube" : "direct", chunk >> 10, (NG - 1) * chunk >> 10, us, (NG - 1) * chunk / us / 1e3, nbad);
    }
  }
  return 0;
}
