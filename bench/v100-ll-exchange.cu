// 8-GPU exchange latency lab (DGX-1 V100 cube). Every GPU contributes `bytes`; every GPU ends with all 8 blocks.
//  A  vx_allgather (the engine's: copies + __threadfence_system + flags, two cube stages)
//  B  LL one-shot: 8-byte (data, seq) words straight into every peer (NVLink and PCIe peers), receivers poll the words
//  C  LL cube: stage 1 to the 3 quad-mates and the twin, stage 2 the twin forwards its block to our quad-mates
// Each runs N exchanges back to back on all GPUs; reports us per exchange (GPU0 events).
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels v100-ll-exchange.cu -o ll-exchange
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dsx.cu"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

struct LLPeers { uint2 *rx[8]; };
// one 8-byte store / load that is never split or cached in L1 (the receiver sees data and seq together)
__device__ __forceinline__ void st2(uint2 *p, uint2 v) { asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory"); }
__device__ __forceinline__ uint2 ld2(const uint2 *p) { uint2 v; asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory"); return v; }  // each GPU's receive area: [2 parities][8 src][words]

// B: one-shot. grid G blocks; words = bytes / 4
__device__ void ll_oneshot_d(const LLPeers &P, int me, const unsigned *src, unsigned *out, int words, unsigned seq) {
  const int par = seq & 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nt = gridDim.x * blockDim.x;
  for (int i = tid; i < words; i += nt) {
    const unsigned v = src[i];
    out[me * words + i] = v;
    const uint2 w = make_uint2(v, seq);
#pragma unroll
    for (int p = 0; p < 8; p++)
      if (p != me) st2(P.rx[p] + ((size_t)par * 8 + me) * words + i, w);
  }
  const uint2 *rx = P.rx[me] + (size_t)par * 8 * words;
  for (int s = 0; s < 8; s++) {
    if (s == me) continue;
    for (int i = tid; i < words; i += nt) {
      uint2 w;
      do { w = ld2(rx + (size_t)s * words + i); } while (w.y != seq);
      out[s * words + i] = w.x;
    }
  }
}

// C: two stages over NVLink only
__device__ void ll_cube_d(const LLPeers &P, int me, const unsigned *src, unsigned *out, int words, unsigned seq) {
  const int par = seq & 1, q = me & ~3, tw = me ^ 4;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nt = gridDim.x * blockDim.x;
  for (int i = tid; i < words; i += nt) {
    const unsigned v = src[i];
    out[me * words + i] = v;
    const uint2 w = make_uint2(v, seq);
    for (int k = 0; k < 4; k++) { int d = q + k; if (d != me) st2(P.rx[d] + ((size_t)par * 8 + me) * words + i, w); }
    st2(P.rx[tw] + ((size_t)par * 8 + me) * words + i, w);
  }
  const uint2 *rx = P.rx[me] + (size_t)par * 8 * words;
  // the twin's block: forward each word to our quad-mates as it lands
  for (int i = tid; i < words; i += nt) {
    uint2 w;
    do { w = ld2(rx + (size_t)tw * words + i); } while (w.y != seq);
    out[tw * words + i] = w.x;
    for (int k = 0; k < 4; k++) { int d = q + k; if (d != me) st2(P.rx[d] + ((size_t)par * 8 + tw) * words + i, w); }
  }
  for (int s = 0; s < 8; s++) {
    if (s == me || s == tw) continue;
    for (int i = tid; i < words; i += nt) {
      uint2 w;
      do { w = ld2(rx + (size_t)s * words + i); } while (w.y != seq);
      out[s * words + i] = w.x;
    }
  }
}


// A in a loop: vx_allgather's body (volta.cuh) with the sequence number passed in
__device__ void vx_ag(const VxPeers &P, int me, size_t slot, const int *bytes, unsigned seq) {
  const int q = me & ~3, tw = me ^ 4;
  unsigned char *d1[4];
  volatile unsigned *f1[4];
  int n = 0;
  for (int i = 0; i < 4; i++) { int d = q + i; if (d == me) continue; d1[n] = P.buf[d] + me * slot; f1[n++] = VXF(d, 0, me); }
  d1[3] = P.buf[tw] + me * slot; f1[3] = VXF(tw, 0, me);
  vx_copy(d1, 4, P.buf[me] + me * slot, bytes[me]);
  vx_signal(f1, 4, seq);
  vx_wait(VXF(me, 0, tw), seq);
  unsigned char *d2[3];
  volatile unsigned *f2[3];
  n = 0;
  for (int i = 0; i < 4; i++) { int d = q + i; if (d == me) continue; d2[n] = P.buf[d] + tw * slot; f2[n++] = VXF(d, 1, tw); }
  vx_copy(d2, 3, P.buf[me] + tw * slot, bytes[tw]);
  vx_signal(f2, 3, seq);
  for (int i = 0; i < 4; i++) { int s = q + i; if (s == me) continue; vx_wait(VXF(me, 0, s), seq); }
  for (int i = 0; i < 4; i++) { int s = (q ^ 4) + i; if (s == tw) continue; vx_wait(VXF(me, 1, s), seq); }
}
__global__ void __launch_bounds__(256) loopA(const VxPeers *Pp, int me, size_t slot, const int *bytes, unsigned base, int reps) {
  for (int r = 1; r <= reps; r++) { vx_ag(*Pp, me, slot, bytes, base + r); __syncthreads(); }
}


// the blocks of one GPU must not run ahead of each other into the next exchange's parity: a grid-wide counter
__device__ void gsync(unsigned *ctr, unsigned target) {
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence(); atomicAdd(ctr, 1); while (*(volatile unsigned *)ctr < target) {} }
  __syncthreads();
}
__global__ void __launch_bounds__(256) loopB(LLPeers P, int me, const unsigned *src, unsigned *out, int words, unsigned base, int reps, unsigned *ctr) {
  for (int r = 1; r <= reps; r++) { ll_oneshot_d(P, me, src, out, words, base + r); gsync(ctr, r * gridDim.x); }
}
__global__ void __launch_bounds__(256) loopC(LLPeers P, int me, const unsigned *src, unsigned *out, int words, unsigned base, int reps, unsigned *ctr) {
  for (int r = 1; r <= reps; r++) { ll_cube_d(P, me, src, out, words, base + r); gsync(ctr, r * gridDim.x); }
}
int main(int argc, char **argv) {
  const int N = 1000;
  for (int bytes : {10240, 20480, 40960, 81920, 163840, 327680}) {
    const int words = bytes / 4;
    std::vector<unsigned *> src(8), out(8), xbuf(8), relay(8), flags(8), seqd(8), xbytes(8);
    std::vector<uint2 *> rx(8);
    std::vector<VxPeers *> vp(8);
    std::vector<cudaStream_t> st(8);
    for (int g = 0; g < 8; g++) {
      CK(cudaSetDevice(g));
      for (int p = 0; p < 8; p++) if (p != g) { cudaDeviceEnablePeerAccess(p, 0); cudaGetLastError(); }
      CK(cudaStreamCreate(&st[g]));
      CK(cudaMalloc(&src[g], bytes)); CK(cudaMemset(src[g], g + 1, bytes));
      CK(cudaMalloc(&out[g], 8 * bytes));
      CK(cudaMalloc(&rx[g], 2 * 8 * (size_t)words * 8)); CK(cudaMemset(rx[g], 0, 2 * 8 * (size_t)words * 8));
      CK(cudaMalloc(&xbuf[g], 8 * (size_t)bytes)); CK(cudaMalloc(&relay[g], 32 * (size_t)bytes));
      CK(cudaMalloc(&flags[g], 3 * 8 * VX_BLOCKS * 4)); CK(cudaMemset(flags[g], 0, 3 * 8 * VX_BLOCKS * 4));
      CK(cudaMalloc(&seqd[g], 4)); CK(cudaMemset(seqd[g], 0, 4));
      CK(cudaMalloc(&xbytes[g], 64)); std::vector<int> b(16, bytes); cudaMemcpy(xbytes[g], b.data(), 64, cudaMemcpyHostToDevice);
      CK(cudaMalloc(&vp[g], sizeof(VxPeers)));
    }
    LLPeers P; for (int g = 0; g < 8; g++) P.rx[g] = rx[g];
    VxPeers hv; for (int g = 0; g < 8; g++) { hv.buf[g] = (unsigned char *)xbuf[g]; hv.relay[g] = (unsigned char *)relay[g]; hv.flag[g] = flags[g]; }
    for (int g = 0; g < 8; g++) { cudaSetDevice(g); cudaMemcpy(vp[g], &hv, sizeof hv, cudaMemcpyHostToDevice); }
    std::vector<unsigned *> ctr(8);
    for (int g = 0; g < 8; g++) { cudaSetDevice(g); CK(cudaMalloc(&ctr[g], 4)); }
    auto run = [&](const char *name, auto launch) {
      for (int g = 0; g < 8; g++) { cudaSetDevice(g); CK(cudaMemset(ctr[g], 0, 4)); CK(cudaDeviceSynchronize()); }
      cudaEvent_t a, b; cudaSetDevice(0); cudaEventCreate(&a); cudaEventCreate(&b);
      cudaEventRecord(a, st[0]);
      for (int g = 0; g < 8; g++) { cudaSetDevice(g); launch(g); }
      cudaSetDevice(0); cudaEventRecord(b, st[0]);
      for (int g = 0; g < 8; g++) { cudaSetDevice(g); CK(cudaStreamSynchronize(st[g])); }
      float ms; cudaEventElapsedTime(&ms, a, b);
      printf("%6d B  %-28s %6.2f us per exchange\n", bytes, name, ms * 1000 / N);
    };
    static unsigned base = 0;
    base += N + 1;
    unsigned b0 = base;
    run("A vx_allgather (cube, flags)", [&](int g) { loopA<<<VX_BLOCKS, 256, 0, st[g]>>>(vp[g], g, bytes, (const int *)xbytes[g], b0, N); });
    for (int G : {80}) {
      char nm[64];
      snprintf(nm, 64, "C LL cube, %d blocks", G);
      base += N + 1;
      unsigned b1 = base;
      run(nm, [&](int g) { loopC<<<G, 256, 0, st[g]>>>(P, g, src[g], out[g], words, b1, N, ctr[g]); });
    }
    // check the last exchange: GPU g's out[s] block is all (s + 1) bytes
    std::vector<unsigned> h(8 * words);
    int bad = 0;
    for (int g = 0; g < 8; g++) {
      cudaSetDevice(g); cudaMemcpy(h.data(), out[g], 8 * bytes, cudaMemcpyDeviceToHost);
      for (int s = 0; s < 8; s++) { unsigned e = 0x01010101u * (s + 1); for (int i = 0; i < words; i++) bad += h[s * words + i] != e; }
    }
    printf("check: %d wrong words\n", bad);
  }
}
