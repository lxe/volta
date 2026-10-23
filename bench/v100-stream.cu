// nvcc -O3 -arch=sm_70 v100-stream.cu -o stream
// Streaming-read lab: 7.2 MB read once, coalesced, graph-timed; 4-byte loads (8 in flight per thread) and 16-byte
// loads, at 90 / 360 / 720 / 1440 blocks of 256. Shows what a 1-token expert pass could reach.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
template <class F> float graph_us(F body, int n = 200) {
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < n; i++) body(st);
  CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st); for (int r = 0; r < 5; r++) CK(cudaGraphLaunch(ge, st)); cudaEventRecord(b, st); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000 / (5 * n);
}
__global__ void rd4(const unsigned *p, size_t n, unsigned *out) {
  unsigned acc = 0;
  const size_t nt = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += 8 * nt) {
#pragma unroll
    for (int k = 0; k < 8; k++) if (i + k * nt < n) acc ^= __ldg(p + i + k * nt);
  }
  if (acc == 0x12345678) out[0] = acc;
}
__global__ void rd16(const uint4 *p, size_t n, unsigned *out) {
  unsigned acc = 0;
  const size_t nt = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += 4 * nt) {
#pragma unroll
    for (int k = 0; k < 4; k++) if (i + k * nt < n) { uint4 v = __ldg(p + i + k * nt); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
  }
  if (acc == 0x12345678) out[0] = acc;
}
int main() {
  const size_t bytes = 7190000 / 64 * 64;
  unsigned *p, *o; CK(cudaMalloc(&p, 256 << 20)); CK(cudaMalloc(&o, 4)); cudaMemset(p, 1, 256 << 20);
  // rotate over a 256 MB region so every launch reads from DRAM, not L2
  for (int g : {90, 360, 720, 1440}) {
    int r = 0;
    float a = graph_us([&](cudaStream_t s) { rd4<<<g, 256, 0, s>>>(p + (size_t)(r++ % 32) * (8 << 20) / 4, bytes / 4, o); });
    r = 0;
    float b = graph_us([&](cudaStream_t s) { rd16<<<g, 256, 0, s>>>((const uint4 *)(p + (size_t)(r++ % 32) * (8 << 20) / 4), bytes / 16, o); });
    printf("%4d blocks: 4-byte loads %6.2f us (%4.0f GB/s), 16-byte loads %6.2f us (%4.0f GB/s)\n", g, a, bytes / a / 1e3, b, bytes / b / 1e3);
  }
}
