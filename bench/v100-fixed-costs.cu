// Fixed costs on V100 for the batch-1 step model. nvcc -O3 -arch=sm_70 v100-fixed-costs.cu -o fixed-costs
// 1. graph node cost: N empty 1-block kernels chained in one graph, and N tiny kernels of grid 80 x 256
// 2. dependent global load latency (pointer chase): DRAM (64 MB ring), L2 (1 MB ring), in clocks and ns
// 3. NVLink flag ping-pong between GPU 0 and GPU g (peer stores + spin), one-way latency
// 4. a 1024-thread single-block kernel that reads 20480 floats and block-reduces (the norm / hc shape)
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void empty() {}
__global__ void touch(float *p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.f; }
__global__ void chase(const unsigned *next, int n, unsigned *out, long long *clk) {
  unsigned j = 0;
  long long t0 = clock64();
  for (int i = 0; i < n; i++) j = next[j];
  long long t1 = clock64();
  out[0] = j;
  clk[0] = t1 - t0;
}
__global__ void ping(volatile unsigned *mine, volatile unsigned *peer, int n, int lead, long long *clk) {
  long long t0 = clock64();
  for (int i = 1; i <= n; i++) {
    if (lead) { *peer = i; __threadfence_system(); while (*mine < (unsigned)i) {} }
    else { while (*mine < (unsigned)i) {} *peer = i; __threadfence_system(); }
  }
  clk[0] = clock64() - t0;
}
__global__ void __launch_bounds__(1024) blockread(const float *x, float *y, int n) {
  __shared__ float sh[32];
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += 1024) s += x[i] * x[i];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) { s = sh[threadIdx.x]; for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o); }
  if (threadIdx.x == 0) y[0] = s;
}

template <class F> float graph_us(F body, int reps = 20) {
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  body(st);
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, st);
  for (int r = 0; r < reps; r++) CK(cudaGraphLaunch(ge, st));
  cudaEventRecord(b, st); CK(cudaEventSynchronize(b));
  float ms; cudaEventElapsedTime(&ms, a, b);
  return ms * 1000 / reps;
}

int main() {
  int clk_khz; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  printf("SM clock %.0f MHz\n", clk_khz / 1e3);
  float *buf; CK(cudaMalloc(&buf, 1 << 20)); CK(cudaMemset(buf, 0, 1 << 20));
  const int N = 1000;
  printf("graph of %d empty 1-block kernels: %.2f us per node\n", N, graph_us([&](cudaStream_t s) { for (int i = 0; i < N; i++) empty<<<1, 32, 0, s>>>(); }) / N);
  printf("graph of %d 1-block kernels with a dependent global RMW: %.2f us per node\n", N, graph_us([&](cudaStream_t s) { for (int i = 0; i < N; i++) touch<<<1, 32, 0, s>>>(buf); }) / N);
  printf("graph of %d empty 80x256 kernels: %.2f us per node\n", N, graph_us([&](cudaStream_t s) { for (int i = 0; i < N; i++) empty<<<80, 256, 0, s>>>(); }) / N);
  printf("graph of %d empty 640x256 kernels: %.2f us per node\n", N, graph_us([&](cudaStream_t s) { for (int i = 0; i < N; i++) empty<<<640, 256, 0, s>>>(); }) / N);
  printf("graph of %d empty 5000x256 kernels: %.2f us per node\n", N, graph_us([&](cudaStream_t s) { for (int i = 0; i < N; i++) empty<<<5000, 256, 0, s>>>(); }) / N);
  float *x; CK(cudaMalloc(&x, 20480 * 4)); CK(cudaMemset(x, 0, 20480 * 4));
  printf("graph of %d single-block 1024-thread reads of 80 KB + reduce: %.2f us per node\n", N, graph_us([&](cudaStream_t s) { for (int i = 0; i < N; i++) blockread<<<1, 1024, 0, s>>>(x, buf, 20480); }) / N);
  // pointer chase
  for (size_t bytes : {(size_t)1 << 20, (size_t)64 << 20}) {
    size_t n = bytes / 4;
    std::vector<unsigned> h(n);
    const size_t stride = 4096 / 4 + 32;  // a new DRAM page / L2 line each hop
    for (size_t i = 0; i < n; i++) h[i] = (unsigned)((i + stride) % n);
    unsigned *d, *o; long long *c, hc;
    CK(cudaMalloc(&d, bytes)); CK(cudaMalloc(&o, 4)); CK(cudaMalloc(&c, 8));
    cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice);
    chase<<<1, 1>>>(d, 1000, o, c); CK(cudaDeviceSynchronize());
    chase<<<1, 1>>>(d, 10000, o, c); CK(cudaDeviceSynchronize());
    cudaMemcpy(&hc, c, 8, cudaMemcpyDeviceToHost);
    printf("dependent load, %zu MB ring: %.0f clocks = %.0f ns\n", bytes >> 20, hc / 10000.0, hc / 10000.0 / (clk_khz / 1e6));
    cudaFree(d);
  }
  // NVLink ping-pong 0 <-> g
  int ng; cudaGetDeviceCount(&ng);
  for (int g = 1; g < ng; g++) {
    int ok; cudaDeviceCanAccessPeer(&ok, 0, g);
    if (!ok) { printf("GPU0 - GPU%d: no peer access\n", g); continue; }
    unsigned *f0, *fg; long long *c0, *cg, h0;
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(g, 0); CK(cudaMalloc(&f0, 4)); CK(cudaMemset(f0, 0, 4)); CK(cudaMalloc(&c0, 8));
    cudaSetDevice(g); cudaDeviceEnablePeerAccess(0, 0); CK(cudaMalloc(&fg, 4)); CK(cudaMemset(fg, 0, 4)); CK(cudaMalloc(&cg, 8));
    const int n = 2000;
    cudaSetDevice(g); ping<<<1, 1>>>(fg, f0, n, 0, cg);
    cudaSetDevice(0); ping<<<1, 1>>>(f0, fg, n, 1, c0);
    cudaSetDevice(0); CK(cudaDeviceSynchronize()); cudaSetDevice(g); CK(cudaDeviceSynchronize());
    cudaSetDevice(0); cudaMemcpy(&h0, c0, 8, cudaMemcpyDeviceToHost);
    printf("GPU0 <-> GPU%d flag ping-pong: %.2f us one way\n", g, h0 / (double)n / 2 / (clk_khz / 1e3));
    cudaSetDevice(0); cudaDeviceDisablePeerAccess(g); cudaSetDevice(g); cudaDeviceDisablePeerAccess(0); cudaFree(fg); cudaSetDevice(0); cudaFree(f0);
  }
}
