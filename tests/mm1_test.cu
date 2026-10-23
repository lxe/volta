// One-token dense matmul lab: the megakernel's prefetching kernel (vmoe_b4_pf: every unit's weights loaded before
// x, one K chunk per SM) against the graph path's vmoe_b4_n1_r1 (K chunks for >= 3 blocks per SM), for the
// head-parallel layer's shapes. Graph-timed per launch over 8 copies of the weights (64 MB: each launch reads from
// DRAM, not L2); floor = bytes / 690 GB/s (a measured stream).
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels mm1_test.cu -o mm1_test
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dsx.cu"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
template <class F> float graph_us(F body, int n = 100) {
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
int main() {
  struct S { const char *name; int n, k, kc_g, kc_pf; } shapes[] = {
      {"qkv_a", 1792, 5120, 128, 512}, {"q_b/8", 4096, 1280, 128, 256}, {"o_a/8", 1024, 4096, 128, 256}, {"o_b/8", 5120, 1024, 128, 256}};
  uint4 *W; half *Sc, *X; float *Y;
  CK(cudaMalloc(&W, 8 * (8 << 20))); CK(cudaMalloc(&Sc, 8 << 20)); CK(cudaMalloc(&X, 1 << 20)); CK(cudaMalloc(&Y, 8 << 20));
  CK(cudaMemset(W, 0x37, 8 * (8 << 20))); CK(cudaMemset(Sc, 0, 8 << 20)); CK(cudaMemset(X, 0, 1 << 20));
  // SPREAD=1: the 8 weight copies 3.5 GB apart (as the engine's ~30 GB per GPU spreads them): TLB reach
  std::vector<uint4 *> Wc(8, nullptr);
  const bool spread = getenv("SPREAD") && atoi(getenv("SPREAD"));
  for (int r = 0; r < 8; r++) {
    if (spread) { void *f; CK(cudaMalloc(&f, (size_t)3500 << 20)); CK(cudaMalloc(&Wc[r], 8 << 20)); CK(cudaMemset(Wc[r], 0x37, 8 << 20)); }
    else Wc[r] = W + (size_t)r * (8 << 20) / 16;
  }
  VWork *wk; CK(cudaMalloc(&wk, sizeof(VWork)));
  VWork w0{0, 0, 0, 1};
  CK(cudaMemcpy(wk, &w0, sizeof w0, cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(vmoe_b4_pf, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024));
  for (auto &s : shapes) {
    const double bytes = (double)s.n * s.k * (0.5 + 2.0 / 16);
    const int rt = (s.n + 255) / 256;
    auto run = [&](bool pf, int kc) {
      dim3 g(rt, 1, s.k / kc);
      size_t sm = 8 * (kc + 8) * 2;
      int r = 0;
      return graph_us([&](cudaStream_t st) {
        const uint4 *Wr = Wc[r++ % 8];
        if (pf) vmoe_b4_pf<<<g, 256, sm, st>>>(Wr, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, kc, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr, 0);
        else vmoe_b4_n1_r1<<<g, 256, sm, st>>>(Wr, Sc, wk, nullptr, nullptr, nullptr, X, Y, s.n, s.k, kc, s.n, nullptr, nullptr, 0, 0, 0, s.k, nullptr, 0);
      });
    };
    // the same prefetching matmul as a one-op megakernel table (volta_mega's code generation)
    auto mega = [&]() {
      MkOp op{};
      op.code = MK_MMP; op.gx = rt; op.gy = 1; op.gz = s.k / s.kc_pf; op.dep2 = -1; op.dep = -1;
      unsigned long long a[20] = {0, (unsigned long long)Sc, (unsigned long long)wk, 0, 0, 0, (unsigned long long)X, (unsigned long long)Y, (unsigned)s.n, (unsigned)s.k,
                                  (unsigned)s.kc_pf, (unsigned)s.n, 0, 0, 0, 0, 0, (unsigned)s.k, 0, 0};
      std::vector<MkOp> tabs(8);
      for (int r = 0; r < 8; r++) { tabs[r] = op; for (int i = 0; i < 20; i++) tabs[r].a[i] = a[i]; tabs[r].a[0] = (unsigned long long)Wc[r]; }
      MkOp *dt; unsigned *bar, *done;
      CK(cudaMalloc(&dt, sizeof(MkOp) * 8)); CK(cudaMalloc(&bar, 64)); CK(cudaMalloc(&done, 64)); cudaMemset(bar, 0, 64); cudaMemset(done, 0, 64);
      cudaMemcpy(dt, tabs.data(), sizeof(MkOp) * 8, cudaMemcpyHostToDevice);
      CK(cudaFuncSetAttribute(volta_mega, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
      int r = 0;
      return graph_us([&](cudaStream_t st) { volta_mega<<<80, 256, 8 * (s.kc_pf + 8) * 2, st>>>(dt + (r++ % 8), 1, bar, done, nullptr); });
    };
    if (getenv("KCSWEEP")) {
      printf("%-6s", s.name);
      for (int kc : {32, 64, 128, 256}) if (s.k % kc == 0) printf("  KC%d (%d blk) %.1f", kc, rt * s.k / kc, run(false, kc));
      printf("\n");
      continue;
    }
    const float a = run(false, s.kc_g), b = run(true, s.kc_pf), c = run(false, s.kc_pf), d = mega();
    printf("%-6s %.2f MB  graph kernel KC%d (%3d blocks) %5.1f us | prefetch KC%d (%3d blocks) %5.1f us | r1 at that KC %5.1f us | as a megakernel op %5.1f us | floor %.1f us\n", s.name,
           bytes / 1e6, s.kc_g, rt * s.k / s.kc_g, a, s.kc_pf, rt * s.k / s.kc_pf, b, c, d, bytes / 690e3);
  }
}
