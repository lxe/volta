// vattn_mma (tensor cores) against vattn_tile5 (CUDA cores): random q, window and compressed rows (some slots empty),
// both merged by sparse_attn_merge; max error of the merged outputs, then the time of each at T tokens.
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels attn_test.cu -o attn_test
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_bf16.h>
#include "dsx.cu"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

int main() {
  const int H = 64, hd = 512, W = 128, K = 512, POOL = 4096;
  const float scale = 1.f / sqrtf((float)hd);
  srand(3);
  auto rnd = [] { return ((rand() % 20001) - 10000) * 1e-4f; };
  for (int T : {8, 32}) {
    std::vector<float> q((size_t)T * H * hd);
    for (auto& v : q) v = rnd() * 2.f;
    std::vector<__nv_bfloat16> win((size_t)POOL * hd), comp((size_t)POOL * hd);
    for (auto& v : win) v = __float2bfloat16(rnd());
    for (auto& v : comp) v = __float2bfloat16(rnd());
    std::vector<int> wi((size_t)T * W), ci((size_t)T * K);
    for (int t = 0; t < T; t++) {
      for (int j = 0; j < W; j++) wi[t * W + j] = (t % 3 == 0 && j < 40) ? -1 : rand() % POOL;  // some windows not full
      for (int j = 0; j < K; j++) ci[t * K + j] = (j % 7 == 3) ? -1 : rand() % POOL;
    }
    float *dq, *dpart, *dsinks, *do1, *do2; __nv_bfloat16 *dwin, *dcomp; int *dwi, *dci; half *dq16;
    const int S = (W + K + 159) / 160;
    CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dq16, q.size() * 2)); CK(cudaMalloc(&dwin, win.size() * 2)); CK(cudaMalloc(&dcomp, comp.size() * 2));
    CK(cudaMalloc(&dwi, wi.size() * 4)); CK(cudaMalloc(&dci, ci.size() * 4)); CK(cudaMalloc(&dpart, (size_t)T * S * H * (hd + 2) * 4));
    CK(cudaMalloc(&dsinks, H * 4)); CK(cudaMalloc(&do1, (size_t)T * H * hd * 4)); CK(cudaMalloc(&do2, (size_t)T * H * hd * 4));
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dwin, win.data(), win.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dcomp, comp.data(), comp.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dwi, wi.data(), wi.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dci, ci.data(), ci.size() * 4, cudaMemcpyHostToDevice); cudaMemset(dsinks, 0, H * 4);
    CK(cudaFuncSetAttribute(vattn_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, VAM_SMEM));
    dim3 g(T, H / 16, S);
    auto old_ = [&] { vattn_tile5<<<g, 256>>>(dq, dwin, dwi, W, dcomp, dci, K, dpart, H, hd, scale); };
    auto new_ = [&] { vq16<<<(T * H * hd + 255) / 256, 256>>>(dq, dq16, scale, T * H * hd); vattn_mma<<<dim3(H / 16, T, S), 256, VAM_SMEM>>>(dq16, dwin, dwi, W, dcomp, dci, K, dpart, H, hd); };
    old_(); sparse_attn_merge<<<dim3(T, H), 256>>>(dpart, S, dsinks, do1, H, hd);
    new_(); sparse_attn_merge<<<dim3(T, H), 256>>>(dpart, S, dsinks, do2, H, hd);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> o1((size_t)T * H * hd), o2(o1.size());
    cudaMemcpy(o1.data(), do1, o1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(o2.data(), do2, o2.size() * 4, cudaMemcpyDeviceToHost);
    double me = 0, mx = 0;
    for (size_t i = 0; i < o1.size(); i++) { me = fmax(me, fabs(o1[i] - o2[i])); mx = fmax(mx, fabs(o1[i])); }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms1, ms2;
    cudaEventRecord(a); for (int i = 0; i < 50; i++) old_(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms1, a, b);
    cudaEventRecord(a); for (int i = 0; i < 50; i++) new_(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms2, a, b);
    printf("T %d: max |diff| %.2e of max |out| %.2e; tile5 %.1f us, mma %.1f us\n", T, me, mx, ms1 * 20, ms2 * 20);
  }
}
