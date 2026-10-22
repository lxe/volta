// vindexer_mma (tensor cores) against vindexer_score4 (CUDA cores) on random FP4 keys: max error of the scores and
// the time of each at T tokens over N keys.
// nvcc -O3 -arch=sm_70 -std=c++17 -I../engine/src/kernels indexer_test.cu -o indexer_test
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "dsx.cu"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)
static const float E2M1[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
int main() {
  srand(5);
  for (int N : {16384, 131072}) for (int T : {8, 64}) {
    std::vector<unsigned char> keys((size_t)N * 68);
    for (size_t r = 0; r < (size_t)N; r++) { for (int b = 0; b < 64; b++) keys[r * 68 + b] = rand() & 255; for (int s = 0; s < 4; s++) keys[r * 68 + 64 + s] = 120 + rand() % 6; }
    std::vector<float> q((size_t)T * 32 * 128), w(T * 32);
    for (auto& v : q) v = E2M1[rand() % 8] * (rand() & 1 ? 1 : -1) * ldexpf(1.f, -(rand() % 4));
    for (auto& v : w) v = (rand() % 1000) * 1e-3f;
    std::vector<int> koff(T, 0), nv(T);
    for (int t = 0; t < T; t++) nv[t] = N - (t * 37) % 500;
    unsigned char* dk; float *dq, *dw, *ds1, *ds2; int *doff, *dnv;
    CK(cudaMalloc(&dk, keys.size())); CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dw, w.size() * 4));
    CK(cudaMalloc(&ds1, (size_t)T * N * 4)); CK(cudaMalloc(&ds2, (size_t)T * N * 4)); CK(cudaMalloc(&doff, T * 4)); CK(cudaMalloc(&dnv, T * 4));
    cudaMemcpy(dk, keys.data(), keys.size(), cudaMemcpyHostToDevice); cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(doff, koff.data(), T * 4, cudaMemcpyHostToDevice); cudaMemcpy(dnv, nv.data(), T * 4, cudaMemcpyHostToDevice);
    auto a = [&] { vindexer_score4<<<dim3((N + 7) / 8, T), 256>>>(dq, dk, doff, dw, ds1, dnv, N, 32, 128); };
    auto b = [&] { vindexer_mma<<<dim3((N + 127) / 128, T), 256>>>(dq, dk, doff, dw, ds2, dnv, N); };
    a(); b(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> s1((size_t)T * N), s2(s1.size());
    cudaMemcpy(s1.data(), ds1, s1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(s2.data(), ds2, s2.size() * 4, cudaMemcpyDeviceToHost);
    double me = 0, mx = 0;
    for (size_t i = 0; i < s1.size(); i++) { if (std::isinf(s1[i]) != std::isinf(s2[i])) { me = 1e9; break; } if (!std::isinf(s1[i])) { me = fmax(me, fabs(s1[i] - s2[i])); mx = fmax(mx, fabs(s1[i])); } }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float m1, m2;
    cudaEventRecord(e0); for (int i = 0; i < 10; i++) a(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&m1, e0, e1);
    cudaEventRecord(e0); for (int i = 0; i < 10; i++) b(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&m2, e0, e1);
    printf("N %6d T %2d: max |diff| %.2e of %.2e; score4 %.0f us, mma %.0f us\n", N, T, me, mx, m1 * 100, m2 * 100);
    cudaFree(dk); cudaFree(dq); cudaFree(dw); cudaFree(ds1); cudaFree(ds2); cudaFree(doff); cudaFree(dnv);
  }
}
