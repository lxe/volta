// Discover m8n8k4 (row.col, f16 in, f32 acc) fragment layout on sm_70.
// A[r][k] = (k==0) ? r : 0 ; B[k][c] = (k==0) ? 1 : 0 plus k==1 row: c*16 -> D[r][c] = r + 16*c ... use A[r][1]=1.
#include <cstdio>
#include <cuda_fp16.h>
__global__ void k(float* out, int* arow, int* bcol) {
  int lane = threadIdx.x;
  // A fragment: thread holds 4 halves of one row. Encode row id guess via lane; fill A so that A[row][0]=row_id, A[row][1]=1.
  // We don't know which row a lane holds, so give each lane a unique tag in a0 and 1 in a1: D[row][col] = tagA(row)*B0(col) + 1*B1(col).
  half2 a01 = __halves2half2(__int2half_rn(lane), __float2half(1.f)), a23 = __float2half2_rn(0.f);
  half2 b01 = __halves2half2(__float2half(1.f), __int2half_rn(lane * 32)), b23 = __float2half2_rn(0.f);
  unsigned A0 = *(unsigned*)&a01, A1 = *(unsigned*)&a23, B0 = *(unsigned*)&b01, B1 = *(unsigned*)&b23;
  float d[8] = {0};
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "r"(A0), "r"(A1), "r"(B0), "r"(B1));
  for (int i = 0; i < 8; i++) out[lane * 8 + i] = d[i];
}
int main() {
  float* o; cudaMallocManaged(&o, 256 * 4); k<<<1, 32>>>(o, 0, 0); cudaDeviceSynchronize();
  // D = tagA(row-lane) + 32*tagB(col-lane): decode which A-lane and B-lane fed each accumulator
  for (int l = 0; l < 32; l++) { printf("lane %2d:", l); for (int i = 0; i < 8; i++) { int v = (int)o[l * 8 + i]; printf(" A%02d/B%02d", v % 32, v / 32); } printf("\n"); }
}
