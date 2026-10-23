// ---------------------------------------------------------------- the megakernel
// One persistent launch runs a table of ops, each the body of one of the engine's kernels (the *_b functions) over
// its virtual grid: physical block b takes virtual blocks b, b + gridDim.x, ... An op whose flags bit 0 is set ends
// in a grid barrier, so the next op sees its writes. The host (gpu.rs) records the table from the kernels' own launch
// arguments, so an op's args are the kernel's parameters in order, one 8-byte slot each.
// Rules for bodies: any block size (they run with 256 threads), no early return before a __syncthreads, and no
// __ldg of data an earlier op in the same launch wrote (the non-coherent cache is not invalidated within a kernel;
// the barrier's fence makes ordinary loads see it).
// bench/v100-persistent.cu: barrier 0.66 us with 80 blocks, against 1.16 us per kernel boundary in a graph
struct MkOp {
  unsigned code, gx, gy, gz, flags, pad;
  unsigned long long a[22];
};
enum MkCode {
  MK_PP1 = 1, MK_PP2, MK_MIX, MK_PREP, MK_MM4, MK_RMS, MK_KV, MK_Q16R, MK_ATTN, MK_MERGE, MK_FP8, MK_LL,
};
// generation barrier (bar[0] arrivals, bar[1] generation): reusable across launches without a reset
__device__ __forceinline__ void mk_barrier(unsigned *bar) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned *gen = bar + 1;
    const unsigned g = *gen;
    __threadfence();
    if (atomicAdd(bar, 1) == gridDim.x - 1) {
      atomicExch(bar, 0);
      __threadfence();
      atomicAdd(bar + 1, 1);
    } else {
      while (*gen == g) {}
    }
  }
  __syncthreads();
  __threadfence();
}
#define MP(i, T) reinterpret_cast<T>(o.a[i])
#define MI(i) ((int)(unsigned)o.a[i])
#define MU(i) ((unsigned)o.a[i])
#define MF(i) __uint_as_float((unsigned)o.a[i])
// prof (VOLTA_MEGA_PROF): block 0 stamps the global timer at the start and after each op (and its barrier)
extern "C" __global__ void __launch_bounds__(256, 1) volta_mega(const MkOp *ops, int n, unsigned *bar, unsigned long long *prof) {
  if (prof && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); prof[0] = t; }
  for (int i = 0; i < n; i++) {
    const MkOp &o = ops[i];
    const int gx = o.gx, gy = o.gy, total = o.gx * o.gy * o.gz;
    for (int v = blockIdx.x; v < total; v += gridDim.x) {
      const VB vb{v % gx, (v / gx) % gy, v / (gx * gy), gx, gy, (int)o.gz};
      switch (o.code) {
        case MK_PP1: vhc_pp1_b(vb, MP(0, float *), MP(1, const float *), MP(2, const float *), MP(3, float *), MP(4, float *), MI(5)); break;
        case MK_PP2: vhc_pp2_b(vb, MP(0, float *), MP(1, const float *), MP(2, const float *), MI(3), MF(4)); break;
        case MK_MIX:
          vhc_mix_b(vb, MP(0, const float *), MP(1, const float *), MP(2, float *), MI(3), MI(4), MI(5), MP(6, float *), MF(7), MP(8, const float *),
                    MP(9, const float *), MF(10), MI(11), MP(12, int *));
          break;
        case MK_PREP: vprep_b(vb, MP(0, const float *), MI(1), MP(2, half *), MI(3), MI(4), MP(5, float *), MI(6)); break;
        case MK_MM4:
          vmoe_mm<4, 1, 1>(vb, MP(0, const uint4 *), MP(1, const half *), MP(2, const VWork *), MP(3, const int *), MP(4, const int *), MP(5, const float *),
                           MP(6, const half *), MP(7, float *), MI(8), MI(9), MI(10), MI(11), MP(12, const int *), MP(13, const int *), MI(14), MI(15),
                           MI(16), MI(17), MP(18, const int *), (long long)o.a[19]);
          break;
        case MK_RMS: rmsnorm_b(vb, MP(0, const float *), MP(1, const float *), MP(2, float *), MI(3), MI(4), MI(5), MF(6), MI(7)); break;
        case MK_KV:
          vkv_store_b(vb, MP(0, float *), MP(1, const float *), MI(2), MF(3), MP(4, const float *), MI(5), MP(6, const int *), MP(7, __nv_bfloat16 *),
                      MP(8, const int *), MI(9));
          break;
        case MK_Q16R:
          vq16_rope_b(vb, MP(0, const float *), MP(1, half *), MF(2), MI(3), MI(4), MI(5), MI(6), MP(7, const float *), MP(8, const int *));
          break;
        case MK_ATTN:
          vattn_mma_b(vb, MP(0, const half *), MP(1, const __nv_bfloat16 *), MP(2, const int *), MI(3), MP(4, const void *), MP(5, const int *), MI(6),
                      MP(7, float *), MI(8), MI(9), MI(10), MI(11));
          break;
        case MK_MERGE:
          vattn_merge_rope_b(vb, MP(0, const float *), MI(1), MP(2, const float *), MP(3, float *), MI(4), MI(5), MI(6), MP(7, const float *),
                             MP(8, const int *));
          break;
        case MK_FP8: fake_fp8_b(vb, MP(0, float *), MI(1)); break;
        case MK_LL:
          vx_llreduce_b(vb, MP(0, uint2 *const *), MI(1), MP(2, const unsigned *), MI(3), MI(4), MP(5, float *), MI(6), MP(7, const unsigned *), MU(8));
          break;
      }
      __syncthreads();
    }
    if (o.flags & 1) mk_barrier(bar);
    if (prof && blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); prof[i + 1] = t; }
  }
}
#undef MP
#undef MI
#undef MU
#undef MF
