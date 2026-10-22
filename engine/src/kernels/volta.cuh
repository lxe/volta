// Batched routed-expert kernels for Volta (sm_70+): mma.m8n8k4 with weights unpacked straight into registers.
//
// Weight layout per expert matrix [N rows][K]: rows in groups of 32 ("row groups"); within a group, 16-byte units
// along K, each unit stored for the 32 lanes that consume it: [group][unit][lane] uint4. Lane L owns row
// 8q + (L&3) + 4h of the group (q = (L>>2)&3, h = L>>4). A unit holds 128/BITS weights of one row as offset-binary
// codes (value = (code - 2^(BITS-1)) * scale), field f of 32-bit word w is weight w*(32/BITS) + f.
// Scales: fp16, two per unit (words 0-1 and 2-3): one per GS = 64/BITS weights (16 at 4-bit, 32 at 2-bit),
// [group][K/GS][lane].
//
// Activations: fp16 rows X[r][K]. Work entries pick an expert matrix and a run of (input row, output row, weight)
// triples; each block of a launch handles one entry, one 32*R-row slice of N and one KC slice of K, and adds
// tw * (W x) into Y[out_row] (f32) with atomics, so K splits and several experts per token simply accumulate.

#include <mma.h>

struct VWork {
  unsigned w_off;   // expert matrix start, in uint4 units
  unsigned s_off;   // its scales start, in halves
  unsigned t0;      // first index into the token arrays
  unsigned nt;      // tokens (<= 8 * NT)
};

__device__ __forceinline__ void v_mma884(float *d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

template <int BITS> __device__ __forceinline__ uint32_t v_deq(uint32_t v, int i, half2 s2) {
  constexpr uint32_t MASK = ((1u << BITS) - 1) * 0x00010001u;
  uint32_t t = ((v >> (BITS * i)) & MASK) | 0x64006400u;
  const half2 off = __float2half2_rn(1024.f + (1 << (BITS - 1)));
  half2 h = __hmul2(__hsub2(*reinterpret_cast<half2 *>(&t), off), s2);
  return *reinterpret_cast<uint32_t *>(&h);
}
// unpacked stream order inside a 32-bit word: pairs (f_i, f_{i + F/2})
template <int BITS> __device__ __forceinline__ int v_perm(int j) { constexpr int F = 32 / BITS; int i = j >> 1; return (j & 1) ? i + F / 2 : i; }

// cnt != null: device-built token lists. Entry wk.nt is then the list's index: its tokens are cnt[wk.nt], its rows
// start at wk.t0, and the launch takes the lists with lo < count <= hi, in tiles of 8 * NT tokens (grid.x =
// row tiles x token tiles, row tiles = rtiles). in_rows / out_rows null = the row is the pair index itself.
template <int BITS, int NT, int R>
__device__ __forceinline__ void vmoe_mm(const uint4 *__restrict__ W, const half *__restrict__ S, const VWork *__restrict__ work,
                                        const int *__restrict__ in_rows, const int *__restrict__ out_rows, const float *__restrict__ tw,
                                        const half *__restrict__ X, float *__restrict__ Y, int N, int K, int KC, int ldy,
                                        const int *__restrict__ cnt, const int *__restrict__ off, int lo, int hi, int rtiles, int ldx, const int *__restrict__ nwork) {
  constexpr int WPU = 128 / BITS, F = 32 / BITS, WARPS = 8;
  extern __shared__ __align__(16) unsigned char vsm[];
  half *xs = reinterpret_cast<half *>(vsm);
  const int LDX = KC + 8;
  // nwork: the entries are a compacted list of *nwork (hot experts); blocks loop over them by gridDim.y
  const int nent = nwork ? *nwork : (int)gridDim.y;
  for (int ei = blockIdx.y; ei < nent; ei += gridDim.y) {
  const VWork wk0 = work[ei];
  int bx = blockIdx.x, n_all = (int)wk0.nt, t_base = (int)wk0.t0, tt0 = 0, tstep = 1 << 30;
  if (cnt) {
    // device-built lists: this block takes token tiles tt0, tt0 + tstep, ... of its list (bounded grid, any count)
    n_all = cnt[wk0.nt];
    if (n_all <= lo || n_all > hi) continue;
    tt0 = blockIdx.x / rtiles;
    tstep = gridDim.x / rtiles;
    bx = blockIdx.x % rtiles;
    t_base = off[wk0.nt];
  }
  const int ntiles = (n_all + 8 * NT - 1) / (8 * NT);
  for (int tt = tt0; tt < ntiles; tt += tstep) {
  VWork wk = wk0;
  wk.t0 = t_base + tt * 8 * NT;
  wk.nt = min(8 * NT, n_all - tt * 8 * NT);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q = (lane >> 2) & 3, h = lane >> 4;
  const int k0 = blockIdx.z * KC;
  if (k0 >= K) continue;
  const int kc = min(KC, K - k0), units = kc / WPU;
  const int tok = (lane & 3) + 4 * h;
  const int rg0 = (bx * WARPS + warp) * R;              // first row group of this warp
  const bool active = rg0 * 32 < N;                      // warps past N still help load x
  const int rv = active ? min(R, N / 32 - rg0) : 0;      // this warp's row groups inside N
  const int units_all = K / WPU, u0 = k0 / WPU;
  const uint4 *wp[R];
  const half *sp[R];
#pragma unroll
  for (int j = 0; j < R; j++) {
    size_t rg = (size_t)(rg0 + j);
    wp[j] = W + wk.w_off + (rg * units_all + u0) * 32 + lane;
    sp[j] = S + wk.s_off + (rg * (K / (WPU / 2)) + k0 / (WPU / 2)) * 32 + lane;
  }
  uint4 qn[R];
#pragma unroll
  for (int j = 0; j < R; j++) qn[j] = j < rv ? __ldg(wp[j]) : make_uint4(0, 0, 0, 0);
  // x tile: NT*8 token rows x kc, k permuted per 32-bit word group to match the unpack order
  for (int i = threadIdx.x; i < NT * 8 * (kc / 8); i += WARPS * 32) {
    int t = i / (kc / 8), c8 = (i % (kc / 8)) * 8;
    half v[8];
    if (t < (int)wk.nt) {
      const half *xr = X + (size_t)(in_rows ? in_rows[wk.t0 + t] : (int)wk.t0 + t) * ldx + k0;
#pragma unroll
      for (int e = 0; e < 8; e++) { int k = c8 + e, g = k / F, j = k % F; v[e] = xr[g * F + v_perm<BITS>(j)]; }
    } else {
#pragma unroll
      for (int e = 0; e < 8; e++) v[e] = __float2half(0.f);
    }
    *reinterpret_cast<uint4 *>(&xs[t * LDX + c8]) = *reinterpret_cast<uint4 *>(v);
  }
  __syncthreads();
  if (active) {
  float acc[R][NT][8];
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][n][e] = 0.f;
  for (int u = 0; u < units; u++) {
    uint4 qc[R];
#pragma unroll
    for (int j = 0; j < R; j++) { qc[j] = qn[j]; if (u + 1 < units && j < rv) qn[j] = __ldg(wp[j] + (size_t)(u + 1) * 32); }
    half2 s2[R][2];
#pragma unroll
    for (int j = 0; j < R; j++)
#pragma unroll
      for (int g = 0; g < 2; g++) s2[j][g] = __half2half2(sp[j][(size_t)(u * 2 + g) * 32]);
#pragma unroll
    for (int w = 0; w < 4; w++) {
      const int g = w >> 1;
#pragma unroll
      for (int m = 0; m < F / 4; m++) {
        const int kpos = u * WPU + w * F + m * 4;
        uint32_t b[NT][2];
#pragma unroll
        for (int n = 0; n < NT; n++) { uint2 v = *reinterpret_cast<const uint2 *>(&xs[(n * 8 + tok) * LDX + kpos]); b[n][0] = v.x; b[n][1] = v.y; }
#pragma unroll
        for (int j = 0; j < R; j++) {
          uint32_t v = (w == 0) ? qc[j].x : (w == 1) ? qc[j].y : (w == 2) ? qc[j].z : qc[j].w;
          uint32_t a0 = v_deq<BITS>(v, 2 * m, s2[j][g]), a1 = v_deq<BITS>(v, 2 * m + 1, s2[j][g]);
#pragma unroll
          for (int n = 0; n < NT; n++) v_mma884(acc[j][n], a0, a1, b[n][0], b[n][1]);
        }
      }
    }
  }
  // accumulator e of lane L: row slot 8q + (L&1) + 2((e>>1)&1) + 4h, token 2((L>>1)&1) + (e&1) + 4(e>>2)
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        int rs = 8 * q + (lane & 1) + 2 * ((e >> 1) & 1) + 4 * h, t = n * 8 + 2 * ((lane >> 1) & 1) + (e & 1) + 4 * (e >> 2);
        if (t < (int)wk.nt && j < rv) {
          int ti = wk.t0 + t;
          atomicAdd(&Y[(size_t)(out_rows ? out_rows[ti] : ti) * ldy + (rg0 + j) * 32 + rs], acc[j][n][e] * (tw ? tw[ti] : 1.f));
        }
      }
  }
  __syncthreads();  // the next tile reuses the x tile
  }
  }
}

#define VMOE(BITS, NT, R)                                                                                                     \
  extern "C" __global__ void __launch_bounds__(256) vmoe_b##BITS##_n##NT(                                                     \
      const uint4 *W, const half *S, const VWork *work, const int *in_rows, const int *out_rows, const float *tw, const half *X, \
      float *Y, int N, int K, int KC, int ldy, const int *cnt, const int *off, int lo, int hi, int rtiles, int ldx,             \
      const int *nwork) {                                                                                                     \
    vmoe_mm<BITS, NT, R>(W, S, work, in_rows, out_rows, tw, X, Y, N, K, KC, ldy, cnt, off, lo, hi, rtiles, ldx, nwork);      \
  }
VMOE(2, 1, 4)
VMOE(2, 2, 2)
VMOE(2, 4, 1)
VMOE(4, 1, 4)
VMOE(4, 2, 2)
VMOE(4, 4, 1)
// one row group per warp: dense matrices whose rows are not a multiple of 32 * 4 * 8
#define VMOE1(BITS, NT)                                                                                                       \
  extern "C" __global__ void __launch_bounds__(256) vmoe_b##BITS##_n##NT##_r1(                                                \
      const uint4 *W, const half *S, const VWork *work, const int *in_rows, const int *out_rows, const float *tw, const half *X, \
      float *Y, int N, int K, int KC, int ldy, const int *cnt, const int *off, int lo, int hi, int rtiles, int ldx,             \
      const int *nwork) {                                                                                                     \
    vmoe_mm<BITS, NT, 1>(W, S, work, in_rows, out_rows, tw, X, Y, N, K, KC, ldy, cnt, off, lo, hi, rtiles, ldx, nwork);      \
  }
VMOE1(4, 1)
VMOE1(4, 2)

// gate/up rows [P][2*inter] (f32, gate first) -> h[P][inter] fp16 = silu(min(g, lim)) * clamp(u, +-lim) * wt[p]
extern "C" __global__ void vswiglu(const float *gu, const float *wt, half *hout, int P, int inter, float lim) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)P * inter) return;
  int p = i / inter, c = i % inter;
  float g = fminf(gu[(size_t)p * 2 * inter + c], lim), u = fminf(fmaxf(gu[(size_t)p * 2 * inter + inter + c], -lim), lim);
  hout[i] = __float2half(bf16r(g / (1.f + expf(-g)) * u * (wt ? wt[p] : 1.f)));
}

// f32 rows -> fp16 rows (n values)
extern "C" __global__ void vto_f16(const float *x, half *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half(x[i]);
}

// ---------------------------------------------------------------- batched per-session attention helpers
// Session state lives in per-GPU pools ([slot][...]); per-token arrays give each token's pool rows.

// kv_norm_rope_store for T tokens: x[t][n] normalized, roped at pos[t], FP8 fake-quantized, stored as bf16 at
// pool row dst_row[t]. grid T, block 1024 (n <= 1024)
extern "C" __global__ void vkv_store(float *x, const float *w, int n, float eps, const float *freqs, int rd, const int *pos,
                                     __nv_bfloat16 *pool, const int *dst_row) {
  __shared__ float sh[32];
  float *xt = x + (size_t)blockIdx.x * n;
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += xt[i] * xt[i];
  s = block_sum(s, sh);
  float r = rsqrtf(s / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) xt[i] = w[i] * (xt[i] * r);
  __syncthreads();
  int half = rd / 2, p0 = pos[blockIdx.x];
  for (int p = threadIdx.x; p < half; p += blockDim.x) {
    float sn, cs;
    sincosf((float)p0 * freqs[p], &sn, &cs);
    float *v = xt + n - rd + 2 * p;
    float a = v[0], b = v[1];
    v[0] = a * cs - b * sn;
    v[1] = a * sn + b * cs;
  }
  __syncthreads();
  int i = threadIdx.x;
  if (i < n) {
    float v = xt[i];
    float amax = fmaxf(warp_max(fabsf(v)), 1e-4f);
    float sc = pow2_ceil(amax / 448.f);
    v = e4m3_round(fminf(fmaxf(v / sc, -448.f), 448.f)) * sc;
    xt[i] = v;
    pool[(size_t)dst_row[blockIdx.x] * n + i] = __float2bfloat16(v);
  }
}

// rows src[t][n] (f32) -> pool rows dst_row[t] (f32 pool, row width n); grid T, block 256
extern "C" __global__ void vrows_scatter(const float *src, float *pool, const int *dst_row, int n) {
  const float *s = src + (size_t)blockIdx.x * n;
  float *d = pool + (size_t)dst_row[blockIdx.x] * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i];
}

// compress_pool for T sessions: state rows state_row[t] .. + ratio of the kv / score pools -> out[t][hd]
extern "C" __global__ void vcompress(const float *kv_pool, const float *sc_pool, const int *state_row, float *out, int ratio, int hd) {
  if (state_row[blockIdx.x] < 0) return;  // this token closes no group this step
  const float *kv = kv_pool + (size_t)state_row[blockIdx.x] * hd, *ss = sc_pool + (size_t)state_row[blockIdx.x] * hd;
  float *o = out + (size_t)blockIdx.x * hd;
  for (int j = threadIdx.x; j < hd; j += blockDim.x) {
    float mx = -INFINITY;
    for (int s = 0; s < ratio; s++) mx = fmaxf(mx, ss[s * hd + j]);
    float den = 0.f, num = 0.f;
    for (int s = 0; s < ratio; s++) { float e = expf(ss[s * hd + j] - mx); den += e; num += kv[s * hd + j] * e; }
    o[j] = bf16r(num / den);
  }
}

// f32 rows -> bf16 pool rows dst_row[t]; grid T, block 256
extern "C" __global__ void vstore_bf16_rows(const float *src, __nv_bfloat16 *pool, const int *dst_row, int n) {
  if (dst_row[blockIdx.x] < 0) return;  // masked token
  const float *s = src + (size_t)blockIdx.x * n;
  __nv_bfloat16 *d = pool + (size_t)dst_row[blockIdx.x] * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) d[i] = __float2bfloat16(s[i]);
}

// indexer_score with per-token key bases: token t scores rows [0, n_vis[t]) of pool rows koff[t] + s; grid (ceil(n_max/8), T)
extern "C" __global__ void vindexer_score(const float *q, const __nv_bfloat16 *kpool, const int *koff, const float *w, float *score,
                                          const int *n_vis, int n_max, int IH, int ID) {
  int t = blockIdx.y, s = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (s >= n_max) return;
  float *sc = score + (size_t)t * n_max;
  if (s >= n_vis[t]) {
    if (lane == 0) sc[s] = -INFINITY;
    return;
  }
  const __nv_bfloat16 *kr = kpool + ((size_t)koff[t] + s) * ID + lane * 4;
  float kv[4];
  for (int j = 0; j < 4; j++) kv[j] = __bfloat162float(kr[j]);
  float tot = 0.f;
  for (int h = 0; h < IH; h++) {
    const float *qh = q + ((size_t)t * IH + h) * ID + lane * 4;
    float d = qh[0] * kv[0] + qh[1] * kv[1] + qh[2] * kv[2] + qh[3] * kv[3];
    d = warp_sum(d);
    tot += fmaxf(d, 0.f) * w[t * IH + h];
  }
  if (lane == 0) sc[s] = tot;
}

// selected positions -> pool rows: idx[t][k] += off[t] where idx >= 0; grid T, block 256
extern "C" __global__ void vidx_offset(int *idx, const int *off, int k) {
  int *r = idx + (size_t)blockIdx.x * k;
  for (int i = threadIdx.x; i < k; i += blockDim.x) if (r[i] >= 0) r[i] += off[blockIdx.x];
}

// shared-expert SwiGLU on rows: h[t][i] = bf16(silu(min(g, lim)) * clamp(u, +-lim)); grid (ceil(inter/256), T)
extern "C" __global__ void vswiglu_f32(const float *g, const float *u, float *h, int inter, float lim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inter) return;
  size_t o = (size_t)blockIdx.y * inter + i;
  float gv = fminf(g[o], lim), uv = fminf(fmaxf(u[o], -lim), lim);
  h[o] = bf16r(gv / (1.f + expf(-gv)) * uv);
}

// a[t][d] = bf16(a[t][d] + sum_s parts[s][t][d]) over S sources; grid (ceil(T*d/256))
extern "C" __global__ void vsum_parts(float *a, const float *parts, int S, int stride, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = a[i];
  for (int s = 0; s < S; s++) v += parts[(size_t)s * stride + i];
  a[i] = bf16r(v);
}

// ---------------------------------------------------------------- device-side routing for the local experts

// Compact token lists per local expert from the exchanged picks, in three passes: vroute_count (cnt[j]),
// vroute_scan (off[j] = exclusive prefix, off[L] = total pairs, cur zeroed), vroute_fill (pair rows off[j] + i get
// the token and its weight). lidx[e] = local list of expert e (-1 = another GPU's); tokens of GPU g are rows
// g * tmax .. + ntok[g]. grid ceil(TG * k / 256) for count / fill, 1 x 32 for the scan.
// picks of token t are selg[t * lds + 0..k), weights wtg[t * lds + 0..k) (strided exchange rows)
extern "C" __global__ void vroute_count(const int *selg, int lds, const int *ntok, int tmax, int ngpu, int k, const int *lidx, int *cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ngpu * tmax * k) return;
  int t = i / k, g = t / tmax;
  if (t - g * tmax >= ntok[g]) return;
  int j = lidx[selg[t * lds + i % k]];
  if (j >= 0) atomicAdd(&cnt[j], 1);
}
extern "C" __global__ void vroute_scan(const int *cnt, int *off, int *cur, int L) {
  if (threadIdx.x == 0) {
    int a = 0;
    for (int j = 0; j < L; j++) { off[j] = a; a += cnt[j]; cur[j] = 0; }
    off[L] = a;
  }
}
extern "C" __global__ void vroute_fill(const int *selg, const float *wtg, int lds, const int *ntok, int tmax, int ngpu, int k, const int *lidx,
                                       const int *off, int *cur, int *tok, float *wt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ngpu * tmax * k) return;
  int t = i / k, g = t / tmax;
  if (t - g * tmax >= ntok[g]) return;
  int j = lidx[selg[t * lds + i % k]];
  if (j < 0) return;
  int p = off[j] + atomicAdd(&cur[j], 1);
  tok[p] = t;
  wt[p] = wtg[t * lds + i % k];
}

// one exchange row per token: [x fp16 (d) | picks i32 (k) | weights f32 (k)], rb bytes apart; grid T, block 256
extern "C" __global__ void vpack_x(const half *x16, const int *sel, const float *wts, unsigned char *dst, int d, int k, int rb) {
  int t = blockIdx.x;
  unsigned char *row = dst + (size_t)t * rb;
  const uint4 *src = reinterpret_cast<const uint4 *>(x16 + (size_t)t * d);
  uint4 *out = reinterpret_cast<uint4 *>(row);
  for (int i = threadIdx.x; i < d / 8; i += blockDim.x) out[i] = src[i];
  if (threadIdx.x < k) {
    reinterpret_cast<int *>(row + d * 2)[threadIdx.x] = sel[t * k + threadIdx.x];
    reinterpret_cast<float *>(row + d * 2 + k * 4)[threadIdx.x] = wts[t * k + threadIdx.x];
  }
}

// routed SwiGLU over the pair rows p < *total
// grid-stride over the pair rows p < *total (the launch is sized for a typical step, not the worst case)
extern "C" __global__ void vswiglu_c(const float *gu, const float *wt, half *hout, const int *total, int P, int inter, float lim) {
  const size_t n = (size_t)min(*total, P) * inter;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int p = i / inter, c = i % inter;
    float g = fminf(gu[(size_t)p * 2 * inter + c], lim), u = fminf(fmaxf(gu[(size_t)p * 2 * inter + inter + c], -lim), lim);
    hout[i] = __float2half(bf16r(g / (1.f + expf(-g)) * u * wt[p]));
  }
}

// fwht_rows_f16 over the pair rows p < *total; grid (cols / 128, P)
extern "C" __global__ void vfwht_c(__half *w, int cols, const int *total) {
  __shared__ float fb[128];
  const int t = threadIdx.x, n = *total;
  for (int p = blockIdx.y; p < n; p += gridDim.y) {  // block-uniform loop: the barriers stay safe
    size_t i = (size_t)p * cols + blockIdx.x * 128 + t;
    fb[t] = __half2float(w[i]);
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
      float a = fb[t], b = fb[t ^ h];
      __syncthreads();
      fb[t] = (t & h) ? b - a : a + b;
      __syncthreads();
    }
    w[i] = __float2half(fb[t] * 0.08838834764831845f);
    __syncthreads();
  }
}

// zero the pair rows p < *total (width floats each); grid (ceil(width / 256), P)
extern "C" __global__ void vzero_c(float *rows, int width, const int *total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, n = *total;
  if (i >= width) return;
  for (int p = blockIdx.y; p < n; p += gridDim.y) rows[(size_t)p * width + i] = 0.f;
}

// f32 rows (stride ldx) -> fp16 rows (stride n); grid (ceil(n / 256), T)
extern "C" __global__ void vto_f16_rows(const float *x, int ldx, half *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[(size_t)blockIdx.y * n + i] = __float2half(x[(size_t)blockIdx.y * ldx + i]);
}

// zero T rows of n floats with stride ld; grid (ceil(n / 256), T)
extern "C" __global__ void vzero_rows(float *y, int ld, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[(size_t)blockIdx.y * ld + i] = 0.f;
}

// ---------------------------------------------------------------- NVLink-only exchanges on the DGX-1 hybrid cube
// GPUs form two quads {0..3}, {4..7}; every GPU has NVLink to its 3 quad-mates and its twin g ^ 4. Peer buffers are
// reached through UVA pointers (peer access enabled). Flags: each block of the sender bumps a per-(stage, src, block)
// u32 on the receiver to the exchange's sequence number after a system fence; receivers spin on them.
#define VX_BLOCKS 8
struct VxPeers {
  unsigned char *buf[8];    // each GPU's exchange buffer (all-gather: [8 src][slot] blocks; all-to-all: see below)
  unsigned char *relay[8];  // each GPU's relay buffer (all-to-all only: [4][slot], blocks forwarded for the far quad)
  volatile unsigned *flag[8];  // each GPU's flags [3 stages][8 src][VX_BLOCKS]
};

__device__ __forceinline__ void vx_copy(unsigned char *const *dst, int nd, const unsigned char *src, size_t bytes) {
  size_t n16 = bytes / 16, per = (n16 + VX_BLOCKS - 1) / VX_BLOCKS, lo = blockIdx.x * per, hi = min(n16, lo + per);
  for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    uint4 v = reinterpret_cast<const uint4 *>(src)[i];
    for (int k = 0; k < nd; k++) reinterpret_cast<uint4 *>(dst[k])[i] = v;
  }
}
__device__ __forceinline__ void vx_signal(volatile unsigned *const *f, int nf, unsigned seq) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    for (int k = 0; k < nf; k++) *f[k] = seq;
  }
}
__device__ __forceinline__ void vx_wait(volatile unsigned *f, unsigned seq) {
  if (threadIdx.x == 0) while (*f < seq) {}
  __syncthreads();
}
#define VXF(g, stage, src) (P.flag[g] + ((stage) * 8 + (src)) * VX_BLOCKS + blockIdx.x)

// All-gather: GPU me's block (bytes[me] bytes at buf[me] + me * slot) to every GPU's same offset. Stage 1: own block
// to the 3 quad-mates and the twin; stage 2: the twin's block (arrived in stage 1) on to the quad-mates.
extern "C" __global__ void __launch_bounds__(256) vx_allgather(const VxPeers *Pp, int me, size_t slot, const int *bytes, const unsigned *seqp, unsigned li) {
  const VxPeers &P = *Pp;
  const unsigned seq = *seqp + li;  // step base (device memory, so a captured graph replays) + layer
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

// All-to-all: GPU me sends src_part[d] (bytes[d] bytes, from its own memory) to GPU d's buf[d] + me * slot.
// NVLink neighbours (quad-mates, twin) directly; the far quad's 3 GPUs through the twin: stage 1 puts them in the
// twin's relay[tw] + j * slot (j = the far GPU's index in its quad), stage 2 has the twin forward them.
extern "C" __global__ void __launch_bounds__(256) vx_alltoall(const VxPeers *Pp, int me, size_t slot, const unsigned char *src_base,
                                                               size_t src_stride, const int *bytes, const unsigned *seqp, unsigned li) {
  const VxPeers &P = *Pp;
  const unsigned seq = *seqp + li;
  const int q = me & ~3, tw = me ^ 4, fq = q ^ 4;
  // own partials stay local; stage 1: direct to the 3 quad-mates and the twin; far quad -> the twin's relay
  { unsigned char *dst = P.buf[me] + me * slot; vx_copy(&dst, 1, src_base + me * src_stride, bytes[me]); }
  for (int i = 0; i < 4; i++) {
    int d = q + i;
    if (d == me) continue;
    unsigned char *dst = P.buf[d] + me * slot;
    vx_copy(&dst, 1, src_base + d * src_stride, bytes[d]);
  }
  { unsigned char *dst = P.buf[tw] + me * slot; vx_copy(&dst, 1, src_base + tw * src_stride, bytes[tw]); }
  for (int i = 0; i < 4; i++) {
    int d = fq + i;
    if (d == tw) continue;
    unsigned char *dst = P.relay[tw] + (size_t)(i * 8 + me) * slot;  // relay[tw][far index i][from me]
    vx_copy(&dst, 1, src_base + d * src_stride, bytes[d]);
  }
  {
    volatile unsigned *f[5];
    int n = 0;
    for (int i = 0; i < 4; i++) { int d = q + i; if (d != me) f[n++] = VXF(d, 0, me); }
    f[n++] = VXF(tw, 0, me);
    f[n++] = VXF(tw, 2, me);  // the relay data for the twin to forward
    vx_signal(f, n, seq);
  }
  // stage 2: forward what our twin left for each of our quad-mates (we are the far quad's relay for twin's quad)
  vx_wait(VXF(me, 2, tw), seq);
  for (int i = 0; i < 4; i++) {
    int d = q + i;
    if (d == me) continue;
    unsigned char *dst = P.buf[d] + tw * slot;
    vx_copy(&dst, 1, P.relay[me] + (size_t)(i * 8 + tw) * slot, bytes[d]);  // every sender's message to d is d's size
  }
  {
    volatile unsigned *f[3];
    int n = 0;
    for (int i = 0; i < 4; i++) { int d = q + i; if (d != me) f[n++] = VXF(d, 1, tw); }
    vx_signal(f, n, seq);
  }
  // done when the 4 neighbours (stage 0) and the far quad's 3 (stage 1, relayed by their twins) have landed here
  for (int i = 0; i < 4; i++) { int s = q + i; if (s != me) vx_wait(VXF(me, 0, s), seq); }
  vx_wait(VXF(me, 0, tw), seq);
  for (int i = 0; i < 4; i++) { int s = fq + i; if (s != tw) vx_wait(VXF(me, 1, s), seq); }
}

// a[t][d] = bf16(a[t][d] + sum_s parts[s][t][d]) over S sources of fp16 partials; grid (ceil(T*d/256))
extern "C" __global__ void vsum_parts_h(float *a, const half *parts, int S, int stride, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = a[i];
  for (int s = 0; s < S; s++) v += __half2float(parts[(size_t)s * stride + i]);
  a[i] = bf16r(v);
}

// sparse_attn_tile with NSUB 32-row sub-tiles per block (online softmax across them), so a token's 640 rows make 4
// partials per head instead of 20 (the partial traffic dominated). Same partial layout as sparse_attn_tile: part
// [T][S][H][hd + 2] (acc, max, sum), S = gridDim.z, merged by sparse_attn_merge. Sub-tiles with no valid row are
// skipped. grid (T, H / AP_H, ceil((W + K) / (32 * NSUB))), block 256.
template <int NSUB>
__device__ void vattn_tile_t(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W, const __nv_bfloat16 *comp,
                             const int *comp_idx, int K, float *part, int H, int hd, float scale) {
  __shared__ __align__(16) __nv_bfloat16 R[AP_R][AP_LD];
  __shared__ float P[AP_H][AP_R];
  __shared__ float Mt[AP_H], Lt[AP_H];
  __shared__ int ok[AP_R];
  __shared__ int any;
  const int t = blockIdx.x, h0 = blockIdx.y * AP_H, z = blockIdx.z, S = gridDim.z, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int nrows = W + K;
  float acc[AP_H][2], M[AP_H], L[AP_H];  // running per head (every thread keeps the scalars for all heads)
#pragma unroll
  for (int h = 0; h < AP_H; h++) { acc[h][0] = acc[h][1] = 0.f; M[h] = -1e30f; L[h] = 0.f; }
  const float *qa = q + ((size_t)t * H + h0 + 2 * w) * hd, *qb = qa + hd;
  for (int sub = 0; sub < NSUB; sub++) {
    const int r0 = (z * NSUB + sub) * AP_R;
    if (r0 >= nrows) break;
    if (tid == 0) any = 0;
    __syncthreads();
    for (int c = tid; c < AP_R * 64; c += 256) {
      int r = c >> 6, part16 = c & 63, i = r0 + r;
      int idx = i >= nrows ? -1 : (i < W ? win_idx[t * W + i] : comp_idx[t * K + (i - W)]);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (idx >= 0) v = *(const uint4 *)((i < W ? win : comp) + (size_t)idx * hd + part16 * 8);
      *(uint4 *)&R[r][part16 * 8] = v;
      if (part16 == 0) { ok[r] = idx >= 0; if (idx >= 0) any = 1; }
    }
    __syncthreads();
    if (!any) continue;  // block-uniform: a sub-tile of empty slots adds nothing
    {
      float sa = 0.f, sb = 0.f;
      for (int d = 0; d < 512; d += 8) {
        uint4 rv = *(const uint4 *)&R[lane][d];
        const __nv_bfloat162 *r2 = (const __nv_bfloat162 *)&rv;
        float4 a0 = __ldg((const float4 *)(qa + d)), a1 = __ldg((const float4 *)(qa + d + 4));
        float4 b0 = __ldg((const float4 *)(qb + d)), b1 = __ldg((const float4 *)(qb + d + 4));
        float2 x0 = __bfloat1622float2(r2[0]), x1 = __bfloat1622float2(r2[1]), x2 = __bfloat1622float2(r2[2]), x3 = __bfloat1622float2(r2[3]);
        sa += a0.x * x0.x + a0.y * x0.y + a0.z * x1.x + a0.w * x1.y + a1.x * x2.x + a1.y * x2.y + a1.z * x3.x + a1.w * x3.y;
        sb += b0.x * x0.x + b0.y * x0.y + b0.z * x1.x + b0.w * x1.y + b1.x * x2.x + b1.y * x2.y + b1.z * x3.x + b1.w * x3.y;
      }
      bool v = ok[lane];
      sa = v ? sa * scale : -1e30f;
      sb = v ? sb * scale : -1e30f;
      float ma = warp_max(sa), mb = warp_max(sb);
      float pa = v ? expf(sa - ma) : 0.f, pb = v ? expf(sb - mb) : 0.f;
      float la = warp_sum(pa), lb = warp_sum(pb);
      P[2 * w][lane] = pa;
      P[2 * w + 1][lane] = pb;
      if (lane == 0) { Mt[2 * w] = ma; Lt[2 * w] = la; Mt[2 * w + 1] = mb; Lt[2 * w + 1] = lb; }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < AP_H; h++) {
      // fold this sub-tile into the running (max, sum, acc) of head h
      float mt = Mt[h], mn = fmaxf(M[h], mt), a = expf(M[h] - mn), b = expf(mt - mn);
      float s0 = 0.f, s1 = 0.f;
      for (int r = 0; r < AP_R; r++) {
        float2 x = __bfloat1622float2(*(const __nv_bfloat162 *)&R[r][2 * tid]);
        float p = P[h][r];
        s0 += p * x.x;
        s1 += p * x.y;
      }
      acc[h][0] = acc[h][0] * a + s0 * b;
      acc[h][1] = acc[h][1] * a + s1 * b;
      L[h] = L[h] * a + Lt[h] * b;
      M[h] = mn;
    }
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < AP_H; h++) {
    float *ph = part + (((size_t)t * S + z) * H + h0 + h) * (hd + 2);
    *(float2 *)(ph + 2 * tid) = make_float2(acc[h][0], acc[h][1]);
    if (tid == 0) { ph[hd] = M[h]; ph[hd + 1] = L[h]; }
  }
}
extern "C" __global__ void __launch_bounds__(256) vattn_tile5(const float *q, const __nv_bfloat16 *win, const int *win_idx, int W,
                                                              const __nv_bfloat16 *comp, const int *comp_idx, int K, float *part,
                                                              int H, int hd, float scale) {
  vattn_tile_t<5>(q, win, win_idx, W, comp, comp_idx, K, part, H, hd, scale);
}

// ---------------------------------------------------------------- lookup-table expert formats (MXFP4, cb3, cb2)
// Per 32 weights of a row (one "unit", one scale): MXFP4 = 4 words of nibbles (weight 2p = low nibble of byte p) + an
// E8M0 scale byte; cb3 = 3 words of 6-bit pair codes (pairs 0-4, 5-9, 10-14 at bits 6m of words 0-2; pair 15 in the
// top 2 bits of each word) + an E4M3 scale byte; cb2 = 2 words of 4-bit pair codes (pairs 0-7, 8-15) + E4M3.
// A pair code indexes a table of half2 (both weights' values): 256 / 64 / 16 entries in shared memory, so one lookup
// decodes two weights. Layout: words [row group][unit][word][lane] u32, scales [row group][unit][lane] u8; cb scales
// are multiplied by a per-matrix factor (VWork2::mul). Weight order inside a unit is natural (no x permutation).
#define VF_FP4 0
#define VF_CB3 1
#define VF_CB2 2
struct VWork2 {
  unsigned w_off;  // offset of the matrix's words, in 32-word (128-byte) units
  unsigned s_off;  // byte offset of its scales
  unsigned t0, nt;
  float mul;
  unsigned pad[3];
};

template <int FMT> __device__ __forceinline__ int vlut_idx(const uint32_t *q, int p) {
  if (FMT == VF_FP4) return (q[p >> 2] >> (8 * (p & 3))) & 255;
  if (FMT == VF_CB2) return (q[p >> 3] >> (4 * (p & 7))) & 15;
  if (p < 15) return (q[p / 5] >> (6 * (p % 5))) & 63;
  return (q[0] >> 30) | ((q[1] >> 30) << 2) | ((q[2] >> 30) << 4);
}

template <int FMT, int NT, int R>
__device__ __forceinline__ void vlut_mm(const uint32_t *__restrict__ W, const uint8_t *__restrict__ S, const VWork2 *__restrict__ work,
                                        const half2 *__restrict__ tab, const half *__restrict__ e43,
                                        const int *__restrict__ in_rows, const int *__restrict__ out_rows, const float *__restrict__ tw,
                                        const half *__restrict__ X, float *__restrict__ Y, int N, int K, int KC, int ldy,
                                        const int *__restrict__ cnt, const int *__restrict__ off, int lo, int hi, int rtiles, int ldx, const int *__restrict__ nwork) {
  constexpr int UW = FMT == VF_FP4 ? 4 : FMT == VF_CB3 ? 3 : 2, NTAB = FMT == VF_FP4 ? 256 : FMT == VF_CB3 ? 64 : 16, WARPS = 8;
  extern __shared__ __align__(16) unsigned char vsm[];
  half2 *ts = reinterpret_cast<half2 *>(vsm);                 // value-pair table
  half *es = reinterpret_cast<half *>(vsm + NTAB * 4);        // E4M3 scale table (cb formats)
  half *xs = reinterpret_cast<half *>(vsm + NTAB * 4 + 512);  // x tile
  const int LDX = KC + 8;
  // nwork: the entries are a compacted list of *nwork (hot experts); blocks loop over them by gridDim.y
  const int nent = nwork ? *nwork : (int)gridDim.y;
  for (int ei = blockIdx.y; ei < nent; ei += gridDim.y) {
  const VWork2 wk0 = work[ei];
  int bx = blockIdx.x, n_all = (int)wk0.nt, t_base = (int)wk0.t0, tt0 = 0, tstep = 1 << 30;
  if (cnt) {
    // device-built lists: this block takes token tiles tt0, tt0 + tstep, ... of its list (bounded grid, any count)
    n_all = cnt[wk0.nt];
    if (n_all <= lo || n_all > hi) continue;
    tt0 = blockIdx.x / rtiles;
    tstep = gridDim.x / rtiles;
    bx = blockIdx.x % rtiles;
    t_base = off[wk0.nt];
  }
  const int ntiles = (n_all + 8 * NT - 1) / (8 * NT);
  for (int tt = tt0; tt < ntiles; tt += tstep) {
  VWork2 wk = wk0;
  wk.t0 = t_base + tt * 8 * NT;
  wk.nt = min(8 * NT, n_all - tt * 8 * NT);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, q4 = (lane >> 2) & 3, h = lane >> 4;
  const int k0 = blockIdx.z * KC;
  if (k0 >= K) continue;
  const int kc = min(KC, K - k0), units = kc / 32;
  const int tok = (lane & 3) + 4 * h;
  const int rg0 = (bx * WARPS + warp) * R;
  const bool active = rg0 * 32 < N;
  const int rv = active ? min(R, N / 32 - rg0) : 0;  // this warp's row groups inside N
  const int units_all = K / 32, u0 = k0 / 32;
  for (int i = threadIdx.x; i < NTAB; i += WARPS * 32) ts[i] = tab[i];
  if (FMT != VF_FP4) for (int i = threadIdx.x; i < 256; i += WARPS * 32) es[i] = e43[i];
  const uint32_t *wp[R];
  const uint8_t *sp[R];
#pragma unroll
  for (int j = 0; j < R; j++) {
    size_t rg = (size_t)(rg0 + j);
    wp[j] = W + (size_t)wk.w_off * 32 + ((rg * units_all + u0) * UW) * 32 + lane;
    sp[j] = S + wk.s_off + (rg * units_all + u0) * 32 + lane;
  }
  uint32_t qn[R][UW];
  uint8_t sn[R];
#pragma unroll
  for (int j = 0; j < R; j++) {
#pragma unroll
    for (int w = 0; w < UW; w++) qn[j][w] = j < rv ? __ldg(wp[j] + w * 32) : 0u;
    sn[j] = j < rv ? __ldg(sp[j]) : 0;
  }
  for (int i = threadIdx.x; i < NT * 8 * (kc / 8); i += WARPS * 32) {
    int t = i / (kc / 8), c8 = (i % (kc / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (t < (int)wk.nt) v = *reinterpret_cast<const uint4 *>(X + (size_t)(in_rows ? in_rows[wk.t0 + t] : (int)wk.t0 + t) * ldx + k0 + c8);
    *reinterpret_cast<uint4 *>(&xs[t * LDX + c8]) = v;
  }
  __syncthreads();
  if (active) {
  const half mulh = __float2half(wk.mul);
  float acc[R][NT][8];
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][n][e] = 0.f;
  for (int u = 0; u < units; u++) {
    uint32_t qc[R][UW];
    half2 s2[R];
#pragma unroll
    for (int j = 0; j < R; j++) {
#pragma unroll
      for (int w = 0; w < UW; w++) qc[j][w] = qn[j][w];
      const uint8_t sb = sn[j];
      if (u + 1 < units && j < rv) {
#pragma unroll
        for (int w = 0; w < UW; w++) qn[j][w] = __ldg(wp[j] + ((size_t)(u + 1) * UW + w) * 32);
        sn[j] = __ldg(sp[j] + (size_t)(u + 1) * 32);
      }
      half s;
      if (FMT == VF_FP4) {
        // E8M0 2^(b - 127) as fp16 (clamped to fp16's normal range)
        unsigned short bits = (unsigned short)(min(max((int)sb - 112, 1), 30) << 10);
        s = *reinterpret_cast<half *>(&bits);
      } else {
        s = __hmul(es[sb], mulh);
      }
      s2[j] = __half2half2(s);
    }
#pragma unroll
    for (int m = 0; m < 8; m++) {  // 4 k per mma: pairs 2m, 2m+1
      const int kpos = u * 32 + 4 * m;
      uint32_t b[NT][2];
#pragma unroll
      for (int n = 0; n < NT; n++) { uint2 v = *reinterpret_cast<const uint2 *>(&xs[(n * 8 + tok) * LDX + kpos]); b[n][0] = v.x; b[n][1] = v.y; }
#pragma unroll
      for (int j = 0; j < R; j++) {
        half2 v0 = __hmul2(ts[vlut_idx<FMT>(qc[j], 2 * m)], s2[j]), v1 = __hmul2(ts[vlut_idx<FMT>(qc[j], 2 * m + 1)], s2[j]);
        uint32_t a0 = *reinterpret_cast<uint32_t *>(&v0), a1 = *reinterpret_cast<uint32_t *>(&v1);
#pragma unroll
        for (int n = 0; n < NT; n++) v_mma884(acc[j][n], a0, a1, b[n][0], b[n][1]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        int rs = 8 * q4 + (lane & 1) + 2 * ((e >> 1) & 1) + 4 * h, t = n * 8 + 2 * ((lane >> 1) & 1) + (e & 1) + 4 * (e >> 2);
        if (t < (int)wk.nt && j < rv) {
          int ti = wk.t0 + t;
          atomicAdd(&Y[(size_t)(out_rows ? out_rows[ti] : ti) * ldy + (rg0 + j) * 32 + rs], acc[j][n][e] * (tw ? tw[ti] : 1.f));
        }
      }
  }
  __syncthreads();  // the next tile reuses the x tile
  }
  }
}

#define VLUT(FMT, NAME, NT, R)                                                                                                 \
  extern "C" __global__ void __launch_bounds__(256) vlut_##NAME##_n##NT(                                                      \
      const uint32_t *W, const uint8_t *S, const VWork2 *work, const half2 *tab, const half *e43, const int *in_rows,           \
      const int *out_rows, const float *tw, const half *X, float *Y, int N, int K, int KC, int ldy, const int *cnt, const int *off, \
      int lo, int hi, int rtiles, int ldx, const int *nwork) {                                                                 \
    vlut_mm<FMT, NT, R>(W, S, work, tab, e43, in_rows, out_rows, tw, X, Y, N, K, KC, ldy, cnt, off, lo, hi, rtiles, ldx, nwork); \
  }
VLUT(VF_FP4, fp4, 1, 4)
VLUT(VF_FP4, fp4, 2, 2)
VLUT(VF_FP4, fp4, 4, 1)
VLUT(VF_CB3, cb3, 1, 4)
VLUT(VF_CB3, cb3, 2, 2)
VLUT(VF_CB3, cb3, 4, 1)
VLUT(VF_CB2, cb2, 1, 4)
VLUT(VF_CB2, cb2, 2, 2)
VLUT(VF_CB2, cb2, 4, 1)
// one row group per warp: 64 registers instead of 170, so 4x the warps per SM (tests/split_test.cu: 680 GB/s vs 550)
VLUT(VF_FP4, fp4r1, 1, 1)
VLUT(VF_FP4, fp4r1, 2, 1)
VLUT(VF_CB3, cb3r1, 1, 1)
VLUT(VF_CB3, cb3r1, 2, 1)
VLUT(VF_CB2, cb2r1, 1, 1)
VLUT(VF_CB2, cb2r1, 2, 1)

// OCP E4M3 (bias 7, max 448) encode of a non-negative value, round to nearest
__device__ __forceinline__ uint32_t v_e4m3_enc(float x) {
  if (!(x > 0.f)) return 0;
  if (x >= 448.f) return 0x7E;
  int e;
  float m = frexpf(x, &e);  // x = m * 2^e, m in [0.5, 1)
  int E = e - 1 + 7;        // biased exponent of 1.mmm form
  if (E <= 0) {             // subnormal: x / 2^-9
    int q = __float2int_rn(x * 512.f);
    return (uint32_t)min(q, 8) & 0x7F;  // q == 8 is the smallest normal
  }
  int mant = __float2int_rn((m * 2.f - 1.f) * 8.f);
  if (mant == 8) { mant = 0; E++; }
  if (E > 15 || (E == 15 && mant > 6)) return 0x7E;
  return (uint32_t)((E << 3) | mant);
}
__device__ __forceinline__ float v_e4m3_dec(uint32_t c) {
  int E = (c >> 3) & 15, M = c & 7;
  return E == 0 ? M * (1.f / 512.f) : ldexpf(1.f + M / 8.f, E - 7);
}

// MXFP4 rows (nibbles [n][k/2], E8M0 [n][k/32]) -> nearest codebook entries with the E4M3 scale (x mul) that
// minimizes squared error over 24 candidates (dsx-amd cb_quant, OCP E4M3 here). codes [n][k], sbytes [n][k/32].
extern "C" __global__ void vcb_quant(const uint8_t *w, const uint8_t *sc, int n, int k, const float *cb, int nl, float mul,
                                     uint8_t *codes, uint8_t *sbytes) {
  const int gi = blockIdx.x * blockDim.x + threadIdx.x, G = k / 32;
  if (gi >= n * G) return;
  const int r = gi / G, g = gi % G;
  const float E2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  float v[32];
  const float s = __uint_as_float((uint32_t)sc[(size_t)r * G + g] << 23);
  float amax = 0;
  for (int i = 0; i < 16; i++) {
    const uint8_t b = w[(size_t)r * (k / 2) + g * 16 + i];
    v[2 * i] = E2M1[b & 15] * s;
    v[2 * i + 1] = E2M1[b >> 4] * s;
    amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
  }
  float cbl[8], mid[7], cmax = 0;
  for (int c = 0; c < nl; c++) { cbl[c] = cb[c]; cmax = fmaxf(cmax, fabsf(cb[c])); }
  for (int c = 0; c + 1 < nl; c++) mid[c] = 0.5f * (cbl[c] + cbl[c + 1]);
  float best = INFINITY, ba = 0.f;
  uint32_t bcode = 0;
  for (int f = 0; f < 24; f++) {
    const float fac = 0.5f + f * (0.8f / 23.f);
    const uint32_t code = v_e4m3_enc(amax * fac / cmax / mul);
    const float a = v_e4m3_dec(code) * mul;
    if (a <= 0.f) continue;
    const float ia = 1.f / a;
    float err = 0;
    for (int i = 0; i < 32; i++) {
      const float x = v[i] * ia;
      int q = 0;
      for (int c = 0; c + 1 < nl; c++) q += x > mid[c];
      const float e = v[i] - a * cbl[q];
      err += e * e;
    }
    if (err < best) { best = err; bcode = code; ba = a; }
  }
  sbytes[(size_t)r * G + g] = (uint8_t)bcode;
  const float ia = ba > 0 ? 1.f / ba : 0.f;
  for (int i = 0; i < 32; i++) {
    const float x = v[i] * ia;
    int q = 0;
    for (int c = 0; c + 1 < nl; c++) q += x > mid[c];
    codes[(size_t)r * k + g * 32 + i] = (uint8_t)q;
  }
}

// lane of row slot rs inside a 32-row group (the mma A-operand owner, see vmoe_mm)
__device__ __forceinline__ int v_lane_of(int rs) { return 4 * (rs >> 3) + (rs & 3) + 16 * ((rs >> 2) & 1); }

// pack one matrix into the LUT layout: thread per (row, unit). fmt VF_FP4: nibbles [n][k/2] as is; VF_CB3 / VF_CB2:
// codes [n][k] (one byte each) into pair codes. scales [n][k/32] bytes.
extern "C" __global__ void vlut_pack(int fmt, const uint8_t *src, const uint8_t *scales, int n, int k, uint32_t *outw, uint8_t *outs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, units = k / 32;
  if (i >= n * units) return;
  const int r = i / units, u = i % units, rg = r / 32, lane = v_lane_of(r % 32);
  const int UW = fmt == VF_FP4 ? 4 : fmt == VF_CB3 ? 3 : 2;
  uint32_t q[4] = {0, 0, 0, 0};
  if (fmt == VF_FP4) {
    const uint8_t *b = src + (size_t)r * (k / 2) + u * 16;
    for (int w = 0; w < 4; w++) q[w] = b[4 * w] | (b[4 * w + 1] << 8) | (b[4 * w + 2] << 16) | ((uint32_t)b[4 * w + 3] << 24);
  } else {
    const uint8_t *c = src + (size_t)r * k + u * 32;
    const int bits = fmt == VF_CB3 ? 3 : 2;
    for (int p = 0; p < 16; p++) {
      const uint32_t pc = c[2 * p] | (c[2 * p + 1] << bits);
      if (fmt == VF_CB2) q[p >> 3] |= pc << (4 * (p & 7));
      else if (p < 15) q[p / 5] |= pc << (6 * (p % 5));
      else { q[0] |= (pc & 3) << 30; q[1] |= ((pc >> 2) & 3) << 30; q[2] |= ((pc >> 4) & 3) << 30; }
    }
  }
  const size_t base = (size_t)rg * units + u;
  for (int w = 0; w < UW; w++) outw[(base * UW + w) * 32 + lane] = q[w];
  outs[base * 32 + lane] = scales[(size_t)r * units + u];
}

// entries of a work list whose count is > lo, copied (stride words each) to out in list order, their number in *n_out
extern "C" __global__ void vcompact(const unsigned *work, int n, int stride, const int *cnt, int lo, unsigned *out, int *n_out) {
  __shared__ int k;
  if (threadIdx.x == 0) k = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    if (cnt[work[i * stride + 3]] > lo) {
      const int j = atomicAdd(&k, 1);
      for (int w = 0; w < stride; w++) out[j * stride + w] = work[i * stride + w];
    }
  __syncthreads();
  if (threadIdx.x == 0) *n_out = k;
}
