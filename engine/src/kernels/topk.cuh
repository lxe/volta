// Indexer selection on the GPU (model.rs indexer(): candidate blocks, masking, top-k), no host round trip.
// One kernel, topk_level, run as a tree: a block takes up to TK_CHUNK elements of a row and keeps its top kk by
// (score desc, index asc), compacted in index order into kcap slots; the next level reads those slots, until one
// block per row is left, which writes the result. Selection is a radix select on order-preserving u32 keys
// (11 + 11 + 10 bits; elements live in registers, the threshold bin's keys move to shared memory after the first
// pass); the threshold key's ties go to the first elements in array order, which is index order because every
// level compacts in order. So each block keeps exactly its share of the global top-k.

#define TK_THREADS 1024
#define TK_PER 16 // max elements per thread
#define TK_CHUNK (TK_THREADS * TK_PER)
#define TK_LIST 8192 // threshold-bin keys kept in shared memory; more (heavy ties) -> passes stay on registers
#define TK_NONE 0xffffffffu // idx of an empty slot
#define TK_KEY_NEGINF 0x007fffffu // tk_key(-inf)

__device__ __forceinline__ uint32_t tk_key(float f) {
  if (f == 0.f) f = 0.f; // -0 ties +0, as partial_cmp
  uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// exclusive scan of v over the block (1024 threads); *total = block sum. `sh` (33 ints) is free again after the
// next barrier.
__device__ __forceinline__ int tk_scan(int v, int *sh, int *total) {
  int w = threadIdx.x >> 5, l = threadIdx.x & 31, x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffff, x, o);
    if (l >= o) x += y;
  }
  if (l == 31) sh[w] = x;
  __syncthreads();
  if (w == 0) {
    int q = sh[l], z = q;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_up_sync(0xffffffff, z, o);
      if (l >= o) z += y;
    }
    sh[l] = z - q;
    if (l == 31) sh[32] = z;
  }
  __syncthreads();
  *total = sh[32];
  return sh[w] + x - v;
}

// Warp 0 over hist[2048] (lane l: bins [64l, 64l+64)): find the bin b with above(b) < r <= above(b) + hist[b],
// above = sum of the higher bins; r = min(k, total) when first. Writes pk = {b, r - above(b), hist[b]}.
__device__ __forceinline__ void tk_pick(const int *hist, int *pk, int r, bool first) {
  if (threadIdx.x >= 32) return;
  const int l = threadIdx.x;
  const int4 *h = (const int4 *)hist + 16 * l;
  int v = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) { // rotated: no bank conflicts between lanes
    int4 q = h[(i + l) & 15];
    v += q.x + q.y + q.z + q.w;
  }
  int x = v; // suffix sum over lanes >= l
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_down_sync(0xffffffff, x, o);
    if (l + o < 32) x += y;
  }
  if (first) r = min(r, __shfl_sync(0xffffffff, x, 0));
  // the lane L whose range holds the threshold, then its 64 bins two per lane
  unsigned hit = __ballot_sync(0xffffffff, x - v < r && r <= x);
  if (!hit) { // r == 0: nothing to take
    if (l == 0) pk[0] = 0, pk[1] = 0, pk[2] = 0;
    return;
  }
  int L = __ffs(hit) - 1, above = __shfl_sync(0xffffffff, x - v, L);
  int2 c = ((const int2 *)hist)[32 * L + l];
  v = c.x + c.y, x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_down_sync(0xffffffff, x, o);
    if (l + o < 32) x += y;
  }
  int ex1 = above + x - v, ex0 = ex1 + c.y; // bins above 64L+2l+1, above 64L+2l
  if (ex1 < r && r <= ex1 + c.y) pk[0] = 64 * L + 2 * l + 1, pk[1] = r - ex1, pk[2] = c.y;
  if (ex0 < r && r <= ex0 + c.x) pk[0] = 64 * L + 2 * l, pk[1] = r - ex0, pk[2] = c.x;
}

// One block's selection with E elements per thread; thread t holds elements [tE, tE+E) of the chunk (index order).
template <int E>
__device__ __forceinline__ void topk_body(const float *__restrict__ s, const uint8_t *__restrict__ mask,
                                          const uint2 *__restrict__ in, int n, int nb, int m, int k, int bs, int kcap,
                                          int in_mode, int out_mode, uint2 *__restrict__ out, int *__restrict__ pos,
                                          uint8_t *__restrict__ keep, int *hist, uint32_t *list, int *sh) {
  int *pk = sh + 33, &nlist = sh[36];
  const int l = threadIdx.x & 31;
  const int e0 = blockIdx.x * TK_CHUNK + threadIdx.x * E;
  uint32_t key[E], idx[E];
  if (in_mode == 2) {
#pragma unroll
    for (int j = 0; j < E; j++) {
      uint2 p = e0 + j < m ? in[e0 + j] : make_uint2(0, TK_NONE);
      key[j] = p.x, idx[j] = p.y;
    }
  } else if (in_mode == 0) {
    float v[E];
    if (E >= 4 && e0 + E <= m && !((uintptr_t)(s + e0) & 15)) {
#pragma unroll
      for (int j = 0; j < E; j += 4) *(float4 *)&v[j] = *(const float4 *)(s + e0 + j);
    } else {
#pragma unroll
      for (int j = 0; j < E; j++) v[j] = e0 + j < m ? s[e0 + j] : 0.f;
    }
#pragma unroll
    for (int j = 0; j < E; j++) {
      int e = e0 + j;
      bool kept = !mask || (e < m && mask[e / bs]);
      key[j] = tk_key(kept ? v[j] : -INFINITY);
      idx[j] = e < m ? e : TK_NONE;
    }
  } else {
#pragma unroll
    for (int j = 0; j < E; j++) {
      int e = e0 + j, lo = e * bs;
      float v = -INFINITY;
      if (bs == 8 && lo + 8 <= n && !((uintptr_t)(s + lo) & 15)) {
        float4 a = *(const float4 *)(s + lo), b = *(const float4 *)(s + lo + 4);
        v = fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)), fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w)));
      } else if (e < m) {
        for (int i = lo, hi = min(lo + bs, n); i < hi; i++) v = fmaxf(v, s[i]);
      }
      key[j] = tk_key(e == nb - 1 ? INFINITY : v);
      idx[j] = e < m ? e : TK_NONE;
    }
  }
  // radix select: prefix P under mask M; rem = how many of the elements matching P are still to take
  uint32_t P = 0, M = 0;
  int rem = k;
  bool on_list = false;
  for (int pass = 0; pass < 3; pass++) {
    const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0;
    const uint32_t dm = pass == 2 ? 1023u : 2047u;
    ((int2 *)hist)[threadIdx.x] = make_int2(0, 0);
    __syncthreads();
    if (on_list) {
      for (int i = threadIdx.x; i < nlist; i += TK_THREADS)
        if ((list[i] & M) == P) atomicAdd(&hist[(list[i] >> shift) & dm], 1);
    } else {
#pragma unroll
      for (int j = 0; j < E; j++) {
        int d = (idx[j] != TK_NONE && (key[j] & M) == P) ? (int)((key[j] >> shift) & dm) : -1;
        int d0 = __shfl_sync(0xffffffff, d, 0);
        if (__all_sync(0xffffffff, d == d0)) { // one bin for the whole warp (ties): one atomic
          if (l == 0 && d0 >= 0) atomicAdd(&hist[d0], 32);
        } else if (d >= 0)
          atomicAdd(&hist[d], 1);
      }
    }
    __syncthreads();
    tk_pick(hist, pk, rem, pass == 0);
    if (pass == 0 && threadIdx.x == 0) nlist = 0;
    __syncthreads();
    P |= (uint32_t)pk[0] << shift;
    M |= dm << shift;
    rem = pk[1];
    const int cnt = pk[2]; // pk, hist: rewritten only after the next pass's barriers
    if (cnt == rem) break; // every element left at P is taken
    if (pass == 0 && cnt <= TK_LIST) { // move the threshold bin's keys to shared memory (any order)
      unsigned inb = 0;
#pragma unroll
      for (int j = 0; j < E; j++) inb |= (idx[j] != TK_NONE && (key[j] & M) == P) ? 1u << j : 0;
      int c = __popc(inb), x = c;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, o);
        if (l >= o) x += y;
      }
      int o = 0;
      if (l == 31 && x) o = atomicAdd(&nlist, x);
      o = __shfl_sync(0xffffffff, o, 31) + x - c;
#pragma unroll
      for (int j = 0; j < E; j++)
        if ((inb >> j) & 1) list[o++] = key[j];
      __syncthreads();
      on_list = true;
    }
  }
  // take all above P and the first rem equal to it; one block scan of (above, equal) counts gives each its slot
  unsigned hib = 0, eqb = 0;
#pragma unroll
  for (int j = 0; j < E; j++) {
    bool v = idx[j] != TK_NONE;
    hib |= v && (key[j] & M) > P ? 1u << j : 0;
    eqb |= v && (key[j] & M) == P ? 1u << j : 0;
  }
  int tot;
  int o = tk_scan(__popc(hib) | __popc(eqb) << 16, sh, &tot);
  int h = o & 0xffff, q = o >> 16;
  const int kk = (tot & 0xffff) + rem;
#pragma unroll
  for (int j = 0; j < E; j++) {
    bool hi = (hib >> j) & 1, eq = (eqb >> j) & 1;
    if (hi || (eq && q < rem)) {
      int slot = h + min(q, rem);
      if (out_mode == 0) out[(size_t)blockIdx.x * kcap + slot] = make_uint2(key[j], idx[j]);
      else if (out_mode == 1) pos[slot] = key[j] > TK_KEY_NEGINF ? (int)idx[j] : -1;
      else if (key[j] > TK_KEY_NEGINF) keep[idx[j]] = 1; // after topk_level's zeroing: ordered by the barriers
    }
    h += hi;
    q += eq;
  }
  for (int t = kk + threadIdx.x; t < kcap; t += TK_THREADS) {
    if (out_mode == 0) out[(size_t)blockIdx.x * kcap + t] = make_uint2(0, TK_NONE);
    else if (out_mode == 1) pos[t] = -1;
  }
}

// Rows: blockIdx.y; row n = ns[row] (or n_max when ns is null); elements of a row: n scores, ceil(n/bs) blocks, or
// ldin pairs. k = min(kcap, n) for top-k, min(kcap, nblocks) for candidate blocks.
// in_mode:  0 = scores s[row*lds + i], -inf where mask (if given) has mask[row*ldm + i/bs] == 0
//           1 = candidate blocks: max of s over [b*bs, b*bs+bs) ∩ [0, n), the block holding n-1 pinned to +inf
//           2 = (key, idx) pairs in[row*ldin + e] from the previous level
// out_mode: 0 = pairs out[row*ldout + blockIdx.x*kcap + j], empty slots idx = TK_NONE
//           1 = top-k positions ascending in pos[row*ldp + j], j < k (-1 where the score is -inf; -1 pads to kcap)
//           2 = keep[row*ldk + b] = 1 for kept blocks, 0 for the others (b < nblocks); -inf blocks are not kept
extern "C" __global__ void __launch_bounds__(TK_THREADS, 1)
    topk_level(const float *__restrict__ s, int lds, const uint8_t *__restrict__ mask, int ldm,
               const uint2 *__restrict__ in, int ldin, const int *__restrict__ ns, int n_max, int bs, int kcap,
               int in_mode, int out_mode, uint2 *__restrict__ out, int ldout, int *__restrict__ pos, int ldp,
               uint8_t *__restrict__ keep, int ldk) {
  __shared__ __align__(16) int hist[2048];
  __shared__ int sh[37];
  __shared__ uint32_t list[TK_LIST];
  const size_t row = blockIdx.y;
  const int n = ns ? ns[row] : n_max;
  const int nb = (n + bs - 1) / bs;
  const int m = in_mode == 0 ? n : in_mode == 1 ? nb : ldin;
  const int k = min(kcap, in_mode == 0 ? n : nb);
  s += row * lds;
  mask = mask ? mask + row * ldm : mask;
  in += row * ldin;
  out += row * ldout;
  pos += row * ldp;
  keep += row * ldk;
  if (out_mode == 2)
    for (int b = threadIdx.x; b < nb; b += TK_THREADS) keep[b] = 0;
  // E per thread: the smallest power of two that holds this block's elements, so all warps share the work
  const int mc = min(TK_CHUNK, m - (int)blockIdx.x * TK_CHUNK);
#define TK_BODY(E) topk_body<E>(s, mask, in, n, nb, m, k, bs, kcap, in_mode, out_mode, out, pos, keep, hist, list, sh)
  if (mc <= TK_THREADS) TK_BODY(1);
  else if (mc <= 2 * TK_THREADS) TK_BODY(2);
  else if (mc <= 4 * TK_THREADS) TK_BODY(4);
  else if (mc <= 8 * TK_THREADS) TK_BODY(8);
  else TK_BODY(TK_PER);
#undef TK_BODY
}

// Candidate-block keys as pairs for topk_level in_mode 2, spread over the grid (one block reading all of a long row
// is bandwidth-bound): out[row*ldout + b] = (key of the block max, b) for b < ceil(n/bs), empty past it.
extern "C" __global__ void topk_blockmax(const float *__restrict__ s, int lds, const int *__restrict__ ns, int n_max,
                                         int bs, uint2 *__restrict__ out, int ldout) {
  const size_t row = blockIdx.y;
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= ldout) return;
  const int n = ns ? ns[row] : n_max, nb = (n + bs - 1) / bs, lo = b * bs;
  s += row * lds;
  uint2 r = make_uint2(0, TK_NONE);
  if (b < nb) {
    float v = -INFINITY;
    if (bs == 8 && lo + 8 <= n && !((uintptr_t)(s + lo) & 15)) {
      float4 a = *(const float4 *)(s + lo), c = *(const float4 *)(s + lo + 4);
      v = fmaxf(fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)), fmaxf(fmaxf(c.x, c.y), fmaxf(c.z, c.w)));
    } else {
      for (int i = lo, hi = min(lo + bs, n); i < hi; i++) v = fmaxf(v, s[i]);
    }
    r = make_uint2(tk_key(b == nb - 1 ? INFINITY : v), b);
  }
  out[row * ldout + b] = r;
}
