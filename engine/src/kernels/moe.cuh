// Device-driven routed experts: the router's picks (sel/wts in device memory) select which resident experts run,
// through a residency table, so a layer's GPU experts are two launches with no host round trip.
#pragma once

// one resident expert: byte offsets into the store + block types (ggml ids, or 2000 + id when repacked for
// expert_fast.cuh); gate == 0xffff... = not resident here
struct ExpSlot {
  unsigned long long gate, up, down;
  int gu, dn;
};

// partial dot of one row (k weights, ggml type t) with x, lanes stride over blocks; caller warp-reduces
__device__ __forceinline__ float row_dot_lane(int t, const uint8_t *row, const float *x, int k, int lane) {
  float acc = 0.f;
  if (t == 42) {  // Q2_0
    int nb = k / 64;
    for (int b = lane; b < nb; b += 32) {
      const uint8_t *blk = row + b * 18;
      float d = ld_h(blk), s = 0.f;
      const float4 *xp = (const float4 *)(x + b * 64);
#pragma unroll
      for (int m = 0; m < 16; m++) {
        uint8_t q = blk[2 + m];
        float4 v = xp[m];
        s += (float)((q & 3) - 1) * v.x + (float)(((q >> 2) & 3) - 1) * v.y + (float)(((q >> 4) & 3) - 1) * v.z +
             (float)(((q >> 6) & 3) - 1) * v.w;
      }
      acc += d * s;
    }
  } else if (t == 143) {  // PTQ1_0
    const uint32_t p3[5] = {1, 3, 9, 27, 81};
    int nb = k / 128;
    for (int b = lane; b < nb; b += 32) {
      const uint8_t *blk = row + b * 28;
      const float *xb = x + b * 128;
      float d = ld_h(blk + 26), s = 0.f;
#pragma unroll
      for (int m = 0; m < 16; m++)
#pragma unroll
        for (int n = 0; n < 5; n++) s += trit(blk[m], p3[n]) * xb[m + 16 * n];
#pragma unroll
      for (int m = 0; m < 8; m++)
#pragma unroll
        for (int n = 0; n < 5; n++) s += trit(blk[16 + m], p3[n]) * xb[80 + m + 8 * n];
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int n = 0; n < 4; n++) s += trit(blk[24 + h], p3[n]) * xb[120 + h + 2 * n];
      acc += d * s;
    }
  } else if (t == 11) {  // Q3_K, lanes over 16-weight sub-blocks
    int nb = k / 256;
    for (int sb = lane; sb < nb * 16; sb += 32) {
      int b = sb >> 4, s16 = sb & 15;
      const uint8_t *blk = row + b * 110;
      const uint8_t *hm = blk, *qs = blk + 32, *raw = blk + 96;
      float d = ld_h(blk + 108);
      int i4 = s16 & 3, grp = s16 >> 2;
      int a = raw[i4], c = raw[4 + i4], tmp = raw[8 + i4], sc;
      if (grp == 0) sc = (a & 15) | ((tmp & 3) << 4);
      else if (grp == 1) sc = (c & 15) | (((tmp >> 2) & 3) << 4);
      else if (grp == 2) sc = (a >> 4) | (((tmp >> 4) & 3) << 4);
      else sc = (c >> 4) | (((tmp >> 6) & 3) << 4);
      const float *xb = x + b * 256 + 16 * s16;
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        int e = 16 * s16 + i, h = e >> 7, j = (e >> 5) & 3, l = e & 31;
        int low = (qs[32 * h + l] >> (2 * j)) & 3, hb = (hm[l] >> (4 * h + j)) & 1;
        s += (float)(low - (hb ? 0 : 4)) * xb[i];
      }
      acc += d * (float)(sc - 32) * s;
    }
  } else if (t == 8) {  // Q8_0
    int nb = k / 32;
    for (int b = lane; b < nb; b += 32) {
      const uint8_t *blk = row + b * 34;
      float d = ld_h(blk), s = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) s += (float)(int8_t)blk[2 + j] * x[b * 32 + j];
      acc += d * s;
    }
  }
  return acc;
}

__device__ __forceinline__ int row_bytes(int t, int k) {
  return t == 42 ? k / 64 * 18 : t == 143 ? k / 128 * 28 : t == 11 ? k / 256 * 110 : k / 32 * 34;
}

// h[j][r] = bf16(silu(min(gate_j[r].x, lim)) * clamp(up_j[r].x, -lim, lim) * w_j) for the picks j resident here.
// tab: [layers][E] slots; sel/wts: [K] for this token. grid (ceil(inter/4), K), block 128 (warp per row).
extern "C" __global__ void moe_gu(const ExpSlot *tab, int layer, int E, const int *sel, const float *wts, int K,
                                  const uint8_t *store, const float *x, float *h, int inter, int dim, float lim) {
  int j = blockIdx.y, r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= inter) return;
  ExpSlot s = tab[layer * E + sel[j]];
  if (s.gate == ~0ull) return;
  int rb = row_bytes(s.gu, dim);
  float g = warp_sum(row_dot_lane(s.gu, store + s.gate + (size_t)r * rb, x, dim, lane));
  float u = warp_sum(row_dot_lane(s.gu, store + s.up + (size_t)r * rb, x, dim, lane));
  if (lane == 0) {
    g = fminf(g, lim);
    u = fminf(fmaxf(u, -lim), lim);
    h[(size_t)j * inter + r] = bf16r(g / (1.f + expf(-g)) * u * wts[j]);
  }
}

// out[r] = sum over resident picks j of down_j[r] . h[j]; grid ceil(dim/4), block 128
extern "C" __global__ void moe_down(const ExpSlot *tab, int layer, int E, const int *sel, int K, const uint8_t *store,
                                    const float *h, float *out, int inter, int dim) {
  int r = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= dim) return;
  float acc = 0.f;
  for (int j = 0; j < K; j++) {
    ExpSlot s = tab[layer * E + sel[j]];
    if (s.gate == ~0ull) continue;
    int rb = row_bytes(s.dn, inter);
    acc += row_dot_lane(s.dn, store + s.down + (size_t)r * rb, h + (size_t)j * inter, inter, lane);
  }
  acc = warp_sum(acc);
  if (lane == 0) out[r] = acc;
}
