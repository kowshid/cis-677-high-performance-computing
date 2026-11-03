// Project 3 baseline -- solution.cpp. YOURS to change: everything in this file
// is timed, and it is the only file you edit (plus build.sh).
//
// Two parts:
//   1. THE TRAINER (struct Trainer): already fast. It carries the Project 2
//      techniques that won: CSR input, SoA Adam fused into one pass, every
//      buffer allocated once, threads that own their outputs, fp32, and the
//      Gram form of layer 4 (the 38,606-wide reconstruction is never formed).
//   2. THE LOADER (hpcbench::solve, at the bottom): naive on purpose. One
//      thread, one batch at a time: two small pread calls per cell, then build
//      the batch, then train on it. Training waits while it reads.
//
// The loader is where the time goes. Profile it, then make it fast:
// overlap reading with training, keep several reads in flight, read fewer and
// larger pieces, build cheaper -- within the memory rule in task.hpp.
#include "task.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// One batch as the trainer wants it: local CSR, values already log1p(count).
struct Batch {
  u32 n = 0;
  std::vector<u32> ptr, idx;
  std::vector<float> val;
};

struct Adam1 {
  std::vector<float> w, m, v;
  void init(std::vector<float> src) { w = std::move(src); m.assign(w.size(), 0.f); v.assign(w.size(), 0.f); }
};
static inline void adam_upd(float& w, float& m, float& v, float g, float c1, float c2) {
  m = 0.9f * m + 0.1f * g;
  v = 0.999f * v + 0.001f * g * g;
  w -= 1e-3f * (m / c1) / (std::sqrt(v / c2) + 1e-8f);
}


struct Trainer {
  static constexpr u64 H = hpcbench::kHidden, K = hpcbench::kLatent;  // compile-time widths: fixed-trip loops vectorize fully
  u64 D, BM;
  std::vector<unsigned char> owner;
  Adam1 W1, b1, W2, b2, W3, b3, W4T, b4;
  std::vector<float> G, u;  // G = W4 W4^T (H x H), u = W4 b4 (H) for the current weights
  // per-step buffers, allocated once
  std::vector<float> A1, H1, Z, A3, H3, P, dA3, dZ, dH1, M, c, gW2, gb2, gW3, gb3, gb1;
  std::vector<float> gW1, gW4s, xs;  // dense D x H gradient rows, written only for touched genes
  std::vector<unsigned char> touched;
  float b4sq = 0.f;
  std::vector<float> Gpart, upart;
  double bt1 = 1, bt2 = 1;
  int T;

  void init(const hpcbench::Params& p) {
    D = p.n_genes; BM = p.batch;
    T = omp_get_max_threads();
    owner.resize(D);
    for (u64 g = 0; g < D; ++g) owner[g] = (unsigned char)((g >> 6) % u64(T));
    W1.init(std::vector<float>(p.W1, p.W1 + D * H));
    W2.init(std::vector<float>(p.W2, p.W2 + H * K));
    W3.init(std::vector<float>(p.W3, p.W3 + K * H));
    W4T.init(std::vector<float>(p.W4T, p.W4T + D * H));
    b1.init(std::vector<float>(p.b1, p.b1 + H)); b2.init(std::vector<float>(p.b2, p.b2 + K));
    b3.init(std::vector<float>(p.b3, p.b3 + H)); b4.init(std::vector<float>(p.b4, p.b4 + D));
    A1.resize(BM * H); H1.resize(BM * H); Z.resize(BM * K); A3.resize(BM * H); H3.resize(BM * H);
    P.resize(BM * H); dA3.resize(BM * H); dZ.resize(BM * K); dH1.resize(BM * H);
    M.resize(H * H); c.resize(H); gW2.resize(H * K); gb2.resize(K); gW3.resize(K * H); gb3.resize(H); gb1.resize(H);
    gW1.assign(D * H, 0.f); gW4s.assign(D * H, 0.f); xs.assign(D, 0.f); touched.assign(D, 0);
    G.assign(H * H, 0.f); u.assign(H, 0.f);
    Gpart.resize(size_t(T) * H * H); upart.resize(size_t(T) * H);
    b4sq = 0.f;
    for (u64 g = 0; g < D; ++g) {
      const float* w = &W4T.w[g * H];
      for (u64 k = 0; k < H; ++k) {
        u[k] += b4.w[g] * w[k];
        for (u64 l = 0; l < H; ++l) G[k * H + l] += w[k] * w[l];
      }
      b4sq += b4.w[g] * b4.w[g];
    }
  }

  // one Adam step on one batch; returns the batch loss (mean squared error) before the step
  double step(const Batch& b) {
    const u64 B = b.n;
    const float s = 2.0f / float(double(B) * double(D));
    const u32* bp = b.ptr.data();
    const u32* bi = b.idx.data();
    const float* bv = b.val.data();
    double loss = 0.0;

    // forward: A1 = Xb W1 + b1, P = Xb W4^T (one pass over each row's non-zeros)
#pragma omp parallel for schedule(static) reduction(+ : loss)
    for (u64 i = 0; i < B; ++i) {
      float* a = &A1[i * H];
      float* p = &P[i * H];
      for (u64 k = 0; k < H; ++k) { a[k] = b1.w[k]; p[k] = 0.f; }
      double xx = 0.0, xb = 0.0;
      for (u32 e = bp[i]; e < bp[i + 1]; ++e) {
        const float x = bv[e];
        const float* w1 = &W1.w[u64(bi[e]) * H];
        const float* w4 = &W4T.w[u64(bi[e]) * H];
#pragma omp simd
        for (u64 k = 0; k < H; ++k) { a[k] += x * w1[k]; p[k] += x * w4[k]; }
        xx += double(x) * x;
        xb += double(x) * b4.w[bi[e]];
      }
      float* h1 = &H1[i * H];
      for (u64 k = 0; k < H; ++k) h1[k] = a[k] > 0.f ? a[k] : 0.f;
      // layers 2 and 3
      float* z = &Z[i * K];
      for (u64 j = 0; j < K; ++j) z[j] = b2.w[j];
      for (u64 k = 0; k < H; ++k) {
        const float hv = h1[k];
        if (hv == 0.f) continue;
        const float* w = &W2.w[k * K];
        for (u64 j = 0; j < K; ++j) z[j] += hv * w[j];
      }
      float* a3 = &A3[i * H];
      float* h3 = &H3[i * H];
      for (u64 k = 0; k < H; ++k) a3[k] = b3.w[k];
      for (u64 j = 0; j < K; ++j) {
        const float zv = z[j];
        const float* w = &W3.w[j * H];
        for (u64 k = 0; k < H; ++k) a3[k] += zv * w[k];
      }
      for (u64 k = 0; k < H; ++k) h3[k] = a3[k] > 0.f ? a3[k] : 0.f;
      // dA3 = s (G h + u - p), and this row's loss from the same quantities
      float* d = &dA3[i * H];
      double hGh = 0.0, hu = 0.0, hp = 0.0;
      for (u64 k = 0; k < H; ++k) {
        const float* gk = &G[k * H];
        float t = 0.f;
#pragma omp simd reduction(+ : t)
        for (u64 l = 0; l < H; ++l) t += gk[l] * h3[l];
        hGh += double(h3[k]) * t;
        hu += double(h3[k]) * u[k];
        hp += double(h3[k]) * p[k];
        d[k] = a3[k] > 0.f ? s * (t + u[k] - p[k]) : 0.f;
      }
      loss += xx - 2.0 * (hp + xb) + hGh + 2.0 * hu + b4sq;
    }
    loss /= double(B) * double(D);

    // back through layers 3 and 2 (small): threads own rows of each gradient
#pragma omp parallel for schedule(static)
    for (u64 j = 0; j < K; ++j) {
      float* gw = &gW3[j * H];
      for (u64 k = 0; k < H; ++k) gw[k] = 0.f;
      for (u64 i = 0; i < B; ++i) {
        const float zv = Z[i * K + j];
        const float* d = &dA3[i * H];
        for (u64 k = 0; k < H; ++k) gw[k] += zv * d[k];
      }
    }
#pragma omp parallel for schedule(static)
    for (u64 i = 0; i < B; ++i) {
      const float* d = &dA3[i * H];
      float* dz = &dZ[i * K];
      for (u64 j = 0; j < K; ++j) {
        const float* w = &W3.w[j * H];
        float t = 0.f;
        for (u64 k = 0; k < H; ++k) t += d[k] * w[k];
        dz[j] = t;
      }
      float* dh = &dH1[i * H];
      const float* a1 = &A1[i * H];
      for (u64 k = 0; k < H; ++k) {
        const float* w = &W2.w[k * K];
        float t = 0.f;
        for (u64 j = 0; j < K; ++j) t += dz[j] * w[j];
        dh[k] = a1[k] > 0.f ? t : 0.f;
      }
    }
#pragma omp parallel for schedule(static)
    for (u64 k = 0; k < H; ++k) {
      float* gw = &gW2[k * K];
      for (u64 j = 0; j < K; ++j) gw[j] = 0.f;
      for (u64 i = 0; i < B; ++i) {
        const float hv = H1[i * H + k];
        if (hv == 0.f) continue;
        const float* dz = &dZ[i * K];
        for (u64 j = 0; j < K; ++j) gw[j] += hv * dz[j];
      }
    }
    for (u64 k = 0; k < H; ++k) { gb1[k] = 0.f; gb3[k] = 0.f; c[k] = 0.f; }
    for (u64 j = 0; j < K; ++j) gb2[j] = 0.f;
    for (u64 i = 0; i < B; ++i) {
      for (u64 k = 0; k < H; ++k) { gb1[k] += dH1[i * H + k]; gb3[k] += dA3[i * H + k]; c[k] += H3[i * H + k]; }
      for (u64 j = 0; j < K; ++j) gb2[j] += dZ[i * K + j];
    }
    // M = H3^T H3 (H x H), threads own rows
#pragma omp parallel for schedule(static)
    for (u64 k = 0; k < H; ++k) {
      float* mk = &M[k * H];
      for (u64 l = 0; l < H; ++l) mk[l] = 0.f;
      for (u64 i = 0; i < B; ++i) {
        const float hk = H3[i * H + k];
        if (hk == 0.f) continue;
        const float* h = &H3[i * H];
        for (u64 l = 0; l < H; ++l) mk[l] += hk * h[l];
      }
    }
    // sparse gradient rows: thread t owns genes whose 64-gene block is t mod T, scans every non-zero
    const u64 nz = bp[B];
#pragma omp parallel
    {
      const unsigned t = omp_get_thread_num();
      for (u64 i = 0; i < B; ++i) {
        const float* dh = &dH1[i * H];
        const float* h3 = &H3[i * H];
        for (u32 e = bp[i]; e < bp[i + 1]; ++e) {
          const u64 g = bi[e];
          if (owner[g] != t) continue;
          float* g1 = &gW1[g * H];
          float* g4 = &gW4s[g * H];
          const float x = bv[e];
          if (!touched[g]) {
            touched[g] = 1;
#pragma omp simd
            for (u64 k = 0; k < H; ++k) { g1[k] = x * dh[k]; g4[k] = x * h3[k]; }
            xs[g] = x;
          } else {
#pragma omp simd
            for (u64 k = 0; k < H; ++k) { g1[k] += x * dh[k]; g4[k] += x * h3[k]; }
            xs[g] += x;
          }
        }
      }
    }
    (void)nz;

    // Adam
    bt1 *= 0.9; bt2 *= 0.999;
    const float c1 = float(1.0 - bt1), c2 = float(1.0 - bt2);  // bias corrections
    for (u64 k = 0; k < H; ++k) adam_upd(b1.w[k], b1.m[k], b1.v[k], gb1[k], c1, c2);
    for (u64 k = 0; k < H; ++k) adam_upd(b3.w[k], b3.m[k], b3.v[k], gb3[k], c1, c2);
    for (u64 j = 0; j < K; ++j) adam_upd(b2.w[j], b2.m[j], b2.v[j], gb2[j], c1, c2);
    for (u64 e = 0; e < H * K; ++e) adam_upd(W2.w[e], W2.m[e], W2.v[e], gW2[e], c1, c2);
    for (u64 e = 0; e < K * H; ++e) adam_upd(W3.w[e], W3.m[e], W3.v[e], gW3[e], c1, c2);
    // W1: genes absent from the batch have zero gradient but still move (Adam momentum)
    const float a1c = 1e-3f / c1, ic2 = 1.0f / c2;
#pragma omp parallel for schedule(static)
    for (u64 g = 0; g < D; ++g) {
      float* __restrict w = &W1.w[g * H];
      float* __restrict m = &W1.m[g * H];
      float* __restrict v = &W1.v[g * H];
      if (touched[g]) {
        const float* __restrict gr = &gW1[g * H];
#pragma omp simd
        for (u64 k = 0; k < H; ++k) {
          const float gg = gr[k];
          const float mm = 0.9f * m[k] + 0.1f * gg, vv = 0.999f * v[k] + 0.001f * gg * gg;
          m[k] = mm; v[k] = vv; w[k] -= a1c * mm / (std::sqrt(vv * ic2) + 1e-8f);
        }
      } else {
#pragma omp simd
        for (u64 k = 0; k < H; ++k) {
          const float mm = 0.9f * m[k], vv = 0.999f * v[k];
          m[k] = mm; v[k] = vv; w[k] -= a1c * mm / (std::sqrt(vv * ic2) + 1e-8f);
        }
      }
    }
    // W4T and b4: gradient, Adam, and the next step's G and u in one pass over the genes
    const float fB = float(B);
    float b4sq_acc = 0.f;
#pragma omp parallel
    {
      const int t = omp_get_thread_num();
      float* Gp = &Gpart[size_t(t) * H * H];
      float* up = &upart[size_t(t) * H];
      std::fill(Gp, Gp + H * H, 0.f);
      std::fill(up, up + H, 0.f);
      alignas(64) float gr[H];
      float bsq = 0.f;
#pragma omp for schedule(static)
      for (u64 g = 0; g < D; ++g) {
        float* __restrict w = &W4T.w[g * H]; float* __restrict m = &W4T.m[g * H]; float* __restrict v = &W4T.v[g * H];
        const bool q = touched[g];
        const float bg = b4.w[g];
        float wc = 0.f;
        for (u64 k = 0; k < H; ++k) wc += w[k] * c[k];
        for (u64 l = 0; l < H; ++l) gr[l] = bg * c[l];
        for (u64 k = 0; k < H; ++k) {
          const float wk = w[k];
          const float* mk = &M[k * H];
#pragma omp simd
          for (u64 l = 0; l < H; ++l) gr[l] += wk * mk[l];
        }
        float gb = wc + fB * bg;
        if (q) {
          const float* r = &gW4s[g * H];
#pragma omp simd
          for (u64 l = 0; l < H; ++l) gr[l] -= r[l];
          gb -= xs[g];
          touched[g] = 0;
        }
#pragma omp simd
        for (u64 l = 0; l < H; ++l) {
          const float gg = s * gr[l];
          const float mm = 0.9f * m[l] + 0.1f * gg, vv = 0.999f * v[l] + 0.001f * gg * gg;
          m[l] = mm; v[l] = vv; w[l] -= a1c * mm / (std::sqrt(vv * ic2) + 1e-8f);
        }
        adam_upd(b4.w[g], b4.m[g], b4.v[g], s * gb, c1, c2);
        const float nb4 = b4.w[g];
        bsq += nb4 * nb4;
        for (u64 k = 0; k < H; ++k) {
          const float wk = w[k];
          float* Gk = &Gp[k * H];
#pragma omp simd
          for (u64 l = 0; l < H; ++l) Gk[l] += wk * w[l];
          up[k] += nb4 * wk;
        }
      }
#pragma omp atomic
      b4sq_acc += bsq;
    }
    for (u64 e = 0; e < H * H; ++e) { float t = 0.f; for (int q = 0; q < T; ++q) t += Gpart[size_t(q) * H * H + e]; G[e] = t; }
    for (u64 k = 0; k < H; ++k) { float t = 0.f; for (int q = 0; q < T; ++q) t += upart[size_t(q) * H + k]; u[k] = t; }
    b4sq = b4sq_acc;
    return loss;
  }
};


}  // namespace

void hpcbench::solve(const Corpus& c, const Params& p, Result& out) {
  const u64 N = p.n_cells, BM = p.batch, H = kHidden, K = kLatent, D = p.n_genes;
  // The row index of the corpus: 8 bytes per cell.
  std::vector<u64> rp(N + 1);
  c.read_at(rp.data(), 8 * (N + 1), c.off_indptr);

  Trainer tr;
  tr.init(p);
  Batch b;
  for (u64 start = 0; start < N; start += BM) {
    const u32 n = u32(std::min(BM, N - start));
    const u32* rows = &p.order[start];
    // READ: two small reads per cell, in batch order.
    u64 tot = 0;
    for (u32 r = 0; r < n; ++r) tot += rp[rows[r] + 1] - rp[rows[r]];
    b.n = n;
    b.ptr.resize(n + 1);
    b.idx.resize(tot);
    b.val.resize(tot);
    u64 q = 0;
    for (u32 r = 0; r < n; ++r) {
      const u64 a = rp[rows[r]], m = rp[rows[r] + 1] - a;
      c.read_at(&b.idx[q], 4 * m, c.off_indices + 4 * a);
      c.read_at(&b.val[q], 4 * m, c.off_data + 4 * a);
      b.ptr[r] = u32(q);
      q += m;
    }
    b.ptr[n] = u32(q);
    // BUILD: the network's input is log1p(count).
    for (u64 e = 0; e < tot; ++e) b.val[e] = std::log1p(b.val[e]);
    // TRAIN: one Adam step.
    tr.step(b);
  }

  std::memcpy(out.W1, tr.W1.w.data(), D * H * sizeof(float));
  std::memcpy(out.b1, tr.b1.w.data(), H * sizeof(float));
  std::memcpy(out.W2, tr.W2.w.data(), H * K * sizeof(float));
  std::memcpy(out.b2, tr.b2.w.data(), K * sizeof(float));
  std::memcpy(out.W3, tr.W3.w.data(), K * H * sizeof(float));
  std::memcpy(out.b3, tr.b3.w.data(), H * sizeof(float));
  std::memcpy(out.W4T, tr.W4T.w.data(), D * H * sizeof(float));
  std::memcpy(out.b4, tr.b4.w.data(), D * sizeof(float));
}
