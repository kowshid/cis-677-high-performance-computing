// Project 3 -- iteration-02: a cheaper loader and a parallel trainer init.
// YOURS to change: everything in this file is timed (plus build.sh).
//
// Where iteration-01 left off (c008 Gold 6230R, same session: baseline 9,165 ms,
// iteration-01 3,672 ms, 2.50x). The trainer waits less than 1 ms in total for
// data: the loader is hidden. The bottleneck moved to CPU 0. OMP_PROC_BIND pins
// the main thread (OpenMP's master) to CPU 0, every loader thread inherits it,
// so CPU 0 runs the master's share of training plus all of the loader's CPU
// work, and the other three cores wait for it at barriers. With 8 loader
// threads, training per batch rose from 27.1 ms (1 loader thread) to 32.1 ms,
// and ~450 ms of the main thread's time fell outside every timed stage.
// Init: ~5 ms of it is the Gram matrix; most is page faults on ~40 MB of fresh
// memory, all taken serially on CPU 0.
//
// This iteration (loader threads still inherit CPU 0; placement comes later):
//   * log1p by table lookup. Every
//     count in the corpus is an integer <= 12,420 and 3/4 of them are 1, so a
//     16,384-entry table (64 KB) covers them. Entries are made by the same
//     std::log1p calls, so values are bit-identical; anything else (zero, a
//     fraction, a larger count) falls back to std::log1p.
//   * Trainer init on all cores, same bits. The big arrays are allocated
//     without zero-filling and then filled in parallel, so each thread takes
//     the page faults for its own part. The
//     Gram matrix G = W4 W4^T and u = W4 b4 are computed with each thread owning
//     whole rows and summing over genes
//     in the original order, so every element gets the same additions in the
//     same order as one thread would give it.
//   * More timers, to test the CPU-0 explanation: the trainer's release of
//     each batch, the teardown at the end of solve, the CPU time and context
//     switches of the main thread and of the loader threads.
// Batches and parameters are bit-identical to the baseline's. Trainer::step is
// the baseline's, unchanged.
//
// Knobs, read once from the environment (localbench: --env KEY=VALUE):
//   P3_IO_THREADS   loader threads                 default 8
//   P3_SLOTS        batches in memory at once      default 4
//   P3_CHUNK        cells per work item            default 16
//   P3_LUT          1 table lookup, 0 std::log1p   default 1
//   P3_PROFILE      1 prints "p3: <name> <value>" timers on stderr
//
// Profile names (p3prof.py relies on them):
//   solve.<stage>_ms   main-thread timeline; stages add up to the solve time
//   train.<phase>_ms   inside Trainer::step; phases add up to solve.train_ms
//   loader.*           summed over loader threads, which run concurrently
//   anything else      counters and distributions (io.*, batch.*, cpu.*, mem.*, aff.*)
#include "task.hpp"

#include <omp.h>
#include <sched.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// ---- profiling ------------------------------------------------------------------
namespace prof {
using Clock = std::chrono::steady_clock;
bool on = false;

inline double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Trainer::step phases, in the order step() runs them.
enum Phase { kForward, kBackL3, kBackL2, kGradW2, kBiasGrads, kGramM, kSparseGrads,
             kAdamSmall, kAdamW1, kW4Pass, kGReduce, kNumPhases };
const char* const kPhaseName[kNumPhases] = {"forward", "back_l3", "back_l2", "grad_w2",
                                            "bias_grads", "gram_m", "sparse_grads", "adam_small",
                                            "adam_w1", "w4_pass", "g_reduce"};
double phase_ms[kNumPhases] = {};

// A lap timer on the calling thread: mark(p) adds the time since the last mark to p.
struct Lap {
  Clock::time_point t;
  Lap() { if (on) t = Clock::now(); }
  void mark(Phase p) {
    if (!on) return;
    const Clock::time_point n = Clock::now();
    phase_ms[p] += ms(t, n);
    t = n;
  }
};

void emit(const std::string& name, double v) { std::fprintf(stderr, "p3: %s %.10g\n", name.c_str(), v); }
void emit(const std::string& name, const std::string& v) { std::fprintf(stderr, "p3: %s %s\n", name.c_str(), v.c_str()); }

// q-quantile of v (nearest rank); v is taken by value and sorted.
double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t i = std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5));
  return v[i];
}
double mean(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / double(v.size());
}

// CPU time and context switches of the calling thread so far.
struct ThreadUse {
  double cpu_ms = 0.0;
  double voluntary = 0.0, involuntary = 0.0;
};
ThreadUse thread_use() {
  struct rusage ru {};
  getrusage(RUSAGE_THREAD, &ru);
  ThreadUse t;
  t.cpu_ms = (double(ru.ru_utime.tv_sec) + double(ru.ru_stime.tv_sec)) * 1e3 +
             (double(ru.ru_utime.tv_usec) + double(ru.ru_stime.tv_usec)) * 1e-3;
  t.voluntary = double(ru.ru_nvcsw);
  t.involuntary = double(ru.ru_nivcsw);
  return t;
}

double peak_rss_mb() {
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  return double(ru.ru_maxrss) / 1024.0;
}

// CPUs the calling thread may run on, as a compact list such as "4-7" or "4,6".
std::string cpus() {
  cpu_set_t s;
  CPU_ZERO(&s);
  if (sched_getaffinity(0, sizeof(s), &s) != 0) return "?";
  std::string out;
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (!CPU_ISSET(c, &s)) continue;
    int e = c;
    while (e + 1 < CPU_SETSIZE && CPU_ISSET(e + 1, &s)) ++e;
    if (!out.empty()) out += ',';
    out += std::to_string(c);
    if (e > c) out += "-" + std::to_string(e);
    c = e;
  }
  return out;
}

// Where threads may run: the main thread, each OpenMP thread (allowed set @ current
// cpu), the main thread again after OpenMP has started, and a std::thread created
// after that -- the kind of thread a loader would use.
void affinity_probe() {
  emit("aff.main_start", cpus());
  const int T = omp_get_max_threads();
  std::vector<std::string> mask(static_cast<size_t>(T));
  std::vector<int> now(static_cast<size_t>(T), -1);
#pragma omp parallel
  {
    const int t = omp_get_thread_num();
    mask[size_t(t)] = cpus();
    now[size_t(t)] = sched_getcpu();
  }
  for (int t = 0; t < T; ++t)
    emit("aff.omp" + std::to_string(t), mask[size_t(t)] + "@" + std::to_string(now[size_t(t)]));
  emit("aff.main_after_omp", cpus());
  std::string th;
  std::thread([&th] { th = cpus() + "@" + std::to_string(sched_getcpu()); }).join();
  emit("aff.std_thread", th);
}
}  // namespace prof

// One batch as the trainer wants it: local CSR, values already log1p(count).
struct Batch {
  u32 n = 0;
  std::vector<u32> ptr, idx;
  std::vector<float> val;
};

// A vector whose resize() leaves new floats uninitialized (no zero-fill), so the
// first write -- and with it the page fault -- can happen on any thread.
template <class T>
struct NoInitAlloc : std::allocator<T> {
  template <class U>
  struct rebind { using other = NoInitAlloc<U>; };
  NoInitAlloc() = default;
  template <class U>
  NoInitAlloc(const NoInitAlloc<U>&) noexcept {}
  template <class U>
  void construct(U* q) noexcept { ::new (static_cast<void*>(q)) U; }
  template <class U, class... A>
  void construct(U* q, A&&... a) { ::new (static_cast<void*>(q)) U(std::forward<A>(a)...); }
};
using fvec = std::vector<float, NoInitAlloc<float>>;

// Run f(i) for i in [0, n) on all threads, in chunks of 64 Ki floats (256 KB):
// each thread first-touches the pages it writes. Small n runs serially.
template <class F>
void par_for_fill(std::uint64_t n, F f) {
  constexpr std::uint64_t kChunk = std::uint64_t(1) << 16;
  if (n <= kChunk) {
    for (std::uint64_t i = 0; i < n; ++i) f(i);
    return;
  }
  const std::int64_t chunks = std::int64_t((n + kChunk - 1) / kChunk);
#pragma omp parallel for schedule(dynamic, 1)
  for (std::int64_t ch = 0; ch < chunks; ++ch) {
    const std::uint64_t a = std::uint64_t(ch) * kChunk, e = std::min(n, a + kChunk);
    for (std::uint64_t i = a; i < e; ++i) f(i);
  }
}

struct Adam1 {
  fvec w, m, v;
  void init(const float* src, std::uint64_t n) {
    w.resize(n); m.resize(n); v.resize(n);
    float* pw = w.data(); float* pm = m.data(); float* pv = v.data();
    par_for_fill(n, [=](std::uint64_t i) { pw[i] = src[i]; pm[i] = 0.f; pv[i] = 0.f; });
  }
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
  fvec gW1, gW4s;                     // dense D x H gradient rows, written only for touched genes
  std::vector<float> xs;
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
    // The big arrays are filled in parallel: every thread takes its own page faults.
    W1.init(p.W1, D * H);
    W2.init(p.W2, H * K);
    W3.init(p.W3, K * H);
    W4T.init(p.W4T, D * H);
    b1.init(p.b1, H); b2.init(p.b2, K);
    b3.init(p.b3, H); b4.init(p.b4, D);
    A1.resize(BM * H); H1.resize(BM * H); Z.resize(BM * K); A3.resize(BM * H); H3.resize(BM * H);
    P.resize(BM * H); dA3.resize(BM * H); dZ.resize(BM * K); dH1.resize(BM * H);
    M.resize(H * H); c.resize(H); gW2.resize(H * K); gb2.resize(K); gW3.resize(K * H); gb3.resize(H); gb1.resize(H);
    gW1.resize(D * H); gW4s.resize(D * H);
    {
      float* g1 = gW1.data(); float* g4 = gW4s.data();
      par_for_fill(D * H, [=](u64 i) { g1[i] = 0.f; g4[i] = 0.f; });
    }
    xs.assign(D, 0.f); touched.assign(D, 0);
    G.assign(H * H, 0.f); u.assign(H, 0.f);
    Gpart.resize(size_t(T) * H * H); upart.resize(size_t(T) * H);
    // G = W4 W4^T and u = W4 b4. A thread owns whole rows k (and u[k]) and adds
    // the genes in order g = 0, 1, ..., D-1, starting from zero: each element
    // sees the same additions in the same order as in the one-thread loop, so
    // the result has the same bits.
    {
      const float* w4 = W4T.w.data();
      const float* bb = b4.w.data();
      float* Gp = G.data();
      float* up = u.data();
      const u64 nD = D;
#pragma omp parallel for schedule(dynamic, 1)
      for (u64 k = 0; k < H; ++k) {
        // local accumulators: rows of G share cache lines, and writing them on
        // every gene from different threads would be false sharing
        alignas(64) float gk[H] = {};
        float uk = 0.f;
        for (u64 g = 0; g < nD; ++g) {
          const float* w = &w4[g * H];
          uk += bb[g] * w[k];
          for (u64 l = 0; l < H; ++l) gk[l] += w[k] * w[l];
        }
        for (u64 l = 0; l < H; ++l) Gp[k * H + l] = gk[l];
        up[k] = uk;
      }
    }
    b4sq = 0.f;
    for (u64 g = 0; g < D; ++g) b4sq += b4.w[g] * b4.w[g];
  }

  // one Adam step on one batch; returns the batch loss (mean squared error) before the step
  double step(const Batch& b) {
    const u64 B = b.n;
    const float s = 2.0f / float(double(B) * double(D));
    const u32* bp = b.ptr.data();
    const u32* bi = b.idx.data();
    const float* bv = b.val.data();
    double loss = 0.0;
    prof::Lap lap;

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
    lap.mark(prof::kForward);

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
    lap.mark(prof::kBackL3);
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
    lap.mark(prof::kBackL2);
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
    lap.mark(prof::kGradW2);
    for (u64 k = 0; k < H; ++k) { gb1[k] = 0.f; gb3[k] = 0.f; c[k] = 0.f; }
    for (u64 j = 0; j < K; ++j) gb2[j] = 0.f;
    for (u64 i = 0; i < B; ++i) {
      for (u64 k = 0; k < H; ++k) { gb1[k] += dH1[i * H + k]; gb3[k] += dA3[i * H + k]; c[k] += H3[i * H + k]; }
      for (u64 j = 0; j < K; ++j) gb2[j] += dZ[i * K + j];
    }
    lap.mark(prof::kBiasGrads);
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
    lap.mark(prof::kGramM);
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
    lap.mark(prof::kSparseGrads);

    // Adam
    bt1 *= 0.9; bt2 *= 0.999;
    const float c1 = float(1.0 - bt1), c2 = float(1.0 - bt2);  // bias corrections
    for (u64 k = 0; k < H; ++k) adam_upd(b1.w[k], b1.m[k], b1.v[k], gb1[k], c1, c2);
    for (u64 k = 0; k < H; ++k) adam_upd(b3.w[k], b3.m[k], b3.v[k], gb3[k], c1, c2);
    for (u64 j = 0; j < K; ++j) adam_upd(b2.w[j], b2.m[j], b2.v[j], gb2[j], c1, c2);
    for (u64 e = 0; e < H * K; ++e) adam_upd(W2.w[e], W2.m[e], W2.v[e], gW2[e], c1, c2);
    for (u64 e = 0; e < K * H; ++e) adam_upd(W3.w[e], W3.m[e], W3.v[e], gW3[e], c1, c2);
    lap.mark(prof::kAdamSmall);
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
    lap.mark(prof::kAdamW1);
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
    lap.mark(prof::kW4Pass);
    for (u64 e = 0; e < H * H; ++e) { float t = 0.f; for (int q = 0; q < T; ++q) t += Gpart[size_t(q) * H * H + e]; G[e] = t; }
    for (u64 k = 0; k < H; ++k) { float t = 0.f; for (int q = 0; q < T; ++q) t += upart[size_t(q) * H + k]; u[k] = t; }
    b4sq = b4sq_acc;
    lap.mark(prof::kGReduce);
    return loss;
  }
};



// ---- the loader ---------------------------------------------------------------------

int env_int(const char* name, int dflt, int lo, int hi) {
  const char* e = std::getenv(name);
  if (!e || !*e) return dflt;
  char* end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (end == e || *end != '\0') return dflt;
  return int(std::min<long>(hi, std::max<long>(lo, v)));
}

// log1p by table: the counts are integers, almost all small. kLut entries cover every count in the class corpus
// (max 12,420); anything else falls back to std::log1p.
constexpr u32 kLut = 16384;

// One loader thread's own counters (one cache line each: no false sharing).
struct alignas(64) LoaderStats {
  double read_ms = 0.0, build_ms = 0.0, slot_wait_ms = 0.0, last_done_ms = 0.0;
  u64 chunks = 0, bytes = 0, lut_miss = 0;
  prof::ThreadUse use;
  std::vector<double> lat_us;
  std::string cpus;
};

struct Loader {
  const hpcbench::Corpus* c = nullptr;
  const u64* rp = nullptr;      // corpus indptr, n_cells + 1
  const u32* order = nullptr;   // batch order from the host
  u64 N = 0, BM = 0, nb = 0;    // cells, batch size, batches
  u32 R = 4, C = 16;            // slots, cells per work item
  int T = 8;                    // loader threads
  bool use_lut = true;
  std::vector<float> lut;       // lut[k] = log1p(k), made before the threads start
  prof::Clock::time_point t_begin;

  // The plan. Batch k's local row pointers are plan_ptr[plan_off[k] .. + n(k)];
  // its work items are chunks chunk_first[k] .. chunk_first[k + 1].
  std::vector<u64> plan_off;
  std::vector<u32> plan_ptr, chunk_first, chunk_batch;
  u32 max_nnz = 0;

  std::vector<Batch> slot;                      // the ring: batch k lives in slot k % R
  std::unique_ptr<std::atomic<u32>[]> remaining; // work items left, per batch
  std::atomic<u64> next_chunk{0};               // the shared work counter
  std::atomic<u64> consumed{0};                 // batches the trainer is done with
  std::mutex mu;
  std::condition_variable cv_ready, cv_free;
  std::vector<std::thread> threads;
  std::vector<LoaderStats> stats;

  u32 n_of(u64 k) const { return u32(std::min(BM, N - k * BM)); }
  const u32* ptr_of(u64 k) const { return &plan_ptr[plan_off[k]]; }

  void plan() {
    nb = (N + BM - 1) / BM;
    plan_off.resize(nb + 1);
    chunk_first.resize(nb + 1);
    u64 off = 0, chunks = 0;
    for (u64 k = 0; k < nb; ++k) {
      plan_off[k] = off;
      off += n_of(k) + 1;
      chunk_first[k] = u32(chunks);
      chunks += (n_of(k) + C - 1) / C;
    }
    plan_off[nb] = off;
    chunk_first[nb] = u32(chunks);
    plan_ptr.resize(off);
    chunk_batch.resize(chunks);
    for (u64 k = 0; k < nb; ++k) {
      u32* pp = &plan_ptr[plan_off[k]];
      const u32* rows = order + k * BM;
      const u32 n = n_of(k);
      u64 q = 0;
      for (u32 r = 0; r < n; ++r) {
        pp[r] = u32(q);
        q += rp[rows[r] + 1] - rp[rows[r]];
      }
      pp[n] = u32(q);
      max_nnz = std::max(max_nnz, u32(q));
      for (u32 g = chunk_first[k]; g < chunk_first[k + 1]; ++g) chunk_batch[g] = u32(k);
    }
  }

  // Allocate every buffer once, then start the threads. Slots are capped so the
  // ring fits the memory rule: a quarter of the corpus, less ~80 MB for the
  // trainer's state and the host's arrays (the baseline peaks at 72 MB).
  void start() {
    const u64 per_slot = 8ull * max_nnz + 4ull * (BM + 1);
    const u64 budget = c->bytes / 4, fixed = 80ull << 20;
    const u64 fit = budget > fixed + per_slot ? (budget - fixed) / per_slot : 1;
    R = u32(std::max<u64>(1, std::min<u64>({u64(R), fit, nb})));
    slot.resize(R);
    for (Batch& b : slot) {
      b.ptr.reserve(BM + 1);
      b.idx.resize(max_nnz);
      b.val.resize(max_nnz);
    }
    lut.resize(kLut);
    for (u32 k = 0; k < kLut; ++k) lut[k] = std::log1p(float(k));
    remaining.reset(new std::atomic<u32>[nb]);
    for (u64 k = 0; k < nb; ++k) remaining[k].store(chunk_first[k + 1] - chunk_first[k], std::memory_order_relaxed);
    stats.resize(size_t(T));
    threads.reserve(size_t(T));
    for (int w = 0; w < T; ++w) threads.emplace_back([this, w] { work(w); });
  }

  void read_at(LoaderStats& st, void* dst, u64 n, u64 off) {
    if (!prof::on) { c->read_at(dst, n, off); return; }
    const prof::Clock::time_point a = prof::Clock::now();
    c->read_at(dst, n, off);
    st.lat_us.push_back(prof::ms(a, prof::Clock::now()) * 1000.0);
    st.bytes += n;
  }

  void work(int w) {
    LoaderStats& st = stats[size_t(w)];
    if (prof::on) st.cpus = prof::cpus() + "@" + std::to_string(sched_getcpu());
    const u64 total = chunk_batch.size();
    for (;;) {
      const u64 g = next_chunk.fetch_add(1, std::memory_order_relaxed);
      if (g >= total) break;
      const u64 k = chunk_batch[g];
      // Bounded queue: slot k % R is free once the trainer is done with batch k - R.
      if (k >= consumed.load(std::memory_order_acquire) + R) {
        const prof::Clock::time_point a = prof::on ? prof::Clock::now() : prof::Clock::time_point{};
        std::unique_lock<std::mutex> lk(mu);
        cv_free.wait(lk, [&] { return k < consumed.load(std::memory_order_acquire) + R; });
        if (prof::on) st.slot_wait_ms += prof::ms(a, prof::Clock::now());
      }
      Batch& b = slot[k % R];
      const u32* pp = ptr_of(k);
      const u32* rows = order + k * BM;
      const u32 r0 = (u32(g) - chunk_first[k]) * C, r1 = std::min(n_of(k), r0 + C);
      // READ: straight into this chunk's part of the slot.
      const prof::Clock::time_point t0 = prof::on ? prof::Clock::now() : prof::Clock::time_point{};
      for (u32 r = r0; r < r1; ++r) {
        const u64 a = rp[rows[r]], m = rp[rows[r] + 1] - a;
        read_at(st, &b.idx[pp[r]], 4 * m, c->off_indices + 4 * a);
        read_at(st, &b.val[pp[r]], 4 * m, c->off_data + 4 * a);
      }
      const prof::Clock::time_point t1 = prof::on ? prof::Clock::now() : prof::Clock::time_point{};
      // BUILD: the network's input is log1p(count). A whole count below kLut is
      // looked up; the table holds the same std::log1p results, bit for bit.
      float* v = b.val.data();
      const u32 e0 = pp[r0], e1 = pp[r1];
      if (use_lut) {
        const float* t = lut.data();
        u64 miss = 0;
        for (u32 e = e0; e < e1; ++e) {
          const float x = v[e];
          if (x > 0.f && x < float(kLut)) {
            const u32 k = u32(x);
            if (float(k) == x) { v[e] = t[k]; continue; }
          }
          v[e] = std::log1p(x);
          ++miss;
        }
        st.lut_miss += miss;
      } else {
        for (u32 e = e0; e < e1; ++e) v[e] = std::log1p(v[e]);
      }
      if (prof::on) {
        const prof::Clock::time_point t2 = prof::Clock::now();
        st.read_ms += prof::ms(t0, t1);
        st.build_ms += prof::ms(t1, t2);
        st.last_done_ms = prof::ms(t_begin, t2);
        ++st.chunks;
      }
      // The last work item of a batch tells the trainer.
      if (remaining[k].fetch_sub(1, std::memory_order_acq_rel) == 1) {
        { std::lock_guard<std::mutex> lk(mu); }
        cv_ready.notify_all();
      }
    }
    if (prof::on) st.use = prof::thread_use();
  }

  // Trainer side: wait until batch k is complete, then hand out its slot.
  Batch& acquire(u64 k) {
    if (remaining[k].load(std::memory_order_acquire) != 0) {
      std::unique_lock<std::mutex> lk(mu);
      cv_ready.wait(lk, [&] { return remaining[k].load(std::memory_order_acquire) == 0; });
    }
    Batch& b = slot[k % R];
    const u32 n = n_of(k);
    const u32* pp = ptr_of(k);
    b.n = n;
    b.ptr.assign(pp, pp + n + 1);
    return b;
  }

  // Trainer side: done with batch k; its slot may be refilled.
  void release(u64 k) {
    {
      std::lock_guard<std::mutex> lk(mu);
      consumed.store(k + 1, std::memory_order_release);
    }
    cv_free.notify_all();
  }

  void join() {
    for (std::thread& t : threads) t.join();
  }
};

}  // namespace

void hpcbench::solve(const Corpus& c, const Params& p, Result& out) {
  using prof::Clock;
  const Clock::time_point t_begin = Clock::now();
  {
    const char* e = std::getenv("P3_PROFILE");
    prof::on = e && *e && std::strcmp(e, "0") != 0;
  }
  const prof::ThreadUse main0 = prof::on ? prof::thread_use() : prof::ThreadUse{};
  if (prof::on) prof::affinity_probe();
  const u64 N = p.n_cells, BM = p.batch, H = kHidden, K = kLatent, D = p.n_genes;

  // The row index of the corpus: 8 bytes per cell.
  Clock::time_point t = Clock::now();
  const double probe_ms = prof::ms(t_begin, t);
  std::vector<u64> rp(N + 1);
  c.read_at(rp.data(), 8 * (N + 1), c.off_indptr);
  Clock::time_point t2 = Clock::now();
  const double indptr_ms = prof::ms(t, t2);

  // Plan every batch, allocate the slots, start reading.
  std::unique_ptr<Loader> Lp(new Loader);
  Loader& L = *Lp;
  L.c = &c; L.rp = rp.data(); L.order = p.order; L.N = N; L.BM = BM;
  L.T = env_int("P3_IO_THREADS", 8, 1, 256);
  L.R = u32(env_int("P3_SLOTS", 4, 1, 64));
  L.C = u32(env_int("P3_CHUNK", 16, 1, int(BM)));
  L.use_lut = env_int("P3_LUT", 1, 0, 1) != 0;
  L.t_begin = t_begin;
  L.plan();
  L.start();
  t = Clock::now();
  const double setup_ms = prof::ms(t2, t);

  // The trainer's init overlaps the first reads.
  std::unique_ptr<Trainer> trp(new Trainer);
  Trainer& tr = *trp;
  tr.init(p);
  t2 = Clock::now();
  const double init_ms = prof::ms(t, t2);
  const double mem_after_init = prof::on ? prof::peak_rss_mb() : 0.0;

  double wait_ms = 0.0, train_ms = 0.0, release_ms = 0.0;
  std::vector<double> b_wait, b_train, b_release, b_nnz;
  for (u64 k = 0; k < L.nb; ++k) {
    const Clock::time_point w0 = Clock::now();
    Batch& b = L.acquire(k);
    const Clock::time_point w1 = Clock::now();
    tr.step(b);
    const Clock::time_point w2 = Clock::now();
    L.release(k);
    const Clock::time_point w3 = Clock::now();
    wait_ms += prof::ms(w0, w1);
    train_ms += prof::ms(w1, w2);
    release_ms += prof::ms(w2, w3);
    if (prof::on) {
      b_wait.push_back(prof::ms(w0, w1));
      b_train.push_back(prof::ms(w1, w2));
      b_release.push_back(prof::ms(w2, w3));
      b_nnz.push_back(double(b.ptr[b.n]));
    }
  }
  t = Clock::now();
  L.join();
  t2 = Clock::now();
  const double join_ms = prof::ms(t, t2);

  t = Clock::now();
  std::memcpy(out.W1, tr.W1.w.data(), D * H * sizeof(float));
  std::memcpy(out.b1, tr.b1.w.data(), H * sizeof(float));
  std::memcpy(out.W2, tr.W2.w.data(), H * K * sizeof(float));
  std::memcpy(out.b2, tr.b2.w.data(), K * sizeof(float));
  std::memcpy(out.W3, tr.W3.w.data(), K * H * sizeof(float));
  std::memcpy(out.b3, tr.b3.w.data(), H * sizeof(float));
  std::memcpy(out.W4T, tr.W4T.w.data(), D * H * sizeof(float));
  std::memcpy(out.b4, tr.b4.w.data(), D * sizeof(float));
  t2 = Clock::now();
  const double copyout_ms = prof::ms(t, t2);

  // The loader's numbers, taken before its memory is freed.
  double l_read = 0.0, l_build = 0.0, l_wait = 0.0, l_done = 0.0, l_chunks = 0.0;
  double l_cpu = 0.0, l_vcs = 0.0, l_ics = 0.0;
  u64 bytes = 0, lut_miss = 0;
  std::vector<double> lat;
  std::string l_cpus = "?";
  const int l_threads = L.T;
  const u32 l_slots = L.R, l_chunk = L.C;
  const bool l_lut = L.use_lut;
  const u32 l_max_nnz = L.max_nnz;
  const u64 nb = L.nb;
  if (prof::on) {
    for (const LoaderStats& st : L.stats) {
      l_read += st.read_ms; l_build += st.build_ms; l_wait += st.slot_wait_ms;
      l_done = std::max(l_done, st.last_done_ms); l_chunks += double(st.chunks);
      l_cpu += st.use.cpu_ms; l_vcs += st.use.voluntary; l_ics += st.use.involuntary;
      bytes += st.bytes; lut_miss += st.lut_miss;
      lat.insert(lat.end(), st.lat_us.begin(), st.lat_us.end());
    }
    if (!L.stats.empty()) l_cpus = L.stats[0].cpus;
  }

  // TEARDOWN: freeing the trainer's and the loader's memory is timed too.
  t = Clock::now();
  trp.reset();
  Lp.reset();
  t2 = Clock::now();
  const double teardown_ms = prof::ms(t, t2);

  if (!prof::on) return;
  const prof::ThreadUse main1 = prof::thread_use();
  const double total_ms = prof::ms(t_begin, t2);
  using prof::emit;
  emit("solve.probe_ms", probe_ms);
  emit("solve.indptr_ms", indptr_ms);
  emit("solve.setup_ms", setup_ms);
  emit("solve.init_ms", init_ms);
  emit("solve.wait_ms", wait_ms);
  emit("solve.train_ms", train_ms);
  emit("solve.release_ms", release_ms);
  emit("solve.join_ms", join_ms);
  emit("solve.copyout_ms", copyout_ms);
  emit("solve.teardown_ms", teardown_ms);
  emit("total.solve_ms", total_ms);
  for (int i = 0; i < prof::kNumPhases; ++i)
    emit(std::string("train.") + prof::kPhaseName[i] + "_ms", prof::phase_ms[i]);

  emit("batch.count", double(b_wait.size()));
  emit("batch.nnz_mean", prof::mean(b_nnz));
  emit("batch.nnz_max", double(l_max_nnz));
  emit("batch.first_wait_ms", b_wait.empty() ? 0.0 : b_wait[0]);
  emit("batch.wait_ms_p50", prof::quantile(b_wait, 0.5));
  emit("batch.wait_ms_max", prof::quantile(b_wait, 1.0));
  double waited = 0.0;
  for (double x : b_wait) waited += x > 0.05 ? 1.0 : 0.0;
  emit("batch.waited_count", waited);
  emit("batch.train_ms_p50", prof::quantile(b_train, 0.5));
  emit("batch.train_ms_max", prof::quantile(b_train, 1.0));
  emit("batch.release_ms_p50", prof::quantile(b_release, 0.5));
  emit("batch.release_ms_p90", prof::quantile(b_release, 0.9));
  emit("batch.release_ms_max", prof::quantile(b_release, 1.0));

  // The main thread runs OpenMP's master: time it spent off its CPU, other than
  // waiting for a batch, is time it was preempted.
  const double main_cpu = main1.cpu_ms - main0.cpu_ms;
  emit("cpu.main_thread_cpu_ms", main_cpu);
  emit("cpu.main_thread_off_cpu_ms", total_ms - main_cpu);
  emit("cpu.main_thread_preempted", main1.involuntary - main0.involuntary);
  emit("cpu.main_thread_blocked", main1.voluntary - main0.voluntary);
  emit("cpu.loader_cpu_ms_sum", l_cpu);
  emit("cpu.loader_preempted_sum", l_ics);
  emit("cpu.loader_blocked_sum", l_vcs);

  emit("loader.threads", double(l_threads));
  emit("loader.slots", double(l_slots));
  emit("loader.chunk_cells", double(l_chunk));
  emit("loader.lut", l_lut ? 1.0 : 0.0);
  emit("loader.lut_misses", double(lut_miss));
  emit("loader.cpus_thread0", l_cpus);
  emit("loader.work_items", l_chunks);
  emit("loader.read_ms_sum", l_read);
  emit("loader.build_ms_sum", l_build);
  emit("loader.slot_wait_ms_sum", l_wait);
  emit("loader.all_read_by_ms", l_done);
  emit("io.reads", double(lat.size()) + 1.0);
  emit("io.mb", double(bytes + 8 * (N + 1)) / 1e6);
  emit("io.read_us_mean", prof::mean(lat));
  emit("io.read_us_p50", prof::quantile(lat, 0.50));
  emit("io.read_us_p90", prof::quantile(lat, 0.90));
  emit("io.read_us_p99", prof::quantile(lat, 0.99));
  emit("io.read_us_max", prof::quantile(lat, 1.0));
  emit("mem.after_init_mb", mem_after_init);
  emit("mem.end_of_solve_mb", prof::peak_rss_mb());
  emit("omp.threads", double(omp_get_max_threads()));
  (void)nb;
}