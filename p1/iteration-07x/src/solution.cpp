// src/solution.cpp -- RUNG 7x (ablation): rung 7 WITHOUT rung 5.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// Not a new idea -- a controlled experiment. Rungs 6 and 7 were built on top
// of rung 5's packed format, so their numbers include rung 5's decoding cost.
// This file is rung 7's solve with exactly one thing removed: the packed
// format. It reads rung 4's plain CSR (u16 gene + u16 count per non-zero,
// csr_narrow.bin) directly, as rung 4 did, and keeps everything else:
// rung 4's register panels, rung 6's prefetch policy, rung 7's gene blocks.
//
//   rung 7  vs  rung 7x   = the effect of rung 5, with everything else on top
//   rung 4  vs  rung 7x   = the effect of rungs 6 + 7 without rung 5
//
// With a plain CSR the blocking needs no decoder at all: a row's cursor is
// just the index of its next non-zero, advanced while the gene is inside the
// current block. (It is rung 7's wide-fallback path, generalised over the
// index and count types.)
//
// Correctness: as rung 7 -- same terms, same order, same fused multiply-adds.
// The result is bit-for-bit the baseline's.
//
// Knobs: P1_W_BLOCK_BYTES and P1_PREFETCH_DIST, as in rung 7.
#include "task.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#define P1_OPENMP 1
#elif !defined(__wasm__) && !defined(__EMSCRIPTEN__)
#define P1_STD_THREAD 1
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

#ifndef P1_PANEL  // override with -DP1_PANEL=... (up to 64) in CXXFLAGS to experiment
#define P1_PANEL 32
#endif

namespace {

constexpr std::uint64_t kMinWorkPerThread = std::uint64_t(1) << 19;  // multiply-adds
constexpr std::uint64_t kChunksPerThread = 32;
constexpr std::uint64_t kMaxBlocks = 64;  // each block costs a barrier

// ---- rung 4's kernel, unchanged -------------------------------------------

typedef double v4d __attribute__((vector_size(32)));  // four doubles

constexpr int kPanel = P1_PANEL;

// acc + x * w, rounded exactly once, the way the baseline's `y += x * w` is.
//
// Where the target has a hardware fused multiply-add (__FP_FAST_FMA: every
// arm64, x86 with FMA), the compiler fuses the baseline's statement into one
// instruction. It does not reliably fuse ours: GCC's tuning for recent Intel
// cores ("avoid FMA chains") splits a loop-carried accumulation into a multiply
// and an add when only one or two accumulator vectors carry it -- exactly our
// narrow panels -- which rounds twice. So the fusion is spelled out here.
// Without hardware FMA the baseline rounds twice, and so does this.
inline double madd(double x, double w, double acc) {
#if defined(__FP_FAST_FMA)
  return __builtin_fma(x, w, acc);
#else
  return x * w + acc;
#endif
}

inline v4d vmadd(v4d x, v4d w, v4d acc) {
#if defined(__FP_FAST_FMA)
  // Four scalar fused multiply-adds; every compiler tested turns them into one
  // vector FMA (vfmadd231pd on x86, two fmla on arm64).
  return v4d{__builtin_fma(x[0], w[0], acc[0]), __builtin_fma(x[1], w[1], acc[1]),
             __builtin_fma(x[2], w[2], acc[2]), __builtin_fma(x[3], w[3], acc[3])};
#else
  return x * w + acc;
#endif
}

// One row, one panel of P output columns:  yc[0:P] = sum over the row's
// non-zeros of x * Wc[g*k + 0:P], accumulated in registers. With `accumulate`
// the sums start from what yc already holds -- the partial sums left there by
// the previous gene blocks -- instead of from zero.
//
// noinline: each width is compiled as its own function. Measured necessity,
// not style: GCC 13 inlined all 32 prefetching widths into the switch below and
// the result ran k = 16 at 18.1 ms instead of 10.1 ms (k = 13: 15.1 vs 8.5) at
// every prefetch distance from 1 to 64 -- a code-generation effect, not a
// memory one. The same attribute on rungs 4 and 5 measured no difference.
template <int P, bool PF, class Idx, class Val>
__attribute__((noinline)) void row_panel(const Idx* col, const Val* val, std::uint32_t n,
                                         const double* Wc, std::uint64_t k, double* yc,
                                         std::uint32_t dist, bool accumulate) {
  constexpr int V = P / 4;  // full 4-wide vectors
  constexpr int R = P % 4;  // 1-3 leftover columns, scalar
  v4d acc[V > 0 ? V : 1];
  double tail[R > 0 ? R : 1];
  if (accumulate) {
#pragma GCC unroll 16
    for (int v = 0; v < V; ++v) std::memcpy(&acc[v], yc + 4 * v, sizeof(v4d));
    for (int r = 0; r < R; ++r) tail[r] = yc[4 * V + r];
  } else {
#pragma GCC unroll 16
    for (int v = 0; v < V; ++v) acc[v] = v4d{0.0, 0.0, 0.0, 0.0};
    for (int r = 0; r < R; ++r) tail[r] = 0.0;
  }

  for (std::uint32_t q = 0; q < n; ++q) {
    if (PF && q + dist < n) {
      // The W segment this row will need `dist` non-zeros from now: one
      // prefetch per 64 bytes (a cache line on x86, half of one on Apple).
      const char* ahead =
          reinterpret_cast<const char*>(Wc + static_cast<std::uint64_t>(col[q + dist]) * k);
#pragma GCC unroll 16
      for (int off = 0; off < P * 8; off += 64) __builtin_prefetch(ahead + off, 0, 3);
    }
    const double x = static_cast<double>(val[q]);
    const v4d xv = v4d{x, x, x, x};
    const double* w = Wc + static_cast<std::uint64_t>(col[q]) * k;
    // Fully unrolled: GCC stops at 8 vectors on its own and would otherwise
    // keep a 64-wide panel's accumulators on the stack.
#pragma GCC unroll 16
    for (int v = 0; v < V; ++v) {
      v4d wv;
      std::memcpy(&wv, w + 4 * v, sizeof(wv));  // unaligned vector load
      acc[v] = vmadd(xv, wv, acc[v]);
    }
    for (int r = 0; r < R; ++r) tail[r] = madd(x, w[4 * V + r], tail[r]);
  }

#pragma GCC unroll 16
  for (int v = 0; v < V; ++v) std::memcpy(yc + 4 * v, &acc[v], sizeof(v4d));
  for (int r = 0; r < R; ++r) yc[4 * V + r] = tail[r];
}

// The switch: a run-time width selects a compile-time kernel.
template <bool PF, class Idx, class Val>
void row_panel_any(int width, const Idx* col, const Val* val, std::uint32_t n,
                   const double* Wc, std::uint64_t k, double* yc, std::uint32_t dist,
                   bool accumulate) {
  switch (width) {
#define P1_CASE(P)                                                  \
  case P:                                                           \
    row_panel<P, PF>(col, val, n, Wc, k, yc, dist, accumulate);     \
    return;
    P1_CASE(1)
    P1_CASE(2)
    P1_CASE(3)
    P1_CASE(4)
    P1_CASE(5)
    P1_CASE(6)
    P1_CASE(7)
    P1_CASE(8)
    P1_CASE(9)
    P1_CASE(10)
    P1_CASE(11)
    P1_CASE(12)
    P1_CASE(13)
    P1_CASE(14)
    P1_CASE(15)
    P1_CASE(16)
    P1_CASE(17)
    P1_CASE(18)
    P1_CASE(19)
    P1_CASE(20)
    P1_CASE(21)
    P1_CASE(22)
    P1_CASE(23)
    P1_CASE(24)
    P1_CASE(25)
    P1_CASE(26)
    P1_CASE(27)
    P1_CASE(28)
    P1_CASE(29)
    P1_CASE(30)
    P1_CASE(31)
    P1_CASE(32)
#if P1_PANEL > 32
    P1_CASE(33)
    P1_CASE(34)
    P1_CASE(35)
    P1_CASE(36)
    P1_CASE(37)
    P1_CASE(38)
    P1_CASE(39)
    P1_CASE(40)
    P1_CASE(41)
    P1_CASE(42)
    P1_CASE(43)
    P1_CASE(44)
    P1_CASE(45)
    P1_CASE(46)
    P1_CASE(47)
    P1_CASE(48)
    P1_CASE(49)
    P1_CASE(50)
    P1_CASE(51)
    P1_CASE(52)
    P1_CASE(53)
    P1_CASE(54)
    P1_CASE(55)
    P1_CASE(56)
    P1_CASE(57)
    P1_CASE(58)
    P1_CASE(59)
    P1_CASE(60)
    P1_CASE(61)
    P1_CASE(62)
    P1_CASE(63)
    P1_CASE(64)
#endif
#undef P1_CASE
    default:
      hpcbench::fatal("panel width out of range");
  }
}

// Per-thread scratch of rung 5's decoder. Nothing is decoded here, so it is
// allocated with capacity 0 (one element); kept so run_rows is rung 7's.
struct Scratch {
  std::unique_ptr<std::uint32_t[]> genes;
  std::unique_ptr<double[]> counts;
  explicit Scratch(std::uint64_t cap)
      : genes(new std::uint32_t[cap > 0 ? cap : 1]), counts(new double[cap > 0 ? cap : 1]) {}
};

// ---- rung 7: gene blocks --------------------------------------------------

// The genes [b * genes, min((b + 1) * genes, n_genes)) form block b of W.
struct Blocks {
  std::uint32_t genes;  // genes per block (the last one may be smaller)
  std::uint32_t count;  // number of blocks; 1 = W fits, no blocking
};

// Where row i stopped at the end of the previous block: the index of its
// next non-zero. (Rung 7's cursor also carries decoder state; not needed.)
struct Cursor {
  std::uint32_t next;
};

// Plain CSR rows: u16 gene + u16 count (format 1), or the u32 + f64 fallback
// (format 2) that prepare writes when a gene index or count does not fit.
template <class Idx, class Val>
struct CsrRows {
  const std::uint32_t* row_nnz;
  const Idx* col;
  const Val* val;
  const double* W;
  double* Y;
  std::uint64_t k;
  std::uint32_t dist;
  std::uint64_t n_genes;
  Blocks blocks;
  Cursor* cursors;

  template <bool PF>
  void rows(std::uint64_t r0, std::uint64_t r1, std::uint32_t b) const {
    const std::uint64_t full = k - k % kPanel;
    const int rem = static_cast<int>(k % kPanel);
    const std::uint64_t gene_end =
        std::min<std::uint64_t>(n_genes, std::uint64_t(b + 1) * blocks.genes);
    const bool accumulate = b > 0;
    for (std::uint64_t i = r0; i < r1; ++i) {
      std::uint32_t q0, n;
      if (blocks.count == 1) {
        q0 = row_nnz[i];
        n = row_nnz[i + 1] - q0;
      } else {
        Cursor& cur = cursors[i];
        if (b == 0) cur.next = row_nnz[i];
        q0 = cur.next;
        std::uint32_t q = q0;
        while (q < row_nnz[i + 1] && col[q] < gene_end) ++q;
        n = q - q0;
        cur.next = q;
        if (n == 0 && accumulate) continue;
      }
      double* yi = Y + i * k;
      for (std::uint64_t c0 = 0; c0 < full; c0 += kPanel)
        row_panel<kPanel, PF>(col + q0, val + q0, n, W + c0, k, yi + c0, dist, accumulate);
      if (rem)
        row_panel_any<PF>(rem, col + q0, val + q0, n, W + full, k, yi + full, dist, accumulate);
    }
  }
  void operator()(std::uint64_t r0, std::uint64_t r1, std::uint32_t b, Scratch&) const {
    if (dist > 0) rows<true>(r0, r1, b);
    else rows<false>(r0, r1, b);
  }
};

// ---- rung 6: cache sizes and the prefetch policy ---------------------------

struct Caches {
  std::uint64_t l1d, l2, llc;  // bytes; llc = the largest cache shared by a cluster/socket
  std::uint64_t l2_min;        // rung 7: the smallest L2 any thread may run on
};

#if defined(__APPLE__)
std::uint64_t sysctl_u64(const char* name) {
  std::uint64_t v = 0;
  std::size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
  return len == 4 ? static_cast<std::uint32_t>(v) : v;
}
#endif

// This machine's cache sizes. On Apple silicon the performance cluster's
// values (perflevel0: 128 KB L1d, 16 MB L2 on an M4) -- the L2 is the last
// level each cluster owns. On Linux, glibc's sysconf. Defaults where unknown.
Caches query_caches() {
  Caches c{0, 0, 0, 0};
#if defined(__APPLE__)
  c.l1d = sysctl_u64("hw.perflevel0.l1dcachesize");
  c.l2 = sysctl_u64("hw.perflevel0.l2cachesize");
  if (c.l1d == 0) c.l1d = sysctl_u64("hw.l1dcachesize");
  if (c.l2 == 0) c.l2 = sysctl_u64("hw.l2cachesize");
  c.llc = c.l2;
  c.l2_min = sysctl_u64("hw.perflevel1.l2cachesize");  // efficiency cluster, if any
#elif defined(__linux__) && defined(_SC_LEVEL1_DCACHE_SIZE)
  const long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
  const long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
  c.l1d = l1 > 0 ? static_cast<std::uint64_t>(l1) : 0;
  c.l2 = l2 > 0 ? static_cast<std::uint64_t>(l2) : 0;
  c.llc = l3 > 0 ? static_cast<std::uint64_t>(l3) : c.l2;
#endif
  if (c.l1d == 0) c.l1d = 32 << 10;  // unknown: a typical L1d
  if (c.l2 == 0) c.l2 = 1 << 20;     // unknown: a typical per-core L2
  if (c.llc == 0) c.llc = 8 << 20;   // unknown: a typical shared L3
  if (c.llc < c.l2) c.llc = c.l2;
  if (c.l2_min == 0 || c.l2_min > c.l2) c.l2_min = c.l2;
  return c;
}

// Prefetch distance for this run: 0 (off) when W fits in half of L2, else
// P1_PREFETCH_DIST or 8 non-zeros ahead.
std::uint32_t prefetch_distance(std::uint64_t n_genes, std::uint64_t k, const Caches& c) {
  std::uint32_t dist = 8;
  if (const char* s = std::getenv("P1_PREFETCH_DIST")) {
    const long v = std::strtol(s, nullptr, 10);
    if (v >= 0) dist = static_cast<std::uint32_t>(v);
  }
  const std::uint64_t w_bytes = n_genes * k * sizeof(double);
  return w_bytes > c.l2 / 2 ? dist : 0;
}

// Rung 7: how to cut W into gene blocks. No blocking while W fits in the
// smallest L2; otherwise blocks of a quarter of that L2. P1_W_BLOCK_BYTES sets
// the block size and blocks whenever W is larger. Blocks are made equal in
// size, and there are never more than kMaxBlocks of them.
Blocks plan_blocks(std::uint64_t n_genes, std::uint64_t k, const Caches& c) {
  const std::uint64_t row_bytes = k * sizeof(double);
  const std::uint64_t w_bytes = n_genes * row_bytes;
  std::uint64_t budget = c.l2_min / 4, fits = c.l2_min;
  if (const char* s = std::getenv("P1_W_BLOCK_BYTES")) {
    const long long v = std::strtoll(s, nullptr, 10);
    if (v > 0) budget = fits = static_cast<std::uint64_t>(v);
  }
  if (n_genes == 0 || row_bytes == 0 || w_bytes <= fits)
    return Blocks{static_cast<std::uint32_t>(n_genes), 1};
  const std::uint64_t max_genes = std::max<std::uint64_t>(1, budget / row_bytes);
  std::uint64_t count = std::min<std::uint64_t>((n_genes + max_genes - 1) / max_genes, kMaxBlocks);
  const std::uint64_t genes = (n_genes + count - 1) / count;
  count = (n_genes + genes - 1) / genes;
  return Blocks{static_cast<std::uint32_t>(genes), static_cast<std::uint32_t>(count)};
}

// ---- threading: rung 3's, with one Scratch per thread and, new in rung 7, a
// barrier between gene blocks ----------------------------------------------

unsigned threads_granted() {
#if defined(P1_OPENMP)
  return static_cast<unsigned>(std::max(1, omp_get_max_threads()));
#elif defined(P1_STD_THREAD)
  if (const char* s = std::getenv("OMP_NUM_THREADS")) {
    const long v = std::strtol(s, nullptr, 10);
    if (v > 0) return static_cast<unsigned>(v);
  }
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0)
    return static_cast<unsigned>(CPU_COUNT(&set));
#endif
  const unsigned hc = std::thread::hardware_concurrency();
  return hc > 0 ? hc : 1;
#else
  return 1;
#endif
}

#if defined(P1_STD_THREAD)
// A reusable barrier (C++17 has none): each thread waits here until all have
// finished the current gene block. The party count starts at the number of
// threads planned and is lowered if creating one of them fails.
//
// A waiter first spins for up to 50 us: the other threads are usually only
// microseconds behind, and a sleep plus wake-up through the kernel costs tens
// of microseconds on macOS -- with 27 blocks at k = 256, that adds up. After
// 50 us it sleeps, so an oversubscribed machine does not burn cores.
class Barrier {
 public:
  explicit Barrier(unsigned parties) : parties_(parties) {}
  void set_parties(unsigned parties) {
    std::lock_guard<std::mutex> lock(m_);
    parties_ = parties;
    if (arrived_ > 0 && arrived_ >= parties_) release();
  }
  void wait() {
    std::unique_lock<std::mutex> lock(m_);
    const unsigned gen = generation_.load(std::memory_order_relaxed);
    if (++arrived_ >= parties_) {
      release();
      return;
    }
    lock.unlock();
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
    while (std::chrono::steady_clock::now() < until)
      if (generation_.load(std::memory_order_acquire) != gen) return;
    lock.lock();
    cv_.wait(lock, [&] { return generation_.load(std::memory_order_relaxed) != gen; });
  }

 private:
  void release() {  // called with m_ held
    arrived_ = 0;
    generation_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
  }
  std::mutex m_;
  std::condition_variable cv_;
  unsigned parties_, arrived_ = 0;
  std::atomic<unsigned> generation_{0};
};
#endif

template <class Rows>
void run_rows(const Rows& rows, const std::uint32_t* row_nnz, std::uint64_t n_cells,
              std::uint64_t k, std::uint64_t scratch_cap, std::uint32_t n_blocks) {
  const std::uint64_t nnz = row_nnz[n_cells];
  const std::uint64_t floor_threads = std::max<std::uint64_t>(1, nnz * k / kMinWorkPerThread);
  const unsigned nt =
      static_cast<unsigned>(std::min<std::uint64_t>(threads_granted(), floor_threads));
  if (nt <= 1) {
    Scratch s(scratch_cap);
    for (std::uint32_t b = 0; b < n_blocks; ++b) rows(0, n_cells, b, s);
    return;
  }

  std::vector<std::uint32_t> bounds;
  bounds.reserve(static_cast<std::size_t>(nt * kChunksPerThread + 2));
  bounds.push_back(0);
  const std::uint64_t target = std::max<std::uint64_t>(1, nnz / (nt * kChunksPerThread));
  std::uint64_t chunk_start = row_nnz[0];
  for (std::uint64_t i = 0; i < n_cells; ++i)
    if (row_nnz[i + 1] - chunk_start >= target) {
      bounds.push_back(static_cast<std::uint32_t>(i + 1));
      chunk_start = row_nnz[i + 1];
    }
  if (bounds.back() != n_cells) bounds.push_back(static_cast<std::uint32_t>(n_cells));
  const std::size_t n_chunks = bounds.size() - 1;

#if defined(P1_OPENMP)
  const long nc = static_cast<long>(n_chunks);
#pragma omp parallel num_threads(nt)
  {
    Scratch s(scratch_cap);  // one per thread, allocated once
    for (std::uint32_t b = 0; b < n_blocks; ++b) {
      // The implicit barrier at the end of this loop keeps all threads on the
      // same block of W: nobody starts block b + 1 before every row has done b.
#pragma omp for schedule(dynamic, 1)
      for (long c = 0; c < nc; ++c) rows(bounds[c], bounds[c + 1], b, s);
    }
  }
#elif defined(P1_STD_THREAD)
  // One chunk counter per block, so a block's chunks are handed out afresh.
  std::unique_ptr<std::atomic<std::size_t>[]> next(new std::atomic<std::size_t>[n_blocks]);
  for (std::uint32_t b = 0; b < n_blocks; ++b) next[b].store(0, std::memory_order_relaxed);
  Barrier barrier(nt);
  auto worker = [&]() {
    Scratch s(scratch_cap);  // one per thread, allocated once
    for (std::uint32_t b = 0; b < n_blocks; ++b) {
      for (;;) {
        const std::size_t c = next[b].fetch_add(1, std::memory_order_relaxed);
        if (c >= n_chunks) break;
        rows(bounds[c], bounds[c + 1], b, s);
      }
      if (b + 1 < n_blocks) barrier.wait();  // one block: never waits, as rung 6
    }
  };
  std::vector<std::thread> helpers;
  helpers.reserve(nt - 1);
  for (unsigned t = 1; t < nt; ++t) {
    try {
      helpers.emplace_back(worker);
    } catch (...) {
      break;
    }
  }
  if (helpers.size() + 1 < nt) barrier.set_parties(static_cast<unsigned>(helpers.size() + 1));
  worker();
  for (std::thread& h : helpers) h.join();
#endif
}

}  // namespace

void hpcbench::solve(const Work& w, const Params& p, Result& y) {
  const Mapped m = w.map("csr_narrow.bin");
  if (m.bytes() < 64) fatal("csr_narrow.bin is truncated; re-run prepare");
  const std::uint64_t* hdr = m.as<std::uint64_t>(0);
  const std::uint64_t n_cells = hdr[0], n_genes = hdr[1], format = hdr[3];
  const std::uint64_t off_row_ptr = hdr[4], off_col = hdr[5], off_val = hdr[6];
  if (n_cells != p.n_cells || n_genes != p.n_genes)
    fatal("csr_narrow.bin dimensions do not match the harness; re-run prepare");
  if (m.bytes() != hdr[7]) fatal("csr_narrow.bin has the wrong size; re-run prepare");
  const std::uint32_t* row_nnz = m.as<std::uint32_t>(off_row_ptr);

  const Caches caches = query_caches();
  const Blocks blocks = plan_blocks(n_genes, p.k, caches);
  // Prefetch when the working set -- now one block of W -- outgrows L2.
  const std::uint32_t dist = prefetch_distance(blocks.genes, p.k, caches);
  std::unique_ptr<Cursor[]> cursors(blocks.count > 1 ? new Cursor[n_cells] : nullptr);

  if (format == 1) {
    const CsrRows<std::uint16_t, std::uint16_t> rows{
        row_nnz, m.as<std::uint16_t>(off_col), m.as<std::uint16_t>(off_val), p.W, y.data,
        p.k, dist, n_genes, blocks, cursors.get()};
    run_rows(rows, row_nnz, n_cells, p.k, 0, blocks.count);
  } else if (format == 2) {
    const CsrRows<std::uint32_t, double> rows{
        row_nnz, m.as<std::uint32_t>(off_col), m.as<double>(off_val), p.W, y.data,
        p.k, dist, n_genes, blocks, cursors.get()};
    run_rows(rows, row_nnz, n_cells, p.k, 0, blocks.count);
  } else {
    fatal("csr_narrow.bin has an unknown format; re-run prepare");
  }
}