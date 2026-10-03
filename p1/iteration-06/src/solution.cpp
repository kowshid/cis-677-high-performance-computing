// src/solution.cpp -- RUNG 6: talk to the caches -- size them, prefetch into them.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// The ONE change from rung 5: explicit cache hints in the kernel.
//
// A CPU does not let a program place data in L1 or L2 -- unlike a GPU's shared
// memory, the caches are managed by hardware. What a program CAN do is
//   (a) ask how big they are, and
//   (b) tell the hardware what it is about to need.
//
// (a) query_caches() asks the OS for this machine's L1d / L2 / last-level
//     sizes: sysctl on macOS (the performance-core values: 128 KB L1d and a
//     16 MB L2 shared by the 4 P-cores on an M4), sysconf on Linux, with
//     conservative defaults elsewhere. Nothing is hard-coded to one machine.
//
// (b) Each non-zero jumps to an unpredictable row of W -- the gene order is
//     sparse, so the hardware prefetcher cannot guess the next address. But
//     rung 5 decodes a whole row before computing it, so the kernel DOES know:
//     the gene D non-zeros ahead is sitting in the buffer. The kernel issues a
//     prefetch for that gene's W segment (one per 64 bytes of the panel), so
//     it is on its way while the current non-zero is being multiplied.
//
// Prefetching only pays when W does not fit in cache; when it does, it is
// pure overhead (extra instructions, extra address arithmetic). So it is
// switched on only when W is larger than half of L2 -- Lecture 3's rule of
// thumb for the usable share of a cache. On an M4 that is k >= 128; on a
// 1-2 MB-L2 x86 core it is k >= 16.
//
// Prefetch distance: 8 non-zeros ahead. Measured on a 1-core Xeon (W in L3),
// ratio vs no prefetch at k = 128 / 256:  distance 8: 1.14x / 1.13x,
// 16: 1.08x / 1.03x, 32: 0.89x / 0.91x. Too far ahead is worse than too near:
// lines fetched too early are evicted before use. A machine whose misses go
// to DRAM (longer latency) may prefer a larger distance -- measure it.
//
// Knob for experiments (read once per solve): P1_PREFETCH_DIST=<non-zeros
// ahead>, default 8; 0 turns prefetching off.
//
// Correctness: a prefetch changes when data arrives, never what is computed.
// The result is bit-for-bit rung 5's, i.e. the baseline's.
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
// non-zeros of x * Wc[g*k + 0:P], accumulated in registers.
//
// noinline: each width is compiled as its own function. Measured necessity,
// not style: GCC 13 inlined all 32 prefetching widths into the switch below and
// the result ran k = 16 at 18.1 ms instead of 10.1 ms (k = 13: 15.1 vs 8.5) at
// every prefetch distance from 1 to 64 -- a code-generation effect, not a
// memory one. The same attribute on rungs 4 and 5 measured no difference.
template <int P, bool PF, class Idx, class Val>
__attribute__((noinline)) void row_panel(const Idx* col, const Val* val, std::uint32_t n,
                                         const double* Wc, std::uint64_t k, double* yc,
                                         std::uint32_t dist) {
  constexpr int V = P / 4;  // full 4-wide vectors
  constexpr int R = P % 4;  // 1-3 leftover columns, scalar
  v4d acc[V > 0 ? V : 1];
  double tail[R > 0 ? R : 1];
#pragma GCC unroll 16
  for (int v = 0; v < V; ++v) acc[v] = v4d{0.0, 0.0, 0.0, 0.0};
  for (int r = 0; r < R; ++r) tail[r] = 0.0;

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
                   const double* Wc, std::uint64_t k, double* yc, std::uint32_t dist) {
  switch (width) {
#define P1_CASE(P)                                      \
  case P:                                               \
    row_panel<P, PF>(col, val, n, Wc, k, yc, dist);     \
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

// ---- rung 5: the packed representation -------------------------------------

// One row's worth of decoded non-zeros, owned by one thread.
struct Scratch {
  std::unique_ptr<std::uint32_t[]> genes;
  std::unique_ptr<double[]> counts;
  explicit Scratch(std::uint64_t cap)
      : genes(new std::uint32_t[cap > 0 ? cap : 1]), counts(new double[cap > 0 ? cap : 1]) {}
};

struct PackedRows {
  const std::uint32_t* code_ptr;  // byte offset of each row in the code stream
  const std::uint32_t* val_ptr;   // byte offset of each row in the value stream
  const std::uint8_t* code;
  const std::uint8_t* vals;
  const double* W;
  double* Y;
  std::uint64_t k;
  std::uint32_t dist;  // prefetch distance in non-zeros; 0 = off

  // Decode row i into s.genes / s.counts; returns its number of non-zeros.
  std::uint32_t decode(std::uint64_t i, Scratch& s) const {
    const std::uint8_t* cp = code + code_ptr[i];
    const std::uint8_t* const end = code + code_ptr[i + 1];
    const std::uint8_t* vp = vals + val_ptr[i];
    std::uint32_t* genes = s.genes.get();
    double* counts = s.counts.get();
    std::uint32_t g = 0xFFFFFFFFu;  // "gene -1": the first gap is gene + 1
    std::uint32_t n = 0;
    while (cp < end) {
      const std::uint32_t b = *cp++;
      std::uint32_t gap = b & 0x7Fu;
      if (gap == 0) {  // escape: two-byte gap
        gap = std::uint32_t(cp[0]) | (std::uint32_t(cp[1]) << 8);
        cp += 2;
      }
      g += gap;
      // Count: 1 if bit 7 is set, else the next byte of the value stream.
      // Written as arithmetic on purpose: GCC compiles `one ? 1 : peek` into a
      // branch, and a 70/30 coin flip per non-zero is a branch no predictor can
      // learn (measured: ~9 cycles per non-zero instead of ~3).
      const std::uint32_t uses_val = (b >> 7) ^ 1u;   // 1 if the count is in the value stream
      const std::uint32_t peek = *vp;                 // always in bounds: stream is padded
      std::uint32_t v = (peek & (0u - uses_val)) | (uses_val ^ 1u);  // peek, or 1
      vp += uses_val;                                 // consume a value byte only if used
      if (v == 0) {                                   // escape: two-byte count
        v = std::uint32_t(vp[0]) | (std::uint32_t(vp[1]) << 8);
        vp += 2;
      }
      genes[n] = g;
      counts[n] = static_cast<double>(v);
      ++n;
    }
    return n;
  }

  template <bool PF>
  void rows(std::uint64_t r0, std::uint64_t r1, Scratch& s) const {
    const std::uint64_t full = k - k % kPanel;
    const int rem = static_cast<int>(k % kPanel);
    for (std::uint64_t i = r0; i < r1; ++i) {
      const std::uint32_t n = decode(i, s);
      const std::uint32_t* genes = s.genes.get();
      const double* counts = s.counts.get();
      double* yi = Y + i * k;
      for (std::uint64_t c0 = 0; c0 < full; c0 += kPanel)
        row_panel<kPanel, PF>(genes, counts, n, W + c0, k, yi + c0, dist);
      if (rem) row_panel_any<PF>(rem, genes, counts, n, W + full, k, yi + full, dist);
    }
  }
  void operator()(std::uint64_t r0, std::uint64_t r1, Scratch& s) const {
    if (dist > 0) rows<true>(r0, r1, s);
    else rows<false>(r0, r1, s);
  }
};

// The wide fallback prepare writes when the packed encoding cannot represent
// the input exactly: plain u32 / f64 CSR, handled exactly as rung 4 did.
struct WideRows {
  const std::uint32_t* row_nnz;
  const std::uint32_t* col;
  const double* val;
  const double* W;
  double* Y;
  std::uint64_t k;
  std::uint32_t dist;

  template <bool PF>
  void rows(std::uint64_t r0, std::uint64_t r1) const {
    const std::uint64_t full = k - k % kPanel;
    const int rem = static_cast<int>(k % kPanel);
    for (std::uint64_t i = r0; i < r1; ++i) {
      const std::uint32_t q0 = row_nnz[i];
      const std::uint32_t n = row_nnz[i + 1] - q0;
      double* yi = Y + i * k;
      for (std::uint64_t c0 = 0; c0 < full; c0 += kPanel)
        row_panel<kPanel, PF>(col + q0, val + q0, n, W + c0, k, yi + c0, dist);
      if (rem) row_panel_any<PF>(rem, col + q0, val + q0, n, W + full, k, yi + full, dist);
    }
  }
  void operator()(std::uint64_t r0, std::uint64_t r1, Scratch&) const {
    if (dist > 0) rows<true>(r0, r1);
    else rows<false>(r0, r1);
  }
};

// ---- rung 6: cache sizes and the prefetch policy ---------------------------

struct Caches {
  std::uint64_t l1d, l2, llc;  // bytes; llc = the largest cache shared by a cluster/socket
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
  Caches c{0, 0, 0};
#if defined(__APPLE__)
  c.l1d = sysctl_u64("hw.perflevel0.l1dcachesize");
  c.l2 = sysctl_u64("hw.perflevel0.l2cachesize");
  if (c.l1d == 0) c.l1d = sysctl_u64("hw.l1dcachesize");
  if (c.l2 == 0) c.l2 = sysctl_u64("hw.l2cachesize");
  c.llc = c.l2;
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

// ---- threading: rung 3's, with one Scratch per thread ----------------------

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

template <class Rows>
void run_rows(const Rows& rows, const std::uint32_t* row_nnz, std::uint64_t n_cells,
              std::uint64_t k, std::uint64_t scratch_cap) {
  const std::uint64_t nnz = row_nnz[n_cells];
  const std::uint64_t floor_threads = std::max<std::uint64_t>(1, nnz * k / kMinWorkPerThread);
  const unsigned nt =
      static_cast<unsigned>(std::min<std::uint64_t>(threads_granted(), floor_threads));
  if (nt <= 1) {
    Scratch s(scratch_cap);
    rows(0, n_cells, s);
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
#pragma omp for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) rows(bounds[c], bounds[c + 1], s);
  }
#elif defined(P1_STD_THREAD)
  std::atomic<std::size_t> next{0};
  auto worker = [&]() {
    Scratch s(scratch_cap);  // one per thread, allocated once
    for (;;) {
      const std::size_t c = next.fetch_add(1, std::memory_order_relaxed);
      if (c >= n_chunks) return;
      rows(bounds[c], bounds[c + 1], s);
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
  worker();
  for (std::thread& h : helpers) h.join();
#endif
}

}  // namespace

void hpcbench::solve(const Work& w, const Params& p, Result& y) {
  const Mapped m = w.map("csr_packed.bin");
  if (m.bytes() < 128) fatal("csr_packed.bin is truncated; re-run prepare");
  const std::uint64_t* hdr = m.as<std::uint64_t>(0);
  const std::uint64_t n_cells = hdr[0], n_genes = hdr[1], format = hdr[3];
  if (n_cells != p.n_cells || n_genes != p.n_genes)
    fatal("csr_packed.bin dimensions do not match the harness; re-run prepare");
  if (m.bytes() != hdr[10]) fatal("csr_packed.bin has the wrong size; re-run prepare");
  const std::uint32_t* row_nnz = m.as<std::uint32_t>(hdr[5]);

  const std::uint32_t dist = prefetch_distance(n_genes, p.k, query_caches());

  if (format == 3) {
    const PackedRows rows{m.as<std::uint32_t>(hdr[6]), m.as<std::uint32_t>(hdr[7]),
                          m.as<std::uint8_t>(hdr[8]),  m.as<std::uint8_t>(hdr[9]),
                          p.W, y.data, p.k, dist};
    run_rows(rows, row_nnz, n_cells, p.k, hdr[4]);
  } else if (format == 2) {
    const WideRows rows{row_nnz, m.as<std::uint32_t>(hdr[6]), m.as<double>(hdr[7]),
                        p.W, y.data, p.k, dist};
    run_rows(rows, row_nnz, n_cells, p.k, 0);
  } else {
    fatal("csr_packed.bin has an unknown format; re-run prepare");
  }
}