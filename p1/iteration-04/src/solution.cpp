// src/solution.cpp -- RUNG 4: stepwise caching at the register level.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// The ONE change from rung 3: the inner kernel. Everything else -- the file,
// the mapping, the threads, the chunking -- is rung 3's.
//
// What rung 3's kernel did, per non-zero:
//     for (c = 0; c < k; ++c) Y[i*k + c] += x * W[g*k + c];
// k is only known at run time, so the compiler emits a generic loop with a
// trip-count check, a vector body, a remainder and an overlap test between Y
// and W -- per non-zero. And every non-zero loads AND stores the Y row: at
// k = 256 that is 256 loads + 256 stores of Y next to 256 loads of W.
//
// What this kernel does instead:
//   * Split k into panels of P columns, with P a COMPILE-TIME constant.
//   * For one row and one panel, keep the P partial sums of Y in registers
//     across all of that row's non-zeros, and store them once at the end.
//     Registers are the fastest level of the hierarchy; Y now costs one store
//     per row per panel instead of a load and a store per non-zero.
//   * Because P is a template argument, the P-wide update is fully unrolled:
//     no loop control, no remainder, no overlap test inside the hot loop.
//   * A switch on the width left over (k mod P) picks a pre-compiled kernel for
//     it, so any k -- odd, prime, 257 -- runs one exact specialised pass.
//
// Panel width: 32 columns = 16 NEON q registers or 8 AVX2/AVX-512 ymm
// registers of accumulators, leaving room for the W loads. On NEON, 64 would
// spill accumulators to the stack. On AVX-512, 64 fits, but measured no faster
// than 32 (within noise at k = 128, 256; 8% slower at k = 64), so 32 everywhere.
//
// Vector type: v4d is a GCC/Clang vector extension, four doubles. Clang
// already kept plain `double acc[32]` in registers, but GCC 13 did not (it
// vectorised across non-zeros and bounced the accumulators through the stack
// every iteration); spelling the accumulators as vectors fixes that on both.
//
// Correctness: every Y element is still accumulated from zero, in ascending
// gene order, one fused multiply-add per non-zero -- the same sequence of
// operations as the baseline -- so the result is bit-for-bit the baseline's.
#include "task.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

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

constexpr std::uint64_t kMinWorkPerThread = std::uint64_t(1) << 19;  // multiply-adds
constexpr std::uint64_t kChunksPerThread = 32;

// One row, one panel of P output columns:  yc[0:P] = sum over the row's
// non-zeros of x * Wc[g*k + 0:P], accumulated in registers.
template <int P, class Idx, class Val>
void row_panel(const Idx* col, const Val* val, std::uint32_t n, const double* Wc,
               std::uint64_t k, double* yc) {
  constexpr int V = P / 4;  // full 4-wide vectors
  constexpr int R = P % 4;  // 1-3 leftover columns, scalar
  v4d acc[V > 0 ? V : 1];
  double tail[R > 0 ? R : 1];
#pragma GCC unroll 16
  for (int v = 0; v < V; ++v) acc[v] = v4d{0.0, 0.0, 0.0, 0.0};
  for (int r = 0; r < R; ++r) tail[r] = 0.0;

  for (std::uint32_t q = 0; q < n; ++q) {
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
template <class Idx, class Val>
void row_panel_any(int width, const Idx* col, const Val* val, std::uint32_t n,
                   const double* Wc, std::uint64_t k, double* yc) {
  switch (width) {
#define P1_CASE(P)                             \
  case P:                                      \
    row_panel<P>(col, val, n, Wc, k, yc);      \
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

// Rows [r0, r1): full panels first, then one pass for whatever width is left.
template <class Idx, class Val>
void spmm_rows(std::uint64_t r0, std::uint64_t r1, const std::uint32_t* row_ptr,
               const Idx* col, const Val* val, const double* W, double* Y,
               std::uint64_t k) {
  const std::uint64_t full = k - k % kPanel;
  const int rem = static_cast<int>(k % kPanel);
  for (std::uint64_t i = r0; i < r1; ++i) {
    const std::uint32_t q0 = row_ptr[i];
    const std::uint32_t n = row_ptr[i + 1] - q0;
    double* yi = Y + i * k;
    for (std::uint64_t c0 = 0; c0 < full; c0 += kPanel)
      row_panel<kPanel>(col + q0, val + q0, n, W + c0, k, yi + c0);
    if (rem) row_panel_any(rem, col + q0, val + q0, n, W + full, k, yi + full);
  }
}

// ---- threading: unchanged from rung 3 ------------------------------------

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

template <class Idx, class Val>
void spmm_parallel(const std::uint32_t* row_ptr, const Idx* col, const Val* val,
                   const double* W, double* Y, std::uint64_t n_cells, std::uint64_t k) {
  const std::uint64_t nnz = row_ptr[n_cells];
  const std::uint64_t floor_threads = std::max<std::uint64_t>(1, nnz * k / kMinWorkPerThread);
  const unsigned nt =
      static_cast<unsigned>(std::min<std::uint64_t>(threads_granted(), floor_threads));
  if (nt <= 1) {
    spmm_rows(0, n_cells, row_ptr, col, val, W, Y, k);
    return;
  }

  std::vector<std::uint32_t> bounds;
  bounds.reserve(static_cast<std::size_t>(nt * kChunksPerThread + 2));
  bounds.push_back(0);
  const std::uint64_t target = std::max<std::uint64_t>(1, nnz / (nt * kChunksPerThread));
  std::uint64_t chunk_start = row_ptr[0];
  for (std::uint64_t i = 0; i < n_cells; ++i)
    if (row_ptr[i + 1] - chunk_start >= target) {
      bounds.push_back(static_cast<std::uint32_t>(i + 1));
      chunk_start = row_ptr[i + 1];
    }
  if (bounds.back() != n_cells) bounds.push_back(static_cast<std::uint32_t>(n_cells));
  const std::size_t n_chunks = bounds.size() - 1;

#if defined(P1_OPENMP)
  const long nc = static_cast<long>(n_chunks);
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
  for (long c = 0; c < nc; ++c)
    spmm_rows(bounds[c], bounds[c + 1], row_ptr, col, val, W, Y, k);
#elif defined(P1_STD_THREAD)
  std::atomic<std::size_t> next{0};
  auto worker = [&]() {
    for (;;) {
      const std::size_t c = next.fetch_add(1, std::memory_order_relaxed);
      if (c >= n_chunks) return;
      spmm_rows(bounds[c], bounds[c + 1], row_ptr, col, val, W, Y, k);
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
  const Mapped m = w.map("csr_narrow.bin");
  if (m.bytes() < 64) fatal("csr_narrow.bin is truncated; re-run prepare");
  const std::uint64_t* hdr = m.as<std::uint64_t>(0);
  const std::uint64_t n_cells = hdr[0], n_genes = hdr[1], format = hdr[3];
  const std::uint64_t off_row_ptr = hdr[4], off_col = hdr[5], off_val = hdr[6];
  if (n_cells != p.n_cells || n_genes != p.n_genes)
    fatal("csr_narrow.bin dimensions do not match the harness; re-run prepare");
  if (m.bytes() != hdr[7]) fatal("csr_narrow.bin has the wrong size; re-run prepare");

  const std::uint32_t* row_ptr = m.as<std::uint32_t>(off_row_ptr);
  if (format == 1)
    spmm_parallel(row_ptr, m.as<std::uint16_t>(off_col), m.as<std::uint16_t>(off_val), p.W,
                  y.data, n_cells, p.k);
  else if (format == 2)
    spmm_parallel(row_ptr, m.as<std::uint32_t>(off_col), m.as<double>(off_val), p.W, y.data,
                  n_cells, p.k);
  else
    fatal("csr_narrow.bin has an unknown format; re-run prepare");
}