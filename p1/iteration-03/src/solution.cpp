// src/solution.cpp -- RUNG 3: split the cells across threads.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// The ONE change from rung 2: rows of X (cells) are processed by several
// threads at once. The file, the kernel and the arithmetic are unchanged.
//
// Why rows parallelise cleanly: row i of X builds row i of Y and nothing else,
// so every Y row has exactly one writer -- no locks, no atomics on Y, no
// reduction. Each Y element still receives its terms in ascending gene order
// from a single thread, so the result is bit-for-bit the baseline's whatever
// the thread count or the schedule.
//
// How the rows are split:
//   * Chunks hold roughly equal numbers of NON-ZEROS, not rows: cells carry
//     212 to 3,400 non-zeros each, a 16x spread.
//   * About 32 chunks per thread, handed out dynamically. A fast core simply
//     takes more chunks than a slow one, which matters on the M4's mix of
//     4 performance and 6 efficiency cores, and keeps the last chunk short.
//
// How many threads:
//   * Whatever the environment grants: OMP_NUM_THREADS if it is set, otherwise
//     every hardware thread (on Linux, every CPU in the affinity mask). No
//     count is hard-coded.
//   * Minus a work floor: a thread is only started for at least 2^19
//     multiply-adds (~0.1 ms). That caps tiny problems (k = 1 on a 48-thread
//     node) and never binds at the published widths on a laptop.
//
// Mechanism: OpenMP when the compiler has it (build.sh adds -fopenmp when the
// probe succeeds -- GCC on Linux), std::thread otherwise (Apple clang has no
// OpenMP out of the box), and a plain serial loop in the browser build, which
// has no threads.
#include "task.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#define RUNG3_OPENMP 1
#elif !defined(__wasm__) && !defined(__EMSCRIPTEN__)
#define RUNG3_STD_THREAD 1
#include <atomic>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

namespace {

constexpr std::uint64_t kMinWorkPerThread = std::uint64_t(1) << 19;  // multiply-adds
constexpr std::uint64_t kChunksPerThread = 32;

// Rows [r0, r1): exactly rung 2's kernel.
template <class Idx, class Val>
void spmm_rows(std::uint64_t r0, std::uint64_t r1, const std::uint32_t* row_ptr,
               const Idx* col, const Val* val, const double* W, double* Y,
               std::uint64_t k) {
  for (std::uint64_t i = r0; i < r1; ++i)
    for (std::uint64_t q = row_ptr[i]; q < row_ptr[i + 1]; ++q) {
      const double x = static_cast<double>(val[q]);
      const std::uint64_t g = col[q];
      for (std::uint64_t c = 0; c < k; ++c)
        Y[i * k + c] += x * W[g * k + c];
    }
}

// The thread count the environment grants. Nothing here imposes a limit.
unsigned threads_granted() {
#if defined(RUNG3_OPENMP)
  return static_cast<unsigned>(std::max(1, omp_get_max_threads()));
#elif defined(RUNG3_STD_THREAD)
  if (const char* s = std::getenv("OMP_NUM_THREADS")) {  // same knob as OpenMP
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

  // Chunk boundaries (row indices) with ~equal non-zeros per chunk.
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

#if defined(RUNG3_OPENMP)
  const long nc = static_cast<long>(n_chunks);
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
  for (long c = 0; c < nc; ++c)
    spmm_rows(bounds[c], bounds[c + 1], row_ptr, col, val, W, Y, k);
#elif defined(RUNG3_STD_THREAD)
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
      break;  // fewer helpers is slower, never wrong: the queue drains regardless
    }
  }
  worker();  // the calling thread works too
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