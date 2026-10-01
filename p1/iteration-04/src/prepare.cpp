// src/prepare.cpp -- RUNG 4: unchanged from rung 3. UNTIMED.
//
// Rung 4 changes only solution.cpp. This file is rung 3's, which is rung 2's
// plus one diagnostic line.
//
// ---- rung 3 notes, unchanged ----
//
// RUNG 3: unchanged from rung 2, plus one diagnostic line.
//
// The file written here is byte-for-byte rung 2's. The only addition is a line
// on stderr, printed here because prepare is off the clock, that reports how
// many threads solve will be granted and where that number comes from. It tells
// you whether the runner sets OMP_NUM_THREADS.
//
// ---- rung 2 notes, unchanged ----
//
// RUNG 2: CSR with the narrowest types the data allows.
//
//     void hpcbench::prepare(const Input& in, Sink& out);
//
// The ONE change from rung 1b: element types. Rung 1 stored every non-zero as
// an 8-byte index plus an 8-byte value. The data needs far less:
//
//     gene index   13,714 genes          -> fits u16 (max 65,535)
//     count        integers 1..419       -> fits u16 exactly, no rounding
//     row pointer  2,282,976 non-zeros   -> fits u32 (max 4.29 billion)
//
// 16 bytes per non-zero become 4, and the file shrinks from 36.5 MB to 9.1 MB.
// The counts are converted back to double inside the kernel, which is exact
// for integers, so the arithmetic is unchanged.
//
// What would make the narrowing unsafe (Fundamentals, Question 1.8): more than
// 65,536 genes, or a value that is not an integer in 1..65,535. prepare checks
// both and, if either fails, writes a wide fallback (u32 index, f64 value) so
// the entry stays correct on any input. The header says which one was written.
//
// csr_narrow.bin layout, little-endian:
//
//     u64 n_cells, n_genes, nnz, format      format 1 = u16/u16, 2 = u32/f64
//     u64 off_row_ptr, off_col, off_val, total_bytes
//     u32 row_ptr[n_cells + 1]                  at off_row_ptr
//     col[nnz]    u16 or u32                    at off_col
//     val[nnz]    u16 or f64                    at off_val
//
// Every section starts on a 64-byte boundary, so every typed pointer into the
// mapping is aligned for its type, and later rungs can use aligned vector loads.
#include "task.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#elif !defined(__wasm__) && !defined(__EMSCRIPTEN__)
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

namespace {

constexpr std::uint64_t kFormatU16 = 1;   // u16 gene index + u16 count
constexpr std::uint64_t kFormatWide = 2;  // u32 gene index + f64 value

// Diagnostic only: mirrors threads_granted() in solution.cpp.
void report_threads() {
  const char* env = std::getenv("OMP_NUM_THREADS");
#if defined(_OPENMP)
  std::fprintf(stderr, "prepare: solve will be granted %d threads (OpenMP; OMP_NUM_THREADS=%s)\n",
               omp_get_max_threads(), env ? env : "unset");
#elif !defined(__wasm__) && !defined(__EMSCRIPTEN__)
  unsigned n = std::thread::hardware_concurrency();
  const char* source = "hardware threads";
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) {
    n = static_cast<unsigned>(CPU_COUNT(&set));
    source = "CPUs in the affinity mask";
  }
#endif
  if (env && std::strtol(env, nullptr, 10) > 0) {
    n = static_cast<unsigned>(std::strtol(env, nullptr, 10));
    source = "OMP_NUM_THREADS";
  }
  std::fprintf(stderr, "prepare: solve will be granted %u threads (std::thread; from %s)\n", n,
               source);
#else
  std::fprintf(stderr, "prepare: solve will run serially (browser build, no threads)\n");
#endif
}

std::uint64_t pad64(std::uint64_t x) { return (x + 63) & ~std::uint64_t(63); }

void pad_to(hpcbench::Writer& w, std::uint64_t& pos, std::uint64_t target) {
  static const unsigned char zeros[64] = {};
  if (target > pos) w.write(zeros, static_cast<std::size_t>(target - pos));
  pos = target;
}

}  // namespace

void hpcbench::prepare(const Input& in, Sink& out) {
  const std::uint64_t n_cells = in.n_cells;
  const std::uint64_t n_genes = in.n_genes;

  // Gather the non-zeros once, row-major, so genes stay ascending within a row.
  std::vector<std::uint32_t> row_ptr(n_cells + 1, 0);
  std::vector<std::uint32_t> col;
  std::vector<double> val;
  bool narrow = n_genes <= 65536;  // u16 holds gene indices 0..65,535
  for (std::uint64_t i = 0; i < n_cells; ++i) {
    const double* xi = in.X + i * n_genes;
    for (std::uint64_t g = 0; g < n_genes; ++g) {
      const double x = xi[g];
      if (x == 0.0) continue;
      col.push_back(static_cast<std::uint32_t>(g));
      val.push_back(x);
      if (!(x >= 1.0 && x <= 65535.0 && x == std::floor(x))) narrow = false;
    }
    if (col.size() > UINT32_MAX) fatal("more than 2^32 non-zeros: u32 row pointers overflow");
    row_ptr[i + 1] = static_cast<std::uint32_t>(col.size());
  }
  const std::uint64_t nnz = col.size();
  const std::uint64_t format = narrow ? kFormatU16 : kFormatWide;
  const std::uint64_t idx_bytes = narrow ? 2 : 4;
  const std::uint64_t val_bytes = narrow ? 2 : 8;

  const std::uint64_t off_row_ptr = 64;  // header is 8 x u64 = 64 bytes
  const std::uint64_t off_col = pad64(off_row_ptr + 4 * (n_cells + 1));
  const std::uint64_t off_val = pad64(off_col + idx_bytes * nnz);
  const std::uint64_t total = off_val + val_bytes * nnz;

  Writer w = out.create("csr_narrow.bin");
  const std::uint64_t header[8] = {n_cells, n_genes, nnz, format,
                                   off_row_ptr, off_col, off_val, total};
  w.write(header, sizeof(header));
  std::uint64_t pos = sizeof(header);

  pad_to(w, pos, off_row_ptr);
  w.write(row_ptr.data(), row_ptr.size() * sizeof(std::uint32_t));
  pos += 4 * (n_cells + 1);

  pad_to(w, pos, off_col);
  if (narrow) {
    std::vector<std::uint16_t> c16(col.begin(), col.end());
    w.write(c16.data(), c16.size() * sizeof(std::uint16_t));
  } else {
    w.write(col.data(), col.size() * sizeof(std::uint32_t));
  }
  pos += idx_bytes * nnz;

  pad_to(w, pos, off_val);
  if (narrow) {
    std::vector<std::uint16_t> v16(val.size());
    for (std::size_t q = 0; q < val.size(); ++q) v16[q] = static_cast<std::uint16_t>(val[q]);
    w.write(v16.data(), v16.size() * sizeof(std::uint16_t));
  } else {
    w.write(val.data(), val.size() * sizeof(double));
  }
  w.close();

  std::fprintf(stderr, "prepare: CSR %s, %llu nonzeros (%.2f%% dense), %.1f MB\n",
               narrow ? "u16 index + u16 count" : "u32 index + f64 value (wide fallback)",
               (unsigned long long)nnz, 100.0 * double(nnz) / double(n_cells * n_genes),
               double(total) / 1e6);
  report_threads();
}