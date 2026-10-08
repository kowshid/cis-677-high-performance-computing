// src/prepare.cpp -- RUNG 9a: unchanged from rung 5. UNTIMED.
//
// Rungs 6, 7 and 9 change only solution.cpp. Rung 7 blocks W by gene without
// any new offsets in this file: blocks are visited in ascending gene order, so
// each row's decoder simply stops at a block boundary and resumes there.
//
// ---- rung 5 notes, unchanged ----
//
// RUNG 5: a data structure designed from the data.
//
//     void hpcbench::prepare(const Input& in, Sink& out);
//
// The ONE change from rung 4: how each non-zero is encoded. Rung 2 spent 4
// bytes on every non-zero (u16 gene + u16 count). The data says we can do with
// about 1.3, by using what Lecture 3 (slides 28-34) points out about it:
//
//   * 69.8% of the counts are exactly 1.          -> say "1" with a single bit
//   * genes are sorted within a row, gaps are small -> store the gap, not the
//     (median 10, 99.8% under 128)                   gene (delta encoding)
//   * counts other than 1 are small (max 419)      -> one byte each, rarely more
//
// Encoding, per row, in gene order:
//
//   code stream, one byte per non-zero:
//     bit 7     1 if the count is 1, else 0 (count is in the value stream)
//     bits 0-6  gap to the previous gene, 1..127 (the first gap is gene + 1)
//     0 in bits 0-6 is an escape: the gap follows as two bytes (u16)
//   value stream, one byte per non-zero whose count is not 1:
//     2..255    the count itself
//     0         escape: the count follows as two bytes (u16)
//
// Unlike the lecture's grouped-by-value layout, this keeps every row in gene
// order, so the kernel adds the terms of each Y element in exactly the same
// order as before: the compression costs no exactness.
//
// Each row starts at a recorded byte offset in both streams, so any row can be
// decoded on its own -- which is what lets rung 3's threads keep splitting the
// rows freely. A per-row count of non-zeros is kept for the thread balancing.
//
// csr_packed.bin layout, little-endian:
//
//     u64 header[16]:
//       [0] n_cells  [1] n_genes  [2] nnz  [3] format  [4] max_row_nnz
//       [5] off_row_nnz  [6] off_a  [7] off_b  [8] off_c  [9] off_d
//       [10] total_bytes
//     format 3 (packed): a = u32 code_ptr[n_cells + 1]  (byte offset of each row)
//                        b = u32 val_ptr[n_cells + 1]   (byte offset of each row)
//                        c = u8 code stream, d = u8 value stream (+64 pad bytes)
//     format 2 (wide fallback, when a gene index or count does not fit):
//                        a = u32 gene index[nnz], b = f64 value[nnz]
//     u32 row_nnz[n_cells + 1] at off_row_nnz: prefix count of non-zeros
//
// Every section starts on a 64-byte boundary.
#include "task.hpp"

#include <algorithm>
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

constexpr std::uint64_t kFormatWide = 2;
constexpr std::uint64_t kFormatPacked = 3;

// Diagnostic only (unchanged from rung 3): mirrors threads_granted() in solve.
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
  while (pos < target) {
    const std::uint64_t n = std::min<std::uint64_t>(64, target - pos);
    w.write(zeros, static_cast<std::size_t>(n));
    pos += n;
  }
}

void put(hpcbench::Writer& w, std::uint64_t& pos, const void* data, std::uint64_t bytes) {
  w.write(data, static_cast<std::size_t>(bytes));
  pos += bytes;
}

}  // namespace

void hpcbench::prepare(const Input& in, Sink& out) {
  const std::uint64_t n_cells = in.n_cells;
  const std::uint64_t n_genes = in.n_genes;

  // Pass 1: can the packed encoding represent this input exactly?
  // Gaps must fit u16 (gene index <= 65,534) and counts must be integers
  // 1..65,535. Otherwise fall back to u32 index + f64 value.
  bool packable = n_genes <= 65535;
  std::uint64_t nnz = 0, max_row = 0;
  std::vector<std::uint32_t> row_nnz(n_cells + 1, 0);
  for (std::uint64_t i = 0; i < n_cells; ++i) {
    const double* xi = in.X + i * n_genes;
    std::uint64_t row = 0;
    for (std::uint64_t g = 0; g < n_genes; ++g) {
      const double x = xi[g];
      if (x == 0.0) continue;
      ++row;
      if (!(x >= 1.0 && x <= 65535.0 && x == std::floor(x))) packable = false;
    }
    nnz += row;
    if (nnz > UINT32_MAX) fatal("more than 2^32 non-zeros: u32 offsets overflow");
    max_row = std::max(max_row, row);
    row_nnz[i + 1] = static_cast<std::uint32_t>(nnz);
  }

  std::uint64_t header[16] = {};
  header[0] = n_cells;
  header[1] = n_genes;
  header[2] = nnz;
  header[4] = max_row;
  header[5] = 128;  // off_row_nnz
  const std::uint64_t after_row_nnz = pad64(header[5] + 4 * (n_cells + 1));

  Writer w = out.create("csr_packed.bin");
  std::uint64_t pos = 0;

  if (packable) {
    // Pass 2: encode.
    std::vector<std::uint8_t> code, vals;
    std::vector<std::uint32_t> code_ptr(n_cells + 1, 0), val_ptr(n_cells + 1, 0);
    code.reserve(static_cast<std::size_t>(nnz + nnz / 64));
    vals.reserve(static_cast<std::size_t>(nnz / 2));
    std::uint64_t gap_escapes = 0, val_escapes = 0;
    for (std::uint64_t i = 0; i < n_cells; ++i) {
      const double* xi = in.X + i * n_genes;
      std::uint64_t prev = ~std::uint64_t(0);  // "gene -1": the first gap is gene + 1
      for (std::uint64_t g = 0; g < n_genes; ++g) {
        const double x = xi[g];
        if (x == 0.0) continue;
        const std::uint64_t gap = g - prev;  // 1..65,535
        prev = g;
        const std::uint32_t v = static_cast<std::uint32_t>(x);
        const std::uint8_t one = (v == 1) ? 0x80 : 0x00;
        if (gap < 128) {
          code.push_back(static_cast<std::uint8_t>(one | gap));
        } else {
          code.push_back(one);  // gap field 0 = escape
          code.push_back(static_cast<std::uint8_t>(gap & 0xFF));
          code.push_back(static_cast<std::uint8_t>(gap >> 8));
          ++gap_escapes;
        }
        if (v != 1) {
          if (v < 256) {
            vals.push_back(static_cast<std::uint8_t>(v));
          } else {
            vals.push_back(0);  // escape
            vals.push_back(static_cast<std::uint8_t>(v & 0xFF));
            vals.push_back(static_cast<std::uint8_t>(v >> 8));
            ++val_escapes;
          }
        }
      }
      if (code.size() > UINT32_MAX || vals.size() > UINT32_MAX)
        fatal("packed streams exceed 4 GB");
      code_ptr[i + 1] = static_cast<std::uint32_t>(code.size());
      val_ptr[i + 1] = static_cast<std::uint32_t>(vals.size());
    }

    header[3] = kFormatPacked;
    header[6] = after_row_nnz;                               // code_ptr
    header[7] = pad64(header[6] + 4 * (n_cells + 1));        // val_ptr
    header[8] = pad64(header[7] + 4 * (n_cells + 1));        // code stream
    header[9] = pad64(header[8] + code.size());              // value stream
    header[10] = header[9] + vals.size() + 64;               // + pad: the decoder
                                                             // may peek one byte ahead
    put(w, pos, header, sizeof(header));
    pad_to(w, pos, header[5]);
    put(w, pos, row_nnz.data(), 4 * (n_cells + 1));
    pad_to(w, pos, header[6]);
    put(w, pos, code_ptr.data(), 4 * (n_cells + 1));
    pad_to(w, pos, header[7]);
    put(w, pos, val_ptr.data(), 4 * (n_cells + 1));
    pad_to(w, pos, header[8]);
    put(w, pos, code.data(), code.size());
    pad_to(w, pos, header[9]);
    put(w, pos, vals.data(), vals.size());
    pad_to(w, pos, header[10]);
    w.close();

    std::fprintf(stderr,
                 "prepare: packed CSR, %llu nonzeros, %.2f MB (%.2f bytes/nnz); "
                 "code %.2f MB, values %.2f MB, %llu gap and %llu value escapes\n",
                 (unsigned long long)nnz, double(header[10]) / 1e6,
                 double(header[10]) / double(nnz), double(code.size()) / 1e6,
                 double(vals.size()) / 1e6, (unsigned long long)gap_escapes,
                 (unsigned long long)val_escapes);
  } else {
    std::vector<std::uint32_t> col;
    std::vector<double> val;
    col.reserve(static_cast<std::size_t>(nnz));
    val.reserve(static_cast<std::size_t>(nnz));
    for (std::uint64_t i = 0; i < n_cells; ++i) {
      const double* xi = in.X + i * n_genes;
      for (std::uint64_t g = 0; g < n_genes; ++g)
        if (xi[g] != 0.0) {
          col.push_back(static_cast<std::uint32_t>(g));
          val.push_back(xi[g]);
        }
    }
    header[3] = kFormatWide;
    header[6] = after_row_nnz;                      // u32 gene index
    header[7] = pad64(header[6] + 4 * nnz);         // f64 value
    header[10] = header[7] + 8 * nnz;
    put(w, pos, header, sizeof(header));
    pad_to(w, pos, header[5]);
    put(w, pos, row_nnz.data(), 4 * (n_cells + 1));
    pad_to(w, pos, header[6]);
    put(w, pos, col.data(), 4 * nnz);
    pad_to(w, pos, header[7]);
    put(w, pos, val.data(), 8 * nnz);
    w.close();
    std::fprintf(stderr, "prepare: wide fallback CSR (u32 + f64), %llu nonzeros, %.2f MB\n",
                 (unsigned long long)nnz, double(header[10]) / 1e6);
  }
  report_threads();
}