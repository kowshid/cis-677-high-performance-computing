// src/solution.cpp -- RUNG 2: the rung-1b kernel over narrow types.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// The ONE change from rung 1b: the element types in the file (see prepare.cpp).
// The mapping, the traversal and the arithmetic are the same. Each count is a
// u16 converted to double in a register, which is exact, and each gene index is
// a u16 widened to 64 bits for the address of its W row.
//
// What shrinks: the file is 9.1 MB instead of 36.5 MB, so the mapping touches a
// quarter of the pages, and the index/value stream the kernel pulls per
// non-zero drops from 16 bytes to 4. What does not shrink: the W row each
// non-zero reads (8k bytes) and the Y row each cell writes.
//
// The kernel is a template over the index and value types, so the wide
// fallback prepare writes for data that does not fit u16 runs the same code.
//
// Correctness: same order, same `y += x * w` expression, exact conversions, so
// the result is bit-for-bit the baseline's.
#include "task.hpp"

#include <cstdint>

namespace {

template <class Idx, class Val>
void spmm(const std::uint32_t* row_ptr, const Idx* col, const Val* val,
          const double* W, double* Y, std::uint64_t n_cells, std::uint64_t k) {
  for (std::uint64_t i = 0; i < n_cells; ++i)
    for (std::uint64_t q = row_ptr[i]; q < row_ptr[i + 1]; ++q) {
      const double x = static_cast<double>(val[q]);
      const std::uint64_t g = col[q];
      for (std::uint64_t c = 0; c < k; ++c)
        Y[i * k + c] += x * W[g * k + c];
    }
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
    spmm(row_ptr, m.as<std::uint16_t>(off_col), m.as<std::uint16_t>(off_val), p.W, y.data,
         n_cells, p.k);
  else if (format == 2)
    spmm(row_ptr, m.as<std::uint32_t>(off_col), m.as<double>(off_val), p.W, y.data, n_cells,
         p.k);
  else
    fatal("csr_narrow.bin has an unknown format; re-run prepare");
}