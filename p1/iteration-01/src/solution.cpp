// src/solution.cpp -- RUNG 1: CSR traversal, same read mechanism as baseline.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// The ONE change from rung 0: X arrives in CSR (see prepare.cpp), so solve
// reads 36.5 MB instead of 296 MB and walks only the non-zeros -- no scan over
// zeros, no branch per entry. How the bytes are read is kept exactly as the
// baseline does it (a streaming Reader into zero-initialised std::vectors) so
// this rung measures the format alone; the read path is rung 1b.
//
// Loop order (Lecture 3, slide 8): row i of X builds row i of Y. Writes to Y
// run in order; each non-zero jumps to the W row of its gene.
//
// Correctness: each Y[i][c] still receives its terms in ascending gene order
// through the same `y += x * w` expression, so the result is bit-for-bit the
// baseline's.
#include "task.hpp"

#include <cstdint>
#include <vector>

void hpcbench::solve(const Work& w, const Params& p, Result& y) {
  // On the clock: stream csr.bin into freshly allocated, zero-filled vectors.
  Reader r = w.open("csr.bin");
  const std::uint64_t n_cells = r.read_value<std::uint64_t>();
  const std::uint64_t n_genes = r.read_value<std::uint64_t>();
  const std::uint64_t nnz = r.read_value<std::uint64_t>();
  if (n_cells != p.n_cells || n_genes != p.n_genes)
    fatal("csr.bin dimensions do not match the harness; re-run prepare");

  std::vector<std::uint64_t> row_ptr(static_cast<std::size_t>(n_cells + 1));
  std::vector<std::uint64_t> col(static_cast<std::size_t>(nnz));
  std::vector<double> val(static_cast<std::size_t>(nnz));
  r.read(row_ptr.data(), row_ptr.size() * sizeof(std::uint64_t));
  r.read(col.data(), col.size() * sizeof(std::uint64_t));
  r.read(val.data(), val.size() * sizeof(double));
  r.close();

  const std::uint64_t k = p.k;
  for (std::uint64_t i = 0; i < n_cells; ++i)
    for (std::uint64_t q = row_ptr[i]; q < row_ptr[i + 1]; ++q) {
      const double x = val[q];
      const std::uint64_t g = col[q];
      for (std::uint64_t c = 0; c < k; ++c)
        y.data[i * k + c] += x * p.W[g * k + c];
    }
}