// src/prepare.cpp -- RUNG 1: write X in CSR instead of dense. UNTIMED.
//
//     void hpcbench::prepare(const Input& in, Sink& out);
//
// The ONE change from the baseline: store only the non-zeros, grouped by row
// (cell), in Compressed Sparse Row form. The element types are deliberately kept
// as wide as possible -- 8-byte indices, 8-byte values -- so this rung measures
// the format alone; narrowing the types is rung 2.
//
// csr.bin layout, all little-endian, every field 8 bytes:
//
//     u64 n_cells, u64 n_genes, u64 nnz
//     u64 row_ptr[n_cells + 1]   row i owns slots [row_ptr[i], row_ptr[i+1])
//     u64 col[nnz]               gene index of each non-zero, ascending per row
//     f64 val[nnz]               the count itself
//
// 2,282,976 non-zeros x 16 bytes + 21.6 KB of row pointers = 36.5 MB, down
// from 296.2 MB dense.
#include "task.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

void hpcbench::prepare(const Input& in, Sink& out) {
  const std::uint64_t n_cells = in.n_cells;
  const std::uint64_t n_genes = in.n_genes;

  std::vector<std::uint64_t> row_ptr(n_cells + 1, 0);
  std::vector<std::uint64_t> col;
  std::vector<double> val;

  // Row-major scan: genes come out ascending within each row, which keeps the
  // per-element summation order identical to the dense baseline.
  for (std::uint64_t i = 0; i < n_cells; ++i) {
    const double* xi = in.X + i * n_genes;
    for (std::uint64_t g = 0; g < n_genes; ++g) {
      if (xi[g] != 0.0) {
        col.push_back(g);
        val.push_back(xi[g]);
      }
    }
    row_ptr[i + 1] = col.size();
  }
  const std::uint64_t nnz = col.size();

  Writer w = out.create("csr.bin");
  w.write_value(n_cells);
  w.write_value(n_genes);
  w.write_value(nnz);
  w.write(row_ptr.data(), row_ptr.size() * sizeof(std::uint64_t));
  w.write(col.data(), col.size() * sizeof(std::uint64_t));
  w.write(val.data(), val.size() * sizeof(double));
  w.close();

  const double bytes = 24.0 + 8.0 * double(n_cells + 1) + 16.0 * double(nnz);
  std::fprintf(stderr, "prepare: CSR u64/f64, %llu nonzeros (%.2f%% dense), %.1f MB\n",
               (unsigned long long)nnz, 100.0 * double(nnz) / double(n_cells * n_genes),
               bytes / 1e6);
}