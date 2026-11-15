// src/prepare.cpp -- YOURS. UNTIMED.
//
//     void hpcbench::prepare(const Input& in, Sink& out);
//
// `in.X` is the corpus AFTER the host's transform: row-major n_cells x n_genes
// doubles, every value log1p(raw count), 6.17% non-zero.
//
// Instead of the dense 296 MB matrix, this writes the non-zeros once, as
// compressed sparse rows (one row per cell), into a single file that `solve`
// maps with mmap:
//
//     offset 0                 char[8]  magic "P2CSRv01"
//            8                 u64      n_cells
//            16                u64      n_genes
//            24                u64      nnz
//            32                u64      row_ptr[n_cells + 1]
//            32 + 8(N+1)       double   val[nnz]      (exact host values)
//            ...               u16      col[nnz]      (gene index, ascending per row)
//
// Doubles come before the 16-bit indices so every array is naturally aligned
// in the mapping. About 23 MB for pbmc3k instead of 296 MB.
//
// Nothing here depends on the seed: the batches are regrouped inside solve.
// KEEP THE LAYOUT IN SYNC WITH solution.cpp.
#include "task.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr char kFileName[] = "xcsr.bin";
constexpr char kMagic[8] = {'P', '2', 'C', 'S', 'R', 'v', '0', '1'};

}  // namespace

void hpcbench::prepare(const Input& in, Sink& out) {
  const std::uint64_t N = in.n_cells, D = in.n_genes;
  if (D > 65536) hpcbench::fatal("prepare: n_genes does not fit the 16-bit column index");

  // Pass 1: row pointers.
  std::vector<std::uint64_t> row_ptr(N + 1, 0);
  for (std::uint64_t i = 0; i < N; ++i) {
    const double* row = in.X + i * D;
    std::uint64_t n = 0;
    for (std::uint64_t j = 0; j < D; ++j) n += (row[j] != 0.0);
    row_ptr[i + 1] = row_ptr[i] + n;
  }
  const std::uint64_t nnz = row_ptr[N];

  // Pass 2: values and gene indices, ascending gene order within each cell.
  std::vector<double> val(nnz);
  std::vector<std::uint16_t> col(nnz);
  for (std::uint64_t i = 0; i < N; ++i) {
    const double* row = in.X + i * D;
    std::uint64_t t = row_ptr[i];
    for (std::uint64_t j = 0; j < D; ++j)
      if (row[j] != 0.0) {
        val[t] = row[j];
        col[t] = static_cast<std::uint16_t>(j);
        ++t;
      }
  }

  Writer w = out.create(kFileName);
  w.write(kMagic, sizeof(kMagic));
  w.write_value(N);
  w.write_value(D);
  w.write_value(nnz);
  w.write(row_ptr.data(), row_ptr.size() * sizeof(std::uint64_t));
  w.write(val.data(), nnz * sizeof(double));
  w.write(col.data(), nnz * sizeof(std::uint16_t));
  w.close();

  const double mb = double(32 + (N + 1) * 8 + nnz * 10) / 1e6;
  std::fprintf(stderr, "prepare: wrote %s, %llu non-zeros of %llu (%.2f%% dense), %.1f MB\n",
               kFileName, (unsigned long long)nnz, (unsigned long long)(N * D),
               100.0 * double(nnz) / double(N * D), mb);
}