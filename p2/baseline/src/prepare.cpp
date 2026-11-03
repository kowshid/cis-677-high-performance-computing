// src/prepare.cpp -- YOURS. UNTIMED.
//
//     void hpcbench::prepare(const Input& in, Sink& out);
//
// `in.X` is the corpus AFTER the host's transform: row-major n_cells x n_genes
// doubles, every value log1p(raw count). It is 6.17% non-zero, because
// log1p(0) = 0. `out.create("name")` hands you a Writer into your workdir.
// Whatever you write here is what `solve` has to read.
//
// Nothing here is on the clock. Note what is NOT in `Input`: the seed. The
// initial weights and the batch order do not exist yet, so the epoch cannot be
// run here.
//
// The baseline writes the dense matrix straight back out -- 296 MB of doubles,
// 94% of them zero -- which is the Project 1 mistake made again on purpose.
#include "task.hpp"

#include <cstdint>
#include <cstdio>

void hpcbench::prepare(const Input& in, Sink& out) {
  Writer w = out.create("X.bin");
  w.write_value(in.n_cells);
  w.write_value(in.n_genes);
  w.write(in.X, static_cast<std::size_t>(in.n_cells * in.n_genes) * sizeof(double));
  w.close();

  std::uint64_t nnz = 0;
  const std::uint64_t n = in.n_cells * in.n_genes;
  for (std::uint64_t i = 0; i < n; ++i)
    if (in.X[i] != 0.0) ++nnz;
  std::fprintf(stderr, "prepare: %llu nonzeros of %llu (%.2f%% dense)\n",
               (unsigned long long)nnz, (unsigned long long)n,
               100.0 * double(nnz) / double(n));
}
