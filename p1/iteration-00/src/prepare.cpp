// src/prepare.cpp -- YOURS. UNTIMED.
//
// One function, no main(), no argv, no file format you have to guess:
//
//     void hpcbench::prepare(const Input& in, Sink& out);
//
// `in` is the raw corpus, already read for you: `in.X` is row-major
// n_cells x n_genes doubles. `out` is your workdir; `out.create("name")` hands
// you a Writer. Whatever you write here is what `solve` has to read, and that
// decision is most of the project.
//
// Nothing here is on the clock. Sort it, compress it, quantise it, shard it,
// build an index. Note what is NOT in `Input`: `k`. The sweep value does not
// exist yet, so it cannot be tabled.
//
// The baseline writes the dense matrix straight back out, which is the worst
// possible answer and is the point.
#include "task.hpp"

#include <cstdint>
#include <cstdio>

void hpcbench::prepare(const Input& in, Sink& out) {
  Writer w = out.create("matrix.bin");
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
