// src/solution.cpp -- RUNG 1b: read the CSR file through w.map, not a Reader.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// The ONE change from rung 1: how the bytes get from the workdir into the
// kernel. Rung 1 allocated three std::vectors (the allocator zero-fills them,
// touching every page once) and then read() copied the file into them (touching
// every page a second time). Here the file is memory-mapped and the kernel reads
// the page cache in place: no allocation, no zero-fill, no copy. The file
// format and the arithmetic are identical to rung 1.
//
// Work::map is a real mmap natively and falls back to a single read into memory
// in the browser, where there is no mmap; the code below is the same either way.
// csr.bin's fields are all 8 bytes wide and start at multiples of 8, so the
// typed pointers into the mapping are aligned.
//
// Correctness: same traversal and expression as rungs 0 and 1, so the result is
// bit-for-bit the baseline's.
#include "task.hpp"

#include <cstdint>

void hpcbench::solve(const Work& w, const Params& p, Result& y) {
  // On the clock, but now only page mappings: nothing is copied.
  const Mapped m = w.map("csr.bin");
  if (m.bytes() < 24) fatal("csr.bin is truncated; re-run prepare");
  const std::uint64_t* hdr = m.as<std::uint64_t>(0);
  const std::uint64_t n_cells = hdr[0];
  const std::uint64_t n_genes = hdr[1];
  const std::uint64_t nnz = hdr[2];
  if (n_cells != p.n_cells || n_genes != p.n_genes)
    fatal("csr.bin dimensions do not match the harness; re-run prepare");
  if (m.bytes() != 24 + 8 * (n_cells + 1) + 16 * nnz)
    fatal("csr.bin has the wrong size; re-run prepare");

  const std::uint64_t* row_ptr = m.as<std::uint64_t>(24);
  const std::uint64_t* col = row_ptr + (n_cells + 1);
  const double* val = reinterpret_cast<const double*>(col + nnz);

  const std::uint64_t k = p.k;
  for (std::uint64_t i = 0; i < n_cells; ++i)
    for (std::uint64_t q = row_ptr[i]; q < row_ptr[i + 1]; ++q) {
      const double x = val[q];
      const std::uint64_t g = col[q];
      for (std::uint64_t c = 0; c < k; ++c)
        y.data[i * k + c] += x * p.W[g * k + c];
    }
}