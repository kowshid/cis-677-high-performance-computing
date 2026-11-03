// task.hpp -- HOST-OWNED. Read-only for competitors.  (CIS 677 Project 3)
//
// THE TASK. Train the autoencoder
//
//     n_genes -> 32 -> 16 -> 32 -> n_genes        (38,606 genes)
//
// for exactly ONE epoch over every cell of a single-cell corpus that lives on
// Clipper, then write the TRAINED parameters into `out`. ReLU after the two
// 32-wide layers; the 16-wide latent and the output are linear. The input to
// the network is log1p(count). Loss: mean squared error against that input,
// averaged over every element of the batch. Adam, lr 1e-3, beta1 0.9, beta2
// 0.999, eps 1e-8, bias-corrected, one step per batch of 1,024 cells, in the
// batch order the host hands you (the last batch may be short).
//
// The baseline already trains fast: it carries the Project 2 techniques that
// won (CSR input, SoA Adam, threads that own their outputs, fp32, the Gram form
// of layer 4). Its time goes to the LOADER: reading each batch's cells from the
// corpus file and building the batch. That is the part to make fast.
//
// THE CORPUS. One file, CSR, cells x genes, little-endian, read with the
// helpers below or however you like:
//
//     u64 n_cells, u64 n_genes, u64 nnz
//     u64 indptr[n_cells + 1]        cell i is entries [indptr[i], indptr[i+1])
//     u32 indices[nnz]               gene of each entry, ascending within a cell
//     f32 data[nnz]                  raw UMI count (apply log1p yourself)
//
// The format is fixed: there is no prepare step and you cannot rewrite the
// corpus. The host empties the node's page cache for this file before the
// clock starts, so every run reads it cold.
//
// MEMORY RULE. Peak resident memory of the whole process during solve must stay
// below a quarter of the corpus file size. Project 3 is out-of-core by rule:
// reading the whole corpus into RAM fails the run. Mapping it with mmap counts
// the pages you touch.
//
// HOW IT IS CHECKED. After solve returns (clock stopped), the harness computes
// the full-data reconstruction MSE of the parameters you wrote, every cell and
// every gene, in double precision. That one number must agree with the
// baseline's to within the competition's relative tolerance.
//
// LAYOUTS (row-major, fp32): W1 [n_genes][32], b1 [32], W2 [32][16], b2 [16],
// W3 [16][32], b3 [32], W4T [n_genes][32] (layer 4's weights TRANSPOSED: row g
// holds the 32 weights into output gene g), b4 [n_genes].
#ifndef HPCBENCH_TASK_HPP
#define HPCBENCH_TASK_HPP

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace hpcbench {

constexpr std::uint64_t kHidden = 32;   // width of layers 1 and 3
constexpr std::uint64_t kLatent = 16;   // width of the bottleneck
constexpr std::uint64_t kBatch = 1024;  // cells per optimiser step
constexpr double kLearningRate = 1e-3, kBeta1 = 0.9, kBeta2 = 0.999, kEps = 1e-8;

[[noreturn]] inline void fatal(const std::string& msg) {
  std::fprintf(stderr, "hpcbench: %s\n", msg.c_str());
  std::exit(1);
}

// The corpus file, already opened read-only by the host.
struct Corpus {
  std::string path;
  int fd = -1;
  std::uint64_t n_cells = 0, n_genes = 0, nnz = 0;
  std::uint64_t off_indptr = 24;  // byte offsets of the three arrays
  std::uint64_t off_indices = 0;
  std::uint64_t off_data = 0;
  std::uint64_t bytes = 0;        // file size

  // pread exactly n bytes at byte offset off (loops over short reads).
  void read_at(void* dst, std::uint64_t n, std::uint64_t off) const {
    char* p = static_cast<char*>(dst);
    while (n) {
      const ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
      if (r <= 0) fatal("read failed on " + path);
      p += r; n -= static_cast<std::uint64_t>(r); off += static_cast<std::uint64_t>(r);
    }
  }
};

struct Params {
  std::uint64_t seed = 0, n_cells = 0, n_genes = 0;
  std::uint64_t hidden = kHidden, latent = kLatent, batch = kBatch;
  double lr = kLearningRate, beta1 = kBeta1, beta2 = kBeta2, eps = kEps;
  const std::uint32_t* order = nullptr;  // batch b = order[b*batch .. ), n_cells entries
  // initial parameters, layouts as above
  const float *W1 = nullptr, *b1 = nullptr, *W2 = nullptr, *b2 = nullptr;
  const float *W3 = nullptr, *b3 = nullptr, *W4T = nullptr, *b4 = nullptr;
};

// Host-allocated, zero-filled. Write the trained parameters here.
struct Result {
  float *W1 = nullptr, *b1 = nullptr, *W2 = nullptr, *b2 = nullptr;
  float *W3 = nullptr, *b3 = nullptr, *W4T = nullptr, *b4 = nullptr;
};

// THE ONE FUNCTION AN ENTRY PROVIDES (src/solution.cpp). Timed.
void solve(const Corpus& c, const Params& p, Result& out);

}  // namespace hpcbench

#endif
