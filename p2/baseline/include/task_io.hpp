// task_io.hpp — HOST-OWNED. Read-only for competitors.
//
// The pieces of CIS 677 Project 2 that are the host's DEFINITION OF THE PROBLEM
// rather than anyone's optimisation:
//
//   hpcbench::read_raw_input   reading the raw corpus (same file as Project 1)
//   hpcbench::transform_input  the input transform: x -> log1p(x), elementwise
//   hpcbench::SplitMix64       the one random stream everything is drawn from
//   hpcbench::init_params      the initial weights, from the seed
//   hpcbench::batch_order      the order cells are visited in, from the seed
//
// Why these live here and not in an entry: the answer to this task is a
// deterministic function of (corpus, seed). If an entry generated its own
// initial weights or its own batch order, it could quietly train a different
// model. With one host-owned definition there is exactly one right answer per
// seed, and `seed` does not exist while `prepare` runs.
//
// The runners stage this file LAST, after an entry's own sources, so an entry
// that ships a file at this path cannot replace it.
#ifndef HPCBENCH_TASK_IO_HPP
#define HPCBENCH_TASK_IO_HPP

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hpcbench {

// ---------------------------------------------------------------------------
// The architecture and the optimiser. FIXED: these are the problem.
// ---------------------------------------------------------------------------
constexpr std::uint64_t kHidden = 256;      // width of both hidden layers
constexpr std::uint64_t kLatent = 32;       // width of the bottleneck
constexpr std::uint64_t kBatch = 256;       // cells per optimiser step
constexpr double kLearningRate = 1e-3;      // Adam
constexpr double kBeta1 = 0.9;
constexpr double kBeta2 = 0.999;
constexpr double kEps = 1e-8;

// A bounded open()/read() loop. REGRESSION GUARD -- do NOT replace this with
// buffered stdio (std::ifstream/fread). hpcbench also runs entries in a browser
// tab, and the vendored memfs there has a broken fd_read that throws on large
// single reads. A bounded loop never asks for more than remains.
inline bool read_exact_fd(int fd, void* dst, std::size_t n) {
  unsigned char* p = static_cast<unsigned char*>(dst);
  const std::size_t chunk = std::size_t(1) << 20;
  while (n > 0) {
    const std::size_t want = n < chunk ? n : chunk;
    const ssize_t got = ::read(fd, p, want);
    if (got <= 0) return false;
    p += got;
    n -= static_cast<std::size_t>(got);
  }
  return true;
}

/** Reads the raw corpus: two u64 dimensions, then n_cells * n_genes doubles,
 *  row-major. Byte-for-byte the Project 1 input file. */
inline bool read_raw_input(const char* path, std::uint64_t& n_cells, std::uint64_t& n_genes,
                           std::vector<double>& X) {
  const int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return false;
  }
  n_cells = 0;
  n_genes = 0;
  if (!read_exact_fd(fd, &n_cells, 8) || !read_exact_fd(fd, &n_genes, 8)) {
    std::fprintf(stderr, "truncated input\n");
    ::close(fd);
    return false;
  }
  X.assign(n_cells * n_genes, 0.0);
  if (!read_exact_fd(fd, X.data(), X.size() * sizeof(double))) {
    std::fprintf(stderr, "truncated input\n");
    ::close(fd);
    return false;
  }
  ::close(fd);
  return true;
}

/** The input transform: every raw count x becomes log1p(x). Zeros stay zero,
 *  so the matrix is exactly as sparse as the raw counts (6.17% non-zero). */
inline void transform_input(std::vector<double>& X) {
  for (double& x : X) x = std::log1p(x);
}

/** splitmix64 (Steele, Lea, Flood 2014), the textbook form: add the golden
 *  gamma, then mix. The same generator hpcbench uses everywhere. */
struct SplitMix64 {
  std::uint64_t state;
  explicit SplitMix64(std::uint64_t s) : state(s) {}
  std::uint64_t next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  /** Uniform on [0, 1) with 53 random bits. */
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
};

/** Independent streams for the weights and the batch order, from one seed. */
inline std::uint64_t weight_stream(std::uint64_t seed) { return seed ^ 0x5157A8D1C0DE0002ull; }
inline std::uint64_t order_stream(std::uint64_t seed) { return seed ^ 0x0BDE5EED0677A11Full; }

/** The initial parameters. Every weight matrix is row-major [fan_in][fan_out],
 *  so layer 1 is exactly Project 1's product: A1 = X * W1 with W1 n_genes x 256.
 *  Weights are Glorot-uniform on [-a, a), a = sqrt(6 / (fan_in + fan_out)),
 *  drawn in the order W1, W2, W3, W4 from one stream, row-major within each.
 *  Every bias starts at zero. */
struct InitialParams {
  std::vector<double> W1, b1;  // n_genes x 256, 256
  std::vector<double> W2, b2;  // 256 x 32,      32
  std::vector<double> W3, b3;  // 32 x 256,      256
  std::vector<double> W4, b4;  // 256 x n_genes, n_genes
};

inline void fill_glorot(SplitMix64& rng, std::vector<double>& W, std::uint64_t fan_in,
                        std::uint64_t fan_out) {
  const double a = std::sqrt(6.0 / double(fan_in + fan_out));
  W.resize(fan_in * fan_out);
  for (double& w : W) w = (2.0 * rng.uniform() - 1.0) * a;
}

inline InitialParams init_params(std::uint64_t seed, std::uint64_t n_genes) {
  InitialParams P;
  SplitMix64 rng(weight_stream(seed));
  fill_glorot(rng, P.W1, n_genes, kHidden);
  fill_glorot(rng, P.W2, kHidden, kLatent);
  fill_glorot(rng, P.W3, kLatent, kHidden);
  fill_glorot(rng, P.W4, kHidden, n_genes);
  P.b1.assign(kHidden, 0.0);
  P.b2.assign(kLatent, 0.0);
  P.b3.assign(kHidden, 0.0);
  P.b4.assign(n_genes, 0.0);
  return P;
}

/** The order cells are visited in: a Fisher-Yates shuffle of 0..n_cells-1,
 *  for i = n-1 down to 1, j = next() % (i + 1), swap(order[i], order[j]).
 *  Batch b is order[b*256 .. min((b+1)*256, n_cells) - 1]; the last batch is
 *  short (2700 = 10 * 256 + 140). */
inline std::vector<std::uint32_t> batch_order(std::uint64_t seed, std::uint64_t n_cells) {
  std::vector<std::uint32_t> order(n_cells);
  for (std::uint64_t i = 0; i < n_cells; ++i) order[i] = static_cast<std::uint32_t>(i);
  SplitMix64 rng(order_stream(seed));
  for (std::uint64_t i = n_cells; i-- > 1;) {
    const std::uint64_t j = rng.next() % (i + 1);
    const std::uint32_t t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  return order;
}

// ---------------------------------------------------------------------------
// The score's correctness gate: the loss the trained model reaches.
// ---------------------------------------------------------------------------

/** The corpus in compressed sparse rows, as the HARNESS keeps it for judging.
 *  `prepare` mode writes it into the workdir after your prepare has run; `solve`
 *  mode reads it before the clock starts. It is the host's copy, not yours:
 *  your own representation is whatever your prepare wrote. */
struct CsrMatrix {
  std::uint64_t n_rows = 0, n_cols = 0;
  std::vector<std::uint64_t> row_ptr;  // n_rows + 1
  std::vector<std::uint32_t> col;      // nnz
  std::vector<double> val;             // nnz
};

inline CsrMatrix dense_to_csr(const double* X, std::uint64_t n_rows, std::uint64_t n_cols) {
  CsrMatrix M;
  M.n_rows = n_rows;
  M.n_cols = n_cols;
  M.row_ptr.assign(n_rows + 1, 0);
  for (std::uint64_t i = 0; i < n_rows; ++i) {
    const double* row = X + i * n_cols;
    for (std::uint64_t j = 0; j < n_cols; ++j)
      if (row[j] != 0.0) {
        M.col.push_back(static_cast<std::uint32_t>(j));
        M.val.push_back(row[j]);
      }
    M.row_ptr[i + 1] = M.col.size();
  }
  return M;
}

/** Four independent partial sums in a fixed order: the same answer from every
 *  compiler that does not reassociate, and fast enough without vectorising. */
inline double dot_fixed(const double* a, const double* b, std::uint64_t n) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  std::uint64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

/** THE NUMBER THAT IS CHECKED. The mean squared reconstruction error of the
 *  whole corpus (every cell, every gene) under a set of parameters:
 *
 *      MSE = sum_{i,j} (Xh_ij - X_ij)^2 / (n_cells * n_genes)
 *      Xh  = relu(relu(X W1 + b1) W2 + b2 ... )  -- the model in task.hpp
 *
 *  The harness calls this AFTER your solve returns, on the parameters you wrote
 *  into `Result`, outside the timed region. It never looks at how you trained.
 *
 *  It is exact algebra, arranged so it is cheap enough to run after every
 *  execution: for each cell with decoder input h (256 wide),
 *      sum_j Xh_j^2   = h' G h + 2 h'c + |b4|^2,   G = W4 W4', c = W4 b4
 *      sum_j Xh_j X_j = over the cell's non-zeros only
 *      sum_j X_j^2    = over the cell's non-zeros only
 *  which is about 1.7 G multiply-adds instead of the 9.5 G of forming Xh. */
inline double reconstruction_mse(const CsrMatrix& X, const double* W1, const double* b1,
                                 const double* W2, const double* b2, const double* W3,
                                 const double* b3, const double* W4, const double* b4) {
  const std::uint64_t N = X.n_rows, D = X.n_cols, H = kHidden, K = kLatent;
  std::vector<double> G(H * H), c(H);
  for (std::uint64_t a = 0; a < H; ++a) {
    c[a] = dot_fixed(W4 + a * D, b4, D);
    for (std::uint64_t b = a; b < H; ++b) G[a * H + b] = G[b * H + a] = dot_fixed(W4 + a * D, W4 + b * D, D);
  }
  const double bb = dot_fixed(b4, b4, D);

  std::vector<double> a1(H), z(K), h3(H), Gh(H), acc;
  double total = 0.0;
  for (std::uint64_t i = 0; i < N; ++i) {
    const std::uint64_t lo = X.row_ptr[i], hi = X.row_ptr[i + 1], nz = hi - lo;
    // encoder
    for (std::uint64_t k = 0; k < H; ++k) a1[k] = b1[k];
    for (std::uint64_t t = lo; t < hi; ++t) {
      const double x = X.val[t];
      const double* w = W1 + std::uint64_t(X.col[t]) * H;
      for (std::uint64_t k = 0; k < H; ++k) a1[k] += x * w[k];
    }
    for (std::uint64_t k = 0; k < H; ++k) a1[k] = a1[k] > 0.0 ? a1[k] : 0.0;
    for (std::uint64_t q = 0; q < K; ++q) z[q] = b2[q];
    for (std::uint64_t k = 0; k < H; ++k) {
      const double h = a1[k];
      if (h == 0.0) continue;
      for (std::uint64_t q = 0; q < K; ++q) z[q] += h * W2[k * K + q];
    }
    // decoder input
    for (std::uint64_t k = 0; k < H; ++k) h3[k] = b3[k];
    for (std::uint64_t q = 0; q < K; ++q) {
      const double v = z[q];
      for (std::uint64_t k = 0; k < H; ++k) h3[k] += v * W3[q * H + k];
    }
    for (std::uint64_t k = 0; k < H; ++k) h3[k] = h3[k] > 0.0 ? h3[k] : 0.0;
    // sum_j Xh_j^2
    for (std::uint64_t a = 0; a < H; ++a) Gh[a] = dot_fixed(G.data() + a * H, h3.data(), H);
    const double xhxh = dot_fixed(h3.data(), Gh.data(), H) + 2.0 * dot_fixed(h3.data(), c.data(), H) + bb;
    // sum_j Xh_j X_j and sum_j X_j^2, over this cell's non-zeros
    acc.assign(nz, 0.0);
    for (std::uint64_t t = 0; t < nz; ++t) acc[t] = b4[X.col[lo + t]];
    for (std::uint64_t k = 0; k < H; ++k) {
      const double h = h3[k];
      if (h == 0.0) continue;
      const double* w = W4 + k * D;
      for (std::uint64_t t = 0; t < nz; ++t) acc[t] += h * w[X.col[lo + t]];
    }
    double xhx = 0.0, xx = 0.0;
    for (std::uint64_t t = 0; t < nz; ++t) {
      const double x = X.val[lo + t];
      xhx += acc[t] * x;
      xx += x * x;
    }
    total += xhxh - 2.0 * xhx + xx;
  }
  return total / (double(N) * double(D));
}

}  // namespace hpcbench

#endif  // HPCBENCH_TASK_IO_HPP
