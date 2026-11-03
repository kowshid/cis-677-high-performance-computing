// ============================================================================
// hpcbench fixed harness -- CIS 677 P2, "The Epoch". HOST-OWNED. DO NOT EDIT:
// every entry is compiled and linked against this file exactly as written.
//
// One binary, two modes:
//
//     ./build/task prepare <raw_input> <workdir>         UNTIMED
//     ./build/task solve   <workdir> <result.bin> <seed> TIMED (solve only)
//
// `prepare` reads the raw corpus (the same file as Project 1: u64 n_cells,
// u64 n_genes, then n_cells*n_genes float64 counts, row-major), applies the
// input transform x -> log1p(x), and calls the entry's `hpcbench::prepare`,
// which writes whatever representation it wants into the workdir. AFTER that,
// the harness writes two host-owned files of its own into the workdir:
//
//     .hpcbench-dims   two little-endian u64: n_cells, n_genes
//     .hpcbench-csr    the transformed corpus as CSR (~27 MB), for judging
//
// `solve` reads those two files, generates the initial weights and the batch
// order from the seed, and allocates and zero-fills space for the TRAINED
// parameters. Then it starts a clock, calls the entry's `hpcbench::solve`, and
// stops it. Reading the entry's own representation happens inside `solve` and
// is therefore on the clock.
//
// After the clock stops, the harness evaluates the parameters the entry wrote:
// the mean squared reconstruction error of the whole corpus (every cell, every
// gene), computed by hpcbench::reconstruction_mse in task_io.hpp. That single
// number is the answer written to result.bin and checked against the
// reference. The entry never reports its own loss.
//
// Nothing outside `solve` is charged to anyone: process start-up, reading the
// host's files, generating the weights and the order, allocating the output,
// evaluating the loss, writing result.bin and digesting it.
//
// The timed region is announced on stdout, always, around the call:
//
//     HPCBENCH_SOLVE_BEGIN
//     HPCBENCH_SOLVE_END
//     HPCBENCH_TIME_MS <milliseconds>
//
// The last line is printed only where the platform has a monotonic clock. The
// browser's wasm sysroot does not implement clock_time_get, so the in-tab runner
// timestamps the two markers instead. The loss is printed on stderr.
// ============================================================================
#define HPCBENCH_HARNESS_TU 1

#include "canon.hpp"
#include "task.hpp"

#if defined(__wasm__) || defined(__EMSCRIPTEN__)
// No clock_time_get in this sysroot: calling steady_clock::now() traps.
#define HPCBENCH_NO_STEADY_CLOCK 1
#else
#include <chrono>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kDimsFile = "/.hpcbench-dims";
const char* kCsrFile = "/.hpcbench-csr";

void write_dims(const std::string& workdir, std::uint64_t n_cells, std::uint64_t n_genes) {
  hpcbench::Writer w(workdir + kDimsFile);
  w.write_value(n_cells);
  w.write_value(n_genes);
  w.close();
}

void read_dims(const std::string& workdir, std::uint64_t& n_cells, std::uint64_t& n_genes) {
  hpcbench::Reader r(workdir + kDimsFile);
  n_cells = r.read_value<std::uint64_t>();
  n_genes = r.read_value<std::uint64_t>();
  if (n_cells == 0 || n_genes == 0) hpcbench::fatal("workdir dimensions are empty; re-run prepare");
}

void write_csr(const std::string& workdir, const hpcbench::CsrMatrix& M) {
  hpcbench::Writer w(workdir + kCsrFile);
  const std::uint64_t nnz = M.col.size();
  w.write_value(M.n_rows);
  w.write_value(M.n_cols);
  w.write_value(nnz);
  w.write(M.row_ptr.data(), M.row_ptr.size() * sizeof(std::uint64_t));
  w.write(M.col.data(), nnz * sizeof(std::uint32_t));
  w.write(M.val.data(), nnz * sizeof(double));
  w.close();
}

hpcbench::CsrMatrix read_csr(const std::string& workdir, std::uint64_t n_cells, std::uint64_t n_genes) {
  hpcbench::Reader r(workdir + kCsrFile);
  hpcbench::CsrMatrix M;
  M.n_rows = r.read_value<std::uint64_t>();
  M.n_cols = r.read_value<std::uint64_t>();
  const std::uint64_t nnz = r.read_value<std::uint64_t>();
  if (M.n_rows != n_cells || M.n_cols != n_genes || nnz > n_cells * n_genes)
    hpcbench::fatal("the host's copy of the corpus in the workdir is damaged; re-run prepare");
  M.row_ptr.resize(n_cells + 1);
  M.col.resize(nnz);
  M.val.resize(nnz);
  r.read(M.row_ptr.data(), M.row_ptr.size() * sizeof(std::uint64_t));
  r.read(M.col.data(), nnz * sizeof(std::uint32_t));
  r.read(M.val.data(), nnz * sizeof(double));
  if (M.row_ptr[0] != 0 || M.row_ptr[n_cells] != nnz)
    hpcbench::fatal("the host's copy of the corpus in the workdir is damaged; re-run prepare");
  return M;
}

bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

int usage() {
  std::fprintf(stderr,
               "usage: task prepare <raw_input> <workdir>\n"
               "       task solve   <workdir> <result.bin> <seed>\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string mode = argv[1];

  if (mode == "prepare") {
    if (argc < 4) return usage();
    const std::string workdir = argv[3];

    std::uint64_t n_cells = 0, n_genes = 0;
    std::vector<double> X;
    if (!hpcbench::read_raw_input(argv[2], n_cells, n_genes, X)) return 1;
    hpcbench::transform_input(X);

    hpcbench::Input in{n_cells, n_genes, X.data()};
    hpcbench::Sink sink{workdir};
    hpcbench::prepare(in, sink);

    // The host's own files are written LAST, so nothing the entry's prepare
    // wrote can stand in for them.
    write_dims(workdir, n_cells, n_genes);
    write_csr(workdir, hpcbench::dense_to_csr(X.data(), n_cells, n_genes));

    std::fprintf(stderr, "prepare: %llu x %llu, log1p applied\n", (unsigned long long)n_cells,
                 (unsigned long long)n_genes);
    return 0;
  }

  if (mode == "solve") {
    if (argc < 5) return usage();
    const std::string workdir = argv[2];
    const std::string result_path = argv[3];
    char* end = nullptr;
    const std::uint64_t seed = std::strtoull(argv[4], &end, 10);
    if (end == argv[4] || *end != '\0') hpcbench::fatal("seed must be a non-negative integer");

    std::uint64_t n_cells = 0, n_genes = 0;
    read_dims(workdir, n_cells, n_genes);
    const std::uint64_t H = hpcbench::kHidden, K = hpcbench::kLatent, D = n_genes;

    // Untimed: the host's copy of the corpus (read before the entry runs, so
    // nothing the entry does can change what it is judged on), the problem
    // definition, and zeroed space for the answer.
    const hpcbench::CsrMatrix corpus = read_csr(workdir, n_cells, n_genes);
    const hpcbench::InitialParams init = hpcbench::init_params(seed, n_genes);
    const std::vector<std::uint32_t> order = hpcbench::batch_order(seed, n_cells);
    std::vector<double> W1(D * H, 0.0), b1(H, 0.0), W2(H * K, 0.0), b2(K, 0.0), W3(K * H, 0.0),
        b3(H, 0.0), W4(H * D, 0.0), b4(D, 0.0);

    hpcbench::Work work{workdir};
    hpcbench::Params params{};
    params.seed = seed;
    params.n_cells = n_cells;
    params.n_genes = n_genes;
    params.hidden = H;
    params.latent = K;
    params.batch = hpcbench::kBatch;
    params.lr = hpcbench::kLearningRate;
    params.beta1 = hpcbench::kBeta1;
    params.beta2 = hpcbench::kBeta2;
    params.eps = hpcbench::kEps;
    params.order = order.data();
    params.W1 = init.W1.data();
    params.b1 = init.b1.data();
    params.W2 = init.W2.data();
    params.b2 = init.b2.data();
    params.W3 = init.W3.data();
    params.b3 = init.b3.data();
    params.W4 = init.W4.data();
    params.b4 = init.b4.data();
    hpcbench::Result out{};
    out.n_genes = n_genes;
    out.hidden = H;
    out.latent = K;
    out.W1 = W1.data();
    out.b1 = b1.data();
    out.W2 = W2.data();
    out.b2 = b2.data();
    out.W3 = W3.data();
    out.b3 = b3.data();
    out.W4 = W4.data();
    out.b4 = b4.data();

    // The markers bracket the timed region for a runner with no clock inside
    // the program. They are flushed so the timestamps are the real boundaries.
    std::printf("HPCBENCH_SOLVE_BEGIN\n");
    std::fflush(stdout);
#ifndef HPCBENCH_NO_STEADY_CLOCK
    const auto t0 = std::chrono::steady_clock::now();
#endif
    hpcbench::solve(work, params, out);
#ifndef HPCBENCH_NO_STEADY_CLOCK
    const auto t1 = std::chrono::steady_clock::now();
#endif
    std::printf("HPCBENCH_SOLVE_END\n");
    std::fflush(stdout);
#ifndef HPCBENCH_NO_STEADY_CLOCK
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("HPCBENCH_TIME_MS %.6f\n", ms);
    std::fflush(stdout);
#endif

    // Untimed: judge the trained parameters by the loss they reach on the
    // whole corpus, and write that one number as the answer.
    for (const std::vector<double>* v : {&W1, &b1, &W2, &b2, &W3, &b3, &W4, &b4})
      if (!all_finite(*v)) hpcbench::fatal("the trained parameters contain NaN or Inf");
    const double mse = hpcbench::reconstruction_mse(corpus, W1.data(), b1.data(), W2.data(),
                                                    b2.data(), W3.data(), b3.data(), W4.data(),
                                                    b4.data());
    std::fprintf(stderr, "loss: full-data reconstruction MSE after the epoch = %.12g\n", mse);
    const std::vector<double> answer{mse};
    hpcbench::write_result(result_path.c_str(), answer, {1});
    return 0;
  }

  return usage();
}
