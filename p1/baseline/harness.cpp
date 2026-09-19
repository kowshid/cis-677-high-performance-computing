// ============================================================================
// hpcbench fixed harness -- CIS 677 P1. HOST-OWNED. DO NOT EDIT: every entry is
// compiled and linked against this file exactly as written.
//
// One binary, two modes:
//
//     ./build/task prepare <raw_input> <workdir>      UNTIMED
//     ./build/task solve   <workdir> <result.bin> <k> TIMED
//
// `prepare` reads the raw corpus and calls the entry's `hpcbench::prepare`,
// which writes whatever representation it wants into the workdir.
//
// `solve` generates the weights, allocates and zero-fills the output, then
// starts a clock, calls the entry's `hpcbench::solve`, and stops it. Reading the
// entry's own representation happens inside `solve` and is therefore on the
// clock -- that is the point of the design. Everything the host does is outside
// it: process start-up, generating W, allocating Y, writing result.bin and
// digesting it are no longer charged to anyone.
//
// The timed region is announced on stdout, always, around the call:
//
//     HPCBENCH_SOLVE_BEGIN
//     HPCBENCH_SOLVE_END
//     HPCBENCH_TIME_MS <milliseconds>
//
// The last line is printed only where the platform has a monotonic clock. The
// browser's wasm sysroot does not implement clock_time_get, so a `steady_clock`
// call there traps; the in-tab runner instead timestamps the two markers as it
// receives them, which measures the same region from outside the program.
//
// DIMENSIONS. The workdir belongs to the entry and the host does not parse it,
// so `prepare` mode records the corpus dimensions in a host-owned sidecar,
// <workdir>/.hpcbench-dims (two little-endian u64: n_cells, n_genes). `solve`
// reads that, not the entry's files, to size W and Y. Reading it is untimed.
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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* kDimsFile = "/.hpcbench-dims";

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

int usage() {
  std::fprintf(stderr,
               "usage: task prepare <raw_input> <workdir>\n"
               "       task solve   <workdir> <result.bin> <k>\n");
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

    hpcbench::Input in{n_cells, n_genes, X.data()};
    hpcbench::Sink sink{workdir};
    hpcbench::prepare(in, sink);
    write_dims(workdir, n_cells, n_genes);

    std::fprintf(stderr, "prepare: %llu x %llu\n", (unsigned long long)n_cells,
                 (unsigned long long)n_genes);
    return 0;
  }

  if (mode == "solve") {
    if (argc < 5) return usage();
    const std::string workdir = argv[2];
    const std::string result_path = argv[3];
    const std::uint64_t k = std::strtoull(argv[4], nullptr, 10);
    if (k == 0) hpcbench::fatal("k must be a positive integer");

    std::uint64_t n_cells = 0, n_genes = 0;
    read_dims(workdir, n_cells, n_genes);

    // Untimed: the problem definition and the output buffer.
    const std::vector<double> W = hpcbench::gen_W(n_genes, k);
    std::vector<double> Y(n_cells * k, 0.0);

    hpcbench::Work work{workdir};
    hpcbench::Params params{k, W.data(), n_cells, n_genes};
    hpcbench::Result out{Y.data(), n_cells, k};

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

    // Untimed: writing and digesting the answer.
    hpcbench::write_result(result_path.c_str(), Y, {n_cells, k});
    return 0;
  }

  return usage();
}
