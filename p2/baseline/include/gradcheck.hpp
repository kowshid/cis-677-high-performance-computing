// gradcheck.hpp -- HOST-SHIPPED helper. Optional; include it if you want it.
//
// Finite-difference gradient checking: the oracle for your backward pass, and
// it costs one run. Use it BEFORE you optimise anything, and again after every
// change to the backward pass.
//
// For a handful of weights w_i it compares your analytic gradient g_i with
//
//     (L(w_i + h) - L(w_i - h)) / 2h
//
// where L is the loss of ONE batch recomputed by a forward pass. It samples
// `pool` random indices and checks the `samples` with the largest |g_i|, since
// a gradient that is ~0 tells you nothing about a sign or a factor of two.
//
//   #include "gradcheck.hpp"
//   hpcbench::gradcheck("W4", n_W4, grad_W4,
//       [&](std::size_t i) { return W4[i]; },          // read weight i
//       [&](std::size_t i, double v) { W4[i] = v; },   // write weight i
//       [&]() { return batch_loss(...); });            // forward only
//
// The shipped baseline calls this for all eight parameter arrays when it is
// built with -DHPCBENCH_GRADCHECK, e.g.
//
//   CXXFLAGS="-O2 -std=c++17 -DHPCBENCH_GRADCHECK" sh ./build.sh
//   ./build/task prepare input.bin work && ./build/task solve work r.bin 677
//
// and prints one line per checked weight to stderr before training. On the
// shipped baseline all 32 checks agree to better than 1e-5 relative; anything
// above ~1e-4 is a bug (a wrong index, a missing factor of 2/(B*D), a ReLU mask
// on the wrong side). A step that straddles a ReLU kink can occasionally
// disagree; if exactly one weight fails, try -DHPCBENCH_GRADCHECK_H=3e-6.
//
// It never runs inside a scored measurement unless you compile it in, and it
// is far too slow to leave on: every checked weight costs two forward passes.
#ifndef HPCBENCH_GRADCHECK_HPP
#define HPCBENCH_GRADCHECK_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "task_io.hpp"

// Step size. 1e-5 is small enough that a step rarely crosses a ReLU kink and
// large enough that rounding in the loss stays well below the tolerance.
#ifndef HPCBENCH_GRADCHECK_H
#define HPCBENCH_GRADCHECK_H 1e-5
#endif

namespace hpcbench {

struct GradCheckReport {
  std::size_t checked = 0;
  std::size_t failed = 0;
  double max_rel_err = 0.0;
};

template <class Get, class Set, class Loss>
inline GradCheckReport gradcheck(const char* name, std::size_t n, const double* analytic, Get get,
                                 Set set, Loss loss, std::size_t samples = 4,
                                 double h = HPCBENCH_GRADCHECK_H,
                                 double tol = 1e-4, std::size_t pool = 256,
                                 std::uint64_t seed = 12345) {
  GradCheckReport rep;
  if (n == 0) return rep;
  SplitMix64 rng(seed ^ n);
  std::vector<std::size_t> idx;
  for (std::size_t t = 0; t < pool && t < n; ++t) idx.push_back(std::size_t(rng.next() % n));
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    return std::fabs(analytic[a]) > std::fabs(analytic[b]);
  });
  idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
  if (idx.size() > samples) idx.resize(samples);

  for (std::size_t i : idx) {
    const double w0 = get(i);
    set(i, w0 + h);
    const double lp = loss();
    set(i, w0 - h);
    const double lm = loss();
    set(i, w0);
    const double fd = (lp - lm) / (2.0 * h);
    const double an = analytic[i];
    const double scale = std::max(std::fabs(fd), std::fabs(an));
    const double rel = scale > 0 ? std::fabs(fd - an) / scale : 0.0;
    const bool bad = !(rel <= tol);  // NaN fails
    rep.checked++;
    if (bad) rep.failed++;
    rep.max_rel_err = std::max(rep.max_rel_err, rel);
    std::fprintf(stderr, "gradcheck %-3s [%8zu]  analytic % .9e  numeric % .9e  rel %.2e%s\n",
                 name, i, an, fd, rel, bad ? "  <-- MISMATCH" : "");
  }
  return rep;
}

}  // namespace hpcbench

#endif  // HPCBENCH_GRADCHECK_HPP
