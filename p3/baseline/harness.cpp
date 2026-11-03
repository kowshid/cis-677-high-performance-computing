// ============================================================================
// hpcbench fixed harness -- CIS 677 P3, "The Loader". HOST-OWNED. DO NOT EDIT:
// every entry is compiled and linked against this file exactly as written.
//
//     ./build/task solve <workdir> <result.bin> <seed>     TIMED (solve only)
//
// The corpus lives on Clipper. Its path is HPCBENCH_P3_CORPUS if set, else the
// class copy below. Before the clock starts the harness opens the corpus,
// generates the initial weights and the batch order from the seed, allocates
// zeroed space for the answer, and asks the kernel to drop this file from the
// node's page cache, so every run reads the corpus cold.
//
// The timed region is announced on stdout around the call:
//     HPCBENCH_SOLVE_BEGIN / HPCBENCH_SOLVE_END / HPCBENCH_TIME_MS <ms>
//
// After the clock stops (untimed): the memory rule is checked (peak resident
// memory below a quarter of the corpus file), the parameters must be finite,
// and the full-data reconstruction MSE is computed in double precision. That
// one number is the answer written to result.bin.
// ============================================================================
#include <fcntl.h>
#include <omp.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "canon.hpp"
#include "task.hpp"

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;

const char* kDefaultCorpus = "/mnt/projects/debruinz_project/debruinz/p3corpus/out/corpus-100k.csr";

// Platform-independent uniform in [0,1): the top 53 bits of mt19937_64.
double unit(std::mt19937_64& g) { return double(g() >> 11) * (1.0 / 9007199254740992.0); }

std::vector<float> glorot(std::mt19937_64& g, u64 fin, u64 fout) {
  std::vector<float> w(fin * fout);
  const double a = std::sqrt(6.0 / double(fin + fout));
  for (auto& x : w) x = float((2.0 * unit(g) - 1.0) * a);
  return w;
}

// Fisher-Yates with an explicit index draw, so every platform agrees.
std::vector<u32> batch_order(u64 seed, u64 n) {
  std::vector<u32> o(n);
  for (u64 i = 0; i < n; ++i) o[i] = u32(i);
  std::mt19937_64 g(seed * 0x9E3779B97F4A7C15ull + 0x5EEDull);
  for (u64 i = n - 1; i > 0; --i) {
    const u64 j = g() % (i + 1);
    std::swap(o[i], o[j]);
  }
  return o;
}

bool finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

// Full-data MSE of the trained parameters, every cell and every gene, in double.
// ||xh - x||^2 = ||x||^2 - 2 (h.p + x.b4) + h^T G h + 2 h.u + ||b4||^2, with
// p = sum_g x_g W4T[g], G = W4 W4^T, u = W4 b4.
double reconstruction_mse(const hpcbench::Corpus& c, const std::vector<float>& W1, const std::vector<float>& b1,
                          const std::vector<float>& W2, const std::vector<float>& b2, const std::vector<float>& W3,
                          const std::vector<float>& b3, const std::vector<float>& W4T, const std::vector<float>& b4) {
  const u64 D = c.n_genes, H = hpcbench::kHidden, K = hpcbench::kLatent;
  std::vector<double> G(H * H, 0.0), u(H, 0.0);
  double bb = 0.0;
  for (u64 g = 0; g < D; ++g) {
    const float* w = &W4T[g * H];
    for (u64 k = 0; k < H; ++k) {
      u[k] += double(b4[g]) * w[k];
      for (u64 l = 0; l < H; ++l) G[k * H + l] += double(w[k]) * w[l];
    }
    bb += double(b4[g]) * b4[g];
  }
  std::vector<u64> rp(c.n_cells + 1);
  c.read_at(rp.data(), 8 * (c.n_cells + 1), c.off_indptr);
  const u64 CH = 8192;
  double total = 0.0;
  std::vector<u32> idx;
  std::vector<float> val;
  for (u64 s = 0; s < c.n_cells; s += CH) {
    const u64 e = std::min(c.n_cells, s + CH), a = rp[s], m = rp[e] - a;
    idx.resize(m); val.resize(m);
    c.read_at(idx.data(), 4 * m, c.off_indices + 4 * a);
    c.read_at(val.data(), 4 * m, c.off_data + 4 * a);
    double part = 0.0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : part)
    for (u64 i = s; i < e; ++i) {
      double h1[64], z[64], h3[64], p[64];
      for (u64 k = 0; k < H; ++k) { h1[k] = b1[k]; p[k] = 0.0; }
      double xx = 0.0, xb = 0.0;
      for (u64 q = rp[i] - a; q < rp[i + 1] - a; ++q) {
        const double x = std::log1p(double(val[q]));
        const u64 g = idx[q];
        for (u64 k = 0; k < H; ++k) { h1[k] += x * W1[g * H + k]; p[k] += x * W4T[g * H + k]; }
        xx += x * x;
        xb += x * b4[g];
      }
      for (u64 k = 0; k < H; ++k) h1[k] = h1[k] > 0.0 ? h1[k] : 0.0;
      for (u64 j = 0; j < K; ++j) {
        double t = b2[j];
        for (u64 k = 0; k < H; ++k) t += h1[k] * W2[k * K + j];
        z[j] = t;
      }
      for (u64 k = 0; k < H; ++k) {
        double t = b3[k];
        for (u64 j = 0; j < K; ++j) t += z[j] * W3[j * H + k];
        h3[k] = t > 0.0 ? t : 0.0;
      }
      double hGh = 0.0, hu = 0.0, hp = 0.0;
      for (u64 k = 0; k < H; ++k) {
        double t = 0.0;
        for (u64 l = 0; l < H; ++l) t += G[k * H + l] * h3[l];
        hGh += h3[k] * t;
        hu += h3[k] * u[k];
        hp += h3[k] * p[k];
      }
      part += xx - 2.0 * (hp + xb) + hGh + 2.0 * hu + bb;
    }
    total += part;
  }
  return total / (double(c.n_cells) * double(D));
}

int usage() {
  std::fprintf(stderr, "usage: task solve <workdir> <result.bin> <seed>\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || std::string(argv[1]) != "solve") return usage();
  const std::string result_path = argv[3];
  char* end = nullptr;
  const u64 seed = std::strtoull(argv[4], &end, 10);
  if (end == argv[4] || *end != '\0') hpcbench::fatal("seed must be a non-negative integer");

  hpcbench::Corpus c;
  const char* env = std::getenv("HPCBENCH_P3_CORPUS");
  c.path = env && *env ? env : kDefaultCorpus;
  c.fd = ::open(c.path.c_str(), O_RDONLY);
  if (c.fd < 0)
    hpcbench::fatal("cannot open the corpus at " + c.path +
                    ". Project 3 runs on Clipper, where the corpus lives (or set HPCBENCH_P3_CORPUS).");
  struct stat st {};
  fstat(c.fd, &st);
  c.bytes = u64(st.st_size);
  u64 h[3];
  c.read_at(h, 24, 0);
  c.n_cells = h[0]; c.n_genes = h[1]; c.nnz = h[2];
  c.off_indptr = 24;
  c.off_indices = 24 + 8 * (c.n_cells + 1);
  c.off_data = c.off_indices + 4 * c.nnz;
  if (c.off_data + 4 * c.nnz != c.bytes) hpcbench::fatal("the corpus file is damaged: " + c.path);

  const u64 D = c.n_genes, H = hpcbench::kHidden, K = hpcbench::kLatent;
  std::mt19937_64 g(seed);
  const std::vector<float> W1i = glorot(g, D, H), W2i = glorot(g, H, K), W3i = glorot(g, K, H), W4i = glorot(g, D, H);
  const std::vector<float> zH(H, 0.f), zK(K, 0.f), zD(D, 0.f);
  const std::vector<u32> order = batch_order(seed, c.n_cells);
  std::vector<float> W1(D * H, 0.f), b1(H, 0.f), W2(H * K, 0.f), b2(K, 0.f), W3(K * H, 0.f), b3(H, 0.f),
      W4T(D * H, 0.f), b4(D, 0.f);

  hpcbench::Params p;
  p.seed = seed; p.n_cells = c.n_cells; p.n_genes = D;
  p.order = order.data();
  p.W1 = W1i.data(); p.b1 = zH.data(); p.W2 = W2i.data(); p.b2 = zK.data();
  p.W3 = W3i.data(); p.b3 = zH.data(); p.W4T = W4i.data(); p.b4 = zD.data();
  hpcbench::Result out;
  out.W1 = W1.data(); out.b1 = b1.data(); out.W2 = W2.data(); out.b2 = b2.data();
  out.W3 = W3.data(); out.b3 = b3.data(); out.W4T = W4T.data(); out.b4 = b4.data();

  // Cold start: drop this file from the node's page cache.
  posix_fadvise(c.fd, 0, 0, POSIX_FADV_DONTNEED);

  std::printf("HPCBENCH_SOLVE_BEGIN\n");
  std::fflush(stdout);
  const auto t0 = std::chrono::steady_clock::now();
  hpcbench::solve(c, p, out);
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("HPCBENCH_SOLVE_END\n");
  std::printf("HPCBENCH_TIME_MS %.6f\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
  std::fflush(stdout);

  // Untimed checks.
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
  const double peak_mb = double(ru.ru_maxrss) / 1024.0, budget_mb = double(c.bytes) / 4.0 / 1048576.0;
  std::fprintf(stderr, "memory: peak resident %.1f MB, budget %.1f MB (a quarter of the corpus)\n", peak_mb, budget_mb);
  if (peak_mb > budget_mb)
    hpcbench::fatal("peak resident memory exceeded a quarter of the corpus; Project 3 must stream the data");
  for (const std::vector<float>* v : {&W1, &b1, &W2, &b2, &W3, &b3, &W4T, &b4})
    if (!finite(*v)) hpcbench::fatal("the trained parameters contain NaN or Inf");
  const double mse = reconstruction_mse(c, W1, b1, W2, b2, W3, b3, W4T, b4);
  std::fprintf(stderr, "loss: full-data reconstruction MSE after the epoch = %.12g\n", mse);
  const std::vector<double> answer{mse};
  hpcbench::write_result(result_path, answer, {1});
  return 0;
}
