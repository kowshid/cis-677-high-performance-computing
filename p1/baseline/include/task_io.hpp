// task_io.hpp — HOST-OWNED. Read-only for competitors.
//
// This header holds the two pieces of the task that are the host's definition of
// the problem rather than anyone's optimisation:
//
//   hpcbench::gen_W          the weights every entry must multiply by
//   hpcbench::read_raw_input reading the raw corpus in prepare
//
// Both bodies are lifted BYTE FOR BYTE from the baseline they replace, and the
// refactored baseline is gated on producing bit-identical results at every
// published sweep value. Nothing about the answer changed.
//
// Why it is here instead of copied into every entry: while `gen_W` lives in each
// entry's own solution.cpp, it is a pure function of `k`, and a pure function of
// a small published set of `k` can be precomputed into a lookup table. The fix
// is to make W depend on a value that does not exist when `prepare` runs, and
// that is only possible if there is ONE `gen_W` that the host controls.
//
// ANTICIPATED CHANGE — the session nonce. `gen_W` already takes a trailing
// `session_nonce` with a default of 0, and today it is RESERVED: nothing reads
// it, so the weights are exactly the historical ones. When the host turns the nonce on, the runner will pass the session's
// value and nothing at any call site has to change. Do not add parameters before
// that one, and do not change what `session_nonce == 0` computes: every stored
// reference digest depends on it.
//
// The runners stage this file LAST, after an entry's own sources, so an entry
// that ships a file at this path cannot replace it.
#ifndef HPCBENCH_TASK_IO_HPP
#define HPCBENCH_TASK_IO_HPP

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hpcbench {

// A bounded open()/read() loop. REGRESSION GUARD -- do NOT replace this with
// buffered stdio (std::ifstream/fread). hpcbench also runs entries in a browser
// tab, and the vendored memfs there (public/wasm-clang/shared.js) has a broken
// fd_read: its copy_out helper builds a typed-array view over the memfs
// WebAssembly.Memory using the caller's full requested length, past the end of
// the file region, and throws "Invalid typed array length: <file size>".
// Buffered stdio asks for large readahead chunks and trips it on this ~296 MB
// input.
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
 *  row-major. Returns false and leaves a message on stderr on any failure. */
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

/** The weights, identical for every entry. Body lifted byte for byte from the
 *  baseline's `gen_W`. `session_nonce == 0` is the historical definition and is
 *  what every stored reference digest was produced with; see the note above. */
inline std::vector<double> gen_W(std::uint64_t n_genes, std::uint64_t k,
                                 std::uint64_t session_nonce = 0) {
  std::vector<double> W(n_genes * k);
  std::uint64_t s = 0x9E3779B97F4A7C15ull ^ (k * 0xBF58476D1CE4E5B9ull);
  // The nonce is RESERVED and deliberately unused: turning it on changes the
  // answer for every existing entry, so it is the host's call and his timing.
  // When he makes it, this is the one line that changes:
  //     if (session_nonce != 0) s ^= session_nonce * 0x94D049BB133111EBull;
  (void)session_nonce;
  for (std::uint64_t i = 0; i < W.size(); ++i) {
    s ^= s >> 30; s *= 0xBF58476D1CE4E5B9ull;
    s ^= s >> 27; s *= 0x94D049BB133111EBull;
    s ^= s >> 31;
    W[i] = (double(s >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
  }
  return W;
}

}  // namespace hpcbench

#endif  // HPCBENCH_TASK_IO_HPP
