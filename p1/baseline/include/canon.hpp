// ============================================================================
// hpcbench canonical result writer.
//
// Emit your answer with write_result(). The grader hashes the file after
// quantizing to the task's declared significant digits, so a vectorized or
// threaded reduction that reorders floating-point additions still produces the
// same checksum as the scalar reference -- provided it agrees to that
// precision. Bit-exactness is NOT required (and would be unachievable).
//
// Usage:
//     #include "canon.hpp"
//     std::vector<double> out = ...;
//     hpcbench::write_result(argv[2], out, {n_rows, n_cols});
//
// DO NOT MODIFY. The grader has its own copy.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpcbench {

constexpr std::uint32_t kCanonVersion = 1;

inline void write_result(const std::string& path,
                         const double* data,
                         std::uint64_t n,
                         std::initializer_list<std::uint64_t> dims = {}) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open result file: " + path);

  std::fwrite("HPCBENCH", 1, 8, f);
  std::uint32_t version = kCanonVersion;
  std::uint32_t ndim = static_cast<std::uint32_t>(dims.size());
  std::fwrite(&version, sizeof(version), 1, f);
  std::fwrite(&ndim, sizeof(ndim), 1, f);
  for (std::uint64_t d : dims) std::fwrite(&d, sizeof(d), 1, f);
  std::fwrite(&n, sizeof(n), 1, f);
  std::fwrite(data, sizeof(double), n, f);

  if (std::fclose(f) != 0)
    throw std::runtime_error("failed to close result file: " + path);
}

inline void write_result(const std::string& path,
                         const std::vector<double>& v,
                         std::initializer_list<std::uint64_t> dims = {}) {
  write_result(path, v.data(), static_cast<std::uint64_t>(v.size()), dims);
}

}  // namespace hpcbench
