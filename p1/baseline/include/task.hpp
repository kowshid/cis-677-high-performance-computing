// task.hpp -- HOST-OWNED. Read-only for competitors.
//
// This header is the whole contract between the harness and an entry. An entry
// is two files, each containing exactly one function and nothing else:
//
//     // src/prepare.cpp    UNTIMED
//     void hpcbench::prepare(const Input& in, Sink& out);
//
//     // src/solution.cpp   TIMED, and only this
//     void hpcbench::solve(const Work& w, const Params& p, Result& y);
//
// Everything else -- main, argv, reading the raw corpus, generating W,
// allocating and zeroing the output, writing result.bin, computing the digest --
// lives in harness.cpp and is not the competitor's problem.
//
// WHAT IS DELIBERATELY STILL YOURS. The host owns the *mechanism* of reading the
// workdir; you own *when and how much* you read, because you call these helpers
// yourself from inside `solve`, inside the timed region. Streaming versus
// mapping versus slurping, lazy versus eager, one file versus shards, are real
// levers and they stay with you.
//
// WHAT prepare CANNOT SEE. `k` is not in `Input` and not in `Sink`, on purpose.
// The sweep value does not exist while your representation is being built, so it
// cannot be tabled.
//
// Failures here are fatal and say why, on stderr, with the path. Nothing in the
// harness throws: a browser tab runs these binaries against a libc++abi with no
// unwinder, where a throw becomes an unexplained abort.
#ifndef HPCBENCH_TASK_HPP
#define HPCBENCH_TASK_HPP

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// `read_exact_fd`, `read_raw_input` and `gen_W` are host-owned too and live next
// door; they are included here so an entry needs exactly one #include.
#include "task_io.hpp"

// mmap exists natively and does not exist in the wasm sysroot the in-tab runner
// uses, so `Work::map` is a real mapping on one platform and a read into memory
// on the other. Either way the bytes and the lifetime are the same.
#if !defined(__wasm__) && !defined(__EMSCRIPTEN__) && !defined(_WIN32)
#define HPCBENCH_HAVE_MMAP 1
#include <sys/mman.h>
#endif

namespace hpcbench {

[[noreturn]] inline void fatal(const std::string& what) {
  std::fprintf(stderr, "hpcbench: %s\n", what.c_str());
  std::fflush(stderr);
  std::exit(1);
}

// The write-side twin of `read_exact_fd`: bounded chunks, same memfs reason.
inline bool write_exact_fd(int fd, const void* src, std::size_t n) {
  const unsigned char* p = static_cast<const unsigned char*>(src);
  const std::size_t chunk = std::size_t(1) << 20;
  while (n > 0) {
    const std::size_t want = n < chunk ? n : chunk;
    const ssize_t put = ::write(fd, p, want);
    if (put <= 0) return false;
    p += put;
    n -= static_cast<std::size_t>(put);
  }
  return true;
}

// ---------------------------------------------------------------------------
// prepare side: untimed
// ---------------------------------------------------------------------------

/** One file being written into the workdir by `prepare`. Untimed. */
class Writer {
 public:
  explicit Writer(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) fatal("cannot create " + path_);
  }
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  Writer(Writer&& other) noexcept : fd_(other.fd_), path_(std::move(other.path_)) {
    other.fd_ = -1;
  }
  ~Writer() { close(); }

  /** Writes n bytes, or ends the process saying which file failed. */
  void write(const void* data, std::size_t bytes) {
    if (fd_ < 0) fatal("write after close on " + path_);
    if (bytes && !write_exact_fd(fd_, data, bytes)) fatal("short write to " + path_);
  }
  template <typename T>
  void write_value(const T& value) {
    write(&value, sizeof(T));
  }
  void close() {
    if (fd_ >= 0) {
      if (::close(fd_) != 0) {
        fd_ = -1;
        fatal("failed to close " + path_);
      }
      fd_ = -1;
    }
  }
  const std::string& path() const { return path_; }

 private:
  int fd_ = -1;
  std::string path_;
};

/** The workdir, as `prepare` sees it: a place to create files. Untimed. */
struct Sink {
  std::string dir;
  Writer create(const char* name) const { return Writer(dir + "/" + name); }
  const char* path() const { return dir.c_str(); }
};

/** The raw corpus, already read for you by the harness. Untimed, and `k` is
 *  deliberately absent. */
struct Input {
  std::uint64_t n_cells;
  std::uint64_t n_genes;
  const double* X;  // row-major n_cells x n_genes
};

// ---------------------------------------------------------------------------
// solve side: every call below happens INSIDE the timed region
// ---------------------------------------------------------------------------

/** A bounded streaming read of one file you wrote in prepare. */
class Reader {
 public:
  explicit Reader(std::string path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) fatal("cannot open " + path_);
    struct stat st{};
    size_ = ::fstat(fd_, &st) == 0 ? static_cast<std::uint64_t>(st.st_size) : 0;
  }
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  Reader(Reader&& other) noexcept : fd_(other.fd_), size_(other.size_), path_(std::move(other.path_)) {
    other.fd_ = -1;
  }
  ~Reader() { close(); }

  /** Reads exactly n bytes; a short read is fatal and names the file. */
  void read(void* dst, std::size_t bytes) {
    if (fd_ < 0) fatal("read after close on " + path_);
    if (bytes && !read_exact_fd(fd_, dst, bytes)) fatal("truncated " + path_);
  }
  /** Reads exactly n bytes, returning false instead of exiting at end of file. */
  bool try_read(void* dst, std::size_t bytes) {
    return fd_ >= 0 && (bytes == 0 || read_exact_fd(fd_, dst, bytes));
  }
  template <typename T>
  T read_value() {
    T value{};
    read(&value, sizeof(T));
    return value;
  }
  std::uint64_t size() const { return size_; }
  void close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }

 private:
  int fd_ = -1;
  std::uint64_t size_ = 0;
  std::string path_;
};

/** A whole file addressable as bytes: mmap where the platform has it, a read
 *  into memory where it does not. Owns whatever it took to get there. */
class Mapped {
 public:
  Mapped() = default;
  explicit Mapped(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fatal("cannot open " + path);
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      fatal("cannot size " + path);
    }
    bytes_ = static_cast<std::size_t>(st.st_size);
#ifdef HPCBENCH_HAVE_MMAP
    if (bytes_ > 0) {
      void* p = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
      if (p != MAP_FAILED) {
        mapped_ = p;
        data_ = static_cast<const std::uint8_t*>(p);
        ::close(fd);
        return;
      }
    }
#endif
    owned_.resize(bytes_);
    if (bytes_ && !read_exact_fd(fd, owned_.data(), bytes_)) {
      ::close(fd);
      fatal("truncated " + path);
    }
    ::close(fd);
    data_ = owned_.data();
  }
  Mapped(const Mapped&) = delete;
  Mapped& operator=(const Mapped&) = delete;
  Mapped(Mapped&& o) noexcept
      : data_(nullptr), bytes_(o.bytes_), owned_(std::move(o.owned_)), mapped_(o.mapped_) {
    data_ = o.mapped_ ? static_cast<const std::uint8_t*>(o.mapped_) : owned_.data();
    o.mapped_ = nullptr;
    o.data_ = nullptr;
    o.bytes_ = 0;
  }
  ~Mapped() {
#ifdef HPCBENCH_HAVE_MMAP
    if (mapped_) ::munmap(mapped_, bytes_);
#endif
  }
  const std::uint8_t* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }
  /** The mapping reinterpreted, for a file you wrote as fixed-width records. */
  template <typename T>
  const T* as(std::size_t byte_offset = 0) const {
    return reinterpret_cast<const T*>(data_ + byte_offset);
  }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t bytes_ = 0;
  std::vector<std::uint8_t> owned_;
  void* mapped_ = nullptr;
};

/** The workdir as `solve` sees it. Everything you call on it is on the clock:
 *  reading your own representation is part of your entry. */
struct Work {
  std::string dir_path;

  Reader open(const char* name) const { return Reader(dir_path + "/" + name); }
  Mapped map(const char* name) const { return Mapped(dir_path + "/" + name); }
  std::vector<std::uint8_t> read_all(const char* name) const {
    Reader r = open(name);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(r.size()));
    r.read(bytes.data(), bytes.size());
    return bytes;
  }
  const char* dir() const { return dir_path.c_str(); }
};

/** The host's definition of the problem for this run. W is already generated:
 *  there is no gen_W in an entry any more, and no time to be spent on it. */
struct Params {
  std::uint64_t k;
  const double* W;  // row-major n_genes x k
  std::uint64_t n_cells;
  std::uint64_t n_genes;
};

/** The output, preallocated and zero-filled by the harness. Row-major
 *  n_cells x k. Writing it to disk is the harness's job, after the clock stops. */
struct Result {
  double* data;
  std::uint64_t n_cells;
  std::uint64_t k;
};

// The two functions an entry provides. Nothing else is linked from an entry.
void prepare(const Input& in, Sink& out);
void solve(const Work& w, const Params& p, Result& y);

}  // namespace hpcbench

// An entry file must not define main(): the harness owns the process. Defining
// one would either collide with the harness or, worse, replace it and skip the
// timing boundary entirely. `main` is renamed onto an object, so a definition
// becomes a compile error whose text says which rule it broke. (Renaming it
// onto a *type* is not enough: C++ lets a function hide a class name, so the
// definition would compile and sit there dead.)
namespace hpcbench {
namespace {
const int hpcbench_error_entry_files_must_not_define_main_the_harness_owns_it = 0;
}
}  // namespace hpcbench
using hpcbench::hpcbench_error_entry_files_must_not_define_main_the_harness_owns_it;
#ifndef HPCBENCH_HARNESS_TU
#define main hpcbench_error_entry_files_must_not_define_main_the_harness_owns_it
#endif


#endif  // HPCBENCH_TASK_HPP
