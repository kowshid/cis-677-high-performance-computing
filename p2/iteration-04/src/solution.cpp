// src/solution.cpp -- YOURS, and the ONLY thing timed.
//
//     void hpcbench::solve(const Work& w, const Params& p, Result& out);
//
// Train the autoencoder n_genes -> 256 -> 32 -> 256 -> n_genes for ONE epoch,
// then write the trained parameters into out.W1, out.b1, ..., out.W4, out.b4
// (row-major [fan_in][fan_out], the same layout as the initial weights in p).
//
// ---------------------------------------------------------------------------
// The whole computation, for one batch of B cells (B = 256, the last is 140):
//
//   forward      A1 = Xb W1 + b1     H1 = relu(A1)             B x 256
//                Z  = H1 W2 + b2                               B x 32
//                A3 = Z  W3 + b3     H3 = relu(A3)             B x 256
//                Xh = H3 W4 + b4                               B x n_genes
//   loss         L  = sum((Xh - Xb)^2) / (B * n_genes)
//   backward     dXh = 2 (Xh - Xb) / (B * n_genes)
//                dW4 = H3^T dXh    db4 = colsum(dXh)   dH3 = dXh W4^T
//                dA3 = dH3 * [A3 > 0]
//                dW3 = Z^T  dA3    db3 = colsum(dA3)   dZ  = dA3 W3^T
//                dW2 = H1^T dZ     db2 = colsum(dZ)    dH1 = dZ  W2^T
//                dA1 = dH1 * [A1 > 0]
//                dW1 = Xb^T dA1    db1 = colsum(dA1)
//   Adam, step t = 1, 2, ... (one per batch), for every parameter w with
//   gradient g:
//                m = beta1 m + (1 - beta1) g
//                v = beta2 v + (1 - beta2) g^2
//                w = w - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
// ---------------------------------------------------------------------------
//
// Changes from the baseline (ideas 1, 3, 7 and B; T5 in iteration-03; T4 in iteration-04):
//
//   [1] X is never dense. prepare writes a CSR file, solve maps it with mmap.
//       * Layer-1 forward A1 = Xb W1 is the Project 1 kernel: one CSR row of
//         the cell times the dense W1, rows of W1 picked by gene.
//       * dW1 = Xb^T dA1 is computed gene-major: at the start of solve every
//         gene's non-zeros are regrouped by batch (the batches depend on the
//         seed, so this is on the clock, O(nnz)), and dW1 row j is the sum of
//         x * dA1[slot] over the batch's cells that express gene j.
//       * dXh = 2 (Xh - Xb)/(B D) only subtracts at the non-zeros.
//   [7] Every product runs on 4 threads, split into blocks that each thread
//       owns outright, so no two threads write the same element.
//       The threads are a small std::thread team that lives for the whole
//       solve, not OpenMP: the shipped build.sh only enables OpenMP where
//       -fopenmp works, and Apple clang rejects it, so OpenMP pragmas would
//       silently run on one thread on a Mac. std::thread needs no flags.
//       Work is handed out dynamically, so a thread that lands on a slower
//       (efficiency) core simply takes fewer blocks.
//   [3] dA3 = dXh W4^T reads a transposed copy of W4 (lecture Target 3).
//       Before, every element of dA3 was one long running sum,
//           s += dXh[i][j] * W4[k][j]          for j = 0 .. 13713,
//       a loop-carried dependency: each add waits for the previous one, so
//       the loop could not vectorise (Week 5) and ran at the latency of one
//       FMA per step. Now, once per batch, W4 is copied into
//       W4T (n_genes x 256), and row i of dA3 is built as
//           dA3[i][:] += dXh[i][j] * W4T[j][:]  for j = 0 .. 13713,
//       whose inner loop runs over the 256 hidden units: 256 independent
//       sums, contiguous, so it vectorises (Lecture 4 Target 2: the loop
//       order now matches the storage order). Each element of dA3 is still
//       summed over j in ascending order, the same additions in the same
//       order as before. Built with -ffp-contract=off the two kernels agree
//       bit for bit; by default the compiler may fuse a * b + c into one FMA
//       (one rounding instead of two) in one kernel and not the other, which
//       moves dA3 by ~1e-10 relative and the final loss by far less than the
//       3e-5 tolerance. Only the L4 product uses this; the tiny L3 and L2
//       input gradients keep the old kernel.
//   [B] Register and cache blocking for the three layer-4 products
//       (Week 5, Cache blocking; Lecture 3, the roofline):
//           fwd  Xh  = H3 W4 + b4     (B x 256)  (256 x D)
//           bwd  dW4 = H3^T dXh       (256 x B)  (B x D)
//           bwd  dA3 = dXh W4^T       (B x D)    (D x 256), from W4T [3]
//       All three run on one micro-kernel that holds a 4 x 8 block of the
//       result in registers (32 sums) while the inner index runs. Per step
//       it loads 8 values of one operand and 4 of the other and does 32
//       multiply-adds with them; the old loops did one multiply-add per
//       value loaded, so they were limited by loads, not arithmetic.
//       Around it, the loops are ordered so the pieces being reused sit in
//       L1 (a 16 KB strip of W4 or dXh, a 16-row block of dXh) or L2 (H3),
//       and strided pieces are first copied into small contiguous buffers
//       ("packing"). Packing W4 into the strip also kept the forward
//       product from reading m and v in its inner loop while Adam's state
//       was an array of structs (it is separate arrays since [T4]).
//       Each element of each product starts from the same value and adds
//       its terms in the same ascending order as before, so the sums are
//       the baseline's (up to FMA fusing, as in [3]).
//
//   [T5] Allocate once (Lecture 4 Target 5). Measured in iteration-02 at
//       4 threads: "make_layer (AoS)" 80 ms and "alloc + zero-fill per
//       batch" 64 ms, both serial, together 8% of the epoch.
//       * Every activation, gradient and scratch buffer now lives in one
//         Workspace, allocated once per solve, sized for the largest batch.
//         Before, a fresh `Grads` and fresh activations were built for every
//         batch, so every batch page-faulted and zero-filled ~112 MB (Xh,
//         dXh, dW1, dW4) on one thread, and freed it again.
//       * Nothing is zero-filled. Buffer<T> uses `new T[n]`, which leaves
//         doubles uninitialised; every kernel writes each element of its
//         output before reading it (each one starts from the bias, from 0,
//         or overwrites), so a zero-fill was pure overhead. The first write
//         to each page happens inside a parallel kernel, so the page faults
//         are spread over the threads instead of serialised (and, on a
//         two-socket node, each page lands on the socket that uses it).
//       * make_layer no longer push_backs 3.5 M structs into a vector that
//         grows by copying itself; it allocates the array once and fills it
//         in parallel.
//       * The small per-call buffers (the packed strips of the layer-4
//         kernels, the per-row losses) are per-thread scratch that grows
//         once and is then reused.
//       The arithmetic is untouched: the trained weights are bit-identical
//       to iteration-02.
//
//   [T4] Struct of arrays (Lecture 4 Target 4, slides 50-51 and 62-63).
//       Before, each parameter was a struct {w, m, v}, so every kernel that
//       reads a weight dragged its two Adam moments through the cache with
//       it: 24 bytes moved for 8 used. Measured in iteration-02, the
//       layer-1 forward gathers ~850 W1 rows per cell this way (~14 GB per
//       epoch, bandwidth-bound), and the W4 strip packing and the W4 -> W4T
//       copy read 84 MB per batch for 28 MB of weights.
//       * w, m and v are now three separate arrays, so a kernel that needs
//         only w reads only w, and Adam streams three clean contiguous
//         arrays (plus the gradient) that the prefetcher follows and the
//         compiler vectorises.
//       * The weights are trained in place in the host's output arrays
//         (out.W1 ... out.b4). They are already in the required row-major
//         layout, already allocated and already page-faulted by the harness
//         before the clock started. The initial weights are copied in once
//         (in parallel), and the copy-out at the end is gone.
//       * m and v are never zero-filled. Step 1 writes them without reading
//         them: with m = v = 0 the update is m = (1 - beta1) g + 0.0 and
//         v = (1 - beta2) g g + 0.0, identical to the general formula in
//         every rounding and contraction (the + 0.0 reproduces 0 + x for
//         x = -0.0). That removes a 112 MB zero-fill and a 112 MB read.
//       * The Adam loop is a plain function with __restrict parameters,
//         called from the parallel_for lambda: inside the lambda GCC lost the
//         no-alias guarantee and would not vectorise it.
//       * Adam takes std::sqrt(std::fabs(vhat)) rather than std::sqrt(vhat).
//         GCC (on Linux, unlike Apple clang on macOS) keeps errno semantics
//         for sqrt by default, so every sqrt carried a branch to a library
//         call that sets errno for negative inputs, and the loop could not
//         vectorise. With fabs the compiler can prove the input is not
//         negative, drops the branch, and vectorises with no extra compiler
//         flag (build.sh stays the original). The value is unchanged: vhat
//         is never negative (it is at least +0), so fabs returns it as is.
//         (-fno-math-errno, a #pragma, or an optimize attribute were the
//         alternatives; GCC 13 ignored the pragma and the attribute.)
//       Same arithmetic in the same order: built with -ffp-contract=off it is
//       bit-identical to iteration-03. With default flags the compiler may
//       fuse the other product of beta1*m + (1-beta1)*g into an FMA once the
//       loop vectorises, which moves the 17th significant digit of the loss.
//
// Kept on purpose, so each later idea can be measured on its own:
//   the layer-1 forward cell-major (CSR gather), the layer-4 products dense
//   (ReLU zeros included), the per-batch W4 -> W4T copy, fp64 throughout,
//   no intrinsics (only the generic vector type in [B]).
//
// Every dot product and accumulation runs in the same order as the baseline,
// so the trained weights should agree with it to the last bit or very nearly.
//
// Compile-time switches (pass them through CXXFLAGS; build.sh is unchanged):
//   -DP2_PROFILE          per-phase timing table on stderr
//   -DP2_THREADS=n        use n threads instead of 4, to measure scaling
//   -DHPCBENCH_GRADCHECK  the host's finite-difference gradient check
// e.g.  CXXFLAGS="-O3 -std=c++17 -DNDEBUG -DP2_PROFILE" sh ./build.sh
#include "task.hpp"

#ifdef HPCBENCH_GRADCHECK
#include "gradcheck.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// The in-browser runner (wasm) has no threads: everything runs serially there.
#if defined(__wasm__) || defined(__EMSCRIPTEN__)
#define P2_SERIAL 1
#endif

#ifndef P2_SERIAL
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

#if defined(__APPLE__) && !defined(P2_SERIAL)
#include <pthread/qos.h>
#endif

#ifdef P2_PROFILE
#include <chrono>
#endif

namespace {

using u64 = std::uint64_t;
using i64 = std::int64_t;

// TODO: use parallelism
#ifndef P2_THREADS
#define P2_THREADS 4
#endif
#ifdef P2_SERIAL
constexpr int kThreads = 1;
#else
constexpr int kThreads = P2_THREADS;
#endif

// Block sizes for the dense products.
constexpr i64 kTileRows = 64;    // rows of the batch per block
constexpr i64 kTileCols = 128;   // output columns per block
constexpr i64 kTileK = 64;       // fan-in rows per block (weight_grad, input_grad)
constexpr i64 kDotChunk = 512;   // reduction chunk inside input_grad
constexpr i64 kDaRows = 16;      // batch rows per task in input_grad_t
constexpr i64 kTransCols = 32;   // columns of W per block in transpose_weights

// Register and cache blocking for the layer-4 products (idea B).
constexpr i64 kMR = 4;           // micro-tile rows held in registers
constexpr i64 kNR = 8;           // micro-tile columns held in registers (4 x 8 = 32 sums)
constexpr i64 kNC = 64;          // output columns per task (fwd L4, dW4)
constexpr i64 kKC = 128;         // genes per L1 block in input_grad_t (16 x 128 of dXh = 16 KB)

// Element-wise loops are handed out in pieces of this many elements.
constexpr i64 kElemChunk = 16384;

inline i64 ceil_div(i64 a, i64 b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Memory that is allocated once and never zero-filled ([T5]).
// ---------------------------------------------------------------------------

// A fixed-size array. `new T[n]` default-initialises, which for double and
// other trivial types writes nothing at all: no memset, no page touched. The
// first write to each page happens in whichever kernel produces that data,
// on whichever thread produces it. Every buffer below is fully written before
// it is read, so nothing ever needs to be zero.
template <class T>
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(u64 n) : p_(n ? new T[n] : nullptr), n_(n) {}
  T* data() { return p_.get(); }
  const T* data() const { return p_.get(); }
  u64 size() const { return n_; }
  T& operator[](u64 i) { return p_[i]; }
  const T& operator[](u64 i) const { return p_[i]; }

 private:
  std::unique_ptr<T[]> p_;
  u64 n_ = 0;
};

// Per-thread scratch for the packed strips of the layer-4 kernels. Each
// thread's buffer grows once, the first time it is needed, and is reused by
// every later call on that thread (the team's threads live for the whole
// solve), so the epoch loop never calls the allocator.
inline double* thread_scratch(u64 n) {
  thread_local std::vector<double> buf;
  if (buf.size() < n) buf.resize(n);
  return buf.data();
}

// ---------------------------------------------------------------------------
// Optional per-phase timing (-DP2_PROFILE). Compiles to nothing otherwise.
// ---------------------------------------------------------------------------
#ifdef P2_PROFILE
enum Phase {
  kSetup, kLayers, kAlloc, kFwdL1, kFwdMid, kFwdL4, kLoss, kBwdL4dW, kBwdL4T, kBwdL4dA, kBwdMid,
  kBwdL1, kAdam, kCopyOut, kNumPhases
};
const char* const kPhaseName[kNumPhases] = {
    "setup: threads, map, buckets", "setup: params in (SoA)",   "alloc workspace (once)",
    "fwd L1 sparse + relu",         "fwd L2, L3 + relu",        "fwd L4 dense",
    "loss + dXh",                   "bwd L4: dW4, db4",         "bwd L4: W4 -> W4T copy",
    "bwd L4: dA3 + mask",           "bwd L3, L2",               "bwd L1 sparse: dW1, db1",
    "adam (SoA)",                   "copy out (none: in place)"};
double g_phase_ms[kNumPhases] = {};

struct PhaseTimer {
  Phase phase;
  std::chrono::steady_clock::time_point t0;
  explicit PhaseTimer(Phase ph) : phase(ph), t0(std::chrono::steady_clock::now()) {}
  ~PhaseTimer() {
    g_phase_ms[phase] +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
};
#define PHASE(p) PhaseTimer p2_phase_timer_(p)
#else
#define PHASE(p) ((void)0)
#endif

// ---------------------------------------------------------------------------
// Threads (idea 7): a fork-join team that lives for the whole solve.
// ---------------------------------------------------------------------------
#ifndef P2_SERIAL

// A hint to the core that this thread is busy-waiting.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield");
#endif
}

// On Apple silicon, ask the scheduler to keep this thread on a performance
// core. A hint only; elsewhere it does nothing.
inline void prefer_fast_cores() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

// TODO Thread team
class Team {
 public:
  explicit Team(int n) : n_(n < 1 ? 1 : n) {
    prefer_fast_cores();
    for (int t = 1; t < n_; ++t) threads_.emplace_back([this, t] { worker(t); });
  }
  ~Team() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      gen_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (std::thread& th : threads_) th.join();
  }
  Team(const Team&) = delete;
  Team& operator=(const Team&) = delete;

  int size() const { return n_; }

  // Runs job(tid) for tid = 0 .. size()-1, one per thread, and waits for all.
  template <class Job>
  void run(Job& job) {
    if (n_ == 1) {
      job(0);
      return;
    }
    fn_ = [](void* ctx, int tid) { (*static_cast<Job*>(ctx))(tid); };
    ctx_ = &job;
    pending_.store(n_ - 1, std::memory_order_relaxed);
    bool anyone_asleep;
    {
      std::lock_guard<std::mutex> lock(mu_);
      gen_.fetch_add(1, std::memory_order_release);
      anyone_asleep = sleepers_ > 0;
    }
    if (anyone_asleep) wake_.notify_all();
    job(0);
    for (int spins = 0; pending_.load(std::memory_order_acquire) != 0; ++spins) {
      if (spins < kSpinPause)
        cpu_relax();
      else
        std::this_thread::yield();
    }
  }

 private:
  static constexpr int kSpinPause = 4096;  // busy-wait rounds before yielding
  static constexpr int kSpinYield = 256;   // yield rounds before sleeping

  void worker(int tid) {
    prefer_fast_cores();
    std::uint64_t seen = 0;
    for (;;) {
      std::uint64_t g = gen_.load(std::memory_order_acquire);
      for (int spins = 0; g == seen; ++spins) {
        if (spins < kSpinPause) {
          cpu_relax();
        } else if (spins < kSpinPause + kSpinYield) {
          std::this_thread::yield();
        } else {
          std::unique_lock<std::mutex> lock(mu_);
          ++sleepers_;
          wake_.wait(lock, [&] { return gen_.load(std::memory_order_acquire) != seen; });
          --sleepers_;
        }
        g = gen_.load(std::memory_order_acquire);
      }
      seen = g;
      if (stop_) return;  // written before the generation bump that got us here
      fn_(ctx_, tid);
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  int n_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_;
  std::atomic<std::uint64_t> gen_{0};
  std::atomic<int> pending_{0};
  int sleepers_ = 0;   // guarded by mu_
  bool stop_ = false;  // set under mu_ before the final generation bump
  void (*fn_)(void*, int) = nullptr;
  void* ctx_ = nullptr;
};

Team* g_team = nullptr;  // the team of the running solve

// Creates the team for the duration of solve.
struct TeamScope {
  Team team;
  explicit TeamScope(int n) : team(n) { g_team = &team; }
  ~TeamScope() { g_team = nullptr; }
};

#endif  // P2_SERIAL

// Calls body(lo, hi) on consecutive ranges [lo, hi) covering [0, n), each at
// most `chunk` long, handing the next range to whichever thread asks first.
// Dynamic hand-out keeps every thread busy when the work per index is uneven
// (cells, genes) and when the cores are not equally fast.
template <class Body>
void parallel_for(i64 n, i64 chunk, Body&& body) {
  if (n <= 0) return;
#ifndef P2_SERIAL
  if (g_team != nullptr && g_team->size() > 1 && n > chunk) {
    std::atomic<i64> next{0};
    auto job = [&](int) {
      for (;;) {
        const i64 lo = next.fetch_add(chunk, std::memory_order_relaxed);
        if (lo >= n) break;
        body(lo, std::min(n, lo + chunk));
      }
    };
    g_team->run(job);
    return;
  }
#endif
  body(i64(0), n);
}

// ---------------------------------------------------------------------------
// The corpus, as written by prepare.cpp. KEEP THE LAYOUT IN SYNC WITH IT.
// ---------------------------------------------------------------------------
constexpr char kFileName[] = "xcsr.bin";
constexpr char kMagic[8] = {'P', '2', 'C', 'S', 'R', 'v', '0', '1'};
constexpr u64 kHeaderBytes = 32;

// The mapped CSR plus, for every gene, its non-zeros regrouped by batch.
// Not copyable or movable: the CSR pointers point into `file`.
struct Corpus {
  hpcbench::Mapped file;

  // CSR, one row per cell (mapped, read-only).
  u64 n_cells = 0, n_genes = 0, nnz = 0;
  const u64* row_ptr = nullptr;         // n_cells + 1
  const double* val = nullptr;          // nnz
  const std::uint16_t* col = nullptr;   // nnz

  // Gene-major, batch-bucketed copy (built in solve, because batches depend on
  // the seed). Gene j's non-zeros from batch b are the entries
  //     [bptr[j * n_batches + b], bptr[j * n_batches + b + 1])
  // of bslot / bval, in ascending slot order, where slot is the cell's row
  // inside its batch. bptr has n_genes * n_batches + 1 entries.
  u64 n_batches = 0;
  std::vector<std::uint32_t> bptr;
  std::unique_ptr<std::uint16_t[]> bslot;
  std::unique_ptr<double[]> bval;

  Corpus(const hpcbench::Work& w, const hpcbench::Params& p);
  Corpus(const Corpus&) = delete;
  Corpus& operator=(const Corpus&) = delete;
};

Corpus::Corpus(const hpcbench::Work& w, const hpcbench::Params& p) : file(w.map(kFileName)) {
  PHASE(kSetup);
  const char* bad = "xcsr.bin is missing or damaged; re-run prepare";
  // TODO: Use memory map
  if (file.bytes() < kHeaderBytes || std::memcmp(file.data(), kMagic, sizeof(kMagic)) != 0)
    hpcbench::fatal(bad);
  n_cells = *file.as<u64>(8);
  n_genes = *file.as<u64>(16);
  nnz = *file.as<u64>(24);
  if (n_cells != p.n_cells || n_genes != p.n_genes) hpcbench::fatal(bad);
  const u64 val_off = kHeaderBytes + (n_cells + 1) * sizeof(u64);
  const u64 col_off = val_off + nnz * sizeof(double);
  if (file.bytes() != col_off + nnz * sizeof(std::uint16_t)) hpcbench::fatal(bad);
  row_ptr = file.as<u64>(kHeaderBytes);
  val = file.as<double>(val_off);
  col = file.as<std::uint16_t>(col_off);
  if (row_ptr[0] != 0 || row_ptr[n_cells] != nnz) hpcbench::fatal(bad);
  if (nnz >= (u64(1) << 32)) hpcbench::fatal("more non-zeros than 32-bit bucket offsets hold");
  if (p.batch == 0 || p.batch > 65536) hpcbench::fatal("batch size does not fit a 16-bit slot");

  // --- Regroup the non-zeros by (gene, batch, slot). ---
  const u64 N = n_cells, D = n_genes, batch = p.batch;
  const u64 NB = (N + batch - 1) / batch;
  n_batches = NB;
  const std::uint32_t* order = p.order;

  // Count per (batch, gene). Batch-major, so threads working on different
  // batches never write the same cache line.
  std::vector<std::uint32_t> cnt(NB * D, 0);
  parallel_for((i64)NB, 1, [&](i64 lo, i64 hi) {
    for (i64 b = lo; b < hi; ++b) {
      std::uint32_t* c = cnt.data() + u64(b) * D;
      const u64 p0 = u64(b) * batch, p1 = std::min(N, p0 + batch);
      for (u64 q = p0; q < p1; ++q) {
        const u64 cell = order[q];
        for (u64 t = row_ptr[cell]; t < row_ptr[cell + 1]; ++t) ++c[col[t]];
      }
    }
  });

  // Exclusive prefix sum in (gene, batch) order. cnt becomes the write cursor.
  bptr.resize(D * NB + 1);
  std::uint32_t run = 0;
  for (u64 j = 0; j < D; ++j)
    for (u64 b = 0; b < NB; ++b) {
      bptr[j * NB + b] = run;
      const std::uint32_t n = cnt[b * D + j];
      cnt[b * D + j] = run;
      run += n;
    }
  bptr[D * NB] = run;

  // Place. Walking each batch's cells in the host's order puts every bucket in
  // ascending slot order, which is the order the baseline sums dW1 in.
  bslot.reset(new std::uint16_t[nnz]);
  bval.reset(new double[nnz]);
  std::uint16_t* slot_out = bslot.get();
  double* val_out = bval.get();
  parallel_for((i64)NB, 1, [&](i64 lo, i64 hi) {
    for (i64 b = lo; b < hi; ++b) {
      std::uint32_t* cur = cnt.data() + u64(b) * D;
      const u64 p0 = u64(b) * batch, p1 = std::min(N, p0 + batch);
      for (u64 q = p0; q < p1; ++q) {
        const u64 cell = order[q];
        const std::uint16_t slot = static_cast<std::uint16_t>(q - p0);
        for (u64 t = row_ptr[cell]; t < row_ptr[cell + 1]; ++t) {
          const std::uint32_t dst = cur[col[t]]++;
          slot_out[dst] = slot;
          val_out[dst] = val[t];
        }
      }
    }
  });
}

// One optimiser step's slice of the data.
struct Batch {
  const Corpus* X;
  const std::uint32_t* cells;  // the B cell ids of this batch, in the host's order
  u64 B;                       // rows in this batch
  u64 index;                   // which batch, 0-based
};

// ---------------------------------------------------------------------------
// Model and optimiser state: struct of arrays ([T4]), trained in place in
// the host's output arrays, allocated once ([T5]).
// ---------------------------------------------------------------------------

struct Layer {
  u64 in = 0;
  u64 out = 0;
  double* W = nullptr;    // in x out, row-major: the host's out.Wn, trained in place
  double* b = nullptr;    // out: the host's out.bn, trained in place
  Buffer<double> mW, vW;  // Adam moments of W, same layout as W
  Buffer<double> mb, vb;  // Adam moments of b
};

// W and b are the host's output arrays. The initial values are copied in by
// all threads; the moments are allocated but not written (step 1 writes them,
// see adam()).
Layer make_layer(const double* W0, const double* b0, double* W, double* b, u64 in, u64 out) {
  Layer L;
  L.in = in;
  L.out = out;
  L.W = W;
  L.b = b;
  parallel_for((i64)(in * out), kElemChunk, [&](i64 lo, i64 hi) {
    std::memcpy(W + lo, W0 + lo, u64(hi - lo) * sizeof(double));
  });
  std::memcpy(b, b0, out * sizeof(double));
  L.mW = Buffer<double>(in * out);
  L.vW = Buffer<double>(in * out);
  L.mb = Buffer<double>(out);
  L.vb = Buffer<double>(out);
  return L;
}

struct Model {
  Layer l1, l2, l3, l4;
};

struct Grads {
  Buffer<double> W1, b1, W2, b2, W3, b3, W4, b4;
};

// Every buffer one batch needs, allocated once per solve for the largest
// batch (B_max = 256; the short last batch uses the first B rows).
struct Workspace {
  Buffer<double> A1, H1, Z, A3, H3;  // forward activations
  Buffer<double> Xh, dXh;            // B x n_genes each: 28 MB apiece
  Buffer<double> dA3, dZ, dA1;       // backward activation gradients
  Buffer<double> w4t;                // W4 transposed, n_genes x 256 (idea 3)
  Buffer<double> row_loss;           // per-row share of the batch loss
  Grads g;

  Workspace(u64 B, u64 D, u64 H, u64 K)
      : A1(B * H), H1(B * H), Z(B * K), A3(B * H), H3(B * H), Xh(B * D), dXh(B * D),
        dA3(B * H), dZ(B * K), dA1(B * H), w4t(D * H), row_loss(B) {
    g.W1 = Buffer<double>(D * H);
    g.b1 = Buffer<double>(H);
    g.W2 = Buffer<double>(H * K);
    g.b2 = Buffer<double>(K);
    g.W3 = Buffer<double>(K * H);
    g.b3 = Buffer<double>(H);
    g.W4 = Buffer<double>(H * D);
    g.b4 = Buffer<double>(D);
  }
};

// ---------------------------------------------------------------------------
// Dense kernels, threaded. Each block of the output belongs to one thread, and
// inside a block the arithmetic runs in the baseline's order.
// ---------------------------------------------------------------------------

// C = A W + b, with A rows x in and C rows x out. Blocks are (column tile,
// row tile). Used for the small layers 2 and 3 only; layer 4 has
// affine_blocked (idea B).
void affine(const double* A, u64 rows, const Layer& L, double* C) {
  const i64 in = (i64)L.in, out = (i64)L.out, R = (i64)rows;
  const double* W = L.W;
  const double* b = L.b;
  const i64 n_ct = ceil_div(out, kTileCols), n_rt = ceil_div(R, kTileRows);
  const i64 chunk = n_ct >= 2 * kThreads ? n_rt : 1;
  parallel_for(n_ct * n_rt, chunk, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const i64 ct = task / n_rt, rt = task % n_rt;
      const i64 j0 = ct * kTileCols, j1 = std::min(out, j0 + kTileCols);
      const i64 i0 = rt * kTileRows, i1 = std::min(R, i0 + kTileRows);
      for (i64 i = i0; i < i1; ++i) {
        double* c = C + i * out;
        for (i64 j = j0; j < j1; ++j) c[j] = b[j];
        for (i64 k = 0; k < in; ++k) {
          const double a = A[i * in + k];
          const double* wk = W + k * out;
          for (i64 j = j0; j < j1; ++j) c[j] += a * wk[j];
        }
      }
    }
  });
}

void relu(const double* A, double* H, u64 n) {
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 e = lo; e < hi; ++e) H[e] = A[e] > 0.0 ? A[e] : 0.0;
  });
}

// dA *= [A > 0]
void relu_mask(double* dA, const double* A, u64 n) {
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 e = lo; e < hi; ++e)
      if (!(A[e] > 0.0)) dA[e] = 0.0;
  });
}

// gW = A^T dC (in x out) and gb = colsum(dC), for A rows x in, dC rows x out.
// Blocks are (column tile, k tile) of gW. Used for the small layers 2 and 3
// only; layer 4 has weight_grad_blocked (idea B). gW and gb must already have
// the right size; every element is overwritten.
void weight_grad(const double* A, const double* dC, u64 rows, u64 in, u64 out, double* gW,
                 double* gb) {
  const i64 R = (i64)rows, IN = (i64)in, OUT = (i64)out;
  const i64 n_ct = ceil_div(OUT, kTileCols), n_kt = ceil_div(IN, kTileK);
  const i64 chunk = n_ct >= 2 * kThreads ? n_kt : 1;
  parallel_for(n_ct * n_kt, chunk, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const i64 ct = task / n_kt, kt = task % n_kt;
      const i64 j0 = ct * kTileCols, j1 = std::min(OUT, j0 + kTileCols);
      const i64 k0 = kt * kTileK, k1 = std::min(IN, k0 + kTileK);
      for (i64 k = k0; k < k1; ++k) {
        double* g = gW + k * OUT;
        for (i64 j = j0; j < j1; ++j) g[j] = 0.0;
        for (i64 i = 0; i < R; ++i) {
          const double a = A[i * IN + k];
          const double* d = dC + i * OUT;
          for (i64 j = j0; j < j1; ++j) g[j] += a * d[j];
        }
      }
      if (kt == 0) {  // the column sums of this tile, owned by the k-tile-0 block
        for (i64 j = j0; j < j1; ++j) gb[j] = 0.0;
        for (i64 i = 0; i < R; ++i) {
          const double* d = dC + i * OUT;
          for (i64 j = j0; j < j1; ++j) gb[j] += d[j];
        }
      }
    }
  });
}

// dA = dC W^T (rows x in), for dC rows x out. Blocks are (row tile, k tile) of
// dA. The reduction over `out` runs in chunks so the pieces of dC and W stay
// in cache; the running sum is carried across chunks, so each element is still
// summed over j = 0, 1, ..., out-1 in order, exactly like the baseline.
void input_grad(const double* dC, u64 rows, const Layer& L, double* dA) {
  const i64 R = (i64)rows, IN = (i64)L.in, OUT = (i64)L.out;
  const double* W = L.W;
  const i64 n_rt = ceil_div(R, kTileRows), n_kt = ceil_div(IN, kTileK);
  parallel_for(n_rt * n_kt, 1, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const i64 rt = task / n_kt, kt = task % n_kt;
      const i64 i0 = rt * kTileRows, i1 = std::min(R, i0 + kTileRows);
      const i64 k0 = kt * kTileK, k1 = std::min(IN, k0 + kTileK);
      for (i64 i = i0; i < i1; ++i)
        for (i64 k = k0; k < k1; ++k) dA[i * IN + k] = 0.0;
      for (i64 j0 = 0; j0 < OUT; j0 += kDotChunk) {
        const i64 j1 = std::min(OUT, j0 + kDotChunk);
        for (i64 i = i0; i < i1; ++i) {
          const double* d = dC + i * OUT;
          for (i64 k = k0; k < k1; ++k) {
            const double* wk = W + k * OUT;
            double s = dA[i * IN + k];
            for (i64 j = j0; j < j1; ++j) s += d[j] * wk[j];
            dA[i * IN + k] = s;
          }
        }
      }
    }
  });
}

// WT = L.W^T, so WT is L.out x L.in and WT[j][k] = W[k][j]. With SoA this
// reads 28 MB of weights for layer 4, not the 84 MB of {w, m, v} structs.
// Blocks of kTransCols columns of W: each block is a few rows of WT (a few
// KB), which stay in cache while all L.in rows of W are walked past them.
void transpose_weights(const Layer& L, double* WT) {
  const i64 IN = (i64)L.in, OUT = (i64)L.out;
  const double* W = L.W;
  parallel_for(ceil_div(OUT, kTransCols), 4, [&](i64 lo, i64 hi) {
    for (i64 bk = lo; bk < hi; ++bk) {
      const i64 j0 = bk * kTransCols, j1 = std::min(OUT, j0 + kTransCols);
      for (i64 k = 0; k < IN; ++k) {
        const double* wk = W + k * OUT;
        for (i64 j = j0; j < j1; ++j) WT[j * IN + k] = wk[j];
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Register- and cache-blocked kernels for the layer-4 products (idea B).
//
// Each product is C = A B, a sum over an inner index p. The micro-kernel
// below keeps a kMR x kNR = 4 x 8 block of C in registers while p runs. Per
// step it loads 8 values of B and 4 of A, and does 32 multiply-adds with
// them, so it is limited by arithmetic rather than by loads.
//
// A(r, p) = a[r * ars + p * aps] and B(p, c) = b[p * bps + c]: the strides
// let the same kernel read H3, H3^T, dXh and the packed strips.
//
// Why vector types and not plain loops: written as plain loops over c, GCC
// vectorised the p loop instead wherever A is contiguous along p (fwd L4 and
// dA3). Without fast-math it may only do that as an in-order reduction, which
// rebuilt a serial dependency chain and made those two products slower than
// before. v4d, the compilers' generic vector type (GCC and Clang, no
// intrinsics), states which loop is the vector one: across the 8 columns.
// It becomes one AVX register on x86 and two NEON registers on Arm, so the
// 32 sums live in 8 AVX or 16 NEON registers. Every lane does exactly the
// scalar arithmetic, so the order of each sum is unchanged.
// ---------------------------------------------------------------------------
typedef double v4d __attribute__((vector_size(32)));

static_assert(kMR == 4 && kNR == 8, "micro_kernel is written out for a 4 x 8 tile");

// acc (kMR x kNR, row-major) += sum over p = 0 .. np-1 of A(r, p) * B(p, c).
// memcpy is the portable unaligned vector load and store; it compiles to one
// instruction.
inline void micro_kernel(i64 np, const double* __restrict a, i64 ars, i64 aps,
                         const double* __restrict b, i64 bps, double* __restrict acc) {
  v4d c00, c01, c10, c11, c20, c21, c30, c31;
  std::memcpy(&c00, acc + 0, sizeof(v4d));
  std::memcpy(&c01, acc + 4, sizeof(v4d));
  std::memcpy(&c10, acc + 8, sizeof(v4d));
  std::memcpy(&c11, acc + 12, sizeof(v4d));
  std::memcpy(&c20, acc + 16, sizeof(v4d));
  std::memcpy(&c21, acc + 20, sizeof(v4d));
  std::memcpy(&c30, acc + 24, sizeof(v4d));
  std::memcpy(&c31, acc + 28, sizeof(v4d));
  for (i64 p = 0; p < np; ++p) {
    const double* bp = b + p * bps;
    const double* ap = a + p * aps;
    v4d b0, b1;
    std::memcpy(&b0, bp, sizeof(v4d));
    std::memcpy(&b1, bp + 4, sizeof(v4d));
    const double a0 = ap[0], a1 = ap[ars], a2 = ap[2 * ars], a3 = ap[3 * ars];
    c00 += a0 * b0;
    c01 += a0 * b1;
    c10 += a1 * b0;
    c11 += a1 * b1;
    c20 += a2 * b0;
    c21 += a2 * b1;
    c30 += a3 * b0;
    c31 += a3 * b1;
  }
  std::memcpy(acc + 0, &c00, sizeof(v4d));
  std::memcpy(acc + 4, &c01, sizeof(v4d));
  std::memcpy(acc + 8, &c10, sizeof(v4d));
  std::memcpy(acc + 12, &c11, sizeof(v4d));
  std::memcpy(acc + 16, &c20, sizeof(v4d));
  std::memcpy(acc + 20, &c21, sizeof(v4d));
  std::memcpy(acc + 24, &c30, sizeof(v4d));
  std::memcpy(acc + 28, &c31, sizeof(v4d));
}

// The same sum for a partial tile (mr <= kMR rows, nr <= kNR columns) at the
// edges, e.g. the last 2 of the 13,714 genes. Same order: p ascending.
void micro_kernel_edge(i64 mr, i64 nr, i64 np, const double* a, i64 ars, i64 aps,
                       const double* b, i64 bps, double* acc) {
  for (i64 p = 0; p < np; ++p)
    for (i64 r = 0; r < mr; ++r) {
      const double ar = a[r * ars + p * aps];
      for (i64 c = 0; c < nr; ++c) acc[r * kNR + c] += ar * b[p * bps + c];
    }
}

inline void micro(i64 mr, i64 nr, i64 np, const double* a, i64 ars, i64 aps, const double* b,
                  i64 bps, double* acc) {
  if (mr == kMR && nr == kNR)
    micro_kernel(np, a, ars, aps, b, bps, acc);
  else
    micro_kernel_edge(mr, nr, np, a, ars, aps, b, bps, acc);
}

// Forward, wide layer: C = A W + b, A rows x in, C rows x out (layer 4:
// 256 x 256 times 256 x 13,714). Tasks are kNC-column blocks of C. For each
// 8-column strip, W's strip (in x 8, 16 KB) is packed out of the AoS into a
// contiguous buffer that stays in L1 while all of A (512 KB, in L2) streams
// past it, 4 rows at a time. Each element starts from b[j] and adds
// A[i][p] * W[p][j] for p = 0, 1, ..., in-1, as in affine().
void affine_blocked(const double* A, u64 rows, const Layer& L, double* C) {
  const i64 R = (i64)rows, IN = (i64)L.in, OUT = (i64)L.out;
  const double* W = L.W;
  const double* b = L.b;
  parallel_for(ceil_div(OUT, kNC), 1, [&](i64 lo, i64 hi) {
    double* const strip = thread_scratch(u64(IN * kNR));  // reused, never reallocated
    double acc[kMR * kNR] = {};
    for (i64 blk = lo; blk < hi; ++blk) {
      const i64 j0 = blk * kNC, j1 = std::min(OUT, j0 + kNC);
      for (i64 jr = j0; jr < j1; jr += kNR) {
        const i64 nr = std::min(kNR, j1 - jr);
        for (i64 p = 0; p < IN; ++p) {
          const double* wp = W + p * OUT + jr;
          for (i64 c = 0; c < nr; ++c) strip[p * kNR + c] = wp[c];
        }
        for (i64 ir = 0; ir < R; ir += kMR) {
          const i64 mr = std::min(kMR, R - ir);
          for (i64 r = 0; r < mr; ++r)
            for (i64 c = 0; c < nr; ++c) acc[r * kNR + c] = b[jr + c];
          micro(mr, nr, IN, A + ir * IN, IN, 1, strip, kNR, acc);
          for (i64 r = 0; r < mr; ++r)
            for (i64 c = 0; c < nr; ++c) C[(ir + r) * OUT + jr + c] = acc[r * kNR + c];
        }
      }
    }
  });
}

// Backward, wide layer: gW = A^T dC (in x out) and gb = colsum(dC), for A
// rows x in and dC rows x out (layer 4: 256 x 256 and 256 x 13,714). Tasks
// are kNC-column blocks. For each 8-column strip, dC's strip (rows x 8,
// 16 KB) is packed contiguously -- in dC its rows are 110 KB apart -- and
// stays in L1 while every 4-row block of gW is formed from it. Each element
// starts from 0 and adds A[i][k] * dC[i][j] for i = 0, 1, ..., rows-1, and
// gb[j] adds dC[i][j] in the same order, as in weight_grad().
void weight_grad_blocked(const double* A, const double* dC, u64 rows, u64 in, u64 out,
                         double* gW, double* gb) {
  const i64 R = (i64)rows, IN = (i64)in, OUT = (i64)out;
  parallel_for(ceil_div(OUT, kNC), 1, [&](i64 lo, i64 hi) {
    double* const strip = thread_scratch(u64(R * kNR));  // reused, never reallocated
    double acc[kMR * kNR] = {};
    for (i64 blk = lo; blk < hi; ++blk) {
      const i64 j0 = blk * kNC, j1 = std::min(OUT, j0 + kNC);
      for (i64 jr = j0; jr < j1; jr += kNR) {
        const i64 nr = std::min(kNR, j1 - jr);
        for (i64 i = 0; i < R; ++i) {
          const double* d = dC + i * OUT + jr;
          for (i64 c = 0; c < nr; ++c) strip[i * kNR + c] = d[c];
        }
        for (i64 c = 0; c < nr; ++c) {
          double s = 0.0;
          for (i64 i = 0; i < R; ++i) s += strip[i * kNR + c];
          gb[jr + c] = s;
        }
        for (i64 kr = 0; kr < IN; kr += kMR) {
          const i64 mr = std::min(kMR, IN - kr);
          for (i64 r = 0; r < mr; ++r)
            for (i64 c = 0; c < nr; ++c) acc[r * kNR + c] = 0.0;
          // A(r, p) = A[p][kr + r]: rows of the tile are adjacent columns of A.
          micro(mr, nr, R, A + kr, 1, IN, strip, kNR, acc);
          for (i64 r = 0; r < mr; ++r)
            for (i64 c = 0; c < nr; ++c) gW[(kr + r) * OUT + jr + c] = acc[r * kNR + c];
        }
      }
    }
  });
}

// dA = dC W^T (rows x in), for dC rows x out, from the TRANSPOSED weights WT
// (out x in, see transpose_weights): layer 4, 256 x 13,714 times 13,714 x 256.
// Element (i, k) is the sum over j = 0, 1, ..., out-1 of dC[i][j] * WT[j][k]:
// for each output element its own running sum, so there is no long
// dependency chain (idea 3), and the micro-kernel computes 32 of them at once.
//
// Tasks are kDaRows-row blocks of dA, each owned by one thread. Inside one,
// the genes go by in blocks of kKC: the block's 16 x 128 piece of dC (16 KB)
// stays in L1 while all 32 eight-column strips of WT are applied to it, and
// each strip (128 x 8) is reused by the 4 four-row tiles. The partial sums
// wait in dA between gene blocks, which are taken in ascending order, so each
// element is still summed over j = 0, 1, ..., out-1 in order from 0.
void input_grad_t(const double* dC, u64 rows, u64 out, u64 in, const double* WT, double* dA) {
  const i64 R = (i64)rows, IN = (i64)in, OUT = (i64)out;
  parallel_for(ceil_div(R, kDaRows), 1, [&](i64 lo, i64 hi) {
    double acc[kMR * kNR] = {};
    for (i64 rt = lo; rt < hi; ++rt) {
      const i64 i0 = rt * kDaRows, i1 = std::min(R, i0 + kDaRows);
      std::fill(dA + i0 * IN, dA + i1 * IN, 0.0);
      for (i64 jc = 0; jc < OUT; jc += kKC) {
        const i64 np = std::min(kKC, OUT - jc);
        for (i64 k0 = 0; k0 < IN; k0 += kNR) {
          const i64 nr = std::min(kNR, IN - k0);
          for (i64 ir = i0; ir < i1; ir += kMR) {
            const i64 mr = std::min(kMR, i1 - ir);
            for (i64 r = 0; r < mr; ++r)
              for (i64 c = 0; c < nr; ++c) acc[r * kNR + c] = dA[(ir + r) * IN + k0 + c];
            micro(mr, nr, np, dC + ir * OUT + jc, OUT, 1, WT + jc * IN + k0, IN, acc);
            for (i64 r = 0; r < mr; ++r)
              for (i64 c = 0; c < nr; ++c) dA[(ir + r) * IN + k0 + c] = acc[r * kNR + c];
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Sparse kernels (idea 1).
// ---------------------------------------------------------------------------

// A1 = Xb W1 + b1: the Project 1 kernel. One cell per index; its output row is
// written by one thread. Summing over the cell's genes in ascending order is
// the baseline's order with the zero terms left out.
void sparse_forward_l1(const Batch& bt, const Layer& L, double* A1) {
  const Corpus& X = *bt.X;
  const u64 H = L.out;
  const double* W = L.W;
  const double* b = L.b;
  parallel_for((i64)bt.B, 4, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; ++i) {
      double* a = A1 + u64(i) * H;
      for (u64 k = 0; k < H; ++k) a[k] = b[k];
      const u64 cell = bt.cells[i];
      for (u64 t = X.row_ptr[cell]; t < X.row_ptr[cell + 1]; ++t) {
        const double x = X.val[t];
        const double* wj = W + u64(X.col[t]) * H;  // 2 KB of weights, no moments
        for (u64 k = 0; k < H; ++k) a[k] += x * wj[k];
      }
    }
  });
}

// dXh = 2 (Xh - Xb) / (B D). Xh - Xb equals Xh wherever Xb is zero, so only
// the non-zeros are subtracted (in place: Xh is not needed afterwards).
// Returns the batch loss, which is only printed. Each row's share is kept and
// summed in row order afterwards, so the printed loss does not depend on the
// threads.
double loss_and_dxh(const Batch& bt, u64 D, double* Xh, double* dXh, double* row_loss) {
  const Corpus& X = *bt.X;
  const double scale = 1.0 / (double(bt.B) * double(D));
  parallel_for((i64)bt.B, 4, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; ++i) {
      double* xh = Xh + u64(i) * D;
      const u64 cell = bt.cells[i];
      for (u64 t = X.row_ptr[cell]; t < X.row_ptr[cell + 1]; ++t) xh[X.col[t]] -= X.val[t];
      double* g = dXh + u64(i) * D;
      double s = 0.0;
      for (u64 j = 0; j < D; ++j) {
        const double d = xh[j];
        s += d * d;
        g[j] = 2.0 * scale * d;
      }
      row_loss[i] = s;
    }
  });
  double loss = 0.0;
  for (u64 i = 0; i < bt.B; ++i) loss += row_loss[i];
  return loss * scale;
}

// dW1 = Xb^T dA1 and db1 = colsum(dA1), gene-major: row j of dW1 is the sum of
// x * dA1[slot] over this batch's cells that express gene j. dA1 (B x 256)
// stays in cache, each dW1 row is written once by one thread, and a gene that
// is absent from the batch gets an exact zero row.
void sparse_weight_grad_l1(const Batch& bt, const double* dA1, u64 H, double* gW, double* gb) {
  const Corpus& X = *bt.X;
  const u64 D = X.n_genes, NB = X.n_batches, b = bt.index;
  const std::uint32_t* bptr = X.bptr.data();
  const std::uint16_t* slot = X.bslot.get();
  const double* val = X.bval.get();
  parallel_for((i64)D, 64, [&](i64 lo, i64 hi) {
    for (i64 j = lo; j < hi; ++j) {
      double* g = gW + u64(j) * H;
      for (u64 k = 0; k < H; ++k) g[k] = 0.0;
      const u64 t0 = bptr[u64(j) * NB + b], t1 = bptr[u64(j) * NB + b + 1];
      for (u64 t = t0; t < t1; ++t) {
        const double x = val[t];
        const double* d = dA1 + u64(slot[t]) * H;
        for (u64 k = 0; k < H; ++k) g[k] += x * d[k];
      }
    }
  });
  for (u64 k = 0; k < H; ++k) gb[k] = 0.0;
  for (u64 i = 0; i < bt.B; ++i)
    for (u64 k = 0; k < H; ++k) gb[k] += dA1[i * H + k];
}

// ---------------------------------------------------------------------------
// One batch: forward, loss, and (when with_grad) backward. Every buffer comes
// from the Workspace allocated once per solve ([T5]); the gradients land in
// ws.g. Nothing is zero-filled: each kernel writes all of its output.
// ---------------------------------------------------------------------------
double batch_pass(const Model& M, const Batch& bt, Workspace& ws, bool with_grad) {
  const u64 B = bt.B, D = bt.X->n_genes, H = M.l1.out;
  Grads* g = &ws.g;
  double* A1 = ws.A1.data();
  double* H1 = ws.H1.data();
  double* Z = ws.Z.data();
  double* A3 = ws.A3.data();
  double* H3 = ws.H3.data();
  double* Xh = ws.Xh.data();
  double* dXh = ws.dXh.data();
  double* dA3 = ws.dA3.data();
  double* dZ = ws.dZ.data();
  double* dA1 = ws.dA1.data();

  {
    PHASE(kFwdL1);
    sparse_forward_l1(bt, M.l1, A1);
    relu(A1, H1, B * H);
  }
  {
    PHASE(kFwdMid);
    affine(H1, B, M.l2, Z);
    affine(Z, B, M.l3, A3);
    relu(A3, H3, B * H);
  }
  {
    PHASE(kFwdL4);
    affine_blocked(H3, B, M.l4, Xh);
  }
  double loss;
  {
    PHASE(kLoss);
    loss = loss_and_dxh(bt, D, Xh, dXh, ws.row_loss.data());
  }
  if (!with_grad) return loss;

  {
    PHASE(kBwdL4dW);
    weight_grad_blocked(H3, dXh, B, H, D, g->W4.data(), g->b4.data());
  }
  {
    // W4 as it is now: the Adam step for this batch has not happened yet.
    PHASE(kBwdL4T);
    transpose_weights(M.l4, ws.w4t.data());
  }
  {
    PHASE(kBwdL4dA);
    input_grad_t(dXh, B, D, H, ws.w4t.data(), dA3);
    relu_mask(dA3, A3, B * H);
  }
  {
    PHASE(kBwdMid);
    const u64 K = M.l2.out;
    weight_grad(Z, dA3, B, K, H, g->W3.data(), g->b3.data());
    input_grad(dA3, B, M.l3, dZ);
    weight_grad(H1, dZ, B, H, K, g->W2.data(), g->b2.data());
    input_grad(dZ, B, M.l2, dA1);
    relu_mask(dA1, A1, B * H);
  }
  {
    PHASE(kBwdL1);
    sparse_weight_grad_l1(bt, dA1, H, g->W1.data(), g->b1.data());
  }
  return loss;
}

// One Adam step over parameters [lo, hi) held as separate arrays: w, its
// moments m and v, and the gradient g. Each is one contiguous stream ([T4]).
// A plain function with __restrict parameters rather than code inside the
// parallel_for lambda: GCC loses the no-alias guarantee through the lambda's
// captures and then will not vectorise the loop.
//
// On step 1 (first == true) m and v have never been written. With m = v = 0
// the general update is
//     m = beta1 * 0 + (1 - beta1) g      v = beta2 * 0 + (1 - beta2) g g,
// which is (1 - beta1) g + 0.0 and (1 - beta2) g g + 0.0 exactly: adding
// beta1 * 0 = +0 changes nothing except turning -0 into +0, which + 0.0 also
// does, and an FMA contraction of either form rounds the same product once.
// So step 1 writes m and v without reading them, and they are never zeroed.
template <bool kFirst>
void adam_range(double* __restrict w, double* __restrict m, double* __restrict v,
                const double* __restrict grad, i64 lo, i64 hi, double beta1, double beta2,
                double lr, double eps, double bc1, double bc2) {
  for (i64 i = lo; i < hi; ++i) {
    const double g = grad[i];
    const double mi = kFirst ? (1.0 - beta1) * g + 0.0 : beta1 * m[i] + (1.0 - beta1) * g;
    const double vi = kFirst ? (1.0 - beta2) * g * g + 0.0 : beta2 * v[i] + (1.0 - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    const double mhat = mi / bc1;
    const double vhat = vi / bc2;
    w[i] -= lr * mhat / (std::sqrt(std::fabs(vhat)) + eps);  // fabs: no errno branch, see [T4]
  }
}

void adam(double* w, double* m, double* v, const double* grad, u64 n, const hpcbench::Params& p,
          double bc1, double bc2, bool first) {
  const double beta1 = p.beta1, beta2 = p.beta2, lr = p.lr, eps = p.eps;
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    if (first)
      adam_range<true>(w, m, v, grad, lo, hi, beta1, beta2, lr, eps, bc1, bc2);
    else
      adam_range<false>(w, m, v, grad, lo, hi, beta1, beta2, lr, eps, bc1, bc2);
  });
}

void adam_layer(Layer& L, const Buffer<double>& gW, const Buffer<double>& gb,
                const hpcbench::Params& p, double bc1, double bc2, bool first) {
  adam(L.W, L.mW.data(), L.vW.data(), gW.data(), L.in * L.out, p, bc1, bc2, first);
  adam(L.b, L.mb.data(), L.vb.data(), gb.data(), L.out, p, bc1, bc2, first);
}

#ifdef HPCBENCH_GRADCHECK
// The analytic gradient is in ws.g; the forward-only passes that the finite
// differences need write the activations and Xh/dXh only, never ws.g.
void check_layer(const char* wname, const char* bname, Layer& L, const Buffer<double>& gW,
                 const Buffer<double>& gb, const Model& M, const Batch& bt, Workspace& ws,
                 hpcbench::GradCheckReport& total) {
  auto loss = [&]() { return batch_pass(M, bt, ws, false); };
  for (int which = 0; which < 2; ++which) {
    double* P = which == 0 ? L.W : L.b;
    const u64 n = which == 0 ? L.in * L.out : L.out;
    const Buffer<double>& G = which == 0 ? gW : gb;
    const hpcbench::GradCheckReport r = hpcbench::gradcheck(
        which == 0 ? wname : bname, n, G.data(), [&](std::size_t i) { return P[i]; },
        [&](std::size_t i, double v) { P[i] = v; }, loss);
    total.checked += r.checked;
    total.failed += r.failed;
    if (r.max_rel_err > total.max_rel_err) total.max_rel_err = r.max_rel_err;
  }
}
#endif

}  // namespace

void hpcbench::solve(const Work& w, const Params& p, Result& out) {
#ifdef P2_PROFILE
  const auto t_solve = std::chrono::steady_clock::now();
#endif
#ifndef P2_SERIAL
  TeamScope threads(kThreads);  // workers live until solve returns
#endif

  // On the clock: map the CSR written by prepare and regroup it by batch.
  const Corpus X(w, p);
  const u64 n_cells = X.n_cells, D = X.n_genes;

  const u64 H = p.hidden, K = p.latent;
  Model M;
  {
    PHASE(kLayers);
    // Trained in place in the host's output arrays ([T4]).
    M.l1 = make_layer(p.W1, p.b1, out.W1, out.b1, D, H);
    M.l2 = make_layer(p.W2, p.b2, out.W2, out.b2, H, K);
    M.l3 = make_layer(p.W3, p.b3, out.W3, out.b3, K, H);
    M.l4 = make_layer(p.W4, p.b4, out.W4, out.b4, H, D);
  }
  // Every per-batch buffer, once. No page is touched here: new T[n] does not
  // initialise doubles, so this phase costs microseconds and the page faults
  // happen in parallel inside the first batch's kernels.
  const u64 B_max = std::min<u64>(p.batch, n_cells);
  std::unique_ptr<Workspace> ws;
  {
    PHASE(kAlloc);
    ws.reset(new Workspace(B_max, D, H, K));
  }
  Grads& g = ws->g;

  double beta1_t = 1.0, beta2_t = 1.0, loss_sum = 0.0;
  u64 steps = 0;
  for (u64 start = 0; start < n_cells; start += p.batch) {
    const u64 B = start + p.batch <= n_cells ? p.batch : n_cells - start;
    const Batch bt{&X, p.order + start, B, steps};

    loss_sum += batch_pass(M, bt, *ws, true);

#ifdef HPCBENCH_GRADCHECK
    if (steps == 0) {
      hpcbench::GradCheckReport total;
      check_layer("W1", "b1", M.l1, g.W1, g.b1, M, bt, *ws, total);
      check_layer("W2", "b2", M.l2, g.W2, g.b2, M, bt, *ws, total);
      check_layer("W3", "b3", M.l3, g.W3, g.b3, M, bt, *ws, total);
      check_layer("W4", "b4", M.l4, g.W4, g.b4, M, bt, *ws, total);
      std::fprintf(stderr, "gradcheck: %zu weights checked, %zu mismatched, max rel err %.2e\n",
                   total.checked, total.failed, total.max_rel_err);
    }
#endif

    const bool first = steps == 0;  // m and v not written yet
    ++steps;
    beta1_t *= p.beta1;
    beta2_t *= p.beta2;
    const double bc1 = 1.0 - beta1_t, bc2 = 1.0 - beta2_t;
    {
      PHASE(kAdam);
      adam_layer(M.l1, g.W1, g.b1, p, bc1, bc2, first);
      adam_layer(M.l2, g.W2, g.b2, p, bc1, bc2, first);
      adam_layer(M.l3, g.W3, g.b3, p, bc1, bc2, first);
      adam_layer(M.l4, g.W4, g.b4, p, bc1, bc2, first);
    }
  }
  std::fprintf(stderr, "epoch: %llu steps, mean batch loss %.9f\n", (unsigned long long)steps,
               loss_sum / double(steps));

  // The answer is already in out.W1 ... out.b4: the parameters were trained
  // there ([T4]), so there is nothing to copy back.

#ifdef P2_PROFILE
  const double total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_solve)
          .count();
  double sum = 0.0;
  for (int i = 0; i < kNumPhases; ++i) sum += g_phase_ms[i];
  std::fprintf(stderr, "profile: %d threads\n", kThreads);
  std::fprintf(stderr, "profile: %-30s %10s %7s\n", "phase", "ms", "share");
  for (int i = 0; i < kNumPhases; ++i)
    std::fprintf(stderr, "profile: %-30s %10.2f %6.1f%%\n", kPhaseName[i], g_phase_ms[i],
                 100.0 * g_phase_ms[i] / total_ms);
  std::fprintf(stderr, "profile: %-30s %10.2f %6.1f%%\n", "(not in any phase)", total_ms - sum,
               100.0 * (total_ms - sum) / total_ms);
  std::fprintf(stderr, "profile: %-30s %10.2f\n", "solve total", total_ms);
#endif
}
