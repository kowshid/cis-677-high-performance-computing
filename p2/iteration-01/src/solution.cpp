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
// Changes from the baseline:
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
//
// Kept on purpose, so each later idea can be measured on its own:
//   Adam state as an array of structs {w, m, v}, every buffer allocated and
//   zero-filled per batch, the dense layer-4 products, no explicit SIMD.
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

// The thread budget for this task is 4.
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

// Element-wise loops are handed out in pieces of this many elements.
constexpr i64 kElemChunk = 16384;

inline i64 ceil_div(i64 a, i64 b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Optional per-phase timing (-DP2_PROFILE). Compiles to nothing otherwise.
// ---------------------------------------------------------------------------
#ifdef P2_PROFILE
enum Phase {
  kSetup, kLayers, kAlloc, kFwdL1, kFwdMid, kFwdL4, kLoss, kBwdL4, kBwdMid, kBwdL1, kAdam,
  kCopyOut, kNumPhases
};
const char* const kPhaseName[kNumPhases] = {
    "setup: threads, map, buckets", "setup: make_layer (AoS)",  "alloc + zero-fill per batch",
    "fwd L1 sparse + relu",         "fwd L2, L3 + relu",        "fwd L4 dense",
    "loss + dXh",                   "bwd L4: dW4, dA3, mask",   "bwd L3, L2",
    "bwd L1 sparse: dW1, db1",      "adam",                     "copy out"};
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

// n threads: the caller is thread 0, and n - 1 workers are created once. Each
// run() wakes them, runs the same job on every thread, and returns when all
// are done. Between jobs a worker spins briefly (jobs follow each other within
// microseconds), then yields, then sleeps on a condition variable.
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
// Model and optimiser state: unchanged from the baseline (AoS, idea 5 later).
// ---------------------------------------------------------------------------

// One parameter and its two Adam moments, side by side.
struct AdamParam {
  double w;
  double m;
  double v;
};

struct Layer {
  u64 in = 0;
  u64 out = 0;
  std::vector<AdamParam> W;  // in x out, row-major
  std::vector<AdamParam> b;  // out
};

Layer make_layer(const double* W, const double* b, u64 in, u64 out) {
  Layer L;
  L.in = in;
  L.out = out;
  for (u64 i = 0; i < in * out; ++i) L.W.push_back(AdamParam{W[i], 0.0, 0.0});
  for (u64 j = 0; j < out; ++j) L.b.push_back(AdamParam{b[j], 0.0, 0.0});
  return L;
}

struct Model {
  Layer l1, l2, l3, l4;
};

struct Grads {
  std::vector<double> W1, b1, W2, b2, W3, b3, W4, b4;
};

// ---------------------------------------------------------------------------
// Dense kernels, threaded. Each block of the output belongs to one thread, and
// inside a block the arithmetic runs in the baseline's order.
// ---------------------------------------------------------------------------

// C = A W + b, with A rows x in and C rows x out. Blocks are (column tile,
// row tile). For the wide layer 4 a thread takes a whole column tile (all row
// tiles) at once, so each piece of W4 is read from memory once.
void affine(const double* A, u64 rows, const Layer& L, double* C) {
  const i64 in = (i64)L.in, out = (i64)L.out, R = (i64)rows;
  const AdamParam* W = L.W.data();
  const AdamParam* b = L.b.data();
  const i64 n_ct = ceil_div(out, kTileCols), n_rt = ceil_div(R, kTileRows);
  const i64 chunk = n_ct >= 2 * kThreads ? n_rt : 1;
  parallel_for(n_ct * n_rt, chunk, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const i64 ct = task / n_rt, rt = task % n_rt;
      const i64 j0 = ct * kTileCols, j1 = std::min(out, j0 + kTileCols);
      const i64 i0 = rt * kTileRows, i1 = std::min(R, i0 + kTileRows);
      for (i64 i = i0; i < i1; ++i) {
        double* c = C + i * out;
        for (i64 j = j0; j < j1; ++j) c[j] = b[j].w;
        for (i64 k = 0; k < in; ++k) {
          const double a = A[i * in + k];
          const AdamParam* wk = W + k * out;
          for (i64 j = j0; j < j1; ++j) c[j] += a * wk[j].w;
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
// Blocks are (column tile, k tile) of gW; for the wide layer 4 a thread takes
// a whole column tile, so each piece of dC is read from memory once. gW and gb
// must already have the right size; every element is overwritten.
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
  const AdamParam* W = L.W.data();
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
            const AdamParam* wk = W + k * OUT;
            double s = dA[i * IN + k];
            for (i64 j = j0; j < j1; ++j) s += d[j] * wk[j].w;
            dA[i * IN + k] = s;
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
  const AdamParam* W = L.W.data();
  const AdamParam* b = L.b.data();
  parallel_for((i64)bt.B, 4, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; ++i) {
      double* a = A1 + u64(i) * H;
      for (u64 k = 0; k < H; ++k) a[k] = b[k].w;
      const u64 cell = bt.cells[i];
      for (u64 t = X.row_ptr[cell]; t < X.row_ptr[cell + 1]; ++t) {
        const double x = X.val[t];
        const AdamParam* wj = W + u64(X.col[t]) * H;
        for (u64 k = 0; k < H; ++k) a[k] += x * wj[k].w;
      }
    }
  });
}

// dXh = 2 (Xh - Xb) / (B D). Xh - Xb equals Xh wherever Xb is zero, so only
// the non-zeros are subtracted (in place: Xh is not needed afterwards).
// Returns the batch loss, which is only printed. Each row's share is kept and
// summed in row order afterwards, so the printed loss does not depend on the
// threads.
double loss_and_dxh(const Batch& bt, u64 D, double* Xh, double* dXh) {
  const Corpus& X = *bt.X;
  const double scale = 1.0 / (double(bt.B) * double(D));
  std::vector<double> row_loss(bt.B, 0.0);
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
  for (double s : row_loss) loss += s;
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
// One batch: forward, loss, and (when g is given) backward.
// ---------------------------------------------------------------------------
double batch_pass(const Model& M, const Batch& bt, Grads* g) {
  const u64 B = bt.B, D = bt.X->n_genes, H = M.l1.out, K = M.l2.out;

  // Every buffer, fresh for every batch, as in the baseline (idea 5 later).
  std::vector<double> A1, H1, Z, A3, H3, Xh, dXh, dA3, dZ, dA1;
  {
    PHASE(kAlloc);
    A1.assign(B * H, 0.0);
    H1.assign(B * H, 0.0);
    Z.assign(B * K, 0.0);
    A3.assign(B * H, 0.0);
    H3.assign(B * H, 0.0);
    Xh.assign(B * D, 0.0);
    dXh.assign(B * D, 0.0);
    if (g) {
      dA3.assign(B * H, 0.0);
      dZ.assign(B * K, 0.0);
      dA1.assign(B * H, 0.0);
      g->W1.assign(D * H, 0.0);
      g->b1.assign(H, 0.0);
      g->W2.assign(H * K, 0.0);
      g->b2.assign(K, 0.0);
      g->W3.assign(K * H, 0.0);
      g->b3.assign(H, 0.0);
      g->W4.assign(H * D, 0.0);
      g->b4.assign(D, 0.0);
    }
  }

  {
    PHASE(kFwdL1);
    sparse_forward_l1(bt, M.l1, A1.data());
    relu(A1.data(), H1.data(), B * H);
  }
  {
    PHASE(kFwdMid);
    affine(H1.data(), B, M.l2, Z.data());
    affine(Z.data(), B, M.l3, A3.data());
    relu(A3.data(), H3.data(), B * H);
  }
  {
    PHASE(kFwdL4);
    affine(H3.data(), B, M.l4, Xh.data());
  }
  double loss;
  {
    PHASE(kLoss);
    loss = loss_and_dxh(bt, D, Xh.data(), dXh.data());
  }
  if (!g) return loss;

  {
    PHASE(kBwdL4);
    weight_grad(H3.data(), dXh.data(), B, H, D, g->W4.data(), g->b4.data());
    input_grad(dXh.data(), B, M.l4, dA3.data());
    relu_mask(dA3.data(), A3.data(), B * H);
  }
  {
    PHASE(kBwdMid);
    weight_grad(Z.data(), dA3.data(), B, K, H, g->W3.data(), g->b3.data());
    input_grad(dA3.data(), B, M.l3, dZ.data());
    weight_grad(H1.data(), dZ.data(), B, H, K, g->W2.data(), g->b2.data());
    input_grad(dZ.data(), B, M.l2, dA1.data());
    relu_mask(dA1.data(), A1.data(), B * H);
  }
  {
    PHASE(kBwdL1);
    sparse_weight_grad_l1(bt, dA1.data(), H, g->W1.data(), g->b1.data());
  }
  return loss;
}

void adam(std::vector<AdamParam>& P, const std::vector<double>& G, const hpcbench::Params& p,
          double bc1, double bc2) {
  AdamParam* q = P.data();
  const double* grad = G.data();
  const double beta1 = p.beta1, beta2 = p.beta2, lr = p.lr, eps = p.eps;
  parallel_for((i64)P.size(), kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; ++i) {
      const double g = grad[i];
      q[i].m = beta1 * q[i].m + (1.0 - beta1) * g;
      q[i].v = beta2 * q[i].v + (1.0 - beta2) * g * g;
      const double mhat = q[i].m / bc1;
      const double vhat = q[i].v / bc2;
      q[i].w -= lr * mhat / (std::sqrt(vhat) + eps);
    }
  });
}

void put(const std::vector<AdamParam>& P, double* dst) {
  const AdamParam* q = P.data();
  parallel_for((i64)P.size(), kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; ++i) dst[i] = q[i].w;
  });
}

#ifdef HPCBENCH_GRADCHECK
void check_layer(const char* wname, const char* bname, Layer& L, const std::vector<double>& gW,
                 const std::vector<double>& gb, const Model& M, const Batch& bt,
                 hpcbench::GradCheckReport& total) {
  auto loss = [&]() { return batch_pass(M, bt, nullptr); };
  for (int which = 0; which < 2; ++which) {
    std::vector<AdamParam>& P = which == 0 ? L.W : L.b;
    const std::vector<double>& G = which == 0 ? gW : gb;
    const hpcbench::GradCheckReport r = hpcbench::gradcheck(
        which == 0 ? wname : bname, P.size(), G.data(),
        [&](std::size_t i) { return P[i].w; }, [&](std::size_t i, double v) { P[i].w = v; },
        loss);
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
    M.l1 = make_layer(p.W1, p.b1, D, H);
    M.l2 = make_layer(p.W2, p.b2, H, K);
    M.l3 = make_layer(p.W3, p.b3, K, H);
    M.l4 = make_layer(p.W4, p.b4, H, D);
  }

  double beta1_t = 1.0, beta2_t = 1.0, loss_sum = 0.0;
  u64 steps = 0;
  for (u64 start = 0; start < n_cells; start += p.batch) {
    const u64 B = start + p.batch <= n_cells ? p.batch : n_cells - start;
    const Batch bt{&X, p.order + start, B, steps};

    Grads g;
    loss_sum += batch_pass(M, bt, &g);

#ifdef HPCBENCH_GRADCHECK
    if (steps == 0) {
      hpcbench::GradCheckReport total;
      check_layer("W1", "b1", M.l1, g.W1, g.b1, M, bt, total);
      check_layer("W2", "b2", M.l2, g.W2, g.b2, M, bt, total);
      check_layer("W3", "b3", M.l3, g.W3, g.b3, M, bt, total);
      check_layer("W4", "b4", M.l4, g.W4, g.b4, M, bt, total);
      std::fprintf(stderr, "gradcheck: %zu weights checked, %zu mismatched, max rel err %.2e\n",
                   total.checked, total.failed, total.max_rel_err);
    }
#endif

    ++steps;
    beta1_t *= p.beta1;
    beta2_t *= p.beta2;
    const double bc1 = 1.0 - beta1_t, bc2 = 1.0 - beta2_t;
    {
      PHASE(kAdam);
      adam(M.l1.W, g.W1, p, bc1, bc2);
      adam(M.l1.b, g.b1, p, bc1, bc2);
      adam(M.l2.W, g.W2, p, bc1, bc2);
      adam(M.l2.b, g.b2, p, bc1, bc2);
      adam(M.l3.W, g.W3, p, bc1, bc2);
      adam(M.l3.b, g.b3, p, bc1, bc2);
      adam(M.l4.W, g.W4, p, bc1, bc2);
      adam(M.l4.b, g.b4, p, bc1, bc2);
    }
  }
  std::fprintf(stderr, "epoch: %llu steps, mean batch loss %.9f\n", (unsigned long long)steps,
               loss_sum / double(steps));

  // The answer: the trained parameters, back in the host's layout.
  {
    PHASE(kCopyOut);
    put(M.l1.W, out.W1);
    put(M.l1.b, out.b1);
    put(M.l2.W, out.W2);
    put(M.l2.b, out.b2);
    put(M.l3.W, out.W3);
    put(M.l3.b, out.b3);
    put(M.l4.W, out.W4);
    put(M.l4.b, out.b4);
  }

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