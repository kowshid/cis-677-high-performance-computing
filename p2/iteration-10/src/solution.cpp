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
//
// Since [GRAM] (iteration-08) the layer-4 lines are computed WITHOUT forming
// Xh or dXh, by exact algebra (c = 2 / (B * n_genes), W4T = W4^T):
//                G = W4T^T W4T     u = W4T^T b4            256 x 256, 256
//                S = H3^T H3       s = colsum(H3)          256 x 256, 256
//                P = Xb W4T        Q = Xb^T H3             B x 256, n_genes x 256
//                dA3  = c (H3 G + 1 u^T - P) * [A3 > 0]
//                dW4T = c (W4T S + b4 s^T - Q)
//                db4  = c (W4T s + B b4 - colsum(Xb)^T)
// ---------------------------------------------------------------------------
//
// Changes from the baseline: ideas 1, 3, 7 and B (iteration-02), T5, T4 and
// T2 (iteration-06; first measured one at a time as iteration-03, -04, -05),
// W4T (iteration-07), GRAM (iteration-08), FUSE (iteration-09) and PREC
// (iteration-10). Only src/solution.cpp differs from iteration-02;
// src/prepare.cpp, build.sh, the harness and include/ are unchanged. M4
// numbers below are from the collect.sh profiles of iteration-02 .. -09.
//
//   [1] X is never dense. prepare writes a CSR file, solve maps it with mmap.
//       * Layer-1 forward A1 = Xb W1 was the Project 1 kernel: one CSR row
//         of the cell times the dense W1 (gene-major since [T2]).
//       * dW1 = Xb^T dA1 is computed gene-major: at the start of solve the
//         non-zeros are regrouped by batch and gene (the batches depend on
//         the seed, so this is on the clock, O(nnz)), and dW1 row j is the sum
//         of x * dA1[slot] over the batch's cells that express gene j.
//       * (dXh used to subtract Xb only at the non-zeros; since [GRAM] there
//         is no dXh, and Xb enters layer 4 only through sparse products.)
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
//       (Superseded by [GRAM]: dA3 no longer reads dXh, which is not formed.)
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
//       (Since [GRAM] those three products are gone; the same micro-kernel
//       now computes the Gram products G, S, H3 G and W4T S.)
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
//       to iteration-02. On the M4 at 4 threads the two serial phases went
//       from 80 + 64 ms to 13 + 0 ms.
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
//         (in parallel), and the copy-out at the end is gone. (Until [PREC]:
//         since then the layers own fp32 copies, converted in and out.)
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
//       bit-identical to iteration-02. With default flags the compiler may
//       fuse the other product of beta1*m + (1-beta1)*g into an FMA once the
//       loop vectorises, which moves the 17th significant digit of the loss
//       (on the M4: identical at seed 677, 1.6e-16 relative at seed 12345).
//       On the M4: layer-1 forward 839 -> 261 ms and Adam 179 -> 142 ms at
//       1 thread. At 4 threads Adam stayed ~110 ms: it is bandwidth-bound,
//       and it moves the same 56 bytes per parameter (w, m, v, g read; w, m,
//       v written) whether the state is interleaved or separate. SoA pays
//       in the kernels that read only w.
//
//   [T2] Loop order matches storage order for layer 1 (Lecture 4 Target 2).
//       Measured in iteration-02: the layer-1 forward and dW1 do the same
//       0.58 G multiply-adds per epoch, but took 317 ms and 71 ms. The
//       forward was cell-major: each cell gathered its ~850 rows of W1
//       (2 KB each, from a 28 MB matrix that does not fit in L2), so ~440 MB
//       of scattered row reads per batch. dW1 was gene-major: each gene's
//       row touched once, the B x 256 operand kept in cache.
//       * The forward is now gene-major too. For each gene present in the
//         batch, its W1 row is read once and x * W1[j] is added into row
//         `slot` of A1 for every cell of the batch that expresses gene j.
//         A1 (256 x 256, 512 KB) stays in cache; W1 streams through once.
//       * No two threads write the same element: the 256 hidden columns
//         are split into one block per thread (64 at 4 threads), and a
//         block's owner computes those columns of A1 for every cell. Bias,
//         the sum over genes and the ReLU are all done by the owner, so H1
//         needs no separate pass. Every block has the same work, so equal
//         blocks are balanced by construction.
//       * Each A1[i][k] still starts from b1[k] and adds its genes in
//         ascending gene order, exactly the CSR order: same sums, same
//         order, bit-identical.
//       * The buckets are now stored batch-major (batch, gene, slot)
//         instead of (gene, batch, slot). Every kernel walks one batch at a
//         time, gene by gene; before, a batch's non-zeros were scattered in
//         13,714 small pieces across 23 MB, now they are one contiguous
//         2.2 MB run. Same entries in the same order within each gene, so
//         dW1 is unchanged bit for bit; it just reads them sequentially.
//       On the M4: layer-1 forward 261 -> 175 ms at 1 thread (800 ms in
//       iteration-02, with the AoS gather). At 4 threads it measured only
//       160 ms (1.1x over 1 thread) on a busy machine: one block per thread
//       leaves no slack for a slow core, and every thread re-reads all of
//       the batch's non-zeros. To re-check on a quiet machine.
//
//   [W4T] W4 is kept transposed for the whole epoch (extends [3]; Lecture 4
//       Target 3). Layer 4's weights live as W4T, n_genes x 256 row-major,
//       the same layout as W1: one row of 256 weights per gene.
//       * Before, W4 was stored like the host's (256 x n_genes) and copied
//         into a transposed W4T every batch for dA3 (28 MB read + 28 MB
//         written, 11 times; 16 ms at 4 threads on the M4 in iteration-06).
//         Now W4T is the only copy: dA3 reads it directly, the forward
//         product packs its 8-column strips from 8 contiguous rows of W4T,
//         and dW4 is produced directly as dW4T (same layout), so Adam, which
//         is element-wise, updates W4T with no layout change at all.
//       * The cost moves to two transposes per solve: p.W4 -> W4T before the
//         first batch and W4T -> out.W4 after the last (in "setup: params
//         in" and "copy out").
//       * Same values, same order of additions in every product: the
//         trained weights are bit-identical to iteration-06.
//       * This is the layout the Gram-matrix backward ([GRAM]) needs: with
//         one gene per row, every layer-4 gradient becomes row-local.
//
//   [GRAM] Stop computing the reconstruction: the layer-4 backward from Gram
//       matrices. Measured in iteration-06 at 4 threads on the M4, layer 4
//       was 708 of 895 ms (79%): three dense 256 x 256 x 13,714 products
//       (forward Xh, dW4, dA3; 0.9 G multiply-adds each per batch, 2.7 G
//       together) plus the loss pass over the 3.5 M-element Xh.
//       * Xh is only ever used to form dXh = c (Xh - Xb), with
//         Xh = H3 W4 + 1 b4^T. Substituting that into the three gradients
//         (formulas at the top) leaves no product with n_genes on both
//         sides except two, and both have 256 x 256 outputs:
//             G = W4T^T W4T   256 x 256 over 13,714 genes, symmetric, so only
//                             the upper triangle: ~0.47 G multiply-adds
//             W4T S           13,714 x 256 by 256 x 256: 0.9 G
//         Everything else is small (S = H3^T H3, H3 G: 17 M each) or sparse
//         (P = Xb W4T and Q = Xb^T H3: 55 M each). Per batch that is ~1.5 G
//         multiply-adds instead of 2.7 G, and no B x n_genes array at all:
//         Xh, dXh (28 MB each) and the loss pass are gone.
//       * This is exact algebra, the identity the harness itself uses in
//         reconstruction_mse (G = W4 W4^T). The rounding differs from forming
//         Xh: on batch 1, 86% of the dW4T entries differ from iteration-07's,
//         by up to ~1e-15 of the gradient's scale. Yet the final loss came out
//         identical to all 17 digits at seeds 677 and 12345 (GCC, x86): Adam
//         divides each gradient by its own running magnitude (on step 1 the
//         update is lr * sign(g)), so relative noise of 1e-14 in a gradient
//         barely moves the weights.
//       * With W4 kept as W4T ([W4T]) everything is row-local per gene: row
//         j of dW4T needs row j of W4T, b4[j] and gene j's non-zeros only.
//         P is the same sparse kernel as the layer-1 forward (and runs in the
//         same pass over the batch's non-zeros), Q the same as dW1.
//       * G is a reduction over 13,714 genes. The genes are cut into fixed
//         parts of kGramRows, each part summed into its own partial matrix,
//         and the partials added in part order, so every element is the same
//         sum in the same order on any number of threads (Week 5 / Lecture 6:
//         private partial sums, fixed combining order).
//       * The batch loss is still printed: it is recomputed from the same
//         pieces, sum (Xh - Xb)^2 = tr(G S) + 2 s.u + B |b4|^2
//         - 2 (sum H3 * P + sum x b4[j]) + sum x^2, which also checks the
//         algebra against the earlier entries' printed loss at no cost.
//       * The gradient check (-DHPCBENCH_GRADCHECK) evaluates its loss the
//         direct way, from the full reconstruction Xh, so the oracle shares no
//         algebra with the Gram-matrix gradients it is checking.
//
//   [FUSE] Each gene's gradient rows are fused with their Adam update.
//       After [GRAM], every big gradient is row-local per gene: row j of dW1
//       needs gene j's non-zeros and dA1; row j of dW4T needs row j of W4T,
//       b4[j], gene j's non-zeros, H3 and S; db4[j] likewise. Before, each
//       was written to a full-size gradient array (dW1, dW4T: 28 MB each),
//       and a separate Adam pass read them back: 56 MB written and 56 MB
//       read per batch only to hand numbers from one loop to the next.
//       * Now one pass over blocks of kGeneBlk genes does, per block: walk
//         the genes' non-zeros once, building the dW1 rows and the Q rows
//         together; form the dW4T rows (the W4T S tiles of [GRAM]) and db4;
//         then apply Adam to those rows of W1, W4T and b4 while they are
//         still in cache. The gradients exist only as a small per-thread
//         block (3 x 32 x 256 doubles), never as full arrays.
//       * Safe to update in place: a block reads only its own rows of W4T
//         and its own b4 entries, and G, u, P and dA3 (which needed the old
//         W4T) were all formed earlier in the batch.
//       * Same sums in the same order and the same Adam arithmetic as
//         iteration-08: the trained weights are the same (up to which
//         product the compiler fuses into an FMA).
//       * The gradient check needs the gradients before Adam changes the
//         weights, so the check build runs the same pass once with the
//         update switched off (writing the rows into gradient arrays that
//         exist only in that build), checks them, recomputes the batch (the
//         check's forward passes leave perturbed activations behind), and
//         then trains.
//
//   [PREC] fp32 working precision, with fp64 wherever a decision is made
//       (Lecture 4 Target 7; Lecture 5's tolerance study). The weights and
//       biases, the Adam moments, the input values, the activations, the
//       Gram matrices and every gradient are fp32, and so is the arithmetic
//       of every product and of Adam. Half the bytes everywhere, and twice
//       the multiply-adds per vector instruction: the micro-kernel's tile
//       becomes 4 x 16 in the same 8 vector registers. Measured at 1 thread
//       (GCC, x86) against the same code in fp64: 2.80 s -> 1.50 s; G and
//       the fused pass 2.0-2.1x faster, the sparse forward 1.5x.
//       * A plain all-fp32 build was NOT safe. It passed the public matrix
//         (<= 1.9e-6 relative to fp64 over 12 seeds), but on a holdout-style
//         matrix (make_p1_input.py --variant holdout: thinned counts, rows
//         and columns permuted, like the gradebook's input) seed 12345 came
//         out 3.8e-5 from fp64, outside the 3e-5 tolerance. Diagnosis, by
//         injecting fp32 behaviour into the fp64 build one piece at a time:
//         storing X, weights, moments, gradients or activations in fp32
//         moved the loss by <= 2e-9, and the fp64 entries with different
//         summation orders agree to 1e-16, so the trajectory is not chaotic
//         and storage is not the problem. Two operations were:
//       * (1) Adam's 1 - beta2. In fp32, 0.999 is stored as 0.99900001287,
//         and 1 - beta2 cancels three digits: 0.00099998713, 1.3e-5 too
//         small. That biased v, and so every Adam step, by ~6e-6 (which is
//         also why plain fp32 always came out slightly low). 1 - beta1 and
//         1 - beta2 are now formed in fp64 and rounded once (error 4.7e-8).
//       * (2) ReLU decisions. Summing only A1 = Xb W1 + b1 (~850 terms) in
//         fp32 reproduced the full 3.7e-5 by itself: a few pre-activations
//         lie within fp32 rounding of zero and land on the other side,
//         which switches that unit's gradient on or off, a discrete change
//         the rest of the epoch carries forward. The same path ran through
//         Z -> A3 (holdout seed 2023896566: 7.3e-6). So every ReLU decision
//         is made in fp64: A1, Z and A3 are summed in fp64 from fp32 inputs,
//         and A1 and A3 are kept in fp64 for the masks. W1 and x are still
//         read as fp32, so the sparse pass still moves half the bytes.
//       * Result, fp32 vs the same code in fp64 (GCC, x86): public matrix,
//         12 seeds: <= 4.5e-8 relative (median 3.0e-8). Holdout-style
//         matrix, 84 seeds: median 2.5e-8, all <= 7.9e-8 except seed 12345
//         at 1.0e-6, 30x inside the tolerance. That seed is a near-tie:
//         fp32 noise from any one of the Gram products, the sparse sums or
//         the small layers' backward alone gives the same ~1e-6.
//       * The gradient check builds in fp64: its finite differences need
//         the loss to ~1e-12, which fp32 cannot give, and it validates the
//         algorithm. The precision is validated by the sweep above.
//       * The host's arrays are fp64, so the layers own fp32 copies:
//         converted in before the first batch, converted (and W4T
//         transposed) back into out.* after the last.
//
// Kept on purpose, so each later idea can be measured on its own:
//   no intrinsics (only the generic vector type in [B]), the micro-kernel's
//   4 x 2-vector tile the same on every instruction set, S and W4T read in
//   place by the micro-kernel (the 2 KB stride noted after iteration-08).
//
// In fp64 (-DP2_FP64) every dot product and accumulation runs in the same
// order as the baseline (up to [GRAM]'s algebra), so the trained weights agree
// with it to the last bits. In fp32 they agree to ~3e-8 relative; see [PREC].
//
// Compile-time switches (pass them through CXXFLAGS; build.sh is unchanged):
//   -DP2_PROFILE          per-phase timing table on stderr
//   -DP2_THREADS=n        use n threads instead of 4, to measure scaling
//   -DHPCBENCH_GRADCHECK  the host's finite-difference gradient check (always fp64)
//   -DP2_FP64             fp64 working precision: iteration-09's arithmetic ([PREC])
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

// [PREC] The working precision of every weight, moment, activation and
// product. fp32 by default; -DP2_FP64 builds the same code in fp64, which is
// iteration-09's arithmetic. The gradient-check build is always fp64: its
// finite differences (h = 1e-5) need the batch loss to ~1e-12, which fp32
// cannot give, and the check validates the algorithm, not the precision.
#if defined(P2_FP64) || defined(HPCBENCH_GRADCHECK)
using real = double;
#define P2_PRECISION "fp64"
#else
using real = float;
#define P2_PRECISION "fp32"
#endif

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
constexpr i64 kDaRows = 16;      // batch rows per task in dA3_from_gram

// Register blocking (idea B): the micro-kernel's tile of the result. A row
// of the tile is two 32-byte vectors: 8 columns in fp64, 16 in fp32 ([PREC]),
// so the same registers hold twice the sums in fp32.
constexpr i64 kVecBytes = 32;                           // one AVX / two NEON registers
constexpr i64 kLanes = kVecBytes / i64(sizeof(real));  // 4 (fp64) or 8 (fp32)
constexpr i64 kMR = 4;                                  // micro-tile rows
constexpr i64 kNR = 2 * kLanes;                         // micro-tile columns: 8 or 16

// Gram products ([GRAM]).
constexpr i64 kGramRows = 512;   // rows (genes) per fixed part of G = W4T^T W4T: 27 parts
constexpr i64 kGramKc = 64;      // rows per cache sub-chunk inside a part (64 x 2 KB = 128 KB)
constexpr i64 kGeneBlk = 32;     // genes per task in dW4T = c (W4T S + b4 s^T - Q)

// Layer-1 forward ([T2]): hidden columns are split into one block per thread
// (256 / 4 = 64), rounded up to a multiple of this, so blocks stay aligned to
// whole vectors. Every block does exactly the same work (all of the batch's
// non-zeros, the same number of columns), so equal blocks are balanced by
// construction; smaller blocks would only make each extra block re-read the
// batch's 2.2 MB of non-zeros. Measured at 1 thread (x86): 16 columns per
// block 430 ms, 32: 350 ms, 64: 285 ms, 128: 240 ms.
constexpr u64 kFwdColAlign = 8;

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
inline real* thread_scratch(u64 n) {
  thread_local std::vector<real> buf;
  if (buf.size() < n) buf.resize(n);
  return buf.data();
}

// ---------------------------------------------------------------------------
// Optional per-phase timing (-DP2_PROFILE). Compiles to nothing otherwise.
// ---------------------------------------------------------------------------
#ifdef P2_PROFILE
enum Phase {
  kSetup, kLayers, kAlloc, kFwdL1, kFwdMid, kGramG, kGramS, kBwdL4dA, kBwdMid, kFused, kAdam,
  kCopyOut, kNumPhases
};
const char* const kPhaseName[kNumPhases] = {
    "setup: threads, map, buckets", "setup: params in, convert",  "alloc workspace (once)",
    "fwd L1 + P = Xb W4T (sparse)", "fwd L2, L3 + relu",          "L4: G = W4T'W4T, u = W4T'b4",
    "L4: S = H3'H3, s, batch loss", "L4: dA3 = c(H3 G + u - P)",  "bwd L3, L2, db1",
    "fused: dW1,dW4T,db4 + adam",   "adam: layers 2, 3, b1",      "copy out: convert, W4T->W4"};
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

// The mapped CSR plus, for every batch, its non-zeros regrouped by gene.
// Not copyable or movable: the CSR pointers point into `file`.
struct Corpus {
  hpcbench::Mapped file;

  // CSR, one row per cell (mapped, read-only).
  u64 n_cells = 0, n_genes = 0, nnz = 0;
  const u64* row_ptr = nullptr;         // n_cells + 1
  const double* val = nullptr;        // nnz
  const std::uint16_t* col = nullptr;   // nnz

  // Batch-major, gene-bucketed copy ([T2]; built in solve, because batches
  // depend on the seed). Batch b's non-zeros are the contiguous entries
  //     [bptr[b * n_genes], bptr[(b + 1) * n_genes])
  // of bslot / bval, grouped by gene in ascending order; gene j's share is
  //     [bptr[b * n_genes + j], bptr[b * n_genes + j + 1])
  // in ascending slot order, where slot is the cell's row inside its batch.
  // bptr has n_batches * n_genes + 1 entries.
  u64 n_batches = 0;
  std::vector<std::uint32_t> bptr;
  std::unique_ptr<std::uint16_t[]> bslot;
  std::unique_ptr<real[]> bval;

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

  // --- Regroup the non-zeros by (batch, gene, slot). ---
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

  // Exclusive prefix sum in (batch, gene) order, the order cnt is already
  // in. cnt becomes the write cursor.
  bptr.resize(NB * D + 1);
  std::uint32_t run = 0;
  for (u64 e = 0; e < NB * D; ++e) {
    bptr[e] = run;
    const std::uint32_t n = cnt[e];
    cnt[e] = run;
    run += n;
  }
  bptr[NB * D] = run;

  // Place. Walking each batch's cells in the host's order puts every bucket in
  // ascending slot order, which is the order the baseline sums dW1 in. Each
  // batch writes only its own contiguous 2.2 MB region.
  bslot.reset(new std::uint16_t[nnz]);
  bval.reset(new real[nnz]);
  std::uint16_t* slot_out = bslot.get();
  real* val_out = bval.get();
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
// Model and optimiser state: struct of arrays ([T4]) in the working
// precision ([PREC]), allocated once ([T5]).
// ---------------------------------------------------------------------------

struct Layer {
  u64 in = 0;
  u64 out = 0;
  real* W = nullptr;    // in x out, row-major (layer 4: W4T, out x in -- see make_layer_t)
  real* b = nullptr;    // out
  Buffer<real> own;     // the weights, in the working precision ([PREC])
  Buffer<real> bown;    // the bias, likewise
  Buffer<real> mW, vW;  // Adam moments of W, same layout as W
  Buffer<real> mb, vb;  // Adam moments of b
};

// dst[i] = T(src[i]) for i < n, by all threads: the conversions between the
// host's fp64 arrays and the working precision ([PREC]).
template <class S, class T>
void convert_into(const S* src, u64 n, T* dst) {
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 i = lo; i < hi; ++i) dst[i] = T(src[i]);
  });
}

// dst (cols x rows) = src (rows x cols)^T, converted to T on the way, in
// 32 x 32 tiles (both the reads and the writes of a tile stay in L1). Tasks
// are rows of tiles, handed out to the threads.
template <class S, class T>
void transpose_into(const S* src, u64 rows, u64 cols, T* dst) {
  constexpr i64 kT = 32;
  const i64 R = (i64)rows, C = (i64)cols;
  parallel_for(ceil_div(R, kT), 1, [&](i64 lo, i64 hi) {
    for (i64 bi = lo; bi < hi; ++bi) {
      const i64 i0 = bi * kT, i1 = std::min(R, i0 + kT);
      for (i64 j0 = 0; j0 < C; j0 += kT) {
        const i64 j1 = std::min(C, j0 + kT);
        for (i64 i = i0; i < i1; ++i)
          for (i64 j = j0; j < j1; ++j) dst[j * R + i] = T(src[i * C + j]);
      }
    }
  });
}

// The host's fp64 weights W0 (in x out) and b0, converted into buffers this
// layer owns ([PREC]; until iteration-09 they were trained in place in the
// host's output arrays, which are fp64). Converted back at the end of solve.
// The moments are allocated but not written (step 1 writes them, see adam()).
Layer make_layer(const double* W0, const double* b0, u64 in, u64 out) {
  Layer L;
  L.in = in;
  L.out = out;
  L.own = Buffer<real>(in * out);
  L.bown = Buffer<real>(out);
  L.W = L.own.data();
  L.b = L.bown.data();
  convert_into(W0, in * out, L.W);
  convert_into(b0, out, L.b);
  L.mW = Buffer<real>(in * out);
  L.vW = Buffer<real>(in * out);
  L.mb = Buffer<real>(out);
  L.vb = Buffer<real>(out);
  return L;
}

// Layer 4, kept TRANSPOSED for the whole epoch ([W4T]): W points at WT
// (out x in = n_genes x 256); the host's W0 (in x out) is transposed (and
// converted, [PREC]) into it once, and back into out.W4 at the end of solve.
Layer make_layer_t(const double* W0, const double* b0, u64 in, u64 out) {
  Layer L;
  L.in = in;
  L.out = out;
  L.own = Buffer<real>(in * out);
  L.bown = Buffer<real>(out);
  L.W = L.own.data();
  L.b = L.bown.data();
  transpose_into(W0, in, out, L.W);
  convert_into(b0, out, L.b);
  L.mW = Buffer<real>(in * out);
  L.vW = Buffer<real>(in * out);
  L.mb = Buffer<real>(out);
  L.vb = Buffer<real>(out);
  return L;
}

struct Model {
  Layer l1, l2, l3, l4;  // l4.W is W4T ([W4T])
};

// The gradients that exist as arrays. Since [FUSE], dW1, dW4T and db4 exist
// only inside the fused pass, a block at a time; the arrays below for them
// are allocated only in the gradient-check build.
struct Grads {
  Buffer<real> W1, b1, W2, b2, W3, b3, W4, b4;  // W4 holds dW4T ([W4T])
};

// Every buffer one batch needs, allocated once per solve for the largest
// batch (B_max = 256; the short last batch uses the first B rows). Since
// [GRAM] there is no B x n_genes buffer: the largest are the Gram partials.
struct Workspace {
  Buffer<double> A1, A3;           // pre-activations of the ReLUs, always fp64 ([PREC])
  Buffer<real> H1, Z, H3;          // forward activations
  Buffer<real> P;                  // Xb W4T, B x 256 ([GRAM])
  Buffer<real> G, u, S, s;         // Gram matrices and vectors ([GRAM])
  Buffer<real> partG, partU;       // per-part partial sums of G, u (and S, s)
  Buffer<real> dA3, dZ, dA1;       // backward activation gradients
  Buffer<real> row_loss;           // per-row loss (gradient check only)
  Grads g;

  Workspace(u64 B, u64 D, u64 H, u64 K)
      : A1(B * H), A3(B * H), H1(B * H), Z(B * K), H3(B * H), P(B * H), G(H * H), u(H),
        S(H * H), s(H), partG(u64(ceil_div(i64(std::max(D, B)), kGramRows)) * H * H),
        partU(u64(ceil_div(i64(std::max(D, B)), kGramRows)) * H), dA3(B * H), dZ(B * K),
        dA1(B * H), row_loss(B) {
    g.b1 = Buffer<real>(H);
    g.W2 = Buffer<real>(H * K);
    g.b2 = Buffer<real>(K);
    g.W3 = Buffer<real>(K * H);
    g.b3 = Buffer<real>(H);
#ifdef HPCBENCH_GRADCHECK
    g.W1 = Buffer<real>(D * H);  // [FUSE]: only for the check
    g.W4 = Buffer<real>(H * D);
    g.b4 = Buffer<real>(D);
#endif
  }
};

// ---------------------------------------------------------------------------
// Dense kernels, threaded. Each block of the output belongs to one thread, and
// inside a block the arithmetic runs in the baseline's order.
// ---------------------------------------------------------------------------

// C = A W + b, with A rows x in and C rows x out. Blocks are (column tile,
// row tile). Used for the small layers 2 and 3 only; layer 4 is done by
// Gram matrices ([GRAM]).
template <class TC>
void affine(const real* A, u64 rows, const Layer& L, TC* C) {
  const i64 in = (i64)L.in, out = (i64)L.out, R = (i64)rows;
  const real* W = L.W;
  const real* b = L.b;
  const i64 n_ct = ceil_div(out, kTileCols), n_rt = ceil_div(R, kTileRows);
  const i64 chunk = n_ct >= 2 * kThreads ? n_rt : 1;
  parallel_for(n_ct * n_rt, chunk, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const i64 ct = task / n_rt, rt = task % n_rt;
      const i64 j0 = ct * kTileCols, j1 = std::min(out, j0 + kTileCols);
      const i64 i0 = rt * kTileRows, i1 = std::min(R, i0 + kTileRows);
      for (i64 i = i0; i < i1; ++i) {
        // Summed in fp64 whatever the output type ([PREC]): Z feeds A3, and
        // A3 decides the layer-3 ReLU. A tile row is at most kTileCols wide.
        double acc[kTileCols];
        for (i64 j = j0; j < j1; ++j) acc[j - j0] = double(b[j]);
        for (i64 k = 0; k < in; ++k) {
          const double a = double(A[i * in + k]);
          const real* wk = W + k * out;
          for (i64 j = j0; j < j1; ++j) acc[j - j0] += a * double(wk[j]);
        }
        TC* c = C + i * out;
        for (i64 j = j0; j < j1; ++j) c[j] = TC(acc[j - j0]);
      }
    }
  });
}

template <class TA>
void relu(const TA* A, real* H, u64 n) {
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 e = lo; e < hi; ++e) H[e] = A[e] > TA(0) ? real(A[e]) : real(0);
  });
}

// dA *= [A > 0]
template <class TA>
void relu_mask(real* dA, const TA* A, u64 n) {
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    for (i64 e = lo; e < hi; ++e)
      if (!(A[e] > TA(0))) dA[e] = real(0);
  });
}

// gW = A^T dC (in x out) and gb = colsum(dC), for A rows x in, dC rows x out.
// Blocks are (column tile, k tile) of gW. Used for the small layers 2 and 3
// only; layer 4 is done by Gram matrices ([GRAM]). gW and gb must already
// have the right size; every element is overwritten.
void weight_grad(const real* A, const real* dC, u64 rows, u64 in, u64 out, real* gW,
                 real* gb) {
  const i64 R = (i64)rows, IN = (i64)in, OUT = (i64)out;
  const i64 n_ct = ceil_div(OUT, kTileCols), n_kt = ceil_div(IN, kTileK);
  const i64 chunk = n_ct >= 2 * kThreads ? n_kt : 1;
  parallel_for(n_ct * n_kt, chunk, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const i64 ct = task / n_kt, kt = task % n_kt;
      const i64 j0 = ct * kTileCols, j1 = std::min(OUT, j0 + kTileCols);
      const i64 k0 = kt * kTileK, k1 = std::min(IN, k0 + kTileK);
      for (i64 k = k0; k < k1; ++k) {
        real* g = gW + k * OUT;
        for (i64 j = j0; j < j1; ++j) g[j] = 0.0;
        for (i64 i = 0; i < R; ++i) {
          const real a = A[i * IN + k];
          const real* d = dC + i * OUT;
          for (i64 j = j0; j < j1; ++j) g[j] += a * d[j];
        }
      }
      if (kt == 0) {  // the column sums of this tile, owned by the k-tile-0 block
        for (i64 j = j0; j < j1; ++j) gb[j] = 0.0;
        for (i64 i = 0; i < R; ++i) {
          const real* d = dC + i * OUT;
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
void input_grad(const real* dC, u64 rows, const Layer& L, real* dA) {
  const i64 R = (i64)rows, IN = (i64)L.in, OUT = (i64)L.out;
  const real* W = L.W;
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
          const real* d = dC + i * OUT;
          for (i64 k = k0; k < k1; ++k) {
            const real* wk = W + k * OUT;
            real s = dA[i * IN + k];
            for (i64 j = j0; j < j1; ++j) s += d[j] * wk[j];
            dA[i * IN + k] = s;
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Register- and cache-blocked micro-kernel (idea B; since [GRAM] it computes
// the Gram products G, S, H3 G and W4T S).
//
// Each product is C = A B, a sum over an inner index p. The micro-kernel
// below keeps a kMR x kNR block of C in registers while p runs: 4 x 8 in
// fp64, 4 x 16 in fp32 ([PREC]; the same 8 vector accumulators). Per step it
// loads two vectors of B and 4 values of A, and does 32 (fp64) or 64 (fp32)
// multiply-adds with them, so it is limited by arithmetic rather than loads.
//
// A(r, p) = a[r * ars + p * aps] and B(p, c) = b[p * bps + c]: the strides
// let the same kernel read W4T, H3, S and G in place, transposed or not.
//
// Why vector types and not plain loops: written as plain loops over c, GCC
// vectorised the p loop instead wherever A is contiguous along p (fwd L4 and
// dA3). Without fast-math it may only do that as an in-order reduction, which
// rebuilt a serial dependency chain and made those two products slower than
// before. vreal, the compilers' generic vector type (GCC and Clang, no
// intrinsics), states which loop is the vector one: across the columns.
// 32 bytes becomes one AVX register on x86 and two NEON registers on Arm,
// so the tile lives in 8 AVX or 16 NEON registers. Every lane does exactly
// the scalar arithmetic, so the order of each sum is unchanged.
// ---------------------------------------------------------------------------
typedef real vreal __attribute__((vector_size(kVecBytes)));  // [PREC]: 4 doubles or 8 floats

static_assert(kMR == 4 && kNR == 2 * kLanes, "micro_kernel is written out for 4 x (2 vectors)");

// acc (kMR x kNR, row-major) += sum over p = 0 .. np-1 of A(r, p) * B(p, c).
// memcpy is the portable unaligned vector load and store; it compiles to one
// instruction.
inline void micro_kernel(i64 np, const real* __restrict a, i64 ars, i64 aps,
                         const real* __restrict b, i64 bps, real* __restrict acc) {
  vreal c00, c01, c10, c11, c20, c21, c30, c31;
  std::memcpy(&c00, acc + 0 * kNR, sizeof(vreal));
  std::memcpy(&c01, acc + 0 * kNR + kLanes, sizeof(vreal));
  std::memcpy(&c10, acc + 1 * kNR, sizeof(vreal));
  std::memcpy(&c11, acc + 1 * kNR + kLanes, sizeof(vreal));
  std::memcpy(&c20, acc + 2 * kNR, sizeof(vreal));
  std::memcpy(&c21, acc + 2 * kNR + kLanes, sizeof(vreal));
  std::memcpy(&c30, acc + 3 * kNR, sizeof(vreal));
  std::memcpy(&c31, acc + 3 * kNR + kLanes, sizeof(vreal));
  for (i64 p = 0; p < np; ++p) {
    const real* bp = b + p * bps;
    const real* ap = a + p * aps;
    vreal b0, b1;
    std::memcpy(&b0, bp, sizeof(vreal));
    std::memcpy(&b1, bp + kLanes, sizeof(vreal));
    const real a0 = ap[0], a1 = ap[ars], a2 = ap[2 * ars], a3 = ap[3 * ars];
    c00 += a0 * b0;
    c01 += a0 * b1;
    c10 += a1 * b0;
    c11 += a1 * b1;
    c20 += a2 * b0;
    c21 += a2 * b1;
    c30 += a3 * b0;
    c31 += a3 * b1;
  }
  std::memcpy(acc + 0 * kNR, &c00, sizeof(vreal));
  std::memcpy(acc + 0 * kNR + kLanes, &c01, sizeof(vreal));
  std::memcpy(acc + 1 * kNR, &c10, sizeof(vreal));
  std::memcpy(acc + 1 * kNR + kLanes, &c11, sizeof(vreal));
  std::memcpy(acc + 2 * kNR, &c20, sizeof(vreal));
  std::memcpy(acc + 2 * kNR + kLanes, &c21, sizeof(vreal));
  std::memcpy(acc + 3 * kNR, &c30, sizeof(vreal));
  std::memcpy(acc + 3 * kNR + kLanes, &c31, sizeof(vreal));
}

// The same sum for a partial tile (mr <= kMR rows, nr <= kNR columns) at the
// edges, e.g. the last 2 of the 13,714 genes. Same order: p ascending.
void micro_kernel_edge(i64 mr, i64 nr, i64 np, const real* a, i64 ars, i64 aps,
                       const real* b, i64 bps, real* acc) {
  for (i64 p = 0; p < np; ++p)
    for (i64 r = 0; r < mr; ++r) {
      const real ar = a[r * ars + p * aps];
      for (i64 c = 0; c < nr; ++c) acc[r * kNR + c] += ar * b[p * bps + c];
    }
}

inline void micro(i64 mr, i64 nr, i64 np, const real* a, i64 ars, i64 aps, const real* b,
                  i64 bps, real* acc) {
  if (mr == kMR && nr == kNR)
    micro_kernel(np, a, ars, aps, b, bps, acc);
  else
    micro_kernel_edge(mr, nr, np, a, ars, aps, b, bps, acc);
}

// ---------------------------------------------------------------------------
// Sparse kernels (idea 1).
// ---------------------------------------------------------------------------

// A1 = Xb W1 + b1, H1 = relu(A1), and P = Xb W4T ([GRAM]), gene-major
// ([T2]), for the hidden columns [k0, k1) of every row of the batch. Both
// products walk the same non-zeros, so they share one pass over them. A plain
// function with __restrict parameters, so the compiler knows the arrays do
// not overlap and vectorises the column loops without run-time checks.
//
// For a fixed (row i, column k), A1[i][k] = b1[k], then + x * W1[j][k] for the
// genes j of cell i in ascending order: the gene loop is ascending and each
// cell appears at most once per gene. That is the CSR kernel's order. P is
// the same sum with W4T in place of W1 and no bias or ReLU.
void forward_sparse_columns(const std::uint32_t* __restrict bp,
                            const std::uint16_t* __restrict slot, const real* __restrict val,
                            u64 D, u64 B, u64 H, const real* __restrict W1,
                            const real* __restrict b1, const real* __restrict W4T,
                            double* __restrict A1, real* __restrict H1, real* __restrict P,
                            u64 k0, u64 k1) {
  for (u64 i = 0; i < B; ++i) {
    double* a = A1 + i * H;
    real* q = P + i * H;
    for (u64 k = k0; k < k1; ++k) a[k] = double(b1[k]);
    for (u64 k = k0; k < k1; ++k) q[k] = 0.0;
  }
  for (u64 j = 0; j < D; ++j) {
    const u64 t0 = bp[j], t1 = bp[j + 1];
    if (t0 == t1) continue;  // gene j is not expressed in this batch
    const real* w1 = W1 + j * H;   // read once per batch, not once per cell
    const real* w4 = W4T + j * H;  // gene j's row of W4T, likewise
    for (u64 t = t0; t < t1; ++t) {
      const real x = val[t];
      double* a = A1 + u64(slot[t]) * H;
      real* q = P + u64(slot[t]) * H;
      const double xd = double(x);
      for (u64 k = k0; k < k1; ++k) a[k] += xd * double(w1[k]);
      for (u64 k = k0; k < k1; ++k) q[k] += x * w4[k];
    }
  }
  for (u64 i = 0; i < B; ++i) {
    const double* a = A1 + i * H;
    real* h = H1 + i * H;
    for (u64 k = k0; k < k1; ++k) h[k] = a[k] > 0.0 ? real(a[k]) : real(0);
  }
}

// The sparse forward: one block of hidden columns per thread, each block
// owned by one thread for every row, so no element is written by two threads.
void sparse_forward(const Batch& bt, const Layer& L1, const real* W4T, double* A1, real* H1,
                    real* P) {
  const Corpus& X = *bt.X;
  const u64 H = L1.out, D = X.n_genes;
  const std::uint32_t* bp = X.bptr.data() + bt.index * D;  // this batch's gene offsets
  const std::uint16_t* slot = X.bslot.get();
  const real* val = X.bval.get();
  const u64 cols = u64(ceil_div(ceil_div((i64)H, kThreads), (i64)kFwdColAlign)) * kFwdColAlign;
  parallel_for(ceil_div((i64)H, (i64)cols), 1, [&](i64 lo, i64 hi) {
    for (i64 task = lo; task < hi; ++task) {
      const u64 k0 = u64(task) * cols, k1 = std::min<u64>(H, k0 + cols);
      forward_sparse_columns(bp, slot, val, D, bt.B, H, L1.W, L1.b, W4T, A1, H1, P, k0, k1);
    }
  });
}

// db1 = colsum(dA1), summed over the batch rows in order (the dW1 rows
// themselves are formed in the fused pass, [FUSE]).
void colsum(const real* A, u64 rows, u64 H, real* out) {
  for (u64 k = 0; k < H; ++k) out[k] = 0.0;
  for (u64 i = 0; i < rows; ++i)
    for (u64 k = 0; k < H; ++k) out[k] += A[i * H + k];
}

// ---------------------------------------------------------------------------
// Layer 4 from Gram matrices ([GRAM]).
// ---------------------------------------------------------------------------

// G = X^T X (H x H, symmetric) and u = X^T y (H; y == nullptr means all ones,
// i.e. u = colsum(X)), for X rows x H, row-major.
//
// The rows are cut into fixed parts of kGramRows rows, independent of the
// thread count. Part t sums into its own partial matrix partG[t] and vector
// partU[t], and the partials are added in part order at the end, so every
// element is the same sum in the same order on any number of threads. Only
// 4 x 8 tiles on or above the diagonal are computed (1,056 of 2,048 for
// H = 256); the lower triangle is mirrored from the upper one.
//
// Inside a part the rows go by in sub-chunks of kGramKc (64 rows x 2 KB =
// 128 KB). For each 8-column strip, every tile at or above the diagonal runs
// the micro-kernel over the sub-chunk with both operands read in place:
// A(r, p) = X[p][a0 + r] and B(p, c) = X[p][b0 + c]; the strip stays in L1
// while the tiles of that column go by. The tile's sums wait in the partial
// between sub-chunks, so each element is summed over the part's rows in
// ascending order.
//
// G = W4T^T W4T has 27 parts, one task each. S = H3^T H3 (B <= 512 rows) has a
// single part, so its 32 column strips are separate tasks instead.
void gram(const real* X, u64 rows, u64 H, const real* y, real* partG, real* partU,
          real* G, real* u) {
  const i64 R = (i64)rows, NH = (i64)H;
  const i64 n_parts = std::max<i64>(1, ceil_div(R, kGramRows));
  const i64 n_strips = ceil_div(NH, kNR);
  const i64 strips_per_task = n_parts == 1 ? 1 : n_strips;
  const i64 tasks_per_part = ceil_div(n_strips, strips_per_task);
  parallel_for(n_parts * tasks_per_part, 1, [&](i64 lo, i64 hi) {
    real acc[kMR * kNR];
    for (i64 task = lo; task < hi; ++task) {
      const i64 t = task / tasks_per_part, sg = task % tasks_per_part;
      const i64 r0 = t * kGramRows, r1 = std::min(R, r0 + kGramRows);
      const i64 st0 = sg * strips_per_task, st1 = std::min(n_strips, st0 + strips_per_task);
      real* Gt = partG + t * NH * NH;
      real* ut = partU + t * NH;
      for (i64 c0 = r0; c0 < r1; c0 += kGramKc) {
        const i64 np = std::min(kGramKc, r1 - c0);
        const real* Xc = X + c0 * NH;
        for (i64 st = st0; st < st1; ++st) {
          const i64 b0 = st * kNR, nr = std::min(kNR, NH - b0);
          for (i64 a0 = 0; a0 < std::min(NH, b0 + kNR); a0 += kMR) {
            const i64 mr = std::min(kMR, NH - a0);
            for (i64 r = 0; r < mr; ++r)
              for (i64 c = 0; c < nr; ++c)
                acc[r * kNR + c] = c0 == r0 ? real(0) : Gt[(a0 + r) * NH + b0 + c];
            micro(mr, nr, np, Xc + a0, 1, NH, Xc + b0, NH, acc);
            for (i64 r = 0; r < mr; ++r)
              for (i64 c = 0; c < nr; ++c) Gt[(a0 + r) * NH + b0 + c] = acc[r * kNR + c];
          }
        }
      }
      // u, for this task's columns, over the part's rows in ascending order.
      for (i64 st = st0; st < st1; ++st) {
        const i64 b0 = st * kNR, b1 = std::min(NH, b0 + kNR);
        for (i64 k = b0; k < b1; ++k) ut[k] = 0.0;
        for (i64 r = r0; r < r1; ++r) {
          const real* xr = X + r * NH;
          if (y) {
            const real yr = y[r];
            for (i64 k = b0; k < b1; ++k) ut[k] += yr * xr[k];
          } else {
            for (i64 k = b0; k < b1; ++k) ut[k] += xr[k];
          }
        }
      }
    }
  });
  // Add the partials in part order; mirror the upper triangle. Row a's task
  // writes G[a][b] and G[b][a] for b >= a: no element is written twice.
  parallel_for(NH, 8, [&](i64 lo, i64 hi) {
    for (i64 a = lo; a < hi; ++a)
      for (i64 b = a; b < NH; ++b) {
        real v = partG[a * NH + b];
        for (i64 t = 1; t < n_parts; ++t) v += partG[t * NH * NH + a * NH + b];
        G[a * NH + b] = v;
        G[b * NH + a] = v;
      }
  });
  for (i64 k = 0; k < NH; ++k) {
    real v = partU[k];
    for (i64 t = 1; t < n_parts; ++t) v += partU[t * NH + k];
    u[k] = v;
  }
}

// The batch loss from the same pieces, without Xh:
//   sum (Xh - Xb)^2 = sum Xh^2 - 2 sum Xh Xb + sum Xb^2, where
//   sum Xh^2  = tr(G S) + 2 s.u + B |b4|^2                (Xh = H3 W4 + 1 b4^T)
//   sum Xh Xb = sum H3 * P + sum over non-zeros of x b4[j]
// It is only printed; serial, ~0.4 M operations per batch.
double gram_loss(const Batch& bt, const real* G, const real* S, const real* s, const real* u,
                 const real* H3, const real* P, const real* b4, u64 H) {
  const Corpus& X = *bt.X;
  const u64 B = bt.B, D = X.n_genes;
  double tr_gs = 0.0, su = 0.0, bb = 0.0, hp = 0.0, xb = 0.0, xx = 0.0;
  for (u64 e = 0; e < H * H; ++e) tr_gs += double(G[e]) * double(S[e]);
  for (u64 k = 0; k < H; ++k) su += double(s[k]) * double(u[k]);
  for (u64 j = 0; j < D; ++j) bb += double(b4[j]) * double(b4[j]);
  for (u64 e = 0; e < B * H; ++e) hp += double(H3[e]) * double(P[e]);
  const std::uint32_t* bp = X.bptr.data() + bt.index * D;
  for (u64 j = 0; j < D; ++j)
    for (u64 t = bp[j]; t < bp[j + 1]; ++t) {
      const double x = double(X.bval[t]);
      xb += x * double(b4[j]);
      xx += x * x;
    }
  const double sum_sq = (tr_gs + 2.0 * su + double(B) * bb) - 2.0 * (hp + xb) + xx;
  return sum_sq / (double(B) * double(D));
}

// dA3 = c (H3 G + 1 u^T - P), masked by [A3 > 0]: B x 256 times 256 x 256.
// Tasks are kDaRows-row blocks, each tile on the micro-kernel with G read in
// place (an 8-column strip of G is 16 KB, reused by the block's 4-row tiles).
void dA3_from_gram(const real* H3, const real* G, const real* u, const real* P,
                   const double* A3, u64 rows, u64 H, real c, real* dA3) {
  const i64 R = (i64)rows, NH = (i64)H;
  parallel_for(ceil_div(R, kDaRows), 1, [&](i64 lo, i64 hi) {
    real acc[kMR * kNR];
    for (i64 rt = lo; rt < hi; ++rt) {
      const i64 i0 = rt * kDaRows, i1 = std::min(R, i0 + kDaRows);
      for (i64 k0 = 0; k0 < NH; k0 += kNR) {
        const i64 nr = std::min(kNR, NH - k0);
        for (i64 ir = i0; ir < i1; ir += kMR) {
          const i64 mr = std::min(kMR, i1 - ir);
          for (i64 e = 0; e < kMR * kNR; ++e) acc[e] = 0.0;
          micro(mr, nr, NH, H3 + ir * NH, NH, 1, G + k0, NH, acc);
          for (i64 r = 0; r < mr; ++r)
            for (i64 cc = 0; cc < nr; ++cc) {
              const i64 e = (ir + r) * NH + k0 + cc;
              const real d = c * ((acc[r * kNR + cc] + u[k0 + cc]) - P[e]);
              dA3[e] = A3[e] > 0.0 ? d : real(0);
            }
        }
      }
    }
  });
}

// One walk over gene j's non-zeros in this batch gives row j of Xb^T dA1
// (dW1, into g1), row j of Xb^T H3 (Q, into q), and the sum of gene j's x
// (for db4). Each sum runs over the non-zeros in ascending slot order.
inline real sparse_rows2(const std::uint32_t* __restrict bp,
                           const std::uint16_t* __restrict slot,
                           const real* __restrict val, u64 j, const real* __restrict dA1,
                           const real* __restrict H3, u64 H, real* __restrict g1,
                           real* __restrict q) {
  for (u64 k = 0; k < H; ++k) g1[k] = 0.0;
  for (u64 k = 0; k < H; ++k) q[k] = 0.0;
  real xs = 0.0;
  for (u64 t = bp[j]; t < bp[j + 1]; ++t) {
    const real x = val[t];
    const real* d = dA1 + u64(slot[t]) * H;
    const real* h = H3 + u64(slot[t]) * H;
    for (u64 k = 0; k < H; ++k) g1[k] += x * d[k];
    for (u64 k = 0; k < H; ++k) q[k] += x * h[k];
    xs += x;
  }
  return xs;
}

// ---------------------------------------------------------------------------
// One batch. Every buffer comes from the Workspace allocated once per solve
// ([T5]); the gradients land in ws.g. Nothing is zero-filled: each kernel
// writes all of its output.
// ---------------------------------------------------------------------------

// Forward through layer 3, plus P = Xb W4T for layer 4.
void forward(const Model& M, const Batch& bt, Workspace& ws) {
  const u64 B = bt.B, H = M.l1.out;
  {
    PHASE(kFwdL1);
    sparse_forward(bt, M.l1, M.l4.W, ws.A1.data(), ws.H1.data(), ws.P.data());
  }
  {
    PHASE(kFwdMid);
    affine(ws.H1.data(), B, M.l2, ws.Z.data());
    affine(ws.Z.data(), B, M.l3, ws.A3.data());
    relu(ws.A3.data(), ws.H3.data(), B * H);
  }
}

// Forward and backward for one batch, up to everything the fused pass
// ([FUSE]) needs: G, u, S, s, P, dA3, dA1, and the small layers' gradients.
// Returns the batch loss (printed only).
double batch_backward(const Model& M, const Batch& bt, Workspace& ws) {
  const u64 B = bt.B, D = bt.X->n_genes, H = M.l1.out;
  const real c = real(2.0 * (1.0 / (double(B) * double(D))));  // d loss / d Xh
  Grads* g = &ws.g;
  const real* W4T = M.l4.W;
  const real* b4 = M.l4.b;

  forward(M, bt, ws);
  {
    // W4T and b4 as they are now: this batch's Adam step has not happened.
    PHASE(kGramG);
    gram(W4T, D, H, b4, ws.partG.data(), ws.partU.data(), ws.G.data(), ws.u.data());
  }
  double loss;
  {
    PHASE(kGramS);
    gram(ws.H3.data(), B, H, nullptr, ws.partG.data(), ws.partU.data(), ws.S.data(),
         ws.s.data());
    loss = gram_loss(bt, ws.G.data(), ws.S.data(), ws.s.data(), ws.u.data(), ws.H3.data(),
                     ws.P.data(), b4, H);
  }
  {
    PHASE(kBwdL4dA);
    dA3_from_gram(ws.H3.data(), ws.G.data(), ws.u.data(), ws.P.data(), ws.A3.data(), B, H, c,
                  ws.dA3.data());
  }
  {
    PHASE(kBwdMid);
    const u64 K = M.l2.out;
    weight_grad(ws.Z.data(), ws.dA3.data(), B, K, H, g->W3.data(), g->b3.data());
    input_grad(ws.dA3.data(), B, M.l3, ws.dZ.data());
    weight_grad(ws.H1.data(), ws.dZ.data(), B, H, K, g->W2.data(), g->b2.data());
    input_grad(ws.dZ.data(), B, M.l2, ws.dA1.data());
    relu_mask(ws.dA1.data(), ws.A1.data(), B * H);
    colsum(ws.dA1.data(), B, H, g->b1.data());
  }
  return loss;
}

#ifdef HPCBENCH_GRADCHECK
// The finite-difference oracle needs the batch loss as a function of the
// weights. It is computed here the direct way, from the full reconstruction
// Xh = H3 W4 + 1 b4^T, row by row, so that it shares no algebra with the
// Gram-matrix gradients it checks. Slow (B x n_genes dot products of 256),
// and compiled only into the gradient-check build.
double reference_loss(const Model& M, const Batch& bt, Workspace& ws) {
  forward(M, bt, ws);
  const Corpus& X = *bt.X;
  const u64 B = bt.B, D = X.n_genes, H = M.l1.out;
  const real* H3 = ws.H3.data();
  const real* W4T = M.l4.W;
  const real* b4 = M.l4.b;
  real* row_loss = ws.row_loss.data();
  parallel_for((i64)B, 1, [&](i64 lo, i64 hi) {
    real* xh = thread_scratch(D);
    for (i64 i = lo; i < hi; ++i) {
      const real* h = H3 + u64(i) * H;
      for (u64 j = 0; j < D; ++j) xh[j] = b4[j] + hpcbench::dot_fixed(h, W4T + j * H, H);
      const u64 cell = bt.cells[i];
      for (u64 t = X.row_ptr[cell]; t < X.row_ptr[cell + 1]; ++t) xh[X.col[t]] -= X.val[t];
      double sum = 0.0;
      for (u64 j = 0; j < D; ++j) sum += xh[j] * xh[j];
      row_loss[i] = sum;
    }
  });
  double loss = 0.0;
  for (u64 i = 0; i < B; ++i) loss += row_loss[i];
  return loss / (double(B) * double(D));
}
#endif

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
void adam_range(real* __restrict w, real* __restrict m, real* __restrict v,
                const real* __restrict grad, i64 lo, i64 hi, real beta1, real beta2,
                real omb1, real omb2, real lr, real eps, real bc1, real bc2) {
  for (i64 i = lo; i < hi; ++i) {
    const real g = grad[i];
    const real mi =
        kFirst ? omb1 * g + real(0) : beta1 * m[i] + omb1 * g;
    const real vi =
        kFirst ? omb2 * g * g + real(0) : beta2 * v[i] + omb2 * g * g;
    m[i] = mi;
    v[i] = vi;
    const real mhat = mi / bc1;
    const real vhat = vi / bc2;
    w[i] -= lr * mhat / (std::sqrt(std::fabs(vhat)) + eps);  // fabs: no errno branch, see [T4]
  }
}

void adam(real* w, real* m, real* v, const real* grad, u64 n, const hpcbench::Params& p,
          real bc1, real bc2, bool first) {
  const real beta1 = real(p.beta1), beta2 = real(p.beta2), lr = real(p.lr), eps = real(p.eps);
  const real omb1 = real(1.0 - p.beta1), omb2 = real(1.0 - p.beta2);  // in fp64, see [PREC]
  parallel_for((i64)n, kElemChunk, [&](i64 lo, i64 hi) {
    if (first)
      adam_range<true>(w, m, v, grad, lo, hi, beta1, beta2, omb1, omb2, lr, eps, bc1, bc2);
    else
      adam_range<false>(w, m, v, grad, lo, hi, beta1, beta2, omb1, omb2, lr, eps, bc1, bc2);
  });
}

void adam_layer(Layer& L, const Buffer<real>& gW, const Buffer<real>& gb,
                const hpcbench::Params& p, real bc1, real bc2, bool first) {
  adam(L.W, L.mW.data(), L.vW.data(), gW.data(), L.in * L.out, p, bc1, bc2, first);
  adam(L.b, L.mb.data(), L.vb.data(), gb.data(), L.out, p, bc1, bc2, first);
}

// One Adam step's constants.
struct AdamStep {
  real beta1, beta2, omb1, omb2, lr, eps, bc1, bc2;  // omb = 1 - beta, formed in fp64
  bool first;  // step 1: m and v not written yet
};

// The fused pass ([FUSE]): for every gene j, rows j of dW1 and dW4T and
// db4[j], and (kApply) the Adam update of rows j of W1 and W4T and of b4[j].
// Tasks are blocks of kGeneBlk genes; per block, with per-thread scratch
// g1, q, g4 (kGeneBlk x 256 each) and gb:
//   1. g1 = dW1 rows and q = Q rows, one walk over each gene's non-zeros;
//   2. g4 = c ((W4T S + b4 s^T) - q): 4 x 8 tiles on the micro-kernel, A =
//      4 rows of W4T (in L1), B = an 8-column strip of S (reused by all of
//      the block's 4-row tiles); gb = c ((W4T s + B b4) - sum x);
//   3. kApply: Adam on those rows of W1 and W4T and those entries of b4,
//      straight from the scratch. Otherwise (gradient check only) the rows
//      are copied out to gW1, gW4T, gb4 and no weight changes.
// A block reads and writes only its own rows of W4T and entries of b4, so
// updating them in place is safe; everything else it reads is fixed.
template <bool kApply, bool kFirst>
void fused_rows(const Batch& bt, const real* dA1, const real* H3, const real* S,
                const real* s, real c, Layer& L1, Layer& L4, const AdamStep& st, real* gW1,
                real* gW4T, real* gb4) {
  const Corpus& X = *bt.X;
  const u64 D = X.n_genes, H = L1.out;
  const std::uint32_t* bp = X.bptr.data() + bt.index * D;  // this batch's gene offsets
  const std::uint16_t* slot = X.bslot.get();
  const real* val = X.bval.get();
  const real Bd = real(bt.B);
  const i64 NH = (i64)H;
  real* const W4T = L4.W;
  real* const b4 = L4.b;
  parallel_for(ceil_div((i64)D, kGeneBlk), 1, [&](i64 lo, i64 hi) {
    real* const g1 = thread_scratch(3 * u64(kGeneBlk) * H + u64(kGeneBlk));
    real* const q = g1 + u64(kGeneBlk) * H;
    real* const g4 = q + u64(kGeneBlk) * H;
    real* const gb = g4 + u64(kGeneBlk) * H;
    real xs[kGeneBlk];
    real acc[kMR * kNR];
    for (i64 blk = lo; blk < hi; ++blk) {
      const i64 j0 = blk * kGeneBlk, nj = std::min<i64>((i64)D - j0, kGeneBlk);
      // 1. dW1 rows and Q rows.
      for (i64 r = 0; r < nj; ++r)
        xs[r] = sparse_rows2(bp, slot, val, u64(j0 + r), dA1, H3, H, g1 + r * NH, q + r * NH);
      // 2. dW4T rows and db4, from the old W4T and b4.
      for (i64 k0 = 0; k0 < NH; k0 += kNR) {
        const i64 nr = std::min(kNR, NH - k0);
        for (i64 r0 = 0; r0 < nj; r0 += kMR) {
          const i64 mr = std::min(kMR, nj - r0);
          for (i64 e = 0; e < kMR * kNR; ++e) acc[e] = 0.0;
          micro(mr, nr, NH, W4T + (j0 + r0) * NH, NH, 1, S + k0, NH, acc);
          for (i64 r = 0; r < mr; ++r) {
            const i64 j = j0 + r0 + r;
            for (i64 cc = 0; cc < nr; ++cc) {
              const i64 k = k0 + cc;
              g4[(r0 + r) * NH + k] =
                  c * ((acc[r * kNR + cc] + b4[j] * s[k]) - q[(r0 + r) * NH + k]);
            }
          }
        }
      }
      for (i64 r = 0; r < nj; ++r) {
        const i64 j = j0 + r;
        const real* w = W4T + j * NH;
        real ws = 0.0;
        for (i64 k = 0; k < NH; ++k) ws += w[k] * s[k];
        gb[r] = c * ((ws + Bd * b4[j]) - xs[r]);
      }
      // 3. Adam on these rows, or hand the gradients to the check.
      if constexpr (kApply) {
        for (i64 r = 0; r < nj; ++r) {
          const i64 j = j0 + r;
          adam_range<kFirst>(L1.W + j * NH, L1.mW.data() + j * NH, L1.vW.data() + j * NH,
                             g1 + r * NH, 0, NH, st.beta1, st.beta2, st.omb1, st.omb2, st.lr,
                             st.eps, st.bc1, st.bc2);
          adam_range<kFirst>(W4T + j * NH, L4.mW.data() + j * NH, L4.vW.data() + j * NH,
                             g4 + r * NH, 0, NH, st.beta1, st.beta2, st.omb1, st.omb2, st.lr,
                             st.eps, st.bc1, st.bc2);
          adam_range<kFirst>(b4 + j, L4.mb.data() + j, L4.vb.data() + j, gb + r, 0, 1, st.beta1,
                             st.beta2, st.omb1, st.omb2, st.lr, st.eps, st.bc1, st.bc2);
        }
      } else {
        std::memcpy(gW1 + j0 * NH, g1, u64(nj * NH) * sizeof(real));
        std::memcpy(gW4T + j0 * NH, g4, u64(nj * NH) * sizeof(real));
        std::memcpy(gb4 + j0, gb, u64(nj) * sizeof(real));
      }
    }
  });
}

// The training form of the fused pass: Adam applied, no gradient arrays.
void fused_step(const Batch& bt, Workspace& ws, real c, Model& M, const AdamStep& st) {
  const real* dA1 = ws.dA1.data();
  const real* H3 = ws.H3.data();
  if (st.first)
    fused_rows<true, true>(bt, dA1, H3, ws.S.data(), ws.s.data(), c, M.l1, M.l4, st, nullptr,
                           nullptr, nullptr);
  else
    fused_rows<true, false>(bt, dA1, H3, ws.S.data(), ws.s.data(), c, M.l1, M.l4, st, nullptr,
                            nullptr, nullptr);
}

#ifdef HPCBENCH_GRADCHECK
// The analytic gradient is in ws.g; the reference loss that the finite
// differences need writes the activations only, never ws.g.
void check_layer(const char* wname, const char* bname, Layer& L, const Buffer<real>& gW,
                 const Buffer<real>& gb, const Model& M, const Batch& bt, Workspace& ws,
                 hpcbench::GradCheckReport& total) {
  auto loss = [&]() { return reference_loss(M, bt, ws); };
  for (int which = 0; which < 2; ++which) {
    real* P = which == 0 ? L.W : L.b;
    const u64 n = which == 0 ? L.in * L.out : L.out;
    const Buffer<real>& G = which == 0 ? gW : gb;
    const hpcbench::GradCheckReport r = hpcbench::gradcheck(
        which == 0 ? wname : bname, n, G.data(), [&](std::size_t i) { return P[i]; },
        [&](std::size_t i, double v) { P[i] = real(v); }, loss);
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
    // Converted to the working precision into buffers the layers own ([PREC]).
    M.l1 = make_layer(p.W1, p.b1, D, H);
    M.l2 = make_layer(p.W2, p.b2, H, K);
    M.l3 = make_layer(p.W3, p.b3, K, H);
    M.l4 = make_layer_t(p.W4, p.b4, H, D);  // W4T ([W4T])
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

    const real c = real(2.0 * (1.0 / (double(B) * double(D))));  // as in batch_backward
    loss_sum += batch_backward(M, bt, *ws);

#ifdef HPCBENCH_GRADCHECK
    if (steps == 0) {
      // [FUSE]: materialise dW1, dW4T and db4 for the check, without Adam.
      AdamStep none{};
      fused_rows<false, false>(bt, ws->dA1.data(), ws->H3.data(), ws->S.data(), ws->s.data(),
                               c, M.l1, M.l4, none, g.W1.data(), g.W4.data(), g.b4.data());
      hpcbench::GradCheckReport total;
      check_layer("W1", "b1", M.l1, g.W1, g.b1, M, bt, *ws, total);
      check_layer("W2", "b2", M.l2, g.W2, g.b2, M, bt, *ws, total);
      check_layer("W3", "b3", M.l3, g.W3, g.b3, M, bt, *ws, total);
      check_layer("W4", "b4", M.l4, g.W4, g.b4, M, bt, *ws, total);
      std::fprintf(stderr, "gradcheck: %zu weights checked, %zu mismatched, max rel err %.2e\n",
                   total.checked, total.failed, total.max_rel_err);
      // The checks' forward passes ended on a perturbed weight; recompute
      // this batch from the restored weights before training on it.
      batch_backward(M, bt, *ws);
    }
#endif

    const bool first = steps == 0;  // m and v not written yet
    ++steps;
    beta1_t *= p.beta1;
    beta2_t *= p.beta2;
    const double bc1 = 1.0 - beta1_t, bc2 = 1.0 - beta2_t;
    {
      PHASE(kFused);
      const AdamStep st{real(p.beta1),       real(p.beta2),       real(1.0 - p.beta1),
                        real(1.0 - p.beta2), real(p.lr),          real(p.eps),
                        real(bc1),           real(bc2),           first};
      fused_step(bt, *ws, c, M, st);  // W1, W4T, b4 ([FUSE])
    }
    {
      PHASE(kAdam);
      adam(M.l1.b, M.l1.mb.data(), M.l1.vb.data(), g.b1.data(), H, p, bc1, bc2, first);
      adam_layer(M.l2, g.W2, g.b2, p, bc1, bc2, first);
      adam_layer(M.l3, g.W3, g.b3, p, bc1, bc2, first);
    }
  }
  std::fprintf(stderr, "epoch: %llu steps, mean batch loss %.9f (%s)\n", (unsigned long long)steps,
               loss_sum / double(steps), P2_PRECISION);

  // The answer, converted back to the host's fp64 ([PREC]); W4 also goes
  // back from W4T into the host's layout ([W4T]).
  {
    PHASE(kCopyOut);
    convert_into(M.l1.W, D * H, out.W1);
    convert_into(M.l1.b, H, out.b1);
    convert_into(M.l2.W, H * K, out.W2);
    convert_into(M.l2.b, K, out.b2);
    convert_into(M.l3.W, K * H, out.W3);
    convert_into(M.l3.b, H, out.b3);
    transpose_into(M.l4.W, D, H, out.W4);
    convert_into(M.l4.b, D, out.b4);
  }

#ifdef P2_PROFILE
  const double total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_solve)
          .count();
  double sum = 0.0;
  for (int i = 0; i < kNumPhases; ++i) sum += g_phase_ms[i];
  std::fprintf(stderr, "profile: %d threads, %s\n", kThreads, P2_PRECISION);
  std::fprintf(stderr, "profile: %-30s %10s %7s\n", "phase", "ms", "share");
  for (int i = 0; i < kNumPhases; ++i)
    std::fprintf(stderr, "profile: %-30s %10.2f %6.1f%%\n", kPhaseName[i], g_phase_ms[i],
                 100.0 * g_phase_ms[i] / total_ms);
  std::fprintf(stderr, "profile: %-30s %10.2f %6.1f%%\n", "(not in any phase)", total_ms - sum,
               100.0 * (total_ms - sum) / total_ms);
  std::fprintf(stderr, "profile: %-30s %10.2f\n", "solve total", total_ms);
#endif
}
