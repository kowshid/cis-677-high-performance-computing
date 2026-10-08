#!/usr/bin/env python3
"""
Build the P1 raw corpus from the packed pbmc3k matrix.

Raw format -- what `prepare` reads, and the only thing the grader ships:

    u64 n_cells
    u64 n_genes
    f64 values[n_cells * n_genes]      row-major, cells x genes

It is deliberately the worst possible layout. Choosing something better is the
first project.

Two variants:

  public   pbmc3k exactly as singlet ships it, transposed to cells x genes.
           This is the matrix the leaderboard ranks on.

  holdout  the same matrix under a seeded transformation: counts are binomially
           thinned at p=0.9 and the rows and columns are permuted. Thinning a
           negative-binomial count is still negative-binomial, so the holdout
           has the same shape, density regime and dispersion -- but different
           bytes, so an answer memorised from the public input is worth
           nothing. Released after the deadline; this is what goes in the
           gradebook.
"""
import argparse, os, struct
import numpy as np

HOLDOUT_SEED = 1729
THIN_P = 0.9


def load_csc(npz):
    z = np.load(npz)
    ngenes, ncells = (int(v) for v in z["shape"])
    return ngenes, ncells, z["indices"], z["indptr"], z["data"]


def densify(ngenes, ncells, indices, indptr, data):
    """CSC (genes x cells) -> dense float64 (cells x genes)."""
    X = np.zeros((ncells, ngenes), dtype=np.float64)
    for c in range(ncells):
        lo, hi = indptr[c], indptr[c + 1]
        X[c, indices[lo:hi]] = data[lo:hi]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="data/p1-kernel/pbmc3k.npz")
    ap.add_argument("--variant", choices=["public", "holdout"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ngenes, ncells, idx, ptr, dat = load_csc(a.packed)
    X = densify(ngenes, ncells, idx, ptr, dat)

    if a.variant == "holdout":
        rng = np.random.default_rng(HOLDOUT_SEED)
        nz = X > 0
        X[nz] = rng.binomial(X[nz].astype(np.int64), THIN_P).astype(np.float64)
        X = X[rng.permutation(ncells)][:, rng.permutation(ngenes)]

    nnz = int(np.count_nonzero(X))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<QQ", ncells, ngenes))
        X.tofile(f)

    print(f"{a.out}: {ncells} cells x {ngenes} genes, {nnz} nnz "
          f"({100*nnz/X.size:.3f}% dense), {os.path.getsize(a.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
