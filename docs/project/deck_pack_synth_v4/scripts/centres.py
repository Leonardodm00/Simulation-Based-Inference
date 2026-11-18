"""The synth v4 condition centres as the cluster built them.

The pipeline places the three condition centres with the DSN generator's
`_class_center_vectors(3, 8, "simplex")` (SBI hpc/dsn/latent_burst_generator.py),
scaled about 0.5 by s = 0.562743 (D-132, `regular_centres` in
hpc/joint/stage1/latent_sbi_simulator.py). That construction takes the simplex
basis from an SVD of the centring projector I - 11^T/3, whose singular value 1
is repeated: the two basis vectors are fixed only up to a rotation or a sign,
and LAPACK builds may differ in the one they return. The basis this sandbox
returns does not fit the cohorts' recorded truths; the same construction with
the sign of the second basis vector flipped does (`main` below prints the
evidence). The deck's figures use the flipped basis, `SIGNS_CLUSTER`.

Usage:
    python3 centres.py <SBI checkout root> <metrics folder>
"""
import itertools
import os
import sys

import numpy as np

S_SCALE = 0.562743          # v4 centre scale (D-132)
N_CLASSES, N_LABEL = 3, 8   # v4: 3 conditions, label axes 0-7 (D-132)
TAU = 0.10                  # within-condition SD on each label axis (D-132)
SIGNS_CLUSTER = (1.0, -1.0)  # basis signs that fit the cohorts (see main)


def simplex_centres(signs=(1.0, 1.0), scale=S_SCALE, C=N_CLASSES, L=N_LABEL):
    """The construction of `_class_center_vectors(C, L, "simplex")` followed by
    `regular_centres`' scaling, with the SVD basis vectors multiplied by `signs`.

    Returns (C, L) float64, centres in unit-box coordinates (dimensionless).
    """
    proj = np.eye(C) - 1.0 / C
    u, _s, _vt = np.linalg.svd(proj)
    y = u[:, :C - 1] * np.asarray(signs, dtype=float)[None, :]
    z = np.zeros((C, L))
    z[:, :C - 1] = y
    k = np.arange(L)
    q = np.cos(np.pi * (k[:, None] + 0.5) * k[None, :] / L)      # DCT-II
    q /= np.linalg.norm(q, axis=0, keepdims=True)
    z = z @ q.T
    half = 0.5 * (C - 1.0) / (C + 1.0)
    z = z * (half / float(np.max(np.abs(z))))
    m = np.clip(0.5 + z, 0.0, 1.0)
    return 0.5 + float(scale) * (m - 0.5)


def donor_truths(metrics_dir, stem):
    r = np.load(os.path.join(metrics_dir, stem))
    th, donor, cls = r["theta_ref"], r["donor"], r["cls"]
    ids = np.unique(donor)
    first = np.array([np.flatnonzero(donor == d)[0] for d in ids])
    return th[first][:, :N_LABEL], cls[first]


def sse_best_perm(mu, T, C):
    """Sum of squared deviations of the donors from their centres, minimised
    over the assignment of conditions to centres."""
    return min((sum(float(((T[C == c] - mu[p[c]]) ** 2).sum()) for c in range(N_CLASSES)), p)
               for p in itertools.permutations(range(N_CLASSES)))


def main(sbi_root, metrics_dir):
    sys.path.insert(0, os.path.join(sbi_root, "hpc", "joint", "stage1"))
    sys.path.insert(0, os.path.join(sbi_root, "hpc", "joint"))
    from latent_sbi_simulator import load_dsn_modules, regular_centres
    lbg, _gbd = load_dsn_modules()
    here = regular_centres(lbg, N_CLASSES, N_LABEL, S_SCALE)
    assert np.allclose(here, simplex_centres((1.0, 1.0)), atol=1e-12), \
        "the reimplementation does not reproduce the pipeline's construction"
    print("reimplementation reproduces regular_centres here: OK")

    T0, C0 = donor_truths(metrics_dir, "S-A1_A1_R0.npz")
    T3, C3 = donor_truths(metrics_dir, "S-A2-R3_A2_R3.npz")
    T, C = np.vstack([T0, T3]), np.concatenate([C0, C3])
    n = T.size
    print("donors: %d (R0 %d, R3 %d), label coordinates %d" % (len(T), len(T0), len(T3), n))
    print("expected SSE if the centres are right (tau = %.2f, truncation ignored): "
          "%.2f +- %.2f" % (TAU, n * TAU ** 2, np.sqrt(2.0 * n) * TAU ** 2))
    for signs in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]:
        mu = simplex_centres(signs)
        v, p = sse_best_perm(mu, T, C)
        d = np.linalg.norm(mu[0] - mu[1])
        print("signs %+d %+d: SSE %.3f (residual SD %.4f), best assignment %s, "
              "centre distance %.6f" % (signs[0], signs[1], v, np.sqrt(v / n), p, d))
    mu = simplex_centres(SIGNS_CLUSTER)
    v, p = sse_best_perm(mu, T, C)
    print("SIGNS_CLUSTER assignment (condition -> row):", p)
    print(np.round(mu, 4))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
