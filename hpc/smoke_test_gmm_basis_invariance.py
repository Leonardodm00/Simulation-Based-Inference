#!/usr/bin/env python3
"""smoke_test_gmm_basis_invariance.py -- 8 checks.

Guards the fix for the D7_contraction machine-dependence: GMMBenchmark's
component means must depend on A only, never on which orthonormal basis of
null(A) the local LAPACK build's SVD happens to return.

The test does not need two machines. It simulates the disagreement directly:
any orthogonal rotation Q of null_basis is an equally valid SVD output, so if
rebuilding the geometry from a rotated basis changes an observable, that
observable is machine-dependent.

Expect: ALL 8 CHECKS PASSED

Run:
    python3 smoke_test_gmm_basis_invariance.py
    python3 smoke_test_gmm_basis_invariance.py -k B3
"""
import argparse
import sys

import numpy as np

from gmm_benchmark import GMMBenchmark
from npe_diagnostics import posterior_contraction

N_DIM, N_OBS, N_COMP = 6, 3, 3
SEP, PRIOR_SCALE, OBS_NOISE = 6.0, 1.0, 0.4
N_CALIB, N_DRAWS = 400, 128
N_ROTATIONS = 40

_passed, _failed = [], []


def check(cid, desc, fn):
    if ARGS.only and ARGS.only not in cid:
        return
    try:
        detail = fn()
        _passed.append(cid)
        print("  PASS  %-28s %s" % (cid, detail))
    except AssertionError as exc:
        _failed.append((cid, str(exc)))
        print("  FAIL  %-28s %s" % (cid, exc))


def bench(seed=0, **kw):
    kw.setdefault("separate_in_nullspace", True)
    return GMMBenchmark(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                        separation=SEP, prior_scale=PRIOR_SCALE,
                        obs_noise=OBS_NOISE, seed=seed, **kw)


def rotations(n=N_ROTATIONS, seed=20260914):
    """n orthogonal 3x3 matrices, each giving an equally valid null basis."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        Q, R = np.linalg.qr(rng.normal(size=(N_DIM - N_OBS, N_DIM - N_OBS)))
        yield Q * np.sign(np.diag(R))[None, :]


def null_projector(A):
    return np.eye(A.shape[1]) - np.linalg.pinv(A) @ A


class rotated_svd(object):
    """Context manager: make np.linalg.svd return a DIFFERENT, equally valid
    null-space basis, exactly as another LAPACK build would.

    Only the full_matrices=True call on an (n_obs, n_dim) matrix is touched, so
    np.linalg.pinv -- which calls svd with full_matrices=False -- is unaffected
    and the projector stays the reference object.
    """

    def __init__(self, Q, rank=N_OBS):
        self.Q, self.rank, self.real = Q, rank, np.linalg.svd

    def __enter__(self):
        real, Q, rank = self.real, self.Q, self.rank

        def patched(a, *args, **kw):
            U, s, Vt = real(a, *args, **kw)
            if np.ndim(a) == 2 and a.shape == (N_OBS, N_DIM) \
                    and Vt.shape == (N_DIM, N_DIM):
                Vt = Vt.copy()
                Vt[rank:] = Q @ Vt[rank:]
            return U, s, Vt

        np.linalg.svd = patched
        return self

    def __exit__(self, *exc):
        np.linalg.svd = self.real
        return False


def calibration(b, seed=0, n_calib=N_CALIB, n_draws=N_DRAWS):
    rng = np.random.default_rng(seed)
    th = b.prior_sample(n_calib, rng)
    Z = b.simulate(th, rng)
    ex = np.stack([b.posterior(Z[i]).sample(n_draws, rng)
                   for i in range(n_calib)], axis=0)
    return th, ex


# --- B1: the premise -- the basis really is non-unique ----------------------
def b1():
    b = bench()
    P = null_projector(b.A)
    worst = 0.0
    for Q in rotations():
        nb = Q @ b.null_basis
        assert np.abs(b.A @ nb.T).max() < 1e-10, \
            "a rotated basis left null(A); the test premise is broken"
        worst = max(worst, np.abs(nb.T @ nb - P).max())
    assert worst < 1e-10, \
        "projector rebuilt from a rotated basis differs by %.2e" % worst
    return ("%d rotated bases: all valid, all give the same projector "
            "(max dev %.1e)" % (N_ROTATIONS, worst))


# --- B2: the means are built inside null(A) ---------------------------------
def b2():
    b = bench()
    d = b.means - b.means.mean(axis=0, keepdims=True)
    resid = np.abs(b.A @ d.T).max()
    assert resid < 1e-10, "mode separation leaks out of null(A) by %.2e" % resid
    in_null = float(np.sum((d @ b.null_basis.T) ** 2))
    in_row = float(np.sum((d @ b.row_basis.T) ** 2))
    frac = in_null / (in_null + in_row)
    assert frac > 1 - 1e-9, "only %.4f of the separation energy is in null(A)" % frac
    return "100%% of mode-separation energy inside null(A); ||A d^T|| = %.1e" % resid


# --- B3: THE REGRESSION GUARD -- means do not depend on the basis ----------
def b3():
    ref = bench().means.copy()
    worst = 0.0
    for Q in rotations():
        with rotated_svd(Q):
            c = bench()                      # BUILT under the rotated basis
        assert np.abs(c.A @ (c.means - c.means.mean(axis=0)).T).max() < 1e-10, \
            "rotated-basis benchmark left null(A); the injection is wrong"
        worst = max(worst, np.abs(c.means - ref).max())
    assert worst < 1e-12, (
        "component means moved by %.2e when the null basis was rotated -- the "
        "geometry is basis-dependent and every per-axis quantity is "
        "machine-dependent" % worst)
    return "means identical across %d bases (max dev %.1e)" % (N_ROTATIONS, worst)


# --- B4: the construction is reproducible from A alone ----------------------
def b4():
    b = bench()
    P = null_projector(b.A)
    # replay the MAIN stream to recover `centre`: A, then b, then the retained
    # offs draw, then centre. The directions come from the SEPARATE stream.
    rng = np.random.default_rng(0)
    rng.normal(size=(N_OBS, N_DIM))
    rng.normal(size=(N_OBS,))
    rng.normal(size=(N_COMP, N_DIM - N_OBS))
    centre = rng.normal(size=(N_DIM,)) * 0.5
    drng = np.random.default_rng(0 + 1000003)
    gen = drng.normal(size=(N_COMP, N_DIM))
    gen -= gen.mean(axis=0, keepdims=True)
    d = gen @ P
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    expect = centre[None, :] + SEP * PRIOR_SCALE * d
    dev = np.abs(expect - b.means).max()
    assert dev < 1e-12, "independent reconstruction differs by %.2e" % dev
    return "means reproduced from A and the seed alone (dev %.1e)" % dev


# --- B5: the fix did not disturb A, b or Sigma ------------------------------
def b5():
    b = bench()
    rng = np.random.default_rng(0)
    A_expect = rng.normal(size=(N_OBS, N_DIM)) / np.sqrt(N_DIM)
    b_expect = rng.normal(size=(N_OBS,))
    assert np.abs(b.A - A_expect).max() < 1e-15, "A changed"
    assert np.abs(b.b - b_expect).max() < 1e-15, "b changed"
    return "A and b bit-identical to the pre-fix rng stream"


# --- B6: weight preservation still holds (the design property) --------------
def b6():
    b = bench()
    d = b.means - b.means.mean(axis=0, keepdims=True)
    spread = np.abs(b.A @ d.T).max()
    assert spread < 1e-10, "A mu_k differs across components by %.2e" % spread
    rng = np.random.default_rng(3)
    th = b.prior_sample(64, rng)
    Z = b.simulate(th, rng)
    worst = 0.0
    for i in range(Z.shape[0]):
        w = np.asarray(b.posterior(Z[i]).weights, dtype=np.float64)
        worst = max(worst, np.abs(w - b.weights).max())
    assert worst < 1e-8, "posterior weights drifted from the prior by %.2e" % worst
    return "posterior weights == prior weights over 64 observations (%.1e)" % worst


# --- B7: separation preservation still holds --------------------------------
def b7():
    b = bench()
    s2 = PRIOR_SCALE ** 2
    C = np.linalg.inv(np.eye(N_DIM) / s2
                      + b.A.T @ np.linalg.inv(b.Sigma) @ b.A)
    worst = 0.0
    for v in b.null_basis:
        worst = max(worst, np.linalg.norm(C @ v - s2 * v))
    assert worst < 1e-10, "C v != s^2 v on null(A); off by %.2e" % worst
    return "C v = s^2 v for every v in null(A) (max dev %.1e)" % worst


# --- B8: D7's own statistic is now basis-invariant --------------------------
def b8():
    b = bench()
    th, ex = calibration(b)
    ref = float(np.max(posterior_contraction(th, ex).contraction))
    vals = [ref]
    for Q in rotations(n=5):
        with rotated_svd(Q):
            c = bench()                      # BUILT under the rotated basis
        th2, ex2 = calibration(c)
        vals.append(float(np.max(posterior_contraction(th2, ex2).contraction)))
    spread = max(vals) - min(vals)
    assert spread < 1e-9, (
        "max contraction still varies by %.3f across valid bases (values %s)"
        % (spread, ["%.3f" % v for v in vals]))
    return "max contraction %.4f, spread %.1e over 5 bases" % (ref, spread)


CHECKS = [
    ("B1_basis_is_non_unique", "rotated bases are valid and share a projector", b1),
    ("B2_separation_in_null", "mode separation lies inside null(A)", b2),
    ("B3_means_basis_free", "REGRESSION GUARD: means do not move with the basis", b3),
    ("B4_reproducible_from_A", "means reconstructible from A and the seed", b4),
    ("B5_stream_undisturbed", "A and b unchanged by the fix", b5),
    ("B6_weight_preservation", "posterior weights equal prior weights", b6),
    ("B7_separation_preservation", "C v = s^2 v on null(A)", b7),
    ("B8_contraction_invariant", "D7's statistic no longer moves with the basis", b8),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", dest="only", default=None,
                    help="run only checks whose id contains this string")
    ARGS = ap.parse_args()

    print("=" * 74)
    print("GMMBenchmark geometry is independent of the SVD null-space basis")
    print("=" * 74)
    print()
    for cid, _desc, fn in CHECKS:
        check(cid, _desc, fn)
    print()
    print("=" * 74)
    n = len(_passed) + len(_failed)
    if _failed:
        print("%d passed, %d failed, %d total" % (len(_passed), len(_failed), n))
        print("=" * 74)
        print()
        print("Failures:")
        for cid, msg in _failed:
            print("  %s: %s" % (cid, msg))
        sys.exit(1)
    print("ALL %d CHECKS PASSED  %d/%d checks passed" % (n, n, n))
    print("=" * 74)
