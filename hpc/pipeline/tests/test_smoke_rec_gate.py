#!/usr/bin/env python3
"""Smoke test of block 34: the gate (specs/SPEC.md block 34, "Test
oracles", checks R34*).

Each check is the oracle of the same name in specs/SPEC.md block 34. A check
that cannot run here is not deleted: it raises unittest.SkipTest("<reason>"),
so the summary shows why. Inputs are synthetic embeddings built from the
contract; no other block's code is imported (only ev_gate and npe_misspec,
which ev_gate calls).

Checks
    test_r34a_reference    member_statistic equals npe_misspec.mmd2_multiscale point by point
    test_r34b_by_hand      t, r, T and p from loops over (R5)-(R8) on a tiny case
    test_r34c_null_as_test the null rows as test rows: T = T_null, p by hand
    test_r34d_size         the p-value's size on toy embeddings of one law
    test_r34e_power        the flagged share against the displacement
    test_r34f_roles        assign_roles: groups, caps, order, determinism, refusal
    test_r34g_edges        non-finite input refused; no test row
    test_r34h_size_law     the size given the null set, Beta(10, 90), on p_member

Run
    python3 hpc/pipeline/tests/test_smoke_rec_gate.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. Needs numpy and scipy; about 25 s on 2 cores.

Pure ASCII, LF only.
"""

import os
import sys
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 20261123  # every random number of this file derives from it

import numpy as np  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import spearmanr, special_ortho_group  # noqa: E402

import ev_gate  # noqa: E402  (puts hpc/ on sys.path through _evalpath)
import npe_misspec  # noqa: E402

FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)   # D-163 (4)


def _rng(k):
    """The generator of sub-check k, derived from SEED."""
    return np.random.default_rng([SEED, k])


# ---------------------------------------------------------------------------
# helpers: the definitions (R5)-(R8) written as loops, independent of ev_gate
# ---------------------------------------------------------------------------

def _k_loop(a, b, h):
    """(R5) for one pair, by a loop over the bandwidths."""
    s = 0.0
    for hb in h:
        d2 = 0.0
        for c in range(len(a)):
            d2 += (float(a[c]) - float(b[c])) ** 2
        s += np.exp(-d2 / (2.0 * hb * hb))
    return s


def _gate_loop(z_ref, z_rank, z_null, z_test, factors):
    """(R5)-(R8) by loops over the definitions, float64; the median by
    scipy's pdist of the reference alone."""
    M = len(z_ref)
    N_rank, N_null, n = len(z_rank[0]), len(z_null[0]), len(z_test[0])
    out = {"t": [], "r": [], "p_member": [], "t_null": [], "r_null": [], "median": []}
    for m in range(M):
        med = float(np.median(pdist(np.asarray(z_ref[m], dtype=np.float64))))
        h = [f * med for f in factors]
        R = z_ref[m]
        N_ref = len(R)
        self_term = sum(_k_loop(R[i], R[j], h) for i in range(N_ref)
                        for j in range(N_ref)) / N_ref ** 2

        def t_of(x):
            return (self_term + _k_loop(x, x, h)
                    - 2.0 / N_ref * sum(_k_loop(R[i], x, h) for i in range(N_ref)))
        t_rank = [t_of(x) for x in z_rank[m]]
        t_null = [t_of(x) for x in z_null[m]]
        t_test = [t_of(x) for x in z_test[m]]

        def r_of(tx):
            return sum(1 for ta in t_rank if ta < tx) / float(N_rank)
        out["t"].append(t_test)
        out["t_null"].append(t_null)
        out["r"].append([r_of(tx) for tx in t_test])
        out["r_null"].append([r_of(tx) for tx in t_null])
        out["p_member"].append([(1 + sum(1 for tv in t_null if tv >= tx)) / (1.0 + N_null)
                                for tx in t_test])
        out["median"].append(med)
    T = [sum(out["r"][m][j] for m in range(M)) / M for j in range(n)]
    T_null = [sum(out["r_null"][m][v] for m in range(M)) / M for v in range(N_null)]
    out["T"], out["T_null"] = T, T_null
    out["p"] = [(1 + sum(1 for tv in T_null if tv >= T[j])) / (1.0 + N_null)
                for j in range(n)]
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _toy(rng, n, Q, shift=0.0, xi=None, u=None):
    """R34d's toy: latent u ~ N(e_1, 0.35^2 I_16) (+ shift * 0.35 e_2);
    member m's embedding (Q_m u + 0.08 xi) / ||Q_m u + 0.08 xi||.
    Returns (M, n, 16) and the (u, xi) used."""
    d = Q.shape[1]
    if u is None:
        u = rng.normal(0.0, 0.35, size=(n, d))
        u[:, 0] += 1.0
    if xi is None:
        xi = rng.normal(size=(Q.shape[0], n, d))
    us = u.copy()
    us[:, 1] += shift * 0.35
    z = np.einsum("mij,nj->mni", Q, us) + 0.08 * xi
    z /= np.linalg.norm(z, axis=2, keepdims=True)
    return z, u, xi


_TOY = {}


def _toy_sets():
    """Reference, rank and null sets of R34d (M = 5), shared by R34d and R34e."""
    if not _TOY:
        rng = _rng(4)
        Q = special_ortho_group.rvs(16, size=5, random_state=rng)
        _TOY["Q"] = Q
        _TOY["ref"] = _toy(rng, 800, Q)[0]
        _TOY["rank"] = _toy(rng, 500, Q)[0]
        _TOY["null"] = _toy(rng, 500, Q)[0]
        _TOY["rng"] = rng
    return _TOY


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def test_r34a_reference():
    """member_statistic equals npe_misspec.mmd2_multiscale point by point (specs/SPEC.md block 34)."""
    rng = _rng(1)
    z_ref = rng.normal(size=(300, 6)).astype(np.float32)
    z = (rng.normal(size=(50, 6)) * 1.5).astype(np.float32)
    t, bw, med = ev_gate.member_statistic(z_ref, z, FACTORS)
    assert t.shape == (50,) and t.dtype == np.float64, (t.shape, t.dtype)
    assert bw.shape == (5,) and bw.dtype == np.float64, (bw.shape, bw.dtype)
    # the bandwidths of the reference alone: scipy's median pairwise distance
    # (float64 on the float32 input) times the factors; 1e-12 relative is
    # float64 roundoff of the two distance formulas (pdist and _sqdist)
    med_ref = float(np.median(pdist(z_ref.astype(np.float64))))
    assert abs(med - med_ref) <= 1e-12 * med_ref, (med, med_ref)
    assert np.allclose(bw, np.asarray(FACTORS) * med_ref, rtol=1e-12, atol=0), bw
    z_ref64, z64 = z_ref.astype(np.float64), z.astype(np.float64)
    kaa = npe_misspec._self_term(z_ref64, bw)
    ref = np.array([npe_misspec.mmd2_multiscale(z_ref64, z64[i][None, :], bw, kaa)
                    for i in range(50)])
    # 1e-12: float64 roundoff of two orders of the same sums of O(1) terms
    # (n_h = 5 kernels, values <= n_h)
    err = float(np.max(np.abs(t - ref)))
    assert err <= 1e-12, "max |t - mmd2_multiscale| = %.3e" % err
    # t orders the points as their mean kernel similarity to the reference,
    # reversed: Spearman -1 (exact up to float64 roundoff of the coefficient)
    d2 = ((z64[:, None, :] - z_ref64[None, :, :]) ** 2).sum(-1)
    sim = sum(np.exp(-d2 / (2.0 * h * h)).mean(axis=1) for h in bw)
    rho = spearmanr(t, sim)[0]
    assert abs(rho + 1.0) <= 1e-12, "Spearman(t, similarity) = %r" % rho


def _check_close(got, want, name, tol=1e-12):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, "%s: shape %s, want %s" % (name, got.shape, want.shape)
    err = float(np.max(np.abs(got - want))) if got.size else 0.0
    assert err <= tol, "%s: max error %.3e\n got %r\n want %r" % (name, err, got, want)


def test_r34b_by_hand():
    """t, r, T and p from loops over (R5)-(R8) on a tiny case (specs/SPEC.md block 34)."""
    # M = 1, d_z = 2, factors (1.0,). The reference's pairwise distances are
    # 1, 1 and sqrt 2: median 1, bandwidth 1. t grows with the distance to the
    # triangle: (0.2, 0.1) < (0.5, 0.5) < (1, 1) < (2, 2) < (3, 3).
    ref = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
    rank = [[[0.5, 0.5], [1.0, 1.0]]]
    null = [[[0.2, 0.1], [2.0, 2.0]]]
    # test 0 is rank point 1 itself (t tied with it: the rank counts strictly
    # below); test 1 lies above both rank points (T = 1, tied with null 1:
    # the tie counts against the window in (R8))
    test = [[[1.0, 1.0], [3.0, 3.0]]]
    out = ev_gate.gate(np.array(ref), np.array(rank), np.array(null), np.array(test), (1.0,))
    want = _gate_loop(ref, rank, null, test, (1.0,))
    # 1e-12: float64 roundoff of sums of a few O(1) terms in two orders
    for k in ("t", "r", "p_member", "t_null", "r_null", "T", "T_null", "p", "median"):
        _check_close(out[k], want[k], "M=1 " + k)
    _check_close(out["bandwidths"], [[1.0]], "M=1 bandwidths")
    # the values the hand calculation above gives (exact rationals)
    _check_close(out["r"], [[0.5, 1.0]], "M=1 r, values", tol=0.0)
    _check_close(out["r_null"], [[0.0, 1.0]], "M=1 r_null, values", tol=0.0)
    _check_close(out["p"], [2.0 / 3.0, 2.0 / 3.0], "M=1 p, values", tol=1e-15)

    # M = 2: member 2 sees the same windows with other coordinates, in which
    # test 0 is nearer the reference than both rank points and test 1
    # between them: the members disagree and T is the mean of their ranks
    ref2 = [ref[0], [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]
    rank2 = [rank[0], [[3.0, 3.0], [0.8, 0.9]]]
    null2 = [null[0], [[0.5, 0.4], [4.0, 4.0]]]
    test2 = [test[0], [[0.6, 0.6], [1.5, 1.5]]]
    out2 = ev_gate.gate(np.array(ref2), np.array(rank2), np.array(null2),
                        np.array(test2), (0.5, 1.0))
    want2 = _gate_loop(ref2, rank2, null2, test2, (0.5, 1.0))
    for k in ("t", "r", "p_member", "t_null", "r_null", "T", "T_null", "p", "median"):
        _check_close(out2[k], want2[k], "M=2 " + k)
    assert not np.array_equal(out2["r"][0], out2["r"][1]), "the members agree: %r" % out2["r"]
    _check_close(out2["T"], out2["r"].mean(axis=0), "M=2 T = mean of r", tol=1e-15)


def test_r34c_null_as_test():
    """the null rows as test rows: T = T_null, p by hand (specs/SPEC.md block 34)."""
    rng = _rng(3)
    M, d = 3, 4
    z_ref = rng.normal(size=(M, 60, d))
    z_rank = rng.normal(size=(M, 40, d)) * 1.2
    z_null = rng.normal(size=(M, 30, d)) * 1.2
    out = ev_gate.gate(z_ref, z_rank, z_null, z_null, FACTORS)
    # exact: the null rows are ranked by the function that ranks a tested window
    assert np.array_equal(out["T"], out["T_null"]), (out["T"], out["T_null"])
    assert np.array_equal(out["r"], out["r_null"]), (out["r"], out["r_null"])
    assert np.array_equal(out["t"], out["t_null"])
    T = out["T_null"]
    p_hand = np.array([(1 + sum(1 for v in range(T.size) if T[v] >= T[j])) / (1.0 + T.size)
                       for j in range(T.size)])
    _check_close(out["p"], p_hand, "p", tol=0.0)
    # the case must contain ties of T for the convention to be exercised
    assert np.unique(T).size < T.size, "no tie of T among the null rows: %r" % T


def test_r34d_size():
    """the p-value's size on toy embeddings of one law (specs/SPEC.md block 34)."""
    S = _toy_sets()
    z_test = _toy(_rng(41), 4000, S["Q"])[0]
    out = ev_gate.gate(S["ref"], S["rank"], S["null"], z_test, FACTORS)
    p = out["p"]
    lines = []
    ok = True
    N_null = S["null"].shape[1]
    for a in (0.05, 0.1, 0.5):
        share = float(np.mean(p <= a))
        # bounds of SPEC R34d (amended, D-165): given the null set the size is
        # Beta(k_a, N_null + 1 - k_a) (R10), of mean k_a / (N_null + 1) and SD
        # about sqrt(a (1 - a) / (N_null + 2)); the 4,000 test points add the
        # binomial term a (1 - a) / 4000; 3 SDs of the sum. The lower bound is
        # 0.01 wider for the ties of T (multiples of 1 / 2500), which count
        # against the window in (R8) and make the gate more conservative
        k_a = int(np.floor(a * (N_null + 1)))
        s_a = np.sqrt(a * (1 - a) * (1.0 / 4000 + 1.0 / (N_null + 2)))
        lo = k_a / (N_null + 1.0) - 3 * s_a - 0.01
        hi = k_a / (N_null + 1.0) + 3 * s_a
        lines.append("a=%.2f share=%.4f bounds [%.4f, %.4f]" % (a, share, lo, hi))
        ok = ok and (lo <= share <= hi)
    print("  R34d: " + "; ".join(lines))
    assert ok, "\n".join(lines)


def test_r34h_size_law():
    """the size given the null set, (R10), on the member's own p-value (specs/SPEC.md block 34)."""
    # The toy of R34d with M = 1; one reference set (800) and one rank set
    # (500); 300 independent null sets of N_null = 99; one set of 2,000 test
    # points shared by all null sets. For each null set, the share of test
    # points with p_member <= 0.1, from ev_gate.gate itself.
    # (R10): k = floor(0.1 x 100) = 10, the size given the null set is
    # Beta(10, 90), mean 0.1, SD sqrt(10 x 90 / (100^2 x 101)) = 0.0299.
    # Tolerance of the mean, 0.026: 3 x 0.0299 / sqrt(300) = 0.0052 for the
    # 300 null sets, plus 3 sqrt(0.1 x 0.9 / 2000) = 0.020 for the one set of
    # test points, whose binomial error shifts every share alike.
    # Tolerance of the SD, 20 %: the relative standard error of an SD over
    # 300 sets is about 1 / sqrt(2 x 299) = 4 % (more with the Beta's mild
    # excess kurtosis); the shared test points move the mean, not the spread
    # (their own binomial term within a share is sqrt(q (1 - q) / 2000) per
    # set, small against 0.0299). t_m is continuous: no ties.
    rng = _rng(8)
    Q = special_ortho_group.rvs(16, random_state=rng)[None]
    z_ref = _toy(rng, 800, Q)[0]
    z_rank = _toy(rng, 500, Q)[0]
    z_test = _toy(rng, 2000, Q)[0]
    shares = np.empty(300)
    for b in range(300):
        z_null = _toy(rng, 99, Q)[0]
        pm = ev_gate.gate(z_ref, z_rank, z_null, z_test, FACTORS)["p_member"][0]
        shares[b] = np.mean(pm <= 0.1)
    sd_beta = np.sqrt(10.0 * 90.0 / (100.0 ** 2 * 101.0))
    mean, sd = float(shares.mean()), float(shares.std(ddof=1))
    msg = ("mean of 300 shares %.4f (want 0.1 +- 0.026); SD %.4f (want %.4f +- 20 %%, "
           "ratio %.3f)" % (mean, sd, sd_beta, sd / sd_beta))
    print("  R34h: " + msg)
    assert abs(mean - 0.1) <= 0.026, msg
    assert abs(sd / sd_beta - 1.0) <= 0.20, msg


def test_r34e_power():
    """the flagged share against the displacement (specs/SPEC.md block 34)."""
    S = _toy_sets()
    rng = _rng(51)
    _, u, xi = _toy(rng, 4000, S["Q"])
    shares = []
    for delta in (0.0, 2.0, 4.0, 8.0, 16.0):
        # the same latents and noise at every delta, only the displacement
        # changes (common random numbers)
        z_test = _toy(rng, 4000, S["Q"], shift=delta, xi=xi, u=u)[0]
        p = ev_gate.gate(S["ref"], S["rank"], S["null"], z_test, FACTORS)["p"]
        shares.append(float(np.mean(p < 0.05)))      # flagged: p < alpha_gate
    msg = "flagged share at deltas (0, 2, 4, 8, 16): %s" % ", ".join("%.4f" % s for s in shares)
    print("  R34e: " + msg)
    assert all(b >= a for a, b in zip(shares, shares[1:])), "not non-decreasing: " + msg
    assert shares[-1] > 0.9, "share at 16 not above 0.9: " + msg


def _roles_loop(donor, fracs, n_ref, n_rank, n_null, rpd, seed):
    """The rule of SPEC block 34 written plainly, with Python lists."""
    nets = sorted(set(int(d) for d in donor))
    perm = [int(v) for v in np.random.default_rng(seed).permutation(np.array(nets))]
    G = len(nets)
    sizes = [int(np.floor(f * G + 1e-9)) for f in fracs[:3]]
    sizes.append(G - sum(sizes))
    groups, s = [], 0
    for k in range(4):
        groups.append(set(perm[s:s + sizes[k]]))
        s += sizes[k]
    occ, seen = [], {}
    for d in donor:
        occ.append(seen.get(int(d), 0))
        seen[int(d)] = occ[-1] + 1
    role = [0] * len(donor)
    for k, (cap, code) in enumerate(((n_ref, 1), (n_rank, 2), (n_null, 3), (None, 4))):
        rows = sorted((occ[i], i) for i in range(len(donor)) if int(donor[i]) in groups[k])
        if k > 0:
            rows = [x for x in rows if x[0] < rpd]
        if cap is not None:
            rows = rows[:cap]
        for _, i in rows:
            role[i] = code
    return np.array(role, dtype=np.int8), sizes, groups


def test_r34f_roles():
    """assign_roles: groups, caps, order, determinism, refusal (specs/SPEC.md block 34)."""
    rng = _rng(6)
    G = 40
    counts = rng.integers(1, 6, size=G)                   # 1 to 5 rows each
    ids = rng.choice(np.arange(100, 1000), size=G, replace=False)   # unsorted ids
    donor = rng.permutation(np.repeat(ids, counts)).astype(np.int64)
    fracs = (0.33, 0.22, 0.22, 0.23)                      # 13.2, 8.8, 8.8 -> 13, 8, 8, 11
    big = 10 ** 6

    def per_net(role):
        return {int(n): set(role[donor == n].tolist()) - {0} for n in ids}

    # groups: no cap binding, every row of a group in the group's role
    role = ev_gate.assign_roles(donor, fracs, big, big, big, 5, seed=7)
    assert role.dtype == np.int8 and role.shape == donor.shape, (role.dtype, role.shape)
    pn = per_net(role)
    assert all(len(s) == 1 for s in pn.values()), "a network in two roles or none: %r" % pn
    n_per_role = [sum(1 for s in pn.values() if s == {c}) for c in (1, 2, 3, 4)]
    assert n_per_role == [13, 8, 8, 11], n_per_role
    want, sizes, _ = _roles_loop(donor, fracs, big, big, big, 5, 7)
    assert sizes == [13, 8, 8, 11] and np.array_equal(role, want)

    # caps and rows_per_donor
    for args in ((10, 5, 6, 2), (3, 100, 3, 1), (big, 7, big, 3)):
        role = ev_gate.assign_roles(donor, fracs, *args, seed=11)
        want, _, groups = _roles_loop(donor, fracs, *args, seed=11)
        assert np.array_equal(role, want), ("caps %r" % (args,), role, want)
        assert all(len(s) <= 1 for s in per_net(role).values())
        for c, cap in ((1, args[0]), (2, args[1]), (3, args[2])):
            avail = sum(min(int(cnt), args[3] if c > 1 else big)
                        for n, cnt in zip(ids, counts) if int(n) in groups[c - 1])
            assert int(np.sum(role == c)) == min(cap, avail), (c, int(np.sum(role == c)), cap, avail)
        for c in (2, 3, 4):
            for n in ids:
                assert int(np.sum((donor == n) & (role == c))) <= args[3], (c, n)
        avail_chk = sum(min(int(cnt), args[3]) for n, cnt in zip(ids, counts)
                        if int(n) in groups[3])
        assert int(np.sum(role == 4)) == avail_chk
        if args[0] < len(groups[0]):                      # only first rows in the reference
            seen = {}
            occ = []
            for d in donor:
                occ.append(seen.get(int(d), 0))
                seen[int(d)] = occ[-1] + 1
            assert all(occ[i] == 0 for i in np.flatnonzero(role == 1))

    # determinism
    a = ev_gate.assign_roles(donor, fracs, 10, 5, 6, 2, seed=11)
    b = ev_gate.assign_roles(donor, fracs, 10, 5, 6, 2, seed=11)
    c = ev_gate.assign_roles(donor, fracs, 10, 5, 6, 2, seed=12)
    assert np.array_equal(a, b) and not np.array_equal(a, c)

    # moving the rows of one network together: the roles follow the rows.
    # The caps do not bind here (with a binding cap the round-robin position
    # of the moved network, and so its rows' roles, may change)
    n0 = ids[int(np.argmax(counts))]
    keep = np.flatnonzero(donor != n0)
    move = np.flatnonzero(donor == n0)
    order = np.concatenate([keep, move])
    r1 = ev_gate.assign_roles(donor, fracs, big, big, big, 2, seed=11)
    r2 = ev_gate.assign_roles(donor[order], fracs, big, big, big, 2, seed=11)
    assert np.array_equal(r1[order], r2)

    # 3 networks: floor(0.33 * 3) = 0 networks in the reference group
    try:
        ev_gate.assign_roles(np.array([5, 6, 7, 5]), fracs, 10, 10, 10, 1, seed=0)
    except ValueError as exc:
        assert "3" in str(exc), str(exc)
    else:
        raise AssertionError("3 networks were not refused")


def test_r34g_edges():
    """non-finite input refused; no test row (specs/SPEC.md block 34)."""
    rng = _rng(7)
    M, d = 2, 3
    z_ref = rng.normal(size=(M, 20, d))
    z_rank = rng.normal(size=(M, 10, d))
    z_null = rng.normal(size=(M, 12, d))
    z_test = rng.normal(size=(M, 5, d))
    for name, bad_val in (("z_ref", np.nan), ("z_rank", np.inf), ("z_null", -np.inf),
                          ("z_test", np.nan)):
        args = {"z_ref": z_ref.copy(), "z_rank": z_rank.copy(), "z_null": z_null.copy(),
                "z_test": z_test.copy()}
        args[name][1, 2, 0] = bad_val
        try:
            ev_gate.gate(args["z_ref"], args["z_rank"], args["z_null"], args["z_test"], FACTORS)
        except ValueError:
            pass
        else:
            raise AssertionError("%s with %r not refused" % (name, bad_val))
    for zr, zz in ((z_ref[0], np.array([[np.nan, 0.0, 0.0]])),
                   (np.vstack([z_ref[0], [[np.inf, 0, 0]]]), z_test[0])):
        try:
            ev_gate.member_statistic(zr, zz, FACTORS)
        except ValueError:
            pass
        else:
            raise AssertionError("member_statistic accepted a non-finite input")
    out = ev_gate.gate(z_ref, z_rank, z_null, z_test[:, :0], FACTORS)
    shapes = {"t": (M, 0), "r": (M, 0), "p_member": (M, 0), "T": (0,), "p": (0,),
              "t_rank": (M, 10), "t_null": (M, 12), "r_null": (M, 12), "T_null": (12,),
              "bandwidths": (M, 5), "median": (M,)}
    assert set(out) == set(shapes), sorted(out)
    for k, s in shapes.items():
        assert out[k].shape == s and out[k].dtype == np.float64, (k, out[k].shape, out[k].dtype)
    t0, _, _ = ev_gate.member_statistic(z_ref[0], z_test[0, :0], FACTORS)
    assert t0.shape == (0,) and t0.dtype == np.float64


CHECKS = [("test_r34a_reference", test_r34a_reference), ("test_r34b_by_hand", test_r34b_by_hand), ("test_r34c_null_as_test", test_r34c_null_as_test), ("test_r34d_size", test_r34d_size), ("test_r34e_power", test_r34e_power), ("test_r34f_roles", test_r34f_roles), ("test_r34g_edges", test_r34g_edges), ("test_r34h_size_law", test_r34h_size_law)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except NotImplementedError as exc:
            bad += 1
            rows.append((name, "TODO", str(exc), time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except AssertionError as exc:
            bad += 1
            rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
        except Exception:
            bad += 1
            rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
