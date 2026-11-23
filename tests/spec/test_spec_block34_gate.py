"""Spec tests of block 34, the gate (specs/SPEC.md sec. 2.6 "The gate",
(R5)-(R10), and block 34's entry). Integration tester, 2026-11-23, incremental
run on dc096e5.

Every expected value is computed here from the definitions, by loops or with
scipy, never by calling the code under test to produce the reference.
"""

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist, pdist

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", "..", "hpc", "pipeline", "evaluation"))
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
sys.dont_write_bytecode = True

import ev_gate  # noqa: E402


# ---------------------------------------------------------------- oracles

def kernel(a, b, h):
    """(R5) by definition: sum_b exp(-||a - b||^2 / (2 h_b^2)), (len a, len b)."""
    d2 = cdist(a, b, "sqeuclidean")
    return sum(np.exp(-d2 / (2.0 * hb ** 2)) for hb in h)


def t_oracle(z_ref, z, factors):
    """(R5), (R6): med = median of the Euclidean distances between the pairs
    of DISTINCT reference points (scipy pdist), h_b = f_b med."""
    med = float(np.median(pdist(z_ref)))
    h = np.asarray(factors, float) * med
    n_h = len(factors)
    first = kernel(z_ref, z_ref, h).mean()
    cross = kernel(z_ref, z, h).mean(axis=0)
    return first + n_h - 2.0 * cross, h, med


def gate_oracle(z_ref, z_rank, z_null, z_test, factors):
    """(R6)-(R8) by loops over the definitions."""
    M = z_ref.shape[0]
    N_rank, N_null, n = z_rank.shape[1], z_null.shape[1], z_test.shape[1]
    r = np.zeros((M, n))
    r_null = np.zeros((M, N_null))
    pm = np.zeros((M, n))
    for m in range(M):
        t_rank, _, _ = t_oracle(z_ref[m], z_rank[m], factors)
        t_null, _, _ = t_oracle(z_ref[m], z_null[m], factors)
        t_test, _, _ = t_oracle(z_ref[m], z_test[m], factors)
        for j in range(n):
            r[m, j] = sum(1 for a in range(N_rank) if t_rank[a] < t_test[j]) / N_rank
            pm[m, j] = (1 + sum(1 for v in range(N_null) if t_null[v] >= t_test[j])) / (1 + N_null)
        for v in range(N_null):
            r_null[m, v] = sum(1 for a in range(N_rank) if t_rank[a] < t_null[v]) / N_rank
    T = r.mean(axis=0)
    T_null = r_null.mean(axis=0)
    p = np.array([(1 + sum(1 for v in range(N_null) if T_null[v] >= T[j])) / (1 + N_null)
                  for j in range(n)])
    return dict(r=r, r_null=r_null, T=T, T_null=T_null, p=p, p_member=pm)


def toy(rng, M, n, d=16, shift=0.0):
    """Toy embeddings of one law (block 34's R34d): u ~ N(e_1, 0.35^2 I),
    member m's embedding normalised (Q_m u + 0.08 xi)."""
    Qs = [np.linalg.qr(np.random.default_rng(100 + m).standard_normal((d, d)))[0]
          for m in range(M)]
    u = rng.standard_normal((n, d)) * 0.35
    u[:, 0] += 1.0
    u[:, 1] += shift * 0.35
    out = []
    for m in range(M):
        e = u @ Qs[m].T + 0.08 * rng.standard_normal((n, d))
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.stack(out)


# ---------------------------------------------------------------- (R5)-(R6)

def test_member_statistic_equals_definition():
    rng = np.random.default_rng(1)
    z_ref = rng.standard_normal((37, 4))
    z = rng.standard_normal((23, 4)) * 1.5
    fac = (0.25, 0.5, 1.0, 2.0, 4.0)
    t, bw, med = ev_gate.member_statistic(z_ref, z, fac)
    t_o, h_o, med_o = t_oracle(z_ref, z, fac)
    # 1e-12: float64 roundoff of two summation orders of <= 37*37*5 terms of size <= 1
    assert_allclose(t, t_o, rtol=0, atol=1e-12)
    assert_allclose(bw, h_o, rtol=1e-14)
    assert abs(med - med_o) < 1e-14
    assert t.dtype == np.float64 and t.shape == (23,)
    # t >= 0: a biased squared MMD is a squared RKHS norm
    assert np.all(t >= -1e-12)


def test_member_statistic_empty_and_float32_cast():
    rng = np.random.default_rng(2)
    z_ref = rng.standard_normal((10, 3)).astype(np.float32)
    t, bw, _ = ev_gate.member_statistic(z_ref, np.zeros((0, 3), np.float32), (1.0,))
    assert t.shape == (0,)
    # the embeddings are cast to float64 before the bandwidths (D-165): the
    # median equals pdist on the float64 cast within float64 roundoff
    assert abs(bw[0] - np.median(pdist(z_ref.astype(np.float64)))) < 1e-12


# ---------------------------------------------------------------- (R7), (R8)

def test_gate_equals_loops_M2():
    rng = np.random.default_rng(3)
    M, d = 2, 3
    z_ref = rng.standard_normal((M, 12, d))
    z_rank = rng.standard_normal((M, 9, d)) * 1.2
    z_null = rng.standard_normal((M, 7, d)) * 1.2
    z_test = rng.standard_normal((M, 11, d)) * 1.6
    out = ev_gate.gate(z_ref, z_rank, z_null, z_test, (0.5, 1.0, 2.0))
    o = gate_oracle(z_ref, z_rank, z_null, z_test, (0.5, 1.0, 2.0))
    for k in ("r", "r_null", "T", "T_null", "p", "p_member"):
        # the ranks and p-values are ratios of small integers: exact up to one rounding
        assert_allclose(out[k], o[k], rtol=0, atol=1e-12, err_msg=k)
    assert out["t"].shape == (M, 11) and out["T"].shape == (11,)
    assert out["bandwidths"].shape == (M, 3) and out["median"].shape == (M,)


def test_gate_null_rows_as_test_rows():
    """R34c: the null rows tested as windows are ranked by the same function."""
    rng = np.random.default_rng(4)
    z_ref, z_rank, z_null = (rng.standard_normal((2, k, 3)) for k in (15, 10, 8))
    out = ev_gate.gate(z_ref, z_rank, z_null, z_null, (1.0,))
    assert_array_equal(out["T"], out["T_null"])
    assert_array_equal(out["r"], out["r_null"])
    p = np.array([(1 + np.sum(out["T_null"] >= T)) / 9.0 for T in out["T_null"]])
    assert_allclose(out["p"], p, rtol=0, atol=0)


def test_tie_counts_against_window():
    """(R8): a null value equal to T(x) counts (>=). A test row identical to a
    null row (same embedding in every member) has the same T; its p must count
    that null row."""
    rng = np.random.default_rng(5)
    z_ref, z_rank, z_null = (rng.standard_normal((1, k, 2)) for k in (6, 5, 4))
    z_test = z_null[:, [0], :].copy()
    out = ev_gate.gate(z_ref, z_rank, z_null, z_test, (1.0,))
    c = int(np.sum(out["T_null"] >= out["T"][0]))
    assert c >= 1
    assert out["p"][0] == (1 + c) / 5.0


def test_gate_refusals():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((1, 5, 2))
    bad = a.copy(); bad[0, 0, 0] = np.nan
    for args in ((bad, a, a, a), (a, bad, a, a), (a, a, bad, a), (a, a, a, bad)):
        with pytest.raises(ValueError):
            ev_gate.gate(*args, (1.0,))
    inf = a.copy(); inf[0, 1, 1] = np.inf
    with pytest.raises(ValueError):
        ev_gate.gate(a, a, a, inf, (1.0,))
    with pytest.raises(ValueError):           # M differs
        ev_gate.gate(a, np.concatenate([a, a]), a, a, (1.0,))
    with pytest.raises(ValueError):           # a non-positive factor
        ev_gate.gate(a, a, a, a, (1.0, 0.0))
    with pytest.raises(ValueError):           # an empty null set
        ev_gate.gate(a, a, a[:, :0], a, (1.0,))
    out = ev_gate.gate(a, a, a, a[:, :0], (1.0, 2.0))
    assert out["T"].shape == (0,) and out["p"].shape == (0,)
    assert out["t"].shape == (1, 0) and out["p_member"].shape == (1, 0)


# ---------------------------------------------------------------- (R9), (R10)

def test_flag_iff_T_exceeds_kgate_largest_null():
    """(R9): p < alpha  <=>  T > the k_gate-th largest null value, k_gate =
    ceil(alpha (N+1)) - 1; checked on the gate's own T and T_null."""
    rng = np.random.default_rng(7)
    z = toy(rng, 3, 200 + 150 + 99 + 400)
    z_ref, z_rank, z_null, z_test = z[:, :200], z[:, 200:350], z[:, 350:449], z[:, 449:]
    out = ev_gate.gate(z_ref, z_rank, z_null, z_test, (0.25, 0.5, 1, 2, 4))
    for alpha in (0.05, 0.1, 0.2):
        k = int(np.ceil(alpha * 100)) - 1
        thr = np.sort(out["T_null"])[::-1][k - 1]
        assert_array_equal(out["p"] < alpha, out["T"] > thr)


def test_size_given_null_set_member_pvalue():
    """(R10) on the member's own p-value (no ties): over 150 independent null
    sets of 99 points, the share of 1,000 fixed test points with p_member <= 0.1
    has mean ~0.1 and SD ~ Beta(10, 90)'s 0.0299. A smaller replica of R34h:
    tolerance on the mean 3*0.0299/sqrt(150) + 3*sqrt(0.09/1000) = 0.0073+0.028
    = 0.036; on the SD 25 % (the relative SE of an SD over 150 sets is ~6 %,
    plus the shared test points)."""
    rng = np.random.default_rng(8)
    z_ref = toy(rng, 1, 400)
    z_rank = toy(rng, 1, 200)
    z_test = toy(rng, 1, 1000)
    shares = []
    for _ in range(150):
        z_null = toy(rng, 1, 99)
        out = ev_gate.gate(z_ref, z_rank, z_null, z_test, (0.25, 0.5, 1, 2, 4))
        shares.append(np.mean(out["p_member"][0] <= 0.1))
    shares = np.asarray(shares)
    assert abs(shares.mean() - 0.1) < 0.036, shares.mean()
    assert abs(shares.std(ddof=1) / 0.029851 - 1) < 0.25, shares.std(ddof=1)


def test_power_monotone_toy():
    """R34e, reduced: shifting the test latents moves the flagged share up."""
    rng = np.random.default_rng(9)
    z_ref, z_rank, z_null = toy(rng, 3, 300), toy(rng, 3, 200), toy(rng, 3, 200)
    shares = []
    for delta in (0, 4, 16):
        zt = toy(np.random.default_rng(10), 3, 800, shift=delta)
        out = ev_gate.gate(z_ref, z_rank, z_null, zt, (0.25, 0.5, 1, 2, 4))
        shares.append(np.mean(out["p"] < 0.05))
    assert shares[0] < 0.12 and shares[0] <= shares[1] <= shares[2] and shares[2] > 0.9, shares


# ---------------------------------------------------------------- assign_roles

def roles_oracle(donor, fracs, n_ref, n_rank, n_null, rpd, seed):
    """Block 34's rule written from the entry, with plain loops."""
    nets = sorted(set(int(d) for d in donor))
    G = len(nets)
    perm = [int(v) for v in np.random.default_rng(seed).permutation(np.asarray(nets))]
    sizes = [int(np.floor(f * G + 1e-9)) for f in fracs[:3]]
    sizes.append(G - sum(sizes))
    groups, s = [], 0
    for k in range(4):
        groups.append(set(perm[s:s + sizes[k]])); s += sizes[k]
    occ, seen = [], {}
    for d in donor:
        occ.append(seen.get(int(d), 0)); seen[int(d)] = occ[-1] + 1
    role = [0] * len(donor)
    caps = (n_ref, n_rank, n_null, None)
    for k in range(4):
        rows = sorted((i for i in range(len(donor)) if int(donor[i]) in groups[k]),
                      key=lambda i: (occ[i], i))
        if k > 0:
            rows = [i for i in rows if occ[i] < rpd]
        if caps[k] is not None:
            rows = rows[:caps[k]]
        for i in rows:
            role[i] = k + 1
    return np.asarray(role), sizes


def test_assign_roles_by_hand_rule():
    rng = np.random.default_rng(11)
    donor = np.repeat(np.arange(40) * 3 + 7, rng.integers(1, 6, 40))
    rng.shuffle(donor)
    for caps in ((800, 1000, 1000), (5, 4, 3), (1000, 2, 1000)):
        for rpd in (1, 2):
            for seed in (0, 5):
                fr = (0.4, 0.25, 0.25, 0.1)
                got = ev_gate.assign_roles(donor, fr, *caps, rpd, seed)
                exp, sizes = roles_oracle(donor, fr, *caps, rpd, seed)
                assert got.dtype == np.int8 and got.shape == donor.shape
                assert_array_equal(got, exp)
                assert sizes == [16, 10, 10, 4]
                # no network in two groups
                for d in np.unique(donor):
                    rs = set(got[donor == d].tolist()) - {0}
                    assert len(rs) <= 1


def test_assign_roles_floor_roundoff_and_refusals():
    donor = np.arange(100)
    got = ev_gate.assign_roles(donor, (0.29, 0.31, 0.3, 0.1), 1000, 1000, 1000, 1, 0)
    # floor(0.29 * 100 + 1e-9) = 29, not 28 (0.29*100 = 28.999999999999996)
    assert [(got == k).sum() for k in (1, 2, 3, 4)] == [29, 31, 30, 10]
    with pytest.raises(ValueError):
        ev_gate.assign_roles(np.arange(3), (0.4, 0.25, 0.25, 0.1), 10, 10, 10, 1, 0)
    with pytest.raises(ValueError):
        ev_gate.assign_roles(np.arange(40), (0.4, 0.25, 0.25, 0.05), 10, 10, 10, 1, 0)
    with pytest.raises(ValueError):
        ev_gate.assign_roles(np.arange(40), (0.4, 0.25, 0.25, 0.1), 10, 10, 10, 0, 0)
    with pytest.raises(ValueError):
        ev_gate.assign_roles(np.arange(40.0), (0.4, 0.25, 0.25, 0.1), 10, 10, 10, 1, 0)
    with pytest.raises(ValueError):
        ev_gate.assign_roles(np.arange(40), (0.4, 0.25, 0.25, 0.1), -1, 10, 10, 1, 0)


def test_assign_roles_nan_fraction_refused():
    """A NaN fraction must not pass the sum check silently."""
    with pytest.raises(ValueError):
        ev_gate.assign_roles(np.arange(40), (0.4, float("nan"), 0.25, 0.1), 10, 10, 10, 1, 0)
