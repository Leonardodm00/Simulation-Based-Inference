"""Block 3 (e3_numbers.py): spec-derived tests.

Oracles (SPEC Block 3): simplex ETF for C in {2,3,4} (unit vectors, pairwise
cosine -1/(C-1), sum zero); r_eff = (tr S)^2 / tr(S^2) is 1 for a rank-one
cloud, d for an isotropic cloud, scale invariant, and on a constant cloud each
implementation behaves as E3 reports (F-bb); the numpy loss transcription
agrees with P2 eqs. (P2.3)-(P2.10) term by term on small hand-checkable
inputs; [RAN] numbers of E3 equal the printed ones; two identical runs.
"""
import itertools
import math
import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conftest import TOOLS, doc_text, floats, git_show, line_with, load_functions, \
    run_script, script_output

sys.path.insert(0, TOOLS)
import e3_numbers as e3          # noqa: E402  (only its loss transcription is unit-tested)


@pytest.fixture(scope="module")
def out():
    return script_output("e3_numbers.py")


@pytest.fixture(scope="module")
def impls():
    d = load_functions(git_show("hpc/dsn/metrics.py"), ("_to_numpy", "_as_2d", "effective_rank"))
    j = load_functions(git_show("hpc/joint/stage3/joint_diagnostics.py"), ("effective_rank",))
    p = load_functions(git_show("hpc/joint/stage3b/encoder_probes.py"), ("effective_rank",))
    return {"dsn": d["effective_rank"], "joint": j["effective_rank"], "probes": p["effective_rank"]}


# ---------------------------------------------------------------- B1
@pytest.mark.parametrize("C", [2, 3, 4])
def test_b1_etf_printed(out, C):
    target = -1.0 / (C - 1)
    ln = line_with(out, "C = %d: pairwise cosine" % C)
    assert "pairwise cosine %.6f" % target in ln
    assert "%d non-zero eigenvalues, each %.6f" % (C - 1, C / (C - 1)) in ln
    # independent ETF: vertices of the regular simplex from the centred identity via SVD
    M = np.eye(C) - 1.0 / C
    U, s, _ = np.linalg.svd(M)
    V = U[:, :C - 1] * s[:C - 1]
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    G = V @ V.T
    assert_allclose(G[~np.eye(C, dtype=bool)], target, atol=1e-12)   # machine precision
    assert_allclose(V.sum(0), 0.0, atol=1e-12)
    assert "|sum of vertices| = " in ln and floats(ln.split("|sum of vertices| =")[1])[0] < 1e-12


# ---------------------------------------------------------------- B2: r_eff
def test_r_eff_rank_one_isotropic_scale(impls):
    rng = np.random.default_rng(7)
    line = rng.standard_normal((300, 1)) @ rng.standard_normal((1, 6))
    for name, f in impls.items():
        assert f(line) == pytest.approx(1.0, abs=1e-9), name
        # isotropic cloud in d = 6: exact isotropy built from an orthogonal design
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        iso = np.concatenate([Q, -Q]) * 3.0          # covariance proportional to I_6
        assert f(iso) == pytest.approx(6.0, rel=1e-9), name
        cloud = rng.standard_normal((200, 6)) * np.arange(1, 7)
        assert f(1e3 * cloud) == pytest.approx(f(cloud), rel=1e-9), name   # scale invariance
        # definition (tr S)^2 / tr(S^2) against numpy's covariance
        S = np.cov(cloud, rowvar=False)
        assert f(cloud) == pytest.approx(np.trace(S) ** 2 / np.trace(S @ S), rel=1e-9), name


def test_r_eff_constant_cloud_as_e3_reports(impls, out):
    """E3 F-bb: exactly zero covariance -> DSN 0.0, joint and Stage 3b 1.0."""
    z = np.zeros((50, 12))
    z[:, 0] = 1.0
    assert impls["dsn"](z) == 0.0
    assert impls["joint"](z) == 1.0
    assert impls["probes"](z) == 1.0
    blk = out.split("a constant cloud")[1]
    first = blk.split("on a generic unit vector")[0]
    assert "DSN metrics.effective_rank               r_eff = 0.0" in first
    assert first.count("r_eff = 1.0") == 2


def test_b2_unequal_masses_closed_form(out):
    # ETF collapse with masses w: Sigma = sum w_c u_c u_c^T - mu mu^T (population cov,
    # the n/(n-1) factor cancels in the ratio)
    for counts, quoted in (((72, 108, 108), "1.960000"), ((1, 1, 8), "1.710059")):
        C = len(counts)
        M = np.eye(C) - 1.0 / C
        U = M / np.linalg.norm(M, axis=1, keepdims=True)
        w = np.asarray(counts, float) / sum(counts)
        mu = w @ U
        S = (U.T * w) @ U - np.outer(mu, mu)
        r = np.trace(S) ** 2 / np.sum(S * S)
        assert "%.6f" % r == quoted
        assert "counts %s: r_eff = %s" % (counts, quoted) in out


def test_b2_rim_cap_formula_by_monte_carlo():
    # E3 eq. (E3.11): two antipodal rim caps, equal masses, E = 10 -> 1/(c^4 + s^4/(E-1))
    rng = np.random.default_rng(11)
    E, rho = 10, math.radians(5.0)
    u = np.zeros(E)
    u[0] = 1.0
    pts = []
    for sgn in (1, -1):
        g = rng.standard_normal((40000, E))
        g[:, 0] = 0.0
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        pts.append(sgn * math.cos(rho) * u + math.sin(rho) * g)
    z = np.concatenate(pts)
    S = np.cov(z, rowvar=False)
    mc = np.trace(S) ** 2 / np.sum(S * S)
    closed = 1 / (math.cos(rho) ** 4 + math.sin(rho) ** 4 / (E - 1))
    assert mc == pytest.approx(closed, abs=2e-4)   # 80k points: MC error ~1e-5 on r_eff


# ---------------------------------------------------------------- B4: the loss transcription
def _Q(z):
    return np.array([[float(np.sum((a - b) ** 2)) for b in z] for a in z])


def _mined_ref(Q, y, strategy):
    """P2 eq. (P2.3) by brute-force enumeration of the set definitions."""
    n = len(y)
    T = []
    for i in range(n):
        P = [j for j in range(n) if j != i and y[j] == y[i]]
        O = [k for k in range(n) if y[k] != y[i]]
        if not P or not O:
            continue
        if strategy == "hard":
            T += [(i, j, k) for j in P for k in O if Q[i, k] <= Q[i, j]]
            continue
        jp = min(P, key=lambda j: (Q[i, j], j))
        if strategy == "easy_positive":
            T.append((i, jp, min(O, key=lambda k: (Q[i, k], k))))
        else:
            adm = [k for k in O if Q[i, k] > Q[i, jp]]
            if adm:
                T.append((i, jp, min(adm, key=lambda k: (Q[i, k], k))))
    return T


def _l_joint_ref(z, y, strategy, strict, m_cos, alpha_deg):
    Q = _Q(z)
    m_sq = 2 * m_cos
    tot = []
    for (i, j, k) in _mined_ref(Q, y, strategy):
        trip = max(0.0, Q[i, j] - min(Q[i, k], Q[j, k]) + m_sq)                      # (P2.4)
        mid = 0.5 * (z[i] + z[j])
        ang = max(0.0, Q[i, j] - 4 * math.tan(math.radians(alpha_deg)) ** 2
                  * float(np.sum((z[k] - mid) ** 2)))                                   # (P2.5)
        keep = (not strict) or (Q[i, j] < Q[i, k] < Q[i, j] + m_sq
                                and Q[i, j] < Q[j, k] < Q[i, j] + m_sq)                 # (P2.6a)
        if keep:
            tot.append(trip + ang)
    n_act = sum(1 for t in tot if t > 0)
    return sum(tot) / max(1, n_act)                                                     # (P2.6)


def _circle(angles_deg):
    a = np.radians(angles_deg)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


CASES = [
    # (angles in degrees, labels); chosen with no distance ties, because P2.3's
    # `<=` / argmin on an exact geometric tie is decided by float rounding
    ([0, 30, 50, 180], [0, 0, 1, 1]),
    ([0, 9, 21, 27, 43, 200], [0, 0, 0, 1, 1, 1]),
    ([0, 5, 115, 125, 240, 250], [0, 0, 1, 1, 2, 2]),
    ([0, 60, 70, 100, 110, 295], [0, 1, 0, 1, 2, 2]),
]


@pytest.mark.parametrize("angles,labels", CASES)
@pytest.mark.parametrize("strategy", ["hard", "easy_positive", "easy_pos_semihard_neg"])
def test_miner_matches_p2_3(angles, labels, strategy):
    z, y = _circle(angles), np.asarray(labels)
    got = sorted(map(tuple, e3._mine(e3._sqdist(z), y, strategy).tolist()))
    want = sorted(_mined_ref(_Q(z), y, strategy))
    assert got == want


@pytest.mark.parametrize("angles,labels", CASES)
@pytest.mark.parametrize("strategy,strict,m,alpha", [
    ("easy_pos_semihard_neg", True, 0.2, 18.0),
    ("easy_pos_semihard_neg", False, 0.2, 18.0),
    ("hard", False, 0.3, 10.6873),
    ("easy_positive", False, 0.5, 20.0),
])
def test_l_joint_matches_p2_4_to_6(angles, labels, strategy, strict, m, alpha):
    z, y = _circle(angles), np.asarray(labels)
    got, _, _ = e3._l_joint(z, y, strategy, strict, m, alpha)
    # tolerance: sums of O(1) terms in float64 -> 1e-12
    assert got == pytest.approx(_l_joint_ref(z, y, strategy, strict, m, alpha), abs=1e-12)


def test_hand_example_triplet_and_angle():
    """One triplet worked by hand: anchor at 0 deg, positive at 30 deg, negative at
    50 deg (labels 0,0,1), easy_positive, m_cos = 0.5, alpha = 20 deg."""
    z, y = _circle([0, 30, 50]), np.array([0, 0, 1])
    c30, c20, c50 = math.cos(math.radians(30)), math.cos(math.radians(20)), math.cos(math.radians(50))
    Qap, Qan, Qpn = 2 - 2 * c30, 2 - 2 * c50, 2 - 2 * c20
    # anchor 0 -> (0,1,2); anchor 1 -> (1,0,2); anchor 2 has no positive
    trip0 = max(0, Qap - min(Qan, Qpn) + 1.0)
    trip1 = max(0, Qap - min(Qpn, Qan) + 1.0)
    mid = 0.5 * (z[0] + z[1])
    qmid = float(np.sum((z[2] - mid) ** 2))
    # Apollonius: Q_mid = (2 Qan + 2 Qpn - Qap) / 4
    assert qmid == pytest.approx((2 * Qan + 2 * Qpn - Qap) / 4, abs=1e-14)
    ang = max(0, Qap - 4 * math.tan(math.radians(20)) ** 2 * qmid)
    terms = [trip0 + ang, trip1 + ang]
    want = sum(terms) / sum(1 for t in terms if t > 0)
    got, n_mined, _ = e3._l_joint(z, y, "easy_positive", False, 0.5, 20.0)
    assert n_mined == 2
    assert got == pytest.approx(want, abs=1e-12)


def test_triplet_p2_7_is_half_of_margin_hinge():
    z, y = _circle([0, 9, 21, 27, 43, 200]), np.array([0, 0, 0, 1, 1, 1])
    Q = _Q(z)
    T = _mined_ref(Q, y, "easy_positive")
    d = Q / 2
    ls = [max(0.0, d[i, j] - min(d[i, k], d[j, k]) + 0.2) for i, j, k in T]
    pos = [v for v in ls if v > 0]
    want = sum(pos) / len(pos) if pos else 0.0
    assert e3._l_triplet(z, y, "easy_positive", 0.2) == pytest.approx(want, abs=1e-12)


def test_l_sep_p2_8_hand_values():
    # K = 2, class mean directions at 0 and 90 deg: L = 2 (0 + 1)^2 / (2*1) = 1
    z = _circle([-10, 10, 80, 100])
    y = np.array([0, 0, 1, 1])
    val, K = e3._l_sep(z, y, 2)
    assert K == 2 and val == pytest.approx(1.0, abs=1e-12)
    # K = 3 at 0/120/240: exactly the ETF -> 0
    val, K = e3._l_sep(_circle([0, 0, 120, 120, 240, 240]), np.array([0, 0, 1, 1, 2, 2]), 3)
    assert K == 3 and val == pytest.approx(0.0, abs=1e-24)
    # a class with one row is not valid (min_per_class = 2): K = 2 among classes 0, 1
    z = _circle([0, 2, 90, 92, 200])
    y = np.array([0, 0, 1, 1, 2])
    val, K = e3._l_sep(z, y, 3)
    v0 = z[:2].mean(0) / np.linalg.norm(z[:2].mean(0))
    v1 = z[2:4].mean(0) / np.linalg.norm(z[2:4].mean(0))
    assert K == 2 and val == pytest.approx((v0 @ v1 + 1.0) ** 2, abs=1e-12)
    # raw (not centred) means: a uniform shrink of one class's cap does not change v_c
    # only through normalisation -- check against the formula with a non-unit mean
    z = _circle([0, 40, 130, 170])
    m0, m1 = z[:2].mean(0), z[2:].mean(0)
    want = ((m0 / np.linalg.norm(m0)) @ (m1 / np.linalg.norm(m1)) + 1.0) ** 2
    assert e3._l_sep(z, np.array([0, 0, 1, 1]), 2)[0] == pytest.approx(want, abs=1e-12)


def test_b4_printed_table_values(out):
    e3d = doc_text("E3_THE_SUMMARY_NETWORK.md")
    rows = [("C = 3, 10 rows/class, rho =  5.0", "3.8 \\times 10^{-6}"),
            ("C = 2,  9 rows/class, rho = 30.0", "2.4 \\times 10^{-3}")]
    for key, quoted in rows:
        comp = floats(line_with(out, key).split("composite")[1])[0]
        mant, exp = quoted.split(" \\times 10^{")
        assert "%.1e" % comp == "%se%+03d" % (mant, int(exp.rstrip("}")))
        assert quoted in e3d
    assert "log-log slope 4.00" in line_with(out, "C = 2: E[L_sep] at rho")
    assert "log-log slope 2.00" in line_with(out, "C = 3: E[L_sep] at rho")


def test_b5_b6_closed_forms(out):
    for ratio, s in ((1.0, "0.347"), (0.1, "0.00498"), (0.01, "5e-05")):
        assert ("%.3g" % (0.5 * math.log(1 + ratio ** 2))) == s
        assert "I = 0.5 ln(1 + ratio^2) = %s nats" % s in out
    # Fano bound at C = 3, tau 0.30: h_b(e) + e ln 2 with the printed Bayes error
    ln = line_with(out, "C = 3, bench simplex_centres      tau_ov 0.30")
    e = floats(ln.split("Bayes error")[1])[0]
    hb = -e * math.log(e) - (1 - e) * math.log(1 - e)
    fano = floats(ln.split("h_b + e ln(C-1) =")[1])[0]
    assert fano == pytest.approx(hb + e * math.log(2), abs=6e-4)   # e printed to 4 decimals


def test_two_runs_identical(out):
    assert run_script("e3_numbers.py").stdout == out
