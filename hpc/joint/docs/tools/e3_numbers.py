#!/usr/bin/env python3
"""e3_numbers.py -- every [RAN] number of E3_THE_SUMMARY_NETWORK.md, recomputed.

Torch-free: numpy, scipy and scikit-learn only. E3 states what the DSN's
metric loss asks of the embedding geometry, what a collapsed code can carry
about theta, and what the participation-ratio rank r_eff can and cannot see.
Every number it quotes is printed here.

Sources, kept apart as in e2_numbers.py:

  * [REPO] functions READ from the repository at the freeze commit with
    `git show` and executed alone (AST extraction, nothing else of the module
    runs): the DSN's `effective_rank` (hpc/dsn/metrics.py, with its two
    helpers), the joint stack's `effective_rank` and `cluster_scores`
    (hpc/joint/stage3/joint_diagnostics.py), the Stage 3b duplicate of
    `effective_rank` (hpc/joint/stage3b/encoder_probes.py) and the DSN
    generator's `_class_center_vectors` (hpc/dsn/latent_burst_generator.py).
    The modules themselves import torch, which the sandbox does not have.
  * [REPO] the bench prior: hpc/joint/stage1 is imported from the working tree
    after checking it is byte-identical to the freeze (`git diff --quiet`).
  * a numpy TRANSCRIPTION of the composite metric loss, eqs. (P2.3)-(P2.10)
    of P2_DSN_LOSS_AXES.md -- not the repository's torch code. Its numbers are
    statements about the equations as P2 transcribes them from
    hpc/dsn/dsn_joint_loss.py and pytorch_metric_learning 1.6.3.

Blocks:
  B1  the simplex equiangular tight frame (ETF) for C = 2, 3, 4
  B2  r_eff of collapsed and nearly collapsed clouds on the unit sphere; the
      three implementations on a constant cloud (finding F-bb)
  B3  what the logged cluster scores and the trace add to r_eff
  B4  the zero sets of the composite loss (and of the plain triplet loss),
      and the separation term's pull on within-class spread (finding F-bc)
  B5  the label code on the bench: I(theta; c), the Bayes error, the Fano bound
      on the free axes; the bench's centres against the DSN's (F-ba)
  B6  the residual channel: size against information

Run from hpc/joint/docs/tools (needs git and the repository's history):

    python e3_numbers.py

Pure ASCII, LF only.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import warnings

sys.dont_write_bytecode = True      # importing the stage1 modules must not leave caches

import numpy as np
from scipy import optimize, stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # repo root
_STAGE1 = os.path.join(_ROOT, "hpc", "joint", "stage1")

FREEZE = "834eb41"
METRICS = "hpc/dsn/metrics.py"
JDIAG = "hpc/joint/stage3/joint_diagnostics.py"
PROBES = "hpc/joint/stage3b/encoder_probes.py"
LBG = "hpc/dsn/latent_burst_generator.py"

E_RUN = 12            # embedding dimension of the running example (search anchor E = 12)
E_R2 = 10             # embedding dimension of the r2 encoder (SBI_PIPELINE.md S4)
PRINTED = 1.0005      # r_eff below this prints as 1.000 at three decimals


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", _ROOT] + list(args), check=False,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _load(path: str, names: tuple) -> tuple:
    """Execute the named top-level functions of `path` as read at the freeze."""
    out = _git("show", "%s:%s" % (FREEZE, path))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode())
    src = out.stdout.decode("utf-8")
    tree = ast.parse(src)
    ns = {"np": np}
    lines = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, path, "exec"), ns)      # noqa: S102 -- the repository's own code
            lines[node.name] = (node.lineno, node.end_lineno)
    missing = [n for n in names if n not in lines]
    if missing:
        raise RuntimeError("not found in %s: %r" % (path, missing))
    return ns, lines


def etf(n_classes: int, dim: int) -> np.ndarray:
    """C unit vectors in R^dim with pairwise inner products -1/(C - 1): the rows
    of the centred identity of R^C, normalised, placed in the first C axes."""
    m = np.eye(n_classes) - 1.0 / n_classes
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    u = np.zeros((n_classes, dim))
    u[:, :n_classes] = m
    return u


def tangent_dirs(u: np.ndarray, n: int, rng) -> np.ndarray:
    """n unit vectors orthogonal to the unit vector u, uniform on that sphere."""
    g = rng.standard_normal((n, u.shape[0]))
    g = g - np.outer(g @ u, u)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def cap(u: np.ndarray, n: int, rho_deg: float, rng, rim: bool = False) -> np.ndarray:
    """n points on the unit sphere at angle t from u, t uniform on [0, rho]
    (or t = rho for every point when rim=True), tangent directions uniform."""
    rho = math.radians(rho_deg)
    t = np.full(n, rho) if rim else rho * rng.uniform(size=n)
    d = tangent_dirs(u, n, rng)
    return np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * d


def r_eff_cov(cov: np.ndarray) -> float:
    """(tr S)^2 / tr(S^2) of a covariance S (the participation ratio)."""
    return float(np.trace(cov) ** 2 / np.sum(cov * cov))


# --------------------------------------------------------------------------- #
# B1  the simplex ETF
# --------------------------------------------------------------------------- #
def block_b1() -> None:
    print("== B1 the simplex ETF in R^%d" % E_RUN)
    for C in (2, 3, 4):
        u = etf(C, E_RUN)
        g = u @ u.T
        off = g[~np.eye(C, dtype=bool)]
        frame = np.linalg.eigvalsh(u.T @ u)
        nz = np.sort(frame[frame > 1e-12])
        print("C = %d: pairwise cosine %.6f (target -1/(C-1) = %.6f, max dev %.1e);"
              " angle %.2f deg; |sum of vertices| = %.1e; frame operator: %d non-zero"
              " eigenvalues, each %.6f (C/(C-1) = %.6f)"
              % (C, off.mean(), -1.0 / (C - 1), np.abs(off + 1.0 / (C - 1)).max(),
                 math.degrees(math.acos(-1.0 / (C - 1))), np.linalg.norm(u.sum(0)),
                 nz.size, nz.mean(), C / (C - 1.0)))
        assert np.allclose(off, -1.0 / (C - 1)) and np.allclose(nz, C / (C - 1.0))
        assert nz.size == C - 1


# --------------------------------------------------------------------------- #
# B2  r_eff of collapsed and nearly collapsed clouds
# --------------------------------------------------------------------------- #
def _r_eff_impls() -> tuple:
    nsd, ld = _load(METRICS, ("_to_numpy", "_as_2d", "effective_rank"))
    nsj, lj = _load(JDIAG, ("effective_rank", "cluster_scores"))
    nsp, lp = _load(PROBES, ("effective_rank",))
    impls = (("DSN metrics.effective_rank", nsd["effective_rank"],
              "%s:%d-%d" % ((METRICS,) + ld["effective_rank"])),
             ("joint joint_diagnostics.effective_rank", nsj["effective_rank"],
              "%s:%d-%d" % ((JDIAG,) + lj["effective_rank"])),
             ("Stage 3b encoder_probes.effective_rank", nsp["effective_rank"],
              "%s:%d-%d" % ((PROBES,) + lp["effective_rank"])))
    return impls, nsj["cluster_scores"], "%s:%d-%d" % ((JDIAG,) + lj["cluster_scores"])


def _rim_closed(rho: float, dim: int) -> float:
    """r_eff of two antipodal rim caps, equal masses: 1/(cos^4 + sin^4/(dim-1))."""
    return 1.0 / (math.cos(rho) ** 4 + math.sin(rho) ** 4 / (dim - 1))


def _arc_closed(beta: float) -> tuple:
    """Eigenvalues and r_eff of the uniform great-circle arc t in [-beta, beta]."""
    s2 = math.sin(2 * beta) / (2 * beta)
    lam_a = 0.5 * (1 + s2) - (math.sin(beta) / beta) ** 2     # along the arc's chord direction
    lam_b = 0.5 * (1 - s2)                                    # along the tangent at its midpoint
    return lam_a, lam_b, (lam_a + lam_b) ** 2 / (lam_a ** 2 + lam_b ** 2)


def block_b2(impls) -> None:
    print("\n== B2 r_eff of collapsed and nearly collapsed clouds")
    for name, _f, where in impls:
        print("   %-40s read from %s at %s" % (name, where, FREEZE))
    rng = np.random.default_rng(20261005)
    joint = impls[1][1]

    print("-- exact collapse onto the ETF, equal masses (100 rows per class, E = %d)" % E_RUN)
    for C in (2, 3, 4):
        z = np.repeat(etf(C, E_RUN), 100, axis=0)
        vals = [f(z) for _n, f, _w in impls]
        print("   C = %d: r_eff = %s (C - 1 = %d)" % (C, ", ".join("%.6f" % v for v in vals), C - 1))
        assert all(abs(v - (C - 1)) < 1e-9 for v in vals)

    print("-- exact collapse, unequal masses")
    for counts in ((72, 108, 108), (10, 90), (1, 1, 8)):
        C = len(counts)
        u = etf(C, E_RUN)
        z = np.concatenate([np.repeat(u[c:c + 1], k, axis=0) for c, k in enumerate(counts)])
        w = np.asarray(counts, float) / sum(counts)
        mean = w @ u
        cov = (u.T * w) @ u - np.outer(mean, mean)
        print("   counts %s: r_eff = %.6f (joint), closed form %.6f; C - 1 = %d"
              % (counts, joint(z), r_eff_cov(cov), C - 1))
        assert abs(joint(z) - r_eff_cov(cov)) < 1e-9

    print("-- two antipodal caps at C = 2, equal masses, every point at angle rho"
          " from its vertex (rim caps), E = %d" % E_R2)
    u = etf(2, E_R2)
    for rho_deg in (0.5, 1.0, 2.0, 5.0):
        z = np.concatenate([cap(u[0], 20000, rho_deg, rng, rim=True),
                            cap(u[1], 20000, rho_deg, rng, rim=True)])
        print("   rho = %4.1f deg: r_eff = %.5f (closed form 1/(cos^4 + sin^4/(E-1)) = %.5f);"
              " trace %.4f" % (rho_deg, joint(z), _rim_closed(math.radians(rho_deg), E_R2),
                               np.trace(np.cov(z, rowvar=False))))
    rho_star = optimize.brentq(lambda r: _rim_closed(r, E_R2) - PRINTED, 1e-6, 0.5)
    print("   r_eff < %.4f (prints as 1.000) needs rho < %.3f deg" % (PRINTED, math.degrees(rho_star)))

    print("-- one great-circle arc, uniform on t in [-beta, beta] (a continuum)")
    for beta_deg in (2.0, 3.5, 10.0, 90.0):
        b = math.radians(beta_deg)
        t = rng.uniform(-b, b, size=40000)
        z = np.zeros((t.size, E_R2))
        z[:, 0], z[:, 1] = np.cos(t), np.sin(t)
        la, lb, rc = _arc_closed(b)
        print("   beta = %4.1f deg: r_eff = %.5f (closed form %.5f); trace %.6f (closed %.6f)"
              % (beta_deg, joint(z), rc, np.trace(np.cov(z, rowvar=False)), la + lb))
    beta_star = optimize.brentq(lambda b: _arc_closed(b)[2] - PRINTED, 1e-4, 1.0)
    la, lb, _ = _arc_closed(beta_star)
    print("   r_eff < %.4f needs beta < %.3f deg (an arc %.2f deg long), trace %.5f;"
          " two antipodal points of equal mass: trace 1"
          % (PRINTED, math.degrees(beta_star), 2 * math.degrees(beta_star), la + lb))
    sim_val = 1.017                       # the r2 encoder's simulated arm (SBI_PIPELINE.md S4)
    rho_s = optimize.brentq(lambda r: _rim_closed(r, E_R2) - sim_val, 1e-6, 0.5)
    beta_s = optimize.brentq(lambda b: _arc_closed(b)[2] - sim_val, 1e-4, 1.5)
    print("   r_eff = %.3f is two antipodal rim caps of rho = %.2f deg, or one arc of beta ="
          " %.2f deg (%.1f deg long); off-axis share of the variance (r_eff - 1)/2 = %.4f"
          % (sim_val, math.degrees(rho_s), math.degrees(beta_s), 2 * math.degrees(beta_s),
             (sim_val - 1) / 2))

    print("-- the joint stack's smoke test R3d cloud: rng(0), (200 x 1) @ (1 x 8), a Gaussian line")
    g = np.random.default_rng(0)
    z = g.standard_normal((200, 1)) @ g.standard_normal((1, 8))
    print("   r_eff = %.6f; distinct rows %d; trace %.3f"
          % (joint(z), len(np.unique(np.round(z, 12), axis=0)), np.trace(np.cov(z, rowvar=False))))

    print("-- a constant cloud (total collapse: one point, 50 rows), finding F-bb")
    e1 = np.zeros((1, E_RUN))
    e1[0, 0] = 1.0
    for label, row in (("on a coordinate axis, so the covariance is exactly zero", e1),
                       ("on a generic unit vector, so rounding leaves a tiny covariance",
                        etf(2, E_RUN)[:1])):
        z = np.repeat(row, 50, axis=0)
        cov = np.cov(z, rowvar=False)
        print("   %s (largest |entry| %.1e):" % (label, np.abs(cov).max()))
        for name, f, _w in impls:
            print("      %-40s r_eff = %.1f" % (name, f(z)))
    print("-- C = 3 ETF collapse plus within-class caps (angles uniform on [0, rho]), E = %d" % E_RUN)
    u = etf(3, E_RUN)
    for rho_deg in (5.0, 15.0):
        z = np.concatenate([cap(u[c], 20000, rho_deg, rng) for c in range(3)])
        print("   rho = %4.1f deg: r_eff = %.4f (C - 1 = 2)" % (rho_deg, joint(z)))


# --------------------------------------------------------------------------- #
# B3  cluster scores and the trace
# --------------------------------------------------------------------------- #
def block_b3(impls, cluster_scores, where) -> None:
    print("\n== B3 what the logged cluster scores and the trace add (%s at %s)" % (where, FREEZE))
    joint = impls[1][1]
    rng = np.random.default_rng(20261006)
    u = etf(2, E_R2)
    n = 400
    lab = np.repeat([0, 1], n // 2)
    clouds = []
    clouds.append(("constant cloud, labels split in half", np.repeat(u[:1], n, axis=0), lab))
    clouds.append(("two-point collapse (antipodal)", np.repeat(u, n // 2, axis=0), lab))
    for beta_deg in (3.5, 90.0):
        b = math.radians(beta_deg)
        t = np.sort(rng.uniform(-b, b, size=n))
        z = np.zeros((n, E_R2))
        z[:, 0], z[:, 1] = np.cos(t), np.sin(t)
        clouds.append(("arc, beta = %.1f deg, labels = its two halves" % beta_deg, z,
                       (t > 0).astype(int)))
    for name, z, y in clouds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = cluster_scores(z, y, seed=0)
        print("   %-46s r_eff %.4f  trace %.5f  ARI %.3f  silhouette %.3f"
              % (name, joint(z), np.trace(np.cov(z, rowvar=False)), sc["ari"], sc["silhouette"]))


# --------------------------------------------------------------------------- #
# B4  the composite loss, transcribed from P2 eqs. (P2.3)-(P2.10)
# --------------------------------------------------------------------------- #
def _sqdist(z: np.ndarray) -> np.ndarray:
    return np.clip(2.0 - 2.0 * (z @ z.T), 0.0, 4.0)      # Q on the unit sphere, eq. (P2.2)


def _mine(q: np.ndarray, y: np.ndarray, strategy: str) -> np.ndarray:
    """Eq. (P2.3): the triplets (i, i', i'') the miner returns."""
    n = len(y)
    out = []
    for i in range(n):
        pos = np.where((y == y[i]) & (np.arange(n) != i))[0]
        neg = np.where(y != y[i])[0]
        if pos.size == 0 or neg.size == 0:
            continue
        if strategy == "hard":
            qp, qn = q[i, pos][:, None], q[i, neg][None, :]
            pi, ni = np.nonzero(qn <= qp)
            out.extend((i, pos[a], neg[b]) for a, b in zip(pi, ni))
            continue
        p = pos[np.argmin(q[i, pos])]                       # the easy (closest) positive
        if strategy == "easy_positive":
            cand = neg
        else:                                               # easy_pos_semihard_neg
            cand = neg[q[i, neg] > q[i, p]]                 # strictly farther than that positive
            if cand.size == 0:
                continue
        out.append((i, p, cand[np.argmin(q[i, cand])]))
    return np.asarray(out, dtype=int).reshape(-1, 3)


def _l_joint(z, y, strategy, strict, m_cos, alpha_deg) -> tuple:
    """Eqs. (P2.4)-(P2.6): the margin-plus-angular part of one batch."""
    q = _sqdist(z)
    t = _mine(q, y, strategy)
    if t.size == 0:
        return 0.0, 0, 0
    i, p, k = t.T
    m_sq = 2.0 * m_cos
    qap, qan, qpn = q[i, p], q[i, k], q[p, k]
    l_trip = np.maximum(0.0, qap - np.minimum(qan, qpn) + m_sq)            # swap = True
    mid = 0.5 * (z[i] + z[p])
    qmid = np.sum((z[k] - mid) ** 2, axis=1)
    l_ang = np.maximum(0.0, qap - 4.0 * math.tan(math.radians(alpha_deg)) ** 2 * qmid)
    if strict:
        keep = (qap < qan) & (qan < qap + m_sq) & (qap < qpn) & (qpn < qap + m_sq)
    else:
        keep = np.ones(len(t), dtype=bool)
    tot = (l_trip + l_ang)[keep]
    n_act = int(np.sum(tot > 0))
    return float(tot.sum() / max(1, n_act)), len(t), int(keep.sum())


def _l_triplet(z, y, strategy, m_cos) -> float:
    """Eq. (P2.7): the library's triplet-margin loss in cosine units, swap on,
    mean over the positive terms (`loss_type = triplet`; no filter)."""
    q = _sqdist(z)
    t = _mine(q, y, strategy)
    if t.size == 0:
        return 0.0
    i, p, k = t.T
    d = q / 2.0                                              # d_cos, eq. (P2.2)
    l = np.maximum(0.0, d[i, p] - np.minimum(d[i, k], d[p, k]) + m_cos)
    pos = l[l > 0]
    return float(pos.mean()) if pos.size else 0.0


def _l_sep(z, y, n_classes, min_per_class=2) -> tuple:
    """Eq. (P2.8): the centroid-separation penalty of one batch (raw means)."""
    v = []
    for c in range(n_classes):
        idx = y == c
        if idx.sum() >= min_per_class:
            m = z[idx].mean(axis=0)
            v.append(m / np.linalg.norm(m))
    k = len(v)
    if k < 2:
        return 0.0, k
    v = np.asarray(v)
    g = v @ v.T
    off = ~np.eye(k, dtype=bool)
    return float(np.sum((g[off] + 1.0 / (k - 1)) ** 2) / (k * (k - 1))), k


CONFIGS = (
    # name, strategy, strict, m_cos, alpha_deg, lambda_sep, rows per class
    ("joint defaults (easy_pos_semihard_neg, strict, joint_sep, m 0.2, alpha 18, lambda_sep 0.1)",
     "easy_pos_semihard_neg", True, 0.2, 18.0, 0.1, None),
    ("r2's constants (hard, filter off, joint_sep, m 0.3, alpha 10.69, lambda_sep 16.87, 9 per class)",
     "hard", False, 0.3, 10.6872999998721, 16.866728721243334, 9),
)


def block_b4() -> None:
    print("\n== B4 the composite loss on caps around the ETF (numpy transcription of P2"
          " eqs. (P2.3)-(P2.10); E = %d)" % E_RUN)
    b_met = 32
    n_batch = 300
    for name, strat, strict, m_cos, alpha, lam, per in CONFIGS:
        print("-- " + name)
        for C in (2, 3):
            n_c = per if per is not None else b_met // C          # eq. (P2.1)
            u = etf(C, E_RUN)
            y = np.repeat(np.arange(C), n_c)
            for rho_deg in (0.0, 5.0, 15.0, 30.0):
                rng = np.random.default_rng(1000 * C + int(rho_deg))
                lj, ls, lt, nz, mined, kept = [], [], [], 0, 0, 0
                for _ in range(n_batch):
                    z = np.concatenate([cap(u[c], n_c, rho_deg, rng) for c in range(C)])
                    a, nm, nk = _l_joint(z, y, strat, strict, m_cos, alpha)
                    s, _k = _l_sep(z, y, C)
                    lj.append(a)
                    ls.append(s)
                    lt.append(_l_triplet(z, y, strat, m_cos))
                    nz += a > 0
                    mined += nm
                    kept += nk
                lj, ls, lt = np.asarray(lj), np.asarray(ls), np.asarray(lt)
                print("   C = %d, %2d rows/class, rho = %4.1f deg: mean L_joint %.4g (batches > 0:"
                      " %3d/%d; mean mined %.1f, kept %.1f); mean L_sep %.3g; composite %.4g;"
                      " triplet loss (P2.7) mean %.4g, batches > 0: %d"
                      % (C, n_c, rho_deg, lj.mean(), nz, n_batch, mined / n_batch,
                         kept / n_batch, ls.mean(), lj.mean() + lam * ls.mean(),
                         lt.mean(), int(np.sum(lt > 0))))
                if rho_deg == 0.0:
                    assert lj.max() == 0.0 and ls.max() < 1e-20

    # the separation term's pull on spread: its batch mean against rho, C = 2 and 3
    print("-- E[L_sep] against rho (joint batcher, B_met = 32, 4000 batches): the order in rho")
    for C in (2, 3):
        n_c = b_met // C
        u = etf(C, E_RUN)
        y = np.repeat(np.arange(C), n_c)
        vals = []
        for rho_deg in (2.0, 4.0, 8.0):
            rng = np.random.default_rng(77 + C)
            s = [_l_sep(np.concatenate([cap(u[c], n_c, rho_deg, rng) for c in range(C)]), y, C)[0]
                 for _ in range(4000)]
            vals.append(np.mean(s))
        slope = np.polyfit(np.log([2.0, 4.0, 8.0]), np.log(vals), 1)[0]
        print("   C = %d: E[L_sep] at rho 2, 4, 8 deg = %s; log-log slope %.2f"
              % (C, ", ".join("%.3g" % v for v in vals), slope))

    # pre-training batches (A0/A0s): B_met rows drawn uniformly from the whole source
    print("-- encoder-only pre-training: B_met = 32 rows drawn uniformly, equal class masses")
    for C in (2, 3):
        p_less = 0.0
        if C == 2:
            pmf = stats.binom.pmf(np.arange(33), 32, 0.5)
            p_less = float(pmf[[0, 1, 31, 32]].sum())
        else:
            for n0 in range(33):
                for n1 in range(33 - n0):
                    n2 = 32 - n0 - n1
                    if min(n0, n1, n2) < 2:
                        p_less += float(stats.multinomial.pmf([n0, n1, n2], 32, [1 / 3.0] * 3))
        print("   C = %d: Pr(some class has fewer than 2 rows) = %.3g" % (C, p_less))
        if C == 3:
            u = etf(3, E_RUN)
            z = np.repeat(u[:2], 5, axis=0)
            s2, k2 = _l_sep(z, np.repeat([0, 1], 5), 3)
            print("   at the exact C = 3 ETF collapse a batch with K = %d gives L_sep = %.3f"
                  " (target -1, cosine -0.5); expected excess %.2g" % (k2, s2, s2 * p_less))


# --------------------------------------------------------------------------- #
# B5  the label code on the bench
# --------------------------------------------------------------------------- #
def _dsn_centres(n_classes: int, n_axes: int) -> np.ndarray:
    ns, _l = _load(LBG, ("_class_center_vectors",))
    return ns["_class_center_vectors"](n_classes, n_axes, "simplex")


def _hb(e: float) -> float:
    return 0.0 if e <= 0.0 or e >= 1.0 else -e * math.log(e) - (1 - e) * math.log(1 - e)


def block_b5() -> None:
    print("\n== B5 the label code on the bench (prior of plan eq. (7), provider `bench`)")
    if _git("diff", "--quiet", FREEZE, "--", "hpc/joint/stage1").returncode != 0:
        raise SystemExit("hpc/joint/stage1 differs from %s; refusing to import it" % FREEZE)
    print("hpc/joint/stage1 is identical to %s (git diff --quiet): importing it" % FREEZE)
    sys.path.insert(0, _STAGE1)
    import bench_burst_provider as bbp            # noqa: E402
    import latent_sbi_simulator as lss            # noqa: E402

    k_lab = len(bbp.BENCH_LABEL_IDX)
    n_draw = 2_000_000
    for C in (3, 2):
        sets = (("bench simplex_centres", lss.simplex_centres(C, k_lab)),
                ("DSN _class_center_vectors", _dsn_centres(C, k_lab)))
        for cname, centres in sets:
            dist = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
            print("   C = %d, %s: pairwise centre distances %s (in units of 0.10: %s)"
                  % (C, cname, np.round(dist[np.triu_indices(C, 1)], 4).tolist(),
                     np.round(dist[np.triu_indices(C, 1)] / 0.10, 2).tolist()))
            for tau in (0.10, 0.20, 0.30):
                try:
                    spec = lss.LatentSBISpec(n_latent=len(bbp.BENCH_AXES),
                                             label_idx=bbp.BENCH_LABEL_IDX,
                                             class_centres=centres, tau_ov=tau)
                except ValueError as exc:
                    print("   C = %d, %s: not a valid spec (%s)" % (C, cname, exc))
                    break
                rng = np.random.default_rng(int(1000 * tau) + 17 * C)
                cls, phi = lss.sample_prior(spec, n_draw, rng)
                post = lss.class_posterior(spec, phi)
                lp = np.log(post[np.arange(n_draw), cls])
                h_c = float(-lp.mean())
                h_c = 0.0 if abs(h_c) < 5e-9 else h_c      # -0.0000 from rounding is printed as 0
                se = float(lp.std(ddof=1) / math.sqrt(n_draw))
                err = float(np.mean(np.argmax(post, axis=1) != cls))
                fano = _hb(err) + err * math.log(C - 1) if C > 2 else _hb(err)
                print("   C = %d, %-26s tau_ov %.2f: H(c|theta) = %.4f +- %.4f; I(theta; c) ="
                      " ln C - H = %.4f (ln C = %.4f); Bayes error %.4f; Fano bound on the"
                      " free axes h_b + e ln(C-1) = %.4f"
                      % (C, cname, tau, h_c, se, math.log(C) - h_c, math.log(C), err, fano))


# --------------------------------------------------------------------------- #
# B6  the residual channel
# --------------------------------------------------------------------------- #
def block_b6() -> None:
    print("\n== B6 the residual channel: a scalar Gaussian signal through additive Gaussian noise")
    for ratio in (1.0, 0.1, 0.01):
        print("   amplitude / noise = %5.2f: I = 0.5 ln(1 + ratio^2) = %.3g nats"
              % (ratio, 0.5 * math.log1p(ratio ** 2)))
    print("   without noise, any injective rescaling of the residual leaves I unchanged"
          " (invariance of mutual information)")
    print("   ln 2 = %.4f, ln 3 = %.4f" % (math.log(2), math.log(3)))


def main() -> int:
    block_b1()
    impls, cluster_scores, where = _r_eff_impls()
    block_b2(impls)
    block_b3(impls, cluster_scores, where)
    block_b4()
    block_b5()
    block_b6()
    return 0


if __name__ == "__main__":
    sys.exit(main())
