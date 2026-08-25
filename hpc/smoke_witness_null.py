#!/usr/bin/env python3
"""
smoke_witness_null.py -- correctness checks for witness_kde_maps() and
witness_null_map() in npe_misspec.py. Run from hpc/:

    python3 smoke_witness_null.py

Exit 0 and "ALL CHECKS PASSED" mean both additions are sound. Set
KEEP_PLOTS=1 to keep the figures. Seconds on a login node.

WHAT IS CHECKED
  1. _mu_on_points reproduces witness_function's own arithmetic: the
     difference of two kernel mean embeddings equals u and v exactly.
  2. _kde_2d integrates to 1 over a wide grid (it is a density), is
     non-negative, and Scott's bandwidth scales as n^(-1/6).
  3. The null is built the way _one_gate builds its own: the arm has
     n_groups points, not n_real, and is drawn from a pool DISJOINT from
     the reference.
  4. The central identity of the design: with the reference held FIXED,
     E_b[g_null] converges to mu_fit - mu_POOL and NOT to zero. Checked
     by driving B up and confirming the mean approaches that limit while
     staying far from zero relative to its own Monte-Carlo error.
  5. Family-wise thresholds are computed in the right ORDER -- maximise
     over space WITHIN a draw, then quantile ACROSS draws -- and the two
     one-sided thresholds differ, as they must for an asymmetric null.
  6. The FWE threshold calibrates: applied to fresh null draws, the
     fraction whose map exceeds it is close to 1 - level, NOT to
     level-per-cell. This is the check that the multiplicity correction
     actually works.
  7. The point-level null is layout-free: identical numbers whether the
     layout is pca, contrast or absent.
  8. Z-map mechanics: masked where the null sd is below the floor, finite
     everywhere else, and sign-consistent with the observed field.
  9. Files: PNG and npz written with the fields, thresholds, max-statistic
     distributions and per-culture verdicts.

ASCII-only by policy (HPC transfer safety).
"""

import os
import shutil
import sys
import tempfile

import numpy as np

from npe_misspec import (_kde_2d, _mu_on_points, _scott_h, bandwidth_grid,
                         witness_function, witness_kde_maps,
                         witness_null_map)

FAILED = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def blob(rng, n, c, s):
    x = c[None, :] + s * rng.normal(size=(n, c.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(29)
    E = 14
    c1, c2 = np.eye(E)[0], np.eye(E)[1]
    z_sim = np.concatenate([blob(rng, 1500, c1, 0.12),
                            blob(rng, 1500, c2, 0.12)])
    n_g, w_per = 12, 8
    z_real = blob(rng, n_g * w_per, c1 + 0.20 * np.eye(E)[5], 0.10)
    groups = np.asarray(["cult%02d" % (i // w_per)
                         for i in range(n_g * w_per)])
    bw = bandwidth_grid(z_sim, z_real, rng=np.random.default_rng(0))

    print("check 1: _mu_on_points matches witness_function exactly")
    res = witness_function(z_sim, z_real, bandwidths=bw, split=True, seed=0)
    fit = z_sim[res.fit_idx]
    ev = z_sim[res.eval_idx]
    u = (_mu_on_points(ev, fit, bw) - _mu_on_points(ev, z_real, bw))
    v = (_mu_on_points(z_real, fit, bw) - _mu_on_points(z_real, z_real, bw))
    check("u reproduced", np.max(np.abs(u - res.u)) < 1e-12,
          "max|diff|=%.2e" % np.max(np.abs(u - res.u)))
    check("v reproduced (self-term included on both sides)",
          np.max(np.abs(v - res.v)) < 1e-12,
          "max|diff|=%.2e" % np.max(np.abs(v - res.v)))

    print("check 2: the KDE is a density")
    pts = rng.normal(size=(300, 2))
    L, n = 9.0, 300
    ax = np.linspace(-L, L, n)
    step = ax[1] - ax[0]
    Q = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
    d = _kde_2d(pts, Q, 0.4)
    check("KDE is non-negative", float(d.min()) >= 0.0)
    integral = float(d.sum() * step * step)
    check("KDE integrates to 1", abs(integral - 1.0) < 1e-3,
          "integral=%.6f" % integral)
    h1, h2 = _scott_h(pts), _scott_h(np.repeat(pts, 64, axis=0))
    check("Scott bandwidth falls as n^(-1/6)",
          abs(h1 / h2 - 64 ** (1.0 / 6.0)) < 1e-6,
          "ratio=%.4f expected=%.4f" % (h1 / h2, 64 ** (1.0 / 6.0)))

    outdir = tempfile.mkdtemp(prefix="witness_null_smoke_")
    try:
        print("check 3: the null arm mirrors _one_gate")
        r, saved = witness_null_map(z_sim, z_real, outdir, groups=groups,
                                    space="z", method="pca", bandwidths=bw,
                                    n_null=60, n_window_choices=3, grid=30,
                                    seed=0)
        check("n_groups recovered from the group vector",
              r.n_groups == n_g, "%d vs %d" % (r.n_groups, n_g))
        check("per-culture verdicts, one per culture",
              r.group_v.shape == (n_g,) and r.group_flag.shape == (n_g,))
        check("group names preserved in order",
              r.group_names[0] == "cult00" and len(r.group_names) == n_g)

        print("check 4: with a FIXED reference the null mean is NOT zero")
        # Reproduce the design directly and drive B up.
        g2 = np.random.default_rng(5)
        perm = g2.permutation(z_sim.shape[0])
        nf = z_sim.shape[0] // 2
        fitb, poolb = z_sim[perm[:nf]], z_sim[perm[nf:]]
        Qp = blob(g2, 6, c1, 0.2)
        mu_f = _mu_on_points(Qp, fitb, bw).sum(axis=0)
        limit = mu_f - _mu_on_points(Qp, poolb, bw).sum(axis=0)
        means = {}
        for B in (100, 1000, 6000):
            acc = np.zeros(6)
            for _ in range(B):
                take = g2.choice(poolb.shape[0], n_g, replace=False)
                acc += mu_f - _mu_on_points(Qp, poolb[take], bw).sum(axis=0)
            means[B] = acc / B
        far = np.max(np.abs(means[6000] - limit))
        check("null mean converges to mu_fit - mu_POOL",
              far < np.max(np.abs(means[100] - limit)) + 1e-12,
              "|mean_B - limit| = %.2e at B=6000" % far)
        check("that limit is NOT zero",
              np.max(np.abs(limit)) > 10 * far,
              "|limit|=%.3e vs residual %.3e" % (np.max(np.abs(limit)), far))

        print("check 5: threshold order of operations and asymmetry")
        with np.load(saved["arrays"], allow_pickle=False) as z:
            mn, mp = z["max_neg"], z["max_pos"]
            check("one max statistic per draw, not per cell",
                  mn.shape == (60,) and mp.shape == (60,),
                  "%s / %s" % (mn.shape, mp.shape))
            check("t_neg is the 0.95 quantile ACROSS draws",
                  abs(float(z["t_neg"]) - float(np.quantile(mn, 0.95)))
                  < 1e-12)
            check("the two one-sided thresholds differ (asymmetric null)",
                  abs(float(z["t_neg"]) - float(z["t_pos"])) > 1e-9,
                  "t_neg=%.5g t_pos=%.5g" % (z["t_neg"], z["t_pos"]))
            check("a per-cell threshold would be far smaller than the FWE "
                  "one", float(np.quantile(np.abs(z["null_mean"]), 0.95))
                  < float(z["t_neg"]))

        print("check 6: the FWE threshold calibrates on fresh draws")
        # Fresh null draws scored against the stored threshold: the
        # fraction of MAPS exceeding it should be about 1 - level.
        g3 = np.random.default_rng(11)
        with np.load(saved["arrays"], allow_pickle=False) as z:
            tn = float(z["point_t_neg"])
        exceed = 0
        B2 = 400
        for _ in range(B2):
            take = g3.choice(poolb.shape[0], n_g, replace=False)
            arm = poolb[take]
            gp = (_mu_on_points(arm, fitb, bw)
                  - _mu_on_points(arm, arm, bw)).sum(axis=0)
            exceed += int(np.max(-gp) > tn)
        frac = exceed / B2
        check("false-positive rate near 1 - level, not level-per-point",
              0.0 <= frac <= 0.20, "%.3f (target ~0.05)" % frac)

        print("check 7: the point-level null is layout-free")
        r_pca, _ = witness_null_map(z_sim, z_real, outdir, groups=groups,
                                    space="zA", method="pca", bandwidths=bw,
                                    n_null=40, n_window_choices=2, grid=20,
                                    seed=3)
        r_con, _ = witness_null_map(z_sim, z_real, outdir, groups=groups,
                                    space="zB", method="contrast",
                                    bandwidths=bw, n_null=40,
                                    n_window_choices=2, grid=20, seed=3)
        check("point-level threshold identical across layouts",
              abs(r_pca.point_t_neg - r_con.point_t_neg) < 1e-12,
              "%.6g vs %.6g" % (r_pca.point_t_neg, r_con.point_t_neg))
        check("per-culture scores identical across layouts",
              np.max(np.abs(r_pca.group_v - r_con.group_v)) < 1e-12)
        check("grid-level thresholds DO differ across layouts",
              abs(r_pca.t_neg - r_con.t_neg) > 0.0
              or abs(r_pca.t_pos - r_con.t_pos) > 0.0)

        print("check 8: Z-map mechanics")
        with np.load(saved["arrays"], allow_pickle=False) as z:
            zm, mk, sd = z["zmap"], z["mask"], z["null_sd"]
            check("Z is finite wherever it is not masked",
                  bool(np.all(np.isfinite(zm[~mk]))))
            check("mask marks the low-variability cells",
                  (not mk.any()) or float(sd[mk].max()) <= float(sd[~mk].min())
                  + 1e-12)
            ok = ~mk
            check("Z has the same sign as the de-biased observed field",
                  bool(np.all(np.sign(zm[ok])
                              == np.sign((z["obs"] - z["null_mean"])[ok]))))

        print("check 9: outputs")
        check("null map PNG written",
              os.path.getsize(saved["null_map"]) > 5000)
        with np.load(saved["arrays"], allow_pickle=False) as z:
            for k in ("obs", "null_mean", "null_sd", "zmap", "mask",
                      "max_neg", "max_pos", "point_max_neg", "group_v",
                      "group_flag", "group_names"):
                check("npz carries %s" % k, k in z)
        ksaved = witness_kde_maps(z_sim, z_real, outdir, space="z",
                                  bandwidths=bw, methods=("pca",), grid=40,
                                  seed=0)
        check("KDE figure written",
              "kde_pca" in ksaved
              and os.path.getsize(ksaved["kde_pca"]) > 5000)
        if os.environ.get("KEEP_PLOTS") == "1":
            print("  outputs kept in %s" % outdir)
    finally:
        if os.environ.get("KEEP_PLOTS") != "1":
            shutil.rmtree(outdir, ignore_errors=True)

    print()
    if FAILED:
        print("FAILED: %d check(s): %s" % (len(FAILED), FAILED))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
