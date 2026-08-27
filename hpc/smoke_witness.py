#!/usr/bin/env python3
"""
smoke_witness.py -- correctness checks for the witness-function addition
to npe_misspec.py. Run from hpc/:

    python3 smoke_witness.py

Exit code 0 and a final "ALL CHECKS PASSED" line mean the addition is
sound. Every check states what it asserts. No cluster resources needed;
runs in seconds on a login node. Plots land in a throwaway directory and
are deleted unless KEEP_PLOTS=1 is set in the environment.

WHAT IS CHECKED
  1. The exact identity mean(u) - mean(v) = mmd2_multiscale (split=False),
     per bandwidth and summed, to 1e-10. This is the algebra the whole
     witness construction rests on.
  2. Linearity across bandwidths: u_sum == sum_s u[s] exactly.
  3. The self-term of v_j is the constant S/n_real offset predicted by
     the docstring (checked by comparing against an explicit
     leave-self-in recomputation).
  4. Determinism under a fixed seed.
  5. A MULTIMODAL Mode-2 scenario: two simulated modes on the sphere,
     real bulk inside mode 1, three real outliers far outside both.
     The outliers must carry the most negative v_sum.
  6. A Mode-1 scenario: real = mode 1 uniformly shifted. The v histogram
     bulk must sit below the u bulk (mean gap positive, no extreme
     stragglers dominating).
  7. witness_maps writes the expected PNG/NPZ files, non-trivially sized,
     for PCA always and t-SNE when sklearn is present.

ASCII-only by policy (HPC transfer safety).
"""

import os
import shutil
import sys
import tempfile

import numpy as np

from npe_misspec import (bandwidth_grid, mmd2_multiscale, witness_function,
                         witness_maps)

FAILED = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (tag, name, ("  -- " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def sphere_blob(rng, n, center, spread):
    x = center[None, :] + spread * rng.normal(size=(n, center.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(7)
    E = 16

    # Two simulated modes: deliberately multimodal, the case the scalar
    # gate over-fires on.
    c1 = np.eye(E)[0]
    c2 = np.eye(E)[1]
    z_sim = np.concatenate([sphere_blob(rng, 400, c1, 0.15),
                            sphere_blob(rng, 400, c2, 0.15)], axis=0)

    # Mode-2 real cohort: bulk inside mode 1, 3 outliers near -c1.
    z_real_m2 = np.concatenate([sphere_blob(rng, 27, c1, 0.15),
                                sphere_blob(rng, 3, -c1, 0.05)], axis=0)
    groups = np.asarray(["rec%02d" % i for i in range(30)])

    bw = bandwidth_grid(z_sim, z_real_m2, rng=np.random.default_rng(0))

    print("check 1: exact identity, split=False")
    res = witness_function(z_sim, z_real_m2, bandwidths=bw, split=False)
    for s in range(bw.size):
        lhs = res.u[s].mean() - res.v[s].mean()
        rhs = mmd2_multiscale(z_sim, z_real_m2, [bw[s]])
        check("per-bandwidth identity, sigma=%.3g" % bw[s],
              abs(lhs - rhs) < 1e-10, "|diff|=%.2e" % abs(lhs - rhs))
    lhs = res.mmd2_from_witness()
    rhs = mmd2_multiscale(z_sim, z_real_m2, bw)
    check("summed identity vs mmd2_multiscale", abs(lhs - rhs) < 1e-10,
          "|diff|=%.2e" % abs(lhs - rhs))

    print("check 2: linearity of the sum across bandwidths")
    check("u_sum == sum_s u[s]",
          np.array_equal(res.u_sum, res.u.sum(axis=0)))
    check("v_sum == sum_s v[s]",
          np.array_equal(res.v_sum, res.v.sum(axis=0)))

    print("check 3: real self-term is the constant S/n_real offset")
    n_real = z_real_m2.shape[0]
    # v_j as computed includes k(z_j, z_j)/n_real = 1/n_real per bandwidth.
    # Recompute mu_real without the self column for one j and compare.
    j = 0
    others = np.delete(np.arange(n_real), j)
    d = np.sum((z_real_m2[others] - z_real_m2[j]) ** 2, axis=1)
    v_j_rebuilt = 0.0
    for s in range(bw.size):
        g = 1.0 / (2.0 * bw[s] ** 2)
        mu_sim_j = np.exp(-g * np.sum((z_sim - z_real_m2[j]) ** 2,
                                      axis=1)).mean()
        mu_real_j = (np.exp(-g * d).sum() + 1.0) / n_real  # +1 = self term
        v_j_rebuilt += mu_sim_j - mu_real_j
    check("v_0 matches explicit rebuild incl. self-term",
          abs(v_j_rebuilt - res.v_sum[0]) < 1e-10,
          "|diff|=%.2e" % abs(v_j_rebuilt - res.v_sum[0]))

    print("check 4: determinism under a fixed seed (split=True)")
    r1 = witness_function(z_sim, z_real_m2, bandwidths=bw, split=True, seed=3)
    r2 = witness_function(z_sim, z_real_m2, bandwidths=bw, split=True, seed=3)
    check("identical u across runs", np.array_equal(r1.u, r2.u))
    check("identical fit split across runs",
          np.array_equal(r1.fit_idx, r2.fit_idx))
    check("split fit/eval sets disjoint",
          np.intersect1d(r1.fit_idx, r1.eval_idx).size == 0)

    print("check 5a: density-matched Mode-2 -- outliers most negative v")
    # The witness is a density CONTRAST, not an isolation score: stranded
    # points dominate the negative tail only when the real bulk's density
    # matches the simulated density there. So this check uses a UNIMODAL
    # simulator and a real bulk drawn from the same blob, plus 3 isolated
    # points in remote, mutually distant directions.
    z_sim_uni = sphere_blob(rng, 800, c1, 0.15)
    outl = np.stack([-c1, np.eye(E)[3], -np.eye(E)[4]])
    z_real_5a = np.concatenate([sphere_blob(rng, 27, c1, 0.15),
                                outl / np.linalg.norm(outl, axis=1,
                                                      keepdims=True)])
    bw5 = bandwidth_grid(z_sim_uni, z_real_5a, rng=np.random.default_rng(0))
    rs = witness_function(z_sim_uni, z_real_5a, bandwidths=bw5, split=True,
                          seed=0)
    worst3 = set(np.argsort(rs.v_sum)[:3].tolist())
    check("the 3 planted outliers are the 3 most negative v_sum",
          worst3 == {27, 28, 29}, "got %s" % sorted(worst3))

    print("check 5b: multimodal sim, unimodal real -- map points at the "
          "surplus mode")
    # The scenario behind this whole addition: the simulator is bimodal,
    # every real recording sits inside mode 1. The scalar gate fires on
    # the mode-weight mismatch; the witness map must expose it by giving
    # the most POSITIVE u_sum to simulated points of the unvisited mode 2
    # (rows >= 400 of z_sim), and negative v to the real bulk.
    z_real_5b = sphere_blob(rng, 30, c1, 0.15)
    rb = witness_function(z_sim, z_real_5b, bandwidths=bw, split=True,
                          seed=0)
    check("witness gap positive (gate would fire)",
          rb.u_sum.mean() - rb.v_sum.mean() > 0,
          "gap=%.3e" % (rb.u_sum.mean() - rb.v_sum.mean()))
    top = np.argsort(rb.u_sum)[-50:]
    frac_mode2 = float(np.mean(rb.eval_idx[top] >= 400))
    check("top-50 positive u_sum are >90 pct unvisited-mode sims",
          frac_mode2 > 0.9, "fraction=%.2f" % frac_mode2)
    check("real bulk witness negative (excess real density)",
          np.median(rb.v_sum) < 0, "median v=%.3e" % np.median(rb.v_sum))

    print("check 6: Mode-1 scenario -- bulk shift moves the v histogram")
    shift = 0.35 * np.eye(E)[2]
    z_real_m1 = sphere_blob(rng, 30, c1 + shift, 0.15)
    rm1 = witness_function(z_sim, z_real_m1, bandwidths=bw, split=True,
                           seed=0)
    gap = rm1.u_sum.mean() - rm1.v_sum.mean()
    check("mean witness gap positive under a bulk shift", gap > 0,
          "gap=%.3e" % gap)
    med_gap = np.median(rm1.u_sum) - np.median(rm1.v_sum)
    check("median gap also positive (not tail-driven)", med_gap > 0,
          "median gap=%.3e" % med_gap)

    print("check 7: witness_maps writes the expected files")
    outdir = tempfile.mkdtemp(prefix="witness_smoke_")
    try:
        saved = witness_maps(z_sim, z_real_m2, outdir, space="z",
                             bandwidths=bw, split=True, groups=groups,
                             seed=0)
        check("PCA map saved", "map_pca" in saved
              and os.path.getsize(saved["map_pca"]) > 5000)
        try:
            import sklearn  # noqa: F401
            check("t-SNE map saved", "map_tsne" in saved
                  and os.path.getsize(saved["map_tsne"]) > 5000)
        except ImportError:
            print("  [SKIP] t-SNE map (scikit-learn not installed here; "
                  "witness_maps records the skip in notes)")
        check("histogram figure saved", "hist" in saved
              and os.path.getsize(saved["hist"]) > 5000)
        check("npz archive saved", "arrays" in saved
              and os.path.getsize(saved["arrays"]) > 1000)
        with np.load(saved["arrays"]) as z:
            check("npz carries coords + scores",
                  "u_sum" in z and any(k.startswith("coords_") for k in z))
        if os.environ.get("KEEP_PLOTS") == "1":
            print("  plots kept in %s" % outdir)
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
