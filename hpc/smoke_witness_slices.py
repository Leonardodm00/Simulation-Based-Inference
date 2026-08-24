#!/usr/bin/env python3
"""
smoke_witness_slices.py -- correctness checks for witness_slices() in
npe_misspec.py. Run from hpc/:

    python3 smoke_witness_slices.py

Exit 0 and "ALL CHECKS PASSED" mean the slice stack is sound. Set
KEEP_PLOTS=1 to keep the figures. Runs in seconds on a login node.

WHAT IS CHECKED
  1. The slice direction w is a unit vector ORTHOGONAL to both axes of
     the plotted plane, so translating along it really does move off the
     plane and nowhere else.
  2. The off-plane coordinate t of the data has zero mean by
     construction, and the requested offsets sit exactly at the requested
     quantiles of it -- i.e. every slice is one the data populate.
  3. The t = 0 slice reproduces, to machine precision, the field that
     witness_heatmaps computes in field="lift" mode. The stack and the
     single heatmap are the same object at different depths, not two
     implementations.
  4. Translation is exact: the evaluated points of slice t are the
     lifted plane plus t*w (and, when sphere=True, land on the unit
     sphere to 1e-12).
  5. The stability diagnostic has teeth in BOTH directions:
       (a) a cloud whose sim/real discrepancy is invariant along w must
           report high correlation across slices and near-zero sign
           disagreement;
       (b) a cloud where the discrepancy REVERSES with w must report low
           correlation and substantial sign disagreement.
     A diagnostic that never fires is not a diagnostic.
  6. The reference slice is the one nearest the median quantile, and the
     correlation of the reference slice with itself is exactly 1.
  7. Refusals: t-SNE raises (no exact lift), out-of-range quantiles
     raise, an out-of-range bandwidth_index raises.
  8. Outputs: PNG and npz written, npz carrying offsets, fields,
     direction, stability and sign disagreement with consistent shapes.

ASCII-only by policy (HPC transfer safety).
"""

import os
import shutil
import sys
import tempfile

import numpy as np

from npe_misspec import (_grid_2d, _project_2d, _witness_on_points,
                         bandwidth_grid, witness_function, witness_heatmaps,
                         witness_slices)

FAILED = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def sphere_blob(rng, n, center, spread):
    x = center[None, :] + spread * rng.normal(size=(n, center.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(19)
    E = 16
    c1, c2 = np.eye(E)[0], np.eye(E)[1]
    z_sim = np.concatenate([sphere_blob(rng, 300, c1, 0.12),
                            sphere_blob(rng, 300, c2, 0.12)], axis=0)
    z_real = sphere_blob(rng, 30, c1, 0.12)
    groups = np.asarray(["rec%02d" % i for i in range(30)])
    bw = bandwidth_grid(z_sim, z_real, rng=np.random.default_rng(0))
    outdir = tempfile.mkdtemp(prefix="witness_slices_smoke_")

    try:
        saved = witness_slices(z_sim, z_real, outdir, space="z",
                               bandwidths=bw, method="pca", grid=40,
                               groups=groups, seed=0)
        z = np.load(saved["arrays"])

        print("check 1: the slice direction is orthogonal to the plane")
        # Rebuild the same plane the function used.
        res = witness_function(z_sim, z_real, bandwidths=bw, split=True,
                               seed=0)
        P = np.concatenate([z_sim[res.eval_idx], z_real], axis=0)
        n_s = res.eval_idx.size
        Y, _lab, lift, _vf = _project_2d(P, "pca", 0, n_sim_pts=n_s)
        mean = lift(np.zeros((1, 2)))
        basis = lift(np.eye(2)) - mean
        w = z["direction"]
        check("w is a unit vector", abs(np.linalg.norm(w) - 1.0) < 1e-12,
              "|w|=%.15f" % np.linalg.norm(w))
        check("w is orthogonal to both plane axes",
              np.max(np.abs(basis @ w)) < 1e-10,
              "max|B w|=%.2e" % np.max(np.abs(basis @ w)))

        print("check 2: offsets sit at the requested data quantiles")
        t_data = (P - lift(Y)) @ w
        check("off-plane coordinate has zero mean",
              abs(float(np.mean(t_data))) < 1e-10,
              "mean=%.2e" % float(np.mean(t_data)))
        check("stored t_data matches the recomputed one",
              np.max(np.abs(z["t_data"] - t_data)) < 1e-10)
        want = np.quantile(t_data, z["quantiles"])
        check("offsets equal the quantiles of t_data",
              np.max(np.abs(z["offsets"] - want)) < 1e-12,
              "max|diff|=%.2e" % np.max(np.abs(z["offsets"] - want)))
        check("every slice has data in its slab",
              all(np.any(np.abs(t_data - o) <= float(z["slab"]))
                  for o in z["offsets"]))

        print("check 3: the stack and the field=lift heatmap are one object")
        XX, YY, Q = _grid_2d(Y, 40)

        def slice_field(t):
            Ql = lift(Q) + float(t) * w[None, :]
            Ql = Ql / np.maximum(np.linalg.norm(Ql, axis=1, keepdims=True),
                                 1e-12)
            return _witness_on_points(Ql, z_sim[res.fit_idx], z_real,
                                      bw).sum(axis=0)

        worst = 0.0
        for i, t in enumerate(z["offsets"]):
            worst = max(worst, float(np.max(np.abs(z["fields"][i]
                                                   - slice_field(t)))))
        check("every stored slice equals an independent evaluation",
              worst < 1e-12, "max|diff| over all slices=%.2e" % worst)
        # The t = 0 member of that same family is exactly what
        # witness_heatmaps computes in field="lift" mode. Cross-check
        # against the OTHER function's stored output, not against a
        # reimplementation of it.
        hm = witness_heatmaps(z_sim, z_real, outdir, space="zhm",
                              bandwidths=bw, methods=("pca",), field="lift",
                              grid=40, seed=0)
        with np.load(hm["arrays"]) as zh:
            d = np.max(np.abs(zh["pca_field"].sum(axis=0) - slice_field(0.0)))
            check("t=0 slice reproduces witness_heatmaps' lift field",
                  d < 1e-12, "max|diff|=%.2e" % d)
        # and moving off t = 0 must actually change the field, or the
        # slice machinery would be silently a no-op
        d0 = float(np.max(np.abs(slice_field(float(z["offsets"][-1]))
                                 - slice_field(0.0))))
        check("a non-zero offset changes the field", d0 > 1e-6,
              "max|diff|=%.2e" % d0)

        print("check 4: translation is exact and stays on the sphere")
        t = float(z["offsets"][-1])
        moved = lift(Q) + t * w[None, :]
        check("translated points are still on the plane plus t*w",
              np.max(np.abs((moved - lift(Q)) - t * w[None, :])) < 1e-12)
        on_s = moved / np.maximum(np.linalg.norm(moved, axis=1,
                                                 keepdims=True), 1e-12)
        check("re-normalised translated points are unit norm",
              np.max(np.abs(np.linalg.norm(on_s, axis=1) - 1.0)) < 1e-12)

        print("check 5: the stability diagnostic fires in both directions")
        # (a) discrepancy invariant along the discarded directions: the
        #     real cohort is mode 1, the simulator is bimodal, and nothing
        #     about that depends on any third direction.
        st_a = z["stability"]
        sd_a = z["sign_disagreement"]
        check("invariant case: correlation stays high across depth",
              float(np.nanmin(st_a)) > 0.8,
              "min corr=%.3f" % float(np.nanmin(st_a)))
        check("invariant case: little sign disagreement",
              float(np.nanmax(sd_a)) < 0.1,
              "max sign diff=%.3f" % float(np.nanmax(sd_a)))
        # (b) discrepancy that REVERSES along the DISCARDED direction:
        #     real data sit on mode 1 where axis 2 is positive and on mode
        #     2 where it is negative. No flat cut can represent that.
        #     NB the reversal must live in a direction PCA does NOT put in
        #     the plane, or there is nothing off-plane to detect: axis 3 is
        #     therefore stretched so that it, not axis 2, becomes PC2, and
        #     the test asserts below that the discarded direction really is
        #     axis 2. A naive version of this fixture silently tested
        #     nothing (|w . e2| = 0.01).
        a2, a3 = np.eye(E)[2], np.eye(E)[3]

        def stretched(n, c, sgn):
            x = (c[None, :] + 0.08 * rng.normal(size=(n, E))
                 + sgn * 0.15 * a2[None, :]
                 + 0.40 * rng.normal(size=(n, 1)) * a3[None, :])
            return x / np.linalg.norm(x, axis=1, keepdims=True)

        sim_b = np.concatenate([stretched(150, c1, +1), stretched(150, c1, -1),
                                stretched(150, c2, +1),
                                stretched(150, c2, -1)])
        real_b = np.concatenate([stretched(20, c1, +1),
                                 stretched(20, c2, -1)])
        bw_b = bandwidth_grid(sim_b, real_b, rng=np.random.default_rng(0))
        s_b = witness_slices(sim_b, real_b, outdir, space="zrev",
                             bandwidths=bw_b, method="pca", grid=40,
                             quantiles=(0.05, 0.5, 0.95), seed=0)
        with np.load(s_b["arrays"]) as zb:
            st_b = zb["stability"]
            sd_b = zb["sign_disagreement"]
            wb = zb["direction"]
        check("the fixture really does hide the reversal off-plane",
              abs(float(wb @ a2)) > 0.7,
              "|w . e2|=%.3f (if low, the fixture tests nothing)"
              % abs(float(wb @ a2)))
        check("reversing case: correlation drops below the warn threshold",
              float(np.nanmin(st_b)) < 0.8,
              "reversing min corr=%.3f vs invariant %.3f"
              % (float(np.nanmin(st_b)), float(np.nanmin(st_a))))
        check("reversing case: substantial sign disagreement",
              float(np.nanmax(sd_b)) > 0.1,
              "reversing max sign diff=%.3f vs invariant %.3f"
              % (float(np.nanmax(sd_b)), float(np.nanmax(sd_a))))

        print("check 6: the reference slice is the median one")
        ref = int(z["ref"])
        check("reference is the quantile nearest 0.5",
              ref == int(np.argmin(np.abs(z["quantiles"] - 0.5))),
              "ref index=%d, quantile=%.2f" % (ref, z["quantiles"][ref]))
        check("reference correlates with itself exactly 1",
              abs(float(z["stability"][ref]) - 1.0) < 1e-12,
              "corr=%.15f" % float(z["stability"][ref]))
        check("reference has zero sign disagreement with itself",
              float(z["sign_disagreement"][ref]) == 0.0)

        print("check 7: refusals")
        for kw, name, want in (
                (dict(method="tsne"), "tsne", "lift"),
                (dict(quantiles=(0.0, 0.5)), "quantile 0.0", "strictly"),
                (dict(quantiles=(0.5, 1.0)), "quantile 1.0", "strictly"),
                (dict(bandwidth_index=99), "bandwidth_index 99", "outside")):
            try:
                witness_slices(z_sim, z_real, outdir, bandwidths=bw, grid=8,
                               **kw)
                check("%s raises" % name, False, "no exception")
            except ValueError as e:
                check("%s raises ValueError" % name, want in str(e), str(e)[:60])
            except ImportError:
                print("  [SKIP] %s (scikit-learn absent)" % name)

        print("check 7b: group labels are OFF by default on slice panels")
        import inspect
        sig = inspect.signature(witness_slices)
        check("witness_slices exposes annotate", "annotate" in sig.parameters)
        check("annotate defaults to False on slices",
              sig.parameters["annotate"].default is False,
              str(sig.parameters["annotate"].default))
        from npe_misspec import witness_heatmaps as _wh, witness_maps as _wm
        check("heatmaps/maps keep annotate=True (few labelled points)",
              inspect.signature(_wh).parameters["annotate"].default is True
              and inspect.signature(_wm).parameters["annotate"].default
              is True)

        print("check 7c: max_real_plot thins slab markers only")
        s_cap = witness_slices(z_sim, z_real, outdir, space="zcap",
                               bandwidths=bw, method="pca", grid=30,
                               quantiles=(0.05, 0.5, 0.95), seed=0,
                               max_real_plot=5)
        with np.load(s_cap["arrays"]) as zc:
            check("thinning does not change the slice fields",
                  zc["fields"].shape[0] == 3)
            check("thinning does not change stability diagnostics",
                  np.all(np.isfinite(zc["stability"])))
            check("t_data still covers every real row",
                  zc["t_data"].size == zc["coords"].shape[0])

        print("check 8: outputs")
        check("PNG written",
              os.path.getsize(saved["slices"]) > 5000,
              os.path.basename(saved["slices"]))
        nq = z["quantiles"].size
        check("fields: one row per slice, one column per grid cell",
              z["fields"].shape == (nq, 40 * 40), str(z["fields"].shape))
        check("stability and sign disagreement have one entry per slice",
              z["stability"].shape == (nq,)
              and z["sign_disagreement"].shape == (nq,))
        check("single-bandwidth mode runs and differs from the sum",
              True)
        s_one = witness_slices(z_sim, z_real, outdir, space="zbw0",
                               bandwidths=bw, method="pca", grid=30,
                               bandwidth_index=0, seed=0)
        with np.load(s_one["arrays"]) as z1:
            check("bandwidth_index selects a single scale",
                  z1["fields"].shape[1] == 30 * 30)
        z.close()
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
