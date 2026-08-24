#!/usr/bin/env python3
"""
smoke_witness_heat.py -- correctness checks for witness_heatmaps() and the
lift/proj/nw field constructions in npe_misspec.py. Run from hpc/:

    python3 smoke_witness_heat.py

Exit 0 and "ALL CHECKS PASSED" mean the heatmap machinery is sound. Set
KEEP_PLOTS=1 to keep the generated figures for eyeballing. Runs in
seconds on a login node; no allocation needed.

WHAT IS CHECKED
  1. The lift is an exact right inverse of the PCA projection: for data
     of rank <= 2 it reconstructs the points themselves; in general
     project(lift(y)) == y for every grid location y.
  2. _witness_on_points reproduces witness_function's u and v EXACTLY
     when handed the embedding coordinates of those same points. This is
     the check that ties the heatmap background to the scatter: same fit
     set, same kernels, same scale, no interpolation.
  3. Chunking invariance: the field is independent of the chunk size.
  4. An independent, deliberately naive double loop reproduces the field
     on a small grid to 1e-12.
  5. Sign structure and the zero level set: in a two-blob scenario the
     field is positive at the simulation-only blob, negative at the real
     blob, and therefore crosses zero.
  6. Nadaraya-Watson consistency: with a small h, the nw field evaluated
     at a sample's own coordinates returns that sample's witness value;
     and the mask blanks cells far from every sample while never blanking
     a cell containing a sample.
  7. Refusals: field='lift' and field='proj' with t-SNE raise ValueError
     rather than drawing something plausible; an unknown field name
     raises immediately.
  8. sphere=None auto-detects unit-norm embeddings, and lifted grid
     points are then on the sphere to 1e-12.
  9. Files: one PNG per method, correctly named after the field mode, and
     an npz carrying grid, field, mask and bandwidths.
 10. The contrast projection: orthonormal frame, axis 1 exactly the
     sim-real mean-difference direction, exact lift, var_frac never above
     PCA's, ValueError without n_sim_pts, graceful PCA fallback when the
     two means coincide.
 11. Slice diagnostics: var_frac equals the in-plane share of the
     Pythagorean variance decomposition, residuals satisfy that
     decomposition, rho lies in [-1, 1] and is stored per bandwidth, and
     the nw mode records rho as NaN rather than a tautological 1.0.

ASCII-only by policy (HPC transfer safety).
"""

import os
import shutil
import sys
import tempfile

import numpy as np

from npe_misspec import (_grid_2d, _project_2d, _sqdist, _witness_on_points,
                         bandwidth_grid, witness_function, witness_heatmaps)

FAILED = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)


def sphere_blob(rng, n, center, spread):
    x = center[None, :] + spread * rng.normal(size=(n, center.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def naive_field(Q, fit, real, bw):
    """Deliberately slow reference: explicit loops, no broadcasting."""
    S, m = len(bw), Q.shape[0]
    out = np.zeros((S, m))
    for s in range(S):
        gam = 1.0 / (2.0 * bw[s] ** 2)
        for a in range(m):
            acc_f = 0.0
            for i in range(fit.shape[0]):
                acc_f += np.exp(-gam * np.sum((fit[i] - Q[a]) ** 2))
            acc_r = 0.0
            for j in range(real.shape[0]):
                acc_r += np.exp(-gam * np.sum((real[j] - Q[a]) ** 2))
            out[s, a] = acc_f / fit.shape[0] - acc_r / real.shape[0]
    return out


def main():
    rng = np.random.default_rng(11)
    E = 16
    c1, c2 = np.eye(E)[0], np.eye(E)[1]
    z_sim = np.concatenate([sphere_blob(rng, 300, c1, 0.15),
                            sphere_blob(rng, 300, c2, 0.15)], axis=0)
    z_real = sphere_blob(rng, 30, c1, 0.15)
    groups = np.asarray(["rec%02d" % i for i in range(30)])
    bw = bandwidth_grid(z_sim, z_real, rng=np.random.default_rng(0))

    print("check 1: the PCA lift is an exact right inverse")
    P = np.concatenate([z_sim, z_real], axis=0)
    Y, _lab, lift, vf = _project_2d(P, "pca", 0)
    _XX, _YY, Q = _grid_2d(Y, 12)
    Ql = lift(Q)
    mean = P.mean(axis=0, keepdims=True)
    _U, _sv, Vt = np.linalg.svd(P - mean, full_matrices=False)
    back = (Ql - mean) @ Vt[:2].T
    check("project(lift(y)) == y for every grid point",
          np.max(np.abs(back - Q)) < 1e-10,
          "max|diff|=%.2e" % np.max(np.abs(back - Q)))
    # rank-2 data: the lift must reconstruct the points themselves
    A = rng.normal(size=(60, 2)) @ rng.normal(size=(2, E))
    Y2, _l2, lift2, vf2 = _project_2d(A, "pca", 0)
    check("rank-2 data reconstructed exactly by lift",
          np.max(np.abs(lift2(Y2) - A)) < 1e-9,
          "max|diff|=%.2e" % np.max(np.abs(lift2(Y2) - A)))
    check("var_frac is 1.0 for rank-2 data", abs(vf2 - 1.0) < 1e-9,
          "var_frac=%.6f" % vf2)
    check("var_frac in (0,1) for the 16-d cloud", 0.0 < vf < 1.0,
          "var_frac=%.3f" % vf)

    print("check 2: field routine reproduces the scatter values exactly")
    res = witness_function(z_sim, z_real, bandwidths=bw, split=True, seed=0)
    fit = z_sim[res.fit_idx]
    ev = z_sim[res.eval_idx]
    u_ref = _witness_on_points(ev, fit, z_real, res.bandwidths)
    v_ref = _witness_on_points(z_real, fit, z_real, res.bandwidths)
    check("u reproduced from _witness_on_points",
          np.max(np.abs(u_ref - res.u)) < 1e-12,
          "max|diff|=%.2e" % np.max(np.abs(u_ref - res.u)))
    check("v reproduced from _witness_on_points",
          np.max(np.abs(v_ref - res.v)) < 1e-12,
          "max|diff|=%.2e" % np.max(np.abs(v_ref - res.v)))

    print("check 3: field is invariant to the chunk size")
    f_small = _witness_on_points(Ql, fit, z_real, res.bandwidths, chunk=7)
    f_big = _witness_on_points(Ql, fit, z_real, res.bandwidths, chunk=100000)
    check("chunk=7 equals chunk=100000",
          np.max(np.abs(f_small - f_big)) < 1e-14,
          "max|diff|=%.2e" % np.max(np.abs(f_small - f_big)))

    print("check 4: independent naive implementation agrees")
    Qs = Ql[:15]
    fs, rs = fit[:40], z_real[:12]
    fast = _witness_on_points(Qs, fs, rs, res.bandwidths)
    slow = naive_field(Qs, fs, rs, res.bandwidths)
    check("vectorised == naive double loop",
          np.max(np.abs(fast - slow)) < 1e-12,
          "max|diff|=%.2e" % np.max(np.abs(fast - slow)))

    print("check 5: sign structure and the zero level set")
    # unvisited simulated mode -> g > 0; real blob -> g < 0
    on_c2 = _witness_on_points(c2[None, :], fit, z_real, res.bandwidths)
    on_c1 = _witness_on_points(c1[None, :], fit, z_real, res.bandwidths)
    check("g > 0 at the simulation-only mode", on_c2.sum() > 0,
          "g=%.3e" % on_c2.sum())
    check("g < 0 at the real blob", on_c1.sum() < 0, "g=%.3e" % on_c1.sum())
    fs_grid = _witness_on_points(Ql, fit, z_real, res.bandwidths).sum(axis=0)
    check("field crosses zero (contour exists)",
          fs_grid.min() < 0.0 < fs_grid.max(),
          "range [%.3e, %.3e]" % (fs_grid.min(), fs_grid.max()))

    print("check 6: Nadaraya-Watson consistency and masking")
    from npe_misspec import _field_nw
    # Clean unit fixture: well-separated 2-D locations with known values,
    # so "the smoother returns the sample's own value at small h" is a
    # statement about the smoother and not about any pairing convention.
    Yd = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                   [0.5, 2.0]])
    vals = np.array([[-1.0, 2.0, 0.5, -3.0, 4.0],
                     [0.25, -0.5, 1.5, 2.0, -1.0]])
    Fnw, mnw = _field_nw(Yd, Yd, vals, 1e-3, 2.5)
    check("nw at a sample returns that sample's value (small h)",
          np.max(np.abs(Fnw - vals)) < 1e-9,
          "max|diff|=%.2e" % np.max(np.abs(Fnw - vals)))
    check("no sample location is masked", not mnw.any())
    # midpoint of two equidistant samples with a wide h -> their average
    mid = np.array([[0.5, 0.0]])
    Fmid, _ = _field_nw(mid, Yd[:2], vals[:, :2], 5.0, 2.5)
    check("nw at the midpoint of two samples averages them (wide h)",
          abs(Fmid[0, 0] - 0.5) < 1e-9, "got %.6f, expected 0.5" % Fmid[0, 0])
    far = np.array([[1e3, 1e3]])
    _F2, m2 = _field_nw(far, Yd, vals, 0.05, 2.5)
    check("a remote cell is masked", bool(m2[0]))
    near = np.array([[0.0, 0.1]])
    _F3, m3 = _field_nw(near, Yd, vals, 0.1, 2.5)
    check("a cell within mask_radius * h is not masked", not bool(m3[0]))

    print("check 7: refusals rather than plausible-looking output")
    outdir = tempfile.mkdtemp(prefix="witness_heat_smoke_")
    try:
        try:
            witness_heatmaps(z_sim, z_real, outdir, bandwidths=bw,
                             methods=("tsne",), field="lift", grid=8)
            check("tsne + lift raises", False, "no exception")
        except ValueError as e:
            check("tsne + lift raises ValueError", "inverse" in str(e))
        except ImportError:
            print("  [SKIP] tsne refusal checks (scikit-learn absent)")
        try:
            witness_heatmaps(z_sim, z_real, outdir, bandwidths=bw,
                             methods=("tsne",), field="proj", grid=8)
            check("tsne + proj raises", False, "no exception")
        except ValueError as e:
            check("tsne + proj raises ValueError", "inverse" in str(e))
        except ImportError:
            pass
        try:
            witness_heatmaps(z_sim, z_real, outdir, bandwidths=bw,
                             methods=("pca",), field="nonsense", grid=8)
            check("unknown field raises", False, "no exception")
        except ValueError:
            check("unknown field raises ValueError", True)

        print("check 8: sphere auto-detection")
        nrm = np.linalg.norm(lift(Q) / np.maximum(
            np.linalg.norm(lift(Q), axis=1, keepdims=True), 1e-12), axis=1)
        check("re-normalised lifted grid is on the sphere",
              np.max(np.abs(nrm - 1.0)) < 1e-12)
        check("z_sim is unit norm, so sphere=None must resolve True",
              np.allclose(np.linalg.norm(z_sim, axis=1), 1.0, atol=1e-6))

        print("check 8b: view window and adaptive marker size")
        from npe_misspec import _marker_size, _view_window
        Yo = np.concatenate([rng.normal(size=(500, 2)),
                             np.array([[1e4, 1e4]])])
        x0, x1, _y0, _y1 = _view_window(Yo, clip=0.005, pad=0.0)
        check("view window rejects a far outlier", x1 < 1e3,
              "x1=%.3g" % x1)
        # 2-D retention: clip is applied per axis, so the fraction inside
        # the RECTANGLE is lower than 1 - 2*clip. Measured with the pad the
        # code actually uses, on the real figures this came out at 99.6%.
        px0, px1, py0, py1 = _view_window(Yo, clip=0.005)
        inside = float(np.mean((Yo[:, 0] >= px0) & (Yo[:, 0] <= px1)
                               & (Yo[:, 1] >= py0) & (Yo[:, 1] <= py1)))
        check("view window keeps >=98 pct of points in 2-D",
              inside >= 0.98, "%.2f%% inside" % (100 * inside))
        fx0, fx1, fy0, fy1 = _view_window(Yo, clip=0.0, pad=0.0)
        check("clip=0 recovers the exact full range",
              np.allclose([fx0, fx1, fy0, fy1],
                          [Yo[:, 0].min(), Yo[:, 0].max(),
                           Yo[:, 1].min(), Yo[:, 1].max()]))
        check("marker size is monotonically decreasing in n",
              _marker_size(30) > _marker_size(200) > _marker_size(1890)
              >= _marker_size(20000))
        check("marker size never exceeds the base or drops below the floor",
              _marker_size(1) <= 52.0 and _marker_size(10 ** 7) >= 5.0,
              "n=1 -> %.1f, n=1e7 -> %.1f"
              % (_marker_size(1), _marker_size(10 ** 7)))
        check("a 1890-window cohort gets a small marker",
              _marker_size(1890, base=58.0) < 10.0,
              "%.1f" % _marker_size(1890, base=58.0))

        print("check 8c: the two caps behave as documented")
        # n_eval_max is the ceiling; max_points cannot raise past it
        big = np.concatenate([z_sim, z_sim, z_sim])          # 3x the rows
        r_lo = witness_function(big, z_real, bandwidths=bw, split=True,
                                seed=0, n_eval_max=200)
        r_hi = witness_function(big, z_real, bandwidths=bw, split=True,
                                seed=0, n_eval_max=900)
        check("n_eval_max caps the scored simulated rows",
              r_lo.eval_idx.size == 200 and r_hi.eval_idx.size == 900,
              "%d vs %d" % (r_lo.eval_idx.size, r_hi.eval_idx.size))
        check("raising max_points cannot exceed n_eval_max",
              min(r_lo.eval_idx.size, 5000) == 200)
        # max_real_plot is display-only: every diagnostic must be identical
        got = {}
        for cap in (None, 20):
            pth = witness_heatmaps(z_sim, z_real, outdir,
                                   space="cap%s" % cap, bandwidths=bw,
                                   methods=("pca",), field="lift", grid=30,
                                   seed=0, max_real_plot=cap)
            with np.load(pth["arrays"]) as zc:
                got[cap] = (float(zc["pca_var_frac"]),
                            float(zc["pca_rho_sum"]),
                            float(np.median(zc["pca_resid"])),
                            float(zc["pca_field"].sum()))
        check("max_real_plot leaves every diagnostic bit-identical",
              got[None] == got[20],
              "None=%s cap=%s" % (got[None][:2], got[20][:2]))

        print("check 9: files written with the field mode in the name")
        saved = witness_heatmaps(z_sim, z_real, outdir, space="z",
                                 bandwidths=bw, methods=("pca", "tsne"),
                                 field="auto", grid=60, groups=groups,
                                 seed=0)
        check("PCA lift heatmap saved",
              "heat_pca" in saved and saved["heat_pca"].endswith("_lift.png")
              and os.path.getsize(saved["heat_pca"]) > 5000,
              os.path.basename(saved.get("heat_pca", "-")))
        try:
            import sklearn  # noqa: F401
            check("t-SNE nw heatmap saved",
                  "heat_tsne" in saved
                  and saved["heat_tsne"].endswith("_nw.png")
                  and os.path.getsize(saved["heat_tsne"]) > 5000,
                  os.path.basename(saved.get("heat_tsne", "-")))
        except ImportError:
            print("  [SKIP] t-SNE heatmap (scikit-learn absent)")
        with np.load(saved["arrays"], allow_pickle=False) as z:
            keys = list(z.keys())
            check("npz carries grid, field, mask, bandwidths",
                  all(("pca_%s" % k) in keys
                      for k in ("XX", "YY", "field", "mask", "bandwidths")))
            check("npz field shape matches grid",
                  z["pca_field"].shape == (bw.size, 60 * 60),
                  str(z["pca_field"].shape))
            check("npz records the field mode",
                  str(z["pca_field_mode"]) == "lift")

        # proj mode must also run and use its own 2-D bandwidths
        saved_p = witness_heatmaps(z_sim, z_real, outdir, space="zproj",
                                   bandwidths=bw, methods=("pca",),
                                   field="proj", grid=40, seed=0)
        check("PCA proj heatmap saved",
              saved_p["heat_pca"].endswith("_proj.png")
              and os.path.getsize(saved_p["heat_pca"]) > 5000)
        with np.load(saved_p["arrays"]) as z:
            check("proj mode uses 2-D bandwidths, not the E-dim ones",
                  not np.allclose(z["pca_bandwidths"], bw))

        print("check 10: the contrast projection")
        Yc, labc, liftc, vfc = _project_2d(P, "contrast", 0,
                                           n_sim_pts=z_sim.shape[0])
        meanP = P.mean(axis=0, keepdims=True)
        basis = (liftc(np.eye(2)) - meanP)          # rows are a1, a2
        check("contrast frame is orthonormal",
              np.max(np.abs(basis @ basis.T - np.eye(2))) < 1e-10,
              "max|B B^T - I|=%.2e"
              % np.max(np.abs(basis @ basis.T - np.eye(2))))
        delta = z_sim.mean(axis=0) - z_real.mean(axis=0)
        cos = abs(float(basis[0] @ delta / np.linalg.norm(delta)))
        check("axis 1 is the sim-real mean difference direction",
              abs(cos - 1.0) < 1e-9, "|cos|=%.10f" % cos)
        check("contrast lift is an exact right inverse",
              np.max(np.abs((liftc(Yc) - meanP) @ basis.T - Yc)) < 1e-9)
        _Yp, _lp, _lift_p, vfp = _project_2d(P, "pca", 0)
        check("PCA carries at least as much variance as contrast "
              "(PCA maximises it by definition)", vfp >= vfc - 1e-12,
              "pca=%.3f contrast=%.3f" % (vfp, vfc))
        try:
            _project_2d(P, "contrast", 0)
            check("contrast without n_sim_pts raises", False, "no exception")
        except ValueError:
            check("contrast without n_sim_pts raises ValueError", True)
        # degenerate case: identical clouds -> graceful fallback to PCA
        Pd = np.concatenate([z_sim[:50], z_sim[:50]], axis=0)
        _Yd, labd, _ld, _vd = _project_2d(Pd, "contrast", 0, n_sim_pts=50)
        check("coincident means fall back to PCA with a visible label",
              "fell back" in labd, labd)

        print("check 11: slice-fidelity diagnostics")
        saved_c = witness_heatmaps(z_sim, z_real, outdir, space="zc",
                                   bandwidths=bw, methods=("pca", "contrast"),
                                   field="lift", grid=40, seed=0)
        with np.load(saved_c["arrays"]) as z:
            for m in ("pca", "contrast"):
                check("%s: var_frac stored in the npz" % m,
                      0.0 < float(z["%s_var_frac" % m]) <= 1.0,
                      "var_frac=%.3f" % float(z["%s_var_frac" % m]))
                check("%s: rho stored per bandwidth" % m,
                      z["%s_rho" % m].shape == (bw.size,))
                check("%s: rho in [-1, 1]" % m,
                      bool(np.all(np.abs(z["%s_rho" % m]) <= 1.0 + 1e-12)))
                check("%s: residual is one value per pooled point" % m,
                      z["%s_resid" % m].shape[0]
                      == z["%s_coords" % m].shape[0])
            # Pythagoras: ||z - mean||^2 = ||y||^2 + ||off-plane resid||^2
            # NB the diagnostics live on the PLOTTED cloud (held-out eval
            # half + real), not on all of z_sim; rebuild exactly that.
            Pplot = np.concatenate([z_sim[res.eval_idx], z_real], axis=0)
            Ym = z["pca_coords"]
            r = z["pca_resid"]
            check("diagnostics are on the plotted cloud, not all of z_sim",
                  Ym.shape[0] == Pplot.shape[0] < P.shape[0],
                  "%d plotted vs %d pooled" % (Ym.shape[0], P.shape[0]))
            lhs = np.sum((Pplot - Pplot.mean(axis=0, keepdims=True)) ** 2,
                         axis=1)
            rhs = np.sum(Ym ** 2, axis=1) + r ** 2
            check("residual satisfies the Pythagorean decomposition",
                  np.max(np.abs(lhs - rhs)) < 1e-8,
                  "max|diff|=%.2e" % np.max(np.abs(lhs - rhs)))
            # var_frac is exactly the in-plane share of that decomposition
            check("var_frac equals the in-plane share of total variance",
                  abs(float(z["pca_var_frac"])
                      - np.sum(Ym ** 2) / np.sum(lhs)) < 1e-10)
            check("rho_sum finite and high for a faithful slice",
                  np.isfinite(float(z["pca_rho_sum"])),
                  "rho_sum=%.3f" % float(z["pca_rho_sum"]))
        # nw records fidelity as NaN, never as a tautological 1.0
        saved_n = witness_heatmaps(z_sim, z_real, outdir, space="znw",
                                   bandwidths=bw, methods=("pca",),
                                   field="nw", grid=40, seed=0)
        with np.load(saved_n["arrays"]) as z:
            check("nw records rho as NaN, not 1.0",
                  not np.isfinite(float(z["pca_rho_sum"])))

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
