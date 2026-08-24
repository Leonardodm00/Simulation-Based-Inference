#!/usr/bin/env python3
"""Run the witness-function analysis on the arrays gate_run.py already wrote.

Analysis only. This script does NOT re-run the gate and does NOT touch the
parquet files: it reads <stem>_arrays.npz, which gate_run.py saves precisely
so that figures can be redrawn without repeating a minutes-long computation.
Whatever conditioning the gate ran under (activity filter, theta dedup, row
caps) is inherited automatically, because those arrays are post-filter. That
is the point -- the witness map must describe the SAME comparison the gate
delivered a verdict on, or the two disagree for uninteresting reasons.

WHAT IT PRODUCES, PER SPACE
---------------------------
  witness_heat_<space>_pca_lift.png        exact g on the variance-optimal plane
  witness_heat_<space>_contrast_lift.png   exact g on the discrepancy-aligned plane
  witness_heat_<space>_tsne_nw.png         g smoothed over a t-SNE layout
  witness_slices_<space>_pca.png           g at several depths off the PCA plane
  witness_slices_<space>_contrast.png      idem, contrast plane
  witness_map_<space>_*.png                the scatter version (no field)
  witness_hist_<space>.png                 the Mode-1 / Mode-2 histograms
  witness_summary.json                     every diagnostic, machine-readable

THE COMPARISON THIS IS FOR
--------------------------
PCA and t-SNE answer different questions and neither is the "right" one:

  PCA (lift)      exact evaluation of g on a flat plane. Trustworthy where
                  it applies, blind to whatever the plane discards. Comes
                  with rho and resid/sigma, which say whether it applies.
  contrast (lift) same, on the plane whose first axis IS the sim/real mean
                  difference. Read this when PCA's leading directions are
                  the between-mode axis of a multimodal simulator, i.e.
                  exactly the nuisance structure the gate over-fires on.
  t-SNE (nw)      no exact lift exists, so the field is interpolated from
                  the sampled g. Good at showing cluster STRUCTURE that a
                  linear plane cannot, useless for reading distances or
                  for any quantitative claim about the field between
                  points.

Agreement between the three is evidence the picture is real. Disagreement
is informative, not a failure -- see docs/witness_interpretation.md, which
is the document to read before drawing any conclusion from these figures.

USAGE
-----
    python3 witness_run.py \
        --arrays results/gate_arrays.npz \
        --misspec_dir /path/Simulation-Based-Inference/hpc \
        --out results/witness

    # faster structural check: no t-SNE, coarse grid, fewer slices
    python3 witness_run.py --arrays ... --out ... --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def _verdict(rho_sum: float, resid_over_sigma, sigma_min: float,
             slice_min_corr: float, slice_max_signdiff: float) -> str:
    """The decision rule, applied mechanically so it cannot drift between
    figures and prose. See docs/witness_interpretation.md section 5.

    rho is the criterion; the residual is an ANNOTATION saying at which
    scales it holds. An earlier version of this rule vetoed on the
    residual at the SMALLEST bandwidth alone, which let the narrowest and
    least informative kernel override a rho of 0.995. A datum sitting one
    bandwidth off-plane attenuates its own kernel contribution by
    exp(-1/2) = 0.61 -- real, but it is rho that measures whether that
    attenuation actually distorts the picture. So rho decides, and the
    residual reports scope. Only when NO scale resolves the data (every
    resid/sigma > 1) is the plane downgraded irrespective of rho.
    """
    ros = (np.asarray(resid_over_sigma, dtype=np.float64).ravel()
           if resid_over_sigma is not None else np.asarray([np.nan]))
    n_tot = int(np.sum(np.isfinite(ros)))
    n_res = int(np.sum(np.isfinite(ros) & (ros <= 1.0)))
    scope = ("" if n_tot == 0 else
             " [%d/%d bandwidths resolve the data; at the others the plane "
             "cuts through empty space and those panels mean less]"
             % (n_res, n_tot))
    faithful = (np.isfinite(rho_sum) and rho_sum >= 0.9
                and (n_tot == 0 or n_res > 0))
    if np.isfinite(slice_min_corr):
        stable = (slice_min_corr >= 0.8 and slice_max_signdiff <= 0.1)
    else:
        stable = None
    if faithful and stable is not False:
        return ("READ THE PLANE: the flat cut reproduces g at the data"
                + ("" if stable is None else " and survives translation")
                + scope)
    if not faithful and stable is False:
        return ("DO NOT READ THE PLANE: the cut neither reproduces g at the "
                "data nor survives translation; use field='proj'/'nw' and "
                "treat the discarded directions as carrying discrepancy"
                + scope)
    if not faithful:
        return ("READ WITH CAUTION: the flat cut does not track g at the "
                "data; cross-check against field='proj' and the t-SNE panel"
                + scope)
    return ("READ WITH CAUTION: the cut tracks g at the data but does NOT "
            "survive translation; structure exists in the discarded "
            "directions" + scope)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", required=True,
                    help="the <stem>_arrays.npz written by gate_run.py")
    ap.add_argument("--misspec_dir", default=None,
                    help="dir holding npe_misspec.py; defaults to this "
                         "script's own directory")
    ap.add_argument("--out", required=True, help="output DIRECTORY")
    ap.add_argument("--spaces", default="z,zraw")
    ap.add_argument("--methods", default="pca,contrast,tsne",
                    help="comma list from pca, contrast, tsne")
    ap.add_argument("--grid", type=int, default=140,
                    help="heatmap grid points per axis")
    ap.add_argument("--slice_grid", type=int, default=100)
    ap.add_argument("--quantiles", default="0.05,0.25,0.5,0.75,0.95",
                    help="slice depths, as quantiles of the data's own "
                         "off-plane coordinate")
    ap.add_argument("--max_points", type=int, default=2000,
                    help="cap on PLOTTED simulated points; scores are "
                         "computed on all evaluated points regardless")
    ap.add_argument("--no_split", action="store_true",
                    help="estimate mu_sim on the same simulated rows it is "
                         "evaluated on. Makes mean(u)-mean(v) exactly the "
                         "gate's MMD^2, at the cost of in-sample scores")
    ap.add_argument("--no_slices", action="store_true")
    ap.add_argument("--bandwidth_index", type=int, default=None,
                    help="restrict the slice stack to one bandwidth of the "
                         "grid; use when the heatmap panels disagreed "
                         "across scales")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="structural check: no t-SNE, coarse grids, 3 slices")
    args = ap.parse_args()

    if args.quick:
        args.methods = ",".join(m for m in args.methods.split(",")
                                if m.strip() != "tsne")
        args.grid = 60
        args.slice_grid = 40
        args.quantiles = "0.05,0.5,0.95"

    misspec_dir = args.misspec_dir or os.path.dirname(
        os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(misspec_dir))
    import npe_misspec as M                                    # noqa: E402

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    npz = np.load(args.arrays, allow_pickle=False)
    keys = list(npz.keys())
    print("[1/3] loaded %s" % args.arrays)
    print("      arrays present: %s"
          % ", ".join(k for k in keys if not k.startswith("z::")
                      and not k.startswith("zraw::")))

    groups = None
    if "real_groups" in keys:
        groups = np.asarray([str(g) for g in npz["real_groups"]])
        print("      real recordings carry %d distinct group labels"
              % len(set(groups.tolist())))

    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    quantiles = tuple(float(q) for q in args.quantiles.split(",")
                      if q.strip())
    linear = [m for m in methods if m in ("pca", "contrast")]

    summary: Dict[str, Dict] = {}
    print("[2/3] witness maps, heatmaps and slices")
    for space in spaces:
        ks, kr = "sim_%s" % space, "real_%s" % space
        if ks not in keys or kr not in keys:
            print("      SKIP %s: %s or %s absent from the npz"
                  % (space, ks, kr))
            continue
        z_sim = np.asarray(npz[ks], dtype=np.float64)
        z_real = np.asarray(npz[kr], dtype=np.float64)
        g_use = groups if (groups is not None
                           and groups.shape[0] == z_real.shape[0]) else None
        if groups is not None and g_use is None:
            print("      NOTE %s: %d group labels but %d real rows; labels "
                  "not used" % (space, groups.shape[0], z_real.shape[0]))
        print("      %s: sim %s, real %s" % (space, z_sim.shape, z_real.shape))

        rec: Dict[str, object] = {"n_sim": int(z_sim.shape[0]),
                                  "n_real": int(z_real.shape[0]),
                                  "E": int(z_sim.shape[1])}

        # scatter + histograms: the Mode-1 / Mode-2 readout
        try:
            paths = M.witness_maps(z_sim, z_real, args.out, space=space,
                                   split=not args.no_split,
                                   methods=tuple(methods), groups=g_use,
                                   seed=args.seed,
                                   max_points=args.max_points)
            rec["maps"] = {k: os.path.basename(v) for k, v in paths.items()}
        except Exception as exc:                               # noqa: BLE001
            rec["maps_error"] = str(exc)
            print("      witness_maps FAILED: %s" % exc)

        # the identity: mean(u) - mean(v) is the gate's own statistic
        res = M.witness_function(z_sim, z_real, space=space,
                                 split=not args.no_split, seed=args.seed)
        rec["bandwidths"] = _jsonable(res.bandwidths)
        rec["witness_gap"] = float(res.mmd2_from_witness())
        rec["witness_gap_is_exact_mmd2"] = bool(args.no_split)
        rec["most_negative_real"] = _jsonable(
            [(str(g_use[j]) if g_use is not None else int(j),
              float(res.v_sum[j]))
             for j in np.argsort(res.v_sum)[:5]])
        rec["notes"] = list(res.notes)

        heat: Dict[str, Dict] = {}
        try:
            hp = M.witness_heatmaps(z_sim, z_real, args.out, space=space,
                                    split=not args.no_split,
                                    methods=tuple(methods), field="auto",
                                    grid=args.grid, groups=g_use,
                                    seed=args.seed,
                                    max_points=args.max_points)
            with np.load(hp["arrays"], allow_pickle=False) as zh:
                for m in methods:
                    if "%s_field_mode" % m not in zh:
                        continue
                    d = {"field": str(zh["%s_field_mode" % m]),
                         "var_frac": float(zh["%s_var_frac" % m])
                         if "%s_var_frac" % m in zh else float("nan")}
                    if "%s_rho_sum" % m in zh:
                        d["rho_sum"] = float(zh["%s_rho_sum" % m])
                    if "%s_rho" % m in zh:
                        d["rho_per_bandwidth"] = _jsonable(zh["%s_rho" % m])
                    if "%s_resid" % m in zh:
                        d["resid_median"] = float(
                            np.median(zh["%s_resid" % m]))
                        d["resid_over_sigma"] = _jsonable(
                            zh["%s_resid_over_sigma" % m])
                    heat[m] = d
            rec["heatmaps"] = heat
            rec["heatmap_files"] = {k: os.path.basename(v)
                                    for k, v in hp.items()}
        except Exception as exc:                               # noqa: BLE001
            rec["heatmaps_error"] = str(exc)
            print("      witness_heatmaps FAILED: %s" % exc)

        sl: Dict[str, Dict] = {}
        if not args.no_slices:
            for m in linear:
                try:
                    sp = M.witness_slices(
                        z_sim, z_real, args.out, space=space, method=m,
                        split=not args.no_split, quantiles=quantiles,
                        grid=args.slice_grid, groups=g_use, seed=args.seed,
                        max_points=args.max_points,
                        bandwidth_index=args.bandwidth_index)
                    with np.load(sp["arrays"], allow_pickle=False) as zs:
                        sl[m] = {
                            "offsets": _jsonable(zs["offsets"]),
                            "quantiles": _jsonable(zs["quantiles"]),
                            "stability": _jsonable(zs["stability"]),
                            "sign_disagreement":
                                _jsonable(zs["sign_disagreement"]),
                            "min_corr": float(np.nanmin(zs["stability"])),
                            "max_sign_disagreement":
                                float(np.nanmax(zs["sign_disagreement"])),
                            "file": os.path.basename(sp["slices"]),
                        }
                except Exception as exc:                       # noqa: BLE001
                    sl[m] = {"error": str(exc)}
                    print("      witness_slices[%s] FAILED: %s" % (m, exc))
            rec["slices"] = sl

        # mechanical verdict per linear method
        verdicts = {}
        smin = float(np.min(res.bandwidths))
        for m in linear:
            h = heat.get(m, {})
            s = sl.get(m, {})
            verdicts[m] = _verdict(
                float(h.get("rho_sum", float("nan"))),
                h.get("resid_over_sigma"), smin,
                float(s.get("min_corr", float("nan"))),
                float(s.get("max_sign_disagreement", float("nan"))))
            print("      VERDICT %s/%s: %s" % (space, m, verdicts[m]))
        rec["verdicts"] = verdicts
        summary[space] = rec

    print("[3/3] writing the summary")
    doc = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "args": {k: v for k, v in vars(args).items()},
        "source_arrays": os.path.abspath(args.arrays),
        "spaces": _jsonable(summary),
        "how_to_read": (
            "See docs/witness_interpretation.md. In short: rho_sum >= 0.9 "
            "means the lifted plane reproduces g at the data, and "
            "resid_over_sigma says at WHICH bandwidths that holds (entries "
            "<= 1 are resolved); slice min_corr >= 0.8 with "
            "max_sign_disagreement <= 0.1 means it survives translation. "
            "Both hold -> read the plane. Otherwise the t-SNE and proj "
            "views carry the information the plane discards."),
        "caveats": [
            "The witness is estimated from the same data it is evaluated "
            "on unless --no_split is off (it is on by default, which "
            "splits); any 'these recordings are anomalous' claim read off "
            "these figures is EXPLORATORY. The gate's permutation p-value "
            "is the inferential statement, not the map.",
            "With split=True the identity mean(u)-mean(v) = MMD^2 holds in "
            "expectation, not exactly; pass --no_split for the exact "
            "identity at the cost of in-sample scores.",
            "t-SNE panels use field='nw', which INTERPOLATES the sampled "
            "witness rather than evaluating it. No quantitative claim "
            "about the field between points is licensed by a t-SNE panel.",
            "These arrays inherit whatever conditioning gate_run.py ran "
            "under (activity filter, theta dedup, row caps). State that "
            "conditioning alongside any conclusion drawn here.",
        ],
    }
    path = os.path.join(args.out, "witness_summary.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print("")
    print("wrote %s" % path)
    print("figures in %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
