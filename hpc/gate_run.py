#!/usr/bin/env python3
"""Run the group-aware misspecification gate on the exported arms.

Analysis only: loading lives in gate_data.py, figures in gate_plots.py. This
script writes a results JSON plus an .npz of the arrays the figures need, so
plots can be redrawn without re-running the gate (which is minutes, not
seconds, at n_null=500 x n_window_choices=100).

WHAT THE GATE DOES, AND WHAT IT DOES NOT
----------------------------------------
It tests the SIMULATOR against reality, not the inference against the
simulator: if real embeddings do not lie inside the simulated distribution,
a perfectly calibrated posterior is a posterior for a different world.

The verdict that counts is p_group, the group-aware one. Each real recording
is cut into disjoint windows sharing one culture, so real rows are clustered;
comparing clustered observations against an i.i.d. null makes the null too
tight and the gate fires on a well-specified model. The suite's own G3 test
measures this (over 6 seeds the i.i.d. null falsely rejected 6/6 while the
group-aware verdict passed 5/6). p_iid is recorded here for contrast and is
never used to decide.

ASYMMETRY THAT REMAINS. misspecification_gate() accepts groups for the REAL
arm only; z_sim is treated as i.i.d. Simulated rows sharing a topology are
equally non-independent. Deduplicating theta (gate_data.load_sim) removes the
seed-replay duplication, but the residual topology clustering cannot be
corrected here and is reported instead -- see 'sim_topology_diversity' in the
results JSON.

USAGE
-----
    python3 gate_run.py \
        --real  /path/sbi_real_cohort.parquet \
        --sim   '/path/sbi_campaign_*rho1300v3__*.parquet' \
        --misspec_dir /path/Simulation-Based-Inference/hpc \
        --out   results/gate

Add --quick for a fast structural check (n_null=50, n_window_choices=10);
the numbers are NOT publishable, it only proves the plumbing.
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
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="real cohort parquet")
    ap.add_argument("--sim", required=True,
                    help="glob for simulated shards (quote it)")
    ap.add_argument("--misspec_dir", required=True,
                    help="dir holding npe_misspec.py and npe_contract.py")
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--group_col", default="culture")
    ap.add_argument("--class_col", default="condition")
    ap.add_argument("--n_null", type=int, default=500)
    ap.add_argument("--n_window_choices", type=int, default=100,
                    help="the default in the gate (20) is tuned for ~6 rows "
                         "per group; the real cohort has 54 (9 subregions x "
                         "6 windows), so more choices are needed to average "
                         "the verdict properly")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_dedup", action="store_true",
                    help="keep duplicated theta rows (NOT recommended; see "
                         "gate_data.load_sim)")
    ap.add_argument("--max_sim_rows", type=int, default=None)
    ap.add_argument("--activity", default=None,
                    help="activity table .npz from build_activity_table.py, "
                         "built against the SAME --sim glob")
    ap.add_argument("--min_rate", type=float, default=None,
                    help="drop simulations below this Hz/electrode. NOTE: "
                         "this CONDITIONS the test -- it then asks whether "
                         "real data lies inside P_sim(. | rate >= min), a "
                         "weaker claim than the unconditioned gate.")
    ap.add_argument("--max_rate", type=float, default=None,
                    help="drop simulations above this Hz/electrode; use with "
                         "--min_rate to restrict to the real observed range")
    ap.add_argument("--spaces", default="z,zraw")
    ap.add_argument("--skip_mde", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_null = 50
        args.n_window_choices = 10

    sys.path.insert(0, os.path.abspath(args.misspec_dir))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import npe_misspec as M                                   # noqa: E402
    import gate_data as D                                     # noqa: E402

    t0 = time.time()
    print("[1/5] loading the real cohort")
    real = D.load_real(args.real, group_col=args.group_col,
                       class_col=args.class_col)
    n_groups = len(set(real.groups.tolist()))
    print("      rows = %d, groups = %d, E = %d"
          % (real.n, n_groups, real.E))
    print("      rows per group: %s"
          % (sorted(set(np.bincount(
              np.unique(real.groups, return_inverse=True)[1]).tolist())),))

    print("[2/5] loading the simulated arm")
    sim = D.load_sim(args.sim, dedup_theta=not args.no_dedup,
                     max_rows=args.max_sim_rows, seed=args.seed,
                     activity_path=args.activity,
                     min_rate=args.min_rate, max_rate=args.max_rate)
    af = sim.meta.get("activity_filter")
    if af:
        print("      activity filter [%s, %s] Hz/electrode: kept %d / %d "
              "(%.1f%%), retained rate median %.4g"
              % (af["min_rate_hz_per_electrode"],
                 af["max_rate_hz_per_electrode"], af["n_after"],
                 af["n_before"], 100.0 * af["fraction_kept"],
                 af["kept_rate_median"]))
    print("      shards = %s, rows raw = %s, rows used = %d"
          % (sim.meta.get("n_shards"), sim.meta.get("n_rows_raw"), sim.n))
    if "duplication_factor" in sim.meta:
        print("      duplication factor removed: %.2fx"
              % sim.meta["duplication_factor"])

    pnames = list(getattr(sim.contract, "param_names", []))
    topo = D.topology_diversity(sim.theta, pnames)
    print("      distinct topology draws behind those rows: %s"
          % topo.get("n_topologies"))

    if sim.E != real.E:
        raise SystemExit("embedding dim differs: sim %d vs real %d"
                         % (sim.E, real.E))

    print("[3/5] embedding geometry")
    diag = {}
    for nm, arm in (("sim", sim), ("real", real)):
        r, ev = D.effective_rank(arm.z)
        diag["%s_r_eff" % nm] = r
        diag["%s_spectrum" % nm] = ev
        print("      %-4s r_eff = %.3f of E = %d" % (nm, r, arm.E))

    spaces = [s.strip() for s in args.spaces.split(",") if s.strip()]
    results: Dict[str, Dict] = {}
    arrays: Dict[str, np.ndarray] = {}

    print("[4/5] running the gate (n_null=%d, n_window_choices=%d)"
          % (args.n_null, args.n_window_choices))
    for space in spaces:
        zs = sim.z if space == "z" else sim.zraw
        zr = real.z if space == "z" else real.zraw
        if zs is None or zr is None:
            print("      SKIP %s: not present in one of the arms" % space)
            continue
        print("      space = %s ..." % space)
        out = M.misspecification_gate(
            zs, zr, groups=real.groups, classes=real.classes,
            space=space, n_null=args.n_null,
            n_window_choices=args.n_window_choices,
            alpha=args.alpha, seed=args.seed, per_group=True)
        for label, gr in out.items():
            key = "%s::%s" % (space, label)
            rec = {
                "space": gr.space, "label": gr.label,
                "n_real_rows": gr.n_real_rows, "n_groups": gr.n_groups,
                "n_sim": gr.n_sim,
                "mmd_group": gr.mmd_group, "p_group": gr.p_group,
                "reject_fraction": gr.reject_fraction,
                "mmd_iid": gr.mmd_iid, "p_iid": gr.p_iid,
                "alpha": gr.alpha, "passes": bool(gr.passes),
                "notes": list(gr.notes),
                "per_group_names": list(gr.per_group_names),
            }
            if gr.per_group_p.size:
                arrays["%s::per_group_p" % key] = np.asarray(gr.per_group_p)
                try:
                    holm, bh = gr.family()
                    rec["holm_rejected"] = _jsonable(
                        getattr(holm, "rejected", None))
                    rec["bh_rejected"] = _jsonable(
                        getattr(bh, "rejected", None))
                except Exception as exc:                       # noqa: BLE001
                    rec["family_error"] = str(exc)
            results[key] = rec
            print("        %-22s p_group=%.4g  p_iid=%.4g  reject_frac=%.2f  %s"
                  % (label, gr.p_group, gr.p_iid, gr.reject_fraction,
                     "PASS" if gr.passes else "FAIL"))

    mde = {}
    if not args.skip_mde:
        print("[5/5] minimum detectable shift (power is set by the number of "
              "RECORDINGS, not windows)")
        for space in spaces:
            zs = sim.z if space == "z" else sim.zraw
            if zs is None:
                continue
            delta, rates = M.minimum_detectable_shift(
                zs, n_groups=n_groups, alpha=args.alpha, seed=args.seed)
            mde[space] = {"mde": None if delta is None else float(delta),
                          "rates": _jsonable(rates)}
            arrays["%s::mde_rates" % space] = np.asarray(rates)
            print("      %-5s MDE = %s   rates = %s"
                  % (space, delta, np.asarray(rates).tolist()))
    else:
        print("[5/5] MDE skipped")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)

    # arrays the figures need, so plotting never re-runs the gate
    arrays["sim_z"] = sim.z
    arrays["real_z"] = real.z
    if sim.zraw is not None:
        arrays["sim_zraw"] = sim.zraw
    if real.zraw is not None:
        arrays["real_zraw"] = real.zraw
    arrays["real_groups"] = np.asarray([str(g) for g in real.groups])
    arrays["real_classes"] = np.asarray([str(c) for c in real.classes])
    arrays["sim_spectrum"] = diag["sim_spectrum"]
    arrays["real_spectrum"] = diag["real_spectrum"]
    np.savez_compressed(args.out + "_arrays.npz", **arrays)

    doc = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "args": {k: v for k, v in vars(args).items()},
        "real": {"path": args.real, "n_rows": real.n, "n_groups": n_groups,
                 "E": real.E},
        "sim": {"meta": _jsonable(sim.meta),
                "n_rows_used": sim.n,
                "topology_diversity": _jsonable(topo)},
        "embedding": {"sim_r_eff": diag["sim_r_eff"],
                      "real_r_eff": diag["real_r_eff"],
                      "note": "A collapsed embedding makes a PASS weak "
                              "evidence while a REJECTION stays decisive."},
        "gate": _jsonable(results),
        "mde": _jsonable(mde),
        "caveats": [
            "misspecification_gate treats z_sim as i.i.d.; simulated rows "
            "sharing a topology are not independent. Distinct topology draws "
            "are reported under sim.topology_diversity.",
            "The gate subsamples the reference cloud to "
            "n_ref = min(n_sim // 2, 800), so n_sim above the few-thousand "
            "range changes which rows are drawn, not how many enter the "
            "statistic.",
            "p_iid is reported for contrast only; the verdict is p_group.",
        ] + ([
            "An activity filter was applied: the gate therefore tests "
            "P_sim(. | rate in the retained band) against P_real, which is a "
            "WEAKER claim than the unconditioned prior predictive check. "
            "State the band alongside any verdict."
        ] if sim.meta.get("activity_filter") else []),
    }
    with open(args.out + "_results.json", "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)

    print("")
    print("wrote %s_results.json" % args.out)
    print("wrote %s_arrays.npz" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
