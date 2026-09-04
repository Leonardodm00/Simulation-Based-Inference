#!/usr/bin/env python3
"""Aggregate Stage 3 runs into the comparison tables (plan v0.6, Stage 3).

Reads every `<arm>_seed<k>.json` and `<arm>_seed<k>_perrow.npz` in a directory
and writes one Markdown report.

The paired interval is NOT computed here
----------------------------------------
`bootstrap_paired.paired_bootstrap` is the delivered, verified module for that
(S2.4a), and it is imported, never reimplemented: a second implementation of an
interval estimator is a second thing to be wrong, and the two would be compared
against each other rather than against the truth. If the module is not
importable this script says so, prints the point estimates WITHOUT intervals,
and marks the table as incomplete -- an interval that silently degrades to a
naive one is worse than no interval, because the S2.4a measurement was that a
naive 95% interval covered 57% of the time and nothing in its output said so.

Verified against the branch on 2026-09-02: `bootstrap_paired.py` is NOT present
on `feat/misspec-gate`, so it has to be delivered before this table can be
completed.

Pure ASCII, LF only.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_runs(out_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_seed*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        stem = path[:-len(".json")]
        pr = stem + "_perrow.npz"
        if os.path.isfile(pr):
            with np.load(pr) as z:
                # PRIMARY endpoint first (D10). Falls back to the simulated
                # split only when arm R was absent, and the table says which.
                if "nll_pseudo_real" in z.files:
                    rec["_nll"] = z["nll_pseudo_real"]
                    rec["_group"] = z["group_pseudo_real"]
                    rec["_endpoint"] = "pseudo-real"
                else:
                    rec["_nll"] = z["nll"]
                    rec["_group"] = z["group"]
                    rec["_endpoint"] = "simulated"
        runs.append(rec)
    return runs


def try_import_bootstrap(extra_dir=None):
    for d in ([extra_dir] if extra_dir else []) + \
             [os.environ.get("SBI_HPC_DIR"), os.getcwd()]:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import bootstrap_paired
        return bootstrap_paired, None
    except ImportError as exc:
        return None, str(exc)


def fmt(v, prec=4):
    if v is None:
        return "-"
    if isinstance(v, float) and not np.isfinite(v):
        return "nan"
    if isinstance(v, float):
        return ("%." + str(prec) + "f") % v
    return str(v)


def arm_table(runs):
    lines = ["| arm | seed | L (primary) | Delta_hat (primary) | L (sim) | "
             "r_eff | ARI | T | p_eff (rep) | p_eff (spec) | T<=0 | rho_grad |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["arm"], r["seed"])):
        rep = r.get("replicate", {})
        rho = r["history"][-1].get("rho_grad") if r.get("history") else None
        prim_L = r.get("L_pseudo_real")
        prim_d = r.get("delta_hat_pseudo_real")
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (r["arm"], r["seed"],
                        fmt(prim_L if prim_L is not None else r["L"]),
                        fmt(prim_d if prim_d is not None else r["delta_hat"]),
                        fmt(r["L"]),
                        fmt(r["r_eff"], 3), fmt(r.get("cluster_ari"), 3),
                        fmt(rep.get("T_mean"), 3),
                        fmt(rep.get("p_eff_mean"), 3),
                        fmt(r.get("p_eff_spectrum"), 3),
                        fmt(rep.get("n_T_nonpositive")), fmt(rho, 3)))
    return "\n".join(lines)


def _primary_L(r):
    v = r.get("L_pseudo_real")
    return r["L"] if v is None else v


def seed_table(runs):
    """Per-arm mean and SD of the primary score across seeds.

    sigma_seed is the across-seed SD and is one of the TWO clauses of the S2.4
    decision rule: "A0 degrades" needs BOTH |D| > 2 sigma_seed AND a
    cluster-bootstrap CI excluding 0. The bootstrap is conditional on the
    trained networks and varies only the evaluation rows; sigma_seed is
    training stochasticity. Neither substitutes for the other.
    """
    by = {}
    for r in runs:
        by.setdefault(r["arm"], []).append(_primary_L(r))
    lines = ["| arm | n_seed | mean L (primary) | sigma_seed | 2 sigma_seed |",
             "|---|---|---|---|---|"]
    sig = {}
    for arm, vals in sorted(by.items()):
        v = np.asarray(vals, dtype=np.float64)
        sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        sig[arm] = sd
        lines.append("| %s | %d | %s | %s | %s |"
                     % (arm, v.size, fmt(float(v.mean())), fmt(sd),
                        fmt(2 * sd if np.isfinite(sd) else sd)))
    return "\n".join(lines), sig


def pair_table(runs, bp, delta_min):
    """Paired comparison of every arm against A1, seed by seed.

    Each arm's seed k is paired with A1's seed k on the SAME report rows
    (same split_hash), giving one D per seed. The table reports the mean D
    over seeds, the pooled cluster-bootstrap interval when bootstrap_paired is
    available, and the two-clause verdict.

    [CORRECTION] The first version paired seed 0 only and ignored every other
    run, so sigma_seed was never formed and the second clause of the S2.4 rule
    could not be applied at all.
    """
    by = {}
    for r in runs:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    if "A1" not in by:
        return "A1 absent; no paired comparison possible.", []
    ref_by_seed = by["A1"]
    if not any("_nll" in r for r in ref_by_seed.values()):
        return "per-row NLL not persisted; rerun with the current runner.", []

    sig_a1 = np.std([_primary_L(r) for r in ref_by_seed.values()], ddof=1) \
        if len(ref_by_seed) > 1 else float("nan")

    lines = ["| arm vs A1 | seeds paired | mean D (nats/row) | sd D | "
             "CI (all, pooled) | rho(d) | verdict |",
             "|---|---|---|---|---|---|---|"]
    notes = []
    for arm, per_seed in sorted(by.items()):
        if arm == "A1":
            continue
        ds, d_rows, groups = [], [], []
        for seed, r in sorted(per_seed.items()):
            ref = ref_by_seed.get(seed)
            if ref is None or "_nll" not in r or "_nll" not in ref:
                continue
            if r["split_hash"] != ref["split_hash"] \
                    or r["_nll"].shape != ref["_nll"].shape:
                notes.append("%s seed %d: split differs from A1 seed %d; the "
                             "pairing of S2.4 does not hold" % (arm, seed, seed))
                continue
            d = r["_nll"] - ref["_nll"]
            ds.append(float(np.mean(d)))
            d_rows.append(d)
            groups.append(ref["_group"])
        if not ds:
            notes.append("%s: no seed could be paired with A1" % arm)
            continue
        D = float(np.mean(ds))
        sdD = float(np.std(ds, ddof=1)) if len(ds) > 1 else float("nan")
        sig = np.std([_primary_L(r) for r in per_seed.values()], ddof=1) \
            if len(per_seed) > 1 else float("nan")
        sigma_seed = np.nanmax([sig, sig_a1])
        clause_seed = (np.isfinite(sigma_seed) and abs(D) > 2 * sigma_seed)

        lo = hi = rho = None
        if bp is not None:
            try:
                d_all = np.concatenate(d_rows)
                g_all = np.concatenate([np.asarray(g) + 10 ** 6 * k
                                        for k, g in enumerate(groups)])
                res = bp.paired_bootstrap(d_all, groups=g_all, within="all")
                icc = bp.intraclass_correlation(d_all, g_all)
                ci = (res.get("ci") if isinstance(res, dict)
                      else getattr(res, "ci", None))
                lo, hi = (ci if ci is not None and len(ci) == 2
                          else (None, None))
                rho = getattr(icc, "rho", icc)
            except Exception as exc:                   # noqa: BLE001
                notes.append("%s: bootstrap_paired failed (%s: %s); its API "
                             "differs from plan S3" % (arm, type(exc).__name__,
                                                      exc))
        clause_ci = (lo is not None and hi is not None
                     and (lo > 0 or hi < 0))
        if bp is None or lo is None:
            verdict = ("seed clause %s; CI unavailable -> NO CALL"
                       % ("met" if clause_seed else "not met"))
        elif clause_seed and clause_ci:
            verdict = "A1 better" if D > 0 else "A1 worse"
        else:
            verdict = ("no call (seed %s, CI %s)"
                       % ("met" if clause_seed else "not met",
                          "excludes 0" if clause_ci else "includes 0"))
        lines.append("| %s | %d | %s | %s | [%s, %s] | %s | %s |"
                     % (arm, len(ds), fmt(D), fmt(sdD), fmt(lo, 3),
                        fmt(hi, 3), fmt(rho, 3), verdict))
    return "\n".join(lines), notes


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", default=None, help="output .md; default stdout")
    p.add_argument("--bootstrap-dir", default=None,
                   help="directory containing bootstrap_paired.py")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit("no runs found in %r" % args.runs_dir)
    bp, bp_err = try_import_bootstrap(args.bootstrap_dir)

    def _gain(r):
        v = r.get("delta_hat_pseudo_real")
        return r["delta_hat"] if v is None else v

    shuffled = [r for r in runs if r["arm"] == "shuffled"]
    delta_min = max((_gain(r) for r in shuffled), default=None)

    endpoints = sorted({r.get("_endpoint", "?") for r in runs})
    out = ["# Stage 3 -- arm comparison", "",
           "Runs: %d, arms: %s" % (len(runs),
                                   ", ".join(sorted({r["arm"] for r in runs}))),
           "",
           "Primary endpoint (D10): **%s** held-out NLL." % "/".join(endpoints),
           ("" if endpoints == ["pseudo-real"] else
            "WARNING: at least one arm was scored on the SIMULATED split "
            "because arm R was absent. P6 predicts the two rankings need not "
            "agree, so a table mixing them cannot be read as one comparison."),
           "", "## Per-arm diagnostics", "", arm_table(runs), ""]

    out += ["## G1: the shuffled control", ""]
    # [CHANGE 2026-09-04, HANDOFF_DELTA_MIN_PER_CONFIG_v1 S2/S3 Tier-1] The
    # single bank-level floor is a PROVISIONAL screen, not a verdict. The
    # shuffled arm runs A1's exact recipe with only the training pairing
    # permuted (arm_config in run_joint_arms.py), and the null gain
    # distribution depends on capacity and recipe, so for every other arm
    # this floor is mis-calibrated in a direction that cannot be signed.
    # The authoritative delta_min is per-config, measured at the tuning
    # stack's finalist tier; Stage 3's binding comparison stays the paired
    # S2.4 rule against A1. Regression: R5h in smoke_test_joint_arms.py.
    if delta_min is None:
        out.append("No `shuffled` arm was run, so delta_min is unknown and no "
                   "arm can be said to have learned anything. Run it.")
    else:
        out.append("delta_min = %s nats/row: max gain over the %d "
                   "shuffled-control seed(s). PROVISIONAL screen "
                   "(HANDOFF_DELTA_MIN_PER_CONFIG_v1), not a gate verdict: "
                   "the shuffled arm runs A1's recipe with only the training "
                   "pairing permuted, so this floor is recipe-matched to A1 "
                   "alone. The null gain distribution depends on capacity "
                   "and recipe, so for arms that freeze the encoder (A0, "
                   "A0s, A_ref), add loss terms (A2, A2s, A3, A5) or "
                   "warm-start (A3) the same floor is mis-calibrated in a "
                   "direction that cannot be signed. Read the table as: for "
                   "A1, a gain at or below the floor means nothing was "
                   "learned; for the other arms it is indicative only, in "
                   "both directions. The authoritative delta_min is "
                   "per-config at the finalist tier. A prior-equal "
                   "posterior is perfectly calibrated, so G2 passing means "
                   "nothing on its own." % (fmt(delta_min), len(shuffled)))
        out.append("")
        out.append("| arm | Delta_hat | above delta_min (provisional) |")
        out.append("|---|---|---|")
        for r in sorted(runs, key=lambda r: r["arm"]):
            g = _gain(r)
            out.append("| %s | %s | %s |" % (r["arm"], fmt(g),
                                             "yes" if g > delta_min else "NO"))
    out.append("")

    out += ["## Across-seed spread (sigma_seed)", ""]
    st, _ = seed_table(runs)
    out += [st, "", "With fewer than 2 seeds per arm sigma_seed is undefined "
            "and the S2.4 rule cannot be applied; the plan asks for "
            "n_seed >= 5.", ""]

    out += ["## Paired comparison against A1 (S2.4a)", ""]
    if bp is None:
        out.append("**Incomplete.** `bootstrap_paired` is not importable "
                   "(%s), so the intervals below are missing. The point "
                   "estimates are shown; do NOT read them as a decision. The "
                   "decision rule of S2.4 needs both the interval AND "
                   "2*sigma_seed, and neither is available here." % bp_err)
        out.append("")
    tbl, notes = pair_table(runs, bp, delta_min)
    out.append(tbl)
    if notes:
        out += ["", "Notes:"] + ["- " + n for n in notes]
    out.append("")

    out += ["## P15: does moving to the full parameter space cost anything?",
            ""]
    for r in sorted(runs, key=lambda r: r["arm"]):
        rep = r.get("replicate")
        if rep is None:
            continue
        out.append("- %s: p_eff from the posterior = %s, from the spectrum = "
                   "%s, gap = %s (%s)"
                   % (r["arm"], fmt(rep["p_eff_mean"], 3),
                      fmt(r.get("p_eff_spectrum"), 3), fmt(rep.get("p_eff_gap"), 3),
                      r.get("spectrum_note", "")))
    out += ["", "The two routes must agree (J13). A persistent gap means the "
            "target the loss is trained against is not the quantity the "
            "diagnostic reports, and S2.5b's identification is broken.", ""]

    out += ["## P16: which directions dominate the disagreement?", ""]
    for r in sorted(runs, key=lambda r: r["arm"]):
        rep = r.get("replicate")
        if rep is None or not rep.get("per_direction_mean"):
            continue
        pd = np.asarray(rep["per_direction_mean"])
        order = np.argsort(-pd)[:5]
        out.append("- %s: top directions %s with normalised contributions %s"
                   % (r["arm"], order.tolist(),
                      np.round(pd[order], 3).tolist()))
    out += ["", "Each term is normalised by its own null share (1 - c_j), so "
            "the 26 directions are comparable. One direction dominating across "
            "donors is evidence against H_0(i) -- the shared-theta premise -- "
            "not against calibration.", ""]

    text = "\n".join(out)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print("written: %s" % args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
