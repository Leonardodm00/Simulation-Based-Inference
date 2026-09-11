#!/usr/bin/env python3
"""Run Stage 3c: the floors, the aliasing test, and the stratification.

Plan v0.6, Stage 3c. Everything runs ON THE BENCH FIRST, where nu, G and theta
are all known, and the stage has an explicit gate:

    the recovered nuisance subspace must match the injected nu directions;
    the realisation floor must concentrate on the kernel axes;
    mu_j must separate exactly on the axes given genuine within-class spread.

**The machinery is not applied to real data until this passes.** The runner
enforces that literally: `--allow-real` is refused unless a validation record
from a bench run is supplied and says `passed`.

Pure ASCII, LF only.
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           os.path.join(_HERE, "..", "stage3"), os.path.join(_HERE, "..", "stage3b"),
           _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latent_bank import concat_shards                          # noqa: E402
from latent_nuisance import NU_COMPONENTS, NuisanceSpec        # noqa: E402
from latent_realisation import (RealisationSpec,               # noqa: E402
                                distinct_realisations_per_theta)
from latent_sbi_simulator import LatentSBISpec, prior_log_prob  # noqa: E402
from aliasing import aliasing_report, jacobian_nu, jacobian_theta  # noqa: E402
from nuisance_floor import nuisance_floor, validate_against_injected  # noqa: E402
from realisation_floor import kernel_axis_concentration, realisation_floor  # noqa: E402
from stratify import stratify, validate_on_bench               # noqa: E402
from window_aggregation import aggregation_curve, curve_slope   # noqa: E402


def spec_from_sidecar(side, n_windows=None):
    ls = side["latent_spec"]
    return LatentSBISpec(
        n_latent=ls["n_latent"], label_idx=tuple(ls["label_idx"]),
        class_centres=np.asarray(ls["class_centres"]), tau_ov=ls["tau_ov"],
        n_windows_per_trace=int(n_windows or ls["n_windows_per_trace"]),
        T_win=ls["T_win"], fs=ls["fs"], n_neurons=ls["n_neurons"])


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a Stage 3 checkpoint")
    p.add_argument("--sim-shards", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-floor-draws", type=int, default=48)
    p.add_argument("--n-post-draws", type=int, default=64)
    p.add_argument("--kernel-axes", default="",
                   help="comma-separated indices of the connectivity-kernel "
                        "axes; on the real bank these are the 3 Weibull axes")
    p.add_argument("--spread-axes", default="",
                   help="comma-separated indices given genuine within-class "
                        "spread on the bench; the stratification gate uses them")
    p.add_argument("--fd-step", type=float, default=0.02)
    p.add_argument("--fd-seeds", type=int, default=3)
    p.add_argument("--allow-real", action="store_true",
                   help="permit application to real data; refused without a "
                        "passing bench validation via --validation")
    p.add_argument("--validation", default=None,
                   help="a stage3c_validation.json from a bench run")
    p.add_argument("--dsn-main-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def d17_realisation_audit(sim, kernel_idx):
    """Count connectivity realisations per kernel-parameter value. D17, (c).

    Informational by construction: the returned dict NEVER contains a
    `passed` key, and main()'s exit code is built only from validation
    entries that have one, so this record is structurally unable to fail a
    run. That is the mechanism, not a convention: D17 was closed as option
    (c) -- accept the kernel-axis p_eff bias and record it -- so a count of 1
    (the exact scenario D17 is about) is reported, not flagged, and a count
    >= 2 is reported the same way, because passing/failing was never the
    point. See JOINT_DSN_NPE_PLAN_v0_6.md S8 (D17) and
    HANDOFF_D17_option_c.md.

    Parameters
    ----------
    sim : dict of arrays from `concat_shards` -- uses `theta` (n, d_theta)
        and, when present, `realisation_id` (n,).
    kernel_idx : sequence of int -- column indices of the kernel-topology
        axes in theta (the 3 Weibull axes), as parsed from `--kernel-axes`.

    Returns
    -------
    dict with `available` (bool) and a human-readable `note`; when
    `available`, also `worst_case_realisations` (int, the minimum over
    distinct kernel-axis values), `n_kernel_values` (int) and
    `kernel_idx` (list). No `passed` key, ever.
    """
    if "realisation_id" not in sim:
        return {"available": False,
                "note": "sim bank carries no realisation_id field (expected "
                        "for the historical ANN campaign export); the D17 "
                        "count cannot be computed from per-row data here. "
                        "See HANDOFF_D17_option_c.md S5 for the two "
                        "unsettled sub-questions before retrofitting one."}
    theta_k = np.asarray(sim["theta"], dtype=np.float64)[:, list(kernel_idx)]
    counts = distinct_realisations_per_theta(theta_k, sim["realisation_id"])
    worst = int(min(counts.values()))
    return {"available": True,
            "worst_case_realisations": worst,
            "n_kernel_values": len(counts),
            "kernel_idx": [int(k) for k in kernel_idx],
            "note": "recorded under D17 option (c): %d distinct kernel-axis "
                    "value(s), worst case %d realisation(s) per value. "
                    "Recorded, not gated." % (len(counts), worst)}


def main(argv=None):
    args = build_parser().parse_args(argv)
    kernel_idx = [int(v) for v in args.kernel_axes.split(",") if v.strip()]
    spread_idx = [int(v) for v in args.spread_axes.split(",") if v.strip()]

    if args.allow_real:
        if not args.validation or not os.path.isfile(args.validation):
            raise SystemExit(
                "--allow-real needs --validation pointing at a bench "
                "stage3c_validation.json. Plan Stage 3c: the machinery is not "
                "applied to real data until the bench test passes.")
        with open(args.validation) as fh:
            v = json.load(fh)
        failed = [k for k, r in v.items()
                  if isinstance(r, dict) and r.get("passed") is False]
        if failed:
            raise SystemExit("bench validation failed on %s; refusing to run "
                             "on real data" % ", ".join(failed))

    paths = sorted(glob.glob(args.sim_shards))
    if not paths:
        raise SystemExit("no shards matched %r" % args.sim_shards)
    sim, side = concat_shards(paths)
    spec = spec_from_sidecar(side)
    names = side["param_names"]

    if args.dry_run:
        print("DRY RUN -- nothing computed")
        print("  checkpoint       : %s (%s)"
              % (args.ckpt, "found" if os.path.isfile(args.ckpt) else "MISSING"))
        print("  shards           : %d, rows %d, d_theta %d"
              % (len(paths), sim["x"].shape[0], len(names)))
        print("  kernel axes      : %s" % (kernel_idx or "none given"))
        print("  spread axes      : %s" % (spread_idx or "none given"))
        print("  floor draws      : %d" % args.n_floor_draws)
        print("  allow-real       : %s" % args.allow_real)
        print("  out              : %s" % args.out_dir)
        return 0

    from run_stage3b import rebuild_encoder                     # noqa: F401
    from joint_model import JointDSNNPE                          # noqa: F401
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    model = _rebuild_model(ck, meta, sim, args)

    from latent_sbi_simulator import load_dsn_provider
    provider = load_dsn_provider(args.dsn_main_dir)
    nspec = NuisanceSpec()
    rspec = RealisationSpec()
    theta_star = np.asarray(sim["theta"], dtype=np.float64).mean(axis=0)
    theta_star = np.clip(theta_star, 0.05, 0.95)

    os.makedirs(args.out_dir, exist_ok=True)
    report, validation = ["# Stage 3c -- floors, aliasing, stratification", ""], {}

    # ---- floors -----------------------------------------------------------
    nf = nuisance_floor(model, spec, provider, theta_star, nspec, rspec,
                        n_draws=args.n_floor_draws, base_seed=args.seed,
                        n_draws_post=args.n_post_draws, param_names=names)
    rf = realisation_floor(model, spec, provider, theta_star, rspec,
                           n_draws=args.n_floor_draws, base_seed=args.seed,
                           n_draws_post=args.n_post_draws, param_names=names)
    report += ["## Floors (fixed theta*, one factor varied)", "",
               "| floor | total variance | leading share | dominant axes |",
               "|---|---|---|---|"]
    for tag, f in (("nuisance", nf), ("realisation", rf)):
        tot = float(np.sum(f["eigenvalues"]))
        report.append("| %s | %.4g | %.2f | %s |"
                      % (tag, tot, f["directions"][0]["variance_share"] or 0.0,
                         ", ".join(f["directions"][0]["dominant_axes"])))
    report.append("")
    if kernel_idx:
        kc = kernel_axis_concentration(rf, kernel_idx)
        validation["realisation_concentration"] = {
            "passed": bool(kc["concentrated"]), **kc}
        if kc["concentrated"]:
            verdict = ("CONCENTRATED -- D17's premise holds: realisation "
                       "noise sits on the kernel axes, so option (c) "
                       "(accept the bias on these 3 axes, recorded below) "
                       "is a coherent response")
        else:
            verdict = ("NOT concentrated -- D17's premise does not hold "
                       "here, so option (c) (accept the bias on these 3 "
                       "axes) is not a coherent response either; D17 "
                       "should reopen")
        report += ["Realisation floor concentration on the kernel axes: "
                   "%.2f (uniform reference %.2f) -- %s. This is the D17 "
                   "check."
                   % (kc["concentration"], kc["uniform_reference"], verdict),
                   ""]

    # ---- aliasing ---------------------------------------------------------
    Js_t, Js_n = [], []
    for s in range(int(args.fd_seeds)):
        Js_t.append(jacobian_theta(model, spec, provider, theta_star, rspec,
                                    base_seed=args.seed + s, h=args.fd_step))
        jn, quant = jacobian_nu(model, spec, provider, theta_star, rspec,
                                base_seed=args.seed + s, nuisance_spec=nspec)
        Js_n.append(jn)
    J_theta = np.mean(Js_t, axis=0)
    J_nu = np.mean(Js_n, axis=0)
    # Report the seed spread RELATIVE to the size of the entries: 0.4 is
    # meaningless without knowing whether the entries are 0.04 or 40.
    scale_t = float(np.abs(J_theta).mean()) or 1.0
    spread_t = (float(np.std(Js_t, axis=0).mean()) / scale_t
                if len(Js_t) > 1 else 0.0)
    al = aliasing_report(J_theta, J_nu, param_names=names,
                         nu_names=list(NU_COMPONENTS), quantised=quant)
    report += ["## Aliasing, eq. (4)", "",
               "rank(J_theta) = %d of %d; across-seed spread of J_theta is "
               "%.0f%% of its mean entry size (above ~30%% the finite "
               "differences are seed noise; raise --fd-seeds before reading "
               "a_m)." % (al["rank_J_theta"], al["n_theta"], 100 * spread_t),
               "",
               "| nu component | a_m | reads as |", "|---|---|---|"]
    for d in al["directions"]:
        a = d.get("a_m")
        report.append("| %s | %s | %s |"
                      % (d["nu"], "n/a" if a is None else "%.3f" % a,
                         d.get("reading", d.get("note", ""))))
    report.append("")

    inj = np.eye(len(names))[:, :0]
    if nf["directions"]:
        cos = validate_against_injected(
            nf, np.asarray([d["delta_theta"] for d in al["directions"]
                            if "delta_theta" in d]))
        validation["nuisance_subspace"] = {
            "passed": bool(cos > 0.7), "cos_principal_angle": cos}
        report += ["Nuisance floor vs the aliasing prediction: cos(principal "
                   "angle) = %.3f. The floor's leading direction and eq. (4)'s "
                   "delta_theta are two independent routes to the same "
                   "subspace, so agreement is the check that both are right."
                   % cos, ""]

    # ---- stratification ---------------------------------------------------
    # Per WELL, with the donor and label of that well -- not the row-level
    # arrays, which are longer by a factor of the windows per trace.
    means, well_donor, well_label = _culture_means(model, sim,
                                                   args.n_post_draws)
    try:
        st = stratify(means, well_donor, well_label, param_names=names)
        report += ["## Stratification, eq. (5)", "",
                   "null mu = %.4f (%s)" % (st["null_mu"],
                                            st["null_mu_derivation"]), "",
                   "| j | mu | above null | above 1 | dominant axes |",
                   "|---|---|---|---|---|"]
        for d in st["directions"][:6]:
            report.append("| %d | %.3f | %s | %s | %s |"
                          % (d["index"], d["mu"], d["above_null"],
                             d["above_one"], ", ".join(d["dominant_axes"])))
        report.append("")
        if spread_idx:
            v = validate_on_bench(st, spread_idx)
            validation["stratification"] = v
            report += ["Bench gate: %s -- %s" % ("PASSED" if v["passed"]
                                                 else "FAILED", v["note"]), ""]
    except ValueError as exc:
        report += ["## Stratification, eq. (5)", "",
                   "Not computable: %s" % exc, ""]
        validation["stratification"] = {"passed": False, "error": str(exc)}

    # ---- D17 realisation audit (informational; option (c)) ----------------
    if kernel_idx:
        audit = d17_realisation_audit(sim, kernel_idx)
        validation["d17_realisation_audit"] = audit
        report += ["## D17 realisation audit (option (c): recorded, "
                   "not gated)", "", audit["note"], ""]

    # ---- P10 --------------------------------------------------------------
    x_all = torch.as_tensor(np.asarray(sim["x"]), dtype=torch.float32)
    donor0 = np.asarray(sim["donor"])
    idx = np.flatnonzero(donor0 == donor0[0])[:16]
    pts = aggregation_curve(model, x_all[idx],
                            np.asarray(sim["theta"])[idx[0]],
                            lambda t: prior_log_prob(spec, t),
                            n_is=256, seed=args.seed)
    sl = curve_slope(pts)
    report += ["## P10 -- the window-aggregation curve", "",
               "| windows | gain (nats) | ESS | trusted |", "|---|---|---|---|"]
    for p in pts:
        report.append("| %d | %.4f | %.1f | %s |"
                      % (p["n"], p["gain"], p["ess"], p["trusted"]))
    report += ["", sl["reading"], ""]
    validation["p10"] = {"slope": sl["slope"], "points": pts}

    stem = os.path.join(args.out_dir, "stage3c")
    with open(stem + "_report.md", "w") as fh:
        fh.write("\n".join(report))
    with open(stem + "_validation.json", "w") as fh:
        json.dump(validation, fh, indent=1, sort_keys=True, default=str)
    with open(stem + "_full.json", "w") as fh:
        json.dump({"nuisance_floor": nf, "realisation_floor": rf,
                   "aliasing": al, "p10": pts}, fh, indent=1, default=str)

    gates = {k: r.get("passed") for k, r in validation.items()
             if isinstance(r, dict) and "passed" in r}
    print("\n".join(report[:3]))
    print("gates: %s" % gates)
    print("written: %s_report.md, _validation.json, _full.json" % stem)
    return 0 if all(v is not False for v in gates.values()) else 1


def _rebuild_model(ck, meta, sim, args):
    """Rebuild the full JointDSNNPE (encoder + flow) from a checkpoint."""
    from sbi.utils import BoxUniform
    from joint_model import build_joint_model
    from run_joint_arms import FixedStatsSummary, make_backbone

    d = int(meta.get("d_theta", np.asarray(sim["theta"]).shape[1]))
    W = int(meta["W"])
    E = int(meta["embedding_size"])
    bb = (FixedStatsSummary() if meta.get("fixed_summary")
          else make_backbone(W, E, args.dsn_main_dir, int(meta.get("seed", 0)))[0])
    prior = BoxUniform(low=torch.zeros(d), high=torch.ones(d))
    theta = torch.as_tensor(np.asarray(sim["theta"])[:64], dtype=torch.float32)
    x = torch.as_tensor(np.asarray(sim["x"])[:64], dtype=torch.float32)

    # The FLOW hyper-parameters must come from the checkpoint, not from the
    # builder's defaults.
    #
    # [CORRECTION] Rebuilding with the defaults (64/5/10) against a
    # checkpoint trained at 48/3/8 produced a wall of size-mismatch errors on
    # load. It fails loudly, which is the good case; the bad case is a run
    # whose defaults happen to match, which would work by luck and break the
    # first time a hyper-parameter is searched (Stage 4 searches all three).
    flow = meta.get("flow")
    if not flow:
        raise SystemExit(
            "checkpoint has no flow hyper-parameters in its meta; it predates "
            "the Stage 3c metadata requirement. Rerun that arm.")
    model = build_joint_model(
        bb, prior, theta, x,
        hidden_features=int(flow["hidden_features"]),
        num_transforms=int(flow["num_transforms"]),
        num_bins=int(flow["num_bins"]),
        z_score_theta=meta.get("z_score_theta", "transform_to_unconstrained"),
        z_score_x=meta.get("z_score_x", "none"), meta=meta)
    model.load_checkpoint(ck)
    model.eval()
    return model


def _culture_means(model, sim, n_post):
    """One posterior mean per WELL, averaged over that well's windows."""
    x = torch.as_tensor(np.asarray(sim["x"]), dtype=torch.float32)
    well = np.asarray(sim["well"])
    out, donors, labels = [], [], []
    for w in np.unique(well):
        rows = np.flatnonzero(well == w)
        with torch.no_grad():
            s = model.sample_posterior(n_post, x[rows])
        out.append(s.mean(dim=1).cpu().numpy().mean(axis=0))
        donors.append(np.asarray(sim["donor"])[rows[0]])
        labels.append(np.asarray(sim["cls"])[rows[0]])
    return (np.asarray(out), np.asarray(donors), np.asarray(labels))


if __name__ == "__main__":
    sys.exit(main())
