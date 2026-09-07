#!/usr/bin/env python3
"""Run Stage 3b: decompose A0's deficit and probe the domain effect directly.

Plan v0.6, Stage 3b. Consumes the Stage 3 run directory -- no training, no new
simulation. Two independent lines of evidence:

  1. eq. (9), the domain/objective decomposition, from the per-row endpoints
     already persisted by `run_joint_arms.py`;
  2. the two direct probes, from an existing checkpoint plus the two banks.

They can disagree, and that is informative rather than a problem: a large
domain TERM with no off-support SIGNATURE points at O2 (the prior predictive
genuinely varying less than the cohort) rather than at the encoder.

Expected output of a correct run: a table with `residual` at 1e-16 or smaller
on every seed. A non-zero residual means the arms were not scored on the same
rows and nothing else in the report can be read.

Pure ASCII, LF only.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           os.path.join(_HERE, "..", "stage3"), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latent_bank import concat_shards                        # noqa: E402
from domain_objective import (REQUIRED_ARMS, decompose, decomposition_table,  # noqa: E402
                              joint_side, per_axis_contraction_split,
                              primary_score, sigma_seed, verdict)
from encoder_probes import (embedding_direction_probe,        # noqa: E402
                            flag_degenerate_layers,
                            layer_activation_stats)


def load_runs(runs_dir):
    """Reuse the Stage 3 loader so the endpoint preference is identical."""
    from report_joint_arms import load_runs as _lr
    return _lr(runs_dir)


def try_import_bootstrap(extra_dir=None):
    for d in ([extra_dir] if extra_dir else []) + [os.environ.get("SBI_HPC_DIR")]:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import bootstrap_paired
        return bootstrap_paired, None
    except ImportError as exc:
        return None, str(exc)


def rebuild_encoder(ckpt_path, dsn_main_dir=None):
    """Rebuild the encoder from a Stage 3 checkpoint and load its weights.

    The checkpoint stores the encoder state separately (`encoder_state`) and
    the meta records W, embedding_size and whether the arm used the fixed
    summary, which is exactly what is needed to reconstruct the module without
    the run's argv.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    W = int(meta.get("W", 0))
    E = int(meta.get("embedding_size", 0))
    if not W or not E:
        raise SystemExit(
            "checkpoint %s has no W/embedding_size in its meta. It predates "
            "the Stage 3b metadata change; rerun that arm." % ckpt_path)
    if meta.get("fixed_summary"):
        from run_joint_arms import FixedStatsSummary
        enc = FixedStatsSummary()
    else:
        from run_joint_arms import make_backbone
        enc, _ = make_backbone(W, E, dsn_main_dir, int(meta.get("seed", 0)))
    enc.load_state_dict(ck["encoder_state"])
    enc.eval()
    return enc, meta


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True,
                   help="the Stage 3 output directory")
    p.add_argument("--sim-shards", default=None,
                   help="glob for arm-S shards; required for the probes")
    p.add_argument("--real-shards", default=None,
                   help="glob for arm-R shards; required for the probes")
    p.add_argument("--probe-ckpt", default=None,
                   help="checkpoint to probe; default <runs-dir>/A0_seed0_ckpt.pt")
    p.add_argument("--max-probe-rows", type=int, default=512)
    p.add_argument("--bootstrap-dir", default=None)
    p.add_argument("--dsn-main-dir", default=None)
    p.add_argument("--out", default=None, help="output .md; default stdout")
    p.add_argument("--out-json", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit("no runs found in %r" % args.runs_dir)

    by = {}
    for r in runs:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    seeds = sorted({r["seed"] for r in runs})
    missing = [a for a in REQUIRED_ARMS if a not in by]

    probe_ckpt = args.probe_ckpt or os.path.join(args.runs_dir,
                                                 "A0_seed0_ckpt.pt")

    if args.dry_run:
        print("DRY RUN -- nothing computed")
        print("  runs dir         : %s (%d runs)" % (args.runs_dir, len(runs)))
        print("  arms present     : %s" % ", ".join(sorted(by)))
        print("  seeds            : %s" % seeds)
        print("  eq. (9) possible : %s" % ("no, missing " + ", ".join(missing)
                                           if missing else "yes"))
        print("  probe checkpoint : %s (%s)"
              % (probe_ckpt, "found" if os.path.isfile(probe_ckpt)
                 else "MISSING"))
        print("  probes possible  : %s"
              % ("yes" if (args.sim_shards and args.real_shards
                           and os.path.isfile(probe_ckpt)) else "no"))
        return 0

    bp, bp_err = try_import_bootstrap(args.bootstrap_dir)
    out = ["# Stage 3b -- decomposing A0's deficit", ""]
    payload = {"runs_dir": args.runs_dir, "seeds": seeds,
               "arms": sorted(by)}

    # ---- eq. (9) ----------------------------------------------------------
    out += ["## The decomposition, eq. (9)", ""]
    if missing:
        out += ["**Not computable.** Stage 3b needs %s; missing: %s. Without "
                "A0s the total A0 - A1 cannot be split, and reading it as "
                "either effect alone is exactly the mistake this stage "
                "exists to prevent."
                % (", ".join(REQUIRED_ARMS), ", ".join(missing)), ""]
        payload["decomposition"] = {"error": "missing arms: %s" % missing}
    else:
        results = [decompose(by, s, bp=bp) for s in seeds]
        sig = {a: sigma_seed(list(by[a].values())) for a in REQUIRED_ARMS}
        payload["decomposition"] = results
        payload["sigma_seed"] = sig

        bad = [r for r in results if "error" not in r
               and abs(r["identity_residual"]) > 1e-9]
        out += [decomposition_table(results, sig), ""]
        if bad:
            out += ["**The identity does not hold on %d seed(s).** eq. (9) is "
                    "algebraic, so a non-zero residual means the three arms "
                    "were not scored on the same rows. Nothing above can be "
                    "read." % len(bad), ""]
        if bp is None:
            out += ["Intervals absent: `bootstrap_paired` is not importable "
                    "(%s). Only the sigma_seed clause of the S2.4 rule can be "
                    "applied." % bp_err, ""]
        out += ["**Reading:**", ""] + verdict(results, sig) + [""]

        js = [joint_side(by, s) for s in seeds]
        js = [j for j in js if j is not None]
        if js:
            payload["joint_side"] = js
            vals = [j["D_joint_domain"] for j in js if "error" not in j]
            if vals:
                out += ["Joint-side analogue L(A2) - L(A2s): mean %.4f "
                        "nats/row over %d seed(s). Same sign as D_domain "
                        "means the domain effect survives joint training."
                        % (float(np.mean(vals)), len(vals)), ""]

        pax = per_axis_contraction_split(by, seeds[0])
        if pax is not None:
            payload["per_axis"] = pax
            tot = np.asarray(pax["total"])
            worst = np.argsort(-np.abs(tot))[:5]
            out += ["### Per-axis split (contraction, not nats)", "",
                    "Deviation from the plan's wording, stated in "
                    "`domain_objective.py`: a per-axis NLL needs the flow's "
                    "marginals, which do not exist in closed form. This is "
                    "the same split applied to per-axis CONTRACTION.", "",
                    "| axis | total | domain | objective |", "|---|---|---|---|"]
            for k in worst:
                out.append("| %s | %.4f | %.4f | %.4f |"
                           % (pax["axes"][k], pax["total"][k],
                              pax["domain"][k], pax["objective"][k]))
            out.append("")

    # ---- the two probes ---------------------------------------------------
    out += ["## Direct probes of the domain effect", ""]
    if not (args.sim_shards and args.real_shards):
        out += ["Skipped: --sim-shards and --real-shards are needed.", ""]
    elif not os.path.isfile(probe_ckpt):
        out += ["Skipped: no checkpoint at %s." % probe_ckpt, ""]
    else:
        import glob
        sim, _ = concat_shards(sorted(glob.glob(args.sim_shards)))
        real, _ = concat_shards(sorted(glob.glob(args.real_shards)))
        x_sim = torch.as_tensor(np.asarray(sim["x"]), dtype=torch.float32)
        x_real = torch.as_tensor(np.asarray(real["x"]), dtype=torch.float32)
        enc, meta = rebuild_encoder(probe_ckpt, args.dsn_main_dir)

        stats = layer_activation_stats(enc, x_sim, x_real,
                                       max_rows=args.max_probe_rows)
        flags = flag_degenerate_layers(stats)
        payload["activation_stats"] = stats
        payload["activation_flags"] = flags

        out += ["### Probe 1 -- per-layer activations (%s, seed %s)"
                % (meta.get("arm"), meta.get("seed")), "",
                "| layer | dead frac sim | dead frac real | mean shift (z) | "
                "std real |", "|---|---|---|---|---|"]
        for name in list(stats)[:12]:
            rec = stats[name]
            out.append("| %s | %.3f | %.3f | %s | %.4g |"
                       % (name, rec["sim"]["dead_frac"],
                          rec["real"]["dead_frac"],
                          "%.2f" % rec["mean_shift_z"]
                          if np.isfinite(rec["mean_shift_z"]) else "n/a",
                          rec["real"]["std"]))
        if len(stats) > 12:
            out.append("| ... %d more layers | | | | |" % (len(stats) - 12))
        out.append("")
        if flags:
            out += ["Flagged: " + ", ".join(
                "%s (%s)" % (f["layer"],
                             "dead" if f["dead"] else "shifted")
                for f in flags[:6]), ""]
        else:
            out += ["No layer flagged as dead or strongly shifted on "
                    "simulated input.", ""]

        with torch.no_grad():
            z_sim = enc(x_sim[:args.max_probe_rows]).cpu().numpy()
            z_real = enc(x_real[:args.max_probe_rows]).cpu().numpy()
        dirp = embedding_direction_probe(z_sim, z_real)
        payload["direction_probe"] = dirp

        out += ["### Probe 2 -- the real cloud's leading direction", "",
                "r_eff: real %.3f, simulated %.3f (of E = %s)"
                % (dirp["r_eff_real"], dirp["r_eff_sim"],
                   meta.get("embedding_size")), ""]
        for d in dirp["directions"]:
            out += ["- direction %d (%.1f%% of real variance): %.1f%% of "
                    "simulated projections below the real range, %.1f%% "
                    "above; KS = %.3f."
                    % (d["index"], 100 * (d["variance_share"] or 0.0),
                       100 * d["sim_frac_below_real_range"],
                       100 * d["sim_frac_above_real_range"], d["ks"]),
                    "  - %s" % d["reading"]]
        out.append("")

    text = "\n".join(out)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print("written: %s" % args.out)
    else:
        print(text)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, default=str)
        print("written: %s" % args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
