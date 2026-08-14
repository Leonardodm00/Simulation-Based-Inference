#!/usr/bin/env python3
"""
replot.py -- regenerate figures and diagnostics from a FINISHED run.

The expensive part of diagnostics is drawing posterior samples; the rank
statistics, coverage, TARP and contraction that follow are cheap numpy. So a
completed run stores its samples in diagnostics_data.npz, and this script
recomputes everything downstream from them in seconds -- no model reload, no
re-sampling, no GPU.

Usage:
    python replot.py synthetic_trial_20260811_150000
    python replot.py <run-dir> --out-dir <somewhere-else>
    python replot.py <run-dir> --n-forms 8 --seed 1

Use it to:
  * recover figures from a run started with --no-plots,
  * redo them after changing npe_plots.py,
  * re-run the diagnostics with different settings (more bilinear test
    quantities, a different seed) without retraining anything.

It also re-prints the text report, so it doubles as a way to re-read the
verdict of an old run without digging through a job log.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_contract import Contract
from npe_diagnostics import (
    data_dependent_sbc, diagnostic_report, embedding_overlap, family_verdict,
    information_spectrum, simulation_based_calibration, tarp,
)
from npe_plots import plots_available, save_all_plots


def _get(d, key, default=None):
    if key not in d:
        return default
    v = d[key]
    return v if v.size else default


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a finished run directory")
    ap.add_argument("--out-dir", default=None,
                    help="where to write figures (default: <run-dir>/figures)")
    ap.add_argument("--n-forms", type=int, default=4,
                    help="bilinear test quantities for the data-dependent SBC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=500,
                    help="bootstrap draws for the overlap null distribution")
    args = ap.parse_args()

    npz_path = os.path.join(args.run_dir, "diagnostics_data.npz")
    if not os.path.exists(npz_path):
        has_ens = os.path.isdir(os.path.join(args.run_dir, "ensemble"))
        print("ERROR: %s not found." % npz_path)
        print()
        print("Runs made before this file was added do not have it. Nothing")
        print("is broken -- the run simply predates the feature.")
        print()
        if has_ens:
            print("This run DID save its ensemble, so it is recoverable")
            print("without retraining:")
            print()
            print("    python recover_npz.py %s" % args.run_dir)
            print("    python replot.py %s" % args.run_dir)
            print()
            print("recover_npz.py reloads the shards, reproduces the exact")
            print("calibration split from the stored seed, and re-samples")
            print("from the saved ensemble. Cost is one sampling pass.")
        else:
            print("This run has no ensemble/ directory, so it was made")
            print("without --save-model and the trained networks are gone.")
            print("The posterior samples cannot be regenerated; re-run:")
            print()
            print("    python run_synthetic_trial.py --save-model --out-dir <new-dir>")
        return 1

    d = np.load(npz_path, allow_pickle=True)
    theta_cal = np.asarray(d["theta_cal"], dtype=np.float64)
    ps = np.asarray(d["posterior_samples"], dtype=np.float64)
    Z_cal = np.asarray(d["Z_cal"], dtype=np.float64)
    z_sim = np.asarray(d["z_sim"], dtype=np.float64)
    z_ok = _get(d, "z_real_ok")
    z_bad = _get(d, "z_real_shifted")
    members = _get(d, "member_samples")

    contract = Contract(
        param_names=[str(x) for x in d["param_names"]],
        coord=[str(x) for x in d["coord"]],
        bounds_theta=np.asarray(d["bounds_theta"], dtype=np.float64),
        embedding_dim=int(d["embedding_dim"]),
    )
    contract.validate()

    print("=" * 70)
    print("REPLOT from %s" % args.run_dir)
    print("=" * 70)
    print("  calibration obs   : %d" % theta_cal.shape[0])
    print("  posterior samples : %s" % (ps.shape,))
    print("  p / E             : %d / %d" % (contract.p, contract.embedding_dim))

    # -- recompute the cheap diagnostics -----------------------------------
    sbc = simulation_based_calibration(theta_cal, ps,
                                       param_names=contract.param_names,
                                       seed=args.seed)
    fam = family_verdict(sbc)
    ddsbc = data_dependent_sbc(theta_cal, ps, Z_cal, n_forms=args.n_forms,
                               seed=args.seed)
    tarp_res = tarp(theta_cal, ps, Z=Z_cal, reference_mode="auto",
                    n_references=3, seed=args.seed)
    spec = information_spectrum(theta_cal, ps, param_names=contract.param_names)

    # Expected coverage needs log q(theta | z), which is a model evaluation
    # and is deliberately NOT stored -- it would mean carrying the ensemble
    # around. Figure 04 is therefore skipped on a replot; re-run the trial
    # if you need it.
    cov = None

    ov_ok = ov_bad = None
    if z_ok is not None and z_ok.size:
        ov_ok = embedding_overlap(z_sim, np.atleast_2d(z_ok),
                                  n_null=args.n_null, seed=args.seed)
        print("\n" + ov_ok.summary())
    if z_bad is not None and z_bad.size:
        ov_bad = embedding_overlap(z_sim, np.atleast_2d(z_bad),
                                   n_null=args.n_null, seed=args.seed)

    print("\n" + diagnostic_report(sbc=sbc, coverage=cov, ddsbc=ddsbc,
                                   tarp_res=tarp_res, contraction=spec,
                                   overlap=ov_ok))
    if cov is None:
        print("\n  NOTE: expected coverage (figure 04) is unavailable on a")
        print("  replot -- it needs log q(theta | z) from the model itself.")
        print("  Everything else is reproduced exactly.")

    # -- figures ------------------------------------------------------------
    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    if not plots_available():
        print("\n  matplotlib not installed; no figures written")
        return 0
    files = save_all_plots(
        out_dir, sbc=sbc, coverage=cov, ddsbc=ddsbc, tarp_res=tarp_res,
        contraction=spec, overlap=ov_ok, overlap_shifted=ov_bad,
        z_sim=z_sim, z_real=z_ok, z_real_shifted=z_bad,
        theta_true=theta_cal, posterior_samples=ps,
        member_samples=list(members) if members is not None and members.size else None,
        embedding_dim=contract.embedding_dim,
        sbc_family=fam, ddsbc_family=family_verdict(ddsbc))
    print("\n  %d figure(s) written to %s" % (len(files), out_dir))
    for f in files:
        print("    %s" % os.path.basename(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
