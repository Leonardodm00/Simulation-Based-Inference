#!/usr/bin/env python3
"""
run_synthetic_trial.py -- full end-to-end dress rehearsal on synthetic data.

Exercises every stage the real data will pass through, at production shapes,
with a KNOWN forward map so the expected answers are available in advance:

    1. write schema-valid parquet shards + JSON sidecars
    2. load them back through the contract (validating assertions A2-A10)
    3. train an ensemble of NPEs
    4. run the full diagnostic battery
    5. run the sim-vs-real overlap gate on BOTH a matched and a deliberately
       shifted query set, so the gate is seen to fire and to stay quiet
    6. write a report and a timing table

Usage:
    python run_synthetic_trial.py --quick          # ~3 min, small everything
    python run_synthetic_trial.py                  # default, ~20-40 min
    python run_synthetic_trial.py --n-train 300000 --n-members 10
    python run_synthetic_trial.py --embedding-dim 8 --p 27

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
The synthetic generator maps theta through a fixed linear map into E
dimensions and then normalises onto the unit sphere, so z carries exactly
E - 1 degrees of freedom.

Read the informativeness section as follows. Per-axis CONTRACTION answers
"is this parameter informed?" and is the number that matters. The effective
RANK of the information spectrum is descriptive only: it is a linear
measure of a possibly curved image, and a nonlinear estimator can show a
rank above E - 1 without anything being wrong. Do not treat rank <= E - 1
as a pass/fail criterion -- see the note in information_spectrum().

What the trial does establish, before any real data exists: the pipeline
runs end to end at production shapes, the diagnostics compose, the overlap
gate fires on a shifted query set and stays quiet on a matched one, and you
get timings to extrapolate from.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_contract import load_shards, make_synthetic_shard, validate_arrays
from npe_diagnostics import (
    data_dependent_sbc, diagnostic_report, embedding_overlap,
    expected_coverage, family_verdict, information_spectrum,
    posterior_contraction, posterior_log_probs, sample_posteriors,
    simulation_based_calibration, tarp,
)
from npe_model import NPEConfig, save_ensemble, train_ensemble
from npe_plots import plots_available, save_all_plots


# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.rows: List[tuple] = []
        self._t0 = None
        self._name = None

    def start(self, name: str):
        self._name, self._t0 = name, time.time()
        print("\n" + "=" * 70)
        print("=== %s" % name)
        print("=" * 70, flush=True)

    def stop(self):
        dt = time.time() - self._t0
        self.rows.append((self._name, dt))
        print("  [%.1f s]" % dt, flush=True)

    def table(self) -> str:
        out = ["  %-40s %10s" % ("STAGE", "SECONDS"),
               "  %-40s %10s" % ("-" * 40, "-" * 10)]
        for name, dt in self.rows:
            out.append("  %-40s %10.1f" % (name, dt))
        out.append("  %-40s %10.1f" % ("TOTAL", sum(d for _, d in self.rows)))
        return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=27, help="parameter dimension")
    ap.add_argument("--embedding-dim", type=int, default=12, help="E")
    ap.add_argument("--n-log-axes", type=int, default=19)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-shards", type=int, default=2,
                    help="split the training set into this many shards, to "
                         "exercise the multi-shard load path")
    ap.add_argument("--n-calib", type=int, default=300,
                    help="calibration observations for the diagnostics")
    ap.add_argument("--n-draws", type=int, default=128,
                    help="posterior draws per calibration observation")
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--n-real", type=int, default=36,
                    help="size of the simulated 'real' query set")
    ap.add_argument("--out-dir", default="synthetic_trial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="tiny settings for a fast end-to-end check")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip figure generation")
    args = ap.parse_args()

    if args.quick:
        args.n_train, args.n_calib, args.n_draws = 3000, 100, 64
        args.n_members, args.n_shards = 2, 2

    os.makedirs(args.out_dir, exist_ok=True)
    T = Timer()
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("SYNTHETIC TRIAL -- full pipeline on data with a known answer")
    print("=" * 70)
    print("  p (parameters)      : %d" % args.p)
    print("  E (embedding dim)   : %d" % args.embedding_dim)
    print("  training rows       : %d across %d shard(s)" % (args.n_train, args.n_shards))
    print("  ensemble members    : %d" % args.n_members)
    print("  calibration set     : %d obs x %d draws" % (args.n_calib, args.n_draws))
    print()
    print("  z carries E - 1 = %d degrees of freedom; the information"
          % (args.embedding_dim - 1))
    print("  budget is capped by that however it is spread across the %d axes."
          % args.p)

    # -- 1. write shards ---------------------------------------------------
    T.start("1. write synthetic shards")
    per = args.n_train // args.n_shards
    paths = []
    for j in range(args.n_shards):
        _, _, _, path = make_synthetic_shard(
            args.out_dir, name="trial_%03d" % j, n_rows=per, p=args.p,
            embedding_dim=args.embedding_dim, n_log_axes=args.n_log_axes,
            seed=args.seed)
        paths.append(path)
        print("  wrote %s (%d rows)" % (os.path.basename(path), per))
    T.stop()

    # -- 2. load through the contract --------------------------------------
    T.start("2. load and validate through the contract")
    Z, theta, contract = load_shards(paths)
    rep = validate_arrays(Z, theta, contract)
    print("  " + rep.summary().replace("\n", "\n  "))
    print("  p from contract     : %d" % contract.p)
    print("  E from contract     : %d" % contract.embedding_dim)
    print("  log axes            : %d / %d" % (int(contract.log_mask.sum()), contract.p))
    if not rep.ok:
        print("\nABORT: shard validation failed.")
        return 1
    T.stop()

    # Hold out a calibration split. Diagnostics computed on training rows are
    # optimistic and would hide exactly the overfitting they exist to catch.
    n = Z.shape[0]
    perm = rng.permutation(n)
    n_cal = min(args.n_calib, n // 4)
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    Z_tr, th_tr = Z[tr_idx], theta[tr_idx]
    Z_cal, th_cal = Z[cal_idx], theta[cal_idx]
    print("  train / calibration : %d / %d" % (Z_tr.shape[0], Z_cal.shape[0]))

    # -- 3. train ----------------------------------------------------------
    T.start("3. train ensemble (%d members)" % args.n_members)
    cfg = NPEConfig(hidden_features=64, num_transforms=4, num_bins=8,
                    training_batch_size=256, max_num_epochs=60,
                    stop_after_epochs=10, show_progress_bars=False)
    prior = contract.prior()
    ensemble = train_ensemble(Z_tr, th_tr, prior, n_members=args.n_members,
                              config=cfg, base_seed=args.seed, verbose=True)
    if args.save_model:
        save_ensemble(ensemble, os.path.join(args.out_dir, "ensemble"),
                      contract=contract, config=cfg)
        print("  saved to %s" % os.path.join(args.out_dir, "ensemble"))
    T.stop()

    # -- 4. posterior draws for the calibration set ------------------------
    T.start("4. sample posteriors for the calibration set")
    ps = sample_posteriors(ensemble, Z_cal, n_draws=args.n_draws)
    print("  posterior samples   : %s" % (ps.shape,))
    # One observation drawn from each member separately, for figure 11. The
    # spread between members is the ensemble's contribution to the reported
    # uncertainty, and it is invisible in the pooled samples.
    member_samples = []
    try:
        for mem in ensemble.posteriors:
            member_samples.append(
                sample_posteriors(mem, Z_cal[:1], n_draws=args.n_draws)[0])
    except Exception as exc:  # noqa: BLE001
        print("  per-member sampling unavailable: %s" % exc)
        member_samples = None
    inside = np.all((ps >= contract.low) & (ps <= contract.high), axis=2)
    print("  fraction in-box     : %.4f" % float(inside.mean()))
    T.stop()

    # -- 4b. persist the arrays needed to re-plot or re-diagnose later -----
    # Diagnostics themselves are cheap pure-numpy; only the posterior
    # SAMPLING in stage 4 is expensive. Saving the samples means a later
    # re-plot costs seconds instead of repeating that stage. Stored as
    # float32: these are inputs to rank statistics and figures, where
    # float64 buys nothing and doubles the file.
    npz_path = os.path.join(args.out_dir, "diagnostics_data.npz")
    rng_sub = np.random.default_rng(0)
    z_keep = Z_tr[rng_sub.choice(Z_tr.shape[0], min(20000, Z_tr.shape[0]),
                                 replace=False)]
    np.savez_compressed(
        npz_path,
        theta_cal=th_cal.astype(np.float32),
        posterior_samples=ps.astype(np.float32),
        Z_cal=Z_cal.astype(np.float32),
        z_sim=z_keep.astype(np.float32),
        z_real_ok=z_real_ok.astype(np.float32) if "z_real_ok" in dir() else np.zeros((0, 0), np.float32),
        param_names=np.array(contract.param_names, dtype=object),
        coord=np.array(contract.coord, dtype=object),
        bounds_theta=contract.bounds_theta,
        embedding_dim=np.array(contract.embedding_dim),
        member_samples=(np.stack(member_samples).astype(np.float32)
                        if member_samples else np.zeros((0, 0, 0), np.float32)),
    )
    print("  arrays for re-plotting -> %s (%.1f MB)"
          % (npz_path, os.path.getsize(npz_path) / 1e6))

    # -- 5. diagnostics ----------------------------------------------------
    T.start("5. diagnostics")
    sbc = simulation_based_calibration(th_cal, ps, param_names=contract.param_names,
                                       seed=args.seed)
    fam = family_verdict(sbc)

    try:
        lp_true = posterior_log_probs(ensemble, th_cal, Z_cal)
        lp_samp = posterior_log_probs(ensemble, ps, Z_cal)
        cov = expected_coverage(lp_true, lp_samp, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        print("  expected coverage unavailable: %s" % exc)
        cov = None

    ddsbc = data_dependent_sbc(th_cal, ps, Z_cal, n_forms=4, seed=args.seed)
    tarp_res = tarp(th_cal, ps, Z=Z_cal, reference_mode="auto",
                    n_references=3, seed=args.seed)
    spec = information_spectrum(th_cal, ps, param_names=contract.param_names)
    T.stop()

    # -- 6. the overlap gate, both directions ------------------------------
    T.start("6. sim-vs-real overlap gate")
    # (a) a query set drawn from the same process but NOT from the training
    # pool. Reusing training rows makes every real-to-sim nearest-neighbour
    # distance exactly zero -- each point matches itself -- which both
    # flatters the gate and produces a degenerate distance distribution.
    z_hold, _, _, _ = make_synthetic_shard(
        args.out_dir, name="holdout_real", n_rows=args.n_real, p=args.p,
        embedding_dim=args.embedding_dim, n_log_axes=args.n_log_axes,
        seed=args.seed, write=False)
    z_real_ok = z_hold
    ov_ok = embedding_overlap(Z_tr, z_real_ok, n_null=300, seed=args.seed)
    print("  [matched query set]")
    print("  " + ov_ok.summary().replace("\n", "\n  "))

    # (b) a deliberately shifted set, standing in for a simulation gap
    base = z_real_ok.mean(axis=0)
    base = base / np.linalg.norm(base)
    z_real_bad = base[None, :] + 0.15 * rng.normal(size=(args.n_real, contract.embedding_dim))
    z_real_bad /= np.linalg.norm(z_real_bad, axis=1, keepdims=True)
    ov_bad = embedding_overlap(Z_tr, z_real_bad, n_null=300, seed=args.seed)
    print("\n  [shifted query set -- stands in for a simulation gap]")
    print("  " + ov_bad.summary().replace("\n", "\n  "))
    T.stop()

    # Append the query sets now that stage 6 has built them.
    try:
        existing = dict(np.load(npz_path, allow_pickle=True))
        existing["z_real_ok"] = z_real_ok.astype(np.float32)
        existing["z_real_shifted"] = z_real_bad.astype(np.float32)
        np.savez_compressed(npz_path, **existing)
    except Exception as exc:  # noqa: BLE001
        print("  could not append query sets to the npz: %s" % exc)

    # -- 6b. figures --------------------------------------------------------
    if not args.no_plots:
        T.start("6b. figures")
        fig_dir = os.path.join(args.out_dir, "figures")
        files = save_all_plots(
            fig_dir,
            sbc=sbc, coverage=cov, ddsbc=ddsbc, tarp_res=tarp_res,
            contraction=spec, overlap=ov_ok, overlap_shifted=ov_bad,
            z_sim=Z_tr, z_real=z_real_ok, z_real_shifted=z_real_bad,
            theta_true=th_cal, posterior_samples=ps,
            member_samples=member_samples,
            embedding_dim=contract.embedding_dim,
            sbc_family=fam, ddsbc_family=family_verdict(ddsbc))
        for f in files:
            print("  %s" % f)
        print("  %d figure(s) in %s" % (len(files), fig_dir))
        T.stop()

    # -- 7. report ---------------------------------------------------------
    print("\n" + diagnostic_report(sbc=sbc, coverage=cov, ddsbc=ddsbc,
                                   tarp_res=tarp_res, contraction=spec,
                                   overlap=ov_ok))

    e_minus_1 = args.embedding_dim - 1
    rank = spec.effective_rank
    n_flat = int(np.sum(spec.contraction < 0.05))
    print("\n" + "=" * 70)
    print("INFORMATION BUDGET")
    print("=" * 70)
    print("  E - 1 (degrees of freedom in z)  : %d" % e_minus_1)
    print("  parameters                       : %d" % args.p)
    print("  effective rank (descriptive)     : %d" % rank)
    print("  axes with contraction < 0.05     : %d" % n_flat)
    print("  mean contraction                 : %.3f" % float(np.mean(spec.contraction)))
    print("  top eigenvalues                  : %s"
          % np.round(spec.eigenvalues[:min(8, args.p)], 3).tolist())
    print("  NOTE: effective rank is a LINEAR measure of a curved image, so a")
    print("        nonlinear estimator can legitimately exceed E-1. Judge")
    print("        informativeness from per-axis contraction, not from rank.")

    print("\n" + "=" * 70)
    print("TIMING")
    print("=" * 70)
    print(T.table())

    summary: Dict = {
        "config": vars(args),
        "contract": {"p": contract.p, "E": contract.embedding_dim},
        "sbc_family_rejected": int(fam.n_rejected),
        "coverage_ks_p": None if cov is None else cov.ks_pvalue,
        "ddsbc_rejected": int(family_verdict(ddsbc).n_rejected),
        "tarp_ks_p": tarp_res.ks_pvalue,
        "effective_rank_descriptive": int(rank),
        "e_minus_1": e_minus_1,
        "n_axes_uninformed": n_flat,
        "mean_contraction": float(np.mean(spec.contraction)),
        "overlap_matched_p": ov_ok.p_value,
        "overlap_shifted_p": ov_bad.p_value,
        "timing_seconds": {k: v for k, v in T.rows},
    }
    out_json = os.path.join(args.out_dir, "trial_summary.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("\n  summary written to %s" % out_json)

    # The gate must fire on the shifted set and stay quiet on the matched
    # one; if not, the gate is not usable on real data and that matters more
    # than any calibration number above.
    gate_ok = (not ov_ok.rejects) and ov_bad.rejects
    if not gate_ok:
        print("\n  WARNING: the overlap gate did not behave as expected "
              "(matched p=%.4f, shifted p=%.4f)." % (ov_ok.p_value, ov_bad.p_value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
