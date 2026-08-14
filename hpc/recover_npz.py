#!/usr/bin/env python3
"""
recover_npz.py -- rebuild diagnostics_data.npz for a run that predates it.

Runs made before diagnostics_data.npz was added store the shards, the saved
ensemble, and trial_summary.json, but not the posterior samples. This script
reconstructs the missing file so replot.py works, without retraining.

    python recover_npz.py synthetic_trial_20260811_152440
    python recover_npz.py <run-dir> --n-draws 128

WHY THIS IS POSSIBLE
--------------------
The train/calibration split is deterministic. The shards are written by
make_synthetic_shard() from a fixed seed, so loading them back reproduces
exactly the same rows in the same order; the split is then the first n_cal
entries of np.random.default_rng(seed).permutation(n). Reading the seed and
n_calib out of trial_summary.json therefore recovers the identical
calibration set the run used.

The only real work is re-drawing posterior samples from the saved ensemble,
which costs the same as stage 4 of the original run -- minutes, not hours,
and no training.

REQUIREMENTS
------------
The run must have been made with --save-model (the default in the PBS job,
but NOT the default when running the script by hand). Without ensemble/ the
posterior samples cannot be regenerated at all and the run must be repeated.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_contract import load_shards
from npe_diagnostics import sample_posteriors
from npe_model import load_ensemble


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n-draws", type=int, default=None,
                    help="posterior draws per observation (default: as the "
                         "original run, from trial_summary.json)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed if trial_summary.json is missing")
    ap.add_argument("--n-calib", type=int, default=None,
                    help="override n_calib if trial_summary.json is missing")
    args = ap.parse_args()

    run = args.run_dir
    out_npz = os.path.join(run, "diagnostics_data.npz")
    if os.path.exists(out_npz):
        print("  %s already exists; nothing to do." % out_npz)
        return 0

    # -- read the original configuration -----------------------------------
    summary_path = os.path.join(run, "trial_summary.json")
    cfg = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh).get("config", {})
        print("  read config from trial_summary.json")
    else:
        print("  no trial_summary.json; relying on command-line overrides")

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    n_calib = args.n_calib if args.n_calib is not None else int(cfg.get("n_calib", 300))
    n_draws = args.n_draws if args.n_draws is not None else int(cfg.get("n_draws", 128))
    quick = bool(cfg.get("quick", False))
    if quick and args.n_calib is None:
        # --quick overrides these AFTER they are parsed, so the stored config
        # records the pre-override values. Mirror the same substitution here
        # or the recovered split will not match the original run.
        n_calib, n_draws = 100, 64
        print("  run used --quick: using n_calib=100, n_draws=64")
    print("  seed=%d  n_calib=%d  n_draws=%d" % (seed, n_calib, n_draws))

    # -- reload the shards --------------------------------------------------
    shards = sorted(glob.glob(os.path.join(run, "trial_*.parquet")))
    if not shards:
        print("ERROR: no trial_*.parquet shards in %s -- cannot recover." % run)
        return 1
    print("  shards: %s" % [os.path.basename(s) for s in shards])
    Z, theta, contract = load_shards(shards)
    print("  loaded %d rows, p=%d, E=%d" % (Z.shape[0], contract.p,
                                            contract.embedding_dim))

    # -- reproduce the split ------------------------------------------------
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    perm = rng.permutation(n)
    n_cal = min(n_calib, n // 4)
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    Z_tr, Z_cal, th_cal = Z[tr_idx], Z[cal_idx], theta[cal_idx]
    print("  reproduced split: %d train / %d calibration"
          % (Z_tr.shape[0], Z_cal.shape[0]))

    # -- reload the ensemble ------------------------------------------------
    ens_dir = os.path.join(run, "ensemble")
    if not os.path.isdir(ens_dir):
        print()
        print("ERROR: %s not found." % ens_dir)
        print("The run was made without --save-model, so the trained")
        print("ensemble no longer exists and posterior samples cannot be")
        print("regenerated. Re-run the trial:")
        print()
        print("    python run_synthetic_trial.py --out-dir <new-dir>")
        return 1
    ensemble, meta = load_ensemble(ens_dir)
    n_members = len(ensemble.posteriors)
    print("  reloaded ensemble: %d members" % n_members)

    # -- re-sample ----------------------------------------------------------
    print("  sampling %d obs x %d draws (the only slow step) ..."
          % (Z_cal.shape[0], n_draws), flush=True)
    ps = sample_posteriors(ensemble, Z_cal, n_draws=n_draws)

    member_samples = []
    try:
        for mem in ensemble.posteriors:
            member_samples.append(sample_posteriors(mem, Z_cal[:1], n_draws=n_draws)[0])
    except Exception as exc:  # noqa: BLE001
        print("  per-member sampling unavailable: %s" % exc)
        member_samples = []

    # -- rebuild the query sets --------------------------------------------
    # These were built in stage 6 from a generator that had already consumed
    # draws, so they are NOT exactly reproducible. Rebuild equivalents: a
    # held-out matched set and a deliberately shifted one. The overlap
    # figures will therefore be statistically equivalent to the original but
    # not identical, and that is stated rather than hidden.
    from npe_contract import make_synthetic_shard
    n_real = int(cfg.get("n_real", 36))
    z_ok, _, _, _ = make_synthetic_shard(
        run, name="recovered_holdout", n_rows=n_real, p=contract.p,
        embedding_dim=contract.embedding_dim,
        n_log_axes=int(cfg.get("n_log_axes", 19)), seed=seed, write=False)
    rng2 = np.random.default_rng(seed + 777)
    base = z_ok.mean(axis=0)
    base = base / np.linalg.norm(base)
    z_bad = base[None, :] + 0.15 * rng2.normal(size=(n_real, contract.embedding_dim))
    z_bad /= np.linalg.norm(z_bad, axis=1, keepdims=True)
    print("  NOTE: the overlap query sets are rebuilt, not recovered -- the")
    print("        original ones were not saved. Figures 01 will be")
    print("        statistically equivalent but not byte-identical.")

    sub = np.random.default_rng(0).choice(Z_tr.shape[0],
                                          min(20000, Z_tr.shape[0]), replace=False)
    np.savez_compressed(
        out_npz,
        theta_cal=th_cal.astype(np.float32),
        posterior_samples=ps.astype(np.float32),
        Z_cal=Z_cal.astype(np.float32),
        z_sim=Z_tr[sub].astype(np.float32),
        z_real_ok=z_ok.astype(np.float32),
        z_real_shifted=z_bad.astype(np.float32),
        param_names=np.array(contract.param_names, dtype=object),
        coord=np.array(contract.coord, dtype=object),
        bounds_theta=contract.bounds_theta,
        embedding_dim=np.array(contract.embedding_dim),
        member_samples=(np.stack(member_samples).astype(np.float32)
                        if member_samples else np.zeros((0, 0, 0), np.float32)),
    )
    print("\n  wrote %s (%.1f MB)" % (out_npz, os.path.getsize(out_npz) / 1e6))
    print("  now run:  python replot.py %s" % run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
