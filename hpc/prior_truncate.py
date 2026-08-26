#!/usr/bin/env python3
"""
prior_truncate.py -- truncated prior box from amortized per-window HPRs.

Purpose
-------
Train (or reuse) the amortized NPE ensemble q(theta | z) on the exported
simulation bank, evaluate it at EVERY real embedded window z_j, form the
per-window highest-probability region

    HPR_eps(j) = { theta : q(theta | z_j) >= tau_j },                    (1)

with tau_j the eps-quantile of q(theta' | z_j) over the window's own
posterior draws theta' ~ q(. | z_j) (the TSNPE threshold rule, Deistler,
Goncalves, Macke 2022, applied per observation), and report the
axis-aligned truncation box

    T = prod_k [ min_j lo_jk , max_j hi_jk ]  intersect  prior box,      (2)

where (lo_jk, hi_jk) is the per-axis extent of the kept draws of window j.
By construction no window's (1 - eps) region is clipped. The box is padded
by pad_frac of the prior range per axis, then clipped to the prior box.

The box hierarchy is reported at four levels, each an envelope (or pooled
quantile, see --box_mode) of the level below:

    window  <=  culture  <=  condition  <=  common

"common" covers both conditions and is the deliverable for the follow-up
campaign; per-condition and per-culture boxes are saved for later focus.

The prior over inference coordinates is the uniform box (log axes are
already natural-log in theta), so the retained prior MASS equals the
retained VOLUME fraction  prod_k (hi_k - lo_k) / (HI_k - LO_k),  which is
printed per level.

Method notes, stated once
-------------------------
* eps is per window. The union over ~2e3 windows is already generous;
  keep eps at 1e-3 unless there is a reason not to (TSNPE reports
  insensitivity for eps <= 1e-4; errors from over-truncation are
  unrecoverable downstream, so wide beats tight).
* Sample-based extents underestimate the true HPR extent at finite
  n_draws; the padding and the envelope over many windows are what absorb
  that. Do not read a single window's box as inferential.
* Tail sensitivity, the practical failure mode: at eps = 1e-3 the kept
  set is essentially ALL draws, so the extent is a min/max order
  statistic and measures the flow's TAILS, not its core. If the box
  saturates the prior (see n_axes_at_prior in the output), raise --eps:
  the threshold tau_j is the eps-quantile, so a larger eps can only
  tighten every extent (monotone by construction), while each window
  still keeps its own (1 - eps) region.
* The ensemble mixture (arithmetic, npe_model E1) is what inflates the
  credible regions; a single member would give tighter, less honest boxes.
* Everything runs in INFERENCE coordinates (mixed ln/linear, from the
  sidecar contract). Natural-unit bounds are derived for reporting only.

Inputs
------
* --sim     glob of simulated parquet shards (theta labels required).
* --real    the real-cohort parquet (no theta; culture/condition columns).
* --activity  the build_activity_table.py npz; required whenever a rate
  filter is requested. sim_rate must align with the RAW pooled row order
  of --sim (asserted); real_rate with the real parquet rows (asserted).

Outputs (under --out)
---------------------
* ensemble/                the trained ensemble (reused on re-runs).
* truncated_prior.json     all boxes + provenance. THE deliverable.
* truncate_arrays.npz      per-window extents, tau, labels, pooled
                           posterior subsamples (input for the plot stage).

Typical use on the cluster
--------------------------
  # probe first (a few windows, tiny cost):
  python3 prior_truncate.py --sim "$EXP/*.parquet" --real $EXP/real_cohort.parquet \
      --activity $ACT --out results/trunc_r2 --limit_real 25 --n_members 2
  # full run:
  python3 prior_truncate.py --sim "$EXP/*.parquet" --real $EXP/real_cohort.parquet \
      --activity $ACT --out results/trunc_r2

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gate_data                                    # noqa: E402
import npe_contract as C                            # noqa: E402
from npe_model import (NPEConfig, train_ensemble,   # noqa: E402
                       save_ensemble, load_ensemble)
from npe_diagnostics import (sample_posteriors,     # noqa: E402
                             posterior_log_probs, expected_coverage)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Truncated prior box from per-window posterior HPRs.")
    # data
    ap.add_argument("--sim", required=True,
                    help="glob of simulated parquet shards")
    ap.add_argument("--real", required=True,
                    help="real-cohort parquet (no theta columns)")
    ap.add_argument("--activity", default=None,
                    help="activity-table npz from build_activity_table.py")
    ap.add_argument("--min_rate", type=float, default=0.1,
                    help="sim-arm rate floor [Hz/electrode]; needs --activity "
                         "(default 0.1, matching the gate). Pass a negative "
                         "value to disable.")
    ap.add_argument("--real_min_rate", type=float, default=None,
                    help="real-arm rate floor; default: same as --min_rate")
    ap.add_argument("--group_col", default="culture")
    ap.add_argument("--class_col", default="condition")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="subsample the sim arm (development only)")
    # ensemble
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ensemble_dir", default=None,
                    help="default: <out>/ensemble; reused if it exists")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain even if the ensemble directory exists")
    ap.add_argument("--n_members", type=int, default=5)
    ap.add_argument("--n_cal", type=int, default=200,
                    help="sim rows held out of training for --coverage")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    # training overrides (defaults follow npe_model.NPEConfig)
    ap.add_argument("--hidden_features", type=int, default=128)
    ap.add_argument("--num_transforms", type=int, default=8)
    ap.add_argument("--num_bins", type=int, default=10)
    ap.add_argument("--max_epochs", type=int, default=500)
    ap.add_argument("--stop_after", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    # HPR / box
    ap.add_argument("--n_draws", type=int, default=2000,
                    help="posterior draws per real window")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="per-window truncation mass, in (0, 0.5)")
    ap.add_argument("--pad_frac", type=float, default=0.05,
                    help="padding per axis, as a fraction of the prior range")
    ap.add_argument("--box_mode", choices=("envelope", "pooled"),
                    default="envelope",
                    help="envelope: min/max over member windows (default, "
                         "conservative). pooled: per-axis [eps/2, 1-eps/2] "
                         "quantiles of pooled posterior draws per unit.")
    ap.add_argument("--pool_keep", type=int, default=256,
                    help="kept draws per window feeding pooled boxes/plots")
    ap.add_argument("--chunk", type=int, default=32,
                    help="windows processed per chunk")
    ap.add_argument("--limit_real", type=int, default=None,
                    help="probe mode: only the first N real windows")
    ap.add_argument("--progress_every", type=int, default=100)
    ap.add_argument("--coverage", action="store_true",
                    help="expected-coverage check on the held-out sim rows")
    args = ap.parse_args(argv)

    if not (0.0 < args.eps < 0.5):
        ap.error("--eps must be in (0, 0.5)")
    if args.pad_frac < 0.0:
        ap.error("--pad_frac must be >= 0")
    if args.min_rate is not None and args.min_rate < 0:
        args.min_rate = None
    if args.real_min_rate is None:
        args.real_min_rate = args.min_rate
    if (args.min_rate is not None or args.real_min_rate is not None) \
            and args.activity is None:
        ap.error("a rate filter is requested but --activity was not given; "
                 "pass the activity npz or disable with --min_rate -1")
    if args.ensemble_dir is None:
        args.ensemble_dir = os.path.join(args.out, "ensemble")
    return args


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_arms(args):
    """Load both arms, rate-filtered; returns (sim Arm, real Arm, filt meta)."""
    sim = gate_data.load_sim(
        args.sim, dedup_theta=True, want_zraw=False,
        max_rows=args.max_rows, seed=args.seed,
        activity_path=args.activity, min_rate=args.min_rate, max_rate=None)
    real = gate_data.load_real(
        args.real, group_col=args.group_col, class_col=args.class_col,
        want_zraw=False)

    # encoder identity across arms (gate_data checks it only across shards)
    sha_s = sim.contract.meta.get("embedding", {}).get("dsn_checkpoint_sha256")
    sha_r = real.contract.meta.get("embedding", {}).get("dsn_checkpoint_sha256")
    if sha_s is not None and sha_r is not None and sha_s != sha_r:
        raise ValueError(
            "sim and real shards come from different DSN checkpoints "
            "(%s vs %s); they cannot be combined." % (sha_s, sha_r))
    if sim.E != real.E:
        raise ValueError("embedding dim differs: sim E=%d, real E=%d"
                         % (sim.E, real.E))

    filt = {"real": None}
    if args.real_min_rate is not None:
        tab = np.load(args.activity, allow_pickle=False)
        if "real_rate" not in tab:
            raise KeyError("activity table %r has no 'real_rate'; rebuild it "
                           "with the real arm included or disable the real "
                           "filter" % args.activity)
        rate = np.asarray(tab["real_rate"], dtype=np.float64)
        if rate.shape[0] != real.n:
            raise RuntimeError(
                "activity real_rate has %d rows but the real parquet gives "
                "%d; the table must be rebuilt against this export."
                % (rate.shape[0], real.n))
        keep = np.isfinite(rate) & (rate >= float(args.real_min_rate))
        filt["real"] = {"min_rate_hz_per_electrode": args.real_min_rate,
                        "n_before": int(real.n), "n_after": int(keep.sum())}
        if not keep.any():
            raise ValueError("the real-arm rate filter removed every window")
        real = gate_data.Arm(
            z=real.z[keep], zraw=None, theta=None,
            groups=real.groups[keep], classes=real.classes[keep],
            contract=real.contract, meta=real.meta)

    if args.limit_real is not None:
        n = min(args.limit_real, real.n)
        print("  PROBE MODE: using the first %d of %d real windows" % (n, real.n))
        real = gate_data.Arm(
            z=real.z[:n], zraw=None, theta=None,
            groups=real.groups[:n], classes=real.classes[:n],
            contract=real.contract, meta=real.meta)

    # every culture must belong to exactly one condition
    cultures = np.asarray([str(g) for g in real.groups])
    classes = np.asarray([str(c) for c in real.classes])
    for cu in np.unique(cultures):
        cl = np.unique(classes[cultures == cu])
        if cl.shape[0] != 1:
            raise ValueError("culture %r appears under conditions %s; the "
                             "hierarchy needs one condition per culture"
                             % (cu, cl.tolist()))
    return sim, real, filt


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def get_ensemble(args, sim):
    """Train, or reload if the directory already holds one. Returns
    (ensemble, cal_idx, trained_now)."""
    contract = sim.contract
    rng = np.random.default_rng(args.seed)
    n = sim.n
    n_cal = min(args.n_cal, max(0, n - 100))
    cal_idx = np.sort(rng.choice(n, size=n_cal, replace=False)) \
        if n_cal > 0 else np.zeros(0, dtype=int)
    train_mask = np.ones(n, dtype=bool)
    train_mask[cal_idx] = False

    meta_path = os.path.join(args.ensemble_dir, "ensemble.json")
    if os.path.isfile(meta_path) and not args.retrain:
        print("  loading existing ensemble from %s" % args.ensemble_dir)
        ensemble, meta = load_ensemble(args.ensemble_dir)
        saved = meta.get("contract", {})
        if list(saved.get("param_names", [])) != list(contract.param_names):
            raise ValueError(
                "the saved ensemble was trained under different param_names "
                "than the current shards; refuse to mix. Use --retrain or a "
                "fresh --ensemble_dir.")
        return ensemble, cal_idx, False

    config = NPEConfig(
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        num_bins=args.num_bins,
        max_num_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after,
        training_batch_size=args.batch_size,
        device=args.device)
    prior = contract.prior(device=args.device)
    ensemble = train_ensemble(
        sim.z[train_mask], sim.theta[train_mask], prior,
        n_members=args.n_members, config=config, base_seed=args.seed)
    save_ensemble(ensemble, args.ensemble_dir,
                  contract=contract, config=config)
    return ensemble, cal_idx, True


# ---------------------------------------------------------------------------
# Per-window HPR extents
# ---------------------------------------------------------------------------

def window_extents(ensemble, Z, n_draws, eps, pool_keep, chunk, seed,
                   progress_every):
    """Sample-based HPR extents per real window.

    Returns (lo (n,p), hi (n,p), tau (n,), pooled (n_pool, p),
    pooled_win (n_pool,) window index of each pooled draw).
    """
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = Z.shape[0]
    lo = hi = None
    tau = np.empty(n, dtype=np.float64)
    pooled, pooled_win = [], []

    done = 0
    for start in range(0, n, chunk):
        Zc = Z[start:start + chunk]
        S = sample_posteriors(ensemble, Zc, n_draws=n_draws)     # (c, M, p)
        L = posterior_log_probs(ensemble, S, Zc)                 # (c, M)
        if not np.all(np.isfinite(L)):
            bad = np.where(~np.all(np.isfinite(L), axis=1))[0] + start
            raise FloatingPointError(
                "non-finite posterior log-probs at real windows %s"
                % bad[:5].tolist())
        if lo is None:
            p = S.shape[2]
            lo = np.empty((n, p), dtype=np.float64)
            hi = np.empty((n, p), dtype=np.float64)
        for c in range(S.shape[0]):
            j = start + c
            tau[j] = np.quantile(L[c], eps)
            keep = L[c] >= tau[j]
            kept = S[c][keep]
            lo[j] = kept.min(axis=0)
            hi[j] = kept.max(axis=0)
            k = min(pool_keep, kept.shape[0])
            sel = rng.choice(kept.shape[0], size=k, replace=False)
            pooled.append(kept[sel].astype(np.float32))
            pooled_win.append(np.full(k, j, dtype=np.int32))
        done += S.shape[0]
        if progress_every and (done % progress_every < chunk):
            print("    windows %d / %d" % (done, n), flush=True)

    return (lo, hi, tau,
            np.concatenate(pooled, axis=0),
            np.concatenate(pooled_win, axis=0))


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def _envelope(lo, hi, mask):
    return lo[mask].min(axis=0), hi[mask].max(axis=0)


def _pooled_quantiles(samples, eps):
    return (np.quantile(samples, eps / 2.0, axis=0),
            np.quantile(samples, 1.0 - eps / 2.0, axis=0))


def _pad_clip(lo, hi, prior_lo, prior_hi, pad_frac):
    pad = pad_frac * (prior_hi - prior_lo)
    return (np.maximum(lo - pad, prior_lo),
            np.minimum(hi + pad, prior_hi))


def _mass_fraction(lo, hi, prior_lo, prior_hi):
    """Retained prior mass. Exact because the prior is uniform in
    inference coordinates."""
    return float(np.prod((hi - lo) / (prior_hi - prior_lo)))


def _box_dict(contract, lo_raw, hi_raw, lo_pad, hi_pad, n_windows):
    prior_lo, prior_hi = contract.low, contract.high
    tol = 1e-9 * np.maximum(1.0, np.abs(prior_hi - prior_lo))
    at_prior = (lo_pad <= prior_lo + tol) | (hi_pad >= prior_hi - tol)
    return {
        "n_windows": int(n_windows),
        "bounds_theta": np.stack([lo_pad, hi_pad], axis=1).tolist(),
        "bounds_natural": np.stack(
            [contract.to_natural(lo_pad), contract.to_natural(hi_pad)],
            axis=1).tolist(),
        "raw_bounds_theta": np.stack([lo_raw, hi_raw], axis=1).tolist(),
        "prior_mass_fraction": _mass_fraction(lo_pad, hi_pad,
                                              prior_lo, prior_hi),
        "raw_prior_mass_fraction": _mass_fraction(lo_raw, hi_raw,
                                                  prior_lo, prior_hi),
        "n_axes_at_prior": int(at_prior.sum()),
    }


def build_boxes(contract, lo_w, hi_w, pooled, pooled_win,
                cultures, classes, box_mode, eps, pad_frac):
    """The window <= culture <= condition <= common hierarchy."""
    prior_lo, prior_hi = contract.low, contract.high

    def unit_box(mask):
        if box_mode == "envelope":
            lo_r, hi_r = _envelope(lo_w, hi_w, mask)
        else:
            wsel = np.where(mask)[0]
            psel = np.isin(pooled_win, wsel)
            lo_r, hi_r = _pooled_quantiles(pooled[psel], eps)
        lo_p, hi_p = _pad_clip(lo_r, hi_r, prior_lo, prior_hi, pad_frac)
        return lo_r, hi_r, lo_p, hi_p

    boxes = {"per_culture": {}, "per_condition": {}}
    for cu in np.unique(cultures):
        m = cultures == cu
        lo_r, hi_r, lo_p, hi_p = unit_box(m)
        d = _box_dict(contract, lo_r, hi_r, lo_p, hi_p, m.sum())
        d["condition"] = str(classes[m][0])
        boxes["per_culture"][str(cu)] = d
    for cl in np.unique(classes):
        m = classes == cl
        lo_r, hi_r, lo_p, hi_p = unit_box(m)
        boxes["per_condition"][str(cl)] = _box_dict(
            contract, lo_r, hi_r, lo_p, hi_p, m.sum())
    m = np.ones(lo_w.shape[0], dtype=bool)
    lo_r, hi_r, lo_p, hi_p = unit_box(m)
    boxes["common"] = _box_dict(contract, lo_r, hi_r, lo_p, hi_p, m.sum())
    return boxes


def print_summary(contract, boxes, sim_theta):
    prior_lo, prior_hi = contract.low, contract.high
    rng_prior = prior_hi - prior_lo
    common = boxes["common"]
    lo = np.asarray(common["bounds_theta"], dtype=np.float64)[:, 0]
    hi = np.asarray(common["bounds_theta"], dtype=np.float64)[:, 1]
    frac = (hi - lo) / rng_prior

    order = np.argsort(frac)
    print("  common box: prior mass retained = %.3g (raw %.3g), "
          "%d/%d axes at the prior bounds"
          % (common["prior_mass_fraction"],
             common["raw_prior_mass_fraction"],
             common["n_axes_at_prior"], len(order)))
    if common["n_axes_at_prior"] == len(order):
        print("  WARNING: the common box saturates the prior on every axis. "
              "The per-window HPRs at this eps are dominated by the flow's "
              "tails; consider a larger --eps (still per-window, monotone: "
              "larger eps can only tighten) or --box_mode pooled.")
    print("  most-narrowed axes (padded width / prior width):")
    for k in order[:min(10, len(order))]:
        print("    %-14s %.3f   [%.4g, %.4g]"
              % (contract.param_names[k], frac[k], lo[k], hi[k]))
    for cl, d in sorted(boxes["per_condition"].items()):
        print("  condition %-4s: n_windows=%d, prior mass retained = %.3g"
              % (cl, d["n_windows"], d["prior_mass_fraction"]))

    inside = np.all((sim_theta >= lo[None, :]) &
                    (sim_theta <= hi[None, :]), axis=1)
    print("  training simulations inside the common box: %d / %d (%.1f%%)"
          "  -- the expected acceptance if rejection-sampling the old prior"
          % (inside.sum(), sim_theta.shape[0], 100.0 * inside.mean()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print("[1/5] loading arms", flush=True)
    sim, real, filt = load_arms(args)
    contract = sim.contract
    print("  sim : %d rows, p=%d, E=%d" % (sim.n, contract.p, sim.E))
    print("  real: %d windows, %d cultures, conditions %s"
          % (real.n, len(set(real.groups.tolist())),
             sorted(set(str(c) for c in real.classes))))

    print("[2/5] ensemble", flush=True)
    ensemble, cal_idx, trained_now = get_ensemble(args, sim)
    print("  members=%d, held-out cal rows=%d, trained_now=%s"
          % (len(ensemble.posteriors), cal_idx.shape[0], trained_now))

    if args.coverage and cal_idx.shape[0] > 0:
        print("[2b] expected coverage on held-out sims", flush=True)
        Zc, Tc = sim.z[cal_idx], sim.theta[cal_idx]
        Sc = sample_posteriors(ensemble, Zc, n_draws=min(args.n_draws, 500))
        lp_true = posterior_log_probs(ensemble, Tc, Zc)
        lp_samp = posterior_log_probs(ensemble, Sc, Zc)
        print(expected_coverage(lp_true, lp_samp, seed=args.seed).summary())

    print("[3/5] per-window HPR extents (eps=%g, n_draws=%d)"
          % (args.eps, args.n_draws), flush=True)
    cultures = np.asarray([str(g) for g in real.groups])
    classes = np.asarray([str(c) for c in real.classes])
    lo_w, hi_w, tau, pooled, pooled_win = window_extents(
        ensemble, real.z, args.n_draws, args.eps, args.pool_keep,
        args.chunk, args.seed, args.progress_every)

    print("[4/5] boxes (%s mode, pad_frac=%g)"
          % (args.box_mode, args.pad_frac), flush=True)
    boxes = build_boxes(contract, lo_w, hi_w, pooled, pooled_win,
                        cultures, classes, args.box_mode, args.eps,
                        args.pad_frac)
    print_summary(contract, boxes, sim.theta)

    print("[5/5] writing outputs", flush=True)
    out_json = {
        "schema_version": SCHEMA_VERSION,
        "kind": "truncated_prior_box",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "eps": args.eps, "pad_frac": args.pad_frac,
        "box_mode": args.box_mode, "n_draws": args.n_draws,
        "pool_keep": args.pool_keep,
        "n_members": len(ensemble.posteriors),
        "seed": args.seed,
        "param_names": list(contract.param_names),
        "coord": list(contract.coord),
        "prior_bounds_theta": np.asarray(contract.bounds_theta).tolist(),
        "prior_bounds_natural": contract.bounds_natural().tolist(),
        "filters": {"sim": sim.meta.get("activity_filter"),
                    "real": filt["real"]},
        "provenance": {
            "sim_pattern": args.sim, "real_path": args.real,
            "activity_path": args.activity,
            "ensemble_dir": args.ensemble_dir,
            "dsn_checkpoint_sha256": contract.meta.get(
                "embedding", {}).get("dsn_checkpoint_sha256"),
            "limit_real": args.limit_real,
            "n_sim_rows_train": int(sim.n - cal_idx.shape[0]),
            "n_real_windows": int(real.n),
        },
        "boxes": boxes,
    }
    json_path = os.path.join(args.out, "truncated_prior.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(out_json, fh, indent=2, sort_keys=True)

    npz_path = os.path.join(args.out, "truncate_arrays.npz")
    np.savez_compressed(
        npz_path,
        window_lo=lo_w.astype(np.float32),
        window_hi=hi_w.astype(np.float32),
        tau=tau.astype(np.float64),
        culture=cultures, condition=classes,
        pooled_samples=pooled, pooled_window=pooled_win,
        cal_idx=cal_idx.astype(np.int64))
    print("  wrote %s" % json_path)
    print("  wrote %s" % npz_path)
    if args.limit_real is not None:
        print("  NOTE: probe mode -- these boxes cover only the first %d "
              "windows and are NOT the deliverable." % args.limit_real)
    return 0


if __name__ == "__main__":
    sys.exit(main())
