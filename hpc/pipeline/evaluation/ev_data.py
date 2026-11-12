"""The evaluation's data sets (block 19; specs/SPEC.md sec. 2.5, "The data
sets", and block 19).

    index, paths = ev_data.sim_index(cfg)          # arm S, the simulated bank
    rows = ev_data.report_rows(index, fracs, seed, split_hash)
    calib = ev_data.calibration_rows(index.donor[rows])
    coh = ev_data.cohort(cfg, "R3")                # arm R: x, theta*, ids, sidecar
    paths = ev_data.cohort_paths(cfg, "R3")        # its shards alone
    theta_dag, in_box, delta = ev_data.pseudo_true(coh["theta"], coh["sidecar"]["gap_spec"])

theta^dag is the parameter at which the unshifted generator gives the
physics the range-shifted generator gave at theta*: theta* + delta on the
free axes, the duty (axis 7) capped where `phi_to_bench_params` clips it.
delta is the gap code's own (`latent_gap.free_axis_range_shift`), read from
the bank's sidecar, and the axes and their ranges are the provider's
(`BENCH_AXES`, `BENCH_FREE_IDX`), so nothing here restates them.

Pure ASCII, LF only.
"""

import glob
import os

import _evalpath  # noqa: F401

import numpy as np

from stages import common as CM
from stages.synthetic import RunView, bank_dir

import run_joint_arms as R
from bench_burst_provider import BENCH_AXES, BENCH_FREE_IDX
from latent_gap import GapSpec, free_axis_range_shift

DUTY_AXIS = 7      # the axis phi_to_bench_params clips at 1 (block 13, F-av)


def sim_index(cfg):
    """(ShardIndex, paths) of the simulated bank (every search's)."""
    q = cfg.data["synthetic"]["searches"][0]
    sim, _real = CM.shard_globs(RunView(cfg, q))
    return R.load_arm_index(sim)


def report_rows(index, fracs, seed, split_hash):
    """The report rows (split 2) of the split (fracs, seed), sorted; the
    split's hash must be `split_hash` (the members' records')."""
    split, h = R.grouped_split(index.donor, fracs=tuple(fracs), seed=int(seed))
    if h != split_hash:
        raise ValueError("the split of the bank is %s, the members' %s"
                         % (h[:12], str(split_hash)[:12]))
    return np.flatnonzero(split == 2)


def calibration_rows(donor):
    """Mask of the first row of each donor (Q-E2: rows sharing theta make the
    uniformity tests anti-conservative)."""
    donor = np.asarray(donor).ravel()
    _u, first = np.unique(donor, return_index=True)
    mask = np.zeros(donor.size, dtype=bool)
    mask[first] = True
    return mask


def cohort_paths(cfg, name):
    """The sorted shards of cohort `name` of `synthetic.real`."""
    pattern = os.path.join(bank_dir(cfg, name), "shard_*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError("no shards match %s" % pattern)
    return paths


def cohort(cfg, name):
    """Cohort `name` of `synthetic.real`: dict(x (n, W) float32, theta (n, 10)
    float64, donor, well, cls, window_idx (n,) int64, contaminated (n,) bool,
    sidecar, paths). [block 23: window_idx and contaminated added]"""
    paths = cohort_paths(cfg, name)
    pattern = os.path.join(bank_dir(cfg, name), "shard_*.npz")
    arrays, sidecar = R.concat_shards(paths)
    if not sidecar.get("theta_withheld", False):
        raise ValueError("%s is not an arm-R bank (theta not withheld)" % pattern)
    return {"x": np.asarray(arrays["x"], dtype=np.float32),
            "theta": np.asarray(arrays["theta"], dtype=np.float64),
            "donor": np.asarray(arrays["donor"], dtype=np.int64),
            "well": np.asarray(arrays["well"], dtype=np.int64),
            "cls": np.asarray(arrays["cls"], dtype=np.int64),
            "window_idx": np.asarray(arrays["window_idx"], dtype=np.int64),
            "contaminated": np.asarray(arrays["contaminated"], dtype=bool),
            "sidecar": sidecar, "paths": paths}


def range_shift_of(gap_spec):
    """delta, the free axes' range shift of a bank's `gap_spec` (0 when the
    mode is off or pi = 0)."""
    if not gap_spec:
        return 0.0
    g = GapSpec(modes=tuple(gap_spec["modes"]), pi=float(gap_spec["pi"]),
                shift_max=float(gap_spec["shift_max"]),
                drift_amp_max=float(gap_spec["drift_amp_max"]),
                drift_period_s=float(gap_spec["drift_period_s"]),
                contam_frac_max=float(gap_spec["contam_frac_max"]))
    return float(free_axis_range_shift(g))


def pseudo_true(theta, gap_spec):
    """(theta_dag (n, 10) float64, in_box (n,) bool, delta): theta* + delta on
    the free axes, the duty capped at (1 - L_7) / (U_7 - L_7); in_box when every
    coordinate is in the open box (0, 1)."""
    delta = range_shift_of(gap_spec)
    t = np.array(theta, dtype=np.float64, copy=True)
    for k in BENCH_FREE_IDX:
        t[:, k] += delta
    _name, lo, hi = BENCH_AXES[DUTY_AXIS]
    t[:, DUTY_AXIS] = np.minimum(t[:, DUTY_AXIS], (1.0 - lo) / (hi - lo))
    in_box = np.all((t > 0.0) & (t < 1.0), axis=1)
    return t, in_box, delta
