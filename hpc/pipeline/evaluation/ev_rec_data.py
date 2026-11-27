"""The data sets of a recorded project's evaluation (block 33; specs/SPEC.md
sec. 2.6, "The data sets of an ensemble", and block 33).

    index, paths = ev_rec_data.sim_index(view)       # the simulated bank
    rows = ev_rec_data.report_rows(index, fracs, seed, split_hash)
    calib = ev_rec_data.calibration_rows(index.donor[rows])
    draw = ev_rec_data.draw_rows(index.donor[rows], k)
    rec = ev_rec_data.recording(view, "real")        # windows and ids, no theta
    names, bounds = ev_rec_data.axes_of(index.sidecar, p)

Runs in `envs.train` (numpy and the runner's bank readers).

CONTRACT (wave A): the signatures, shapes and dtypes are fixed by
specs/SPEC.md block 33.

Pure ASCII, LF only.
"""

import glob
import json
import os

import _evalpath  # noqa: F401

import numpy as np

import run_joint_arms as R
from ev_plan import RefuseError

# never defined again here (block 33)
from ev_data import calibration_rows, report_rows  # noqa: F401

# the ids every recording's dict carries, each (n,) int64 (`rep_group` is -1
# on every row of a bank without that array)
RECORDING_IDS = ("cls", "donor", "well", "batch", "subregion", "window_idx", "rep_group")


def _first_bad_theta(pattern):
    """"; first at shard <file>, row <i>, axis <k> (value <v>)" for the first non-finite
    theta of the shards matching `pattern`, in sorted order; "" when there is none."""
    for path in sorted(glob.glob(pattern)):
        try:
            with np.load(path) as z:
                th = np.asarray(z["theta"], dtype=np.float64)
        except Exception:                      # an unreadable shard: the contract's message stands
            continue
        bad = np.argwhere(~np.isfinite(th))
        if bad.size:
            i, k = (int(v) for v in bad[0][:2]) if th.ndim == 2 else (int(bad[0][0]), 0)
            return ("; first at shard %s, row %d of the shard, axis %d (value %r)"
                    % (os.path.basename(path), i, k, float(th[i, k] if th.ndim == 2 else th[i])))
    return ""


def sim_index(view):
    """(ShardIndex, paths) of the view's simulated bank (view.sim_glob, the
    shards sorted)."""
    if not glob.glob(view.sim_glob):
        raise RefuseError("the simulated bank has no shard: nothing matches %s" % view.sim_glob)
    try:
        return R.load_arm_index(view.sim_glob)
    except ValueError as exc:
        # a bank that breaks its contract (latent_bank.ContractError): a refusal that
        # names the bank and, for a non-finite label, where it is
        if type(exc).__name__ != "ContractError":
            raise
        raise RefuseError("the simulated bank %s breaks its contract: %s%s"
                          % (view.sim_glob, exc, _first_bad_theta(view.sim_glob)))


def draw_rows(donor, k):
    """(n,) bool: True on the first `k` rows of each donor, in the given
    order; every row when `k` is None. draw_rows(donor, 1) equals
    calibration_rows(donor). `donor` (n,) int; `k` None or an int >= 1."""
    donor = np.asarray(donor).ravel()
    n = donor.size
    if k is None:
        return np.ones(n, dtype=bool)
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("draw_rows: k must be None or an integer >= 1, got %r" % (k,))
    if n == 0:
        return np.zeros(0, dtype=bool)
    # each row's occurrence: its position among its own donor's rows, in the given order
    order = np.argsort(donor, kind="stable")
    sd = donor[order]
    start = np.ones(n, dtype=bool)
    start[1:] = sd[1:] != sd[:-1]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    occ = np.empty(n, dtype=np.int64)
    occ[order] = np.arange(n) - first
    return occ < int(k)


def _bank_paths(view, name):
    if name not in view.banks:
        raise RefuseError("no recording bank %r in this view (banks: %s)"
                          % (name, ", ".join(sorted(view.banks)) or "none"))
    paths = sorted(glob.glob(view.banks[name]))
    if not paths:
        raise RefuseError("recording bank %r has no shard: nothing matches %s"
                          % (name, view.banks[name]))
    return paths


def recording(view, name):
    """Recording bank `name` of the view, its rows in bank order (the shards
    sorted, read by run_joint_arms.concat_shards): dict(x (n, W) float32,
    each of RECORDING_IDS (n,) int64, contaminated (n,) bool, sidecar,
    paths). No `theta` key, whatever the bank holds (D-162). RefuseError for
    an unknown name, no shard, or a bank whose sidecar `arm` is not "R"."""
    paths = _bank_paths(view, name)
    with open(os.path.splitext(paths[0])[0] + ".json") as fh:
        arm = json.load(fh).get("arm")
    if arm != "R":
        raise RefuseError("recording bank %r (%s) is of arm %r, not R: a recording bank is an "
                          "arm-R bank" % (name, view.banks[name], arm))
    arrays, sidecar = R.concat_shards(paths)
    if sidecar.get("arm") != "R":
        raise RefuseError("recording bank %r (%s) is of arm %r, not R"
                          % (name, view.banks[name], sidecar.get("arm")))
    n = int(np.asarray(arrays["x"]).shape[0])
    out = {"x": np.asarray(arrays["x"], dtype=np.float32)}
    for k in RECORDING_IDS:
        if k == "rep_group" and k not in arrays:
            out[k] = np.full(n, -1, dtype=np.int64)
        else:
            out[k] = np.asarray(arrays[k], dtype=np.int64)
    out["contaminated"] = np.asarray(arrays["contaminated"], dtype=bool)
    out["sidecar"] = sidecar
    out["paths"] = paths
    return out


def axes_of(sidecar, p):
    """(names, bounds) of the p axes of a simulated bank, from its sidecar:
    `names` a list of p strings and `bounds` a (p, 2) float64 array [lo_k,
    hi_k] in each axis's declared coordinate, the sidecar's `param_names`
    and `bounds_theta` when both have p entries; otherwise the names
    `theta_0`, ..., `theta_{p-1}` and bounds None."""
    p = int(p)
    side = sidecar or {}
    names = side.get("param_names")
    bounds = side.get("bounds_theta")
    if (isinstance(names, list) and isinstance(bounds, list) and len(names) == p
            and len(bounds) == p):
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape == (p, 2):
            return [str(s) for s in names], b
    return ["theta_%d" % k for k in range(p)], None
