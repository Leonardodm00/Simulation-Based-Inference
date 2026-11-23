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
specs/SPEC.md block 33. The functions are stubs.

Pure ASCII, LF only.
"""

import _evalpath  # noqa: F401

# never defined again here (block 33)
from ev_data import calibration_rows, report_rows  # noqa: F401

# the ids every recording's dict carries, each (n,) int64 (`rep_group` is -1
# on every row of a bank without that array)
RECORDING_IDS = ("cls", "donor", "well", "batch", "subregion", "window_idx", "rep_group")


def sim_index(view):
    """(ShardIndex, paths) of the view's simulated bank (view.sim_glob, the
    shards sorted)."""
    raise NotImplementedError("block 33")


def draw_rows(donor, k):
    """(n,) bool: True on the first `k` rows of each donor, in the given
    order; every row when `k` is None. draw_rows(donor, 1) equals
    calibration_rows(donor). `donor` (n,) int; `k` None or an int >= 1."""
    raise NotImplementedError("block 33")


def recording(view, name):
    """Recording bank `name` of the view, its rows in bank order (the shards
    sorted, read by run_joint_arms.concat_shards): dict(x (n, W) float32,
    each of RECORDING_IDS (n,) int64, contaminated (n,) bool, sidecar,
    paths). No `theta` key, whatever the bank holds (D-162). RefuseError for
    an unknown name, no shard, or a bank whose sidecar `arm` is not "R"."""
    raise NotImplementedError("block 33")


def axes_of(sidecar, p):
    """(names, bounds) of the p axes of a simulated bank, from its sidecar:
    `names` a list of p strings and `bounds` a (p, 2) float64 array [lo_k,
    hi_k] in each axis's declared coordinate, the sidecar's `param_names`
    and `bounds_theta` when both have p entries; otherwise the names
    `theta_0`, ..., `theta_{p-1}` and bounds None."""
    raise NotImplementedError("block 33")
