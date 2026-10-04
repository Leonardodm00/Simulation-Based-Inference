#!/usr/bin/env python3
"""e1_numbers.py -- every [RAN] number of E1_THE_PROBLEM.md, recomputed.

Torch-free and numpy-free: E1 derives nothing new, so every number it tags
[RAN] is arithmetic on values that come from two places, and this script
keeps the two apart.

  * [REPO] values are READ here, from the repository at the freeze commit
    834eb41, with `git show` -- the DSN refit config of the r2 encoder
    (hpc/dsn/hpc/Config/refit_mea_joint_full_r2_l0_t82.json): its cohort
    block (w_size, gaussian_window, n_subsets, electrodes_per_subset,
    fs_raw, grid_width, the two class roots and class names), its window
    length and stride (data.window_s, data.train_stride_s) and its
    embedding size (backbone.embedding_size).
  * [KB] values are CONSTANTS below, each with the project document and
    section it is quoted from; they are not re-derived (the cohort's
    recording length, the bank's row counts, the parameter census, the
    kernel axis bounds, the gate's p-value).

Blocks:
  B1  the window grid: f_s, W, the smoothing width in bins, windows per
      subregion, per culture and in the cohort, raw samples per well,
      electrodes pooled per well against the grid
  B2  the parameter vector: 23 + 3, 17 + 9, the linear neuron/synapse axes,
      and the log-axis rule applied to the two bounds E1 quotes
  B3  the simulated domain: the share of rows the activity floor keeps
  B4  the gate's floor: the permutation count a 1/(B+1) floor of 0.001996
      implies
  B5  the class roots of the r2 config, counted

Run from hpc/joint/docs/tools (needs git and the repository's history):

    python e1_numbers.py

Pure ASCII, LF only.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # repo root

FREEZE = "834eb41"
R2_CONFIG = "hpc/dsn/hpc/Config/refit_mea_joint_full_r2_l0_t82.json"

# --------------------------------------------------------------------------- #
# [KB] constants, each with its source
# --------------------------------------------------------------------------- #
T_REC_S = 1200.0          # HPC_PATHS.md sec. 3b, manifest per well `T_rec 1200.0`;
                          # SBI_PIPELINE.md sec. 5 (T_rec = 1200 s)
N_WELLS = 35              # HPC_PATHS.md sec. 3b `n_wells 35`; SBI_PIPELINE.md sec. 5
N_SAMPLES_RAW = 12132108  # HPC_PATHS.md sec. 3b `n_samples_raw 12132108`
SIM_ROWS_RAW = 86251      # SBI_PIPELINE.md sec. 5, 6 (54 shards of rho1300v3, r2 export)
SIM_ROWS_MFR = 29616      # SBI_PIPELINE.md sec. 6 (MFR >= 0.1 Hz/electrode)
REAL_WINDOWS = 1890       # SBI_PIPELINE.md sec. 5
N_ACTIVE_NS = 23          # SBI_PIPELINE.md sec. 3; HPC_PATHS.md sec. 4a (23 active_indices)
N_KERNEL = 3              # p0_conn, d0_conn, beta_conn (SBI_PIPELINE.md sec. 3)
N_LN, N_LIN = 17, 9       # SBI_PIPELINE.md sec. 3 (p = 26: 17 ln, 9 linear)
P0_CONN_BOUNDS = (0.1, 1.0)     # EXTRACTOR_USAGE.md sec. 6.4
CONN_PROB_BOUNDS = (0.1, 0.6)   # EXTRACTOR_USAGE.md sec. 5.1 (rho1300 job_args)
P_GROUP = 0.001996        # SBI_PIPELINE.md sec. 8 (floor 1/(B+1))


def _git_show(path: str) -> str:
    out = subprocess.run(["git", "-C", _ROOT, "show", "%s:%s" % (FREEZE, path)],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out.stdout.decode("utf-8")


def _line_of(text: str, needle: str) -> int:
    for k, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return k
    raise KeyError(needle)


def log_axis(lo: float, hi: float) -> bool:
    """The rule as SBI_PIPELINE.md conv. (i) states it, WITHOUT the exception."""
    return lo > 0 and hi > 0 and math.log10(hi / lo) >= 1.0


def main() -> int:
    raw = _git_show(R2_CONFIG)
    cfg = json.loads(raw)
    coh = cfg["cohort"]
    data = cfg["data"]

    print("== source: %s @ %s" % (R2_CONFIG, FREEZE))
    for key in ("w_size", "gaussian_window", "n_subsets", "electrodes_per_subset",
                "fs_raw", "grid_width"):
        print("   cohort.%-22s = %r   (line %d)"
              % (key, coh[key], _line_of(raw, '"%s": %s' % (key, json.dumps(coh[key])))))
    for key in ("window_s", "train_stride_s"):
        print("   data.%-24s = %r   (line %d)"
              % (key, data[key], _line_of(raw, '"%s": %s' % (key, json.dumps(data[key])))))
    print("   backbone.embedding_size       = %r   (line %d)"
          % (cfg["backbone"]["embedding_size"], _line_of(raw, '"embedding_size": ')))

    # ---------------------------------------------------------------- B1
    print("\n== B1 the window grid")
    dt = float(coh["w_size"])
    sig = float(coh["gaussian_window"])
    t_win = float(data["window_s"])
    stride = float(data["train_stride_s"])
    n_sub = int(coh["n_subsets"])
    n_e = int(coh["electrodes_per_subset"])
    fs_raw = float(coh["fs_raw"])
    grid = int(coh["grid_width"])
    f_s = 1.0 / dt
    W = int(round(t_win * f_s))
    print("f_s = 1/dt = 1/%.2f = %.1f Hz" % (dt, f_s))
    print("W = round(T_win f_s) = round(%.1f x %.1f) = %d samples" % (t_win, f_s, W))
    print("sigma_sm / dt = %.2f / %.2f = %.1f bins" % (sig, dt, sig / dt))
    n_per_sub = int(math.floor((T_REC_S - t_win) / stride)) + 1
    left = T_REC_S - (n_per_sub - 1) * stride - t_win
    print("windows per subregion, back to back from t = 0: floor((T_rec - T_win)/stride) + 1"
          " = floor((%.0f - %.0f)/%.0f) + 1 = %d; %.0f s left over"
          % (T_REC_S, t_win, stride, n_per_sub, left))
    n_win = n_sub * n_per_sub
    print("n_win = %d subregions x %d windows = %d windows per culture" % (n_sub, n_per_sub, n_win))
    print("G x n_win = %d x %d = %d windows (KB: %d)" % (N_WELLS, n_win, N_WELLS * n_win, REAL_WINDOWS))
    assert N_WELLS * n_win == REAL_WINDOWS
    print("units = G x n_subsets = %d x %d = %d subregion traces" % (N_WELLS, n_sub, N_WELLS * n_sub))
    print("raw samples per well = fs_raw x T_rec = %.2f x %.0f = %.2f -> %d (KB: %d)"
          % (fs_raw, T_REC_S, fs_raw * T_REC_S, int(round(fs_raw * T_REC_S)), N_SAMPLES_RAW))
    assert int(round(fs_raw * T_REC_S)) == N_SAMPLES_RAW
    pooled = n_sub * n_e
    print("electrodes pooled per well = %d x %d = %d of %d x %d = %d grid sites (%.2f %%)"
          % (n_sub, n_e, pooled, grid, grid, grid * grid, 100.0 * pooled / (grid * grid)))

    # ---------------------------------------------------------------- B2
    print("\n== B2 the parameter vector")
    d_theta = N_ACTIVE_NS + N_KERNEL
    print("d_theta = %d neuron/synapse + %d kernel = %d" % (N_ACTIVE_NS, N_KERNEL, d_theta))
    assert d_theta == N_LN + N_LIN == 26
    ns_lin = N_LIN - N_KERNEL
    print("17 ln + 9 linear = %d; the 3 kernel axes are linear, so the neuron/synapse block"
          " is %d ln + %d linear = %d" % (N_LN + N_LIN, N_LN, ns_lin, N_LN + ns_lin))
    assert N_LN + ns_lin == N_ACTIVE_NS
    for name, (lo, hi) in (("p0_conn", P0_CONN_BOUNDS), ("conn_prob", CONN_PROB_BOUNDS)):
        dec = math.log10(hi / lo)
        print("rule without the exception on %s [%.1f, %.1f]: decades = log10(%.1f/%.1f) = %.6f"
              " -> %s" % (name, lo, hi, hi, lo, dec, "ln" if log_axis(lo, hi) else "linear"))

    # ---------------------------------------------------------------- B3
    print("\n== B3 the simulated domain")
    kept = SIM_ROWS_MFR / SIM_ROWS_RAW
    print("MFR floor keeps %d / %d = %.4f (%.1f %%); drops %.1f %%"
          % (SIM_ROWS_MFR, SIM_ROWS_RAW, kept, 100 * kept, 100 * (1 - kept)))
    print("real windows kept: %d / %d" % (REAL_WINDOWS, REAL_WINDOWS))

    # ---------------------------------------------------------------- B4
    print("\n== B4 the gate's floor")
    b_impl = 1.0 / P_GROUP - 1.0
    print("1/p_grp - 1 = 1/%.6f - 1 = %.2f; 1/(500 + 1) = %.6f"
          % (P_GROUP, b_impl, 1.0 / 501.0))

    # ---------------------------------------------------------------- B5
    print("\n== B5 the class roots of the r2 config")
    roots = coh["class_roots"]
    names = coh["class_names"]
    for key in sorted(roots):
        folders = [r.rstrip("/").split("/Deep_bio/")[-1] for r in roots[key]]
        print("class key %r -> class_names[%s] = %r: %d roots %s"
              % (key, key, names[int(key)], len(folders), folders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
