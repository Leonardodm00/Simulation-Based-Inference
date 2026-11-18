"""Example synthetic windows for the deck, made with the pipeline's own bench
generator (SBI `hpc/joint/stage1/bench_burst_provider.py`), its nuisance
model (`latent_nuisance.py`) and its gap (`latent_gap.py`).

Standalone illustration script, not pipeline code. It writes
`data/traces.npz` for the figure scripts:

- `cond_x` (3, W): one window per condition, at the condition's centre on the
  eight label axes (synth v4, D-132: s = 0.562743, the DSN generator's simplex
  centres on 8 axes, in the basis the cluster used: `centres.py`) and 0.5 on
  the two free axes, after the default nuisance;
- `well_clean`, `well_gap` (6, W): one donor's well, six windows, without and
  with R3's gap (pi = 0.3: range shift, drift, contamination), the same
  realisation and nuisance; `contaminated` (6,) marks the replaced window;
- `theta_cond` (3, 10), `fs`, `axes` (names).

Usage:
    python3 gen_traces.py <SBI checkout root> <output npz>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centres import SIGNS_CLUSTER, simplex_centres   # noqa: E402

# ---- configuration -------------------------------------------------------
FS = 100.0            # Hz, the bench's sampling rate (SPEC 2.4)
T_WIN = 180.0         # s, one window
N_NEURONS = 100       # units (SPEC 2.4)
S_SCALE = 0.562743    # v4 centre scale (D-132)
PI_GAP = 0.3          # R3's gap severity
SEED = 20261118       # base seed of this illustration
J_WELL = 6            # windows per well (arm R)


def main(sbi_root, out_path):
    stage1 = os.path.join(sbi_root, "hpc", "joint", "stage1")
    sys.path.insert(0, stage1)
    sys.path.insert(0, os.path.join(sbi_root, "hpc", "joint"))
    import bench_burst_provider as BP
    import latent_gap as LG
    import latent_nuisance as LN

    mu = simplex_centres(SIGNS_CLUSTER, scale=S_SCALE)   # (3, 8), row c = condition c
    d01 = np.linalg.norm(mu[0] - mu[1])
    print("centre distance %.6f (handoff: 0.441934)" % d01)

    provider = BP.load_bench_provider(label_idx=tuple(range(8)))   # v4: K_L = 0..7, K_F = 8, 9 (D-132)
    W = int(round(FS * T_WIN))
    nspec = LN.NuisanceSpec()
    t_win = np.arange(W) / FS

    theta_cond = np.full((3, 10), 0.5)
    theta_cond[:, :8] = mu
    cond_x = np.empty((3, W))
    for c in range(3):
        x = provider(theta_cond[c], 1, W, FS, N_NEURONS, SEED + c)   # (1, W)
        nu = LN.sample_nuisance(nspec, [0], [c], [0], SEED)[0]
        cond_x[c] = LN.apply_nuisance(nspec, x, nu, t_win)[0]

    # one well of a condition-0 donor, six windows, with and without the gap;
    # look for a realisation whose gap draws one contaminated window (p = 0.36)
    gspec = LG.GapSpec(modes=("range_shift", "drift", "contamination"), pi=PI_GAP)
    print("gap spec:", gspec.to_dict())
    shift = LG.free_axis_range_shift(gspec)
    t_well = (np.arange(J_WELL)[:, None] * T_WIN) + t_win[None, :]
    for k in range(200):
        rng = np.random.default_rng(SEED + 1000 + k)
        clean = provider(theta_cond[0], J_WELL, W, FS, N_NEURONS, SEED + 77)
        shifted = provider(theta_cond[0], J_WELL, W, FS, N_NEURONS, SEED + 77,
                           param_overrides={"free_axis_range_shift": shift})
        gap, contaminated = LG.apply_trace_gap(gspec, shifted, t_well, rng)
        if contaminated.sum() == 1 and not contaminated[0] and not contaminated[-1]:
            break
    nu = LN.sample_nuisance(nspec, [0], [5], [0], SEED)[0]
    well_clean = LN.apply_nuisance(nspec, clean, nu, t_well)
    well_gap = LN.apply_nuisance(nspec, gap, nu, t_well)
    print("contaminated window:", np.flatnonzero(contaminated), "after", k + 1, "draws")

    np.savez_compressed(out_path, cond_x=cond_x, theta_cond=theta_cond,
                        well_clean=well_clean, well_gap=well_gap,
                        contaminated=contaminated, fs=FS,
                        axes=np.array([a[0] for a in BP.BENCH_AXES]))
    print("wrote", out_path, cond_x.shape, well_gap.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
