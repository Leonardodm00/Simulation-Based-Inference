"""Spike raster and population trace of one simulated stretch, for the deck's
"how the simulator works" figure. Uses the pipeline's bench generator
(SBI hpc/joint/stage1: `phi_to_bench_params`, `generate_spike_times`) and the
DSN tree's IFR (`compute_ifr_trace`), as `BenchBurstProvider` does.

Standalone illustration script, not pipeline code. Writes `data/raster.npz`:
`t_spk`, `i_spk` (spike times in s and neuron indices), `x` (per-unit IFR at
100 Hz, the provider's normalisation), `fs`, `T`, `phi`.

Usage:
    python3 gen_raster.py <SBI checkout root> <output npz>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centres import SIGNS_CLUSTER, simplex_centres   # noqa: E402

FS = 100.0          # Hz
T = 40.0            # s of simulated activity shown
N_NEURONS = 100
COND = 2            # condition 3 (index 2): the most bursts per minute
SEED = 20261118


def main(sbi_root, out_path):
    sys.path.insert(0, os.path.join(sbi_root, "hpc", "joint", "stage1"))
    sys.path.insert(0, os.path.join(sbi_root, "hpc", "joint"))
    import bench_burst_provider as BP
    from bench_burst_generator import generate_spike_times
    from latent_sbi_simulator import load_dsn_modules

    _lbg, gbd = load_dsn_modules()
    phi = np.full(10, 0.5)
    phi[:8] = simplex_centres(SIGNS_CLUSTER)[COND]
    params = BP.phi_to_bench_params(phi, n_neurons=N_NEURONS, duration_s=T + 0.02,
                                    free_idx=(8, 9))
    rng = np.random.default_rng(SEED)
    spikes, _stats = generate_spike_times(params, rng)
    ifr = gbd.BurstParams(n_neurons=N_NEURONS, duration_s=T + 0.02, w_size=1.0 / FS,
                          gaussian_window=0.04)
    x, fs_out = gbd.compute_ifr_trace(spikes, ifr)
    assert abs(fs_out - FS) < 1e-9
    x = np.asarray(x[: int(T * FS)], float) / N_NEURONS
    t_spk = np.concatenate(spikes)
    i_spk = np.concatenate([np.full(len(s), i) for i, s in enumerate(spikes)])
    keep = t_spk < T
    np.savez_compressed(out_path, t_spk=t_spk[keep], i_spk=i_spk[keep], x=x, fs=FS, T=T,
                        phi=phi)
    print("params:", params)
    print("spikes: %d in %.0f s; x mean %.4f, max %.3f" % (keep.sum(), T, x.mean(), x.max()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
