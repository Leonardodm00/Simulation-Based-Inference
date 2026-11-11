"""The bench generator at given parameters, and the eight fixed window
statistics (block 19; specs/SPEC.md block 19, the posterior-predictive check).

    s = ev_simulate.fixed_stats(x)                          # (n, 8)
    x_sim = ev_simulate.windows_at(theta, latent_spec, seed, ncpus=8)

`windows_at` simulates ONE window per row of theta, each as its own well --
its own realisation seed and a fresh nuisance draw at ids of its own (batch,
donor and well), no gap: the simulated arm's world, which is the model's.
It composes the bank builder's own pieces (`simulate_windows`, the bench
provider, `sample_nuisance`), so a simulated window is a window of arm S at
that theta.

The statistics are the runner's `FixedStatsSummary` (arm A_ref's summary):
mean, SD, max, 90 % quantile, burst fraction, lag-1 autocorrelation,
skewness, roughness -- fixed, untrained, the same for every model.

Pure ASCII, LF only.
"""

import os
from multiprocessing import get_context

import _evalpath  # noqa: F401

import numpy as np
import torch

import run_joint_arms as R
from bench_burst_provider import load_bench_provider
from latent_nuisance import NuisanceSpec, sample_nuisance
from latent_realisation import RealisationSpec
from latent_sbi_simulator import LatentSBISpec, simulate_windows

STAT_NAMES = ("mean", "sd", "max", "q90", "burst_frac", "acf1", "skew", "roughness")


def fixed_stats(x, batch=512):
    """(n, 8) float32: the eight fixed statistics of each window of x (n, W)."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == 0:
        return np.zeros((0, len(STAT_NAMES)), dtype=np.float32)
    f = R.FixedStatsSummary()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], int(batch)):
            out.append(f(torch.as_tensor(x[i:i + int(batch)])).numpy())
    return np.concatenate(out).astype(np.float32)


def _spec_of(latent):
    """A one-window LatentSBISpec from a bank sidecar's `latent_spec`."""
    return LatentSBISpec(n_latent=int(latent["n_latent"]),
                         label_idx=tuple(latent["label_idx"]),
                         class_centres=np.asarray(latent["class_centres"], dtype=np.float64),
                         tau_ov=float(latent["tau_ov"]), n_windows_per_trace=1,
                         T_win=float(latent["T_win"]), fs=float(latent["fs"]),
                         n_neurons=int(latent["n_neurons"]))


_WORKER = {}


def _init(latent, dsn_main_dir):
    _WORKER["spec"] = _spec_of(latent)
    _WORKER["provider"] = load_bench_provider(dsn_main_dir)
    _WORKER["nspec"] = NuisanceSpec()
    _WORKER["rspec"] = RealisationSpec(n_per_theta=1)


def _one(job):
    """job = (i, theta_i, nu_i, base_seed): window of row i, float32 (W,)."""
    i, th, nu, base_seed = job
    x, _info = simulate_windows(_WORKER["spec"], th, _WORKER["provider"], donor=int(i),
                                well=int(i), subregion=0, realisation_spec=_WORKER["rspec"],
                                base_seed=int(base_seed), nuisance_spec=_WORKER["nspec"],
                                nu_row=nu, gap_spec=None,
                                rng=np.random.default_rng(int(base_seed) + 7919 * (int(i) + 1)))
    return x[0].astype(np.float32)


def windows_at(theta, latent, seed, ncpus=1, dsn_main_dir=None, chunksize=8):
    """(n, W) float32: one window of the bench generator per row of theta
    (n, n_latent), at base seed `seed`; `latent` a bank sidecar's latent_spec
    (T_win, fs, n_neurons). Deterministic in (theta, seed), whatever ncpus."""
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.shape[0]
    spec = _spec_of(latent)
    if n == 0:
        return np.zeros((0, spec.W), dtype=np.float32)
    ids = ["e%d" % i for i in range(n)]
    nu, _parts = sample_nuisance(NuisanceSpec(), ["b" + s for s in ids], ["d" + s for s in ids],
                                 ["w" + s for s in ids], base_seed=int(seed))
    jobs = [(i, theta[i], nu[i], int(seed)) for i in range(n)]
    if int(ncpus) <= 1:
        _init(latent, dsn_main_dir)
        out = [_one(j) for j in jobs]
    else:
        ctx = get_context("fork")
        with ctx.Pool(int(ncpus), initializer=_init, initargs=(latent, dsn_main_dir)) as pool:
            out = pool.map(_one, jobs, chunksize=int(chunksize))
    x = np.stack(out)
    if x.shape != (n, spec.W) or not np.all(np.isfinite(x)):
        raise ValueError("the simulated windows have shape %r or non-finite values" % (x.shape,))
    return x


def ncpus_default():
    """The job's cores (PBS's NCPUS) or 1."""
    v = os.environ.get("NCPUS", "").strip()
    return int(v) if v.isdigit() and int(v) >= 1 else 1
