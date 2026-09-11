#!/usr/bin/env python3
"""Generate a 3-class demonstration set from the bench generator.

GENERATION ONLY. Writes one .npz; every figure is produced by
`demo_classes_plot.py` reading that file, so a plotting change never
re-runs the simulator and a simulator change never silently alters a
figure's meaning.

What this is, and is not
------------------------
Not a bank: no shards, no sidecar, no contract digest -- use
`build_latent_bank.py` for anything downstream. This is a demonstration
that the class structure of eq. (7) survives the map
phi -> spikes -> IFR, i.e. that the three classes are visible in the
observable rather than only in the prior.

The classes come from the SAME sampler the bank uses
(`sample_prior`, eq. 7): the 7 class-bearing axes are drawn truncated-
normal around the class centre m_{c,k} with spread tau_ov, the 3 free
axes uniform on (0, 1) regardless of class. So class separation is a
property of the prior being pushed through the generator, not something
this script arranges.

Nuisance is OFF by default (`--nuisance` turns it on). With it off the
figures show dynamics alone; with it on they additionally show the
observation layer, including the negative values the additive baseline
produces -- see the [OPEN] note in bench_burst_provider.py about the
absolute nuisance scales against per-unit-mean traces.

Pure ASCII, LF only.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bench_burst_provider import (BENCH_AXES, BENCH_LABEL_IDX,        # noqa: E402
                                  load_bench_provider)
from latent_nuisance import NuisanceSpec, sample_nuisance             # noqa: E402
from latent_realisation import RealisationSpec                        # noqa: E402
from latent_sbi_simulator import (LatentSBISpec, sample_prior,        # noqa: E402
                                  simplex_centres, simulate_windows)


def build_spec(args):
    """The bench spec: 10 axes, 7 class-bearing, structural (spec S4)."""
    return LatentSBISpec(
        n_latent=len(BENCH_AXES), label_idx=BENCH_LABEL_IDX,
        class_centres=simplex_centres(args.n_classes, len(BENCH_LABEL_IDX)),
        tau_ov=args.tau_ov, n_windows_per_trace=args.n_windows,
        T_win=args.T_win, fs=args.fs, n_neurons=args.n_neurons,
        seed=args.seed)


def generate(spec, provider, n_per_class, base_seed, nuisance=False,
             n_background=0):
    """Simulate `n_per_class` traces for each class, plus a background.

    The BACKGROUND cohort is drawn uniformly on the full box [0, 1]^p --
    NOT from eq. (7) -- and carries class label -1. It exists so the
    t-SNE map has something to mean: "where the class-generating
    parameters fall in the full space" needs the full space to be
    populated, and needs it populated in the SAME way for the phi
    embedding and the x embedding. Simulating it (rather than drawing
    phi only) is what makes the x-space version of that question
    answerable at all.

    Returns
    -------
    out : dict of arrays, one ROW PER TRACE (not per window):
        x     (n, J, W) float64   the observable
        phi   (n, p)    float64   the generating coordinates, in [0, 1]^p
        cls   (n,)      int       class label; -1 marks a background draw
        rid   (n,)      uint64    realisation id
    Each trace is its own well, so each carries an independent
    realisation draw (S2.5c) -- the repeated-realisation structure D17
    could not get from the ANN campaign bank.
    """
    rng = np.random.default_rng(base_seed)
    C, n_pc = spec.n_classes, int(n_per_class)

    # Draw from eq. (7) until each class has n_pc members, then take the
    # first n_pc of each -- a rejection-free way to get a BALANCED demo
    # without altering the sampler (class is uniform in sample_prior, so
    # conditioning on the label leaves phi | c untouched).
    c_all, phi_all = sample_prior(spec, 40 * C * n_pc, rng)
    idx = np.concatenate([np.flatnonzero(c_all == k)[:n_pc]
                          for k in range(C)])
    if idx.size != C * n_pc:
        raise RuntimeError("prior draw did not fill every class; raise the "
                           "oversampling factor")
    cls, phi = c_all[idx], phi_all[idx]

    n_bg = int(n_background)
    if n_bg:
        eps = np.finfo(np.float64).eps
        phi_bg = rng.uniform(eps, 1.0 - eps, size=(n_bg, spec.n_latent))
        phi = np.vstack([phi, phi_bg])
        cls = np.concatenate([cls, np.full(n_bg, -1, dtype=cls.dtype)])

    n = cls.size
    J, W = spec.n_windows_per_trace, spec.W
    rspec = RealisationSpec(n_per_theta=1)
    nspec = NuisanceSpec() if nuisance else None
    if nuisance:
        nu = sample_nuisance(nspec, np.zeros(n, dtype=int), np.arange(n),
                             np.arange(n), base_seed + 101)
    x = np.empty((n, J, W), dtype=np.float64)
    rid = np.empty(n, dtype=np.uint64)
    for i in range(n):
        xi, info = simulate_windows(
            spec, phi[i], provider, donor=int(i), well=int(i), subregion=0,
            realisation_spec=rspec, base_seed=base_seed,
            nuisance_spec=nspec, nu_row=(nu[i] if nuisance else None),
            gap_spec=None,
            rng=np.random.default_rng(base_seed + 7919 * (i + 1)))
        x[i], rid[i] = xi, info["realisation_id"]
    return {"x": x, "phi": phi, "cls": cls, "rid": rid}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="demo_classes.npz")
    p.add_argument("--n-per-class", type=int, default=30)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--T-win", type=float, default=15.0)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--n-neurons", type=int, default=100)
    p.add_argument("--tau-ov", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-background", type=int, default=60,
                   help="uniform-box traces (class -1) giving the t-SNE map "
                        "its 'full space'; 0 disables")
    p.add_argument("--nuisance", action="store_true",
                   help="apply the observation layer (eq. N1); off by default")
    p.add_argument("--dsn-main-dir", default=None,
                   help="overrides $DSN_MAIN_DIR (compute_ifr_trace lives there)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = build_spec(args)
    provider = load_bench_provider(args.dsn_main_dir)
    out = generate(spec, provider, args.n_per_class, args.seed,
                   nuisance=args.nuisance, n_background=args.n_background)

    meta = dict(spec.to_dict())
    meta.update({"axis_names": [a[0] for a in BENCH_AXES],
                 "axis_lo": [a[1] for a in BENCH_AXES],
                 "axis_hi": [a[2] for a in BENCH_AXES],
                 "nuisance": bool(args.nuisance),
                 "scale_convention": "per_unit_mean"})
    np.savez_compressed(args.out, meta=np.array(repr(meta)), **out)

    print("wrote %s" % args.out)
    print("  traces           : %d (%d classes x %d, plus %d background)"
          % (out["cls"].size, args.n_classes, args.n_per_class,
             int(np.sum(out["cls"] < 0))))
    print("  x                : %r  (trace, window, sample)" % (out["x"].shape,))
    print("  phi              : %r in [0, 1]^%d" % (out["phi"].shape, spec.n_latent))
    print("  trace duration   : %.1f s  (J = %d windows of %.1f s at %.0f Hz)"
          % (spec.n_windows_per_trace * spec.T_win, spec.n_windows_per_trace,
             spec.T_win, spec.fs))
    print("  nuisance         : %s" % ("ON" if args.nuisance else "off"))
    print("  x range          : [%.4f, %.4f] counts/bin/unit"
          % (out["x"].min(), out["x"].max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
