#!/usr/bin/env python3
"""Stage E of the bench burst model: the BurstProvider adapter.

Wires the three pieces together, per STAGE_A_BENCH_GENERATOR_SPEC_v1.md S6:

    phi --(affine map, S2/S4)--> BenchBurstParams          this module, new
    BenchBurstParams --(bench_burst_generator)--> spikes   Stages B-D, ours
    spikes --(gbd.compute_ifr_trace, UNCHANGED)--> IFR     imported from
                                                           DSN_MAIN_DIR
    IFR --(O-2 normalisation, windowing)--> (J, W)         this module, new

The IFR step is imported rather than reimplemented because its constants
(Delta_t, sigma_sm) are hard fields of the sim/real parity contract
(EXTRACTOR_USAGE.md S4); a second copy of that code is the failure the
contract exists to prevent (spec S6). Spike generation carries no such
constraint and has no code lineage to Deep-Summary-Network (D-A3).

O-2, resolved here: the scale convention is the PER-UNIT MEAN
-----------------------------------------------------------------
`compute_ifr_trace` returns the smoothed POPULATION count per bin, summed
over all N spike trains, undivided. The real pipeline's observable is a
per-electrode mean, R_norm[k] = R_tilde[k] / n_e for each fixed bin k
(EXTRACTOR_USAGE.md S4.1 eq. 3), and the plan declares that convention as
the parity target (plan v0.6, Stage 6: "Scale parity stays the
per-electrode-mean convention"). The bench has no electrodes and no
detection model; its pooled units ARE the N neurons. This provider
therefore divides the trace by `n_neurons` -- the structural analogue of
eq. (3) with the unit being a neuron -- so that

    x[j, s] = R_tilde[s] / N          (per-unit mean, counts per bin)

and the time-mean of x * fs estimates the per-neuron MFR of spec eq. (14)
(`bench_burst_generator.expected_mfr`), which the smoke test asserts.
The divisor is recorded in the sidecar as scale_convention
"per_unit_mean" by build_latent_bank. A bench unit (one neuron) and a
real unit (one electrode) are different populations; comparability across
that boundary is a parity-contract question, never an implicit factor.
`DSNBurstProvider` deliberately retains the undivided sum (its smoke test
S4 asserts bit-for-bit agreement with the raw primitives, and a DSN bank
can never pool with a bench bank -- different n_latent, different
contract digest); its sidecar record is corrected to "sum_over_units".

[OPEN] O-2 x nuisance-scale interaction, measured, not resolved here:
NuisanceSpec's `baseline` and `drift_amp` are ABSOLUTE ("in units of x",
combined sd ~ 0.0245 across levels). Against per-unit-mean traces
(~ 0.02-0.06 counts/bin at mid-box rates) that is an order-one relative
perturbation; against the old sum-scale traces it was ~ 0.6%, i.e.
effectively off. The default scales are therefore only now acting at
what appears to be their intended (real-scale) strength -- consistent
with the sidecar's long-declared per-electrode-mean intent -- but their
calibration provenance is undocumented, so this is recorded rather than
retuned. Negative x_obs values after the additive nuisance are expected
and deliberate (the map must stay invertible; see latent_nuisance).

Pure ASCII, LF only.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bench_burst_generator import BenchBurstParams, generate_spike_times  # noqa: E402
from latent_sbi_simulator import BurstProvider, load_dsn_modules          # noqa: E402

__all__ = [
    "BENCH_AXES", "BENCH_LABEL_IDX", "BENCH_FREE_IDX",
    "phi_to_bench_params", "BenchBurstProvider", "load_bench_provider",
]

# ---------------------------------------------------------------------------
# The axis table, spec S4. Order IS the phi coordinate order (0-based here;
# the spec numbers axes 1..10). Ranges are the [L_k, U_k] of the affine map
# a_k = L_k + phi_k (U_k - L_k); every axis is parameterised in the
# interpretable quantity (spec convention ii), so the map is linear for all.
# ---------------------------------------------------------------------------

BENCH_AXES = (
    # (name,                 L_k,   U_k)          units          spec axis
    ("burst_rate",           0.10,  0.40),      # bursts/s            1
    ("irregularity",         0.30,  0.90),      # dimensionless       2
    ("ibi_cv",               0.25,  1.00),      # dimensionless       3
    ("n_fragments",          1.0,   4.0),       # count               4
    ("intraburst_rate",      60.0,  140.0),     # spikes/s            5
    ("participation_mean",   0.45,  0.80),      # probability         6
    ("burst_duration",       0.15,  0.35),      # s                   7
    ("fragment_duty",        0.35,  1.00),      # dimensionless       8
    ("participation_kappa",  3.0,   15.0),      # dimensionless       9
    ("background",           0.01,  0.06),      # spikes/s/neuron    10
)

BENCH_LABEL_IDX = (0, 1, 2, 3, 4, 5, 6)   # class-bearing, spec S4
BENCH_FREE_IDX = (7, 8, 9)                # label-irrelevant, spec S4


def phi_to_bench_params(phi, n_neurons, duration_s, overlap="merge",
                        free_axis_shift=0.0):
    """The affine map of spec S2: phi in [0, 1]^10 -> BenchBurstParams.

    `free_axis_shift` implements gap perturbation (a) with the same
    semantics as `DSNBurstProvider.burst_params`: each FREE axis range
    [L_k, U_k] is displaced by that fraction of its own width BEFORE the
    map, so the same phi maps to physics the unshifted prior box cannot
    reach, while every recorded coordinate stays in [0, 1]. Class-bearing
    axes are never shifted (latent_gap, perturbation (a)).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if phi.size != len(BENCH_AXES):
        raise ValueError("phi must have %d entries, got %d"
                         % (len(BENCH_AXES), phi.size))
    a = np.empty(len(BENCH_AXES), dtype=np.float64)
    shift = float(free_axis_shift)
    for k, (_name, lo, hi) in enumerate(BENCH_AXES):
        if shift and k in BENCH_FREE_IDX:
            width = hi - lo
            lo, hi = lo + shift * width, hi + shift * width
        # convex-combination form: exact at BOTH endpoints, unlike
        # lo + phi * (hi - lo), which misses U_k by 1 ulp (e.g.
        # 0.10 + (0.40 - 0.10) != 0.40 in binary64). Caught by E1b.
        a[k] = lo * (1.0 - phi[k]) + hi * phi[k]
    return BenchBurstParams(
        lambda_b=float(a[0]), sigma_d=float(a[1]), ibi_cv=float(a[2]),
        n_frag_mean=float(a[3]), lambda_burst=float(a[4]),
        p_mean=float(a[5]), d_med=float(a[6]), duty=float(a[7]),
        kappa=float(a[8]), lambda_bg=float(a[9]),
        n_neurons=int(n_neurons), duration_s=float(duration_s),
        overlap=overlap)


class BenchBurstProvider(BurstProvider):
    """The Stage E provider: bench spikes, DSN IFR, per-unit-mean scale.

    Satisfies the BurstProvider protocol: a PURE function of its arguments
    (same arguments, same bytes -- all randomness flows through `seed`), and
    `seed` is where the realisation enters (S2.5c; smoke test property S10).
    For the bench generator the "realisation" is the spike-pattern draw at
    fixed phi -- there is no connectivity graph -- so distinct seeds at one
    phi are exactly the repeated realisations D17 could not get from the
    ANN campaign bank.

    Parameters
    ----------
    gbd : module -- the DSN's `generate_burst_data`, imported from
        DSN_MAIN_DIR by `load_dsn_modules`. Only `compute_ifr_trace` and
        `BurstParams` are used, both unchanged (spec S6).
    gaussian_window : float [s] -- IFR smoothing sd sigma_sm. Default 0.04,
        the DSN generator's own default (the value the DSN bench runs at);
        recorded per bank via the sidecar's latent_spec + this provider's
        source digest. w_size (Delta_t) is NOT a constructor parameter: it
        is 1/fs from each call, matching DSNBurstProvider, so the
        "generator's bin width is authoritative" rule is unchanged.
    overlap : str -- envelope overlap policy, "merge" (O-5 default) or
        "clamp".
    """

    #: extra IFR bins simulated beyond n_windows * W. int(T / Delta_t)
    #: inside compute_ifr_trace can land one sample short of the target in
    #: float arithmetic (the DSN provider raises on exactly this); padding
    #: the simulated duration and slicing x[:need] removes that error path
    #: deterministically. The discarded tail changes no kept sample.
    PAD_BINS = 2

    def __init__(self, gbd, gaussian_window=0.04, overlap="merge"):
        self.gbd = gbd
        self.gaussian_window = float(gaussian_window)
        if not (self.gaussian_window > 0):
            raise ValueError("gaussian_window must be > 0, got %r"
                             % (self.gaussian_window,))
        self.overlap = str(overlap)

    def __call__(self, phi, n_windows, W, fs, n_neurons, seed,
                 param_overrides=None):
        ov = dict(param_overrides or {})
        shift = float(ov.pop("free_axis_range_shift", 0.0))
        if ov:
            raise NotImplementedError(
                "gap perturbation emits %r, which BenchBurstParams does not "
                "accept. Bind these keys in latent_gap.param_overrides "
                "before enabling the mode that produces them."
                % (sorted(ov),))

        n_windows, W, n_neurons = int(n_windows), int(W), int(n_neurons)
        dt = 1.0 / float(fs)
        need = n_windows * W
        duration_s = (need + self.PAD_BINS) * dt

        params = phi_to_bench_params(phi, n_neurons=n_neurons,
                                     duration_s=duration_s,
                                     overlap=self.overlap,
                                     free_axis_shift=shift)
        rng = np.random.default_rng(int(seed) % (2 ** 63))
        spikes, _stats = generate_spike_times(params, rng)

        ifr_params = self.gbd.BurstParams(
            n_neurons=n_neurons, duration_s=duration_s, w_size=dt,
            gaussian_window=self.gaussian_window)
        x, fs_out = self.gbd.compute_ifr_trace(spikes, ifr_params)

        if abs(float(fs_out) - float(fs)) > 1e-9:
            raise ValueError("compute_ifr_trace returned f_s = %r, expected %r"
                             % (fs_out, fs))
        if x.shape[0] < need:
            raise ValueError(
                "compute_ifr_trace returned %d samples, need %d even after "
                "padding %d bins" % (x.shape[0], need, self.PAD_BINS))

        # O-2: per-unit mean -- the structural analogue of EXTRACTOR_USAGE
        # S4.1 eq. (3), with the pooled unit being a neuron. See the module
        # docstring for why this and not the sum, and not / n_e.
        x = np.asarray(x[:need], dtype=np.float64) / float(n_neurons)

        # Disjoint windows from s = 0 with stride W -- the same rule as
        # DSNBurstProvider, sim_observable.window_trace and MEAWindowDataset.
        return x.reshape(n_windows, W)


def load_bench_provider(dsn_main_dir=None, gaussian_window=0.04,
                        overlap="merge"):
    """Build a BenchBurstProvider, mirroring `load_dsn_provider`.

    DSN_MAIN_DIR is still required -- not for the generator (ours), but for
    `compute_ifr_trace`, which is imported unchanged for parity (spec S6).
    """
    _lbg, gbd = load_dsn_modules(dsn_main_dir)
    return BenchBurstProvider(gbd, gaussian_window=gaussian_window,
                              overlap=overlap)
