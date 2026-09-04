"""Nuisance latent nu: sampling with batch/donor/well nesting, and the
observation-level transform it induces on an IFR trace.

Scope (plan v0.6, S2.6 Lever 3 and S4.0). nu acts AFTER the dynamics, on the
observation map only. That is what makes eq. (6) nuisance-invariant and what
distinguishes nu from the connectivity realisation G, which acts ON the
dynamics and is handled in latent_realisation.py instead.

Modelling choice, stated rather than hidden
-------------------------------------------
The bench observable is a pooled instantaneous-firing-rate (IFR) trace, not a
spike raster. On a rate trace every nuisance component in the plan's list
reduces to an affine, strictly monotone map:

    gain              : multiplicative
    baseline offset   : additive
    detection threshold shift : multiplicative attenuation exp(-kappa * dthr),
                        since raising the threshold removes a roughly constant
                        FRACTION of the detected events at fixed waveform
                        amplitude distribution
    electrode dropout : multiplicative, (n_active / n_e), because the observable
                        is a per-electrode MEAN (the scale convention in the
                        pipeline, D5)
    slow drift        : additive, a smooth function of time

so the whole transform is

    x'(t) = A(nu) * x(t) + b(nu) + drift(nu, t),        A(nu) > 0.        (N1)

(N1) is invertible exactly, which is what smoke test S9 checks. A rectifying
or clipping nuisance would not be, and would also not be a nuisance in the
sense the plan needs: it would destroy information rather than relabel it.

Pure ASCII, LF only. numpy only.
"""

import hashlib

import numpy as np

# Component order is part of the contract: nu arrays are (n_units, N_COMPONENTS)
# in exactly this order, and the sidecar records it.
NU_COMPONENTS = (
    "log_gain",        # log multiplicative gain
    "baseline",        # additive offset, in units of x
    "dthr",            # detection-threshold shift, arbitrary units
    "dropout_logit",   # logit of the fraction of electrodes lost
    "drift_amp",       # amplitude of the slow drift, in units of x
)
N_COMPONENTS = len(NU_COMPONENTS)

# Levels of the nesting, outermost first. A well's nu is the sum of the level
# contributions, so variance adds and each level is identifiable by design.
NU_LEVELS = ("batch", "donor", "well")


class NuisanceSpec(object):
    """Scales of the per-level Gaussian contributions to nu.

    Parameters
    ----------
    scales : dict level -> array (N_COMPONENTS,)
        Standard deviation of each component at each level. A zero switches
        that component off at that level, which is how the controls in S4.5
        are built.
    kappa : float
        Sensitivity of detected rate to the threshold shift, in (N1).
    n_electrodes : int
        n_e; dropout removes a fraction of these.
    drift_period_s : float
        Period of the slow drift, in seconds. Long relative to T_win by
        construction, or it is not a drift.
    """

    def __init__(self, scales=None, kappa=0.5, n_electrodes=9,
                 drift_period_s=600.0, dropout_logit0=-4.0):
        if scales is None:
            scales = {
                "batch": np.array([0.10, 0.02, 0.10, 0.30, 0.02]),
                "donor": np.array([0.05, 0.01, 0.05, 0.15, 0.01]),
                "well": np.array([0.05, 0.01, 0.05, 0.15, 0.01]),
            }
        self.scales = {}
        for lvl in NU_LEVELS:
            s = np.asarray(scales[lvl], dtype=np.float64).ravel()
            if s.size != N_COMPONENTS:
                raise ValueError("scales[%r] must have %d entries"
                                 % (lvl, N_COMPONENTS))
            if np.any(s < 0):
                raise ValueError("scales must be non-negative")
            self.scales[lvl] = s
        self.kappa = float(kappa)
        self.n_electrodes = int(n_electrodes)
        self.drift_period_s = float(drift_period_s)
        # Baseline logit of the lost-electrode fraction.
        #
        # [CORRECTION] Without this, nu = 0 gave a logit of 0, hence a lost
        # fraction of 1/2 and a gain factor of 4/9: the ZERO nuisance vector
        # was not the identity map, and the mean electrode loss across the
        # cohort was 48%. Every nuisance component must be centred so that
        # nu = 0 leaves the observable untouched, or the "nuisance floor"
        # measured in S2.6 Lever 3 is measuring a fixed 55% attenuation as
        # well as the variation, and the invertibility test passes while the
        # transform is not the identity at its own centre.
        self.dropout_logit0 = float(dropout_logit0)

    def to_dict(self):
        return {
            "components": list(NU_COMPONENTS),
            "levels": list(NU_LEVELS),
            "scales": {k: self.scales[k].tolist() for k in NU_LEVELS},
            "kappa": self.kappa,
            "n_electrodes": self.n_electrodes,
            "drift_period_s": self.drift_period_s,
            "dropout_logit0": self.dropout_logit0,
        }


def _level_rng(base_seed, level, key):
    """Deterministic RNG for one (level, key) pair.

    Derived by hashing rather than by arithmetic on the seed, so that adding a
    level or renaming a key cannot silently alias two units onto one stream.
    """
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("ascii"))
    h.update(b"|")
    h.update(str(level).encode("ascii"))
    h.update(b"|")
    h.update(str(key).encode("ascii"))
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "big"))


def sample_nuisance(spec, batch_ids, donor_ids, well_ids, base_seed):
    """Draw nu for each (batch, donor, well) triple, with the nesting.

    Parameters
    ----------
    spec : NuisanceSpec
    batch_ids, donor_ids, well_ids : array-like of hashable, same length n
        One entry per unit (well). Units sharing a batch share that batch's
        contribution exactly; likewise for donor.
    base_seed : int

    Returns
    -------
    nu : (n, N_COMPONENTS) float64
    parts : dict level -> (n, N_COMPONENTS), the additive decomposition, so a
        test can check that variance adds and a diagnostic can attribute.
    """
    batch_ids = list(batch_ids)
    donor_ids = list(donor_ids)
    well_ids = list(well_ids)
    n = len(well_ids)
    if not (len(batch_ids) == len(donor_ids) == n):
        raise ValueError("batch_ids, donor_ids, well_ids must be the same length")

    ids_by_level = {"batch": batch_ids, "donor": donor_ids, "well": well_ids}
    parts = {}
    for lvl in NU_LEVELS:
        s = spec.scales[lvl]
        cache = {}
        out = np.empty((n, N_COMPONENTS), dtype=np.float64)
        for i, key in enumerate(ids_by_level[lvl]):
            if key not in cache:
                rng = _level_rng(base_seed, lvl, key)
                cache[key] = rng.standard_normal(N_COMPONENTS) * s
            out[i] = cache[key]
        parts[lvl] = out

    nu = parts["batch"] + parts["donor"] + parts["well"]
    return nu, parts


def nuisance_affine(spec, nu_row, t_seconds):
    """The (A, b_total) of eq. (N1) for one unit, at the given time grid.

    Returns
    -------
    A : float, > 0
    b : (T,) float64, the additive part including the drift
    """
    nu_row = np.asarray(nu_row, dtype=np.float64).ravel()
    if nu_row.size != N_COMPONENTS:
        raise ValueError("nu_row must have %d entries" % N_COMPONENTS)
    log_gain, baseline, dthr, dropout_logit, drift_amp = nu_row

    frac_lost = 1.0 / (1.0 + np.exp(-(spec.dropout_logit0 + dropout_logit)))
    # At least one electrode always survives, or the observable is undefined.
    n_active = max(1, int(round(spec.n_electrodes * (1.0 - frac_lost))))
    dropout_factor = n_active / float(spec.n_electrodes)

    A = float(np.exp(log_gain) * np.exp(-spec.kappa * dthr) * dropout_factor)
    if not (A > 0.0):
        raise FloatingPointError("non-positive gain; nu is out of range")

    t = np.asarray(t_seconds, dtype=np.float64)
    drift = drift_amp * np.sin(2.0 * np.pi * t / spec.drift_period_s)
    b = baseline + drift
    return A, b


def apply_nuisance(spec, x, nu_row, t_seconds):
    """x' = A x + b, eq. (N1). x is (..., T); the last axis is time."""
    A, b = nuisance_affine(spec, nu_row, t_seconds)
    return np.asarray(x, dtype=np.float64) * A + b


def invert_nuisance(spec, x_obs, nu_row, t_seconds):
    """The exact inverse of apply_nuisance. Smoke test S9 uses it."""
    A, b = nuisance_affine(spec, nu_row, t_seconds)
    return (np.asarray(x_obs, dtype=np.float64) - b) / A
