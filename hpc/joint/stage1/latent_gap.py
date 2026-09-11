"""The simulation gap: the four perturbations of plan S4.0, with severity pi.

The pseudo-real arm R is drawn from a perturbed generator p_pi; the simulated
arm S from p_0. At pi = 0 the two arms must be indistinguishable (smoke test
S7), and the gap must grow monotonically in pi (S8). Everything here is a
deliberate misspecification, so each perturbation states which of the plan's
four it implements and at which layer it acts.

Layers, because they are not interchangeable
--------------------------------------------
(a) free-axis range shift  -> AXIS layer: shifts the PHYSICAL range [a_k, b_k]
    of the free axes, so the same phi in [0, 1] maps to physical values partly
    outside the ones the simulated prior box can produce. The flow is then
    asked to explain data from off its prior-predictive support.

    [CORRECTION, found while wiring Stage 3] An earlier version shifted phi
    itself. That is wrong twice over. It makes phi leave [0, 1], which the DSN
    generator refuses outright (`LatentAxis.value` raises), and it would make
    the RECORDED theta lie outside the prior box, so `prior_log_prob` returns
    -inf and the Monte Carlo floor L_0 -- hence every information gain on the
    pseudo-real arm -- becomes undefined. Shifting the range keeps every
    recorded coordinate inside the box while moving the physics, which is what
    "a range shifted partly outside the simulated prior box" actually means.
(b) heavy-tailed burst durations -> GENERATOR layer: REMOVED 2026-09-09.
    It was never bound to the generator: `param_overrides` emitted placeholder
    keys, and `DSNBurstProvider.__call__` raises NotImplementedError on any key
    it does not recognise -- so selecting it crashed the job rather than
    degrading. An unimplemented mode listed as available is worse than an
    absent one. To reinstate it, bind real `BurstParams` field names first,
    THEN re-add the mode here.
(c) slow background drift  -> TRACE layer: additive, absent from p_0.
(d) contaminated windows   -> TRACE layer: a fraction of windows replaced.

Pure ASCII, LF only. numpy only.
"""

import numpy as np

GAP_MODES = ("range_shift", "drift", "contamination")


class GapSpec(object):
    """Which perturbations are active and how hard.

    Parameters
    ----------
    modes : tuple of str, subset of GAP_MODES
    pi : float in [0, 1]
        Severity. pi = 0 must reduce every perturbation to the identity, and
        that is asserted, not assumed (smoke test S7).
    shift_max : float
        At pi = 1, the free-axis sampling range is shifted by this much in phi
        units (phi lives in (0,1)), so part of it falls outside the box.
    drift_amp_max, drift_period_s : float
        The (c) drift. Distinct from the nuisance drift in latent_nuisance.py:
        that one is a nuisance present in BOTH arms, this one is a
        misspecification present only in R.
    contam_frac_max : float
        At pi = 1, this fraction of windows is contaminated.
    """

    def __init__(self, modes=("range_shift",), pi=0.0, shift_max=0.35,
                 drift_amp_max=0.5, drift_period_s=120.0,
                 contam_frac_max=0.20):
        bad = [m for m in modes if m not in GAP_MODES]
        if bad:
            raise ValueError("unknown gap modes: %r" % (bad,))
        self.modes = tuple(modes)
        self.pi = float(pi)
        if not (0.0 <= self.pi <= 1.0):
            raise ValueError("pi must be in [0, 1]")
        self.shift_max = float(shift_max)
        self.drift_amp_max = float(drift_amp_max)
        self.drift_period_s = float(drift_period_s)
        self.contam_frac_max = float(contam_frac_max)

    @property
    def active(self):
        """No mode does anything at pi = 0. This is the S7 guarantee."""
        return self.pi > 0.0 and len(self.modes) > 0

    def to_dict(self):
        return {
            "modes": list(self.modes),
            "pi": self.pi,
            "shift_max": self.shift_max,
            "drift_amp_max": self.drift_amp_max,
            "drift_period_s": self.drift_period_s,
            "contam_frac_max": self.contam_frac_max,
        }


# ---------------------------------------------------------------------------
# (a) parameter layer
# ---------------------------------------------------------------------------

def free_axis_range_shift(spec):
    """The fraction of its own width each FREE axis range is displaced by.

    0.0 when the mode is inactive, so a provider that always reads this key
    behaves correctly at pi = 0 without a branch.
    """
    if not spec.active or "range_shift" not in spec.modes:
        return 0.0
    return float(spec.pi * spec.shift_max)


# ---------------------------------------------------------------------------
# (b) generator layer -- intentionally empty, see the module docstring
# ---------------------------------------------------------------------------

def param_overrides(spec):
    """Overrides for the burst generator: perturbation (a) only.

    Returns an empty dict when inactive, so a provider that ignores the
    argument entirely still behaves correctly at pi = 0. Perturbation (b) was
    removed; this function must never emit a key no provider consumes.
    """
    out = {}
    shift = free_axis_range_shift(spec)
    if shift:
        out["free_axis_range_shift"] = shift
    return out


# ---------------------------------------------------------------------------
# (c) and (d) trace layer
# ---------------------------------------------------------------------------

def apply_trace_gap(spec, x, t_seconds, rng):
    """Apply (c) drift and (d) contamination to a window block.

    Parameters
    ----------
    x : (J, T) float64, J windows of one trace
    t_seconds : (T,) or (J, T) float64 -- ABSOLUTE time. Pass (J, T) built
        from the per-window absolute grids so the drift is one continuous
        slow function across the whole trace.

        [CORRECTION] This was called once per window with a fresh random
        phase each time, so the "slow drift" restarted at a random phase at
        every window boundary: a within-window artefact at the drift period,
        not a slow drift at all, and a misspecification the simulator could
        never see as such because it was resampled faster than the window.
        One call per trace, one phase, absolute time.
    rng : numpy Generator

    Returns
    -------
    x_out : (J, T) float64
    contaminated : (J,) bool, which windows were replaced -- recorded so that a
        diagnostic can condition on it rather than guess.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    if x.ndim != 2:
        raise ValueError("x must be (J, T)")
    J = x.shape[0]
    contaminated = np.zeros(J, dtype=bool)
    if not spec.active:
        return x, contaminated

    if "drift" in spec.modes:
        t = np.asarray(t_seconds, dtype=np.float64)
        if t.ndim == 1:
            t = np.broadcast_to(t, x.shape)
        if t.shape != x.shape:
            raise ValueError("t_seconds must be (T,) or match x's (J, T)")
        phase = rng.uniform(0.0, 2.0 * np.pi)          # ONE phase per trace
        amp = spec.pi * spec.drift_amp_max * float(np.mean(x))
        x = x + amp * np.sin(2.0 * np.pi * t / spec.drift_period_s + phase)

    if "contamination" in spec.modes:
        frac = spec.pi * spec.contam_frac_max
        n_bad = int(np.floor(frac * J))
        # Round the remainder stochastically, so that a small frac is not
        # silently truncated to zero on short traces.
        if rng.random() < (frac * J - n_bad):
            n_bad += 1
        n_bad = min(n_bad, J)
        if n_bad > 0:
            idx = rng.choice(J, size=n_bad, replace=False)
            contaminated[idx] = True
            for i in idx:
                # Contamination model: amplitude rescaling plus a burst of
                # broadband noise. Deliberately something the simulator cannot
                # produce, which is what makes it a misspecification.
                scale = rng.uniform(2.0, 5.0)
                noise = rng.standard_normal(x.shape[1]) * np.std(x[i])
                x[i] = np.maximum(x[i] * scale + noise, 0.0)

    return x, contaminated
