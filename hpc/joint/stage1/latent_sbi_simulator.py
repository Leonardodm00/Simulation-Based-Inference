"""The bench simulator wrapper: prior, densities, and window synthesis.

Implements eqs. (7) and (8) of plan v0.6 S4.1, plus the two deviations from the
DSN's own `sample_latents` that the plan requires:

  1. a TRUNCATED normal on the label axes, not a clipped one. Clipping puts
     point masses on the box boundary, and `z_score_theta =
     "transform_to_unconstrained"` maps a boundary point to +/- infinity. This
     is not cosmetic: one clipped row poisons a training batch.
  2. the NPE prior is the MIXTURE p(phi) = (1/C) sum_c p(phi | c), so the
     analytic box floor does not apply and L_0 needs Monte Carlo.

and one deviation new in v0.6:

  3. a per-well realisation draw inside the likelihood (latent_realisation.py),
     so that two wells of one donor share theta and differ in their graph.

The DSN generator is NOT reimplemented. It enters through `DSNBurstProvider`,
which reuses `latent_to_burst_params`, `generate_spike_times` and
`compute_ifr_trace` unchanged -- the same three primitives the DSN's own
`LatentBurstProvider` calls. Smoke test S4 asserts bit-for-bit equality
against a direct call to them.

Pure ASCII, LF only. numpy + scipy.
"""

import hashlib
import os
import sys

import numpy as np
from scipy import stats

from latent_gap import apply_trace_gap, param_overrides
from latent_nuisance import apply_nuisance
from latent_realisation import realisation_key, realisation_seed


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

class LatentSBISpec(object):
    """Everything that defines one bench bank.

    Parameters
    ----------
    n_latent : int
        Dimension of phi. n = 6 in the plan.
    label_idx : sequence of int
        The label-carrying axes S. The rest are the free axes F.
    class_centres : (C, len(label_idx)) array in (0, 1)
        m_{c,k} of eq. (7).
    tau_ov : float > 0
        Within-class spread on the label axes. tau_ov > 0 is deliberate: the
        bench must be able to FAIL the heterogeneity hypothesis, not just
        illustrate it.
    n_windows_per_trace : int
        J of eq. (8).
    T_win : float
        Window length in seconds.
    fs : float
        Sampling rate of the IFR trace, Hz. W = round(T_win * fs).
    n_neurons : int
    """

    def __init__(self, n_latent=6, label_idx=(0, 1, 2), class_centres=None,
                 tau_ov=0.10, n_windows_per_trace=8, T_win=60.0, fs=50.0,
                 n_neurons=100, seed=0):
        self.n_latent = int(n_latent)
        self.label_idx = tuple(int(i) for i in label_idx)
        if any(i < 0 or i >= self.n_latent for i in self.label_idx):
            raise ValueError("label_idx out of range")
        if len(set(self.label_idx)) != len(self.label_idx):
            raise ValueError("label_idx has duplicates")
        self.free_idx = tuple(i for i in range(self.n_latent)
                              if i not in self.label_idx)

        if class_centres is None:
            class_centres = simplex_centres(3, len(self.label_idx))
        self.class_centres = np.asarray(class_centres, dtype=np.float64)
        if self.class_centres.ndim != 2 or \
                self.class_centres.shape[1] != len(self.label_idx):
            raise ValueError("class_centres must be (C, len(label_idx))")
        if np.any(self.class_centres <= 0.0) or np.any(self.class_centres >= 1.0):
            raise ValueError("class_centres must lie strictly inside (0, 1)")

        self.tau_ov = float(tau_ov)
        if self.tau_ov <= 0.0:
            raise ValueError("tau_ov must be > 0; see S4.0")
        self.n_windows_per_trace = int(n_windows_per_trace)
        self.T_win = float(T_win)
        self.fs = float(fs)
        self.n_neurons = int(n_neurons)
        self.seed = int(seed)

    @property
    def n_classes(self):
        return self.class_centres.shape[0]

    @property
    def W(self):
        return int(round(self.T_win * self.fs))

    def to_dict(self):
        return {
            "n_latent": self.n_latent,
            "label_idx": list(self.label_idx),
            "free_idx": list(self.free_idx),
            "class_centres": self.class_centres.tolist(),
            "tau_ov": self.tau_ov,
            "n_windows_per_trace": self.n_windows_per_trace,
            "T_win": self.T_win,
            "fs": self.fs,
            "n_neurons": self.n_neurons,
            "seed": self.seed,
        }


def simplex_centres(n_classes, n_label_axes, radius=0.30):
    """Class centres on a regular simplex, mapped into (0, 1).

    Mirrors the DSN generator's convention (centres on a regular simplex)
    without importing it, so that a spec can be built with no DSN present.
    Degenerate cases are handled explicitly rather than by accident.
    """
    C, k = int(n_classes), int(n_label_axes)
    if C < 2 or k < 1:
        raise ValueError("need C >= 2 and at least one label axis")
    # Vertices of a regular (C-1)-simplex in R^{C-1}, then embed in R^k.
    v = np.eye(C)[:, : C - 1] - np.ones((C, C - 1)) / (C + np.sqrt(C))
    v = v - v.mean(axis=0, keepdims=True)
    v = v / np.max(np.linalg.norm(v, axis=1))
    emb = np.zeros((C, k))
    m = min(k, C - 1)
    emb[:, :m] = v[:, :m]
    if k > C - 1:
        # Pad the remaining axes by cycling the used ones, so no axis is
        # constant across classes for k > C-1 -- a constant axis would be a
        # free axis mislabelled as a label axis.
        for j in range(C - 1, k):
            emb[:, j] = v[:, j % max(1, C - 1)]
    return 0.5 + radius * emb


# ---------------------------------------------------------------------------
# eq. (7): prior
# ---------------------------------------------------------------------------

def _truncnorm(mu, sigma):
    """scipy truncnorm on (0, 1) with the given mean and scale."""
    a = (0.0 - mu) / sigma
    b = (1.0 - mu) / sigma
    return stats.truncnorm(a, b, loc=mu, scale=sigma)


def sample_prior(spec, n, rng):
    """Draw (c, phi) from eq. (7).

    Returns
    -------
    c : (n,) int
    phi : (n, n_latent) float64, strictly inside (0, 1) on every axis --
        asserted, because a boundary coordinate maps to +/- infinity under
        `transform_to_unconstrained`.
    """
    n = int(n)
    c = rng.integers(0, spec.n_classes, size=n)
    phi = np.empty((n, spec.n_latent), dtype=np.float64)

    for j, ax in enumerate(spec.label_idx):
        for cls in range(spec.n_classes):
            mask = (c == cls)
            k = int(np.sum(mask))
            if k:
                d = _truncnorm(spec.class_centres[cls, j], spec.tau_ov)
                phi[mask, ax] = d.rvs(size=k, random_state=rng)

    for ax in spec.free_idx:
        phi[:, ax] = rng.uniform(0.0, 1.0, size=n)

    # scipy's truncnorm can return exactly 0 or 1 at extreme tails in float64.
    eps = np.finfo(np.float64).eps
    phi = np.clip(phi, eps, 1.0 - eps)
    if np.any(phi <= 0.0) or np.any(phi >= 1.0):
        raise FloatingPointError("phi escaped the open unit box")
    return c, phi


def prior_log_prob(spec, phi):
    """log p(phi) for the MIXTURE prior, eq. (7).

    p(phi) = (1/C) sum_c prod_{k in S} truncnorm(phi_k; m_ck, tau) *
                          prod_{k in F} 1_{(0,1)}(phi_k)

    Vectorised over rows. Returns -inf outside the box.
    """
    phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    n = phi.shape[0]
    inside = np.all((phi > 0.0) & (phi < 1.0), axis=1)

    comp = np.full((n, spec.n_classes), -np.inf)
    for cls in range(spec.n_classes):
        lp = np.zeros(n)
        for j, ax in enumerate(spec.label_idx):
            d = _truncnorm(spec.class_centres[cls, j], spec.tau_ov)
            lp = lp + d.logpdf(phi[:, ax])
        comp[:, cls] = lp                      # free axes contribute log 1 = 0
    out = np.logaddexp.reduce(comp, axis=1) - np.log(spec.n_classes)
    out = np.where(inside, out, -np.inf)
    return out


def class_posterior(spec, phi):
    """p(c | phi), rows summing to 1. Equal class priors, so it is the
    normalised component density."""
    phi = np.atleast_2d(np.asarray(phi, dtype=np.float64))
    n = phi.shape[0]
    comp = np.empty((n, spec.n_classes))
    for cls in range(spec.n_classes):
        lp = np.zeros(n)
        for j, ax in enumerate(spec.label_idx):
            d = _truncnorm(spec.class_centres[cls, j], spec.tau_ov)
            lp = lp + d.logpdf(phi[:, ax])
        comp[:, cls] = lp
    comp = comp - comp.max(axis=1, keepdims=True)
    w = np.exp(comp)
    return w / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# eq. (8): windows
# ---------------------------------------------------------------------------

class BurstProvider(object):
    """Protocol for signal synthesis. The DSN generator is adapted to this.

    A provider must implement

        __call__(phi, n_windows, W, fs, n_neurons, seed, param_overrides)
            -> (n_windows, W) float64, non-negative

    and must be a PURE function of its arguments: same arguments, same bytes.
    `seed` is where the connectivity realisation enters, so a provider that
    ignores it silently destroys the S2.5c construction -- smoke test S10
    checks that it does not.
    """

    def __call__(self, phi, n_windows, W, fs, n_neurons, seed,
                 param_overrides=None):
        raise NotImplementedError


def load_dsn_modules(dsn_main_dir=None):
    """Import the DSN generator modules from DSN_MAIN_DIR.

    Returns (latent_burst_generator, generate_burst_data). The path is put on
    sys.path rather than copied, so the bench can never drift from the module
    the DSN itself uses.
    """
    dsn_main_dir = dsn_main_dir or os.environ.get("DSN_MAIN_DIR")
    if not dsn_main_dir:
        raise RuntimeError(
            "DSN_MAIN_DIR is not set. Point it at the DSN repo's Main/ "
            "directory (use the dsn_main symlink -- the real path contains a "
            "space).")
    if not os.path.isfile(os.path.join(dsn_main_dir, "latent_burst_generator.py")):
        raise RuntimeError("no latent_burst_generator.py under %r" % dsn_main_dir)
    if dsn_main_dir not in sys.path:
        sys.path.insert(0, dsn_main_dir)
    import latent_burst_generator as lbg
    import generate_burst_data as gbd
    return lbg, gbd


class DSNBurstProvider(BurstProvider):
    """Adapter from the DSN generator to the BurstProvider protocol.

    The DSN's own `LatentBurstProvider.__call__(condition, trace_id)` samples
    phi INTERNALLY from (condition, trace_id) and returns a whole trace. That
    signature cannot be used here for two independent reasons:

      1. the bench must supply phi, because on arm R phi is drawn from the
         perturbed generator (gap (a)) and because two wells of one donor must
         be given the SAME phi (S2.5c);
      2. the realisation seed must be settable per well, and
         `LatentBurstProvider` derives its spike RNG from (seed, condition,
         trace_id) with no way to vary it at fixed phi.

    So this adapter goes one level down and reuses the same three primitives
    `LatentBurstProvider` uses, unchanged:

        latent_to_burst_params(spec, phi, condition)   phi -> BurstParams
        generate_spike_times(params, rng)              dynamics
        compute_ifr_trace(spikes, params)              observable

    Nothing is reimplemented. Smoke test S4 asserts bit-for-bit equality
    against a direct call to those three, and against `window_trace` from
    Sbi-extractor for the windowing.
    """

    def __init__(self, lbg, gbd, dsn_spec):
        self.lbg = lbg
        self.gbd = gbd
        self.dsn_spec = dsn_spec

    def burst_params(self, phi, n_windows, W, fs, n_neurons,
                     free_axis_shift=0.0):
        """The BurstParams for one call. Exposed so S4 can rebuild it.

        `free_axis_shift` implements gap (a): each FREE axis range [a_k, b_k]
        is displaced by that fraction of its own width, so the same phi maps
        to physics the simulated prior box cannot reach. The axes are rebuilt
        through the DSN's own `resolve_axes` override hook rather than by
        editing LatentAxis instances, so the validation in `__post_init__`
        (lo < hi, orientation) still runs.
        """
        import dataclasses
        spec = dataclasses.replace(
            self.dsn_spec,
            duration_s=float(n_windows * W) / float(fs),
            w_size=1.0 / float(fs),
            n_neurons=int(n_neurons))
        if free_axis_shift:
            names = [a.name for a in spec.axes]
            overrides = []
            for k in spec.free_axes:
                a = spec.axes[k]
                delta = free_axis_shift * (a.hi - a.lo)
                overrides.append({"name": a.name, "lo": a.lo + delta,
                                  "hi": a.hi + delta})
            spec = dataclasses.replace(
                spec, axes=self.lbg.resolve_axes(names, overrides))
        return self.lbg.latent_to_burst_params(spec, np.asarray(phi,
                                                                dtype=np.float64),
                                               condition=0, tag="bench")

    def __call__(self, phi, n_windows, W, fs, n_neurons, seed,
                 param_overrides=None):
        ov = dict(param_overrides or {})
        shift = float(ov.pop("free_axis_range_shift", 0.0))
        if ov:
            raise NotImplementedError(
                "gap perturbation (b) emits %r, which BurstParams does not "
                "accept. Bind these keys to BurstParams fields in "
                "latent_gap.param_overrides before enabling 'heavy_tail'."
                % (sorted(ov),))

        params = self.burst_params(phi, n_windows, W, fs, n_neurons,
                                   free_axis_shift=shift)
        rng = np.random.default_rng(int(seed) % (2 ** 63))
        spikes = self.gbd.generate_spike_times(params, rng)
        x, fs_out = self.gbd.compute_ifr_trace(spikes, params)

        if abs(float(fs_out) - float(fs)) > 1e-9:
            raise ValueError("generator returned f_s = %r, expected %r"
                             % (fs_out, fs))
        need = n_windows * W
        if x.shape[0] < need:
            raise ValueError(
                "generator returned %d samples, need %d. K = floor(T/Dt) can "
                "fall one sample short of n_windows * W; lengthen duration_s."
                % (x.shape[0], need))
        # Disjoint windows from s = 0 with stride W -- the same rule as
        # sim_observable.window_trace and MEAWindowDataset.
        return np.asarray(x[:need], dtype=np.float64).reshape(n_windows, W)


def load_dsn_provider(dsn_main_dir=None, dsn_spec=None):
    """Build a DSNBurstProvider. `dsn_spec` defaults to the DSN's own
    LatentSpec() defaults, which is what `latent_spec_from_dsn` mirrors."""
    lbg, gbd = load_dsn_modules(dsn_main_dir)
    if dsn_spec is None:
        dsn_spec = lbg.LatentSpec()
    return DSNBurstProvider(lbg, gbd, dsn_spec)


def latent_spec_from_dsn(dsn_spec, lbg, n_windows_per_trace=8, T_win=60.0,
                         seed=0):
    """Build a LatentSBISpec that agrees with a DSN LatentSpec by construction.

    Deriving the bench spec from the DSN spec rather than restating it is the
    only way the two class-centre conventions cannot drift apart: the DSN's
    `_class_center_vectors(C, L, mode)` is used directly, so "simplex" here
    means exactly what it means there.

    The one deliberate difference remains the prior: the DSN clips, this
    truncates (S4.1). Smoke test S1e checks that the two agree conditional on
    no clipping, which is the strongest statement that can be true.
    """
    centres = lbg._class_center_vectors(dsn_spec.n_classes,
                                        len(dsn_spec.label_axes),
                                        dsn_spec.class_center_mode)
    eps = 1e-6
    centres = np.clip(centres, eps, 1.0 - eps)
    return LatentSBISpec(
        n_latent=dsn_spec.n_latent,
        label_idx=tuple(int(k) for k in dsn_spec.label_axes),
        class_centres=centres,
        tau_ov=float(dsn_spec.class_overlap),
        n_windows_per_trace=int(n_windows_per_trace),
        T_win=float(T_win),
        fs=float(dsn_spec.fs),
        n_neurons=int(dsn_spec.n_neurons),
        seed=int(seed))


def provider_sha256(provider):
    """A digest identifying the provider, for the sidecar's generator_sha256.

    Falls back to the class name when the source is not introspectable, and
    says so in the digest string rather than pretending.
    """
    try:
        import inspect
        src = inspect.getsource(type(provider))
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return "unavailable:" + type(provider).__name__


def time_grid(spec, window_idx):
    """Absolute time of each sample of window `window_idx`, in seconds.

    Absolute rather than window-relative, so that the slow drift in nu and the
    gap's drift are continuous ACROSS windows of one trace. A per-window reset
    would make the drift a within-window artefact instead of a slow one.
    """
    t0 = window_idx * spec.T_win
    return t0 + np.arange(spec.W, dtype=np.float64) / spec.fs


def simulate_windows(spec, phi, provider, donor, well, subregion,
                     realisation_spec, base_seed, nuisance_spec=None,
                     nu_row=None, gap_spec=None, rng=None):
    """One trace: J windows for one well, eq. (8).

    Order of operations, which matters and is easy to get wrong:

        gap (a) --(axis range overrides)--> provider        [misspecification]
        phi, realisation seed --(provider)-->  x            [dynamics]
        x  --(gap (c),(d), trace layer)-->  x               [misspecification]
        x  --(nu, eq. N1)-->  x_obs                         [observation]

    The nuisance is applied LAST because it is an observation-level map, which
    is the property the whole nuisance/mechanism separation rests on (S2.6).
    Applying it before the gap would make it dynamics-dependent and silently
    invalidate eq. (6).

    Returns
    -------
    x_obs : (J, W) float64
    info : dict with `realisation_id`, `contaminated` (J,), `phi_eff`
    """
    rng = rng if rng is not None else np.random.default_rng(base_seed)
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if phi.size != spec.n_latent:
        raise ValueError("phi must have %d entries" % spec.n_latent)

    # phi is NEVER shifted: perturbation (a) moves the axis RANGES inside the
    # provider instead, so every recorded coordinate stays in [0, 1] and the
    # prior density of a recorded theta stays finite. See latent_gap.
    phi_eff = phi
    overrides = None
    if gap_spec is not None:
        overrides = param_overrides(gap_spec) or None

    key = realisation_key(realisation_spec, donor, well, subregion)
    seed = realisation_seed(base_seed, key)

    x = np.asarray(provider(phi_eff, spec.n_windows_per_trace, spec.W,
                            spec.fs, spec.n_neurons, seed,
                            param_overrides=overrides), dtype=np.float64)
    if x.shape != (spec.n_windows_per_trace, spec.W):
        raise ValueError("provider returned %r, expected %r"
                         % (x.shape, (spec.n_windows_per_trace, spec.W)))
    if np.any(x < 0.0):
        raise ValueError("provider returned negative rates")

    contaminated = np.zeros(spec.n_windows_per_trace, dtype=bool)
    if gap_spec is not None:
        # Whole trace at once, on the absolute time grid: one drift phase for
        # the trace and a contamination count over J windows, not J
        # independent coin flips at frac*1.
        t_all = np.stack([time_grid(spec, j)
                          for j in range(spec.n_windows_per_trace)])
        x, contaminated = apply_trace_gap(gap_spec, x, t_all, rng)

    if nuisance_spec is not None and nu_row is not None:
        for j in range(spec.n_windows_per_trace):
            x[j] = apply_nuisance(nuisance_spec, x[j], nu_row,
                                  time_grid(spec, j))

    return x, {"realisation_id": np.uint64(seed),
               "contaminated": contaminated,
               "phi_eff": phi_eff}
