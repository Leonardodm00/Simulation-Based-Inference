#!/usr/bin/env python3
"""
bench_burst_generator.py -- Stages B, C and D of the bench burst model.

Implements, from scratch and with no code lineage to Deep-Summary-Network,
the spike-time generator specified in STAGE_A_BENCH_GENERATOR_SPEC_v1.md:

    S3.1  Gamma-renewal burst onsets with burn-in         eqs. (1)-(3)
    S3.2  lognormal envelope durations, overlap policy    eqs. (4)-(5) / O-5(b)
    S3.3  fragmentation: Poisson-shifted count, tiling    eqs. (6)-(8)    [Stage D]
    S3.4  Beta participation, per-fragment gating         eqs. (9)-(10)   [Stage C]
    S3.5  Poisson spikes: background + within-fragment    eqs. (11)-(13)

All five components of the spec are implemented. No NotImplementedError
guards remain.

Reduction discipline (every stage). Each component's reduction setting must
reproduce the previous stage's output EXACTLY for the same seed, which means
it must not consume RNG state. Stage C draws nothing when kappa is infinite
and p_mean is 1; Stage D draws nothing when n_frag_mean is 1. The smoke test
asserts the RNG state is untouched in both cases and that Stage B output is
reproduced bit-for-bit.

Overlap policy (spec O-5). Two bursts whose envelopes overlap are handled by
`overlap`:

    "merge"  (default, O-5 option b)  union the intervals into one longer
             envelope. Preserves every duration DRAW; the realised envelope
             count J_eff is <= the onset count J.
    "clamp"  (spec eq. 5, option a)   truncate each duration at the next
             onset. Preserves J; shortens the realised duration marginal.

Both are reported per trace via `EnvelopeStats` so the choice is auditable in
the bank, never silent.

This module produces SPIKE TIMES ONLY. IFR construction (binning, smoothing)
is imported from DSN_MAIN_DIR unchanged, in Stage E, for parity reasons
stated in spec S6.

Pure ASCII, LF only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

__all__ = [
    "BenchBurstParams", "EnvelopeStats", "FragmentStats",
    "generate_onsets", "draw_durations", "build_envelopes",
    "fragment_envelopes", "draw_participation_probs",
    "generate_spike_times", "expected_mfr",
]

OVERLAP_POLICIES = ("merge", "clamp")


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchBurstParams:
    """Physical parameters of one trace. Units as in spec S1.

    The ten latent axes appear here as their PHYSICAL images a_k (spec
    convention i); the affine map from phi lives in the provider, not here.
    """
    # -- S3.1 onsets --------------------------------------------------------
    lambda_b: float          # bursts/s, axis 1
    ibi_cv: float            # dimensionless, axis 3
    # -- S3.2 durations -----------------------------------------------------
    d_med: float             # s, axis 7 (median)
    sigma_d: float           # dimensionless, axis 2
    # -- S3.3 fragmentation, eqs. (6)-(8) ----------------------------------
    n_frag_mean: float = 1.0     # count, axis 4; 1.0 = unfragmented
    duty: float = 1.0            # dimensionless, axis 8; ignored when m = 1
    # -- S3.4 participation, eq. (9) ----------------------------------------
    p_mean: float = 1.0          # probability, axis 6
    kappa: float = float("inf")  # dimensionless, axis 9; inf = homogeneous
    # -- S3.5 spikes --------------------------------------------------------
    lambda_burst: float = 100.0  # spikes/s, axis 5
    lambda_bg: float = 0.03      # spikes/s per neuron, axis 10
    # -- recording geometry (fixed, never latent) ---------------------------
    n_neurons: int = 100
    duration_s: float = 1080.0
    # -- policy -------------------------------------------------------------
    overlap: str = "merge"
    burn_in_intervals: float = 10.0   # T_burn = burn_in_intervals / lambda_b

    def __post_init__(self):
        if not (self.lambda_b > 0):
            raise ValueError("lambda_b must be > 0, got %r" % (self.lambda_b,))
        if not (self.ibi_cv > 0):
            raise ValueError("ibi_cv must be > 0, got %r" % (self.ibi_cv,))
        if not (self.d_med > 0):
            raise ValueError("d_med must be > 0, got %r" % (self.d_med,))
        if not (self.sigma_d >= 0):
            raise ValueError("sigma_d must be >= 0, got %r" % (self.sigma_d,))
        if not (self.lambda_burst >= 0 and self.lambda_bg >= 0):
            raise ValueError("rates must be >= 0")
        if not (self.n_neurons >= 1 and self.duration_s > 0):
            raise ValueError("n_neurons >= 1 and duration_s > 0 required")
        if self.overlap not in OVERLAP_POLICIES:
            raise ValueError("overlap must be one of %r, got %r"
                             % (OVERLAP_POLICIES, self.overlap))
        if not (self.n_frag_mean >= 1.0):
            raise ValueError("n_frag_mean must be >= 1, got %r"
                             % (self.n_frag_mean,))
        if not (0.0 < self.duty <= 1.0):
            raise ValueError("duty must be in (0, 1], got %r" % (self.duty,))
        if not (0.0 < self.p_mean <= 1.0):
            raise ValueError("p_mean must be in (0, 1], got %r" % (self.p_mean,))
        if not (self.kappa > 0):
            raise ValueError("kappa must be > 0 (inf allowed), got %r"
                             % (self.kappa,))

    # derived, spec eq. (2) and (4)
    @property
    def k_gamma(self) -> float:
        return self.ibi_cv ** -2

    @property
    def theta_gamma(self) -> float:
        return 1.0 / (self.k_gamma * self.lambda_b)

    @property
    def mu_d(self) -> float:
        return float(np.log(self.d_med))

    @property
    def t_burn(self) -> float:
        return self.burn_in_intervals / self.lambda_b

    # spec eq. (6): P(m = 1) = exp(-(n_frag_mean - 1))
    @property
    def p_single_fragment(self) -> float:
        return float(np.exp(-(self.n_frag_mean - 1.0)))

    # spec eq. (9)
    @property
    def alpha_p(self) -> float:
        return self.kappa * self.p_mean

    @property
    def beta_p(self) -> float:
        return self.kappa * (1.0 - self.p_mean)


@dataclass
class FragmentStats:
    """What fragmentation did to this trace, spec S3.3."""
    n_envelopes: int
    n_fragments: int
    mean_fragments_per_envelope: float
    total_fragment_s: float       # sum of fragment lengths
    total_envelope_s: float       # sum of envelope lengths

    @property
    def active_fraction(self) -> float:
        """Realised sum(fragments)/sum(envelopes); equals 1 when unfragmented
        and tends to P(m=1) + duty*(1 - P(m=1)) otherwise."""
        return (self.total_fragment_s / self.total_envelope_s
                if self.total_envelope_s > 0 else float("nan"))


@dataclass
class EnvelopeStats:
    """What the overlap policy did to this trace. Reported, never silent."""
    n_onsets: int                 # J: onsets that fell in [0, T)
    n_envelopes: int              # J_eff after the overlap policy
    n_affected: int               # onsets whose envelope was merged or clamped
    total_envelope_s: float       # sum of realised envelope durations
    total_drawn_s: float          # sum of the raw duration draws (pre-policy)
    policy: str
    extra: dict = field(default_factory=dict)

    @property
    def affected_rate(self) -> float:
        return self.n_affected / self.n_onsets if self.n_onsets else 0.0


# ---------------------------------------------------------------------------
# S3.1 -- onsets, eqs. (1)-(3)
# ---------------------------------------------------------------------------

def generate_onsets(params: BenchBurstParams,
                    rng: np.random.Generator) -> np.ndarray:
    """Gamma-renewal onsets in [0, T), spec eqs. (1)-(3).

    Intervals are i.i.d. Gamma(k_gamma, theta_gamma). The process is started
    at -T_burn and only onsets in [0, T) are kept, which defeats the
    inspection paradox (spec S3.1): without burn-in the interval straddling
    t = 0 is stochastically longer than a typical one.
    """
    T = params.duration_s
    k, th = params.k_gamma, params.theta_gamma
    t = -params.t_burn
    # expected number of draws needed, with headroom; extend in chunks if short
    n_expect = int((T + params.t_burn) * params.lambda_b) + 16
    onsets: List[float] = []
    while True:
        gaps = rng.gamma(k, th, size=n_expect)
        cum = t + np.cumsum(gaps)
        past = cum >= T
        keep = cum[(cum >= 0.0) & ~past]
        onsets.extend(keep.tolist())
        if past.any():
            break
        t = float(cum[-1])
    out = np.asarray(onsets, dtype=np.float64)
    out.sort()
    return out


# ---------------------------------------------------------------------------
# S3.2 -- durations, eqs. (4)-(5)
# ---------------------------------------------------------------------------

def draw_durations(n: int, params: BenchBurstParams,
                   rng: np.random.Generator) -> np.ndarray:
    """Raw lognormal duration draws, spec eq. (4). No clamping here."""
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return rng.lognormal(params.mu_d, params.sigma_d, size=n).astype(np.float64)


def build_envelopes(onsets: np.ndarray, durations: np.ndarray,
                    duration_s: float, overlap: str
                    ) -> Tuple[np.ndarray, EnvelopeStats]:
    """Non-overlapping envelopes from onsets + raw durations.

    Returns (env, stats) where env is (J_eff, 2) float64 of [start, end).

    "merge"  -- union overlapping intervals (O-5 option b). A duration draw is
                never shortened by another burst; only by the recording end.
    "clamp"  -- spec eq. (5): D <- min(D, next_onset - onset, T - onset).
    """
    if overlap not in OVERLAP_POLICIES:
        raise ValueError("unknown overlap policy %r" % (overlap,))
    n = int(onsets.size)
    if n == 0:
        st = EnvelopeStats(0, 0, 0, 0.0, 0.0, overlap)
        return np.zeros((0, 2), dtype=np.float64), st
    if durations.shape != (n,):
        raise ValueError("durations must have shape (%d,), got %r"
                         % (n, durations.shape))
    T = float(duration_s)
    starts = onsets.astype(np.float64)
    ends = np.minimum(starts + durations, T)        # recording end, always

    if overlap == "clamp":
        nxt = np.append(starts[1:], T)
        ends_c = np.minimum(ends, nxt)
        n_aff = int(np.sum(ends_c < ends))
        env = np.stack([starts, ends_c], axis=1)
        st = EnvelopeStats(n, n, n_aff, float(np.sum(env[:, 1] - env[:, 0])),
                           float(np.sum(durations)), overlap)
        return env, st

    # merge: single pass, onsets already sorted
    merged: List[List[float]] = [[float(starts[0]), float(ends[0])]]
    n_aff = 0
    for s, e in zip(starts[1:], ends[1:]):
        if s < merged[-1][1]:                       # overlaps the open envelope
            merged[-1][1] = max(merged[-1][1], float(e))
            n_aff += 1
        else:
            merged.append([float(s), float(e)])
    env = np.asarray(merged, dtype=np.float64)
    st = EnvelopeStats(n, int(env.shape[0]), n_aff,
                       float(np.sum(env[:, 1] - env[:, 0])),
                       float(np.sum(durations)), overlap)
    return env, st


# ---------------------------------------------------------------------------
# S3.3 / S3.4 -- reduction forms only, Stage B
# ---------------------------------------------------------------------------

def fragment_envelopes(env: np.ndarray, params: BenchBurstParams,
                       rng: np.random.Generator
                       ) -> Tuple[np.ndarray, np.ndarray, FragmentStats]:
    """Split each envelope into fragments, spec eqs. (6)-(8).

    Returns (frags, owner, stats): frags is (n_frag, 2) float64 [start, end);
    owner[r] is the envelope index each fragment belongs to.

    Per envelope j with duration D:
        m = 1 + Poisson(n_frag_mean - 1)                         eq. (6)
        m == 1 : the single fragment IS the envelope; duty ignored (S3.3 rule)
        m >= 2 : f = duty*D/m, g = (1-duty)*D/(m-1)              eq. (7)
                 fragment r = [t0 + r(f+g), t0 + r(f+g) + f)       eq. (8)
    which tiles the envelope exactly: m*f + (m-1)*g = D.

    Reduction: at n_frag_mean = 1 no draw is made and frags is env itself,
    so Stage C output is reproduced bit-for-bit for the same seed.
    """
    J = int(env.shape[0])
    if J == 0:
        return (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64),
                FragmentStats(0, 0, float("nan"), 0.0, 0.0))
    D = env[:, 1] - env[:, 0]
    if params.n_frag_mean == 1.0:
        m = np.ones(J, dtype=np.int64)
    else:
        m = 1 + rng.poisson(params.n_frag_mean - 1.0, size=J).astype(np.int64)

    out: List[Tuple[float, float]] = []
    owner: List[int] = []
    for j in range(J):
        t0, mj, Dj = float(env[j, 0]), int(m[j]), float(D[j])
        if mj == 1:
            out.append((t0, t0 + Dj)); owner.append(j)
            continue
        f = params.duty * Dj / mj
        g = (1.0 - params.duty) * Dj / (mj - 1)
        # Boundaries as ONE cumulative sequence, so that adjacent fragments
        # share the identical float when g == 0 (contiguous by construction,
        # not by tolerance), and the last end is pinned to the stored
        # envelope end rather than to an accumulated sum. Without this,
        # t0 + r*f + f and t0 + (r+1)*f differ by one ulp at t ~ 1e4, and a
        # 1-ulp "overlap" appears between contiguous fragments.
        steps = np.empty(2 * mj - 1, dtype=np.float64)
        steps[0::2] = f
        steps[1::2] = g
        edges = t0 + np.concatenate(([0.0], np.cumsum(steps)))
        edges[-1] = float(env[j, 1])
        for r in range(mj):
            out.append((float(edges[2 * r]), float(edges[2 * r + 1])))
            owner.append(j)
    frags = np.asarray(out, dtype=np.float64)
    own = np.asarray(owner, dtype=np.int64)
    st = FragmentStats(J, int(frags.shape[0]), float(m.mean()),
                       float(np.sum(frags[:, 1] - frags[:, 0])),
                       float(np.sum(D)))
    return frags, own, st


def draw_participation_probs(params: BenchBurstParams,
                             rng: np.random.Generator) -> np.ndarray:
    """Per-neuron participation probabilities p_i, spec eq. (9), drawn ONCE
    per trace (spec assumption A-1). Shape (n_neurons,).

    kappa = inf is the homogeneous limit p_i = p_bar for every i and consumes
    no RNG state, so that the Stage B reduction is exact for a fixed seed.
    p_mean = 1 with finite kappa would put beta_p = 0, which Beta does not
    accept; it is treated as the same degenerate point p_i = 1.
    """
    N = params.n_neurons
    if not np.isfinite(params.kappa) or params.p_mean >= 1.0:
        return np.full(N, params.p_mean, dtype=np.float64)
    p = rng.beta(params.alpha_p, params.beta_p, size=N).astype(np.float64)
    # Beta can return exact 0.0 or 1.0 at extreme shapes; keep p_i in (0,1)
    # so that a neuron is never certainly-out or certainly-in by rounding.
    return np.clip(p, np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).eps)


def _participation(params: BenchBurstParams, n_frag: int,
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Recruitment indicators Z_i^(j,r), spec eq. (10): Bernoulli(p_i)
    independently per (neuron, fragment). Returns (Z, p_i) with Z of shape
    (n_neurons, n_frag) bool.

    When every p_i is exactly 1 no draw is made, so the all-recruited case
    leaves the RNG untouched (reduction discipline).
    """
    p_i = draw_participation_probs(params, rng)
    if n_frag == 0:
        return np.zeros((params.n_neurons, 0), dtype=bool), p_i
    if np.all(p_i >= 1.0):
        return np.ones((params.n_neurons, n_frag), dtype=bool), p_i
    u = rng.random(size=(params.n_neurons, n_frag))
    return u < p_i[:, None], p_i


# ---------------------------------------------------------------------------
# S3.5 -- spikes, eqs. (11)-(13)
# ---------------------------------------------------------------------------

def generate_spike_times(params: BenchBurstParams,
                         rng: np.random.Generator
                         ) -> Tuple[List[np.ndarray], EnvelopeStats]:
    """Spike times for one trace, spec eqs. (11)-(13).

    Returns (spike_times, stats): spike_times is a list of n_neurons sorted
    float64 arrays in [0, T), matching the interface compute_ifr_trace
    expects; stats records what the overlap policy did.
    """
    T = params.duration_s
    onsets = generate_onsets(params, rng)
    durs = draw_durations(onsets.size, params, rng)
    env, stats = build_envelopes(onsets, durs, T, params.overlap)
    frags, _owner, fstats = fragment_envelopes(env, params, rng)
    stats.extra["fragments"] = fstats
    n_frag = int(frags.shape[0])
    Z, p_i = _participation(params, n_frag, rng)    # (N, n_frag) bool, (N,)
    stats.extra["p_i"] = p_i
    stats.extra["recruit_frac"] = float(Z.mean()) if Z.size else float("nan")
    flen = frags[:, 1] - frags[:, 0] if n_frag else np.zeros(0)

    spikes: List[np.ndarray] = []
    for i in range(params.n_neurons):
        parts: List[np.ndarray] = []
        # eq. (12): background
        n_bg = int(rng.poisson(params.lambda_bg * T))
        if n_bg > 0:
            parts.append(rng.uniform(0.0, T, size=n_bg))
        # eq. (13): within each fragment the neuron was recruited into
        for r in range(n_frag):
            if not Z[i, r]:
                continue
            n_b = int(rng.poisson(params.lambda_burst * flen[r]))
            if n_b > 0:
                parts.append(rng.uniform(frags[r, 0], frags[r, 1], size=n_b))
        if parts:
            s = np.concatenate(parts).astype(np.float64)
            s.sort()
        else:
            s = np.zeros(0, dtype=np.float64)
        spikes.append(s)
    return spikes, stats


# ---------------------------------------------------------------------------
# spec eq. (14) -- consistency identity, for the smoke test
# ---------------------------------------------------------------------------

def expected_mfr(params: BenchBurstParams) -> float:
    """Spec eq. (14): expected pooled mean firing rate per neuron, ignoring the
    overlap policy. The participation factor is E[p_i] = p_bar exactly, by
    eq. (9), independent of kappa.

    delta_eff refines the spec's statement: because duty is ignored when
    m = 1 (S3.3 rule) and m is random by eq. (6), the expected active
    fraction of an envelope is P(m=1) + duty*(1-P(m=1)), not duty. The two
    agree only in the limit n_frag_mean -> inf. Recorded as a spec
    refinement (O-6)."""
    p1 = params.p_single_fragment
    delta_eff = p1 + params.duty * (1.0 - p1)
    mean_d = params.d_med * np.exp(0.5 * params.sigma_d ** 2)
    return (params.lambda_bg
            + params.lambda_b * params.p_mean * params.lambda_burst
              * delta_eff * mean_d)
