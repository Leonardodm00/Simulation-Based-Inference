#!/usr/bin/env python3
"""
npe_tune_score.py -- the selection objective and its reference floor.

Scope boundary: pure numpy. This module turns log-densities into scores. It
does not train, does not sample, does not load files, and imports neither
torch nor sbi -- every function takes arrays that somebody else produced.
That is deliberate: it makes the objective testable against a fixture whose
answer is known analytically, on a machine with no deep-learning stack.

Quantities implemented, with the equation numbers of the protocol document
(NPE_TUNING_PROTOCOL_v1.md):

  eq. (3)  held-out negative log-likelihood, in nats per row

               L(lambda; D', M') = -(1/|D'|) sum_i log qbar^(M')(theta_i | z_i)

  eq. (4)  Jensen bound: the mixture score never exceeds the members' mean

               L(lambda; D', M) <= (1/M) sum_m L_m(lambda; D')

  eq. (5)  analytic prior floor = differential entropy of the box prior

               L_0 = -E_p(theta)[log p(theta)] = sum_k log(U_k - L_k)

  eq. (6)  empirical information gain

               Delta_hat = L_0 - L

  eq. (7)  population identity (not computed here, but the reason (6) is the
           statistic of record):

               Delta = I(theta; z) - E_p(z)[ KL( p(theta|z) || q(theta|z) ) ]
                     <= I(theta; z)

  so Delta_hat is a lower bound on the mutual information between parameters
  and embedding, whose slack is exactly the estimator's average posterior
  error. Delta_hat ~ 0 therefore means EITHER the embedding carries nothing
  about theta OR the estimator failed to extract it, and those are the only
  two options.

Units: nats per row throughout, with theta in INFERENCE coordinates. A score
computed under one coordinate convention is not comparable to one computed
under another; the information gain is (in population) invariant and is the
quantity to carry across such a change.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "ScoreResult",
    "prior_floor",
    "heldout_nll",
    "information_gain",
    "bootstrap_ci",
    "score_from_log_probs",
    "mixture_log_prob",
    "check_jensen",
]


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------

def prior_floor(bounds_theta: np.ndarray) -> float:
    """L_0, eq. (5): the differential entropy of the uniform box prior.

    Parameters
    ----------
    bounds_theta : ndarray, shape (p, 2)
        Prior box in inference coordinates; row k is (L_k, U_k), L_k < U_k.

    Returns
    -------
    float
        sum_k log(U_k - L_k), in nats. This is exactly the score an estimator
        achieves if it ignores z and returns the prior, so it is the number
        every trial has to beat to have done anything at all. It is exact:
        no estimation error enters it, which is why all the noise in the
        information gain comes from the NLL term.

    Notes
    -----
    Widths are widths in INFERENCE coordinates. On the log-stored axes that
    is a span in ln-units (a number of decades), not a physical width, so
    L_0 is not comparable to any density quoted in natural units.
    Differential entropy may be negative; nothing here requires L_0 > 0.
    """
    b = np.asarray(bounds_theta, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds_theta must have shape (p, 2), got %r"
                         % (b.shape,))
    width = b[:, 1] - b[:, 0]
    if np.any(width <= 0):
        bad = np.flatnonzero(width <= 0).tolist()
        raise ValueError("non-positive prior width on axes %r" % bad)
    return float(np.sum(np.log(width)))


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

def heldout_nll(log_prob_true: np.ndarray) -> float:
    """L, eq. (3): mean negative log-density of the true theta, in nats/row.

    Parameters
    ----------
    log_prob_true : ndarray, shape (N,)
        log q(theta_i | z_i) for each row i of the evaluation split, where q
        is whatever estimator is being scored (a single member or a mixture).
        Rows must come from a split disjoint from the training rows; that is
        the caller's responsibility and is enforced upstream by the split
        manifest.

    Raises
    ------
    ValueError
        If any entry is NaN, or if every entry is -inf. A NaN here is a
        training or evaluation failure, and silently averaging it away
        would report a plausible-looking score for a broken model.
    """
    lp = np.asarray(log_prob_true, dtype=np.float64).reshape(-1)
    if lp.size == 0:
        raise ValueError("heldout_nll: empty log-probability array")
    if np.any(np.isnan(lp)):
        n_nan = int(np.sum(np.isnan(lp)))
        raise ValueError(
            "heldout_nll: %d of %d log-probabilities are NaN. This is a "
            "failed fit or a mismatched split, not a score to average."
            % (n_nan, lp.size))
    if np.all(np.isneginf(lp)):
        raise ValueError("heldout_nll: every log-probability is -inf")
    return float(-np.mean(lp))


def information_gain(nll: float, floor: float) -> float:
    """Delta_hat, eq. (6): how many nats per row the estimator beats the
    prior by. Positive means informative; ~0 means the posterior is the
    prior in disguise; negative means worse than knowing nothing."""
    return float(floor - nll)


def bootstrap_ci(log_prob_true: np.ndarray,
                 n_boot: int = 2000,
                 alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap interval for the NLL, over evaluation rows.

    Returns a dict with 'lo' and 'hi' for the NLL itself. Because the floor
    is exact, the interval for the information gain is obtained by mirroring:
    Delta in [floor - hi, floor - lo].

    The resample is over ROWS of the evaluation split. Rows within one
    topology group are not independent, so this interval is mildly
    optimistic; it is used as a decision aid on top of the across-seed
    spread, never as the sole evidence for a gate.
    """
    lp = np.asarray(log_prob_true, dtype=np.float64).reshape(-1)
    n = lp.shape[0]
    if n < 2:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = -np.mean(lp[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {"lo": lo, "hi": hi, "n_boot": int(n_boot)}


@dataclass
class ScoreResult:
    """Everything one evaluation of the objective produced."""

    nll: float                       # eq. (3), nats/row
    floor: float                     # eq. (5), nats/row
    delta: float                     # eq. (6), nats/row
    n_rows: int
    n_members: int
    nll_ci_lo: float = float("nan")
    nll_ci_hi: float = float("nan")
    delta_ci_lo: float = float("nan")
    delta_ci_hi: float = float("nan")
    member_nll: Optional[Sequence[float]] = None    # per-member, eq. (4) lhs
    member_nll_mean: float = float("nan")
    jensen_ok: Optional[bool] = None

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        if d.get("member_nll") is not None:
            d["member_nll"] = [float(x) for x in d["member_nll"]]
        return d

    def summary(self) -> str:
        return ("NLL %.4f  floor %.4f  gain %.4f  (gain CI [%.4f, %.4f])  "
                "n=%d  M=%d"
                % (self.nll, self.floor, self.delta, self.delta_ci_lo,
                   self.delta_ci_hi, self.n_rows, self.n_members))


def mixture_log_prob(member_log_probs: np.ndarray) -> np.ndarray:
    """log of the equal-weight mixture, eq. (2), from per-member logs.

    Parameters
    ----------
    member_log_probs : ndarray, shape (M, N)
        Row m is log q_m(theta_i | z_i) over the evaluation rows.

    Returns
    -------
    ndarray, shape (N,)
        logsumexp_m(.) - log M, computed stably.

    This exists so the mixture identity can be checked against a fixture
    without instantiating an sbi EnsemblePosterior. When a real ensemble is
    available its own log_prob is used instead, and the smoke test asserts
    the two agree rather than trusting either.
    """
    lp = np.asarray(member_log_probs, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError("member_log_probs must be 2-D (M, N), got %r"
                         % (lp.shape,))
    m = lp.shape[0]
    mx = np.max(lp, axis=0)
    finite = np.isfinite(mx)
    out = np.full(lp.shape[1], -np.inf, dtype=np.float64)
    if np.any(finite):
        shifted = lp[:, finite] - mx[finite]
        out[finite] = mx[finite] + np.log(np.sum(np.exp(shifted), axis=0))
    return out - np.log(float(m))


def check_jensen(member_log_probs: np.ndarray,
                 tol: float = 1e-9) -> Dict[str, float]:
    """Verify eq. (4) numerically on the given evaluation rows.

    The mixture NLL must not exceed the mean of the member NLLs. Returns the
    two quantities and their difference; the caller decides what to do with a
    violation. A violation beyond floating-point tolerance means the mixture
    was not formed as an arithmetic mixture -- most likely a geometric mean
    (product of experts), which is sharper than any member and would make
    overconfidence worse rather than better.
    """
    lp = np.asarray(member_log_probs, dtype=np.float64)
    mix_nll = heldout_nll(mixture_log_prob(lp))
    member_nlls = [-float(np.mean(row)) for row in lp]
    mean_member = float(np.mean(member_nlls))
    return {
        "mixture_nll": mix_nll,
        "member_nll_mean": mean_member,
        "slack": mean_member - mix_nll,     # must be >= -tol
        "ok": bool(mix_nll <= mean_member + tol),
    }


def score_from_log_probs(log_prob_mixture: np.ndarray,
                         bounds_theta: np.ndarray,
                         member_log_probs: Optional[np.ndarray] = None,
                         n_members: int = 1,
                         n_boot: int = 2000,
                         alpha: float = 0.05,
                         seed: int = 0) -> ScoreResult:
    """Assemble a full ScoreResult from evaluated log-densities.

    Parameters
    ----------
    log_prob_mixture : ndarray, shape (N,)
        log qbar^(M')(theta_i | z_i) on the evaluation split.
    bounds_theta : ndarray, shape (p, 2)
        For the exact floor, eq. (5).
    member_log_probs : ndarray, shape (M', N), optional
        Per-member log-densities. When supplied, the Jensen check of eq. (4)
        is run and recorded, which is the cheap standing guarantee that the
        ensemble is being combined as an arithmetic mixture.
    """
    floor = prior_floor(bounds_theta)
    nll = heldout_nll(log_prob_mixture)
    ci = bootstrap_ci(log_prob_mixture, n_boot=n_boot, alpha=alpha, seed=seed)
    res = ScoreResult(
        nll=nll,
        floor=floor,
        delta=information_gain(nll, floor),
        n_rows=int(np.asarray(log_prob_mixture).reshape(-1).shape[0]),
        n_members=int(n_members),
        nll_ci_lo=ci["lo"], nll_ci_hi=ci["hi"],
        delta_ci_lo=floor - ci["hi"], delta_ci_hi=floor - ci["lo"],
    )
    if member_log_probs is not None:
        j = check_jensen(member_log_probs)
        res.member_nll = [-float(np.mean(r))
                          for r in np.asarray(member_log_probs, dtype=np.float64)]
        res.member_nll_mean = j["member_nll_mean"]
        res.jensen_ok = bool(j["ok"])
    return res
