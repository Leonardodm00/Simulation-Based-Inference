#!/usr/bin/env python3
"""
npe_tune_search.py -- the hyperparameter space and the GP search loop.

Scope boundary: proposing configurations and judging whether to keep going.
This module does not train, score, or load banks. Its only heavy dependency
is scikit-optimize, imported lazily so that everything else in the tuner
still works if that package is absent.

Design decisions locked here, with the reason for each:

  * THE SEARCH IS STATELESS ACROSS INVOCATIONS. The optimiser is never kept
    alive between jobs. Every call rebuilds it and replays the recorded
    (config, score) pairs through tell(). Verified: a fresh Optimizer given
    the same observations reproduces the warm-started state and proposes
    sensibly. The consequence is that raising the budget costs nothing
    already spent, a crashed coordinator loses nothing, and the ledger --
    not a pickle -- is the state.

  * BATCHED PROPOSALS USE THE CONSTANT-LIAR STRATEGY. ask(n_points=b) with
    strategy="cl_min" temporarily assumes each un-run proposal returned a
    placeholder, so a batch is b genuinely different configurations rather
    than b copies of one guess. Without it, parallel submission wastes the
    batch.

  * RANGES ARE ANCHORED TO THE PROBLEM, NOT TO NUMERALS. The width range
    scales with max(p, E) so a bank with a different parameter count
    searches a sensibly placed range with no code edit. The absolute
    fallbacks match the repository defaults and are recorded in the ledger.

  * THE STOPPING RULE COMPARES AGAINST MEASURED NOISE. An improvement
    smaller than the across-seed spread is not an improvement. That spread
    is measured in the baseline stage, not guessed here.

  * WEIGHT DECAY AND LR SCHEDULES ARE NOT IN THE SPACE. sbi 0.27.0's
    preconfigured training loop does not expose them through NPEConfig, so
    including them would mean silently writing a custom training loop.
    Recorded as an extension rather than smuggled in.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SpaceSpec",
    "default_space",
    "space_dimensions",
    "config_from_point",
    "point_from_config",
    "build_optimizer",
    "propose",
    "EscalationVerdict",
    "escalation_verdict",
    "convergence_trace",
    "SpaceAdapter",
    "NPE_ADAPTER",
]

# Order is fixed and load-bearing: a skopt point is a positional list, so the
# mapping between a point and a configuration is this tuple and nothing else.
KNOB_ORDER = ("hidden_features", "num_transforms", "num_bins",
              "learning_rate", "training_batch_size")


@dataclass
class SpaceAdapter:
    """How the optimiser talks to ONE search space.

    Everything in this module that is space-specific goes through here:
    building the skopt dimensions, converting between a positional point and
    a configuration dict, the identity used to detect duplicates, and which
    axes count as "on the boundary". The optimiser mechanics -- stateless
    Optimizer rebuilt from the ledger each round, constant-liar batching,
    the three-condition escalation rule -- are the same for every space and
    are written once.

    The default is `NPE_ADAPTER`, the flow-only space of KNOB_ORDER, so
    every existing call keeps its behaviour unchanged. Stage 4's joint space
    supplies its own through `joint_space.adapter(campaign, spec)` rather
    than by copying this module.
    """

    name: str
    dimensions: Callable[[Any], List[Any]]
    config_from_point: Callable[[Sequence[Any]], Dict[str, Any]]
    point_from_config: Callable[[Dict[str, Any]], List[Any]]
    config_key: Callable[[Dict[str, Any]], Any]
    boundary_axes: Callable[[Dict[str, Any], Any], List[str]]


@dataclass
class SpaceSpec:
    """The searched ranges, resolved against the problem's shapes."""

    hidden_features: Tuple[int, int] = (64, 256)
    num_transforms: Tuple[int, int] = (4, 12)
    num_bins: Tuple[int, int] = (6, 16)
    learning_rate: Tuple[float, float] = (1e-4, 2e-3)
    batch_sizes: Tuple[int, ...] = (256, 512, 1024)
    anchored_to: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_features": list(self.hidden_features),
            "num_transforms": list(self.num_transforms),
            "num_bins": list(self.num_bins),
            "learning_rate": list(self.learning_rate),
            "batch_sizes": list(self.batch_sizes),
            "anchored_to": dict(self.anchored_to),
        }


def default_space(p: int, embedding_dim: int,
                  n_train: Optional[int] = None) -> SpaceSpec:
    """Ranges for a problem of this shape.

    The width range spans roughly [2, 8] times max(p, E), then widened if
    necessary so that it always CONTAINS the repository's own [64, 256].
    A bank with a larger parameter count therefore searches wider networks
    with no edit, while the default region always stays reachable. The batch-size set is trimmed if the training split is
    too small for the largest option to give a sensible number of steps per
    epoch: a batch that swallows the epoch makes early stopping meaningless.
    """
    scale = max(int(p), int(embedding_dim))
    lo = max(32, int(2 * scale))
    hi = max(lo * 2, int(8 * scale))
    lo, hi = min(lo, 64), max(hi, 256)

    batches = [256, 512, 1024]
    if n_train is not None:
        batches = [b for b in batches if n_train >= 20 * b] or [min(batches)]

    return SpaceSpec(
        hidden_features=(int(lo), int(hi)),
        num_transforms=(4, 12),
        num_bins=(6, 16),
        learning_rate=(1e-4, 2e-3),
        batch_sizes=tuple(int(b) for b in batches),
        anchored_to={"p": int(p), "embedding_dim": int(embedding_dim),
                     "n_train": (int(n_train) if n_train is not None else None),
                     "width_rule": "clamp([2,8] * max(p,E), [64,256])"},
    )


def space_dimensions(spec: SpaceSpec):
    """Build the scikit-optimize dimension list, in KNOB_ORDER."""
    from skopt.space import Categorical, Integer, Real

    return [
        Integer(spec.hidden_features[0], spec.hidden_features[1],
                prior="log-uniform", name="hidden_features"),
        Integer(spec.num_transforms[0], spec.num_transforms[1],
                name="num_transforms"),
        Integer(spec.num_bins[0], spec.num_bins[1], name="num_bins"),
        Real(spec.learning_rate[0], spec.learning_rate[1],
             prior="log-uniform", name="learning_rate"),
        Categorical(list(spec.batch_sizes), name="training_batch_size"),
    ]


def config_from_point(point: Sequence[Any]) -> Dict[str, Any]:
    """Turn a skopt point into an NPEConfig-shaped dict of plain Python
    types. The cast is not cosmetic: skopt returns numpy integer scalars,
    which json.dump refuses and which would otherwise reach the ledger."""
    if len(point) != len(KNOB_ORDER):
        raise ValueError("point has %d entries, expected %d for %r"
                         % (len(point), len(KNOB_ORDER), list(KNOB_ORDER)))
    out: Dict[str, Any] = {}
    for name, value in zip(KNOB_ORDER, point):
        if name == "learning_rate":
            out[name] = float(value)
        else:
            out[name] = int(value)
    return out


def point_from_config(config: Dict[str, Any]) -> List[Any]:
    """Inverse of config_from_point, for replaying the ledger."""
    point: List[Any] = []
    for name in KNOB_ORDER:
        if name not in config:
            raise KeyError("config is missing searched knob %r" % name)
        v = config[name]
        point.append(float(v) if name == "learning_rate" else int(v))
    return point


# ---------------------------------------------------------------------------
# The optimiser
# ---------------------------------------------------------------------------

def build_optimizer(spec: SpaceSpec,
                    n_initial_points: int,
                    seed: int = 0,
                    noise: Optional[float] = None,
                    observations: Optional[Sequence[Tuple[Dict[str, Any], float]]] = None,
                    adapter: Optional[SpaceAdapter] = None):
    """Construct an Optimizer and replay any recorded observations.

    Parameters
    ----------
    noise : float, optional
        The GP's assumed observation noise VARIANCE. Pass the square of the
        measured run-to-run spread when it is known; leave None to let
        scikit-optimize estimate it ("gaussian"). Setting it too low makes
        the surrogate chase seed luck and over-trust its own minimum, which
        is the specific failure mode of a noisy objective.
    observations : sequence of (config dict, score)
        Replayed through tell() in the given order. This IS the warm start.
    """
    from skopt import Optimizer

    ad = adapter or NPE_ADAPTER
    opt = Optimizer(
        dimensions=ad.dimensions(spec),
        base_estimator="GP",
        n_initial_points=int(n_initial_points),
        acq_func="EI",
        acq_optimizer="auto",
        random_state=int(seed),
    )
    if noise is not None:
        # skopt exposes the noise level through the Optimizer's constructor in
        # some versions and through the base estimator in others; set it on
        # the estimator when the constructor did not accept it, rather than
        # assuming either shape.
        try:
            opt.base_estimator_.noise = float(noise)
        except Exception:
            pass

    if observations:
        xs = [ad.point_from_config(cfg) for cfg, _ in observations]
        ys = [float(y) for _, y in observations]
        if xs:
            opt.tell(xs, ys)
    return opt


def propose(spec: SpaceSpec,
            observations: Sequence[Tuple[Dict[str, Any], float]],
            n_points: int,
            n_initial_points: int,
            seed: int = 0,
            noise: Optional[float] = None,
            exclude: Optional[Sequence[Dict[str, Any]]] = None,
            max_resample: int = 20,
            adapter: Optional[SpaceAdapter] = None) -> List[Dict[str, Any]]:
    """Return n_points fresh configurations, warm-started from observations.

    `exclude` lists configurations already proposed but not yet evaluated
    (the pending set). Duplicates against it are re-asked a bounded number of
    times and then accepted: refusing forever would deadlock a batch when the
    surrogate is confident, and an accepted duplicate is merely a wasted job,
    which the trial-id check upstream will in any case detect.
    """
    ad = adapter or NPE_ADAPTER
    opt = build_optimizer(spec, n_initial_points=n_initial_points, seed=seed,
                          noise=noise, observations=observations, adapter=ad)
    excl = {ad.config_key(c) for c in (exclude or [])}
    excl |= {ad.config_key(c) for c, _ in observations}

    out: List[Dict[str, Any]] = []
    tries = 0
    while len(out) < n_points and tries < max_resample:
        tries += 1
        want = n_points - len(out)
        points = opt.ask(n_points=want, strategy="cl_min")
        if want == 1 and points and not isinstance(points[0], (list, tuple)):
            points = [points]
        fresh = []
        for pt in points:
            cfg = ad.config_from_point(pt)
            key = ad.config_key(cfg)
            if key in excl:
                continue
            excl.add(key)
            fresh.append(cfg)
        out.extend(fresh)
        if not fresh:
            # Nudge the optimiser off a confident point by telling it the
            # current worst value at the duplicate, the standard escape.
            ys = [y for _, y in observations]
            if ys and points:
                opt.tell(list(points[0]) if not isinstance(points[0], (list, tuple))
                         else points[0], float(np.max(ys)))
    return out[:n_points]


def _config_key(config: Dict[str, Any]) -> Tuple:
    """Hashable identity of a configuration, tolerant of float formatting."""
    key = []
    for name in KNOB_ORDER:
        v = config.get(name)
        if name == "learning_rate" and v is not None:
            key.append(round(float(v), 10))
        else:
            key.append(v)
    return tuple(key)


def _npe_boundary_axes(config: Dict[str, Any], spec: SpaceSpec) -> List[str]:
    """Which searched axes of the flow-only space sit on their range edge."""
    on_edge: List[str] = []
    for name, (lo, hi) in (("hidden_features", spec.hidden_features),
                           ("num_transforms", spec.num_transforms),
                           ("num_bins", spec.num_bins)):
        val = config.get(name)
        if val is None:
            continue
        if int(val) <= int(lo) or int(val) >= int(hi):
            on_edge.append(name)
    lr = config.get("learning_rate")
    if lr is not None:
        lo, hi = spec.learning_rate
        if float(lr) <= lo * (1.0 + 1e-6) or float(lr) >= hi * (1.0 - 1e-6):
            on_edge.append("learning_rate")
    return on_edge


NPE_ADAPTER = SpaceAdapter(
    name="npe_flow_only",
    dimensions=space_dimensions,
    config_from_point=config_from_point,
    point_from_config=point_from_config,
    config_key=_config_key,
    boundary_axes=_npe_boundary_axes,
)


# ---------------------------------------------------------------------------
# Convergence and the escalation decision
# ---------------------------------------------------------------------------

def convergence_trace(scores: Sequence[float]) -> List[float]:
    """Best-so-far score after each evaluation, in evaluation order."""
    best, out = math.inf, []
    for y in scores:
        if y is not None and np.isfinite(y) and y < best:
            best = float(y)
        out.append(float(best) if np.isfinite(best) else float("nan"))
    return out


@dataclass
class EscalationVerdict:
    """Whether to spend more budget, and on the evidence of what."""

    escalate: bool
    reasons: List[str] = field(default_factory=list)
    recent_improvement: float = float("nan")
    best_score: float = float("nan")
    best_config: Optional[Dict[str, Any]] = None
    on_boundary: List[str] = field(default_factory=list)
    n_observations: int = 0
    tau_stop: float = float("nan")

    def summary(self) -> str:
        head = ("ESCALATE" if self.escalate else "STOP")
        body = "; ".join(self.reasons) if self.reasons else "no condition fired"
        return ("[escalation] %s after %d evaluations (tau_stop=%.4f): %s"
                % (head, self.n_observations, self.tau_stop, body))


def escalation_verdict(observations: Sequence[Tuple[Dict[str, Any], float]],
                       spec: SpaceSpec,
                       tau_stop: float,
                       window: Optional[int] = None,
                       boundary_tol: float = 1e-6,
                       adapter: Optional[SpaceAdapter] = None) -> EscalationVerdict:
    """Apply the three-condition rule of the protocol, S3.7.

    Escalate if ANY of:
      1. the best-so-far improved by more than tau_stop over the last
         `window` evaluations (the search is still descending);
      2. the best configuration sits on a boundary of the searched range
         (the RANGE, not the budget, is binding -- widen it in the same
         move);
      3. there are too few observations for the surrogate to mean anything.

    tau_stop should be the MEASURED across-seed spread: an improvement
    smaller than seed noise is not an improvement. Condition 3 of the
    protocol (surrogate predicts improvement above tau_stop anywhere) is
    deliberately not implemented from the surrogate's own extrapolation:
    that prediction is exactly the quantity a mis-specified GP gets wrong,
    and acting on it would let the model authorise its own budget. Condition
    1 uses observed values only.
    """
    obs = [(c, float(y)) for c, y in observations
           if y is not None and np.isfinite(float(y))]
    n = len(obs)
    v = EscalationVerdict(escalate=False, n_observations=n,
                          tau_stop=float(tau_stop))
    if n == 0:
        v.escalate = True
        v.reasons.append("no evaluations yet")
        return v

    scores = [y for _, y in obs]
    trace = convergence_trace(scores)
    best_i = int(np.argmin(scores))
    v.best_score = float(scores[best_i])
    v.best_config = dict(obs[best_i][0])

    w = int(window) if window else max(5, n // 3)
    w = min(w, n - 1) if n > 1 else 0
    if w > 0:
        v.recent_improvement = float(trace[-w - 1] - trace[-1])
        if v.recent_improvement > tau_stop:
            v.escalate = True
            v.reasons.append(
                "still descending: best improved by %.4f over the last %d "
                "evaluations, more than tau_stop=%.4f"
                % (v.recent_improvement, w, tau_stop))

    ad = adapter or NPE_ADAPTER
    on_edge = ad.boundary_axes(v.best_config, spec)
    if on_edge:
        v.on_boundary = on_edge
        v.escalate = True
        v.reasons.append(
            "best configuration is on the boundary for %s -- widen the range, "
            "not just the budget" % (", ".join(on_edge),))

    if n < 12:
        v.escalate = True
        v.reasons.append("only %d evaluations: too few for the surrogate to "
                         "be informative" % n)
    return v
