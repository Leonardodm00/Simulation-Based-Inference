#!/usr/bin/env python3
"""
npe_tune_benchmark.py -- ground-truth substrate for the Stage 1 synthetic
full-pipeline benchmark.

Scope boundary: this module adapts gmm_benchmark.GMMBenchmark (a K-component
Gaussian problem with an EXACT analytic posterior) so the rest of the tuner
-- npe_tune_data, npe_tune_train, npe_tune_search, npe_tune_gates,
npe_tune_ledger -- can be driven against it UNCHANGED, plus the ground-truth
comparisons (C2ST, mode recovery) that only make sense when the truth is
known and that the real-bank pipeline therefore never needed.

WHY THIS IS A SEPARATE MODULE, NOT A GENERALISATION OF THE EXISTING ONES.
GMMBenchmark's prior is an unbounded Gaussian MIXTURE, not the uniform box
the real pipeline's Contract assumes throughout:

  * npe_contract.Contract.prior() returns a BOX prior; the box formula for
    the floor (npe_tune_score.prior_floor, eq. 5 of the protocol) is exact
    only for that family. A Gaussian mixture's differential entropy has no
    closed form in general.
  * npe_model's z_score_theta="transform_to_unconstrained" builds a
    bijection from the prior's SUPPORT, i.e. from the box; it is the wrong
    choice for an already-unconstrained R^n prior. The repository's own GMM
    smoke test (smoke_test_gmm.py, function _train) uses
    z_score_theta="independent" instead, with exactly this reasoning in a
    comment. This module makes the same choice, for the same reason.

Bending Contract or prior_floor to cover both cases would blur a
distinction the rest of the codebase depends on being sharp (see
npe_tune_score.py's docstring: "theta is always in inference coordinates").
Keeping the two prior families in two modules, with the shared training and
scoring machinery reused unmodified, is what the design permits: NOTHING in
npe_tune_data.py, npe_tune_train.py, npe_tune_search.py, npe_tune_gates.py,
or npe_tune_ledger.py is touched by this file. Only npe_tune_score.py's
GENERIC pieces are reused (heldout_nll, bootstrap_ci, information_gain,
check_jensen, mixture_log_prob) -- prior_floor() itself, which is box-only,
is deliberately bypassed in favour of benchmark_prior_floor() below.

What GMMBenchmark supplies, verified against its source before writing this:

  bench.prior_sample(n, rng)        -> theta, (n, n_dim)
  bench.prior_log_prob(theta)       -> (n,)
  bench.simulate(theta, rng)        -> x, (n, n_obs); x IS the conditioner,
                                        playing the role "z" plays in the
                                        real pipeline -- there is no
                                        separate embedding step here
  bench.torch_prior(device)         -> the TRUE mixture prior as a torch
                                        Distribution, support declared as
                                        R^n_dim (see its own docstring)
  bench.posterior(x_o)              -> GMMPosterior, EXACT, for one fixed
                                        x_o; .sample(n, rng), .assign(theta)
  npe_model.train_single(z, theta, prior, config, seed)
                                        -- confirmed signature; "z" is
                                        already the parameter name used
                                        for the conditioner in this
                                        repository, matching this module's
                                        usage throughout

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import npe_tune_data as TD
import npe_tune_score as TS

__all__ = [
    "BenchmarkContract",
    "_RealVectorMixturePrior",
    "PriorFloorMC",
    "make_benchmark_contract",
    "benchmark_bank",
    "benchmark_prior_floor",
    "score_against_mc_floor",
    "c2st",
    "npe_samples_at",
    "mode_recovery",
    "default_shapes",
]


# ---------------------------------------------------------------------------
# BenchmarkContract -- the minimal duck-typed stand-in
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.distributions import (Categorical, Distribution,
                                     MixtureSameFamily, MultivariateNormal,
                                     constraints)
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

if _HAVE_TORCH:
    class _RealVectorMixturePrior(Distribution):
        """A picklable equivalent of GMMBenchmark.torch_prior()'s return
        value.

        torch_prior() wraps its MixtureSameFamily in a class defined INSIDE
        the method body (to declare full support R^n, since a bare
        MixtureSameFamily's own .support is a MixtureSameFamilyConstraint
        sbi cannot build a bijection from). A class statement nested inside
        a function gets a __qualname__ containing "<locals>", which has no
        resolvable module-level path, so torch.save() on any object holding
        a reference to it -- an sbi posterior does, via its own .prior
        attribute -- fails with "Can't pickle local object ...". Confirmed
        directly: running this module's own training-plus-save path,
        training succeeds and only torch.save() on the resulting posterior
        fails, which pins the problem to picklability specifically.

        THIS class statement sits at module level (inside an `if` block,
        which does not introduce a new scope in Python, so __qualname__ is
        plain "_RealVectorMixturePrior" with no "<locals>" -- verified
        below). An earlier version of this fix wrapped the class in a
        factory function to defer the torch import; that reintroduced the
        identical bug one level removed, since the class statement was then
        nested inside THAT function instead. The conditional-import pattern
        used here is what avoids a hard torch dependency at module import
        time (needed so the fast smoke-test tier still imports this module
        with no torch installed) without paying for it with unpicklability.

        sbi's check_prior() requires isinstance(prior, Distribution)
        (confirmed the hard way: an earlier duck-typed, non-Distribution
        version of this class failed that assertion immediately on the
        first real training call), so this subclasses Distribution
        properly, mirroring torch_prior()'s own
        __init__(batch_shape, event_shape, validate_args=False) call.

        Equivalence to both the repository's own torch_prior() and to the
        independently-implemented analytic prior_log_prob is not assumed:
        see test_benchmark_prior_matches in smoke_test_tune.py, and the
        interactive check that produced max|mine - bench.torch_prior| = 0.0
        and max|mine - analytic| ~ 2e-6 over 500 draws.
        """

        arg_constraints: Dict[str, Any] = {}
        has_rsample = False

        def __init__(self, weights: np.ndarray, means: np.ndarray,
                    covs: np.ndarray, device: str = "cpu"):
            mix = Categorical(torch.as_tensor(weights, dtype=torch.float32,
                                              device=device))
            comp = MultivariateNormal(
                torch.as_tensor(means, dtype=torch.float32, device=device),
                covariance_matrix=torch.as_tensor(covs, dtype=torch.float32,
                                                  device=device))
            self._inner = MixtureSameFamily(mix, comp)
            super().__init__(self._inner.batch_shape, self._inner.event_shape,
                             validate_args=False)

        @constraints.dependent_property
        def support(self):
            return constraints.independent(constraints.real, 1)

        def sample(self, sample_shape=torch.Size()):
            return self._inner.sample(sample_shape)

        def log_prob(self, value):
            return self._inner.log_prob(value)

        @property
        def mean(self):
            return self._inner.mean

        @property
        def variance(self):
            return self._inner.variance

    assert "<locals>" not in _RealVectorMixturePrior.__qualname__, (
        "the picklability fix regressed: _RealVectorMixturePrior is nested "
        "again (%r)" % _RealVectorMixturePrior.__qualname__)
else:
    _RealVectorMixturePrior = None           # fast tier: torch absent, fine


@dataclass
class BenchmarkContract:
    """Exposes exactly what npe_tune_data / npe_tune_train / npe_tune_ledger
    touch on a Contract, and nothing more -- so a new dependency on a
    Contract attribute this class does not have fails loudly at import or
    call time, rather than silently returning a plausible-looking wrong
    value.

    bounds_theta here is NOMINAL: a wide box that contains most of the
    prior's mass, present only so identity hashing (_contract_digest in
    npe_tune_data.py) and any incidental shape check have something of the
    right shape to hash. It is NEVER passed to npe_tune_score.prior_floor
    in this module -- see benchmark_prior_floor() for the real floor.
    """

    param_names: List[str]
    coord: List[str]
    bounds_theta: np.ndarray            # nominal only; see docstring
    embedding_dim: int
    meta: Dict[str, Any] = field(default_factory=dict)
    _bench: Any = field(default=None, repr=False)   # the live GMMBenchmark

    @property
    def p(self) -> int:
        return len(self.param_names)

    def prior(self, device: str = "cpu"):
        """The TRUE Gaussian-mixture prior, not a box approximation to it.

        Uses _RealVectorMixturePrior rather than bench.torch_prior()
        directly: same distribution (identical weights/means/covs and
        construction), but picklable, which bench.torch_prior()'s own
        return value is not -- see that class's docstring. Training against
        anything other than the bench's true weights/means/covs would mean
        the NPE is fit to a different prior than the one that actually
        generated the data, and the analytic posterior comparison would no
        longer be comparing like with like; the module-level test
        test_benchmark_prior_matches checks the two give the same log_prob.
        """
        return _RealVectorMixturePrior(self._bench.weights, self._bench.means,
                                       self._bench.covs, device=device)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_names": list(self.param_names),
            "coord": list(self.coord),
            "bounds_theta": np.asarray(self.bounds_theta).tolist(),
            "embedding_dim": int(self.embedding_dim),
            "meta": dict(self.meta),
            "source": "npe_tune_benchmark.BenchmarkContract",
        }


def make_benchmark_contract(bench, nominal_sigma: float = 6.0) -> BenchmarkContract:
    """Build the BenchmarkContract for one GMMBenchmark instance.

    nominal_sigma sets the width of the NOMINAL box (see the class
    docstring) in units of the prior's marginal standard deviation, wide
    enough to contain essentially all prior mass without pretending to be
    the true support.
    """
    p = int(bench.n_dim)
    marg_var = np.einsum("k,kii->i", bench.weights, bench.covs) \
             + np.average((bench.means - np.average(bench.means, axis=0,
                                                     weights=bench.weights))**2,
                          axis=0, weights=bench.weights)
    sd = np.sqrt(np.maximum(marg_var, 1e-12))
    centre = np.average(bench.means, axis=0, weights=bench.weights)
    lo = centre - nominal_sigma * sd
    hi = centre + nominal_sigma * sd
    return BenchmarkContract(
        param_names=["theta_%02d" % k for k in range(p)],
        coord=["linear"] * p,                 # GMM theta is never log-stored
        bounds_theta=np.stack([lo, hi], axis=1),
        embedding_dim=int(bench.n_obs),
        meta={"embedding": {"dsn_checkpoint_sha256": "synthetic-gmm-benchmark"},
             "gmm_benchmark": {"n_dim": p, "n_obs": int(bench.n_obs),
                               "n_components": int(bench.n_components),
                               "separate_in_nullspace": bool(bench.separate_in_nullspace)}},
        _bench=bench,
    )


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

def benchmark_bank(bench, n_rows: int, seed: int = 0) -> TD.Bank:
    """A Bank drawn from the GMM benchmark's TRUE generative process.

    theta ~ prior (the real mixture), z = simulate(theta) (the real
    likelihood). No activity filter, no deduplication: those are properties
    of the real pipeline's upstream simulator, not of this one.

    GROUPING. GMMBenchmark has no topology structure (no analogue of a
    connectivity-kernel draw), so each row is its own group -- a row-level
    (iid) split. This is a deliberate, documented choice, not an oversight:
    the grouped-split MACHINERY (whole-group assignment, disjointness,
    hashing, tamper detection) is independently exercised against a fixture
    WITH real multi-row groups in smoke_test_tune.py (S4, S5, S10). This
    module's job is to validate the DRIVER end to end against known ground
    truth, which does not require re-testing grouping mechanics that are
    already covered elsewhere.
    """
    rng = np.random.default_rng(seed)
    theta = bench.prior_sample(n_rows, rng)
    z = bench.simulate(theta, rng)
    groups = np.arange(n_rows, dtype=np.int64)     # iid: one row per group
    contract = make_benchmark_contract(bench)
    meta = {
        "grouping": {"grouping": "iid_fallback", "axes_used": [],
                    "n_groups": int(n_rows),
                    "note": "GMMBenchmark has no topology structure; the "
                            "split is row-level by construction, not a "
                            "fallback from a missing axis."},
        "sim_glob": "gmm_benchmark(n_dim=%d, n_obs=%d, n_components=%d)"
                   % (bench.n_dim, bench.n_obs, bench.n_components),
        "n_rows_raw": int(n_rows), "n_rows_dedup": int(n_rows),
        "duplication_factor": 1.0,
    }
    return TD.Bank(z=z.astype(np.float32), theta=theta, groups=groups,
                   contract=contract, meta=meta)


def default_shapes() -> List[Dict[str, Any]]:
    """Two deliberately different (n_dim, n_obs) pairs for the
    shape-agnosticism assertion. Shape A matches the repository's own
    smoke_test_gmm.py exactly (n_dim=6, n_obs=3, n_components=3), so its
    numbers are directly cross-checkable against that suite's published
    thresholds (G4 worst C2ST < 0.75; G6 negative control C2ST > 0.75,
    reported there at 0.906 in one run). Shape B is larger and unrelated,
    to prove nothing here depends on shape A's specific dimensions.
    """
    return [
        {"name": "shape_A_repo_reference", "n_dim": 6, "n_obs": 3,
         "n_components": 3, "separation": 6.0, "prior_scale": 1.0,
         "obs_noise": 0.4, "seed": 0},
        {"name": "shape_B_larger", "n_dim": 10, "n_obs": 4,
         "n_components": 3, "separation": 5.0, "prior_scale": 1.0,
         "obs_noise": 0.5, "seed": 1},
    ]


# ---------------------------------------------------------------------------
# The Monte Carlo floor
# ---------------------------------------------------------------------------

@dataclass
class PriorFloorMC:
    """The floor for a non-box prior, eq. (5) of the protocol in its
    general form L_0 = -E_p(theta)[log p(theta)], estimated by Monte Carlo
    rather than closed form.

    Unlike the box case this carries its own standard error: se should be
    reported and checked against the scale of scores it will be compared
    to (nats/row), not assumed negligible.
    """

    value: float
    se: float
    n_mc: int
    method: str = "monte_carlo"

    def summary(self) -> str:
        return ("L_0 = %.6f +/- %.6f nats/row (Monte Carlo, n=%d)"
                % (self.value, self.se, self.n_mc))


def benchmark_prior_floor(bench, n_mc: int = 200_000,
                          seed: int = 0) -> PriorFloorMC:
    """Estimate L_0 = -E_p(theta)[log p(theta)] for the mixture prior.

    Both prior_sample and prior_log_prob are exact (no MCMC, no
    approximation in the sampling or the density), so the only source of
    error is the finite Monte Carlo sample -- a plain average of i.i.d.
    draws, with a standard error that shrinks as 1/sqrt(n_mc) and is
    reported rather than assumed away.
    """
    rng = np.random.default_rng(seed)
    theta = bench.prior_sample(int(n_mc), rng)
    nlp = -bench.prior_log_prob(theta)
    return PriorFloorMC(value=float(np.mean(nlp)),
                        se=float(np.std(nlp, ddof=1) / np.sqrt(nlp.shape[0])),
                        n_mc=int(n_mc))


def score_against_mc_floor(log_prob_mixture: np.ndarray,
                           mc_floor: PriorFloorMC,
                           member_log_probs: Optional[np.ndarray] = None,
                           n_members: int = 1,
                           n_boot: int = 500,
                           seed: int = 0) -> TS.ScoreResult:
    """The benchmark's equivalent of TS.score_from_log_probs, sourcing the
    floor from a PriorFloorMC instead of a box. Every OTHER piece --
    heldout_nll, bootstrap_ci, information_gain, check_jensen -- is the
    same generic function npe_tune_score.py already provides and tests;
    only prior_floor() itself is bypassed, which is the one function in
    that module that is box-specific.
    """
    nll = TS.heldout_nll(log_prob_mixture)
    ci = TS.bootstrap_ci(log_prob_mixture, n_boot=n_boot, seed=seed)
    res = TS.ScoreResult(
        nll=nll, floor=mc_floor.value,
        delta=TS.information_gain(nll, mc_floor.value),
        n_rows=int(np.asarray(log_prob_mixture).reshape(-1).shape[0]),
        n_members=int(n_members),
        nll_ci_lo=ci["lo"], nll_ci_hi=ci["hi"],
        delta_ci_lo=mc_floor.value - ci["hi"],
        delta_ci_hi=mc_floor.value - ci["lo"],
    )
    if member_log_probs is not None:
        j = TS.check_jensen(member_log_probs)
        res.member_nll = [-float(np.mean(r)) for r in
                          np.asarray(member_log_probs, dtype=np.float64)]
        res.member_nll_mean = j["member_nll_mean"]
        res.jensen_ok = bool(j["ok"])
    return res


# ---------------------------------------------------------------------------
# C2ST -- mirrors the repository's own pinned metric
# ---------------------------------------------------------------------------

def c2st(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test accuracy, 5-fold, standardised features.

    0.5 = indistinguishable, 1.0 = trivially separable. Deliberately
    reimplemented rather than imported from smoke_test_gmm.py: that file's
    _c2st is a leading-underscore, test-file-local function with no
    stability contract, and importing production code from a test module
    is backwards. This copy uses the IDENTICAL method (sklearn
    MLPClassifier, 5-fold cross_val_score, standardised features) so
    numbers stay comparable to that suite's own published thresholds
    (G4: worst < 0.75; G6 negative control: > 0.75, one run at 0.906).
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    if n < 10:
        raise ValueError("c2st needs at least 10 samples per side, got %d/%d"
                         % (a.shape[0], b.shape[0]))
    X = np.concatenate([a[:n], b[:n]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    X = (X - mu) / sd
    clf = MLPClassifier(hidden_layer_sizes=(48,), max_iter=150,
                        random_state=seed, early_stopping=False)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="accuracy")))


# ---------------------------------------------------------------------------
# NPE samples and mode recovery at one fixed observation
# ---------------------------------------------------------------------------

def npe_samples_at(ensemble_or_posterior, x_o: np.ndarray,
                   n_draws: int) -> np.ndarray:
    """Draw n_draws samples from the trained estimator at ONE fixed x_o.

    Wraps npe_diagnostics.sample_posteriors, which is built for many
    observations at once (returns (N, n_draws, p)); here N=1 and the
    leading axis is dropped, since analytic comparisons are made at one
    observation at a time (a fresh classifier per x_o, so looping over many
    x_o is the expensive part, not this call).
    """
    import npe_diagnostics as D

    x_o = np.asarray(x_o, dtype=np.float64).reshape(1, -1)
    samples = D.sample_posteriors(ensemble_or_posterior, x_o,
                                  n_draws=int(n_draws))
    return np.asarray(samples)[0]


def mode_recovery(bench, x_o: np.ndarray, npe_samples: np.ndarray,
                  weight_tol: float = 0.30,
                  centre_tol_sd: float = 1.5) -> Dict[str, Any]:
    """Did the trained estimator find every mode of the exact posterior?

    Mirrors smoke_test_gmm.py's g5_mode_recovery, with DELIBERATELY LOOSER
    tolerances: this module validates that the harness and the pipeline
    detect a genuine multi-modal recovery, not that a small, fast-budget
    training run matches the repository's own tight, larger-budget recovery
    thresholds. The repository's G4/G5 suite is the place tolerances that
    tight are meaningful.
    """
    exact = bench.posterior(x_o)
    assign = exact.assign(npe_samples)
    k_range = range(exact.n_components)     # property, not a method
    found = np.array([int(np.sum(assign == k)) for k in k_range])
    scale = float(np.sqrt(np.mean(np.diag(exact.covs[0]))))

    weight_err, centre_err = [], []
    for k in k_range:
        if found[k] == 0:
            weight_err.append(float("inf"))
            centre_err.append(float("inf"))
            continue
        emp_w = float(found[k]) / float(npe_samples.shape[0])
        weight_err.append(abs(emp_w - float(exact.weights[k])))
        centre = npe_samples[assign == k].mean(axis=0)
        centre_err.append(float(np.linalg.norm(centre - exact.means[k])) / scale)

    all_found = bool(np.all(found > 0))
    weight_ok = bool(all_found and max(weight_err) < weight_tol)
    centre_ok = bool(all_found and max(centre_err) < centre_tol_sd)
    return {
        "n_components": exact.n_components, "found_per_mode": found.tolist(),
        "all_modes_found": all_found,
        "weight_error": weight_err, "weight_ok": weight_ok,
        "centre_error_sd": centre_err, "centre_ok": centre_ok,
        "passed": bool(all_found and weight_ok and centre_ok),
    }
