#!/usr/bin/env python3
"""
smoke_test_gmm.py -- does the NPE recover three known n-dimensional Gaussians?

Run:
    python3 smoke_test_gmm.py                # full suite
    python3 smoke_test_gmm.py --fast         # skip the NPE training tests
    python3 smoke_test_gmm.py -k G4          # single test
    python3 smoke_test_gmm.py --demo         # print the problem, train, report

What each test establishes:

  G1  THE ANALYTIC POSTERIOR IS ITSELF CORRECT. Equations (G4)-(G6) in
      gmm_benchmark.py were derived, not copied, so before using them as
      ground truth they are re-verified numerically by self-normalised
      importance sampling from the prior -- a different route to the same
      quantity. If this fails, every other test in this file is measuring
      against a wrong reference and means nothing.
  G2  The null-space construction does what it claims: posterior weights
      equal prior weights EXACTLY, and mode separation is preserved.
  G3  The problem is discriminating: the modes are far apart in units of
      the within-mode standard deviation, so a unimodal fit cannot pass by
      accident. Also confirms a deliberately unimodal control does NOT
      exhibit the null-space weight identity.
  G4  RECOVERY. The trained NPE is compared against exact draws from the
      analytic posterior using a classifier two-sample test (C2ST). 0.5 is
      indistinguishable; 1.0 is trivially separable.
  G5  Mode-level recovery: every one of the K modes is found, each mode's
      mass is recovered, and each mode's centre is recovered.
  G6  Negative control. The same metrics applied to a deliberately WRONG
      posterior (the prior itself) must FAIL the G4/G5 thresholds. Without
      this, a loose threshold could let a broken estimator pass.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gmm_benchmark import GMMBenchmark, reference_posterior_by_importance  # noqa: E402
from npe_model import NPEConfig, train_single  # noqa: E402

RESULTS: List[Tuple[str, bool, str]] = []

# Problem size used throughout. n=6 parameters seen through d=3 observations,
# so null(A) is 3-dimensional and the three Gaussians are unresolvable.
N_DIM, N_OBS, N_COMP = 6, 3, 3


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, True, detail))
        print("  PASS  %-26s %s" % (test_id, detail), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, False, "%s | %s" % (exc, tb)))
        print("  FAIL  %-26s %s" % (test_id, exc), flush=True)


def _bench(**kw) -> GMMBenchmark:
    opts = dict(n_dim=N_DIM, n_obs=N_OBS, n_components=N_COMP,
                separation=6.0, prior_scale=1.0, obs_noise=0.4, seed=0)
    opts.update(kw)
    return GMMBenchmark(**opts)


def _observation(bench: GMMBenchmark, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta_star = bench.prior_sample(1, rng)
    return bench.simulate(theta_star, rng)[0]


# ---------------------------------------------------------------------------
# G1 -- validate the analytic posterior against brute force
# ---------------------------------------------------------------------------

def g1_analytic_posterior_is_correct() -> str:
    worst_mean, worst_cov, worst_ess = 0.0, 0.0, 1.0
    for trial in range(3):
        bench = _bench(seed=trial)
        x_o = _observation(bench, seed=10 + trial)
        post = bench.posterior(x_o)

        ref_mean, ref_cov, diag = reference_posterior_by_importance(
            bench, x_o, n_samples=300000, rng=np.random.default_rng(99 + trial))

        scale = float(np.sqrt(np.mean(np.diag(post.cov()))))
        e_mean = float(np.max(np.abs(post.mean() - ref_mean))) / scale
        e_cov = float(np.max(np.abs(post.cov() - ref_cov))) / (scale ** 2)
        worst_mean = max(worst_mean, e_mean)
        worst_cov = max(worst_cov, e_cov)
        worst_ess = min(worst_ess, diag["ess_frac"])

        check(e_mean < 0.05,
              "analytic vs importance-sampled posterior MEAN disagree: "
              "max rel dev %.4f (trial %d). Equations (G4)-(G6) are wrong."
              % (e_mean, trial))
        check(e_cov < 0.15,
              "analytic vs importance-sampled posterior COVARIANCE disagree: "
              "max rel dev %.4f (trial %d)." % (e_cov, trial))
    return ("analytic == importance sampling: mean dev <=%.4f, cov dev <=%.4f, "
            "min ESS frac %.3f" % (worst_mean, worst_cov, worst_ess))


# ---------------------------------------------------------------------------
# G2 -- the null-space construction behaves as derived
# ---------------------------------------------------------------------------

def g2_nullspace_properties() -> str:
    bench = _bench()
    max_w_dev, min_sep = 0.0, np.inf
    for trial in range(5):
        x_o = _observation(bench, seed=20 + trial)
        post = bench.posterior(x_o)
        # (G6) with A m_k + b identical across k and equal S_k => wt == w
        dev = float(np.max(np.abs(post.weights - bench.weights)))
        max_w_dev = max(max_w_dev, dev)
        min_sep = min(min_sep, post.min_pairwise_separation())
        check(dev < 1e-9,
              "posterior weights differ from prior weights by %.2e; the "
              "component means are not confined to null(A)" % dev)

    # component means must genuinely lie in null(A): A m_k identical for all k
    proj = bench.means @ bench.A.T
    spread = float(np.max(np.abs(proj - proj[0][None, :])))
    check(spread < 1e-9,
          "A m_k is not identical across components (spread %.2e)" % spread)
    return ("posterior weights == prior weights to %.1e; A m_k identical to "
            "%.1e; mode separation >=%.1f sd" % (max_w_dev, spread, min_sep))


# ---------------------------------------------------------------------------
# G3 -- the problem is discriminating
# ---------------------------------------------------------------------------

def g3_problem_is_discriminating() -> str:
    bench = _bench()
    x_o = _observation(bench, seed=3)
    post = bench.posterior(x_o)
    sep = post.min_pairwise_separation()
    check(sep > 4.0,
          "modes are only %.2f sd apart; a unimodal fit could pass by "
          "accident, so the recovery test would not be discriminating" % sep)

    # every component must actually receive mass
    check(float(np.min(post.weights)) > 0.05,
          "a component carries weight %.4f, too small to test recovery"
          % float(np.min(post.weights)))

    # control: separating in a generic direction must BREAK the weight identity
    ctrl = _bench(separate_in_nullspace=False)
    x_c = _observation(ctrl, seed=3)
    dev = float(np.max(np.abs(ctrl.posterior(x_c).weights - ctrl.weights)))
    check(dev > 1e-3,
          "the generic-direction control also shows weights == prior weights "
          "(dev %.2e), so G2 is not testing the null-space property" % dev)
    return ("separation %.1f sd, min weight %.3f; generic-direction control "
            "breaks the identity (dev %.2e)" % (sep, float(np.min(post.weights)), dev))


# ---------------------------------------------------------------------------
# NPE training shared by G4-G6
# ---------------------------------------------------------------------------

_TRAINED = {}


def _train(n_train: int = 12000, seed: int = 0):
    key = (n_train, seed)
    if key in _TRAINED:
        return _TRAINED[key]
    bench = _bench()
    rng = np.random.default_rng(seed)
    theta = bench.prior_sample(n_train, rng)
    x = bench.simulate(theta, rng)

    cfg = NPEConfig(
        hidden_features=64, num_transforms=5, num_bins=8,
        # The prior here is an unbounded Gaussian mixture, not a box, so the
        # unconstrained transform does not apply; plain z-scoring is correct.
        z_score_theta="independent",
        training_batch_size=512, max_num_epochs=80, stop_after_epochs=12,
        show_progress_bars=False,
    )
    post = train_single(x, theta, bench.torch_prior(), config=cfg, seed=seed)
    _TRAINED[key] = (bench, post, cfg)
    return _TRAINED[key]


def _c2st(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test accuracy, 5-fold, standardised features.

    0.5 means the two sample sets are indistinguishable; 1.0 means trivially
    separable. Implemented directly rather than imported so the metric is
    pinned and cannot shift under a library update.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    n = min(a.shape[0], b.shape[0])
    X = np.concatenate([a[:n], b[:n]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    X = (X - mu) / sd
    clf = MLPClassifier(hidden_layer_sizes=(48,), max_iter=150,
                        random_state=seed, early_stopping=False)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="accuracy")))


def _npe_samples(post, x_o: np.ndarray, n: int) -> np.ndarray:
    import torch
    return post.sample((n,), x=torch.as_tensor(x_o, dtype=torch.float32),
                       show_progress_bars=False).numpy()


# ---------------------------------------------------------------------------
# G4 -- recovery, measured by C2ST against exact posterior draws
# ---------------------------------------------------------------------------

def g4_c2st_recovery() -> str:
    bench, post, _ = _train()
    rng = np.random.default_rng(7)
    scores = []
    for trial in range(3):
        x_o = _observation(bench, seed=40 + trial)
        exact = bench.posterior(x_o).sample(2000, rng)
        approx = _npe_samples(post, x_o, 2000)
        scores.append(_c2st(exact, approx, seed=trial))
    worst = max(scores)
    check(worst < 0.75,
          "C2ST %.3f: NPE samples are easily separable from exact posterior "
          "draws (0.5 = indistinguishable)" % worst)
    return "C2ST vs exact posterior: %s (worst %.3f, 0.5 is ideal)" % (
        ["%.3f" % s for s in scores], worst)


# ---------------------------------------------------------------------------
# G5 -- mode-level recovery
# ---------------------------------------------------------------------------

def g5_mode_recovery() -> str:
    bench, post, _ = _train()
    worst_w, worst_m = 0.0, 0.0
    for trial in range(3):
        x_o = _observation(bench, seed=50 + trial)
        exact = bench.posterior(x_o)
        s = _npe_samples(post, x_o, 8000)

        assign = exact.assign(s)
        found = np.array([int(np.sum(assign == k)) for k in range(N_COMP)])
        check(np.all(found > 0),
              "mode(s) %s received ZERO posterior samples -- the flow has "
              "dropped a mode" % np.flatnonzero(found == 0).tolist())

        emp_w = found / found.sum()
        w_err = float(np.max(np.abs(emp_w - exact.weights)))
        worst_w = max(worst_w, w_err)
        check(w_err < 0.10,
              "mode weights off by %.3f (exact %s, npe %s)"
              % (w_err, np.round(exact.weights, 3).tolist(), np.round(emp_w, 3).tolist()))

        scale = float(np.sqrt(np.mean(np.diag(exact.covs[0]))))
        for k in range(N_COMP):
            centre = s[assign == k].mean(axis=0)
            err = float(np.linalg.norm(centre - exact.means[k])) / scale
            worst_m = max(worst_m, err)
            check(err < 0.6,
                  "mode %d centre off by %.3f sd" % (k, err))
    return ("all %d modes found; max weight error %.3f, max centre error "
            "%.2f sd" % (N_COMP, worst_w, worst_m))


# ---------------------------------------------------------------------------
# G6 -- negative control
# ---------------------------------------------------------------------------

def g6_negative_control() -> str:
    """The prior is a WRONG posterior. It must fail the G4/G5 thresholds.

    Without this, a threshold loose enough to accommodate training noise
    might also accommodate an estimator that has learned nothing.
    """
    bench, _, _ = _train()
    rng = np.random.default_rng(11)
    x_o = _observation(bench, seed=60)
    exact = bench.posterior(x_o)

    wrong = bench.prior_sample(4000, rng)
    score = _c2st(exact.sample(4000, rng), wrong, seed=0)
    check(score > 0.75,
          "C2ST cannot even distinguish the exact posterior from the PRIOR "
          "(%.3f); the metric or the problem is not discriminating" % score)

    # and the wrong answer must also fail the mode-centre criterion
    assign = exact.assign(wrong)
    scale = float(np.sqrt(np.mean(np.diag(exact.covs[0]))))
    errs = []
    for k in range(N_COMP):
        sel = wrong[assign == k]
        if sel.shape[0] > 0:
            errs.append(float(np.linalg.norm(sel.mean(axis=0) - exact.means[k])) / scale)
    check(max(errs) > 0.6,
          "the prior passes the mode-centre criterion (max err %.2f sd); the "
          "threshold in G5 is too loose to be meaningful" % max(errs))
    return ("prior vs exact posterior: C2ST %.3f, max centre error %.2f sd -- "
            "both correctly rejected" % (score, max(errs)))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    bench = _bench()
    x_o = _observation(bench, seed=40)
    print(bench.describe(x_o))
    print("\ntraining NPE ...")
    _, post, cfg = _train()
    exact = bench.posterior(x_o)
    rng = np.random.default_rng(0)
    s = _npe_samples(post, x_o, 8000)
    assign = exact.assign(s)
    emp_w = np.array([np.mean(assign == k) for k in range(N_COMP)])
    scale = float(np.sqrt(np.mean(np.diag(exact.covs[0]))))

    print("\n%-6s %-10s %-10s %-14s" % ("mode", "w_exact", "w_npe", "centre err (sd)"))
    for k in range(N_COMP):
        centre = s[assign == k].mean(axis=0) if np.any(assign == k) else np.full(N_DIM, np.nan)
        err = float(np.linalg.norm(centre - exact.means[k])) / scale
        print("%-6d %-10.4f %-10.4f %-14.3f" % (k, exact.weights[k], emp_w[k], err))
    print("\nC2ST vs exact posterior draws: %.3f  (0.5 = indistinguishable)"
          % _c2st(exact.sample(4000, rng), s[:4000]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip NPE training (G4-G6)")
    ap.add_argument("-k", dest="selector", default="", help="substring filter on test id")
    ap.add_argument("--demo", action="store_true", help="print a worked example instead")
    args = ap.parse_args()

    if args.demo:
        demo()
        return 0

    print("=" * 74)
    print("Recover %d Gaussians in %d dims from %d-dim observations"
          % (N_COMP, N_DIM, N_OBS))
    print("=" * 74)

    print("\n[ground truth is itself correct]")
    run("G1_analytic_is_correct", g1_analytic_posterior_is_correct, args.selector)
    run("G2_nullspace_properties", g2_nullspace_properties, args.selector)
    run("G3_discriminating", g3_problem_is_discriminating, args.selector)

    if not args.fast:
        print("\n[NPE recovery -- trains a network, slower]")
        run("G4_c2st_recovery", g4_c2st_recovery, args.selector)
        run("G5_mode_recovery", g5_mode_recovery, args.selector)
        run("G6_negative_control", g6_negative_control, args.selector)
    else:
        print("\n[NPE recovery -- SKIPPED (--fast)]")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 74)
    print("%d passed, %d failed, %d total" % (n_pass, n_fail, len(RESULTS)))
    print("=" * 74)
    if n_fail:
        print("\nFailures:")
        for tid, ok, detail in RESULTS:
            if not ok:
                print("  %s: %s" % (tid, detail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
