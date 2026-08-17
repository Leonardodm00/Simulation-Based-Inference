#!/usr/bin/env python3
"""
smoke_test_npe.py -- correctness and shape checks for the NPE stage.

Run:
    python3 smoke_test_npe.py            # full suite
    python3 smoke_test_npe.py --fast     # skip the slow end-to-end tests
    python3 smoke_test_npe.py -k T5      # run only tests whose id contains T5

What each test actually establishes:

  T1  Contract coordinate algebra is an exact involution on the box edges,
      and the mixed ln/linear split is applied to the right axes.
  T2  Assertion A3 fires on a degenerate (frozen) axis, and A2 fires on a
      length mismatch. These are the two failures that would otherwise
      surface as a divergent loss rather than an error.
  T3  Array validation catches a constant column (A4), an out-of-box row
      (A5), a non-unit-norm embedding (A7), and a NaN (A9).
  T4  Shard write/read is lossless and column order comes from the sidecar,
      not from the file, so a reordered export cannot permute the axes.
  T5  Two shards that disagree on the prior box or on the DSN checkpoint
      digest are refused rather than pooled (A10).
  T6  CORRECTNESS ANCHOR. On a linear-Gaussian problem with a box prior,
      where the true posterior is computable exactly on a grid, the trained
      NPE recovers it. This is the only test that validates the estimator
      itself rather than the plumbing: if it passes, the training loop, the
      unconstrained transform, and log_prob are all behaving.
  T7  The ensemble combines as the ARITHMETIC mixture, not the geometric
      mean. Asserted numerically against both formulas, since getting this
      backwards makes the posterior sharper and the overconfidence worse.
  T8  End-to-end at the real shapes (p=27, E=12, unit-norm z): posterior
      samples land inside the box, log_prob is finite, sampling works.
  T9  Simulation-based calibration ranks on the T6 toy are consistent with
      uniform. A calibrated estimator is a necessary, not sufficient,
      condition -- it says nothing about whether the summary is informative.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_contract import (  # noqa: E402
    Contract,
    load_shard,
    load_shards,
    make_synthetic_shard,
    validate_arrays,
)
from npe_model import NPEConfig, train_single  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------

RESULTS: List[Tuple[str, bool, str]] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, True, detail))
        print("  PASS  %-32s %s" % (test_id, detail), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, False, "%s | %s" % (exc, tb)))
        print("  FAIL  %-32s %s" % (test_id, exc), flush=True)


# ---------------------------------------------------------------------------
# T1 -- coordinate algebra
# ---------------------------------------------------------------------------

def t1_coordinate_algebra() -> str:
    p, n_log = 6, 4
    lo = np.array([-2.0, -1.0, 0.5, -3.0, 10.0, -50.0])
    hi = np.array([1.0, 2.0, 3.0, 0.0, 40.0, -10.0])
    c = Contract(
        param_names=["a", "b", "c", "d", "e", "f"],
        coord=["ln"] * n_log + ["linear"] * (p - n_log),
        bounds_theta=np.stack([lo, hi], axis=1),
        embedding_dim=4,
    )
    c.validate()

    # involution at both box edges and at interior points
    for edge in (c.low, c.high, 0.5 * (c.low + c.high)):
        back = c.to_inference(c.to_natural(edge))
        check(np.allclose(back, edge, rtol=0, atol=1e-12),
              "to_inference(to_natural(x)) != x at %s" % edge)

    # the ln transform must hit exactly the log axes and nothing else
    nat = c.to_natural(c.low)
    check(np.allclose(nat[:n_log], np.exp(c.low[:n_log])), "log axes not exponentiated")
    check(np.allclose(nat[n_log:], c.low[n_log:]), "linear axes were transformed")

    # batched form agrees with the single-vector form
    batch = np.stack([c.low, c.high, 0.5 * (c.low + c.high)])
    check(np.allclose(c.to_natural(batch)[0], c.to_natural(c.low)),
          "batched to_natural disagrees with single-row")

    # a non-positive value on a log axis must raise, not silently produce nan
    bad = c.to_natural(c.low).copy()
    bad[0] = -1.0
    try:
        c.to_inference(bad)
        raise AssertionError("to_inference accepted a non-positive log-axis value")
    except ValueError:
        pass
    return "involution exact to 1e-12; ln applied to %d/%d axes" % (n_log, p)


# ---------------------------------------------------------------------------
# T2 -- contract-level assertions
# ---------------------------------------------------------------------------

def t2_contract_assertions() -> str:
    # A3: a frozen axis with a point interval must be refused
    try:
        Contract(param_names=["ok", "frozen"], coord=["linear", "linear"],
                 bounds_theta=np.array([[0.0, 1.0], [2.0, 2.0]]),
                 embedding_dim=4).validate()
        raise AssertionError("A3 did not fire on a point interval")
    except ValueError as exc:
        check("A3" in str(exc), "wrong error for degenerate bounds: %s" % exc)

    # A2: length mismatch
    try:
        Contract(param_names=["a", "b"], coord=["linear"],
                 bounds_theta=np.array([[0.0, 1.0], [0.0, 1.0]]),
                 embedding_dim=4).validate()
        raise AssertionError("A2 did not fire on a length mismatch")
    except ValueError as exc:
        check("A2" in str(exc), "wrong error for length mismatch: %s" % exc)

    # A2: unknown coordinate label
    try:
        Contract(param_names=["a"], coord=["log10"],
                 bounds_theta=np.array([[0.0, 1.0]]), embedding_dim=4).validate()
        raise AssertionError("A2 did not fire on an unknown coord label")
    except ValueError as exc:
        check("A2" in str(exc), "wrong error for bad coord: %s" % exc)

    # A2: duplicate names
    try:
        Contract(param_names=["a", "a"], coord=["linear", "linear"],
                 bounds_theta=np.array([[0.0, 1.0], [0.0, 1.0]]),
                 embedding_dim=4).validate()
        raise AssertionError("A2 did not fire on duplicate names")
    except ValueError as exc:
        check("A2" in str(exc), "wrong error for duplicates: %s" % exc)

    return "A3 and A2 fire on all four malformed contracts"


# ---------------------------------------------------------------------------
# T3 -- array validation
# ---------------------------------------------------------------------------

def t3_array_validation() -> str:
    z, theta, c, _ = make_synthetic_shard("", n_rows=256, p=5, embedding_dim=6,
                                          n_log_axes=3, seed=3, write=False)
    check(validate_arrays(z, theta, c).ok, "clean shard failed validation")

    # A4 exactly constant column. Note the column is set from a single row,
    # so numpy's std returns O(1e-16) round-off rather than 0.0; the check
    # must be relative to the prior width or it never fires.
    th_bad = theta.copy()
    th_bad[:, 2] = th_bad[0, 2]
    rep = validate_arrays(z, th_bad, c)
    check(not rep.ok and any("A4" in f for f in rep.failures),
          "A4 not detected on an exactly constant column")

    # A4 near-constant column: a frozen axis jittered by numerical noise
    # should still be caught, and a genuinely varying one must not be.
    th_bad = theta.copy()
    th_bad[:, 3] = th_bad[0, 3] + 1e-14 * np.arange(th_bad.shape[0])
    rep = validate_arrays(z, th_bad, c)
    check(not rep.ok and any("A4" in f for f in rep.failures),
          "A4 not detected on a near-constant column")
    check(validate_arrays(z, theta, c).ok, "A4 false-positived on a healthy shard")

    # A5 out of box
    th_bad = theta.copy()
    th_bad[0, 1] = c.high[1] + 1.0
    rep = validate_arrays(z, th_bad, c)
    check(not rep.ok and any("A5" in f for f in rep.failures), "A5 not detected")

    # A7 broken normalisation
    z_bad = z.copy()
    z_bad[5] *= 1.5
    rep = validate_arrays(z_bad, theta, c)
    check(not rep.ok and any("A7" in f for f in rep.failures), "A7 not detected")

    # A9 nan
    z_bad = z.copy()
    z_bad[7, 0] = np.nan
    rep = validate_arrays(z_bad, theta, c)
    check(not rep.ok and any("A9" in f for f in rep.failures), "A9 not detected")

    # real-data shard: theta is None, A4/A5 must be skipped without error
    check(validate_arrays(z, None, c).ok, "label-free shard failed validation")
    return "A4, A5, A7, A9 each detected; label-free shard accepted"


# ---------------------------------------------------------------------------
# T4 -- shard io round trip
# ---------------------------------------------------------------------------

def t4_shard_roundtrip() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        z, theta, c, path = make_synthetic_shard(tmp, name="rt_000", n_rows=512,
                                                 p=9, embedding_dim=7, n_log_axes=5,
                                                 seed=11)
        z2, th2, c2 = load_shard(path)
        check(np.allclose(z, z2, rtol=0, atol=0), "z changed on round trip")
        check(np.allclose(theta, th2, rtol=0, atol=0), "theta changed on round trip")
        check(c2.param_names == c.param_names, "param_names changed on round trip")
        check(c2.coord == c.coord, "coord changed on round trip")
        check(np.allclose(c2.bounds_theta, c.bounds_theta), "bounds changed on round trip")

        # column order must come from the sidecar, not the file: shuffle the
        # parquet columns and confirm the loaded arrays are unchanged
        import pandas as pd
        df = pd.read_parquet(path)
        df = df[list(reversed(list(df.columns)))]
        df.to_parquet(path, index=False)
        z3, th3, _ = load_shard(path)
        check(np.allclose(z, z3, rtol=0, atol=0), "z permuted by column reorder")
        check(np.allclose(theta, th3, rtol=0, atol=0), "theta permuted by column reorder")
    return "lossless round trip; immune to parquet column reordering"


# ---------------------------------------------------------------------------
# T5 -- cross-shard compatibility (A10)
# ---------------------------------------------------------------------------

def t5_cross_shard() -> str:
    import json as _json
    with tempfile.TemporaryDirectory() as tmp:
        _, _, c, p1 = make_synthetic_shard(tmp, name="s0", n_rows=128, p=4,
                                           embedding_dim=5, n_log_axes=2, seed=1)
        _, _, _, p2 = make_synthetic_shard(tmp, name="s1", n_rows=128, p=4,
                                           embedding_dim=5, n_log_axes=2, seed=1)
        # identical settings pool fine
        z, th, _ = load_shards([p1, p2])
        check(z.shape[0] == 256, "pooling two compatible shards gave %d rows" % z.shape[0])

        # perturb the second shard's prior box -> must be refused
        side2 = os.path.splitext(p2)[0] + ".json"
        with open(side2, "r", encoding="utf-8") as fh:
            d = _json.load(fh)
        d["bounds_theta"][0][1] += 0.5
        with open(side2, "w", encoding="utf-8") as fh:
            _json.dump(d, fh)
        try:
            load_shards([p1, p2])
            raise AssertionError("A10 did not fire on differing bounds_theta")
        except ValueError as exc:
            check("A10" in str(exc), "wrong error for incompatible bounds: %s" % exc)

        # restore bounds, perturb the checkpoint digest -> must also be refused
        d["bounds_theta"][0][1] -= 0.5
        d["embedding"]["dsn_checkpoint_sha256"] = "a-different-encoder"
        with open(side2, "w", encoding="utf-8") as fh:
            _json.dump(d, fh)
        try:
            load_shards([p1, p2])
            raise AssertionError("A10 did not fire on differing checkpoint digest")
        except ValueError as exc:
            check("A10" in str(exc), "wrong error for differing digest: %s" % exc)
    return "compatible shards pool; differing box and differing digest both refused"


# ---------------------------------------------------------------------------
# T6 -- CORRECTNESS ANCHOR: linear-Gaussian with a box prior
# ---------------------------------------------------------------------------

def _lingauss_problem(seed: int = 7, sigma: float = 0.15, dim_x: int = 3):
    """theta ~ U([0,1]^2);  x = A theta + b + N(0, sigma^2 I).

    The posterior is p(theta | x) proportional to
        exp( -||x - A theta - b||^2 / (2 sigma^2) ) * 1{theta in [0,1]^2}
    which has no closed form over the box but is computable exactly on a
    grid. That grid is the reference the NPE is scored against.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2, dim_x))
    b = rng.normal(size=(dim_x,))
    return A, b, sigma


def _lingauss_reference(x_o, A, b, sigma, n_grid=200):
    g = (np.arange(n_grid) + 0.5) / n_grid
    T1, T2 = np.meshgrid(g, g, indexing="ij")
    theta_grid = np.stack([T1.ravel(), T2.ravel()], axis=1)
    mu = theta_grid @ A + b[None, :]
    ll = -0.5 * np.sum((x_o[None, :] - mu) ** 2, axis=1) / sigma ** 2
    ll -= ll.max()
    w = np.exp(ll)
    w /= w.sum()
    return theta_grid, w.reshape(n_grid, n_grid), g


def t6_correctness_anchor() -> str:
    import torch
    from sbi.utils import BoxUniform

    A, b, sigma = _lingauss_problem()
    n_train, n_grid = 20000, 160
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, 1.0, size=(n_train, 2))
    x = theta @ A + b[None, :] + sigma * rng.normal(size=(n_train, A.shape[1]))

    prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
    cfg = NPEConfig(hidden_features=64, num_transforms=5, num_bins=8,
                    training_batch_size=256, max_num_epochs=150,
                    stop_after_epochs=15, show_progress_bars=False)
    post = train_single(x, theta, prior, config=cfg, seed=0)

    tv_all, err_all = [], []
    for trial in range(3):
        theta_star = np.array([0.3 + 0.2 * trial, 0.7 - 0.2 * trial])
        x_o = theta_star @ A + b + sigma * np.random.default_rng(100 + trial).normal(size=A.shape[1])

        grid, ref, g = _lingauss_reference(x_o, A, b, sigma, n_grid=n_grid)
        with torch.no_grad():
            lp = post.log_prob(
                torch.as_tensor(grid, dtype=torch.float32),
                x=torch.as_tensor(x_o, dtype=torch.float32),
            ).numpy()
        lp = np.nan_to_num(lp, neginf=-1e30)
        lp -= lp.max()
        q = np.exp(lp)
        q /= q.sum()
        q = q.reshape(n_grid, n_grid)

        tv = 0.5 * np.sum(np.abs(q - ref))
        ref_mean = np.array([np.sum(ref.sum(axis=1) * g), np.sum(ref.sum(axis=0) * g)])
        npe_mean = np.array([np.sum(q.sum(axis=1) * g), np.sum(q.sum(axis=0) * g)])
        tv_all.append(tv)
        err_all.append(float(np.linalg.norm(ref_mean - npe_mean)))

    tv_max, err_max = max(tv_all), max(err_all)
    check(tv_max < 0.25, "total variation vs analytic reference too large: %.3f" % tv_max)
    check(err_max < 0.05, "posterior mean error vs reference too large: %.4f" % err_max)
    return "TV<=%.3f, |mean error|<=%.4f vs exact grid posterior (3 observations)" % (tv_max, err_max)


# ---------------------------------------------------------------------------
# T7 -- ensemble combination rule
# ---------------------------------------------------------------------------

def t7_ensemble_is_mixture() -> str:
    import torch
    from sbi.utils import BoxUniform
    from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

    A, b, sigma = _lingauss_problem()
    rng = np.random.default_rng(1)
    n = 4000
    theta = rng.uniform(0.0, 1.0, size=(n, 2))
    x = theta @ A + b[None, :] + sigma * rng.normal(size=(n, A.shape[1]))
    prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
    cfg = NPEConfig(hidden_features=32, num_transforms=3, num_bins=6,
                    training_batch_size=256, max_num_epochs=40,
                    stop_after_epochs=8, show_progress_bars=False)

    members = [train_single(x, theta, prior, config=cfg, seed=s) for s in (0, 1, 2)]
    ens = EnsemblePosterior(members)

    x_o = torch.as_tensor(np.array([0.4, 0.6]) @ A + b, dtype=torch.float32)
    pts = torch.as_tensor(rng.uniform(0.05, 0.95, size=(64, 2)), dtype=torch.float32)

    with torch.no_grad():
        per_member = torch.stack([m.log_prob(pts, x=x_o) for m in members])
        got = ens.log_prob(pts, x=x_o)

    mixture = torch.logsumexp(per_member, dim=0) - math.log(len(members))
    geometric = per_member.mean(dim=0)

    d_mix = float((got - mixture).abs().max())
    d_geo = float((got - geometric).abs().max())
    check(d_mix < 1e-4,
          "ensemble log_prob is NOT the arithmetic mixture (max dev %.3e)" % d_mix)
    check(d_geo > 1e-3,
          "ensemble log_prob coincides with the geometric mean; the mixture "
          "and product-of-experts rules are indistinguishable on this test, "
          "so the test is not discriminating")

    # the mixture must never be sharper than the sharpest member at its own mode
    check(float(mixture.max()) <= float(per_member.max()) + 1e-5,
          "mixture peak exceeds the sharpest member -- not a mixture")
    return "matches logsumexp - log n (dev %.1e), differs from geometric mean (dev %.2e)" % (d_mix, d_geo)


# ---------------------------------------------------------------------------
# T8 -- end to end at production shapes
# ---------------------------------------------------------------------------

def t8_end_to_end_shapes() -> str:
    import torch

    # Sized to run inside ~4 GB. This test checks shapes, support
    # confinement, and the back-map -- not statistical quality -- so the
    # training set only needs to be large enough for the flow to be
    # well-defined. Scale n_rows and n_draw up on the cluster.
    n_rows, n_draw = 3000, 512
    z, theta, c, _ = make_synthetic_shard("", n_rows=n_rows, p=27, embedding_dim=12,
                                          n_log_axes=19, seed=42, write=False)
    check(validate_arrays(z, theta, c).ok, "synthetic production-shape shard failed validation")

    prior = c.prior()
    cfg = NPEConfig(hidden_features=48, num_transforms=3, num_bins=8,
                    training_batch_size=256, max_num_epochs=25,
                    stop_after_epochs=6, show_progress_bars=False)
    post = train_single(z, theta, prior, config=cfg, seed=0)

    z_o = torch.as_tensor(z[0], dtype=torch.float32)
    samples = post.sample((n_draw,), x=z_o, show_progress_bars=False).numpy()
    check(samples.shape == (n_draw, 27), "sample shape %s != (%d, 27)" % (samples.shape, n_draw))
    check(np.all(np.isfinite(samples)), "non-finite posterior samples")

    inside = np.all((samples >= c.low[None, :]) & (samples <= c.high[None, :]), axis=1)
    frac_in = float(inside.mean())
    check(frac_in > 0.999,
          "only %.4f of posterior samples lie inside the prior box; the "
          "unconstrained transform is not confining the flow" % frac_in)

    with torch.no_grad():
        lp = post.log_prob(torch.as_tensor(samples[:128], dtype=torch.float32), x=z_o)
    check(bool(torch.isfinite(lp).all()), "non-finite log_prob on in-box samples")

    # posterior samples mapped back to natural units must be finite and, on
    # log axes, strictly positive
    nat = c.to_natural(samples[:128])
    check(np.all(np.isfinite(nat)), "non-finite natural-unit samples")
    check(np.all(nat[:, c.log_mask] > 0.0), "non-positive natural value on a log axis")
    return "p=27, E=12: %.4f of samples in-box, log_prob finite, back-map valid" % frac_in


# ---------------------------------------------------------------------------
# T9 -- simulation-based calibration on the toy
# ---------------------------------------------------------------------------

def t9_sbc_ranks() -> str:
    import torch
    from scipy import stats
    from sbi.utils import BoxUniform

    A, b, sigma = _lingauss_problem()
    rng = np.random.default_rng(5)
    n = 12000
    theta = rng.uniform(0.0, 1.0, size=(n, 2))
    x = theta @ A + b[None, :] + sigma * rng.normal(size=(n, A.shape[1]))
    prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
    cfg = NPEConfig(hidden_features=64, num_transforms=5, num_bins=8,
                    training_batch_size=256, max_num_epochs=150,
                    stop_after_epochs=15, show_progress_bars=False)
    post = train_single(x, theta, prior, config=cfg, seed=0)

    n_sbc, n_post = 200, 128
    th_sbc = rng.uniform(0.0, 1.0, size=(n_sbc, 2))
    x_sbc = th_sbc @ A + b[None, :] + sigma * rng.normal(size=(n_sbc, A.shape[1]))

    ranks = np.zeros((n_sbc, 2), dtype=int)
    for i in range(n_sbc):
        s = post.sample((n_post,), x=torch.as_tensor(x_sbc[i], dtype=torch.float32),
                        show_progress_bars=False).numpy()
        ranks[i] = np.sum(s < th_sbc[i][None, :], axis=0)

    pvals = []
    for k in range(2):
        u = (ranks[:, k] + rng.uniform(size=n_sbc)) / (n_post + 1)
        pvals.append(float(stats.kstest(u, "uniform").pvalue))
    check(min(pvals) > 0.005,
          "SBC ranks reject uniformity (KS p = %s); the estimator is miscalibrated"
          % ["%.4f" % p for p in pvals])
    return "KS p-values vs uniform: %s (n_sbc=%d)" % (["%.3f" % p for p in pvals], n_sbc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip the tests that train networks (T6-T9)")
    ap.add_argument("-k", dest="selector", default="",
                    help="run only tests whose id contains this substring")
    args = ap.parse_args()

    print("=" * 74)
    print("NPE smoke test suite")
    print("=" * 74)

    print("\n[contract and data plumbing]")
    run("T1_coordinate_algebra", t1_coordinate_algebra, args.selector)
    run("T2_contract_assertions", t2_contract_assertions, args.selector)
    run("T3_array_validation", t3_array_validation, args.selector)
    run("T4_shard_roundtrip", t4_shard_roundtrip, args.selector)
    run("T5_cross_shard", t5_cross_shard, args.selector)

    if not args.fast:
        print("\n[estimator correctness -- trains networks, slower]")
        run("T6_correctness_anchor", t6_correctness_anchor, args.selector)
        run("T7_ensemble_is_mixture", t7_ensemble_is_mixture, args.selector)
        run("T8_end_to_end_shapes", t8_end_to_end_shapes, args.selector)
        run("T9_sbc_ranks", t9_sbc_ranks, args.selector)
    else:
        print("\n[estimator correctness -- SKIPPED (--fast)]")

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
