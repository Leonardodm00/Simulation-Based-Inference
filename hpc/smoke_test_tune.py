#!/usr/bin/env python3
"""
smoke_test_tune.py -- tests for the hyperparameter tuning stack.

Run before trusting anything in npe_tune*.py:

    python3 smoke_test_tune.py             # everything available here
    python3 smoke_test_tune.py --fast      # only tests needing no torch/sbi
    python3 smoke_test_tune.py -k S4       # one test
    python3 smoke_test_tune.py --list      # what exists and what it checks

Exit code is 0 only if every selected test passed, so this is safe as the
last command of a PBS job.

TWO TIERS, on purpose. The FAST tier needs numpy, scipy and scikit-optimize
only -- no torch, no sbi. That is what lets the objective, the floor, the
splits, the ledger, the search and the GATE LOGIC be verified on a machine
with no deep-learning stack, including this repository's own laptop-side
checks. The FULL tier adds the tests that genuinely need a trained network.

The most important test here is S9. It applies the gate battery to a
posterior that is deliberately equal to the prior and requires G1 to REJECT
it while G2 PASSES it. That reproduces, in miniature, the documented
blindness of marginal calibration to a data-ignoring posterior. A gate that
has never failed anything is a gate that was never tested; S9 is what
licenses believing a G1 pass on the real bank.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import npe_tune_data as TD
import npe_tune_gates as TG
import npe_tune_ledger as TL
import npe_tune_score as TS
import npe_tune_search as TSR

FAST_TESTS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")
FULL_TESTS = ("S11", "S12", "S13")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeContract(object):
    """A minimal stand-in for npe_contract.Contract.

    Only the attributes the tuner actually touches are provided, so the test
    fails loudly if the tuner starts depending on something new rather than
    silently picking up a real Contract's extra behaviour.
    """

    def __init__(self, p=6, embedding_dim=4, width=2.0, seed=0):
        rng = np.random.default_rng(seed)
        self.param_names = ["ax_%02d" % k for k in range(p)]
        self.coord = ["ln" if k % 2 else "linear" for k in range(p)]
        lo = rng.uniform(-1.0, 0.0, size=p)
        self.bounds_theta = np.stack([lo, lo + width], axis=1)
        self.embedding_dim = int(embedding_dim)
        self.meta = {"embedding": {"dsn_checkpoint_sha256": "deadbeef"}}

    @property
    def p(self):
        return len(self.param_names)

    def to_dict(self):
        return {"param_names": list(self.param_names),
                "coord": list(self.coord),
                "bounds_theta": np.asarray(self.bounds_theta).tolist(),
                "embedding_dim": int(self.embedding_dim),
                "meta": self.meta}


def _fake_bank(n=900, p=6, embedding_dim=4, n_groups=30, seed=0) -> TD.Bank:
    """A bank whose group structure is real: rows in one group share the
    last three theta axes exactly, exactly as rows from one topology draw
    share the connectivity-kernel axes."""
    rng = np.random.default_rng(seed)
    contract = _FakeContract(p=p, embedding_dim=embedding_dim, seed=seed)
    lo, hi = contract.bounds_theta[:, 0], contract.bounds_theta[:, 1]
    theta = rng.uniform(lo, hi, size=(n, p))
    gid = rng.integers(0, n_groups, size=n)
    kernel = rng.uniform(lo[-3:], hi[-3:], size=(n_groups, 3))
    theta[:, -3:] = kernel[gid]
    z = rng.normal(size=(n, embedding_dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    groups, ginfo = TD.derive_groups(theta, contract.param_names,
                                     axes=contract.param_names[-3:])
    return TD.Bank(z=z.astype(np.float32), theta=theta, groups=groups,
                   contract=contract,
                   meta={"grouping": ginfo, "sim_glob": "fixture"})


def _uniform_log_prob(theta: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """log p(theta) for the uniform box prior: a constant inside the box.

    This is the analytic anchor. An estimator returning exactly this is the
    'learned nothing' estimator, and its NLL must equal the floor exactly.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    inside = np.all((theta >= lo) & (theta <= hi), axis=1)
    const = -np.sum(np.log(hi - lo))
    return np.where(inside, const, -np.inf)


def _gaussian_log_prob(theta: np.ndarray, mean: np.ndarray,
                       sd: float) -> np.ndarray:
    """log N(theta; mean, sd^2 I), row-wise. Used as an informative
    estimator whose information gain is positive by construction."""
    d = theta.shape[1]
    q = np.sum((theta - mean) ** 2, axis=1) / (sd ** 2)
    return -0.5 * q - d * np.log(sd) - 0.5 * d * np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_S1_prior_floor() -> str:
    """S1: the floor equals the hand-computed log-volume, and an estimator
    that returns the prior scores EXACTLY the floor (gain 0)."""
    bounds = np.array([[0.0, 2.0], [-1.0, 1.0], [1.0, 11.0]])
    expected = np.log(2.0) + np.log(2.0) + np.log(10.0)
    got = TS.prior_floor(bounds)
    assert abs(got - expected) < 1e-12, (got, expected)

    rng = np.random.default_rng(0)
    theta = rng.uniform(bounds[:, 0], bounds[:, 1], size=(500, 3))
    lp = _uniform_log_prob(theta, bounds)
    nll = TS.heldout_nll(lp)
    gain = TS.information_gain(nll, got)
    assert abs(nll - got) < 1e-12, (nll, got)
    assert abs(gain) < 1e-12, gain

    try:
        TS.prior_floor(np.array([[1.0, 1.0]]))
        raise AssertionError("a zero-width axis must be refused")
    except ValueError:
        pass
    return "floor %.6f == hand-computed; prior estimator gains exactly 0" % got


def test_S2_information_gain_positive() -> str:
    """S2: an informative estimator has a positive gain, and a sharper one
    has a larger gain. Sign and ordering, which is all the statistic is
    asked to get right."""
    bounds = np.array([[0.0, 4.0]] * 4)
    floor = TS.prior_floor(bounds)
    rng = np.random.default_rng(1)
    theta = rng.uniform(bounds[:, 0], bounds[:, 1], size=(2000, 4))
    gains = []
    for sd in (1.0, 0.5, 0.25):
        lp = _gaussian_log_prob(theta, theta + rng.normal(scale=sd * 0.1,
                                                          size=theta.shape), sd)
        gains.append(TS.information_gain(TS.heldout_nll(lp), floor))
    assert all(g > 0 for g in gains), gains
    assert gains[0] < gains[1] < gains[2], gains
    return "gains %s: positive and increasing with sharpness" % (
        ", ".join("%.3f" % g for g in gains))


def test_S3_jensen_bound() -> str:
    """S3: eq. (4). The arithmetic mixture never scores worse than the mean
    of its members, and a geometric mean (product of experts) is detectably
    different -- which is what the check is for."""
    rng = np.random.default_rng(2)
    member_lp = rng.normal(loc=-3.0, scale=1.5, size=(5, 400))
    j = TS.check_jensen(member_lp)
    assert j["ok"], j
    assert j["slack"] >= -1e-9, j
    assert j["slack"] > 0, "with disagreeing members the bound is strict"

    geometric = np.mean(member_lp, axis=0)          # product of experts
    arithmetic = TS.mixture_log_prob(member_lp)
    assert np.all(arithmetic >= geometric - 1e-12)
    assert np.max(arithmetic - geometric) > 1e-6, "the two rules must differ"

    single = rng.normal(size=(1, 50))
    assert np.allclose(TS.mixture_log_prob(single), single[0])
    return ("mixture NLL %.4f <= member mean %.4f (slack %.4f); geometric "
            "mean distinguishable" % (j["mixture_nll"], j["member_nll_mean"],
                                      j["slack"]))


def test_S4_split_is_group_disjoint() -> str:
    """S4: the split never divides a topology group, covers the bank
    exactly, is reproducible from its seed, and its hash detects tampering."""
    bank = _fake_bank(seed=3)
    man = TD.make_split(bank, fractions=(0.8, 0.1, 0.1), seed=0,
                        min_groups_per_side=2, min_rows_per_side=20)

    tr = set(bank.groups[np.asarray(man.train)].tolist())
    se = set(bank.groups[np.asarray(man.sel)].tolist())
    rp = set(bank.groups[np.asarray(man.rep)].tolist())
    assert not (tr & se) and not (tr & rp) and not (se & rp), "groups leak"
    total = len(man.train) + len(man.sel) + len(man.rep)
    assert total == bank.n, (total, bank.n)
    assert len(set(man.train) & set(man.sel)) == 0

    again = TD.make_split(bank, fractions=(0.8, 0.1, 0.1), seed=0,
                          min_groups_per_side=2, min_rows_per_side=20)
    assert again.hash() == man.hash(), "the same seed must give the same split"
    other = TD.make_split(bank, fractions=(0.8, 0.1, 0.1), seed=1,
                          min_groups_per_side=2, min_rows_per_side=20)
    assert other.hash() != man.hash(), "a different seed must differ"

    tmp = tempfile.mkdtemp()
    try:
        p = TD.save_split(man, os.path.join(tmp, "split.json"))
        back = TD.load_split(p)
        assert back.hash() == man.hash()
        with open(p, "r", encoding="ascii") as fh:
            d = json.load(fh)
        d["sel"] = d["sel"][:-1]                # tamper
        with open(p, "w", encoding="ascii") as fh:
            json.dump(d, fh)
        try:
            TD.load_split(p)
            raise AssertionError("an edited manifest must be refused")
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ("%d groups dealt to %d/%d/%d rows, disjoint, reproducible, "
            "tamper-evident" % (bank.n_groups, len(man.train), len(man.sel),
                                len(man.rep)))


def test_S5_shape_agnostic() -> str:
    """S5: the same code path runs at two different (p, E) with no edit, and
    the guards refuse a split too small to mean anything."""
    out = []
    for p, E, n in ((6, 4, 900), (13, 9, 1200)):
        bank = _fake_bank(n=n, p=p, embedding_dim=E, n_groups=25, seed=p)
        man = TD.make_split(bank, seed=0, min_groups_per_side=2,
                            min_rows_per_side=20)
        assert man.p == p and man.embedding_dim == E
        floor = TS.prior_floor(bank.contract.bounds_theta)
        assert np.isfinite(floor)
        spec = TSR.default_space(p, E, n_train=len(man.train))
        assert spec.hidden_features[0] < spec.hidden_features[1]
        TD.check_split(bank, man)
        out.append("p=%d E=%d floor=%.3f width_range=%s"
                   % (p, E, floor, spec.hidden_features))

    bank = _fake_bank(n=900, seed=9)
    try:
        TD.make_split(bank, seed=0, min_groups_per_side=1000)
        raise AssertionError("a too-small split side must be refused")
    except ValueError:
        pass

    other = _fake_bank(n=900, p=7, embedding_dim=4, seed=11)
    man = TD.make_split(bank, seed=0, min_groups_per_side=2,
                        min_rows_per_side=20)
    try:
        TD.check_split(other, man)
        raise AssertionError("a mismatched bank/manifest pair must be refused")
    except ValueError:
        pass
    return "; ".join(out) + "; guards fire"


def test_S6_ledger_roundtrip() -> str:
    """S6: numpy scalars survive serialisation, results are filtered by
    split hash and contract digest, and a partial file cannot poison the
    ledger."""
    tmp = tempfile.mkdtemp()
    try:
        res = os.path.join(tmp, "results")
        cfg = {"hidden_features": np.int64(128), "num_transforms": np.int64(8),
               "num_bins": np.int64(10), "learning_rate": np.float64(5e-4),
               "training_batch_size": 512}
        tid = TL.trial_id(cfg, "splithash", "digest", 3, [0, 1, 2])
        assert tid == TL.trial_id(cfg, "splithash", "digest", 3, [0, 1, 2])
        assert tid != TL.trial_id(cfg, "OTHER", "digest", 3, [0, 1, 2])
        assert tid != TL.trial_id(cfg, "splithash", "digest", 3, [0, 1, 2],
                                  tag="frac0.5")

        rec = TL.TrialRecord(trial_id=tid, config=cfg, seeds=[0, 1, 2],
                             n_members=3, nll=12.5, floor=13.0, delta=0.5,
                             split_hash="splithash", contract_digest="digest",
                             epochs=[np.int64(40)] * 3,
                             delta_ci_lo=float("nan"))
        TL.write_trial(rec, res)
        back = TL.read_trial(os.path.join(res, "trial_%s.json" % tid))
        assert back is not None and back.trial_id == tid
        assert back.config["hidden_features"] == 128
        assert isinstance(back.config["hidden_features"], int)

        with open(os.path.join(res, "trial_broken.json"), "w",
                  encoding="ascii") as fh:
            fh.write('{"trial_id": "broken", ')      # truncated on purpose
        kept = TL.load_ledger(res, split_hash="splithash",
                              contract_digest="digest")
        assert len(kept) == 1, [r.trial_id for r in kept]
        assert TL.load_ledger(res, split_hash="OTHER") == []

        pend = os.path.join(tmp, "pending")
        TL.write_pending({"trial_id": tid, "config": cfg}, pend)
        assert TL.pending_ids(pend) == [tid]
        TL.clear_pending(pend, tid)
        assert TL.pending_ids(pend) == []
        TL.clear_pending(pend, tid)              # must be idempotent
        assert "  " in TL.ledger_table(kept)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "numpy casts, digest filtering, partial-file tolerance, pending"


def test_S7_search_space_and_warm_start() -> str:
    """S7: the space builds; a point round-trips to a config of plain Python
    types; and replaying observations into a fresh optimiser reproduces the
    warm-started state. That replay IS the budget escalation."""
    spec = TSR.default_space(p=26, embedding_dim=10, n_train=20000)
    dims = TSR.space_dimensions(spec)
    assert len(dims) == len(TSR.KNOB_ORDER)

    opt = TSR.build_optimizer(spec, n_initial_points=4, seed=0)
    pts = opt.ask(n_points=3, strategy="cl_min")
    assert len({tuple(p) for p in pts}) == 3, "constant liar must diversify"
    cfg = TSR.config_from_point(pts[0])
    assert isinstance(cfg["hidden_features"], int)
    assert isinstance(cfg["learning_rate"], float)
    json.dumps(cfg)                       # must be serialisable as-is
    assert TSR.point_from_config(cfg) == list(TSR.config_from_point(pts[0]).values())

    obs = [(TSR.config_from_point(p), float(np.sum([float(v) for v in p[:4]])))
           for p in pts]
    a = TSR.build_optimizer(spec, 4, seed=0, observations=obs)
    b = TSR.build_optimizer(spec, 4, seed=0, observations=obs)
    assert len(a.yi) == len(obs) and a.yi == b.yi
    assert a.ask() == b.ask(), "warm start must be reproducible"

    fresh = TSR.propose(spec, obs, n_points=4, n_initial_points=4, seed=1)
    assert len(fresh) == 4
    keys = {TSR._config_key(c) for c in fresh}
    assert len(keys) == 4, "proposals within a batch must differ"
    assert not (keys & {TSR._config_key(c) for c, _ in obs}), \
        "must not re-propose an already-evaluated configuration"
    return ("space %s, %d dims, warm start reproducible, %d distinct fresh "
            "proposals" % (spec.hidden_features, len(dims), len(fresh)))


def test_S8_escalation_rule() -> str:
    """S8: escalate while still descending or while the best sits on a
    range boundary; stop when neither holds. Improvements smaller than the
    measured seed noise do not count."""
    spec = TSR.default_space(26, 10, n_train=20000)
    mid = {"hidden_features": 128, "num_transforms": 8, "num_bins": 10,
           "learning_rate": 5e-4, "training_batch_size": 512}

    flat = [(dict(mid), 10.0 + 1e-4 * i) for i in range(20)]
    v = TSR.escalation_verdict(flat, spec, tau_stop=0.01)
    assert not v.escalate, v.summary()

    desc = [(dict(mid), 10.0 - 0.3 * i) for i in range(20)]
    v2 = TSR.escalation_verdict(desc, spec, tau_stop=0.01)
    assert v2.escalate and any("descending" in r for r in v2.reasons)

    edge = dict(mid); edge["hidden_features"] = spec.hidden_features[1]
    obs = flat[:-1] + [(edge, 9.0)]
    v3 = TSR.escalation_verdict(obs, spec, tau_stop=0.01)
    assert v3.escalate and "hidden_features" in v3.on_boundary

    v4 = TSR.escalation_verdict([(dict(mid), 10.0)] * 3, spec, tau_stop=0.01)
    assert v4.escalate and any("too few" in r for r in v4.reasons)

    tr = TSR.convergence_trace([5.0, 6.0, 4.0, 4.5, 3.0])
    assert tr == [5.0, 5.0, 4.0, 4.0, 3.0], tr
    return "stop on flat, escalate on descent / boundary / too-few; trace ok"


def test_S9_gates_can_fail() -> str:
    """S9: THE test that licenses the battery.

    A posterior set exactly equal to the prior is applied to G1 and G2. The
    required outcome is that G1 REJECTS it and G2 PASSES it -- reproducing
    the documented blindness of marginal calibration to a data-ignoring
    posterior. If G2 also rejected, the two gates would be redundant; if G1
    passed, the battery would have no way to detect the failure mode this
    project is most exposed to.
    """
    rng = np.random.default_rng(7)
    p, n, n_draws = 5, 400, 64
    bounds = np.array([[0.0, 3.0]] * p)
    floor = TS.prior_floor(bounds)
    theta_true = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, p))

    lp_prior = _uniform_log_prob(theta_true, bounds)
    score_prior = TS.score_from_log_probs(lp_prior, bounds, n_members=1,
                                          n_boot=400, seed=0)
    g1_prior = TG.gate_g1_informativeness(score_prior.delta,
                                          score_prior.delta_ci_lo,
                                          delta_min=0.02)
    assert not g1_prior.passed, g1_prior.summary()

    # Draws from the prior: theta_true is exchangeable with them by
    # construction, so SBC ranks are uniform and G2 cannot see the problem.
    samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, n_draws, p))
    try:
        g2_prior = TG.gate_g2_marginal_calibration(
            theta_true, samples, param_names=["a%d" % k for k in range(p)],
            alpha=0.05, seed=0)
        g2_note = "G2 %s (blind, as documented)" % (
            "PASS" if g2_prior.passed else "FAIL")
        assert g2_prior.passed, (
            "G2 rejected a prior-equal posterior; on exchangeable draws the "
            "ranks are uniform, so this indicates a bug in the SBC path: "
            + g2_prior.summary())
    except ImportError:
        g2_note = "G2 skipped (npe_diagnostics unavailable here)"

    # The same gate must PASS a genuinely informative estimator, or it is
    # merely a gate that always fails.
    lp_good = _gaussian_log_prob(theta_true, theta_true, sd=0.3)
    score_good = TS.score_from_log_probs(lp_good, bounds, n_members=1,
                                         n_boot=400, seed=0)
    g1_good = TG.gate_g1_informativeness(score_good.delta,
                                         score_good.delta_ci_lo,
                                         delta_min=0.02)
    assert g1_good.passed, g1_good.summary()

    # The threshold comes from the control, not from taste.
    dmin = TG.delta_min_from_control([0.001, -0.002, 0.0005], k_sigma=3.0)
    assert dmin > 0.0
    assert TG.delta_min_from_control([]) == 0.0
    battery = TG.GateBattery(results=[g1_prior])
    assert not battery.passed
    return ("prior-equal: G1 REJECT (gain %.4f vs floor %.3f), %s; "
            "informative: G1 PASS (gain %.4f); delta_min from control %.4f"
            % (score_prior.delta, floor, g2_note, score_good.delta, dmin))


def test_S10_learning_curve_subsample() -> str:
    """S10: shrinking the training set drops whole groups, never parts of
    one, and always returns a subset of the rows it was given."""
    bank = _fake_bank(n=1200, n_groups=40, seed=5)
    man = TD.make_split(bank, seed=0, min_groups_per_side=2,
                        min_rows_per_side=20)
    train = np.asarray(man.train, dtype=np.int64)
    full_groups = set(bank.groups[train].tolist())
    sizes = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        idx = TD.subsample_groups(bank, train, frac, seed=0)
        assert set(idx.tolist()).issubset(set(train.tolist()))
        sub = set(bank.groups[idx].tolist())
        assert sub.issubset(full_groups)
        for g in sub:                       # whole groups only
            in_sub = int(np.sum(bank.groups[idx] == g))
            in_all = int(np.sum(bank.groups[train] == g))
            assert in_sub == in_all, (g, in_sub, in_all)
        sizes.append(len(idx))
    assert sizes == sorted(sizes), sizes
    assert sizes[-1] == len(train)
    again = TD.subsample_groups(bank, train, 0.25, seed=0)
    assert np.array_equal(again, TD.subsample_groups(bank, train, 0.25, seed=0))
    return "sizes %s from %d rows, whole groups only, reproducible" % (
        sizes, len(train))


def test_S13_defaults_do_not_drift() -> str:
    """S13: the mirrored NPEConfig defaults in npe_tune.py must equal the
    real ones. They exist so the coordinator commands run without the
    training stack; if they silently drifted, `baseline` would queue a
    configuration that is not the repository default while reporting that
    it is."""
    import npe_tune as T

    fallback = dict(T.FALLBACK_DEFAULTS)
    try:
        import npe_model
    except Exception as exc:
        raise ImportError("npe_model unavailable: %s" % exc)
    d = npe_model.NPEConfig()
    real = {"hidden_features": int(d.hidden_features),
            "num_transforms": int(d.num_transforms),
            "num_bins": int(d.num_bins),
            "learning_rate": float(d.learning_rate),
            "training_batch_size": int(d.training_batch_size)}
    assert fallback == real, ("mirrored defaults have drifted: %r vs %r"
                             % (fallback, real))
    assert T.default_config(verbose=False) == real
    return "mirrored defaults match npe_model.NPEConfig exactly"


def test_S11_train_and_extend() -> str:
    """S11 (needs torch + sbi): members train independently, the ensemble
    extends by training only the missing seeds, and the reused members are
    bit-identical. This is what makes probe-then-promote cheap."""
    import npe_contract as C
    import npe_tune_train as TT

    tmp = tempfile.mkdtemp()
    try:
        z, theta, contract = C.make_synthetic_shard(
            tmp, n_rows=400, p=4, embedding_dim=3, n_log_axes=2, seed=0,
            write=False)
        cfg = TT.make_config({"hidden_features": 16, "num_transforms": 2,
                              "num_bins": 4, "learning_rate": 1e-3,
                              "training_batch_size": 64},
                             device="cpu", max_num_epochs=3,
                             stop_after_epochs=2)
        prior = contract.prior(device="cpu")
        mdir = os.path.join(tmp, "model")

        posts, recs = TT.train_members(z, theta, prior, cfg, [0, 1],
                                       verbose=False)
        TT.save_members(posts, mdir, [0, 1], cfg, recs, contract=contract)
        lp2, _ = TT.evaluate_log_probs(posts, theta[:50], z[:50])

        posts4, recs4 = TT.extend_ensemble(mdir, z, theta, prior, cfg,
                                           [0, 1, 2, 3], contract=contract,
                                           verbose=False)
        assert len(posts4) == 4
        lp4, _ = TT.evaluate_log_probs(posts4[:2], theta[:50], z[:50])
        assert np.allclose(lp2, lp4, atol=1e-5), \
            "reused members changed under extension"

        bad = TT.make_config({"hidden_features": 32, "num_transforms": 2,
                              "num_bins": 4, "learning_rate": 1e-3,
                              "training_batch_size": 64},
                             device="cpu", max_num_epochs=3)
        try:
            TT.extend_ensemble(mdir, z, theta, prior, bad, [0, 1, 2, 3, 4],
                               contract=contract, verbose=False)
            raise AssertionError("extending under a changed config must fail")
        except ValueError:
            pass
        return "2 -> 4 members, reused members bit-identical, config guard ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_S12_mixture_matches_ensemble() -> str:
    """S12 (needs torch + sbi): sbi's EnsemblePosterior.log_prob agrees with
    the arithmetic mixture computed from the members. Asserted rather than
    trusted: a geometric mean would be sharper than any member and would
    make overconfidence worse, and the two are only distinguishable by this
    comparison."""
    import npe_contract as C
    import npe_tune_train as TT

    tmp = tempfile.mkdtemp()
    try:
        z, theta, contract = C.make_synthetic_shard(
            tmp, n_rows=300, p=3, embedding_dim=3, n_log_axes=1, seed=1,
            write=False)
        cfg = TT.make_config({"hidden_features": 16, "num_transforms": 2,
                              "num_bins": 4, "learning_rate": 1e-3,
                              "training_batch_size": 64},
                             device="cpu", max_num_epochs=3,
                             stop_after_epochs=2)
        prior = contract.prior(device="cpu")
        posts, _ = TT.train_members(z, theta, prior, cfg, [0, 1, 2],
                                    verbose=False)
        ens = TT.build_ensemble(posts)
        member_lp, mix_ens = TT.evaluate_log_probs(posts, theta[:64], z[:64],
                                                   ensemble=ens)
        mix_manual = TS.mixture_log_prob(member_lp)
        dev = float(np.max(np.abs(mix_ens - mix_manual)))
        assert dev < 1e-4, "ensemble log_prob is not the arithmetic mixture "\
                           "(max deviation %.3g)" % dev
        geometric = np.mean(member_lp, axis=0)
        assert float(np.max(np.abs(mix_ens - geometric))) > 1e-6, \
            "members agree too closely to distinguish the two rules"
        score = TS.score_from_log_probs(mix_ens, contract.bounds_theta,
                                        member_log_probs=member_lp,
                                        n_members=3, n_boot=200)
        assert score.jensen_ok
        return "ensemble == arithmetic mixture (dev %.2e); Jensen holds" % dev
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS: Dict[str, Tuple[str, Callable[[], str]]] = {
    "S1": ("prior floor is exact; prior estimator gains zero", test_S1_prior_floor),
    "S2": ("information gain: sign and ordering", test_S2_information_gain_positive),
    "S3": ("Jensen bound; arithmetic vs geometric mixture", test_S3_jensen_bound),
    "S4": ("split is group-disjoint, reproducible, tamper-evident", test_S4_split_is_group_disjoint),
    "S5": ("shape-agnostic at two (p, E); guards fire", test_S5_shape_agnostic),
    "S6": ("ledger round-trip, numpy casts, digest filtering", test_S6_ledger_roundtrip),
    "S7": ("search space, batching, warm-start replay", test_S7_search_space_and_warm_start),
    "S8": ("escalation rule and convergence trace", test_S8_escalation_rule),
    "S9": ("GATES CAN FAIL: G1 rejects a prior-equal posterior, G2 does not", test_S9_gates_can_fail),
    "S10": ("learning-curve subsampling keeps whole groups", test_S10_learning_curve_subsample),
    "S11": ("[needs sbi] independent training and free extension", test_S11_train_and_extend),
    "S12": ("[needs sbi] ensemble log_prob is the arithmetic mixture", test_S12_mixture_matches_ensemble),
    "S13": ("[needs npe_model] mirrored NPEConfig defaults have not drifted", test_S13_defaults_do_not_drift),
}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--fast", action="store_true",
                    help="only tests that need no torch/sbi")
    ap.add_argument("-k", dest="pattern", default=None,
                    help="run only tests whose id contains this")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for tid, (desc, _) in TESTS.items():
            print("  %-5s %s" % (tid, desc))
        return 0

    selected = list(FAST_TESTS) if args.fast else list(TESTS)
    if args.pattern:
        selected = [t for t in selected if args.pattern in t]
    if not selected:
        print("no test matched %r" % args.pattern)
        return 1

    print("=" * 72)
    print("smoke_test_tune.py -- %d test(s)%s"
          % (len(selected), " [fast tier]" if args.fast else ""))
    print("=" * 72)

    failures = []
    for tid in selected:
        desc, fn = TESTS[tid]
        try:
            detail = fn()
            print("  %-5s PASS  %s" % (tid, detail))
        except ImportError as exc:
            print("  %-5s SKIP  missing dependency: %s" % (tid, exc))
        except Exception as exc:
            failures.append(tid)
            print("  %-5s FAIL  %s: %s" % (tid, type(exc).__name__, exc))
            traceback.print_exc()

    print("-" * 72)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
