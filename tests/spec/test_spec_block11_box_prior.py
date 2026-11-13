"""Spec-derived tests of block 11 (the box prior for arm S; SPEC.md sec. 2.4
arm S, block 11). Oracles O11.1-O11.7 of the t1-B report.

Expected values come from the spec (the box prior U((0,1)^p), the clip at the
float64 machine epsilon, cls -1, the latent_spec rule), from SciPy (KS tests)
and from the code before block 11 (`git archive 59465e1^`), which is the
spec's own reference for "the default path is unchanged draw for draw".
"""
import json
import os

import numpy as np
import pytest
from scipy import stats

import t1b_helpers as H

H.joint_paths()
import latent_sbi_simulator as LS      # noqa: E402  (code under test)

EPS = np.finfo(np.float64).eps


def _bench_like_spec(n_latent=10, seed=0):
    # the bench's structure (10 axes, label axes 0-6); centres are irrelevant to the box draw
    return LS.LatentSBISpec(n_latent=n_latent, label_idx=tuple(range(7)), tau_ov=0.1,
                            n_windows_per_trace=1, T_win=12.8, fs=10.0, n_neurons=100, seed=seed)


def test_o11_1_contract():
    spec = _bench_like_spec()
    c, phi = LS.sample_box_prior(spec, 1000, np.random.default_rng(1))
    assert LS.BOX_CLASS == -1
    assert c.shape == (1000,) and c.dtype == np.int64
    assert np.all(c == -1)
    assert phi.shape == (1000, 10) and phi.dtype == np.float64
    assert np.all(phi >= EPS) and np.all(phi <= 1.0 - EPS)


def test_o11_1_clip_at_eps():
    """A generator that returns the box's ends exactly: the spec clips to
    [eps, 1 - eps] with eps the float64 machine epsilon."""
    class Ends(object):
        def uniform(self, lo, hi, size=None):
            a = np.zeros(size, dtype=np.float64)
            a[..., 1::2] = 1.0
            return a
    c, phi = LS.sample_box_prior(_bench_like_spec(), 4, Ends())
    np.testing.assert_array_equal(phi[:, 0::2], EPS)
    np.testing.assert_array_equal(phi[:, 1::2], 1.0 - EPS)


def test_o11_2_uniform_400_donors():
    """The spec's own oracle: KS against U(0, 1) on each axis over 400 donors,
    p > 1e-3 (6 axes in the spec; all 10 here)."""
    c, phi = LS.sample_box_prior(_bench_like_spec(), 400, np.random.default_rng(400))
    for k in range(10):
        p = stats.kstest(phi[:, k], "uniform").pvalue
        assert p > 1e-3, (k, p)


def test_o11_2_uniform_and_independent_20000():
    """A stronger reference: 20,000 draws. KS p > 1e-3 per axis (fixed seed;
    family-wise false alarm 1e-2 under the null); each axis mean within 5
    standard errors of 1/2 (SE = sqrt(1/12/n) = 0.002); every pairwise
    correlation within 5/sqrt(n) = 0.035 of 0 (SE 1/sqrt(n))."""
    n = 20000
    c, phi = LS.sample_box_prior(_bench_like_spec(), n, np.random.default_rng(2026))
    for k in range(10):
        assert stats.kstest(phi[:, k], "uniform").pvalue > 1e-3, k
        assert abs(phi[:, k].mean() - 0.5) < 5 * np.sqrt(1.0 / 12.0 / n), k
    r = np.corrcoef(phi.T)
    off = r[~np.eye(10, dtype=bool)]
    assert np.max(np.abs(off)) < 5.0 / np.sqrt(n)


def test_o11_3_seeded_and_uses_rng():
    spec = _bench_like_spec()
    a = LS.sample_box_prior(spec, 50, np.random.default_rng(7))[1]
    b = LS.sample_box_prior(spec, 50, np.random.default_rng(7))[1]
    d = LS.sample_box_prior(spec, 50, np.random.default_rng(8))[1]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, d)
    g = np.random.default_rng(7)
    first = LS.sample_box_prior(spec, 50, g)[1]
    second = LS.sample_box_prior(spec, 50, g)[1]
    assert not np.array_equal(first, second)        # the generator is consumed


BOX_ARGS = ["--arm", "S", "--prior", "box", "--provider", "bench", "--fs", "10.0",
            "--T-win", "12.8", "--n-traces", "16", "--wells-per-donor", "2",
            "--n-windows", "2", "--seed", "5"]


@pytest.fixture(scope="module")
def box_shard():
    d = H.tmpdir("t1b_b11_")
    rc, out = H.run_builder(d, BOX_ARGS)
    assert rc == 0, out
    yield H.load_shard(os.path.join(d, "shard_0000.npz"))
    H.rmtree(d)


def test_o11_4_box_bank_rows(box_shard):
    arrays, side = box_shard
    assert arrays["x"].shape == (16 * 2, 128)               # n_traces x n_windows, W = 12.8 x 10
    assert np.all(arrays["cls"] == -1)
    th = arrays["theta"].astype(np.float64)
    assert np.all((th > 0.0) & (th < 1.0))
    # every donor: exactly two wells, one theta shared by all its rows
    for d in np.unique(arrays["donor"]):
        rows = arrays["donor"] == d
        assert np.unique(arrays["well"][rows]).size == 2, d
        np.testing.assert_array_equal(th[rows], np.broadcast_to(th[rows][0], th[rows].shape))
    # different donors, different theta (a continuous draw)
    firsts = np.array([th[arrays["donor"] == d][0] for d in np.unique(arrays["donor"])])
    assert np.unique(firsts, axis=0).shape[0] == firsts.shape[0]
    assert side["arm"] == "S" and side["theta_withheld"] is False


def test_o11_4_box_latent_spec(box_shard):
    _, side = box_shard
    latent = side["latent_spec"]
    assert latent.get("kind") == "box"
    rest = dict(latent)
    rest.pop("kind")
    # the spec's rule: LatentSBISpec.to_dict() plus "kind": "box" -- the same
    # spec built from the command's own values
    from bench_burst_provider import BENCH_LABEL_IDX
    import latent_sbi_simulator as LS2
    ref = LS2.LatentSBISpec(n_latent=10, label_idx=BENCH_LABEL_IDX,
                            class_centres=LS2.simplex_centres(3, len(BENCH_LABEL_IDX)),
                            tau_ov=0.1, n_windows_per_trace=2, T_win=12.8, fs=10.0,
                            n_neurons=100, seed=5).to_dict()
    assert set(rest) == set(ref), (sorted(rest), sorted(ref))
    for k in ref:
        assert rest[k] == ref[k], k                       # JSON round trip of repr floats: exact


def test_o11_4_box_refused_on_arm_R():
    d = H.tmpdir("t1b_b11r_")
    try:
        args = list(BOX_ARGS)
        args[args.index("S")] = "R"
        rc, out = H.run_builder(d, args)
        assert rc != 0
        assert "box" in out
        assert not os.listdir(d)                                # nothing written
    finally:
        H.rmtree(d)


DEFAULT_BENCH = ["--arm", "S", "--provider", "bench", "--n-traces", "6", "--n-windows", "1",
                 "--T-win", "12.8", "--fs", "10.0", "--seed", "4", "--shard-index", "0"]


def _build_with(tree, args):
    d = H.tmpdir("t1b_old_")
    rc, out = H.run_builder(d, args, tree=tree)
    assert rc == 0, out
    res = H.load_shard(os.path.join(d, "shard_%04d.npz" % int(args[args.index("--shard-index") + 1])))
    H.rmtree(d)
    return res


@pytest.mark.parametrize("args", [DEFAULT_BENCH], ids=["bench"])
def test_o11_5_default_path_unchanged_shard0(args):
    """The default path (`--prior mixture`) at shard 0: every array byte for
    byte as the code before block 11 built it, and the latent_spec's keys
    unchanged (the spec: unchanged draw for draw; latent_spec exactly
    to_dict()'s keys)."""
    old_a, old_s = _build_with(H.old_tree(H.PRE_BLOCK11), args)
    new_a, new_s = _build_with(None, args)
    assert sorted(old_a) == sorted(new_a)
    for k in old_a:
        assert old_a[k].dtype == new_a[k].dtype, k
        np.testing.assert_array_equal(old_a[k], new_a[k], err_msg=k)
    assert sorted(old_s["latent_spec"]) == sorted(new_s["latent_spec"])
    assert "kind" not in new_s["latent_spec"]


def test_o11_5_default_digest_unchanged_by_blocks_11_14():
    """Block 11 (and block 12): the default bank's contract digest is unchanged.
    Compared on the code of blocks 11-14 (44f812b^, before blocks 25-26) with
    the code before them (59465e1^)."""
    old_a, old_s = _build_with(H.old_tree(H.PRE_BLOCK11), DEFAULT_BENCH)
    mid_a, mid_s = _build_with(H.old_tree(H.PRE_BLOCK25), DEFAULT_BENCH)
    assert old_s["contract_digest"] == mid_s["contract_digest"]
    assert old_s["generator_sha256"] == mid_s["generator_sha256"]


def test_o11_5_default_digest_unchanged_at_commit():
    """The same claim at the commit under test (blocks 11-13's entries: a
    default bank's digest unchanged; block 13: provider_sha256 unchanged,
    as is every shard the code could build before)."""
    old_a, old_s = _build_with(H.old_tree(H.PRE_BLOCK11), DEFAULT_BENCH)
    new_a, new_s = _build_with(None, DEFAULT_BENCH)
    assert old_s["generator_sha256"] == new_s["generator_sha256"], (
        old_s["generator_sha256"], new_s["generator_sha256"])
    assert old_s["contract_digest"] == new_s["contract_digest"]


def test_o11_6_unit_box_kind():
    torch = pytest.importorskip("torch")       # noqa: F841  (the runner imports torch)
    import run_joint_arms as RJA
    assert RJA.unit_box_kind({"latent_spec": {"kind": "campaign"}}) == "campaign"
    assert RJA.unit_box_kind({"latent_spec": {"kind": "box"}}) == "box"
    assert RJA.unit_box_kind({"latent_spec": {"n_latent": 10}}) is None
    assert RJA.unit_box_kind({"latent_spec": {"kind": "mixture"}}) is None
    assert RJA.unit_box_kind({}) is None
    assert RJA.unit_box_kind({"latent_spec": None}) is None
    assert tuple(RJA.UNIT_BOX_KINDS) == ("campaign", "box")


def test_o11_7_campaign_log_prob():
    pytest.importorskip("torch")
    import run_joint_arms as RJA
    t = np.array([[0.5, 0.5], [1e-12, 1 - 1e-12], [0.0, 0.5], [0.5, 1.0], [-0.1, 0.5], [0.5, 1.1]])
    lp = RJA.campaign_log_prob(t)
    assert lp.shape == (6,) and lp.dtype == np.float64
    np.testing.assert_array_equal(lp, [0.0, 0.0, -np.inf, -np.inf, -np.inf, -np.inf])
