"""Spec-derived tests of block 12 (arm R's cohort and bank-wide ids; SPEC.md
sec. 2.4 arm R, block 12). Oracles O12.1-O12.6 of the t1-B report.

References: the DSN generator's `_class_center_vectors` (the spec names it as
the source of m_c), scipy.stats.truncnorm for TN(mu, tau^2; (0, 1)), the
numbers of sec. 2.4, and the code before blocks 11-14 (`git archive
59465e1^`) for "unchanged byte for byte".
"""
import importlib.util
import itertools
import os

import numpy as np
import pytest
from scipy import stats

import t1b_helpers as H

H.joint_paths()
import latent_sbi_simulator as LS      # noqa: E402  (code under test)


def _dsn_lbg():
    """The DSN generator module, loaded from hpc/dsn by path (not through the
    code under test's loader)."""
    path = os.path.join(H.DSN, "latent_burst_generator.py")
    H.add_paths(H.DSN)
    spec = importlib.util.spec_from_file_location("t1b_lbg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LBG = _dsn_lbg()


@pytest.mark.parametrize("C,L,s", [(3, 7, 0.6), (3, 8, 0.562743), (4, 5, 0.5), (2, 3, 1.0)])
def test_o12_1_regular_centres(C, L, s):
    m = np.asarray(LBG._class_center_vectors(C, L, "simplex"), dtype=np.float64)
    want = 0.5 + s * (m - 0.5)
    got = LS.regular_centres(LBG, C, L, s)
    assert got.shape == (C, L) and got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    dist = [np.linalg.norm(got[a] - got[b]) for a, b in itertools.combinations(range(C), 2)]
    np.testing.assert_allclose(dist, dist[0], rtol=0, atol=1e-12)     # equidistant


def test_o12_2_simplex_geometry_of_sec_2_4():
    """sec. 2.4: ||m_c - m_c'|| = 0.7366, centred cosine -1/2, ||mu_c - mu_c'||
    = 0.442 at s = 0.6 (the spec's 4 and 3 decimals)."""
    m = np.asarray(LBG._class_center_vectors(3, 7, "simplex"), dtype=np.float64)
    for a, b in itertools.combinations(range(3), 2):
        assert round(float(np.linalg.norm(m[a] - m[b])), 4) == 0.7366
    mc = m - m.mean(axis=0)
    for a, b in itertools.combinations(range(3), 2):
        cos = mc[a] @ mc[b] / (np.linalg.norm(mc[a]) * np.linalg.norm(mc[b]))
        assert abs(cos + 0.5) < 1e-12
    mu = LS.regular_centres(LBG, 3, 7, 0.6)
    for a, b in itertools.combinations(range(3), 2):
        assert round(float(np.linalg.norm(mu[a] - mu[b])), 3) == 0.442


def _cohort_spec(tau=0.1):
    return LS.LatentSBISpec(n_latent=10, label_idx=tuple(range(7)),
                            class_centres=LS.regular_centres(LBG, 3, 7, 0.6), tau_ov=tau,
                            n_windows_per_trace=1, T_win=12.8, fs=10.0, n_neurons=100, seed=0)


def test_o12_3_sample_prior_given_classes_law():
    """theta_k | c ~ TN(mu_{c,k}, tau^2; (0, 1)) on the label axes, U(0, 1) on
    the free axes. 3,000 rows per class, fixed seed: KS p > 1e-3 against
    scipy's truncnorm (21 label tests and 3 free tests per class: family-wise
    false alarm about 0.07 under the null, but the seed is fixed), and each
    mean within 5 standard errors of the law's mean."""
    spec = _cohort_spec()
    n = 3000
    c = np.repeat(np.arange(3), n)
    phi = LS.sample_prior_given_classes(spec, c, np.random.default_rng(12))
    assert phi.shape == (3 * n, 10) and phi.dtype == np.float64
    eps = np.finfo(np.float64).eps
    assert np.all(phi >= eps) and np.all(phi <= 1 - eps)
    mu = spec.class_centres
    for cls in range(3):
        rows = phi[c == cls]
        for j, ax in enumerate(range(7)):
            law = stats.truncnorm((0 - mu[cls, j]) / 0.1, (1 - mu[cls, j]) / 0.1,
                                  loc=mu[cls, j], scale=0.1)
            assert stats.kstest(rows[:, ax], law.cdf).pvalue > 1e-3, (cls, ax)
            assert abs(rows[:, ax].mean() - law.mean()) < 5 * law.std() / np.sqrt(n), (cls, ax)
        for ax in (7, 8, 9):
            assert stats.kstest(rows[:, ax], "uniform").pvalue > 1e-3, (cls, ax)


def test_o12_3_classes_out_of_range_refused():
    spec = _cohort_spec()
    with pytest.raises(ValueError):
        LS.sample_prior_given_classes(spec, np.array([0, 3]), np.random.default_rng(0))
    with pytest.raises(ValueError):
        LS.sample_prior_given_classes(spec, np.array([-1, 0]), np.random.default_rng(0))


COHORT = ["--arm", "R", "--provider", "reference", "--n-latent", "10", "--n-label-axes", "7",
          "--class-centres", "regular", "--centre-scale", "0.6", "--donors-per-class", "150",
          "--n-traces", "900", "--wells-per-donor", "2", "--n-windows", "1", "--T-win", "6.0",
          "--fs", "10.0", "--seed", "9"]


@pytest.fixture(scope="module")
def cohort_shard():
    d = H.tmpdir("t1b_b12_")
    rc, out = H.run_builder(d, COHORT)
    assert rc == 0, out
    yield H.load_shard(os.path.join(d, "shard_0000.npz"))
    H.rmtree(d)


def test_o12_4_cohort_design(cohort_shard):
    a, side = cohort_shard
    donors = np.unique(a["donor"])
    assert donors.size == 450
    cls_of = {}
    for d in donors:
        cl = np.unique(a["cls"][a["donor"] == d])
        assert cl.size == 1, d                                   # one condition per donor
        cls_of[int(d)] = int(cl[0])
    counts = np.bincount(list(cls_of.values()), minlength=3)
    np.testing.assert_array_equal(counts, [150, 150, 150])
    # theta given the class: the per-class mean on each label axis within 5 SE
    mu = LS.regular_centres(LBG, 3, 7, 0.6)
    th = np.array([a["theta"][a["donor"] == d][0] for d in donors], dtype=np.float64)
    cl = np.array([cls_of[int(d)] for d in donors])
    for c in range(3):
        for k in range(7):
            law = stats.truncnorm((0 - mu[c, k]) / 0.1, (1 - mu[c, k]) / 0.1, loc=mu[c, k], scale=0.1)
            assert abs(th[cl == c, k].mean() - law.mean()) < 5 * law.std() / np.sqrt(150), (c, k)
    # batches of 8 consecutive donor indices mix the classes (all but at most one in ten)
    order = [cls_of[int(d)] for d in sorted(donors)]
    full = [order[i:i + 8] for i in range(0, 448, 8)]
    unmixed = sum(1 for b in full if len(set(b)) == 1)
    assert unmixed <= len(full) // 10, unmixed
    latent = side["latent_spec"]
    assert latent["class_centres_mode"] == "regular"
    assert latent["centre_scale"] == 0.6
    assert latent["donors_per_class"] == 150
    np.testing.assert_allclose(np.asarray(latent["class_centres"]), mu, rtol=0, atol=1e-12)
    assert side["arm"] == "R" and side["theta_withheld"] is True


@pytest.mark.parametrize("args,why", [
    (["--arm", "R", "--provider", "reference", "--donors-per-class", "2", "--n-traces", "10",
      "--wells-per-donor", "2"], "n-traces != C N wells"),
    (["--arm", "S", "--provider", "reference", "--prior", "box", "--donors-per-class", "2",
      "--n-traces", "12", "--wells-per-donor", "2"], "with --prior box"),
    (["--arm", "R", "--provider", "reference", "--donors-per-class", "2", "--n-traces", "12",
      "--wells-per-donor", "2", "--max-records", "6"], "with --max-records"),
    (["--arm", "R", "--provider", "dsn", "--class-centres", "regular", "--n-traces", "4"],
     "dsn provider and regular centres"),
    (["--arm", "R", "--provider", "dsn", "--centre-scale", "0.6", "--n-traces", "4"],
     "dsn provider and a centre scale"),
    (["--arm", "R", "--provider", "reference", "--centre-scale", "0.6", "--n-traces", "4"],
     "a centre scale without regular centres"),
])
def test_o12_4_refusals(args, why):
    d = H.tmpdir("t1b_b12r_")
    try:
        rc, out = H.run_builder(d, args + ["--n-windows", "1", "--T-win", "6.0", "--fs", "10.0"])
        assert rc != 0, (why, out)
        assert not os.listdir(d), why
    finally:
        H.rmtree(d)


IDS = ["--arm", "S", "--provider", "reference", "--n-traces", "8", "--wells-per-donor", "2",
       "--donors-per-batch", "2", "--n-windows", "2", "--T-win", "6.0", "--fs", "10.0", "--seed", "3"]


def _two_shards(tree):
    d = H.tmpdir("t1b_ids_")
    out = []
    for k in (0, 1):
        rc, txt = H.run_builder(d, IDS + ["--shard-index", str(k)], tree=tree)
        assert rc == 0, txt
        out.append(os.path.join(d, "shard_%04d.npz" % k))
    return d, out


def test_o12_5_bank_wide_ids():
    stride = 1000000
    d, paths = _two_shards(None)
    try:
        s = [H.load_shard(p)[0] for p in paths]
        for k in (0, 1):
            for key in ("donor", "well", "batch"):
                v = s[k][key]
                assert np.all((v >= k * stride) & (v < (k + 1) * stride)), (k, key)
        for key in ("donor", "well", "batch"):
            assert not set(s[0][key].tolist()) & set(s[1][key].tolist()), key
            np.testing.assert_array_equal(s[1][key] - stride, s[0][key])   # the local layouts equal
        import latent_bank as LB
        merged, _ = LB.concat_shards(paths)
        assert np.unique(merged["donor"]).size == 8                # 4 donors per shard, none merged
        assert np.unique(merged["well"]).size == 16
    finally:
        H.rmtree(d)


def test_o12_5_unchanged_byte_for_byte_but_ids():
    """Against the code before blocks 11-14: shard 0 equal in every array;
    shard 1 equal in x, theta, cls, nu, realisation_id (and the rest), its
    donor, well and batch the old ones + 10^6; one contract digest."""
    d_old, old = _two_shards(H.old_tree(H.PRE_BLOCK11))
    d_new, new = _two_shards(None)
    try:
        for k in (0, 1):
            a_old, s_old = H.load_shard(old[k])
            a_new, s_new = H.load_shard(new[k])
            assert sorted(a_old) == sorted(a_new)
            for key in a_old:
                assert a_old[key].dtype == a_new[key].dtype, key
                if k == 1 and key in ("donor", "well", "batch"):
                    np.testing.assert_array_equal(a_new[key], a_old[key] + 1000000, err_msg=key)
                else:
                    np.testing.assert_array_equal(a_new[key], a_old[key], err_msg=(k, key))
            assert s_old["contract_digest"] == s_new["contract_digest"], k
    finally:
        H.rmtree(d_old)
        H.rmtree(d_new)


def test_o12_6_design_keys_only_off_default():
    d = H.tmpdir("t1b_keys_")
    try:
        base = ["--arm", "R", "--provider", "reference", "--n-traces", "6", "--wells-per-donor", "2",
                "--n-windows", "1", "--T-win", "6.0", "--fs", "10.0", "--seed", "1"]
        rc, out = H.run_builder(os.path.join(d, "a"), base)
        assert rc == 0, out
        lat = H.load_shard(os.path.join(d, "a", "shard_0000.npz"))[1]["latent_spec"]
        for key in ("class_centres_mode", "centre_scale", "donors_per_class", "kind"):
            assert key not in lat, key
        rc, out = H.run_builder(os.path.join(d, "b"), base + ["--class-centres", "regular"])
        assert rc == 0, out
        lat = H.load_shard(os.path.join(d, "b", "shard_0000.npz"))[1]["latent_spec"]
        assert lat["class_centres_mode"] == "regular" and lat["centre_scale"] == 1.0
        assert "donors_per_class" not in lat
        rc, out = H.run_builder(os.path.join(d, "c"), base[:5] + ["6", "--wells-per-donor", "2",
                                                                    "--donors-per-class", "1"]
                                + base[8:])
        assert rc == 0, out
        lat = H.load_shard(os.path.join(d, "c", "shard_0000.npz"))[1]["latent_spec"]
        assert lat["donors_per_class"] == 1 and "class_centres_mode" not in lat
    finally:
        H.rmtree(d)


def test_o12_3_seeded_and_uses_rng():
    spec = _cohort_spec()
    c = np.repeat(np.arange(3), 20)
    a = LS.sample_prior_given_classes(spec, c, np.random.default_rng(5))
    b = LS.sample_prior_given_classes(spec, c, np.random.default_rng(5))
    e = LS.sample_prior_given_classes(spec, c, np.random.default_rng(6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, e)
