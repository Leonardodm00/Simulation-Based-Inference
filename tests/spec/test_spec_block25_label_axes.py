"""Block 25 -- the bench's label axes from the configuration (SPEC.md sec. 5,
block 25; D-132). Spec-derived tests of the integration tester, run t1-D of
2026-11-13. Oracles O25.1-O25.13 of the report.

The expected values are computed here: the affine map of sec. 2.4 from the
`BENCH_AXES` ranges, the centres mu_c = 0.5 + s (m_c - 0.5) of sec. 2.4 /
block 12 from the DSN generator's simplex (unchanged by block 25), the
truncated normal's moments from scipy.stats.truncnorm, the uniform law by
scipy.stats.kstest, and -- where the spec says "unchanged" -- the functions
as they stood at 0e62645 (before blocks 25-26).

    python3 -m pytest tests/spec/test_spec_block25_label_axes.py -v
"""

import argparse
import ast
import contextlib
import io
import itertools
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers as H

H.add_paths(H.PIPE, H.PTESTS, H.STAGE1, H.EVAL)

import bench_burst_provider as bbp      # noqa: E402
import config as C                      # noqa: E402
import latent_sbi_simulator as lss      # noqa: E402

FIELDS = ("lambda_b", "sigma_d", "ibi_cv", "n_frag_mean", "lambda_burst", "p_mean", "d_med",
          "duty", "kappa", "lambda_bg")
L8 = list(range(8))
D_KEEP = 0.441934           # sec. 5 block 25: the centres' distance kept by the scale 0.562743


# ------------------------------------------------------------------ oracles ---

def affine(phi, delta=0.0, free=(7, 8, 9)):
    """Sec. 2.4: a_k = L_k (1 - theta_k) + U_k theta_k, the range displaced by
    delta (U_k - L_k) on the free axes, then a_7 = min(a_7, 1)."""
    phi = np.asarray(phi, dtype=np.float64)
    out = np.empty(10)
    for k, (_n, lo, hi) in enumerate(bbp.BENCH_AXES):
        w = hi - lo
        if delta and k in free:
            lo, hi = lo + delta * w, hi + delta * w
        out[k] = lo * (1.0 - phi[k]) + hi * phi[k]
    out[7] = min(out[7], 1.0)
    return out


def params_vec(p):
    return np.array([getattr(p, f) for f in FIELDS], dtype=np.float64)


def centres(n_axes, scale):
    """mu_c = 0.5 + s (m_c - 0.5), m_c the DSN generator's regular simplex
    for 3 classes on `n_axes` axes (sec. 2.4, block 12)."""
    lbg, _gbd = lss.load_dsn_modules(None)
    m = np.asarray(lbg._class_center_vectors(3, n_axes, "simplex"), dtype=np.float64)
    return 0.5 + scale * (m - 0.5)


def dists(mu):
    return [float(np.linalg.norm(mu[i] - mu[j])) for i, j in itertools.combinations(range(len(mu)), 2)]


def build(out, argv, env=None):
    rc, text = H.run([H.PYTHON, H.BUILDER, "--out-dir", out] + list(argv), env or dict(os.environ),
                     timeout=900)
    return rc, text


def arg_set(argv, flag, value):
    argv = list(argv)
    argv[argv.index(flag) + 1] = value
    return argv


@pytest.fixture(scope="module")
def project():
    """A reduced project on the committed v4 (test_smoke_synthetic._small)."""
    import test_smoke_synthetic as TS
    root = tempfile.mkdtemp(prefix="t1d_b25_")
    path, d = TS._small(root)
    cfg = C.load_config(path, synthetic_ok=True)
    yield {"root": root, "path": path, "cfg": cfg, "d": d}
    shutil.rmtree(root, ignore_errors=True)


def argv_of(cfg, bank, k=0):
    from stages import synthetic as SY
    return SY.build_of(cfg, bank)["argv"][k]


# ------------------------------------------------------------------- checks ---

def test_o25_1_bench_free_idx_every_subset():
    """O25.1: for every subset L of the ten axes, bench_free_idx(L) is the
    complement in increasing order when 0 < |L| < 10, and a ValueError for
    the empty set and the full set; the defaults."""
    assert tuple(bbp.BENCH_LABEL_IDX) == (0, 1, 2, 3, 4, 5, 6)
    assert tuple(bbp.BENCH_FREE_IDX) == (7, 8, 9)
    n_ok = 0
    for r in range(0, 11):
        for sub in itertools.combinations(range(10), r):
            want = tuple(k for k in range(10) if k not in sub)
            if 0 < r < 10:
                assert tuple(bbp.bench_free_idx(list(sub))) == want, sub
                assert tuple(bbp.bench_free_idx(np.array(sub, dtype=np.int64))) == want, sub
                n_ok += 1
            else:
                with pytest.raises(ValueError):
                    bbp.bench_free_idx(list(sub))
    assert n_ok == 2 ** 10 - 2


@pytest.mark.parametrize("bad", [[1, 1], [2, 1], [0, 3, 2], [-1, 2], [0, 10], [0.5], [2.5, 3],
                                 ["3"], [True, 2], None, 5])
def test_o25_1_bench_free_idx_refusals(bad):
    """O25.1: duplicates, unsorted, out of range, non-integers -> ValueError."""
    with pytest.raises(ValueError):
        bbp.bench_free_idx(bad)


@pytest.mark.parametrize("free", [(8, 9), (7, 8, 9), (0, 8, 9), (9,), (0, 1, 2, 3, 4, 5, 6, 7, 8)])
@pytest.mark.parametrize("delta", [0.0, 0.105, 0.35])
def test_o25_2_affine_map_on_the_given_free_axes(free, delta):
    """O25.2: phi_to_bench_params displaces exactly the ranges of the axes in
    free_idx; the duty clipped at 1. Tolerance: two float64 evaluations of
    one affine map of values below 150 -- 1e-12 relative is 1e4 ulps."""
    rng = np.random.default_rng(25)
    phis = np.vstack([rng.uniform(0.0, 1.0, size=(300, 10)), np.zeros((1, 10)), np.ones((1, 10))])
    for phi in phis:
        got = params_vec(bbp.phi_to_bench_params(phi, 100, 12.8, free_axis_shift=delta,
                                                 free_idx=free))
        assert_allclose(got, affine(phi, delta, free), rtol=1e-12, atol=1e-15)


def test_o25_2_known_answers():
    """O25.2: the duty's known answers (block 13's numbers; sec. 2.4)."""
    assert bbp.BENCH_AXES[7][1:] == (0.35, 1.0)
    phi = np.full(10, 0.5)
    phi[7] = 0.95
    with_duty = bbp.phi_to_bench_params(phi, 100, 12.8, free_axis_shift=0.105, free_idx=(7, 8, 9))
    assert with_duty.duty == 1.0                             # 1.0357 clipped
    without = bbp.phi_to_bench_params(phi, 100, 12.8, free_axis_shift=0.105, free_idx=(8, 9))
    assert abs(without.duty - (0.35 * 0.05 + 1.0 * 0.95)) < 1e-15       # 0.9675, not shifted
    p9 = bbp.phi_to_bench_params(np.full(10, 0.9), 100, 12.8, free_axis_shift=0.105, free_idx=(8, 9))
    assert abs(p9.duty - 0.935) < 1e-15                      # the unshifted map's value


def test_o25_3_default_free_axes_unchanged_bit_for_bit():
    """O25.3: with the default free axes, or none passed, the map equals the
    function at 0e62645 bit for bit."""
    old = H.module_at(H.BEFORE_B25_26, "hpc/joint/stage1/bench_burst_provider.py", "bbp_0e62645",
                      [H.STAGE1])
    rng = np.random.default_rng(250)
    for phi in rng.uniform(0.0, 1.0, size=(500, 10)):
        for delta in (0.0, 0.105, 0.3):
            want = params_vec(old.phi_to_bench_params(phi, 100, 12.8, free_axis_shift=delta))
            assert_array_equal(params_vec(bbp.phi_to_bench_params(phi, 100, 12.8,
                                                                  free_axis_shift=delta)), want)
            assert_array_equal(params_vec(bbp.phi_to_bench_params(
                phi, 100, 12.8, free_axis_shift=delta, free_idx=bbp.BENCH_FREE_IDX)), want)


def test_o25_4_provider_holds_and_passes_its_free_axes():
    """O25.4: the provider built with L holds F = bench_free_idx(L) and passes
    it, with the gap's shift, on every call (a spy on the map)."""
    lbg, gbd = lss.load_dsn_modules(None)
    prov = bbp.BenchBurstProvider(gbd, label_idx=range(8))
    assert tuple(prov.free_idx) == (8, 9)
    assert tuple(bbp.load_bench_provider(None, label_idx=L8).free_idx) == (8, 9)
    assert tuple(bbp.load_bench_provider(None).free_idx) == (7, 8, 9)
    for bad in ([3, 2], [], list(range(10))):
        with pytest.raises(ValueError):
            bbp.BenchBurstProvider(gbd, label_idx=bad)
    calls = []
    real = bbp.phi_to_bench_params

    def spy(*a, **k):
        calls.append((tuple(k.get("free_idx", ())), k.get("free_axis_shift")))
        return real(*a, **k)
    bbp.phi_to_bench_params = spy
    try:
        phi = np.full(10, 0.4)
        prov(phi, 1, 128, 10.0, 20, 3, param_overrides={"free_axis_range_shift": 0.105})
        prov(phi, 1, 128, 10.0, 20, 4)
    finally:
        bbp.phi_to_bench_params = real
    assert calls == [((8, 9), 0.105), ((8, 9), 0.0)], calls


def test_o25_5_builder_sidecar_and_centres(project):
    """O25.5, O25.10: the builder with --label-axes 0..7 records label_idx,
    free_idx and centres (3, 8) equal to 0.5 + s (m - 0.5) (1e-12: float64
    roundoff of an affine map), three equal distances 0.441934 (1e-6)."""
    cfg = project["cfg"]
    out = os.path.join(project["root"], "r3")
    rc, text = build(out, argv_of(cfg, "R3"))
    assert rc == 0, text
    with open(os.path.join(out, "shard_0000.json")) as fh:
        lat = json.load(fh)["latent_spec"]
    assert lat["label_idx"] == L8 and lat["free_idx"] == [8, 9], lat
    s = cfg.data["synthetic"]["builder"]["centre_scale"]
    assert s == 0.562743
    mu = np.asarray(lat["class_centres"], dtype=np.float64)
    assert mu.shape == (3, 8)
    assert_allclose(mu, centres(8, s), rtol=0, atol=1e-12)
    d = dists(mu)
    assert all(abs(x - D_KEEP) < 1e-6 for x in d), d
    # O25.10: on 7 axes at 0.6 the same distance
    assert all(abs(x - D_KEEP) < 1e-6 for x in dists(centres(7, 0.6)))
    # without the flag: the bench's default split, centres (3, 7)
    argv = argv_of(cfg, "R3")
    i = argv.index("--label-axes")
    out7 = os.path.join(project["root"], "r3_default")
    rc, text = build(out7, argv[:i] + argv[i + 2:])
    assert rc == 0, text
    with open(os.path.join(out7, "shard_0000.json")) as fh:
        lat7 = json.load(fh)["latent_spec"]
    assert lat7["label_idx"] == list(range(7)) and lat7["free_idx"] == [7, 8, 9]
    assert np.asarray(lat7["class_centres"]).shape == (3, 7)


@pytest.mark.parametrize("value,provider", [("0,1,2,3,4,5,6,7", "reference"),
                                            ("0,1,2,3,4,5,6,7", "dsn"),
                                            ("0,0", "bench"), ("3,2", "bench"),
                                            ("0,1,2,3,4,5,6,7,8,9", "bench"), ("", "bench"),
                                            ("a", "bench"), ("1.5", "bench"), ("-1,2", "bench"),
                                            ("0,10", "bench")])
def test_o25_5_builder_refusals(project, value, provider):
    """O25.5: --label-axes with another provider, or a list bench_free_idx
    refuses: a non-zero exit (SystemExit) and nothing written."""
    out = os.path.join(project["root"], "bad_%s_%s" % (provider, value.replace(",", "_") or "empty"))
    argv = arg_set(arg_set(argv_of(project["cfg"], "R0"), "--label-axes", value), "--provider",
                   provider)
    rc, text = build(out, argv)
    assert rc != 0, text
    assert not os.path.exists(os.path.join(out, "shard_0000.npz")), text


def test_o25_6_arm_s_does_not_depend_on_the_split(project):
    """O25.6: arm S built with L = 0..7 and with the default L at one seed:
    every array of the shard byte-identical (the box prior ignores L)."""
    a8 = argv_of(project["cfg"], "sim")
    a7 = arg_set(a8, "--label-axes", "0,1,2,3,4,5,6")
    o8, o7 = os.path.join(project["root"], "s8"), os.path.join(project["root"], "s7")
    assert build(o8, a8)[0] == 0 and build(o7, a7)[0] == 0
    with np.load(os.path.join(o8, "shard_0000.npz")) as z8, \
            np.load(os.path.join(o7, "shard_0000.npz")) as z7:
        assert sorted(z8.files) == sorted(z7.files)
        for k in z8.files:
            assert z8[k].tobytes() == z7[k].tobytes(), k


def test_o25_7_arm_r_theta_law_with_the_duty_class_bearing(project):
    """O25.7: arm R built with L = 0..7, 150 donors per class: per class c
    and axis k in 0..7, the mean of theta_k within 5 standard errors of the
    truncated normal TN(mu_ck, 0.1^2; (0, 1)) (a two-sided 5-SE band: a false
    alarm 5.7e-7 per check); theta_8, theta_9 uniform (KS p > 1e-3), and
    theta_7 not uniform in any class (KS p < 1e-3)."""
    from scipy import stats
    n = 150
    argv = argv_of(project["cfg"], "R0")
    argv = arg_set(argv, "--donors-per-class", str(n))
    argv = arg_set(argv, "--n-traces", str(3 * n * 2))
    argv = arg_set(argv, "--n-windows", "1")
    out = os.path.join(project["root"], "rbig")
    rc, text = build(out, argv)
    assert rc == 0, text
    with open(os.path.join(out, "shard_0000.json")) as fh:
        lat = json.load(fh)["latent_spec"]
    mu = np.asarray(lat["class_centres"], dtype=np.float64)
    tau = float(lat["tau_ov"])
    with np.load(os.path.join(out, "shard_0000.npz")) as z:
        theta, cls, donor = z["theta"].astype(np.float64), z["cls"], z["donor"]
    _u, first = np.unique(donor, return_index=True)
    th, cl = theta[first], cls[first]
    assert sorted(np.bincount(cl).tolist()) == [n, n, n]
    for c in range(3):
        tc = th[cl == c]
        for j, k in enumerate(L8):
            a, b = (0.0 - mu[c, j]) / tau, (1.0 - mu[c, j]) / tau
            m = stats.truncnorm.mean(a, b, loc=mu[c, j], scale=tau)
            se = stats.truncnorm.std(a, b, loc=mu[c, j], scale=tau) / np.sqrt(len(tc))
            assert abs(tc[:, k].mean() - m) < 5 * se, (c, k, tc[:, k].mean(), m, se)
        assert stats.kstest(tc[:, 7], "uniform").pvalue < 1e-3, (c, "axis 7 looks uniform")
    for k in (8, 9):
        assert stats.kstest(th[:, k], "uniform").pvalue > 1e-3, k


def _builder_spy(argv, out):
    """build_latent_bank.main in this process with a spy on the map: the
    (free_idx, shift) of every call."""
    import build_latent_bank as BLB
    calls = []
    real = bbp.phi_to_bench_params

    def spy(*a, **k):
        calls.append((tuple(int(x) for x in k.get("free_idx", ())), float(k.get("free_axis_shift", 0))))
        return real(*a, **k)
    bbp.phi_to_bench_params = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            BLB.main(["--out-dir", out] + list(argv))
    finally:
        bbp.phi_to_bench_params = real
    return calls


def test_o25_7_range_shift_moves_the_banks_own_free_axes(project):
    """O25.7: building R3 (pi 0.3, the three modes) with L = 0..7 calls the
    map with free_idx (8, 9) and the shift 0.35 x 0.3 on every window; with
    the default L, (7, 8, 9)."""
    argv = argv_of(project["cfg"], "R3")
    calls = _builder_spy(argv, os.path.join(project["root"], "spy8"))
    assert calls and set(calls) == {((8, 9), 0.35 * 0.3)}, set(calls)
    calls7 = _builder_spy(arg_set(argv, "--label-axes", "0,1,2,3,4,5,6"),
                          os.path.join(project["root"], "spy7"))
    assert calls7 and set(calls7) == {((7, 8, 9), 0.35 * 0.3)}, set(calls7)


@pytest.mark.parametrize("value", ["__missing__", None, "0,1", [], [1, 1], [2, 1], [0, 10], [-1, 0],
                                   list(range(10)), [0.5], [1.0, 2.0], ["0"], [False, True]])
def test_o25_8_configuration_refusals(value):
    """O25.8: synthetic.builder.label_axes required, a non-empty list of
    distinct increasing integer indices 0-9 leaving a free axis; exactly one
    problem naming the key for each violation."""
    import test_smoke_synthetic as TS
    d = TS._committed()
    if value == "__missing__":
        del d["synthetic"]["builder"]["label_axes"]
    else:
        d["synthetic"]["builder"]["label_axes"] = value
    probs = [p for p in C.validate(d) if "label_axes" in p]
    assert len(probs) == 1, (value, C.validate(d))


@pytest.mark.parametrize("value", [[0], [9], list(range(9)), [3, 5, 7], L8])
def test_o25_8_configuration_accepts(value):
    import test_smoke_synthetic as TS
    d = TS._committed()
    d["synthetic"]["builder"]["label_axes"] = value
    assert not [p for p in C.validate(d) if "label_axes" in p]


def test_o25_8_9_committed_v4_and_build_of():
    """O25.8, O25.9: the committed synth.json loads with L = 0..7, the scale
    0.562743 and regular centres; every shard argv of every bank, and the
    calibration bank's, carry --label-axes 0,1,2,3,4,5,6,7 once."""
    from stages import synthetic as SY
    import ev_config
    import ev_plan
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    b = cfg.data["synthetic"]["builder"]
    assert b["label_axes"] == L8 and b["centre_scale"] == 0.562743 and b["class_centres"] == "regular"
    banks = ["sim"] + [r["name"] for r in cfg.data["synthetic"]["real"]]
    assert banks == ["sim", "R0", "R3"]
    for bank in banks:
        for argv in SY.build_of(cfg, bank)["argv"]:
            assert argv.count("--label-axes") == 1 and \
                argv[argv.index("--label-axes") + 1] == "0,1,2,3,4,5,6,7", (bank, argv)
    for argv in ev_plan.calbank_build(cfg, ev_config.load(H.SYNTH_EVAL))["argv"]:
        assert argv[argv.index("--label-axes") + 1] == "0,1,2,3,4,5,6,7", argv


def test_o25_11_free_axes_of_and_pseudo_true(project):
    """O25.11: free_axes_of; pseudo_true adds delta on the given free axes
    only, caps the duty only when it is one of them; delta = 0.35 pi read
    from a real R3 sidecar; the default equals 0e62645's function; the
    identity -- the unshifted map at theta-dagger equals the shifted map at
    theta* (1e-12 relative: float64 roundoff of affine maps)."""
    import ev_data
    assert tuple(ev_data.free_axes_of({"latent_spec": {"free_idx": [8, 9]}})) == (8, 9)
    for side in ({}, {"latent_spec": {}}, {"latent_spec": None}, None):
        assert tuple(ev_data.free_axes_of(side)) == (7, 8, 9), side
    out = os.path.join(project["root"], "r3_side")
    assert build(out, argv_of(project["cfg"], "R3"))[0] == 0
    with open(os.path.join(out, "shard_0000.json")) as fh:
        side = json.load(fh)
    gap = side["gap_spec"]
    free = tuple(ev_data.free_axes_of(side))
    assert free == (8, 9)
    rng = np.random.default_rng(2511)
    theta = rng.uniform(1e-9, 1 - 1e-9, size=(2000, 10))
    theta[0, 7] = 0.99
    dag, in_box, delta = ev_data.pseudo_true(theta, gap, free)
    assert abs(delta - 0.35 * 0.3) < 1e-15, delta
    want = theta.copy()
    want[:, [8, 9]] += delta
    assert_array_equal(dag, want)
    assert dag[0, 7] == 0.99
    assert_array_equal(in_box, np.all((want > 0) & (want < 1), axis=1))
    dag7, _i7, _d7 = ev_data.pseudo_true(theta, gap, (7, 8, 9))
    want7 = theta.copy()
    want7[:, [7, 8, 9]] += delta
    want7[:, 7] = np.minimum(want7[:, 7], 1.0)
    assert_array_equal(dag7, want7)
    old = H.module_at(H.BEFORE_B25_26, "hpc/pipeline/evaluation/ev_data.py", "ev_data_0e62645",
                      [H.EVAL, H.STAGE1])
    o_dag, o_in, o_d = old.pseudo_true(theta, gap)
    n_dag, n_in, n_d = ev_data.pseudo_true(theta, gap)
    assert_array_equal(n_dag, o_dag)
    assert_array_equal(n_in, o_in)
    assert n_d == o_d
    for f, dg in (((8, 9), dag), ((7, 8, 9), dag7)):
        for i in range(0, 2000, 7):
            a = affine(theta[i], delta, f)
            b = affine(np.clip(dg[i], -1, 2), 0.0, f)
            assert_allclose(b, a, rtol=1e-12, atol=1e-15)


def test_o25_12_callers_pass_the_cohorts_own_free_axes():
    """O25.12: every call of pseudo_true in the evaluation passes, third,
    free_axes_of(<the sidecar the gap_spec is read from>)."""
    seen = 0
    for name in sorted(os.listdir(H.EVAL)):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(H.EVAL, name)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) \
                    == "pseudo_true":
                seen += 1
                assert len(node.args) == 3, (name, ast.dump(node))
                third = node.args[2]
                assert isinstance(third, ast.Call) and getattr(third.func, "attr", "") == "free_axes_of"
                gap_src = ast.dump(node.args[1])
                assert ast.dump(third.args[0]) in gap_src, (name, ast.dump(node))
    assert seen >= 2, seen


@pytest.mark.parametrize("seed,outside", [(33, 0), (34, 3)])
def test_o25_13_evaluation_fixture_seed(project, seed, outside):
    """O25.13: the evaluation fixture's R3 (2 donors per class, 2 windows,
    pi 0.3) under v4's split: seed 33 leaves all 6 donors' theta-dagger in
    the box, seed 34 puts 3 outside (theta-dagger by hand: + 0.105 on 8, 9)."""
    import test_smoke_synthetic as TS
    root = tempfile.mkdtemp(prefix="t1d_b25_fix_")
    try:
        real = [{"name": "R0", "seed": 22, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
                 "gap_modes": ["range_shift", "drift", "contamination"]},
                {"name": "R3", "seed": seed, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
                 "gap_modes": ["range_shift", "drift", "contamination"]}]
        path, _d = TS._small(root, **{"synthetic.real": real})
        cfg = C.load_config(path, synthetic_ok=True)
        out = os.path.join(root, "r3")
        rc, text = build(out, argv_of(cfg, "R3"))
        assert rc == 0, text
        with np.load(os.path.join(out, "shard_0000.npz")) as z:
            theta, donor = z["theta"].astype(np.float64), z["donor"]
        _u, first = np.unique(donor, return_index=True)
        th = theta[first]
        assert len(th) == 6
        dag = th.copy()
        dag[:, [8, 9]] += 0.35 * 0.3
        got = int(np.sum(~np.all((dag > 0) & (dag < 1), axis=1)))
        assert got == outside, (seed, got)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_og_1_ascii_lf_python36():
    """OG.1: the changed modules ASCII and LF; the sequencer's (config.py,
    stages/synthetic.py) in Python 3.6 syntax (vermin)."""
    files = [os.path.join(H.STAGE1, "bench_burst_provider.py"),
             os.path.join(H.STAGE1, "build_latent_bank.py"),
             os.path.join(H.PIPE, "config.py"), os.path.join(H.PIPE, "stages", "synthetic.py"),
             os.path.join(H.EVAL, "ev_data.py"), os.path.join(H.EVAL, "ev_dataview.py"),
             os.path.join(H.EVAL, "ev_sample.py"), os.path.join(H.EVAL, "ev_data_figures.py"),
             H.SYNTH]
    for f in files:
        raw = open(f, "rb").read()
        assert b"\r" not in raw and all(c < 128 for c in raw), f
    vermin = shutil.which("vermin")
    if vermin is None:
        pytest.skip("vermin not installed")
    for f in (os.path.join(H.PIPE, "config.py"), os.path.join(H.PIPE, "stages", "synthetic.py")):
        rc, out = H.run([vermin, "-t=3.6-", "--no-tips", "--violations", f], dict(os.environ))
        assert rc == 0, (f, out)


def test_step6_same_argv_same_bytes(project):
    """Step 6 (determinism): a v4 bank shard built twice from one argv is
    byte-identical in every array (the builder is a pure function of its
    arguments and seed)."""
    for bank in ("sim", "R3"):
        argv = argv_of(project["cfg"], bank)
        a, b = (os.path.join(project["root"], "det_%s_%d" % (bank, i)) for i in (1, 2))
        assert build(a, argv)[0] == 0 and build(b, argv)[0] == 0
        with np.load(os.path.join(a, "shard_0000.npz")) as za, \
                np.load(os.path.join(b, "shard_0000.npz")) as zb:
            for k in za.files:
                assert za[k].tobytes() == zb[k].tobytes(), (bank, k)


def test_step8_swap_the_label_axes_through_the_bank_chain():
    """Step 8 (swappability): another split through the configuration --
    L = (0, 2, 4, 6, 8), F = (1, 3, 5, 7, 9), the duty free again -- runs the
    chain configuration -> build_of -> the builder -> the record tool (block
    16) -> free_axes_of -> pseudo_true, with theta-dagger shifted on F and the
    duty capped."""
    import test_smoke_synthetic as TS
    from stages import synthetic as SY
    import ev_data
    root = tempfile.mkdtemp(prefix="t1d_b25_swap_")
    try:
        path, _d = TS._small(root, **{"synthetic.builder.label_axes": [0, 2, 4, 6, 8]})
        cfg = C.load_config(path, synthetic_ok=True)
        for bank in ("sim", "R3"):
            bd = SY.bank_dir(cfg, bank)
            for argv in SY.build_of(cfg, bank)["argv"]:
                assert argv[argv.index("--label-axes") + 1] == "0,2,4,6,8"
                rc, text = build(bd, argv)
                assert rc == 0, text
            exp = os.path.join(root, "expect_%s.json" % bank)
            H.write(exp, json.dumps(SY.SynthBankStage(bank).expect(cfg)))
            rc, text = H.run([H.PYTHON, os.path.join(H.TOOLS, "synth_bank_record.py"), "--bank-dir",
                              bd, "--expect", exp, "--out", os.path.join(bd, "BANK_RECORD.json")],
                             dict(os.environ))
            assert rc == 0 and "SYNTH BANK RECORD OK" in text, text
        with open(os.path.join(SY.bank_dir(cfg, "R3"), "shard_0000.json")) as fh:
            side = json.load(fh)
        free = tuple(ev_data.free_axes_of(side))
        assert free == (1, 3, 5, 7, 9), free
        with np.load(os.path.join(SY.bank_dir(cfg, "R3"), "shard_0000.npz")) as z:
            theta = z["theta"].astype(np.float64)
        dag, _inb, delta = ev_data.pseudo_true(theta, side["gap_spec"], free)
        want = theta.copy()
        want[:, list(free)] += delta
        want[:, 7] = np.minimum(want[:, 7], 1.0)
        assert_array_equal(dag, want)
    finally:
        shutil.rmtree(root, ignore_errors=True)
