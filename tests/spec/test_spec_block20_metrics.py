"""Block 20 (the evaluation's quantities), derived from specs/SPEC.md sec. 2.5
and block 20 (oracles O20.1-O20.9 of the t1-C report).

Pure ASCII, LF only.
"""

import glob
import json
import math
import os
import shutil

import numpy as np
import pytest
from scipy.stats import kstwo, multivariate_normal, norm

from conftest import EVAL, load_json, run_py, write_eval_config

BETAS = np.round(np.arange(101) / 100.0, 10)


# ------------------------------------------------- O20.1 HPD rank, C(beta)

def test_hpd_rank_by_hand():
    import ev_metrics as EM
    lq = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [5.0, 6.0, 7.0, 8.0]])
    ref = np.array([2.5, 0.0, -np.inf])
    # strict >: 3, 4 exceed 2.5; equal values do not count; every draw exceeds -inf
    np.testing.assert_array_equal(EM.hpd_ranks(lq, ref), [0.5, 0.0, 1.0])


def test_coverage_curve_by_hand_and_grid_ends():
    import ev_metrics as EM
    g = np.array([0.0, 0.5, 0.5, 1.0])
    np.testing.assert_array_equal(EM.coverage_curve(g, np.array([0.0, 0.5, 0.99, 1.0])),
                                  [0.25, 0.75, 0.75, 1.0])
    np.testing.assert_array_equal(EM.BETAS, BETAS)
    # a rank equal to a grid point counts at that point (<=), whatever its float route
    for k in range(101):
        hits = np.zeros(100, dtype=bool)
        hits[:k] = True
        gamma = np.array([np.mean(hits)])
        assert EM.coverage_curve(gamma)[k] == 1.0, k


def _gaussian_case(scale, n=2000, P=200, seed=3):
    """theta ~ N(0, I), x = theta + N(0, 0.25 I) on 10 axes; the exact
    posterior N(x / 1.25, 0.2 I); `scale` multiplies its SD."""
    rng = np.random.default_rng(seed)
    d = 10
    th = rng.standard_normal((n, d))
    x = th + 0.5 * rng.standard_normal((n, d))
    mu, sd = x / 1.25, math.sqrt(0.2) * scale
    draws = mu[:, None, :] + sd * rng.standard_normal((n, P, d))
    lq_draws = norm.logpdf(draws, loc=mu[:, None, :], scale=sd).sum(axis=2)
    lq_ref = norm.logpdf(th, loc=mu, scale=sd).sum(axis=1)
    return lq_draws, lq_ref


def test_exact_posterior_is_calibrated():
    import ev_metrics as EM
    lq, ref = _gaussian_case(1.0)
    c = EM.coverage_curve(EM.hpd_ranks(lq, ref))
    # KS bound at 0.999 for n rows, plus one draw's resolution (1/P)
    assert np.max(np.abs(c - BETAS)) <= kstwo.ppf(0.999, lq.shape[0]) + 1.0 / lq.shape[1]


@pytest.mark.parametrize("scale,sign", [(0.5, -1), (2.0, +1)])
def test_too_narrow_and_too_wide(scale, sign):
    import ev_metrics as EM
    lq, ref = _gaussian_case(scale)
    s = EM.coverage_summary(EM.coverage_curve(EM.hpd_ranks(lq, ref)))
    assert sign * s["signed_area"] > 0.1, s


def test_binomial_band_at_one_half():
    """E20a: the binomial band's half-width 1.96 sqrt(0.25 / n) (at beta = 1/2;
    binomial quantiles are on a grid of 1/n)."""
    import ev_metrics as EM
    for n in (100, 400, 1000):
        lo, hi = EM.binomial_band(n, np.array([0.5]), 0.95)
        hw = 1.96 * math.sqrt(0.25 / n)
        assert abs((hi[0] - lo[0]) / 2.0 - hw) <= 1.0 / n + 1e-12, (n, lo, hi, hw)


def test_donor_bootstrap_band_unchanged_by_copies():
    import ev_metrics as EM
    rng = np.random.default_rng(7)
    G = 27
    donor = np.repeat(np.arange(G), 12)
    gamma = rng.uniform(size=donor.size)
    lo1, hi1 = EM.donor_bootstrap_band(gamma, donor, BETAS, 0.95, 300, 11)
    rep = np.repeat(np.arange(donor.size), 12)        # every row twelve times
    lo2, hi2 = EM.donor_bootstrap_band(gamma[rep], donor[rep], BETAS, 0.95, 300, 11)
    np.testing.assert_allclose(lo1, lo2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hi1, hi2, rtol=0, atol=1e-12)


# --------------------------------------------------------- O20.5 recovery

@pytest.mark.parametrize("shift,cov,z", [(0.0, 1.0, 0.0), (0.2, 0.0, -2.0)])
def test_recovery_known_posterior(shift, cov, z):
    import ev_metrics as EM
    rng = np.random.default_rng(9)
    n, P = 50, 4000
    mu = rng.uniform(0.3, 0.7, size=(n, 10))
    draws = mu[:, None, :] + 0.1 * rng.standard_normal((n, P, 10))
    ref = mu + shift
    rec = EM.recovery_arrays(ref, draws)
    s = EM.recovery_summary(ref, rec, np.ones(n, dtype=bool))
    # contraction 1 - 0.01 / (1/12) = 0.88; Monte Carlo of the SD over 4000 draws: 0.01
    np.testing.assert_allclose(s["contraction"], 0.88, atol=0.01)
    np.testing.assert_allclose(s["interval90_coverage"], cov, atol=0)
    # z-scores: Monte Carlo of the mean and SD, 0.1 / sqrt(4000) / 0.1 and 2 x 0.011
    np.testing.assert_allclose(s["zeta_mean"], z, atol=0.08)


# ------------------------------------------------- O20.4, O20.6-7 by hand

def test_nll_block_by_hand():
    import ev_metrics as EM
    lq = np.array([-1.0, -2.0, -np.inf, -4.0, -5.0])
    mem = np.array([[-1.0, -1.2], [-2.0, -2.2], [-np.inf, -np.inf], [-4.0, -4.4], [-5.0, -5.6]])
    ib = np.array([True, True, False, True, True])
    donor = np.array([0, 0, 1, 1, 2])
    s, a = EM.nll_block(lq, mem, ib, donor)
    assert s["n_rows"] == 4 and s["n_out_of_box"] == 1
    assert abs(s["mean"] - np.mean([1, 2, 4, 5])) < 1e-15
    np.testing.assert_allclose(a["donor_nll"], [1.5, 4.0, 5.0])
    np.testing.assert_allclose(s["members"], [3.0, (1.2 + 2.2 + 4.4 + 5.6) / 4])


def test_ppc_block_by_hand():
    import ev_metrics as EM
    obs = np.array([[5.0], [0.0]])
    sim = np.array([[[1.0], [2.0], [3.0], [4.0], [6.0]], [[1.0], [2.0], [3.0], [4.0], [6.0]]])
    s, a = EM.ppc_block(obs, sim)
    np.testing.assert_array_equal(a["ppc_rank"][:, 0], [4, 0])
    lo, hi = np.quantile(sim[0, :, 0], [0.05, 0.95])
    assert a["ppc_lo"][0, 0] == lo and a["ppc_hi"][0, 0] == hi
    np.testing.assert_array_equal(a["ppc_outside"][:, 0], [False, True])
    assert s["frac_outside_90"] == [0.5]


def _paired(d_rows, sd_t, sd_a, seed=1):
    import ev_metrics as EM
    donor = np.repeat(np.arange(27), 12)
    l_a1 = np.full(donor.size, 5.0)
    return EM.paired_block(l_a1 + d_rows, l_a1, donor, np.ones(donor.size, bool), sd_t, sd_a,
                           500, 0.95, seed)[0]


def test_paired_verdicts():
    rng = np.random.default_rng(4)
    noise = 0.01 * rng.standard_normal(27 * 12)
    s = _paired(-1.0 + noise, 0.1, 0.2)            # |Dbar| 1 > 2 x 0.2, interval < 0
    assert s["resolved"] and s["lower_nll"] == "term arm"
    assert abs(s["Dbar"] - np.mean(-1.0 + noise)) < 1e-12 and s["sigma_seed"] == 0.2
    s = _paired(-1.0 + noise, 0.6, 0.1)            # interval excludes 0, |Dbar| < 2 x 0.6
    assert s["ci_excludes_0"] and not s["beyond_2_sigma_seed"] and not s["resolved"]
    s = _paired(rng.standard_normal(27 * 12), 0.01, 0.01)    # beyond 2 sigma_seed? interval on 0
    assert s["ci"][0] < 0 < s["ci"][1] or s["beyond_2_sigma_seed"]
    assert s["resolved"] == (s["beyond_2_sigma_seed"] and s["ci_excludes_0"])


def test_running_min():
    import ev_metrics as EM
    assert EM.running_min([3.0, None, float("nan"), 2.0, 5.0, 1.0]) == [3.0, 3.0, 3.0, 2.0, 2.0, 1.0]
    assert EM.running_min([None, 1.0]) == [None, 1.0]


def test_example_donors_median_per_condition():
    import ev_metrics as EM
    ids = np.array([10, 11, 12, 13, 20, 21, 22])
    nll = np.array([4.0, 1.0, 3.0, 2.0, 7.0, 5.0, 6.0])
    cls = np.array([0, 0, 0, 0, 1, 1, 1])
    ex = EM.example_donors(ids, nll, cls)
    # condition 0: sorted 1, 2, 3, 4 -> the lower median 2 (donor 13); condition 1: 6 (donor 22)
    assert [(x["cls"], x["donor"]) for x in ex] == [(0, 13), (1, 22)]


def test_json_writer_nan_inf_to_null(tmp_path):
    import ev_metrics as EM
    p = str(tmp_path / "x.json")
    EM.write_json(p, {"a": float("nan"), "b": [np.float64(np.inf), -np.inf, 1.5], "c": np.int64(3)})
    txt = open(p).read()
    assert "NaN" not in txt and "Infinity" not in txt
    assert json.loads(txt) == {"a": None, "b": [None, None, 1.5], "c": 3}


# ------------------------------------------------------- the fixture's run

def _mdir(evaluation):
    return os.path.join(evaluation["out"], "metrics")


def _sample(evaluation, stem, D):
    return np.load(os.path.join(evaluation["out"], "samples", "%s_%s.npz" % (stem, D)))


def test_metrics_cli_and_index(evaluation):
    rc, txt = evaluation["logs"]["metrics"]
    assert rc == 0 and "EVAL METRICS OK" in txt, txt[-3000:]
    idx = load_json(os.path.join(_mdir(evaluation), "METRICS.json"))
    assert sorted(idx["ensembles"]) == sorted(evaluation["names"])
    assert not [x for x in idx["left_out"] if not x["why"].startswith("no BEST_FIT")], idx["left_out"]
    for f in idx["files"]:
        assert os.path.isfile(f), f
    for f in glob.glob(os.path.join(_mdir(evaluation), "*.json")):
        t = open(f).read()
        assert "NaN" not in t and "Infinity" not in t, f


def test_nll_and_coverage_recomputed(evaluation):
    """From the samples alone: the NLL over the in-box rows, the donor means,
    C(beta) over calib & in-box rows, with D_sim's binomial band; SBC over the
    calibration rows (theta-dagger's out-of-box rows included)."""
    from scipy.stats import binom
    ecfg = load_json(evaluation["ecfg"])
    for e in evaluation["names"]:
        stem = e.replace(":", "_")
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % stem))
        for f in prov["files"]:
            if f.endswith("_cal.npz"):
                continue
            D = os.path.basename(f)[len(stem) + 1:-4]
            z = np.load(f)
            js = load_json(os.path.join(_mdir(evaluation), "%s_%s.json" % (stem, D)))
            a = np.load(os.path.join(_mdir(evaluation), "%s_%s.npz" % (stem, D)))
            ib, cal = z["in_box"], z["calib"]
            l = -z["logq_ref"][ib]
            assert js["nll"]["n_rows"] == int(ib.sum())
            assert abs(js["nll"]["mean"] - np.mean(l)) <= 1e-12 * max(1, abs(np.mean(l)))
            for k, d in enumerate(a["donor_ids"]):
                sel = ib & (z["donor"] == d)
                if sel.any():
                    assert abs(a["donor_nll"][k] - np.mean(-z["logq_ref"][sel])) < 1e-12
                else:
                    assert np.isnan(a["donor_nll"][k])
            rows = cal & ib
            g = np.mean(z["logq_draws"][rows].astype(np.float64)
                        > z["logq_ref"][rows][:, None], axis=1)
            c = np.array([np.mean(g <= b) for b in BETAS])
            np.testing.assert_allclose(a["coverage"], c, rtol=0, atol=1e-15)
            assert js["coverage"]["n_rows"] == int(rows.sum())
            if D == "sim":
                n = int(rows.sum())
                np.testing.assert_allclose(a["coverage_lo"], binom.ppf(0.025, n, BETAS) / n)
                np.testing.assert_allclose(a["coverage_hi"], binom.ppf(0.975, n, BETAS) / n)
            assert js["sbc"]["n_rows"] == int(cal.sum())
            ranks = np.sum(z["draws"][cal].astype(np.float64)
                           < z["theta_ref"][cal][:, None, :], axis=1)
            np.testing.assert_array_equal(a["sbc_ranks"], ranks)
            # recovery: contraction 1 - mean sigma^2 / (1/12) over the calibration rows (the
            # SD's ddof is not in the spec: either convention accepted)
            got = np.asarray(js["recovery"]["contraction"])
            fits = []
            for ddof in (0, 1):
                sig = z["draws"][cal].astype(np.float64).std(axis=1, ddof=ddof)
                fits.append(np.max(np.abs(got - (1.0 - np.mean(sig ** 2, axis=0) * 12.0))))
            assert min(fits) <= 1e-6, fits
            assert ecfg["level"] == js["level"]


def test_paired_recomputed(evaluation):
    for f in sorted(glob.glob(os.path.join(_mdir(evaluation), "paired_*.json"))):
        js = load_json(f)
        eT, eA = js["ensembles"]["term"], js["ensembles"]["A1"]
        D = js["dataset"]
        zT = _sample(evaluation, eT.replace(":", "_"), D)
        zA = _sample(evaluation, eA.replace(":", "_"), D)
        ok = zT["in_box"] & np.isfinite(zT["logq_ref"]) & np.isfinite(zA["logq_ref"])
        d = -zT["logq_ref"][ok] + zA["logq_ref"][ok]
        assert abs(js["Dbar"] - np.mean(d)) <= 1e-12
        sds = []
        for z in (zT, zA):
            okz = z["in_box"] & np.isfinite(z["logq_ref"])
            sds.append(np.std(-np.mean(z["logq_ref_members"][okz], axis=0), ddof=1))
        assert abs(js["sigma_seed"] - max(sds)) <= 1e-12
        assert js["resolved"] == (abs(js["Dbar"]) > 2 * js["sigma_seed"]
                                  and (js["ci"][0] > 0 or js["ci"][1] < 0))
        assert js["n_donors"] == np.unique(zT["donor"][ok]).size


def test_search_summary_order_arms_running_min(evaluation):
    cfg = load_json(evaluation["cfg"])
    for q in cfg["synthetic"]["searches"]:
        p = os.path.join(_mdir(evaluation), "search_%s.json" % q["name"])
        js = load_json(p)
        sdir = os.path.join(cfg["paths"]["tune_root"], q["name"], q["campaign"])
        rounds = load_json(os.path.join(sdir, "pipeline_rounds.json"))["rounds"]
        order = [t for r in rounds for t in r["trial_ids"]]
        assert [t["trial_id"] for t in js["trials"]] == order
        cur = {}
        for t in js["trials"]:
            rec = load_json(os.path.join(sdir, "trials", "%s.json" % t["trial_id"]))
            dsn, rep = int(rec["config"]["dsn_on"]), int(rec["config"]["rep_on"])
            arm = "A2" if dsn and not rep else ("A5" if rep and not dsn else ("A1" if not dsn else None))
            assert t["arm"] == arm and t["nll"] == rec["nll"]
            if rec["status"] == "ok" and arm:
                cur[arm] = min(cur.get(arm, float("inf")), rec["nll"])
            for a, v in t["running_min_nll"].items():
                assert v == cur.get(a), (t, cur)
        sc = load_json(os.path.join(sdir, "SEARCH_COMPLETE.json"))
        assert js["finalist"] == sc["finalist"]["trial_id"]


def test_left_out_and_refusals(evaluation, tmp_path):
    ec = write_eval_config(str(tmp_path / "e.json"), load_json(evaluation["ecfg"])["out_root"],
                           n_post=80)
    rc, txt = run_py([os.path.join(EVAL, "ev_metrics.py"), "--config", evaluation["cfg"],
                      "--eval-config", ec])
    assert rc == 3 and "REFUSE" in txt, txt[-1500:]
    rc, txt = run_py([os.path.join(EVAL, "ev_metrics.py"), "--config", evaluation["cfg"],
                      "--eval-config", evaluation["ecfg"], "--ensemble", "S-A1:A5"])
    assert rc == 3 and "REFUSE" in txt, txt[-1500:]


def _strip_times(obj):
    if isinstance(obj, dict):
        return dict((k, _strip_times(v)) for k, v in obj.items() if k != "seconds")
    if isinstance(obj, list):
        return [_strip_times(v) for v in obj]
    return obj


def test_one_worker_and_three_give_the_same_files(evaluation, tmp_path):
    mdir = _mdir(evaluation)
    keep = str(tmp_path / "one")
    shutil.copytree(mdir, keep)
    try:
        rc, txt = run_py([os.path.join(EVAL, "ev_metrics.py"), "--config", evaluation["cfg"],
                          "--eval-config", evaluation["ecfg"], "--ncpus", "3"])
        assert rc == 0, txt[-2000:]
        for f in sorted(glob.glob(os.path.join(keep, "*"))):
            g = os.path.join(mdir, os.path.basename(f))
            if f.endswith(".json"):
                assert _strip_times(load_json(f)) == _strip_times(load_json(g)), f
            else:
                a, b = np.load(f), np.load(g)
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k], err_msg="%s %s" % (f, k))
    finally:
        shutil.rmtree(mdir)
        shutil.copytree(keep, mdir)


def test_swap_the_level(evaluation, tmp_path):
    """Directive audit, swappability: the bands' level through the
    configuration alone (0.90): the metrics run, D_sim's band is the binomial
    band at that level."""
    from scipy.stats import binom
    out = str(tmp_path / "ev")
    shutil.copytree(os.path.join(evaluation["out"], "samples"), os.path.join(out, "samples"))
    # the samples' values are the session's (cal_glob is one of SAMPLE_PARAMS)
    ec = write_eval_config(str(tmp_path / "e.json"), out, level=0.9,
                           cal_glob=load_json(evaluation["ecfg"])["cal_glob"])
    rc, txt = run_py([os.path.join(EVAL, "ev_metrics.py"), "--config", evaluation["cfg"],
                      "--eval-config", ec])
    assert rc == 0 and "EVAL METRICS OK" in txt, txt[-2000:]
    stem = evaluation["names"][0].replace(":", "_")
    js = load_json(os.path.join(out, "metrics", "%s_sim.json" % stem))
    a = np.load(os.path.join(out, "metrics", "%s_sim.npz" % stem))
    n = js["coverage"]["n_rows"]
    assert js["level"] == 0.9
    np.testing.assert_allclose(a["coverage_lo"], binom.ppf(0.05, n, BETAS) / n)


def test_sbc_rejects_the_shifted_axis_alone():
    """E20a: exact posterior draws on ten axes, one axis shifted by 1.5 SD:
    the Holm family rejects that axis and no other (fixed seed)."""
    import ev_metrics as EM
    rng = np.random.default_rng(12)
    n, P = 400, 200
    mu = rng.uniform(0.3, 0.7, size=(n, 10))
    theta = mu + 0.05 * rng.standard_normal((n, 10))
    draws = mu[:, None, :] + 0.05 * rng.standard_normal((n, P, 10))
    draws[:, :, 4] += 0.075
    s, _a = EM.sbc_block(theta, draws, 0.95, 5)
    assert [i for i, r in enumerate(s["holm_rejected"]) if r] == [4], s["holm_adjusted_p"]


def test_tarp_x_references_see_a_posterior_equal_to_the_prior():
    """E20a: a 'posterior' that ignores x (prior draws): TARP with references
    read out from x rejects it, with uniform references it does not."""
    import ev_metrics as EM
    rng = np.random.default_rng(21)
    n, P = 600, 200
    theta = rng.uniform(size=(n, 10))
    stats = theta[:, :8] + 0.05 * rng.standard_normal((n, 8))     # informative windows
    draws = rng.uniform(size=(n, P, 10))                           # the prior, whatever x
    s, _a = EM.tarp_block(theta, draws, stats, 3)
    assert s["x"]["ks_p"] < 1e-3 and s["x"]["reference_mode"] == "x-dependent", s
    assert s["random"]["ks_p"] > 1e-3, s


def test_embedding_mmd_one_law_and_two():
    import ev_metrics as EM
    rng = np.random.default_rng(8)
    z_sim = rng.standard_normal((1, 400, 4))
    same = rng.standard_normal((1, 60, 4))
    other = rng.standard_normal((1, 60, 4)) + 1.0
    cls = np.zeros(400), np.zeros(60)
    s1, _ = EM.embedding_block(z_sim, same, cls[0], cls[1], 2)
    s2, _ = EM.embedding_block(z_sim, other, cls[0], cls[1], 2)
    assert s1["members"][0]["p"] > 0.01 and s2["members"][0]["p"] <= 0.01
    assert abs(s2["members"][0]["mmd2"] - s2["members"][0]["mmd"] ** 2) < 1e-15
