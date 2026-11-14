"""Chain C3 (the evaluation), run t2-C3 of D-131: the quantities that must
carry through the chain unchanged, checked on what each producer actually
wrote (step 7 of the tester's procedure) -- the runner's per-row files and
records and the best fits' BEST_FIT_COMPLETE.json (block 7, block 18), the
banks (block 16), the samples (block 19), the metrics (block 20) and the data
overview (block 23). Expected values from SPEC.md (secs. 2.3-2.5, blocks 7 and
16-23), numpy and scipy; never from the code under test.

    python3 -m pytest tests/spec/test_spec_chain_C3_carry.py -q
    C3_FIXTURE=<dir> python3 -m pytest tests/spec/test_spec_chain_C3_carry.py -q

Pure ASCII, LF only.
"""

import glob
import json
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy.special import logsumexp

from chain_c3_support import fixture

ID_STRIDE = 1000000          # block 12: bank-wide ids, shard k's are k * 10^6 + local


# ------------------------------------------------------------------ helpers

def _cfg(F):
    with open(F["cfg"]) as fh:
        return json.load(fh)


def _ecfg(F, run="A"):
    with open(F["rec"][run]["ecfg"]) as fh:
        return json.load(fh)


def _ensembles(F):
    """[(name, stem, folder, search dict, records)] read from the best-fit
    folders (block 19's rule), in the configuration's order."""
    cfg = _cfg(F)
    out = []
    for q in cfg["synthetic"]["searches"]:
        for suffix in ["_best"] + (["_best_alt"] if q["alt_best_fit"] else []):
            folder = os.path.join(cfg["paths"]["runs_root"], q["name"], cfg["dataset"] + suffix)
            with open(os.path.join(folder, "BEST_FIT_COMPLETE.json")) as fh:
                bf = json.load(fh)
            recs = []
            for s in bf["member_seeds"]:
                recs.append(glob.glob(os.path.join(folder, "member_seed%d" % s,
                                                   "*_seed%d.json" % s))[0])
            arm = json.load(open(recs[0]))["arm"]
            out.append(("%s:%s" % (q["name"], arm), "%s_%s" % (q["name"], arm), folder, q, recs, bf))
    return out


def _sample(F, stem, D, run="A"):
    return np.load(os.path.join(F["rec"][run]["out_root"], "samples", "%s_%s.npz" % (stem, D)))


def _bank(cfg, name):
    """Every array of a bank's shards, concatenated in sorted path order
    (block 5's concatenated order), read with numpy; and the first sidecar."""
    d = os.path.join(cfg["paths"]["banks_root"], "%s_%s" % (cfg["dataset"], name))
    paths = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    assert paths, d
    parts, sizes = {}, []
    for p in paths:
        with np.load(p) as z:
            n = z["theta"].shape[0]
            sizes.append(n)
            for k in ("theta", "donor", "well", "cls", "x"):
                parts.setdefault(k, []).append(np.asarray(z[k]))
    arr = dict((k, np.concatenate(v)) for k, v in parts.items())
    with open(paths[0][:-4] + ".json") as fh:
        side = json.load(fh)
    return arr, side, sizes


def _perrow(rec_path):
    return np.load(rec_path[:-len(".json")] + "_perrow.npz")


# ------------------------------------------------------------------ D_sim (blocks 7 -> 19)

def test_dsim_rows_are_every_members_report_rows():
    """Sec. 2.5: D_sim = the report split of split seed 1, the rows every
    member scores in its _perrow.npz -- row for row: theta (the runner's
    float32, widened exactly) and donor (`group`)."""
    F = fixture()
    for name, stem, _f, _q, recs, _bf in _ensembles(F):
        z = _sample(F, stem, "sim")
        for r in recs:
            pr = _perrow(r)
            assert_array_equal(z["theta_ref"], pr["theta"].astype(np.float64), err_msg=name)
            assert_array_equal(z["donor"], np.asarray(pr["group"]).astype(np.int64), err_msg=name)


def test_dsim_is_one_split_for_every_ensemble():
    """Sec. 2.5: the same split for every member, every arm and every
    search: identical rows in every ensemble's D_sim."""
    F = fixture()
    ens = _ensembles(F)
    ref = _sample(F, ens[0][1], "sim")["rows"]
    for name, stem, *_ in ens[1:]:
        assert_array_equal(_sample(F, stem, "sim")["rows"], ref, err_msg=name)


def test_dsim_rows_against_the_bank():
    """Block 19: D_sim's rows in global row order (bank rows of the
    concatenated shards); theta_ref, donor, well, cls the bank's; the report
    split takes whole donors (sec. 2.3): every row of a report donor is in
    D_sim; n_report donors = n_d - round(0.85 n_d) - round(0.10 n_d)."""
    F = fixture()
    cfg = _cfg(F)
    bank, _side, _sizes = _bank(cfg, "sim")
    z = _sample(F, _ensembles(F)[0][1], "sim")
    rows = z["rows"]
    assert rows.dtype == np.int64 and np.all(np.diff(rows) > 0)
    assert_array_equal(z["theta_ref"], bank["theta"][rows].astype(np.float64))
    assert_array_equal(z["donor"], bank["donor"][rows])
    assert_array_equal(z["well"], bank["well"][rows])
    assert_array_equal(z["cls"], bank["cls"][rows])
    donors = np.unique(z["donor"])
    assert_array_equal(np.sort(rows), np.flatnonzero(np.isin(bank["donor"], donors)))
    n_d = np.unique(bank["donor"]).size
    f = cfg["split"]["fractions"]
    assert donors.size == n_d - int(round(f[0] * n_d)) - int(round(f[1] * n_d)), (donors.size, n_d)


def test_dsim_donor_ids_are_bank_wide():
    """Block 12 (F-at): a donor id of shard k is k * 10^6 + its local id."""
    F = fixture()
    cfg = _cfg(F)
    _bank_arr, _side, sizes = _bank(cfg, "sim")
    z = _sample(F, _ensembles(F)[0][1], "sim")
    shard_of = np.searchsorted(np.cumsum(sizes), z["rows"], side="right")
    assert_array_equal(z["donor"] // ID_STRIDE, shard_of)


# ------------------------------------------------------------------ the NLLs (blocks 7 -> 19 -> 20)

def test_ensemble_dsim_nll_equals_block7_L_bar():
    """Sec. 2.5 (E1): the mixture's mean NLL over the report split is block
    7's L_bar. Tolerance oracle_tol: block 19 bounds every member's rebuilt
    log density row for row by it, and logsumexp is 1-Lipschitz in the max
    norm, so the mixture's mean moves by at most that much."""
    F = fixture()
    tol = _ecfg(F)["oracle_tol"]
    for name, stem, _f, _q, recs, bf in _ensembles(F):
        z = _sample(F, stem, "sim")
        assert z["in_box"].all(), name
        assert bf["ensemble"]["n_report"] == z["logq_ref"].size, name
        L = -float(np.mean(z["logq_ref"]))
        assert abs(L - bf["ensemble"]["L_bar"]) <= tol, (name, L, bf["ensemble"]["L_bar"])
        for k, r in enumerate(recs):
            Lk = -float(np.mean(z["logq_ref_members"][:, k]))
            with open(r) as fh:
                rk = json.load(fh)
            assert abs(Lk - bf["ensemble"]["L_members"][k]) <= tol, (name, k, Lk)
            assert abs(Lk - rk["L"]) <= tol, (name, k, Lk, rk["L"])


def test_metrics_nll_is_the_samples_mean():
    """Block 20: the NLL is -logq_ref, its mean over the in-box rows; the
    metrics file of D_sim carries that mean (1e-9: the same float64 sum)."""
    F = fixture()
    md = os.path.join(F["rec"]["A"]["out_root"], "metrics")
    for name, stem, *_ in _ensembles(F):
        z = _sample(F, stem, "sim")
        with open(os.path.join(md, "%s_sim.json" % stem)) as fh:
            js = json.load(fh)
        assert_allclose(js["nll"]["mean"], -np.mean(z["logq_ref"][z["in_box"]]), rtol=0, atol=1e-9)


def test_members_cohort_nll_equals_their_L_pseudo_real():
    """Sec. 2.4 / block 19: each member's mean NLL on its search's own
    cohort is its record's L_pseudo_real (the mean over the cohort's n_R
    windows of the withheld theta's NLL); oracle_tol as above. The
    ensemble's mean equals the mixture of the members' per-row
    nll_pseudo_real, the same bound."""
    F = fixture()
    tol = _ecfg(F)["oracle_tol"]
    for name, stem, _f, q, recs, _bf in _ensembles(F):
        z = _sample(F, stem, q["real"])
        assert z["in_box"].all(), name
        per = []
        for k, r in enumerate(recs):
            with open(r) as fh:
                rk = json.load(fh)
            Lk = -float(np.mean(z["logq_ref_members"][:, k]))
            assert abs(Lk - rk["L_pseudo_real"]) <= tol, (name, k, Lk, rk["L_pseudo_real"])
            pr = _perrow(r)
            assert_array_equal(np.asarray(pr["group_pseudo_real"]).astype(np.int64), z["donor"],
                               err_msg=name)
            per.append(np.asarray(pr["nll_pseudo_real"], dtype=np.float64))
        M = len(per)
        mix = -np.mean(logsumexp(-np.stack(per), axis=0) - math.log(M))
        assert abs(mix - (-np.mean(z["logq_ref"]))) <= tol, (name, mix)


def test_mixture_formula_and_out_of_box_rows():
    """Sec. 2.5 (E1) and block 19: logq_ref = logsumexp_m(logq_ref_members)
    - log M where theta^ref is in the open box, -inf elsewhere; in_box =
    all(0 < theta^ref < 1). 1e-12: the same float64 formula."""
    F = fixture()
    out = F["rec"]["A"]["out_root"]
    n_files = 0
    for p in sorted(glob.glob(os.path.join(out, "samples", "*.npz"))):
        if p.endswith("_cal.npz"):
            continue
        z = np.load(p)
        n_files += 1
        inb = np.all((z["theta_ref"] > 0) & (z["theta_ref"] < 1), axis=1)
        assert_array_equal(z["in_box"], inb, err_msg=p)
        M = z["logq_ref_members"].shape[1]
        want = logsumexp(z["logq_ref_members"][inb], axis=1) - math.log(M)
        assert_allclose(z["logq_ref"][inb], want, rtol=0, atol=1e-12, err_msg=p)
        assert np.all(np.isneginf(z["logq_ref"][~inb])) and np.all(
            np.isneginf(z["logq_ref_members"][~inb])), p
    assert n_files >= 5


# ------------------------------------------------------------------ the cohorts (blocks 16 -> 19)

def test_cohort_data_sets_are_their_banks():
    """Block 19: D_<r> = every window of cohort r in bank order, theta^ref
    the bank's theta; n_R = n_classes x donors_per_class x wells x
    n_windows; rows 0..n_R-1; every row a calibration and a PPC row."""
    F = fixture()
    cfg = _cfg(F)
    b = cfg["synthetic"]["builder"]
    for name, stem, _f, q, _recs, _bf in _ensembles(F):
        cohorts = [r["name"] for r in cfg["synthetic"]["real"]] if q["campaign"] == "S-A1" \
            else [q["real"]]
        for c in cohorts:
            r = [x for x in cfg["synthetic"]["real"] if x["name"] == c][0]
            bank, _side, _sizes = _bank(cfg, "real_" + c)
            z = _sample(F, stem, c)
            n = b["n_classes"] * r["donors_per_class"] * b["wells_per_donor"] * r["n_windows"]
            assert z["theta_ref"].shape == (n, 10), (name, c, z["theta_ref"].shape)
            assert_array_equal(z["rows"], np.arange(n))
            assert_array_equal(z["theta_ref"], bank["theta"].astype(np.float64))
            for k in ("donor", "well", "cls"):
                assert_array_equal(z[k], bank[k], err_msg="%s %s %s" % (name, c, k))
            assert z["calib"].all() and z["ppc_rows"].size == n


def test_theta_dagger_formula_and_blocks_19_23_agree():
    """Sec. 2.5 D_R3+ with block 25's free axes: theta^dag = theta* + delta
    on the cohort's free axes (its sidecar's latent_spec.free_idx; 8-9 under
    the label axes 0-7), delta = 0.35 pi; the duty capped at 1 only when it
    is free. Block 19's <stem>_R3dag.npz and block 23's data_view.npz must
    hold the same theta^dag and in_box per donor; R0 (pi 0) has none."""
    F = fixture()
    cfg = _cfg(F)
    A = F["rec"]["A"]
    dv = np.load(os.path.join(A["out_root"], "data", "data_view.npz"))
    _bank_arr, side, _s = _bank(cfg, "real_R3")
    free = list(side["latent_spec"]["free_idx"])
    assert free == [8, 9], free                      # block 25: label axes 0-7
    pi = [r for r in cfg["synthetic"]["real"] if r["name"] == "R3"][0]["pi"]
    delta = 0.35 * pi
    assert_allclose(float(dv["R3_delta"]), delta, rtol=0, atol=1e-15)
    assert float(dv["R0_delta"]) == 0.0
    donors = list(dv["R3_donors"])
    for name, stem, _f, q, _r, _bf in _ensembles(F):
        cohorts = ["R0", "R3"] if q["campaign"] == "S-A1" else [q["real"]]
        assert not os.path.exists(os.path.join(A["out_root"], "samples", "%s_R0dag.npz" % stem))
        if "R3" not in cohorts:
            continue
        z = _sample(F, stem, "R3dag")
        base = _sample(F, stem, "R3")["theta_ref"]
        want = base.copy()
        want[:, free] += delta
        assert_allclose(z["theta_ref"], want, rtol=0, atol=1e-15, err_msg=name)
        assert_array_equal(z["in_box"], np.all((want > 0) & (want < 1), axis=1))
        idx = np.array([donors.index(d) for d in z["donor"]])
        assert_allclose(z["theta_ref"], dv["R3_theta_dag_d"][idx], rtol=0, atol=0, err_msg=name)
        assert_array_equal(z["in_box"], dv["R3_in_box_d"][idx])


def test_fixed_statistics_blocks_19_23_agree():
    """The eight fixed statistics of one window are computed by block 19
    (the samples' `stats`) and block 23 (data_view's T): the same windows
    give the same float32 values -- every cohort window, and the D_sim
    rows that lie in block 23's S_sub."""
    F = fixture()
    A = F["rec"]["A"]
    dv = np.load(os.path.join(A["out_root"], "data", "data_view.npz"))
    stem = _ensembles(F)[0][1]                    # S-A1 is scored on every cohort
    for c in ("R0", "R3"):
        assert_array_equal(_sample(F, stem, c)["stats"], dv["%s_T" % c], err_msg=c)
    z = _sample(F, stem, "sim")
    pos = dict((int(r), i) for i, r in enumerate(dv["sim_rows"]))
    common = [(i, pos[int(r)]) for i, r in enumerate(z["rows"]) if int(r) in pos]
    if not common:
        pytest.skip("no D_sim row lies in S_sub")
    a = np.array([i for i, _j in common])
    b = np.array([j for _i, j in common])
    assert_array_equal(z["stats"][a], dv["sim_T"][b])


# ------------------------------------------------------------------ seeds, layout, calibration

def test_seeds_common_to_the_ensembles():
    """Block 19: member k's draws on D from seed_of(seed, D) + k, the PPC
    from seed_of(seed, D, "ppc"), the calibration draws from seed_of(seed,
    "cal"), whatever the ensemble: equal recorded seeds per data set."""
    F = fixture()
    out = F["rec"]["A"]["out_root"]
    seen = {}
    for name, stem, *_ in _ensembles(F):
        with open(os.path.join(out, "samples", "%s_sample.json" % stem)) as fh:
            prov = json.load(fh)
        for D, s in prov["seeds"].items():
            seen.setdefault(D, set()).add(s)
    assert set(seen) >= {"sim", "R0", "R3", "cal"}, seen
    assert all(len(v) == 1 for v in seen.values()), seen


def test_draw_and_ppc_layout():
    """Block 19's npz contract along the chain: draw_member = n_post / M
    draws per member in member order; ppc_draw_idx = j n_post / ppc_draws;
    D_sim's calibration rows the first row of each donor, its PPC rows the
    first ppc_rows of them."""
    F = fixture()
    e = _ecfg(F)
    for name, stem, _f, _q, recs, _bf in _ensembles(F):
        M = len(recs)
        z = _sample(F, stem, "sim")
        assert_array_equal(z["draw_member"], np.repeat(np.arange(M), e["n_post"] // M))
        assert z["draws"].shape == (z["rows"].size, e["n_post"], 10)
        assert_array_equal(z["ppc_draw_idx"],
                           (np.arange(e["ppc_draws"]) * e["n_post"]) // e["ppc_draws"])
        _u, first = np.unique(z["donor"], return_index=True)
        calib = np.zeros(z["donor"].size, dtype=bool)
        calib[first] = True
        assert_array_equal(z["calib"], calib)
        assert_array_equal(z["ppc_rows"], np.flatnonzero(calib)[:e["ppc_rows"]])
        assert z["ppc_stats"].shape == (z["ppc_rows"].size, e["ppc_draws"], 8)
        assert np.all(np.isfinite(z["ppc_stats"]))


def test_calibration_bank_matches_the_simulated_bank():
    """Block 19/22 (Q-E5): cal_n windows of arm S on the box prior, one per
    donor, at cal_seed -- a seed no bank of the project has -- with the
    simulated bank's other options (W, f_s, T_win, n_neurons)."""
    F = fixture()
    cfg = _cfg(F)
    e = _ecfg(F)
    paths = sorted(glob.glob(e["cal_glob"]))
    assert len(paths) == e["cal_shards"], paths
    th, don, W = [], [], set()
    for p in paths:
        with np.load(p) as z:
            th.append(z["theta"])
            don.append(z["donor"])
            W.add(z["x"].shape[1])
    _b, sim_side, _s = _bank(cfg, "sim")
    with open(paths[0][:-4] + ".json") as fh:
        cs = json.load(fh)
    assert sum(t.shape[0] for t in th) == e["cal_n"]
    assert np.unique(np.concatenate(don)).size == e["cal_n"]
    assert W == {int(sim_side["W"])}, (W, sim_side["W"])
    lat, slat = cs["latent_spec"], sim_side["latent_spec"]
    assert lat["kind"] == "box" and int(lat["seed"]) == e["cal_seed"]
    seeds = [cfg["synthetic"]["sim"]["seed"]] + [r["seed"] for r in cfg["synthetic"]["real"]]
    assert e["cal_seed"] not in seeds
    for k in ("fs", "T_win", "n_neurons"):
        assert lat[k] == slat[k], (k, lat[k], slat[k])


# ------------------------------------------------------------------ the searches (blocks 7, 18 -> 20)

def test_search_summaries_follow_the_ledgers():
    """Block 20: the trials in proposal order (pipeline_rounds.json), each
    with its arm, the ledger's nll and L_pseudo_real; the running minimum of
    nll per arm over the trials that ended ok; the search's finalist the
    SEARCH_COMPLETE.json one."""
    F = fixture()
    cfg = _cfg(F)
    md = os.path.join(F["rec"]["A"]["out_root"], "metrics")
    for q in cfg["synthetic"]["searches"]:
        sdir = os.path.join(cfg["paths"]["tune_root"], q["name"], q["campaign"])
        with open(os.path.join(sdir, "pipeline_rounds.json")) as fh:
            rounds = json.load(fh)["rounds"]
        ids = [t for r in rounds for t in r.get("trial_ids") or []]
        with open(os.path.join(md, "search_%s.json" % q["name"])) as fh:
            s = json.load(fh)
        assert [t["trial_id"] for t in s["trials"]] == ids, q["name"]
        best = {}
        for t in s["trials"]:
            with open(os.path.join(sdir, "trials", "%s.json" % t["trial_id"])) as fh:
                r = json.load(fh)
            assert (t["arm"], t["status"], t["nll"], t["L_pseudo_real"]) == \
                (r.get("arm"), r.get("status"), r.get("nll"), r.get("L_pseudo_real")), t
            if r.get("status") == "ok":
                best[r["arm"]] = min(best.get(r["arm"], float("inf")), r["nll"])
            for a, v in (t.get("running_min_nll") or {}).items():
                assert v == best.get(a), (q["name"], t["trial_id"], a, v, best.get(a))
        with open(os.path.join(sdir, "SEARCH_COMPLETE.json")) as fh:
            sc = json.load(fh)
        assert s["finalist"] == sc["finalist"]["trial_id"], q["name"]
        if q["campaign"] == "S-A2-only":
            assert set(t["arm"] for t in s["trials"]) == {"A2"}, q["name"]
        if q["campaign"] == "S-A5-only":
            assert set(t["arm"] for t in s["trials"]) == {"A5"}, q["name"]


def test_members_learning_curves_are_their_records():
    F = fixture()
    md = os.path.join(F["rec"]["A"]["out_root"], "metrics")
    for name, _stem, _f, q, recs, _bf in _ensembles(F):
        with open(os.path.join(md, "search_%s.json" % q["name"])) as fh:
            s = json.load(fh)
        mem = s["ensembles"][name]["members"]
        assert len(mem) == len(recs)
        for m, r in zip(mem, recs):
            with open(r) as fh:
                rk = json.load(fh)
            assert m["val_npe"] == [h.get("val_npe") for h in rk["history"]], name
            assert m["L"] == rk["L"] and m["L_pseudo_real"] == rk["L_pseudo_real"]


# ------------------------------------------------------------------ determinism

def _norm(obj, out_root):
    if isinstance(obj, dict):
        return dict((k, _norm(v, out_root)) for k, v in obj.items() if k != "seconds")
    if isinstance(obj, list):
        return [_norm(v, out_root) for v in obj]
    if isinstance(obj, str):
        return obj.replace(out_root, "<OUT>")
    return obj


def test_two_runs_same_seed_identical():
    """Step 6: the chain run twice with the same values (run A in one
    submission, run B interrupted at every boundary, a resampling killed
    half-way) gives identical arrays -- every sample, metrics and data-view
    array bit for bit -- the same metrics files (paths and timings aside) and
    the same figures."""
    F = fixture()
    A, B = F["rec"]["A"]["out_root"], F["rec"]["B"]["out_root"]
    for sub, pat in (("samples", "*.npz"), ("metrics", "*.npz"), ("data", "data_view.npz")):
        fa = sorted(os.path.basename(p) for p in glob.glob(os.path.join(A, sub, pat)))
        fb = sorted(os.path.basename(p) for p in glob.glob(os.path.join(B, sub, pat)))
        assert fa == fb and fa, (sub, fa, fb)
        for f in fa:
            za, zb = np.load(os.path.join(A, sub, f)), np.load(os.path.join(B, sub, f))
            assert sorted(za.files) == sorted(zb.files), f
            for k in za.files:
                assert_array_equal(za[k], zb[k], err_msg="%s %s" % (f, k))
    for p in sorted(glob.glob(os.path.join(A, "metrics", "*.json"))):
        f = os.path.basename(p)
        with open(p) as fh:
            ja = _norm(json.load(fh), A)
        with open(os.path.join(B, "metrics", f)) as fh:
            jb = _norm(json.load(fh), B)
        if f == "METRICS.json":
            for k in ("report_inputs", "files"):
                ja.pop(k, None)
                jb.pop(k, None)
        assert ja == jb, f
    with open(os.path.join(A, "figures", "FIGURES.md")) as fh:
        fa = [l for l in fh if l.startswith("- `")]
    with open(os.path.join(B, "figures", "FIGURES.md")) as fh:
        fb = [l for l in fh if l.startswith("- `")]
    assert fa == fb
