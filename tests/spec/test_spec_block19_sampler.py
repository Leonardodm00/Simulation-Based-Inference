"""Block 19 (the evaluation's sampler), derived from specs/SPEC.md sec. 2.5
and block 19 (oracles O19.1-O19.11 of the t1-C report), with block 25's
amendment of pseudo_true (free_idx).

Pure ASCII, LF only.
"""

import ast
import glob
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import (EVAL, HPC, PIPE, PY, REPO, build_calbank, glob1, load_json, run_py,
                      seed_of_ref, write_eval_config)

NPZ_CONTRACT = {          # key: (dtype kind, number of dims) from SPEC block 19
    "theta_ref": ("float64", 2), "in_box": ("bool", 1), "rows": ("int64", 1),
    "donor": ("int64", 1), "well": ("int64", 1), "cls": ("int64", 1),
    "logq_ref_members": ("float64", 2), "logq_ref": ("float64", 1),
    "draws": ("float32", 3), "draw_member": ("int64", 1), "logq_draws": ("float32", 2),
    "logq_draws_own": ("float32", 2), "z": ("float32", 3), "stats": ("float32", 2),
    "calib": ("bool", 1), "ppc_rows": ("int64", 1), "ppc_draw_idx": ("int64", 1),
    "ppc_stats": ("float32", 3)}
GAP = {"modes": ["range_shift", "drift", "contamination"], "pi": 0.3, "shift_max": 0.35,
       "drift_amp_max": 0.5, "drift_period_s": 120.0, "contam_frac_max": 0.2}


# ------------------------------------------------------- O19.3 theta-dagger

def _dagger_ref(theta, delta, free):
    """sec. 2.5 D_R3+: + delta on the free axes, the duty (axis 7) capped at
    (1 - L_7) / (U_7 - L_7) = 1 when it is free (block 25)."""
    t = np.array(theta, dtype=np.float64, copy=True)
    for k in free:
        t[:, k] = t[:, k] + delta
    if 7 in free:
        t[:, 7] = np.minimum(t[:, 7], 1.0)
    return t, np.all((t > 0.0) & (t < 1.0), axis=1)


@pytest.mark.parametrize("free", [(7, 8, 9), (8, 9)])
def test_pseudo_true_formula(free):
    import ev_data
    rng = np.random.default_rng(19)
    th = rng.uniform(1e-6, 1 - 1e-6, size=(2000, 10))
    th[:5, 7] = [0.894, 0.895, 0.896, 0.9999, 1e-3]
    dag, in_box, delta = ev_data.pseudo_true(th, GAP, free)
    assert abs(delta - 0.35 * 0.3) <= 1e-15
    want, want_in = _dagger_ref(th, 0.105, free)
    np.testing.assert_allclose(dag, want, rtol=0, atol=1e-15)    # float64 roundoff of one sum
    np.testing.assert_array_equal(in_box, want_in)
    assert dag.dtype == np.float64 and in_box.dtype == bool
    assert np.array_equal(th[:5, 7], [0.894, 0.895, 0.896, 0.9999, 1e-3])     # input untouched


def test_pseudo_true_no_shift():
    import ev_data
    th = np.random.default_rng(5).uniform(0.01, 0.99, size=(50, 10))
    for gap in (dict(GAP, pi=0.0), dict(GAP, modes=["drift", "contamination"]), None, {}):
        dag, in_box, delta = ev_data.pseudo_true(th, gap, (7, 8, 9))
        assert delta == 0.0
        np.testing.assert_array_equal(dag, th)
        assert in_box.all()


def test_pseudo_true_physics_identity():
    """The unshifted affine map at theta-dagger gives the physics the shifted
    map gives at theta* (sec. 2.5), the duty's clip included; float64
    roundoff of the affine map."""
    import ev_data
    from bench_burst_provider import phi_to_bench_params
    rng = np.random.default_rng(31)
    th = rng.uniform(0.02, 0.98, size=(60, 10))
    th[0, 7], th[1, 7] = 0.95, 0.80
    for free in ((7, 8, 9), (8, 9)):
        dag, _in, delta = ev_data.pseudo_true(th, GAP, free)
        for i in range(th.shape[0]):
            a = phi_to_bench_params(dag[i], 100, 12.8, free_axis_shift=0.0, free_idx=free)
            b = phi_to_bench_params(th[i], 100, 12.8, free_axis_shift=delta, free_idx=free)
            va, vb = vars(a), vars(b)
            for k in va:
                if isinstance(va[k], float):
                    assert abs(va[k] - vb[k]) <= 1e-12 * max(1.0, abs(vb[k])), (free, i, k)


def test_free_axes_of():
    import ev_data
    assert tuple(ev_data.free_axes_of({"latent_spec": {"free_idx": [8, 9]}})) == (8, 9)
    assert tuple(ev_data.free_axes_of({"latent_spec": {}})) == (7, 8, 9)


def test_calibration_rows_first_of_each_donor():
    import ev_data
    donor = np.array([5, 5, 3, 7, 3, 5, 9])
    np.testing.assert_array_equal(ev_data.calibration_rows(donor),
                                  [True, False, True, True, False, False, True])


# ------------------------------------------------------ O19.1 import boundary

STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
PIPELINE_OWN = {"config", "stages", "pbs", "state", "envs", "pipeline", "_evalpath",
                "ev_config", "ev_plan"}


def _imports(path):
    tree = ast.parse(open(path).read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= set(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize("mod", ["ev_config.py", "ev_plan.py", "ev_submit.py"])
def test_login_node_modules_import_stdlib_and_pipeline_only(mod):
    bad = sorted(m for m in _imports(os.path.join(EVAL, mod)) if m not in STDLIB | PIPELINE_OWN)
    assert not bad, (mod, bad)


def test_sequencer_imports_no_evaluation_module():
    paths = glob.glob(os.path.join(PIPE, "*.py")) + glob.glob(os.path.join(PIPE, "stages", "*.py"))
    for p in paths:
        ev = sorted(m for m in _imports(p) if m.startswith("ev_") or m == "evaluation")
        assert not ev, (p, ev)


def test_login_node_modules_python_version():
    """ev_config, ev_plan and ev_submit run on the login node's python (block
    22: Python 3.3 or later by vermin)."""
    if not shutil.which("vermin"):
        pytest.skip("vermin is not installed")
    files = [os.path.join(EVAL, m) for m in ("ev_config.py", "ev_plan.py", "ev_submit.py",
                                             "_evalpath.py")]
    r = subprocess.run(["vermin", "-t=3.6-", "--no-tips", "--violations"] + files,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout[-2000:]


# --------------------------------------------------------- O19.7 the config

def _ecfg_errors(**over):
    import ev_config
    vals = json.load(open(os.path.join(PIPE, "projects", "synth", "synth_eval.json")))
    vals.update(over)
    return ev_config.validate(dict((k, v) for k, v in vals.items() if not k.startswith("_")))


def test_committed_eval_config_values():
    import ev_config
    e = ev_config.load(os.path.join(PIPE, "projects", "synth", "synth_eval.json"))
    assert (e.n_post, e.ppc_rows, e.ppc_draws, e.cal_n, e.lc2st_null) == (1000, 324, 20, 10000, 100)
    assert e.level == 0.95 and e.dpi == 200 and tuple(e.formats) == ("png", "pdf")
    assert e.oracle_tol == 2e-3
    assert "oracle_tol" not in ev_config.SAMPLE_PARAMS
    j = e.jobs
    assert (j["calbank"]["ncpus"], j["calbank"]["mem"], j["calbank"]["walltime"]) == (1, "16gb", "04:00:00")
    assert (j["sample"]["ncpus"], j["sample"]["mem"], j["sample"]["walltime"]) == (8, "64gb", "24:00:00")
    assert (j["report"]["ncpus"], j["report"]["mem"], j["report"]["walltime"]) == (8, "64gb", "24:00:00")
    assert (j["data"]["ncpus"], j["data"]["mem"], j["data"]["walltime"]) == (2, "8gb", "01:00:00")
    assert e.data_view["sim_shards"] == 2


@pytest.mark.parametrize("tol,ok", [(2e-3, True), (0.1, True), (1e-12, True), (0.0, False),
                                    (0.1000001, False), (-1e-3, False), (float("nan"), False),
                                    (float("inf"), False)])
def test_oracle_tol_range(tol, ok):
    errs = [e for e in _ecfg_errors(oracle_tol=tol) if "oracle_tol" in e]
    assert (not errs) == ok, errs


# ---------------------------------------------------- the sampler's outputs

def _npz(evaluation, stem, D):
    return np.load(os.path.join(evaluation["out"], "samples", "%s_%s.npz" % (stem, D)))


def _ensembles(evaluation):
    import ev_plan
    return ev_plan.ensembles(ev_plan.load_project(evaluation["cfg"]))


def _shards(path_glob):
    out = {}
    for p in sorted(glob.glob(path_glob)):
        z = np.load(p)
        for k in ("theta", "donor", "well", "cls", "x"):
            out.setdefault(k, []).append(np.asarray(z[k]))
    return dict((k, np.concatenate(v)) for k, v in out.items())


def test_cli_pass_lines(evaluation):
    for n in evaluation["names"]:
        rc, txt = evaluation["logs"]["sample:" + n]
        stem = n.replace(":", "_")
        assert rc == 0 and ("EVAL SAMPLE OK: %s" % stem) in txt, txt[-3000:]


def test_datasets_per_ensemble(evaluation):
    """D_sim and the cohorts (S-A1: every cohort; else its own), and
    <cohort>dag for a cohort with a range shift at pi > 0 (R3 only)."""
    cfg = load_json(evaluation["cfg"])
    shifted = [r["name"] for r in cfg["synthetic"]["real"]
               if r["pi"] > 0 and "range_shift" in r["gap_modes"]]
    for e in _ensembles(evaluation):
        q = e.search
        coh = [r["name"] for r in cfg["synthetic"]["real"]] if q["campaign"] == "S-A1" else [q["real"]]
        want = ["sim"] + coh + [c + "dag" for c in coh if c in shifted]
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % e.stem))
        got = [os.path.basename(f)[len(e.stem) + 1:-4] for f in prov["files"]
               if not f.endswith("_cal.npz")]
        assert sorted(got) == sorted(want), (e.name, got, want)
        assert tuple(e.cohorts) == tuple(coh)


def test_npz_contract(evaluation):
    ecfg = load_json(evaluation["ecfg"])
    for e in _ensembles(evaluation):
        M = len(e.records)
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % e.stem))
        for f in prov["files"]:
            z = np.load(f)
            if f.endswith("_cal.npz"):
                n = ecfg["cal_n"]
                assert z["theta"].shape == (n, 10) and z["stats"].shape == (n, 8)
                assert z["draws"].shape == (n, 10) and z["member"].shape == (n,)
                assert np.all((z["draws"] > 0) & (z["draws"] < 1))
                assert set(np.unique(z["member"])) <= set(range(M))
                continue
            assert sorted(z.files) == sorted(NPZ_CONTRACT), (f, sorted(z.files))
            n = z["theta_ref"].shape[0]
            for k, (dt, nd) in NPZ_CONTRACT.items():
                assert z[k].dtype == np.dtype(dt) and z[k].ndim == nd, (f, k, z[k].dtype, z[k].shape)
            P, k_ppc = ecfg["n_post"], z["ppc_rows"].shape[0]
            assert z["theta_ref"].shape == (n, 10) and z["logq_ref_members"].shape == (n, M)
            assert z["draws"].shape == (n, P, 10) and z["draw_member"].shape == (P,)
            assert z["logq_draws"].shape == (n, P) == z["logq_draws_own"].shape
            assert z["z"].shape[:2] == (M, n) and z["stats"].shape == (n, 8)
            assert z["ppc_draw_idx"].shape == (ecfg["ppc_draws"],)
            assert z["ppc_stats"].shape == (k_ppc, ecfg["ppc_draws"], 8)


def test_mixture_and_box(evaluation):
    """(E1) log q_E = logsumexp_m log q_m - log M in float64; -inf exactly
    outside the open box; draws inside the open box, n_post / M per member in
    member order; the mixture at a draw >= its own member's density / M."""
    from scipy.special import logsumexp
    for e in _ensembles(evaluation):
        M = len(e.records)
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % e.stem))
        for f in prov["files"]:
            if f.endswith("_cal.npz"):
                continue
            z = np.load(f)
            ib = z["in_box"]
            np.testing.assert_array_equal(ib, np.all((z["theta_ref"] > 0) & (z["theta_ref"] < 1),
                                                     axis=1))
            assert np.all(np.isneginf(z["logq_ref_members"][~ib]))
            assert np.all(np.isneginf(z["logq_ref"][~ib]))
            assert np.all(np.isfinite(z["logq_ref_members"][ib]))
            want = logsumexp(z["logq_ref_members"][ib], axis=1) - math.log(M)
            np.testing.assert_allclose(z["logq_ref"][ib], want, rtol=0, atol=1e-12)
            d = z["draws"]
            assert np.all((d > 0) & (d < 1)), f
            P = d.shape[1]
            np.testing.assert_array_equal(z["draw_member"], np.repeat(np.arange(M), P // M))
            # mixture >= own / M; float32 storage of both: 1e-5 nats on values of order 10
            slack = z["logq_draws"].astype(np.float64) - (z["logq_draws_own"].astype(np.float64)
                                                         - math.log(M))
            assert slack.min() >= -1e-5, (f, slack.min())


def test_rebuild_oracle_against_perrow(evaluation):
    """Each member's -log q at theta^ref equals its _perrow.npz nll on D_sim
    and nll_pseudo_real on its search's cohort, within oracle_tol (2e-3)."""
    for e in _ensembles(evaluation):
        sim = _npz(evaluation, e.stem, "sim")
        coh = _npz(evaluation, e.stem, e.search["real"])
        for k, rec in enumerate(e.records):
            pr = np.load(rec[:-len(".json")] + "_perrow.npz")
            assert pr["nll"].shape == sim["logq_ref"].shape
            assert np.max(np.abs(pr["nll"] + sim["logq_ref_members"][:, k])) <= 2e-3
            assert "nll_pseudo_real" in pr.files, rec
            assert np.max(np.abs(pr["nll_pseudo_real"] + coh["logq_ref_members"][:, k])) <= 2e-3


def test_rows_and_references(evaluation):
    """D_sim: the report rows of the members' split, in global row order,
    theta^ref the bank's theta, calib the first row of each donor, the PPC
    rows the first ppc_rows calibration rows; a cohort: every window in bank
    order; D_<r>dag: theta-dagger of the cohort's sidecar, D_<r>'s draws."""
    import run_joint_arms as R
    cfg = load_json(evaluation["cfg"])
    ecfg = load_json(evaluation["ecfg"])
    banks = cfg["paths"]["banks_root"]
    sim = _shards(os.path.join(banks, "synth_sim", "shard_*.npz"))
    for e in _ensembles(evaluation):
        rec = load_json(e.records[0])
        split, h = R.grouped_split(sim["donor"], fracs=tuple(rec["split_fracs"]),
                                   seed=int(rec["split_seed"]))
        assert h == rec["split_hash"]
        rows = np.flatnonzero(split == 2)
        z = _npz(evaluation, e.stem, "sim")
        np.testing.assert_array_equal(z["rows"], rows)
        np.testing.assert_array_equal(z["theta_ref"], sim["theta"][rows].astype(np.float64))
        np.testing.assert_array_equal(z["donor"], sim["donor"][rows])
        first = np.zeros(rows.size, bool)
        seen = set()
        for i, d in enumerate(z["donor"]):
            if d not in seen:
                first[i] = True
                seen.add(d)
        np.testing.assert_array_equal(z["calib"], first)
        np.testing.assert_array_equal(z["ppc_rows"], np.flatnonzero(first)[:ecfg["ppc_rows"]])
        P, J = ecfg["n_post"], ecfg["ppc_draws"]
        np.testing.assert_array_equal(z["ppc_draw_idx"], np.arange(J) * P // J)
        for c in e.cohorts:
            cb = _shards(os.path.join(banks, "synth_real_%s" % c, "shard_*.npz"))
            zc = _npz(evaluation, e.stem, c)
            n = cb["theta"].shape[0]
            np.testing.assert_array_equal(zc["rows"], np.arange(n))
            np.testing.assert_array_equal(zc["theta_ref"], cb["theta"].astype(np.float64))
            np.testing.assert_array_equal(zc["donor"], cb["donor"])
            np.testing.assert_array_equal(zc["cls"], cb["cls"])
            assert zc["calib"].all()
            np.testing.assert_array_equal(zc["ppc_rows"], np.arange(n))
            side = load_json(glob1(os.path.join(banks, "synth_real_%s" % c, "shard_0000.json")))
            gap = side.get("gap_spec") or {}
            delta = float(gap.get("pi", 0.0)) * float(gap.get("shift_max", 0.0)) \
                if "range_shift" in (gap.get("modes") or []) else 0.0
            dag_path = os.path.join(evaluation["out"], "samples", "%s_%sdag.npz" % (e.stem, c))
            if delta > 0:
                zd = np.load(dag_path)
                want, want_in = _dagger_ref(cb["theta"].astype(np.float64), delta,
                                            tuple(side["latent_spec"]["free_idx"]))
                np.testing.assert_allclose(zd["theta_ref"], want, rtol=0, atol=1e-15)
                np.testing.assert_array_equal(zd["in_box"], want_in)
                np.testing.assert_array_equal(zd["draws"], zc["draws"])
                np.testing.assert_array_equal(zd["logq_draws"], zc["logq_draws"])
                assert zd["ppc_rows"].size == 0
            else:
                assert not os.path.exists(dag_path), dag_path


def test_seeds_common_to_ensembles(evaluation):
    ecfg = load_json(evaluation["ecfg"])
    seeds = {}
    for e in _ensembles(evaluation):
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % e.stem))
        for D, s in prov["seeds"].items():
            seeds.setdefault(D, set()).add(int(s))
            assert int(s) == seed_of_ref(ecfg["seed"], D), (e.name, D)
    assert all(len(v) == 1 for v in seeds.values()), seeds


def test_mixture_against_sbi_log_prob(evaluation):
    """The stored mixture density at the draws, recomputed through sbi's own
    estimator.log_prob for every member (the members rebuilt and checked by
    the oracle above), on S-A2-R0's cohort; float32 network outputs: 1e-4."""
    import torch
    import ev_data
    import ev_models
    import ev_plan
    from scipy.special import logsumexp
    cfg = ev_plan.load_project(evaluation["cfg"])
    e = [x for x in _ensembles(evaluation) if x.kind == "best_fit"
         and x.search["name"] == "S-A2-R0"][0]
    index, _p = ev_data.sim_index(cfg)
    z = _npz(evaluation, e.stem, "R0")
    coh = ev_data.cohort(cfg, "R0")
    rows = np.arange(min(6, z["draws"].shape[0]))
    vals = []
    for rec in e.records:
        model, _info = ev_models.rebuild(rec, index)
        est = model.estimator
        with torch.no_grad():
            per = []
            for i in rows:
                th = torch.as_tensor(z["draws"][i], dtype=torch.float32)          # (P, 10)
                x = torch.as_tensor(coh["x"][i:i + 1])                             # (1, W)
                lp = est.log_prob(th.unsqueeze(1), x)                              # (P, 1)
                per.append(lp.squeeze(1).numpy().astype(np.float64))
        vals.append(np.stack(per))
    mix = logsumexp(np.stack(vals), axis=0) - math.log(len(e.records))
    np.testing.assert_allclose(z["logq_draws"][rows].astype(np.float64), mix, rtol=0, atol=1e-4)


def test_same_seed_same_files(evaluation, tmp_path):
    """Two runs at one seed give identical arrays (E19d)."""
    e = [x for x in _ensembles(evaluation) if x.kind == "best_fit_alt"][0]
    out = str(tmp_path / "again")
    rc, txt = run_py([os.path.join(EVAL, "ev_sample.py"), "--config", evaluation["cfg"],
                      "--eval-config", evaluation["ecfg"], "--ensemble", e.name, "--out", out,
                      "--ncpus", "1"])
    assert rc == 0, txt[-2000:]
    for f in sorted(glob.glob(os.path.join(out, "*.npz"))):
        a = np.load(f)
        b = np.load(os.path.join(evaluation["out"], "samples", os.path.basename(f)))
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k], err_msg="%s %s" % (f, k))


# ------------------------------------------------------------- refusals

def _run_sampler(cfg, ecfg, name, out):
    return run_py([os.path.join(EVAL, "ev_sample.py"), "--config", cfg, "--eval-config", ecfg,
                   "--ensemble", name, "--out", out, "--ncpus", "1"])


def test_argument_errors_exit_2(evaluation):
    rc, txt = run_py([os.path.join(EVAL, "ev_sample.py"), "--config", evaluation["cfg"]])
    assert rc == 2, txt[-500:]


@pytest.mark.parametrize("over,name", [({"n_post": 41}, "S-A1:A1"),
                                       ({"oracle_tol": 1e-6}, "S-A1:A1"),
                                       ({}, "S-A1:A5")])
def test_refusals_write_nothing(evaluation, tmp_path, over, name):
    ec = write_eval_config(str(tmp_path / "e.json"),
                           load_json(evaluation["ecfg"])["out_root"], **over)
    out = str(tmp_path / "out")
    rc, txt = _run_sampler(evaluation["cfg"], ec, name, out)
    assert rc == 3 and "REFUSE" in txt, txt[-1500:]
    assert not [f for f in (os.listdir(out) if os.path.isdir(out) else [])], os.listdir(out)


def _calbank_at_seed(evaluation, tmp_path, seed):
    out_root = str(tmp_path / ("ev_seed%d" % seed))
    ec = write_eval_config(str(tmp_path / ("e_seed%d.json" % seed)), out_root, cal_seed=int(seed))
    build_calbank(evaluation["cfg"], ec)
    return ec, out_root


def test_calibration_bank_at_the_simulated_banks_seed_refused(evaluation, tmp_path):
    cfg = load_json(evaluation["cfg"])
    ec, out_root = _calbank_at_seed(evaluation, tmp_path, cfg["synthetic"]["sim"]["seed"])
    out = os.path.join(out_root, "samples")
    rc, txt = _run_sampler(evaluation["cfg"], ec, "S-A2-R0:A1", out)
    assert rc == 3 and "REFUSE" in txt, txt[-1500:]
    assert not (os.path.isdir(out) and os.listdir(out))


def test_calibration_bank_at_another_cohorts_seed_refused(evaluation, tmp_path):
    """SPEC block 19: the bank is checked before any sampling -- 'a builder
    seed no bank of the project has'. R3's seed, for an ensemble scored on R0
    alone (S-A2-R0)."""
    cfg = load_json(evaluation["cfg"])
    r3 = [r for r in cfg["synthetic"]["real"] if r["name"] == "R3"][0]
    ec, out_root = _calbank_at_seed(evaluation, tmp_path, r3["seed"])
    out = os.path.join(out_root, "samples")
    rc, txt = _run_sampler(evaluation["cfg"], ec, "S-A2-R0:A1", out)
    assert rc == 3 and "REFUSE" in txt, (rc, txt[-1500:])


def _copy_project(src_root, dst_root):
    for d in ("banks_root", "runs_root", "tune_root"):
        shutil.copytree(os.path.join(src_root, d), os.path.join(dst_root, d))
    cfg = load_json(os.path.join(src_root, "synth.json"))
    for k in ("banks_root", "runs_root", "tune_root", "work_root"):
        cfg["paths"][k] = os.path.join(dst_root, k)
    p = os.path.join(dst_root, "synth.json")
    with open(p, "w") as fh:
        json.dump(cfg, fh, indent=1)
    return p


def test_cohort_oracle_refusal_writes_nothing(evaluation, tmp_path):
    """A member whose cohort NLL is not reproduced: the job refuses (exit 3)
    and, as every refusal of E19d, writes nothing."""
    root = str(tmp_path / "proj")
    os.makedirs(root)
    cfgp = _copy_project(evaluation["root"], root)
    pr = glob1(os.path.join(root, "runs_root", "S-A2-R0", "synth_best", "member_seed1",
                            "*_seed1_perrow.npz"))
    z = dict(np.load(pr))
    z["nll_pseudo_real"] = z["nll_pseudo_real"] + 0.05      # 25 x oracle_tol
    np.savez(pr[:-len(".npz")], **z)
    out = str(tmp_path / "samples")
    rc, txt = _run_sampler(cfgp, evaluation["ecfg"], "S-A2-R0:A2", out)
    assert rc == 3 and "REFUSE" in txt and "R0" in txt, txt[-1500:]
    left = os.listdir(out) if os.path.isdir(out) else []
    assert not left, left


def test_outputs_finite(evaluation):
    """Every stored quantity finite, -inf only where the spec puts it (log q
    at a theta^ref outside the open box)."""
    for e in _ensembles(evaluation):
        prov = load_json(os.path.join(evaluation["out"], "samples", "%s_sample.json" % e.stem))
        for f in prov["files"]:
            z = np.load(f)
            for k in z.files:
                a = z[k]
                if a.dtype.kind != "f":
                    continue
                if k in ("logq_ref", "logq_ref_members"):
                    ib = z["in_box"] if a.ndim == 1 else z["in_box"][:, None] & np.ones_like(a, bool)
                    assert np.all(np.isfinite(a[ib])) and np.all(np.isneginf(a[~ib])), (f, k)
                else:
                    assert np.all(np.isfinite(a)), (f, k)


def test_ppc_windows_are_arm_s_windows(evaluation):
    """windows_at simulates a window of arm S at a given theta (block 19,
    Q-E4): at the thetas of the simulated bank's own windows its windows'
    eight statistics follow the bank windows' law (two-sample KS per
    statistic, Holm at 0.001; fixed seed), and one process or two give the
    same windows."""
    from scipy.stats import ks_2samp
    import ev_simulate
    cfg = load_json(evaluation["cfg"])
    paths = sorted(glob.glob(os.path.join(cfg["paths"]["banks_root"], "synth_sim", "shard_*.npz")))
    B = _shards(os.path.join(cfg["paths"]["banks_root"], "synth_sim", "shard_*.npz"))
    side = load_json(paths[0][:-4] + ".json")
    th = B["theta"].astype(np.float64)
    x1 = ev_simulate.windows_at(th, side["latent_spec"], 1234, ncpus=1)
    x2 = ev_simulate.windows_at(th, side["latent_spec"], 1234, ncpus=2)
    np.testing.assert_array_equal(x1, x2)
    assert x1.shape == B["x"].shape and x1.dtype == np.float32
    s_sim = ev_simulate.fixed_stats(x1).astype(np.float64)
    s_bank = ev_simulate.fixed_stats(B["x"]).astype(np.float64)
    p = sorted(ks_2samp(s_sim[:, j], s_bank[:, j]).pvalue for j in range(8))
    holm = [min(1.0, (8 - i) * pv) for i, pv in enumerate(p)]
    assert min(holm) > 1e-3, p
