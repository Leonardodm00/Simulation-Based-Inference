"""Block 23 (the data overview), derived from specs/SPEC.md block 23 and sec.
2.5 (oracles O23.1-O23.10 of the t1-C report).

Pure ASCII, LF only.
"""

import ast
import glob
import hashlib
import json
import os
import re
import shutil

import numpy as np
import pytest

from conftest import EVAL, files_under, glob1, load_json, run_py, seed_of_ref, write_eval_config

DATAVIEW = os.path.join(EVAL, "ev_dataview.py")
FIGS = os.path.join(EVAL, "ev_data_figures.py")


# ------------------------------------------------------------ the helpers

def test_frame_widened_two_percent():
    import ev_dataview as V
    lo, hi = V.frame([1.0, 2.0, 3.0])
    assert abs(lo - (1.0 - 0.04)) < 1e-15 and abs(hi - (3.0 + 0.04)) < 1e-15
    lo, hi = V.frame([5.0, 5.0])                    # a constant is widened, not left empty
    assert lo < 5.0 < hi
    lo, hi = V.frame([0.0, 0.0])
    assert lo < 0.0 < hi


def test_heights_area_is_fraction_inside():
    import ev_dataview as V
    edges = np.linspace(0.0, 1.0, 11)
    vals = np.r_[np.linspace(0.05, 0.95, 50), np.full(25, -1.0), np.full(25, 2.0)]
    h = V.heights(vals, edges, vals.size)
    assert abs(np.sum(h * np.diff(edges)) - 0.5) < 1e-12


@pytest.mark.parametrize("amax,want", [(2, 0), (9, 6), (5, 3), (0, 0)])
def test_zoom_start(amax, want):
    import ev_dataview as V
    x = np.zeros(10)
    x[amax] = 1.0
    L = 4
    z0 = V.zoom_start(x, L)
    assert z0 == min(max(amax - L // 2, 0), 10 - L) == want
    assert z0 <= amax < z0 + L


def test_standardise_ddof1_and_constant_refused():
    import ev_dataview as V
    import ev_plan
    t = np.random.default_rng(1).normal(size=(30, 8))
    m, s = V.standardise(t)
    np.testing.assert_allclose(m, t.mean(axis=0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(s, t.std(axis=0, ddof=1), rtol=1e-14)
    t[:, 3] = 2.0
    with pytest.raises(ev_plan.RefuseError):
        V.standardise(t)


def test_figures_module_computes_nothing():
    tree = ast.parse(open(FIGS).read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= set(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert not (mods & {"scipy", "sklearn", "torch", "sbi", "npe_diagnostics", "ev_dataview",
                        "ev_metrics", "ev_simulate"}), mods
    banned = {"mean", "std", "var", "quantile", "percentile", "median", "nanmean", "histogram",
              "histogram2d", "fit", "transform", "fit_transform", "argmax", "argmin", "uniform",
              "normal", "choice", "integers", "random", "default_rng", "permutation", "shuffle"}
    bad = [(n.lineno, n.func.attr) for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in banned]
    assert not bad, bad


# ------------------------------------------------------- the job on the fixture

@pytest.fixture(scope="module")
def dataview(evaluation):
    ecfg = evaluation["ecfg"]
    out = load_json(ecfg)["out_root"]
    ddir = os.path.join(out, "data")
    if os.path.isdir(ddir):
        shutil.rmtree(ddir)
    rc, txt = run_py([DATAVIEW, "--config", evaluation["cfg"], "--eval-config", ecfg, "--ncpus", "1"])
    return {"rc": rc, "txt": txt, "dir": ddir, "ecfg": load_json(ecfg), "cfg": load_json(evaluation["cfg"]),
            "ecfg_path": ecfg, "cfg_path": evaluation["cfg"]}


def _bank(cfg, name):
    return os.path.join(cfg["paths"]["banks_root"], "%s_%s" % (cfg["dataset"], name))


def _read(paths, keys=("x", "theta", "donor", "well", "cls", "window_idx", "contaminated")):
    out = {}
    for p in paths:
        z = np.load(p)
        for k in keys:
            out.setdefault(k, []).append(np.asarray(z[k]))
    return dict((k, np.concatenate(v)) for k, v in out.items())


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_pass_line_and_record(dataview):
    assert dataview["rc"] == 0, dataview["txt"][-3000:]
    cfg, e = dataview["cfg"], dataview["ecfg"]
    sim_paths = sorted(glob.glob(os.path.join(_bank(cfg, "sim"), "shard_*.npz")))[:e["data_view"]["sim_shards"]]
    n_S = sum(np.load(p)["theta"].shape[0] for p in sim_paths)
    names = [r["name"] for r in cfg["synthetic"]["real"]]
    n_R = [sum(np.load(p)["theta"].shape[0] for p in sorted(glob.glob(os.path.join(_bank(cfg, "real_" + n), "shard_*.npz"))))
           for n in names]
    m = re.search(r"EVAL DATA OK: (\d+) simulated windows, (\d+) \+ (\d+) cohort windows, (\d+) figure\(s\)",
                  dataview["txt"])
    assert m, dataview["txt"][-1500:]
    assert [int(m.group(i)) for i in (1, 2, 3)] == [n_S] + n_R
    rec = load_json(os.path.join(dataview["dir"], "DATA_VIEW.json"))
    assert rec["schema"] == "eval_data_view/1"
    banks = {"sim": [[os.path.basename(p), _sha(p[:-4] + ".json")] for p in sim_paths]}
    for n in names:
        banks[n] = [[os.path.basename(p), _sha(p[:-4] + ".json")]
                    for p in sorted(glob.glob(os.path.join(_bank(cfg, "real_" + n), "shard_*.npz")))]
    params = {"seed": e["seed"], "dpi": e["dpi"], "formats": e["formats"], "data_view": e["data_view"]}
    assert rec["inputs"] == {"banks": banks, "params": params}
    # DATA_VIEW.json is written last
    newest = max(os.path.getmtime(os.path.join(dataview["dir"], f)) for f in files_under(dataview["dir"]))
    assert os.path.getmtime(os.path.join(dataview["dir"], "DATA_VIEW.json")) == newest


def test_quantities_recomputed(dataview):
    cfg, e = dataview["cfg"], dataview["ecfg"]
    a = np.load(os.path.join(dataview["dir"], "data_view.npz"))
    sim_paths = sorted(glob.glob(os.path.join(_bank(cfg, "sim"), "shard_*.npz")))[:e["data_view"]["sim_shards"]]
    S = _read(sim_paths, keys=("x", "theta", "donor"))
    T = a["sim_T"].astype(np.float64)
    assert T.shape == (S["x"].shape[0], 8)
    x = S["x"].astype(np.float64)
    # mean and max of every window by definition (float32 statistics: 1e-5 relative)
    np.testing.assert_allclose(T[:, 0], x.mean(axis=1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(T[:, 2], x.max(axis=1), rtol=1e-5, atol=1e-7)
    sd_ok = min(np.max(np.abs(T[:, 1] - x.std(axis=1, ddof=d)) / np.maximum(x.std(axis=1), 1e-12))
                for d in (0, 1))
    assert sd_ok < 1e-4
    lo, hi = np.quantile(x, 0.9, axis=1, method="lower"), np.quantile(x, 0.9, axis=1, method="higher")
    assert np.all((T[:, 3] >= lo - 1e-6) & (T[:, 3] <= hi + 1e-6))
    # the standardisation, ddof 1, float64
    m, s = T.mean(axis=0), T.std(axis=0, ddof=1)
    np.testing.assert_allclose(a["m"], m, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a["s"], s, rtol=1e-12)
    z = (T - m) / s
    # the PCA by eigendecomposition of cov(z) over S_sub alone
    w, V = np.linalg.eigh(np.cov(z.T))
    order = np.argsort(w)[::-1]
    ratios = w[order][:2] / np.sum(w)
    np.testing.assert_allclose(a["pca_evr"], ratios, rtol=1e-8)
    L = V[:, order[:2]].T
    for k in range(2):
        sgn = np.sign(np.dot(a["pca_components"][k], L[k]))
        np.testing.assert_allclose(a["pca_components"][k], sgn * L[k], atol=1e-8)
    load = a["pca_components"]
    np.testing.assert_allclose(a["sim_pc"], (z - z.mean(axis=0)) @ load.T, rtol=0, atol=1e-4)
    q = np.quantile(T, [0.01, 0.5, 0.99], axis=0)
    np.testing.assert_allclose(a["q_sim"], q, rtol=0, atol=1e-12)
    frames, pcs = [T], [(z - z.mean(axis=0)) @ load.T]
    for n in [r["name"] for r in cfg["synthetic"]["real"]]:
        Tc = a[n + "_T"].astype(np.float64)
        c = a[n + "_contam"]
        np.testing.assert_array_equal(a[n + "_below_q"], (Tc < q[0]).sum(axis=0))
        np.testing.assert_array_equal(a[n + "_above_q"], (Tc > q[-1]).sum(axis=0))
        pc = ((Tc - m) / s - z.mean(axis=0)) @ load.T
        np.testing.assert_allclose(a[n + "_pc"], pc, rtol=0, atol=1e-4)
        frames.append(Tc[~c])
        pcs.append(pc[~c])
    allT, allP = np.concatenate(frames), np.concatenate(pcs)
    for j in range(8):
        lo, hi = allT[:, j].min(), allT[:, j].max()
        w_ = hi - lo
        np.testing.assert_allclose(a["frame_T"][j], [lo - 0.02 * w_, hi + 0.02 * w_], rtol=1e-12, atol=1e-12)
    for j in range(2):
        lo, hi = allP[:, j].min(), allP[:, j].max()
        w_ = hi - lo
        np.testing.assert_allclose(a["frame_pc"][j], [lo - 0.02 * w_, hi + 0.02 * w_], rtol=1e-6, atol=1e-6)
    for n in [r["name"] for r in cfg["synthetic"]["real"]] + ["sim"]:
        h = a["hist_sim"] if n == "sim" else a[n + "_hist"]
        TT = T if n == "sim" else a[n + "_T"].astype(np.float64)
        for j in range(8):
            ed = a["edges"][j]
            assert len(ed) == e["data_view"]["bins"] + 1
            inside = np.mean((TT[:, j] >= ed[0]) & (TT[:, j] <= ed[-1]))
            assert abs(np.sum(h[j] * np.diff(ed)) - inside) < 1e-12, (n, j)


def test_theta_and_dagger(dataview):
    cfg = dataview["cfg"]
    a = np.load(os.path.join(dataview["dir"], "data_view.npz"))
    for r in cfg["synthetic"]["real"]:
        n = r["name"]
        paths = sorted(glob.glob(os.path.join(_bank(cfg, "real_" + n), "shard_*.npz")))
        B = _read(paths)
        side = load_json(paths[0][:-4] + ".json")
        ids = np.unique(B["donor"])
        np.testing.assert_array_equal(a[n + "_donors"], ids)
        th = np.array([B["theta"][B["donor"] == d][0] for d in ids], dtype=np.float64)
        for d, t in zip(ids, th):
            assert np.all(B["theta"][B["donor"] == d] == t.astype(B["theta"].dtype))
        np.testing.assert_array_equal(a[n + "_theta_d"], th)
        gap = side.get("gap_spec") or {}
        delta = (float(gap["pi"]) * float(gap["shift_max"])
                 if gap and "range_shift" in gap.get("modes", []) else 0.0)
        assert abs(float(a[n + "_delta"]) - delta) < 1e-15
        free = tuple(side["latent_spec"]["free_idx"])
        dag = th.copy()
        for k in free:
            dag[:, k] += delta
        if 7 in free:
            dag[:, 7] = np.minimum(dag[:, 7], 1.0)
        np.testing.assert_allclose(a[n + "_theta_dag_d"], dag, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(a[n + "_in_box_d"], np.all((dag > 0) & (dag < 1), axis=1))
        if r["pi"] == 0:
            np.testing.assert_array_equal(a[n + "_theta_dag_d"], th)
        np.testing.assert_array_equal(a["centres"], np.asarray(side["latent_spec"]["class_centres"]))


def test_examples_and_zooms(dataview):
    cfg, e = dataview["cfg"], dataview["ecfg"]
    a = np.load(os.path.join(dataview["dir"], "data_view.npz"))
    k = e["data_view"]["examples_per_condition"]
    L = int(round(e["data_view"]["zoom_s"] * float(a["fs"])))
    W = int(a["W"])
    for r in cfg["synthetic"]["real"]:
        n = r["name"]
        B = _read(sorted(glob.glob(os.path.join(_bank(cfg, "real_" + n), "shard_*.npz"))))
        want = []
        for c in sorted(set(B["cls"].tolist())):
            donors_c = np.unique(B["donor"][B["cls"] == c])
            rng = np.random.default_rng(seed_of_ref(e["seed"], "data_view", n, int(c)))
            for d in np.sort(rng.choice(donors_c, size=min(k, donors_c.size), replace=False)):
                rows = np.flatnonzero(B["donor"] == d)
                rows = rows[B["well"][rows] == B["well"][rows].min()]
                want.append(int(rows[np.argmin(B["window_idx"][rows])]))
        np.testing.assert_array_equal(a[n + "_ex_rows"], want)
        for rows_key, z_key in ((n + "_ex_rows", n + "_ex_z0"), (n + "_cx_rows", n + "_cx_z0")):
            for row, z0 in zip(a[rows_key], a[z_key]):
                x = B["x"][row]
                am = int(np.argmax(x))
                assert z0 == min(max(am - L // 2, 0), W - L) and z0 <= am < z0 + L
        assert np.all(B["contaminated"][a[n + "_cx_rows"]])


def test_figures_and_index(dataview):
    cfg, e = dataview["cfg"], dataview["ecfg"]
    a = np.load(os.path.join(dataview["dir"], "data_view.npz"))
    names = [r["name"] for r in cfg["synthetic"]["real"]]
    want = {"windows_cohorts", "windows_cohorts_zoom", "windows_sim", "stats_hist", "stats_strip",
            "theta_cohorts", "stats_pca"}
    if any(a[n + "_contam"].any() for n in names):
        want.add("windows_contaminated")
    if any(float(a[n + "_delta"]) > 0 for n in names):
        want.add("theta_dagger")
    fdir = os.path.join(dataview["dir"], "figures")
    got = set(os.path.splitext(f)[0] for f in files_under(fdir))
    assert got == want, (sorted(got - want), sorted(want - got))
    assert len(files_under(fdir)) == len(want) * len(e["formats"])
    idx = open(os.path.join(dataview["dir"], "FIGURES.md")).read()
    for n in want:
        assert re.search(r"`(?:figures/)?%s\.\{[^`]*\}`: [^\n]{10,}" % re.escape(n), idx), n


@pytest.mark.parametrize("over", [{"sim_shards": 99}, {"zoom_s": 1000.0}])
def test_refusals_write_nothing(dataview, tmp_path, over):
    out = str(tmp_path / "ev")
    dv = dict(dataview["ecfg"]["data_view"], **over)
    ec = write_eval_config(str(tmp_path / "e.json"), out, data_view=dv)
    rc, txt = run_py([DATAVIEW, "--config", dataview["cfg_path"], "--eval-config", ec])
    assert rc == 3 and "REFUSE" in txt, txt[-1500:]
    assert not files_under(out), files_under(out)


def test_argument_error_exit_2(dataview):
    rc, txt = run_py([DATAVIEW, "--config", dataview["cfg_path"]])
    assert rc == 2, txt[-500:]


def test_launcher_data_done_and_stale(dataview, tmp_path):
    sub = os.path.join(EVAL, "ev_submit.py")
    env = dict(os.environ, PIPELINE_SKIP_CONDA="1")
    rc, out = run_py([sub, "--config", dataview["cfg_path"], "--eval-config", dataview["ecfg_path"],
                      "--what", "data", "--dry-run"], env=env)
    assert rc == 0 and "1 done" in out and "data done" in out, out[-1500:]
    dv = dict(dataview["ecfg"]["data_view"], bins=dataview["ecfg"]["data_view"]["bins"] + 1)
    ec = write_eval_config(str(tmp_path / "e.json"), dataview["ecfg"]["out_root"], data_view=dv)
    rc, out = run_py([sub, "--config", dataview["cfg_path"], "--eval-config", ec, "--what", "data",
                      "--dry-run"], env=env)
    assert rc == 0 and "data would be submitted (stale" in out, out[-1500:]
    side = glob1(os.path.join(_bank(dataview["cfg"], "real_R0"), "shard_0000.json"))
    keep = open(side, "rb").read()
    try:
        with open(side, "wb") as fh:
            fh.write(keep + b"\n")
        rc, out = run_py([sub, "--config", dataview["cfg_path"], "--eval-config", dataview["ecfg_path"],
                          "--what", "data", "--dry-run"], env=env)
        assert rc == 0 and "data would be submitted (stale" in out, out[-1500:]
    finally:
        with open(side, "wb") as fh:
            fh.write(keep)


def test_same_arrays_twice_and_other_examples_under_another_seed(dataview, tmp_path):
    a = np.load(os.path.join(dataview["dir"], "data_view.npz"))
    runs = {}
    for seed in (dataview["ecfg"]["seed"], dataview["ecfg"]["seed"] + 1):
        out = str(tmp_path / ("ev%d" % seed))
        ec = write_eval_config(str(tmp_path / ("e%d.json" % seed)), out, seed=seed)
        rc, txt = run_py([DATAVIEW, "--config", dataview["cfg_path"], "--eval-config", ec])
        assert rc == 0, txt[-1500:]
        runs[seed] = np.load(os.path.join(out, "data", "data_view.npz"))
    same = runs[dataview["ecfg"]["seed"]]
    for k in a.files:
        np.testing.assert_array_equal(a[k], same[k], err_msg=k)
    other = runs[dataview["ecfg"]["seed"] + 1]
    np.testing.assert_array_equal(a["sim_T"], other["sim_T"])          # the data do not depend on it
    assert not np.array_equal(a["sim_ex_rows"], other["sim_ex_rows"]) or \
        not np.array_equal(a["R0_jit"], other["R0_jit"])
