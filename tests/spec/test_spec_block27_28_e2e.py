"""End-to-end spec tests of blocks 27 and 28 (and the parts of blocks 20-22
they amend) on a reduced v4-like project with real samples: S-A1 on R0,
S-A2-R3 (S-A2-only) and S-A5-R3 (S-A5-only) on R3, two members each.

The project is the tree `test_smoke_comparisons.py` builds and keeps with
CMP_KEEP=1 (its synth.json, cmp_eval.json and cmp_evaluation/samples); point
SPEC_CMP_FIXTURE at it. The samples are copied into a fresh out_root, then
the real entry points run there: `ev_metrics.py` and `ev_figures.py` (CLI,
exit codes, pass lines), and `submit_eval.sh --dry-run`. Every expected
value is recomputed here from the samples and the spec (numpy; the
library's paired_bootstrap with the arguments the spec names).

Run: SPEC_CMP_FIXTURE=<dir> python3 -m pytest -q tests/spec/test_spec_block27_28_e2e.py
Without SPEC_CMP_FIXTURE every test skips.
Pure ASCII, LF only.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _spec_util as U  # noqa: E402

FIX = os.environ.get("SPEC_CMP_FIXTURE", "").strip()
pytestmark = pytest.mark.skipif(not FIX, reason="SPEC_CMP_FIXTURE not set")

WANT = [("S-A2-R3", "vs_A1", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3", "S-A1", "A2", "A1"),
        ("S-A5-R3", "vs_A1", "S-A5-R3:A5", "S-A1:A1", "S-A5-R3", "S-A1", "A5", "A1"),
        ("S-A2-R3_vs_S-A5-R3", "A2_vs_A5", "S-A2-R3:A2", "S-A5-R3:A5", "S-A2-R3", "S-A5-R3",
         "A2", "A5")]
FIELDS = ("name", "kind", "E", "E_ref", "search", "search_ref", "arm", "arm_ref")
ARM_HEX = {"A1": "#2a78d6", "A2": "#eb6834", "A5": "#1baf7a"}
FAMILIES = ["calibration", "accuracy", "posterior", "embeddings", "comparisons", "searches",
            "across"]


def _env():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    env.pop("PIPELINE_SKIP_CONDA", None)
    return env


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A fresh out_root holding a copy of the fixture's samples; ev_metrics.py and
    ev_figures.py run there once through their CLIs."""
    root = tmp_path_factory.mktemp("e2e")
    cfg = os.path.join(FIX, "synth.json")
    src = json.load(open(os.path.join(FIX, "cmp_eval.json")))
    out = str(root / "evaluation")
    shutil.copytree(os.path.join(src["out_root"], "samples"), os.path.join(out, "samples"))
    e = dict(src, out_root=out)                 # cal_glob kept: a sample parameter
    ecfg = str(root / "eval.json")
    json.dump(e, open(ecfg, "w"), indent=1)
    m = subprocess.run([sys.executable, os.path.join(U.EVAL, "ev_metrics.py"), "--config", cfg,
                        "--eval-config", ecfg, "--ncpus", "2"], capture_output=True, text=True,
                       env=_env(), timeout=3600)
    f = subprocess.run([sys.executable, os.path.join(U.EVAL, "ev_figures.py"), "--config", cfg,
                        "--eval-config", ecfg], capture_output=True, text=True, env=_env(),
                       timeout=3600)
    return {"root": str(root), "cfg": cfg, "ecfg": ecfg, "out": out, "e": e, "m": m, "f": f,
            "mdir": os.path.join(out, "metrics"), "fdir": os.path.join(out, "figures")}


def _sample(run, stem, D):
    return np.load(os.path.join(run["out"], "samples", "%s_%s.npz" % (stem, D)))


def _stem(name):
    return name.replace(":", "_")


def test_metrics_cli_lines_and_index(run):
    m = run["m"]
    assert m.returncode == 0, m.stdout[-3000:] + m.stderr[-3000:]
    lines = m.stdout.strip().splitlines()
    assert lines[-1].startswith("EVAL METRICS OK: 3 ensemble(s), "), lines[-1]
    assert " 9 paired comparison(s)," in lines[-1], lines[-1]
    assert lines[-2] == "comparisons: 3 made, 0 left out, 0 search(es) in none", lines[-2]
    idx = json.load(open(os.path.join(run["mdir"], "METRICS.json")))
    assert [tuple(c[f] for f in FIELDS) for c in idx["comparisons"]] == WANT
    for c in idx["comparisons"]:
        # E's data sets in its provenance order, every one sampled for E_ref too
        assert c["datasets"] == ["sim", "R3", "R3dag"], c
        assert set(c["paired"]) == set(c["datasets"])
        for D, p in c["paired"].items():
            assert os.path.isabs(p) and os.path.isfile(p)
            assert os.path.basename(p) == "paired_%s_%s.json" % (c["name"], D)
    assert idx["comparisons_left_out"] == [] and idx["uncompared"] == []
    assert idx["report_inputs"]["comparisons"] == [[w[0], w[2], w[3]] for w in WANT]


@pytest.mark.parametrize("w", WANT, ids=[w[0] for w in WANT])
@pytest.mark.parametrize("D", ["sim", "R3", "R3dag"])
def test_paired_numbers_recomputed_from_the_samples(run, w, D):
    import bootstrap_paired as BP
    import ev_plan
    name, E, R = w[0], w[2], w[3]
    js = json.load(open(os.path.join(run["mdir"], "paired_%s_%s.json" % (name, D))))
    z = np.load(os.path.join(run["mdir"], "paired_%s_%s.npz" % (name, D)))
    zE, zR = _sample(run, _stem(E), D), _sample(run, _stem(R), D)
    for k in ("rows", "theta_ref", "donor"):
        npt.assert_array_equal(zE[k], zR[k])
    ok = zE["in_box"] & np.isfinite(zE["logq_ref"]) & np.isfinite(zR["logq_ref"])
    d = (-zE["logq_ref"][ok].astype(np.float64)) - (-zR["logq_ref"][ok].astype(np.float64))
    g = zE["donor"][ok]
    donors = np.unique(g)
    delta = np.array([d[g == x].mean() for x in donors])
    sizes = np.array([(g == x).sum() for x in donors])
    # every donor of these data sets carries the same number of paired rows, so E3's mean
    # over donors and the mean over rows coincide here (see the unbalanced case elsewhere)
    assert np.all(sizes == sizes[0]), sizes
    # 1e-12 nats: float64 sums of at most a few hundred O(1) terms in another order
    npt.assert_allclose(js["Dbar"], delta.mean(), rtol=0, atol=1e-12)
    npt.assert_allclose(z["delta_d"], delta, rtol=0, atol=1e-12)
    npt.assert_array_equal(z["donor_ids"], donors)

    def msd(a):
        o = a["in_box"] & np.isfinite(a["logq_ref"])
        return float(np.std((-a["logq_ref_members"][o].astype(np.float64)).mean(axis=0), ddof=1))
    sig = max(msd(zE), msd(zR))
    npt.assert_allclose(js["sigma_seed"], sig, rtol=0, atol=1e-12)
    seed = ev_plan.seed_of(json.load(open(run["ecfg"]))["seed"], "paired", name, D)
    res = BP.paired_bootstrap(d, groups=g, within="all", n_boot=run["e"]["bootstrap_n"],
                              level=run["e"]["level"], seed=seed)
    npt.assert_allclose(js["ci"], res.ci, rtol=0, atol=1e-12)
    assert js["bootstrap_seed"] == seed
    beyond, excl = abs(delta.mean()) > 2 * sig, (res.ci[0] > 0 or res.ci[1] < 0)
    assert (js["beyond_2_sigma_seed"], js["ci_excludes_0"], js["resolved"]) == (
        beyond, excl, beyond and excl)
    assert js["n_rows"] == int(ok.sum()) and js["n_donors"] == donors.size
    if D == "R3dag":
        # theta-dagger outside the open box: those rows are out of the comparison
        assert (~zE["in_box"]).sum() > 0 and js["n_rows"] < zE["in_box"].size


def test_metrics_deterministic_and_restricted_by_ensemble(run, tmp_path):
    out2 = str(tmp_path / "evaluation")
    shutil.copytree(os.path.join(run["out"], "samples"), os.path.join(out2, "samples"))
    e2 = dict(run["e"], out_root=out2)
    ecfg2 = str(tmp_path / "eval.json")
    json.dump(e2, open(ecfg2, "w"))
    r = subprocess.run([sys.executable, os.path.join(U.EVAL, "ev_metrics.py"), "--config",
                        run["cfg"], "--eval-config", ecfg2, "--ensemble", "S-A2-R3:A2",
                        "--ensemble", "S-A5-R3:A5", "--ncpus", "1"],
                       capture_output=True, text=True, env=_env(), timeout=3600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = r.stdout.strip().splitlines()
    for c in ("S-A2-R3", "S-A5-R3"):
        assert ("  comparison %s left out: S-A1:A1 was not requested (--ensemble)" % c) in lines
    assert lines[-2] == "comparisons: 1 made, 2 left out, 0 search(es) in none"
    idx = json.load(open(os.path.join(out2, "metrics", "METRICS.json")))
    assert [c["name"] for c in idx["comparisons"]] == ["S-A2-R3_vs_S-A5-R3"]
    # the same samples, the same seeds: the A2_vs_A5 files equal the full run's
    for D in ("sim", "R3", "R3dag"):
        a = json.load(open(os.path.join(run["mdir"], "paired_S-A2-R3_vs_S-A5-R3_%s.json" % D)))
        b = json.load(open(os.path.join(out2, "metrics", "paired_S-A2-R3_vs_S-A5-R3_%s.json" % D)))
        assert a == b


def _expected_figures(run):
    """The figure set of blocks 21 and 28's rules, from METRICS.json and the
    per-ensemble metrics files (stated here from the spec)."""
    md = run["mdir"]
    idx = json.load(open(os.path.join(md, "METRICS.json")))

    def js(n):
        p = os.path.join(md, n + ".json")
        return json.load(open(p)) if os.path.isfile(p) else None
    want = set()
    for name, info in idx["ensembles"].items():
        stem, dss = info["stem"], info["datasets"]
        b = "ensembles/%s/" % stem
        want |= {b + "cal_coverage", b + "cal_tarp", b + "acc_sigma", b + "acc_nll",
                 b + "post_lc2st"}
        for D in dss:
            want |= {b + "cal_sbc_ecdf_" + D, b + "cal_sbc_hist_" + D, b + "acc_recovery_" + D,
                     b + "acc_zscore_" + D}
            if not D.endswith("dag"):
                want.add(b + "post_ppc_" + D)
            if D != "sim" and not D.endswith("dag"):
                want.add(b + "emb_pca_" + D)
        ex = (js("search_%s" % info["search"]) or {}).get("example_donors") or {}
        for coh, v in ex.items():
            if coh in dss:
                for x in v["donors"]:
                    want.add(b + "post_corner_%s_c%d" % (coh, x["cls"]))
    for c in idx.get("comparisons") or []:
        b = "comparisons/%s/" % c["name"]
        want.add(b + "cmp_coverage")
        eE, eR = idx["ensembles"][c["E"]], idx["ensembles"][c["E_ref"]]
        for D in c["datasets"]:
            want.add(b + "cmp_forest_" + D)
            if D != "sim":
                want.add(b + "cmp_nll_" + D)
            if D != "sim" and not D.endswith("dag"):
                for key in ("ppc", "lc2st"):
                    if (js("%s_%s" % (eE["stem"], D)) or {}).get(key) and \
                            (js("%s_%s" % (eR["stem"], D)) or {}).get(key):
                        want.add(b + "cmp_%s_%s" % (key, D))
    for q in idx["searches"]:
        if js("search_%s" % q) is not None:
            want |= {"searches/%s/%s" % (q, n) for n in ("search_trials", "search_nll_vs_lpr",
                                                         "search_learning")}
    if idx.get("comparisons"):
        want.add("across/dbar")
    want.add("across/lpr")
    return want


def test_figures_set_index_and_captions(run):
    f = run["f"]
    assert f.returncode == 0, f.stdout[-3000:] + f.stderr[-3000:]
    want = _expected_figures(run)
    last = f.stdout.strip().splitlines()[-1]
    assert re.match(r"^EVAL FIGURES OK: %d figure\(s\) in %s \(\d+ s\)$" % (len(want),
                    re.escape(run["fdir"])), last), last
    e = run["e"]
    got = set()
    for dp, dn, fn in os.walk(run["fdir"]):
        for x in fn:
            if x == "FIGURES.md":
                continue
            rel, ext = os.path.splitext(os.path.relpath(os.path.join(dp, x), run["fdir"]))
            assert ext[1:] in e["formats"], x
            got.add(rel)
    assert got == want, ("missing", sorted(want - got)[:8], "extra", sorted(got - want)[:8])
    assert not [g for g in got if "/arms_" in g]
    # no comparison figure on R0, which S-A1:A1 has and no comparison pairs
    assert not [g for g in got if g.startswith("comparisons/") and g.endswith("_R0")]
    for g in got:
        for fmt in e["formats"]:
            assert os.path.getsize(os.path.join(run["fdir"], g + "." + fmt)) > 1000
    md = open(os.path.join(run["fdir"], "FIGURES.md")).read().splitlines()
    heads = [l[3:] for l in md if l.startswith("## ")]
    assert heads == [h for h in FAMILIES if h in heads] and "comparisons" in heads
    header = [l for l in md if l.startswith("Written by")]
    assert len(header) == 1 and "Comparisons left out: none." in header[0], header
    assert "Uncompared searches: none." in header[0], header
    listed = {}
    fam = None
    for l in md:
        if l.startswith("## "):
            fam = l[3:]
        m = re.match(r"^- `([^`]+)\.\{([a-z,]+)\}`: (.*)$", l)
        if m:
            assert m.group(1) not in listed, m.group(1)
            listed[m.group(1)] = (fam, m.group(3))
    assert set(listed) == want
    for rel, (fam, cap) in listed.items():
        assert len(cap) >= 30 and cap.endswith(".") and not cap.endswith(".."), (rel, cap)
        if rel.startswith("comparisons/"):
            assert fam == "comparisons", rel
            name = rel.split("/")[1]
            c = [w for w in WANT if w[0] == name][0]
            assert c[2] in cap and c[3] in cap, (rel, cap)          # both ensembles named
            D = rel.rsplit("_", 1)[1] if "_" in rel.split("/")[2][4:] else None
            if D:
                assert (" on %s" % D) in cap or (" %s" % D) in cap, (rel, cap)


def test_comparison_figures_colours_legends_and_dbar_panels(run):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import to_hex
    import ev_config
    import ev_figures as FG
    ecfg = ev_config.load(run["ecfg"])
    st = FG.Store(ecfg)
    idx = st.index
    rows = []
    for c in idx["comparisons"]:
        eE = dict(idx["ensembles"][c["E"]], name=c["E"])
        eR = dict(idx["ensembles"][c["E_ref"]], name=c["E_ref"])
        fig, cap = FG.fig_cmp_coverage(st, c, eE, eR, c["datasets"])
        ax = fig.axes[0]
        lab = dict((l.get_label(), to_hex(l.get_color())) for l in ax.get_lines()
                   if not l.get_label().startswith("_"))
        assert lab == {"%s (%s)" % (c["arm"], c["search"]): ARM_HEX[c["arm"]],
                       "%s (%s)" % (c["arm_ref"], c["search_ref"]): ARM_HEX[c["arm_ref"]]}, lab
        for D in c["datasets"]:
            rows.append(("%s, %s" % (c["name"], D),
                         json.load(open(c["paired"][D])), c))
    fig, cap = FG.fig_dbar_all(st, rows)
    axes = [a for a in fig.axes if a.get_visible()]
    assert [a.get_title() for a in axes] == ["against A1", "A2 against A5"]
    want_rows = {"against A1": [r[0] for r in rows if r[2]["kind"] == "vs_A1"],
                 "A2 against A5": [r[0] for r in rows if r[2]["kind"] == "A2_vs_A5"]}
    for a in axes:
        labels = [t.get_text() for t in a.get_yticklabels()]
        assert sorted(labels) == sorted(want_rows[a.get_title()]), labels


def test_launcher_submits_the_report_under_the_fake_pbs(tmp_path):
    """Block 22 as block 27 amends it, through its real entry point: on the v4-like
    project with its samples and calibration bank done, `submit_eval.sh` (no dry
    run) under the tests' fake PBS submits the report alone, lists the three
    comparisons after its unit, and the report job ends with REPORT_COMPLETE.json
    recording ev_plan.report_inputs, comparisons included; run again, done."""
    import time
    import ev_config
    import ev_plan
    src = json.load(open(os.path.join(FIX, "cmp_eval.json")))
    out = str(tmp_path / "evaluation")
    shutil.copytree(os.path.join(src["out_root"], "samples"), os.path.join(out, "samples"))
    ecfg = str(tmp_path / "eval.json")
    json.dump(dict(src, out_root=out), open(ecfg, "w"), indent=1)
    cfg = os.path.join(FIX, "synth.json")
    spool = str(tmp_path / "spool")
    os.makedirs(spool)
    env = _env()
    env.update({"PATH": os.path.join(U.TESTS, "fake_pbs") + os.pathsep + env["PATH"],
                "FAKE_PBS_SPOOL": spool, "FAKE_PBS_DELAY": "0.2", "FAKE_PBS_ENFORCE_WALLTIME": "0",
                "PIPELINE_QSTAT_RETRY_WAIT_S": "0.2", "ROADMAP_DISABLE": "1", "SKIP_CONDA": "1"})
    sub = [ "bash", os.path.join(U.EVAL, "submit_eval.sh"), "--config", cfg, "--eval-config", ecfg]
    r = subprocess.run(sub, capture_output=True, text=True, env=env, timeout=600)
    out1 = r.stdout.strip().splitlines()
    assert r.returncode == 0, out1[-20:] + [r.stderr[-2000:]]
    assert out1[-1] == "EVAL SUBMIT OK: 1 job(s) submitted, 5 done, 0 live", out1[-1]
    i = [k for k, l in enumerate(out1) if l.startswith("EVAL SUBMIT: report submitted ")]
    assert len(i) == 1, out1
    want = ["EVAL SUBMIT: comparison %s: %s against %s (%s)" % (w[0], w[2], w[3], w[1]) for w in WANT]
    assert out1[i[0] + 1:i[0] + 4] == want, out1
    led = json.load(open(os.path.join(out, "jobs", "ledger.json")))
    assert sorted(led) == ["report"], led
    rc_path = os.path.join(out, "REPORT_COMPLETE.json")
    t0 = time.time()
    while time.time() - t0 < 1800 and not os.path.isfile(rc_path):
        time.sleep(3)
    assert os.path.isfile(rc_path), open(led["report"]["log"]).read()[-3000:]
    time.sleep(2)
    log = open(led["report"]["log"]).read()
    assert "EVAL REPORT OK: 3 ensemble(s)" in log and log.rstrip().endswith("EVAL JOB EXIT: 0"), \
        log[-2000:]
    rec = json.load(open(rc_path))
    c, e = ev_plan.load_project(cfg), ev_config.load(ecfg)
    assert rec["inputs"] == ev_plan.report_inputs(c, e)
    assert rec["inputs"]["comparisons"] == [[w[0], w[2], w[3]] for w in WANT]
    r = subprocess.run(sub, capture_output=True, text=True, env=env, timeout=600)
    out2 = r.stdout.strip().splitlines()
    assert r.returncode == 0 and out2[-1] == "EVAL SUBMIT OK: 0 job(s) submitted, 6 done, 0 live", \
        out2[-5:]
    j = [k for k, l in enumerate(out2) if l.startswith("EVAL SUBMIT: report done")]
    assert len(j) == 1 and out2[j[0] + 1:j[0] + 4] == want, out2


EVAL_FIX = os.environ.get("SPEC_EVAL_FIXTURE", "").strip()
BASE = "57e294ded7f3c9efca772f4c5945bb89d8475d85"


@pytest.mark.skipif(not EVAL_FIX, reason="SPEC_EVAL_FIXTURE not set")
def test_v3_design_keeps_block20s_names_seeds_and_numbers(tmp_path):
    """Block 27: "A project of v3's design ...: its comparisons are block 20's pairs,
    under the same names and seeds, with the same numbers." The v3-like project of
    test_smoke_evaluation.py (EVAL_KEEP=1; S-A1, and S-A2-R0 with both arms): block
    20's ev_metrics.py of the base commit (git archive) and this one, on copies of the
    same samples; every value of block 20's keys and every per-ensemble file equal."""
    base = tmp_path / "base_tree"
    base.mkdir()
    a = subprocess.run("git -C %s archive %s hpc | tar -x -C %s" % (U.REPO, BASE, base), shell=True)
    assert a.returncode == 0
    src = json.load(open(os.path.join(EVAL_FIX, "synth_eval.json")))
    cfg = os.path.join(EVAL_FIX, "synth.json")
    outs = {}
    for v, prog in (("base", str(base / "hpc/pipeline/evaluation/ev_metrics.py")),
                    ("new", os.path.join(U.EVAL, "ev_metrics.py"))):
        out = str(tmp_path / v / "evaluation")
        shutil.copytree(os.path.join(src["out_root"], "samples"), os.path.join(out, "samples"))
        ecfg = str(tmp_path / v / "eval.json")
        json.dump(dict(src, out_root=out), open(ecfg, "w"))
        r = subprocess.run([sys.executable, prog, "--config", cfg, "--eval-config", ecfg,
                            "--ncpus", "1"], capture_output=True, text=True, env=_env(), timeout=3600)
        assert r.returncode == 0, (v, r.stdout[-2000:], r.stderr[-2000:])
        outs[v] = os.path.join(out, "metrics")
    fb, fn = sorted(os.listdir(outs["base"])), sorted(os.listdir(outs["new"]))
    assert fb == fn
    paired = [f for f in fb if f.startswith("paired_") and f.endswith(".json")]
    assert paired == ["paired_S-A2-R0_R0.json", "paired_S-A2-R0_sim.json"]
    for f in fb:
        A, B = os.path.join(outs["base"], f), os.path.join(outs["new"], f)
        if f.endswith(".npz"):
            za, zb = np.load(A), np.load(B)
            assert set(za.files) == set(zb.files), f
            for k in za.files:
                npt.assert_array_equal(za[k], zb[k], err_msg="%s %s" % (f, k))
        elif f == "METRICS.json":
            ja, jb = json.load(open(A)), json.load(open(B))
            for k in ("ensembles", "left_out", "searches"):
                assert ja[k] == jb[k], k
        else:
            ja, jb = json.load(open(A)), json.load(open(B))
            assert set(ja) <= set(jb), (f, set(ja) - set(jb))
            for k in ja:
                if k != "sample":
                    assert ja[k] == jb[k], (f, k, ja[k], jb[k])
