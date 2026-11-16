#!/usr/bin/env python3
"""Smoke test for block 27 of specs/SPEC.md: the evaluation's comparisons
across searches (sec. 2.5, "The comparisons"; D-132 (5), D-136; the roadmap
synth, step S6; wave W2).

Checks (specs/SPEC.md block 27, "Test oracles")
    test_e27a_plan       ev_plan.comparisons on configurations alone (no
                         training environment): the committed synth.json v4's
                         six comparisons in order, with their names, kinds,
                         ensembles, searches and arms; v3's committed layout:
                         block 20's within-search pairs under their names,
                         then A2 against A5 per cohort; the evaluation
                         fixture's v3-like configuration: S-A2-R0 against
                         S-A2-R0:A1 and no A2_vs_A5; term-only searches with
                         no S-A1 search, and with two, refused naming the
                         search and the S-A1 searches (a nesting search with
                         two is not); S-A25 among the uncompared; an A2 and an
                         A5 search on different cohorts, no A2_vs_A5; the
                         order of (b) over nesting and term-only searches;
                         no file read; no numpy in ev_plan, Python 3.6 syntax
                         (vermin, where installed)
    test_e27b_metrics    paired_all on known answers (block 19's arrays
                         written by hand: d constant or symmetric per donor,
                         so Dbar, the interval, sigma_seed and the verdicts
                         are known; E's data-set order; the out-of-box rows;
                         a reference not complete; no data set shared; rows,
                         donors or theta^ref differing: refused naming both
                         ensembles and the data set); then a reduced v4-like
                         project -- S-A1 on R0, S-A2-only and S-A5-only on
                         R3, two members each, run to PIPELINE DONE under the
                         fake PBS, its calibration bank and samplings made
                         here -- through ev_metrics.compute: the nine paired
                         files of the three comparisons on sim, R3 and R3dag,
                         their keys per kind, Dbar and sigma_seed recomputed
                         from the samples within 1e-12, the interval equal to
                         paired_bootstrap's with seed_of(seed, "paired",
                         name, D); METRICS.json's comparisons,
                         comparisons_left_out and uncompared; the log lines,
                         the pass line last; S-A1's sample provenance
                         removed: the two comparisons against A1 left out and
                         named, A2 against A5 made, and the same whys when
                         --ensemble leaves S-A1:A1 out (contract amendment 1:
                         an ensemble without a current sample is named by its
                         sample's state, named by --ensemble or not);
                         --ensemble on a copy of the configuration with a
                         search not complete and an S-A25 search: "was not
                         requested" (a current sample left out), "not
                         complete", the search in none; one sample's rows
                         altered: the CLI refuses (exit 3) naming both
                         ensembles and R3
    test_e27c_launcher   report_inputs carries the comparisons; a
                         REPORT_COMPLETE.json recording inputs without them
                         makes ev_submit.plan's report unit stale; the dry
                         run with report lists the three comparisons right
                         after the report's unit (an S-A25 search's line
                         too), without report none; no S-A1 search: --what
                         report refused (exit 3) with nothing written or
                         submitted, --what calbank,sample not; on the fixture
                         (training environment): the report would be
                         submitted, stale without the comparisons, done with
                         them, the comparisons after it each time
    test_convergence     not a discretisation: skipped

Run
    python3 hpc/pipeline/tests/test_smoke_comparisons.py [check ...]
        test_e27a_plan needs no training environment and takes a second;
        test_e27b_metrics and test_e27c_launcher begin with checks that need
        none either, then build once the reduced project (about 2 minutes in
        the sandbox) and sample its three ensembles. CMP_FIXTURE=<dir> reuses a
        tree this test built before (its synth.json, banks and runs; the
        evaluation's files under <dir>/cmp_evaluation), CMP_KEEP=1 keeps the
        tree it builds and prints where. Without the training environment
        (torch, sbi, zuko, scikit-optimize) the fixture's parts skip. Nothing
        touches davinci.

Pure ASCII, LF only.
"""

import ast
import builtins
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_smoke_synthetic as TS      # noqa: E402  (the reduced project; puts the pipeline first)
import test_smoke_evaluation as TE     # noqa: E402  (the evaluation's configuration, calibration bank)

PYTHON = sys.executable
SUBMIT_PY = os.path.join(_EVAL, "ev_submit.py")
METRICS_PY = os.path.join(_EVAL, "ev_metrics.py")

# the reduced v4-like project: S-A1 on R0 (scored on R0 and R3, D-122) and one search per
# term arm on R3, of block 26's term-only campaigns
SEARCHES = [{"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
            {"name": "S-A2-R3", "campaign": "S-A2-only", "real": "R3", "alt_best_fit": False},
            {"name": "S-A5-R3", "campaign": "S-A5-only", "real": "R3", "alt_best_fit": False}]
# arm S and the cohorts of test_smoke_evaluation's fixture: R3's seed 34 leaves 3 of its 6
# donors with theta-dagger outside the box, so R3dag's paired rows are a strict subset
REAL = [{"name": "R0", "seed": 22, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
         "gap_modes": ["range_shift", "drift", "contamination"]},
        {"name": "R3", "seed": 34, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
         "gap_modes": ["range_shift", "drift", "contamination"]}]
SIM = {"seed": 11, "n_theta": 60, "n_windows": 1, "traces_per_shard": 40}
ENSEMBLES = ("S-A1:A1", "S-A2-R3:A2", "S-A5-R3:A5")
# sec. 2.5's comparisons of SEARCHES, stated here independently of ev_plan:
# (name, kind, E, E_ref, search, search_ref, arm, arm_ref)
WANT = [("S-A2-R3", "vs_A1", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3", "S-A1", "A2", "A1"),
        ("S-A5-R3", "vs_A1", "S-A5-R3:A5", "S-A1:A1", "S-A5-R3", "S-A1", "A5", "A1"),
        ("S-A2-R3_vs_S-A5-R3", "A2_vs_A5", "S-A2-R3:A2", "S-A5-R3:A5", "S-A2-R3", "S-A5-R3",
         "A2", "A5")]
FIELDS = ("name", "kind", "E", "E_ref", "search", "search_ref", "arm", "arm_ref")
DATASETS = ["sim", "R3", "R3dag"]      # E's data sets (sim, R3, theta-dagger), all of S-A1:A1's too
# the paired JSON's keys (SPEC block 27): block 20's, every comparison's, kind vs_A1's legacy ones
KEYS_BLOCK20 = {"schema", "dataset", "n_rows", "n_donors", "Dbar", "ci", "se", "level", "n_boot",
                "scheme", "sigma_seed", "two_sigma_seed", "beyond_2_sigma_seed", "ci_excludes_0",
                "resolved"}
KEYS_EVERY = {"comparison", "kind", "E", "E_ref", "search", "search_ref", "arm", "arm_ref",
              "sd_members_E", "sd_members_ref", "lower_nll_ensemble", "bootstrap_seed"}
KEYS_LEGACY = {"term_arm", "ensembles", "sd_members", "lower_nll"}
WHY_S_A25 = "campaign S-A25: its arms are not defined against A1 (D-136)"
PAIRED_LINE = re.compile(r"^  paired (\S+) on (\S+): Dbar [+-]\d+\.\d{4} \[[+-]\d+\.\d{4}, "
                         r"[+-]\d+\.\d{4}\], 2 sigma_seed \d+\.\d{4} -> (resolved|not resolved)$")

_FIX = {}


# --------------------------------------------------------------- helpers ---

def _q(name, campaign, real, alt=False):
    return {"name": name, "campaign": campaign, "real": real, "alt_best_fit": alt}


def _stem(name):
    """`<q>_<arm>` of an ensemble `<q>:<arm>` (block 19's stem)."""
    return name.replace(":", "_")


def _config(root, searches):
    """A reduced synthetic configuration of `searches` in folder `root`
    (TS._small: the bench at W 128, arm S and the cohorts above, the paths
    under `root`, sbi_dir this checkout); nothing is run. Its path."""
    os.makedirs(root, exist_ok=True)
    path, _d = TS._small(root, searches=[dict(q) for q in searches],
                         **{"synthetic.sim": dict(SIM), "synthetic.real": [dict(r) for r in REAL]})
    return path


def _launch(cfg, ecfg, *args, **kw):
    """The launcher (ev_submit.py) in a process; (exit code, output)."""
    e = dict(kw.get("env") or os.environ)
    e.update({"PIPELINE_SKIP_CONDA": "1", "ROADMAP_DISABLE": "1"})
    r = subprocess.run([PYTHON, SUBMIT_PY, "--config", cfg, "--eval-config", ecfg] + list(args),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       env=e, timeout=300)
    return r.returncode, r.stdout


def _build_project(root):
    """The reduced v4-like project in `root`, run to PIPELINE DONE under the
    fake PBS (as test_smoke_evaluation's fixture is). Its synth.json."""
    path = _config(root, SEARCHES)
    saved = dict(os.environ)
    try:
        env = TS._fake_env(os.path.join(root, "spool"))
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                            "--poll-seconds", "0.5"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=3600)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert r.returncode == 0 and "PIPELINE DONE: every stage done or skipped" in r.stdout, \
        r.stdout[-6000:]
    return path


def _current(ecfg, stem):
    """The sample of `stem` exists, every file of it, with the evaluation's values."""
    pj = os.path.join(ecfg.sub("samples"), "%s_sample.json" % stem)
    if not os.path.isfile(pj):
        return False
    with open(pj) as fh:
        prov = json.load(fh)
    return (prov.get("sample_params") == ecfg.sample_params()
            and all(os.path.isfile(f) for f in prov.get("files") or [None]))


def _fixture():
    """dict(root, cfg, ecfg): the reduced v4-like project, its calibration
    bank and its three ensembles' samples, built once per process."""
    if "ready" in _FIX:
        return _FIX
    if "failed" in _FIX:
        raise AssertionError("the fixture could not be built: %s" % _FIX["failed"])
    TE._need_training_env()
    try:
        import ev_config
        import ev_sample
        given = os.environ.get("CMP_FIXTURE", "").strip()
        if given:
            root = os.path.abspath(given)
            cfg = os.path.join(root, "synth.json")
            assert os.path.isfile(cfg), "CMP_FIXTURE=%s has no synth.json" % root
            with open(cfg) as fh:
                got = [q["name"] for q in json.load(fh)["synthetic"]["searches"]]
            assert got == [q["name"] for q in SEARCHES], "CMP_FIXTURE=%s holds %s" % (root, got)
        else:
            root = tempfile.mkdtemp(prefix="cmp_fixture_")
            _FIX.update(root=root, built=True)
            t0 = time.time()
            cfg = _build_project(root)
            print("[fixture] the reduced v4-like project ran to PIPELINE DONE in %.0f s: %s"
                  % (time.time() - t0, root), flush=True)
        ecfg = TE._eval_config(root, "cmp_eval.json", out=os.path.join(root, "cmp_evaluation"))
        TE._build_calbank(cfg, ecfg)
        ec = ev_config.load(ecfg)
        for name in ENSEMBLES:
            if not _current(ec, _stem(name)):
                lines = []
                ev_sample.run(cfg, ecfg, name, log=lines.append)
                assert any(l.startswith("EVAL SAMPLE OK: ") for l in lines), "\n".join(lines[-20:])
        _FIX.update(root=root, cfg=cfg, ecfg=ecfg, ready=True)
        return _FIX
    except Exception as exc:
        _FIX["failed"] = ("%s: %s" % (type(exc).__name__, exc))[:3000]
        raise


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _sd_members(z):
    """The SD over members of the members' mean NLL on the sample's own
    in-box rows with a finite log density (sec. 2.5, sigma_seed's terms)."""
    import numpy as np
    ok = z["in_box"] & np.isfinite(z["logq_ref"])
    return float(np.std(-np.mean(np.asarray(z["logq_ref_members"], dtype=np.float64)[ok], axis=0),
                        ddof=1))


def _check_paired(js, arr, w, D, zE, zR, ecfg, BP, seed_of):
    """One paired file (its JSON and .npz) against the contract and against
    E3 recomputed from the two samples; returns (Dbar, d) for the caller."""
    import numpy as np
    name, kind, E, E_ref, search, search_ref, arm, arm_ref = w
    want_keys = KEYS_BLOCK20 | KEYS_EVERY | (KEYS_LEGACY if kind == "vs_A1" else set())
    assert set(js) == want_keys, (name, D, sorted(set(js) ^ want_keys))
    named = dict((f, js[f if f != "name" else "comparison"]) for f in FIELDS)
    assert named == dict(zip(FIELDS, w)), js
    assert js["schema"] == "eval_metrics/1" and js["dataset"] == D and js["scheme"] == "all", js
    assert js["level"] == ecfg.level and js["n_boot"] == ecfg.bootstrap_n, js
    for k in ("rows", "theta_ref", "donor"):
        assert np.array_equal(zE[k], zR[k]), (name, D, k)
    ok = zE["in_box"] & np.isfinite(zE["logq_ref"]) & np.isfinite(zR["logq_ref"])
    lE = -np.asarray(zE["logq_ref"], dtype=np.float64)
    lR = -np.asarray(zR["logq_ref"], dtype=np.float64)
    d = lE[ok] - lR[ok]
    g = np.asarray(zE["donor"])[ok]
    # Dbar and sigma_seed from the samples: float64 sums of at most a few hundred rows, so 1e-12
    assert abs(js["Dbar"] - float(np.mean(d))) < 1e-12, (name, D, js["Dbar"], float(np.mean(d)))
    assert js["n_rows"] == int(ok.sum()) and js["n_donors"] == int(np.unique(g).size), js
    sE, sR = _sd_members(zE), _sd_members(zR)
    assert abs(js["sd_members_E"] - sE) < 1e-12 and abs(js["sd_members_ref"] - sR) < 1e-12, js
    assert abs(js["sigma_seed"] - max(sE, sR)) < 1e-12, (js["sigma_seed"], sE, sR)
    assert js["two_sigma_seed"] == 2.0 * js["sigma_seed"], js
    # the interval: paired_bootstrap with the comparison's own seed, so equal to the last bit
    seed = seed_of(ecfg.seed, "paired", name, D)
    res = BP.paired_bootstrap(d, groups=g, within="all", n_boot=ecfg.bootstrap_n, level=ecfg.level,
                              seed=seed)
    assert js["bootstrap_seed"] == seed, (name, D, js["bootstrap_seed"], seed)
    assert js["ci"] == [float(res.ci[0]), float(res.ci[1])] and js["se"] == float(res.se), \
        (name, D, js["ci"], res.ci)
    beyond = abs(js["Dbar"]) > js["two_sigma_seed"]
    excl = js["ci"][0] > 0.0 or js["ci"][1] < 0.0
    assert (js["beyond_2_sigma_seed"], js["ci_excludes_0"], js["resolved"]) == (
        beyond, excl, beyond and excl), js
    lower = (E if js["Dbar"] < 0 else E_ref) if (beyond and excl) else None
    assert js["lower_nll_ensemble"] == lower, (js["lower_nll_ensemble"], lower)
    if kind == "vs_A1":
        assert js["term_arm"] == arm and js["ensembles"] == {"term": E, "A1": E_ref}, js
        assert js["sd_members"] == {arm: js["sd_members_E"], "A1": js["sd_members_ref"]}, js
        assert js["lower_nll"] == ({E: "term arm", E_ref: "A1"}[lower] if lower else None), js
    # the .npz: per donor the mean of d over its paired rows (block 20's arrays)
    donors = np.unique(g)
    assert sorted(arr.files) == ["delta_d", "donor_ids", "se_d"], arr.files
    assert arr["donor_ids"].dtype == np.int64 and np.array_equal(arr["donor_ids"], donors)
    assert arr["delta_d"].dtype == np.float64 and arr["se_d"].dtype == np.float64
    assert arr["se_d"].shape == donors.shape
    assert np.allclose(arr["delta_d"], [d[g == x].mean() for x in donors], rtol=0.0, atol=1e-12)
    return js["Dbar"], d


# ---------------------------------------------------------------- checks ---

def test_e27a_plan():
    import ev_plan
    Cmp = ev_plan.Comparison
    assert Cmp._fields == FIELDS, Cmp._fields
    # the committed synth.json v4: sec. 2.5's six comparisons, in order
    v4 = ev_plan.load_project(TS.SYNTH)
    want = [Cmp("S-A2-R0", "vs_A1", "S-A2-R0:A2", "S-A1:A1", "S-A2-R0", "S-A1", "A2", "A1"),
            Cmp("S-A5-R0", "vs_A1", "S-A5-R0:A5", "S-A1:A1", "S-A5-R0", "S-A1", "A5", "A1"),
            Cmp("S-A2-R3", "vs_A1", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3", "S-A1", "A2", "A1"),
            Cmp("S-A5-R3", "vs_A1", "S-A5-R3:A5", "S-A1:A1", "S-A5-R3", "S-A1", "A5", "A1"),
            Cmp("S-A2-R0_vs_S-A5-R0", "A2_vs_A5", "S-A2-R0:A2", "S-A5-R0:A5", "S-A2-R0", "S-A5-R0",
                "A2", "A5"),
            Cmp("S-A2-R3_vs_S-A5-R3", "A2_vs_A5", "S-A2-R3:A2", "S-A5-R3:A5", "S-A2-R3", "S-A5-R3",
                "A2", "A5")]
    got = ev_plan.comparisons(v4)
    assert isinstance(got, list) and got == want, got
    # nothing is read: open() refused while the comparisons are listed
    real_open = builtins.open

    def no_open(*a, **k):
        raise AssertionError("ev_plan.comparisons opened %r" % (a[:1],))
    builtins.open = no_open
    try:
        got = ev_plan.comparisons(v4, with_uncompared=True)
    finally:
        builtins.open = real_open
    assert got == (want, []), got
    tmp = tempfile.mkdtemp(prefix="e27a_")
    try:
        def cfg(tag, searches):
            return ev_plan.load_project(_config(os.path.join(tmp, tag), searches))

        def refused(c, needles):
            for flag in (False, True):
                try:
                    ev_plan.comparisons(c, with_uncompared=flag)
                except ev_plan.RefuseError as exc:
                    assert all(n in str(exc) for n in needles), (needles, str(exc))
                else:
                    raise AssertionError("not refused: %s" % [q["name"] for q in
                                                              c.data["synthetic"]["searches"]])
        # v3's committed layout (S-A2 and S-A5 nest A1, alt_best_fit): block 20's pairs under
        # their names, each against its own A1 ensemble; then A2 against A5 on each cohort
        root = os.path.join(tmp, "v3")
        os.makedirs(root)
        v3 = ev_plan.load_project(TS._v3_like(root))
        got = ev_plan.comparisons(v3, with_uncompared=True)
        assert got == ([Cmp(q, "vs_A1", "%s:%s" % (q, a), "%s:A1" % q, q, q, a, "A1")
                        for q, a in (("S-A2-R0", "A2"), ("S-A5-R0", "A5"), ("S-A2-R3", "A2"),
                                     ("S-A5-R3", "A5"))]
                       + [Cmp("S-A2-R%s_vs_S-A5-R%s" % (r, r), "A2_vs_A5", "S-A2-R%s:A2" % r,
                              "S-A5-R%s:A5" % r, "S-A2-R%s" % r, "S-A5-R%s" % r, "A2", "A5")
                          for r in "03"], []), got
        # the evaluation fixture's v3-like configuration: the within-search pair, no A2_vs_A5
        ev3 = cfg("ev3", [_q("S-A1", "S-A1", "R0"), _q("S-A2-R0", "S-A2", "R0", True)])
        assert ev_plan.comparisons(ev3, with_uncompared=True) == (
            [Cmp("S-A2-R0", "vs_A1", "S-A2-R0:A2", "S-A2-R0:A1", "S-A2-R0", "S-A2-R0", "A2", "A1")],
            []), ev_plan.comparisons(ev3)
        # term-only searches: no S-A1 search, two S-A1 searches -- refused, naming both
        refused(cfg("none", [_q("S-A2-R3", "S-A2-only", "R3"), _q("S-A5-R3", "S-A5-only", "R3")]),
                ["search S-A2-R3 ", "campaign S-A1", "the configuration has none"])
        refused(cfg("two", [_q("S-A1", "S-A1", "R0"), _q("S-A1-R3", "S-A1", "R3"),
                            _q("S-A5-R3", "S-A5-only", "R3")]),
                ["search S-A5-R3 ", "the configuration has 2 (S-A1, S-A1-R3)"])
        # a search that nests A1 needs no S-A1 search, and two do not trouble it
        own = [Cmp("S-A2-R3", "vs_A1", "S-A2-R3:A2", "S-A2-R3:A1", "S-A2-R3", "S-A2-R3", "A2", "A1")]
        assert ev_plan.comparisons(cfg("nest0", [_q("S-A2-R3", "S-A2", "R3", True)])) == own
        nest = cfg("nest", [_q("S-A1", "S-A1", "R0"), _q("S-A1-R3", "S-A1", "R3"),
                            _q("S-A2-R3", "S-A2", "R3", True)])
        assert ev_plan.comparisons(nest) == own
        # S-A25: in none, with its why; S-A1 alone: nothing to compare
        s25 = cfg("s25", [_q("S-A1", "S-A1", "R0"), _q("S-A25-R3", "S-A25", "R3"),
                          _q("S-A2-R3", "S-A2-only", "R3")])
        one = [Cmp("S-A2-R3", "vs_A1", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3", "S-A1", "A2", "A1")]
        assert ev_plan.comparisons(s25, with_uncompared=True) == (one, [("S-A25-R3", WHY_S_A25)])
        assert ev_plan.comparisons(s25) == one
        assert ev_plan.comparisons(cfg("a1", [_q("S-A1", "S-A1", "R0")]),
                                   with_uncompared=True) == ([], [])
        # an A2 and an A5 search on different cohorts: no A2_vs_A5
        dif = ev_plan.comparisons(cfg("dif", [_q("S-A1", "S-A1", "R0"),
                                              _q("S-A2-R0", "S-A2-only", "R0"),
                                              _q("S-A5-R3", "S-A5-only", "R3")]))
        assert [(c.name, c.kind) for c in dif] == [("S-A2-R0", "vs_A1"), ("S-A5-R3", "vs_A1")], dif
        # (b) over nesting and term-only campaigns alike, in the order of q, then of q'
        mix = ev_plan.comparisons(cfg("mix", [
            _q("S-A1", "S-A1", "R0"), _q("A2a", "S-A2", "R0", True), _q("A5a", "S-A5-only", "R0"),
            _q("A2b", "S-A2-only", "R0"), _q("A5b", "S-A5", "R0", True)]))
        assert [(c.name, c.E, c.E_ref) for c in mix] == [
            ("A2a", "A2a:A2", "A2a:A1"), ("A5a", "A5a:A5", "S-A1:A1"), ("A2b", "A2b:A2", "S-A1:A1"),
            ("A5b", "A5b:A5", "A5b:A1"), ("A2a_vs_A5a", "A2a:A2", "A5a:A5"),
            ("A2a_vs_A5b", "A2a:A2", "A5b:A5"), ("A2b_vs_A5a", "A2b:A2", "A5a:A5"),
            ("A2b_vs_A5b", "A2b:A2", "A5b:A5")], mix
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the launcher's module: the standard library and the pipeline's own only, Python 3.6
    tree = ast.parse(open(os.path.join(_EVAL, "ev_plan.py")).read())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= set(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    banned = {"numpy", "scipy", "torch", "sbi", "sklearn", "matplotlib", "ev_metrics"}
    assert not (mods & banned), mods & banned
    vermin = shutil.which("vermin")
    if vermin:
        r = subprocess.run([vermin, "-t=3.6-", "--no-tips", "--violations",
                            os.path.join(_EVAL, "ev_plan.py"), os.path.join(_EVAL, "ev_submit.py")],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120)
        assert r.returncode == 0, r.stdout[-2000:]
    print("[e27a] v4: six comparisons in order; v3: block 20's pairs, then A2 against A5; the "
          "refusals, S-A25, different cohorts, the order of (b); no file read%s"
          % ("; Python 3.6 (vermin)" if vermin else "; vermin not installed: 3.6 syntax not checked"))


def _known_set(n_donors, l_rows, l_members, out_donor=None):
    """One data set's arrays as block 19 writes them (the ones paired_all
    reads): donors 0..n_donors-1 of two rows each, theta^ref 0.5 except the
    rows of `out_donor`, outside the box (log densities -inf, as the sampler
    gives); l_rows (n,) the ensemble's NLL, l_members (n, M) its members'."""
    import numpy as np
    n = 2 * n_donors
    donor = np.repeat(np.arange(n_donors, dtype=np.int64), 2)
    theta = np.full((n, 10), 0.5)
    in_box = np.ones(n, dtype=bool)
    lq = -np.asarray(l_rows, dtype=np.float64)
    lqm = -np.asarray(l_members, dtype=np.float64)
    if out_donor is not None:
        out = donor == out_donor
        theta[out, 9] = 1.5
        in_box[out] = False
        lq[out] = -np.inf
        lqm[out] = -np.inf
    return {"rows": np.arange(100, 100 + n, dtype=np.int64), "theta_ref": theta, "donor": donor,
            "in_box": in_box, "logq_ref": lq, "logq_ref_members": lqm}


def _write_sample(ecfg, stem, sets):
    """A provenance and the .npz of `sets` [(D, arrays)], in that order, as
    block 19's sampler leaves them under <out_root>/samples/."""
    import numpy as np
    sdir = ecfg.sub("samples")
    os.makedirs(sdir, exist_ok=True)
    files = []
    for D, arrays in sets:
        p = os.path.join(sdir, "%s_%s.npz" % (stem, D))
        np.savez_compressed(p, **arrays)
        files.append(p)
    files.append(os.path.join(sdir, "%s_cal.npz" % stem))
    with open(os.path.join(sdir, "%s_sample.json" % stem), "w") as fh:
        json.dump({"schema": "eval_sample/1", "stem": stem, "files": files,
                   "sample_params": ecfg.sample_params()}, fh)


def _known_answers(tmp):
    """paired_all on samples written by hand, whose E3 is known (no training
    environment): d constant on R3 (resolved, the lower ensemble known) and
    symmetric per donor on sim (Dbar 0, not resolved)."""
    import numpy as np
    import bootstrap_paired as BP
    import ev_config
    import ev_metrics as EM
    import ev_plan
    cfg = ev_plan.load_project(_config(os.path.join(tmp, "known"), SEARCHES))
    ecfg = ev_config.load(TE._eval_config(os.path.join(tmp, "known"), "eval.json",
                                          out=os.path.join(tmp, "known", "ev")))
    # per ensemble: its NLL on R3 (6 donors, donor 5 outside the box) and its members' spread
    l_r3 = {"S-A2-R3:A2": (1.0, 0.01), "S-A1:A1": (2.0, 0.02), "S-A5-R3:A5": (0.5, 0.005)}
    sign = np.repeat(np.where(np.arange(8) % 2 == 0, 1.0, -1.0), 2)      # sim: +-1 per donor

    def sets(name, order):
        lv, sp = l_r3[name]
        r3 = _known_set(6, np.full(12, lv), lv + np.outer(np.ones(12), [-sp, sp]), out_donor=5)
        l_sim = 1.0 + (0.3 * sign if name == "S-A2-R3:A2" else 0.0 * sign)
        sim = _known_set(8, l_sim, l_sim[:, None] + np.array([[-sp, sp]]))
        r0 = _known_set(4, np.full(8, 3.0), np.full((8, 2), 3.0))
        have = {"sim": sim, "R3": r3, "R0": r0}
        return [(D, have[D]) for D in order]
    order = {"S-A2-R3:A2": ["R3", "sim"],            # E's order rules, so S-A2-R3's pairs: R3, sim
             "S-A1:A1": ["sim", "R0", "R3"], "S-A5-R3:A5": ["sim", "R3"]}
    for name in ENSEMBLES:
        _write_sample(ecfg, _stem(name), sets(name, order[name]))
    rep = [ev_plan.Ensemble(name=n, stem=_stem(n), search={"name": n.split(":")[0]},
                            arm=n.split(":")[1], folder="", kind="best_fit", seeds=(1, 2),
                            records=(), cohorts=()) for n in ENSEMBLES]
    md = os.path.join(tmp, "known", "metrics")
    os.makedirs(md)
    lines = []
    files, made, left = EM.paired_all(cfg, ecfg, rep, md, lines.append)
    shared = {"S-A2-R3": ["R3", "sim"], "S-A5-R3": ["sim", "R3"], "S-A2-R3_vs_S-A5-R3": ["R3", "sim"]}
    assert left == [] and made == [
        dict(zip(FIELDS, w), datasets=shared[w[0]],
             paired=dict((D, os.path.join(md, "paired_%s_%s.json" % (w[0], D))) for D in shared[w[0]]))
        for w in WANT], made
    assert files == [os.path.join(md, "paired_%s_%s%s" % (w[0], D, x)) for w in WANT
                     for D in shared[w[0]] for x in (".json", ".npz")], files
    assert [PAIRED_LINE.match(l).group(1, 2) for l in lines] == [
        (w[0], D) for w in WANT for D in shared[w[0]]], lines
    # the known answers: on R3 d = l(E) - l(E_ref) on donors 0-4 (donor 5 outside the box)
    known = {"S-A2-R3": (-1.0, 0.02, "S-A2-R3:A2"), "S-A5-R3": (-1.5, 0.02, "S-A5-R3:A5"),
             "S-A2-R3_vs_S-A5-R3": (0.5, 0.01, "S-A5-R3:A5")}
    by = dict((e.name, e) for e in rep)
    for w in WANT:
        dbar, sp, lower = known[w[0]]
        for D in shared[w[0]]:
            base = os.path.join(md, "paired_%s_%s" % (w[0], D))
            js = _load(base + ".json")
            zE = dict(np.load(os.path.join(ecfg.sub("samples"), "%s_%s.npz" % (by[w[2]].stem, D))))
            zR = dict(np.load(os.path.join(ecfg.sub("samples"), "%s_%s.npz" % (by[w[3]].stem, D))))
            with np.load(base + ".npz") as arr:
                _check_paired(js, arr, w, D, zE, zR, ecfg, BP, ev_plan.seed_of)
            if D == "R3":
                # sd over two members at l -+ sp: sp sqrt(2); d constant, so the interval is the
                # point (float64 roundoff of the bootstrap's weighted means, 1e-12)
                assert js["n_rows"] == 10 and js["n_donors"] == 5, js
                assert abs(js["Dbar"] - dbar) < 1e-12 and abs(js["ci"][0] - dbar) < 1e-12 \
                    and abs(js["ci"][1] - dbar) < 1e-12, js
                assert abs(js["sigma_seed"] - sp * math.sqrt(2.0)) < 1e-12, js
                assert js["resolved"] and js["lower_nll_ensemble"] == lower, js
                if w[1] == "vs_A1":
                    assert js["lower_nll"] == "term arm", js
            else:
                # d = +-0.3 by donor (S-A2-R3's) or 0 (S-A5-R3 against S-A1): Dbar 0, not resolved
                assert abs(js["Dbar"]) < 1e-12 and not js["resolved"], js
                assert js["lower_nll_ensemble"] is None and js.get("lower_nll") is None, js
    # A2 against A5 and a comparison against A1 never share a bootstrap seed
    seeds = dict((w[0], _load(os.path.join(md, "paired_%s_R3.json" % w[0]))["bootstrap_seed"])
                 for w in WANT)
    assert len(set(seeds.values())) == 3, seeds
    # a reference not complete (this configuration has no best fit): left out, named, one line
    lines = []
    files, made, left = EM.paired_all(cfg, ecfg, rep[1:], md, lines.append)
    why = "S-A1:A1 is not reported (not complete)"
    assert left == [{"comparison": w[0], "E": w[2], "E_ref": w[3], "why": why} for w in WANT[:2]], left
    assert lines[:2] == ["  comparison %s left out: %s" % (w[0], why) for w in WANT[:2]], lines
    assert [m["name"] for m in made] == [WANT[2][0]] and len(files) == 4, made
    # no data set sampled for both: S-A1's sample on R0 alone
    _write_sample(ecfg, "S-A1_A1", sets("S-A1:A1", ["R0"]))
    lines = []
    files, made, left = EM.paired_all(cfg, ecfg, rep, md, lines.append)
    why = "no data set is sampled for both"
    assert left == [{"comparison": w[0], "E": w[2], "E_ref": w[3], "why": why} for w in WANT[:2]], left
    assert lines[:2] == ["  comparison %s left out: %s" % (w[0], why) for w in WANT[:2]], lines
    assert [m["name"] for m in made] == [WANT[2][0]], made
    # rows, donors or theta^ref differing between the two samples: refused, naming both and D
    _write_sample(ecfg, "S-A1_A1", sets("S-A1:A1", order["S-A1:A1"]))
    for key in ("rows", "donor", "theta_ref"):
        bad = sets("S-A5-R3:A5", order["S-A5-R3:A5"])
        arr = bad[1][1]                                            # R3
        arr[key] = arr[key].copy()
        arr[key][0] = arr[key][0] + (7 if key != "theta_ref" else 0.25)
        _write_sample(ecfg, "S-A5-R3_A5", bad)
        try:
            EM.paired_all(cfg, ecfg, rep, md, lambda t: None)
        except ev_plan.RefuseError as exc:
            assert str(exc) == "S-A5-R3:A5 and S-A1:A1 do not share the rows of R3", (key, exc)
        else:
            raise AssertionError("%s altered: not refused" % key)
    return len(WANT)


def test_e27b_metrics():
    try:
        import numpy as np
        import ev_metrics as EM      # noqa: F401  (numpy, scipy, the hpc/ diagnostics)
    except ImportError as exc:
        raise unittest.SkipTest("numpy and scipy are needed (%s)" % exc)
    tmp = tempfile.mkdtemp(prefix="e27b_")
    try:
        _known_answers(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[e27b] known answers: Dbar, the interval, sigma_seed, the verdicts and the lower "
          "ensemble; E's order; left out (not complete, no data set shared); rows refused")
    try:
        F = _fixture()
    except unittest.SkipTest as exc:
        raise unittest.SkipTest("the known answers passed; the reduced project needs %s" % exc)
    import bootstrap_paired as BP
    import ev_config
    import ev_plan
    ecfg = ev_config.load(F["ecfg"])
    md, sd = ecfg.sub("metrics"), ecfg.sub("samples")
    shutil.rmtree(md, ignore_errors=True)
    t0 = time.time()
    lines = []
    idx = EM.compute(F["cfg"], F["ecfg"], ncpus=1, log=lines.append)
    # the pass line last, the comparisons' line before it (block 20's format kept)
    assert lines[-1].startswith("EVAL METRICS OK: 3 ensemble(s), 10 data-set file(s), 9 paired "
                                "comparison(s), 3 search(es), 0 left out"), lines[-1]
    assert lines[-2] == "comparisons: 3 made, 0 left out, 0 search(es) in none", lines[-2:]
    pl = [l for l in lines if l.startswith("  paired ")]
    assert [PAIRED_LINE.match(l).group(1, 2) for l in pl] == [(w[0], D) for w in WANT
                                                              for D in DATASETS], pl
    # nine paired files of each extension, every one in the index
    for ext in (".json", ".npz"):
        got = sorted(os.path.basename(p) for p in glob.glob(os.path.join(md, "paired_*" + ext)))
        want = sorted("paired_%s_%s%s" % (w[0], D, ext) for w in WANT for D in DATASETS)
        assert got == want, (got, want)
        assert all(os.path.join(md, f) in idx["files"] for f in want), idx["files"]
    # each file: its keys per kind, E3 recomputed from the two samples, the bootstrap's interval
    out_rows = 0
    for w, line in zip([w for w in WANT for D in DATASETS], pl):
        D = PAIRED_LINE.match(line).group(2)
        base = os.path.join(md, "paired_%s_%s" % (w[0], D))
        js = _load(base + ".json")
        with np.load(os.path.join(sd, "%s_%s.npz" % (_stem(w[2]), D))) as a, \
                np.load(os.path.join(sd, "%s_%s.npz" % (_stem(w[3]), D))) as b, \
                np.load(base + ".npz") as arr:
            zE, zR = dict(a), dict(b)
            _check_paired(js, arr, w, D, zE, zR, ecfg, BP, ev_plan.seed_of)
        assert ("Dbar %+.4f [%+.4f, %+.4f], 2 sigma_seed %.4f -> " % (
            js["Dbar"], js["ci"][0], js["ci"][1], js["two_sigma_seed"])) in line, (line, js)
        if D == "R3dag":
            out_rows = int((~zE["in_box"]).sum())
            assert 0 < js["n_rows"] == int(zE["in_box"].sum()) < zE["in_box"].size, js
    assert out_rows > 0, "no theta-dagger outside the box: the paired rows' filter is not exercised"
    # METRICS.json: the comparisons made in order, with their data sets and files
    want_cmp = [dict(zip(FIELDS, w), datasets=list(DATASETS),
                     paired=dict((D, os.path.join(md, "paired_%s_%s.json" % (w[0], D)))
                                 for D in DATASETS)) for w in WANT]
    disk = _load(os.path.join(md, "METRICS.json"))
    for ix in (idx, disk):
        assert ix["comparisons"] == want_cmp, ix["comparisons"]
        assert ix["comparisons_left_out"] == [] and ix["uncompared"] == [], ix
        assert ix["report_inputs"]["comparisons"] == [[w[0], w[2], w[3]] for w in WANT], ix
    assert all(os.path.isabs(p) for c in want_cmp for p in c["paired"].values())
    # S-A1's sample provenance removed: the two comparisons against A1 left out and named
    pj = ev_plan.sample_json(ecfg, "S-A1_A1")
    shutil.move(pj, pj + ".aside")
    try:
        lines = []
        idx2 = EM.compute(F["cfg"], F["ecfg"], ncpus=1, log=lines.append)
        # contract amendment 1: an ensemble that --ensemble did not name and that has no current
        # sample is named by its sample's state (block 20's reason), as when it is named,
        # although compute, which skips what --ensemble does not name, gives it no "left_out"
        lines_x = []
        idx_x = EM.compute(F["cfg"], F["ecfg"], names=["S-A2-R3:A2", "S-A5-R3:A5"], ncpus=1,
                           log=lines_x.append)
    finally:
        shutil.move(pj + ".aside", pj)
    why = "S-A1:A1 is not reported (not sampled (%s absent))" % pj
    gone = [{"comparison": w[0], "E": w[2], "E_ref": w[3], "why": why} for w in WANT[:2]]
    assert idx_x["left_out"] == [] and idx_x["comparisons_left_out"] == gone, idx_x
    assert idx_x["comparisons"] == want_cmp[2:], idx_x["comparisons"]
    assert [l for l in lines_x if l.startswith("  comparison ")] == [
        "  comparison %s left out: %s" % (w[0], why) for w in WANT[:2]], lines_x
    # without --ensemble: the same whys, and compute's own "left_out" names the sample
    assert {"ensemble": "S-A1:A1", "why": "not sampled (%s absent)" % pj} in idx2["left_out"]
    assert idx2["comparisons_left_out"] == gone, idx2["comparisons_left_out"]
    assert idx2["comparisons"] == want_cmp[2:], idx2["comparisons"]
    for w in WANT[:2]:
        assert lines.count("  comparison %s left out: %s" % (w[0], why)) == 1, lines
    assert lines[-2] == "comparisons: 1 made, 2 left out, 0 search(es) in none", lines[-2:]
    assert lines[-1].startswith("EVAL METRICS OK: 2 ensemble(s), 6 data-set file(s), 3 paired "
                                "comparison(s), 2 search(es), 1 left out"), lines[-1]
    assert _load(os.path.join(md, "METRICS.json"))["comparisons_left_out"] == gone
    # --ensemble on a copy of the configuration with a search not complete and an S-A25 search
    d = _load(F["cfg"])
    d["synthetic"]["searches"] += [_q("S-A2-R0", "S-A2-only", "R0"), _q("S-A25-R3", "S-A25", "R3")]
    extra = os.path.join(F["root"], "synth_extra.json")
    with open(extra, "w") as fh:
        json.dump(d, fh, indent=1)
    lines = []
    try:
        idx3 = EM.compute(extra, F["ecfg"], names=["S-A5-R3:A5"], ncpus=1, log=lines.append)
    finally:
        os.remove(extra)
    left3 = [("S-A2-R3", "S-A2-R3:A2", "S-A1:A1", "S-A2-R3:A2 was not requested (--ensemble)"),
             ("S-A5-R3", "S-A5-R3:A5", "S-A1:A1", "S-A1:A1 was not requested (--ensemble)"),
             ("S-A2-R0", "S-A2-R0:A2", "S-A1:A1", "S-A2-R0:A2 is not reported (not complete)"),
             ("S-A2-R3_vs_S-A5-R3", "S-A2-R3:A2", "S-A5-R3:A5",
              "S-A2-R3:A2 was not requested (--ensemble)")]
    assert idx3["comparisons_left_out"] == [dict(zip(("comparison", "E", "E_ref", "why"), x))
                                            for x in left3], idx3["comparisons_left_out"]
    assert [l for l in lines if l.startswith("  comparison ")] == [
        "  comparison %s left out: %s" % (x[0], x[3]) for x in left3], lines
    assert idx3["comparisons"] == [] and idx3["uncompared"] == [
        {"search": "S-A25-R3", "why": WHY_S_A25}], idx3
    assert lines[-2] == "comparisons: 0 made, 4 left out, 1 search(es) in none", lines[-2:]
    assert lines[-1].startswith("EVAL METRICS OK: 1 ensemble(s), 3 data-set file(s), 0 paired "
                                "comparison(s), 1 search(es), 2 left out"), lines[-1]
    # one sample's rows altered: the CLI refuses, naming both ensembles and the data set
    p = os.path.join(sd, "S-A5-R3_A5_R3.npz")
    with open(p, "rb") as fh:
        keep = fh.read()
    try:
        with np.load(p) as z:
            arrays = dict(z)
        arrays["rows"] = arrays["rows"].copy()
        arrays["rows"][0] += 1000
        np.savez_compressed(p, **arrays)
        r = subprocess.run([PYTHON, METRICS_PY, "--config", F["cfg"], "--eval-config", F["ecfg"],
                            "--ensemble", "S-A2-R3:A2", "--ensemble", "S-A5-R3:A5", "--ncpus", "1"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=900)
    finally:
        with open(p, "wb") as fh:
            fh.write(keep)
    assert r.returncode == 3 and r.stdout.rstrip().splitlines()[-1] == (
        "REFUSE: S-A2-R3:A2 and S-A5-R3:A5 do not share the rows of R3"), r.stdout[-2500:]
    assert "  comparison S-A2-R3 left out: S-A1:A1 was not requested (--ensemble)" in r.stdout
    assert "  paired S-A2-R3_vs_S-A5-R3 on sim: " in r.stdout, r.stdout[-2500:]
    print("[e27b] the reduced v4-like project: 9 paired files recomputed from the samples (%d "
          "theta-dagger rows outside the box), METRICS.json's comparisons; left out (not sampled, "
          "not requested, not complete), S-A25 in none; the rows' refusal (%.0f s)"
          % (out_rows, time.time() - t0))


def test_e27c_launcher():
    import ev_config
    import ev_plan
    import ev_submit
    cmp_lines = ["EVAL SUBMIT: comparison %s: %s against %s (%s)" % (w[0], w[2], w[3], w[1])
                 for w in WANT]

    def after(out, start):
        """The output's lines after the unit's line `start` (and its qsub and
        command lines in a dry run)."""
        ll = out.splitlines()
        i = [k for k, l in enumerate(ll) if l.startswith(start)]
        assert len(i) == 1, (start, out[-3000:])
        i = i[0] + 1
        while i < len(ll) and ll[i].startswith("    "):
            i += 1
        return ll[i:]
    tmp = tempfile.mkdtemp(prefix="e27c_")
    try:
        # report_inputs: block 19's dict and the comparisons, [name, E, E_ref] in order
        cfg_a = _config(os.path.join(tmp, "a"), SEARCHES)
        ec_a = TE._eval_config(os.path.join(tmp, "a"), "eval.json", out=os.path.join(tmp, "a", "ev"))
        cfg, ecfg = ev_plan.load_project(cfg_a), ev_config.load(ec_a)
        ri = ev_plan.report_inputs(cfg, ecfg)
        assert ri == {"samples": {}, "params": ecfg.report_params(),
                      "comparisons": [[w[0], w[2], w[3]] for w in WANT]}, ri
        # the dry run with report: the comparisons right after the report's unit; nothing written
        rc, out = _launch(cfg_a, ec_a, "--dry-run")
        assert rc == 0 and "REFUSE" not in out, out[-3000:]
        rest = after(out, "EVAL SUBMIT: report: nothing to report yet")
        assert rest[:3] == cmp_lines and rest[3].startswith("EVAL DRY RUN: 2 job(s) would be "
                                                            "submitted"), rest
        rc, out = _launch(cfg_a, ec_a, "--dry-run", "--what", "report,data")
        rest = after(out, "EVAL SUBMIT: report: nothing to report yet")
        assert rc == 0 and rest[:3] == cmp_lines and rest[3].startswith(
            "EVAL SUBMIT: data would be submitted"), out[-3000:]
        rc, out = _launch(cfg_a, ec_a, "--dry-run", "--what", "calbank,sample,data")
        assert rc == 0 and "comparison" not in out, out[-3000:]
        assert not os.path.exists(ecfg.out_root), "a dry run wrote %s" % ecfg.out_root
        # a REPORT_COMPLETE.json of inputs without the comparisons: the report's unit stale
        os.makedirs(ecfg.out_root)
        rcp = os.path.join(ecfg.out_root, "REPORT_COMPLETE.json")
        for inputs, state in ((dict((k, v) for k, v in ri.items() if k != "comparisons"), "stale"),
                              (ri, "done")):
            with open(rcp, "w") as fh:
                json.dump({"schema": "eval_report/1", "inputs": inputs,
                           "finished": "2026-11-15T12:00:00"}, fh)
            _c, _e, units = ev_submit.plan(cfg_a, ec_a, what=("report",), log=lambda t: None)
            assert [u["unit"] for u in units] == ["report"] and units[0]["state"] == state, units
            if state == "stale":
                assert units[0]["why"] == "its comparisons changed since 2026-11-15T12:00:00", units
            assert [tuple(c) for c in units[0]["comparisons"]] == WANT, units[0]
            assert units[0]["uncompared"] == [], units[0]
        # an S-A25 search: its line among the comparisons'
        cfg_c = _config(os.path.join(tmp, "c"), [_q("S-A1", "S-A1", "R0"),
                                                 _q("S-A25-R3", "S-A25", "R3")])
        ec_c = TE._eval_config(os.path.join(tmp, "c"), "eval.json", out=os.path.join(tmp, "c", "ev"))
        rc, out = _launch(cfg_c, ec_c, "--dry-run", "--what", "report")
        rest = after(out, "EVAL SUBMIT: report: nothing to report yet")
        assert rc == 0 and rest[0] == "EVAL SUBMIT: no comparison for S-A25-R3: " + WHY_S_A25, rest
        assert "EVAL SUBMIT: comparison " not in out, out[-2000:]
        # term-only searches and no S-A1 search: the report refused, nothing written or submitted
        root_b = os.path.join(tmp, "b")
        cfg_b = _config(root_b, [_q("S-A2-R3", "S-A2-only", "R3"), _q("S-A5-R3", "S-A5-only", "R3")])
        ec_b = TE._eval_config(root_b, "eval.json", out=os.path.join(root_b, "ev"))
        spool = os.path.join(tmp, "spool")
        env = TE._fake_pbs_env(spool)
        before = TE._files_under(root_b)
        for args in (("--dry-run", "--what", "report"), ("--what", "report"), ()):
            rc, out = _launch(cfg_b, ec_b, *args, env=env)
            last = out.rstrip().splitlines()[-1]
            assert rc == 3 and last.startswith("REFUSE: search S-A2-R3 (campaign S-A2-only") and \
                last.endswith("the configuration has none"), (args, out[-2000:])
            assert TE._files_under(root_b) == before and TS._records(spool) == [], args
        rc, out = _launch(cfg_b, ec_b, "--dry-run", "--what", "calbank,sample", env=env)
        assert rc == 0 and "REFUSE" not in out and "comparison" not in out, out[-2000:]
        assert "EVAL DRY RUN: 2 job(s) would be submitted, 0 done, 0 live" in out, out[-1500:]
        rc, out = _launch(cfg_b, ec_b, "--dry-run", "--what", "data", env=env)
        assert rc == 0 and "REFUSE" not in out, out[-2000:]
        assert TE._files_under(root_b) == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[e27c] report_inputs with the comparisons; a report without them stale; the dry run's "
          "comparison lines after the report's unit; no S-A1 search: the report refused, "
          "the calibration bank and the samplings not")
    # the fixture: samples present, so the report would be submitted and the comparisons follow it
    try:
        F = _fixture()
    except unittest.SkipTest as exc:
        raise unittest.SkipTest("the configurations' checks passed; the fixture's part needs %s"
                                % exc)
    ecfg = ev_config.load(F["ecfg"])
    rcp = os.path.join(ecfg.out_root, "REPORT_COMPLETE.json")
    assert not os.path.exists(rcp), rcp
    ri = ev_plan.report_inputs(ev_plan.load_project(F["cfg"]), ecfg)
    assert sorted(ri["samples"]) == sorted(_stem(n) for n in ENSEMBLES), ri["samples"]
    before = TE._files_under(ecfg.out_root)
    rc, out = _launch(F["cfg"], F["ecfg"], "--dry-run", "--what", "report")
    rest = after(out, "EVAL SUBMIT: report would be submitted:")
    assert rc == 0 and rest[:3] == cmp_lines and rest[3] == (
        "EVAL DRY RUN: 1 job(s) would be submitted, 0 done, 0 live; nothing written"), out[-3000:]
    assert TE._files_under(ecfg.out_root) == before
    try:
        for inputs, unit, tail in (
                (dict((k, v) for k, v in ri.items() if k != "comparisons"),
                 "EVAL SUBMIT: report would be submitted (stale: its comparisons changed since ",
                 "EVAL DRY RUN: 1 job(s) would be submitted, 0 done, 0 live; nothing written"),
                (ri, "EVAL SUBMIT: report done (",
                 "EVAL DRY RUN: 0 job(s) would be submitted, 1 done, 0 live; nothing written")):
            with open(rcp, "w") as fh:
                json.dump({"schema": "eval_report/1", "inputs": inputs,
                           "finished": "2026-11-15T12:00:00"}, fh)
            rc, out = _launch(F["cfg"], F["ecfg"], "--dry-run", "--what", "report")
            rest = after(out, unit)
            assert rc == 0 and rest[:3] == cmp_lines and rest[3] == tail, out[-3000:]
    finally:
        if os.path.exists(rcp):
            os.remove(rcp)
    print("[e27c] the fixture: the report would be submitted, stale without the comparisons, done "
          "with them; the three comparisons after it each time")


def test_convergence():
    raise unittest.SkipTest("not a discretisation: no convergence order")


# ------------------------------------------------------------------ main ---

CHECKS = [("test_e27a_plan", test_e27a_plan), ("test_e27b_metrics", test_e27b_metrics),
          ("test_e27c_launcher", test_e27c_launcher), ("test_convergence", test_convergence)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    try:
        for name, fn in CHECKS:
            if only and name not in only:
                continue
            t1 = time.time()
            try:
                fn()
                rows.append((name, "PASS", "", time.time() - t1))
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        if _FIX.get("built"):
            if os.environ.get("CMP_KEEP", "").strip() == "1":
                print("[fixture] kept: %s" % _FIX["root"])
            else:
                shutil.rmtree(_FIX["root"], ignore_errors=True)
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
