#!/usr/bin/env python3
"""Smoke test of block 33: the view of a recorded project, its data and the evaluation's configuration (specs/SPEC.md block 33, "Test
oracles", checks R33*).

R33a (its file part), R33b, R33c and R33f run on rec_fixture (the training
environment; a few minutes to build, REC_FIXTURE=<dir> reuses a tree,
REC_KEEP=1 keeps the one built); R33a's rules, R33d and R33e need the
standard library and numpy only. A check that cannot run here raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r33a_config       the configuration: every key rule, sample_params, report_params
    test_r33b_as_recorded  the as-recorded view equals ev_plan's ensembles and comparisons
    test_r33c_recorded     the recorded view: names, bank, missing, refusals
    test_r33d_comparisons  the comparisons of a recorded view, by hand
    test_r33e_draw_rows    draw_rows by hand; k = 1 equals calibration_rows
    test_r33f_recording    recording(): keys, dtypes, order, no theta; axes_of

Run
    python3 hpc/pipeline/tests/test_smoke_rec_plan.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

Pure ASCII, LF only.
"""

import os
import sys
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 20261123  # every random number of this file derives from it

import copy      # noqa: E402
import glob      # noqa: E402
import json      # noqa: E402
import shutil    # noqa: E402
import tempfile  # noqa: E402

import ev_rec_config as RC   # noqa: E402  (standard library only)

LINDDUP = os.path.join(_PIPE, "projects", "linddup", "linddup.json")


# --------------------------------------------------------------- helpers ---

def _fixture():
    import rec_fixture
    return rec_fixture.fixture()


def _write(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    return path


def _read(path):
    with open(path) as fh:
        return json.load(fh)


def _rec_dict(**over):
    """A valid recorded evaluation configuration, written here from the
    entry's values (not from rec_fixture, so that R33a needs no fixture)."""
    d = {"out_root": "/tmp/rec_eval_out", "seed": 0, "n_post": 1000, "batch_rows": 256,
         "oracle_tol": 2e-3, "as_recorded": False, "sim_draw_rows_per_donor": 1,
         "edge_eps": 0.006693,
         "gate": {"donor_fracs": [0.4, 0.25, 0.25, 0.1], "n_ref": 800, "n_rank": 1000,
                  "n_null": 1000, "n_null_min": 50, "rows_per_donor": 1,
                  "factors": [0.25, 0.5, 1, 2, 4], "alpha": 0.05},
         "bootstrap_n": 2000, "level": 0.95, "dpi": 150, "formats": ["png", "pdf"],
         "ncpus": 4, "validate_against": None,
         "jobs": {"queue": "q", "sample": {"ncpus": 8, "mem": "32gb", "walltime": "04:00:00"},
                  "report": {"ncpus": 8, "mem": "32gb", "walltime": "04:00:00"},
                  "validate": {"ncpus": 1, "mem": "8gb", "walltime": "01:00:00"}}}
    d.update(over)
    return d


def _probs_name(data, needle):
    probs = RC.validate(data)
    assert probs, "no problem found for %s" % needle
    assert any(needle in p for p in probs), "%s not named in %r" % (needle, probs)
    return probs


def _cfg_like(base_path, out_path, **top):
    """A copy of a project configuration with top-level keys replaced; a
    dotted key "a.b" replaces a nested value."""
    d = _read(base_path)
    for k, v in top.items():
        cur = d
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = v
    return _write(out_path, d)


def _ecfg(path, **over):
    return RC.load(_write(path, _rec_dict(out_root=os.path.join(os.path.dirname(path), "out"),
                                          **over)))


def _copy_best(src_folder, dst_folder):
    """A best-fit folder's JSON files (its completion and the members'
    records; no checkpoint), copied."""
    for f in glob.glob(os.path.join(src_folder, "*.json")) + glob.glob(
            os.path.join(src_folder, "member_seed*", "*.json")):
        rel = os.path.relpath(f, src_folder)
        os.makedirs(os.path.dirname(os.path.join(dst_folder, rel)), exist_ok=True)
        shutil.copy2(f, os.path.join(dst_folder, rel))


def _expect_refusal(fn, *needles):
    import ev_plan
    try:
        fn()
    except ev_plan.RefuseError as exc:
        msg = str(exc)
        for n in needles:
            assert n in msg, "refusal %r does not name %r" % (msg, n)
        return msg
    raise AssertionError("no refusal (expected one naming %r)" % (needles,))


# ---------------------------------------------------------------- checks ---

def test_r33a_config():
    """the configuration: every key rule, sample_params, report_params (specs/SPEC.md block 33)."""
    ok = _rec_dict()
    assert RC.validate(ok) == [], RC.validate(ok)
    # each missing and each unknown key is named
    for k in RC.KEYS:
        d = _rec_dict()
        del d[k]
        probs = _probs_name(d, k)
        assert any(p.startswith("missing") for p in probs), probs
    d = _rec_dict(extra_key=1)
    assert any("unknown" in p and "extra_key" in p for p in RC.validate(d)), RC.validate(d)
    for k in RC.GATE_KEYS:
        d = _rec_dict()
        del d["gate"][k]
        _probs_name(d, k)
    d = _rec_dict()
    d["gate"]["n_ref_typo"] = 3
    _probs_name(d, "n_ref_typo")
    # one check per value rule (the entry's examples first)
    bad = [
        ("gate.donor_fracs", {"gate": dict(ok["gate"], donor_fracs=[0.4, 0.25, 0.15, 0.1])}),
        ("gate.donor_fracs", {"gate": dict(ok["gate"], donor_fracs=[0.5, 0.25, 0.25])}),
        ("gate.donor_fracs", {"gate": dict(ok["gate"], donor_fracs=[0.5, 0.5, 0.0, 0.0])}),
        ("edge_eps", {"edge_eps": 0.5}),
        ("edge_eps", {"edge_eps": 0.0}),
        ("gate.alpha", {"gate": dict(ok["gate"], alpha=1)}),
        ("gate.alpha", {"gate": dict(ok["gate"], alpha=0.0)}),
        ("n_post", {"n_post": 0}),
        ("gate.factors", {"gate": dict(ok["gate"], factors=[])}),
        ("gate.factors", {"gate": dict(ok["gate"], factors=[1, -2])}),
        ("validate_against", {"validate_against": "/tmp/old_out"}),
        ("validate_against", {"validate_against": "relative/out", "as_recorded": True}),
        ("jobs", {"jobs": dict((k, v) for k, v in ok["jobs"].items() if k != "validate")}),
        ("jobs.sample.mem", {"jobs": dict(ok["jobs"], sample={"ncpus": 1, "mem": "32",
                                                              "walltime": "01:00:00"})}),
        ("jobs.report.walltime", {"jobs": dict(ok["jobs"], report={"ncpus": 1, "mem": "1gb",
                                                                   "walltime": "1h"})}),
        ("jobs.validate.ncpus", {"jobs": dict(ok["jobs"], validate={"ncpus": 0, "mem": "1gb",
                                                                    "walltime": "01:00:00"})}),
        ("jobs.queue", {"jobs": dict(ok["jobs"], queue="")}),
        ("out_root", {"out_root": "relative/out"}),
        ("seed", {"seed": -1}),
        ("seed", {"seed": True}),
        ("batch_rows", {"batch_rows": 0}),
        ("oracle_tol", {"oracle_tol": 0.0}),
        ("oracle_tol", {"oracle_tol": 0.2}),
        ("as_recorded", {"as_recorded": 1}),
        ("sim_draw_rows_per_donor", {"sim_draw_rows_per_donor": 0}),
        ("sim_draw_rows_per_donor", {"sim_draw_rows_per_donor": 1.5}),
        ("gate.n_ref", {"gate": dict(ok["gate"], n_ref=1)}),
        # [2026-11-23, after the final check: F4, F2, F3] above 1,000 reference rows the
        # median of (R5) would be a subsample's; a non-finite factor; a trailing newline
        ("gate.n_ref", {"gate": dict(ok["gate"], n_ref=1001)}),
        ("gate.factors", {"gate": dict(ok["gate"], factors=[1.0, float("inf")])}),
        ("gate.factors", {"gate": dict(ok["gate"], factors=[float("nan")])}),
        ("jobs.sample.mem", {"jobs": dict(ok["jobs"], sample={"ncpus": 1, "mem": "64gb\n",
                                                              "walltime": "01:00:00"})}),
        ("jobs.report.walltime", {"jobs": dict(ok["jobs"], report={
            "ncpus": 1, "mem": "1gb", "walltime": "01:00:00\n"})}),
        ("n_post", {"n_post": 1000.0}),          # an integer key requires an int (S3)
        ("gate.n_rank", {"gate": dict(ok["gate"], n_rank=0)}),
        ("gate.n_null", {"gate": dict(ok["gate"], n_null=0)}),
        ("gate.n_null_min", {"gate": dict(ok["gate"], n_null_min=0)}),
        ("gate.rows_per_donor", {"gate": dict(ok["gate"], rows_per_donor=0)}),
        ("gate", {"gate": [1, 2]}),
        ("bootstrap_n", {"bootstrap_n": 0}),
        ("level", {"level": 1.0}),
        ("dpi", {"dpi": 0}),
        ("formats", {"formats": []}),
        ("formats", {"formats": ["png", "png"]}),
        ("formats", {"formats": ["gif"]}),
        ("ncpus", {"ncpus": 0}),
    ]
    for needle, over in bad:
        _probs_name(_rec_dict(**copy.deepcopy(over)), needle)
    # the values the rules allow at their edges
    for over in ({"sim_draw_rows_per_donor": None}, {"oracle_tol": 0.1}, {"seed": 0},
                 {"validate_against": "/tmp/old_out", "as_recorded": True},
                 {"gate": dict(ok["gate"], donor_fracs=[0.1, 0.2, 0.3, 0.4])},
                 {"gate": dict(ok["gate"], n_ref=1000)}, {"level": 0.9},
                 {"gate": dict(ok["gate"], factors=[1])}):
        assert RC.validate(_rec_dict(**copy.deepcopy(over))) == [], (over, RC.validate(
            _rec_dict(**over)))
    tmp = tempfile.mkdtemp(prefix="r33a_")
    try:
        # `_` keys are comments, at the top and in `gate`; every problem listed, with the path
        d = _rec_dict(_comment="x")
        d["gate"]["_n_ref"] = "doc"
        e = RC.load(_write(os.path.join(tmp, "ok.json"), d))
        assert isinstance(e, RC.RecEvalConfig) and e.path == os.path.join(tmp, "ok.json")
        assert e.formats == ("png", "pdf") and e.gate["donor_fracs"] == (0.4, 0.25, 0.25, 0.1)
        assert e.gate["factors"] == (0.25, 0.5, 1, 2, 4) and "_n_ref" not in e.gate
        assert e.sub("samples") == os.path.join(d["out_root"], "samples")
        p = _write(os.path.join(tmp, "bad.json"), _rec_dict(n_post=0, edge_eps=0.5))
        try:
            RC.load(p)
            raise AssertionError("a bad configuration loaded")
        except RC.RecEvalConfigError as exc:
            assert p in str(exc) and "n_post" in str(exc) and "edge_eps" in str(exc), str(exc)
        # sample_params and report_params hold exactly the keys listed
        sp, rp = e.sample_params(), e.report_params()
        assert sorted(sp) == sorted(["seed", "n_post", "as_recorded", "sim_draw_rows_per_donor",
                                     "gate"]), sorted(sp)
        assert sorted(sp["gate"]) == sorted(["donor_fracs", "n_ref", "n_rank", "n_null",
                                             "n_null_min", "rows_per_donor"]), sorted(sp["gate"])
        assert sorted(rp) == sorted(["seed", "edge_eps", "bootstrap_n", "level", "dpi",
                                     "formats", "gate"]), sorted(rp)
        assert sorted(rp["gate"]) == ["alpha", "factors"], sorted(rp["gate"])
        assert "oracle_tol" not in json.dumps(sp) + json.dumps(rp)
        assert sp["gate"]["donor_fracs"] == [0.4, 0.25, 0.25, 0.1] and rp["formats"] == [
            "png", "pdf"] and rp["gate"]["factors"] == [0.25, 0.5, 1, 2, 4]
        assert json.loads(json.dumps(sp)) == sp and json.loads(json.dumps(rp)) == rp
        assert sp["n_post"] == 1000 and rp["edge_eps"] == 0.006693 and rp["gate"]["alpha"] == 0.05
        assert RC.JOB_KINDS == ("sample", "report", "validate")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the fixture's two files load
    F = _fixture()
    ea, er = RC.load(F["ecfg_as"]), RC.load(F["ecfg_rec"])
    assert ea.as_recorded is True and ea.sim_draw_rows_per_donor is None
    assert ea.gate["n_null_min"] == 1 and ea.validate_against == F["old_out"]
    old = _read(F["ecfg_old"])
    for k in ("seed", "n_post", "batch_rows"):
        assert getattr(ea, k) == old[k], k
    assert er.as_recorded is False and er.sim_draw_rows_per_donor == 1
    # [amendment 2, D-165] both fixture configurations: gate.alpha 0.5, n_null_min 1, and on
    # the fixture's simulated bank a null set of 2 rows, so k_gate (R9) = 1, not a refusal
    import math
    import ev_gate
    import ev_plan
    import ev_rec_data as D
    import ev_rec_plan as P
    for e in (ea, er):
        assert e.gate["alpha"] == 0.5 and e.gate["n_null_min"] == 1, (e.path, e.gate)
        view = P.load_view(F["cfg"] if e.as_recorded else F["cfg_rec"], e)
        index, _paths = D.sim_index(view)
        ens = view.ensembles[0]
        rec = _read(ens.records[0])
        rows = D.report_rows(index, rec["split_fracs"], rec["split_seed"], rec["split_hash"])
        g = e.gate
        role = ev_gate.assign_roles(index.donor[rows], g["donor_fracs"], g["n_ref"], g["n_rank"],
                                    g["n_null"], g["rows_per_donor"],
                                    ev_plan.seed_of(e.seed, "gate"))
        n_null = int((role == ev_gate.ROLE_NULL).sum())
        k_gate = int(math.ceil(g["alpha"] * (n_null + 1))) - 1
        assert n_null == 2 and k_gate == 1, (e.path, n_null, k_gate)
        assert n_null >= g["n_null_min"]


def test_r33b_as_recorded():
    """the as-recorded view equals ev_plan's ensembles and comparisons (specs/SPEC.md block 33)."""
    F = _fixture()
    import ev_plan
    import ev_rec_plan as P
    ea = RC.load(F["ecfg_as"])
    view = P.load_view(F["cfg"], ea)
    cfg = ev_plan.load_project(F["cfg"])
    old, old_missing = ev_plan.ensembles(cfg, with_missing=True)
    assert view.kind == "as_recorded"
    assert len(view.ensembles) == len(old) == 3, (view.ensembles, old)
    for e, o in zip(view.ensembles, old):
        assert (e.name, e.stem, e.folder, tuple(e.seeds), tuple(e.records), tuple(e.banks)) == \
            (o.name, o.stem, o.folder, tuple(o.seeds), tuple(o.records), tuple(o.cohorts)), (e, o)
        assert e.search == o.search["name"] and e.campaign == o.search["campaign"] \
            and e.arm == o.arm, (e, o)
    assert list(view.missing) == list(old_missing), (view.missing, old_missing)
    assert view.sim_glob == ev_plan.sim_bank_glob(cfg)
    assert view.banks == dict((r["name"], ev_plan.cohort_glob(cfg, r["name"]))
                              for r in cfg.data["synthetic"]["real"]), view.banks
    made, not_made = P.comparisons(view)
    assert made == ev_plan.comparisons(cfg), (made, ev_plan.comparisons(cfg))
    assert not_made == ev_plan.comparisons(cfg, with_uncompared=True)[1]
    assert P.find(view, old[0].name) == view.ensembles[0]
    _expect_refusal(lambda: P.find(view, "no:such"), "no:such", old[0].name)
    # the two kind mismatches refused, each naming the other evaluation
    er = RC.load(F["ecfg_rec"])
    _expect_refusal(lambda: P.load_view(F["cfg"], er), "submit_eval.sh", "as_recorded")
    _expect_refusal(lambda: P.load_view(F["cfg_rec"], ea), "as_recorded false")
    # sample_json and report_inputs
    assert P.sample_json(ea, "S-A1_A1") == os.path.join(ea.out_root, "samples",
                                                        "S-A1_A1_sample.json")
    ri = P.report_inputs(view, ea)
    assert ri["params"] == ea.report_params()
    assert ri["comparisons"] == [[c.name, c.E, c.E_ref] for c in made]
    assert set(ri["samples"]) <= set(e.stem for e in view.ensembles)


def test_r33c_recorded():
    """the recorded view: names, bank, missing, refusals (specs/SPEC.md block 33)."""
    F = _fixture()
    import ev_plan
    import ev_rec_plan as P
    er = RC.load(F["ecfg_rec"])
    view = P.load_view(F["cfg_rec"], er)
    syn = _read(F["cfg"])
    runs, banks, tag = syn["paths"]["runs_root"], syn["paths"]["banks_root"], syn["dataset"]
    best = os.path.join(runs, "S-A2-R0", tag + "_best")
    bf = _read(os.path.join(best, "BEST_FIT_COMPLETE.json"))
    arm = set(_read(ev_plan.member_record(best, s))["arm"] for s in bf["member_seeds"])
    assert len(arm) == 1, arm
    arm = arm.pop()
    assert view.kind == "recorded"
    assert [e.name for e in view.ensembles] == ["S-A1:A1", "S-A2-R0:%s" % arm], view.ensembles
    assert [e.stem for e in view.ensembles] == ["S-A1_A1", "S-A2-R0_%s" % arm]
    assert [(e.search, e.campaign) for e in view.ensembles] == [("S-A1", "S-A1"),
                                                                ("S-A2-R0", "S-A2")]
    assert [e.folder for e in view.ensembles] == [os.path.join(runs, "S-A1", tag + "_best"), best]
    assert view.ensembles[1].seeds == tuple(bf["member_seeds"])
    assert all(e.banks == ("real",) for e in view.ensembles)
    assert view.banks == {"real": os.path.join(banks, tag + "_real", "shard_*.npz")}, view.banks
    assert view.sim_glob == ev_plan.sim_bank_glob(ev_plan.load_project(F["cfg"])), view.sim_glob
    assert view.missing == ()
    tmp = tempfile.mkdtemp(prefix="r33c_")
    try:
        # searches null and one best fit: best:<arm>
        r1 = os.path.join(tmp, "runs_null")
        _copy_best(os.path.join(runs, "S-A1", tag + "_best"), os.path.join(r1, tag + "_best"))
        c1 = _cfg_like(F["cfg_rec"], os.path.join(tmp, "null.json"), searches=None,
                       **{"paths.runs_root": r1, "search.campaign": "S-A1", "search.arm": "A1"})
        v1 = P.load_view(c1, er)
        assert [(e.name, e.stem, e.search, e.campaign) for e in v1.ensembles] == [
            ("best:A1", "best_A1", None, "S-A1")], v1.ensembles
        assert P.comparisons(v1) == ([], [])
        # a best fit without BEST_FIT_COMPLETE.json listed in missing
        r2 = os.path.join(tmp, "runs_list")
        for q in ("S-A1", "S-A2-R0"):
            _copy_best(os.path.join(runs, q, tag + "_best"), os.path.join(r2, q, tag + "_best"))
        c2 = _cfg_like(F["cfg_rec"], os.path.join(tmp, "list.json"), **{"paths.runs_root": r2})
        assert [e.name for e in P.load_view(c2, er).ensembles] == [e.name for e in view.ensembles]
        os.remove(os.path.join(r2, "S-A2-R0", tag + "_best", "BEST_FIT_COMPLETE.json"))
        v2 = P.load_view(c2, er)
        assert [e.name for e in v2.ensembles] == ["S-A1:A1"], v2.ensembles
        assert len(v2.missing) == 1 and v2.missing[0][0] == "S-A2-R0" and v2.missing[0][1] == \
            os.path.join(r2, "S-A2-R0", tag + "_best") and "BEST_FIT_COMPLETE" in v2.missing[0][2]
        # a member record of another arm, and one of another split hash, refused
        a1 = os.path.join(r2, "S-A1", tag + "_best")
        rec = ev_plan.member_record(a1, 2)
        orig = _read(rec)
        _write(rec, dict(orig, arm="A2"))
        _expect_refusal(lambda: P.load_view(c2, er), "arm", a1)
        _write(rec, dict(orig, split_hash="0" * 64))
        _expect_refusal(lambda: P.load_view(c2, er), "split_hash", a1)
        _write(rec, orig)
        assert [e.name for e in P.load_view(c2, er).ensembles] == ["S-A1:A1"]
        # a member without its record refused
        os.remove(rec)
        _expect_refusal(lambda: P.load_view(c2, er), "member seed 2 has no record")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _hand_tree(tmp, searches, arms, complete=None):
    """A recorded configuration made from linddup.json with `searches`
    (name, campaign) and, for each search, an empty best-fit folder with
    BEST_FIT_COMPLETE.json and two hand-written member records of arm
    arms[name] (no folder for a name not in `complete`, when given)."""
    runs = os.path.join(tmp, "runs_root")
    for name, _camp in searches:
        if complete is not None and name not in complete:
            continue
        folder = os.path.join(runs, name, "hand_best")
        _write(os.path.join(folder, "BEST_FIT_COMPLETE.json"), {"member_seeds": [1, 2]})
        for s in (1, 2):
            _write(os.path.join(folder, "member_seed%d" % s, "%s_seed%d.json" % (arms[name], s)),
                   {"arm": arms[name], "split_seed": 1, "split_fracs": [0.85, 0.1, 0.05],
                    "split_hash": "ab" * 32})
    d = _read(LINDDUP)
    d["dataset"] = "hand"
    for k in d["paths"]:
        d["paths"][k] = os.path.join(tmp, k)
    d["searches"] = [{"name": n, "campaign": c, "max_cores": None} for n, c in searches]
    return _write(os.path.join(tmp, "hand_%d.json" % len(glob.glob(os.path.join(tmp, "*.json")))),
                  d)


def test_r33d_comparisons():
    """the comparisons of a recorded view, by hand (specs/SPEC.md block 33)."""
    import ev_plan
    import ev_rec_plan as P
    tmp = tempfile.mkdtemp(prefix="r33d_")
    try:
        er = _ecfg(os.path.join(tmp, "rec_eval.json"))
        three = [("A1", "S-A1"), ("A2", "S-A2-only"), ("A5", "S-A5-only")]
        arms = {"A1": "A1", "A2": "A2", "A5": "A5"}
        view = P.load_view(_hand_tree(os.path.join(tmp, "t1"), three, arms), er)
        assert [e.name for e in view.ensembles] == ["A1:A1", "A2:A2", "A5:A5"], view
        made, not_made = P.comparisons(view)
        C = ev_plan.Comparison
        want = [C(name="A2", kind="vs_A1", E="A2:A2", E_ref="A1:A1", search="A2", search_ref="A1",
                  arm="A2", arm_ref="A1"),
                C(name="A5", kind="vs_A1", E="A5:A5", E_ref="A1:A1", search="A5", search_ref="A1",
                  arm="A5", arm_ref="A1"),
                C(name="A2_vs_A5", kind="A2_vs_A5", E="A2:A2", E_ref="A5:A5", search="A2",
                  search_ref="A5", arm="A2", arm_ref="A5")]
        assert made == want, made
        assert not_made == [], not_made
        assert P.report_inputs(view, er)["comparisons"] == [
            ["A2", "A2:A2", "A1:A1"], ["A5", "A5:A5", "A1:A1"], ["A2_vs_A5", "A2:A2", "A5:A5"]]
        # without the S-A1 search, a refusal (block 27's rule)
        v2 = P.load_view(_hand_tree(os.path.join(tmp, "t2"), three[1:], arms), er)
        _expect_refusal(lambda: P.comparisons(v2), "S-A1", "none")
        # a term search whose best fit is not complete: named and left out
        v3 = P.load_view(_hand_tree(os.path.join(tmp, "t3"), three, arms,
                                    complete=("A1", "A2")), er)
        made, not_made = P.comparisons(v3)
        assert [c.name for c in made] == ["A2"], made
        assert [n for n, _w in not_made] == ["A5", "A2_vs_A5"], not_made
        assert all("A5" in w and "not complete" in w for _n, w in not_made), not_made
        assert len(v3.missing) == 1 and v3.missing[0][0] == "A5"
        # a term search whose ensemble is not of its term arm: named with the reason
        v4 = P.load_view(_hand_tree(os.path.join(tmp, "t4"), three, dict(arms, A2="A1")), er)
        made, not_made = P.comparisons(v4)
        assert [c.name for c in made] == ["A5"], made
        assert [n for n, _w in not_made] == ["A2", "A2_vs_A5"], not_made
        assert "arm A1" in not_made[0][1] and "term arm A2" in not_made[0][1], not_made
        # the A1 reference missing: the term searches named, none made
        v5 = P.load_view(_hand_tree(os.path.join(tmp, "t5"), three, arms,
                                    complete=("A2", "A5")), er)
        made, not_made = P.comparisons(v5)
        assert [c.name for c in made] == ["A2_vs_A5"], made
        assert [n for n, _w in not_made] == ["A2", "A5"] and all(
            "reference" in w and "A1" in w for _n, w in not_made), not_made
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_r33e_draw_rows():
    """draw_rows by hand; k = 1 equals calibration_rows (specs/SPEC.md block 33)."""
    import numpy as np
    import ev_rec_data as D
    donor = np.array([5, 5, 7, 5, 7, 9])
    for k, rows in ((1, [0, 2, 5]), (2, [0, 1, 2, 4, 5]), (3, [0, 1, 2, 3, 4, 5]),
                    (None, [0, 1, 2, 3, 4, 5])):
        m = D.draw_rows(donor, k)
        assert m.dtype == np.bool_ and m.shape == (6,), (m.dtype, m.shape)
        assert np.flatnonzero(m).tolist() == rows, (k, np.flatnonzero(m).tolist())
    assert np.array_equal(D.draw_rows(donor, 1), D.calibration_rows(donor))
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        d = rng.integers(0, 15, size=int(rng.integers(1, 80)))
        assert np.array_equal(D.draw_rows(d, 1), D.calibration_rows(d))
        k = int(rng.integers(1, 5))
        m = D.draw_rows(d, k)
        for i in range(d.size):                     # by the definition, row by row
            assert m[i] == (int(np.sum(d[:i] == d[i])) < k), (d.tolist(), k, i)
    assert D.draw_rows(np.zeros(0, dtype=np.int64), 2).shape == (0,)
    for bad in (0, -1, 1.5):
        try:
            D.draw_rows(donor, bad)
            raise AssertionError("k = %r accepted" % (bad,))
        except ValueError:
            pass


def test_r33f_recording():
    """recording(): keys, dtypes, order, no theta; axes_of (specs/SPEC.md block 33)."""
    F = _fixture()
    import numpy as np
    import ev_data
    import ev_plan
    import ev_rec_data as D
    import ev_rec_plan as P
    import run_joint_arms as R
    view = P.load_view(F["cfg_rec"], RC.load(F["ecfg_rec"]))
    rec = D.recording(view, "real")
    want = set(("x", "contaminated", "sidecar", "paths") + D.RECORDING_IDS)
    assert set(rec) == want, sorted(rec)
    assert "theta" not in rec
    assert rec["x"].dtype == np.float32 and rec["x"].ndim == 2
    n = rec["x"].shape[0]
    for k in D.RECORDING_IDS:
        assert rec[k].dtype == np.int64 and rec[k].shape == (n,), (k, rec[k].dtype, rec[k].shape)
    assert rec["contaminated"].dtype == np.bool_ and rec["contaminated"].shape == (n,)
    assert rec["paths"] == sorted(glob.glob(view.banks["real"])) and rec["sidecar"]["arm"] == "R"
    assert np.all(rec["rep_group"] == -1)
    # the rows in bank order, equal to cohort R0's (the bank was written from it)
    cfg = ev_plan.load_project(F["cfg"])
    coh = ev_data.cohort(cfg, "R0")
    arrays, _side = R.concat_shards(ev_data.cohort_paths(cfg, "R0"))
    assert np.array_equal(rec["x"], coh["x"]), "x differs from cohort R0's"
    for k in ("cls", "donor", "well", "window_idx", "contaminated"):
        assert np.array_equal(rec[k], coh[k]), k
    for k in ("batch", "subregion"):
        assert np.array_equal(rec[k], arrays[k].astype(np.int64)), k
    # on the as-recorded view: the same keys, no theta
    va = P.load_view(F["cfg"], RC.load(F["ecfg_as"]))
    for name in sorted(va.banks):
        ra = D.recording(va, name)
        assert set(ra) == want and "theta" not in ra, (name, sorted(ra))
        c = ev_data.cohort(cfg, name)
        assert np.array_equal(ra["x"], c["x"]) and np.array_equal(ra["donor"], c["donor"]), name
    # a bank of arm S refused; an unknown name refused
    vs = view._replace(banks={"real": view.banks["real"], "S": view.sim_glob})
    _expect_refusal(lambda: D.recording(vs, "S"), "arm", "'S'")
    _expect_refusal(lambda: D.recording(view, "nope"), "nope")
    # the simulated bank and axes_of on its sidecar
    index, paths = D.sim_index(view)
    assert paths == sorted(glob.glob(view.sim_glob)) and index.n_rows > 0
    p = int(index.theta.shape[1])
    names, bounds = D.axes_of(index.sidecar, p)
    assert names == list(index.sidecar["param_names"]) and len(names) == p, names
    assert bounds.dtype == np.float64 and bounds.shape == (p, 2)
    assert np.array_equal(bounds, np.asarray(index.sidecar["bounds_theta"], dtype=np.float64))
    # on a sidecar without names (and with names of the wrong length)
    side = dict(index.sidecar)
    del side["param_names"]
    assert D.axes_of(side, p) == (["theta_%d" % k for k in range(p)], None)
    n2, b2 = D.axes_of(dict(index.sidecar, param_names=["a"]), p)
    assert n2 == ["theta_%d" % k for k in range(p)] and b2 is None
    assert D.axes_of({}, 3) == (["theta_0", "theta_1", "theta_2"], None)


CHECKS = [("test_r33a_config", test_r33a_config), ("test_r33b_as_recorded", test_r33b_as_recorded), ("test_r33c_recorded", test_r33c_recorded), ("test_r33d_comparisons", test_r33d_comparisons), ("test_r33e_draw_rows", test_r33e_draw_rows), ("test_r33f_recording", test_r33f_recording)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except NotImplementedError as exc:
            bad += 1
            rows.append((name, "TODO", str(exc), time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except AssertionError as exc:
            bad += 1
            rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
        except Exception:
            bad += 1
            rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    if "rec_fixture" in sys.modules:
        sys.modules["rec_fixture"].cleanup()      # the tree built here, unless REC_KEEP=1
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
