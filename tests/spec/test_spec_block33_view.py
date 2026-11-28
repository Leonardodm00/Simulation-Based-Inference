"""Spec tests of block 33: the evaluation's configuration (`ev_rec_config`),
the draw rows and the axes (`ev_rec_data`), the views (`ev_rec_plan`) and the
recording reader. Integration tester, 2026-11-23, incremental run on dc096e5.

The view tests use a fixture tree built by `rec_fixture` (REC_FIXTURE=<dir>);
they are skipped without it. Expected values come from specs/SPEC.md sec. 2.6
and block 33's entry, and from the bank files read with numpy directly.
"""

import copy
import glob
import json
import os
import sys
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(REPO, "hpc", "pipeline", "evaluation")
if EVAL not in sys.path:
    sys.path.insert(0, EVAL)
sys.dont_write_bytecode = True

import _evalpath  # noqa: E402,F401
import ev_rec_config as RC  # noqa: E402
import ev_rec_data as RD  # noqa: E402

FIX = os.environ.get("REC_FIXTURE", "").strip()
needs_fixture = pytest.mark.skipif(not (FIX and os.path.isdir(FIX)),
                                   reason="REC_FIXTURE not set to a built rec_fixture tree")

VALID = {
    "out_root": "/tmp/x", "seed": 1, "n_post": 1000, "batch_rows": 64, "oracle_tol": 2e-3,
    "as_recorded": False, "sim_draw_rows_per_donor": 1, "edge_eps": 0.006693,
    "gate": {"donor_fracs": [0.4, 0.25, 0.25, 0.1], "n_ref": 800, "n_rank": 1000,
             "n_null": 1000, "n_null_min": 50, "rows_per_donor": 1,
             "factors": [0.25, 0.5, 1, 2, 4], "alpha": 0.05},
    "bootstrap_n": 2000, "level": 0.95, "dpi": 150, "formats": ["png", "pdf"], "ncpus": 4,
    "validate_against": None,
    "jobs": {"queue": "intel",
             "sample": {"ncpus": 8, "mem": "64gb", "walltime": "24:00:00"},
             "report": {"ncpus": 8, "mem": "64gb", "walltime": "24:00:00"},
             "validate": {"ncpus": 1, "mem": "8gb", "walltime": "01:00:00"}},
}


def _with(path, value):
    d = copy.deepcopy(VALID)
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    if value is KeyError:
        del cur[path[-1]]
    else:
        cur[path[-1]] = value
    return d


def test_config_valid_and_keys():
    assert RC.validate(VALID) == []
    assert set(RC.KEYS) == set(VALID) and set(RC.GATE_KEYS) == set(VALID["gate"])
    assert RC.JOB_KINDS == ("sample", "report", "validate")


@pytest.mark.parametrize("path,value", [
    (("gate", "donor_fracs"), [0.4, 0.25, 0.15, 0.1]),        # sums to 0.9
    (("gate", "donor_fracs"), [0.4, 0.25, 0.35, 0.0]),        # not positive
    (("edge_eps",), 0.5), (("edge_eps",), 0.0),
    (("gate", "alpha"), 1), (("gate", "alpha"), 0),
    (("n_post",), 0), (("n_post",), True), (("seed",), -1),
    (("gate", "factors"), []), (("gate", "factors"), [1, -1]),
    (("gate", "n_ref"), 1), (("gate", "n_null_min"), 0), (("gate", "rows_per_donor"), 0),
    (("validate_against",), "/abs/old"),                      # without as_recorded
    (("validate_against",), "rel/old"),
    (("oracle_tol",), 0.2), (("oracle_tol",), 0),
    (("level",), 1.0), (("out_root",), "relative"),
    (("formats",), ["png", "png"]), (("formats",), ["jpg"]), (("formats",), []),
    (("sim_draw_rows_per_donor",), 0), (("as_recorded",), 1),
    (("jobs", "validate"), KeyError), (("jobs", "sample", "mem"), "64GB"),
    (("jobs", "sample", "walltime"), "24:00"),
    (("out_root",), KeyError), (("gate", "alpha"), KeyError),
])
def test_config_each_rule_refused(path, value):
    probs = RC.validate(_with(path, value))
    assert probs, (path, value)


def test_config_unknown_and_missing_named():
    d = copy.deepcopy(VALID); d["extra"] = 1; del d["level"]; d["gate"]["zz"] = 2
    probs = " ".join(RC.validate(d))
    assert "extra" in probs and "level" in probs and "zz" in probs


def test_config_load_params_and_tuples():
    with tempfile.TemporaryDirectory() as td:
        d = copy.deepcopy(VALID); d["_doc"] = "a comment"
        p = os.path.join(td, "e.json"); json.dump(d, open(p, "w"))
        c = RC.load(p)
        sp, rp = c.sample_params(), c.report_params()
        # block 33's lists, exactly
        assert set(sp) == {"seed", "n_post", "as_recorded", "sim_draw_rows_per_donor", "gate"}
        assert set(sp["gate"]) == {"donor_fracs", "n_ref", "n_rank", "n_null", "n_null_min",
                                   "rows_per_donor"}
        assert set(rp) == {"seed", "edge_eps", "bootstrap_n", "level", "dpi", "formats", "gate"}
        assert set(rp["gate"]) == {"factors", "alpha"}
        assert "oracle_tol" not in sp and "oracle_tol" not in rp
        # the params round-trip through JSON unchanged (block 38 compares a
        # sample's recorded params with these)
        assert json.loads(json.dumps(sp)) == sp and json.loads(json.dumps(rp)) == rp


def test_config_nonfinite_factor():
    """Python's json accepts Infinity; a factor of inf is not a usable
    bandwidth factor. Recorded as an observation (see the report)."""
    d = _with(("gate", "factors"), [1.0, float("inf")])
    assert RC.validate(d) != [], "gate.factors [1, Infinity] passes validation"


# ---------------------------------------------------------------- draw rows, axes

def test_draw_rows_by_hand():
    donor = np.array([5, 5, 7, 5, 7, 9])
    assert_array_equal(np.flatnonzero(RD.draw_rows(donor, 1)), [0, 2, 5])
    assert_array_equal(np.flatnonzero(RD.draw_rows(donor, 2)), [0, 1, 2, 4, 5])
    assert RD.draw_rows(donor, None).all()
    assert_array_equal(RD.draw_rows(donor, 1), RD.calibration_rows(donor))
    assert_array_equal(np.flatnonzero(RD.draw_rows(donor, 10)), np.arange(6))
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError):
            RD.draw_rows(donor, bad)


def test_draw_rows_random_against_loop():
    rng = np.random.default_rng(1)
    donor = rng.integers(0, 9, 200)
    for k in (1, 2, 3):
        seen, exp = {}, []
        for d in donor:
            exp.append(seen.get(d, 0) < k); seen[d] = seen.get(d, 0) + 1
        assert_array_equal(RD.draw_rows(donor, k), exp)


def test_axes_of():
    names, b = RD.axes_of({"param_names": ["a", "b"], "bounds_theta": [[0, 1], [-2, 3]]}, 2)
    assert names == ["a", "b"] and b.dtype == np.float64 and b.shape == (2, 2)
    assert RD.axes_of({"param_names": [], "bounds_theta": []}, 3) == (["theta_0", "theta_1",
                                                                       "theta_2"], None)
    assert RD.axes_of({"param_names": ["a"], "bounds_theta": [[0, 1], [0, 1]]}, 2)[1] is None
    assert RD.axes_of(None, 1) == (["theta_0"], None)


# ---------------------------------------------------------------- views (fixture)

def _ecfg(name):
    return RC.load(os.path.join(FIX, name))


@needs_fixture
def test_recorded_view_on_fixture():
    import ev_rec_plan as RP
    v = RP.load_view(os.path.join(FIX, "synth_rec.json"), _ecfg("rec_eval.json"))
    assert v.kind == "recorded"
    names = [e.name for e in v.ensembles]
    assert names[0] == "S-A1:A1" and names[1].startswith("S-A2-R0:") and len(names) == 2
    for e in v.ensembles:
        assert e.banks == ("real",) and e.stem == e.name.replace(":", "_")
        assert e.folder.endswith(os.path.join(e.search, "synth_best"))
        bf = json.load(open(os.path.join(e.folder, "BEST_FIT_COMPLETE.json")))
        assert list(e.seeds) == [int(s) for s in bf["member_seeds"]]
    assert set(v.banks) == {"real"}
    assert sorted(glob.glob(v.sim_glob)) == sorted(glob.glob(os.path.join(
        FIX, "banks_root", "synth_sim", "shard_*.npz")))
    # the kind mismatch refused, naming the other evaluation
    with pytest.raises(Exception) as ei:
        RP.load_view(os.path.join(FIX, "synth.json"), _ecfg("rec_eval.json"))
    assert "submit_eval.sh" in str(ei.value)
    with pytest.raises(Exception):
        RP.load_view(os.path.join(FIX, "synth_rec.json"), _ecfg("rec_eval_as.json"))


@needs_fixture
def test_recording_on_fixture():
    import ev_rec_plan as RP
    v = RP.load_view(os.path.join(FIX, "synth_rec.json"), _ecfg("rec_eval.json"))
    r = RD.recording(v, "real")
    assert "theta" not in r
    paths = sorted(glob.glob(os.path.join(FIX, "banks_root", "synth_real", "shard_*.npz")))
    raw = [np.load(p) for p in paths]
    x = np.concatenate([z["x"] for z in raw])
    assert r["x"].dtype == np.float32 and r["x"].shape == x.shape
    assert_array_equal(r["x"], x)
    r0 = np.concatenate([np.load(p)["x"] for p in sorted(glob.glob(os.path.join(
        FIX, "banks_root", "synth_real_R0", "shard_*.npz")))])
    assert_array_equal(r["x"], r0)
    for k in ("cls", "donor", "well", "batch", "subregion", "window_idx", "rep_group"):
        assert r[k].dtype == np.int64 and r[k].shape == (x.shape[0],), k
    for k in ("cls", "donor", "well", "batch", "subregion", "window_idx"):
        assert_array_equal(r[k], np.concatenate([z[k] for z in raw]))
    assert np.all(r["rep_group"] == -1) and r["contaminated"].dtype == bool
    # the as-recorded view: a cohort has no theta either
    va = RP.load_view(os.path.join(FIX, "synth.json"), _ecfg("rec_eval_as.json"))
    for name in va.banks:
        assert "theta" not in RD.recording(va, name)
    # a bank of arm S refused
    vs = va._replace(banks={"sim": va.sim_glob})
    with pytest.raises(Exception):
        RD.recording(vs, "sim")


# ---------------------------------------------------------------- 2026-11-23_1906 (D-167 fix round)
# Oracles from block 33's amended KEYS list and build notes: gate.n_ref an
# int in [2, 1000]; gate.factors positive FINITE numbers; mem / walltime
# matched whole; an integer key requires an int (n_post 1000.0 refused); a
# non-integer key accepts int or float, never a bool.

def test_n_ref_upper_bound():
    assert RC.validate(_with(("gate", "n_ref"), 1000)) == []
    assert RC.validate(_with(("gate", "n_ref"), 2)) == []
    probs = RC.validate(_with(("gate", "n_ref"), 1001))
    assert probs and any("n_ref" in q for q in probs), probs


@pytest.mark.parametrize("value", [
    [1.0, float("inf")], [float("-inf")], [float("nan")], [1, 0], [True], ["1"]])
def test_factors_finite_positive(value):
    assert RC.validate(_with(("gate", "factors"), value)), value


def test_factors_ints_accepted():
    # a non-integer key accepts ints (build note, clarified 2026-11-23, S3)
    assert RC.validate(_with(("gate", "factors"), [1, 2, 4])) == []


def test_factors_huge_int_does_not_crash():
    """json parses a 400-digit integer literal as a Python int; validate must
    return a list (a problem or none), never raise: math.isfinite(int)
    overflows for ints beyond float range."""
    d = _with(("gate", "factors"), [10 ** 400])
    try:
        out = RC.validate(d)
    except OverflowError as exc:
        pytest.fail("validate raised OverflowError on factors [10**400]: %s" % exc)
    assert isinstance(out, list)


def test_donor_fracs_huge_int_does_not_crash():
    d = _with(("gate", "donor_fracs"), [10 ** 400, 0.25, 0.25, 0.1])
    try:
        out = RC.validate(d)
    except OverflowError as exc:
        pytest.fail("validate raised OverflowError on donor_fracs [10**400, ...]: %s" % exc)
    assert out


@pytest.mark.parametrize("kind,key,value", [
    ("sample", "mem", "64gb\n"), ("report", "mem", " 64gb"), ("validate", "mem", "64gb "),
    ("sample", "walltime", "24:00:00\n"), ("report", "walltime", "\n24:00:00"),
    ("validate", "walltime", "01:00:00 "), ("sample", "mem", "064gb"),
    ("sample", "walltime", "24:60:00")])
def test_jobs_mem_walltime_whole(kind, key, value):
    assert RC.validate(_with(("jobs", kind, key), value)), (kind, key, value)


def test_jobs_mem_walltime_valid_forms():
    for mem in ("1b", "8kb", "64gb", "2tb", "512mb"):
        assert RC.validate(_with(("jobs", "sample", "mem"), mem)) == [], mem
    for w in ("0:00:01", "100:59:59"):
        assert RC.validate(_with(("jobs", "sample", "walltime"), w)) == [], w


@pytest.mark.parametrize("path,value", [
    (("n_post",), 1000.0), (("batch_rows",), 64.0), (("seed",), 1.0),
    (("bootstrap_n",), 2000.0), (("gate", "n_ref"), 800.0), (("gate", "n_rank"), 1000.0),
    (("gate", "n_null"), 1000.0), (("gate", "n_null_min"), 50.0),
    (("gate", "rows_per_donor"), 1.0), (("sim_draw_rows_per_donor",), 1.0),
    (("jobs", "sample", "ncpus"), 8.0)])
def test_integer_keys_refuse_floats(path, value):
    assert RC.validate(_with(path, value)), (path, value)


@pytest.mark.parametrize("path,value", [
    (("edge_eps",), True), (("level",), True), (("oracle_tol",), True),
    (("gate", "alpha"), True), (("gate", "donor_fracs"), [True, 0.25, 0.25, 0.1]),
    (("edge_eps",), float("nan")), (("level",), float("nan")),
    (("oracle_tol",), float("nan")), (("gate", "alpha"), float("nan")),
    (("gate", "donor_fracs"), [0.4, 0.25, 0.25, float("nan")]),
    (("gate", "donor_fracs"), [float("inf"), 0.25, 0.25, 0.1])])
def test_noninteger_keys_refuse_bool_and_nonfinite(path, value):
    assert RC.validate(_with(path, value)), (path, value)


def test_noninteger_keys_accept_int_where_in_range():
    # oracle_tol in (0, 0.1], edge_eps in (0, 0.5), level and alpha in (0, 1):
    # no int lies in those ranges, so the only int form to accept is a
    # donor_fracs entry; [1, ...] cannot sum to 1 with positives, so the
    # int-acceptance is exercised on factors (above) only.
    assert RC.validate(_with(("gate", "donor_fracs"), [0.5, 0.25, 0.125, 0.125])) == []


def test_load_refuses_through_file():
    with tempfile.TemporaryDirectory() as td:
        d = _with(("gate", "n_ref"), 1001)
        p = os.path.join(td, "e.json"); json.dump(d, open(p, "w"))
        with pytest.raises(RC.RecEvalConfigError):
            RC.load(p)
        d = _with(("gate", "factors"), [1.0, float("inf")])
        json.dump(d, open(p, "w"))           # writes Infinity, as json allows
        with pytest.raises(RC.RecEvalConfigError):
            RC.load(p)


@needs_fixture
def test_boundary_copy_edits_only_the_copy(tmp_path):
    """rec_fixture.boundary_copy (block 33's fix round): the copy's simulated
    bank has exactly the named theta entries changed; the fixture's own bank
    is not written."""
    sys.path.insert(0, os.path.join(REPO, "hpc", "pipeline", "tests"))
    os.environ["REC_FIXTURE"] = FIX
    import rec_fixture
    cfg = json.load(open(os.path.join(FIX, "synth_rec.json")))
    src = sorted(glob.glob(os.path.join(cfg["paths"]["banks_root"], "*_sim", "shard_*.npz")))
    before = [np.load(f)["theta"].copy() for f in src]
    mt = [os.path.getmtime(f) for f in src]
    th_all = np.concatenate(before)
    edits = [(3, 0, 1.0), (th_all.shape[0] - 1, th_all.shape[1] - 1, 0.0)]
    path = rec_fixture.boundary_copy(str(tmp_path), edits)
    c2 = json.load(open(path))
    assert c2["paths"]["banks_root"] == os.path.join(str(tmp_path), "banks_root")
    dst = sorted(glob.glob(os.path.join(str(tmp_path), "banks_root", "*_sim", "shard_*.npz")))
    after = np.concatenate([np.load(f)["theta"] for f in dst])
    exp = th_all.copy()
    for r, k, v in edits:
        exp[r, k] = v
    np.testing.assert_array_equal(after, exp)
    assert after.dtype == th_all.dtype
    for f, b, m in zip(src, before, mt):
        np.testing.assert_array_equal(np.load(f)["theta"], b)
        assert os.path.getmtime(f) == m


# ===== fix round of 2026-11-28 (the tester's F7): the search for the first non-finite
# theta skips a shard on the reader's errors only [2026-11-28_0151]

def _shards(tmp_path):
    d = tmp_path / "bank"
    d.mkdir()
    (d / "shard_0000.npz").write_bytes(b"not a zip at all")        # unreadable
    th = np.zeros((4, 3))
    th[2, 1] = np.nan
    np.savez(d / "shard_0001.npz", theta=th)
    return str(d / "shard_*.npz")


def test_first_bad_theta_skips_unreadable_shards_and_names_the_first(tmp_path):
    pat = _shards(tmp_path)
    msg = RD._first_bad_theta(pat)
    assert "shard_0001.npz" in msg and "row 2" in msg and "axis 1" in msg, msg


@pytest.mark.parametrize("exc", ["OSError", "ValueError", "KeyError", "EOFError",
                                 "BadZipFile"])
def test_first_bad_theta_reader_errors_are_skipped(tmp_path, monkeypatch, exc):
    import zipfile
    E = zipfile.BadZipFile if exc == "BadZipFile" else __builtins__[exc] \
        if isinstance(__builtins__, dict) else getattr(__builtins__, exc)
    pat = _shards(tmp_path)
    real = np.load

    def fake(path, *a, **k):
        if str(path).endswith("shard_0000.npz"):
            raise E("planted")
        return real(path, *a, **k)
    monkeypatch.setattr(RD.np, "load", fake)
    assert "shard_0001.npz" in RD._first_bad_theta(pat)


@pytest.mark.parametrize("exc", [RuntimeError, TypeError, MemoryError, AttributeError])
def test_first_bad_theta_other_errors_are_not_swallowed(tmp_path, monkeypatch, exc):
    """Block 33's fix round: not on every exception. Another error of the read propagates."""
    pat = _shards(tmp_path)
    real = np.load

    def fake(path, *a, **k):
        if str(path).endswith("shard_0000.npz"):
            raise exc("planted")
        return real(path, *a, **k)
    monkeypatch.setattr(RD.np, "load", fake)
    with pytest.raises(exc):
        RD._first_bad_theta(pat)
