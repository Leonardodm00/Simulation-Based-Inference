"""Config v11 (specs/SPEC.md sec. 3.1; D-148; blocks 29, 31, 32 and the
shared config.py of blocks 1, 2, 15-18, 24-26).

Oracles (written from the spec before the code was read):
  Ov11.1 config_version 11; real.cohort_map an absolute path or null;
         bank.drop_partial_topologies a bool; replicate.pairs in
         {within_group, cross_well}; replicate.group_key in {rep_group, donor,
         batch, well}; every key required, a missing / unknown / wrong key a
         named problem, every problem listed at once.
  Ov11.2 the synthetic kind keeps replicate.* and refuses `searches`.
  O29.10 `searches`: null or a list of objects with exactly {name, campaign,
         max_cores}; name [A-Za-z0-9][A-Za-z0-9_-]*; campaign one of the six;
         max_cores an int >= 1 or null; unique names and campaigns; an int
         max_cores >= max(search.trial.ncpus, best_fit.ncpus) and <=
         resources.max_cores - driver.ncpus; with a list, search.campaign and
         search.arm are refused; with null they are required.
  Ov11.5 the committed giulia, linddup and synth files load, with the values
         sec. 4 / blocks 29 and 31 state.
"""

import copy
import shutil

import pytest

import _spec_util as U

import config as C   # noqa: E402  (the pipeline's, put first on sys.path by _spec_util)


def _problems(d, root, synthetic=False):
    path = U.write_json(U.os.path.join(root, "cfg.json"), d)
    try:
        C.load_config(path, synthetic_ok=synthetic)
    except C.ConfigError as exc:
        return list(exc.problems)
    return []


@pytest.fixture
def root():
    r = U.tmpdir("spec_cfg_")
    yield r
    shutil.rmtree(r, ignore_errors=True)


def _linddup(root):
    return U.moved(U.strip_docs(U.raw_json(U.LINDDUP)), root)


def _giulia(root):
    return U.moved(U.strip_docs(U.raw_json(U.GIULIA)), root)


def _synth(root):
    d = U.strip_docs(U.raw_json(U.SYNTH))
    for k in list(d["paths"]):
        d["paths"][k] = U.os.path.join(root, k)
    return d


# ---------------------------------------------------------------- Ov11.5 ---

def test_committed_files_load_at_v11():
    assert C.CONFIG_VERSION == 11
    for path, synth in ((U.GIULIA, False), (U.LINDDUP, False), (U.SYNTH, True)):
        cfg = C.load_config(path, synthetic_ok=synth)
        assert cfg.data["config_version"] == 11, path


def test_committed_values_of_v11():
    lin = C.load_config(U.LINDDUP).data
    # sec. 4 D-138..D-151 and block 29: A1, A2, A5 each searched, 148 cores each
    assert lin["searches"] == [{"name": "A1", "campaign": "S-A1", "max_cores": 148},
                               {"name": "A2", "campaign": "S-A2-only", "max_cores": 148},
                               {"name": "A5", "campaign": "S-A5-only", "max_cores": 148}]
    assert "campaign" not in lin["search"] and "arm" not in lin["search"]
    assert lin["search"]["max_total_walltime_h"] is None          # D-141: no clock cap
    assert lin["replicate"] == {"pairs": "within_group", "group_key": "rep_group"}   # D-148
    assert lin["bank"]["drop_partial_topologies"] is True          # D-148 (5)
    cm = lin["real"]["cohort_map"]
    assert isinstance(cm, str) and cm.startswith("/") and cm.endswith("linddup_cohort_map.csv")
    # each cap within the project's share (sec. 3.1)
    room = lin["resources"]["max_cores"] - lin["driver"]["ncpus"]
    largest = max(lin["search"]["trial"]["ncpus"], lin["best_fit"]["ncpus"])
    for q in lin["searches"]:
        assert largest <= q["max_cores"] <= room
    gi = C.load_config(U.GIULIA).data
    assert gi["searches"] is None and gi["real"]["cohort_map"] is None
    assert gi["bank"]["drop_partial_topologies"] is False
    assert gi["replicate"] == {"pairs": "within_group", "group_key": "donor"}
    sy = C.load_config(U.SYNTH, synthetic_ok=True).data
    assert sy["replicate"] == {"pairs": "cross_well", "group_key": "donor"}   # sec. 3.1


def test_cohort_map_csv_is_committed_with_block9_columns():
    with open(U.COHORT_MAP, "rb") as fh:
        raw = fh.read()
    raw.decode("ascii")                       # pure ASCII (sec. 0's rule for shipped files)
    assert b"\r" not in raw
    header = raw.decode("ascii").splitlines()[0].split(",")
    for col in ("out_dir", "donor", "rep_group", "include"):     # sec. 3.1's row of real.cohort_map
        assert col in header, header


# ---------------------------------------------------------------- Ov11.1 ---

def test_version_10_refused(root):
    d = _linddup(root)
    d["config_version"] = 10
    probs = _problems(d, root)
    assert any(p.startswith("config_version") for p in probs), probs


@pytest.mark.parametrize("value,ok", [
    (None, True), ("/abs/map.csv", True), ("rel/map.csv", False), ("~/map.csv", False),
    (5, False), ("", False)])
def test_cohort_map_type(root, value, ok):
    d = _linddup(root)
    d["real"]["cohort_map"] = value
    probs = _problems(d, root)
    named = [p for p in probs if p.startswith("real.cohort_map")]
    assert (not named) == ok, probs


def test_cohort_map_missing_refused(root):
    d = _linddup(root)
    del d["real"]["cohort_map"]
    assert any(p.startswith("real.cohort_map") for p in _problems(d, root))


@pytest.mark.parametrize("value,ok", [(True, True), (False, True), ("true", False), (1, False),
                                      (0, False), (None, False)])
def test_drop_partial_topologies_is_bool(root, value, ok):
    d = _linddup(root)
    d["bank"]["drop_partial_topologies"] = value
    probs = _problems(d, root)
    named = [p for p in probs if p.startswith("bank.drop_partial_topologies")]
    assert (not named) == ok, probs


@pytest.mark.parametrize("pairs", ["within_group", "cross_well"])
@pytest.mark.parametrize("key", ["rep_group", "donor", "batch", "well"])
def test_replicate_values_accepted(root, pairs, key):
    d = _linddup(root)
    d["replicate"] = {"pairs": pairs, "group_key": key}
    assert _problems(d, root) == []


@pytest.mark.parametrize("field,value", [("pairs", "any"), ("pairs", "within-group"),
                                         ("group_key", "cohort"), ("group_key", "Donor"),
                                         ("pairs", None), ("group_key", 3)])
def test_replicate_values_refused(root, field, value):
    d = _linddup(root)
    d["replicate"][field] = value
    probs = _problems(d, root)
    assert any(p.startswith("replicate." + field) for p in probs), probs


@pytest.mark.parametrize("field", ["pairs", "group_key"])
def test_replicate_missing_refused(root, field):
    d = _linddup(root)
    del d["replicate"][field]
    assert any(p.startswith("replicate." + field) for p in _problems(d, root))


def test_every_problem_listed_at_once(root):
    d = _linddup(root)
    d["replicate"]["pairs"] = "any"
    d["bank"]["drop_partial_topologies"] = "yes"
    d["real"]["cohort_map"] = "relative.csv"
    probs = _problems(d, root)
    for key in ("replicate.pairs", "bank.drop_partial_topologies", "real.cohort_map"):
        assert any(p.startswith(key) for p in probs), (key, probs)


# ---------------------------------------------------------------- Ov11.2 ---

def test_synthetic_keeps_replicate_and_refuses_searches(root):
    d = _synth(root)
    del d["replicate"]
    assert any(p.startswith("replicate") for p in _problems(d, root, synthetic=True))
    d = _synth(root)
    d["searches"] = None
    assert any(p.startswith("searches") for p in _problems(d, root, synthetic=True))
    d = _synth(root)
    d["replicate"] = {"pairs": "within_group", "group_key": "rep_group"}
    assert _problems(d, root, synthetic=True) == []


# ---------------------------------------------------------------- O29.10 ---

def _with_searches(root, searches):
    d = _linddup(root)
    d["searches"] = searches
    return d


GOOD = [{"name": "A1", "campaign": "S-A1", "max_cores": 148},
        {"name": "A2", "campaign": "S-A2-only", "max_cores": None}]


def test_searches_good_list_accepted(root):
    assert _problems(_with_searches(root, copy.deepcopy(GOOD)), root) == []


@pytest.mark.parametrize("item", [
    {"name": "A1", "campaign": "S-A1"},                                          # missing max_cores
    {"name": "A1", "campaign": "S-A1", "max_cores": 148, "real": "R0"},          # unknown key
    {"name": "-A1", "campaign": "S-A1", "max_cores": 148},                       # bad first char
    {"name": "A 1", "campaign": "S-A1", "max_cores": 148},
    {"name": "", "campaign": "S-A1", "max_cores": 148},
    {"name": "A1", "campaign": "S-A3", "max_cores": 148},                        # unknown campaign
    {"name": "A1", "campaign": "S-A1", "max_cores": 0},
    {"name": "A1", "campaign": "S-A1", "max_cores": -48},
    {"name": "A1", "campaign": "S-A1", "max_cores": 148.0},
    {"name": "A1", "campaign": "S-A1", "max_cores": True},
    {"name": "A1", "campaign": "S-A1", "max_cores": "148"},
    "A1",
])
def test_searches_bad_item_refused(root, item):
    probs = _problems(_with_searches(root, [item]), root)
    assert any(p.startswith("searches") for p in probs), probs


def test_searches_name_with_trailing_newline_refused(root):
    """The name rule `[A-Za-z0-9][A-Za-z0-9_-]*` has no newline: "A1\\n" is not
    a name (it would become the stage `search_A1\\n` and a folder `A1\\n`)."""
    probs = _problems(_with_searches(root, [{"name": "A1\n", "campaign": "S-A1",
                                             "max_cores": 148}]), root)
    assert any(p.startswith("searches") for p in probs), probs


def test_searches_unique_names_and_campaigns(root):
    probs = _problems(_with_searches(root, [
        {"name": "A1", "campaign": "S-A1", "max_cores": 148},
        {"name": "A1", "campaign": "S-A2-only", "max_cores": 148}]), root)
    assert any(p.startswith("searches") and "A1" in p for p in probs), probs
    probs = _problems(_with_searches(root, [
        {"name": "X", "campaign": "S-A1", "max_cores": 148},
        {"name": "Y", "campaign": "S-A1", "max_cores": 148}]), root)
    assert any(p.startswith("searches") and "S-A1" in p for p in probs), probs


def test_searches_cap_bounds(root):
    d = _linddup(root)
    largest = max(d["search"]["trial"]["ncpus"], d["best_fit"]["ncpus"])        # 48
    room = d["resources"]["max_cores"] - d["driver"]["ncpus"]                     # 440
    for cap, ok in ((largest - 1, False), (largest, True), (room, True), (room + 1, False)):
        dd = _with_searches(root, [{"name": "A1", "campaign": "S-A1", "max_cores": cap}])
        probs = _problems(dd, root)
        assert (probs == []) == ok, (cap, probs)
    # the larger of the two sizes counts: best_fit.ncpus above search.trial.ncpus
    dd = _with_searches(root, [{"name": "A1", "campaign": "S-A1", "max_cores": 60}])
    dd["best_fit"]["ncpus"] = 64
    assert any(p.startswith("searches") for p in _problems(dd, root))


def test_list_refuses_single_search_keys(root):
    for key, val in (("campaign", "S-A1"), ("arm", "A1")):
        d = _linddup(root)
        d["search"][key] = val
        probs = _problems(d, root)
        assert any(p.startswith("search." + key) for p in probs), (key, probs)


def test_null_requires_single_search_keys(root):
    d = _linddup(root)
    d["searches"] = None
    probs = _problems(d, root)
    assert any(p.startswith("search.campaign") for p in probs), probs
    assert any(p.startswith("search.arm") for p in probs), probs
    d["search"]["campaign"] = "S-A1"
    d["search"]["arm"] = "A1"
    assert _problems(d, root) == []


def test_searches_missing_refused(root):
    d = _linddup(root)
    del d["searches"]
    probs = _problems(d, root)
    assert any(p.startswith("searches") for p in probs), probs


# ----------------------------------------------- amendment 5 (2026-11-17_1730) ---
# Sec. 3.1, amendment 5: a `searches` list is non-empty; every name rule --
# searches[].name, synthetic.searches[].name, synthetic.real[].name, the dataset
# tag, the label axes -- matches the WHOLE string (a trailing newline refused).
# Oracle: the spec's patterns, applied here with re.fullmatch on ASCII classes.

import re as _re

_RUN = r"[A-Za-z0-9][A-Za-z0-9_-]*"
_REAL = r"[A-Za-z0-9]+"
_TAGP = r"[a-z0-9_]+"


def _spec_ok(pattern, s):
    return _re.fullmatch(pattern, s, flags=_re.ASCII) is not None


_NAMES = ["A1", "Q-A2-R9", "a_b-c", "0", "A1\n", "A1\r", "A1 ", " A1", "-A1", "_A1", "A\n1",
          "A1٣", "Ａ１", "A.1", "", "A1\t", "A1\x00"]


def test_searches_empty_list_refused(root):
    probs = _problems(_with_searches(root, []), root)
    assert any(p.startswith("searches:") for p in probs), probs


@pytest.mark.parametrize("name", _NAMES)
def test_recorded_search_name_whole_string(root, name):
    probs = _problems(_with_searches(root, [{"name": name, "campaign": "S-A1", "max_cores": 148}]),
                      root)
    assert (not any(p.startswith("searches") for p in probs)) == _spec_ok(_RUN, name), (name, probs)


@pytest.mark.parametrize("name", _NAMES)
def test_synthetic_search_name_whole_string(root, name):
    d = _synth(root)
    d["synthetic"]["searches"][0]["name"] = name
    probs = _problems(d, root, synthetic=True)
    bad = [p for p in probs if p.startswith("synthetic.searches")]
    assert (not bad) == _spec_ok(_RUN, name), (name, probs)


@pytest.mark.parametrize("name", _NAMES)
def test_synthetic_real_name_whole_string(root, name):
    d = _synth(root)
    old = d["synthetic"]["real"][0]["name"]
    d["synthetic"]["real"][0]["name"] = name
    for q in d["synthetic"]["searches"]:          # keep the searches' references valid
        if q["real"] == old:
            q["real"] = name
    probs = _problems(d, root, synthetic=True)
    bad = [p for p in probs if p.startswith("synthetic.real")]
    assert (not bad) == _spec_ok(_REAL, name), (name, probs)


@pytest.mark.parametrize("tag", ["linddup", "lind_dup2", "linddup\n", "LindDUP", "lind-dup", "",
                                 "linddup\r", "linddup ", "lind٣"])
def test_dataset_tag_whole_string(root, tag):
    d = _linddup(root)
    d["dataset"] = tag
    probs = _problems(d, root)
    bad = [p for p in probs if p.startswith("dataset")]
    assert (not bad) == _spec_ok(_TAGP, tag), (tag, probs)


@pytest.mark.parametrize("axis", ["depth_exponent", "depth_exponent\n", "Depth", "depth-x"])
def test_space_override_axis_name_whole_string(root, axis):
    """The axes of search.space_overrides (the code's other name rule, [a-z0-9_]+)."""
    d = _giulia(root)
    d["search"]["space_overrides"] = {axis: [3, 5]}
    probs = _problems(d, root)
    bad = [p for p in probs if p.startswith("search.space_overrides") and "axis name" in p]
    assert (not bad) == _spec_ok(_TAGP, axis), (axis, probs)


def test_three_bad_names_listed_at_once(root):
    d = _synth(root)
    d["dataset"] = "synth\n"
    d["synthetic"]["searches"][0]["name"] = "S-A1\n"
    d["synthetic"]["searches"][1]["name"] = "S A2"
    probs = _problems(d, root, synthetic=True)
    assert any(p.startswith("dataset") for p in probs), probs
    s = [p for p in probs if p.startswith("synthetic.searches")]
    assert len(s) == 1 and "[0]" in s[0] and "[1]" in s[0], probs


# The value formats of sec. 3.1 (`hh:mm:ss`, PBS memory `<n>gb`) are not name
# rules, but the same reading of the table applies: a value with a trailing
# newline is not `hh:mm:ss`. Recorded as a NOTE if refused nowhere.
@pytest.mark.parametrize("key, value", [
    (("search", "trial", "walltime"), "120:00:00\n"),
    (("best_fit", "mem"), "200gb\n"),
    (("driver", "walltime"), "24:00:00\n"),
])
def test_walltime_and_mem_whole_string(root, key, value):
    d = _giulia(root)
    o = d
    for k in key[:-1]:
        o = o[k]
    o[key[-1]] = value
    probs = _problems(d, root)
    assert any(p.startswith(".".join(key)) for p in probs), (key, probs)
