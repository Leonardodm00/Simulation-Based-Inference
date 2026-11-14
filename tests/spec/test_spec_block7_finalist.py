"""Spec-derived tests: block 7 amendment (i) (D-134 (2)) and sec. 2.2 rule 5
(block 1's `--retry`), specs/SPEC.md.

Oracles (written from the spec before the implementation was read):
- rule 5: `--retry s` returns to pending every stage downstream of s
  (direct or transitive dependency) whose status is done or skipped; a
  downstream stage pending or failed is left as it is; a downstream stage
  with a live job refuses the retry (exit 2, nothing changed); the reset
  forgets the stage's jobs and its `sub`; one log line per returned stage.
  Checked here on graphs the smoke test does not use: a diamond, two
  branches, a retry naming two stages one downstream of the other, a
  downstream stage absent from the state.
- done(S8)/done(F_q): artefact exists, SEARCH_COMPLETE exists, the artefact's
  finalist.trial_id is SEARCH_COMPLETE's finalist.trial_id; done(G_q):
  artefact exists and its recorded search_finalist.trial_id is
  SEARCH_COMPLETE's. The reason names the failing condition.
- the start: a root holding a finalist_config.json or BEST_FIT_COMPLETE.json
  naming another trial, or member records without a readable
  finalist_config.json naming the current finalist, or an unreadable /
  trial-less file, is moved aside whole to <root>.set_aside<k> (first free
  k, never overwritten), every file's bytes kept; a root of the current
  finalist is kept.
- the dry run moves nothing and says the root would be moved aside.

Run: python3 -m pytest -q tests/spec/test_spec_block7_finalist.py
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
TESTS = os.path.join(PIPE, "tests")
for p in (PIPE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PIPELINE_SKIP_CONDA", "1")

import config as C                          # noqa: E402
import pipeline as P                        # noqa: E402
import stages as S                          # noqa: E402
from stages.best_fit import BestFitStage    # noqa: E402
from stages import synthetic as SY          # noqa: E402
from state import State                     # noqa: E402

GIULIA = os.path.join(PIPE, "projects", "giulia", "giulia.json")


# --------------------------------------------------------------- fixtures ---

class Log(object):
    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(str(text))

    log = __call__


class _JS(object):
    def __init__(self, finished):
        self.finished = finished


class FakePBS(object):
    def __init__(self, live=()):
        self.live = set(live)

    def status(self, job_id):
        return _JS(job_id not in self.live)


class Toy(S.Stage):
    def __init__(self, name, deps):
        self.name = name
        self.deps = tuple(deps)

    def is_done(self, cfg):
        return True, "toy"

    def start(self, cfg, st, pbs):
        raise AssertionError("started")


class ToyCfg(object):
    def __init__(self, root):
        self.data = {"paths": {"work_root": os.path.join(root, "work")}, "dataset": "toy"}
        self.path = os.path.join(root, "toy.json")
        self.sha256 = "1" * 64
        self.tag = "toy"


def make_state(cfg, graph, statuses, live=()):
    st = State.new(cfg.data["paths"]["work_root"], "toy", cfg.path, cfg.sha256)
    for g in graph:
        if g.name not in statuses:
            continue
        rec = st.stage(g.name)
        rec["status"] = statuses[g.name]
        rec["sub"] = {"mark": g.name}
        rec["note"] = "n-" + g.name
        rec["jobs"] = [{"id": g.name + ".1", "role": "x", "finished": (g.name + ".1") not in live,
                        "cmd": [], "log": None}]
    st.save()
    return st


@pytest.fixture
def root():
    d = tempfile.mkdtemp(prefix="spec_b7i_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------- rule 5: graph ---

def diamond():
    # a -> b -> d, a -> c -> d, d -> e ; x independent ; y depends on x and b
    return [Toy("a", ()), Toy("x", ()), Toy("b", ("a",)), Toy("c", ("a",)),
            Toy("d", ("b", "c")), Toy("e", ("d",)), Toy("y", ("x", "b"))]


def test_downstream_of_by_hand():
    g = diamond()
    assert set(P.downstream_of(g, ["a"])) == {"b", "c", "d", "e", "y"}
    assert set(P.downstream_of(g, ["c"])) == {"d", "e"}
    assert set(P.downstream_of(g, ["x"])) == {"y"}
    assert set(P.downstream_of(g, ["e"])) == set()
    assert set(P.downstream_of(g, ["b"])) == {"d", "e", "y"}


def test_retry_resets_done_and_skipped_leaves_pending_failed(root):
    cfg = ToyCfg(root)
    g = diamond()
    statuses = {"a": "done", "x": "done", "b": "done", "c": "skipped", "d": "failed",
                "e": "done", "y": "pending"}
    make_state(cfg, g, statuses)
    wr = cfg.data["paths"]["work_root"]
    log = Log()
    st, rc = P._prepare_state(cfg, g, wr, ["a"], False, FakePBS(), log, dry_run=False)
    assert rc is None
    got = dict((k, v["status"]) for k, v in st.data["stages"].items())
    # by hand: b (done), c (skipped), e (done, through d) return; d failed and y pending stay; x not downstream
    assert got == {"a": "pending", "x": "done", "b": "pending", "c": "pending", "d": "failed",
                   "e": "pending", "y": "pending"}, got
    for n in ("b", "c", "e"):
        assert st.data["stages"][n]["jobs"] == [] and not st.data["stages"][n].get("sub"), n
    assert st.data["stages"]["d"]["sub"] == {"mark": "d"}       # left as it is
    assert st.data["stages"]["y"]["sub"] == {"mark": "y"}
    assert st.data["stages"]["x"]["sub"] == {"mark": "x"}
    returned = [ln for ln in log.lines if "returned to pending" in ln]
    assert len(returned) == 3, log.lines                         # one line per stage returned
    for n in ("b", "c", "e"):
        assert sum(ln.startswith(n + " returned to pending") for ln in returned) == 1, (n, returned)


def test_retry_refused_by_live_job_in_failed_downstream(root):
    cfg = ToyCfg(root)
    g = diamond()
    statuses = dict((n.name, "done") for n in g)
    statuses["d"] = "failed"
    make_state(cfg, g, statuses, live=("d.1",))
    wr = cfg.data["paths"]["work_root"]
    before = open(State.path_of(wr), "rb").read()
    log = Log()
    rc = P.run(cfg, graph=g, retry=["a"], pbs_module=FakePBS(live=["d.1"]), poll_seconds=0.01,
               max_ticks=2, log=log)
    assert rc == 2, log.lines
    assert any("downstream" in ln and "d.1" in ln and ln.startswith("REFUSE") for ln in log.lines), log.lines
    assert open(State.path_of(wr), "rb").read() == before


def test_retry_two_names_one_downstream_and_absent_stage(root):
    cfg = ToyCfg(root)
    g = diamond()
    statuses = {"a": "done", "x": "done", "b": "done", "c": "done", "d": "done"}  # e, y never recorded
    make_state(cfg, g, statuses)
    wr = cfg.data["paths"]["work_root"]
    log = Log()
    st, rc = P._prepare_state(cfg, g, wr, ["b", "a"], False, FakePBS(), log, dry_run=False)
    assert rc is None, log.lines
    got = dict((k, v["status"]) for k, v in st.data["stages"].items())
    assert got["a"] == "pending" and got["b"] == "pending" and got["c"] == "pending" and got["d"] == "pending"
    assert got["x"] == "done"
    # a stage absent from the state is not created by the retry (no record, no job)
    assert "e" not in st.data["stages"] and "y" not in st.data["stages"], sorted(st.data["stages"])


def test_retry_dry_run_changes_nothing(root):
    cfg = ToyCfg(root)
    g = diamond()
    make_state(cfg, g, dict((n.name, "done") for n in g))
    wr = cfg.data["paths"]["work_root"]
    before = open(State.path_of(wr), "rb").read()
    rc = P.run(cfg, graph=g, retry=["a"], pbs_module=FakePBS(), dry_run=True, log=Log())
    assert rc in (0, 3)
    assert open(State.path_of(wr), "rb").read() == before


# ------------------------------------------------------ done predicate ---

def _write(path, obj):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh)


def rec_cfg(root):
    with open(GIULIA, "rb") as fh:
        d = C._strip_docs(json.loads(fh.read().decode("ascii")))
    for k in ("work_root", "runs_root", "tune_root"):
        d["paths"][k] = os.path.join(root, k)
    path = os.path.join(root, "rec.json")
    _write(path, d)
    return C.load_config(path)


def synth_cfg(root):
    import test_smoke_synthetic as T            # the reduced-project fixture builder
    search = {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": True}
    path, _ = T._small(root, searches=[search])
    return C.load_config(path, synthetic_ok=True)


def test_is_done_table_s8(root):
    cfg = rec_cfg(root)
    stg = BestFitStage()
    dirs = stg._dirs(cfg)
    art, sc = dirs["complete"], dirs["search_complete"]
    rows = [
        # (SEARCH_COMPLETE content or None, artefact content or None, done?, words in the reason)
        ({"finalist": {"trial_id": "T1"}}, None, False, ["absent"]),
        ({"finalist": {"trial_id": "T1"}}, {"finalist": {"trial_id": "T1"}}, True, []),
        ({"finalist": {"trial_id": "T1"}}, {"finalist": {"trial_id": "T2"}}, False, ["T1", "T2"]),
        (None, {"finalist": {"trial_id": "T1"}}, False, ["absent", "SEARCH_COMPLETE"]),
        ({"finalist": {"trial_id": "T1"}}, {"pre_made": True}, False, ["T1"]),
        ({"finalist": {"trial_id": "T1"}}, "{not json", False, ["unreadable"]),
        ({"finalist": {"trial_id": "T1"}}, [1, 2], False, []),
        ({"finalist": {}}, {"finalist": {"trial_id": "T1"}}, False, []),
        ("", {"finalist": {"trial_id": "T1"}}, False, []),
    ]
    for scc, artc, want, words in rows:
        for p in (art, sc):
            if os.path.exists(p):
                os.remove(p)
        if scc is not None:
            _write(sc, scc)
        if artc is not None:
            _write(art, artc)
        done, why = stg.is_done(cfg)
        assert done is want, (scc, artc, done, why)
        for w in words:
            assert w in why, (w, why)


def test_is_done_table_fq_gq(root):
    cfg = synth_cfg(root)
    by = dict((s_.name, s_) for s_ in S.default_graph(cfg))
    fq, gq = by["best_fit_S-A2-R0"], by["best_fit_alt_S-A2-R0"]
    v = fq._view(cfg)
    sc = fq._dirs(v)["search_complete"]
    _write(sc, {"finalist": {"trial_id": "T1"}})
    art_f, art_g = fq._dirs(v)["complete"], gq._dirs(v)["complete"]
    assert art_f != art_g
    _write(art_f, {"finalist": {"trial_id": "T1"}})
    assert fq.is_done(cfg)[0] is True
    _write(art_f, {"finalist": {"trial_id": "T9"}})
    assert fq.is_done(cfg)[0] is False
    # G_q: the recorded search finalist decides, not the fitted trial
    _write(art_g, {"finalist": {"trial_id": "A7", "search_finalist": {"trial_id": "T1"}}})
    assert gq.is_done(cfg)[0] is True
    _write(art_g, {"finalist": {"trial_id": "A7", "search_finalist": {"trial_id": "T2"}}})
    assert gq.is_done(cfg)[0] is False
    _write(art_g, {"finalist": {"trial_id": "A7"}})                 # nothing recorded
    assert gq.is_done(cfg)[0] is False
    # spec-gap probe (reported, not a contract): G_q's artefact of a trial
    # that is no longer the other arm's best still counts as done while the
    # search's finalist is the same; is_done never reads the ledger
    _write(art_g, {"finalist": {"trial_id": "STALE", "search_finalist": {"trial_id": "T1"}}})
    assert gq.is_done(cfg)[0] is True


# --------------------------------------------------------------- start ---

def sha_tree(root):
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def _prep_root(cfg, stg, files):
    dirs = stg._dirs(cfg)
    if os.path.lexists(dirs["root"]):
        shutil.rmtree(dirs["root"])
    for rel, obj in files.items():
        _write(os.path.join(dirs["root"], rel), obj)
    return dirs


MEM = "member_seed1/A1_split1_seed1.json"


@pytest.mark.parametrize("files,moved", [
    ({"finalist_config.json": {"trial_id": "OLD"}}, True),
    ({"BEST_FIT_COMPLETE.json": {"finalist": {"trial_id": "OLD"}}}, True),
    ({"BEST_FIT_COMPLETE.json": {"pre_made": True}}, True),
    ({MEM: {"L": 1.0}}, True),                                            # member record, no config
    ({"finalist_config.json": "garbage{", MEM: {"L": 1.0}}, True),       # unreadable config
    ({"finalist_config.json": "garbage{"}, True),                         # unreadable, alone
    ({"finalist_config.json": {"no": "trial"}}, True),                    # names no trial
    ({"finalist_config.json": {"trial_id": "CUR"}, MEM: {"L": 1.0}}, False),
    ({"finalist_config.json": {"trial_id": "CUR"},
      "BEST_FIT_COMPLETE.json": {"finalist": {"trial_id": "CUR"}}}, False),
    ({"jobs/x.log": "log"}, False),                                       # no record of any finalist
])
def test_follow_finalist_s8(root, files, moved):
    cfg = rec_cfg(root)
    stg = BestFitStage()
    dirs = stg._dirs(cfg)
    _write(dirs["search_complete"], {"finalist": {"trial_id": "CUR", "nll": 1.0, "config": {}},
                                     "campaign": "S-A1", "identity": {}})
    _prep_root(cfg, stg, files)
    before = sha_tree(dirs["root"])
    st = {"sub": {}, "jobs": []}
    log = Log()
    stg._follow_finalist(cfg, st, log)
    aside = dirs["root"] + ".set_aside1"
    if moved:
        assert os.path.isdir(aside) and not os.path.lexists(dirs["root"]), log.lines
        assert sha_tree(aside) == before                       # every file, its bytes unchanged
        assert len(log.lines) == 1 and "CUR" in log.lines[0] and aside in log.lines[0], log.lines
    else:
        assert not os.path.lexists(aside) and sha_tree(dirs["root"]) == before, log.lines


def test_follow_finalist_second_set_aside_never_overwrites(root):
    cfg = rec_cfg(root)
    stg = BestFitStage()
    dirs = stg._dirs(cfg)
    _write(dirs["search_complete"], {"finalist": {"trial_id": "CUR"}})
    _prep_root(cfg, stg, {"finalist_config.json": {"trial_id": "OLD1"}})
    stg._follow_finalist(cfg, {"sub": {}}, Log())
    first = sha_tree(dirs["root"] + ".set_aside1")
    _prep_root(cfg, stg, {"finalist_config.json": {"trial_id": "OLD2"}})
    stg._follow_finalist(cfg, {"sub": {}}, Log())
    assert sha_tree(dirs["root"] + ".set_aside1") == first
    assert json.load(open(dirs["root"] + ".set_aside2/finalist_config.json"))["trial_id"] == "OLD2"


def test_follow_finalist_forgets_cached_other_finalist(root):
    """The state's cached finalist of another trial is not reused (no member
    record of another finalist is ever reused)."""
    cfg = rec_cfg(root)
    stg = BestFitStage()
    dirs = stg._dirs(cfg)
    _write(dirs["search_complete"], {"finalist": {"trial_id": "CUR"}})
    st = {"sub": {"finalist": {"trial_id": "OLD"}, "members": {"1": {"jobs": ["j"]}}}}
    stg._follow_finalist(cfg, st, Log())
    assert "finalist" not in st["sub"] or st["sub"]["finalist"].get("trial_id") == "CUR"
    assert st["sub"].get("members", {}) == {}


def test_dry_run_moves_nothing(root):
    cfg = rec_cfg(root)
    stg = BestFitStage()
    dirs = stg._dirs(cfg)
    _write(dirs["search_complete"], {"finalist": {"trial_id": "CUR"}})
    _prep_root(cfg, stg, {"finalist_config.json": {"trial_id": "OLD"}, MEM: {"L": 1}})
    before = sha_tree(dirs["root"])
    lines = stg.dry_run(cfg, {"sub": {}, "jobs": []})
    assert any("moved aside" in ln and "OLD" in ln and "CUR" in ln for ln in lines), lines
    assert sha_tree(dirs["root"]) == before and not os.path.lexists(dirs["root"] + ".set_aside1")
