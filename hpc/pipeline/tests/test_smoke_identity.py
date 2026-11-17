#!/usr/bin/env python3
"""Smoke test for block 7's amendment (j) of 2026-11-14 (D-134 (1), wave W1,
block 1b) in specs/SPEC.md: the search's identity is re-checked at the
stage's start and at the top of every poll against the configuration, on
the fields that come from it, K_cfg = {split_fracs, split_seed, campaign,
shape_anchors, space_overrides, flow}.

Checks (SPEC block 7, amendment (j), "Test oracles")
    test_known_answer   the comparison on a table: each field of K_cfg
                        changed alone (that key named, both values), two
                        changed (both named), none changed (no refusal),
                        0.85 written 0.8500 in the file (no refusal), the
                        fractions in another order (refused), a key
                        outside K_cfg changed (no refusal); the start and
                        the poll of a stage whose state holds the identity,
                        under a changed configuration, on a scheduler that
                        fails the test if touched
    test_reference      end to end on a reduced search (2 rounds: 2 random
                        trials, then 1 by the GP; one trial at a time) under
                        the fake PBS with the real tuner and runner: stopped
                        while round 0 runs, split.fractions changed, run
                        again with --accept-config-change: the STOP line
                        carries the REFUSE text naming split_fracs, exit 3,
                        no job submitted, pipeline_rounds.json byte for
                        byte, the state's jobs kept; restored, --retry
                        search; the same for search.flow.
                        conditioner_hidden_layers; a changed
                        search.trial.walltime continues to PIPELINE DONE;
                        every ledger record has the identity's split_hash
    test_convergence    not a discretisation: skipped
    test_invariants     after every case of test_reference, every ok
                        record's split_hash is the identity's (checked
                        inside it, at each step); here: same_value is
                        symmetric and the refusal a pure function
    test_contract       for every search of giulia.json, linddup.json and
                        synth.json (each run view of linddup.json's
                        `searches` and of synth.json's), the K_cfg part of
                        the identity as `_identity` writes it (through a
                        JSON round trip, as the state file holds it)
                        compares equal to the file; one identity per search
                        view of each file (stages/runs.py search_views)
    test_determinism    no randomness in the comparison: skipped
    test_edge_cases     the change accepted before the stage has started
                        (the start refuses a folder of another identity as
                        before; inside test_reference); the dry run's line

Run
    python3 hpc/pipeline/tests/test_smoke_identity.py [check ...]
        About 3 minutes at one thread (test_reference runs the real tuner
        and runner; it needs torch, sbi, zuko, scikit-optimize). The other
        checks need the standard library only. Nothing touches davinci.

Pure ASCII, LF only.
"""

import collections
import copy
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C                      # noqa: E402
import stages as S                      # noqa: E402
import stages.common as CM              # noqa: E402
import stages.runs as RN                # noqa: E402
import stages.search as SS              # noqa: E402

PROJECTS = os.path.join(_PIPE, "projects")
GIULIA = os.path.join(PROJECTS, "giulia", "giulia.json")
LINDDUP = os.path.join(PROJECTS, "linddup", "linddup.json")
SYNTH = os.path.join(PROJECTS, "synth", "synth.json")
PYTHON = sys.executable


# --------------------------------------------------------------- helpers ---

def _changed(cfg, edit):
    """A Config whose data is a deep copy of cfg's, `edit(data)` applied."""
    data = copy.deepcopy(cfg.data)
    edit(data)
    return C.Config(data, cfg.path, "x" * 64)


def _ident_of(cfg):
    """The K_cfg part of an identity as `_identity` builds it, through the
    JSON round trip of the state file."""
    return json.loads(json.dumps(dict((k, v) for k, v in SS.config_identity(cfg).items())))


class _Tripwire(object):
    """A scheduler recorder that fails the test when anything but `log` is
    used (the refusal must come before any status, fit or submission)."""

    def __init__(self):
        self.lines = []

    def log(self, text):
        self.lines.append(text)

    def __getattr__(self, name):
        raise AssertionError("the stage used the scheduler (%s) before refusing" % name)


def _fake_split_tool(tmp):
    """CM.E.run_in_env replaced: writes bank_split.json as tools/
    bank_split_hash.py does from its argv (the fractions parsed as floats,
    the seed as an int; lines 51 and 68 of the tool), with a dummy bank."""
    def run_in_env(env_name, argv, cwd=None, log_path=None, timeout=None, env=None):
        argv = [str(a) for a in argv]
        i = argv.index("--split-fracs")
        fracs = [float(f) for f in argv[i + 1:i + 4]]
        seed = int(argv[argv.index("--split-seed") + 1])
        out = argv[argv.index("--out") + 1]
        d_theta = int(run_in_env.d_theta)
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        with open(out, "w") as fh:
            json.dump({"split_hash": "a" * 64, "contract_digest": "b" * 64, "n_train": 10,
                       "n_sel": 2, "n_report": 1, "n_rows": 13, "n_shards": 2, "W": 128,
                       "d_theta": d_theta, "latent_kind": None, "split_fracs": fracs,
                       "split_seed": seed}, fh)
        return 0, "SPLIT OK: hash=%s contract=%s" % ("a" * 64, "b" * 64)
    return run_in_env


def _identity_as_written(stage, view, tmp):
    """`SearchStage._identity` run on `view` with the split tool replaced
    (the bank's numbers are dummies; K_cfg comes from the code under test),
    its campaign folder under `tmp`; the identity through a JSON round trip."""
    camp = os.path.join(tmp, view.data["search"]["campaign"])
    orig_run, orig_dirs = CM.E.run_in_env, stage._dirs
    fake = _fake_split_tool(tmp)
    fake.d_theta = view.data["search"]["shape_anchors"]["d_theta"]
    CM.E.run_in_env = fake
    stage._dirs = lambda cfg: {"campaign": camp}
    try:
        st = {"sub": {}}
        ident = stage._identity(view, st, os.path.join(tmp, "login.log"))
    finally:
        CM.E.run_in_env = orig_run
        try:
            del stage._dirs
        except AttributeError:
            stage._dirs = orig_dirs
    return json.loads(json.dumps(ident))


def _search_stages(cfg):
    """[(stage, the configuration it sees)] of a project's graph."""
    out = []
    for s in S.default_graph(cfg):
        if isinstance(s, SS.SearchStage):
            out.append((s, s._view(cfg) if hasattr(s, "_view") else cfg))
    return out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = C.load_config(GIULIA)
    ident = _ident_of(cfg)
    assert SS.identity_refusal(ident, cfg) is None, "none changed: no refusal"
    one = {"split_fracs": lambda d: d["split"].__setitem__("fractions", [0.8, 0.15, 0.05]),
           "split_seed": lambda d: d["search"].__setitem__("train_seed", 7),
           "campaign": lambda d: d["search"].__setitem__("campaign", "S-A2-only"),
           "shape_anchors": lambda d: d["search"]["shape_anchors"].__setitem__("embedding_dim", 32),
           "space_overrides": lambda d: d["search"]["space_overrides"].__setitem__(
               "depth_exponent", [3, 4]),
           "flow": lambda d: d["search"]["flow"].__setitem__("conditioner_hidden_layers", 3)}
    assert set(one) == set(SS.K_CFG), "the table covers K_cfg exactly"
    for key, edit in sorted(one.items()):
        new = _changed(cfg, edit)
        text = SS.identity_refusal(ident, new)
        assert text is not None, "%s changed alone: no refusal" % key
        assert text.startswith("REFUSE: the configuration no longer matches this search's "
                               "identity ("), text
        assert text.endswith("); a search is not continued under another identity -- restore "
                             "the value(s), or reset the search first"), text
        old_v, new_v = SS._show(ident[key]), SS._show(SS.config_identity(new)[key])
        assert ("%s: search %s, configuration %s" % (key, old_v, new_v)) in text, (key, text)
        others = [k for k in SS.K_CFG if k != key and ("%s: search" % k) in text]
        assert not others, (key, others, text)
    # two changed: both named, in K_CFG's order
    two = _changed(cfg, lambda d: (one["split_seed"](d), one["flow"](d)))
    text = SS.identity_refusal(ident, two)
    assert "split_seed: search 0, configuration 7" in text and "flow: search" in text, text
    assert text.index("split_seed:") < text.index("flow:"), text
    assert [k for k, _, _ in SS.identity_diffs(ident, two)] == ["split_seed", "flow"]
    # 0.85 written 0.8500 in the file: no refusal (numbers as floats)
    tmp = tempfile.mkdtemp(prefix="pipeline_ident_")
    try:
        raw = open(GIULIA, "rb").read().decode("ascii")
        p = os.path.join(tmp, "g.json")
        data = json.loads(raw)
        data["split"]["fractions"] = ["@@F@@"]
        text4 = json.dumps(data, indent=1).replace('"@@F@@"', "0.8500, 0.1000, 0.0500")
        assert "0.8500, 0.1000, 0.0500" in text4
        with open(p, "w") as fh:
            fh.write(text4)
        c4 = C.load_config(p)
        assert c4.sha256 != cfg.sha256, "another file"
        assert SS.identity_refusal(ident, c4) is None, "0.8500 refused"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert SS.same_value(0.85, 0.8500) and SS.same_value(0, 0.0) and SS.same_value(3, 3.0)
    assert SS.same_value({"a": [1, 2.0]}, {"a": [1.0, 2]})
    assert not SS.same_value({"a": 1}, {"a": 1, "b": 2}), "a key on one side only"
    assert not SS.same_value([1, 2], [1, 2, 3]) and not SS.same_value(True, 1)
    # the fractions in another order: refused
    perm = _changed(cfg, lambda d: d["split"].__setitem__("fractions", [0.1, 0.85, 0.05]))
    text = SS.identity_refusal(ident, perm)
    assert text and "split_fracs: search [0.85, 0.1, 0.05], configuration [0.1, 0.85, 0.05]" \
        in text, text
    # a key outside K_cfg: no refusal
    for edit in (lambda d: d["search"]["trial"].__setitem__("walltime", "01:00:00"),
                 lambda d: d["search"].__setitem__("max_concurrent_trials", 9),
                 lambda d: d["resources"].__setitem__("max_cores", 999)):
        assert SS.identity_refusal(ident, _changed(cfg, edit)) is None
    # the start and the poll on a state holding the identity, the configuration changed
    tmp = tempfile.mkdtemp(prefix="pipeline_ident_")
    try:
        base = _changed(cfg, lambda d: d["paths"].__setitem__("tune_root", tmp))
        full = dict(ident, split_hash="a" * 64, contract_digest="b" * 64, n_train=10)
        moved = _changed(base, one["split_fracs"])
        stage = SS.SearchStage()
        st = {"sub": {"identity": dict(full)}, "jobs": [{"id": "7", "finished": False}]}
        pbs = _Tripwire()
        try:
            stage.start(moved, st, pbs)
            raise AssertionError("the start did not refuse")
        except RuntimeError as exc:
            assert str(exc).startswith("REFUSE: the configuration no longer matches") \
                and "split_fracs: search" in str(exc), exc
        st = {"sub": {"identity": dict(full), "phase": "rounds", "rounds": [], "trials": {}},
              "jobs": [{"id": "7", "finished": False}]}
        r = stage.poll(moved, st, pbs)
        assert r == "finished" and st["sub"]["phase"] == "failed", (r, st)
        assert st["sub"]["fail"].startswith("REFUSE: ") and "split_fracs" in st["sub"]["fail"]
        assert st["jobs"] == [{"id": "7", "finished": False}], "the jobs kept, untouched"
        ok, why = stage.verify(moved, st)
        assert not ok and why == st["sub"]["fail"], (ok, why)
        assert os.listdir(tmp) == [], "nothing written: %s" % os.listdir(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- the end-to-end check ---

def _bump(path, edit):
    cfg = json.load(open(path))
    edit(cfg)
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)


def _all_jobs_ended(spool, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(spool, "*.json"))
                if os.path.basename(p)[:-5].isdigit()]
        if all(r.get("finished") for r in recs):
            return True
        time.sleep(0.5)
    return False


def _state(cfg):
    return json.load(open(os.path.join(cfg["paths"]["work_root"], "state.json")))


def _ok_hashes_are_the_identity(camp, ident_hash, where):
    for p in glob.glob(os.path.join(camp, "trials", "*.json")):
        rec = json.load(open(p))
        if rec.get("status") == "ok":
            assert rec["split_hash"] == ident_hash, (where, p, rec["split_hash"], ident_hash)


def _refused_run(B7, root, path, c, spool, env, camp, key):
    """A run with --accept-config-change on a running search whose K_cfg
    field `key` changed: the dry run's line, then the refusal."""
    rounds = os.path.join(camp, "pipeline_rounds.json")
    before_rf = open(rounds, "rb").read()
    st0 = _state(c)["stages"]["search"]
    assert st0["status"] == "running", st0["status"]
    jobs0 = [j["id"] for j in st0["jobs"]]
    n0 = B7._n_submitted(spool)
    state_bytes = open(os.path.join(c["paths"]["work_root"], "state.json"), "rb").read()
    # the dry run prints the REFUSE line and changes nothing
    rc, out = B7._run(root, path, ["--dry-run", "--accept-config-change"], env, timeout=120)
    assert ("    REFUSE: the configuration no longer matches this search's identity (%s: search"
            % key) in out, out
    assert open(os.path.join(c["paths"]["work_root"], "state.json"), "rb").read() == state_bytes
    # the run
    rc, out = B7._run(root, path, ["--accept-config-change"], env, timeout=300)
    assert rc == 3, (rc, out)
    stop = [ln for ln in out.splitlines() if "STOP:" in ln]
    assert stop and ("STOP: stage search failed: REFUSE: the configuration no longer matches "
                     "this search's identity (%s: search" % key) in stop[-1], out
    assert B7._n_submitted(spool) == n0, "a job was submitted after the change"
    assert open(rounds, "rb").read() == before_rf, "pipeline_rounds.json rewritten"
    st1 = _state(c)["stages"]["search"]
    assert st1["status"] == "failed" and st1["note"].startswith("REFUSE: "), st1["status"]
    assert [j["id"] for j in st1["jobs"]] == jobs0, "the stage's jobs forgotten"
    return out


def test_reference():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)
    import test_smoke_block7 as B7
    import test_smoke_block7b as B7b
    root = tempfile.mkdtemp(prefix="pipeline_ident_e2e_")
    try:
        path, c = B7._synthetic_project(
            root, search={"n_initial_points": 2, "trials_per_round": 1, "gp_rounds": 1,
                          "max_concurrent_trials": 1})
        camp = B7b._seed_probes(c)        # the gate's probes reused: no probe job
        spool = os.path.join(root, "spool")
        env = B7._env(spool)
        rounds = os.path.join(camp, "pipeline_rounds.json")

        # (1) stopped while round 0 runs: one tick starts the search and submits trial 1
        rc, out = B7._run(root, path, ["--max-ticks", "1"], env, timeout=300)
        assert rc == 0 and "tick limit 1 reached" in out, out
        assert B7._n_submitted(spool) == 1, B7._n_submitted(spool)
        ident = json.load(open(rounds))["identity"]
        assert SS.identity_refusal(ident, C.load_config(path)) is None

        # (2) split.fractions changed: refused at the poll
        _bump(path, lambda d: d["split"].__setitem__("fractions", [0.8, 0.15, 0.05]))
        _refused_run(B7, root, path, c, spool, env, camp, "split_fracs")
        assert _all_jobs_ended(spool), "trial 1's job never ended"
        _ok_hashes_are_the_identity(camp, ident["split_hash"], "after the split refusal")

        # (edge) the change still in place, --retry search: the stage starts afresh and
        # its start refuses the folder of another identity, as before the amendment
        rf_bytes = open(rounds, "rb").read()
        n = B7._n_submitted(spool)
        rc, out = B7._run(root, path, ["--retry", "search", "--accept-config-change"], env,
                          timeout=300)
        assert rc == 3 and "STOP: stage search refused: REFUSE:" in out and \
            "holds a search of another identity" in out and "split_fracs: folder" in out, out
        assert B7._n_submitted(spool) == n and open(rounds, "rb").read() == rf_bytes

        # (3) restored, --retry search: the search continues (trial 2 submitted)
        _bump(path, lambda d: d["split"].__setitem__("fractions", [0.85, 0.1, 0.05]))
        rc, out = B7._run(root, path, ["--retry", "search", "--accept-config-change",
                                       "--max-ticks", "1"], env, timeout=300)
        assert rc == 0 and "REFUSE" not in out, out
        assert B7._n_submitted(spool) == n + 1, (B7._n_submitted(spool), out)
        assert json.load(open(rounds))["identity"] == ident
        _ok_hashes_are_the_identity(camp, ident["split_hash"], "after the retry")

        # (4) search.flow.conditioner_hidden_layers changed: the same refusal
        _bump(path, lambda d: d["search"]["flow"].__setitem__("conditioner_hidden_layers", 3))
        _refused_run(B7, root, path, c, spool, env, camp, "flow")
        assert _all_jobs_ended(spool)
        _bump(path, lambda d: d["search"]["flow"].__setitem__("conditioner_hidden_layers", 2))
        rc, out = B7._run(root, path, ["--retry", "search", "--accept-config-change",
                                       "--max-ticks", "1"], env, timeout=300)
        assert rc == 0 and "REFUSE" not in out, out
        assert _state(c)["stages"]["search"]["status"] == "running"
        _ok_hashes_are_the_identity(camp, ident["split_hash"], "after the flow retry")

        # (5) search.trial.walltime changed while the search runs: it continues
        _bump(path, lambda d: d["search"]["trial"].__setitem__("walltime", "00:09:00"))
        rc, out = B7._run(root, path, ["--accept-config-change"], env, timeout=900)
        assert rc == 0 and "PIPELINE DONE" in out and "REFUSE" not in out, out
        sc = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert sc["ended_by"] == "rounds" and sc["n_trials"]["ok"] == 3, sc
        assert sc["identity"]["split_hash"] == ident["split_hash"]
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(camp, "trials", "*.json"))]
        assert len(recs) == 3 and all(r["split_hash"] == ident["split_hash"] for r in recs), \
            [(r["trial_id"][:8], r["status"], r["split_hash"][:12]) for r in recs]
        rf = json.load(open(rounds))
        assert rf["identity"] == ident and [len(r["trial_ids"]) for r in rf["rounds"]] == [2, 1]
        # no trial submitted twice: 3 trials + the best fit's members
        n_members = len(c["best_fit"]["member_seeds"])
        assert B7._n_submitted(spool) == 3 + n_members, B7._n_submitted(spool)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        for d in list(B7._SHARED.values()):
            if isinstance(d, str) and os.path.isdir(d) and "banks_" in d:
                shutil.rmtree(d, ignore_errors=True)
        B7._SHARED.clear()


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    # the ok records' split_hash is checked at every step of test_reference;
    # here the comparison's own invariants
    cfg = C.load_config(GIULIA)
    ident = _ident_of(cfg)
    vals = [0, 0.0, 1, 0.85, [0.85, 0.1], [0.1, 0.85], {"a": 1}, {"a": 1.0}, {"b": 1}, "S-A1",
            None, True, [], {}]
    for a in vals:
        for b in vals:
            assert SS.same_value(a, b) == SS.same_value(b, a), (a, b)
        assert SS.same_value(a, a), a
    moved = _changed(cfg, lambda d: d["search"].__setitem__("train_seed", 5))
    assert SS.identity_refusal(ident, moved) == SS.identity_refusal(dict(ident), moved)
    snap = json.dumps(ident, sort_keys=True), json.dumps(moved.data, sort_keys=True)
    SS.identity_refusal(ident, moved)
    assert snap == (json.dumps(ident, sort_keys=True), json.dumps(moved.data, sort_keys=True)), \
        "the check mutated its inputs"


def test_contract():
    tmp = tempfile.mkdtemp(prefix="pipeline_ident_contract_")
    try:
        seen, views = [], []
        for path in (GIULIA, LINDDUP, SYNTH):
            cfg = C.load_config(path, synthetic_ok=True)
            # the file's searches as the pipeline lists them (block 29, stages/runs.py
            # search_views): the single search of a file whose `searches` is null, a run
            # view per entry of `searches` (linddup.json v11) or `synthetic.searches`
            views.extend((os.path.basename(path), n) for n, _v in RN.search_views(cfg))
            pairs = _search_stages(cfg)
            assert pairs, path
            if path == SYNTH:
                names = sorted(q["name"] for q in cfg.data["synthetic"]["searches"])
                assert sorted(s.run["name"] for s, _ in pairs) == names, "every run view"
            for stage, view in pairs:
                ident = _identity_as_written(stage, view, os.path.join(tmp, str(len(seen))))
                assert set(SS.K_CFG) <= set(ident), (path, sorted(ident))
                text = SS.identity_refusal(ident, view)
                assert text is None, (path, stage.name, text)
                # and the comparison of 8d5b162's folder check holds too
                for k in SS.K_CFG:
                    assert SS.same_value(ident[k], SS.config_identity(view)[k]), (path, k)
                seen.append((os.path.basename(path), stage.name, ident["campaign"]))
        # the count: one identity per search view of each file [the final check's F2,
        # 2026-11-17: the count was the files' of before v11, one search in linddup.json]
        assert collections.Counter(f for f, _s, _c in seen) == \
            collections.Counter(f for f, _n in views), (seen, views)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_determinism():
    raise unittest.SkipTest("the comparison has no randomness")


def test_edge_cases():
    # (a) the change accepted before the stage has started: inside test_reference
    #     ("(edge)": the start refuses the folder of another identity)
    # (b) the dry run's line: on a hand-made state, no scheduler
    cfg = C.load_config(GIULIA)
    ident = dict(_ident_of(cfg), split_hash="a" * 64, contract_digest="b" * 64, n_train=10)
    stage = SS.SearchStage()
    lines = stage.dry_run(cfg, {"sub": {"identity": ident}})
    assert not any(ln.startswith("REFUSE") for ln in lines), lines[0]
    moved = _changed(cfg, lambda d: d["search"]["flow"].__setitem__("conditioner_hidden_layers", 4))
    lines = stage.dry_run(moved, {"sub": {"identity": ident}})
    assert lines[0].startswith("REFUSE: the configuration no longer matches this search's "
                               "identity (flow: search ") and "conditioner_hidden_layers" in lines[0]
    assert lines[1].startswith("identity: "), lines[1]
    lines = stage.dry_run(moved, {"sub": {}})
    assert not any(ln.startswith("REFUSE") for ln in lines), "no identity yet: no line"
    # (c) a state identity without some K_cfg field (none exists at 8d5b162): refused,
    #     never taken as equal
    partial = dict(ident)
    del partial["flow"]
    assert "flow: search null" in SS.identity_refusal(partial, cfg)


# ---------------------------------------------------------------- runner ---

CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases)]


def main():
    print("== %s" % os.path.basename(__file__))
    print("python %s | %s" % (platform.python_version(), platform.platform()))
    only = set(a for a in sys.argv[1:] if not a.startswith("-"))
    rows, bad = [], 0
    t0 = time.time()
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except NotImplementedError as exc:
            bad += 1
            rows.append((name, "TODO", str(exc), time.time() - t1))
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
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
