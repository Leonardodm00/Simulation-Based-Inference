#!/usr/bin/env python3
"""Smoke test for block 7, amendment (i) of 2026-11-14 in specs/SPEC.md
(D-134 (2)): the best fit follows its search's finalist, and sec. 2.2
rule 5 (`--retry <s>` returns the done or skipped stages downstream of s
to pending).

Checks (specs/SPEC.md block 7, amendment (i), "Test oracles")
    test_known_answer   is_done's table -- no artefact; the artefact naming
                        the search's finalist; naming another trial; no
                        SEARCH_COMPLETE.json -- for S8 (recorded kind), F_q
                        and G_q (synthetic kind); on a chain a -> b -> c
                        with every stage done, --retry a returns b and c
                        to pending, --retry b returns c alone, --retry c
                        returns none
    test_reference      end to end on block 7's reduced project (the fake
                        PBS, the real tuner and runner): a finished run;
                        SEARCH_COMPLETE.json rewritten to name the
                        ledger's second-best trial (the audit's t2-C2);
                        a run without --retry changes nothing (a done
                        stage is not re-checked: the chain of 8d5b162);
                        --retry search: the best fit returns to pending,
                        the old root is <root>.set_aside1 with every
                        file's sha256 unchanged, M member jobs, the
                        artefact names the new finalist and every member
                        record carries its configuration, PIPELINE DONE;
                        then with the finalist unchanged: skipped, no
                        job, the root untouched
    test_reference_synthetic
                        the same on the synthetic kind (block 18's reduced
                        S-A2-R0 project): --retry search_S-A2-R0 after the
                        finalist changed: best_fit_S-A2-R0 is fitted again
                        in a new root; unchanged, skipped; then G_q's
                        start (alt_best_fit true, block 18's finalist
                        computed by the real tuner in a scratch folder):
                        its own root kept untouched, a root naming
                        another trial moved aside
    test_convergence    not a discretisation: skipped
    test_invariants     no file is deleted by any case of test_reference
                        (every file before is present after, at its path
                        or under the set-aside root); --retry <s> with a
                        downstream job live exits 2 and leaves the state
                        file byte for byte; a retry of a stage with nothing
                        downstream behaves as before; the state file
                        carries no field an older one lacks
    test_contract       the reasons is_done gives name the failing
                        condition (both trial ids, or the missing file);
                        the log lines of the set-aside and of rule 5
    test_determinism    no randomness in the pipeline's part: skipped
    test_edge_cases     the audit's F504 case (BEST_FIT_COMPLETE.json
                        removed, the member folders of another finalist
                        kept, --retry best_fit: the root set aside, M
                        member jobs) which is also the second set-aside
                        (.set_aside2); the dry run after a changed
                        finalist (the line, nothing moved); by hand: an
                        unreadable finalist_config.json beside member
                        records, member records with no
                        finalist_config.json, a root of the current
                        finalist kept

Run
    python3 hpc/pipeline/tests/test_smoke_finalist.py [check ...]
        A few minutes at one thread (two reduced searches); names of checks
        run only those. Needs the training environment (torch, sbi, zuko,
        scikit-optimize) for the end-to-end checks; PIPELINE_SKIP_CONDA=1
        is set by the helpers. Nothing touches davinci.

Pure ASCII, LF only.
"""

import copy
import glob
import hashlib
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
import pipeline as P                    # noqa: E402
import stages as S                      # noqa: E402
from stages.best_fit import BestFitStage    # noqa: E402
from stages import synthetic as SY      # noqa: E402
from state import State                 # noqa: E402

PYTHON = sys.executable
GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_finalist_")


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tree(root):
    """{relative path: sha256} of every file under root."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = _sha(p)
    return out


def _write(path, obj):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        if isinstance(obj, (bytes, str)):
            fh.write(obj if isinstance(obj, str) else obj.decode("ascii"))
        else:
            json.dump(obj, fh, indent=1, sort_keys=True)


def _recorded_cfg(root):
    """The committed recorded-kind configuration with its folders under root."""
    with open(GIULIA, "rb") as fh:
        d = C._strip_docs(json.loads(fh.read().decode("ascii")))
    for k in ("work_root", "runs_root", "tune_root"):
        d["paths"][k] = os.path.join(root, k)
    path = os.path.join(root, "rec.json")
    _write(path, d)
    return C.load_config(path)


def _synth_cfg(root):
    """Block 18's reduced synthetic project, S-A2-R0 with alt_best_fit true."""
    import test_smoke_synthetic as T
    search = {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": True}
    path, _ = T._small(root, searches=[search])
    return C.load_config(path, synthetic_ok=True)


class _Log(object):
    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(str(text))

    log = __call__


class _Job(object):
    def __init__(self, finished):
        self.finished = finished
        self.state = "F" if finished else "R"


class _FakePBSModule(object):
    """pbs_module for _prepare_state / run: `live` ids are not finished."""

    def __init__(self, live=()):
        self.live = set(live)

    def status(self, job_id):
        return _Job(job_id not in self.live)


class _Toy(S.Stage):
    def __init__(self, name, deps):
        self.name = name
        self.deps = tuple(deps)

    def is_done(self, cfg):
        return True, "toy artefact"

    def start(self, cfg, st, pbs):
        raise AssertionError("a toy stage was started")


class _ToyCfg(object):
    def __init__(self, root):
        self.data = {"paths": {"work_root": os.path.join(root, "work")}, "dataset": "toy"}
        self.path = os.path.join(root, "toy.json")
        self.sha256 = "0" * 64

    @property
    def tag(self):
        return "toy"


def _toy_state(cfg, graph, live_in=None):
    st = State.new(cfg.data["paths"]["work_root"], "toy", cfg.path, cfg.sha256)
    for s_ in graph:
        rec = st.stage(s_.name)
        rec["status"] = "done"
        rec["note"] = "toy"
        rec["sub"] = {"kept": s_.name}
        rec["jobs"] = [{"id": "%s.1" % s_.name, "role": "x", "finished": s_.name != live_in,
                        "cmd": [], "log": None, "exit_status": 0, "walltime_s": 1,
                        "submitted_utc": "2026-11-14T00:00:00Z"}]
    st.save()
    return st


def _chain():
    return [_Toy("a", ()), _Toy("b", ("a",)), _Toy("c", ("b",))]


def _statuses(work_root):
    return dict((k, v["status"]) for k, v in State.load(work_root).data["stages"].items())


# -- the end-to-end fixture (recorded kind), run once and shared by checks --

_E2E = {}


def _spool_names(spool):
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                r = json.load(fh)
            out.append((r["num"], r["name"]))
    return [n for _, n in sorted(out)]


def _present_after(before, root, aside_roots):
    """Every file of `before` (relative to `root`) is present after, with the
    same sha256, at its path or under one of the set-aside roots."""
    missing = []
    for rel, h in before.items():
        p = os.path.join(root, rel)
        if os.path.isfile(p) and _sha(p) == h:
            continue
        ok = False
        for best, aside in aside_roots:
            if rel.startswith(best + os.sep):
                q = os.path.join(root, aside + rel[len(best):])
                if os.path.isfile(q) and _sha(q) == h:
                    ok = True
        if not ok:
            missing.append(rel)
    return missing


def _e2e():
    """The recorded-kind sequence; each step's outputs kept in _E2E."""
    if _E2E.get("done"):
        if _E2E.get("error"):
            raise AssertionError("the end-to-end fixture failed: %s" % _E2E["error"])
        return _E2E
    _E2E["done"] = True
    try:
        _e2e_steps()
    except Exception:
        _E2E["error"] = traceback.format_exc()
        raise
    return _E2E


def _e2e_steps():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)
    import test_smoke_block7 as B7
    root = _tmp()
    E = _E2E
    E["root"] = root
    path, c = B7._synthetic_project(root)
    spool = os.path.join(root, "spool")
    env = B7._env(spool)
    E.update(path=path, c=c, spool=spool, env=env)
    runs = c["paths"]["runs_root"]
    tune = c["paths"]["tune_root"]
    best = os.path.join(runs, "synth_best")
    camp = os.path.join(tune, "S-A1")
    scp = os.path.join(camp, "SEARCH_COMPLETE.json")
    E.update(runs=runs, tune=tune, best=best, camp=camp, scp=scp)

    def run(*extra):
        return B7._run(root, path, list(extra), env, timeout=1200)

    # 0. the finished run
    rc, out = run()
    E["run0"] = (rc, out)
    assert rc == 0 and "PIPELINE DONE: every stage done or skipped" in out, out[-4000:]
    sc = json.load(open(scp))
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(camp, "trials", "*.json")))]
    ok = sorted((r for r in recs if r["status"] == "ok"), key=lambda r: r["nll"])
    first = sc["finalist"]["trial_id"]
    assert ok[0]["trial_id"] == first, (first, [r["trial_id"] for r in ok])
    second = [r for r in ok if r["trial_id"] != first][0]
    E.update(first=first, second=second, sc0=copy.deepcopy(sc))
    E["root0"] = _tree(best)

    # 1. the search's finalist changed by hand (the audit's t2-C2): the ledger's second best
    sc2 = copy.deepcopy(sc)
    sc2["finalist"] = {"trial_id": second["trial_id"], "nll": second["nll"],
                       "config": second["config"]}
    _write(scp, sc2)
    E["sc2"] = sc2
    before = {"runs": _tree(runs), "tune": _tree(tune)}
    E["before1"] = before
    n0 = len(_spool_names(spool))
    E["n0"] = n0
    state_path = State.path_of(c["paths"]["work_root"])

    # 1a. no --retry: a done stage is not re-checked (a chain of 8d5b162 resumes unchanged)
    rc, out = run()
    E["run_noretry"] = (rc, out, len(_spool_names(spool)), _tree(runs), _tree(tune))
    # 1b. the dry run with --retry search: the line, nothing moved
    st_bytes = open(state_path, "rb").read()
    rc, out = run("--dry-run", "--retry", "search")
    E["dry"] = (rc, out, len(_spool_names(spool)), _tree(runs), open(state_path, "rb").read() == st_bytes)
    # 1c. --retry search: the best fit follows
    rc, out = run("--retry", "search")
    E["retry1"] = (rc, out, _spool_names(spool)[n0:])
    E["after1"] = {"runs": _tree(runs), "tune": _tree(tune)}
    bf1 = os.path.join(best, "BEST_FIT_COMPLETE.json")
    E["bf1"] = json.load(open(bf1)) if os.path.isfile(bf1) else None
    E["members1"] = dict((m["seed"], json.load(open(m["record"])))
                         for m in (E["bf1"] or {}).get("members", []))
    fc1 = os.path.join(best, "finalist_config.json")
    E["fc1"] = json.load(open(fc1)) if os.path.isfile(fc1) else None
    E["state1"] = json.load(open(state_path))

    # 2. the finalist unchanged: --retry search again
    n1 = len(_spool_names(spool))
    root1 = _tree(best)
    rc, out = run("--retry", "search")
    E["retry2"] = (rc, out, len(_spool_names(spool)) - n1, root1, _tree(best))

    # 3. the audit's F504 case: SEARCH_COMPLETE.json back to the first finalist, the
    # artefact removed (kept by the test outside the runs root), the member folders of
    # the second finalist kept, --retry best_fit: the root moved aside (.set_aside2)
    _write(scp, sc)
    keep = os.path.join(root, "removed_by_the_test")
    os.makedirs(keep)
    shutil.move(os.path.join(best, "BEST_FIT_COMPLETE.json"), os.path.join(keep, "BEST_FIT_COMPLETE.json"))
    before3 = {"runs": _tree(runs), "tune": _tree(tune)}
    E["before3"] = before3
    n2 = len(_spool_names(spool))
    rc, out = run("--retry", "best_fit")
    E["retry3"] = (rc, out, _spool_names(spool)[n2:])
    bf3 = os.path.join(best, "BEST_FIT_COMPLETE.json")
    E["bf3"] = json.load(open(bf3)) if os.path.isfile(bf3) else None
    E["after3"] = {"runs": _tree(runs), "tune": _tree(tune)}


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    root = _tmp()
    try:
        # (a) is_done's table, S8 on the recorded kind
        cfg = _recorded_cfg(root)
        stage = BestFitStage()
        dirs = stage._dirs(cfg)
        art, scp = dirs["complete"], dirs["search_complete"]
        table = []

        def case(label, sc_trial, art_finalist, expect):
            for p in (art, scp):
                if os.path.exists(p):
                    os.remove(p)
            if sc_trial is not None:
                _write(scp, {"finalist": {"trial_id": sc_trial, "nll": 1.0, "config": {}}})
            if art_finalist is not None:
                _write(art, {"finalist": art_finalist})
            done, why = stage.is_done(cfg)
            table.append((label, done, why))
            assert done == expect, (label, done, why)
            return why

        why = case("S8 no artefact", "T1", None, False)
        assert why == art + " absent", why
        why = case("S8 the search's finalist", "T1", {"trial_id": "T1"}, True)
        assert why == art, why
        why = case("S8 another trial", "T1", {"trial_id": "T2"}, False)
        assert "T2" in why and "T1" in why, why
        why = case("S8 no SEARCH_COMPLETE", None, {"trial_id": "T1"}, False)
        assert scp in why and "absent" in why, why

        # (b) F_q and G_q of a synthetic search (stages/synthetic.py's classes)
        sroot = os.path.join(root, "synth")
        os.makedirs(sroot)
        scfg = _synth_cfg(sroot)
        by = dict((s_.name, s_) for s_ in S.default_graph(scfg))
        fq, gq = by["best_fit_S-A2-R0"], by["best_fit_alt_S-A2-R0"]
        assert isinstance(gq, SY.RunAltBestFitStage)
        v = fq._view(scfg)
        scp_q = fq._dirs(v)["search_complete"]
        for stg, label, own, other in (
                (fq, "F_q", {"trial_id": "T1"}, {"trial_id": "T2"}),
                (gq, "G_q", {"trial_id": "A9", "search_finalist": {"trial_id": "T1", "arm": "A2"}},
                 {"trial_id": "A9", "search_finalist": {"trial_id": "T2", "arm": "A2"}})):
            art_q = stg._dirs(v)["complete"]
            for sc_trial, fin, expect, needles in (
                    ("T1", None, False, (art_q + " absent",)),
                    ("T1", own, True, ()),
                    ("T1", other, False, ("T1", "T2")),
                    (None, own, False, (scp_q, "absent"))):
                for p in (art_q, scp_q):
                    if os.path.exists(p):
                        os.remove(p)
                if sc_trial:
                    _write(scp_q, {"finalist": {"trial_id": sc_trial}})
                if fin is not None:
                    _write(art_q, {"finalist": fin})
                done, why = stg.is_done(scfg)
                table.append(("%s sc=%s art=%s" % (label, sc_trial, fin and fin.get("trial_id")),
                              done, why))
                assert done == expect, (label, sc_trial, fin, done, why)
                for n in needles:
                    assert n in why, (label, n, why)
                if expect:
                    assert why == art_q, why

        # (c) rule 5 on a chain a -> b -> c with every stage done
        for retry, returned in ((["a"], ["b", "c"]), (["b"], ["c"]), (["c"], [])):
            croot = os.path.join(root, "chain_" + retry[0])
            os.makedirs(croot)
            tcfg = _ToyCfg(croot)
            graph = _chain()
            _toy_state(tcfg, graph)
            log = _Log()
            st, rc = P._prepare_state(tcfg, graph, tcfg.data["paths"]["work_root"], retry, False,
                                      _FakePBSModule(), log, dry_run=False)
            assert rc is None, log.lines
            got = sorted(n for n, rec in st.data["stages"].items()
                         if rec["status"] == "pending" and n not in retry)
            assert got == returned, (retry, got, log.lines)
            assert st.stage(retry[0])["status"] == "pending"
            for n in ("a", "b", "c"):
                if n not in returned and n not in retry:
                    assert st.stage(n)["status"] == "done" and st.stage(n)["sub"], n
            assert sum(1 for ln in log.lines if "returned to pending" in ln) == len(returned), log.lines
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for label, done, why in table:
        print("    is_done %-40s %-5s %s" % (label, done, why))


def test_reference():
    E = _e2e()
    first, second = E["first"], E["second"]["trial_id"]
    print("    finalists: first %s, second %s" % (first[:12], second[:12]))
    # without --retry: nothing is re-checked, nothing submitted, nothing moved
    rc, out, n, runs, tune = E["run_noretry"]
    assert rc == 0 and "PIPELINE DONE" in out, out[-3000:]
    assert n == E["n0"], "a run without --retry submitted %d job(s)" % (n - E["n0"])
    assert runs == E["before1"]["runs"] and tune == E["before1"]["tune"], "a run without --retry changed files"
    # --retry search: the best fit follows the new finalist
    rc, out, new = E["retry1"]
    assert rc == 0 and "PIPELINE DONE: every stage done or skipped" in out, out[-5000:]
    assert "search: skipped, artefact of record exists" in out, out[-3000:]
    assert "best_fit returned to pending (--retry search" in out, out[-3000:]
    aside = E["best"] + ".set_aside1"
    assert os.path.isdir(aside), os.listdir(E["runs"])
    assert ("moved aside as %s" % aside) in out and first in out and second in out, out[-3000:]
    assert _tree(aside) == E["root0"], "the set-aside root's files changed"
    assert sorted(new) == ["bf_synth_s1", "bf_synth_s2"], new
    bf = E["bf1"]
    assert bf["finalist"]["trial_id"] == second, bf["finalist"]
    assert bf["ensemble"]["M"] == 2
    fc = E["second"]["config"]
    for m in bf["members"]:
        rec = E["members1"][m["seed"]]
        assert m["record"].startswith(E["best"] + os.sep), m["record"]
        for k, rk in (("lr", "lr"), ("weight_decay", "weight_decay"), ("head_pool_ops", "head_pool_ops"),
                      ("head_fusion", "head_fusion"), ("batch_size_npe", "b_sim")):
            assert abs(float(rec["config"][rk]) - float(fc[k])) < 1e-12, (m["seed"], k, rec["config"][rk], fc[k])
    fcfg = E["fc1"]
    assert fcfg["trial_id"] == second and fcfg["config"] == fc
    # the finalist unchanged: skipped, no job, the root untouched
    rc, out, n_new, root1, root2 = E["retry2"]
    assert rc == 0 and "PIPELINE DONE" in out, out[-3000:]
    assert "best_fit returned to pending (--retry search" in out
    assert "best_fit: skipped, artefact of record exists" in out, out[-3000:]
    assert n_new == 0 and root1 == root2, (n_new, root1 == root2)


def test_reference_synthetic():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)
    import test_smoke_synthetic as T
    root = _tmp()
    saved = dict(os.environ)
    try:
        search = {"name": "S-A2-R0", "campaign": "S-A2", "real": "R0", "alt_best_fit": False}
        path, d = T._small(root, searches=[search],
                           **{"synthetic.sim": {"seed": 11, "n_theta": 60, "n_windows": 1,
                                                "traces_per_shard": 40}})
        spool = os.path.join(root, "spool")
        env = T._fake_env(spool)

        def run(*extra):
            r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                                "--poll-seconds", "0.5"] + list(extra), cwd=root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=1800)
            return r.returncode, r.stdout

        rc, out = run()
        assert rc == 0 and "PIPELINE DONE" in out, out[-5000:]
        p = d["paths"]
        camp = os.path.join(p["tune_root"], "S-A2-R0", "S-A2")
        best = os.path.join(p["runs_root"], "S-A2-R0", "synth_best")
        scp = os.path.join(camp, "SEARCH_COMPLETE.json")
        sc = json.load(open(scp))
        recs = [json.load(open(f)) for f in glob.glob(os.path.join(camp, "trials", "*.json"))]
        ok = sorted((r for r in recs if r["status"] == "ok"), key=lambda r: r["nll"])
        first = sc["finalist"]["trial_id"]
        second = [r for r in ok if r["trial_id"] != first][0]
        sc0 = copy.deepcopy(sc)
        root0 = _tree(best)
        sc["finalist"] = {"trial_id": second["trial_id"], "nll": second["nll"], "config": second["config"]}
        _write(scp, sc)
        n0 = len(T._records(spool))
        rc, out = run("--retry", "search_S-A2-R0")
        assert rc == 0 and "PIPELINE DONE" in out, out[-5000:]
        assert "best_fit_S-A2-R0 returned to pending (--retry search_S-A2-R0" in out, out[-3000:]
        assert os.path.isdir(best + ".set_aside1"), ("the root was not set aside", out[-3000:])
        assert _tree(best + ".set_aside1") == root0
        new = [r["name"] for r in T._records(spool)[n0:]]
        assert sorted(new) == ["bf_synth_s1", "bf_synth_s2"], new
        bf = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
        assert bf["finalist"]["trial_id"] == second["trial_id"], bf["finalist"]
        # unchanged: skipped, nothing submitted
        n1 = len(T._records(spool))
        rc, out = run("--retry", "search_S-A2-R0")
        assert rc == 0 and "best_fit_S-A2-R0: skipped, artefact of record exists" in out, out[-3000:]
        assert len(T._records(spool)) == n1
        # G_q's start (block 18's finalist, computed by the tuner): with the search's
        # finalist restored, the other arm's finalist is computed in a scratch folder and
        # the root is examined untouched -- kept when it is that finalist's, moved aside
        # when its finalist_config.json names another trial
        sc["finalist"] = sc0["finalist"]
        _write(scp, sc)
        d_alt = json.load(open(path))
        d_alt["synthetic"]["searches"][0]["alt_best_fit"] = True
        alt_path = os.path.join(root, "synth_alt.json")
        _write(alt_path, d_alt)
        acfg = C.load_config(alt_path, synthetic_ok=True)
        gq = dict((s_.name, s_) for s_ in S.default_graph(acfg))["best_fit_alt_S-A2-R0"]
        v = gq._view(acfg)
        gdirs = gq._dirs(v)
        assert not os.path.exists(gdirs["root"])
        log = _Log()
        st = {"sub": {}}
        gq._follow_finalist(v, st, log)             # a first run: nothing examined
        assert not log.lines and not os.path.exists(gdirs["root"]), log.lines
        fin = gq._finalist(v, st)                   # block 18 writes its root
        alt_id = fin["trial_id"]
        assert alt_id != first, "the other arm's finalist is not the search's"
        _write(os.path.join(gdirs["root"], "member_seed1", "A1_seed1.json"), {"L": 1.0})
        tree = _tree(gdirs["root"])
        log = _Log()
        gq._follow_finalist(v, {"sub": dict(st["sub"])}, log)
        assert not log.lines and _tree(gdirs["root"]) == tree, ("G_q's own root not kept", log.lines)
        fc = json.load(open(gdirs["finalist_config"]))
        fc["trial_id"] = "another_trial"
        _write(gdirs["finalist_config"], fc)
        tree = _tree(gdirs["root"])
        log = _Log()
        st2 = {"sub": {}}
        gq._follow_finalist(v, st2, log)
        assert os.path.isdir(gdirs["root"] + ".set_aside1") and not os.path.exists(gdirs["root"]), log.lines
        assert _tree(gdirs["root"] + ".set_aside1") == tree
        assert len(log.lines) == 1 and "another_trial" in log.lines[0] and alt_id in log.lines[0], log.lines
        print("    G_q: " + log.lines[0][:300])
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the amendment is control flow over files")


def test_invariants():
    root = _tmp()
    try:
        # (a) --retry a with a job of c (downstream) live: exit 2, the state file byte for byte
        tcfg = _ToyCfg(root)
        graph = _chain()
        _toy_state(tcfg, graph, live_in="c")
        wr = tcfg.data["paths"]["work_root"]
        before = open(State.path_of(wr), "rb").read()
        log = _Log()
        rc = P.run(tcfg, graph=graph, retry=["a"], pbs_module=_FakePBSModule(live=["c.1"]),
                   poll_seconds=0.01, max_ticks=2, log=log)
        assert rc == P.EXIT_CONFIG == 2, (rc, log.lines)
        assert any("REFUSE: --retry a: the stage c, downstream of it, has 1 job(s) still queued "
                   "or running (c.1)" in ln for ln in log.lines), log.lines
        assert open(State.path_of(wr), "rb").read() == before, "a refused retry changed the state file"
        rc = P.run(tcfg, graph=graph, retry=["a"], pbs_module=_FakePBSModule(live=["c.1"]),
                   dry_run=True, log=log)
        assert rc == 2 and open(State.path_of(wr), "rb").read() == before
        # (b) --retry c: nothing downstream, behaves as before: c alone reset, then skipped
        log = _Log()
        rc = P.run(tcfg, graph=graph, retry=["c"], pbs_module=_FakePBSModule(),
                   poll_seconds=0.01, max_ticks=2, log=log)
        assert rc == 0 and "c reset to pending (--retry)" in log.lines, log.lines
        assert not any("returned to pending" in ln for ln in log.lines), log.lines
        assert "c: skipped, artefact of record exists: toy artefact" in log.lines, log.lines
        st = State.load(wr)
        assert st.stage("a")["sub"] == {"kept": "a"} and st.stage("b")["sub"] == {"kept": "b"}
        # (c) pending and failed downstream stages are left as they are
        st.stage("b")["status"] = "failed"
        st.stage("c")["status"] = "pending"
        st.save()
        log = _Log()
        st2, rc = P._prepare_state(tcfg, graph, wr, ["a"], False, _FakePBSModule(), log, dry_run=False)
        assert rc is None and st2.stage("b")["status"] == "failed" and st2.stage("c")["status"] == "pending"
    finally:
        shutil.rmtree(root, ignore_errors=True)
    # (d) no file is deleted by any case of the end-to-end sequence
    E = _e2e()
    best_rel = os.path.relpath(E["best"], E["runs"])
    miss = _present_after(E["before1"]["runs"], E["runs"], [(best_rel, best_rel + ".set_aside1")])
    assert not miss, ("files gone in the first retry", miss[:10])
    assert E["after1"]["tune"] == E["before1"]["tune"], "a file of the search changed"
    miss = _present_after(E["before3"]["runs"], E["runs"], [(best_rel, best_rel + ".set_aside2")])
    assert not miss, ("files gone in the F504 case", miss[:10])
    # (e) no field an older state file lacks: the stage records keep their keys
    old_keys = {"status", "started_utc", "finished_utc", "jobs", "resubmits", "note", "sub",
                "waiting_note"}
    for name, rec in E["state1"]["stages"].items():
        assert set(rec) <= old_keys, (name, sorted(set(rec) - old_keys))


def test_contract():
    E = _e2e()
    rc, out, _ = E["retry1"]
    lines = [ln for ln in out.splitlines() if "belongs to another finalist" in ln]
    assert len(lines) == 1, lines
    ln = lines[0]
    assert E["first"] in ln and E["second"]["trial_id"] in ln and ".set_aside1" in ln, ln
    print("    " + ln.strip()[:300])
    # --status shows the reason of a best fit whose artefact names another trial
    _write(E["scp"], E["sc2"])
    try:
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "status", "--config", E["path"]],
                           env=E["env"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120)
        row = [x for x in r.stdout.splitlines() if "| best_fit " in x]
        assert row and "names the finalist" in row[0] and E["first"] in row[0] \
            and E["second"]["trial_id"] in row[0], r.stdout
        print("    " + row[0][:300])
    finally:
        _write(E["scp"], E["sc0"])


def test_determinism():
    raise unittest.SkipTest("no randomness in the pipeline's part (sec. 4); the seeds are the "
                            "configuration's and the trainers' determinism is block 6's")


def test_edge_cases():
    # (a) by hand, on the recorded kind: the start's examination of the root
    root = _tmp()
    try:
        cfg = _recorded_cfg(root)
        stage = BestFitStage()
        dirs = stage._dirs(cfg)
        _write(dirs["search_complete"], {"finalist": {"trial_id": "T1", "nll": 1.0, "config": {}}})
        rec = os.path.join(dirs["root"], "member_seed1", "A1_seed1.json")

        def fresh(files):
            if os.path.exists(dirs["root"]):
                shutil.move(dirs["root"], os.path.join(root, "old_%d" % len(os.listdir(root))))
            for p, body in files.items():
                _write(p, body)

        cases = (
            ("unreadable finalist_config.json beside a member record",
             {dirs["finalist_config"]: "\0\0\0\0", rec: {"L": 1.0}}, True),
            ("a member record and no finalist_config.json", {rec: {"L": 1.0}}, True),
            ("finalist_config.json of another trial", {dirs["finalist_config"]: {"trial_id": "T7"}}, True),
            ("the current finalist's root", {dirs["finalist_config"]: {"trial_id": "T1"}, rec: {"L": 1.0}},
             False),
            ("an artefact naming no trial (pre-made)", {dirs["complete"]: {"pre_made": True}}, True))
        for label, files, moved in cases:
            fresh(files)
            tree = _tree(dirs["root"])
            asides = set(glob.glob(dirs["root"] + ".set_aside*"))
            st = {"sub": {"finalist": {"trial_id": "T7"}, "members": {"1": {"status": "ok"}}}}
            log = _Log()
            stage._follow_finalist(cfg, st, log)
            new = sorted(set(glob.glob(dirs["root"] + ".set_aside*")) - asides)
            if moved:
                assert len(new) == 1 and not os.path.exists(dirs["root"]), (label, new)
                assert _tree(new[0]) == tree, label
                assert len(log.lines) == 1 and "belongs to another finalist" in log.lines[0], log.lines
            else:
                assert not new and _tree(dirs["root"]) == tree and not log.lines, (label, new, log.lines)
            assert "finalist" not in st["sub"] and "members" not in st["sub"], (label, st)
            print("    %-55s %s" % (label, os.path.basename(new[0]) if new else "kept"))
        assert os.path.isdir(dirs["root"] + ".set_aside4"), "set_aside<k> counts up, none overwritten"
        # a first run (no root, no cached finalist): nothing read, nothing logged
        if os.path.exists(dirs["root"]):
            shutil.move(dirs["root"], os.path.join(root, "old_last"))
        log = _Log()
        st = {"sub": {}}
        stage._follow_finalist(cfg, st, log)
        assert not log.lines and st == {"sub": {}}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    # (b) the dry run after a changed finalist: the line, nothing moved, state unchanged
    E = _e2e()
    rc, out, n, runs, same_state = E["dry"]
    assert rc == 0, out[-3000:]
    assert "would be moved aside" in out and E["first"] in out and E["second"]["trial_id"] in out, out[-3000:]
    assert n == E["n0"] and runs == E["before1"]["runs"] and same_state, "the dry run moved or wrote"
    # (c) the audit's F504 case, which is also the second set-aside
    rc, out, new = E["retry3"]
    assert rc == 0 and "PIPELINE DONE" in out, out[-5000:]
    assert "best_fit reset to pending (--retry)" in out, out[-3000:]
    assert os.path.isdir(E["best"] + ".set_aside2") and os.path.isdir(E["best"] + ".set_aside1")
    assert sorted(new) == ["bf_synth_s1", "bf_synth_s2"], new
    bf = E["bf3"]
    assert bf["finalist"]["trial_id"] == E["first"], bf["finalist"]


# ---------------------------------------------------------------- runner ---

def main(argv=None):
    names = list(argv if argv is not None else sys.argv[1:])
    checks = [(n, o) for n, o in globals().items() if n.startswith("test_") and callable(o)]
    if names:
        checks = [(n, o) for n, o in checks if n in names]
    print("== %s" % os.path.basename(__file__))
    print("python %s | %s" % (platform.python_version(), platform.platform()))
    results = []
    try:
        for name, func in checks:
            t0 = time.time()
            try:
                func()
                status, detail = "PASS", ""
            except unittest.SkipTest as exc:
                status, detail = "SKIP", str(exc)
            except NotImplementedError as exc:
                status, detail = "TODO", str(exc)
            except AssertionError as exc:
                status, detail = "FAIL", str(exc) or traceback.format_exc()
            except Exception:
                status, detail = "ERROR", traceback.format_exc()
            results.append((name, status, time.time() - t0, detail))
    finally:
        if _E2E.get("root"):
            shutil.rmtree(_E2E["root"], ignore_errors=True)
    width = max([len(r[0]) for r in results] + [4])
    for name, status, sec, detail in results:
        head = (detail.strip().splitlines() or [""])[0]
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, sec, head[:200])).rstrip())
    for name, status, _, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[-6000:]))
    counts = dict((s, sum(1 for r in results if r[1] == s)) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (counts[s], s.lower()) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
