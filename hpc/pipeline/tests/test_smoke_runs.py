#!/usr/bin/env python3
"""Smoke test for block 29 of specs/SPEC.md: several searches in one
recorded project, each with its own core cap (D-148 (1), D-140, D-141).

Checks (specs/SPEC.md block 29, "Test oracles")
    test_t29a_graph       a recorded configuration with the searches A1
                          (S-A1), A2 (S-A2-only) and A5 (S-A5-only): the 12
                          stages real_extraction ... real_bank, search_A1,
                          best_fit_A1, ..., best_fit_A5 in that order, with
                          the deps of sec. 2.1 and each run's core group;
                          `searches` null (and giulia.json) the eight stages
                          of before; the committed linddup.json gives the 12
                          and its dry run exits 0 with a cores line per
                          search and no REFUSE line
    test_t29b_views       each search's view: tune_root, runs_root and
                          campaign; shard_globs gives no real bank on A1's
                          view and the real bank on A2's and A5's; the
                          folders of record; the renamed log lines; a view
                          of another run refused
    test_t29c_group_cap   cores_in_flight with a names filter against
                          hand-counted values; under the fake PBS, three
                          searches capped at 148 with 48-core jobs,
                          resources.max_cores 444, driver 4: at every tick no
                          search has more than 3 jobs in flight and the
                          project never more than 440 cores, and the three
                          searches each hold a running job at once (none
                          starves); a cap of 40 < 48 stops the stage at its
                          start with CapExceeded naming the search; a search
                          with max_cores null is limited by the project's
                          cap only
    test_t29d_end_to_end  the stand-in davinci (tests/standin.py: the
                          Sbi-extractor bank tools on the bank fixture, the
                          real tuner and runner): a recorded project with
                          the searches A1 (S-A1) and A2 (S-A2-only), each
                          with max_cores 1 (one job at a time though
                          max_concurrent_trials is 3: the cap binds), runs
                          to PIPELINE DONE; each SEARCH_COMPLETE.json and
                          BEST_FIT_COMPLETE.json in its own folder; A1's
                          trials without the real bank, A2's with it; no
                          search over its cap; no job twice; `status`
                          prints each search's cores line (SKIP without
                          SBIX_CLONE or the training environment)
    test_t29e_tools       search_report.py with --search on a list
                          configuration (its view's folder, its stage's
                          trials, the pass line) and without it on a null
                          one; the refusals (no --search, an unknown name,
                          --search on a null configuration);
                          search_reset.py the same; perrow_check.py and
                          purge_outputs.py refuse a list configuration with
                          one REFUSED line (exit 2), and keep their behaviour
                          on a null one
    test_convergence      not a discretisation: skipped

Run
    python3 hpc/pipeline/tests/test_smoke_runs.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table and the message of
        every failure; names of checks run only those. Exit code 1 if any
        check failed, errored or is TODO. About 1 minute without
        test_t29d_end_to_end; that one takes several more minutes and needs
        SBIX_CLONE=<Sbi-extractor checkout>, torch, sbi, zuko and
        scikit-optimize.
    MUTATION=<name> python3 hpc/pipeline/tests/test_smoke_runs.py
        Re-runs the checks with one piece of block 29 disabled; the check
        named in _PATCHES must then FAIL: group_ignored, group_whole_project,
        no_group_check, names_ignored, report_accepts.

Everything runs under the fake PBS (tests/fake_pbs/, first on PATH) in
temporary folders. Nothing touches davinci.

Pure ASCII, LF only.
"""

import contextlib
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
_TOOLS = os.path.join(_PIPE, "tools")
_FAKE = os.path.join(_HERE, "fake_pbs")
for _p in (_TOOLS, _HERE, _PIPE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

MUTATION = os.environ.pop("MUTATION", "")      # not handed to the tools' own suites' imports

import config as C              # noqa: E402  (the pipeline's, before anything else)
import pbs as PBS               # noqa: E402
import pipeline as P            # noqa: E402
import stages as S              # noqa: E402
import stages.common as CM      # noqa: E402
import stages.runs as R         # noqa: E402
import state as ST              # noqa: E402
import search_report as SREP    # noqa: E402
import search_reset as SRES     # noqa: E402

LINDDUP = os.path.join(_PIPE, "projects", "linddup", "linddup.json")
GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
SYNTH = os.path.join(_PIPE, "projects", "synth", "synth.json")
PYTHON = sys.executable

THREE = [{"name": "A1", "campaign": "S-A1", "max_cores": 148},
         {"name": "A2", "campaign": "S-A2-only", "max_cores": 148},
         {"name": "A5", "campaign": "S-A5-only", "max_cores": 148}]
WANT12 = ["real_extraction", "sim_redetection", "label_freeze", "bank_plan", "sim_bank",
          "real_bank", "search_A1", "best_fit_A1", "search_A2", "best_fit_A2", "search_A5",
          "best_fit_A5"]

# A mutation disables one piece of block 29; the check named after it must FAIL.
_PATCHES = {
    # test_t29c_group_cap: the Submitter as before block 29 (the group ignored): a search
    # takes every slot the project's cap leaves, and a cap of 40 never stops its stage
    "group_ignored": ("def _fits(self, n):\n"
                      "    b = self.budget()\n"
                      "    if b is None or n is None:\n"
                      "        return True\n"
                      "    return int(n) <= b - self.in_flight()\n"
                      "P.Submitter.fits = _fits\n"),
    # test_t29c_group_cap: the group's cap charged with the whole project's cores: the
    # second and third searches starve
    "group_whole_project": "P.Submitter.group_in_flight = lambda self: self.in_flight()\n",
    # test_t29c_group_cap: no can-never-start check of the group: the stage of a search
    # capped below its job waits for ever instead of stopping
    "no_group_check": ("_orig_check = P.Submitter.check\n"
                       "def _check(self, n):\n"
                       "    g, self._group = self._group, None\n"
                       "    try:\n"
                       "        return _orig_check(self, n)\n"
                       "    finally:\n"
                       "        self._group = g\n"
                       "P.Submitter.check = _check\n"),
    # test_t29c_group_cap: cores_in_flight's names filter ignored
    "names_ignored": ("_orig_cif = P.cores_in_flight\n"
                      "def _cif(state, names=None):\n"
                      "    return _orig_cif(state)\n"
                      "P.cores_in_flight = _cif\n"),
    # test_t29e_tools: without --search, the first search of a list taken silently
    "report_accepts": ("_orig_pick = R.pick_search\n"
                       "def _pick(cfg, name):\n"
                       "    if name is None and isinstance(cfg.data.get('searches'), list):\n"
                       "        n, v = R.search_views(cfg)[0]\n"
                       "        return v, 'search_' + n\n"
                       "    return _orig_pick(cfg, name)\n"
                       "R.pick_search = _pick\n"),
}
if MUTATION:
    if MUTATION not in _PATCHES:
        raise SystemExit("unknown MUTATION %r; known: %s" % (MUTATION, ", ".join(sorted(_PATCHES))))
    exec(_PATCHES[MUTATION], {"P": P, "R": R, "PBS": PBS})


# --------------------------------------------------------------- helpers ---

def _tmp():
    return tempfile.mkdtemp(prefix="pipeline_runs_")


def _load_json(path):
    with open(path, "rb") as fh:
        return C._strip_docs(json.loads(fh.read().decode("ascii")))


def _moved(d, root):
    """`d` with every path under `root` (the committed files name davinci)."""
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    d["real"]["cohort_config"] = os.path.join(root, "cohort.json")
    d["real"]["extract_root"] = os.path.join(root, "extracted")
    if d["real"].get("cohort_map"):
        d["real"]["cohort_map"] = os.path.join(root, "cohort_map.csv")
    for k in ("sim_main", "out_root", "tools"):
        d["sim"][k] = os.path.join(root, k)
    return d


def _write(root, d, name="project.json"):
    path = os.path.join(root, name)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path


def _list_config(root, searches=None, name="project.json"):
    """The committed linddup.json (v11: the three searches capped at 148,
    48-core trials and members, resources.max_cores 444, driver 4) with every
    path under `root`; `searches` replaces its list. (path, dict)."""
    d = _moved(_load_json(LINDDUP), root)
    if searches is not None:
        d["searches"] = [dict(q) for q in searches]
    return _write(root, d, name), d


def _null_config(root, name="null.json"):
    """The same project with `searches` null: the single search of
    search.campaign S-A1 (arm A1), as before v11. (path, dict)."""
    d = _moved(_load_json(LINDDUP), root)
    d["searches"] = None
    d["search"]["campaign"] = "S-A1"
    d["search"]["arm"] = "A1"
    return _write(root, d, name), d


@contextlib.contextmanager
def _fake_pbs(spool, delay="0.2"):
    """The fake PBS first on PATH for this process and its children."""
    saved = dict(os.environ)
    os.environ["PATH"] = _FAKE + os.pathsep + saved.get("PATH", "")
    os.environ["FAKE_PBS_SPOOL"] = spool
    os.environ["FAKE_PBS_DELAY"] = delay
    os.environ["FAKE_PBS_ENFORCE_WALLTIME"] = "0"
    os.environ["PIPELINE_SKIP_CONDA"] = "1"
    os.environ.pop("PBS_JOBID", None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _records(spool):
    """The fake PBS's job records, in job-number order."""
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        if os.path.basename(p)[:-5].isdigit():
            with open(p) as fh:
                out.append(json.load(fh))
    return sorted(out, key=lambda r: r["num"])


def _cores(rec):
    argv = ["qsub"]
    for x in rec.get("resources", []):
        argv += ["-l", x]
    return PBS.cores_of_argv(argv + ["x"]) or 0


def _peak(intervals):
    """The largest sum of weights of the intervals (start, end, weight) that
    overlap at one time (an end before a start at the same time)."""
    ev = []
    for a, b, w in intervals:
        if a is None:
            continue
        b = float("inf") if b is None else b
        if a < b:
            ev += [(a, 1, w), (b, 0, -w)]
    peak, now = 0, 0
    for _t, _k, w in sorted(ev):
        now += w
        peak = max(peak, now)
    return peak


def _capture(fn, *args):
    """(return value, stdout) of fn(*args)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args)
    return rc, buf.getvalue()


def _check_graph12(graph, cap):
    """The 12 stages of a recorded configuration with the three searches,
    their deps (sec. 2.1) and core groups (`cap` for each search)."""
    names = [s.name for s in graph]
    assert names == WANT12, names
    assert "search" not in names and "best_fit" not in names, names
    assert [s.name for s in S.topological_order(graph)] == names, "the order is a dependency order"
    by = dict((s.name, s) for s in graph)
    for n in WANT12[:6]:
        assert tuple(by[n].deps) == S.DEPS[n], (n, by[n].deps)
        assert type(by[n]) is S.BUILT[n], (n, type(by[n]))       # S1-S6 as in the default graph
        assert getattr(by[n], "core_group", None) is None and R.core_group_of(by[n]) is None, n
    search_cls, best_fit_cls, _alt = R.stage_classes()
    for q in ("A1", "A2", "A5"):
        s7, s8 = by["search_" + q], by["best_fit_" + q]
        assert type(s7) is search_cls and type(s8) is best_fit_cls, (type(s7), type(s8))
        assert tuple(s7.deps) == ("sim_bank", "real_bank"), s7.deps
        assert tuple(s8.deps) == ("search_" + q,), s8.deps
        want = (cap, ("search_" + q, "best_fit_" + q))
        assert s7.core_group == s8.core_group == want, (q, s7.core_group, s8.core_group)
        assert R.core_group_of(s7) == R.core_group_of(s8) == want


def _skip_heavy():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("the training environment is needed (%s)" % exc)


# ---------------------------------------------------------------- checks ---

def test_t29a_graph():
    root = _tmp()
    try:
        # (1) a recorded configuration with the three searches
        path, d = _list_config(root, THREE)
        cfg = C.load_config(path)
        assert C.has_search_list(cfg.data) and "campaign" not in cfg.data["search"]
        graph = S.default_graph(cfg)
        _check_graph12(graph, 148)
        assert [s.name for s in R.recorded_set_graph(cfg)] == WANT12
        # a search with max_cores null has no core group; the others keep theirs
        mixed = [dict(q) for q in THREE]
        mixed[1]["max_cores"] = None
        cfg_m = C.load_config(_list_config(root, mixed, name="mixed.json")[0])
        by = dict((s.name, s) for s in S.default_graph(cfg_m))
        assert by["search_A2"].core_group is None and by["best_fit_A2"].core_group is None
        assert R.core_group_of(by["search_A2"]) is None
        assert by["search_A1"].core_group == (148, ("search_A1", "best_fit_A1"))
        # status's and the dry run's lines: a cores line per search with a cap, and per
        # search its stop line when block 30's status_line gives one (its stub raises
        # NotImplementedError: nothing printed then; with no request it returns None)
        import stages.search_stop as SS
        st0 = ST.State.new(d["paths"]["work_root"], "linddup", cfg_m.path, cfg_m.sha256)
        cores = "cores of search %s: at most 148 (searches[].max_cores); in flight 0 in 0 job(s)"
        assert P.search_lines(cfg_m, st0) == [cores % "A1", cores % "A5"], \
            P.search_lines(cfg_m, st0)
        orig = SS.status_line
        try:
            SS.status_line = lambda view: ("stop requested 2026-11-17T12:00:00Z (kill running: no)"
                                           if view.run["name"] == "A2" else None)
            assert P.search_lines(cfg_m, st0) == [
                cores % "A1", "search A2: stop requested 2026-11-17T12:00:00Z (kill running: no)",
                cores % "A5"], P.search_lines(cfg_m, st0)
        finally:
            SS.status_line = orig
        assert P.search_lines(C.load_config(GIULIA), st0) == [], "no lines without a list"
        # (2) searches null: the eight stages of before (regression)
        for p in (GIULIA, _null_config(root)[0]):
            rec = C.load_config(p)
            g = S.default_graph(rec)
            assert [s.name for s in g] == list(S.STAGE_ORDER), [s.name for s in g]
            assert all(type(s) is S.BUILT[s.name] for s in g), [type(s) for s in g]
            assert all(getattr(s, "core_group", None) is None for s in g)
        try:
            R.recorded_set_graph(C.load_config(GIULIA))
            raise AssertionError("recorded_set_graph accepted a configuration without a list")
        except ValueError:
            pass
        # the synthetic kind's runs have no core group (no max_cores)
        syn = C.load_config(SYNTH, synthetic_ok=True)
        assert all(getattr(s, "core_group", None) is None for s in S.default_graph(syn))
        # (3) the committed linddup.json: the 12 stages, and its dry run
        lind = C.load_config(LINDDUP)
        _check_graph12(S.default_graph(lind), 148)
        dry_cfg = LINDDUP
        if os.path.isdir(lind.data["paths"]["work_root"]):
            # davinci after the first run: the dry run would read that run's state (a
            # configuration change there is refused): the same file with its work root here
            dd = _load_json(LINDDUP)
            dd["paths"]["work_root"] = os.path.join(root, "work_root")
            dry_cfg = _write(root, dd, "linddup_here.json")
        env = dict(os.environ)
        env["PIPELINE_SKIP_CONDA"] = "1"
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", dry_cfg,
                            "--dry-run"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
        out = r.stdout
        assert r.returncode == 0, out[-4000:]
        assert "REFUSE" not in out and "STOP" not in out, out[-4000:]
        lines = [ln.split(" ", 2)[2] if ln[:2] == "20" else ln for ln in out.splitlines()]
        table = [ln for ln in lines if ln.startswith("| ")]
        assert [ln.split("|")[1].strip() for ln in table[1:]] == WANT12, table
        k = lines.index("cores: at most 444 at once (resources.max_cores, D-099): the sequencer's "
                        "job 4, the stages' jobs 440; in flight per the state file: 0 in 0 job(s)")
        assert lines[k + 1:k + 4] == ["cores of search %s: at most 148 (searches[].max_cores); in "
                                      "flight 0 in 0 job(s)" % q for q in ("A1", "A2", "A5")], \
            lines[k:k + 5]
        tune = lind.data["paths"]["tune_root"]
        banks = lind.data["paths"]["banks_root"]
        assert ("    run A1: campaign S-A1, folder %s/A1/S-A1; the runner gets the simulated bank "
                "%s/linddup_sim/shard_*.npz and no real bank (S-A1 trains on the simulated "
                "windows alone, D-039)" % (tune, banks)) in lines, out[-4000:]
        for q, camp in (("A2", "S-A2-only"), ("A5", "S-A5-only")):
            assert ("    run %s: campaign %s, folder %s/%s/%s; the runner gets the simulated bank "
                    "%s/linddup_sim/shard_*.npz and the real bank %s/linddup_real/shard_*.npz"
                    % (q, camp, tune, q, camp, banks, banks)) in lines, (q, out[-4000:])
        group_line = ("    cores (block 29, D-141): the jobs of search_%s, best_fit_%s share at most "
                      "148 cores (searches[].max_cores), inside resources.max_cores: at most 3 "
                      "job(s) of 48 cores at once")
        for q in ("A1", "A2", "A5"):
            assert lines.count(group_line % (q, q)) == 2, (q, out[-4000:])   # the search, the best fit
        assert out.count("qsub -N t_<trial>") == 3 and out.count("qsub -N bf_linddu_s<k>") == 3, \
            out[-4000:]
        if dry_cfg == LINDDUP:
            assert not os.path.exists(lind.data["paths"]["work_root"]), "a dry run creates nothing"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_t29b_views():
    root = _tmp()
    try:
        path, d = _list_config(root, THREE)
        cfg = C.load_config(path)
        p = d["paths"]
        by = dict((s.name, s) for s in S.default_graph(cfg))
        views = R.search_views(cfg)
        assert [n for n, _v in views] == ["A1", "A2", "A5"], views
        ident = {"n_train": 10, "split_hash": "h" * 64, "contract_digest": "c" * 64}
        for (q, camp, real), (name, sv) in zip((("A1", "S-A1", False), ("A2", "S-A2-only", True),
                                                ("A5", "S-A5-only", True)), views):
            assert name == q
            s7, s8 = by["search_" + q], by["best_fit_" + q]
            v = s7._view(cfg)
            assert isinstance(v, R.RunView) and v.run == dict(THREE[["A1", "A2", "A5"].index(q)])
            assert s7._view(v) is v and s7._view(cfg) is v, "a view is itself; the base's is cached"
            for view in (v, sv, s8._view(cfg)):
                vd = view.data
                assert vd["paths"]["tune_root"] == os.path.join(p["tune_root"], q), vd["paths"]
                assert vd["paths"]["runs_root"] == os.path.join(p["runs_root"], q), vd["paths"]
                assert vd["search"]["campaign"] == camp, vd["search"]
                for k in ("work_root", "banks_root", "sbix_dir", "sbi_dir"):
                    assert vd["paths"][k] == p[k], k
                assert view.sha256 == cfg.sha256 and view.path == cfg.path
            assert cfg.data["paths"]["tune_root"] == p["tune_root"] and \
                "campaign" not in cfg.data["search"], "the base is not changed"
            sim, rg = CM.shard_globs(v)
            assert sim == os.path.join(p["banks_root"], "linddup_sim", "shard_*.npz"), sim
            want = os.path.join(p["banks_root"], "linddup_real", "shard_*.npz") if real else None
            assert rg == want, (q, rg)
            # the folders and artefacts of record
            sc = os.path.join(p["tune_root"], q, camp, "SEARCH_COMPLETE.json")
            assert s7.is_done(cfg) == (False, sc + " absent"), s7.is_done(cfg)
            assert s7._dirs(v)["campaign"] == os.path.join(p["tune_root"], q, camp)
            bf = os.path.join(p["runs_root"], q, "linddup_best", "BEST_FIT_COMPLETE.json")
            assert s8.is_done(cfg) == (False, bf + " absent"), s8.is_done(cfg)
            assert s8._dirs(s8._view(cfg))["search_complete"] == sc
            args = CM.tuner_common_args(v, ident)
            assert args[args.index("--campaign") + 1] == camp
            assert args[args.index("--results-dir") + 1] == os.path.join(p["tune_root"], q)
            # a stage refuses another run's view
            other = by["search_A5" if q != "A5" else "search_A1"]
            try:
                other._view(v)
                raise AssertionError("%s accepted the view of %s" % (other.name, q))
            except ValueError:
                pass
        # the log lines carry the stage's name
        seen = []

        class Rec(object):
            def log(self, t):
                seen.append(t)

            def fits(self, n):
                return "fits %d" % n

        rec = Rec()
        by["search_A2"]._rec(rec).log("search: trial abc ok")
        by["best_fit_A5"]._rec(rec).log("best_fit: BEST FIT COMPLETE: x")
        by["search_A2"]._rec(rec).log("another line")
        assert seen == ["search_A2: trial abc ok", "best_fit_A5: BEST FIT COMPLETE: x",
                        "another line"], seen
        assert by["search_A1"]._rec(rec).fits(48) == "fits 48", "everything else is the recorder's"
        # searches null: the single search, the configuration itself
        null = C.load_config(_null_config(root)[0])
        assert R.search_views(null) == [(None, null)]
        assert CM.shard_globs(null)[1] is None, "the single S-A1 search gets no real bank"
    finally:
        shutil.rmtree(root, ignore_errors=True)


class ThrottleStage(S.Stage):
    """A stand-in for a search's (or a best fit's) stage: `n_jobs` jobs of
    `ncpus` cores, as many at once as its Submitter's fits() allows -- the
    search's throttle -- each a sleep under the fake PBS. Its core group
    is given (a run stage's, or one made by hand)."""

    max_resubmits = 0

    def __init__(self, name, deps, n_jobs, ncpus, core_group, sleep):
        self.name = name
        self.deps = tuple(deps)
        self.n_jobs = int(n_jobs)
        self.ncpus = int(ncpus)
        self.core_group = core_group
        self.sleep = float(sleep)

    def _dir(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "throttle", self.name)

    def is_done(self, cfg):
        p = os.path.join(self._dir(cfg), "DONE")
        return os.path.isfile(p), p

    def _qsub(self, cfg, k):
        d = self._dir(cfg)
        if not os.path.isdir(d):
            os.makedirs(d)
        script = os.path.join(d, "job_%d.sh" % k)
        with open(script, "w") as fh:
            fh.write("#!/bin/bash\nsleep %g\necho done\n" % self.sleep)
        return ["qsub", "-N", "%s_%d" % (self.name, k), "-q", "cpu",
                "-l", "select=1:ncpus=%d:mem=1gb" % self.ncpus, "-l", "walltime=00:10:00",
                "-j", "oe", "-o", os.path.join(d, "job_%d.log" % k), script]

    def _submit_waiting(self, cfg, st, pbs):
        sub = st["sub"]
        while sub["next"] < self.n_jobs:
            if not pbs.fits(self.ncpus):
                return
            k = sub["next"]
            pbs.submit(self._qsub(cfg, k), role="job %d" % k,
                       log_path=os.path.join(self._dir(cfg), "job_%d.log" % k), ncpus=self.ncpus)
            sub["next"] = k + 1

    def start(self, cfg, st, pbs):
        st["sub"]["next"] = 0
        if not pbs.fits(self.ncpus):            # check() first: a job that can never start raises
            raise S.Deferred("a job of %d cores does not fit now" % self.ncpus)
        self._submit_waiting(cfg, st, pbs)

    def poll(self, cfg, st, pbs):
        S.Stage.poll(self, cfg, st, pbs)
        self._submit_waiting(cfg, st, pbs)
        if st["sub"]["next"] < self.n_jobs or any(not j.get("finished") for j in st["jobs"]):
            return "running"
        with open(os.path.join(self._dir(cfg), "DONE"), "w") as fh:
            fh.write("done\n")
        return "finished"

    def verify(self, cfg, st):
        ok, why = self.is_done(cfg)
        return (True, why) if ok else (False, "killed: " + why)


def _throttle_graph(cfg, plan, sleep=1.5):
    """ThrottleStage stand-ins for the run stages of `cfg`'s graph, with the
    real stages' names and core groups; `plan` {name: (jobs, core group
    override or None)}; a search after nothing, its best fit after it."""
    real = dict((s.name, s) for s in S.default_graph(cfg))
    out = []
    for name, (n_jobs, override) in plan.items():
        grp = override if override is not None else real[name].core_group
        deps = () if name.startswith("search_") else ("search_" + name[len("best_fit_"):],)
        out.append(ThrottleStage(name, deps, n_jobs, 48, grp, sleep))
    return out


def _run_throttle(cfg, graph, spool, max_ticks=None):
    """The sequencer in this process on `graph` under the fake PBS; the
    probe at every full tick reads the state file: (rc, log lines, ticks),
    each tick {group: (cores, jobs)} for the searches with a cap, '*' the
    project's, and the per-search lines of status."""
    wr = cfg.data["paths"]["work_root"]
    groups = dict((s.name[len("search_"):], s.core_group) for s in graph
                  if s.name.startswith("search_") and s.core_group is not None)
    lines, ticks = [], []

    def log(text):
        lines.append(str(text))

    def probe():
        st = ST.State.load(wr)
        row = {"*": P.cores_in_flight(st)[:2]}
        for q, grp in groups.items():
            row[q] = P.cores_in_flight(st, names=grp[1])[:2]
        row["lines"] = P.search_lines(cfg, st)
        ticks.append(row)

    with _fake_pbs(spool):
        rc = P.run(cfg, graph=graph, poll_seconds=0.3, log=log, on_tick=probe,
                   max_ticks=max_ticks)
    return rc, lines, ticks


def _t29c_reference():
    """(a) the names filter of cores_in_flight on a hand-made state, counted as
    the whole project's -- a job recorded `after` another counted once (the
    larger); the Submitter of a group against hand-counted values."""
    root = _tmp()
    try:
        st = ST.State.new(os.path.join(root, "wr"), "linddup", "/x.json", "a" * 64)

        def job(i, n, after=None, finished=False):
            return {"id": "%d.pbs" % i, "role": "j", "ncpus": n, "after": after,
                    "finished": finished}

        st.data["stages"] = {
            "search_A1": {"status": "running", "jobs": [job(1, 48), job(2, 48), job(3, 48, finished=True)]},
            "best_fit_A1": {"status": "running", "jobs": [job(4, 48)]},
            "search_A2": {"status": "running", "jobs": [job(5, 48), job(6, 8, after="5.pbs"),
                                                         job(7, 4, after="1.pbs")]},
            "best_fit_A2": {"status": "done", "jobs": [job(8, 48)]},       # a done stage: no cores
            "sim_bank": {"status": "running", "jobs": [{"id": "9.pbs", "role": "x", "cmd": []}]}}
        assert P.cores_in_flight(st, names=("search_A1", "best_fit_A1")) == (144, 3, [])
        # job 7 is `after` job 1 of another group: alone in its group it is a root
        assert P.cores_in_flight(st, names=["search_A2", "best_fit_A2"]) == (52, 3, [])
        assert P.cores_in_flight(st, names=["sim_bank"]) == (0, 0, ["9.pbs"])
        assert P.cores_in_flight(st, names="search_A1") == (96, 2, []), "one name as a string"
        assert P.cores_in_flight(st, names=()) == (0, 0, [])
        # the whole project: job 7 inside job 1's footprint, max(48, 4) = 48
        assert P.cores_in_flight(st) == P.cores_in_flight(st, names=None) == (192, 6, ["9.pbs"]), \
            P.cores_in_flight(st)
        # the Submitter of a group: a job above the cap can never start; one that fits
        path, d = _list_config(root, THREE)
        cfg = C.load_config(path)
        lines = []
        sub = P.Submitter(None, st, "search_A1", lines.append, cfg,
                          group=(148, ("search_A1", "best_fit_A1")))
        assert sub.group_in_flight() == 144 and sub.group_free() == 4
        assert sub.fits(4) and not sub.fits(48), "48 > 148 - 144"
        assert sub.budget() == 440 and sub.free() == 440 - 192
        small = P.Submitter(None, st, "search_A1", lines.append, cfg,
                            group=(40, ("search_A1", "best_fit_A1")))
        for call in (small.check, small.fits):
            try:
                call(48)
                raise AssertionError("a 48-core job under a cap of 40 was accepted")
            except P.CapExceeded as exc:
                assert "search A1" in str(exc) and "max_cores 40" in str(exc), str(exc)
        assert small.check(40) is True
        none = P.Submitter(None, st, "search_A1", lines.append, cfg)
        assert none.group_in_flight() is None and none.fits(48) and none.check(48)
        free = P.Submitter(None, st, "search_A1", lines.append)
        assert free.fits(10 ** 6), "no configuration, no group: everything fits"
        # the configuration refuses such a cap (so the run below sets it on the stage by hand)
        bad = _load_json(path)
        bad["searches"][0]["max_cores"] = 40
        assert any("A1: max_cores 40 cannot hold its largest job (48 cores" in x
                   for x in C.validate(bad)), C.validate(bad)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _t29c_three():
    """(b) three searches capped at 148, 48-core jobs, max_cores 444, driver 4:
    the project's jobs share 440 cores (9 jobs of 48); each search may hold 3
    at once, and all three do at once (none starves)."""
    root = _tmp()
    try:
        path, d = _list_config(root, THREE)
        cfg = C.load_config(path)
        plan = {}
        for q in ("A1", "A2", "A5"):
            plan["search_" + q] = (5, None)
            plan["best_fit_" + q] = (2, None)
        graph = _throttle_graph(cfg, plan)
        spool = os.path.join(root, "spool")
        rc, lines, ticks = _run_throttle(cfg, graph, spool)
        text = "\n".join(lines)
        assert rc == 0 and "PIPELINE DONE: every stage done or skipped" in text, text[-3000:]
        assert ticks, "no full tick"
        for row in ticks:
            for q in ("A1", "A2", "A5"):
                assert row[q][1] <= 3 and row[q][0] <= 148, (q, row)
            assert row["*"][0] <= 440, row
        full = [row for row in ticks if all(row[q][1] == 3 for q in ("A1", "A2", "A5"))]
        assert full, ("no tick had each search holding its 3 jobs at once",
                      [(r["A1"], r["A2"], r["A5"]) for r in ticks[:6]])
        assert full[0]["lines"] == ["cores of search %s: at most 148 (searches[].max_cores); in "
                                    "flight 144 in 3 job(s)" % q for q in ("A1", "A2", "A5")], \
            full[0]["lines"]
        recs = _records(spool)
        assert len(recs) == 3 * (5 + 2), "every job once: %d submitted" % len(recs)
        for q in ("A1", "A2", "A5"):
            mine = [r for r in recs if r["name"].startswith(("search_%s_" % q, "best_fit_%s_" % q))]
            assert len(mine) == 7, (q, [r["name"] for r in mine])
            held = _peak([(r["submitted"], r["finished"], 1) for r in mine])
            assert held <= 3, "search %s held %d jobs at once (the scheduler's records)" % (q, held)
        assert _peak([(r["submitted"], r["finished"], _cores(r)) for r in recs]) <= 440
        # none starves: a moment at which each of the three searches has a job running
        run_ev = []
        for r in recs:
            q = r["name"].split("_")[1]
            if r["name"].startswith("search_") and r["started"] is not None:
                run_ev += [(r["started"], 1, q), (r["finished"], 0, q)]
        now, together = {"A1": 0, "A2": 0, "A5": 0}, False
        for _t, kind, q in sorted(run_ev):
            now[q] += 1 if kind else -1
            together = together or all(v > 0 for v in now.values())
        assert together, "the three searches never ran a job at the same time"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _t29c_small_cap():
    """(c) a cap of 40 below the 48-core job: the stage stops at its start with
    CapExceeded naming the search, nothing submitted."""
    root = _tmp()
    try:
        path, d = _list_config(root, THREE)
        cfg = C.load_config(path)
        plan = {"search_A1": (2, (40, ("search_A1", "best_fit_A1"))), "search_A2": (2, None)}
        spool = os.path.join(root, "spool")
        rc, lines, ticks = _run_throttle(cfg, _throttle_graph(cfg, plan), spool, max_ticks=8)
        text = "\n".join(lines)
        assert rc == P.EXIT_FAILED, (rc, text[-3000:])
        stop = [ln for ln in lines if ln.startswith("STOP: stage search_A1 refused:")]
        assert stop and "search A1" in stop[0] and "max_cores 40" in stop[0] and \
            "a job of 48 cores can never start" in stop[0], text[-3000:]
        assert _records(spool) == [], "nothing was submitted"
        state = ST.State.load(cfg.data["paths"]["work_root"])
        assert state.stage("search_A1")["status"] == "failed", state.stage("search_A1")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _t29c_null_cap():
    """(d) a search with max_cores null is limited by the project's cap only,
    beside a capped one."""
    root = _tmp()
    try:
        mixed = [dict(THREE[0]), {"name": "A2", "campaign": "S-A2-only", "max_cores": None}]
        path, d = _list_config(root, mixed)
        cfg = C.load_config(path)
        plan = {"search_A1": (3, None), "search_A2": (8, None)}
        graph = _throttle_graph(cfg, plan)
        assert [s.core_group for s in graph] == [(148, ("search_A1", "best_fit_A1")), None]
        spool = os.path.join(root, "spool")
        rc, lines, ticks = _run_throttle(cfg, graph, spool)
        text = "\n".join(lines)
        assert rc == 0 and "PIPELINE DONE" in text, text[-3000:]
        recs = _records(spool)
        a2 = [r for r in recs if r["name"].startswith("search_A2_")]
        held = _peak([(r["submitted"], r["finished"], 1) for r in a2])
        # A1 takes its 3 (144 cores) in the first tick; A2 the 296 left: 6 of 48
        assert held >= 6, "search A2 (max_cores null) held at most %d jobs at once" % held
        assert _peak([(r["submitted"], r["finished"], _cores(r)) for r in recs]) <= 440
        for row in ticks:
            assert row["*"][0] <= 440 and row["A1"][1] <= 3, row
            assert [ln for ln in row["lines"] if "search A2" in ln] == [], row["lines"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


T29C_PARTS = (_t29c_reference, _t29c_three, _t29c_small_cap, _t29c_null_cap)


def test_t29c_group_cap():
    for part in T29C_PARTS:
        part()


def _standin_project(root):
    """The stand-in davinci of block 8 (tests/standin.py) with a searches
    list: A1 (S-A1) and A2 (S-A2-only), each with max_cores 1, small budgets
    (two batch levels: two probes per gate). One job per search at a time also
    keeps the runners' memory (about 2 GB each on these 18,000-sample
    windows) inside a shared sandbox's. (path, dict, env)."""
    import standin as SI
    clone = SI.sbix_clone()
    if clone is None:
        raise unittest.SkipTest("no Sbi-extractor checkout (set SBIX_CLONE)")
    sbix = SI.build_sbix(root, clone)
    fakebin = SI.fake_conda(root)
    # block 8's data set: 12 more topologies of 2 simulations per task
    extra = dict((i, dict((k, [0, 1]) for k in range(3, 15))) for i in range(3))
    ds = SI.write_dataset(root, sbix, extra_topos=extra)
    cfg = SI.synthetic_config(root, sbix, _REPO, ds)
    cfg["split"] = {"fractions": [0.6, 0.2, 0.2]}
    del cfg["search"]["campaign"]
    del cfg["search"]["arm"]
    cfg["searches"] = [{"name": "A1", "campaign": "S-A1", "max_cores": 1},
                       {"name": "A2", "campaign": "S-A2-only", "max_cores": 1}]
    cfg["search"].update({
        "shape_anchors": {"p": 23, "d_theta": 23, "embedding_dim": 8},
        "space_overrides": {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                            "embedding_size": [8, 8], "hidden_features": [24, 24],
                            "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                            "head_pool_ops": [0, 1]},
        "n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 1,
        "max_concurrent_trials": 3, "gp_seed": 3, "train_seed": 0,
        "budget_gate": {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5},
        "max_total_walltime_h": 400})
    cfg["search"]["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "8gb",
                                   "walltime": "00:20:00", "queue": "test", "max_resubmits": 1})
    cfg["best_fit"].update({"split_seed": 1, "member_seeds": [1, 2], "passes": 1.0, "epochs": 2,
                            "patience_epochs": 10, "n_posterior_draws": 16, "ncpus": 1,
                            "mem": "8gb", "walltime": "00:20:00", "queue": "test",
                            "max_resubmits": 1})
    cfg["loader"] = {"buffer_shards": 2}
    path = _write(root, cfg, "standin.json")
    env = dict(os.environ)
    env["PATH"] = _FAKE + os.pathsep + fakebin + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = os.path.join(root, "spool")
    env["FAKE_PBS_DELAY"] = "0.3"
    env["PIPELINE_SKIP_CONDA"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = _PIPE + os.pathsep + env.get("PYTHONPATH", "")
    for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PROFILE", "QUEUE", "CONCURRENCY", "PBS_JOBID"):
        env.pop(k, None)
    return path, cfg, env


def test_t29d_end_to_end():
    _skip_heavy()
    root = _tmp()
    try:
        path, cfg, env = _standin_project(root)
        C.load_config(path)                    # a valid v11 configuration with a searches list
        p = cfg["paths"]
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                            "--dry-run"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=300)
        assert r.returncode == 0 and "REFUSE" not in r.stdout, r.stdout[-4000:]
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "run", "--config", path,
                            "--poll-seconds", "0.5"], cwd=root, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=3000)
        out = r.stdout
        assert r.returncode == 0 and "PIPELINE DONE: every stage done or skipped" in out, \
            out[-6000:]
        for q, camp, arm, real in (("A1", "S-A1", "A1", None),
                                   ("A2", "S-A2-only", "A2",
                                    os.path.join(p["banks_root"], "synth_real", "shard_*.npz"))):
            folder = os.path.join(p["tune_root"], q, camp)
            sc = json.load(open(os.path.join(folder, "SEARCH_COMPLETE.json")))
            assert sc["campaign"] == camp and sc["ended_by"] == "rounds", sc
            assert sc["n_trials"]["ok"] == 3, sc["n_trials"]
            bf = json.load(open(os.path.join(p["runs_root"], q, "synth_best",
                                             "BEST_FIT_COMPLETE.json")))
            assert bf["ensemble"]["M"] == 2, bf
            assert bf["finalist"]["trial_id"] == sc["finalist"]["trial_id"], (bf["finalist"],
                                                                               sc["finalist"])
            ledger = sorted(glob.glob(os.path.join(folder, "trials", "*.json")))
            assert len(ledger) == 3, ledger
            for lp in ledger:
                rec = json.load(open(lp))
                assert rec["status"] == "ok" and rec["arm"] == arm, \
                    (q, lp, rec.get("status"), rec.get("arm"), rec.get("error"),
                     "(runner exited -9: a SIGKILL, the out-of-memory killer of a shared sandbox?)")
                argv = rec["argv"]
                got = argv[argv.index("--real-shards") + 1] if "--real-shards" in argv else None
                assert got == real, (q, got)
            assert "search_%s: SEARCH COMPLETE" % q in out and \
                "best_fit_%s: BEST FIT COMPLETE" % q in out, (q, out[-4000:])
        assert not os.path.exists(os.path.join(p["tune_root"], "S-A1")), "no single-search folder"
        assert not os.path.exists(os.path.join(p["runs_root"], "synth_best"))
        state = json.load(open(os.path.join(p["work_root"], "state.json")))
        statuses = dict((k, v["status"]) for k, v in state["stages"].items())
        assert statuses == {"real_extraction": "skipped", "sim_redetection": "skipped",
                            "label_freeze": "skipped", "bank_plan": "done", "sim_bank": "done",
                            "real_bank": "done", "search_A1": "done", "best_fit_A1": "done",
                            "search_A2": "done", "best_fit_A2": "done"}, statuses
        # the scheduler's records: no job twice; no search over its own cap
        recs = _records(os.path.join(root, "spool"))
        keys = [(r["name"], os.path.normpath(r["out"])) for r in recs]
        assert len(keys) == len(set(keys)), "a job submitted twice"
        for q in ("A1", "A2"):
            tune_q = os.path.join(p["tune_root"], q) + os.sep
            mine = [r for r in recs if r["out"].startswith((tune_q, os.path.join(p["runs_root"], q)
                                                            + os.sep))]
            names = [r["name"] for r in mine]
            assert sum(1 for n in names if n.startswith("t_")) == 3 and \
                sum(1 for n in names if n.startswith("bf_")) == 2 and \
                sum(1 for n in names if n.startswith("gate_b")) == 2, (q, names)
            held = _peak([(r["submitted"], r["finished"], _cores(r)) for r in mine])
            # round 0's two trials would run together (max_concurrent_trials 3): the cap binds
            assert held == 1, "search %s held %d cores at once; its max_cores is 1" % (q, held)
        # status: each search's cores line, nothing in flight
        r = subprocess.run([PYTHON, os.path.join(_PIPE, "pipeline.py"), "status", "--config", path],
                           cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        assert r.returncode == 0, r.stdout[-3000:]
        for q in ("A1", "A2"):
            assert ("cores of search %s: at most 1 (searches[].max_cores); in flight 0 in 0 job(s)"
                    % q) in r.stdout, r.stdout[-3000:]
        assert r.stdout.count("| done     | yes") == 7 and r.stdout.count("| skipped  | yes") == 3, \
            r.stdout[-3000:]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _campaign(view, stage_name, work_root, salt):
    """A campaign folder of three trials -- `ok`, `failed`, `running` -- in
    `view`'s folders, and the state's record of its search stage
    `stage_name` (the trials' ids depend on `salt`). (folder, {kind: id})."""
    import hashlib
    d = view.data
    camp = os.path.join(d["paths"]["tune_root"], d["search"]["campaign"])
    for sub in ("trials", "pending", "runs", "jobs"):
        os.makedirs(os.path.join(camp, sub))
    ids = dict((k, hashlib.sha256((salt + k).encode("ascii")).hexdigest())
               for k in ("ok", "failed", "running"))
    with open(os.path.join(camp, "trials", ids["ok"] + ".json"), "w") as fh:
        json.dump({"trial_id": ids["ok"], "config": {"lr": 0.001}, "status": "ok", "nll": 0.5,
                   "error": ""}, fh)
    with open(os.path.join(camp, "trials", ids["failed"] + ".json"), "w") as fh:
        json.dump({"trial_id": ids["failed"], "config": {"lr": 0.004}, "status": "failed",
                   "error": "runner exited 1"}, fh)
    with open(os.path.join(camp, "pending", ids["running"] + ".json"), "w") as fh:
        json.dump({"trial_id": ids["running"], "config": {"lr": 0.002}}, fh)
    st = (ST.State.load(work_root) if os.path.isdir(work_root) else None) or \
        ST.State.new(work_root, d["dataset"], view.path, view.sha256)
    trials, jobs = {}, []
    for i, k in enumerate(("ok", "failed", "running")):
        jid = "%d.pbsserver01" % (1900000 + i)
        trials[ids[k]] = {"round": 0, "jobs": [jid], "resubmits": 0, "status": k,
                          "fault_checked": True}
        jobs.append({"id": jid, "role": "trial %s" % ids[k][:12], "finished": True,
                     "exit_status": 0 if k == "ok" else 1, "walltime_s": 60.0})
    st.data["stages"][stage_name] = {
        "status": "running", "jobs": jobs, "resubmits": 0, "note": "", "started_utc": None,
        "finished_utc": None, "sub": {"phase": "rounds", "round": 0, "trials": trials,
                                      "rounds": [{"round": 0, "trial_ids": sorted(trials)}]}}
    st.save()
    return camp, ids


def test_t29e_tools():
    import perrow_check as PCK
    import purge_outputs as PUR
    root = _tmp()
    try:
        path, d = _list_config(root, THREE)
        cfg = C.load_config(path)
        wr = d["paths"]["work_root"]
        views = dict(R.search_views(cfg))
        camp, ids = _campaign(views["A2"], "search_A2", wr, "list")
        os.makedirs(os.path.join(root, "null"))
        npath, nd = _null_config(os.path.join(root, "null"))       # its own folders
        ncfg = C.load_config(npath)
        ncamp, nids = _campaign(ncfg, "search", nd["paths"]["work_root"], "null")
        refusal = "REFUSED: this configuration has the searches A1, A2, A5: give --search <name>"

        # search_report.py: --search names a search of the list
        rc, out = _capture(SREP.main, ["--config", path, "--search", "A2"])
        assert rc == 0, out
        lines = out.splitlines()
        assert "campaign folder %s" % camp in lines, out
        assert any(ln.startswith("search A2 (campaign S-A2-only), the stage search_A2 of a "
                                 "configuration with the searches A1, A2, A5") for ln in lines), out
        assert any(ln.startswith("state ") and "the search stage running, phase rounds, round 0"
                   in ln for ln in lines), out
        for k in ("ok", "failed", "running"):
            assert sum(1 for ln in lines if ln.startswith("trial %s " % ids[k][:12])) == 1, (k, out)
        assert lines[-1] == "SEARCH REPORT OK: 3 trial(s) read; nothing was written", lines[-1]
        rc, out = _capture(SREP.main, ["--config", path, "--search", "A5"])
        assert rc == 0 and out.splitlines()[-1] == \
            "SEARCH REPORT OK: 0 trial(s) read; nothing was written", out
        assert "campaign folder %s" % os.path.join(d["paths"]["tune_root"], "A5", "S-A5-only") \
            in out, out
        # the refusals: no --search, an unknown name, --search on a null configuration
        for args in (["--config", path], ["--config", path, "--search", "A9"]):
            rc, out = _capture(SREP.main, args)
            assert rc == 2 and out.splitlines() == [refusal], (args, out)
        rc, out = _capture(SREP.main, ["--config", npath, "--search", "A1"])
        assert rc == 2 and len(out.splitlines()) == 1 and out.startswith("REFUSED: ") and \
            "--search" in out, out
        # a null configuration without --search: its single search, as before
        rc, out = _capture(SREP.main, ["--config", npath])
        assert rc == 0 and "campaign folder %s" % ncamp in out and \
            out.splitlines()[-1] == "SEARCH REPORT OK: 3 trial(s) read; nothing was written", out

        # search_reset.py: the same rule; the listing of the view, its stage, its --retry
        spool = os.path.join(root, "spool")
        with _fake_pbs(spool):
            rc, out = _capture(SRES.main, ["--config", path, "--search", "A2"])
            assert rc == 0, out
            assert "SEARCH RESET (nothing done yet): campaign folder %s" % camp in out, out
            assert "the state's search stage search_A2: running, 1 round(s) recorded" in out, out
            assert "the submission's --retry search_A2 resets the stage" in out, out
            code = out.split("CODE ")[1].split()[0]
            rc, out = _capture(SRES.main, ["--config", path, "--search", "A2", "--confirm", code])
            assert rc == 0 and "next: submit with --retry search_A2 --accept-config-change" in out, out
            assert not os.path.isdir(os.path.join(camp, "trials")), "the round was set aside"
            aside = glob.glob(camp + "_set_aside_*")
            assert len(aside) == 1 and os.path.isfile(os.path.join(aside[0], "trials",
                                                                   ids["ok"] + ".json")), aside
            for args in (["--config", path], ["--config", path, "--search", "A9"]):
                rc, out = _capture(SRES.main, args)
                assert rc == 2 and out.splitlines() == [refusal], (args, out)
            rc, out = _capture(SRES.main, ["--config", npath, "--search", "A1"])
            assert rc == 2 and out.startswith("REFUSED: ") and len(out.splitlines()) == 1, out
            rc, out = _capture(SRES.main, ["--config", npath])
            assert rc == 0 and "the state's search stage: running, 1 round(s) recorded" in out and \
                "the submission's --retry search resets the stage" in out, out

            # perrow_check.py and purge_outputs.py: a list configuration refused, one line
            for fn, args, tool in ((PCK.main, ["--config", path], "perrow_check.py"),
                                   (PUR.main, ["--config", path, "--what", "search,best_fit"],
                                    "purge_outputs.py")):
                rc, out = _capture(fn, args)
                assert rc == 2, (tool, rc, out)
                assert len(out.splitlines()) == 1 and out.startswith("REFUSED: ") and \
                    "not supported for several searches" in out and tool in out, (tool, out)
            assert os.path.isdir(views["A2"].data["paths"]["tune_root"]), "nothing deleted"
            # a null configuration: their behaviour of before (the per-row check finds no
            # per-row file, exit 3; the purge lists its targets and its CODE, exit 0)
            r = subprocess.run([PYTHON, os.path.join(_TOOLS, "perrow_check.py"), "--config", npath],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, timeout=300)
            assert r.returncode == 3 and "PERROW CHECK: no per-row file was read" in r.stdout, \
                r.stdout[-3000:]
            rc, out = _capture(PUR.main, ["--config", npath, "--what", "search"])
            assert rc == 0 and "CODE " in out and "REFUSED" not in out, out
            assert os.path.isdir(ncamp), "a listing deletes nothing"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the graph, the caps and the tools have no "
                            "convergence order")


CHECKS = [("test_t29a_graph", test_t29a_graph), ("test_t29b_views", test_t29b_views),
          ("test_t29c_group_cap", test_t29c_group_cap),
          ("test_t29d_end_to_end", test_t29d_end_to_end), ("test_t29e_tools", test_t29e_tools),
          ("test_convergence", test_convergence)]


def main():
    only = set(sys.argv[1:])
    print("== %s" % os.path.basename(__file__))
    print("python %s | fake PBS %s | SBIX_CLONE %s" % (sys.version.split()[0], _FAKE,
                                                       os.environ.get("SBIX_CLONE", "(unset)")))
    if MUTATION:
        print("MUTATION=%s: the named piece is disabled; the check it names must FAIL" % MUTATION)
    results = []
    for name, func in CHECKS:
        if only and name not in only:
            continue
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(r[0]) for r in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:200]))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[:6000]))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP"))
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
