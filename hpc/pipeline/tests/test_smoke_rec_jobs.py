#!/usr/bin/env python3
"""Smoke test of block 38: the jobs and the run sheet of a recorded project's evaluation (specs/SPEC.md block 38, "Test
oracles", checks R38*).

Checks
    test_r38a_dry_run          the dry run on the committed synth.json and synth_rec_eval.json; the wrapper
    test_r38b_as_recorded      under the fake PBS, as recorded: samplings, report, validation; done, stale, live
    test_r38c_recorded         under the fake PBS, the recorded kind: no validate unit; an incomplete best fit
    test_r38d_static           ASCII and LF, bash -n, the launcher's imports

R38b and R38c run the checkout's ev_rec_report.py and ev_rec_validate.py
when both are implemented; while either is a stub (block 37 written in
parallel) they run on a copy of hpc/pipeline in a temporary directory in
which this file replaces the two programs by stand-ins of a few lines
(STANDIN_REPORT, STANDIN_VALIDATE: the contract of wave C), so that the
launcher's rules are tested. The samplings are block 35's real ones in both
cases. The summary line of each check says which programs ran.

Nothing is written into the fixture's tree (it may be shared): every
output -- the evaluation's out_root, the fake PBS's spool, the copies of
the configurations, a best fit copied without its BEST_FIT_COMPLETE.json --
is under a temporary directory of this file's, removed at the end.

Run
    python3 hpc/pipeline/tests/test_smoke_rec_jobs.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built
        before (rec_fixture.py); REC_KEEP=1 keeps the one this run builds.

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

import rec_fixture  # noqa: E402

import ast  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

SEED = 20261123  # every random number of this file derives from it (none is drawn)

PYTHON = sys.executable
_REPO = os.path.abspath(os.path.join(_PIPE, "..", ".."))
SUBMIT = os.path.join(_EVAL, "submit_rec_eval.sh")
LAUNCHER = os.path.join(_EVAL, "ev_rec_submit.py")
SYNTH = os.path.join(_PIPE, "projects", "synth", "synth.json")
SYNTH_EVAL = os.path.join(_PIPE, "projects", "synth", "synth_eval.json")
SYNTH_REC_EVAL = os.path.join(_PIPE, "projects", "synth", "synth_rec_eval.json")
_FAKE = os.path.join(_HERE, "fake_pbs")

# The fake PBS's jobs: with the stand-ins, R38b's five jobs (block 35's
# three samplings on the fixture among them) ended in 8 s and R38c's three in
# 6 s in the sandbox (2 cores, shared with another process); the real report
# (block 37: metrics and figures) takes longer. An hour allows for the real
# programs on a machine many times slower; a job that has not ended by then
# is a failure, not a hang.
WAIT_JOBS_S = 3600
WAIT_POLL_S = 2.0
SLOW_DELAY_S = 30

_TMP = {}


def _tmp():
    """This file's temporary directory (one per process)."""
    if not _TMP.get("root"):
        _TMP["root"] = tempfile.mkdtemp(prefix="rec_jobs_")
    return _TMP["root"]


def _cleanup_tmp():
    root = _TMP.pop("root", None)
    if root:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------ helpers ---

def _read(path):
    with open(path) as fh:
        return json.load(fh)


def _write(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


def _cfg_copy(src, dst, sbi, **paths):
    """A copy of a project configuration with paths.sbi_dir = sbi and the
    other paths given replaced."""
    d = _read(src)
    d["paths"]["sbi_dir"] = sbi
    d["paths"].update(paths)
    return _write(dst, d)


def _ecfg_copy(src, dst, **over):
    """A copy of an evaluation's configuration with `over` replacing
    top-level keys (`jobs` and `gate` merged)."""
    d = _read(src)
    for k, v in over.items():
        if k in ("jobs", "gate") and isinstance(v, dict):
            d[k].update(v)
        else:
            d[k] = v
    return _write(dst, d)


_LIGHT_JOBS = {"sample": {"ncpus": 1, "mem": "2gb", "walltime": "01:00:00"},
               "report": {"ncpus": 1, "mem": "2gb", "walltime": "01:00:00"},
               "validate": {"ncpus": 1, "mem": "1gb", "walltime": "00:20:00"}}


def _submit(*argv, env=None, timeout=600):
    """The wrapper with SKIP_CONDA=1; (exit code, output)."""
    e = dict(env or os.environ)
    e["SKIP_CONDA"] = "1"
    e.setdefault("ROADMAP_DISABLE", "1")
    r = subprocess.run(["bash", SUBMIT] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout, env=e)
    return r.returncode, r.stdout


def _direct(*argv, skip="1", env=None, timeout=300):
    """The launcher itself; (exit code, output)."""
    e = dict(env or os.environ)
    e["PIPELINE_SKIP_CONDA"] = skip
    r = subprocess.run([PYTHON, LAUNCHER] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, env=e, timeout=timeout)
    return r.returncode, r.stdout


def _strip(text):
    return [l for l in text.splitlines() if not l.startswith("[rec evaluation]")]


def _fake_pbs_env(spool, delay="0.2"):
    env = dict(os.environ)
    env.update({"PATH": _FAKE + os.pathsep + env.get("PATH", ""), "FAKE_PBS_SPOOL": spool,
                "FAKE_PBS_DELAY": delay, "FAKE_PBS_ENFORCE_WALLTIME": "0",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PIPELINE_QSTAT_RETRY_WAIT_S": "0.2", "ROADMAP_DISABLE": "1"})
    os.makedirs(spool, exist_ok=True)
    return env


def _records(spool):
    import test_smoke_synthetic as TS
    return TS._records(spool)


def _wait(spool, n):
    t0 = time.time()
    while time.time() - t0 < WAIT_JOBS_S:
        recs = _records(spool)
        if len(recs) == n and all(r["finished"] for r in recs):
            return dict((r["name"], r) for r in recs), time.time() - t0
        time.sleep(WAIT_POLL_S)
    raise AssertionError("the fake PBS's %d job(s) did not all end within %d s: %s"
                         % (n, WAIT_JOBS_S, [(r["name"], r["finished"]) for r in _records(spool)]))


def _log_of(rec):
    with open(rec["out"]) as fh:
        return fh.read()


def _files_under(d):
    out = set()
    for a, _ds, fs in os.walk(d):
        for f in fs:
            out.add(os.path.join(a, f))
    return out


def _sha(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------- real or stand-in ---

STANDIN_REPORT = '''#!/usr/bin/env python3
"""Stand-in for ev_rec_report.py (test_smoke_rec_jobs.py, block 38): block 37 is a stub in
the checkout. Writes metrics/METRICS.json and REC_REPORT_COMPLETE.json with the current
report_inputs, prints REC REPORT OK."""
import argparse, datetime, json, os, time
import _evalpath  # noqa: F401
import ev_rec_config, ev_rec_plan
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--eval-config"); ap.add_argument("--ncpus")
a = ap.parse_args()
ecfg = ev_rec_config.load(a.eval_config)
inputs = ev_rec_plan.report_inputs(ev_rec_plan.load_view(a.config, ecfg), ecfg)
os.makedirs(ecfg.sub("metrics"), exist_ok=True)
with open(os.path.join(ecfg.sub("metrics"), "METRICS.json"), "w") as fh:
    json.dump({"standin": True, "ensembles": sorted(inputs["samples"]), "t": time.time()}, fh)
with open(os.path.join(ecfg.out_root, "REC_REPORT_COMPLETE.json"), "w") as fh:
    json.dump({"schema": "rec_eval_report/1", "inputs": inputs, "standin": True,
               "finished": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}, fh)
print("REC REPORT OK (stand-in)")
'''

STANDIN_VALIDATE = '''#!/usr/bin/env python3
"""Stand-in for ev_rec_validate.py (test_smoke_rec_jobs.py, block 38): writes the record of
block 36's --record from the stand-in report's metrics, prints REC VALIDATION OK."""
import argparse, datetime, hashlib, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--old"); ap.add_argument("--new"); ap.add_argument("--record")
a = ap.parse_args()
p = os.path.join(a.new, "metrics", "METRICS.json")
with open(p, "rb") as fh:
    blob = fh.read()
n = len(json.loads(blob.decode("ascii"))["ensembles"])
rec = {"schema": "rec_eval_validation/1", "old": os.path.abspath(a.old),
       "new": os.path.abspath(a.new), "ok": True, "n_ensembles": n, "failures": [],
       "nll_tol": 0.002, "coverage_tol": 0.01,
       "metrics_index_sha256": hashlib.sha256(blob).hexdigest(),
       "finished": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
with open(a.record + ".tmp", "w") as fh:
    json.dump(rec, fh)
os.replace(a.record + ".tmp", a.record)
print("REC VALIDATION OK: %d ensemble(s)" % n)
'''


def _real_programs():
    """True when the checkout's ev_rec_report.run and ev_rec_validate.validate
    are implemented: each is called on paths that do not exist, which an
    implementation refuses at once (no report is run) and a stub answers
    with NotImplementedError."""
    import ev_rec_report
    import ev_rec_validate
    nowhere = os.path.join(_tmp(), "nowhere")
    real = True
    for fn, args in ((ev_rec_report.run, (os.path.join(nowhere, "none.json"),
                                          os.path.join(nowhere, "none_rec_eval.json"))),
                     (ev_rec_validate.validate, (os.path.join(nowhere, "old"),
                                                 os.path.join(nowhere, "new")))):
        try:
            if fn is ev_rec_report.run:
                fn(*args, log=lambda t: None)
            else:
                fn(*args)
        except NotImplementedError:
            real = False
        except Exception:  # noqa: BLE001  (an implementation's refusal of absent files)
            pass
    assert not os.path.exists(nowhere), "the probe wrote %s" % nowhere
    return real


def _sbi_dir():
    """(sbi_dir, which): the checkout with the real programs, or a copy of
    its hpc/pipeline with the stand-ins (built once per process)."""
    if "sbi" in _TMP:
        return _TMP["sbi"], _TMP["which"]
    if _real_programs():
        _TMP["sbi"], _TMP["which"] = _REPO, ("the real programs: ev_rec_sample.py, "
                                             "ev_rec_report.py, ev_rec_validate.py")
        return _TMP["sbi"], _TMP["which"]
    sbi = os.path.join(_tmp(), "sbi")
    hpc = os.path.join(sbi, "hpc")
    os.makedirs(hpc)
    src_hpc = os.path.join(_REPO, "hpc")
    for name in os.listdir(src_hpc):
        if name == "pipeline":
            shutil.copytree(os.path.join(src_hpc, name), os.path.join(hpc, name),
                            ignore=shutil.ignore_patterns("__pycache__"))
        else:
            os.symlink(os.path.join(src_hpc, name), os.path.join(hpc, name))
    ev = os.path.join(hpc, "pipeline", "evaluation")
    for name, text in (("ev_rec_report.py", STANDIN_REPORT),
                       ("ev_rec_validate.py", STANDIN_VALIDATE)):
        with open(os.path.join(ev, name), "w") as fh:
            fh.write(text)
    _TMP["sbi"], _TMP["which"] = sbi, ("block 35's real sampler, the stand-ins of ev_rec_report.py "
                                       "and ev_rec_validate.py (block 37's report is a stub here)")
    return _TMP["sbi"], _TMP["which"]


# ------------------------------------------------------------- checks ---

def test_r38a_dry_run():
    """the dry run on the committed synth.json and synth_rec_eval.json; the wrapper (specs/SPEC.md block 38)."""
    import ev_config
    import ev_rec_config
    import ev_rec_plan
    # synth_rec_eval.json loads, with the values block 38 states
    rec = ev_rec_config.load(SYNTH_REC_EVAL)
    old = ev_config.load(SYNTH_EVAL)
    for k in ("seed", "n_post", "batch_rows", "oracle_tol", "bootstrap_n", "level", "dpi",
              "formats", "ncpus"):
        assert getattr(rec, k) == getattr(old, k), (k, getattr(rec, k), getattr(old, k))
    assert rec.as_recorded is True and rec.sim_draw_rows_per_donor is None
    assert rec.validate_against == old.out_root, rec.validate_against
    assert rec.out_root == "/davinci-1/home/ldellamea/runs/synth/evaluation_rec", rec.out_root
    assert rec.out_root != old.out_root and not rec.out_root.startswith(old.out_root + "/")
    assert rec.edge_eps == 0.006693
    assert rec.gate == {"donor_fracs": (0.4, 0.25, 0.25, 0.1), "n_ref": 800, "n_rank": 1000,
                        "n_null": 1000, "n_null_min": 50, "rows_per_donor": 1,
                        "factors": (0.25, 0.5, 1, 2, 4), "alpha": 0.05}, rec.gate
    assert rec.jobs == {"queue": "intel",
                        "sample": {"ncpus": 8, "mem": "64gb", "walltime": "24:00:00"},
                        "report": {"ncpus": 8, "mem": "64gb", "walltime": "24:00:00"},
                        "validate": {"ncpus": 1, "mem": "8gb", "walltime": "01:00:00"}}, rec.jobs
    runs_root = _read(SYNTH)["paths"]["runs_root"]
    if os.path.isdir(runs_root):
        raise unittest.SkipTest("the project's runs root %s exists (davinci): its dry run is the "
                                "run sheet's" % runs_root)
    tmp = os.path.join(_tmp(), "r38a")
    os.makedirs(tmp)
    # (1) the committed files, paths.sbi_dir at this checkout: nothing is complete here
    here_cfg = _cfg_copy(SYNTH, os.path.join(tmp, "synth.json"), _REPO)
    e = dict(os.environ)
    e.pop("ROADMAP_DISABLE", None)
    rc, out = _direct("--config", here_cfg, "--eval-config", SYNTH_REC_EVAL, "--dry-run", skip="0",
                      env=e)
    assert rc == 0 and "REFUSE" not in out and "ABORT" not in out, out[-3000:]
    n_fits = len(_read(SYNTH)["synthetic"]["searches"]) + sum(
        1 for q in _read(SYNTH)["synthetic"]["searches"] if q["alt_best_fit"])
    assert out.count("REC EVAL SUBMIT: not complete: ") == n_fits == 5, out[-3000:]
    assert "view as_recorded" in out, out[-2000:]
    assert ("REC EVAL SUBMIT: report: nothing to report yet (no ensemble sampled or being sampled)"
            in out), out[-2000:]
    assert ("REC EVAL SUBMIT: validate: nothing to validate yet (the report is neither done, live "
            "nor submitted now)") in out, out[-2000:]
    assert ("REC EVAL DRY RUN: 0 job(s) would be submitted, 0 done, 0 live; nothing written"
            in out), out[-1000:]
    assert not os.path.exists(rec.out_root), "the dry run wrote %s" % rec.out_root
    e2 = dict(os.environ)
    e2["ROADMAP_DISABLE"] = "0"
    rc2, out2 = _submit("--config", here_cfg, "--eval-config", SYNTH_REC_EVAL, "--dry-run", env=e2)
    rc1, out1 = _direct("--config", here_cfg, "--eval-config", SYNTH_REC_EVAL, "--dry-run",
                        skip="1", env=e)
    assert rc2 == 0 and rc1 == 0 and _strip(out2) == _strip(out1), "the wrapper gave another plan"
    # (2) the committed values on the fixture's as-recorded project (complete best fits):
    # every unit's qsub line with the committed resources and its ROADMAP_STAGE
    F = rec_fixture.fixture()
    out_root = os.path.join(tmp, "out")
    cfg = _cfg_copy(F["cfg"], os.path.join(tmp, "fix_synth.json"), _REPO)
    ec = _ecfg_copy(SYNTH_REC_EVAL, os.path.join(tmp, "fix_rec_eval.json"), out_root=out_root,
                    validate_against=F["old_out"])
    rc, out = _direct("--config", cfg, "--eval-config", ec, "--dry-run", skip="0", env=e)
    assert rc == 0 and "REFUSE" not in out and "ABORT" not in out, out[-3000:]
    ecfg = ev_rec_config.load(ec)
    view = ev_rec_plan.load_view(cfg, ecfg)
    assert len(view.ensembles) == 3, view.ensembles
    ev = os.path.join(_REPO, "hpc", "pipeline", "evaluation")
    script = os.path.join(ev, "jobs", "eval_run.pbs")
    logs = os.path.join(out_root, "jobs", "logs")
    j = rec.jobs
    for i, en in enumerate(view.ensembles):
        safe = "sample_" + en.name.replace(":", "_")
        q = ("qsub -N er_s%d -q intel -l select=1:ncpus=%d:mem=%s -l walltime=%s -j oe -o %s -v "
             "ARGV_FILE=%s," % (i, j["sample"]["ncpus"], j["sample"]["mem"],
                                j["sample"]["walltime"], os.path.join(logs, safe + ".log"),
                                os.path.join(out_root, "jobs", safe + ".argv")))
        assert ("REC EVAL SUBMIT: sample:%s would be submitted:\n    %s" % (en.name, q)) in out, \
            (q, out[-3000:])
        cmd = "command: python %s --config %s --eval-config %s --ensemble %s --ncpus 8" % (
            os.path.join(ev, "ev_rec_sample.py"), cfg, ec, en.name)
        assert cmd in out, (cmd, out[-3000:])
    smp = ":".join("<sample_%s>" % en.name.replace(":", "_") for en in view.ensembles)
    q = ("qsub -N er_report -q intel -l select=1:ncpus=8:mem=64gb -l walltime=24:00:00 -j oe -o %s "
         "-W depend=afterok:%s -v " % (os.path.join(logs, "report.log"), smp))
    assert q in out, (q, out[-3000:])
    assert ("command: python %s --config %s --eval-config %s --ncpus 8"
            % (os.path.join(ev, "ev_rec_report.py"), cfg, ec)) in out, out[-3000:]
    q = ("qsub -N er_valid -q intel -l select=1:ncpus=1:mem=8gb -l walltime=01:00:00 -j oe -o %s "
         "-W depend=afterok:<report> -v " % os.path.join(logs, "validate.log"))
    assert q in out, (q, out[-3000:])
    assert ("command: python %s --old %s --new %s --record %s"
            % (os.path.join(ev, "ev_rec_validate.py"), F["old_out"], out_root,
               os.path.join(out_root, "VALIDATION.json"))) in out, out[-3000:]
    stages = [l.split("ROADMAP_STAGE=")[1].split()[0] for l in out.splitlines()
              if "ROADMAP_STAGE=" in l]
    assert stages == ["I35", "I35", "I35", "I37", "I36"], stages
    assert out.count("SKIP_CONDA=0,") == 5 and "ROADMAP_DISABLE" not in out, out[-2000:]
    assert out.count(script) == 5 and out.count("WORKDIR=%s," % _REPO) == 5, out[-2000:]
    assert out.count("ENV_NAME=%s," % _read(cfg)["envs"]["train"]) == 5, out[-2000:]
    assert ("REC EVAL DRY RUN: 5 job(s) would be submitted, 0 done, 0 live; nothing written"
            in out), out[-1000:]
    assert not os.path.exists(out_root), "the dry run wrote %s" % out_root
    rc2, out2 = _submit("--config", cfg, "--eval-config", ec, "--dry-run", env=e2)
    rc1, out1 = _direct("--config", cfg, "--eval-config", ec, "--dry-run", skip="1", env=e)
    assert rc2 == 0 and rc1 == 0 and _strip(out2) == _strip(out1), "the wrapper gave another plan"
    # the committed --project: its sbi_dir is davinci's, absent here
    rc5, out5 = _submit("--project", "synth", "--dry-run")
    assert rc5 == 3 and ("REFUSE: report's program /davinci-1/home/ldellamea/SBI/hpc/pipeline/"
                         "evaluation/ev_rec_report.py is not in this checkout (git pull)") in out5, out5
    rc3, out3 = _submit("--project", "synth", "--what", "sample,bogus", "--dry-run")
    assert rc3 == 2 and "ABORT: --what takes a subset of sample,report,validate" in out3, out3
    rc4, out4 = _submit("--project", "nosuch")
    assert rc4 == 2 and "ABORT: no configuration at" in out4, out4
    print("[r38a] the committed plan: 5 folders not complete, 0 jobs, nothing written; the committed "
          "values on the fixture: 3 samplings (I35), the report (I37) and the validation (I36) "
          "with the committed resources; the wrapper's plan the launcher's")


def test_r38b_as_recorded():
    """under the fake PBS, as recorded: samplings, report, validation; done, stale, live (specs/SPEC.md block 38)."""
    import ev_rec_config
    import ev_rec_plan
    F = rec_fixture.fixture()
    sbi, which = _sbi_dir()
    tmp = os.path.join(_tmp(), "r38b")
    os.makedirs(tmp)
    out_root = os.path.join(tmp, "out")
    cfg = _cfg_copy(F["cfg"], os.path.join(tmp, "synth.json"), sbi)
    ec = _ecfg_copy(F["ecfg_as"], os.path.join(tmp, "rec_eval_as.json"), out_root=out_root,
                    jobs=_LIGHT_JOBS)
    spool = os.path.join(tmp, "spool")
    env = _fake_pbs_env(spool)
    ecfg = ev_rec_config.load(ec)
    view = ev_rec_plan.load_view(cfg, ecfg)
    names = [en.name for en in view.ensembles]
    assert len(names) == 3 and ecfg.validate_against == F["old_out"], (names, ecfg)
    rc, out = _submit("--config", cfg, "--eval-config", ec, env=env)
    assert rc == 0 and "REC EVAL SUBMIT OK: 5 job(s) submitted, 0 done, 0 live" in out, out[-3000:]
    recs, dt = _wait(spool, 5)
    want = ["er_report", "er_s0", "er_s1", "er_s2", "er_valid"]
    assert sorted(recs) == want, sorted(recs)
    for n, r in recs.items():
        log = _log_of(r)
        assert r["exit_status"] == 0, (n, r.get("comment"), log[-3000:])
        assert log.rstrip().endswith("EVAL JOB EXIT: 0"), (n, log[-500:])
    smp = sorted(recs["er_s%d" % i]["id"] for i in range(3))
    for i, name in enumerate(names):
        r = recs["er_s%d" % i]
        assert r["depend"] is None and r["vars"]["ROADMAP_STAGE"] == "I35", (name, r)
        assert ("REC SAMPLE OK: %s" % view.ensembles[i].stem) in _log_of(r), _log_of(r)[-2000:]
    rep, val = recs["er_report"], recs["er_valid"]
    assert sorted((rep["depend"] or "").split(":")) == smp and rep["vars"]["ROADMAP_STAGE"] == "I37", rep
    assert val["depend"] == rep["id"] and val["vars"]["ROADMAP_STAGE"] == "I36", val
    assert "REC REPORT OK" in _log_of(rep), _log_of(rep)[-3000:]
    assert "REC VALIDATION OK: 3 ensemble(s)" in _log_of(val), _log_of(val)[-3000:]
    ledger = _read(os.path.join(out_root, "jobs", "ledger.json"))
    assert sorted(ledger) == sorted(["sample:" + n for n in names] + ["report", "validate"]), ledger
    rcj = _read(os.path.join(out_root, "REC_REPORT_COMPLETE.json"))
    assert rcj["inputs"] == ev_rec_plan.report_inputs(view, ecfg), rcj["inputs"]
    vrec_path = os.path.join(out_root, "VALIDATION.json")
    vrec = _read(vrec_path)
    assert vrec["ok"] is True and vrec["n_ensembles"] == 3 and vrec["old"] == F["old_out"], vrec
    assert vrec["metrics_index_sha256"] == _sha(os.path.join(out_root, "metrics", "METRICS.json"))
    # run again: every unit done, the validation's verdict printed
    rc, out = _submit("--config", cfg, "--eval-config", ec, env=env)
    assert rc == 0 and "REC EVAL SUBMIT OK: 0 job(s) submitted, 5 done, 0 live" in out, out[-3000:]
    assert "REC EVAL SUBMIT: validate done (validated: OK, 3 ensemble(s))" in out, out[-2000:]
    # n_post changed: the three samplings stale, the report after them, the validation
    # after the report (its record would be stale once that report ends)
    ec2 = _ecfg_copy(ec, os.path.join(tmp, "rec_eval_n20.json"), n_post=20)
    rc, out = _submit("--config", cfg, "--eval-config", ec2, "--dry-run", env=env)
    assert rc == 0 and out.count("(stale: n_post was 40, now 20)") == 3, out[-3000:]
    assert "report would be submitted (stale: the sampling of this run comes first)" in out, out
    assert ("depend=afterok:%s " % ":".join("<sample_%s>" % n.replace(":", "_") for n in names)
            in out), out[-3000:]
    assert "validate would be submitted (stale: the report of this run comes first)" in out, out
    assert "depend=afterok:<report> " in out, out[-3000:]
    assert "REC EVAL DRY RUN: 5 job(s) would be submitted, 0 done, 0 live" in out, out[-1000:]
    # a report value changed (dpi): the report stale and the validation after it, the
    # samplings done
    ec3 = _ecfg_copy(ec, os.path.join(tmp, "rec_eval_dpi.json"), dpi=61)
    rc, out = _submit("--config", cfg, "--eval-config", ec3, "--dry-run", env=env)
    assert rc == 0 and "REC EVAL DRY RUN: 2 job(s) would be submitted, 3 done, 0 live" in out, out
    assert "report would be submitted (stale: its samples or its values changed" in out, out[-1500:]
    line = [l for l in out.splitlines() if l.startswith("    qsub -N er_valid ")]
    assert len(line) == 1 and "depend=afterok:<report> " in line[0], line
    # --resample: that sampling, the report after it, the validation after the report
    rc, out = _submit("--config", cfg, "--eval-config", ec, "--resample", names[0], "--dry-run",
                      env=env)
    assert rc == 0 and "REC EVAL DRY RUN: 3 job(s) would be submitted, 2 done, 0 live" in out, out
    assert ("sample:%s would be submitted (stale: --resample)" % names[0]) in out, out[-2000:]
    assert "depend=afterok:<sample_%s> " % names[0].replace(":", "_") in out, out[-2000:]
    # the validation's record: stale when the metrics changed (submitted at once, the report
    # being done), its FAILED verdict printed when done
    keep = open(vrec_path, "rb").read()
    try:
        _write(vrec_path, dict(vrec, metrics_index_sha256="0" * 64))
        rc, out = _submit("--config", cfg, "--eval-config", ec, "--dry-run", env=env)
        assert rc == 0 and "REC EVAL DRY RUN: 1 job(s) would be submitted, 4 done, 0 live" in out, out
        line = [l for l in out.splitlines() if l.startswith("    qsub -N er_valid ")]
        assert len(line) == 1 and "depend" not in line[0], line
        assert "validate would be submitted (stale: the metrics changed since" in out, out[-1500:]
        _write(vrec_path, dict(vrec, ok=False, failures=["S-A1:A1: nll row 3: 1.0 vs 2.0"]))
        rc, out = _submit("--config", cfg, "--eval-config", ec, "--dry-run", env=env)
        assert rc == 0 and ("validate done (validated: FAILED, 1 failure(s), see %s)" % vrec_path
                            ) in out, out[-1500:]
    finally:
        with open(vrec_path, "wb") as fh:
            fh.write(keep)
    # a queued job is live and not submitted twice; the report goes after the live sampling.
    # The job waits SLOW_DELAY_S before it starts: the three launcher runs below take about 1 s
    # each here, so it is still queued during them on a machine ten times slower; once it is
    # deleted, its runner wakes after that delay, sees the deletion and ends, and the test waits
    # for that before removing the spool (the runner would otherwise create it again).
    slow = _fake_pbs_env(spool, delay=str(SLOW_DELAY_S))
    t_slow = time.time()
    rc, out = _submit("--config", cfg, "--eval-config", ec, "--what", "sample", "--resample",
                      names[0], env=slow)
    assert rc == 0 and "REC EVAL SUBMIT OK: 1 job(s) submitted, 2 done, 0 live" in out, out[-2000:]
    rc, out = _submit("--config", cfg, "--eval-config", ec, "--what", "sample", "--resample",
                      names[0], env=slow)
    assert rc == 0 and "REC EVAL SUBMIT OK: 0 job(s) submitted, 2 done, 1 live" in out, out[-2000:]
    led = _read(os.path.join(out_root, "jobs", "ledger.json"))
    live_id = led["sample:" + names[0]]["job_id"]
    assert ("sample:%s live: job %s" % (names[0], live_id)) in out, out[-2000:]
    rc, out = _submit("--config", cfg, "--eval-config", ec, "--dry-run", env=slow)
    assert rc == 0 and "REC EVAL DRY RUN: 2 job(s) would be submitted, 2 done, 1 live" in out, out
    assert ("report would be submitted (stale: the sampling of this run comes first)" in out
            and "depend=afterok:%s " % live_id in out), out[-2000:]
    assert time.time() - t_slow < SLOW_DELAY_S, "the queued job may have started: raise SLOW_DELAY_S"
    subprocess.run(["qdel", live_id], env=slow, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # refusals, nothing written
    before = _files_under(out_root)
    rc, out = _submit("--config", cfg, "--eval-config", ec, "--resample", "S-A9:A1", env=env)
    assert rc == 3 and "REFUSE: --resample names no complete ensemble: S-A9:A1" in out, out
    assert _files_under(out_root) == before
    time.sleep(max(0.0, t_slow + SLOW_DELAY_S + 3.0 - time.time()))
    print("[r38b] %s: 3 samplings, the report after them and the validation after the report "
          "through the fake PBS in %.0f s (REC VALIDATION OK: 3 ensemble(s)); done, stale, "
          "--resample, live as specified" % (which, dt))


def test_r38c_recorded():
    """under the fake PBS, the recorded kind: no validate unit; an incomplete best fit (specs/SPEC.md block 38)."""
    import ev_rec_config
    import ev_rec_plan
    F = rec_fixture.fixture()
    sbi, which = _sbi_dir()
    tmp = os.path.join(_tmp(), "r38c")
    os.makedirs(tmp)
    out_root = os.path.join(tmp, "out")
    cfg = _cfg_copy(F["cfg_rec"], os.path.join(tmp, "synth_rec.json"), sbi)
    ec = _ecfg_copy(F["ecfg_rec"], os.path.join(tmp, "rec_eval.json"), out_root=out_root,
                    jobs=_LIGHT_JOBS)
    spool = os.path.join(tmp, "spool")
    env = _fake_pbs_env(spool)
    ecfg = ev_rec_config.load(ec)
    view = ev_rec_plan.load_view(cfg, ecfg)
    assert view.kind == "recorded" and len(view.ensembles) == 2, view
    names = [en.name for en in view.ensembles]
    # the plan: one sampling per complete ensemble and the report, no validate unit
    rc, out = _submit("--config", cfg, "--eval-config", ec, "--dry-run", env=env)
    assert rc == 0 and "REC EVAL DRY RUN: 3 job(s) would be submitted, 0 done, 0 live" in out, out
    assert "validat" not in out and "er_valid" not in out, out[-2000:]
    for w in ("validate", "sample,report,validate"):
        before = _files_under(tmp)
        rc, out = _submit("--config", cfg, "--eval-config", ec, "--what", w, env=env)
        assert rc == 3 and "REFUSE: --what validate: the view of " in out and \
            "is recorded" in out, out[-1500:]
        assert _files_under(tmp) == before
    rc, out = _submit("--config", cfg, "--eval-config", ec, env=env)
    assert rc == 0 and "REC EVAL SUBMIT OK: 3 job(s) submitted, 0 done, 0 live" in out, out[-3000:]
    recs, dt = _wait(spool, 3)
    assert sorted(recs) == ["er_report", "er_s0", "er_s1"], sorted(recs)
    for n, r in recs.items():
        assert r["exit_status"] == 0, (n, r.get("comment"), _log_of(r)[-3000:])
    for i, en in enumerate(view.ensembles):
        r = recs["er_s%d" % i]
        assert r["vars"]["ROADMAP_STAGE"] == "I35" and r["depend"] is None, r
        assert ("REC SAMPLE OK: %s" % en.stem) in _log_of(r), _log_of(r)[-2000:]
    rep = recs["er_report"]
    assert sorted((rep["depend"] or "").split(":")) == sorted(recs["er_s%d" % i]["id"] for i in range(2))
    assert rep["vars"]["ROADMAP_STAGE"] == "I37" and "REC REPORT OK" in _log_of(rep), rep
    assert _read(os.path.join(out_root, "REC_REPORT_COMPLETE.json"))["inputs"] == \
        ev_rec_plan.report_inputs(view, ecfg)
    assert not os.path.exists(os.path.join(out_root, "VALIDATION.json"))
    rc, out = _submit("--config", cfg, "--eval-config", ec, env=env)
    assert rc == 0 and "REC EVAL SUBMIT OK: 0 job(s) submitted, 3 done, 0 live" in out, out[-2000:]
    # an ensemble whose BEST_FIT_COMPLETE.json is removed (a copy of its best fit, under a
    # runs root of this test's): named and not sampled
    runs = os.path.join(tmp, "runs_root")
    gone = view.ensembles[1]
    for name, v in __import__("stages.runs", fromlist=["search_views"]).search_views(view.cfg):
        src = v.data["paths"]["runs_root"]
        dst = os.path.join(runs, name)
        if name == gone.search:
            shutil.copytree(src, dst, symlinks=True,
                            ignore=lambda d, fs: [f for f in fs if f == "BEST_FIT_COMPLETE.json"])
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
    cfg2 = _cfg_copy(cfg, os.path.join(tmp, "synth_rec_gone.json"), sbi, runs_root=runs)
    rc, out = _submit("--config", cfg2, "--eval-config", ec, "--dry-run", env=env)
    folder = os.path.join(runs, gone.search, os.path.basename(gone.folder))
    assert rc == 0 and ("REC EVAL SUBMIT: not complete: %s (%s: no BEST_FIT_COMPLETE.json)"
                        % (gone.search, folder)) in out, out[-2000:]
    assert ("sample:%s" % gone.name) not in out, out[-2000:]
    assert ("REC EVAL SUBMIT: sample:%s done" % names[0]) in out, out[-2000:]
    assert "validat" not in out, out[-2000:]
    print("[r38c] %s: 2 samplings and the report after them through the fake PBS in %.0f s, no "
          "validate unit (--what validate refused); an ensemble without BEST_FIT_COMPLETE.json "
          "named and not sampled" % (which, dt))


def test_r38d_static():
    """ASCII and LF, bash -n, the launcher's imports (specs/SPEC.md block 38)."""
    import stat
    for f in (LAUNCHER, SUBMIT, SYNTH_REC_EVAL):
        b = open(f, "rb").read()
        assert b.count(b"\r") == 0 and all(c < 128 for c in b) and b"\t" not in b, f
    assert os.stat(SUBMIT).st_mode & stat.S_IXUSR, "%s is not executable" % SUBMIT
    r = subprocess.run(["bash", "-n", SUBMIT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    assert r.returncode == 0, r.stdout
    # the launcher's own imports: the standard library and the pipeline's modules; no f-string
    # or walrus (Python 3.6 syntax, the login node's python)
    tree = ast.parse(open(LAUNCHER).read())
    own = set()
    for d in (_PIPE, _EVAL, os.path.join(_PIPE, "stages")):
        own |= set(f[:-3] for f in os.listdir(d) if f.endswith(".py"))
    own |= {"stages"}
    std = set(getattr(sys, "stdlib_module_names", ()))
    if not std:
        raise unittest.SkipTest("sys.stdlib_module_names needs Python 3.10 or later")
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops |= set(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            tops.add((node.module or "").split(".")[0])
        assert not isinstance(node, ast.JoinedStr), "an f-string in the launcher"
        assert type(node).__name__ != "NamedExpr", "a walrus in the launcher"
    foreign = sorted(t for t in tops if t not in std and t not in own)
    assert not foreign, foreign
    # and everything it imports, transitively: a fresh interpreter that refuses any module
    # outside the standard library and the pipeline's tree (numpy, scipy, torch, ...)
    probe = (
        "import sys, os\n"
        "STD = set(sys.stdlib_module_names)\n"
        "ROOTS = (%r,)\n"
        "class Block(object):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.')[0]\n"
        "        if top in STD or top in sys.builtin_module_names:\n"
        "            return None\n"
        "        import importlib.machinery as M\n"
        "        spec = M.PathFinder.find_spec(name, path)\n"
        "        if spec is not None and spec.origin and any(\n"
        "                os.path.abspath(spec.origin).startswith(r + os.sep) for r in ROOTS):\n"
        "            return None\n"
        "        raise ImportError('not the standard library or the pipeline: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, %r)\n"
        "import ev_rec_submit\n"
        "print('IMPORTS OK', sorted(m for m in sys.modules if m.split('.')[0] in "
        "('numpy', 'scipy', 'torch')))\n" % (os.path.join(_REPO, "hpc"), _EVAL))
    r = subprocess.run([PYTHON, "-c", probe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=120)
    assert r.returncode == 0 and "IMPORTS OK []" in r.stdout, r.stdout[-3000:]
    print("[r38d] the launcher, the wrapper and synth_rec_eval.json ASCII and LF; bash -n; the "
          "launcher imports %d module(s) of the standard library and the pipeline only"
          % len(tops))


CHECKS = [("test_r38a_dry_run", test_r38a_dry_run),
          ("test_r38b_as_recorded", test_r38b_as_recorded),
          ("test_r38c_recorded", test_r38c_recorded),
          ("test_r38d_static", test_r38d_static)]


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
    finally:
        _cleanup_tmp()
        rec_fixture.cleanup()
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
