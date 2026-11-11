#!/usr/bin/env python3
"""Block 22: the evaluation's launcher on davinci's PBS -- the calibration
bank, one sampling job per complete ensemble, the report (specs/SPEC.md
block 22; D-126, D-127).

    bash hpc/pipeline/evaluation/submit_eval.sh --project synth [--dry-run] \\
        [--what calbank,sample,report] [--resample <q>:<arm>|all]

(the wrapper runs `python3 hpc/pipeline/evaluation/ev_submit.py --config
<synth.json> --eval-config <synth_eval.json> ...` in envs.extract). Each
unit is done, live (its recorded job queued, held or running), or submitted:

    calbank:<k>       shard k of the calibration bank (ev_plan.calbank_build):
                      done when shard_<k>.npz and its sidecar exist with the
                      build's arguments; a sidecar of other arguments, or a
                      shard index the build does not have, is refused and
                      never overwritten
    sample:<q>:<arm>  one complete ensemble (ev_plan.ensembles): done when its
                      <stem>_sample.json lists files that all exist, with the
                      sampling's current values (ev_config.SAMPLE_PARAMS) and
                      its best fit's current BEST_FIT_COMPLETE.json digest;
                      otherwise absent or stale; after the calibration bank's
                      jobs (afterok)
    report            ev_report.py over every sampled ensemble: submitted when a
                      sampling job is, or when REPORT_COMPLETE.json records
                      other inputs (ev_plan.report_inputs); after the sampling
                      jobs (afterok); not while no ensemble is sampled or
                      being sampled

An ensemble whose best fit is not complete is named; run this again once it
is. Every job is jobs/eval_run.pbs with its command in <out_root>/jobs/
<unit>.argv and its log in <out_root>/jobs/logs/, recorded in <out_root>/
jobs/ledger.json. --dry-run prints the plan and every qsub line and writes
nothing.

Exit 0 (EVAL SUBMIT OK / EVAL DRY RUN), 2 on the arguments or the
configurations, 3 on a refusal (nothing submitted), 4 when a qsub fails
(the jobs submitted before it are in the ledger).

Standard library only (the login node's python). Pure ASCII, LF only.
"""

import argparse
import datetime
import json
import os
import re
import sys

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

import ev_config
import ev_plan
import pbs as PBS
from stages import common as CM
from stages.synthetic import _normalise

EXIT_ARGS, EXIT_REFUSE, EXIT_QSUB = 2, 3, 4
KINDS = ("calbank", "sample", "report")
STAGE_OF = {"calbank": "I19", "sample": "I19", "report": "I22"}
REPORT_COMPLETE = "REPORT_COMPLETE.json"
_SHARD = re.compile(r"^shard_(\d+)\.(npz|json)$")


class Refuse(RuntimeError):
    """The launcher cannot go on; nothing is submitted (exit 3)."""


# ---------------------------------------------------------------- places

def code_paths(cfg):
    """The checkout the jobs run from (paths.sbi_dir) and its programs."""
    sbi = cfg.data["paths"]["sbi_dir"]
    ev = os.path.join(sbi, "hpc", "pipeline", "evaluation")
    return {"sbi_dir": sbi, "jobs_dir": os.path.join(sbi, "hpc", "pipeline", "jobs"),
            "script": os.path.join(ev, "jobs", "eval_run.pbs"),
            "builder": os.path.join(sbi, "hpc", "joint", "stage1", "build_latent_bank.py"),
            "sampler": os.path.join(ev, "ev_sample.py"),
            "report": os.path.join(ev, "ev_report.py")}


def jobs_dir(ecfg):
    return ecfg.sub("jobs")


def ledger_path(ecfg):
    return os.path.join(jobs_dir(ecfg), "ledger.json")


def read_ledger(ecfg):
    obj, _why = CM.try_read_json(ledger_path(ecfg))
    return obj or {}


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- the units

def calbank_units(cfg, ecfg, py):
    """[(unit, state, argv, why)] for the calibration bank's shards; Refuse
    on a folder holding a bank of another build."""
    build = ev_plan.calbank_build(cfg, ecfg)
    d = build["out_dir"]
    out, foreign = [], []
    for k, argv in enumerate(build["argv"]):
        npz = os.path.join(d, "shard_%04d.npz" % k)
        js = os.path.join(d, "shard_%04d.json" % k)
        state, why = "absent", ""
        if os.path.isfile(js):
            obj, problem = CM.try_read_json(js)
            got = _normalise(((obj or {}).get("provenance") or {}).get("argv") or [])
            if obj is None:
                foreign.append("shard %d: its sidecar cannot be read (%s)" % (k, problem))
            elif got != list(argv):
                foreign.append("shard %d: built with `%s`, the evaluation's build is `%s`"
                               % (k, " ".join(got), " ".join(argv)))
            elif os.path.isfile(npz):
                state, why = "done", npz
        out.append(("calbank:%d" % k, state,
                    [py, code_paths(cfg)["builder"], "--out-dir", d] + list(argv), why))
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            m = _SHARD.match(f)
            if m and int(m.group(1)) >= build["n_shards"]:
                foreign.append("%s: a shard index the build's %d shard(s) do not have"
                               % (f, build["n_shards"]))
    if foreign:
        raise Refuse("%s holds a calibration bank of another build (%s); nothing in it is "
                     "overwritten -- move the folder aside or change cal_glob"
                     % (d, "; ".join(foreign[:3])))
    return out


def sample_state(e, ecfg):
    """(state, why): done, absent or stale, for ensemble e's sampling."""
    pj = ev_plan.sample_json(ecfg, e.stem)
    if not os.path.isfile(pj):
        return "absent", pj
    obj, problem = CM.try_read_json(pj)
    if obj is None or obj.get("schema") != "eval_sample/1":
        return "stale", "%s cannot be read (%s)" % (pj, problem or "another schema")
    missing = [f for f in obj.get("files") or [] if not os.path.isfile(f)]
    if missing or not obj.get("files"):
        return "stale", "%d of its file(s) missing (%s)" % (len(missing), ", ".join(missing[:2]))
    now, then = ecfg.sample_params(), obj.get("sample_params") or {}
    diff = ["%s was %r, now %r" % (k, then.get(k), now[k]) for k in sorted(now)
            if then.get(k) != now[k]]
    if diff:
        return "stale", "; ".join(diff)
    bf = ev_plan.sha256_file(os.path.join(e.folder, "BEST_FIT_COMPLETE.json"))
    if obj.get("best_fit_complete_sha256") != bf:
        return "stale", "its best fit's BEST_FIT_COMPLETE.json changed since it was sampled"
    return "done", "n_post %s, %d member(s), %s" % (then.get("n_post"), len(obj.get("members") or []),
                                                   pj)


def report_state(cfg, ecfg):
    """(state, why): done, absent or stale, for the report."""
    rc = os.path.join(ecfg.out_root, REPORT_COMPLETE)
    if not os.path.isfile(rc):
        return "absent", rc
    obj, problem = CM.try_read_json(rc)
    if obj is None:
        return "stale", "%s cannot be read (%s)" % (rc, problem)
    if obj.get("inputs") != ev_plan.report_inputs(cfg, ecfg):
        return "stale", "its samples or its values changed since %s" % obj.get("finished", "?")
    return "done", rc


def ensemble_job_name(cfg, e):
    """ev_s<i><b|a>: search i of the configuration, its best fit or the
    other arm's (PBS's 15 characters cannot hold the stem)."""
    names = [q["name"] for q in cfg.data["synthetic"]["searches"]]
    return CM.job_name("ev_s%d%s" % (names.index(e.search["name"]),
                                     "a" if e.kind == "best_fit_alt" else "b"))


# ---------------------------------------------------------------- qsub

def qsub_line(cfg, ecfg, kind, name, log, argv_file, depend):
    j = ecfg.jobs[kind]
    p = code_paths(cfg)
    variables = ("ARGV_FILE=%s,ENV_NAME=%s,WORKDIR=%s,SKIP_CONDA=%s,JOBS_DIR=%s,WORK_ROOT=%s,"
                 "ROADMAP_STAGE=%s" % (argv_file, cfg.data["envs"]["train"], p["sbi_dir"],
                                       "1" if CM.E.skip_conda() else "0", p["jobs_dir"],
                                       cfg.data["paths"]["work_root"], STAGE_OF[kind]))
    if os.environ.get("ROADMAP_DISABLE", "") == "1":
        variables += ",ROADMAP_DISABLE=1"
    for v in (argv_file, log, p["sbi_dir"], cfg.data["paths"]["work_root"]):
        if "," in v:
            raise Refuse("a path with a comma cannot be passed through qsub -v: %r" % v)
    line = ["qsub", "-N", name, "-q", str(ecfg.jobs["queue"]),
            "-l", "select=1:ncpus=%d:mem=%s" % (int(j["ncpus"]), j["mem"]),
            "-l", "walltime=%s" % j["walltime"], "-j", "oe", "-o", log]
    if depend:
        line += ["-W", "depend=afterok:" + ":".join(depend)]
    return line + ["-v", variables, p["script"]]


# ---------------------------------------------------------------- the plan

def plan(cfg_path, ecfg_path, what=KINDS, resample=(), log=print):
    """(cfg, ecfg, units): every unit as dict(unit, kind, name, state, why,
    argv, live) -- nothing submitted, nothing written. Refuse when the
    launcher cannot go on."""
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    py = CM.E.python_in_env()
    p = code_paths(cfg)
    ledger = read_ledger(ecfg)
    units = []

    def live(unit):
        rec = ledger.get(unit)
        if not rec:
            return None
        try:
            st = PBS.status(rec["job_id"])
        except PBS.SchedulerError as exc:
            raise Refuse("the scheduler did not answer for %s (%s)" % (rec["job_id"], exc))
        return None if st.finished else "%s (%s)" % (rec["job_id"], st.state)

    if "calbank" in what:
        for unit, state, argv, why in calbank_units(cfg, ecfg, py):
            units.append({"unit": unit, "kind": "calbank", "state": state, "why": why,
                          "name": CM.job_name("ev_cal%s" % unit.split(":")[1]), "argv": argv,
                          "live": live(unit)})
    found, missing = ev_plan.ensembles(cfg, with_missing=True)
    if "sample" in what:
        names = set(e.name for e in found)
        unknown = sorted(set(resample) - names - {"all"})
        if unknown:
            raise Refuse("--resample names no complete ensemble: %s (complete: %s)"
                         % (", ".join(unknown), ", ".join(sorted(names)) or "none"))
        for e in found:
            state, why = sample_state(e, ecfg)
            if state == "done" and ("all" in resample or e.name in resample):
                state, why = "stale", "--resample"
            unit = "sample:%s" % e.name
            argv = [py, p["sampler"], "--config", cfg.path, "--eval-config", ecfg.path,
                    "--ensemble", e.name, "--ncpus", str(int(ecfg.jobs["sample"]["ncpus"]))]
            units.append({"unit": unit, "kind": "sample", "state": state, "why": why,
                          "name": ensemble_job_name(cfg, e), "argv": argv, "live": live(unit)})
        for q, folder, why in missing:
            log("EVAL SUBMIT: not complete: %s (%s: %s); run this again once it is"
                % (q, folder, why))
    if "report" in what:
        if not os.path.isfile(p["report"]):
            raise Refuse("the report's program %s is not in this checkout (git pull)" % p["report"])
        state, why = report_state(cfg, ecfg)
        argv = [py, p["report"], "--config", cfg.path, "--eval-config", ecfg.path,
                "--ncpus", str(int(ecfg.jobs["report"]["ncpus"]))]
        units.append({"unit": "report", "kind": "report", "state": state, "why": why,
                      "name": CM.job_name("ev_report"), "argv": argv, "live": live("report")})
    for u in units:
        if not os.path.isfile(u["argv"][1]):
            raise Refuse("%s's program %s is not in this checkout (git pull)"
                         % (u["unit"], u["argv"][1]))
    if not os.path.isfile(p["script"]):
        raise Refuse("the job script %s is not in this checkout (git pull)" % p["script"])
    return cfg, ecfg, units


def run(cfg_path, ecfg_path, what=KINDS, resample=(), dry_run=False, log=print):
    """Submits what the plan says; returns the exit code."""
    try:
        cfg, ecfg, units = plan(cfg_path, ecfg_path, what, resample, log)
    except ev_config.EvalConfigError as exc:
        log("ABORT: %s" % exc)
        return EXIT_ARGS
    except (Refuse, ev_plan.RefuseError) as exc:
        log("REFUSE: %s" % exc)
        return EXIT_REFUSE
    log("EVAL SUBMIT: %s; %s; out_root %s" % (cfg.path, ecfg.path, ecfg.out_root))
    jd = jobs_dir(ecfg)
    ledger = read_ledger(ecfg)
    ids = {"calbank": [], "sample": []}          # this run's ids, or placeholders
    live_ids = {"calbank": [], "sample": []}
    n_sub, n_done, n_live = 0, 0, 0
    sampling_now = False                         # a sampling job submitted now, or live
    has_samples = bool(ev_plan.report_inputs(cfg, ecfg)["samples"])
    for u in units:
        kind = u["kind"]
        if kind == "report" and not (has_samples or sampling_now or u["live"]):
            log("EVAL SUBMIT: report: nothing to report yet (no ensemble sampled or being sampled)")
            continue
        if u["live"]:
            n_live += 1
            note = ""
            if kind == "report" and ids["sample"]:
                note = ("; it does not include the sampling submitted now -- run this again "
                        "once it has ended")
            log("EVAL SUBMIT: %s live: job %s%s" % (u["unit"], u["live"], note))
            if kind in live_ids:
                live_ids[kind].append(u["live"].split(" ")[0])
            if kind == "sample":
                sampling_now = True
            continue
        if u["state"] == "done" and not (kind == "report" and sampling_now):
            n_done += 1
            log("EVAL SUBMIT: %s done (%s)" % (u["unit"], u["why"]))
            continue
        if u["state"] == "done":
            u = dict(u, state="stale", why="the sampling of this run comes first")
        depend = []
        if kind == "sample":
            depend = ids["calbank"] + live_ids["calbank"]
            sampling_now = True
        elif kind == "report":
            depend = ids["sample"] + live_ids["sample"]
        safe = u["unit"].replace(":", "_")
        argv_file = os.path.join(jd, "%s.argv" % safe)
        log_path = os.path.join(jd, "logs", "%s.log" % safe)
        line = qsub_line(cfg, ecfg, kind, u["name"], log_path, argv_file, depend)
        why = "" if u["state"] == "absent" else " (%s: %s)" % (u["state"], u["why"])
        if dry_run:
            n_sub += 1
            if kind in ids:
                ids[kind].append("<%s>" % u["unit"].replace(":", "_"))
            log("EVAL SUBMIT: %s would be submitted%s:\n    %s\n    command: %s"
                % (u["unit"], why, " ".join(line), " ".join(u["argv"])))
            continue
        os.makedirs(os.path.join(jd, "logs"), exist_ok=True)
        with open(argv_file + ".tmp", "w") as fh:
            fh.write("\n".join(u["argv"]) + "\n")
        os.replace(argv_file + ".tmp", argv_file)
        try:
            job = PBS.submit(line)
        except PBS.SubmitError as exc:
            log("EVAL SUBMIT FAILED: %s: %s" % (u["unit"], exc))
            return EXIT_QSUB
        ledger[u["unit"]] = {"job_id": job, "submitted": _now(), "argv_file": argv_file,
                             "log": log_path, "qsub": line}
        _write_json(ledger_path(ecfg), ledger)
        if kind in ids:
            ids[kind].append(job)
        n_sub += 1
        log("EVAL SUBMIT: %s submitted %s%s%s, log %s"
            % (u["unit"], job, why, (" after " + ", ".join(depend)) if depend else "", log_path))
    if dry_run:
        log("EVAL DRY RUN: %d job(s) would be submitted, %d done, %d live; nothing written"
            % (n_sub, n_done, n_live))
    else:
        log("EVAL SUBMIT OK: %d job(s) submitted, %d done, %d live" % (n_sub, n_done, n_live))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True, help="the evaluation's file (<tag>_eval.json)")
    ap.add_argument("--what", default=",".join(KINDS),
                    help="a comma-separated subset of %s (default: all)" % ",".join(KINDS))
    ap.add_argument("--resample", default="",
                    help="comma-separated ensembles <q>:<arm> to sample again although done, "
                         "or all")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; submit nothing")
    args = ap.parse_args(argv)
    what = tuple(w for w in args.what.split(",") if w)
    bad = sorted(set(what) - set(KINDS))
    if bad or not what:
        print("ABORT: --what takes a subset of %s, got %r" % (",".join(KINDS), args.what))
        return EXIT_ARGS
    resample = tuple(r for r in args.resample.split(",") if r)
    return run(os.path.abspath(args.config), os.path.abspath(args.eval_config), what, resample,
               args.dry_run, log=lambda t: print(t, flush=True))


if __name__ == "__main__":
    sys.exit(main())
