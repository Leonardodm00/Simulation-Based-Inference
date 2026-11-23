#!/usr/bin/env python3
"""Block 38: the launcher of a recorded project's evaluation on davinci's
PBS -- one sampling job per complete ensemble, the report after them, and
for a synthetic project read as recorded the validation after the report
(specs/SPEC.md sec. 2.6 and block 38; decisions D-159, D-162).

    bash hpc/pipeline/evaluation/submit_rec_eval.sh --project <tag> [--dry-run] \\
        [--what sample,report,validate] [--resample <name>,...|all]

(the wrapper runs `python3 hpc/pipeline/evaluation/ev_rec_submit.py
--config <tag>.json --eval-config <tag>_rec_eval.json ...` in
envs.extract). Each unit is done, live (its job in <out_root>/jobs/
ledger.json queued, held or running) or submitted; the rule is block 22's
(ev_submit.py), for these units:

    sample:<name>  one complete ensemble of the view (ev_rec_sample.py):
                   done when its <stem>_sample.json lists files that all
                   exist, with the current sample_params and its best fit's
                   current BEST_FIT_COMPLETE.json digest; otherwise absent
                   or stale (the reason printed), or --resample. An
                   ensemble whose best fit is not complete is named, not
                   sampled
    report         ev_rec_report.py, after the samplings: submitted when a
                   sampling is submitted now or live, or when
                   REC_REPORT_COMPLETE.json records other inputs than
                   ev_rec_plan.report_inputs; not while no ensemble is
                   sampled or being sampled
    validate       ev_rec_validate.py --old <validate_against> --new
                   <out_root> --record <out_root>/VALIDATION.json, after the
                   report; only for an as-recorded view with validate_against.
                   Done when that record names validate_against and the
                   sha256 of the current metrics/METRICS.json: its verdict
                   is printed and it is not submitted again (to repeat it,
                   delete the record)

Every job is jobs/eval_run.pbs (block 22's, unchanged) in envs.train, its
command in <out_root>/jobs/<unit>.argv, its log in <out_root>/jobs/logs/.
--dry-run prints the plan and every qsub line and writes nothing.

Exit 0 (REC EVAL SUBMIT OK / REC EVAL DRY RUN), 2 on the arguments or the
configurations, 3 on a refusal (nothing submitted), 4 when a qsub fails
(the jobs submitted before it are in the ledger).

CONTRACT (wave C): the constants and the signatures below are fixed by
specs/SPEC.md block 38; the bodies are the implementer's.

Standard library and the pipeline's own modules only (the login node's
python). Pure ASCII, LF only; Python 3.6 syntax.
"""

import argparse
import datetime
import os
import sys

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

import config as C
import ev_plan
import ev_rec_config
import ev_rec_plan
import pbs as PBS
from stages import common as CM

EXIT_ARGS, EXIT_REFUSE, EXIT_QSUB = 2, 3, 4
KINDS = ("sample", "report", "validate")
STAGE_OF = {"sample": "I35", "report": "I37", "validate": "I36"}   # specs/roadmap/tool.yaml
REPORT_COMPLETE = "REC_REPORT_COMPLETE.json"      # block 37's (ev_rec_report.REPORT_COMPLETE)
VALIDATION_RECORD = "VALIDATION.json"             # block 36's ev_rec_validate.py --record
JOB_NAMES = {"report": "er_report", "validate": "er_valid"}        # a sampling: er_s<i>

SAMPLE_SCHEMA = "rec_eval_sample/1"               # block 35's <stem>_sample.json
VALIDATION_SCHEMA = "rec_eval_validation/1"       # block 36's --record


class Refuse(RuntimeError):
    """The launcher cannot go on; nothing is submitted (exit 3)."""


# ---------------------------------------------------------------- places

def code_paths(cfg):
    """The checkout the jobs run from (paths.sbi_dir), its job script and
    its programs."""
    sbi = cfg.data["paths"]["sbi_dir"]
    ev = os.path.join(sbi, "hpc", "pipeline", "evaluation")
    return {"sbi_dir": sbi, "jobs_dir": os.path.join(sbi, "hpc", "pipeline", "jobs"),
            "script": os.path.join(ev, "jobs", "eval_run.pbs"),
            "sample": os.path.join(ev, "ev_rec_sample.py"),
            "report": os.path.join(ev, "ev_rec_report.py"),
            "validate": os.path.join(ev, "ev_rec_validate.py")}


def ledger_path(ecfg):
    return os.path.join(ecfg.sub("jobs"), "ledger.json")


def read_ledger(ecfg):
    obj, _why = CM.try_read_json(ledger_path(ecfg))
    return obj or {}


def validation_record(ecfg):
    return os.path.join(ecfg.out_root, VALIDATION_RECORD)


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- what

def default_what(view, ecfg):
    """The units of a run without --what: ("sample", "report"), and
    "validate" as well for an as-recorded view whose evaluation has
    `validate_against`."""
    if _no_validation(view, ecfg) is None:
        return ("sample", "report", "validate")
    return ("sample", "report")


def _no_validation(view, ecfg):
    """None when the view has a validation unit, else why it has none."""
    if view.kind != "as_recorded":
        return ("the view of %s is recorded: the validation compares a synthetic project read as "
                "recorded with its synthetic evaluation (D-162)" % view.cfg.path)
    if not ecfg.validate_against:
        return ("%s has validate_against null: there is no synthetic evaluation to compare with "
                "(D-162)" % ecfg.path)
    return None


# ---------------------------------------------------------------- the states

def sample_state(e, ecfg):
    """(state, why): done, absent or stale, for ensemble e's sampling."""
    pj = ev_rec_plan.sample_json(ecfg, e.stem)
    if not os.path.isfile(pj):
        return "absent", pj
    obj, problem = CM.try_read_json(pj)
    if obj is None or obj.get("schema") != SAMPLE_SCHEMA:
        return "stale", "%s cannot be read (%s)" % (pj, problem or "another schema")
    files = obj.get("files") or []
    missing = [f for f in files if not os.path.isfile(f)]
    if missing or not files:
        return "stale", "%d of its file(s) missing (%s)" % (len(missing), ", ".join(missing[:2]))
    now, then = ecfg.sample_params(), obj.get("sample_params") or {}
    diff = []
    for k in sorted(now):
        if k == "gate":
            g_now, g_then = now["gate"], then.get("gate") or {}
            diff += ["gate.%s was %r, now %r" % (g, g_then.get(g), g_now[g]) for g in sorted(g_now)
                     if g_then.get(g) != g_now[g]]
        elif then.get(k) != now[k]:
            diff.append("%s was %r, now %r" % (k, then.get(k), now[k]))
    if diff:
        return "stale", "; ".join(diff)
    bf = ev_plan.sha256_file(os.path.join(e.folder, "BEST_FIT_COMPLETE.json"))
    if obj.get("best_fit_complete_sha256") != bf:
        return "stale", "its best fit's BEST_FIT_COMPLETE.json changed since it was sampled"
    return "done", "n_post %s, %d member(s), %s" % (then.get("n_post"), len(obj.get("members") or []),
                                                   pj)


def report_state(view, ecfg):
    """(state, why): done, absent or stale, for the report."""
    rc = os.path.join(ecfg.out_root, REPORT_COMPLETE)
    if not os.path.isfile(rc):
        return "absent", rc
    obj, problem = CM.try_read_json(rc)
    if obj is None:
        return "stale", "%s cannot be read (%s)" % (rc, problem)
    if obj.get("inputs") != ev_rec_plan.report_inputs(view, ecfg):
        return "stale", "its samples or its values changed since %s" % obj.get("finished", "?")
    return "done", rc


def _sha256_or_none(path):
    return ev_plan.sha256_file(path) if os.path.isfile(path) else None


def validate_state(ecfg):
    """(state, why, verdict): done, absent or stale, for the validation; the
    verdict line of a done one (`validated: OK, <n> ensemble(s)` or
    `validated: FAILED, <k> failure(s), see <record>`), else None."""
    rec = validation_record(ecfg)
    if not os.path.isfile(rec):
        return "absent", rec, None
    obj, problem = CM.try_read_json(rec)
    if obj is None or obj.get("schema") != VALIDATION_SCHEMA:
        return "stale", "%s cannot be read (%s)" % (rec, problem or "another schema"), None
    if os.path.abspath(str(obj.get("old"))) != os.path.abspath(ecfg.validate_against):
        return "stale", ("it compared with %s, validate_against is now %s"
                         % (obj.get("old"), ecfg.validate_against)), None
    digest = _sha256_or_none(os.path.join(ecfg.sub("metrics"), "METRICS.json"))
    if digest is None or obj.get("metrics_index_sha256") != digest:
        return "stale", ("the metrics changed since it ran (%s): %s"
                         % (obj.get("finished", "?"),
                            "metrics/METRICS.json is absent" if digest is None else
                            "metrics/METRICS.json has another sha256")), None
    if obj.get("ok") is True:
        verdict = "validated: OK, %s ensemble(s)" % obj.get("n_ensembles")
    else:
        verdict = "validated: FAILED, %d failure(s), see %s" % (len(obj.get("failures") or []), rec)
    return "done", rec, verdict


# ---------------------------------------------------------------- qsub

def job_name_of(i):
    """er_s<i>: the i-th complete ensemble of the view, in the
    configuration's order (PBS's 15 characters cannot hold the stem)."""
    return CM.job_name("er_s%d" % i)


def qsub_line(view, ecfg, kind, name, log, argv_file, depend):
    """The qsub command line of one unit's job (eval_run.pbs)."""
    cfg = view.cfg
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

def plan(cfg_path, ecfg_path, what=None, resample=(), log=print):
    """(view, ecfg, units): every unit as dict(unit, kind, name, state, why,
    argv, live) -- nothing submitted, nothing written. `what` None is
    default_what. Refuse (or ev_plan.RefuseError) when the launcher cannot
    go on: an unknown --resample name, `validate` named for a view that
    has none, a program or the job script missing from the checkout, the
    scheduler not answering for a recorded job."""
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(cfg_path, ecfg)
    cfg = view.cfg
    if what is None:
        what = default_what(view, ecfg)
    what = tuple(what)
    bad = sorted(set(what) - set(KINDS))
    if bad or not what:
        raise Refuse("--what takes a subset of %s, got %r" % (",".join(KINDS), ",".join(what)))
    if "validate" in what:
        why = _no_validation(view, ecfg)
        if why is not None:
            raise Refuse("--what validate: %s" % why)
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

    if "sample" in what:
        names = set(e.name for e in view.ensembles)
        unknown = sorted(set(resample) - names - {"all"})
        if unknown:
            raise Refuse("--resample names no complete ensemble: %s (complete: %s)"
                         % (", ".join(unknown), ", ".join(sorted(names)) or "none"))
        for i, e in enumerate(view.ensembles):
            state, why = sample_state(e, ecfg)
            if state == "done" and ("all" in resample or e.name in resample):
                state, why = "stale", "--resample"
            unit = "sample:%s" % e.name
            argv = [py, p["sample"], "--config", cfg.path, "--eval-config", ecfg.path,
                    "--ensemble", e.name, "--ncpus", str(int(ecfg.jobs["sample"]["ncpus"]))]
            units.append({"unit": unit, "kind": "sample", "state": state, "why": why,
                          "name": job_name_of(i), "argv": argv, "live": live(unit)})
        for q, folder, why in view.missing:
            log("REC EVAL SUBMIT: not complete: %s (%s: %s); run this again once it is"
                % (q, folder, why))
    if "report" in what or "validate" in what:
        # the validation's order needs the report's state even when --what leaves it out
        state, why = report_state(view, ecfg)
        argv = [py, p["report"], "--config", cfg.path, "--eval-config", ecfg.path,
                "--ncpus", str(int(ecfg.jobs["report"]["ncpus"]))]
        report = {"unit": "report", "kind": "report", "state": state, "why": why,
                  "name": CM.job_name(JOB_NAMES["report"]), "argv": argv, "live": live("report")}
        if "report" in what:
            units.append(report)
    if "validate" in what:
        state, why, verdict = validate_state(ecfg)
        argv = [py, p["validate"], "--old", ecfg.validate_against, "--new", ecfg.out_root,
                "--record", validation_record(ecfg)]
        u = {"unit": "validate", "kind": "validate", "state": state, "why": why,
             "name": CM.job_name(JOB_NAMES["validate"]), "argv": argv, "live": live("validate"),
             "verdict": verdict}
        if "report" not in what:
            u["report"] = report                  # its state and liveness only; not submitted
        units.append(u)
    for u in units:
        if not os.path.isfile(u["argv"][1]):
            raise Refuse("%s's program %s is not in this checkout (git pull)"
                         % (u["unit"], u["argv"][1]))
    if not os.path.isfile(p["script"]):
        raise Refuse("the job script %s is not in this checkout (git pull)" % p["script"])
    return view, ecfg, units


# ---------------------------------------------------------------- the run

def run(cfg_path, ecfg_path, what=None, resample=(), dry_run=False, log=print):
    """Submits what the plan says; returns the exit code."""
    try:
        view, ecfg, units = plan(cfg_path, ecfg_path, what, resample, log)
    except (ev_rec_config.RecEvalConfigError, C.ConfigError, OSError) as exc:
        log("ABORT: %s" % exc)
        return EXIT_ARGS
    except (Refuse, ev_plan.RefuseError) as exc:
        log("REFUSE: %s" % exc)
        return EXIT_REFUSE
    log("REC EVAL SUBMIT: %s; %s; view %s; out_root %s"
        % (view.cfg.path, ecfg.path, view.kind, ecfg.out_root))
    jd = ecfg.sub("jobs")
    ledger = read_ledger(ecfg)
    ids = {"sample": [], "report": []}            # this run's ids, or placeholders
    live_ids = {"sample": [], "report": []}
    n_sub, n_done, n_live = 0, 0, 0
    sampling_now = False                          # a sampling job submitted now, or live
    report_done = False
    has_samples = any(os.path.isfile(ev_rec_plan.sample_json(ecfg, e.stem))
                      for e in view.ensembles)
    for u in units:
        kind = u["kind"]
        if kind == "validate" and "report" in u:
            # --what without report: the report as it stands, never submitted here
            r = u["report"]
            if r["live"]:
                live_ids["report"].append(r["live"].split(" ")[0])
            elif r["state"] == "done":
                report_done = True
        if kind == "report" and not (has_samples or sampling_now or u["live"]):
            log("REC EVAL SUBMIT: report: nothing to report yet (no ensemble sampled or being "
                "sampled)")
            continue
        if kind == "validate" and u["state"] != "done" and not u["live"] and not (
                report_done or ids["report"] or live_ids["report"]):
            log("REC EVAL SUBMIT: validate: nothing to validate yet (the report is neither done, "
                "live nor submitted now)")
            continue
        if u["live"]:
            n_live += 1
            note = ""
            if kind == "report" and ids["sample"]:
                note = ("; it does not include the sampling submitted now -- run this again "
                        "once it has ended")
            if kind == "validate" and ids["report"]:
                note = ("; it does not cover the report submitted now -- run this again "
                        "once it has ended")
            log("REC EVAL SUBMIT: %s live: job %s%s" % (u["unit"], u["live"], note))
            if kind in live_ids:
                live_ids[kind].append(u["live"].split(" ")[0])
            if kind == "sample":
                sampling_now = True
            continue
        # what this run redoes upstream comes first: the report after a sampling
        # submitted now or live, the validation after a report submitted now or live
        # (that report rewrites the metrics the validation's record names)
        upstream = ((kind == "report" and sampling_now)
                    or (kind == "validate" and bool(ids["report"] or live_ids["report"])))
        if u["state"] == "done" and not upstream:
            n_done += 1
            if kind == "report":
                report_done = True
            if kind == "validate":
                log("REC EVAL SUBMIT: validate done (%s)" % u["verdict"])
            else:
                log("REC EVAL SUBMIT: %s done (%s)" % (u["unit"], u["why"]))
            continue
        if u["state"] == "done":
            u = dict(u, state="stale", why=("the sampling of this run comes first"
                                            if kind == "report" else
                                            "the report of this run comes first"))
        depend = []
        if kind == "sample":
            sampling_now = True
        elif kind == "report":
            depend = ids["sample"] + live_ids["sample"]
        elif kind == "validate":
            depend = ids["report"] + live_ids["report"]
        safe = u["unit"].replace(":", "_")
        argv_file = os.path.join(jd, "%s.argv" % safe)
        log_path = os.path.join(jd, "logs", "%s.log" % safe)
        line = qsub_line(view, ecfg, kind, u["name"], log_path, argv_file, depend)
        why = "" if u["state"] == "absent" else " (%s: %s)" % (u["state"], u["why"])
        if dry_run:
            n_sub += 1
            if kind in ids:
                ids[kind].append("<%s>" % safe)
            log("REC EVAL SUBMIT: %s would be submitted%s:\n    %s\n    command: %s"
                % (u["unit"], why, " ".join(line), " ".join(u["argv"])))
            continue
        os.makedirs(os.path.join(jd, "logs"), exist_ok=True)
        with open(argv_file + ".tmp", "w") as fh:
            fh.write("\n".join(u["argv"]) + "\n")
        os.replace(argv_file + ".tmp", argv_file)
        try:
            job = PBS.submit(line)
        except PBS.SubmitError as exc:
            log("REC EVAL SUBMIT FAILED: %s: %s" % (u["unit"], exc))
            return EXIT_QSUB
        ledger[u["unit"]] = {"job_id": job, "submitted": _now(), "argv_file": argv_file,
                             "log": log_path, "qsub": line}
        CM.write_json(ledger_path(ecfg), ledger)
        if kind in ids:
            ids[kind].append(job)
        n_sub += 1
        log("REC EVAL SUBMIT: %s submitted %s%s%s, log %s"
            % (u["unit"], job, why, (" after " + ", ".join(depend)) if depend else "", log_path))
    if dry_run:
        log("REC EVAL DRY RUN: %d job(s) would be submitted, %d done, %d live; nothing written"
            % (n_sub, n_done, n_live))
    else:
        log("REC EVAL SUBMIT OK: %d job(s) submitted, %d done, %d live" % (n_sub, n_done, n_live))
    return 0


def main(argv=None):
    """The command line; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's file (<tag>_rec_eval.json)")
    ap.add_argument("--what", default=None,
                    help="a comma-separated subset of %s (default: sample,report, and validate "
                         "for an as-recorded view with validate_against)" % ",".join(KINDS))
    ap.add_argument("--resample", default="",
                    help="comma-separated ensembles to sample again although done, or all")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; submit nothing")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    what = None
    if args.what is not None:
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
