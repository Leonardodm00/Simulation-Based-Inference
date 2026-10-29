#!/usr/bin/env python3
"""A read-only report of the search stage's trials (SPEC block 7, an
unreadable record, amended 2026-10-19): what the state file, the ledger,
the pending specs, the run folders and the job logs say of each trial, and
what the stage's next poll does with it by the stage's own rules.

    python3 hpc/pipeline/tools/search_report.py --config <project>.json
        [--qstat] [--trial <id prefix>]

    --qstat    ask the scheduler (one `qstat -x -f` of every trial's last
               job, a query) for its state, node, exit, walltime and
               memory; a job the state file does not know to have ended is
               judged by it; without it a job counts as ended when the
               state says so or its log has its `RUN_ARGV EXIT:` line
    --trial    one trial in full: its facts, its job logs, the last 60
               lines of the latest

Per trial, one line: the state's status, round, jobs, resubmissions; the
ledger record (status, nll, error) or why it cannot be read; the records
set aside; the pending spec; the run folder's checkpoint and runner record;
the last job log (host, `RUN_ARGV EXIT`, its last error line); the failure's
cause (a heuristic label; the stage's own rule is `common.node_fault`); and
`next:` -- what the next poll does, by `_finish_trial_job`, `_examine` and
`_sweep_failed`. Then why the failed records failed -- per cause, one
trial's traceback tail -- and the knobs of the failed records against the
ok ones and the trials still open; the counts, the failures grouped by
cause, the unreadable records, and the last line

    SEARCH REPORT OK: <n> trial(s) read; nothing was written

Exit 0 with that line; 2 when the configuration does not load or holds no
search. Writes nothing: every file is opened for reading; the scheduler is
only queried.

Stdlib only. Pure ASCII, LF only; Python 3.6 syntax.
"""

import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

import config as C                      # noqa: E402
import pbs as PBS                       # noqa: E402
import state as ST                      # noqa: E402
import stages.common as CM              # noqa: E402
from stages.search import SearchStage   # noqa: E402

STATUS_ORDER = ("examining", "running", "waiting", "failed", "killed", "ok")

_RUN_ARGV_EXIT = re.compile(r"^RUN_ARGV EXIT: ([0-9]+)\s*$", re.M)
_ERRORISH = re.compile(r"(Error|Exception|Traceback|Killed|killed|exceeded|Bus error|"
                       r"Segmentation|FATAL|REFUSE|=>> PBS|out of memory|NaN)")
_EXC_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit))"
                       r"(?::\s*(.*))?$", re.M)
_RUNNER_EXITED = re.compile(r"^runner exited (-?[0-9]+)$")


# ---------------------------------------------------------------------------
# Loading: facts only
# ---------------------------------------------------------------------------

def load_search(cfg):
    """(stage, state or None, the search stage's record, {job id: job record})."""
    stage = SearchStage()
    wr = cfg.data["paths"]["work_root"]
    state = ST.State.load(wr) if os.path.isdir(wr) else None
    rec = (state.data.get("stages", {}).get("search") if state is not None else None) or {}
    jobs = dict((j.get("id"), j) for j in rec.get("jobs", []))
    return stage, state, rec, jobs


def ledger_only_trials(stage, cfg):
    """The trials the ledger and the pending folder name (no state file)."""
    dirs = stage._dirs(cfg)
    out = {}
    for p in glob.glob(os.path.join(glob.escape(dirs["ledger"]), "*.json")):
        out[os.path.basename(p)[:-5]] = {"status": "?", "jobs": [], "resubmits": 0}
    for p in glob.glob(os.path.join(glob.escape(dirs["pending"]), "*.json")):
        out.setdefault(os.path.basename(p)[:-5], {"status": "?", "jobs": [], "resubmits": 0})
    return out


def last_error_line(text):
    """The last line of a job log that names an error, or None."""
    for line in reversed((text or "").splitlines()):
        s = line.strip()
        if not s or s.startswith("RUN_ARGV EXIT"):
            continue
        if _ERRORISH.search(s):
            return s[:200]
    return None


def traceback_tail(text, n=12):
    """The last n lines of a log's last traceback (its deepest frames and the
    exception), without the tuner's and the job script's own lines; the
    last 8 lines of the log when it has no traceback."""
    lines = (text or "").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")]
    keep = [ln for ln in (lines[starts[-1]:] if starts else lines)
            if ln.strip() and not ln.startswith(("[evaluate]", "RUN_ARGV"))]
    return keep[-n:] if starts else keep[-8:]


def trial_facts(cfg, stage, tid, tr, jobs, qstat=None):
    """Everything on disk (and, with qstat -- {job id: its record} -- in the
    scheduler) about one trial."""
    dirs = stage._dirs(cfg)
    f = {"tid": tid, "status": str(tr.get("status", "?")), "round": tr.get("round"),
         "jobs": list(tr.get("jobs", [])), "resubmits": int(tr.get("resubmits", 0) or 0),
         "fault_checked": bool(tr.get("fault_checked")),
         "resume_next": bool(tr.get("resume_next")),
         "unreadable_since": tr.get("record_unreadable_since"),
         "node_faults_before": len(tr.get("node_faults", [])),
         "unreadable_before": len(tr.get("unreadable_records", []))}
    last = f["jobs"][-1] if f["jobs"] else None
    j = jobs.get(last) if last else None
    f["last_job"] = last
    f["job_finished"] = bool(j and j.get("finished"))
    f["job_exit"] = j.get("exit_status") if j else None
    f["job_walltime_s"] = j.get("walltime_s") if j else None
    f["q"] = None
    if qstat is not None and last:
        f["q"] = qstat.get(last) or qstat.get(last.split(".")[0])
    # the scheduler's word counts only for a job the state does not know to have ended
    f["qstat"] = None
    if qstat is not None and last and not f["job_finished"]:
        f["qstat"] = ((f["q"]["state"], f["q"]["exit"]) if f["q"] is not None
                      else ("?", "not in qstat's answer"))
    rp = stage._record_path(cfg, tid)
    f["record_path"] = rp
    f["record_exists"] = os.path.isfile(rp)
    f["record"], f["record_problem"] = (CM.try_read_json(rp) if f["record_exists"]
                                        else (None, None))
    f["aside"] = sorted(os.path.basename(p) for p in glob.glob(glob.escape(rp) + ".*"))
    f["pending"] = os.path.isfile(os.path.join(dirs["pending"], tid + ".json"))
    run = os.path.join(dirs["runs"], tid)
    f["ckpt"] = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(glob.escape(run), "*_train_ckpt.pt")))
    f["runner_records"] = sorted(os.path.basename(p) for p in
                                 glob.glob(os.path.join(glob.escape(run), "*_seed*.json")))
    f["log"], text = stage._latest_log(cfg, tid)
    f["host"] = CM.host_of_log(text)
    m = _RUN_ARGV_EXIT.findall(text or "")
    f["log_exit"] = int(m[-1]) if m else None
    f["last_error"] = last_error_line(text)
    f["tail"] = traceback_tail(text)
    f["node_fault"] = CM.node_fault((f["record"] or {}).get("error"), text)
    # D-111: a job that ended without a record and could not start on its node
    f["env_fault"] = CM.env_fault({"exit_status": f["job_exit"]}, text) \
        if f["job_finished"] else None
    f["env_faults_before"] = len(tr.get("env_faults", []))
    f["cause"] = failure_cause((f["record"] or {}).get("error"), text, f["node_fault"])
    return f


def _gb(value):
    """A PBS size (`78400000kb`) in GB, as text; the value itself otherwise."""
    m = re.match(r"^([0-9]+)(b|kb|mb|gb|tb)?$", str(value or "").strip().lower())
    if not m:
        return value or "-"
    mult = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3,
            "tb": 1024 ** 4}[m.group(2) or "b"]
    return "%.1fGB" % (int(m.group(1)) * mult / 1024.0 ** 3)


def qstat_many(job_ids, timeout=300):
    """({job id: record}, problem): one `qstat -x -f` of the jobs (a query);
    a record holds the state letter, exit, node(s), walltime, cput, memory
    used and asked, and the scheduler's comment. A job the scheduler no
    longer knows is absent."""
    ids = sorted(set(j for j in job_ids if j))
    if not ids:
        return {}, None
    try:
        proc = subprocess.run([PBS.QSTAT, "-x", "-f"] + ids, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, "qstat could not be asked: %s" % exc
    blocks, cur = [], None
    for line in proc.stdout.splitlines():
        if line.startswith("Job Id:"):
            cur = [line]
            blocks.append(cur)
        elif cur is not None:
            cur.append(line)
    out = {}
    for b in blocks:
        jid = b[0].split(":", 1)[1].strip()
        text = "\n".join(b)
        js = PBS.parse_qstat(jid, text)
        fields = {}
        for line in text.replace("\n\t", "").splitlines():
            if " = " in line:
                k, v = line.split(" = ", 1)
                fields[k.strip()] = v.strip()
        rec = {"state": js.state, "exit": js.exit_status,
               "host": ",".join(sorted(set(re.findall(r"([^/+]+)/", fields.get("exec_host", "")))))
               or "-", "walltime": fields.get("resources_used.walltime", "-"),
               "cput": fields.get("resources_used.cput", "-"),
               "mem": _gb(fields.get("resources_used.mem")),
               "mem_asked": fields.get("Resource_List.mem", "-"),
               "comment": fields.get("comment", "")}
        out[jid] = rec
        out[jid.split(".")[0]] = rec
    problem = proc.stderr.strip().replace("\n", " | ")[:300] or None
    return out, problem


# ---------------------------------------------------------------------------
# The stage's rules (analysis)
# ---------------------------------------------------------------------------

def failure_cause(record_error, log_text, why_node):
    """A label for why a trial failed (a heuristic for the report; the stage
    decides by node_fault alone)."""
    if why_node:
        return "node fault: " + why_node[:90]
    m = _RUNNER_EXITED.match(str(record_error or "").strip())
    if m and int(m.group(1)) == -9:
        return "SIGKILL: memory (the OOM killer) or an operator"
    if m and int(m.group(1)) < 0:
        return "signal %d" % -int(m.group(1))
    text = log_text or ""
    if re.search(r"walltime[^\n]*exceeded|exceeded[^\n]*walltime", text):
        return "walltime exceeded"
    exc = _EXC_LINE.findall(text)
    if exc:
        name, msg = exc[-1]
        return ("%s: %s" % (name, msg.strip()))[:110] if msg else name
    if record_error:
        return str(record_error)[:110]
    return "no error text"


def cause_key(cause):
    """The cause with its numbers blanked, to group like failures."""
    return re.sub(r"[0-9]+", "#", cause)[:90]


def job_ended(f):
    """True / False / None (unknown) -- by the scheduler when asked, else
    the state file, else the log's last line."""
    if f["qstat"] is not None and f["qstat"][0] != "?":
        # as the stage reads it: a job the scheduler no longer knows has ended
        return f["qstat"][0] in PBS.FINISHED_STATES or f["qstat"][0] == "unknown"
    if f["job_finished"] or f["log_exit"] is not None:
        return True
    return None


def _after_no_record(cfg, f, at_job_end):
    """What follows once a trial has no usable record (search.py)."""
    s = cfg.data["search"]
    mx = int(s["trial"]["max_resubmits"])
    r = f["resubmits"]
    if not f["pending"]:
        return "recorded killed (no pending spec to resubmit from)"
    if at_job_end and not s["resubmit_killed"]:
        return "recorded killed (search.resubmit_killed false)"
    if r < mx:
        return "%s with --resume (%d of %d) when the cores allow" % (
            "resubmitted" if at_job_end else "queued again", r + 1, mx)
    return "recorded killed (no resubmission left after %d)" % r


def next_action(cfg, f):
    """What the stage's next poll does with the trial."""
    s = cfg.data["search"]
    mx = int(s["trial"]["max_resubmits"])
    st = f["status"]
    if st in ("ok", "killed"):
        return "final (%s)" % st
    if st == "failed" and (f["fault_checked"] or not s["resubmit_node_faults"]):
        return "final (failed)"
    if st == "waiting":
        return "waits for a slot and the cores%s" % (" (then --resume)" if f["resume_next"] else "")
    if st == "?":
        return "unknown to the state file"
    at_job_end = (st == "running")
    if at_job_end:
        ended = job_ended(f)
        if ended is False:
            return "nothing until its job %s ends (the scheduler: %s)" % (f["last_job"], f["qstat"][0])
        if ended is None:
            return ("nothing until its job %s ends (not ended by the state file, no RUN_ARGV EXIT "
                    "in its log: running, or killed before its last line; --qstat asks)"
                    % f["last_job"])
    if f["record_exists"] and f["record"] is None:
        return ("its record cannot be read: read again for %d s, then set aside as .unreadable%d; "
                "then %s" % (int(CM.RECORD_GRACE_S), f["unreadable_before"] + 1,
                             _after_no_record(cfg, f, at_job_end)))
    rec = f["record"]
    if rec is not None:
        rs = str(rec.get("status", "ok"))
        if rs == "failed" and s["resubmit_node_faults"]:
            if f["node_fault"]:
                if not f["pending"] and not rec.get("config"):
                    return "NODE FAULT, but no configuration and no pending spec: final"
                if f["resubmits"] < mx:
                    return ("NODE FAULT: record set aside, queued again with --resume (%d of %d)"
                            % (f["resubmits"] + 1, mx))
                return "NODE FAULT: record set aside, no resubmission left: recorded killed"
            if f["log_exit"] is None:
                return ("examined: waits up to %d s for its log's RUN_ARGV EXIT line, then final "
                        "unless that log shows a node fault" % int(CM.LOG_GRACE_S))
            return "final failure: not a node fault"
        if rs == "failed":
            return "final failure (search.resubmit_node_faults false)"
        nll = rec.get("nll")
        return "marked %s%s" % (rs, (" nll %.6f" % float(nll)) if nll is not None else "")
    if at_job_end and CM.held_off_node({"exit_status": f.get("job_exit")}):
        return ("held off an avoided node (exit %d): submitted again at once, not a resubmission"
                % CM.AVOID_EXIT)
    if at_job_end and f.get("env_fault"):
        return ("could not start (an environment fault: %s): submitted again at once, not a "
                "resubmission (D-111)" % f["env_fault"])
    if at_job_end:
        return "ended without a record: " + _after_no_record(cfg, f, True)
    return ("no ledger record any more: %s" % ("queued again" if f["pending"]
                                                 else "left failed (no pending spec)"))


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _record_text(f):
    if not f["record_exists"]:
        return "none"
    if f["record"] is None:
        return "UNREADABLE (%s)" % f["record_problem"]
    r = f["record"]
    out = str(r.get("status", "ok"))
    if r.get("nll") is not None:
        out += " nll %.6f" % float(r["nll"])
    if r.get("error"):
        out += " error %r" % str(r["error"])[:100]
    return out


def trial_line(cfg, f):
    mx = int(cfg.data["search"]["trial"]["max_resubmits"])
    job = "no job"
    if f["last_job"]:
        job = "last %s %s" % (f["last_job"], "ended" if f["job_finished"] else "not ended per state")
        if f["job_exit"] is not None:
            job += " exit %s" % f["job_exit"]
        if f["job_walltime_s"]:
            job += " %.1f h" % (float(f["job_walltime_s"]) / 3600.0)
        if f["q"] is not None:
            job += " qstat %s on %s exit %s, %s, mem %s of %s" % (
                f["q"]["state"], f["q"]["host"], f["q"]["exit"], f["q"]["walltime"],
                f["q"]["mem"], f["q"]["mem_asked"])
        elif f["qstat"] is not None:
            job += " qstat %s" % (f["qstat"][0],)
    parts = ["trial %s r%s %-9s jobs %d (%s) resub %d/%d"
             % (f["tid"][:12], f["round"], f["status"], len(f["jobs"]), job, f["resubmits"], mx),
             "record " + _record_text(f)]
    if f["aside"]:
        parts.append("aside " + ",".join(a[len(f["tid"]) + len(".json."):] for a in f["aside"]))
    parts.append("pending %s" % ("yes" if f["pending"] else "no"))
    parts.append("ckpt %s, runner record %s" % ("yes" if f["ckpt"] else "no",
                                               "yes" if f["runner_records"] else "no"))
    if f["log"]:
        parts.append("log %s exit %s%s" % (f["host"] or "?", f["log_exit"] if f["log_exit"] is not None
                                           else "-", (": " + f["last_error"]) if f["last_error"]
                                           else ""))
    else:
        parts.append("no job log")
    if f["status"] in ("failed", "examining", "killed") or (f["record"] or {}).get("status") \
            in ("failed",):
        parts.append("cause " + f["cause"])
    parts.append("next: " + next_action(cfg, f))
    return " | ".join(parts)


def summary_lines(cfg, facts, state):
    out = []
    by_status = {}
    for f in facts:
        by_status[f["status"]] = by_status.get(f["status"], 0) + 1
    rounds = sorted(set(f["round"] for f in facts if f["round"] is not None))
    when = ""
    if state is not None:
        when = " (state updated %s by %s on %s)" % (state.data.get("updated_utc"),
                                                    state.data.get("job") or state.data.get("pid"),
                                                    state.data.get("host"))
    out.append("trials: %d in %d round(s); by the state%s: %s" % (
        len(facts), len(rounds), when,
        ", ".join("%s %d" % (k, by_status[k]) for k in sorted(by_status, key=_order_key))))
    readable = [f for f in facts if f["record"] is not None]
    unreadable = [f for f in facts if f["record_exists"] and f["record"] is None]
    rs = {}
    for f in readable:
        k = str(f["record"].get("status", "ok"))
        rs[k] = rs.get(k, 0) + 1
    out.append("ledger records: %d readable (%s), %d UNREADABLE%s; set aside earlier: %d"
               % (len(readable), ", ".join("%s %d" % kv for kv in sorted(rs.items())) or "-",
                  len(unreadable), (": " + ", ".join("%s (%s)" % (f["tid"][:12], f["record_problem"])
                                                     for f in unreadable)) if unreadable else "",
                  sum(len(f["aside"]) for f in facts)))
    failed = [f for f in facts if (f["record"] or {}).get("status") == "failed"]
    groups = {}
    for f in failed:
        groups.setdefault(cause_key(f["cause"]), []).append(f)
    out.append("failed records by cause (a heuristic label; the stage heals only node faults): %d"
               % len(failed))
    for key, fs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        hosts = sorted(set(f["host"] or "?" for f in fs))
        out.append("  %2d x %s -- hosts %s -- %s" % (len(fs), fs[0]["cause"], ",".join(hosts),
                                                     " ".join(f["tid"][:12] for f in fs)))
    acts = {}
    for f in facts:
        a = next_action(cfg, f)
        key = re.sub(r"\(.*$", "", re.sub(r"[0-9]+", "#", a)).strip()
        acts[key] = acts.get(key, 0) + 1
    out.append("the next poll, by the stage's rules:")
    for k, n in sorted(acts.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append("  %2d x %s" % (n, k))
    return out


def _fmt(v):
    return ("%.4g" % v) if isinstance(v, float) else str(v)


def _summary(values):
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if values and len(nums) == len(values):
        s = sorted(nums)
        return "min %s median %s max %s (%d)" % (_fmt(s[0]), _fmt(statistics.median(s)),
                                                _fmt(s[-1]), len(s))
    counts = {}
    for v in values:
        counts[_fmt(v)] = counts.get(_fmt(v), 0) + 1
    return ", ".join("%s x%d" % kv for kv in sorted(counts.items())) or "-"


def why_lines(cfg, stage, facts):
    """Why the failed records failed: per cause, one trial's traceback tail;
    then the knobs of the failed records against the ok ones and the open
    trials (their pending specs)."""
    failed = [f for f in facts if (f["record"] or {}).get("status") == "failed"]
    if not failed:
        return []
    out = ["why the failed records failed -- per cause, one trial's log (--trial <id> shows any "
           "one in full):"]
    groups = {}
    for f in failed:
        groups.setdefault(cause_key(f["cause"]), []).append(f)
    for _key, fs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        f = fs[0]
        q = f["q"] or {}
        out.append("  -- %d x %s; e.g. trial %s on %s (record %r; job exit %s, %s, mem %s)"
                   % (len(fs), f["cause"], f["tid"][:12], f["host"] or q.get("host") or "?",
                      (f["record"] or {}).get("error"), q.get("exit", f["job_exit"]),
                      q.get("walltime", "-"), q.get("mem", "-")))
        for line in f["tail"]:
            out.append("  | " + line[:220])
    dirs = stage._dirs(cfg)
    cfgs = {"failed": [], "ok": [], "open": []}
    for f in facts:
        rec = f["record"]
        if rec is not None and isinstance(rec.get("config"), dict):
            k = str(rec.get("status", "ok"))
            if k in ("failed", "ok"):
                cfgs[k].append(rec["config"])
            continue
        if rec is None and f["pending"]:
            spec, _ = CM.try_read_json(os.path.join(dirs["pending"], f["tid"] + ".json"))
            if spec is not None and isinstance(spec.get("config"), dict):
                cfgs["open"].append(spec["config"])
    out.append("the knobs: %d failed record(s) against %d ok and %d open (pending specs)"
               % (len(cfgs["failed"]), len(cfgs["ok"]), len(cfgs["open"])))
    keys = sorted(set(k for g in cfgs.values() for c in g for k in c))
    same = []
    for k in keys:
        vals = dict((g, [c[k] for c in cs if k in c]) for g, cs in cfgs.items())
        every = vals["failed"] + vals["ok"] + vals["open"]
        if len(set(repr(v) for v in every)) == 1:
            same.append("%s=%s" % (k, _fmt(every[0])))
            continue
        fv = vals["failed"]
        nums = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in fv)
        shown = ", ".join(_fmt(v) for v in (sorted(fv) if nums else fv))
        out.append("  %-18s failed %s | ok %s | open %s" % (k, shown, _summary(vals["ok"]),
                                                           _summary(vals["open"])))
    if same:
        out.append("  the same in all: %s" % ", ".join(same))
    return out


def _order_key(status):
    return STATUS_ORDER.index(status) if status in STATUS_ORDER else len(STATUS_ORDER)


def _one_trial(cfg, stage, f):
    print("== trial %s" % f["tid"])
    for k in sorted(f):
        if k != "record":
            print("   %-18s %r" % (k, f[k]))
    print("   %-18s %r" % ("record", f["record"]))
    logs = sorted(glob.glob(os.path.join(glob.escape(stage._dirs(cfg)["jobs"]), f["tid"] + ".*.log")))
    print("   job logs: %s" % (", ".join(os.path.basename(p) for p in logs) or "none"))
    if f["log"]:
        text = CM.read_log_tail(f["log"]) or ""
        print("   ---- the last 60 lines of %s" % f["log"])
        for line in text.splitlines()[-60:]:
            print("   | " + line[:240])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--qstat", action="store_true")
    ap.add_argument("--trial", default=None)
    args = ap.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print("SEARCH REPORT: the configuration %s does not load: %s" % (args.config, exc.problems))
        return 2
    if "search" not in cfg.data:
        print("SEARCH REPORT: %s has no search block" % args.config)
        return 2
    stage, state, rec, jobs = load_search(cfg)
    dirs = stage._dirs(cfg)
    trials = (rec.get("sub") or {}).get("trials") if rec else None
    print("search report %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("configuration %s (config_version %s)" % (cfg.path, cfg.data.get("config_version")))
    print("campaign folder %s" % dirs["campaign"])
    if state is None:
        print("state: none under %s; the trials the ledger and the pending folder name"
              % cfg.data["paths"]["work_root"])
    else:
        print("state %s: the search stage %s, phase %s, round %s"
              % (ST.State.path_of(state.work_root), rec.get("status"),
                 (rec.get("sub") or {}).get("phase"), (rec.get("sub") or {}).get("round")))
    if not trials:
        trials = ledger_only_trials(stage, cfg)
    qstat = None
    if args.qstat:
        lasts = [tr.get("jobs", [])[-1] for tid, tr in trials.items()
                 if tr.get("jobs") and (not args.trial or tid.startswith(args.trial))]
        qstat, problem = qstat_many(lasts)
        print("qstat: %d of %d last job(s) known to the scheduler%s"
              % (len(set(id(r) for r in qstat.values())), len(set(lasts)),
                 ("; it said: " + problem) if problem else ""))
    facts = []
    for tid, tr in sorted(trials.items()):
        if args.trial and not tid.startswith(args.trial):
            continue
        facts.append(trial_facts(cfg, stage, tid, tr, jobs, qstat=qstat))
    facts.sort(key=lambda f: (_order_key(f["status"]), f["round"] if f["round"] is not None else -1,
                              f["tid"]))
    if args.trial:
        for f in facts:
            _one_trial(cfg, stage, f)
    else:
        for f in facts:
            print(trial_line(cfg, f))
        for line in why_lines(cfg, stage, facts):
            print(line)
    for line in summary_lines(cfg, facts, state):
        print(line)
    print("SEARCH REPORT OK: %d trial(s) read; nothing was written" % len(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
