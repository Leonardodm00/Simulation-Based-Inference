#!/usr/bin/env python3
"""An asynchronous fake PBS Pro for the pipeline's tests (not pipeline code).

`qsub` (the wrapper beside this file) records a job in $FAKE_PBS_SPOOL and
starts a detached runner that, after $FAKE_PBS_DELAY seconds (default 0.5),
runs the job script with bash: the -v variables and PBS_JOBID in its
environment, PBS_ARRAY_INDEX per element of a -J array (elements run in
parallel), stdout+stderr to the -o path (^array_index^ substituted) or to
<spool>/<id>.o. `-W depend=afterok:<id>` waits for that job and refuses to
run when it failed (the job then finishes with Exit_status -1 and a comment),
which is what the sequencer must see as a failure -- for a PLAIN job. For
an ARRAY (`<n>[]`) the dependency is ordering only, as measured on
davinci's PBS Pro (D-006, 2026-09-21): the dependent job runs whatever the
elements' exit codes, and the gate tools are what refuse a partial result.
Two switches model a scheduler that ends the dependent of an array unrun,
as davinci did on 2026-10-16 (the bank's gate: no log, no exit status):
FAKE_PBS_ARRAY_DEPEND=gating ends it unrun when an element exited non-zero;
FAKE_PBS_DROP_DEPENDENTS=1 ends it unrun whatever the elements did. Such a
job finishes with state F, no Exit_status, a comment, and no output file.
FAKE_PBS_FAIL_ONCE=<i>[,<j>...] makes array element i fail to start the
first time an array runs it (code -2, PBS's JOB_EXEC_FAIL2, and no log:
davinci's element 63 of 2026-10-16); later arrays run it (a spool flag).

`-W depend=afterany:<id>` waits for that job to finish and then runs,
whatever its exit status (a plain job: the sequencer's driver chain, block 1
amended 2026-10-18). `-h` submits the job held (state H). `qdel <id>` ends
a queued or held job unrun (state F, no Exit_status, the comment `Job deleted
by user`) and kills a running one (Exit_status 271). `qselect [-N name] [-s
states] [-u user] [-x]` prints the ids of the matching jobs, one per line;
as PBS Pro, without -x no finished job. FAKE_PBS_QSELECT_SUFFIX=<text>
appends <text> to every id qselect prints (`1.fake.cluster`): a server that
spells its ids differently in qselect than in qsub and PBS_JOBID, which the
sequencer must still recognise as one job. FAKE_PBS_QSELECT_HISTORY=1
lists the finished jobs too, as if -x were given: a server whose qselect
does, which the sequencer must not mistake for live jobs.

`qstat -x -f <id>` prints a PBS-like record: job_state (Q, H, R, B, F),
Exit_status when finished, resources_used.walltime, array_state_count for
arrays. An unknown id prints `qstat: Unknown Job Id <id>` and exits 153.

With FAKE_PBS_ENFORCE_WALLTIME=1 a non-array job whose `-l walltime=` has
elapsed is killed (its process group, SIGKILL) and finishes with
Exit_status 271, as PBS Pro reports a job killed at its walltime (block 7's
tests: a trial killed and resubmitted). As PBS Pro, a plain job ends when its
script ends: whatever of its process group is still alive then is killed
(SIGKILL) before the job is recorded finished, so a job that depends on it
never starts beside a leftover of it.

Pure ASCII, LF only; stdlib only.
"""

import json
import os
import subprocess
import sys
import time

SPOOL = os.environ.get("FAKE_PBS_SPOOL", "")
DELAY = float(os.environ.get("FAKE_PBS_DELAY", "0.5"))
ENFORCE_WALLTIME = os.environ.get("FAKE_PBS_ENFORCE_WALLTIME", "0") == "1"
ARRAY_DEPEND = os.environ.get("FAKE_PBS_ARRAY_DEPEND", "ordering")
DROP_DEPENDENTS = os.environ.get("FAKE_PBS_DROP_DEPENDENTS", "0") == "1"
FAIL_ONCE = set(x.strip() for x in os.environ.get("FAKE_PBS_FAIL_ONCE", "").split(",")
                if x.strip())
SERVER = "fake"
QSELECT_SUFFIX = os.environ.get("FAKE_PBS_QSELECT_SUFFIX", "")
QSELECT_HISTORY = os.environ.get("FAKE_PBS_QSELECT_HISTORY", "0") == "1"


def _spool():
    if not SPOOL:
        sys.stderr.write("fake_pbs: FAKE_PBS_SPOOL is unset\n")
        sys.exit(2)
    if not os.path.isdir(SPOOL):
        os.makedirs(SPOOL)
    return SPOOL


def _path(job_num):
    return os.path.join(_spool(), "%d.json" % int(job_num))


def _load(job_num):
    with open(_path(job_num)) as fh:
        return json.load(fh)


def _save(rec):
    p = _path(rec["num"])
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, p)


def _next_num():
    counter = os.path.join(_spool(), "counter")
    n = 0
    if os.path.isfile(counter):
        with open(counter) as fh:
            n = int(fh.read().strip() or 0)
    n += 1
    with open(counter, "w") as fh:
        fh.write("%d\n" % n)
    return n


def _walltime(seconds):
    s = int(round(seconds))
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


# ---------------------------------------------------------------------------
# qsub
# ---------------------------------------------------------------------------

def cmd_qsub(argv):
    name, queue, out, vars_, depend, array = "job", "", "", {}, None, None
    depend_kind, held = "afterok", False
    resources = []
    script = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-N":
            name = argv[i + 1]; i += 2
        elif a == "-q":
            queue = argv[i + 1]; i += 2
        elif a == "-l":
            resources.append(argv[i + 1]); i += 2
        elif a == "-J":
            array = argv[i + 1]; i += 2
        elif a == "-j":
            i += 2
        elif a == "-h":
            held = True; i += 1
        elif a == "-o":
            out = argv[i + 1]; i += 2
        elif a == "-v":
            for kv in argv[i + 1].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    vars_[k] = v
                elif kv:
                    vars_[kv] = os.environ.get(kv, "")
            i += 2
        elif a == "-W":
            w = argv[i + 1]
            if w.startswith("depend=afterok:"):
                depend = w[len("depend=afterok:"):]
            elif w.startswith("depend=afterany:"):
                depend, depend_kind = w[len("depend=afterany:"):], "afterany"
            i += 2
        elif a.startswith("-"):
            sys.stderr.write("fake qsub: unknown option %s\n" % a)
            return 2
        else:
            script = a; i += 1
    if script is None or not os.path.isfile(script):
        sys.stderr.write("fake qsub: no script (%r)\n" % (script,))
        return 2
    indices = None
    if array:
        rng = array.split("%")[0]
        lo, hi = rng.split("-")
        indices = list(range(int(lo), int(hi) + 1))
        if len(indices) < 2:
            sys.stderr.write("qsub: Invalid Job Array Range\n")
            return 1
    num = _next_num()
    job_id = ("%d[].%s" if indices else "%d.%s") % (num, SERVER)
    rec = {"num": num, "id": job_id, "name": name, "queue": queue,
           "resources": resources, "script": os.path.abspath(script),
           "cwd": os.getcwd(), "out": out, "vars": vars_, "depend": depend,
           "depend_kind": depend_kind,
           "indices": indices, "array_spec": array,
           "state": "H" if held else "Q", "exit_status": None,
           "submitted": time.time(), "started": None, "finished": None, "comment": "",
           "elements": {}}
    _save(rec)
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--run", str(num)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    print(job_id)
    return 0


# ---------------------------------------------------------------------------
# the detached runner
# ---------------------------------------------------------------------------

def _deleted(num):
    try:
        return _load(num)["state"] == "F"
    except (OSError, ValueError):
        return True


def _wait_dependency(dep, kind="afterok", self_num=None):
    """Block until the dependency finished; afterany: True once it finished;
    afterok: True iff it exited 0 -- or, for
    an array dependency, True once it finished (ordering only, D-006), unless
    one of the two switches of the module docstring says otherwise."""
    num = int(dep.split("[")[0].split(".")[0])
    is_array = "[" in dep
    while True:
        try:
            d = _load(num)
        except (OSError, ValueError):
            return False
        if self_num is not None and _deleted(self_num):
            return False
        if d["state"] == "F":
            if kind == "afterany":
                return True
            if not is_array:
                return d["exit_status"] == 0
            if DROP_DEPENDENTS:
                return False
            if ARRAY_DEPEND == "gating":
                # any element that did not exit 0 -- a code of its own, or one
                # that never started (-2) -- ends the dependent unrun
                return all(int(c) == 0 for c in d["elements"].values())
            return True
        time.sleep(0.1)


def _run_one(rec, index):
    env = dict(os.environ)
    env.update(rec["vars"])
    env["PBS_JOBID"] = rec["id"] if index is None else "%d[%d].%s" % (rec["num"], index, SERVER)
    env["PBS_O_WORKDIR"] = rec["cwd"]
    if index is not None:
        env["PBS_ARRAY_INDEX"] = str(index)
    out = rec["out"] or os.path.join(_spool(), "%d.o" % rec["num"])
    out = out.replace("^array_index^", "" if index is None else str(index))
    if index is not None and not rec["out"]:
        out = os.path.join(_spool(), "%d_%d.o" % (rec["num"], index))
    d = os.path.dirname(out)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(out, "ab") as fh:
        proc = subprocess.Popen(["bash", rec["script"]], cwd=rec["cwd"], env=env,
                                stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    return proc


def _walltime_limit(rec):
    """Seconds of `-l walltime=hh:mm:ss`, or None."""
    for r in rec["resources"]:
        if r.startswith("walltime="):
            h, m, s = r[len("walltime="):].split(":")
            return int(h) * 3600 + int(m) * 60 + int(s)
    return None


def _wait_with_walltime(proc, rec):
    """Wait for a job's process; with the enforcement on, kill its process
    group at the walltime and report 271."""
    limit = _walltime_limit(rec) if ENFORCE_WALLTIME else None
    if limit is None:
        return proc.wait()
    try:
        return proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            pass
        proc.wait()
        rec["comment"] = "job killed: walltime %d exceeded limit %d" % (limit + 1, limit)
        return 271


def _end_group(pgid):
    """Kill what is left of a job's process group, and wait until it is gone."""
    try:
        os.killpg(pgid, 9)
    except OSError:
        return
    for _ in range(200):
        try:
            os.killpg(pgid, 0)
        except OSError:
            return
        time.sleep(0.05)


def cmd_run(num):
    rec = _load(num)
    time.sleep(DELAY)
    while _load(num)["state"] == "H":           # held: wait for a release or a qdel
        time.sleep(0.1)
    if _deleted(num):
        return 0
    rec = _load(num)
    if rec["depend"] and not _wait_dependency(rec["depend"], rec.get("depend_kind", "afterok"), num):
        if _deleted(num):
            return 0
        if "[" in rec["depend"]:
            # ended unrun, as davinci ended the bank's gate (2026-10-16): no exit status, no log
            rec["state"], rec["exit_status"] = "F", None
            rec["comment"] = "Job deleted as result of dependency on job %s" % rec["depend"]
            rec["finished"] = time.time()
            _save(rec)
            return 0
        rec["state"], rec["exit_status"] = "F", -1
        rec["comment"] = "dependency %s did not finish with exit 0" % rec["depend"]
        rec["started"] = rec["finished"] = time.time()
        _save(rec)
        return 0
    if _deleted(num):
        return 0
    rec = _load(num)
    rec["state"] = "B" if rec["indices"] else "R"
    rec["started"] = time.time()
    _save(rec)
    if rec["indices"]:
        procs, codes = {}, {}
        for k in rec["indices"]:
            flag = os.path.join(_spool(), "failed_once_%d" % k)
            if str(k) in FAIL_ONCE and not os.path.exists(flag):
                with open(flag, "w") as fh:
                    fh.write("%s\n" % rec["id"])
                codes[k] = -2                 # never started: no process, no log
                rec["elements"][str(k)] = -2
                continue
            procs[k] = _run_one(rec, k)
        for k, p in procs.items():
            codes[k] = p.wait()
            rec["elements"][str(k)] = codes[k]
            _save(rec)
        # as davinci's record of 2026-10-16: an element that never started
        # leaves the array's Exit_status at the others' maximum
        rec["exit_status"] = max(codes.values())
    else:
        proc = _run_one(rec, None)
        rec["pgid"] = proc.pid
        _save(rec)
        rec["exit_status"] = _wait_with_walltime(proc, rec)
        _end_group(proc.pid)
        if _load(num).get("comment", "").startswith("Job deleted by user"):
            rec["exit_status"], rec["comment"] = 271, "Job deleted by user (running)"
    rec["state"] = "F"
    rec["finished"] = time.time()
    _save(rec)
    return 0


# ---------------------------------------------------------------------------
# qdel, qselect
# ---------------------------------------------------------------------------

def cmd_qdel(argv):
    rc = 0
    for job_id in [a for a in argv if not a.startswith("-")]:
        try:
            num = int(job_id.split("[")[0].split(".")[0])
            rec = _load(num)
        except (ValueError, OSError):
            sys.stderr.write("qdel: Unknown Job Id %s\n" % job_id)
            rc = 153
            continue
        if rec["state"] == "F":
            continue
        if rec["state"] in ("Q", "H"):
            rec["state"], rec["exit_status"] = "F", None
            rec["comment"] = "Job deleted by user"
            rec["finished"] = time.time()
            _save(rec)
            continue
        rec["comment"] = "Job deleted by user"
        _save(rec)
        pgid = rec.get("pgid")
        if pgid:
            try:
                os.killpg(int(pgid), 9)
            except OSError:
                pass
    return rc


def cmd_qselect(argv):
    name, states, history, i = None, None, False, 0
    while i < len(argv):
        if argv[i] == "-N":
            name = argv[i + 1]; i += 2
        elif argv[i] == "-s":
            states = argv[i + 1]; i += 2
        elif argv[i] == "-u":
            i += 2                              # one user in the tests
        elif argv[i] == "-x":
            history = True; i += 1
        else:
            i += 1
    out = []
    for p in sorted(os.listdir(_spool())):
        if not p.endswith(".json") or not p[:-5].isdigit():
            continue
        try:
            rec = _load(int(p[:-5]))
        except (OSError, ValueError):
            continue
        if name is not None and rec["name"] != name:
            continue
        if states is not None and rec["state"] not in states:
            continue
        if not (history or QSELECT_HISTORY) and rec["state"] in ("F", "X"):
            continue
        out.append((rec["num"], rec["id"]))
    for _, job_id in sorted(out):
        print(job_id + QSELECT_SUFFIX)
    return 0


# ---------------------------------------------------------------------------
# qstat
# ---------------------------------------------------------------------------

def cmd_qstat(argv):
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        sys.stderr.write("fake qstat: give a job id\n")
        return 2
    job_id = args[0]
    try:
        num = int(job_id.split("[")[0].split(".")[0])
        rec = _load(num)
    except (ValueError, OSError):
        sys.stderr.write("qstat: Unknown Job Id %s\n" % job_id)
        return 153
    lines = ["Job Id: %s" % rec["id"],
             "    Job_Name = %s" % rec["name"],
             "    job_state = %s" % rec["state"],
             "    queue = %s" % rec["queue"]]
    if rec["state"] == "F" and rec["exit_status"] is not None:
        lines.append("    Exit_status = %d" % rec["exit_status"])
    if rec["started"]:
        end = rec["finished"] or time.time()
        lines.append("    resources_used.walltime = %s" % _walltime(end - rec["started"]))
    if rec["indices"]:
        n = len(rec["indices"])
        fin = len(rec["elements"])
        running = 0 if rec["state"] in ("Q", "F") else n - fin
        lines.append("    array_state_count = Queued:%d Running:%d Exiting:0 Expired:0 Finished:%d"
                     % (0 if rec["state"] != "Q" else n, running, fin))
    if rec["comment"]:
        lines.append("    comment = %s" % rec["comment"])
    if rec["out"]:
        # as PBS Pro: "<submission host>:<path>", wrapped (newline + tab) like davinci's
        v = "fakehost:" + rec["out"]
        parts = [v[i:i + 54] for i in range(0, len(v), 54)]
        lines.append("    Output_Path = " + "\n\t".join(parts))
    print("\n".join(lines))
    return 0


def main(argv):
    if argv and argv[0] == "--run":
        return cmd_run(int(argv[1]))
    if argv and argv[0] == "qsub":
        return cmd_qsub(argv[1:])
    if argv and argv[0] == "qstat":
        return cmd_qstat(argv[1:])
    if argv and argv[0] == "qdel":
        return cmd_qdel(argv[1:])
    if argv and argv[0] == "qselect":
        return cmd_qselect(argv[1:])
    sys.stderr.write("fake_pbs.py: qsub ... | qstat ... | qdel ... | qselect ... | --run N\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
