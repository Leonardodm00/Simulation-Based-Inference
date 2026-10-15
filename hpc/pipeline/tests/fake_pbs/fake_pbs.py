#!/usr/bin/env python3
"""An asynchronous fake PBS Pro for the pipeline's tests (not pipeline code).

`qsub` (the wrapper beside this file) records a job in $FAKE_PBS_SPOOL and
starts a detached runner that, after $FAKE_PBS_DELAY seconds (default 0.5),
runs the job script with bash: the -v variables and PBS_JOBID in its
environment, PBS_ARRAY_INDEX per element of a -J array (elements run in
parallel), stdout+stderr to the -o path (^array_index^ substituted) or to
<spool>/<id>.o. `-W depend=afterok:<id>` waits for that job and refuses to
run when it failed (the job then finishes with Exit_status -1 and a comment),
which is what the sequencer must see as a failure.

`qstat -x -f <id>` prints a PBS-like record: job_state (Q, R, B, F),
Exit_status when finished, resources_used.walltime, array_state_count for
arrays. An unknown id prints `qstat: Unknown Job Id <id>` and exits 153.

Pure ASCII, LF only; stdlib only.
"""

import json
import os
import subprocess
import sys
import time

SPOOL = os.environ.get("FAKE_PBS_SPOOL", "")
DELAY = float(os.environ.get("FAKE_PBS_DELAY", "0.5"))
SERVER = "fake"


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
           "indices": indices, "state": "Q", "exit_status": None,
           "started": None, "finished": None, "comment": "",
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

def _wait_dependency(dep):
    """Block until the dependency finished; True iff it exited 0."""
    num = int(dep.split("[")[0].split(".")[0])
    while True:
        try:
            d = _load(num)
        except (OSError, ValueError):
            return False
        if d["state"] == "F":
            return d["exit_status"] == 0
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
                                stdout=fh, stderr=subprocess.STDOUT)
    return proc


def cmd_run(num):
    rec = _load(num)
    time.sleep(DELAY)
    if rec["depend"] and not _wait_dependency(rec["depend"]):
        rec["state"], rec["exit_status"] = "F", -1
        rec["comment"] = "dependency %s did not finish with exit 0" % rec["depend"]
        rec["started"] = rec["finished"] = time.time()
        _save(rec)
        return 0
    rec["state"] = "B" if rec["indices"] else "R"
    rec["started"] = time.time()
    _save(rec)
    if rec["indices"]:
        procs = dict((k, _run_one(rec, k)) for k in rec["indices"])
        codes = {}
        for k, p in procs.items():
            codes[k] = p.wait()
            rec["elements"][str(k)] = codes[k]
            _save(rec)
        rec["exit_status"] = max(codes.values())
    else:
        rec["exit_status"] = _run_one(rec, None).wait()
    rec["state"] = "F"
    rec["finished"] = time.time()
    _save(rec)
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
    if rec["state"] == "F":
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
    print("\n".join(lines))
    return 0


def main(argv):
    if argv and argv[0] == "--run":
        return cmd_run(int(argv[1]))
    if argv and argv[0] == "qsub":
        return cmd_qsub(argv[1:])
    if argv and argv[0] == "qstat":
        return cmd_qstat(argv[1:])
    sys.stderr.write("fake_pbs.py: qsub ... | qstat ... | --run N\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
