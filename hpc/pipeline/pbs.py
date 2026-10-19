"""The scheduler adapter: `qsub`, `qstat -x -f`, and since 2026-10-18
`qselect -N` and `qdel` (specs/SPEC.md sec. 3.3; block 1 amended: the
sequencer runs as a chain of PBS jobs, D-098).

Thin calls, so the fake PBS of tests/fake_pbs/ can stand in for davinci's
PBS Pro by being first on PATH. Nothing else in the pipeline talks to the
scheduler.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import getpass
import os
import re
import subprocess
import time

__all__ = ["JobStatus", "SubmitError", "SchedulerError", "submit", "status", "select",
           "delete", "job_seq", "job_number", "same_job", "cores_of_select",
           "cores_of_argv", "parse_qstat",
           "walltime_to_seconds", "seconds_to_walltime", "FINISHED_STATES",
           "RUNNING_STATES", "QSUB", "QSTAT", "QSELECT", "QDEL"]

# Overridable for a test that wants to name the executables explicitly;
# otherwise PATH decides (the fake PBS is put first on PATH by the tests).
QSUB = os.environ.get("PIPELINE_QSUB", "qsub")
QSTAT = os.environ.get("PIPELINE_QSTAT", "qstat")
QSELECT = os.environ.get("PIPELINE_QSELECT", "qselect")
QDEL = os.environ.get("PIPELINE_QDEL", "qdel")

# PBS Pro job_state letters. F = finished, X = expired subjob (finished).
FINISHED_STATES = ("F", "X")
# the states of a job that runs (B: an array with an element running)
RUNNING_STATES = ("R", "E", "B")
_KNOWN_STATES = ("Q", "R", "H", "B", "E", "F", "X", "S", "T", "W", "M")

_QSTAT_RETRIES = 3
_QSTAT_RETRY_WAIT_S = 20.0


class SubmitError(RuntimeError):
    """qsub exited non-zero; carries its output."""


class SchedulerError(RuntimeError):
    """The scheduler did not answer: qstat after its retries, or qselect /
    qdel that could not run."""


class JobStatus(object):
    """One job's record as the sequencer reads it."""

    def __init__(self, job_id, state, exit_status=None, walltime_s=None,
                 array_counts=None, raw="", output_path=None):
        self.job_id = job_id
        self.state = state                  # a letter of _KNOWN_STATES or "unknown"
        self.exit_status = exit_status      # int or None
        self.walltime_s = walltime_s        # float seconds or None
        self.array_counts = array_counts or {}
        self.raw = raw
        self.output_path = output_path      # Output_Path without its "host:" prefix, or None

    @property
    def finished(self):
        return self.state in FINISHED_STATES or self.state == "unknown"

    def __repr__(self):
        return "JobStatus(%r, %s, exit=%r, walltime_s=%r)" % (
            self.job_id, self.state, self.exit_status, self.walltime_s)


def walltime_to_seconds(text):
    """'hh:mm:ss' (hours unbounded) -> float seconds."""
    m = re.match(r"^\s*(\d+):([0-5]\d):([0-5]\d)\s*$", text)
    if not m:
        raise ValueError("not a walltime: %r" % (text,))
    h, mi, s = (int(x) for x in m.groups())
    return float(h * 3600 + mi * 60 + s)


def seconds_to_walltime(seconds):
    seconds = int(round(seconds))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def submit(argv, cwd=None, env=None):
    """Run a complete qsub command line; return the job id (first token)."""
    argv = list(argv)
    if os.path.basename(argv[0]) == "qsub":
        argv[0] = QSUB
    proc = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        raise SubmitError("qsub exited %d\nstdout: %s\nstderr: %s"
                          % (proc.returncode, proc.stdout.strip(), proc.stderr.strip()))
    tokens = proc.stdout.split()
    if not tokens:
        raise SubmitError("qsub printed no job id\nstderr: %s" % proc.stderr.strip())
    return tokens[0]


def parse_qstat(job_id, text):
    """Parse one `qstat -x -f` record. Wrapped values (newline + tab) are
    rejoined before the ' = ' split, as qstat wraps at about 80 columns."""
    joined = text.replace("\n\t", "")
    fields = {}
    for line in joined.splitlines():
        if " = " not in line:
            continue
        k, v = line.split(" = ", 1)
        fields[k.strip()] = v.strip()
    state = fields.get("job_state", "unknown")
    if state not in _KNOWN_STATES:
        state = "unknown"
    exit_status = None
    if "Exit_status" in fields:
        try:
            exit_status = int(fields["Exit_status"])
        except ValueError:
            exit_status = None
    walltime_s = None
    if "resources_used.walltime" in fields:
        try:
            walltime_s = walltime_to_seconds(fields["resources_used.walltime"])
        except ValueError:
            walltime_s = None
    counts = {}
    if "array_state_count" in fields:
        for part in fields["array_state_count"].split():
            if ":" in part:
                name, n = part.split(":", 1)
                try:
                    counts[name] = int(n)
                except ValueError:
                    pass
    output_path = None
    if "Output_Path" in fields:
        # "<submission host>:<path>"; the host is the qsub's, so only the path compares
        v = fields["Output_Path"]
        output_path = v.split(":", 1)[1] if ":" in v else v
    return JobStatus(job_id, state, exit_status, walltime_s, counts, raw=text,
                     output_path=output_path)


def status(job_id, retries=_QSTAT_RETRIES, retry_wait_s=_QSTAT_RETRY_WAIT_S):
    """`qstat -x -f <job_id>`, parsed. An unknown job (purged history or a
    bad id) is JobStatus(state='unknown'); a qstat that cannot run is
    retried and then raises SchedulerError."""
    last = ""
    for attempt in range(int(retries)):
        try:
            proc = subprocess.run([QSTAT, "-x", "-f", str(job_id)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True)
        except OSError as exc:
            raise SchedulerError("cannot run %s: %s" % (QSTAT, exc))
        if proc.returncode == 0 and proc.stdout.strip():
            return parse_qstat(job_id, proc.stdout)
        err = (proc.stderr or "") + (proc.stdout or "")
        if "Unknown Job" in err or "Unknown queue" in err or "does not exist" in err:
            return JobStatus(job_id, "unknown", raw=err)
        last = err.strip()
        if attempt + 1 < retries:
            time.sleep(float(retry_wait_s))
    raise SchedulerError("qstat failed %d times for %s: %s" % (retries, job_id, last))


def cores_of_select(text):
    """The cores of one `-l` value: `select=K:ncpus=N[:...][+K2:ncpus=N2...]`
    gives the sum of K x N over the chunks (K = 1 when a chunk has no count),
    an old-style `ncpus=N` gives N; other resources in the same value
    (`,walltime=...`) are ignored. None when no ncpus is named."""
    total, seen = 0, False
    for part in str(text).split(","):
        part = part.strip()
        if part.startswith("select="):
            for chunk in part[len("select="):].split("+"):
                fields = chunk.split(":")
                k = int(fields[0]) if fields and fields[0].isdigit() else 1
                for f in fields:
                    if f.startswith("ncpus=") and f[len("ncpus="):].isdigit():
                        total += k * int(f[len("ncpus="):])
                        seen = True
        elif part.startswith("ncpus=") and part[len("ncpus="):].isdigit():
            total += int(part[len("ncpus="):])
            seen = True
    return total if seen else None


_ARRAY_RANGE = re.compile(r"^([0-9]+)-([0-9]+)(?::([0-9]+))?(?:%([0-9]+))?$")


def cores_of_argv(argv):
    """The cores a qsub command line can hold at once (D-099): the cores of
    its `-l select=...` (or `ncpus=`) values, times, for an array `-J
    a-b[:s][%c]`, min(c, count) with count = (b - a) // s + 1 (all of them
    without `%`). None when the line names no ncpus."""
    argv = [str(a) for a in argv]
    per, seen = 0, False
    elements = 1
    for i, a in enumerate(argv[:-1]):
        if a == "-l":
            c = cores_of_select(argv[i + 1])
            if c is not None:
                per += c
                seen = True
        elif a == "-J":
            m = _ARRAY_RANGE.match(argv[i + 1].strip())
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                step = int(m.group(3) or 1)
                count = (hi - lo) // step + 1 if hi >= lo else 0
                elements = min(int(m.group(4)), count) if m.group(4) else count
    return per * elements if seen else None


def job_seq(job_id):
    """The sequence part of a job id, before the first dot: `1829506` of
    `1829506.davinci-mgt01`, `1829506[]` of an array. Two spellings of one
    job (qsub's, qselect's, PBS_JOBID's may name the server differently)
    have the same sequence part on one server."""
    return str(job_id).strip().split(".", 1)[0]


def job_number(job_id):
    """The integer of the sequence part (-1 when there is none): the
    server numbers its jobs in submission order."""
    m = re.match(r"[0-9]+", job_seq(job_id))
    return int(m.group(0)) if m else -1


def same_job(a, b):
    return bool(a) and bool(b) and job_seq(a) == job_seq(b)


def select(name, user=None):
    """The ids of this user's jobs named `name` that have not finished
    (`qselect -u <user> -N <name>`: without -x PBS Pro lists no finished
    job), in the order qselect prints them; [] when there is none. The
    states are read with status() where they matter."""
    user = user or getpass.getuser()
    try:
        proc = subprocess.run([QSELECT, "-u", user, "-N", str(name)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
    except OSError as exc:
        raise SchedulerError("cannot run %s: %s" % (QSELECT, exc))
    if proc.returncode != 0:
        raise SchedulerError("qselect exited %d: %s" % (proc.returncode,
                                                         (proc.stderr or proc.stdout).strip()))
    return [tok for tok in proc.stdout.split() if tok]


def delete(job_id):
    """`qdel <job_id>`; True when the scheduler accepted it (a job already
    finished gives False, which is not an error here)."""
    try:
        proc = subprocess.run([QDEL, str(job_id)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)
    except OSError as exc:
        raise SchedulerError("cannot run %s: %s" % (QDEL, exc))
    return proc.returncode == 0
