"""The scheduler adapter: `qsub` and `qstat -x -f` (specs/SPEC.md sec. 3.3).

Two calls, kept thin so the fake PBS of tests/fake_pbs/ can stand in for
davinci's PBS Pro by being first on PATH. Nothing else in the pipeline
talks to the scheduler.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import os
import re
import subprocess
import time

__all__ = ["JobStatus", "SubmitError", "SchedulerError", "submit", "status",
           "parse_qstat", "walltime_to_seconds", "seconds_to_walltime",
           "FINISHED_STATES", "QSUB", "QSTAT"]

# Overridable for a test that wants to name the executables explicitly;
# otherwise PATH decides (the fake PBS is put first on PATH by the tests).
QSUB = os.environ.get("PIPELINE_QSUB", "qsub")
QSTAT = os.environ.get("PIPELINE_QSTAT", "qstat")

# PBS Pro job_state letters. F = finished, X = expired subjob (finished).
FINISHED_STATES = ("F", "X")
_KNOWN_STATES = ("Q", "R", "H", "B", "E", "F", "X", "S", "T", "W", "M")

_QSTAT_RETRIES = 3
_QSTAT_RETRY_WAIT_S = 20.0


class SubmitError(RuntimeError):
    """qsub exited non-zero; carries its output."""


class SchedulerError(RuntimeError):
    """qstat could not be run after the retries."""


class JobStatus(object):
    """One job's record as the sequencer reads it."""

    def __init__(self, job_id, state, exit_status=None, walltime_s=None,
                 array_counts=None, raw=""):
        self.job_id = job_id
        self.state = state                  # a letter of _KNOWN_STATES or "unknown"
        self.exit_status = exit_status      # int or None
        self.walltime_s = walltime_s        # float seconds or None
        self.array_counts = array_counts or {}
        self.raw = raw

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
    return JobStatus(job_id, state, exit_status, walltime_s, counts, raw=text)


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
