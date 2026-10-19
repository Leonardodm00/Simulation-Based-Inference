#!/usr/bin/env python3
"""The sequencer behind the one command (specs/SPEC.md secs. 2.1-2.2, block 1).

    python3 pipeline.py submit --config projects/<tag>/<tag>.json [--retry S] [--accept-config-change]
    python3 pipeline.py run    --config projects/<tag>/<tag>.json --dry-run
    python3 pipeline.py status --config projects/<tag>/<tag>.json
    python3 pipeline.py stop   --config projects/<tag>/<tag>.json [--wait S]
    python3 pipeline.py retire --config projects/<tag>/<tag>.json [--wait S]
    python3 pipeline.py paths  --config projects/<tag>/<tag>.json
    python3 pipeline.py drive  --config ...      (the sequencer's own PBS job)
    python3 pipeline.py run    --config ...      (the loop in this shell: tests)

The sequencer drives the stage graph to its end: a stage whose artefact of
record exists is skipped, a stage whose dependencies are done is started,
running stages are polled through the scheduler, and the first refusal stops
the run with its report. It is resumable: the state file records every job,
and a re-run re-attaches to them.

It runs as PBS jobs on compute nodes, never as a process on a login node
(D-098, 2026-10-18: the user's logins land on any login node). `submit`,
from any login node, checks what a run would refuse and submits the first
sequencer job (`pl_<tag>`); each such job (`drive`) first submits its
successor (afterany: it starts when this one ends, whatever the cause),
then ticks until its time budget and exits; the last one deletes its
successor. `status` and `stop` work from any login node: `stop` deletes the
waiting sequencer job and asks the running one to stop at its next tick
boundary -- after the polls and the state's save, never between a
submission and its record (STOP_REQUEST). A signal (PBS ending the job: its
walltime, a qdel) also stops a driver at a tick boundary, but its successor
continues; three sequencer jobs in a row that end uncleanly before a tick
end the chain. `--retry <stage>` is refused while that stage's jobs are
still queued or running, since the reset forgets them. A job about to be
submitted whose name is already queued or running is adopted, not
submitted twice. `retire` stops a sequencer started on a login node before
2026-10-18 from any login node, through the shared file system only.

Exit codes: 0 every stage done or skipped (or a requested stop); 2 a
configuration refusal (a schema problem, a key no built block honours, a
stage not built); 3 a stage failed (its STOP: line is the last line of the
log); 4 another sequencer holds the project's lock, or is already queued.

The pipeline is data-set-agnostic (D-091): nothing here names a project.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import argparse
import errno
import os
import signal
import socket
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as C          # noqa: E402
import pbs as PBS           # noqa: E402
import stages as S          # noqa: E402
from stages import common as CM   # noqa: E402
from state import State     # noqa: E402

LOCK_FILE = "pipeline.lock"
LOG_FILE = "pipeline.log"
STOP_FILE = "STOP_REQUEST"

EXIT_OK, EXIT_CONFIG, EXIT_FAILED, EXIT_LOCKED = 0, 2, 3, 4
EXIT_BUDGET = 5         # run(until=...): a driver job's time budget ended (internal)

# A stop asked by SIGTERM / SIGINT while the sequencer runs: honoured at the
# next tick boundary (a second signal forces the exit at once).
_STOP = {"how": None}
# Why the last run() ended: done, stopped, budget, failed, refused, locked,
# tick limit; and, for a stop, how it was asked (STOP_REQUEST or a signal)
_END = {"why": None, "how": None}

DRIVER_DIR = "driver"
UNCLEAN_FILE = "unclean_ends.json"
# Sequencer jobs in a row that may end uncleanly (an exception, a kill, a
# signal) before one of them completes a tick; the next one ends the chain
# instead of looping (PIPELINE_DRIVER_MAX_UNCLEAN overrides it: the tests).
DRIVER_MAX_UNCLEAN = int(os.environ.get("PIPELINE_DRIVER_MAX_UNCLEAN", "3"))
# Seconds a sequencer waits after replacing a stale lock before it reads the
# lock back: two that replaced it at the same moment, the later write wins.
LOCK_SETTLE_S = 1.0


# ---------------------------------------------------------------------------
# Logging: one line, timestamped, to stdout and the log file
# ---------------------------------------------------------------------------

class Log(object):
    def __init__(self, path=None):
        self.path = path
        if path and not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    def __call__(self, text):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


# ---------------------------------------------------------------------------
# The lock: one sequencer per work root
# ---------------------------------------------------------------------------

def _pid_alive(pid):
    if pid <= 0:                # never os.kill(0 or -1, ...): a process group, everyone
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _holder_token():
    """This sequencer as the lock names it: its PBS job, else its pid."""
    jid = os.environ.get("PBS_JOBID")
    return ("job:%s" % jid) if jid else ("pid:%d" % os.getpid())


def read_lock(work_root):
    """(holder, host) of the lock, or None. The holder is `job:<id>`,
    `pid:<n>`, or a bare pid (a lock written before 2026-10-18)."""
    p = os.path.join(work_root, LOCK_FILE)
    try:
        with open(p, "r") as fh:
            tokens = fh.read().split()
    except OSError:
        return None
    if not tokens:
        return None
    return tokens[0], (tokens[1] if len(tokens) > 1 else "?")


def _state_heartbeat(work_root, quiet_s, log):
    """True as soon as `state.json` is written again within `quiet_s`
    seconds of watching (by this host's clock, two mtime readings: the
    writer's clock need not agree); False after `quiet_s` of silence."""
    p = State.path_of(work_root)

    def mtime():
        try:
            return os.stat(p).st_mtime
        except OSError:
            return None

    first = mtime()
    log("the lock names a sequencer on another host: watching %s for up to %d s for a write "
        "of it" % (p, int(quiet_s)))
    end = time.time() + float(quiet_s)
    while time.time() < end:
        time.sleep(min(5.0, max(0.1, end - time.time())))
        if mtime() != first:
            return True
    return False


def _lock_line(me):
    return "%s %s %s\n" % (me, socket.gethostname(),
                           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _create_lock(p, me):
    """Create the lock only where there is none (O_EXCL): True when this
    call created it, False when another sequencer did first."""
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise
    with os.fdopen(fd, "w") as fh:
        fh.write(_lock_line(me))
    return True


def _replace_lock(p, me):
    tmp = "%s.%d.tmp" % (p, os.getpid())
    with open(tmp, "w") as fh:
        fh.write(_lock_line(me))
    os.replace(tmp, p)


def _lock_is_stale(work_root, lock, log, pbs_module, quiet_s):
    """True when the holder the lock names no longer runs; the reason is
    logged either way."""
    p = os.path.join(work_root, LOCK_FILE)
    holder, host = lock
    if holder.startswith("job:"):
        jid = holder[4:]
        try:
            alive = not pbs_module.status(jid).finished
        except PBS.SchedulerError:
            alive = True                     # cannot tell: do not take it
        if alive:
            log("another sequencer (driver job %s, on %s) holds %s" % (jid, host, p))
            return False
        log("stale lock of driver job %s (finished) replaced" % jid)
        return True
    num = holder[4:] if holder.startswith("pid:") else holder
    pid = int(num) if num.isdigit() else -1
    if host == socket.gethostname():
        if _pid_alive(pid) and pid != os.getpid():
            log("another sequencer (pid %d on %s) holds %s" % (pid, host, p))
            return False
        log("stale lock from pid %d on %s replaced" % (pid, host))
        return True
    if quiet_s is None:
        quiet_s = float(os.environ.get("PIPELINE_HEARTBEAT_QUIET_S", "600"))
    if _state_heartbeat(work_root, quiet_s, log):
        log("another sequencer (pid %d on %s) is still writing the state: it holds %s. Stop it "
            "first (run_pipeline.sh --retire-login-sequencer, README), then submit again"
            % (pid, host, p))
        return False
    log("stale lock from pid %d on %s replaced: no state write for %d s" % (pid, host, int(quiet_s)))
    return True


def acquire_lock(work_root, log, break_lock=False, pbs_module=PBS, quiet_s=None):
    """Return True when this sequencer now holds the lock.

    A lock is stale when its holder is a PBS job the scheduler reports
    finished or unknown, a pid on this host that is not alive, or a pid on
    another host that has not written the state for `quiet_s` seconds
    (PIPELINE_HEARTBEAT_QUIET_S, default 600; amended 2026-10-18, D-098:
    the sequencer's jobs run on any compute node and the user on any login
    node, so liveness is read from the scheduler and the state file, never
    from a process on one host). Where there is no lock it is created with
    O_EXCL; a stale one is replaced and read back after LOCK_SETTLE_S, so
    of two sequencers that replaced it at once the later one keeps it."""
    p = os.path.join(work_root, LOCK_FILE)
    me = _holder_token()
    for _ in range(3):
        lock = read_lock(work_root)
        if lock is None:
            if _create_lock(p, me):
                return True
            continue                         # created by another just now: look again
        if lock[0] == me:
            _replace_lock(p, me)             # this job's own lock (a job PBS ran again)
            return True
        if break_lock:
            log("lock of %s on %s broken (--break-lock)" % lock)
            break_lock = False
        elif not _lock_is_stale(work_root, lock, log, pbs_module, quiet_s):
            return False
        _replace_lock(p, me)
        time.sleep(LOCK_SETTLE_S)
        now = read_lock(work_root)
        if now is None:
            continue                         # removed meanwhile (an old sequencer's exit)
        if now[0] == me:
            return True
        log("the lock was taken by %s on %s at the same moment; this sequencer yields" % now)
        return False
    log("the lock %s changed hands while this sequencer looked at it; it yields" % p)
    return False


def release_lock(work_root):
    """Remove the lock when it is this sequencer's (never another's)."""
    p = os.path.join(work_root, LOCK_FILE)
    lock = read_lock(work_root)
    if lock is not None and lock[0] != _holder_token():
        return
    try:
        os.remove(p)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# A requested stop (amended 2026-10-18): STOP_REQUEST, SIGTERM, SIGINT
# ---------------------------------------------------------------------------

def _on_signal(signum, frame):
    if _STOP["how"] is not None:
        raise SystemExit(128 + signum)      # a second signal: stop now
    _STOP["how"] = "signal %d" % signum


def _stop_requested(work_root):
    """The stop's origin when one was asked, else None; a stop file is
    consumed here, so it stops this sequencer once."""
    if _STOP["how"] is not None:
        return _STOP["how"]
    p = os.path.join(work_root, STOP_FILE)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            note = fh.read().strip()
    except OSError:
        note = ""
    try:
        os.remove(p)
    except OSError:
        pass
    return "%s: %s" % (STOP_FILE, note) if note else STOP_FILE


def _sleep(seconds, work_root):
    """time.sleep(seconds), cut short by a stop request."""
    end = time.time() + float(seconds)
    while True:
        left = end - time.time()
        if left <= 0:
            return
        if _STOP["how"] is not None or os.path.isfile(os.path.join(work_root, STOP_FILE)):
            return
        time.sleep(min(1.0, left))


def _unfinished_jobs(st, pbs_module):
    """The ids of the stage's recorded jobs the scheduler does not report
    finished (an unknown job counts as finished, as everywhere)."""
    live = []
    for job in st.get("jobs", []):
        if job.get("finished"):
            continue
        if not pbs_module.status(job["id"]).finished:
            live.append(job["id"])
    return live


# ---------------------------------------------------------------------------
# What a stage sees of the scheduler and the state
# ---------------------------------------------------------------------------

def _qsub_option(argv, flag):
    """The value after `flag` in a qsub command line, or None."""
    argv = list(argv)
    for i, a in enumerate(argv[:-1]):
        if a == flag:
            return argv[i + 1]
    return None


def _qsub_name(argv):
    """The value of `-N` in a qsub command line, or None."""
    return _qsub_option(argv, "-N")


def _same_output(path_a, path_b):
    """Two output paths name one file (normalised; None never matches)."""
    if not path_a or not path_b:
        return False
    return os.path.normpath(path_a) == os.path.normpath(path_b)


# ---------------------------------------------------------------------------
# The core cap (D-099): what the pipeline's jobs hold at once
# ---------------------------------------------------------------------------

class CapExceeded(RuntimeError):
    """A submission that would take the pipeline past resources.max_cores
    (the stages ask Submitter.fits first: this is a guard)."""


def job_cores(job):
    """The cores a recorded job can hold at once: its `ncpus` (recorded at
    submission since D-099), else read from its qsub line; None when
    neither says."""
    if job.get("ncpus") is not None:
        return int(job["ncpus"])
    return PBS.cores_of_argv(job.get("cmd") or [])


def cores_in_flight(state):
    """(cores, jobs counted, [ids of jobs whose cores no record gives]) over
    the jobs not marked finished of every stage not done or skipped (a done
    stage's record its polls never marked finished holds no cores). Jobs
    recorded `after` one (a gate, an aggregation) start when it ends: the
    group holds the larger of its cores and the sum of theirs, never both."""
    jobs = [j for st in state.data.get("stages", {}).values()
            if st.get("status") not in ("done", "skipped")
            for j in st.get("jobs", []) if not j.get("finished")]
    by_id = dict((j["id"], j) for j in jobs)
    later = {}
    for j in jobs:
        if j.get("after") in by_id:
            later.setdefault(j["after"], []).append(j)
    count = {"n": 0, "unknown": []}
    seen = set()

    def footprint(j):
        if j["id"] in seen:
            return 0
        seen.add(j["id"])
        c = job_cores(j)
        if c is None:
            count["unknown"].append(j["id"])
            c = 0
        else:
            count["n"] += 1
        return max(c, sum(footprint(d) for d in later.get(j["id"], [])))

    total = sum(footprint(j) for j in jobs if j.get("after") not in by_id)
    return total, count["n"], count["unknown"]


def stage_budget(cfg):
    """The cores the stages' jobs may hold: resources.max_cores less the
    sequencer's own job (D-099); None without a cap (an old configuration)."""
    return None if cfg is None else CM.stage_budget(cfg)


def cores_line(cfg, state):
    """The dry run's and --status's line on the cap."""
    cap = int(cfg.data["resources"]["max_cores"])
    drv = int(cfg.data["driver"]["ncpus"])
    used, n, unknown = cores_in_flight(state)
    return ("cores: at most %d at once (resources.max_cores, D-099): the sequencer's job %d, the "
            "stages' jobs %d; in flight per the state file: %d in %d job(s)%s"
            % (cap, drv, cap - drv, used, n,
               ("; %d job(s) of unknown cores, not counted: %s" % (len(unknown), ", ".join(unknown[:6])))
               if unknown else ""))


class Submitter(object):
    """Submits through the adapter and records every job in the state, so
    a stage never has to know the state file's layout. With the
    configuration it also keeps the stages under resources.max_cores
    (D-099): `fits(n)` before a submission."""

    def __init__(self, pbs_module, state, stage_name, log, cfg=None):
        self._pbs = pbs_module
        self._state = state
        self._name = stage_name
        self._log = log
        self._cfg = cfg

    def budget(self):
        return stage_budget(self._cfg)

    def in_flight(self):
        return cores_in_flight(self._state)[0]

    def free(self):
        b = self.budget()
        return None if b is None else b - self.in_flight()

    def check(self, n):
        """CapExceeded when a job of n cores can never start: more than the
        stages' whole share (a stage's start asks it of the jobs it submits
        later, so it stops before its first job, not after); True else."""
        b = self.budget()
        if b is not None and n is not None and int(n) > b:
            d = self._cfg.data
            raise CapExceeded(
                "%s: a job of %d cores can never start: the stages' jobs share %d cores "
                "(resources.max_cores %d less the sequencer's job, %d); raise "
                "resources.max_cores (--stop, then a submission with --accept-config-change) "
                "or ask fewer cores (D-099)"
                % (self._name, int(n), b, int(d["resources"]["max_cores"]),
                   int(d["driver"]["ncpus"])))
        return True

    def fits(self, n):
        """True when a job of n cores fits under the cap now (always without a
        cap, or when n is not known); one that can never start raises
        (check), so its stage stops with the reason instead of waiting for
        ever."""
        b = self.budget()
        if b is None or n is None:
            return True
        self.check(n)
        return int(n) <= b - self.in_flight()

    def submit(self, argv, role, log_path=None, cwd=None, env=None, ncpus=None, after=None):
        # A job already queued, held or running under the same name AND the
        # same output file (-o, unique per project and role; a name alone is
        # not: `bank_verify`, `gate_b16`) is ADOPTED, not submitted again: a
        # sequencer stopped between a qsub and the state's save (a driver job
        # on a failing node, a kill) leaves such a job behind (amended
        # 2026-10-18, D-098). The scheduler not answering: submitted as before.
        name, out = _qsub_name(argv), _qsub_option(argv, "-o")
        select = getattr(self._pbs, "select", None)
        if name and out and select is not None:
            try:
                live = []
                for j in select(name):
                    js = self._pbs.status(j)
                    if not js.finished and _same_output(js.output_path, out):
                        live.append(j)
            except PBS.SchedulerError:
                live = []
            if live:
                job_id = live[-1]
                self._state.add_job(self._name, job_id, role, argv, log=log_path,
                                    ncpus=ncpus if ncpus is not None else PBS.cores_of_argv(argv),
                                    after=after)
                self._state.save()
                self._log("%s: adopted %s -> %s (a job of that name and output file was queued "
                          "or running but not recorded)" % (self._name, role, job_id))
                return job_id
        n = ncpus if ncpus is not None else PBS.cores_of_argv(argv)
        if n is not None and not self.fits(n):
            raise CapExceeded("%s: %s asks %d cores; %s of %s free under resources.max_cores"
                              % (self._name, role, n, self.free(), self.budget()))
        job_id = self._pbs.submit(argv, cwd=cwd, env=env)
        self._state.add_job(self._name, job_id, role, argv, log=log_path, ncpus=n, after=after)
        self._state.save()
        self._log("%s: submitted %s -> %s" % (self._name, role, job_id))
        return job_id

    def record(self, job_id, role, cmd, log_path=None, ncpus=None, after=None):
        """A job another tool submitted (a launcher's `job : <id>` line):
        recorded like one of ours, so it is polled and resumed alike, with
        the cores the stage worked out before it ran the launcher."""
        self._state.add_job(self._name, job_id, role, cmd, log=log_path, ncpus=ncpus, after=after)
        self._state.save()
        self._log("%s: recorded %s -> %s" % (self._name, role, job_id))
        return job_id

    def status(self, job_id):
        return self._pbs.status(job_id)

    def log(self, text):
        self._log(text)


# ---------------------------------------------------------------------------
# The sequencer
# ---------------------------------------------------------------------------

def _deps_satisfied(state, stage):
    return all(state.stage(d)["status"] in ("done", "skipped") for d in stage.deps)


def _refusals(cfg, graph, state):
    """Configuration-level refusals before anything is submitted: keys no
    built block honours, for the stages that are still to run."""
    pending = [s.name for s in graph
               if state.stage(s.name)["status"] not in ("done", "skipped")
               and not s.is_done(cfg)[0]]
    lines = []
    for key, block, what in C.unhonoured_keys(cfg, pending_stages=pending):
        lines.append("REFUSE: this code cannot honour %s (needs block %d: %s)"
                     % (key, block, what))
    return lines


def stage_table(cfg, graph, state, live=False, pbs_module=PBS):
    """Rows (name, status, done?, detail) for the table."""
    rows = []
    for stage in graph:
        st = state.stage(stage.name)
        done, why = stage.is_done(cfg)
        detail = why
        if live and st["status"] == "running" and st["jobs"]:
            parts = []
            if len(st["jobs"]) > 8:
                # a stage with many jobs (the search's trials): counts, not lines
                n_fin = sum(1 for j in st["jobs"] if j.get("finished"))
                parts.append("%d job(s): %d finished, %d not yet" % (len(st["jobs"]), n_fin,
                                                                      len(st["jobs"]) - n_fin))
                sub = st.get("sub") or {}
                if sub.get("trials"):
                    c = {}
                    for tr in sub["trials"].values():
                        c[tr["status"]] = c.get(tr["status"], 0) + 1
                    parts.append("trials " + ", ".join("%s %d" % kv for kv in sorted(c.items())))
                    parts.append("round %s of %s" % (sub.get("round"), cfg.data["search"]["gp_rounds"]))
                if sub.get("phase"):
                    parts.append("phase " + str(sub["phase"]))
            else:
                for job in st["jobs"]:
                    if job.get("finished"):
                        parts.append("%s %s finished (exit %s)" % (job["role"], job["id"],
                                                                    job.get("exit_status")))
                    else:
                        js = pbs_module.status(job["id"])
                        parts.append("%s %s state %s" % (job["role"], job["id"], js.state))
            detail = "; ".join(parts)
        rows.append((stage.name, st["status"], "yes" if done else "no", detail))
    return rows


def print_table(rows, log):
    w = max(len(r[0]) for r in rows)
    log("| %-*s | %-8s | %-4s | detail" % (w, "stage", "status", "done"))
    for name, status, done, detail in rows:
        log("| %-*s | %-8s | %-4s | %s" % (w, name, status, done, detail))


def _prepare_state(cfg, graph, work_root, retry, accept_config_change, pbs_module, log,
                   dry_run, verb="run"):
    """Load (or create) the state, apply an accepted configuration change and
    the --retry resets; return (state, None) or (None, exit code). `submit`
    checks the same refusals with dry_run=True and verb="submit"."""
    state = State.load(work_root, log) if os.path.isdir(work_root) else None
    if state is None:
        state = State.new(work_root, cfg.tag, cfg.path, cfg.sha256)
        if not dry_run:
            state.save()
            log("new state file %s" % State.path_of(work_root))
    elif state.data["config_sha256"] != cfg.sha256:
        if not accept_config_change:
            log("REFUSE: the configuration changed since this run started "
                "(state has %s, file has %s); re-run with --accept-config-change "
                "to continue under the new values"
                % (state.data["config_sha256"][:16], cfg.sha256[:16]))
            return None, EXIT_CONFIG
        state.event("configuration changed: %s -> %s"
                    % (state.data["config_sha256"][:16], cfg.sha256[:16]))
        state.data["config_sha256"] = cfg.sha256
        state.data["config_path"] = cfg.path

    for name in retry:
        if name not in [s.name for s in graph]:
            log("REFUSE: --retry names no stage: %s" % name)
            return None, EXIT_CONFIG
        # the reset forgets the stage's jobs: refused while any still runs,
        # or their work would be submitted a second time (amended 2026-10-18)
        try:
            live = _unfinished_jobs(state.stage(name), pbs_module)
        except PBS.SchedulerError as exc:
            log("REFUSE: --retry %s: cannot tell whether the stage's jobs still run (the "
                "scheduler did not answer: %s); try again" % (name, exc))
            return None, EXIT_CONFIG
        if live:
            log("REFUSE: --retry %s: the stage has %d job(s) still queued or running (%s%s); a "
                "retry forgets them and would submit their work again -- let them end (or qdel "
                "them) first" % (name, len(live), ", ".join(live[:6]),
                                 ", ..." if len(live) > 6 else ""))
            return None, EXIT_CONFIG
        state.reset_stage(name)
        if verb == "submit":
            log("%s: --retry accepted; the first sequencer job resets it to pending" % name)
        else:
            log("%s reset to pending (--retry)" % name)
    return state, None


def _dry_run(cfg, graph, state, log):
    log("DRY RUN -- nothing is submitted (%s)" % cfg.path)
    print_table(stage_table(cfg, graph, state), log)
    failed = []
    for stage in graph:
        st = state.stage(stage.name)
        if st["status"] in ("done", "skipped"):
            continue
        if st["status"] == "failed":
            # the run would stop here: say so, with the way on
            failed.append(stage.name)
            log("%s: FAILED in the state file (%s); the run would stop here -- add "
                "--retry %s" % (stage.name, st.get("note", ""), stage.name))
            continue
        done, _ = stage.is_done(cfg)
        if done:
            log("%s: would be skipped (artefact of record exists)" % stage.name)
            continue
        # a dependency counts as satisfied when its artefact exists (it
        # would be skipped) or the state says done / skipped
        by_name = dict((s.name, s) for s in graph)
        waits = [d for d in stage.deps
                 if state.stage(d)["status"] not in ("done", "skipped")
                 and not by_name[d].is_done(cfg)[0]]
        if waits:
            log("%s: would wait for %s" % (stage.name, ", ".join(waits)))
        else:
            log("%s: would run" % stage.name)
        for line in stage.dry_run(cfg, st):
            log("    %s" % line)
    if "resources" in cfg.data:
        log(cores_line(cfg, state))
    if "driver" in cfg.data:
        d = cfg.data["driver"]
        log("the sequencer: PBS jobs named %s on queue %s (%d cores, %s, walltime %s, a tick every "
            "%d s), one at a time, each submitting its successor (afterany); nothing runs on a "
            "login node (D-098)" % (driver_name(cfg), d["queue"], int(d["ncpus"]), d["mem"],
                                    d["walltime"], int(d["poll_seconds"])))
    refusals = _refusals(cfg, graph, state)
    for line in refusals:
        log(line)
    not_built = [s for s in graph if isinstance(s, S.NotBuiltStage)
                 and state.stage(s.name)["status"] not in ("done", "skipped")
                 and not s.is_done(cfg)[0]]
    for s in not_built:
        log("REFUSE: stage %s is not built yet (block %d)" % (s.name, s.block))
    if refusals or not_built:
        return EXIT_CONFIG
    if failed:
        log("STOP: the run would stop at the failed stage(s) %s; add %s"
            % (", ".join(failed), " ".join("--retry %s" % n for n in failed)))
        return EXIT_FAILED
    return EXIT_OK


def run(cfg, graph=None, poll_seconds=60.0, dry_run=False, retry=(),
        accept_config_change=False, break_lock=False, pbs_module=PBS,
        max_ticks=None, log=None, until=None, on_tick=None):
    """Drive the graph; return an exit code (module docstring). `until`
    (epoch seconds) ends the run at the first tick boundary past it with
    EXIT_BUDGET: a driver job's time budget (D-098). `on_tick()` is called
    at every tick boundary (every stage polled, the state saved). Why the
    run ended is left in _END["why"] (and a stop's origin in _END["how"])."""
    work_root = cfg.data["paths"]["work_root"]
    graph = S.topological_order(graph if graph is not None else S.default_graph(cfg))
    log = log or Log(None if dry_run else os.path.join(work_root, LOG_FILE))
    _END["why"], _END["how"] = None, None

    # ---- dry run: the table, every submission, the refusals; nothing runs --
    if dry_run:
        state, rc = _prepare_state(cfg, graph, work_root, retry, accept_config_change,
                                   pbs_module, log, dry_run=True)
        if rc is not None:
            _END["why"] = "refused"
            return rc
        return _dry_run(cfg, graph, state, log)

    # ---- the real run: the lock first, then the state (read after the lock,
    # so a sequencer that wrote it while this one waited is not overwritten)
    if not os.path.isdir(work_root):
        os.makedirs(work_root)
    if not acquire_lock(work_root, log, break_lock=break_lock, pbs_module=pbs_module):
        _END["why"] = "locked"
        return EXIT_LOCKED
    old_handlers = {}
    _STOP["how"] = None
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            old_handlers[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):      # not the main thread: no handler, no harm
            pass
    try:
        state, rc = _prepare_state(cfg, graph, work_root, retry, accept_config_change,
                                   pbs_module, log, dry_run=False)
        if rc is not None:
            _END["why"] = "refused"
            return rc
        stale = os.path.join(work_root, STOP_FILE)
        if os.path.isfile(stale) and not os.environ.get("PBS_JOBID"):
            # a request left behind by a stop that found no sequencer; a driver
            # job leaves it in place: it may have been asked while it queued
            os.remove(stale)
            log("an earlier stop request (%s) was removed: it was for a sequencer that has "
                "already stopped" % stale)
        refusals = _refusals(cfg, graph, state)
        if refusals:
            for line in refusals:
                log(line)
            _END["why"] = "refused"
            return EXIT_CONFIG
        log("sequencer started: %s on %s, %d stage(s), poll %.1f s%s"
            % (_holder_token().replace(":", " "), socket.gethostname(), len(graph), poll_seconds,
               (", until %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)))
               if until else ""))
        ticks = 0
        while True:
            ticks += 1
            for stage in graph:
                st = state.stage(stage.name)
                status = st["status"]
                if status in ("done", "skipped"):
                    continue
                if status == "failed":
                    log("STOP: stage %s failed: %s" % (stage.name, st.get("note", "")))
                    _END["why"] = "failed"
                    return EXIT_FAILED
                sub = Submitter(pbs_module, state, stage.name, log, cfg)
                if status == "pending":
                    done, why = stage.is_done(cfg)
                    if done:
                        state.set_status(stage.name, "skipped", why)
                        state.save()
                        log("%s: skipped, artefact of record exists: %s" % (stage.name, why))
                        continue
                    if not _deps_satisfied(state, stage):
                        continue
                    try:
                        stage.start(cfg, st, sub)
                    except S.StageNotBuilt as exc:
                        log("REFUSE: %s" % exc)
                        _END["why"] = "refused"
                        return EXIT_CONFIG
                    except S.Deferred as exc:   # its first job does not fit (D-099)
                        if st.get("waiting_note") != str(exc):
                            st["waiting_note"] = str(exc)
                            state.save()
                            log("%s: waits for cores: %s" % (stage.name, exc))
                        continue
                    except Exception as exc:  # a refusal before submission
                        state.set_status(stage.name, "failed", str(exc))
                        state.save()
                        log("STOP: stage %s refused: %s" % (stage.name, exc))
                        _END["why"] = "failed"
                        return EXIT_FAILED
                    st.pop("waiting_note", None)
                    state.set_status(stage.name, "running")
                    state.save()
                    log("%s: running" % stage.name)
                    st = state.stage(stage.name)
                    status = st["status"]
                if status == "running":
                    try:
                        r = stage.poll(cfg, st, sub)
                    except PBS.SchedulerError as exc:
                        # the scheduler is unreachable: keep the state, try again next tick
                        state.save()
                        log("%s: scheduler unreachable (%s); will retry next tick" % (stage.name, exc))
                        continue
                    except CapExceeded as exc:
                        # a job that can never start, or the guard (D-099): stop with the
                        # reason, not a crash of every driver of the chain
                        state.set_status(stage.name, "failed", str(exc))
                        state.save()
                        log("STOP: stage %s failed: %s" % (stage.name, exc))
                        _END["why"] = "failed"
                        return EXIT_FAILED
                    state.save()
                    if r == "running":
                        continue
                    ok, why = stage.verify(cfg, st)
                    if ok:
                        state.set_status(stage.name, "done", why)
                        state.save()
                        log("%s: done: %s" % (stage.name, why))
                        continue
                    if why.startswith("killed:"):
                        try:
                            resubmitted = stage.on_killed(cfg, st, sub)
                        except Exception as exc:
                            resubmitted = False
                            why = "%s; resubmission refused: %s" % (why, exc)
                        if resubmitted:
                            state.save()
                            log("%s: %s -- resubmitted (%d of %d)"
                                % (stage.name, why, st["resubmits"], stage.max_resubmits))
                            continue
                    state.set_status(stage.name, "failed", why)
                    state.save()
                    log("STOP: stage %s failed: %s" % (stage.name, why))
                    _END["why"] = "failed"
                    return EXIT_FAILED
            statuses = state.statuses()
            if all(statuses.get(s.name) in ("done", "skipped") for s in graph):
                log("PIPELINE DONE: every stage done or skipped")
                _END["why"] = "done"
                return EXIT_OK
            if max_ticks is not None and ticks >= max_ticks:
                log("tick limit %d reached; stopping (resumable)" % max_ticks)
                _END["why"] = "tick limit"
                return EXIT_OK
            # a stop is honoured here only: after the polls and the state's save
            if on_tick is not None:
                on_tick()
            how = _stop_requested(work_root)
            if how is None and until is not None and time.time() >= until:
                log("time budget reached; stopping (resumable): the next driver job continues")
                _END["why"] = "budget"
                return EXIT_BUDGET
            if how is None:
                nap = float(poll_seconds)
                if until is not None:
                    nap = max(0.0, min(nap, until - time.time()))
                _sleep(nap, work_root)
                how = _stop_requested(work_root)
            if how is not None:
                n_live = sum(1 for s_ in graph for j in state.stage(s_.name).get("jobs", [])
                             if not j.get("finished"))
                log("stopped on request (%s); resumable: %d job(s) still queued or running "
                    "keep running" % (how, n_live))
                _END["why"], _END["how"] = "stopped", how
                return EXIT_OK
    finally:
        for sig, h in old_handlers.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass
        _STOP["how"] = None
        release_lock(work_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load(path):
    try:
        return C.load_config(path)
    except C.ConfigError as exc:
        print("REFUSE: the configuration %s has %d problem(s):" % (path, len(exc.problems)))
        for p in exc.problems:
            print("  " + p)
        return None


def cmd_run(args, graph=None, entry=None):
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    return run(cfg, graph=graph, poll_seconds=args.poll_seconds, dry_run=args.dry_run,
               retry=args.retry or (), accept_config_change=args.accept_config_change,
               break_lock=args.break_lock, max_ticks=args.max_ticks)


# ---------------------------------------------------------------------------
# The sequencer as PBS jobs (D-098, 2026-10-18): submit, drive
# ---------------------------------------------------------------------------

def driver_name(cfg):
    """The PBS job name of the project's sequencer jobs."""
    return CM.job_name("pl_" + cfg.tag)


def _driver_budget_s(cfg):
    """A driver job's ticking time: its walltime less min(max(900 s,
    5 x poll), walltime / 4), left for the last tick and the exit."""
    d = cfg.data["driver"]
    wall = PBS.walltime_to_seconds(d["walltime"])
    margin = min(max(900.0, 5.0 * float(d["poll_seconds"])), wall / 4.0)
    return wall - margin


def _default_entry():
    """The command line before the verb: this file under the environment's
    python (the tests put their own graph's script here)."""
    return [CM.E.python_in_env(), os.path.join(_HERE, "pipeline.py")]


def submit_driver(cfg, depend=None, extra=(), pbs_module=PBS, entry=None):
    """qsub one sequencer job: jobs/run_argv.pbs in envs.extract with the
    resources of `driver`, running `<entry> drive`; `depend` makes it wait
    for that job (afterany). Returns (job id, argv file, log)."""
    d = cfg.data["driver"]
    work_root = cfg.data["paths"]["work_root"]
    ddir = os.path.join(work_root, DRIVER_DIR)
    if not os.path.isdir(ddir):
        os.makedirs(ddir)
    k = 1 + len([f for f in os.listdir(ddir) if f.startswith("driver.") and f.endswith(".argv")])
    while os.path.exists(os.path.join(ddir, "driver.%d.argv" % k)):
        k += 1
    argv_file = os.path.join(ddir, "driver.%d.argv" % k)
    log_path = os.path.join(ddir, "driver.%d.log" % k)
    argv = (list(entry or _default_entry()) + ["drive", "--config", os.path.abspath(cfg.path)]
            + [str(a) for a in extra])
    CM.write_argv_file(argv_file, argv)
    line = CM.qsub_line(cfg, driver_name(cfg), d["queue"], d["ncpus"], d["mem"], d["walltime"],
                        log_path, argv_file, work_root, env_name=cfg.data["envs"]["extract"])
    if depend:
        line = line[:-1] + ["-W", "depend=afterany:%s" % depend] + line[-1:]
    return pbs_module.submit(line), argv_file, log_path


def _delete(job_id, log, who):
    """qdel that never raises: a scheduler that cannot be asked is logged."""
    try:
        return PBS.delete(job_id)
    except PBS.SchedulerError as exc:
        log("%s: could not delete %s (%s)" % (who, job_id, exc))
        return False


def _drivers(cfg, pbs_module=PBS):
    """[(job id, state)] of the project's sequencer jobs not finished. PBS
    Pro's qselect lists no finished job without -x; one it lists all the
    same (another server's habit) is dropped here, by qstat's answer."""
    out = []
    for j in pbs_module.select(driver_name(cfg)):
        js = pbs_module.status(j)
        if not js.finished:
            out.append((j, js.state))
    return out


def cmd_submit(args, graph=None, entry=None):
    """The one command's start (run_pipeline.sh --project <tag>): from any
    login node, check what a run would refuse, then submit the sequencer's
    first job. Nothing keeps running on the login node."""
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    if not os.path.isdir(work_root):
        os.makedirs(work_root)
    log = Log(os.path.join(work_root, LOG_FILE))
    graph = S.topological_order(graph if graph is not None else S.default_graph(cfg))
    # the refusals of a run, checked here so that none reaches a driver job:
    # the configuration change, --retry (a stage name; none of its jobs live)
    try:
        probe, rc = _prepare_state(cfg, graph, work_root, args.retry, args.accept_config_change,
                                   PBS, log, dry_run=True, verb="submit")
    except ValueError as exc:
        log("REFUSE: %s" % exc)
        return EXIT_CONFIG
    if rc is not None:
        return rc
    lines = _refusals(cfg, graph, probe)
    lines += ["REFUSE: stage %s is not built yet (block %d)" % (s.name, s.block) for s in graph
              if isinstance(s, S.NotBuiltStage)
              and probe.stage(s.name)["status"] not in ("done", "skipped")
              and not s.is_done(cfg)[0]]
    if lines:
        for line in lines:
            log(line)
        return EXIT_CONFIG
    try:
        live = _drivers(cfg)
    except PBS.SchedulerError as exc:
        log("REFUSE: the scheduler did not answer (%s); try again" % exc)
        return EXIT_LOCKED
    if live:
        log("REFUSE: the sequencer is already queued or running: %s; see --status, or --stop it "
            "first" % ", ".join("%s (%s)" % jv for jv in live))
        return EXIT_LOCKED
    lock = read_lock(work_root)
    if lock is not None and not lock[0].startswith("job:"):
        log("note: the lock names %s on %s, a sequencer started on a login node; if it is still "
            "running, the sequencer job ends without ticking (run_pipeline.sh "
            "--retire-login-sequencer first); if it is gone, the job takes the lock after %s s "
            "without a state write" % (lock[0], lock[1],
                                       os.environ.get("PIPELINE_HEARTBEAT_QUIET_S", "600")))
    stale = os.path.join(work_root, STOP_FILE)
    if os.path.isfile(stale):
        os.remove(stale)
        log("an earlier stop request (%s) was removed: this submission starts the sequencer "
            "again" % stale)
    u = _unclean(work_root)
    if u["count"]:
        _clear_unclean(work_root)
        log("the count of sequencer jobs that ended uncleanly in a row (%d) was reset by this "
            "submission" % u["count"])
    extra = []
    for r in args.retry:
        extra += ["--retry", r]
    if args.accept_config_change:
        extra += ["--accept-config-change"]
    if args.break_lock:
        extra += ["--break-lock"]
    try:
        job, argv_file, log_path = submit_driver(cfg, extra=extra, entry=entry)
    except (PBS.SubmitError, OSError, ValueError) as exc:
        log("REFUSE: the sequencer's job could not be submitted: %s" % exc)
        return EXIT_CONFIG
    log("SUBMITTED the sequencer: job %s (%s), log %s; it runs on a compute node and submits "
        "its own successor; --status and --stop work from any login node"
        % (job, driver_name(cfg), log_path))
    return EXIT_OK


def _unclean(work_root):
    """{"count": unclean ends in a row, "ends": [[utc, job, why], ...]}."""
    p = os.path.join(work_root, DRIVER_DIR, UNCLEAN_FILE)
    try:
        d = CM.read_json(p)
        return {"count": int(d.get("count", 0)), "ends": list(d.get("ends", []))}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"count": 0, "ends": []}


def _note_unclean(work_root, who, why):
    d = _unclean(work_root)
    d["count"] += 1
    d["ends"] = (d["ends"] + [[CM.utc_now(), who, why]])[-10:]
    CM.write_json(os.path.join(work_root, DRIVER_DIR, UNCLEAN_FILE), d)
    return d["count"]


def _clear_unclean(work_root):
    d = _unclean(work_root)
    if d["count"]:
        d["count"] = 0
        CM.write_json(os.path.join(work_root, DRIVER_DIR, UNCLEAN_FILE), d)


def _older_running(others, me):
    """The drivers among `others` [(id, state)] that run and were submitted
    before `me`: of two drivers running at once, the older one keeps the
    work root and the younger one exits (a tie broken by the job numbers)."""
    return [j for j, s in others if s in PBS.RUNNING_STATES
            and PBS.job_number(j) < PBS.job_number(me)]


def cmd_drive(args, graph=None, entry=None):
    """The sequencer's job (D-098): keep one successor queued, tick until the
    time budget, hand over; the chain ends with the pipeline."""
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    me = os.environ.get("PBS_JOBID")
    if not me:
        print("REFUSE: `drive` is the sequencer's PBS job (PBS_JOBID is unset); start the "
              "pipeline with run_pipeline.sh --project <tag> from any login node")
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    log = Log(os.path.join(work_root, LOG_FILE))
    host = socket.gethostname()
    # 1. one driver at a time: the OLDEST running one keeps the work root (job
    # ids compared by their sequence part: qselect may spell them otherwise)
    try:
        others = [(j, s) for j, s in _drivers(cfg) if not PBS.same_job(j, me)]
    except PBS.SchedulerError as exc:
        others = []
        log("driver %s on %s: the scheduler did not list the other drivers (%s); going on"
            % (me, host, exc))
    older = _older_running(others, me)
    if older:
        log("driver %s on %s: another driver %s is running; this one exits" % (me, host, older[0]))
        return EXIT_OK
    for j, s in others:
        if s not in PBS.RUNNING_STATES:          # a younger running one exits by itself
            if _delete(j, log, "driver %s" % me):
                log("driver %s: deleted the waiting driver %s (state %s): one successor at a time"
                    % (me, j, s))
    # 2. a predecessor that left its lock was killed (walltime, node, memory)
    lock = read_lock(work_root)
    if lock is not None and lock[0].startswith("job:") and not PBS.same_job(lock[0][4:], me):
        try:
            gone = PBS.status(lock[0][4:]).finished
        except PBS.SchedulerError:
            gone = False
        if gone:
            n = _note_unclean(work_root, lock[0][4:], "ended without releasing the lock")
            log("driver %s: the previous driver %s ended without releasing the lock (killed: its "
                "walltime, its node or its memory); unclean end %d in a row"
                % (me, lock[0][4:], n))
    u = _unclean(work_root)
    if u["count"] >= DRIVER_MAX_UNCLEAN:
        log("driver %s: the last %d sequencer jobs ended uncleanly before completing a tick (%s); "
            "the chain ends here, no successor -- read their logs in %s, then submit again"
            % (me, u["count"], "; ".join("%s %s" % (e[1], e[2]) for e in u["ends"][-3:]),
               os.path.join(work_root, DRIVER_DIR)))
        return EXIT_FAILED
    # 3. the successor, at once: whatever ends this job, the chain goes on
    try:
        succ, _, succ_log = submit_driver(cfg, depend=me, entry=entry)
        log("driver %s on %s: started; its successor %s waits for it (afterany), log %s"
            % (me, host, succ, succ_log))
    except (PBS.SubmitError, OSError, ValueError) as exc:
        succ = None
        log("driver %s on %s: its successor could not be submitted (%s); retried at the end"
            % (me, host, exc))
    # 4. tick until the budget; the first full tick clears the unclean count
    ticked = {"once": False}

    def on_tick():
        if not ticked["once"]:
            ticked["once"] = True
            _clear_unclean(work_root)

    try:
        rc = run(cfg, graph=graph, poll_seconds=float(cfg.data["driver"]["poll_seconds"]),
                 retry=args.retry or (), accept_config_change=args.accept_config_change,
                 break_lock=args.break_lock, log=log,
                 until=time.time() + _driver_budget_s(cfg), on_tick=on_tick)
    except Exception as exc:     # this driver's own failure: its node, a bug
        for line in traceback.format_exc().rstrip().splitlines():
            print("    " + line, flush=True)
        n = _note_unclean(work_root, me, ("".join(traceback.format_exception_only(
            type(exc), exc)).strip())[:300])
        if succ and n < DRIVER_MAX_UNCLEAN:
            log("driver %s: the sequencer raised %s (traceback in this job's log); unclean end %d "
                "of at most %d in a row; the successor %s continues"
                % (me, type(exc).__name__, n, DRIVER_MAX_UNCLEAN, succ))
            return 1
        if succ:
            _delete(succ, log, "driver %s" % me)
        log("driver %s: the sequencer raised %s (traceback in this job's log); unclean end %d in "
            "a row; the chain ends%s -- read the traceback, then submit again"
            % (me, type(exc).__name__, n, "; its successor %s was deleted" % succ if succ else ""))
        return 1
    why, how = _END.get("why"), _END.get("how")
    if rc == EXIT_BUDGET:
        if not succ:
            try:
                succ, _, succ_log = submit_driver(cfg, depend=me, entry=entry)
            except (PBS.SubmitError, OSError, ValueError) as exc:
                log("driver %s: time budget reached and no successor could be submitted (%s); the "
                    "chain ends -- submit again from any login node" % (me, exc))
                return EXIT_FAILED
        log("driver %s: time budget reached; the successor %s continues" % (me, succ))
        return EXIT_OK
    if why == "stopped" and how and how.startswith("signal") and succ:
        n = _note_unclean(work_root, me, how)
        log("driver %s: ended by %s (PBS ending the job: its walltime, a qdel); unclean end %d in "
            "a row; the successor %s continues -- --stop stops the sequencer" % (me, how, n, succ))
        return EXIT_OK
    if succ:
        _delete(succ, log, "driver %s" % me)
    if why in ("done", "stopped"):
        _clear_unclean(work_root)
    log("driver %s: the chain ends (%s, exit %d)%s" % (me, why, rc,
                                                     "; its successor %s was deleted" % succ
                                                     if succ else ""))
    return rc


def cmd_status(args, graph=None, entry=None):
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    log = Log(None)
    state = State.load(work_root, log) if os.path.isdir(work_root) else None
    if state is None:
        log("no state file at %s: the pipeline has not run" % State.path_of(work_root))
        state = State.new(work_root, cfg.tag, cfg.path, cfg.sha256)
    else:
        try:
            age = time.time() - os.stat(State.path_of(work_root)).st_mtime
        except OSError:
            age = float("nan")
        log("state %s: started %s, updated %s (about %.0f s ago by this node's clock, by %s on %s)"
            % (State.path_of(work_root), state.data["started_utc"], state.data["updated_utc"],
               age, ("job %s" % state.data["job"]) if state.data.get("job")
               else "pid %s" % state.data.get("pid"), state.data.get("host")))
        lock = read_lock(work_root)
        log("lock: %s" % ("%s on %s" % lock if lock else "none (no sequencer ticking)"))
    if "resources" in cfg.data:
        log(cores_line(cfg, state))
    if "driver" in cfg.data:
        try:
            live = _drivers(cfg)
            log("sequencer jobs (%s): %s" % (driver_name(cfg), ", ".join(
                "%s %s" % jv for jv in live) if live else "none queued or running"))
        except PBS.SchedulerError as exc:
            log("sequencer jobs: the scheduler did not answer (%s)" % exc)
    graph = S.topological_order(graph if graph is not None else S.default_graph(cfg))
    print_table(stage_table(cfg, graph, state, live=True), log)
    for ev in state.data["events"][-10:]:
        log("event %s %s" % (ev[0], ev[1]))
    return EXIT_OK


def cmd_stop(args, graph=None, entry=None):
    """From any login node: the waiting sequencer jobs are deleted, and the
    running one stops at its next tick boundary (STOP_REQUEST) and deletes
    its successor; with --wait S, wait up to S seconds for it to end."""
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    log = Log(os.path.join(work_root, LOG_FILE) if os.path.isdir(work_root) else None)
    p = os.path.join(work_root, STOP_FILE)
    live = []
    if "driver" in cfg.data:
        try:
            live = _drivers(cfg)
        except PBS.SchedulerError as exc:
            log("stop: the scheduler did not answer (%s); try again" % exc)
            return EXIT_LOCKED
    running = [j for j, s in live if s in PBS.RUNNING_STATES]
    lock = read_lock(work_root)
    other = False                            # a sequencer that is not a job (a pid lock)
    if lock is not None and not lock[0].startswith("job:"):
        num = lock[0][4:] if lock[0].startswith("pid:") else lock[0]
        other = not (lock[1] == socket.gethostname()
                     and not _pid_alive(int(num) if num.isdigit() else -1))
    if not live and not other:
        if os.path.isfile(p):
            os.remove(p)
        log("stop: no sequencer is running")
        return EXIT_OK
    # the request first, so that a waiting job released at this moment stops too
    with open(p, "w") as fh:
        fh.write("asked %s from %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                         socket.gethostname()))
    for j, s in live:
        if j not in running and _delete(j, log, "stop"):
            log("stop: deleted the waiting sequencer job %s (state %s)" % (j, s))
    if running or other:
        log("stop: requested (%s): the running sequencer (%s) stops at its next tick boundary and "
            "deletes its successor; its jobs keep running"
            % (p, ", ".join(running) or "%s on %s" % lock))
    else:
        log("stop: requested (%s): no sequencer job was running; the request stays for one that "
            "was starting at this moment (a submission removes it)" % p)
    if not args.wait:
        return EXIT_OK
    end = time.time() + float(args.wait)
    while time.time() < end:
        time.sleep(2.0)
        try:
            still = [j for j, s in _drivers(cfg) if s in PBS.RUNNING_STATES] \
                if "driver" in cfg.data else []
        except PBS.SchedulerError:
            continue
        if not still and read_lock(work_root) is None:
            log("stop: the sequencer has stopped")
            return EXIT_OK
    log("stop: the sequencer is still running after %g s; it stops at its next tick (a "
        "sequencer started on a login node before 2026-10-18 ignores the request: "
        "run_pipeline.sh --retire-login-sequencer)" % float(args.wait))
    return EXIT_LOCKED


def cmd_retire(args, graph=None, entry=None):
    """The one-time transition of 2026-10-18 (D-098; SPEC block 1, amended
    (c)): stop a sequencer started on a login node before the amendment --
    it has no stop request and no job lock, and that login node may be out
    of reach -- from any login node, through the shared file system only.

    Such a sequencer writes its state through `state.json.tmp` after every
    tick; a directory of that name makes the write raise, so it leaves
    through its `finally`, which removes its lock. The directory stays (an
    empty folder; this code writes its state through `state.json.<pid>.tmp`
    and never uses that name), so the old code can never write again."""
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    log = Log(os.path.join(work_root, LOG_FILE) if os.path.isdir(work_root) else None)
    lock = read_lock(work_root)
    if lock is None:
        log("RETIRE: no lock in %s: no sequencer holds the work root; nothing to retire" % work_root)
        return EXIT_OK
    holder, host = lock
    if holder.startswith("job:") or holder.startswith("pid:"):
        log("RETIRE: the lock names %s on %s, a sequencer of the code of 2026-10-18 or later: "
            "nothing to retire; --stop stops it" % (holder, host))
        return EXIT_OK
    blocker = State.path_of(work_root) + ".tmp"
    for _ in range(10):                      # a file there: the old code is in mid-save
        if not os.path.exists(blocker) or os.path.isdir(blocker):
            break
        time.sleep(1.0)
    if os.path.isfile(blocker):
        # still there after 10 s: not a save in flight but one a crash left
        # behind; moved aside (kept), so that the blocker can take its name
        aside = "%s.stale-%s" % (blocker, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
        os.rename(blocker, aside)
        log("RETIRE: a temporary state file left by an earlier crash was moved aside: %s" % aside)
    if not os.path.isdir(blocker):
        try:
            os.mkdir(blocker)
        except OSError as exc:
            log("RETIRE: FAILED: cannot create %s (%s); nothing was changed; run it again"
                % (blocker, exc))
            return EXIT_FAILED

    def mtime():
        try:
            return os.stat(State.path_of(work_root)).st_mtime
        except OSError:
            return None

    m0 = mtime()
    log("RETIRE: the lock names pid %s on %s, a sequencer started on a login node before "
        "2026-10-18; the blocker %s is in place: at its next state save (after every tick) it "
        "fails there and exits, removing its lock; waiting up to %d s" % (holder, host, blocker,
                                                                         int(args.wait)))
    end = time.time() + float(args.wait)
    while time.time() < end:
        now = read_lock(work_root)
        if now != lock:
            log("RETIRE: OLD SEQUENCER STOPPED: its lock is gone (now: %s); the blocker %s stays "
                "(an empty folder the new code does not use)"
                % ("%s on %s" % now if now else "none", blocker))
            return EXIT_OK
        time.sleep(2.0)
    if mtime() != m0:
        log("RETIRE: UNEXPECTED: the state was written while the blocker was in place, so the lock's "
            "holder does not write through %s; nothing more was done -- paste this output" % blocker)
        return EXIT_FAILED
    log("RETIRE: the lock is still there after %d s and no state could be written: that sequencer "
        "was already gone (its node down?), or is inside a step longer than the wait and stops at "
        "its next save. The blocker stays. The first sequencer job takes this lock after %s s "
        "without a state write" % (int(args.wait),
                                   os.environ.get("PIPELINE_HEARTBEAT_QUIET_S", "600")))
    return EXIT_OK


def cmd_paths(args, graph=None, entry=None):
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    print(cfg.data["paths"]["work_root"])
    print(os.path.join(cfg.data["paths"]["work_root"], LOG_FILE))
    print(State.path_of(cfg.data["paths"]["work_root"]))
    print(cfg.data["envs"]["extract"])
    return EXIT_OK


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    sub.required = True

    def common(q):
        q.add_argument("--config", required=True, help="projects/<tag>/<tag>.json")
        return q

    q = common(sub.add_parser("run", help="the sequencer in this shell (tests; on a cluster "
                                           "only through submit, D-098)"))
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--poll-seconds", type=float, default=60.0)
    q.add_argument("--retry", action="append", default=[],
                   help="reset this failed stage to pending (repeatable)")
    q.add_argument("--accept-config-change", action="store_true")
    q.add_argument("--break-lock", action="store_true")
    q.add_argument("--max-ticks", type=int, default=None,
                   help="stop after this many polls (tests; resumable)")
    for verb in ("submit", "drive"):
        q = common(sub.add_parser(verb))
        q.add_argument("--retry", action="append", default=[],
                       help="reset this failed stage to pending (repeatable; the first driver only)")
        q.add_argument("--accept-config-change", action="store_true")
        q.add_argument("--break-lock", action="store_true")
    common(sub.add_parser("status"))
    q = common(sub.add_parser("stop"))
    q.add_argument("--wait", type=float, default=0.0,
                   help="wait up to this many seconds for the sequencer to stop")
    q = common(sub.add_parser("retire", help="stop a sequencer started on a login node before "
                                             "2026-10-18, from any login node (the transition)"))
    q.add_argument("--wait", type=float, default=300.0,
                   help="wait up to this many seconds for its lock to go (default 300)")
    common(sub.add_parser("paths"))
    return ap


def main(argv=None, graph=None, entry=None):
    """`graph` replaces the default stage graph and `entry` the command line
    before the verb in the sequencer jobs' argv files (the tests)."""
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "submit": cmd_submit, "drive": cmd_drive, "status": cmd_status,
            "stop": cmd_stop, "retire": cmd_retire, "paths": cmd_paths}[args.command](
                args, graph=graph, entry=entry)


if __name__ == "__main__":
    sys.exit(main())
