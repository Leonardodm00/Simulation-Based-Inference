#!/usr/bin/env python3
"""The sequencer behind the one command (specs/SPEC.md secs. 2.1-2.2, block 1).

    python3 pipeline.py run    --config projects/<tag>/<tag>.json [--dry-run]
    python3 pipeline.py status --config projects/<tag>/<tag>.json
    python3 pipeline.py stop   --config projects/<tag>/<tag>.json [--wait S]
    python3 pipeline.py paths  --config projects/<tag>/<tag>.json

`run` drives the stage graph to its end: a stage whose artefact of record
exists is skipped, a stage whose dependencies are done is started, running
stages are polled through the scheduler, and the first refusal stops the run
with its report. It is resumable: the state file records every job, and a
re-run re-attaches to them. `run_pipeline.sh` starts it under nohup.

`stop` asks the running sequencer to stop (the file <work_root>/STOP_REQUEST;
SIGTERM and SIGINT do the same): it stops at its next tick boundary -- after
the polls and the state's save, never between a submission and its record --
and its jobs keep running; a re-run re-attaches to them. `--retry <stage>`
is refused while that stage's jobs are still queued or running, since the
reset forgets them (amended 2026-10-18).

Exit codes: 0 every stage done or skipped; 2 a configuration refusal (a
schema problem, a key no built block honours, a stage not built); 3 a stage
failed (its STOP: line is the last line of the log); 4 another sequencer
holds the project's lock.

The pipeline is data-set-agnostic (D-091): nothing here names a project.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import argparse
import os
import signal
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as C          # noqa: E402
import pbs as PBS           # noqa: E402
import stages as S          # noqa: E402
from state import State     # noqa: E402

LOCK_FILE = "pipeline.lock"
LOG_FILE = "pipeline.log"
STOP_FILE = "STOP_REQUEST"

EXIT_OK, EXIT_CONFIG, EXIT_FAILED, EXIT_LOCKED = 0, 2, 3, 4

# A stop asked by SIGTERM / SIGINT while the sequencer runs: honoured at the
# next tick boundary (a second signal forces the exit at once).
_STOP = {"how": None}


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
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(work_root, log, break_lock=False):
    """Return True when this process now holds the lock."""
    p = os.path.join(work_root, LOCK_FILE)
    if os.path.isfile(p):
        with open(p, "r") as fh:
            tokens = fh.read().split()
        pid = int(tokens[0]) if tokens and tokens[0].isdigit() else -1
        host = tokens[1] if len(tokens) > 1 else "?"
        if host == socket.gethostname() and _pid_alive(pid) and pid != os.getpid():
            log("another sequencer (pid %d on %s) holds %s" % (pid, host, p))
            return False
        if host != socket.gethostname() and not break_lock:
            log("a sequencer on another host (pid %d on %s) may hold %s; "
                "re-run with --break-lock if it is gone" % (pid, host, p))
            return False
        log("stale lock from pid %d on %s replaced" % (pid, host))
    with open(p, "w") as fh:
        fh.write("%d %s %s\n" % (os.getpid(), socket.gethostname(),
                                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    return True


def release_lock(work_root):
    p = os.path.join(work_root, LOCK_FILE)
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

class Submitter(object):
    """Submits through the adapter and records every job in the state, so
    a stage never has to know the state file's layout."""

    def __init__(self, pbs_module, state, stage_name, log):
        self._pbs = pbs_module
        self._state = state
        self._name = stage_name
        self._log = log

    def submit(self, argv, role, log_path=None, cwd=None, env=None):
        job_id = self._pbs.submit(argv, cwd=cwd, env=env)
        self._state.add_job(self._name, job_id, role, argv, log=log_path)
        self._state.save()
        self._log("%s: submitted %s -> %s" % (self._name, role, job_id))
        return job_id

    def record(self, job_id, role, cmd, log_path=None):
        """A job another tool submitted (a launcher's `job : <id>` line):
        recorded like one of ours, so it is polled and resumed alike."""
        self._state.add_job(self._name, job_id, role, cmd, log=log_path)
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


def run(cfg, graph=None, poll_seconds=60.0, dry_run=False, retry=(),
        accept_config_change=False, break_lock=False, pbs_module=PBS,
        max_ticks=None, log=None):
    """Drive the graph; return an exit code (module docstring)."""
    work_root = cfg.data["paths"]["work_root"]
    graph = S.topological_order(graph if graph is not None else S.default_graph(cfg))
    log = log or Log(None if dry_run else os.path.join(work_root, LOG_FILE))

    state = State.load(work_root) if os.path.isdir(work_root) else None
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
            return EXIT_CONFIG
        state.event("configuration changed: %s -> %s"
                    % (state.data["config_sha256"][:16], cfg.sha256[:16]))
        state.data["config_sha256"] = cfg.sha256
        state.data["config_path"] = cfg.path

    for name in retry:
        if name not in [s.name for s in graph]:
            log("REFUSE: --retry names no stage: %s" % name)
            return EXIT_CONFIG
        # the reset forgets the stage's jobs: refused while any still runs,
        # or their work would be submitted a second time (amended 2026-10-18)
        try:
            live = _unfinished_jobs(state.stage(name), pbs_module)
        except PBS.SchedulerError as exc:
            log("REFUSE: --retry %s: cannot tell whether the stage's jobs still run (the "
                "scheduler did not answer: %s); try again" % (name, exc))
            return EXIT_CONFIG
        if live:
            log("REFUSE: --retry %s: the stage has %d job(s) still queued or running (%s%s); a "
                "retry forgets them and would submit their work again -- let them end (or qdel "
                "them) first" % (name, len(live), ", ".join(live[:6]),
                                 ", ..." if len(live) > 6 else ""))
            return EXIT_CONFIG
        state.reset_stage(name)
        log("%s reset to pending (--retry)" % name)

    # ---- dry run: the table, every submission, the refusals; nothing runs --
    if dry_run:
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

    # ---- the real run --------------------------------------------------------
    if not os.path.isdir(work_root):
        os.makedirs(work_root)
    if not acquire_lock(work_root, log, break_lock=break_lock):
        return EXIT_LOCKED
    old_handlers = {}
    _STOP["how"] = None
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            old_handlers[sig] = signal.signal(sig, _on_signal)
        except (ValueError, OSError):      # not the main thread: no handler, no harm
            pass
    try:
        stale = os.path.join(work_root, STOP_FILE)
        if os.path.isfile(stale):
            os.remove(stale)
            log("an earlier stop request (%s) was removed: it was for a sequencer that has "
                "already stopped" % stale)
        refusals = _refusals(cfg, graph, state)
        if refusals:
            for line in refusals:
                log(line)
            return EXIT_CONFIG
        log("sequencer started: pid %d on %s, %d stage(s), poll %.1f s"
            % (os.getpid(), socket.gethostname(), len(graph), poll_seconds))
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
                    return EXIT_FAILED
                sub = Submitter(pbs_module, state, stage.name, log)
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
                        return EXIT_CONFIG
                    except Exception as exc:  # a refusal before submission
                        state.set_status(stage.name, "failed", str(exc))
                        state.save()
                        log("STOP: stage %s refused: %s" % (stage.name, exc))
                        return EXIT_FAILED
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
                    return EXIT_FAILED
            statuses = state.statuses()
            if all(statuses.get(s.name) in ("done", "skipped") for s in graph):
                log("PIPELINE DONE: every stage done or skipped")
                return EXIT_OK
            if max_ticks is not None and ticks >= max_ticks:
                log("tick limit %d reached; stopping (resumable)" % max_ticks)
                return EXIT_OK
            # a stop is honoured here only: after the polls and the state's save
            how = _stop_requested(work_root)
            if how is None:
                _sleep(poll_seconds, work_root)
                how = _stop_requested(work_root)
            if how is not None:
                n_live = sum(1 for s_ in graph for j in state.stage(s_.name).get("jobs", [])
                             if not j.get("finished"))
                log("stopped on request (%s); resumable: %d job(s) still queued or running "
                    "keep running" % (how, n_live))
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


def cmd_run(args):
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    return run(cfg, poll_seconds=args.poll_seconds, dry_run=args.dry_run,
               retry=args.retry or (), accept_config_change=args.accept_config_change,
               break_lock=args.break_lock, max_ticks=args.max_ticks)


def cmd_status(args):
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    log = Log(None)
    state = State.load(work_root) if os.path.isdir(work_root) else None
    if state is None:
        log("no state file at %s: the pipeline has not run" % State.path_of(work_root))
        state = State.new(work_root, cfg.tag, cfg.path, cfg.sha256)
    else:
        log("state %s: started %s, updated %s, pid %s on %s"
            % (State.path_of(work_root), state.data["started_utc"],
               state.data["updated_utc"], state.data["pid"], state.data["host"]))
        lock = os.path.join(work_root, LOCK_FILE)
        if os.path.isfile(lock):
            with open(lock) as fh:
                log("lock: %s" % fh.read().strip())
        else:
            log("lock: none (no sequencer running)")
    graph = S.topological_order(S.default_graph(cfg))
    print_table(stage_table(cfg, graph, state, live=True), log)
    for ev in state.data["events"][-10:]:
        log("event %s %s" % (ev[0], ev[1]))
    return EXIT_OK


def cmd_stop(args):
    """Ask the running sequencer to stop at its next tick boundary; with
    --wait S, wait up to S seconds for its lock to go."""
    cfg = _load(args.config)
    if cfg is None:
        return EXIT_CONFIG
    work_root = cfg.data["paths"]["work_root"]
    log = Log(os.path.join(work_root, LOG_FILE) if os.path.isdir(work_root) else None)
    lock = os.path.join(work_root, LOCK_FILE)
    if not os.path.isfile(lock):
        log("stop: no sequencer is running (no lock at %s); nothing to stop" % lock)
        return EXIT_OK
    with open(lock) as fh:
        tokens = fh.read().split()
    pid = int(tokens[0]) if tokens and tokens[0].isdigit() else -1
    host = tokens[1] if len(tokens) > 1 else "?"
    if host == socket.gethostname() and not _pid_alive(pid):
        log("stop: the lock names pid %d on %s, which is not running; nothing to stop (the next "
            "start replaces the lock)" % (pid, host))
        return EXIT_OK
    p = os.path.join(work_root, STOP_FILE)
    with open(p, "w") as fh:
        fh.write("asked %s by pid %d on %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                 os.getpid(), socket.gethostname()))
    log("stop: requested (%s): the sequencer (pid %d on %s) stops at its next tick boundary; "
        "its jobs keep running" % (p, pid, host))
    if not args.wait:
        return EXIT_OK
    end = time.time() + float(args.wait)
    while time.time() < end:
        if not os.path.isfile(lock):
            log("stop: the sequencer has stopped (its lock is gone)")
            return EXIT_OK
        time.sleep(1.0)
    log("stop: the sequencer still holds %s after %g s (it stops at its next tick, within "
        "--poll-seconds of its last poll; a sequencer started before 2026-10-18 ignores the "
        "request)" % (lock, float(args.wait)))
    return EXIT_LOCKED


def cmd_paths(args):
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

    q = common(sub.add_parser("run"))
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--poll-seconds", type=float, default=60.0)
    q.add_argument("--retry", action="append", default=[],
                   help="reset this failed stage to pending (repeatable)")
    q.add_argument("--accept-config-change", action="store_true")
    q.add_argument("--break-lock", action="store_true")
    q.add_argument("--max-ticks", type=int, default=None,
                   help="stop after this many polls (tests; resumable)")
    common(sub.add_parser("status"))
    q = common(sub.add_parser("stop"))
    q.add_argument("--wait", type=float, default=0.0,
                   help="wait up to this many seconds for the sequencer to stop")
    common(sub.add_parser("paths"))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "status": cmd_status, "stop": cmd_stop,
            "paths": cmd_paths}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
