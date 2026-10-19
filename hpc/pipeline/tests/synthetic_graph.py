#!/usr/bin/env python3
"""A three-stage synthetic graph A -> B -> C for the sequencer's smoke test.

Each stage submits one job through the scheduler adapter (the fake PBS in
the tests): a bash script that writes the stage's artefact of record,
`<work_root>/artefacts/<name>.txt` with its .sha256 sidecar, and prints
`STAGE <name> OK`. Switches make a stage misbehave the way a real runner
can:

    --refuse NAME        the job prints `REFUSED: ...` and exits 4 (a refusal)
    --silent-kill NAME   the job exits 0 with no artefact and no refusal (a
                         kill, as the sequencer must read it)
    --pre-make NAME      the artefact exists before the run (a skip)

The rest of the command line is the sequencer's (`--config`, `--dry-run`,
`--poll-seconds`, `--retry`, `--accept-config-change`, `--max-ticks`).

    synthetic_graph.py submit|drive|stop|status|retire --config ...

runs the pipeline's own command (pipeline.main) on this graph, the
sequencer's jobs running this file again (block 1 amended 2026-10-18 (c):
the driver chain under the fake PBS). There the stages' behaviour comes
from the environment, which the fake PBS hands to its jobs:
SYNTH_REFUSE=A,B, SYNTH_SILENT_KILL=..., SYNTH_CRASH_POLL=... (that stage's
poll raises: a sequencer that crashes), SYNTH_CRASH_ONCE=<path> (the first
poll of any stage raises once, the file <path> marking that it has: a
first driver that crashes in its first tick), SYNTH_JOB_SLEEP=<s> (each job's
sleep, default 0.3), MUTATION_PATCH=<python> (exec'd before the command:
block 1b's mutations). For the core cap (D-099, block 1c's test), in every
mode: SYNTH_PARALLEL=1 makes A and B independent (C after both),
SYNTH_STAGE_NCPUS=<n> makes each stage's job ask n cores and its start
wait (S.Deferred) while they do not fit, SYNTH_NO_CHECK=1 submits without
asking (the sequencer's guard must refuse it), SYNTH_POLL_ASK=<n> makes
every poll ask whether n cores fit (a job that can never start, asked in
a poll).
Not pipeline code: a fixture. Pure ASCII, LF only; stdlib only.
"""

import argparse
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_PIPE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C      # noqa: E402
import pbs as PBS       # noqa: E402
import pipeline as P    # noqa: E402
import stages as S      # noqa: E402
import state as ST      # noqa: E402

VERBS = ("submit", "drive", "stop", "status", "retire")


class SyntheticStage(S.Stage):
    max_resubmits = 1

    def __init__(self, name, deps, mode):
        self.name = name
        self.deps = tuple(deps)
        self.mode = mode            # "ok" | "refuse" | "silent-kill"

    def _artefact(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "artefacts", self.name + ".txt")

    def _log(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "logs", self.name + ".log")

    def is_done(self, cfg):
        return S.artefact_done("sha256", self._artefact(cfg))

    def _script(self, cfg):
        art = self._artefact(cfg)
        d = os.path.join(cfg.data["paths"]["work_root"], "jobs")
        if not os.path.isdir(d):
            os.makedirs(d)
        p = os.path.join(d, self.name + ".sh")
        if self.mode == "refuse":
            body = "echo 'REFUSED: synthetic refusal of %s'\nexit 4\n" % self.name
        elif self.mode == "silent-kill":
            body = "echo 'the job started'\nexit 0\n"
        else:
            body = ("sleep %s\nmkdir -p '%s'\nprintf 'hello %s\\n' > '%s'\n"
                    "sha256sum '%s' | cut -d' ' -f1 > '%s.sha256'\n"
                    "echo 'STAGE %s OK'\n"
                    % (os.environ.get("SYNTH_JOB_SLEEP", "0.3"), os.path.dirname(art),
                       self.name, art, art, art, self.name))
        with open(p, "w") as fh:
            fh.write("#!/bin/bash\nset -u\n" + body)
        return p

    def _qsub(self, cfg):
        n = int(os.environ.get("SYNTH_STAGE_NCPUS", "1"))
        return ["qsub", "-N", "synth_" + self.name, "-q", "cpu",
                "-l", "select=1:ncpus=%d:mem=1gb" % n, "-l", "walltime=00:10:00",
                "-j", "oe", "-o", self._log(cfg), "-v", "STAGE=%s" % self.name,
                self._script(cfg)]

    def start(self, cfg, st, sub):
        d = os.path.dirname(self._log(cfg))
        if not os.path.isdir(d):
            os.makedirs(d)
        n = int(os.environ.get("SYNTH_STAGE_NCPUS", "1"))
        if os.environ.get("SYNTH_NO_CHECK") != "1" and not sub.fits(n):
            raise S.Deferred("job asks %d cores; %s free" % (n, sub.free()))
        sub.submit(self._qsub(cfg), role="job", log_path=self._log(cfg))

    def verify(self, cfg, st):
        ok, why = self.is_done(cfg)
        if ok:
            return True, why
        log = self._log(cfg)
        if os.path.isfile(log):
            with open(log) as fh:
                for line in fh:
                    if line.startswith("REFUSED:"):
                        return False, line.strip()
        return False, "killed: no artefact and no refusal line in %s" % log

    def dry_run(self, cfg, st):
        return [" ".join(self._qsub(cfg))]

    def poll(self, cfg, st, sub):
        if self.name in _env_list("SYNTH_CRASH_POLL"):
            raise RuntimeError("synthetic crash in the poll of %s" % self.name)
        once = os.environ.get("SYNTH_CRASH_ONCE")
        if once and not os.path.exists(once):
            open(once, "w").close()
            raise RuntimeError("synthetic crash, once, in the poll of %s" % self.name)
        if os.environ.get("SYNTH_POLL_ASK"):
            sub.fits(int(os.environ["SYNTH_POLL_ASK"]))
        return S.Stage.poll(self, cfg, st, sub)


def _env_list(var):
    return [x for x in os.environ.get(var, "").split(",") if x]


def _deps():
    """A -> B -> C, or with SYNTH_PARALLEL=1 A and B independent, C after both."""
    if os.environ.get("SYNTH_PARALLEL") == "1":
        return {"A": (), "B": (), "C": ("A", "B")}
    return {"A": (), "B": ("A",), "C": ("B",)}


def env_graph():
    """A -> B -> C with each stage's behaviour from the environment."""
    def mode(name):
        if name in _env_list("SYNTH_REFUSE"):
            return "refuse"
        if name in _env_list("SYNTH_SILENT_KILL"):
            return "silent-kill"
        return "ok"
    deps = _deps()
    return [SyntheticStage(n, deps[n], mode(n)) for n in ("A", "B", "C")]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    patch = os.environ.get("MUTATION_PATCH")
    if patch:
        exec(patch, {"P": P, "PBS": PBS, "ST": ST, "S": S, "C": C})
        sys.stderr.write("MUTATION_PATCH applied\n")
    if argv and argv[0] in VERBS:
        return P.main(argv, graph=env_graph(),
                      entry=[sys.executable, os.path.abspath(__file__)])
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=0.2)
    ap.add_argument("--retry", action="append", default=[])
    ap.add_argument("--accept-config-change", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--refuse", action="append", default=[])
    ap.add_argument("--silent-kill", action="append", default=[])
    ap.add_argument("--pre-make", action="append", default=[])
    args = ap.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print("REFUSE: the configuration has %d problem(s):" % len(exc.problems))
        for p in exc.problems:
            print("  " + p)
        return P.EXIT_CONFIG

    def mode(name):
        if name in args.refuse:
            return "refuse"
        if name in args.silent_kill:
            return "silent-kill"
        return "ok"

    deps = _deps()
    graph = [SyntheticStage(n, deps[n], mode(n)) for n in ("A", "B", "C")]
    for name in args.pre_make:
        st = [s for s in graph if s.name == name][0]
        art = st._artefact(cfg)
        if not os.path.isdir(os.path.dirname(art)):
            os.makedirs(os.path.dirname(art))
        with open(art, "wb") as fh:
            fh.write(b"pre-made\n")
        with open(art + ".sha256", "w") as fh:
            fh.write(hashlib.sha256(b"pre-made\n").hexdigest() + "  " + art + "\n")
    return P.run(cfg, graph=graph, poll_seconds=args.poll_seconds, dry_run=args.dry_run,
                 retry=args.retry, accept_config_change=args.accept_config_change,
                 max_ticks=args.max_ticks)


if __name__ == "__main__":
    sys.exit(main())
