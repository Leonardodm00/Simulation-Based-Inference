"""Stages S4-S6, the bank plan and the two banks (block 2), through
Sbi-extractor's committed `export_windows/launch_bank.sh` (D-062: the
pipeline drives the tool, it does not change it; D-092: the tool's rules
stand).

    bank_plan (S4)   writes the bank profile from the configuration and the
                     digests of S1-S3's artefacts, then `launch_bank.sh plan`
                     on the login node (`envs.extract`); pass line
                     `BANK PLAN OK`; artefact `<sbix_dir>/export_windows/
                     <tag>/bank_plan.json` + .sha256 (the tool writes both)
    sim_bank (S5)    the chain line BC of the first data set, as phases:
                     `test` (one task as a job; skipped when its marker
                     belongs to this plan), passed when its completion
                     marker `tasks/task_0000.json` is this plan's (its log
                     line `TASK <i> OK` is reported with it); `array` with
                     WALLTIME from `bank.walltime` (`auto` = the launcher's
                     rule from the test task's pace) -- the array and its
                     gate job, passed when `BANK_RECORD.json` + .sha256
                     exist (the gate's `wrote ...` line reported with it);
                     artefact `BANK_RECORD.json` + .sha256
    real_bank (S6)   `launch_bank.sh real` on the login node; pass line
                     `REAL BANK OK`; artefact `shard_0000.npz` + .json

The launcher submits the jobs itself and prints `  job   : <id>` and
`  gate  : <id>`; the stages record those ids through the sequencer's
Submitter so they are polled and resumed like any other job. The
launcher's refusals are its `ABORT:` / `REFUSED:` lines, read back from
its output.

A job's success is read from the files the job itself writes (the task's
marker, the gate's record), never from its PBS log alone: the scheduler
delivers the `-o` file when the job ends, a site epilogue may append lines
after the pass line, and a file written on a compute node may reach the
login node's view a little later. The log gives the failure's text. When a
job has finished with neither its file nor its script's own end line
(`[job] <name>_exit=<rc>`) in the log, the stage waits up to
`PIPELINE_LOG_GRACE_S` seconds (default 300) before deciding (2026-10-16).

A gate can end without ever running: on davinci (2026-10-16 12:29) the
gate behind the array finished with no log, no record and no exit status.
The gate's log counts only when it is fresh (absent, or of another size or
mtime, when the gate was submitted); a finished gate with no record and no
fresh log after its grace period never ran, and the stage runs the gate
alone (`launch_bank.sh gate`, no dependency) once the array has finished --
once; a second gate that never runs fails the stage (specs/SPEC.md, block 2).

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import hashlib
import os
import re
import time

from . import DEPS, Stage, artefact_of, artefact_done
from . import common as CM

_JOB_LINE = re.compile(r"^\s*(job|gate)\s*:\s*(\S+)\s*$", re.M)


def _sha256_prefix(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _last_lines(path, n=5):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as fh:
        text = fh.read()[-20000:].decode("ascii", "replace")
    return "\n".join(text.rstrip("\n").split("\n")[-n:])


def _last_line(path):
    return _last_lines(path, 1)


def read_log(path, limit=400000):
    """The last `limit` bytes of a job log ('' when absent)."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - int(limit)))
        return fh.read().decode("ascii", "replace")


def job_exit_line(text, name):
    """The rc of the job script's own end line `[job] <name>_exit=<rc>`
    (the Sbi-extractor job scripts print it last), or None when absent."""
    found = re.findall(r"(?:^|\n)(?:\[job\] )?%s_exit=(-?\d+)" % re.escape(name), text)
    return int(found[-1]) if found else None


def file_stat(path):
    """[size, mtime] of a file (a list, as the state file keeps it), or None."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [int(st.st_size), float(st.st_mtime)]


def log_is_fresh(path, before):
    """True when the log at `path` was written after a stat `before` taken
    at its job's submission: absent then and present now, or present then
    with another size or mtime. `before` is the string "unknown" for a state
    written before the stat was kept: any log counts then."""
    now = file_stat(path)
    if now is None:
        return False
    if before == "unknown":
        return True
    return before is None or list(before) != now


def grace_over(sub, job_id):
    """True once PIPELINE_LOG_GRACE_S seconds have passed since the stage
    first saw `job_id` finished (the time is kept in the stage's state, so
    a restarted sequencer does not wait again from zero)."""
    grace = float(os.environ.get("PIPELINE_LOG_GRACE_S", "300"))
    seen = sub.setdefault("seen_finished", {})
    now = time.time()
    t0 = float(seen.setdefault(job_id, now))
    return (now - t0) >= grace


class _BankBase(Stage):
    """What the three stages share: the launcher's location, the profile,
    the environment and a login-node call."""

    max_resubmits = 0

    def _launcher_dir(self, cfg):
        return os.path.join(cfg.data["paths"]["sbix_dir"], "export_windows")

    def _profile_path(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"],
                            "bank_profile_%s.json" % cfg.data["dataset"])

    def _work_dir(self, cfg):
        """The launcher's work folder: beside the plan."""
        return os.path.dirname(artefact_of("bank_plan", cfg)[1])

    def _log_dir(self, cfg):
        return os.path.join(self._work_dir(cfg), "logs")

    def _login_log(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "logs", "bank_login_node.log")

    def _env(self, cfg, **extra):
        d = cfg.data
        env = dict(os.environ)
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "PROFILE", "QUEUE", "CONCURRENCY"):
            env.pop(k, None)
        env.update({"PROFILE": self._profile_path(cfg), "ENV_NAME": d["envs"]["extract"],
                    "QUEUE": d["bank"]["queue"], "CONCURRENCY": str(int(d["bank"]["concurrency"])),
                    # the DSN tree the Sbi-extractor tools import (env.sh defaults it
                    # to artifacts/sbi_hpc, a per-machine symlink): the configuration knows it
                    "SBI_HPC_DIR": os.path.join(d["paths"]["sbi_dir"], "hpc")})
        env.update(extra)
        return env

    def ensure_profile(self, cfg, pbs):
        """The profile the launcher reads, (re)written from the configuration
        before any launcher call -- whichever stage runs first. When the plan
        stage was skipped (its artefact exists, as for a data set whose plan
        is of record) no earlier stage wrote it. Idempotent: the same
        configuration and artefacts give the same file."""
        path = self._profile_path(cfg)
        before = CM.read_json(path) if os.path.isfile(path) else None
        prof = BankPlanStage().write_profile(cfg)
        after = dict((k, v) for k, v in prof.items() if not k.startswith("_"))
        if before is not None:
            before = dict((k, v) for k, v in before.items() if not k.startswith("_"))
            if before != after:
                changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
                pbs.log("%s: profile %s rewritten; changed: %s" % (self.name, path, ", ".join(changed)))
        else:
            pbs.log("%s: profile %s written (record %s, freeze %s, manifest %s)"
                    % (self.name, path, prof["record_sha256_prefix"], prof["freeze_sha256_prefix"],
                       prof["cohort_manifest_sha256_prefix"]))
        return prof

    def _run_launcher(self, cfg, mode, log_path, **extra):
        """`bash launch_bank.sh <mode>` in envs.extract from export_windows/;
        returns (rc, text)."""
        argv = ["bash", os.path.join(self._launcher_dir(cfg), "launch_bank.sh"), mode]
        return CM.E.run_in_env(cfg.data["envs"]["extract"], argv, cwd=self._launcher_dir(cfg),
                               log_path=log_path, env=self._env(cfg, **extra))

    def _refusal(self, text):
        lines = [l for l in text.split("\n") if l.startswith(("ABORT", "REFUSED", "STOP", "ERROR"))]
        return "; ".join(lines) if lines else text.strip().split("\n")[-1] if text.strip() else "(no output)"

    def verify(self, cfg, st):
        ok, why = self.is_done(cfg)
        if ok:
            return True, why
        fail = st["sub"].get("fail")
        if fail:
            return False, fail
        return False, "killed: " + why

    def on_killed(self, cfg, st, pbs):
        return False


# ---------------------------------------------------------------------------
# S4: the plan
# ---------------------------------------------------------------------------

class BankPlanStage(_BankBase):
    name = "bank_plan"
    deps = DEPS["bank_plan"]

    def write_profile(self, cfg):
        """The bank profile (`bank_plan_config.BankPlanConfig`'s keys) from
        the configuration's `bank` block and S1-S3's artefacts, whose
        16-hex digest prefixes the profile pins."""
        d = cfg.data
        tag = d["dataset"]
        sbix = d["paths"]["sbix_dir"]
        record = artefact_of("sim_redetection", cfg)[1]
        freeze = artefact_of("label_freeze", cfg)[1]
        frozen = os.path.join(sbix, "artifacts", "cohort_manifest_%s" % d["real"]["cohort_tag"],
                              "cohort_manifest.json")
        manifest = frozen if os.path.isfile(frozen) else artefact_of("real_extraction", cfg)[1]
        for p in (record, freeze, manifest):
            if not os.path.isfile(p):
                raise RuntimeError("bank_plan: input missing: %s" % p)
        b = d["bank"]
        prof = {
            "_comment": "Written by the pipeline (hpc/pipeline/stages/bank.py) from %s; the "
                        "digests are those of the artefacts of record of S1-S3." % cfg.path,
            "record_path": record, "record_sha256_prefix": _sha256_prefix(record),
            "freeze_path": freeze, "freeze_sha256_prefix": _sha256_prefix(freeze),
            "cohort_manifest_path": manifest,
            "cohort_manifest_sha256_prefix": _sha256_prefix(manifest),
            "sim_dir": d["sim"]["sim_main"],
            "on_axis": b["on_axis"], "on_declared_box": list(b["on_declared_box"]),
            "topology_axes": list(b["topology_axes"]), "expected_p": int(b["expected_p"]),
            "t_win_s": float(b["t_win_s"]), "n_e": int(b["n_e"]), "floor_hz": float(b["floor_hz"]),
            "floor_rtol": float(b["floor_rtol"]), "sample_seed": int(b["sample_seed"]),
            "bootstrap_resamples": int(b["bootstrap_resamples"]),
            "bootstrap_seed": int(b["bootstrap_seed"]),
            "bytes_sample_rows": int(b["bytes_sample_rows"]),
            "rows_per_shard": int(b["rows_per_shard"]),
            "bank_root": os.path.join(d["paths"]["banks_root"], tag + "_sim"),
            "real_bank_root": os.path.join(d["paths"]["banks_root"], tag + "_real"),
            "plan_out": artefact_of("bank_plan", cfg)[1],
        }
        CM.write_json(self._profile_path(cfg), prof)
        return prof

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        self.ensure_profile(cfg, pbs)
        rc, text = self._run_launcher(cfg, "plan", self._login_log(cfg))
        sub["plan_rc"] = rc
        if rc != 0 or "BANK PLAN OK" not in text:
            sub["fail"] = "the bank plan was refused (exit %d): %s" % (rc, self._refusal(text))
            return
        for line in text.split("\n"):
            if line.startswith("[plan]") or "BANK PLAN OK" in line:
                pbs.log("bank_plan: " + line.strip())
        sub["pass_line"] = "BANK PLAN OK"

    def poll(self, cfg, st, pbs):
        return "finished"

    def dry_run(self, cfg, st):
        return ["profile -> %s (from the configuration's bank block and the digests of S1-S3's "
                "artefacts)" % self._profile_path(cfg),
                "login node (envs.extract): PROFILE=<profile> ENV_NAME=%s bash %s/launch_bank.sh plan"
                % (cfg.data["envs"]["extract"], self._launcher_dir(cfg)),
                "pass line: BANK PLAN OK; artefact %s + .sha256" % artefact_of("bank_plan", cfg)[1]]


# ---------------------------------------------------------------------------
# S5: the simulated bank (the chain line BC)
# ---------------------------------------------------------------------------

class SimBankStage(_BankBase):
    name = "sim_bank"
    deps = DEPS["sim_bank"]

    def _bank_root(self, cfg):
        return os.path.join(cfg.data["paths"]["banks_root"], cfg.data["dataset"] + "_sim")

    def _test_marker_ok(self, cfg):
        """The test task's marker exists and belongs to this plan."""
        m = os.path.join(self._bank_root(cfg), "tasks", "task_0000.json")
        plan = artefact_of("bank_plan", cfg)[1]
        if not (os.path.isfile(m) and os.path.isfile(plan)):
            return False
        try:
            rec = CM.read_json(m)
        except ValueError:
            return False
        h = hashlib.sha256()
        with open(plan, "rb") as fh:
            h.update(fh.read())
        return rec.get("plan_sha256") == h.hexdigest() and bool(rec.get("finished"))

    def _jobs_of(self, text):
        return dict((k, v) for k, v in _JOB_LINE.findall(text))

    def _start_test(self, cfg, st, pbs):
        sub = st["sub"]
        log_path = os.path.join(self._log_dir(cfg), "bank_test_0.log")
        rc, text = self._run_launcher(cfg, "test", self._login_log(cfg), TEST_INDEX="0")
        jobs = self._jobs_of(text)
        if rc != 0 or "job" not in jobs:
            sub["fail"] = "the test task could not be submitted (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        pbs.record(jobs["job"], "bank test task 0", ["launch_bank.sh", "test"], log_path=log_path)
        sub["test_job"] = jobs["job"]
        sub["test_log"] = log_path
        sub["phase"] = "test"

    def _start_array(self, cfg, st, pbs):
        sub = st["sub"]
        gate_log = os.path.join(self._log_dir(cfg), "bank_gate.log")
        before = file_stat(gate_log)
        rc, text = self._run_launcher(cfg, "array", self._login_log(cfg),
                                      WALLTIME=str(cfg.data["bank"]["walltime"]), TEST_INDEX="0")
        jobs = self._jobs_of(text)
        if rc != 0 or "job" not in jobs or "gate" not in jobs:
            sub["fail"] = "the array could not be submitted (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        for line in text.split("\n"):
            if line.startswith("[launch] WALLTIME=auto") or line.startswith("# tasks"):
                pbs.log("sim_bank: " + line.strip())
        pbs.record(jobs["job"], "bank array", ["launch_bank.sh", "array"],
                   log_path=os.path.join(self._log_dir(cfg), "bank_task_<index>.log"))
        pbs.record(jobs["gate"], "bank gate", ["launch_bank.sh", "array"], log_path=gate_log)
        sub["array_job"] = jobs["job"]
        sub["gate_job"] = jobs["gate"]
        sub["gate_log"] = gate_log
        sub["gate_log_stat"] = before
        sub["phase"] = "gate"

    def _start_gate_alone(self, cfg, st, pbs):
        """`launch_bank.sh gate`: the gate with no dependency, recorded like
        the first; the phase stays `gate`."""
        sub = st["sub"]
        gate_log = os.path.join(self._log_dir(cfg), "bank_gate.log")
        before = file_stat(gate_log)
        rc, text = self._run_launcher(cfg, "gate", self._login_log(cfg))
        jobs = self._jobs_of(text)
        if rc != 0 or "gate" not in jobs:
            sub["fail"] = "the gate alone could not be submitted (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        pbs.record(jobs["gate"], "bank gate alone", ["launch_bank.sh", "gate"], log_path=gate_log)
        sub["gate_job"] = jobs["gate"]
        sub["gate_log"] = gate_log
        sub["gate_log_stat"] = before

    def _gate_never_ran(self, cfg, st, pbs, finished, by_id):
        """The finished gate left no record and no fresh log: it never ran.
        Once the array has finished, run the gate alone -- once."""
        sub = st["sub"]
        job_id = sub["gate_job"]
        array_id = sub.get("array_job")
        if array_id and not finished.get(array_id):
            return "running"
        unrun = list(sub.get("gates_unrun", [])) + [job_id]
        sub["gates_unrun"] = unrun
        exit_status = by_id[job_id].get("exit_status")
        if len(unrun) > 1:
            sub["fail"] = ("the gate did not run (jobs %s finished with no record and no log at %s; "
                           "exit %s): read qstat -x -f %s" % (", ".join(unrun), sub["gate_log"],
                                                               exit_status, job_id))
            sub["phase"] = "failed"
            return "finished"
        pbs.log("sim_bank: the gate %s never ran (finished with exit %s, no record, no log at %s; "
                "the array %s has ended): the gate alone" % (job_id, exit_status, sub["gate_log"],
                                                             array_id or "-"))
        self._start_gate_alone(cfg, st, pbs)
        return "running" if sub.get("phase") == "gate" else "finished"

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        self.ensure_profile(cfg, pbs)
        if self._test_marker_ok(cfg):
            pbs.log("sim_bank: the test task's marker belongs to this plan: test skipped")
            sub["test_skipped"] = True
            self._start_array(cfg, st, pbs)
        else:
            self._start_test(cfg, st, pbs)

    def poll(self, cfg, st, pbs):
        sub = st["sub"]
        finished = {}
        for job in st["jobs"]:
            if not job.get("finished"):
                js = pbs.status(job["id"])
                if js.finished:
                    job["finished"] = True
                    job["exit_status"] = js.exit_status
                    job["walltime_s"] = js.walltime_s
                    job["state"] = js.state
            finished[job["id"]] = bool(job.get("finished"))
        by_id = dict((j["id"], j) for j in st["jobs"])
        phase = sub.get("phase")
        if phase == "test":
            job_id = sub["test_job"]
            if not finished.get(job_id):
                return "running"
            text = read_log(sub["test_log"])
            ok_lines = re.findall(r"(?m)^TASK \d+ OK.*$", text)
            if self._test_marker_ok(cfg):
                pbs.log("sim_bank: test task passed: %s (its marker is this plan's)"
                        % (ok_lines[-1].strip() if ok_lines else "no log line yet"))
                self._start_array(cfg, st, pbs)
                return "running" if sub.get("phase") == "gate" else "finished"
            # no marker of this plan: the task failed -- unless the marker or
            # the log is still on its way (the log says OK, or it lacks the
            # job script's end line): wait, at most the grace period
            ended = job_exit_line(text, "bank_task")
            if (ok_lines or ended is None) and not grace_over(sub, job_id):
                return "running"
            refused = [l for l in text.split("\n") if l.startswith(("REFUSED", "ABORT", "Traceback"))]
            sub["fail"] = ("the test task did not pass (job %s, exit %s, no marker of this plan in %s; "
                           "log %s: %s)"
                           % (job_id, by_id[job_id].get("exit_status"),
                              os.path.join(self._bank_root(cfg), "tasks"), sub["test_log"],
                              "; ".join(refused[:5]) if refused
                              else (_last_lines(sub["test_log"], 3) or "absent")))
            sub["phase"] = "failed"
            return "finished"
        if phase == "gate":
            job_id = sub["gate_job"]
            if not finished.get(job_id):
                return "running"
            # only this gate's log counts: a log left by an earlier gate is not read
            fresh = log_is_fresh(sub["gate_log"], sub.get("gate_log_stat", "unknown"))
            text = read_log(sub["gate_log"]) if fresh else ""
            wrote = [l for l in text.split("\n")
                     if l.startswith("wrote ") and "BANK_RECORD.json" in l]
            done, why = self.is_done(cfg)
            if done:
                pbs.log("sim_bank: " + (wrote[-1].strip() if wrote else why))
                sub["phase"] = "complete"
                return "finished"
            ended = job_exit_line(text, "bank_gate")
            if (wrote or ended is None) and not grace_over(sub, job_id):
                return "running"
            if not fresh:
                return self._gate_never_ran(cfg, st, pbs, finished, by_id)
            fails = [l for l in text.split("\n") if l.startswith("[gate] FAIL") or
                     l.startswith("REFUSED")]
            sub["fail"] = ("the gate refused (job %s, exit %s; %s): %s"
                           % (job_id, by_id[job_id].get("exit_status"), sub["gate_log"],
                              "; ".join(fails[:20]) if fails
                              else (_last_lines(sub["gate_log"], 3) or "no log")))
            sub["phase"] = "failed"
            return "finished"
        return "finished"

    def dry_run(self, cfg, st):
        d = cfg.data
        ld = self._launcher_dir(cfg)
        return ["test (skipped when %s/tasks/task_0000.json belongs to the plan): PROFILE=<profile> "
                "QUEUE=%s TEST_INDEX=0 bash %s/launch_bank.sh test -> one job; passed by its marker "
                "(log line TASK 0 OK)" % (self._bank_root(cfg), d["bank"]["queue"], ld),
                "array: WALLTIME=%s CONCURRENCY=%d QUEUE=%s bash %s/launch_bank.sh array -> the array "
                "(-J 0-<n-1>%%%d) and its gate job (depend=afterok)"
                % (d["bank"]["walltime"], int(d["bank"]["concurrency"]), d["bank"]["queue"], ld,
                   int(d["bank"]["concurrency"])),
                "a gate that never runs (no record, no log of its own): bash %s/launch_bank.sh gate "
                "-> the gate alone, once the array has ended" % ld,
                "pass line: the gate's log ends 'wrote %s/BANK_RECORD.json  sha256 <16 hex>'; artefact "
                "BANK_RECORD.json + .sha256" % self._bank_root(cfg)]


# ---------------------------------------------------------------------------
# S6: the real bank
# ---------------------------------------------------------------------------

class RealBankStage(_BankBase):
    name = "real_bank"
    deps = DEPS["real_bank"]

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        self.ensure_profile(cfg, pbs)
        rc, text = self._run_launcher(cfg, "real", self._login_log(cfg))
        if rc != 0 or "REAL BANK OK" not in text:
            sub["fail"] = "the real bank was refused (exit %d): %s" % (rc, self._refusal(text))
            return
        for line in text.split("\n"):
            if line.startswith("[real]") or "REAL BANK OK" in line or line.startswith("wrote "):
                pbs.log("real_bank: " + line.strip())
        sub["pass_line"] = "REAL BANK OK"

    def poll(self, cfg, st, pbs):
        return "finished"

    def dry_run(self, cfg, st):
        kind, path = artefact_of("real_bank", cfg)
        return ["login node (envs.extract): PROFILE=<profile> bash %s/launch_bank.sh real"
                % self._launcher_dir(cfg),
                "pass line: REAL BANK OK; artefact %s + .json" % path]
