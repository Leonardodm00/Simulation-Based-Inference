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
                     gate job; then a verification JOB (tools/bank_verify.py:
                     every shard's sha256 and every zip member's CRC); bad
                     tasks are moved to a quarantine folder (never deleted)
                     and the array runs again, at most bank.max_rerun_rounds
                     times (D-094, D-095); artefact `BANK_RECORD.json` +
                     .sha256 and `BANK_VERIFIED.json` + .sha256
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
gate behind the array finished with no log, no record and no exit status,
because one element had failed to start on a node without its home file
system. The gate's log counts only when it is fresh (absent, or of another
size or mtime, when the gate was submitted); a finished gate with no record
and no fresh log after its grace period never ran. Whatever the gate did,
the round goes on to the verification job; a verified round whose gate
never ran gets the gate alone (`launch_bank.sh gate`, no dependency), once
(specs/SPEC.md, block 2). The same node had finished two other tasks whose
shards were not zip files though their markers existed: hence the job that
re-reads every shard from another node (2026-10-18).

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import hashlib
import os
import re
import time

from . import DEPS, Stage, artefact_of, artefact_done, sha256_sidecar_ok
from . import common as CM

_JOB_LINE = re.compile(r"^\s*(job|gate)\s*:\s*(\S+)\s*$", re.M)
_WALLTIME_USED = re.compile(r"->\s*([0-9]+:[0-5][0-9]:[0-5][0-9])")
MAX_VERIFY_TRIES = 3          # verification jobs per round before a STOP


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """S5, the simulated bank, as phases (specs/SPEC.md block 2; the
    self-healing rules D-094, D-095):

        test        one task as a job (skipped when its marker is this plan's)
        gate        round r: `launch_bank.sh array` -- the array and its gate --
                    until the gate has ended (passed, refused, never ran) and
                    the array too
        verify      the verification job (tools/bank_verify.py, every shard's
                    sha256 and every zip member's CRC); its report decides:
                    OK -> the certificate (or the gate alone first, when the
                    gate never ran); bad tasks -> quarantine (moved, never
                    deleted) and round r + 1, at most bank.max_rerun_rounds;
                    a global problem -> STOP
        gate_alone  the gate with no dependency, after a verified round whose
                    gate never ran
    The artefact of record: BANK_RECORD.json + .sha256 and BANK_VERIFIED.json
    + .sha256 naming that record's sha256."""

    name = "sim_bank"
    deps = DEPS["sim_bank"]

    # -- places -------------------------------------------------------------

    def _bank_root(self, cfg):
        return os.path.join(cfg.data["paths"]["banks_root"], cfg.data["dataset"] + "_sim")

    def _record_path(self, cfg):
        return os.path.join(self._bank_root(cfg), "BANK_RECORD.json")

    def _cert_path(self, cfg):
        return os.path.join(self._bank_root(cfg), "BANK_VERIFIED.json")

    def _verify_dir(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "bank_verify")

    def _quarantine_root(self, cfg):
        return os.path.join(cfg.data["paths"]["banks_root"], cfg.data["dataset"] + "_sim_quarantine")

    # -- the done-predicate: the record AND its certificate ------------------

    def is_done(self, cfg):
        rec = self._record_path(cfg)
        ok, why = artefact_done("sha256", rec)
        if not ok:
            return False, why
        cert = self._cert_path(cfg)
        if not sha256_sidecar_ok(cert):
            return False, ("%s exists, but the bank is not verified: %s or its .sha256 is missing or "
                           "does not match" % (rec, cert))
        try:
            c = CM.read_json(cert)
        except ValueError:
            return False, "%s cannot be read" % cert
        if c.get("record_sha256") != _sha256(rec):
            return False, "%s certifies another record than %s" % (cert, rec)
        return True, "%s (sha256 matches its sidecar), verified (%s)" % (rec, cert)

    def _record_fresh(self, cfg, sub):
        """The record exists with its sidecar AND this round's gate wrote it
        (its stat changed since the gate's submission)."""
        rec = self._record_path(cfg)
        return (artefact_done("sha256", rec)[0]
                and log_is_fresh(rec, sub.get("record_stat", "unknown")))

    # -- helpers ---------------------------------------------------------------

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

    @staticmethod
    def _fail(sub, text):
        sub["fail"] = text
        sub["phase"] = "failed"
        return "finished"

    def _refresh_jobs(self, st, pbs):
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
        return finished, dict((j["id"], j) for j in st["jobs"])

    def _task_failed_line(self, cfg, index):
        """The last `FAILED:` line of a task's log, else its last line."""
        path = os.path.join(self._log_dir(cfg), "bank_task_%d.log" % int(index))
        text = read_log(path, limit=200000)
        lines = [l.strip() for l in text.split("\n") if l.startswith("FAILED:")]
        if lines:
            return lines[-1]
        return _last_line(path).strip()

    # -- submissions -------------------------------------------------------------

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
        """`launch_bank.sh array` for round sub['round']: the array and its
        gate. A rerun round passes the WALLTIME round 0 used (the launcher's
        auto rule needs the test task's marker, which a quarantine may have
        moved)."""
        sub = st["sub"]
        r = int(sub.get("round", 0))
        gate_log = os.path.join(self._log_dir(cfg), "bank_gate.log")
        before = file_stat(gate_log)
        record_before = file_stat(self._record_path(cfg))
        walltime = sub.get("walltime_used") or str(cfg.data["bank"]["walltime"])
        rc, text = self._run_launcher(cfg, "array", self._login_log(cfg),
                                      WALLTIME=walltime, TEST_INDEX="0")
        jobs = self._jobs_of(text)
        if rc != 0 or "job" not in jobs or "gate" not in jobs:
            sub["fail"] = ("the array could not be submitted (round %d, exit %d): %s"
                           % (r, rc, self._refusal(text)))
            sub["phase"] = "failed"
            return
        for line in text.split("\n"):
            if line.startswith("[launch] WALLTIME=auto") or line.startswith("# tasks"):
                pbs.log("sim_bank: " + line.strip())
            if line.startswith("[launch] WALLTIME=auto") and not sub.get("walltime_used"):
                m = _WALLTIME_USED.search(line)
                if m:
                    sub["walltime_used"] = m.group(1)
        if not sub.get("walltime_used") and walltime != "auto":
            sub["walltime_used"] = walltime
        label = "" if r == 0 else " (rerun round %d)" % r
        pbs.record(jobs["job"], "bank array" + label, ["launch_bank.sh", "array"],
                   log_path=os.path.join(self._log_dir(cfg), "bank_task_<index>.log"))
        pbs.record(jobs["gate"], "bank gate" + label, ["launch_bank.sh", "array"], log_path=gate_log)
        for k in ("gate_outcome", "gate_said", "gate_alone_unrun_of"):
            sub.pop(k, None)
        sub.update(array_job=jobs["job"], gate_job=jobs["gate"], gate_log=gate_log,
                   gate_log_stat=before, record_stat=record_before, verify_try=0,
                   round=r, phase="gate")

    def _start_gate_alone(self, cfg, st, pbs):
        """`launch_bank.sh gate`: the gate with no dependency, after a verified
        round whose gate never ran."""
        sub = st["sub"]
        gate_log = os.path.join(self._log_dir(cfg), "bank_gate.log")
        before = file_stat(gate_log)
        record_before = file_stat(self._record_path(cfg))
        rc, text = self._run_launcher(cfg, "gate", self._login_log(cfg))
        jobs = self._jobs_of(text)
        if rc != 0 or "gate" not in jobs:
            sub["fail"] = "the gate alone could not be submitted (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        pbs.record(jobs["gate"], "bank gate alone", ["launch_bank.sh", "gate"], log_path=gate_log)
        sub.update(gate_alone_unrun_of=sub.get("gate_job"), gate_job=jobs["gate"],
                   gate_log=gate_log, gate_log_stat=before, record_stat=record_before,
                   phase="gate_alone")

    def _submit_verify(self, cfg, st, pbs):
        """The verification job of this round (a new report path each try)."""
        sub = st["sub"]
        d = cfg.data
        v = d["bank"]["verify"]
        r = int(sub.get("round", 0))
        a = int(sub.get("verify_try", 0)) + 1
        sub["verify_try"] = a
        base = os.path.join(self._verify_dir(cfg), "round%d_try%d_%s" % (r, a, time.strftime("%Y%m%d_%H%M%S")))
        stem, k = base, 0
        while os.path.exists(stem + ".json") or os.path.exists(stem + ".argv"):
            k += 1
            stem = "%s_%d" % (base, k)
        report, argv_file, log_path = stem + ".json", stem + ".argv", stem + ".log"
        argv = [CM.E.python_in_env(), CM.paths_of(cfg)["verify_tool"],
                "--plan", artefact_of("bank_plan", cfg)[1], "--bank-root", self._bank_root(cfg),
                "--out", report, "--workers", str(int(v["ncpus"]))]
        try:
            CM.write_argv_file(argv_file, argv)
            line = CM.qsub_line(cfg, "bank_verify", d["bank"]["queue"], v["ncpus"], v["mem"],
                                v["walltime"], log_path, argv_file, d["paths"]["work_root"],
                                env_name=d["envs"]["extract"])
            job_id = pbs.submit(line, role="bank verify (round %d, try %d)" % (r, a),
                                log_path=log_path)
        except (RuntimeError, OSError, ValueError) as exc:
            return self._fail(sub, "the verification job could not be submitted: %s" % exc)
        sub.update(verify_job=job_id, verify_report=report, verify_log=log_path, phase="verify")
        return "running"

    # -- the verification's verdict -----------------------------------------------

    def _verify_done(self, cfg, st, pbs, finished, by_id):
        sub = st["sub"]
        job_id = sub["verify_job"]
        if not finished.get(job_id):
            return "running"
        path = sub["verify_report"]
        rep = None
        if os.path.isfile(path):
            try:
                rep = CM.read_json(path)
            except ValueError:
                rep = None
        if rep is None:
            if not grace_over(sub, job_id):
                return "running"
            es = by_id[job_id].get("exit_status")
            tail = (_last_lines(sub["verify_log"], 4) or "no log").replace("\n", " | ")
            if int(sub.get("verify_try", 0)) >= MAX_VERIFY_TRIES:
                return self._fail(sub, "the verification job ended without its report %d times in round "
                                       "%d (the last: job %s, exit %s; %s: %s)"
                                  % (int(sub["verify_try"]), int(sub.get("round", 0)), job_id, es,
                                     sub["verify_log"], tail))
            pbs.log("sim_bank: the verification job %s ended without its report (exit %s; %s): again"
                    % (job_id, es, tail))
            return self._submit_verify(cfg, st, pbs)
        sub["last_report"] = path
        return self._act_on_report(cfg, st, pbs, rep, path)

    def _act_on_report(self, cfg, st, pbs, rep, path):
        sub = st["sub"]
        host = rep.get("host", "?")
        bad = rep.get("bad_tasks") or {}
        glob_p = rep.get("global_problems") or []
        r = int(sub.get("round", 0))
        if rep.get("verdict") == "OK" and not bad and not glob_p:
            pbs.log("sim_bank: VERIFY OK on %s: %s tasks, %s shard files, %s members' CRCs, %s s"
                    % (host, rep.get("n_tasks"), rep.get("n_npz_checked"),
                       rep.get("n_members_checked"), rep.get("seconds")))
            if self._record_fresh(cfg, sub):
                return self._certify(cfg, st, pbs, path)
            if sub.get("gate_outcome") == "never_ran":
                if not sub.get("gate_alone_unrun_of"):
                    pbs.log("sim_bank: every task verified and no record (the gate never ran): the gate alone")
                    self._start_gate_alone(cfg, st, pbs)
                    return "running" if sub.get("phase") == "gate_alone" else "finished"
                return self._fail(sub, "the gate did not run, alone either (jobs %s and %s), though every task "
                                       "verified OK on %s (%s)" % (sub.get("gate_alone_unrun_of"),
                                                                   sub.get("gate_job"), host, path))
            return self._fail(sub, "the gate refused (job %s; %s): %s -- and every task verified OK on %s "
                                   "(%s), so the problem is one the verifier does not test (the contract, "
                                   "the donors, a repeated row)"
                              % (sub.get("gate_job"), sub.get("gate_log"), sub.get("gate_said", "?"),
                                 host, path))
        if glob_p:
            return self._fail(sub, "the bank folder has problems no rerun fixes (%s): %s; nothing was moved"
                              % (path, "; ".join(glob_p[:10])))
        n_tasks = int(rep.get("n_tasks") or 0)
        io_bad = [k for k, b in bad.items() if b.get("io_only")]
        if len(io_bad) >= 3 and 2 * len(io_bad) > n_tasks:
            if int(sub.get("verify_try", 0)) >= MAX_VERIFY_TRIES:
                return self._fail(sub, "the verifier could not read %d of %d tasks (I/O errors alone) %d "
                                       "times in round %d; the last on %s (%s)"
                                  % (len(io_bad), n_tasks, int(sub["verify_try"]), r, host, path))
            pbs.log("sim_bank: the verifier on %s could not read %d of %d tasks (I/O errors alone): the "
                    "node is suspect; verifying again, nothing moved" % (host, len(io_bad), n_tasks))
            return self._submit_verify(cfg, st, pbs)
        keys = sorted(bad, key=int)
        for k in keys:
            pbs.log("sim_bank: VERIFY BAD task %s (%s): %s" % (k, bad[k].get("name"),
                                                               "; ".join(bad[k].get("problems", [])[:4])))
        max_r = int(cfg.data["bank"]["max_rerun_rounds"])
        if r >= max_r:
            parts = []
            for k in keys:
                line = self._task_failed_line(cfg, k)
                parts.append("task %s (%s): %s%s" % (k, bad[k].get("name"),
                                                    "; ".join(bad[k].get("problems", [])[:3]),
                                                    (" [log: %s]" % line) if line else ""))
            return self._fail(sub, "after %d rerun round(s), %d task(s) still bad (%s): %s"
                              % (r, len(keys), path, " | ".join(parts[:20])))
        try:
            self._quarantine(cfg, st, pbs, rep, path)
        except OSError as exc:
            return self._fail(sub, "the quarantine could not move a file (%s); the moves made are listed in "
                                   "its QUARANTINE.json; nothing was deleted" % exc)
        sub["round"] = r + 1
        pbs.log("sim_bank: rerun round %d of at most %d: the array again (finished tasks skip on their "
                "markers)" % (r + 1, max_r))
        self._start_array(cfg, st, pbs)
        return "running" if sub.get("phase") == "gate" else "finished"

    def _quarantine(self, cfg, st, pbs, rep, report_path):
        """Move every file of every bad task, and the record and the
        certificate, out of the bank folder (D-095: moved, never deleted)."""
        sub = st["sub"]
        bank = self._bank_root(cfg)
        r = int(sub.get("round", 0))
        base = os.path.join(self._quarantine_root(cfg), "%s_round%d" % (time.strftime("%Y%m%d_%H%M%S"), r))
        qdir, k = base, 0
        while os.path.exists(qdir):
            k += 1
            qdir = "%s_%d" % (base, k)
        os.makedirs(qdir)
        manifest = {"rule": "D-095: moved out of the bank folder, never deleted; delete through the "
                            "list-then-delete tool", "round": r, "bank_root": bank,
                    "report": report_path, "report_sha256": _sha256(report_path),
                    "written": CM.utc_now(), "moves": []}
        qfile = os.path.join(qdir, "QUARANTINE.json")

        def move(rel, task, reasons):
            src = os.path.join(bank, rel)
            if not os.path.lexists(src):
                return
            dst = os.path.join(qdir, rel)
            d = os.path.dirname(dst)
            if not os.path.isdir(d):
                os.makedirs(d)
            size = os.lstat(src).st_size
            try:
                os.rename(src, dst)          # same file system: banks_root holds both
            finally:
                if os.path.lexists(dst):
                    manifest["moves"].append({"from": src, "to": dst, "bytes": int(size),
                                              "task": task, "reasons": reasons})
                CM.write_json(qfile, manifest)

        bad = rep.get("bad_tasks") or {}
        for key in sorted(bad, key=int):
            b = bad[key]
            for rel in b.get("files", []):
                move(rel, int(key), list(b.get("problems", [])))
            pbs.log("sim_bank: quarantine task %s (%s): %d file(s) moved"
                    % (key, b.get("name"), len(b.get("files", []))))
        for rel in ("BANK_RECORD.json", "BANK_RECORD.json.sha256", "BANK_VERIFIED.json",
                    "BANK_VERIFIED.json.sha256"):
            move(rel, None, ["the bank changes: this file no longer describes it"])
        CM.write_json(qfile, manifest)
        sub.setdefault("quarantines", []).append(qdir)
        pbs.log("sim_bank: quarantined %d file(s) of %d task(s) in %s (QUARANTINE.json lists every move)"
                % (len(manifest["moves"]), len(bad), qdir))

    def _certify(self, cfg, st, pbs, report_path):
        """BANK_VERIFIED.json + .sha256: the record's sha256 and the report."""
        sub = st["sub"]
        rec, cert = self._record_path(cfg), self._cert_path(cfg)
        try:
            rep = CM.read_json(report_path)
            body = {"certificate": "bank_verified", "version": 1, "record": rec,
                    "record_sha256": _sha256(rec), "report": report_path,
                    "report_sha256": _sha256(report_path), "verified_by": rep.get("host"),
                    "n_tasks": rep.get("n_tasks"), "n_npz_checked": rep.get("n_npz_checked"),
                    "n_members_checked": rep.get("n_members_checked"),
                    "rerun_rounds": int(sub.get("round", 0)),
                    "quarantines": list(sub.get("quarantines", [])), "written": CM.utc_now()}
            CM.write_json(cert, body)
            tmp = cert + ".sha256.tmp"
            with open(tmp, "w") as fh:
                fh.write("%s  %s\n" % (_sha256(cert), os.path.basename(cert)))
            os.replace(tmp, cert + ".sha256")
        except (OSError, ValueError) as exc:
            return self._fail(sub, "the certificate %s could not be written: %s" % (cert, exc))
        pbs.log("sim_bank: VERIFIED: %s shard files, every zip member's CRC checked on %s; record sha256 "
                "%s; wrote %s" % (body["n_npz_checked"], body["verified_by"], body["record_sha256"][:16],
                                  cert))
        sub["phase"] = "complete"
        return "finished"

    # -- the stage -----------------------------------------------------------------

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        self.ensure_profile(cfg, pbs)
        sub["round"] = 0
        if self._test_marker_ok(cfg):
            pbs.log("sim_bank: the test task's marker belongs to this plan: test skipped")
            sub["test_skipped"] = True
            self._start_array(cfg, st, pbs)
        else:
            self._start_test(cfg, st, pbs)

    def poll(self, cfg, st, pbs):
        sub = st["sub"]
        finished, by_id = self._refresh_jobs(st, pbs)
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
            return self._fail(sub, "the test task did not pass (job %s, exit %s, no marker of this plan in "
                                   "%s; log %s: %s)"
                              % (job_id, by_id[job_id].get("exit_status"),
                                 os.path.join(self._bank_root(cfg), "tasks"), sub["test_log"],
                                 "; ".join(refused[:5]) if refused
                                 else (_last_lines(sub["test_log"], 3) or "absent")))
        if phase == "gate":
            job_id = sub["gate_job"]
            if not finished.get(job_id):
                return "running"
            outcome = self._gate_outcome(cfg, st)
            if outcome == "wait":
                return "running"
            array_id = sub.get("array_job")
            if array_id and not finished.get(array_id):
                return "running"
            sub["gate_outcome"] = outcome
            es = by_id[job_id].get("exit_status")
            if outcome == "passed":
                pbs.log("sim_bank: " + sub["gate_said"])
            elif outcome == "refused":
                pbs.log("sim_bank: the gate %s refused (exit %s): %s" % (job_id, es, sub["gate_said"]))
            else:
                pbs.log("sim_bank: the gate %s never ran (finished with exit %s, no record, no log at %s; "
                        "the array %s has ended)" % (job_id, es, sub["gate_log"], array_id or "-"))
            pbs.log("sim_bank: verifying the bank, round %d: every shard's sha256 and every zip member's "
                    "CRC, as a job" % int(sub.get("round", 0)))
            return self._submit_verify(cfg, st, pbs)
        if phase == "verify":
            return self._verify_done(cfg, st, pbs, finished, by_id)
        if phase == "gate_alone":
            job_id = sub["gate_job"]
            if not finished.get(job_id):
                return "running"
            outcome = self._gate_outcome(cfg, st)
            if outcome == "wait":
                return "running"
            es = by_id[job_id].get("exit_status")
            if outcome == "passed":
                pbs.log("sim_bank: " + sub["gate_said"])
                return self._certify(cfg, st, pbs, sub["last_report"])
            if outcome == "never_ran":
                return self._fail(sub, "the gate did not run (jobs %s and %s finished with no record and no "
                                       "log at %s; exit %s): read qstat -x -f %s"
                                  % (sub.get("gate_alone_unrun_of"), job_id, sub["gate_log"], es, job_id))
            return self._fail(sub, "the gate refused (job %s, exit %s; %s): %s"
                              % (job_id, es, sub["gate_log"], sub["gate_said"]))
        return "finished"

    def _gate_outcome(self, cfg, st):
        """'passed' (this gate wrote the record), 'refused' (its own log, no
        record), 'never_ran' (no record and no log of its own once the grace
        period is over) or 'wait', for the finished gate sub['gate_job']."""
        sub = st["sub"]
        job_id = sub["gate_job"]
        # only this gate's log counts: a log left by an earlier gate is not read
        fresh = log_is_fresh(sub["gate_log"], sub.get("gate_log_stat", "unknown"))
        text = read_log(sub["gate_log"]) if fresh else ""
        wrote = [l for l in text.split("\n") if l.startswith("wrote ") and "BANK_RECORD.json" in l]
        if self._record_fresh(cfg, sub):
            sub["gate_said"] = (wrote[-1].strip() if wrote
                                else "%s written (its sidecar matches)" % self._record_path(cfg))
            return "passed"
        ended = job_exit_line(text, "bank_gate")
        if (wrote or ended is None) and not grace_over(sub, job_id):
            return "wait"
        if not fresh:
            return "never_ran"
        fails = [l for l in text.split("\n") if l.startswith("[gate] FAIL") or l.startswith("REFUSED")]
        sub["gate_said"] = ("; ".join(fails[:20]) if fails
                            else (_last_lines(sub["gate_log"], 3) or "no log"))
        return "refused"

    def dry_run(self, cfg, st):
        d = cfg.data
        ld = self._launcher_dir(cfg)
        v = d["bank"]["verify"]
        return ["test (skipped when %s/tasks/task_0000.json belongs to the plan): PROFILE=<profile> "
                "QUEUE=%s TEST_INDEX=0 bash %s/launch_bank.sh test -> one job; passed by its marker "
                "(log line TASK 0 OK)" % (self._bank_root(cfg), d["bank"]["queue"], ld),
                "array: WALLTIME=%s CONCURRENCY=%d QUEUE=%s bash %s/launch_bank.sh array -> the array "
                "(-J 0-<n-1>%%%d) and its gate job (depend=afterok)"
                % (d["bank"]["walltime"], int(d["bank"]["concurrency"]), d["bank"]["queue"], ld,
                   int(d["bank"]["concurrency"])),
                "verify, after the gate: qsub -N bank_verify -q %s -l select=1:ncpus=%d:mem=%s -l "
                "walltime=%s ... run_argv.pbs -> %s --plan <plan> --bank-root %s --out %s/round<r>_try<a>_"
                "<time>.json; bad tasks moved to %s/<time>_round<r>/ and the array again, at most %d "
                "rerun round(s)" % (d["bank"]["queue"], int(v["ncpus"]), v["mem"], v["walltime"],
                                    CM.paths_of(cfg)["verify_tool"], self._bank_root(cfg),
                                    self._verify_dir(cfg), self._quarantine_root(cfg),
                                    int(d["bank"]["max_rerun_rounds"])),
                "a gate that never runs (no record, no log of its own): once the round verifies, bash "
                "%s/launch_bank.sh gate -> the gate alone" % ld,
                "pass lines: 'sim_bank: VERIFY OK on <host>: ...' and 'sim_bank: VERIFIED: ...'; artefact "
                "BANK_RECORD.json + .sha256 and BANK_VERIFIED.json + .sha256 in %s" % self._bank_root(cfg)]


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
