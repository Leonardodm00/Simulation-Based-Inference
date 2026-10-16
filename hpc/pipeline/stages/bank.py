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
                     belongs to this plan), its log's last line `TASK <i>
                     OK`; `array` with WALLTIME from `bank.walltime`
                     (`auto` = the launcher's rule from the test task's
                     pace) -- the array and its gate job; the gate's log
                     must hold `wrote <bank_root>/BANK_RECORD.json`;
                     artefact `BANK_RECORD.json` + .sha256
    real_bank (S6)   `launch_bank.sh real` on the login node; pass line
                     `REAL BANK OK`; artefact `shard_0000.npz` + .json

The launcher submits the jobs itself and prints `  job   : <id>` and
`  gate  : <id>`; the stages record those ids through the sequencer's
Submitter so they are polled and resumed like any other job. The
launcher's refusals are its `ABORT:` / `REFUSED:` lines, read back from
its output.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import hashlib
import os
import re

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
        prof = self.write_profile(cfg)
        pbs.log("bank_plan: profile %s (record %s, freeze %s, manifest %s)"
                % (self._profile_path(cfg), prof["record_sha256_prefix"],
                   prof["freeze_sha256_prefix"], prof["cohort_manifest_sha256_prefix"]))
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
        pbs.record(jobs["gate"], "bank gate", ["launch_bank.sh", "array"],
                   log_path=os.path.join(self._log_dir(cfg), "bank_gate.log"))
        sub["array_job"] = jobs["job"]
        sub["gate_job"] = jobs["gate"]
        sub["gate_log"] = os.path.join(self._log_dir(cfg), "bank_gate.log")
        sub["phase"] = "gate"

    def start(self, cfg, st, pbs):
        sub = st["sub"]
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
        phase = sub.get("phase")
        if phase == "test":
            if not finished.get(sub["test_job"]):
                return "running"
            # the pass line `TASK <i> OK`, followed by the job script's own
            # `[job] bank_task_exit=0 ...` line (the chain line BC's wait)
            tail = _last_lines(sub["test_log"], 5).split("\n")
            passed = [l for l in tail if re.match(r"^TASK \d+ OK", l)]
            if not passed:
                sub["fail"] = ("the test task did not pass (job %s; log %s ends: %s)"
                               % (sub["test_job"], sub["test_log"], _last_lines(sub["test_log"], 3)))
                sub["phase"] = "failed"
                return "finished"
            pbs.log("sim_bank: test task passed: %s" % passed[-1])
            self._start_array(cfg, st, pbs)
            return "running" if sub.get("phase") == "gate" else "finished"
        if phase == "gate":
            if not finished.get(sub["gate_job"]):
                return "running"
            text = _last_lines(sub["gate_log"], 40)
            wrote = [l for l in text.split("\n")
                     if l.startswith("wrote ") and "BANK_RECORD.json" in l]
            if not wrote:
                fails = [l for l in text.split("\n") if l.startswith("[gate] FAIL") or
                         l.startswith("REFUSED")]
                sub["fail"] = ("the gate refused (job %s; %s): %s"
                               % (sub["gate_job"], sub["gate_log"],
                                  "; ".join(fails) if fails else _last_lines(sub["gate_log"], 3)))
                sub["phase"] = "failed"
                return "finished"
            pbs.log("sim_bank: " + wrote[-1].strip())
            sub["phase"] = "complete"
            return "finished"
        return "finished"

    def dry_run(self, cfg, st):
        d = cfg.data
        ld = self._launcher_dir(cfg)
        return ["test (skipped when %s/tasks/task_0000.json belongs to the plan): PROFILE=<profile> "
                "QUEUE=%s TEST_INDEX=0 bash %s/launch_bank.sh test -> one job; its log's last line "
                "TASK 0 OK" % (self._bank_root(cfg), d["bank"]["queue"], ld),
                "array: WALLTIME=%s CONCURRENCY=%d QUEUE=%s bash %s/launch_bank.sh array -> the array "
                "(-J 0-<n-1>%%%d) and its gate job (depend=afterok)"
                % (d["bank"]["walltime"], int(d["bank"]["concurrency"]), d["bank"]["queue"], ld,
                   int(d["bank"]["concurrency"])),
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
