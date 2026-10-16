"""Stages S1-S3, the data set's inputs (block 3), through Sbi-extractor's
committed launchers (D-062: driven, not changed). For the first data set
all three artefacts exist, so the stages are skipped; they are what a NEW
data set runs first.

    real_extraction (S1)   `extractor/launch_stage_d.sh <extract_root>` with
                           CONFIG = real.cohort_config, COHORT_TAG =
                           real.cohort_tag: one array (one well per element)
                           and the aggregation job, which writes
                           `<extract_root>/cohort_manifest.json` + .sha256;
                           its log `extractor/out/cohort_manifest_<tag>.log`
                           ends `wrote <root>/cohort_manifest.json  sha256
                           <16 hex>` or names every well under REFUSED.
                           The launcher's own refusals (a new cohort whose
                           flags file it just wrote and must be COMMITTED; a
                           root that already holds a manifest; flags that
                           differ from the tracked file) stop the run with
                           their text: they are the user's to settle.
    sim_redetection (S2)   `sim_reextract/launch_sim_reextract.sh` `plan`,
                           `test` (one task as a job, TEST_INDEX 0), then
                           `array` with RESUME=1 (the array and its gate; or
                           `gate` alone when every task is complete) with
                           the configuration's `sim` block as the launcher's
                           variables; the gate's log `sim_reextract/out/
                           sim_reextract_gate_<tag>.log` ends `wrote
                           <out_root>/REEXTRACTION_RECORD.json  sha256 <16
                           hex>`. The launcher needs the cohort's profile
                           `sim_reextract/profiles/<tag>.sh` (the device
                           geometry lives there): refused when absent.
    label_freeze (S3)      `preflight_label_axes.py --sim_main --campaigns
                           --require-conn-rule --exclude NAME=REASON ...
                           --out <sbix_dir>/<freeze.out>` on the login node;
                           pass lines `wrote <out>` and `sha256 <16 hex>`.

Every launcher is run in `envs.extract` from its own folder with
SBI_HPC_DIR = `<paths.sbi_dir>/hpc` (the DSN tree the tools import).

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import os
import re

from . import DEPS, Stage, artefact_of
from . import common as CM
from .bank import _last_lines, _JOB_LINE, read_log, job_exit_line, grace_over


class _InputBase(Stage):
    max_resubmits = 0

    def _env(self, cfg, **extra):
        d = cfg.data
        env = dict(os.environ)
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "RESUME", "COHORT_TAG", "CONFIG", "QUEUE",
                  "NCPUS", "CONCURRENCY", "PLAN_ARGS", "COHORT_MANIFEST", "SIM_MAIN", "OUT_ROOT",
                  "TOOLS", "CAMPAIGN_GLOB"):
            env.pop(k, None)
        env.update({"ENV_NAME": d["envs"]["extract"],
                    "SBI_HPC_DIR": os.path.join(d["paths"]["sbi_dir"], "hpc")})
        env.update(dict((k, str(v)) for k, v in extra.items()))
        return env

    def _login_log(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "logs", "%s_login_node.log" % self.name)

    def _run(self, cfg, argv, cwd, **extra):
        return CM.E.run_in_env(cfg.data["envs"]["extract"], argv, cwd=cwd,
                               log_path=self._login_log(cfg), env=self._env(cfg, **extra))

    def _refusal(self, text):
        lines = [l for l in text.split("\n")
                 if l.startswith(("ABORT", "REFUSED", "NEW COHORT", "STOP", "ERROR"))]
        if lines:
            return "; ".join(lines)
        t = text.strip().split("\n")
        return t[-1] if t and t[-1] else "(no output)"

    def _poll_jobs(self, st, pbs):
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
        return finished

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
# S1: the real-arm extraction
# ---------------------------------------------------------------------------

_S1_LINE = re.compile(r"^\s*(array|agg)\s*:\s*(\S+)", re.M)


class RealExtractionStage(_InputBase):
    name = "real_extraction"
    deps = DEPS["real_extraction"]

    def _dir(self, cfg):
        return os.path.join(cfg.data["paths"]["sbix_dir"], "extractor")

    def _agg_log(self, cfg):
        tag = cfg.data["real"]["cohort_tag"]
        return os.path.join(self._dir(cfg), "out", "cohort_manifest_%s.log" % tag)

    def start(self, cfg, st, pbs):
        d = cfg.data
        sub = st["sub"]
        if not os.path.isfile(d["real"]["cohort_config"]):
            raise RuntimeError("real_extraction: real.cohort_config not found: %s"
                               % d["real"]["cohort_config"])
        rc, text = self._run(cfg, ["bash", os.path.join(self._dir(cfg), "launch_stage_d.sh"),
                                   d["real"]["extract_root"]], self._dir(cfg),
                             CONFIG=d["real"]["cohort_config"], COHORT_TAG=d["real"]["cohort_tag"])
        jobs = dict(_S1_LINE.findall(text))
        if rc != 0 or "array" not in jobs or "agg" not in jobs:
            sub["fail"] = ("the extraction could not be submitted (exit %d): %s"
                           % (rc, self._refusal(text)))
            sub["phase"] = "failed"
            return
        for line in text.split("\n"):
            if line.startswith("# wells") or line.startswith("# extract root"):
                pbs.log("real_extraction: " + line.strip())
        pbs.record(jobs["array"], "extraction array", ["launch_stage_d.sh", d["real"]["extract_root"]],
                   log_path=os.path.join(self._dir(cfg), "out", "chsub_mea_array_%s_<index>.log"
                                         % d["real"]["cohort_tag"]))
        pbs.record(jobs["agg"], "cohort manifest", ["launch_stage_d.sh", d["real"]["extract_root"]],
                   log_path=self._agg_log(cfg))
        sub["agg_job"] = jobs["agg"]
        sub["phase"] = "jobs"

    def poll(self, cfg, st, pbs):
        sub = st["sub"]
        finished = self._poll_jobs(st, pbs)
        if sub.get("phase") != "jobs":
            return "finished"
        job_id = sub["agg_job"]
        if not finished.get(job_id):
            return "running"
        # success from the file the job writes; the log for the text (bank.py's rule)
        text = read_log(self._agg_log(cfg))
        wrote = [l for l in text.split("\n") if l.startswith("wrote ") and "cohort_manifest.json" in l]
        done, why = self.is_done(cfg)
        if done:
            pbs.log("real_extraction: " + (wrote[-1].strip() if wrote else why))
            sub["phase"] = "complete"
            return "finished"
        ended = job_exit_line(text, "cohort_manifest")
        if (wrote or ended is None) and not grace_over(sub, job_id):
            return "running"
        refused = [l for l in text.split("\n") if "REFUSED" in l or l.startswith("  well")]
        sub["fail"] = ("the cohort manifest was refused (job %s; %s): %s"
                       % (job_id, self._agg_log(cfg),
                          "; ".join(refused[:12]) if refused
                          else (_last_lines(self._agg_log(cfg), 3) or "no log")))
        sub["phase"] = "failed"
        return "finished"

    def dry_run(self, cfg, st):
        d = cfg.data
        return ["login node (envs.extract): CONFIG=%s COHORT_TAG=%s ENV_NAME=%s bash %s/launch_stage_d.sh %s "
                "-> the array (one well per element) and the aggregation job"
                % (d["real"]["cohort_config"], d["real"]["cohort_tag"], d["envs"]["extract"],
                   self._dir(cfg), d["real"]["extract_root"]),
                "pass line: %s ends 'wrote %s/cohort_manifest.json  sha256 <16 hex>'; artefact + .sha256"
                % (self._agg_log(cfg), d["real"]["extract_root"])]


# ---------------------------------------------------------------------------
# S2: the simulated-arm re-detection
# ---------------------------------------------------------------------------

class SimRedetectionStage(_InputBase):
    name = "sim_redetection"
    deps = DEPS["sim_redetection"]

    def _dir(self, cfg):
        return os.path.join(cfg.data["paths"]["sbix_dir"], "sim_reextract")

    def _gate_log(self, cfg):
        return os.path.join(self._dir(cfg), "out",
                            "sim_reextract_gate_%s.log" % cfg.data["real"]["cohort_tag"])

    def _vars(self, cfg):
        d = cfg.data
        s = d["sim"]
        return {"COHORT_TAG": d["real"]["cohort_tag"],
                "COHORT_MANIFEST": artefact_of("real_extraction", cfg)[1],
                "SIM_MAIN": s["sim_main"], "OUT_ROOT": s["out_root"], "TOOLS": s["tools"],
                "CAMPAIGN_GLOB": s["campaign_glob"], "PLAN_ARGS": s["plan_args"],
                "QUEUE": s["queue"], "NCPUS": int(s["ncpus"]), "CONCURRENCY": int(s["concurrency"])}

    def _launch(self, cfg, mode, **extra):
        v = self._vars(cfg)
        v.update(extra)
        return self._run(cfg, ["bash", os.path.join(self._dir(cfg), "launch_sim_reextract.sh"), mode],
                         self._dir(cfg), **v)

    def start(self, cfg, st, pbs):
        d = cfg.data
        sub = st["sub"]
        profile = os.path.join(self._dir(cfg), "profiles", "%s.sh" % d["real"]["cohort_tag"])
        if not os.path.isfile(profile):
            raise RuntimeError("sim_redetection: the cohort's profile %s does not exist; the "
                               "launcher needs it (the device geometry and the cohort's own "
                               "folders): add it to Sbi-extractor first" % profile)
        rc, text = self._launch(cfg, "plan")
        if rc != 0:
            sub["fail"] = "the re-detection plan was refused (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        for line in text.split("\n"):
            if line.startswith("[plan] tasks") or line.startswith("[launch] plan only"):
                pbs.log("sim_redetection: " + line.strip())
        rc, text = self._launch(cfg, "test", WALLTIME=d["sim"]["test_walltime"], TEST_INDEX=0)
        jobs = dict(_JOB_LINE.findall(text))
        if rc != 0 or "job" not in jobs:
            sub["fail"] = "the test task could not be submitted (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        pbs.record(jobs["job"], "re-detection test task 0", ["launch_sim_reextract.sh", "test"])
        sub["test_job"] = jobs["job"]
        sub["phase"] = "test"

    def _start_array(self, cfg, st, pbs):
        d = cfg.data
        sub = st["sub"]
        rc, text = self._launch(cfg, "array", WALLTIME=d["sim"]["walltime"], RESUME=1)
        if rc != 0 and "tasks.tsv is empty" in text:
            pbs.log("sim_redetection: every task complete after the test task; the gate alone")
            rc, text = self._launch(cfg, "gate", RESUME=1)
        jobs = dict(_JOB_LINE.findall(text))
        if rc != 0 or "gate" not in jobs:
            sub["fail"] = "the array could not be submitted (exit %d): %s" % (rc, self._refusal(text))
            sub["phase"] = "failed"
            return
        for line in text.split("\n"):
            if line.startswith("[plan] tasks") or line.startswith("# tasks"):
                pbs.log("sim_redetection: " + line.strip())
        if "job" in jobs:
            pbs.record(jobs["job"], "re-detection array", ["launch_sim_reextract.sh", "array"])
        pbs.record(jobs["gate"], "re-detection gate", ["launch_sim_reextract.sh", "array"],
                   log_path=self._gate_log(cfg))
        sub["gate_job"] = jobs["gate"]
        sub["phase"] = "gate"

    def poll(self, cfg, st, pbs):
        sub = st["sub"]
        finished = self._poll_jobs(st, pbs)
        by_id = dict((j["id"], j) for j in st["jobs"])
        phase = sub.get("phase")
        if phase == "test":
            if not finished.get(sub["test_job"]):
                return "running"
            job = by_id[sub["test_job"]]
            if job.get("exit_status") not in (0, None):
                sub["fail"] = ("the test task failed (job %s exit %s); read its log in the job's "
                               "home folder" % (sub["test_job"], job.get("exit_status")))
                sub["phase"] = "failed"
                return "finished"
            pbs.log("sim_redetection: test task finished (job %s exit %s)"
                    % (sub["test_job"], job.get("exit_status")))
            self._start_array(cfg, st, pbs)
            return "running" if sub.get("phase") == "gate" else "finished"
        if phase == "gate":
            job_id = sub["gate_job"]
            if not finished.get(job_id):
                return "running"
            text = read_log(self._gate_log(cfg))
            wrote = [l for l in text.split("\n")
                     if l.startswith("wrote ") and "REEXTRACTION_RECORD.json" in l]
            done, why = self.is_done(cfg)
            if done:
                pbs.log("sim_redetection: " + (wrote[-1].strip() if wrote else why))
                sub["phase"] = "complete"
                return "finished"
            ended = job_exit_line(text, "sim_reextract_gate")
            if (wrote or ended is None) and not grace_over(sub, job_id):
                return "running"
            refused = [l for l in text.split("\n") if "REFUSED" in l or "[gate] FAIL" in l]
            sub["fail"] = ("the re-detection gate refused (job %s; %s): %s"
                           % (job_id, self._gate_log(cfg),
                              "; ".join(refused[:12]) if refused
                              else (_last_lines(self._gate_log(cfg), 3) or "no log")))
            sub["phase"] = "failed"
            return "finished"
        return "finished"

    def dry_run(self, cfg, st):
        d = cfg.data
        v = self._vars(cfg)
        vs = " ".join("%s=%s" % (k, v[k]) for k in ("COHORT_TAG", "COHORT_MANIFEST", "SIM_MAIN", "OUT_ROOT",
                                                    "TOOLS", "CAMPAIGN_GLOB", "PLAN_ARGS", "QUEUE",
                                                    "NCPUS", "CONCURRENCY"))
        return ["login node (envs.extract): %s bash %s/launch_sim_reextract.sh plan" % (vs, self._dir(cfg)),
                "then: WALLTIME=%s TEST_INDEX=0 ... test -> one job; then WALLTIME=%s RESUME=1 ... array "
                "-> the array and its gate (or `gate` alone when every task is complete)"
                % (d["sim"]["test_walltime"], d["sim"]["walltime"]),
                "pass line: %s ends 'wrote %s/REEXTRACTION_RECORD.json  sha256 <16 hex>'; artefact + .sha256"
                % (self._gate_log(cfg), d["sim"]["out_root"])]


# ---------------------------------------------------------------------------
# S3: the label freeze
# ---------------------------------------------------------------------------

class LabelFreezeStage(_InputBase):
    name = "label_freeze"
    deps = DEPS["label_freeze"]

    def _argv(self, cfg):
        d = cfg.data
        f = d["freeze"]
        argv = [CM.E.python_in_env(), os.path.join(d["paths"]["sbix_dir"], "preflight_label_axes.py"),
                "--sim_main", d["sim"]["sim_main"], "--campaigns", f["campaigns"],
                "--require-conn-rule", f["require_conn_rule"],
                "--out", artefact_of("label_freeze", cfg)[1]]
        for axis, reason in sorted(f["exclude"].items()):
            argv += ["--exclude", "%s=%s" % (axis, reason)]
        return argv

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        rc, text = self._run(cfg, self._argv(cfg), cfg.data["paths"]["sbix_dir"])
        out = artefact_of("label_freeze", cfg)[1]
        wrote = [l for l in text.split("\n") if l.startswith("wrote ") or l.startswith("sha256 ")
                 or l.startswith("theta topology block")]
        if rc != 0 or not any(l.startswith("wrote ") for l in wrote):
            sub["fail"] = "the label freeze was refused (exit %d): %s" % (rc, self._refusal(text))
            return
        for l in wrote:
            pbs.log("label_freeze: " + l.strip())
        sub["out"] = out

    def poll(self, cfg, st, pbs):
        return "finished"

    def dry_run(self, cfg, st):
        return ["login node (envs.extract): " + " ".join(self._argv(cfg)),
                "pass lines: 'wrote %s' and 'sha256 <16 hex>'; artefact + .sha256"
                % artefact_of("label_freeze", cfg)[1]]
