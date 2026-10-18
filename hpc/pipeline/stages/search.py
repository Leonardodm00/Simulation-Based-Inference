"""Stage S7, the GP search over the joint stack's hyper-parameters
(block 7; specs/SPEC.md secs. 2.3 and 5, block 7).

What runs where
    login node, `envs.train` (through `envs.run_in_env`): the bank's split
        hash (`tools/bank_split_hash.py`), the tuner's `costliest`,
        `propose` and `finalists`;
    the scheduler, one job per probe and one job per trial
        (`jobs/run_argv.pbs`, the trial's resources from `search.trial`):
        the budget gate's probes (the runner in probe mode) and the trials
        (the tuner's `evaluate`, which runs the runner and writes the
        ledger record).

The phases, kept in the stage's private state `st["sub"]` and mirrored
(rounds, identity, gate, clock) in `<tune_root>/<campaign>/pipeline_rounds.
json`, which survives a `--retry` and is what makes a results folder of
another search refused:
    identity   the split hash and the contract digest of the bank, once
    gate       `budget_gate`: the space's costliest configuration at each
               searched batch size, `probe_steps` steps and `probe_val_rows`
               validation rows on the trial's resources; the run stops
               unless max_B T_B * safety_factor <= trial walltime and
               max_B maxrss_B <= trial mem, T_B = epochs * (S_B t_step +
               n_sel t_val_row) (the runner's projection)
    rounds     round 0: `n_initial_points` random configurations; round r
               >= 1: `trials_per_round` by expected improvement (constant
               liar), the GP refitted on the ledger; at most
               `max_concurrent_trials` trial jobs at once; a trial whose
               job ends with neither a record nor a refusal is resubmitted
               with --resume up to `trial.max_resubmits` times, then
               recorded `killed`; a trial recorded `failed` is examined
               (D-096): a node fault -- the runner killed by SIGBUS, or a
               file-system error in its log -- sets the record aside
               (`<id>.json.nodefault<k>`) and queues the trial again with
               --resume, within the same `max_resubmits`, else `killed`;
               before any submission a checkpoint whose zip CRCs fail is
               moved aside (`.corrupt<k>`, D-097); a round is complete when
               every trial of it is ok, failed or killed; the clock (D-090, elapsed) starts
               at round 0's first submission; after a complete round the
               search ends when r == gp_rounds (`rounds`) or the clock
               exceeds `max_total_walltime_h` (`budget`) or the tuner
               proposes nothing new (`no_new_proposals`)
    finish     `finalists --top-k 1`, then the artefact of record
               `SEARCH_COMPLETE.json`

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import glob
import os
import re
import time

from . import DEPS, Stage
from . import common as CM

ROUNDS_FILE = "pipeline_rounds.json"
COMPLETE_FILE = "SEARCH_COMPLETE.json"
TERMINAL = ("ok", "failed", "killed")

_PROBE_LINE = re.compile(r"^PROBE OK:")


class SearchStage(Stage):
    name = "search"
    deps = DEPS["search"]
    max_resubmits = 0           # the trials carry their own resubmissions

    # ------------------------------------------------------------ paths ---

    def _dirs(self, cfg):
        d = cfg.data
        camp = os.path.join(d["paths"]["tune_root"], d["search"]["campaign"])
        return {"campaign": camp,
                "pending": os.path.join(camp, "pending"),
                "ledger": os.path.join(camp, "trials"),
                "gate": os.path.join(camp, "gate"),
                "runs": os.path.join(camp, "runs"),
                "jobs": os.path.join(camp, "jobs"),
                "rounds_file": os.path.join(camp, ROUNDS_FILE),
                "finalists": os.path.join(camp, "finalists.json"),
                "complete": os.path.join(camp, COMPLETE_FILE)}

    # --------------------------------------------------------- identity ---

    def _identity(self, cfg, st, log_path):
        """The bank's split hash, contract digest and row counts, computed
        once per run by the runner's own split function (tools/
        bank_split_hash.py) and kept in the state."""
        sub = st["sub"]
        if sub.get("identity"):
            return sub["identity"]
        d = cfg.data
        p = CM.paths_of(cfg)
        sim, _ = CM.shard_globs(cfg)
        out = os.path.join(self._dirs(cfg)["campaign"], "bank_split.json")
        argv = [CM.E.python_in_env(), p["split_tool"], "--sim-shards", sim,
                "--split-fracs"] + [repr(float(f)) for f in d["split"]["fractions"]] + \
               ["--split-seed", str(int(d["search"]["train_seed"])), "--out", out]
        rc, text = CM.E.run_in_env(d["envs"]["train"], argv, log_path=log_path)
        if rc != 0 or not os.path.isfile(out):
            raise RuntimeError("the bank's split hash could not be computed (exit %d): %s"
                               % (rc, text.strip()[-1500:]))
        rec = CM.read_json(out)
        ident = {"split_hash": rec["split_hash"], "contract_digest": rec["contract_digest"],
                 "n_train": int(rec["n_train"]), "n_sel": int(rec["n_sel"]),
                 "n_report": int(rec["n_report"]), "n_rows": int(rec["n_rows"]),
                 "n_shards": int(rec["n_shards"]), "W": int(rec["W"]),
                 "d_theta": int(rec["d_theta"]), "latent_kind": rec.get("latent_kind"),
                 "split_fracs": list(rec["split_fracs"]), "split_seed": int(rec["split_seed"]),
                 "campaign": d["search"]["campaign"],
                 "shape_anchors": dict(d["search"]["shape_anchors"]),
                 "space_overrides": dict(d["search"]["space_overrides"]),
                 "flow": dict(d["search"]["flow"])}
        if ident["d_theta"] != int(d["search"]["shape_anchors"]["d_theta"]):
            raise RuntimeError("the bank's theta has %d columns, search.shape_anchors.d_theta "
                               "is %d" % (ident["d_theta"], d["search"]["shape_anchors"]["d_theta"]))
        sub["identity"] = ident
        return ident

    def _rounds_file(self, cfg, st):
        """Create or check `pipeline_rounds.json`: a results folder that
        holds a search of another identity (another bank, split, space
        or campaign) is refused rather than mixed (block 7's proposal 2)."""
        sub = st["sub"]
        path = self._dirs(cfg)["rounds_file"]
        ident = sub["identity"]
        if os.path.isfile(path):
            rf = CM.read_json(path)
            diffs = []
            for k in ("split_hash", "contract_digest", "n_train", "campaign", "shape_anchors",
                      "space_overrides", "flow", "split_fracs", "split_seed"):
                if rf.get("identity", {}).get(k) != ident.get(k):
                    diffs.append("%s: folder %r, configuration %r"
                                 % (k, rf.get("identity", {}).get(k), ident.get(k)))
            if diffs:
                raise RuntimeError("REFUSE: %s holds a search of another identity; a results "
                                   "folder is not shared between searches (move it or change "
                                   "paths.tune_root): %s" % (path, "; ".join(diffs)))
            if not sub.get("rounds"):
                # a fresh state (a --retry, or a new work root): the rounds,
                # the gate and the clock come back from the folder
                sub["rounds"] = rf.get("rounds", [])
                sub["gate"] = rf.get("gate")
                sub["clock_start_utc"] = rf.get("clock_start_utc")
                sub["ended_by"] = rf.get("ended_by")
                sub["trials"] = {}
                for r in sub["rounds"]:
                    for tid in r["trial_ids"]:
                        sub["trials"][tid] = {"round": r["round"], "jobs": [], "resubmits": 0,
                                              "status": "waiting"}
                sub["round"] = len(sub["rounds"]) - 1
                self._refresh_from_ledger(cfg, st)
                self._revive_killed(cfg, st)
            return rf
        rf = {"schema": "pipeline_rounds/1", "identity": ident, "config_sha256": cfg.sha256,
              "created_utc": CM.utc_now(), "clock_start_utc": None, "gate": None,
              "rounds": [], "ended_by": None}
        CM.write_json(path, rf)
        sub.setdefault("rounds", [])
        sub.setdefault("trials", {})
        sub.setdefault("gate", None)
        sub.setdefault("clock_start_utc", None)
        sub.setdefault("ended_by", None)
        return rf

    def _save_rounds_file(self, cfg, st):
        path = self._dirs(cfg)["rounds_file"]
        rf = CM.read_json(path) if os.path.isfile(path) else {"schema": "pipeline_rounds/1"}
        sub = st["sub"]
        rf.update({"identity": sub["identity"], "config_sha256": cfg.sha256,
                   "clock_start_utc": sub.get("clock_start_utc"), "gate": sub.get("gate"),
                   "rounds": sub.get("rounds", []), "ended_by": sub.get("ended_by"),
                   "updated_utc": CM.utc_now()})
        CM.write_json(path, rf)

    # ------------------------------------------------------------- gate ---

    def _probe_dirs(self, cfg):
        """{batch size: probe dir} from the costliest files already written."""
        out = {}
        for p in sorted(glob.glob(os.path.join(self._dirs(cfg)["gate"], "costliest_b*.json"))):
            b = int(os.path.basename(p)[len("costliest_b"):-5])
            out[b] = os.path.join(self._dirs(cfg)["gate"], "b%d" % b)
        return out

    def _probe_json(self, probe_dir):
        found = sorted(glob.glob(os.path.join(probe_dir, "*_probe.json")))
        return found[0] if found else None

    def _start_gate(self, cfg, st, pbs, log_path):
        """Write the costliest configurations and submit one probe job per
        batch size; reuse a probe file that already exists."""
        d = cfg.data
        s = d["search"]
        dirs = self._dirs(cfg)
        sub = st["sub"]
        ident = sub["identity"]
        CM.run_tuner(cfg, ["costliest"] + CM.tuner_common_args(cfg, ident)
                     + ["--out-dir", dirs["gate"]], log_path=log_path)
        probes = {}
        sim, real = CM.shard_globs(cfg)
        for b, pdir in sorted(self._probe_dirs(cfg).items()):
            existing = self._probe_json(pdir)
            if existing:
                probes[str(b)] = {"job": None, "probe_json": existing, "reused": True}
                continue
            argv_file = os.path.join(dirs["jobs"], "gate_b%d.argv" % b)
            args = (["argv"] + CM.tuner_common_args(cfg, ident)
                    + ["--config-json", os.path.join(dirs["gate"], "costliest_b%d.json" % b),
                       "--sim-shards", sim, "--out-dir", pdir,
                       "--epochs", str(int(s["trial"]["epochs"])),
                       "--passes", repr(float(s["trial"]["passes"])),
                       "--split-fracs"] + [repr(float(f)) for f in d["split"]["fractions"]]
                    + ["--flow-hidden-layers", str(int(s["flow"]["conditioner_hidden_layers"])),
                       "--buffer-shards", str(int(d["loader"]["buffer_shards"])),
                       "--threads", str(int(s["trial"]["ncpus"])),
                       "--patience", "99",
                       "--runner-extra", "--probe-steps %d --probe-val-rows %d"
                       % (int(s["budget_gate"]["probe_steps"]),
                          int(s["budget_gate"]["probe_val_rows"])),
                       "--argv-file", argv_file])
            if real:
                args += ["--real-shards", real]
            CM.run_tuner(cfg, args, log_path=log_path)
            line = CM.qsub_line(cfg, "gate_b%d" % b, s["trial"]["queue"], s["trial"]["ncpus"],
                                s["trial"]["mem"], s["trial"]["walltime"],
                                os.path.join(dirs["jobs"], "gate_b%d.log" % b), argv_file,
                                CM.paths_of(cfg)["stage4"])
            job_id = pbs.submit(line, role="gate_b%d" % b,
                                log_path=os.path.join(dirs["jobs"], "gate_b%d.log" % b))
            probes[str(b)] = {"job": job_id, "probe_json": os.path.join(pdir, "*_probe.json"),
                              "reused": False}
        sub["gate"] = {"probes": probes, "verdict": None, "started_utc": CM.utc_now()}
        sub["phase"] = "gate"

    def _gate_verdict(self, cfg, st):
        """Read every probe file and decide; None while a probe file is
        missing (a probe job ended without one: a failure)."""
        s = cfg.data["search"]
        sub = st["sub"]
        t_wall = CM.walltime_seconds(s["trial"]["walltime"])
        mem_gb = CM.mem_to_gb(s["trial"]["mem"])
        safety = float(s["budget_gate"]["safety_factor"])
        rows = {}
        for b, pr in sorted(sub["gate"]["probes"].items(), key=lambda kv: int(kv[0])):
            pdir = os.path.dirname(pr["probe_json"])
            pj = self._probe_json(pdir)
            if pj is None:
                return {"ok": False, "reason": "probe b%s wrote no *_probe.json in %s (its job "
                                               "failed; see its log)" % (b, pdir), "probes": rows}
            rec = CM.read_json(pj)
            proj = rec.get("projected_trial_seconds")
            rows[b] = {"t_step_s": rec.get("step_seconds"),
                       "t_val_row_s": rec.get("val_row_seconds"),
                       "maxrss_gb": rec.get("maxrss_gb"),
                       "steps_per_epoch": rec.get("steps_per_epoch"),
                       "projected_trial_h": (None if proj is None else proj / 3600.0),
                       "projected_x_safety_h": (None if proj is None
                                                else proj * safety / 3600.0),
                       "file": pj}
        worst_t = max((r["projected_x_safety_h"] or 0.0) for r in rows.values())
        worst_m = max((r["maxrss_gb"] or 0.0) for r in rows.values())
        reasons = []
        if worst_t * 3600.0 > t_wall:
            reasons.append("the costliest configuration projects %.1f h x safety %.2f = %.1f h "
                           "per trial, above search.trial.walltime %s"
                           % (worst_t / safety, safety, worst_t, s["trial"]["walltime"]))
        if worst_m > mem_gb:
            reasons.append("the costliest configuration's peak memory %.1f GB exceeds "
                           "search.trial.mem %s" % (worst_m, s["trial"]["mem"]))
        if any(r["projected_trial_h"] is None for r in rows.values()):
            reasons.append("a probe gave no projection (no validation rows timed)")
        return {"ok": not reasons, "reason": "; ".join(reasons), "probes": rows,
                "worst_projected_x_safety_h": worst_t, "worst_maxrss_gb": worst_m,
                "trial_walltime_h": t_wall / 3600.0, "trial_mem_gb": mem_gb,
                "safety_factor": safety, "decided_utc": CM.utc_now()}

    # ----------------------------------------------------------- rounds ---

    def _pending_ids(self, cfg):
        return set(os.path.basename(p)[:-5]
                   for p in glob.glob(os.path.join(self._dirs(cfg)["pending"], "*.json")))

    def _propose(self, cfg, st, r, log_path):
        """Propose round r; return the NEW trial ids (the pending files the
        call wrote)."""
        s = cfg.data["search"]
        sub = st["sub"]
        before = self._pending_ids(cfg)
        n_points = int(s["n_initial_points"] if r == 0 else s["trials_per_round"])
        args = (["propose"] + CM.tuner_common_args(cfg, sub["identity"])
                + ["--n-points", str(n_points),
                   "--n-initial-points", str(int(s["n_initial_points"])),
                   "--gp-seed", str(int(s["gp_seed"]) + r)])
        if s["gp_noise_sigma"] is not None:
            args += ["--sigma-seed", repr(float(s["gp_noise_sigma"]))]
        CM.run_tuner(cfg, args, log_path=log_path)
        new = sorted(self._pending_ids(cfg) - before)
        return new, len(before)

    def _start_round(self, cfg, st, r, log_path):
        sub = st["sub"]
        new, orphans = self._propose(cfg, st, r, log_path)
        sub["rounds"].append({"round": r, "trial_ids": new, "proposed_utc": CM.utc_now(),
                              "n_requested": int(cfg.data["search"]["n_initial_points"] if r == 0
                                                 else cfg.data["search"]["trials_per_round"]),
                              "orphan_pending_before": orphans})
        for tid in new:
            sub["trials"][tid] = {"round": r, "jobs": [], "resubmits": 0, "status": "waiting"}
        sub["round"] = r
        sub["phase"] = "rounds"
        self._save_rounds_file(cfg, st)
        return new

    def _trial_argv(self, cfg, st, tid, resume):
        d = cfg.data
        s = d["search"]
        dirs = self._dirs(cfg)
        sim, real = CM.shard_globs(cfg)
        p = CM.paths_of(cfg)
        argv = ([CM.E.python_in_env(), p["tuner"], "evaluate"]
                + CM.tuner_common_args(cfg, st["sub"]["identity"])
                + ["--pending-id", tid, "--sim-shards", sim, "--out-dir", dirs["runs"],
                   "--epochs", str(int(s["trial"]["epochs"])),
                   "--passes", repr(float(s["trial"]["passes"])),
                   "--split-fracs"] + [repr(float(f)) for f in d["split"]["fractions"]]
                + ["--flow-hidden-layers", str(int(s["flow"]["conditioner_hidden_layers"])),
                   "--buffer-shards", str(int(d["loader"]["buffer_shards"])),
                   "--threads", str(int(s["trial"]["ncpus"])),
                   "--patience", "99"])
        if real:
            argv += ["--real-shards", real]
        if resume:
            argv += ["--resume"]
        return argv

    def _submit_trial(self, cfg, st, pbs, tid, resume=False):
        s = cfg.data["search"]
        dirs = self._dirs(cfg)
        tr = st["sub"]["trials"][tid]
        n = len(tr["jobs"])
        # D-097: a checkpoint whose zip CRCs fail is moved aside (kept) first,
        # so the job starts from epoch 0 instead of resuming from bad bytes
        CM.set_aside_bad_checkpoints(os.path.join(dirs["runs"], tid), pbs.log, self.name,
                                     "trial %s" % tid[:12])
        # a trial healed of a node fault goes on with --resume (D-096)
        if tr.pop("resume_next", False):
            resume = True
        # a checkpoint left by an earlier job (a --retry after a kill): continue it
        if glob.glob(os.path.join(dirs["runs"], tid, "*_train_ckpt.pt")):
            resume = True
        argv_file = os.path.join(dirs["jobs"], "%s.%d.argv" % (tid, n))
        CM.write_argv_file(argv_file, self._trial_argv(cfg, st, tid, resume))
        log_path = os.path.join(dirs["jobs"], "%s.%d.log" % (tid, n))
        line = CM.qsub_line(cfg, "t_" + tid[:12], s["trial"]["queue"], s["trial"]["ncpus"],
                            s["trial"]["mem"], s["trial"]["walltime"], log_path, argv_file,
                            CM.paths_of(cfg)["stage4"])
        job_id = pbs.submit(line, role="trial %s%s" % (tid[:12], " resume" if resume else ""),
                            log_path=log_path)
        tr["jobs"].append(job_id)
        tr["status"] = "running"
        tr["argv_file"] = argv_file
        if st["sub"].get("clock_start_utc") is None:
            st["sub"]["clock_start_utc"] = CM.utc_now()     # D-090: round 0's first submission
            self._save_rounds_file(cfg, st)
        return job_id

    def _record_path(self, cfg, tid):
        return os.path.join(self._dirs(cfg)["ledger"], tid + ".json")

    def _refresh_from_ledger(self, cfg, st):
        """Trials whose ledger record exists are terminal whatever their
        jobs say (a fresh state after --retry)."""
        for tid, tr in st["sub"]["trials"].items():
            rp = self._record_path(cfg, tid)
            if os.path.isfile(rp):
                rec = CM.read_json(rp)
                tr["status"] = str(rec.get("status", "ok"))
                tr["nll"] = rec.get("nll")

    def _revive_killed(self, cfg, st):
        """A fresh state (a --retry) gives every trial the pipeline recorded
        `killed` another chance: the record is set aside as
        `<id>.json.killed<n>` (kept, out of the ledger's glob), the pending
        spec is rewritten from the record's configuration, and the search
        is re-opened (`ended_by` cleared) so the round completes again."""
        sub = st["sub"]
        dirs = self._dirs(cfg)
        revived = 0
        for tid, tr in sorted(sub["trials"].items()):
            if tr["status"] != "killed":
                continue
            rp = self._record_path(cfg, tid)
            rec = CM.read_json(rp) if os.path.isfile(rp) else {}
            if rec.get("recorded_by") != "pipeline" or not rec.get("config"):
                continue
            n = len(glob.glob(rp + ".killed*")) + 1
            os.replace(rp, "%s.killed%d" % (rp, n))
            pend = os.path.join(dirs["pending"], tid + ".json")
            if not os.path.isfile(pend):
                CM.write_json(pend, {"trial_id": tid, "campaign": rec.get("campaign"),
                                     "config": rec["config"]})
            tr.update({"status": "waiting", "resubmits": 0, "revived": n,
                       "jobs_before": list(rec.get("jobs", []))})
            tr.pop("nll", None)
            revived += 1
        if revived:
            sub["ended_by"] = None
            sub["revived_on_retry"] = sub.get("revived_on_retry", 0) + revived
            self._save_rounds_file(cfg, st)

    def _write_killed_record(self, cfg, st, tid, why):
        """A ledger record with status `killed` (the tuner's loaders drop
        it from the GP's observations and `propose` never re-proposes the
        id); the pending file is removed."""
        dirs = self._dirs(cfg)
        pend = os.path.join(dirs["pending"], tid + ".json")
        cfg_json = CM.read_json(pend).get("config") if os.path.isfile(pend) else None
        rec = {"trial_id": tid, "campaign": cfg.data["search"]["campaign"], "config": cfg_json,
               "status": "killed", "error": why, "tag": "",
               "split_hash": st["sub"]["identity"]["split_hash"],
               "contract_digest": st["sub"]["identity"]["contract_digest"],
               "seeds": [int(cfg.data["search"]["train_seed"])],
               "jobs": list(st["sub"]["trials"][tid]["jobs"]), "recorded_by": "pipeline"}
        CM.write_json(self._record_path(cfg, tid), rec)
        if os.path.isfile(pend):
            os.remove(pend)

    def _finish_trial_job(self, cfg, st, pbs, tid, job, log):
        """The trial's last job ended: ok / failed from its record, or a
        kill: resubmit with --resume, or record it killed."""
        s = cfg.data["search"]
        tr = st["sub"]["trials"][tid]
        rp = self._record_path(cfg, tid)
        if os.path.isfile(rp):
            rec = CM.read_json(rp)
            if str(rec.get("status", "ok")) == "failed" and s["resubmit_node_faults"]:
                # D-096: not final until its evidence says it was not the node's fault
                tr["status"] = "examining"
                tr["examining_since"] = time.time()
                log("search: trial %s failed (%s); examining it for a node fault"
                    % (tid[:12], rec.get("error")))
                self._examine(cfg, st, tid, log)
                return
            tr["status"] = str(rec.get("status", "ok"))
            tr["nll"] = rec.get("nll")
            pend = os.path.join(self._dirs(cfg)["pending"], tid + ".json")
            if os.path.isfile(pend):
                os.remove(pend)
            log("search: trial %s %s%s" % (tid[:12], tr["status"],
                                           (" nll %.6f" % tr["nll"]) if tr.get("nll") is not None
                                           else ""))
            return
        if s["resubmit_killed"] and tr["resubmits"] < int(s["trial"]["max_resubmits"]):
            tr["resubmits"] += 1
            self._submit_trial(cfg, st, pbs, tid, resume=True)
            log("search: trial %s ended without a record (job %s exit %s); resubmitted with "
                "--resume (%d of %d)" % (tid[:12], job["id"], job.get("exit_status"),
                                          tr["resubmits"], int(s["trial"]["max_resubmits"])))
            return
        why = ("killed: %d job(s) ended without a record (last job %s exit %s)"
               % (len(tr["jobs"]), job["id"], job.get("exit_status")))
        self._write_killed_record(cfg, st, tid, why)
        tr["status"] = "killed"
        log("search: trial %s recorded killed after %d resubmission(s)" % (tid[:12], tr["resubmits"]))

    # ------------------------------------------- node faults (D-096) ---

    def _latest_log(self, cfg, tid):
        """(path, text) of the trial's last job log `jobs/<id>.<n>.log`
        (the highest n), text None when there is none or it is unreadable."""
        best, best_n = None, -1
        for p in glob.glob(os.path.join(glob.escape(self._dirs(cfg)["jobs"]), tid + ".*.log")):
            mid = os.path.basename(p)[len(tid) + 1:-len(".log")]
            if mid.isdigit() and int(mid) > best_n:
                best, best_n = p, int(mid)
        return best, CM.read_log_tail(best)

    def _examine(self, cfg, st, tid, log):
        """A trial recorded `failed`: a node fault (the record's SIGBUS, or a
        file-system error in its last job log) is healed; anything else is
        final. Waits for the log's `RUN_ARGV EXIT:` line up to LOG_GRACE_S."""
        sub = st["sub"]
        tr = sub["trials"][tid]
        dirs = self._dirs(cfg)
        rp = self._record_path(cfg, tid)
        pend = os.path.join(dirs["pending"], tid + ".json")
        if not os.path.isfile(rp):
            # the record went away (moved by hand): the trial has none, as a kill
            tr["fault_checked"] = True
            tr["status"] = "waiting" if os.path.isfile(pend) else "failed"
            log("search: trial %s has no ledger record any more; %s" % (
                tid[:12], "queued again" if tr["status"] == "waiting" else "left failed (no pending spec)"))
            return
        rec = CM.read_json(rp)
        if str(rec.get("status", "ok")) != "failed":
            # the record was replaced meanwhile: it decides, as any record
            tr["fault_checked"] = True
            tr.pop("examining_since", None)
            tr["status"] = str(rec.get("status", "ok"))
            tr["nll"] = rec.get("nll")
            return
        log_path, text = self._latest_log(cfg, tid)
        why = CM.node_fault(rec.get("error"), text)
        complete = text is not None and "RUN_ARGV EXIT:" in text
        since = float(tr.get("examining_since") or time.time())
        if why is None and not complete and time.time() - since < CM.LOG_GRACE_S:
            return                       # its log may still be on its way
        tr["fault_checked"] = True
        tr.pop("examining_since", None)
        if why is None:
            tr["status"] = "failed"
            tr["nll"] = rec.get("nll")
            if os.path.isfile(pend):
                os.remove(pend)
            log("search: trial %s failed: not a node fault (%s); final" % (tid[:12], rec.get("error")))
            return
        self._heal(cfg, st, tid, rec, why, CM.host_of_log(text) or "?", log)

    def _heal(self, cfg, st, tid, rec, why, host, log):
        """D-096: the record set aside (kept), the pending spec back, and the
        trial queued again with --resume while resubmissions are left, else
        the pipeline's `killed` record (revivable by --retry search)."""
        s = cfg.data["search"]
        sub = st["sub"]
        tr = sub["trials"][tid]
        dirs = self._dirs(cfg)
        rp = self._record_path(cfg, tid)
        pend = os.path.join(dirs["pending"], tid + ".json")
        if not os.path.isfile(pend):
            if not rec.get("config"):
                tr["status"] = "failed"
                log("search: trial %s NODE FAULT on %s (%s), but its record has no configuration "
                    "and no pending spec is left: final" % (tid[:12], host, why))
                return
            CM.write_json(pend, {"trial_id": tid, "campaign": rec.get("campaign"),
                                 "config": rec["config"]})
        k = len(glob.glob(glob.escape(rp) + ".nodefault*")) + 1
        aside = "%s.nodefault%d" % (rp, k)
        os.replace(rp, aside)
        tr.setdefault("node_faults", []).append({"host": host, "why": why,
                                                 "record": os.path.basename(aside),
                                                 "jobs": list(tr["jobs"]), "utc": CM.utc_now()})
        sub.setdefault("node_fault_hosts", {})
        sub["node_fault_hosts"][host] = sub["node_fault_hosts"].get(host, 0) + 1
        tr.pop("nll", None)
        n_host = sub["node_fault_hosts"][host]
        if tr["resubmits"] < int(s["trial"]["max_resubmits"]):
            tr["resubmits"] += 1
            tr["status"] = "waiting"
            tr["resume_next"] = True
            log("search: trial %s NODE FAULT on %s (%s); record set aside as %s; queued again with "
                "--resume (%d of %d); %d node fault(s) on %s so far"
                % (tid[:12], host, why, os.path.basename(aside), tr["resubmits"],
                   int(s["trial"]["max_resubmits"]), n_host, host))
            return
        self._write_killed_record(cfg, st, tid, "killed: node fault on %s (%s) after %d "
                                                "resubmission(s)" % (host, why, tr["resubmits"]))
        tr["status"] = "killed"
        log("search: trial %s NODE FAULT on %s (%s); record set aside as %s; no resubmission "
            "left: recorded killed; %d node fault(s) on %s so far"
            % (tid[:12], host, why, os.path.basename(aside), n_host, host))

    def _sweep_failed(self, cfg, st, log):
        """Every poll: the trials still being examined, and those recorded
        `failed` without having been examined (before this code existed, or
        restored by a fresh state)."""
        for tid, tr in sorted(st["sub"]["trials"].items()):
            if tr["status"] == "failed" and not tr.get("fault_checked"):
                tr["status"] = "examining"
                tr["examining_since"] = time.time()
                log("search: trial %s is recorded failed and was never examined for a node "
                    "fault; examining it" % tid[:12])
            if tr["status"] == "examining":
                self._examine(cfg, st, tid, log)

    def _round_complete(self, st):
        sub = st["sub"]
        r = sub["round"]
        ids = sub["rounds"][r]["trial_ids"] if r < len(sub["rounds"]) else []
        return all(sub["trials"][t]["status"] in TERMINAL for t in ids)

    def _counts(self, st):
        c = {"ok": 0, "failed": 0, "killed": 0, "running": 0, "waiting": 0}
        for tr in st["sub"]["trials"].values():
            c[tr["status"]] = c.get(tr["status"], 0) + 1
        return c

    def _finish(self, cfg, st, ended_by, log_path, log):
        """finalists --top-k 1, then SEARCH_COMPLETE.json."""
        sub = st["sub"]
        dirs = self._dirs(cfg)
        counts = self._counts(st)
        sub["ended_by"] = ended_by
        self._save_rounds_file(cfg, st)
        if counts["ok"] == 0:
            sub["phase"] = "failed"
            sub["fail"] = ("the search ended (%s) with no completed trial: %d failed, %d killed; "
                           "see %s" % (ended_by, counts["failed"], counts["killed"], dirs["ledger"]))
            return
        try:
            CM.run_tuner(cfg, ["finalists"] + CM.tuner_common_args(cfg, sub["identity"])
                         + ["--top-k", "1", "--rank-split", "report", "--gate-split", "report"],
                         log_path=log_path)
        except CM.ToolError as exc:
            sub["phase"] = "failed"
            sub["fail"] = "finalists failed: %s" % exc
            return
        fin = CM.read_json(dirs["finalists"])
        best = fin["finalists"][0]
        elapsed = (CM.elapsed_hours(sub["clock_start_utc"]) if sub.get("clock_start_utc")
                   else None)
        rec = {"schema": "search_complete/1", "campaign": cfg.data["search"]["campaign"],
               "ended_by": ended_by, "rounds_evaluated": len(sub["rounds"]),
               "gp_rounds_configured": int(cfg.data["search"]["gp_rounds"]),
               "n_trials": counts, "elapsed_h": elapsed,
               "max_total_walltime_h": cfg.data["search"]["max_total_walltime_h"],
               "clock_start_utc": sub.get("clock_start_utc"), "finished_utc": CM.utc_now(),
               "finalists": dirs["finalists"],
               "finalist": {"trial_id": best["trial_id"], "nll": best["nll_rank_split"],
                            "config": best["config"]},
               "identity": sub["identity"], "gate": sub.get("gate"),
               "rounds_file": dirs["rounds_file"], "config_sha256": cfg.sha256}
        CM.write_json(dirs["complete"], rec)
        sub["phase"] = "complete"
        log("search: SEARCH COMPLETE (%s): %d round(s), %d ok / %d failed / %d killed, "
            "finalist %s nll %.6f" % (ended_by, len(sub["rounds"]), counts["ok"],
                                      counts["failed"], counts["killed"], best["trial_id"][:12],
                                      best["nll_rank_split"]))

    # -------------------------------------------------------- interface ---

    def _log_path(self, cfg):
        return os.path.join(self._dirs(cfg)["jobs"], "login_node.log")

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        log_path = self._log_path(cfg)
        self._identity(cfg, st, log_path)
        self._rounds_file(cfg, st)
        self._refresh_from_ledger(cfg, st)
        if sub.get("ended_by"):
            # the folder says the search ended; only the finish is missing
            sub["phase"] = "finishing"
            return
        if sub.get("gate") and sub["gate"].get("verdict", {}) and sub["gate"]["verdict"].get("ok"):
            sub["phase"] = "rounds"
            if not sub["rounds"]:
                self._start_round(cfg, st, 0, log_path)
            return
        self._start_gate(cfg, st, pbs, log_path)

    def poll(self, cfg, st, pbs):
        sub = st["sub"]
        log = pbs.log
        log_path = self._log_path(cfg)
        s = cfg.data["search"]

        # the jobs' states, through the sequencer's records
        finished_now = []
        for job in st["jobs"]:
            if job.get("finished"):
                continue
            js = pbs.status(job["id"])
            if js.finished:
                job["finished"] = True
                job["exit_status"] = js.exit_status
                job["walltime_s"] = js.walltime_s
                job["state"] = js.state
                finished_now.append(job)
        by_id = dict((j["id"], j) for j in st["jobs"])

        phase = sub.get("phase")
        if phase == "gate":
            probes = sub["gate"]["probes"]
            if any(pr["job"] and not by_id[pr["job"]].get("finished") for pr in probes.values()):
                return "running"
            verdict = self._gate_verdict(cfg, st)
            sub["gate"]["verdict"] = verdict
            self._save_rounds_file(cfg, st)
            if not verdict["ok"]:
                sub["phase"] = "failed"
                sub["fail"] = "budget gate: " + verdict["reason"]
                return "finished"
            log("search: budget gate passed: worst projected %.2f h x %.2f = %.2f h <= %s; "
                "peak %.2f GB <= %s"
                % (verdict["worst_projected_x_safety_h"] / verdict["safety_factor"],
                   verdict["safety_factor"], verdict["worst_projected_x_safety_h"],
                   s["trial"]["walltime"], verdict["worst_maxrss_gb"], s["trial"]["mem"]))
            self._start_round(cfg, st, 0, log_path)
            phase = "rounds"

        if phase == "rounds":
            # trials whose job ended
            for tid, tr in sorted(sub["trials"].items()):
                if tr["status"] != "running" or not tr["jobs"]:
                    continue
                job = by_id.get(tr["jobs"][-1])
                if job is not None and job.get("finished"):
                    self._finish_trial_job(cfg, st, pbs, tid, job, log)
            # D-096: failed trials awaiting their log, and failed trials never examined
            if s["resubmit_node_faults"]:
                self._sweep_failed(cfg, st, log)
            # the throttle: fill the free slots with waiting trials, earliest round first
            # (a revived trial of an earlier round after a --retry comes first)
            running = sum(1 for tr in sub["trials"].values() if tr["status"] == "running")
            r = sub["round"]
            for rr in sub["rounds"][:r + 1]:
                for tid in rr["trial_ids"]:
                    if running >= int(s["max_concurrent_trials"]):
                        break
                    if sub["trials"][tid]["status"] == "waiting":
                        self._submit_trial(cfg, st, pbs, tid)
                        running += 1
            if not all(sub["trials"][t_]["status"] in TERMINAL
                       for rr in sub["rounds"][:r + 1] for t_ in rr["trial_ids"]):
                return "running"
            counts = self._counts(st)
            elapsed = CM.elapsed_hours(sub["clock_start_utc"]) if sub.get("clock_start_utc") else 0.0
            log("search: round %d complete (%d ok / %d failed / %d killed in all; %.3f h elapsed)"
                % (r, counts["ok"], counts["failed"], counts["killed"], elapsed))
            cap = s["max_total_walltime_h"]
            if r >= int(s["gp_rounds"]):
                self._finish(cfg, st, "rounds", log_path, log)
                return "finished"
            if cap is not None and elapsed >= float(cap):
                log("search: the elapsed clock %.3f h reached max_total_walltime_h %s: no new "
                    "round (D-090)" % (elapsed, cap))
                self._finish(cfg, st, "budget", log_path, log)
                return "finished"
            new = self._start_round(cfg, st, r + 1, log_path)
            if not new:
                log("search: round %d proposed nothing new; the search ends" % (r + 1))
                sub["rounds"].pop()
                self._finish(cfg, st, "no_new_proposals", log_path, log)
                return "finished"
            log("search: round %d proposed: %d trial(s)" % (r + 1, len(new)))
            return "running"

        if phase == "finishing":
            self._finish(cfg, st, sub["ended_by"], log_path, log)
            return "finished"
        if phase in ("complete", "failed"):
            return "finished"
        raise RuntimeError("search stage in unknown phase %r" % (phase,))

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

    def dry_run(self, cfg, st):
        d = cfg.data
        s = d["search"]
        dirs = self._dirs(cfg)
        p = CM.paths_of(cfg)

        def line_of(name, log, argv_file):
            q = CM.qsub_line(cfg, name, s["trial"]["queue"], s["trial"]["ncpus"], s["trial"]["mem"],
                             s["trial"]["walltime"], log, argv_file, p["stage4"])
            q[2] = name            # the placeholder, not its sanitised form
            return " ".join(q)

        n_total = int(s["n_initial_points"]) + int(s["gp_rounds"]) * int(s["trials_per_round"])
        sim, real = CM.shard_globs(cfg)
        lines = ["identity: %s --sim-shards %s --split-fracs %s --split-seed %d (envs.train, login node)"
                 % (p["split_tool"], sim, " ".join(str(f) for f in d["split"]["fractions"]),
                    int(s["train_seed"])),
                 "gate: %s costliest ... then one probe job per searched batch size "
                 "(%d steps, %d validation rows, safety %.2f against walltime %s and mem %s):"
                 % (p["tuner"], int(s["budget_gate"]["probe_steps"]),
                    int(s["budget_gate"]["probe_val_rows"]), float(s["budget_gate"]["safety_factor"]),
                    s["trial"]["walltime"], s["trial"]["mem"]),
                 "    " + line_of("gate_b<B>", os.path.join(dirs["jobs"], "gate_b<B>.log"),
                                  os.path.join(dirs["jobs"], "gate_b<B>.argv")),
                 "rounds: round 0 = %d random configurations, then %d GP round(s) of %d "
                 "(%d trials in all), at most %d trial jobs at once, each:"
                 % (int(s["n_initial_points"]), int(s["gp_rounds"]), int(s["trials_per_round"]),
                    n_total, int(s["max_concurrent_trials"])),
                 "    " + line_of("t_<trial>", os.path.join(dirs["jobs"], "<trial>.<n>.log"),
                                  os.path.join(dirs["jobs"], "<trial>.<n>.argv")),
                 "    argv: %s %s evaluate --campaign %s --results-dir %s --pending-id <trial> "
                 "--sim-shards %s%s --out-dir %s --epochs %d --passes %g --split-fracs %s "
                 "--flow-hidden-layers %d --buffer-shards %d --threads %d --patience 99 "
                 "--space-overrides ... [--resume on a resubmission, at most %d]"
                 % (CM.E.python_in_env(), p["tuner"], s["campaign"], d["paths"]["tune_root"], sim,
                    (" --real-shards " + real) if real else "", dirs["runs"],
                    int(s["trial"]["epochs"]), float(s["trial"]["passes"]),
                    " ".join(str(f) for f in d["split"]["fractions"]),
                    int(s["flow"]["conditioner_hidden_layers"]), int(d["loader"]["buffer_shards"]),
                    int(s["trial"]["ncpus"]), int(s["trial"]["max_resubmits"])),
                 "node faults (D-096): %s; checkpoints whose zip CRCs fail are moved aside "
                 "before a submission (D-097)"
                 % ("a trial failed of SIGBUS or a file-system error is queued again with --resume "
                    "(within max_resubmits), its record set aside" if s["resubmit_node_faults"]
                    else "off (search.resubmit_node_faults false): a failed trial is final"),
                 "clock: elapsed from round 0's first submission, cap %s h (D-090)"
                 % (s["max_total_walltime_h"],),
                 "end: finalists --top-k 1 -> %s; artefact %s" % (dirs["finalists"], dirs["complete"])]
        return lines
