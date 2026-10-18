"""Stage S8, the best fit: the search's finalist trained as an ensemble
of M members on one shared donor split, and the ensemble's report-split
NLL (block 7; D-089; specs/SPEC.md sec. 2.3).

    members   one job per seed in `best_fit.member_seeds` (the runner
              through the tuner's `argv`, so the finalist's configuration
              reaches the runner by the same translation as in the search),
              `--split-seed best_fit.split_seed`, `--passes`, `--epochs`,
              `--patience`, `--n-posterior-draws` (diagnostics), the
              resources of `best_fit`; a member whose job ends with no
              record is resubmitted with --resume up to `max_resubmits`
              times, then the stage fails (every member is needed); a
              member that failed of a node fault (SIGBUS or a file-system
              error in its log) is resubmitted the same way instead of
              failing as a refusal (D-096, `best_fit.resubmit_node_faults`);
              a checkpoint whose zip CRCs fail is moved aside before a
              submission (D-097); a member whose record exists when the
              stage starts is not re-run (resumable)
    ensemble  `tools/ensemble_nll.py` over the members' per-row files:
              L_bar = -(1/n) sum_i [logsumexp_m(-l_{m,i}) - log M]
    artefact  `<runs_root>/<tag>_best/BEST_FIT_COMPLETE.json`

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import glob
import os

from . import DEPS, Stage
from . import common as CM

COMPLETE_FILE = "BEST_FIT_COMPLETE.json"


class BestFitStage(Stage):
    name = "best_fit"
    deps = DEPS["best_fit"]
    max_resubmits = 0           # the members carry their own resubmissions

    # ------------------------------------------------------------ paths ---

    def _dirs(self, cfg):
        d = cfg.data
        root = os.path.join(d["paths"]["runs_root"], d["dataset"] + "_best")
        return {"root": root, "jobs": os.path.join(root, "jobs"),
                "finalist_config": os.path.join(root, "finalist_config.json"),
                "ensemble": os.path.join(root, "ensemble_nll.json"),
                "complete": os.path.join(root, COMPLETE_FILE),
                "search_complete": os.path.join(d["paths"]["tune_root"], d["search"]["campaign"],
                                                "SEARCH_COMPLETE.json")}

    def _member_dir(self, cfg, seed):
        return os.path.join(self._dirs(cfg)["root"], "member_seed%d" % int(seed))

    def _member_record(self, cfg, seed):
        """The runner's record of a member, or None: `<stem>.json` with the
        stem ending `_seed<k>` (block 6's naming); the per-row file beside it."""
        found = sorted(glob.glob(os.path.join(self._member_dir(cfg, seed),
                                              "*_seed%d.json" % int(seed))))
        return found[0] if found else None

    def _member_perrow(self, cfg, seed):
        found = sorted(glob.glob(os.path.join(self._member_dir(cfg, seed),
                                              "*_seed%d_perrow.npz" % int(seed))))
        return found[0] if found else None

    # --------------------------------------------------------- finalist ---

    def _finalist(self, cfg, st):
        """The finalist's configuration from SEARCH_COMPLETE.json, written
        to the best-fit folder in the tuner's --config-json shape."""
        sub = st["sub"]
        dirs = self._dirs(cfg)
        if sub.get("finalist"):
            return sub["finalist"]
        sc = CM.read_json(dirs["search_complete"])
        fin = sc["finalist"]
        CM.write_json(dirs["finalist_config"],
                      {"trial_id": fin["trial_id"], "campaign": sc["campaign"],
                       "nll_search": fin["nll"], "config": fin["config"],
                       "search_complete": dirs["search_complete"]})
        sub["finalist"] = {"trial_id": fin["trial_id"], "nll_search": fin["nll"],
                           "identity": sc["identity"], "config_file": dirs["finalist_config"]}
        sub.setdefault("members", {})
        return sub["finalist"]

    # ---------------------------------------------------------- members ---

    def _member_argv_file(self, cfg, st, seed, resume, log_path):
        """The runner's argv of one member through the tuner's `argv`."""
        d = cfg.data
        b = d["best_fit"]
        s = d["search"]
        sub = st["sub"]
        dirs = self._dirs(cfg)
        sim, real = CM.shard_globs(cfg)
        n = len(sub["members"][str(seed)]["jobs"])
        if not os.path.isdir(dirs["jobs"]):
            os.makedirs(dirs["jobs"])
        argv_file = os.path.join(dirs["jobs"], "member_seed%d.%d.argv" % (int(seed), n))
        common = CM.tuner_common_args(cfg, sub["finalist"]["identity"])
        # the member's training seed replaces the search's --seed
        i = common.index("--seed")
        common[i + 1] = str(int(seed))
        args = (["argv"] + common
                + ["--config-json", sub["finalist"]["config_file"],
                   "--sim-shards", sim, "--out-dir", self._member_dir(cfg, seed),
                   "--epochs", str(int(b["epochs"])), "--passes", repr(float(b["passes"])),
                   "--patience", str(int(b["patience_epochs"])),
                   "--split-seed", str(int(b["split_seed"])),
                   "--split-fracs"] + [repr(float(f)) for f in d["split"]["fractions"]]
                + ["--flow-hidden-layers", str(int(s["flow"]["conditioner_hidden_layers"])),
                   "--buffer-shards", str(int(d["loader"]["buffer_shards"])),
                   "--threads", str(int(b["ncpus"])),
                   "--runner-extra", "--n-posterior-draws %d" % int(b["n_posterior_draws"]),
                   "--argv-file", argv_file])
        if real:
            args += ["--real-shards", real]
        if resume:
            args += ["--resume"]
        CM.run_tuner(cfg, args, log_path=log_path)
        return argv_file

    def _submit_member(self, cfg, st, pbs, seed, resume=False):
        b = cfg.data["best_fit"]
        dirs = self._dirs(cfg)
        m = st["sub"]["members"][str(seed)]
        # D-097: a checkpoint whose zip CRCs fail is moved aside (kept) first
        CM.set_aside_bad_checkpoints(self._member_dir(cfg, seed), pbs.log, self.name,
                                     "member seed %d" % int(seed))
        # a checkpoint left by an earlier job (a --retry after a kill): continue it
        if glob.glob(os.path.join(self._member_dir(cfg, seed), "*_train_ckpt.pt")):
            resume = True
        argv_file = self._member_argv_file(cfg, st, seed, resume, self._log_path(cfg))
        n = len(m["jobs"])
        log_path = os.path.join(dirs["jobs"], "member_seed%d.%d.log" % (int(seed), n))
        line = CM.qsub_line(cfg, "bf_%s_s%d" % (cfg.data["dataset"][:6], int(seed)), b["queue"],
                            b["ncpus"], b["mem"], b["walltime"], log_path, argv_file,
                            CM.paths_of(cfg)["stage4"])
        job_id = pbs.submit(line, role="member seed %d%s" % (int(seed), " resume" if resume else ""),
                            log_path=log_path)
        m["jobs"].append(job_id)
        m["status"] = "running"
        m["argv_file"] = argv_file
        return job_id

    def _finish_member_job(self, cfg, st, pbs, seed, job, log):
        b = cfg.data["best_fit"]
        m = st["sub"]["members"][str(seed)]
        rec = self._member_record(cfg, seed)
        if rec is not None:
            m["status"] = "ok"
            m["record"] = rec
            m["L"] = CM.read_json(rec).get("L")
            log("best_fit: member seed %d done: L %s (%s)" % (int(seed), m["L"], rec))
            return
        if b["resubmit_node_faults"]:
            # D-096 applied to the members: a node fault is not a refusal
            text = CM.read_log_tail(job.get("log"))
            why = CM.node_fault(None, text)
            if why is not None:
                host = CM.host_of_log(text) or "?"
                m.setdefault("node_faults", []).append({"host": host, "why": why, "job": job["id"],
                                                        "utc": CM.utc_now()})
                if m["resubmits"] < int(b["max_resubmits"]):
                    m["resubmits"] += 1
                    self._submit_member(cfg, st, pbs, seed, resume=True)
                    log("best_fit: member seed %d NODE FAULT on %s (%s); resubmitted with --resume "
                        "(%d of %d)" % (int(seed), host, why, m["resubmits"], int(b["max_resubmits"])))
                    return
                m["status"] = "failed"
                m["why"] = ("member seed %d: node fault on %s (%s) after %d resubmission(s); raise "
                            "best_fit.max_resubmits, then --retry best_fit (finished members are "
                            "kept)" % (int(seed), host, why, m["resubmits"]))
                log("best_fit: member seed %d NODE FAULT on %s (%s); no resubmission left"
                    % (int(seed), host, why))
                return
        if job.get("exit_status") not in (None, 0) and self._refused(job):
            m["status"] = "failed"
            m["why"] = "member seed %d refused (job %s exit %s); see %s" % (
                int(seed), job["id"], job.get("exit_status"), job.get("log"))
            return
        if m["resubmits"] < int(b["max_resubmits"]):
            m["resubmits"] += 1
            self._submit_member(cfg, st, pbs, seed, resume=True)
            log("best_fit: member seed %d ended without a record (job %s exit %s); resubmitted "
                "with --resume (%d of %d)" % (int(seed), job["id"], job.get("exit_status"),
                                               m["resubmits"], int(b["max_resubmits"])))
            return
        m["status"] = "failed"
        m["why"] = ("member seed %d: %d job(s) ended without a record (last %s exit %s); "
                    "raise best_fit.max_resubmits or fix the cause, then --retry best_fit "
                    "(finished members are kept)" % (int(seed), len(m["jobs"]), job["id"],
                                                     job.get("exit_status")))

    def _refused(self, job):
        """A member job that printed a refusal (a traceback or a REFUSE/
        FATAL line) rather than being killed: read from its log."""
        p = job.get("log")
        if not p or not os.path.isfile(p):
            return False
        try:
            with open(p, "rb") as fh:
                tail = fh.read()[-20000:].decode("ascii", "replace")
        except OSError:
            return False
        return any(k in tail for k in ("Traceback (most recent call last)", "REFUSE", "FATAL",
                                        "usage: run_joint_arms"))

    # --------------------------------------------------------- ensemble ---

    def _ensemble(self, cfg, st, log_path, log):
        d = cfg.data
        b = d["best_fit"]
        sub = st["sub"]
        dirs = self._dirs(cfg)
        p = CM.paths_of(cfg)
        files = []
        for seed in b["member_seeds"]:
            pr = self._member_perrow(cfg, seed)
            if pr is None:
                sub["fail"] = "member seed %d has a record but no per-row file" % int(seed)
                return False
            files.append(pr)
        argv = [CM.E.python_in_env(), p["ensemble_tool"], "--out", dirs["ensemble"],
                "--members"] + files
        rc, out = CM.E.run_in_env(d["envs"]["train"], argv, log_path=log_path)
        if rc != 0 or not os.path.isfile(dirs["ensemble"]):
            sub["fail"] = "ensemble NLL failed (exit %d): %s" % (rc, out.strip()[-1500:])
            return False
        ens = CM.read_json(dirs["ensemble"])
        members = []
        for seed in b["member_seeds"]:
            m = sub["members"][str(seed)]
            members.append({"seed": int(seed), "record": m.get("record"), "L": m.get("L"),
                            "jobs": m["jobs"], "resubmits": m["resubmits"],
                            "dir": self._member_dir(cfg, seed)})
        rec = {"schema": "best_fit_complete/1", "dataset": d["dataset"],
               "finalist": sub["finalist"], "split_seed": int(b["split_seed"]),
               "split_fracs": list(d["split"]["fractions"]),
               "member_seeds": [int(x) for x in b["member_seeds"]], "members": members,
               "ensemble": {"M": ens["M"], "n_report": ens["n_report"], "L_bar": ens["L_bar"],
                            "L_members": ens["L_members"], "file": dirs["ensemble"],
                            "formula": ens["formula"]},
               "passes": float(b["passes"]), "epochs": int(b["epochs"]),
               "patience_epochs": int(b["patience_epochs"]),
               "n_posterior_draws": int(b["n_posterior_draws"]),
               "finished_utc": CM.utc_now(), "config_sha256": cfg.sha256}
        CM.write_json(dirs["complete"], rec)
        log("best_fit: BEST FIT COMPLETE: M=%d members, ensemble L_bar %.6f nats/row (members %s)"
            % (ens["M"], ens["L_bar"], ", ".join("%.4f" % x for x in ens["L_members"])))
        return True

    # -------------------------------------------------------- interface ---

    def _log_path(self, cfg):
        return os.path.join(self._dirs(cfg)["jobs"], "login_node.log")

    def start(self, cfg, st, pbs):
        b = cfg.data["best_fit"]
        sub = st["sub"]
        dirs = self._dirs(cfg)
        if not os.path.isfile(dirs["search_complete"]):
            raise RuntimeError("no %s: the search has not completed" % dirs["search_complete"])
        self._finalist(cfg, st)
        for seed in b["member_seeds"]:
            m = sub["members"].setdefault(str(seed), {"jobs": [], "resubmits": 0,
                                                       "status": "waiting"})
            rec = self._member_record(cfg, seed)
            if rec is not None:
                m["status"] = "ok"
                m["record"] = rec
                m["L"] = CM.read_json(rec).get("L")
                pbs.log("best_fit: member seed %d already has a record: %s" % (int(seed), rec))
            elif m["status"] in ("waiting", "failed"):
                self._submit_member(cfg, st, pbs, seed, resume=bool(m["jobs"]))
        sub["phase"] = "members"

    def poll(self, cfg, st, pbs):
        b = cfg.data["best_fit"]
        sub = st["sub"]
        log = pbs.log
        for job in st["jobs"]:
            if job.get("finished"):
                continue
            js = pbs.status(job["id"])
            if js.finished:
                job["finished"] = True
                job["exit_status"] = js.exit_status
                job["walltime_s"] = js.walltime_s
                job["state"] = js.state
        by_id = dict((j["id"], j) for j in st["jobs"])
        if sub.get("phase") == "members":
            for seed in b["member_seeds"]:
                m = sub["members"][str(seed)]
                if m["status"] != "running" or not m["jobs"]:
                    continue
                job = by_id.get(m["jobs"][-1])
                if job is not None and job.get("finished"):
                    self._finish_member_job(cfg, st, pbs, seed, job, log)
            statuses = [sub["members"][str(s)]["status"] for s in b["member_seeds"]]
            if any(s_ == "running" for s_ in statuses):
                return "running"
            failed = [sub["members"][str(s)] for s in b["member_seeds"]
                      if sub["members"][str(s)]["status"] == "failed"]
            if failed:
                sub["phase"] = "failed"
                sub["fail"] = "; ".join(m.get("why", "member failed") for m in failed)
                return "finished"
            sub["phase"] = "ensemble"
        if sub.get("phase") == "ensemble":
            ok = self._ensemble(cfg, st, self._log_path(cfg), log)
            sub["phase"] = "complete" if ok else "failed"
            return "finished"
        return "finished"

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
        b = d["best_fit"]
        dirs = self._dirs(cfg)
        p = CM.paths_of(cfg)
        sim, real = CM.shard_globs(cfg)
        q = CM.qsub_line(cfg, "bf_<tag>_s<k>", b["queue"], b["ncpus"], b["mem"], b["walltime"],
                         os.path.join(dirs["jobs"], "member_seed<k>.<n>.log"),
                         os.path.join(dirs["jobs"], "member_seed<k>.<n>.argv"), p["stage4"])
        q[2] = "bf_%s_s<k>" % d["dataset"][:6]
        return ["finalist: the lowest report-split nll of %s" % dirs["search_complete"],
                "members: one job per seed %s, split seed %d, passes %g in %d epochs, patience %d:"
                % (b["member_seeds"], int(b["split_seed"]), float(b["passes"]), int(b["epochs"]),
                   int(b["patience_epochs"])),
                "    " + " ".join(q),
                "    argv: %s %s --arm <finalist's arm> --sim-shards %s%s --out-dir %s/member_seed<k> "
                "--seed <k> --split-seed %d --split-fracs %s --epochs %d --passes %g --patience %d "
                "--buffer-shards %d --threads %d --flow-hidden-layers %d <the finalist's axes> "
                "--n-posterior-draws %d [--resume on a resubmission, at most %d]"
                % (CM.E.python_in_env(), p["runner"], sim, (" --real-shards " + real) if real else "",
                   dirs["root"], int(b["split_seed"]),
                   " ".join(str(f) for f in d["split"]["fractions"]), int(b["epochs"]),
                   float(b["passes"]), int(b["patience_epochs"]), int(d["loader"]["buffer_shards"]),
                   int(b["ncpus"]), int(d["search"]["flow"]["conditioner_hidden_layers"]),
                   int(b["n_posterior_draws"]), int(b["max_resubmits"])),
                "node faults (D-096): %s; checkpoints whose zip CRCs fail are moved aside "
                "before a submission (D-097)"
                % ("a member failed of SIGBUS or a file-system error is resubmitted with --resume "
                   "(within max_resubmits)" if b["resubmit_node_faults"]
                   else "off (best_fit.resubmit_node_faults false): such a member fails the stage"),
                "ensemble: %s --members <M per-row files> -> %s (L_bar, D-089)"
                % (p["ensemble_tool"], dirs["ensemble"]),
                "artefact: %s" % dirs["complete"]]
