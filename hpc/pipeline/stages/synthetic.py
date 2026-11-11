"""The synthetic kind (blocks 15, 16; specs/SPEC.md secs. 2.1, 3.1 and 5):
a data set whose banks the bench generator builds from the configuration,
searched by S7 and fitted by S8 once per entry of `synthetic.searches`.

The graph (synthetic_set_graph), in this order
    sim_bank            arm S, the simulated bank: K shards, one job each,
                        then the record job (block 16)
    real_bank_<r>       one per entry r of `synthetic.real`: arm R's cohort,
                        one shard, then the record job (block 16)
    search_<q>          one per entry q of `synthetic.searches`: S7 on q's
                        view of the configuration, after sim_bank and
                        real_bank_<q.real> (block 15)
    best_fit_<q>        S8 on the same view, after search_<q>
    best_fit_alt_<q>    [block 18, D-126] for an entry with alt_best_fit
                        true: S8 on the same view for the best trial of
                        the arm the search's finalist is not, in
                        <runs_root>/<q>/<tag>_best_alt, after search_<q>

A run's view (RunView) is the configuration with `paths.tune_root` and
`paths.runs_root` joined with the run's name, `search.campaign` its
campaign and `synthetic.run` the run itself, which common.shard_globs reads
to give the runner the simulated bank and the run's real bank (D-122: for
every campaign, S-A1 included). S7's and S8's code runs on the view
unchanged; their log lines carry the run stage's name.

What runs where: the sequencer's process (the stdlib only) writes the argv
files and submits; every bank job is `jobs/run_argv.pbs` in `envs.train`:
`hpc/joint/stage1/build_latent_bank.py` for a shard, `tools/
synth_bank_record.py` for the record.

Nothing here names a project (D-091): every value comes from the
configuration.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import copy
import glob
import os
import re

import config as C

from . import Stage, artefact_done, artefact_of

if not hasattr(C, "SCHEMA_SYNTHETIC"):       # another config.py earlier on sys.path
    raise ImportError("stages.synthetic imported %s as `config`, not the pipeline's config.py: "
                      "import the pipeline's config first" % getattr(C, "__file__", "?"))
from . import common as CM
from .best_fit import BestFitStage
from .search import SearchStage

__all__ = ["RunView", "RunSearchStage", "RunBestFitStage", "RunAltBestFitStage",
           "SynthBankStage", "build_of", "bank_dir", "synthetic_set_graph", "SIM_BANK",
           "RECORD_FILE"]

SIM_BANK = "sim_bank"
RECORD_FILE = "BANK_RECORD.json"
BUILD_SCHEMA = "synth_bank_build/1"
_SHARD_FILE = re.compile(r"^shard_([0-9]+)\.(npz|json)$")


# ---------------------------------------------------------------------------
# A run's view of the configuration (block 15)
# ---------------------------------------------------------------------------

class RunView(C.Config):
    """The configuration as one search of a synthetic data set sees it: a
    copy of `base.data` with `paths.tune_root` and `paths.runs_root` joined
    with run["name"], `search.campaign` = run["campaign"] and
    `synthetic.run` = run; `base`'s path and sha256."""

    def __init__(self, base, run):
        if isinstance(base, RunView):
            raise ValueError("RunView of a RunView (%s of %s)" % (run.get("name"), base.run["name"]))
        data = copy.deepcopy(base.data)
        paths = data["paths"]
        paths["tune_root"] = os.path.join(paths["tune_root"], run["name"])
        paths["runs_root"] = os.path.join(paths["runs_root"], run["name"])
        data["search"]["campaign"] = run["campaign"]
        data["synthetic"]["run"] = dict(run)
        C.Config.__init__(self, data, base.path, base.sha256)
        self.base = base
        self.run = dict(run)


class _StageLog(object):
    """The sequencer's recorder (pipeline.Submitter) with S7's and S8's log
    lines renamed: a line starting `search: ` (`best_fit: `) starts with the
    run stage's name instead; everything else is the recorder's."""

    def __init__(self, pbs, old, new):
        self._pbs = pbs
        self._old = old + ": "
        self._new = new + ": "

    def log(self, text):
        t = str(text)
        if t.startswith(self._old):
            t = self._new + t[len(self._old):]
        self._pbs.log(t)

    def __getattr__(self, name):
        return getattr(self._pbs, name)


class _RunStage(object):
    """What the run stages share: the view (cached; a view is returned as
    it is, since S7's and S8's methods call is_done with it) and the
    renamed recorder."""

    base_name = ""
    log_name = None             # [block 18] the prefix of the lines renamed; None: base_name

    def _init_run(self, run, prefix):
        self.run = dict(run)
        self.name = prefix + run["name"]
        self._cache = None

    def _view(self, cfg):
        if isinstance(cfg, RunView):
            if cfg.run["name"] != self.run["name"]:
                raise ValueError("stage %s given the view of run %s" % (self.name, cfg.run["name"]))
            return cfg
        hit = self._cache
        if hit is not None and hit[0] is cfg:
            return hit[1]
        view = RunView(cfg, self.run)
        self._cache = (cfg, view)
        return view

    def _rec(self, pbs):
        return _StageLog(pbs, self.log_name or self.base_name, self.name)

    def is_done(self, cfg):
        kind, path = artefact_of(self.base_name, self._view(cfg))
        return artefact_done(kind, path)


class RunSearchStage(_RunStage, SearchStage):
    """S7 on one search's view (sec. 2.1, the synthetic kind)."""

    base_name = "search"
    max_resubmits = 0

    def __init__(self, run):
        self._init_run(run, "search_")
        self.deps = (SIM_BANK, "real_bank_" + run["real"])

    def start(self, cfg, st, pbs):
        return SearchStage.start(self, self._view(cfg), st, self._rec(pbs))

    def poll(self, cfg, st, pbs):
        return SearchStage.poll(self, self._view(cfg), st, self._rec(pbs))

    def verify(self, cfg, st):
        return SearchStage.verify(self, self._view(cfg), st)

    def on_killed(self, cfg, st, pbs):
        return SearchStage.on_killed(self, self._view(cfg), st, self._rec(pbs))

    def dry_run(self, cfg, st):
        v = self._view(cfg)
        sim, real = CM.shard_globs(v)
        return (["run %s: campaign %s, folder %s; the runner gets the simulated bank %s and the "
                 "real bank %s (every campaign, D-122)"
                 % (self.run["name"], self.run["campaign"],
                    os.path.join(v.data["paths"]["tune_root"], self.run["campaign"]), sim, real)]
                + SearchStage.dry_run(self, v, st))


class RunBestFitStage(_RunStage, BestFitStage):
    """S8 on one search's view (sec. 2.1, the synthetic kind)."""

    base_name = "best_fit"
    max_resubmits = 0

    def __init__(self, run):
        self._init_run(run, "best_fit_")
        self.deps = ("search_" + run["name"],)

    def start(self, cfg, st, pbs):
        return BestFitStage.start(self, self._view(cfg), st, self._rec(pbs))

    def poll(self, cfg, st, pbs):
        return BestFitStage.poll(self, self._view(cfg), st, self._rec(pbs))

    def verify(self, cfg, st):
        return BestFitStage.verify(self, self._view(cfg), st)

    def on_killed(self, cfg, st, pbs):
        return BestFitStage.on_killed(self, self._view(cfg), st, self._rec(pbs))

    def dry_run(self, cfg, st):
        v = self._view(cfg)
        return (["run %s: the best fit of %s in %s"
                 % (self.run["name"], self._dirs(v)["search_complete"], self._dirs(v)["root"])]
                + BestFitStage.dry_run(self, v, st))


class RunAltBestFitStage(RunBestFitStage):
    """[block 18, D-126] S8 on one search's view for the best trial, on the
    simulated report split's nll, of the arm the search's finalist is not
    (specs/SPEC.md sec. 2.5): the folder `<runs_root>/<q>/<tag>_best_alt`,
    the member jobs `ba_<tag>_s<k>`, S8's log lines under the stage's own
    name. Only the finalist is its own; the members, their faults and the
    ensemble are S8's, with the same `best_fit.*` values."""

    base_name = "best_fit_alt"
    log_name = "best_fit"
    root_suffix = "_best_alt"
    job_prefix = "ba"

    def __init__(self, run):
        self._init_run(run, "best_fit_alt_")
        self.deps = ("search_" + run["name"],)

    def _arms(self, cfg):
        camp = cfg.data["search"]["campaign"]
        if camp not in C.ALT_ARMS:
            raise RuntimeError("%s: campaign %s has no other arm (config.ALT_ARMS: %s)"
                               % (self.name, camp, ", ".join(sorted(C.ALT_ARMS))))
        return C.ALT_ARMS[camp]

    def _arm_file(self, cfg, arm):
        return os.path.join(self._dirs(cfg)["root"], "finalists_%s.json" % arm)

    def _finalist(self, cfg, st):
        """The tuner's `finalists --arm a` for each arm a of the campaign;
        the search's finalist must be the best of exactly one, and the
        other arm's best is fitted. Written to finalist_config.json in the
        tuner's --config-json shape, as S8's."""
        sub = st["sub"]
        if sub.get("finalist"):
            return sub["finalist"]
        dirs = self._dirs(cfg)
        sc = CM.read_json(dirs["search_complete"])
        main_id = sc["finalist"]["trial_id"]
        if not os.path.isdir(dirs["jobs"]):
            os.makedirs(dirs["jobs"])
        arms = self._arms(cfg)
        best = {}
        for arm in arms:
            out = self._arm_file(cfg, arm)
            CM.run_tuner(cfg, ["finalists"] + CM.tuner_common_args(cfg, sc["identity"])
                         + ["--top-k", "1", "--rank-split", "report", "--gate-split", "report",
                            "--arm", arm, "--out", out], log_path=self._log_path(cfg))
            fin = CM.read_json(out)
            if fin.get("arm") != arm or not fin.get("finalists"):
                raise RuntimeError("%s: %s is not the ranking of arm %s" % (self.name, out, arm))
            best[arm] = fin["finalists"][0]
        own = [a for a in arms if best[a]["trial_id"] == main_id]
        if len(own) != 1:
            raise RuntimeError(
                "%s: the search's finalist %s is the best trial of %d of the arms %s (best %s); "
                "exactly one was expected (%s)"
                % (self.name, main_id[:12], len(own), "/".join(arms),
                   ", ".join("%s %s" % (a, best[a]["trial_id"][:12]) for a in arms),
                   dirs["search_complete"]))
        other = [a for a in arms if a != own[0]][0]
        alt = best[other]
        search_finalist = {"trial_id": main_id, "arm": own[0]}
        CM.write_json(dirs["finalist_config"],
                      {"trial_id": alt["trial_id"], "campaign": sc["campaign"],
                       "nll_search": alt["nll_rank_split"], "config": alt["config"],
                       "search_complete": dirs["search_complete"], "arm": other,
                       "search_finalist": search_finalist,
                       "finalists_files": dict((a, self._arm_file(cfg, a)) for a in arms)})
        sub["finalist"] = {"trial_id": alt["trial_id"], "nll_search": alt["nll_rank_split"],
                           "identity": sc["identity"], "config_file": dirs["finalist_config"],
                           "arm": other, "search_finalist": search_finalist}
        sub.setdefault("members", {})
        return sub["finalist"]

    def dry_run(self, cfg, st):
        v = self._view(cfg)
        arms = C.ALT_ARMS.get(v.data["search"]["campaign"], ())
        lines = []
        for ln in BestFitStage.dry_run(self, v, st):
            if ln.startswith("finalist: "):
                ln = ("finalist: the best trial on the report-split nll of the arm of %s that the "
                      "finalist of %s is not -- the tuner's finalists --arm <a> --out %s for each "
                      "(block 18)" % ("/".join(arms), self._dirs(v)["search_complete"],
                                      self._arm_file(v, "<a>")))
            lines.append(ln)
        return ["run %s: the best fit of the other arm (D-126) in %s"
                % (self.run["name"], self._dirs(v)["root"])] + lines


# ---------------------------------------------------------------------------
# The banks (block 16)
# ---------------------------------------------------------------------------

def _num(x):
    return repr(float(x))


def _real_entry(cfg, name):
    for r in cfg.data["synthetic"]["real"]:
        if r["name"] == name:
            return r
    raise KeyError("synthetic.real has no entry %r" % (name,))


def bank_dir(cfg, bank):
    """`<banks_root>/<tag>_sim` (bank "sim") or `<banks_root>/<tag>_real_<bank>`."""
    d = cfg.data
    suffix = "_sim" if bank == "sim" else "_real_" + bank
    return os.path.join(d["paths"]["banks_root"], d["dataset"] + suffix)


def _builder_options(cfg):
    b = cfg.data["synthetic"]["builder"]
    return ["--provider", str(b["provider"]), "--fs", _num(b["fs"]), "--T-win", _num(b["T_win"]),
            "--n-neurons", str(int(b["n_neurons"])),
            "--wells-per-donor", str(int(b["wells_per_donor"])),
            "--donors-per-batch", str(int(b["donors_per_batch"])),
            "--n-classes", str(int(b["n_classes"])), "--tau-ov", _num(b["tau_ov"]),
            "--class-centres", str(b["class_centres"]), "--centre-scale", _num(b["centre_scale"])]


def build_of(cfg, bank):
    """The build of a bank (sec. 5, block 16): its shards' builder arguments
    without --out-dir, their rows, its arm; strings and integers only, so
    that it compares equal to its JSON copy in the record."""
    syn = cfg.data["synthetic"]
    b = syn["builder"]
    wpd = int(b["wells_per_donor"])
    if bank == "sim":
        s = syn["sim"]
        tps = int(s["traces_per_shard"])
        n_shards = int(s["n_theta"]) * wpd // tps
        argv = [["--arm", "S", "--prior", "box", "--shard-index", str(k), "--n-traces", str(tps),
                 "--n-windows", str(int(s["n_windows"])), "--seed", str(int(s["seed"]))]
                + _builder_options(cfg) for k in range(n_shards)]
        rows = [tps * int(s["n_windows"])] * n_shards
        arm = "S"
    else:
        r = _real_entry(cfg, bank)
        n_tr = int(b["n_classes"]) * int(r["donors_per_class"]) * wpd
        argv = [["--arm", "R", "--shard-index", "0", "--n-traces", str(n_tr),
                 "--donors-per-class", str(int(r["donors_per_class"])),
                 "--n-windows", str(int(r["n_windows"])), "--pi", _num(r["pi"]),
                 "--gap-modes", ",".join(r["gap_modes"]), "--seed", str(int(r["seed"]))]
                + _builder_options(cfg)]
        rows = [n_tr * int(r["n_windows"])]
        arm = "R"
    return {"schema": BUILD_SCHEMA, "bank": bank, "arm": arm, "n_shards": len(argv),
            "rows_per_shard": rows, "argv": argv}


def _normalise(argv):
    """A builder's argument list without `--out-dir` and its value."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--out-dir":
            skip = True
            continue
        out.append(str(a))
    return out


def _tail(path, n=3):
    text = CM.read_log_tail(path, 4000) or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return " | ".join(lines[-n:]) if lines else "(no log)"


class SynthBankStage(Stage):
    """One bank of a synthetic data set (sec. 5, block 16): `build` (one job
    per shard missing, at most what resources.max_cores holds), then
    `record` (one job of tools/synth_bank_record.py), then done; the
    artefact of record is the bank folder's BANK_RECORD.json with its
    .sha256 and the configuration's build."""

    max_resubmits = 0           # the shards and the record carry their own

    def __init__(self, bank):
        self.bank = bank
        self.name = SIM_BANK if bank == "sim" else "real_bank_" + bank
        self.deps = ()

    # -- places ----------------------------------------------------------------

    def _dir(self, cfg):
        return bank_dir(cfg, self.bank)

    def _jobs_dir(self, cfg):
        return os.path.join(cfg.data["paths"]["work_root"], "bank_jobs", self.name)

    def _record(self, cfg):
        return os.path.join(self._dir(cfg), RECORD_FILE)

    def _shard(self, cfg, k, ext):
        return os.path.join(self._dir(cfg), "shard_%04d.%s" % (int(k), ext))

    def _short(self):
        return "sim" if self.bank == "sim" else self.bank

    @staticmethod
    def _code(cfg):
        sbi = cfg.data["paths"]["sbi_dir"]
        return {"stage1": os.path.join(sbi, "hpc", "joint", "stage1"),
                "builder": os.path.join(sbi, "hpc", "joint", "stage1", "build_latent_bank.py"),
                "pipeline": os.path.join(sbi, "hpc", "pipeline"),
                "record_tool": os.path.join(sbi, "hpc", "pipeline", "tools", "synth_bank_record.py")}

    # -- the done-predicate ------------------------------------------------------

    def is_done(self, cfg):
        rec = self._record(cfg)
        ok, why = artefact_done("sha256", rec)
        if not ok:
            return False, why
        obj, problem = CM.try_read_json(rec)
        if obj is None:
            return False, "%s cannot be read (%s)" % (rec, problem)
        if obj.get("build") != build_of(cfg, self.bank):
            return False, ("%s records a bank of another build than the configuration's (move the "
                           "folder aside or change paths.banks_root)" % rec)
        return True, ("%s (sha256 matches its sidecar; the configuration's build: %s shard(s), %s "
                      "rows)" % (rec, obj.get("n_shards"), obj.get("n_rows")))

    # -- the folder ----------------------------------------------------------------

    def _shard_state(self, cfg, k, argv_k):
        """'built', 'absent', 'partial' (one of the pair missing),
        'unreadable' (the sidecar cannot be read) or 'foreign: <why>'."""
        npz, js = self._shard(cfg, k, "npz"), self._shard(cfg, k, "json")
        has_npz, has_js = os.path.isfile(npz), os.path.isfile(js)
        if not has_js:
            return "partial" if has_npz else "absent"
        obj, _ = CM.try_read_json(js)
        if obj is None:
            return "unreadable"
        got = _normalise(((obj.get("provenance") or {}).get("argv")) or [])
        if got != list(argv_k):
            return "foreign: its sidecar's arguments are `%s`, the configuration's `%s`" % (
                " ".join(got), " ".join(argv_k))
        return "built" if has_npz else "partial"

    def _scan(self, cfg, build):
        """({k: state}, [the folder's foreign content])."""
        states, foreign = {}, []
        for k, argv_k in enumerate(build["argv"]):
            s = self._shard_state(cfg, k, argv_k)
            states[k] = s
            if s.startswith("foreign"):
                foreign.append("shard %d: %s" % (k, s[len("foreign: "):]))
        d = self._dir(cfg)
        if os.path.isdir(d):
            for p in sorted(os.listdir(d)):
                m = _SHARD_FILE.match(p)
                if m and int(m.group(1)) >= int(build["n_shards"]):
                    foreign.append("%s: a shard index the configuration's %d shard(s) do not have"
                                   % (p, int(build["n_shards"])))
            rec = self._record(cfg)
            if os.path.isfile(rec):
                obj, _ = CM.try_read_json(rec)
                if obj is not None and obj.get("build") != build:
                    foreign.append("%s records another build" % RECORD_FILE)
        return states, foreign

    def _refuse_foreign(self, cfg, foreign):
        return ("REFUSE: %s holds a bank of another build than the configuration's (%s%s); nothing "
                "in it is overwritten -- move the folder aside or change paths.banks_root, then "
                "--retry %s" % (self._dir(cfg), "; ".join(foreign[:3]),
                                "; and %d more" % (len(foreign) - 3) if len(foreign) > 3 else "",
                                self.name))

    # -- the jobs ------------------------------------------------------------------

    def _ncpus(self, cfg):
        return int(cfg.data["synthetic"]["bank_job"]["ncpus"])

    def _qsub(self, cfg, name, log_path, argv_file, workdir):
        bj = cfg.data["synthetic"]["bank_job"]
        return CM.qsub_line(cfg, name, bj["queue"], bj["ncpus"], bj["mem"], bj["walltime"], log_path,
                            argv_file, workdir, avoid_file=CM.avoid_file(cfg))

    def _shard_argv(self, cfg, build, k):
        return ([CM.E.python_in_env(), self._code(cfg)["builder"], "--out-dir", self._dir(cfg)]
                + list(build["argv"][k]))

    def _submit_shard(self, cfg, st, pbs, k):
        sh = st["sub"]["shards"][str(k)]
        jd = self._jobs_dir(cfg)
        n = len(sh["jobs"])
        argv_file = CM.write_argv_file(os.path.join(jd, "shard_%d.%d.argv" % (k, n)),
                                       self._shard_argv(cfg, build_of(cfg, self.bank), k))
        log_path = os.path.join(jd, "shard_%d.%d.log" % (k, n))
        line = self._qsub(cfg, "sb_%s_%d" % (self._short(), k), log_path, argv_file,
                          self._code(cfg)["stage1"])
        job_id = pbs.submit(line, role="shard %d" % k, log_path=log_path)
        sh["jobs"].append(job_id)
        sh["status"] = "running"
        sh["log"] = log_path
        return job_id

    def _submit_waiting(self, cfg, st, pbs):
        n = self._ncpus(cfg)
        for k in sorted(st["sub"]["shards"], key=int):
            if st["sub"]["shards"][k]["status"] != "waiting":
                continue
            if not pbs.fits(n):
                return
            self._submit_shard(cfg, st, pbs, int(k))

    def expect(self, cfg):
        """What the record tool checks (sec. 5, block 16)."""
        d = cfg.data
        b = d["synthetic"]["builder"]
        build = build_of(cfg, self.bank)
        if self.bank == "sim":
            n_windows = int(d["synthetic"]["sim"]["n_windows"])
            dpc = None
        else:
            r = _real_entry(cfg, self.bank)
            n_windows = int(r["n_windows"])
            dpc = int(r["donors_per_class"])
        return {"schema": "synth_bank_expect/1", "bank": self.bank, "dir": self._dir(cfg),
                "build": build, "arm": build["arm"],
                "W": int(round(float(b["T_win"]) * float(b["fs"]))),
                "d_theta": int(d["search"]["shape_anchors"]["d_theta"]),
                "wells_per_donor": int(b["wells_per_donor"]), "n_windows": n_windows,
                "n_classes": int(b["n_classes"]), "donors_per_class": dpc,
                "latent_kind": "box" if build["arm"] == "S" else None,
                "theta_withheld": build["arm"] == "R"}

    def _record_argv(self, cfg, expect_path):
        return [CM.E.python_in_env(), self._code(cfg)["record_tool"], "--bank-dir", self._dir(cfg),
                "--expect", expect_path, "--out", self._record(cfg)]

    def _submit_record(self, cfg, st, pbs):
        rj = st["sub"]["record"]
        jd = self._jobs_dir(cfg)
        n = len(rj["jobs"])
        expect_path = CM.write_json(os.path.join(jd, "expect.json"), self.expect(cfg))
        argv_file = CM.write_argv_file(os.path.join(jd, "record.%d.argv" % n),
                                       self._record_argv(cfg, expect_path))
        log_path = os.path.join(jd, "record.%d.log" % n)
        line = self._qsub(cfg, "sr_%s" % self._short(), log_path, argv_file,
                          self._code(cfg)["pipeline"])
        job_id = pbs.submit(line, role="record", log_path=log_path)
        rj["jobs"].append(job_id)
        rj["status"] = "running"
        rj["log"] = log_path
        return job_id

    # -- a job's end -----------------------------------------------------------------

    @staticmethod
    def _fail(sub, text):
        sub["fail"] = text
        sub["phase"] = "failed"

    def _not_started(self, cfg, st, holder, job, what, log):
        """True (and `holder` waiting again, not counted) when the job never
        ran its command: held off a listed host (D-102) or an environment
        fault (D-111; the streak guard may stop the stage)."""
        sub = st["sub"]
        text = CM.read_log_tail(job.get("log"))
        if CM.held_off_node(job):
            host = CM.host_of_log(text) or "?"
            holder.setdefault("avoid_holds", []).append({"host": host, "job": job["id"],
                                                         "utc": CM.utc_now()})
            holder["status"] = "waiting"
            log("%s: %s: its job %s landed on %s, which is on the avoid list (D-102), held the slot "
                "and exited; submitted again (not a resubmission)" % (self.name, what, job["id"], host))
            return True
        why = CM.env_fault(job, text)
        if why is None:
            return False
        host = CM.host_of_log(text) or job.get("host") or "?"
        holder.setdefault("env_faults", []).append({"host": host, "job": job["id"], "why": why,
                                                    "utc": CM.utc_now()})
        n, stop = CM.note_env_fault(sub, host, why, self.name, log, CM.avoid_file(cfg))
        holder["status"] = "waiting"
        if stop is not None:
            sub["env_stop"] = stop
        log("%s: %s: its job %s could not start on %s (%s): an environment fault, not the bank's "
            "(D-111); %s" % (self.name, what, job["id"], host, why,
                            "the guard stops the stage" if stop else
                            "submitted again (not a resubmission); %d fault(s) on %s so far"
                            % (n, host)))
        return True

    def _finish_shard(self, cfg, st, pbs, k, job, build, log):
        sub = st["sub"]
        sh = sub["shards"][str(k)]
        s = self._shard_state(cfg, k, build["argv"][k])
        if s == "built":
            sh["status"] = "built"
            CM.clear_env_fault_streak(sub)
            log("%s: shard %d built (job %s)" % (self.name, k, job["id"]))
            return
        if s.startswith("foreign"):
            self._fail(sub, self._refuse_foreign(cfg, ["shard %d: %s" % (k, s[len("foreign: "):])]))
            return
        if s == "unreadable":
            aside = CM.set_aside(self._shard(cfg, k, "json"), "unreadable")
            log("%s: shard %d: its sidecar cannot be read; set aside as %s"
                % (self.name, k, os.path.basename(aside)))
        if self._not_started(cfg, st, sh, job, "shard %d" % k, log):
            return
        CM.clear_env_fault_streak(sub)          # the job ran (D-111)
        mx = int(cfg.data["synthetic"]["bank_job"]["max_resubmits"])
        if sh["resubmits"] < mx:
            sh["resubmits"] += 1
            sh["status"] = "waiting"
            log("%s: shard %d: its job %s ended without the shard (exit %s); submitted again (%d of "
                "%d)" % (self.name, k, job["id"], job.get("exit_status"), sh["resubmits"], mx))
            return
        self._fail(sub, "shard %d: %d job(s) ended without the shard (the last, %s, exit %s; its "
                        "log %s ends: %s); fix the cause or raise synthetic.bank_job, then --retry "
                        "%s (the shards built are kept)"
                   % (k, len(sh["jobs"]), job["id"], job.get("exit_status"), job.get("log"),
                      _tail(job.get("log")), self.name))

    def _finish_record(self, cfg, st, pbs, job, log):
        sub = st["sub"]
        rj = sub["record"]
        ok, why = self.is_done(cfg)
        if ok:
            rj["status"] = "done"
            sub["phase"] = "complete"
            CM.clear_env_fault_streak(sub)
            log("%s: BANK RECORDED: %s" % (self.name, why))
            return
        text = CM.read_log_tail(job.get("log")) or ""
        refusals = [l.strip() for l in text.splitlines() if l.startswith("REFUSE:")]
        if refusals:
            self._fail(sub, "the record tool refused the bank %s: %s%s (its log %s); fix the bank "
                            "(move it aside to rebuild it), then --retry %s"
                       % (self._dir(cfg), " | ".join(refusals[:6]),
                          " | ..." if len(refusals) > 6 else "", job.get("log"), self.name))
            return
        if self._not_started(cfg, st, rj, job, "the record", log):
            return
        CM.clear_env_fault_streak(sub)
        mx = int(cfg.data["synthetic"]["bank_job"]["max_resubmits"])
        if rj["resubmits"] < mx:
            rj["resubmits"] += 1
            rj["status"] = "waiting"
            log("%s: the record job %s ended without the record (exit %s); submitted again (%d of %d)"
                % (self.name, job["id"], job.get("exit_status"), rj["resubmits"], mx))
            return
        self._fail(sub, "the record: %d job(s) ended without it (the last, %s, exit %s; its log %s "
                        "ends: %s); then --retry %s"
                   % (len(rj["jobs"]), job["id"], job.get("exit_status"), job.get("log"),
                      _tail(job.get("log")), self.name))

    # -- the interface ---------------------------------------------------------------

    def start(self, cfg, st, pbs):
        sub = st["sub"]
        build = build_of(cfg, self.bank)
        states, foreign = self._scan(cfg, build)
        if foreign:
            raise RuntimeError(self._refuse_foreign(cfg, foreign))
        pbs.check(self._ncpus(cfg))              # a bank job that can never start (D-099)
        sub.clear()
        sub["shards"] = {}
        for k, s in sorted(states.items()):
            if s == "unreadable":
                aside = CM.set_aside(self._shard(cfg, k, "json"), "unreadable")
                pbs.log("%s: shard %d: its sidecar cannot be read; set aside as %s; built again"
                        % (self.name, k, os.path.basename(aside)))
            sub["shards"][str(k)] = {"status": "built" if s == "built" else "waiting", "jobs": [],
                                     "resubmits": 0}
        n_built = sum(1 for sh in sub["shards"].values() if sh["status"] == "built")
        if n_built:
            pbs.log("%s: %d of %d shard(s) already built in %s; not built again"
                    % (self.name, n_built, len(states), self._dir(cfg)))
        sub["record"] = {"status": "waiting", "jobs": [], "resubmits": 0}
        sub["phase"] = "build"
        self._submit_waiting(cfg, st, pbs)

    def poll(self, cfg, st, pbs):
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
                job["host"] = getattr(js, "exec_host", None)
        by_id = dict((j["id"], j) for j in st["jobs"])

        if sub.get("phase") == "build":
            build = build_of(cfg, self.bank)
            for k in sorted(sub["shards"], key=int):
                sh = sub["shards"][k]
                if sh["status"] != "running" or not sh["jobs"]:
                    continue
                job = by_id.get(sh["jobs"][-1])
                if job is None or not job.get("finished"):
                    continue
                self._finish_shard(cfg, st, pbs, int(k), job, build, log)
                if sub.get("phase") == "failed":
                    return "finished"
            if sub.get("env_stop"):
                self._fail(sub, sub.pop("env_stop"))
                return "finished"
            CM.sync_avoid_list(sub, CM.avoid_file(cfg), log, self.name)
            self._submit_waiting(cfg, st, pbs)
            if any(sh["status"] != "built" for sh in sub["shards"].values()):
                return "running"
            log("%s: every shard built (%d); the record job checks the bank"
                % (self.name, len(sub["shards"])))
            sub["phase"] = "record"

        if sub.get("phase") == "record":
            rj = sub["record"]
            if rj["status"] == "running":
                job = by_id.get(rj["jobs"][-1]) if rj["jobs"] else None
                if job is None or not job.get("finished"):
                    return "running"
                self._finish_record(cfg, st, pbs, job, log)
                if sub.get("phase") in ("complete", "failed"):
                    return "finished"
            if sub.get("env_stop"):
                self._fail(sub, sub.pop("env_stop"))
                return "finished"
            if rj["status"] == "waiting" and pbs.fits(self._ncpus(cfg)):
                self._submit_record(cfg, st, pbs)
            return "running"
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
        build = build_of(cfg, self.bank)
        states, foreign = self._scan(cfg, build)
        n_built = sum(1 for s in states.values() if s == "built")
        jd = self._jobs_dir(cfg)
        k_ex = build["n_shards"] - 1
        shard = self._qsub(cfg, "sb_%s_<k>" % self._short(), os.path.join(jd, "shard_<k>.<n>.log"),
                           os.path.join(jd, "shard_<k>.<n>.argv"), self._code(cfg)["stage1"])
        shard[2] = "sb_%s_<k>" % self._short()
        rec = self._qsub(cfg, "sr_%s" % self._short(), os.path.join(jd, "record.<n>.log"),
                         os.path.join(jd, "record.<n>.argv"), self._code(cfg)["pipeline"])
        bj = cfg.data["synthetic"]["bank_job"]
        lines = ["bank %s: %d shard(s) of %s row(s) (arm %s) in %s"
                 % (self.bank, build["n_shards"], "/".join(sorted(set(str(r) for r in
                                                                     build["rows_per_shard"]))),
                    build["arm"], self._dir(cfg)),
                 "folder: %s" % ("would REFUSE: " + "; ".join(foreign[:3]) if foreign else
                                 "%d of %d shard(s) built already (not built again)"
                                 % (n_built, build["n_shards"])),
                 "cores (D-099): at most %s bank job(s) at once of %d core(s) each; a job that ends "
                 "without its shard is submitted again up to %d time(s) (D-102 holds and D-111 "
                 "faults free)" % (CM.cap_room(cfg, bj["ncpus"], build["n_shards"]),
                                   int(bj["ncpus"]), int(bj["max_resubmits"])),
                 "    " + " ".join(shard),
                 "    argv of shard %d: %s" % (k_ex, " ".join(self._shard_argv(cfg, build, k_ex))),
                 "record: one job when every shard is built (CRCs, digests, rows, classes; pass "
                 "line SYNTH BANK RECORD OK):",
                 "    " + " ".join(rec),
                 "    argv: %s" % " ".join(self._record_argv(cfg, os.path.join(jd, "expect.json"))),
                 "artefact: %s + .sha256, its build the configuration's" % self._record(cfg)]
        return lines


# ---------------------------------------------------------------------------
# The graph (block 15)
# ---------------------------------------------------------------------------

def synthetic_set_graph(cfg):
    """The stages of a synthetic data set (sec. 2.1): the simulated bank, the
    real banks, then each search, its best fit and, with alt_best_fit true,
    the other arm's best fit (block 18), in the configuration's order."""
    syn = cfg.data["synthetic"]
    graph = [SynthBankStage("sim")] + [SynthBankStage(r["name"]) for r in syn["real"]]
    for q in syn["searches"]:
        graph.append(RunSearchStage(q))
        graph.append(RunBestFitStage(q))
        if q.get("alt_best_fit"):
            graph.append(RunAltBestFitStage(q))
    return graph
