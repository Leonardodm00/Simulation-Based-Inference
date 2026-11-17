"""A data set's searches as runs (blocks 15 and 29; specs/SPEC.md secs. 2.1,
3.1 and block 29).

A run is one search of a data set and its best fit: one entry of
`synthetic.searches` (the synthetic kind, block 15) or of `searches` (a
recorded data set with several searches, block 29, config v11; D-148 (1)).
S7 and S8 run on the run's view of the configuration (RunView): the same
configuration with `paths.tune_root` and `paths.runs_root` joined with the
run's name and `search.campaign` the run's campaign, so their code and their
folders are those of a single search.

    RunView(base, run)       the view (implemented in the contract commit)
    search_views(cfg)        [(name or None, view)] of every search
    recorded_set_graph(cfg)  block 29: S1-S6, then search_<q>, best_fit_<q>
    core_group_of(stage)     block 29: (cap, stage names) of a run's stages
    run_core_group(run)      the same, from the run itself
    pick_search(cfg, name)   a tool's --search: (view, stage name), or the
                             REFUSED line as a ValueError
    stage_classes()          (RunSearchStage, RunBestFitStage,
                             RunAltBestFitStage): S7 and S8 on a run's view,
                             both kinds (moved here from stages/synthetic.py,
                             which binds them under their old names)

A run with an integer `max_cores` (recorded `searches` only) caps the cores
of its own stages' jobs, inside `resources.max_cores` (block 29): its
stages carry `core_group` (stages.Stage), which the sequencer's Submitter
reads.

The run stages subclass S7's and S8's classes, so they are built at the
first call of stage_classes() rather than when this module is imported:
this module imports nothing from stages/search.py or stages/best_fit.py at
its top, so that a module those import (the stop request's, block 30) may
import this one without an import cycle.

Nothing here names a project (D-091). Pure ASCII, LF only; stdlib only;
Python 3.6 syntax.
"""

import copy
import os

import config as C

from . import artefact_done, artefact_of
from . import common as CM

__all__ = ["RunView", "search_views", "recorded_set_graph", "core_group_of", "run_core_group",
           "pick_search", "stage_classes"]


class RunView(C.Config):
    """The configuration as one search sees it: a copy of `base.data` with
    `paths.tune_root` and `paths.runs_root` joined with run["name"] and
    `search.campaign` = run["campaign"]; on a synthetic configuration also
    `synthetic.run` = run (common.shard_globs reads it there). `base`'s
    path and sha256; `.base` and `.run`."""

    def __init__(self, base, run):
        if isinstance(base, RunView):
            raise ValueError("RunView of a RunView (%s of %s)" % (run.get("name"), base.run["name"]))
        data = copy.deepcopy(base.data)
        paths = data["paths"]
        paths["tune_root"] = os.path.join(paths["tune_root"], run["name"])
        paths["runs_root"] = os.path.join(paths["runs_root"], run["name"])
        data.setdefault("search", {})["campaign"] = run["campaign"]
        if C.is_synthetic(data):
            data["synthetic"]["run"] = dict(run)
        C.Config.__init__(self, data, base.path, base.sha256)
        self.base = base
        self.run = dict(run)


def search_views(cfg):
    """[(name, view)] of every search of the configuration, in its order:
    [(None, cfg)] for a recorded configuration whose `searches` is null (the
    single search, as before v11); [(q.name, RunView(cfg, q))] for each entry
    q of `searches` or of `synthetic.searches`."""
    d = cfg.data
    if C.is_synthetic(d):
        return [(q["name"], RunView(cfg, q)) for q in d["synthetic"]["searches"]]
    if isinstance(d.get("searches"), list):
        return [(q["name"], RunView(cfg, q)) for q in d["searches"]]
    return [(None, cfg)]


# ---------------------------------------------------------------------------
# The run stages: S7 and S8 on a run's view (blocks 15, 18; block 29: both
# kinds)
# ---------------------------------------------------------------------------

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
    it is, since S7's and S8's methods call is_done with it), the renamed
    recorder and [block 29] the run's core group."""

    base_name = ""
    log_name = None             # [block 18] the prefix of the lines renamed; None: base_name

    def _init_run(self, run, prefix):
        self.run = dict(run)
        self.name = prefix + run["name"]
        self._cache = None
        # [block 29, D-148 (1)] the run's own core cap, read by the sequencer's Submitter
        self.core_group = run_core_group(self.run)

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

    def _group_lines(self, cfg, ncpus_key):
        """[block 29] The dry run's line on the run's own core cap, for a job
        of `ncpus_key` cores; none without a cap."""
        if self.core_group is None:
            return []
        cap, names = self.core_group
        n = int(cfg.get(ncpus_key))
        return ["cores (block 29, D-141): the jobs of %s share at most %d cores "
                "(searches[].max_cores), inside resources.max_cores: at most %d job(s) of %d "
                "cores at once" % (", ".join(names), cap, cap // n, n)]


_CLASSES = []


def stage_classes():
    """(RunSearchStage, RunBestFitStage, RunAltBestFitStage), the run stages
    of both kinds; built at the first call (the module docstring)."""
    if not _CLASSES:
        _CLASSES.extend(_define_stage_classes())
    return tuple(_CLASSES)


def _define_stage_classes():
    from . import DEPS
    from .best_fit import BestFitStage
    from .search import SearchStage

    class RunSearchStage(_RunStage, SearchStage):
        """S7 on one search's view (sec. 2.1: the synthetic kind, block 15;
        a recorded data set with several searches, block 29)."""

        base_name = "search"
        max_resubmits = 0

        def __init__(self, run, deps=None):
            self._init_run(run, "search_")
            if deps is None:
                # a synthetic run names its real bank (block 15; synthetic.SIM_BANK is
                # "sim_bank"); a recorded run's search follows S5 and S6 (block 29)
                deps = (("sim_bank", "real_bank_" + run["real"]) if "real" in run
                        else DEPS["search"])
            self.deps = tuple(deps)

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
            if C.is_synthetic(v.data):
                banks = "the real bank %s (every campaign, D-122)" % (real,)
            elif real:
                banks = "the real bank %s" % (real,)
            else:
                banks = "no real bank (S-A1 trains on the simulated windows alone, D-039)"
            return (["run %s: campaign %s, folder %s; the runner gets the simulated bank %s and %s"
                     % (self.run["name"], self.run["campaign"],
                        os.path.join(v.data["paths"]["tune_root"], self.run["campaign"]), sim,
                        banks)]
                    + self._group_lines(v, "search.trial.ncpus")
                    + SearchStage.dry_run(self, v, st))

    class RunBestFitStage(_RunStage, BestFitStage):
        """S8 on one search's view (sec. 2.1, both kinds)."""

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

        def is_done(self, cfg):
            # [block 7 (i), D-134] the best fit follows its search's finalist: S8's
            # predicate on the view, not _RunStage's artefact-only one (G_q inherits it)
            return BestFitStage.is_done(self, self._view(cfg))

        def dry_run(self, cfg, st):
            v = self._view(cfg)
            return (["run %s: the best fit of %s in %s"
                     % (self.run["name"], self._dirs(v)["search_complete"], self._dirs(v)["root"])]
                    + self._group_lines(v, "best_fit.ncpus")
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
                    ln = ("finalist: the best trial on the report-split nll of the arm of %s that "
                          "the finalist of %s is not -- the tuner's finalists --arm <a> --out %s for "
                          "each (block 18)" % ("/".join(arms), self._dirs(v)["search_complete"],
                                               self._arm_file(v, "<a>")))
                lines.append(ln)
            return ["run %s: the best fit of the other arm (D-126) in %s"
                    % (self.run["name"], self._dirs(v)["root"])] + lines

    classes = (RunSearchStage, RunBestFitStage, RunAltBestFitStage)
    for cls in classes:
        cls.__qualname__ = cls.__name__          # built in this function; named as at module level
    return classes


# ---------------------------------------------------------------------------
# Block 29: a recorded data set with several searches, each with its core cap
# ---------------------------------------------------------------------------

def recorded_set_graph(cfg):
    """[block 29] The stages of a recorded data set with a `searches` list
    (sec. 2.1): S1-S6 as in the default graph, then, for each entry q in
    order, `search_<q.name>` (deps sim_bank, real_bank) and
    `best_fit_<q.name>` (deps search_<q.name>); no stage `search` or
    `best_fit`."""
    from . import BLOCK_OF, BUILT, DEPS, STAGE_ORDER, NotBuiltStage
    if not C.has_search_list(cfg.data):
        raise ValueError("recorded_set_graph: the configuration %s has no searches list (a "
                         "recorded data set with several searches, block 29)" % (cfg.path,))
    graph = []
    for name in STAGE_ORDER:
        if name in ("search", "best_fit"):
            continue
        # S1-S6 exactly as stages.default_graph builds them
        cls = BUILT.get(name)
        if cls is None:
            graph.append(NotBuiltStage(name, BLOCK_OF[name]))
            continue
        stage = cls()
        if stage.name != name or tuple(stage.deps) != DEPS[name]:
            raise RuntimeError("stage class %r does not match the graph for %s" % (cls, name))
        graph.append(stage)
    search_cls, best_fit_cls, _alt = stage_classes()
    for q in cfg.data["searches"]:
        graph.append(search_cls(q, deps=DEPS["search"]))
        graph.append(best_fit_cls(q))
    return graph


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def run_core_group(run):
    """(cap, names) of a run with an integer `max_cores`: the cap and the
    names of the run's stages (search_<q>, best_fit_<q>, and best_fit_alt_<q>
    when the run has one: `alt_best_fit` true, the synthetic kind); None
    for a run whose `max_cores` is null or absent (every synthetic run)."""
    if not isinstance(run, dict):
        return None
    cap = run.get("max_cores")
    if not _is_int(cap):
        return None
    q = run["name"]
    names = ["search_" + q, "best_fit_" + q]
    if run.get("alt_best_fit"):
        names.append("best_fit_alt_" + q)
    return int(cap), tuple(names)


def pick_search(cfg, name):
    """[block 29] The search a tool's `--search <name>` names: (its view,
    the name of its search stage, `search_<name>`) on a configuration with a
    searches list; (cfg, "search") on a recorded one whose `searches` is null,
    given no name. ValueError with the tool's `REFUSED: ...` line otherwise:
    no name or an unknown one on a list, a name on a null configuration."""
    if C.has_search_list(cfg.data):
        views = search_views(cfg)
        for n, v in views:
            if name is not None and n == name:
                return v, "search_" + n
        raise ValueError("REFUSED: this configuration has the searches %s: give --search <name>"
                         % ", ".join(n for n, _v in views))
    if name is not None:
        raise ValueError("REFUSED: this configuration has one search (searches null, the campaign "
                         "%s): --search is for a configuration with a searches list; give no "
                         "--search" % ((cfg.data.get("search") or {}).get("campaign"),))
    return cfg, "search"


def core_group_of(stage):
    """[block 29] (cap, names) when `stage` is a run's stage whose run has an
    integer `max_cores`: cap = that integer, names = the names of the run's
    stages (search_<q>, best_fit_<q>, and best_fit_alt_<q> where the graph
    has it); None otherwise (every other stage, and a run with max_cores
    null). The sequencer's Submitter counts a stage's jobs against its
    group's cap as well as against resources.max_cores."""
    if not isinstance(stage, _RunStage):
        return None
    return run_core_group(stage.run)
