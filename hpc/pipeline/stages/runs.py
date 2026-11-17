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

A run with an integer `max_cores` (recorded `searches` only) caps the cores
of its own stages' jobs, inside `resources.max_cores` (block 29).

Nothing here names a project (D-091). Pure ASCII, LF only; stdlib only;
Python 3.6 syntax.
"""

import copy
import os

import config as C

__all__ = ["RunView", "search_views", "recorded_set_graph", "core_group_of"]


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


def recorded_set_graph(cfg):
    """[block 29] The stages of a recorded data set with a `searches` list
    (sec. 2.1): S1-S6 as in the default graph, then, for each entry q in
    order, `search_<q.name>` (deps sim_bank, real_bank) and
    `best_fit_<q.name>` (deps search_<q.name>); no stage `search` or
    `best_fit`."""
    raise NotImplementedError("block 29")


def core_group_of(stage):
    """[block 29] (cap, names) when `stage` is a run's stage whose run has an
    integer `max_cores`: cap = that integer, names = the names of the run's
    stages (search_<q>, best_fit_<q>, and best_fit_alt_<q> where the graph
    has it); None otherwise (every other stage, and a run with max_cores
    null). The sequencer's Submitter counts a stage's jobs against its
    group's cap as well as against resources.max_cores."""
    raise NotImplementedError("block 29")
