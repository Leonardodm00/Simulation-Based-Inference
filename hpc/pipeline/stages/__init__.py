"""The stages: the interface, the eight names with their dependencies and
done-predicates, and the registry (specs/SPEC.md sec. 2.1, block 1).

A stage is a small state machine the sequencer drives through four calls:
`is_done` (the artefact of record exists), `start` (submit; record the
jobs in the state), `poll` (`running` or `finished`, from the scheduler),
`verify` (the artefact and the refusal lines after the jobs ended). A
stage whose runner a later block builds is registered here as NotBuilt, so
the sequencer can say so instead of guessing.

Nothing here names a project (D-091): every path comes from the config.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import hashlib
import os

__all__ = ["Stage", "NotBuiltStage", "StageNotBuilt", "STAGE_ORDER",
           "DEPS", "default_graph", "topological_order", "sha256_sidecar_ok",
           "artefact_of", "BUILT"]


# ---------------------------------------------------------------------------
# The graph (sec. 2.1)
# ---------------------------------------------------------------------------

STAGE_ORDER = ("real_extraction", "sim_redetection", "label_freeze",
               "bank_plan", "sim_bank", "real_bank", "search", "best_fit")

DEPS = {
    "real_extraction": (),
    "sim_redetection": ("real_extraction",),
    "label_freeze": (),
    "bank_plan": ("real_extraction", "sim_redetection", "label_freeze"),
    "sim_bank": ("bank_plan",),
    "real_bank": ("bank_plan",),
    "search": ("sim_bank", "real_bank"),
    "best_fit": ("search",),
}

# The block that builds each stage's runner. A built stage registers its
# class in BUILT (later blocks); the rest are NotBuilt placeholders.
BLOCK_OF = {
    "real_extraction": 3, "sim_redetection": 3, "label_freeze": 3,
    "bank_plan": 2, "sim_bank": 2, "real_bank": 2,
    "search": 7, "best_fit": 7,
}

BUILT = {}


def sha256_sidecar_ok(path):
    """True iff `path` and `path + '.sha256'` exist and the sidecar's first
    token (64 hex characters) equals the file's SHA-256."""
    side = path + ".sha256"
    if not (os.path.isfile(path) and os.path.isfile(side)):
        return False
    with open(side, "rb") as fh:
        tokens = fh.read().decode("ascii", "replace").split()
    if not tokens or len(tokens[0]) != 64:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == tokens[0].lower()


def artefact_of(name, cfg):
    """The artefact of record of a stage, as (kind, path): kind is
    'sha256' (the file plus a matching sidecar) or 'pair' (the file plus
    its .json beside it) or 'file'."""
    d = cfg.data
    tag = d["dataset"]
    if name == "real_extraction":
        return "sha256", os.path.join(d["real"]["extract_root"], "cohort_manifest.json")
    if name == "sim_redetection":
        return "sha256", os.path.join(d["sim"]["out_root"], "REEXTRACTION_RECORD.json")
    if name == "label_freeze":
        return "sha256", os.path.join(d["paths"]["sbix_dir"], d["freeze"]["out"])
    if name == "bank_plan":
        return "sha256", os.path.join(d["paths"]["sbix_dir"], "export_windows", tag,
                                      "bank_plan.json")
    if name == "sim_bank":
        return "sha256", os.path.join(d["paths"]["banks_root"], tag + "_sim",
                                      "BANK_RECORD.json")
    if name == "real_bank":
        return "pair", os.path.join(d["paths"]["banks_root"], tag + "_real",
                                    "shard_0000.npz")
    if name == "search":
        return "file", os.path.join(d["paths"]["tune_root"], d["search"]["campaign"],
                                    "SEARCH_COMPLETE.json")
    if name == "best_fit":
        return "file", os.path.join(d["paths"]["runs_root"], tag + "_best",
                                    "BEST_FIT_COMPLETE.json")
    raise KeyError(name)


def artefact_done(kind, path):
    """(done, reason) for an artefact of record."""
    if kind == "sha256":
        if sha256_sidecar_ok(path):
            return True, "%s (sha256 matches its sidecar)" % path
        if os.path.isfile(path):
            return False, "%s exists but its .sha256 sidecar is missing or does not match" % path
        return False, "%s absent" % path
    if kind == "pair":
        side = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(path) and os.path.isfile(side):
            return True, "%s and its .json" % path
        return False, "%s or its .json absent" % path
    if kind == "file":
        if os.path.isfile(path):
            return True, path
        return False, "%s absent" % path
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

class Deferred(Exception):
    """Raised by a stage's start when its first job does not fit under
    resources.max_cores (D-099): the stage stays pending, and the sequencer
    calls start again at the next tick."""


class StageNotBuilt(RuntimeError):
    """Raised by a placeholder's start(): the runner belongs to a later block."""


class Stage(object):
    """Base class. Subclasses set `name`, `deps`, `max_resubmits` and
    implement start / verify / dry_run; `is_done` and `poll` have defaults
    that most stages keep."""

    name = ""
    deps = ()
    max_resubmits = 0

    # -- the done-predicate -------------------------------------------------

    def is_done(self, cfg):
        kind, path = artefact_of(self.name, cfg)
        return artefact_done(kind, path)

    # -- the run ---------------------------------------------------------------

    def start(self, cfg, st, pbs):
        """Submit the stage's first job(s), recording them through
        `pbs`/`st` (the sequencer passes a State-bound recorder: see
        pipeline.StageContext). Raise StageNotBuilt or a RuntimeError with
        the refusal text to stop before anything is submitted."""
        raise NotImplementedError

    def poll(self, cfg, st, pbs):
        """Default: 'finished' once every recorded job has finished."""
        running = False
        for job in st["jobs"]:
            if job.get("finished"):
                continue
            js = pbs.status(job["id"])
            if js.finished:
                job["finished"] = True
                job["exit_status"] = js.exit_status
                job["walltime_s"] = js.walltime_s
                job["state"] = js.state
            else:
                running = True
        return "running" if running else "finished"

    def verify(self, cfg, st):
        """(ok, message). A message starting with 'killed:' means the jobs
        ended with neither the artefact nor a refusal line, so on_killed may
        resubmit; any other False is a refusal that stops the run."""
        ok, why = self.is_done(cfg)
        if ok:
            return True, why
        return False, "killed: " + why

    def on_killed(self, cfg, st, pbs):
        """Resubmit after a kill, at most max_resubmits times. Default: a
        fresh start() (a stage with a resume flag overrides this)."""
        if st.get("resubmits", 0) >= self.max_resubmits:
            return False
        st["resubmits"] = st.get("resubmits", 0) + 1
        self.start(cfg, st, pbs)
        return True

    def dry_run(self, cfg, st):
        """The lines --dry-run prints for this stage (the qsub lines)."""
        return []


class NotBuiltStage(Stage):
    """A stage whose runner a later block builds: is_done works, start
    refuses with the block named."""

    def __init__(self, name, block):
        self.name = name
        self.deps = DEPS[name]
        self.block = int(block)

    def start(self, cfg, st, pbs):
        raise StageNotBuilt("stage %s is not built yet (block %d)"
                            % (self.name, self.block))

    def dry_run(self, cfg, st):
        return ["not built (block %d): nothing would be submitted" % self.block]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def default_graph(cfg):
    """The eight stages, built classes where a block registered one."""
    graph = []
    for name in STAGE_ORDER:
        cls = BUILT.get(name)
        if cls is None:
            graph.append(NotBuiltStage(name, BLOCK_OF[name]))
        else:
            stage = cls()
            if stage.name != name or tuple(stage.deps) != DEPS[name]:
                raise RuntimeError("stage class %r does not match the graph for %s"
                                   % (cls, name))
            graph.append(stage)
    return graph


def topological_order(graph):
    """Stages in an order where every dependency precedes its dependents;
    stable with respect to the given order; refuses a cycle or a missing
    dependency."""
    by_name = dict((s.name, s) for s in graph)
    for s in graph:
        for d in s.deps:
            if d not in by_name:
                raise ValueError("stage %s depends on unknown stage %s" % (s.name, d))
    order, seen, visiting = [], set(), set()

    def visit(s):
        if s.name in seen:
            return
        if s.name in visiting:
            raise ValueError("cycle through stage %s" % s.name)
        visiting.add(s.name)
        for d in s.deps:
            visit(by_name[d])
        visiting.discard(s.name)
        seen.add(s.name)
        order.append(s)

    for s in graph:
        visit(s)
    return order


# ---------------------------------------------------------------------------
# The built stages register here (block 7: S7, S8; block 2: S4-S6; block 3: S1-S3). The import sits at the
# end so the stage modules can import the interface above.
# ---------------------------------------------------------------------------

from .search import SearchStage        # noqa: E402
from .best_fit import BestFitStage     # noqa: E402
from .bank import BankPlanStage, SimBankStage, RealBankStage   # noqa: E402
from .inputs import RealExtractionStage, SimRedetectionStage, LabelFreezeStage   # noqa: E402

BUILT["search"] = SearchStage
BUILT["best_fit"] = BestFitStage
BUILT["bank_plan"] = BankPlanStage
BUILT["sim_bank"] = SimBankStage
BUILT["real_bank"] = RealBankStage
BUILT["real_extraction"] = RealExtractionStage
BUILT["sim_redetection"] = SimRedetectionStage
BUILT["label_freeze"] = LabelFreezeStage
