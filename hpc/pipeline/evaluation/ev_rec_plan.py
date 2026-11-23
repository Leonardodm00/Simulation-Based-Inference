"""The view of a project for the evaluation of a recorded project (block 33;
specs/SPEC.md sec. 2.6, "The view of a project", and block 33; decisions
D-159, D-162, D-163 (3)).

    view = ev_rec_plan.load_view(".../<tag>.json", ecfg)
    for e in view.ensembles: ...                  # complete ensembles
    made, not_made = ev_rec_plan.comparisons(view)

The standard library and the pipeline's own modules only (config,
stages.runs, ev_plan): the launcher (block 38) imports this module on the
login node's python.

CONTRACT (wave A): the namedtuples and the names imported from ev_plan are
fixed by specs/SPEC.md block 33. The functions are stubs.

Pure ASCII, LF only; Python 3.6 syntax.
"""

from collections import namedtuple

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

# never defined again here (block 33): the refusal, the seeds, the digests,
# the comparison's record and the campaigns' term arms are ev_plan's
from ev_plan import (Comparison, NESTS_A1, RefuseError, TERM_ARM,  # noqa: F401
                     member_record, seed_of, sha256_file)

# kind: "recorded" (a recorded configuration) or "as_recorded" (a synthetic
# one read as recorded, D-162). ensembles: a tuple of RecEnsemble, in the
# configuration's order. missing: a tuple of (search, folder, why) for each
# configured best fit that is not complete. sim_glob: the simulated bank's
# shards. banks: dict from a recording bank's name to its shards' glob.
View = namedtuple("View", "cfg kind ensembles missing sim_glob banks")

# name "<q>:<arm>" (stem "<q>_<arm>") with a searches list or as recorded;
# "best:<arm>" (stem "best_<arm>") for a recorded configuration whose
# `searches` is null. search: the search's name, or None in that last case.
# campaign: the search's campaign. seeds, records: the members' train seeds
# and record paths, in member order. banks: the names of its recording banks.
RecEnsemble = namedtuple("RecEnsemble",
                         "name stem search campaign arm folder seeds records banks")


def load_view(cfg_path, ecfg):
    """The View of the project configuration at `cfg_path` under the
    evaluation's configuration `ecfg` (a RecEvalConfig). RefuseError for a
    recorded configuration with `as_recorded` true, a synthetic one with
    `as_recorded` false (the message names the other evaluation), members
    whose arm or split differ, a member without its record, or two complete
    ensembles of one name."""
    raise NotImplementedError("block 33")


def find(view, name):
    """The complete RecEnsemble called `name`; RefuseError naming the
    complete ones otherwise."""
    raise NotImplementedError("block 33")


def comparisons(view):
    """(made, not_made): `made` a list of ev_plan.Comparison in sec. 2.6's
    order, (a) then (b); `not_made` a list of (name, why). For an
    as-recorded view `made` equals ev_plan.comparisons(cfg). RefuseError as
    block 27's when a needed S-A1 reference is missing or ambiguous."""
    raise NotImplementedError("block 33")


def sample_json(ecfg, stem):
    """`<out_root>/samples/<stem>_sample.json`, block 35's provenance."""
    raise NotImplementedError("block 33")


def report_inputs(view, ecfg):
    """What a report is made of (block 38): {"samples": {stem: sha256 of its
    sample provenance, for each complete ensemble that has one}, "params":
    ecfg.report_params(), "comparisons": [[name, E, E_ref], ...] of `made`}."""
    raise NotImplementedError("block 33")
