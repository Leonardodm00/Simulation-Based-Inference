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
fixed by specs/SPEC.md block 33.

Pure ASCII, LF only; Python 3.6 syntax.
"""

import json
import os
from collections import namedtuple

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

# never defined again here (block 33): the refusal, the seeds, the digests,
# the comparison's record and the campaigns' term arms are ev_plan's
from ev_plan import (Comparison, NESTS_A1, RefuseError, TERM_ARM,  # noqa: F401
                     member_record, seed_of, sha256_file)
import ev_plan  # noqa: E402

import config as C  # noqa: E402
from stages.runs import search_views  # noqa: E402

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


# the members' records must agree on these (block 33, "Rule")
_MEMBER_AGREE = ("arm", "split_seed", "split_fracs", "split_hash")
SUBMIT_SYNTHETIC = "submit_eval.sh"


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _members(folder):
    """(seeds, records, arm) of the complete best fit in `folder`: the
    members are BEST_FIT_COMPLETE.json's `member_seeds`; RefuseError for a
    member without its record or members whose arm or split differ."""
    bf = _read_json(os.path.join(folder, "BEST_FIT_COMPLETE.json"))
    seeds = [int(s) for s in bf["member_seeds"]]
    if not seeds:
        raise RefuseError("%s: BEST_FIT_COMPLETE.json lists no member" % folder)
    recs, vals = [], dict((k, []) for k in _MEMBER_AGREE)
    for s in seeds:
        r = member_record(folder, s)
        if r is None:
            raise RefuseError("%s: member seed %d has no record" % (folder, s))
        rec = _read_json(r)
        for k in _MEMBER_AGREE:
            v = rec.get(k)
            vals[k].append(json.dumps(v, sort_keys=True))
        recs.append(r)
    for k in _MEMBER_AGREE:
        if len(set(vals[k])) != 1:
            raise RefuseError("%s: the members' %s differ: %s"
                              % (folder, k, ", ".join(sorted(set(vals[k])))))
    arm = json.loads(vals["arm"][0])
    if not isinstance(arm, str) or not arm:
        raise RefuseError("%s: the members' records name no arm" % folder)
    return tuple(seeds), tuple(recs), arm


def _no_duplicates(ensembles):
    names = [e.name for e in ensembles]
    dup = sorted(set(n for n in names if names.count(n) > 1))
    if dup:
        raise RefuseError("two complete ensembles share a name: %s" % ", ".join(dup))


def _recorded_view(cfg):
    d = cfg.data
    tag = d["dataset"]
    banks_root = d["paths"]["banks_root"]
    listed = C.has_search_list(d)
    out, missing = [], []
    for name, v in search_views(cfg):
        folder = os.path.join(v.data["paths"]["runs_root"], tag + "_best")
        if not os.path.isfile(os.path.join(folder, "BEST_FIT_COMPLETE.json")):
            missing.append((name, folder, "no BEST_FIT_COMPLETE.json"))
            continue
        seeds, recs, arm = _members(folder)
        q = name if listed else "best"
        out.append(RecEnsemble(name="%s:%s" % (q, arm), stem="%s_%s" % (q, arm),
                               search=name if listed else None,
                               campaign=v.data["search"]["campaign"], arm=arm, folder=folder,
                               seeds=seeds, records=recs, banks=("real",)))
    _no_duplicates(out)
    return View(cfg=cfg, kind="recorded", ensembles=tuple(out), missing=tuple(missing),
                sim_glob=os.path.join(banks_root, tag + "_sim", "shard_*.npz"),
                banks={"real": os.path.join(banks_root, tag + "_real", "shard_*.npz")})


def _as_recorded_view(cfg):
    found, missing = ev_plan.ensembles(cfg, with_missing=True)
    out = []
    for e in found:
        _seeds, _recs, _arm = _members(e.folder)          # the split's agreement (block 33)
        out.append(RecEnsemble(name=e.name, stem=e.stem, search=e.search["name"],
                               campaign=e.search["campaign"], arm=e.arm, folder=e.folder,
                               seeds=tuple(e.seeds), records=tuple(e.records),
                               banks=tuple(e.cohorts)))
    _no_duplicates(out)
    banks = dict((r["name"], ev_plan.cohort_glob(cfg, r["name"]))
                 for r in cfg.data["synthetic"]["real"])
    return View(cfg=cfg, kind="as_recorded", ensembles=tuple(out),
                missing=tuple((q, f, why) for q, f, why in missing),
                sim_glob=ev_plan.sim_bank_glob(cfg), banks=banks)


def load_view(cfg_path, ecfg):
    """The View of the project configuration at `cfg_path` under the
    evaluation's configuration `ecfg` (a RecEvalConfig). RefuseError for a
    recorded configuration with `as_recorded` true, a synthetic one with
    `as_recorded` false (the message names the other evaluation), members
    whose arm or split differ, a member without its record, or two complete
    ensembles of one name."""
    cfg = C.load_config(cfg_path, synthetic_ok=True)
    synthetic = C.is_synthetic(cfg.data)
    if synthetic and not ecfg.as_recorded:
        raise RefuseError(
            "%s is a synthetic project's configuration and %s has as_recorded false: a synthetic "
            "project read as itself is evaluated by the synthetic evaluation (%s, blocks 19-22); "
            "set as_recorded true to read it as recorded (D-162)"
            % (cfg_path, ecfg.path, SUBMIT_SYNTHETIC))
    if not synthetic and ecfg.as_recorded:
        raise RefuseError(
            "%s is a recorded project's configuration and %s has as_recorded true, which is for "
            "a synthetic project read as recorded (D-162): set as_recorded false for the "
            "evaluation of a recorded project" % (cfg_path, ecfg.path))
    return _as_recorded_view(cfg) if synthetic else _recorded_view(cfg)


def find(view, name):
    """The complete RecEnsemble called `name`; RefuseError naming the
    complete ones otherwise."""
    for e in view.ensembles:
        if e.name == name:
            return e
    raise RefuseError("no complete ensemble %r (complete: %s)"
                      % (name, ", ".join(e.name for e in view.ensembles) or "none"))


def _recorded_comparisons(view):
    d = view.cfg.data
    if not C.has_search_list(d):
        return [], []
    searches = d["searches"]
    by_search = dict((e.search, e) for e in view.ensembles)
    missing = dict((q, folder) for q, folder, _why in view.missing)
    s_a1 = [q["name"] for q in searches if q["campaign"] == "S-A1"]

    def absent(q):
        return "the best fit of search %s is not complete (%s: no BEST_FIT_COMPLETE.json)" % (
            q, missing.get(q, "?"))

    made, not_made = [], []
    # (a) each search with a term arm against its A1 reference
    for q in searches:
        name, camp = q["name"], q["campaign"]
        if camp in ev_plan.UNCOMPARED:
            not_made.append((name, ev_plan.UNCOMPARED[camp]))
            continue
        if camp not in TERM_ARM:
            continue                                   # S-A1: a reference only
        arm = TERM_ARM[camp]
        e = by_search.get(name)
        own = e if (camp in NESTS_A1 and e is not None and e.arm == "A1") else None
        if own is not None:
            ref = own
            ref_search = name
        else:
            if len(s_a1) != 1:
                raise RefuseError(
                    "search %s (campaign %s) has no A1 ensemble of its own in this view: its "
                    "comparison needs exactly one search of campaign S-A1, the configuration has "
                    "%s" % (name, camp, ("%d (%s)" % (len(s_a1), ", ".join(s_a1))) if s_a1
                            else "none"))
            ref_search = s_a1[0]
            ref = by_search.get(ref_search)
        if e is None:
            not_made.append((name, absent(name)))
            continue
        if e.arm != arm:
            not_made.append((name, "the ensemble of search %s is of arm %s, not of its term arm "
                                   "%s (campaign %s)" % (name, e.arm, arm, camp)))
            continue
        if ref is None:
            not_made.append((name, "its A1 reference, search %s, is missing: %s"
                                   % (ref_search, absent(ref_search))))
            continue
        if ref.arm != "A1":
            not_made.append((name, "its A1 reference %s is of arm %s, not A1"
                                   % (ref.name, ref.arm)))
            continue
        made.append(Comparison(name=name, kind="vs_A1", E=e.name, E_ref=ref.name, search=name,
                               search_ref=ref_search, arm=arm, arm_ref="A1"))
    # (b) A2 against A5, in the order of q and then of q'
    for q in searches:
        if TERM_ARM.get(q["campaign"]) != "A2":
            continue
        for q2 in searches:
            if TERM_ARM.get(q2["campaign"]) != "A5":
                continue
            name = "%s_vs_%s" % (q["name"], q2["name"])
            why = []
            for s, want in ((q["name"], "A2"), (q2["name"], "A5")):
                e = by_search.get(s)
                if e is None:
                    why.append(absent(s))
                elif e.arm != want:
                    why.append("the ensemble of search %s is of arm %s, not %s" % (s, e.arm, want))
            if why:
                not_made.append((name, "; ".join(why)))
                continue
            made.append(Comparison(name=name, kind="A2_vs_A5", E=by_search[q["name"]].name,
                                   E_ref=by_search[q2["name"]].name, search=q["name"],
                                   search_ref=q2["name"], arm="A2", arm_ref="A5"))
    return made, not_made


def comparisons(view):
    """(made, not_made): `made` a list of ev_plan.Comparison in sec. 2.6's
    order, (a) then (b); `not_made` a list of (name, why). For an
    as-recorded view `made` equals ev_plan.comparisons(cfg). RefuseError as
    block 27's when a needed S-A1 reference is missing or ambiguous."""
    if view.kind == "as_recorded":
        made, uncompared = ev_plan.comparisons(view.cfg, with_uncompared=True)
        return list(made), list(uncompared)
    return _recorded_comparisons(view)


def sample_json(ecfg, stem):
    """`<out_root>/samples/<stem>_sample.json`, block 35's provenance."""
    return os.path.join(ecfg.sub("samples"), "%s_sample.json" % stem)


def report_inputs(view, ecfg):
    """What a report is made of (block 38): {"samples": {stem: sha256 of its
    sample provenance, for each complete ensemble that has one}, "params":
    ecfg.report_params(), "comparisons": [[name, E, E_ref], ...] of `made`}."""
    samples = {}
    for e in view.ensembles:
        pj = sample_json(ecfg, e.stem)
        if os.path.isfile(pj):
            samples[e.stem] = sha256_file(pj)
    made, _not_made = comparisons(view)
    return {"samples": samples, "params": ecfg.report_params(),
            "comparisons": [[c.name, c.E, c.E_ref] for c in made]}

