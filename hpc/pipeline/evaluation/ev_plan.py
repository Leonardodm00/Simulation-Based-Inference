"""What the evaluation finds and what it builds, in the standard library only
(block 19; specs/SPEC.md sec. 2.5 and blocks 19 and 22): the ensembles of a
synthetic project, the comparisons between them (block 27) and the
calibration bank's build. Imported by the sampler and by the launcher (block
22), which runs on the login node's python without numpy or torch.

    cfg = ev_plan.load_project(".../synth.json")
    for e in ev_plan.ensembles(cfg): ...          # complete ensembles
    for c in ev_plan.comparisons(cfg): ...        # sec. 2.5's comparisons (block 27)
    build = ev_plan.calbank_build(cfg, ecfg)      # the builder's argv per shard
    inputs = ev_plan.data_inputs(cfg, ecfg)       # what the data overview reads (block 23)

An ensemble is one best-fit folder of one search: `<runs_root>/<q>/<tag>_best`
(block 7) or `<tag>_best_alt` (block 18). Its members are found in the folder
itself (`member_seed<k>/*_seed<k>.json`, as stages/best_fit.py finds them),
never through absolute paths recorded on another machine.

Pure ASCII, LF only.
"""

import glob
import hashlib
import json
import os
from collections import namedtuple

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

import config as C
from stages.synthetic import RunView, _builder_options, bank_dir

Ensemble = namedtuple("Ensemble", "name stem search arm folder kind seeds records cohorts")

# [block 27, D-136] a comparison of sec. 2.5 ("The comparisons"): `name` "<q>"
# (kind "vs_A1") or "<q>_vs_<q'>" (kind "A2_vs_A5"); `E` and `E_ref` the two
# ensembles' names "<search>:<arm>", read donor-paired as E3 with E = E and
# E' = E_ref; `search`, `search_ref` their searches; `arm`, `arm_ref` their arms
Comparison = namedtuple("Comparison", "name kind E E_ref search search_ref arm arm_ref")

# [block 27] A(q), the term arm of a search's campaign (sec. 2.5): S-A2 and S-A5
# nest A1 (their A1 reference is their own A1 ensemble); S-A2-only and
# S-A5-only (block 26) do not (theirs is the one S-A1 search's)
TERM_ARM = {"S-A2": "A2", "S-A5": "A5", "S-A2-only": "A2", "S-A5-only": "A5"}
NESTS_A1 = ("S-A2", "S-A5")
# a campaign with a term that takes part in no comparison, and why (D-136)
UNCOMPARED = {"S-A25": "campaign S-A25: its arms are not defined against A1 (D-136)"}


class RefuseError(RuntimeError):
    """The evaluation cannot go on for a stated reason (exit 3)."""


def load_project(path):
    """The pipeline's configuration of a synthetic data set."""
    cfg = C.load_config(path, synthetic_ok=True)
    if not C.is_synthetic(cfg.data):
        raise RefuseError("%s is not a synthetic data set's configuration" % path)
    return cfg


def search_view(cfg, name):
    """RunView of search `name`."""
    for q in cfg.data["synthetic"]["searches"]:
        if q["name"] == name:
            return RunView(cfg, q)
    raise RefuseError("no search %r in %s" % (name, cfg.path))


def member_record(folder, seed):
    """The record of member `seed` in a best-fit folder, or None."""
    found = sorted(glob.glob(os.path.join(folder, "member_seed%d" % int(seed),
                                          "*_seed%d.json" % int(seed))))
    return found[0] if found else None


def _cohorts(cfg, q):
    """D-122: an S-A1 search's ensemble is scored on every cohort; any other
    on its own."""
    if q["campaign"] == "S-A1":
        return [r["name"] for r in cfg.data["synthetic"]["real"]]
    return [q["real"]]


def ensembles(cfg, with_missing=False):
    """Every complete best fit of every search, as Ensembles, in the
    configuration's order (best fit, then the other arm's). With
    with_missing, also (q, folder, why) for each folder that is configured
    and not complete."""
    out, missing = [], []
    d = cfg.data
    for q in d["synthetic"]["searches"]:
        v = RunView(cfg, q)
        kinds = [("best_fit", "_best")]
        if q.get("alt_best_fit"):
            kinds.append(("best_fit_alt", "_best_alt"))
        for kind, suffix in kinds:
            folder = os.path.join(v.data["paths"]["runs_root"], d["dataset"] + suffix)
            complete = os.path.join(folder, "BEST_FIT_COMPLETE.json")
            if not os.path.isfile(complete):
                missing.append((q["name"], folder, "no BEST_FIT_COMPLETE.json"))
                continue
            with open(complete) as fh:
                bf = json.load(fh)
            seeds = [int(s) for s in bf["member_seeds"]]
            recs, arms = [], set()
            for s in seeds:
                r = member_record(folder, s)
                if r is None:
                    raise RefuseError("%s: member seed %d has no record" % (folder, s))
                with open(r) as fh:
                    arms.add(json.load(fh)["arm"])
                recs.append(r)
            if len(arms) != 1:
                raise RefuseError("%s: the members' arms differ: %s" % (folder, sorted(arms)))
            arm = arms.pop()
            out.append(Ensemble(name="%s:%s" % (q["name"], arm), stem="%s_%s" % (q["name"], arm),
                                search=dict(q), arm=arm, folder=folder, kind=kind,
                                seeds=tuple(seeds), records=tuple(recs),
                                cohorts=tuple(_cohorts(cfg, q))))
    names = [e.name for e in out]
    dup = sorted(set(n for n in names if names.count(n) > 1))
    if dup:
        raise RefuseError("two best fits of one search share an arm: %s" % ", ".join(dup))
    return (out, missing) if with_missing else out


def comparisons(cfg, with_uncompared=False):
    """[block 27, D-136] The comparisons of specs/SPEC.md sec. 2.5, from the
    configuration alone (no file is read; standard library only).

    (a) For each search q, in the configuration's order, of campaign S-A2,
    S-A5, S-A2-only or S-A5-only -- A(q) = A2 for S-A2 and S-A2-only, A5 for
    S-A5 and S-A5-only --
        Comparison(name=q, kind="vs_A1", E="<q>:<A(q)>", E_ref=<q's A1 reference>,
                   search=q, search_ref=..., arm=A(q), arm_ref="A1"),
    the reference "<q>:A1" for S-A2 and S-A5 (they nest A1) and "<s>:A1" for
    S-A2-only and S-A5-only, s the configuration's one search of campaign
    S-A1; then (b) for each search q of term arm A2 and each search q' of term
    arm A5 with q.real == q'.real, in the order of q and then of q',
        Comparison(name="<q>_vs_<q'>", kind="A2_vs_A5", E="<q>:A2",
                   E_ref="<q'>:A5", search=q, search_ref=q', arm="A2", arm_ref="A5").
    A search of campaign S-A1 yields no comparison of its own; one of S-A25
    is uncompared, with the why "campaign S-A25: its arms are not defined
    against A1 (D-136)".

    Returns [Comparison], or with `with_uncompared` ([Comparison], [(q, why)]).
    Raises RefuseError when a search of a term-only campaign has no A1
    reference -- no search of campaign S-A1, or two or more -- naming the
    search and the S-A1 searches found."""
    searches = cfg.data["synthetic"]["searches"]
    s_a1 = [q["name"] for q in searches if q["campaign"] == "S-A1"]
    out, uncompared = [], []
    # (a) each search with a term against its A1 reference
    for q in searches:
        camp = q["campaign"]
        if camp in UNCOMPARED:
            uncompared.append((q["name"], UNCOMPARED[camp]))
            continue
        if camp not in TERM_ARM:
            continue                                   # S-A1: a reference only
        if camp in NESTS_A1:
            ref = q["name"]
        elif len(s_a1) == 1:
            ref = s_a1[0]
        else:
            raise RefuseError(
                "search %s (campaign %s, no A1 trial) has no A1 reference: its comparison needs "
                "exactly one search of campaign S-A1, the configuration has %s"
                % (q["name"], camp, ("%d (%s)" % (len(s_a1), ", ".join(s_a1))) if s_a1 else "none"))
        arm = TERM_ARM[camp]
        out.append(Comparison(name=q["name"], kind="vs_A1", E="%s:%s" % (q["name"], arm),
                              E_ref="%s:A1" % ref, search=q["name"], search_ref=ref, arm=arm,
                              arm_ref="A1"))
    # (b) A2 against A5 on one cohort, in the order of q and then of q'
    for q in searches:
        if TERM_ARM.get(q["campaign"]) != "A2":
            continue
        for q2 in searches:
            if TERM_ARM.get(q2["campaign"]) != "A5" or q2["real"] != q["real"]:
                continue
            out.append(Comparison(name="%s_vs_%s" % (q["name"], q2["name"]), kind="A2_vs_A5",
                                  E="%s:A2" % q["name"], E_ref="%s:A5" % q2["name"],
                                  search=q["name"], search_ref=q2["name"], arm="A2",
                                  arm_ref="A5"))
    return (out, uncompared) if with_uncompared else out


def find(cfg, name):
    """The complete ensemble called `name` (`<q>:<arm>`)."""
    found = ensembles(cfg)
    for e in found:
        if e.name == name:
            return e
    raise RefuseError("no complete ensemble %r (complete: %s)"
                      % (name, ", ".join(e.name for e in found) or "none"))


def seed_of(base, *parts):
    """A seed for (base, parts...), stable across machines and runs: the
    first 32 bits of the sha256 of their text."""
    h = hashlib.sha256(("%d|" % int(base) + "|".join(str(p) for p in parts)).encode("ascii"))
    return int(h.hexdigest()[:8], 16)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calbank_build(cfg, ecfg):
    """The calibration bank's build (Q-E5): per shard k = 0, ..., cal_shards
    - 1, the builder's arguments without --out-dir -- arm S on the box prior
    with the simulated bank's options (provider, f_s, T_win, neurons, the
    nuisance's batches), one well per donor and one window per well, so one
    window per theta, `cal_n / cal_shards` donors per shard, seed `cal_seed`.
    Strings only. Returns dict(out_dir, n_shards, rows_per_shard, argv)."""
    opts = list(_builder_options(cfg))
    i = opts.index("--wells-per-donor")
    opts[i + 1] = "1"
    per = int(ecfg.cal_n) // int(ecfg.cal_shards)
    argv = [["--arm", "S", "--prior", "box", "--shard-index", str(k), "--n-traces", str(per),
             "--n-windows", "1", "--n-per-theta", "1", "--seed", str(int(ecfg.cal_seed))] + opts
            for k in range(int(ecfg.cal_shards))]
    return {"out_dir": os.path.dirname(ecfg.cal_glob), "n_shards": len(argv),
            "rows_per_shard": [per] * len(argv), "argv": argv}


def sim_bank_glob(cfg):
    """The simulated bank's shards (every search's)."""
    return os.path.join(bank_dir(cfg, "sim"), "shard_*.npz")


def sample_json(ecfg, stem):
    """`<out_root>/samples/<stem>_sample.json`, block 19's provenance."""
    return os.path.join(ecfg.sub("samples"), "%s_sample.json" % stem)


def report_inputs(cfg, ecfg):
    """What a report is made of (block 22): the sha256 of every complete
    ensemble's sample provenance that exists, by stem, the report's values
    (ev_config.REPORT_PARAMS) and [block 27] the comparisons, [[name, E,
    E_ref], ...] in `comparisons`' order. A report whose recorded inputs
    differ is stale, so a report made before block 27 is. RefuseError as
    `comparisons`."""
    samples = {}
    for e in ensembles(cfg):
        pj = sample_json(ecfg, e.stem)
        if os.path.isfile(pj):
            samples[e.stem] = sha256_file(pj)
    return {"samples": samples, "params": ecfg.report_params(),
            "comparisons": [[c.name, c.E, c.E_ref] for c in comparisons(cfg)]}


def data_dir(ecfg):
    """`<out_root>/data`: the data overview's files (block 23)."""
    return ecfg.sub("data")


def data_view_json(ecfg):
    """`<out_root>/data/DATA_VIEW.json`, the data overview's record, written
    last (block 23)."""
    return os.path.join(data_dir(ecfg), "DATA_VIEW.json")


def cohort_glob(cfg, name):
    """The shards of cohort `name` of `synthetic.real`."""
    return os.path.join(bank_dir(cfg, name), "shard_*.npz")


def data_bank_paths(cfg, ecfg):
    """The shards the data overview reads (block 23), sorted: the first
    `data_view.sim_shards` of the simulated bank under "sim" (fewer when the
    bank has fewer: the job refuses then) and every shard of each cohort of
    `synthetic.real` under its name."""
    out = {"sim": sorted(glob.glob(sim_bank_glob(cfg)))[:int(ecfg.data_view["sim_shards"])]}
    for r in cfg.data["synthetic"]["real"]:
        out[r["name"]] = sorted(glob.glob(cohort_glob(cfg, r["name"])))
    return out


def data_inputs(cfg, ecfg):
    """What the data overview is made of (block 23): per bank, each shard
    read as [file name, sha256 of its sidecar or None], and its values
    (ev_config.DATA_PARAMS). A DATA_VIEW.json whose recorded inputs differ
    is stale."""
    banks = {}
    for name, paths in sorted(data_bank_paths(cfg, ecfg).items()):
        rows = []
        for p in paths:
            side = os.path.splitext(p)[0] + ".json"
            rows.append([os.path.basename(p), sha256_file(side) if os.path.isfile(side) else None])
        banks[name] = rows
    return {"banks": banks, "params": ecfg.data_params()}
