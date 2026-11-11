"""What the evaluation finds and what it builds, in the standard library only
(block 19; specs/SPEC.md sec. 2.5 and blocks 19 and 22): the ensembles of a
synthetic project and the calibration bank's build. Imported by the
sampler and by the launcher (block 22), which runs on the login node's
python without numpy or torch.

    cfg = ev_plan.load_project(".../synth.json")
    for e in ev_plan.ensembles(cfg): ...          # complete ensembles
    build = ev_plan.calbank_build(cfg, ecfg)      # the builder's argv per shard

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


def find(cfg, name):
    """The complete ensemble called `name` (`<q>:<arm>`)."""
    found = ensembles(cfg)
    for e in found:
        if e.name == name:
            return e
    raise RefuseError("no complete ensemble %r (complete: %s)"
                      % (name, ", ".join(e.name for e in found) or "none"))


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
    ensemble's sample provenance that exists, by stem, and the report's
    values (ev_config.REPORT_PARAMS). A report whose recorded inputs differ
    is stale."""
    samples = {}
    for e in ensembles(cfg):
        pj = sample_json(ecfg, e.stem)
        if os.path.isfile(pj):
            samples[e.stem] = sha256_file(pj)
    return {"samples": samples, "params": ecfg.report_params()}
