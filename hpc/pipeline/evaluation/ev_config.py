"""The evaluation's configuration (block 19; specs/SPEC.md block 19,
"EvalConfig"; D-127 for the values of Q-E1..Q-E6).

    cfg = ev_config.load("hpc/pipeline/projects/synth/synth_eval.json")

A JSON object; a key starting with '_' documents the key after it and is
ignored; every other key is required and checked, and there are no defaults
in the code, so the value in force is always the file's.

Pure ASCII, LF only.
"""

import json
import os
from dataclasses import dataclass, fields

FORMATS = ("png", "pdf", "svg")


class EvalConfigError(ValueError):
    """The evaluation's configuration is not usable; the message lists every
    problem found."""


@dataclass(frozen=True)
class EvalConfig:
    out_root: str            # absolute: samples/, metrics/, figures/, calbank/ under it
    seed: int                # the base of every seed of the evaluation
    n_post: int              # posterior draws per row (Q-E1), a multiple of M
    batch_rows: int          # windows per batch through a member
    ppc_rows: int            # D_sim's PPC rows (Q-E4)
    ppc_draws: int           # simulated windows per PPC row (Q-E4)
    cal_glob: str            # the calibration bank's shards (Q-E5)
    cal_n: int               # its windows, one per theta
    cal_shards: int          # its shards (cal_n divisible by them)
    cal_seed: int            # its builder seed: distinct from the banks' seeds
    lc2st_null: int          # L-C2ST's permutation trials (Q-E5)
    lc2st_eval_draws: int    # draws per window at which L-C2ST is evaluated
    bootstrap_n: int         # resamples of every bootstrap
    level: float             # the bands' and intervals' level
    dpi: int                 # the figures' (Q-E6)
    formats: tuple           # the figures' file formats (Q-E6)
    ncpus: int               # worker processes for the simulations
    path: str = ""           # the file it was read from

    def sub(self, name):
        """A folder under out_root."""
        return os.path.join(self.out_root, name)


_INTS = ("seed", "n_post", "batch_rows", "ppc_rows", "ppc_draws", "cal_n", "cal_shards",
         "cal_seed", "lc2st_null", "lc2st_eval_draws", "bootstrap_n", "dpi", "ncpus")


def _strip_docs(obj):
    if isinstance(obj, dict):
        return dict((k, _strip_docs(v)) for k, v in obj.items() if not str(k).startswith("_"))
    return obj


def validate(data):
    """The problems of a configuration dict, every one (empty when usable)."""
    names = [f.name for f in fields(EvalConfig) if f.name != "path"]
    probs = []
    missing = [k for k in names if k not in data]
    unknown = sorted(k for k in data if k not in names)
    if missing:
        probs.append("missing key(s): %s" % ", ".join(missing))
    if unknown:
        probs.append("unknown key(s): %s" % ", ".join(unknown))
    for k in _INTS:
        v = data.get(k)
        if k in data and not (isinstance(v, int) and not isinstance(v, bool)
                              and (v >= 0 if k == "seed" or k == "cal_seed" else v >= 1)):
            probs.append("%s must be an integer >= %d, got %r"
                         % (k, 0 if k in ("seed", "cal_seed") else 1, v))
    if "out_root" in data and not (isinstance(data["out_root"], str)
                                   and os.path.isabs(data["out_root"])):
        probs.append("out_root must be an absolute path, got %r" % (data["out_root"],))
    if "cal_glob" in data and not (isinstance(data["cal_glob"], str)
                                   and data["cal_glob"].endswith(".npz")):
        probs.append("cal_glob must name the calibration bank's .npz shards, got %r"
                     % (data.get("cal_glob"),))
    lv = data.get("level")
    if "level" in data and not (isinstance(lv, float) and 0.0 < lv < 1.0):
        probs.append("level must be a number in (0, 1), got %r" % (lv,))
    fm = data.get("formats")
    if "formats" in data and not (isinstance(fm, list) and fm and all(f in FORMATS for f in fm)
                                  and len(set(fm)) == len(fm)):
        probs.append("formats must be a non-empty list of distinct %s, got %r"
                     % ("/".join(FORMATS), fm))
    if not probs:
        if data["ppc_draws"] > data["n_post"]:
            probs.append("ppc_draws %d exceeds n_post %d" % (data["ppc_draws"], data["n_post"]))
        if data["lc2st_eval_draws"] > data["n_post"]:
            probs.append("lc2st_eval_draws %d exceeds n_post %d"
                         % (data["lc2st_eval_draws"], data["n_post"]))
        if data["cal_n"] % data["cal_shards"]:
            probs.append("cal_n %d is not divisible by cal_shards %d"
                         % (data["cal_n"], data["cal_shards"]))
    return probs


def load(path):
    """The EvalConfig of a file; EvalConfigError listing every problem."""
    with open(path, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    data = _strip_docs(raw)
    probs = validate(data)
    if probs:
        raise EvalConfigError("%s: %s" % (path, "; ".join(probs)))
    data = dict(data)
    data["formats"] = tuple(data["formats"])
    return EvalConfig(path=os.path.abspath(path), **data)
