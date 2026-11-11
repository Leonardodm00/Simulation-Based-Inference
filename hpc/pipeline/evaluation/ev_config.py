"""The evaluation's configuration (block 19; specs/SPEC.md block 19,
"EvalConfig"; D-127 for the values of Q-E1..Q-E6).

    cfg = ev_config.load("hpc/pipeline/projects/synth/synth_eval.json")

A JSON object; a key starting with '_' documents the key after it and is
ignored; every other key is required and checked, and there are no defaults
in the code, so the value in force is always the file's. Standard library
only (block 22's launcher imports it on the login node).

Pure ASCII, LF only.
"""

import json
import os
import re
from collections import namedtuple

FORMATS = ("png", "pdf", "svg")
JOB_KINDS = ("calbank", "sample", "report")
# the values a sampling depends on: a sample of other values is stale (block 22)
SAMPLE_PARAMS = ("seed", "n_post", "ppc_rows", "ppc_draws", "cal_glob", "cal_n", "cal_seed")
# the values a report depends on beyond its samples (block 22)
REPORT_PARAMS = ("seed", "lc2st_null", "lc2st_eval_draws", "bootstrap_n", "level", "dpi",
                 "formats")
_MEM = re.compile(r"^[1-9][0-9]*[kmgt]?b$")
_WALL = re.compile(r"^[0-9]+:[0-5][0-9]:[0-5][0-9]$")

# the keys, in order (a namedtuple, not a dataclass: the launcher runs on the
# login node's python, which may predate 3.7)
KEYS = (
    "out_root",          # absolute: samples/, metrics/, figures/, calbank/, jobs/ under it
    "seed",              # the base of every seed of the evaluation
    "n_post",            # posterior draws per row (Q-E1), a multiple of M
    "batch_rows",        # windows per batch through a member
    "ppc_rows",          # D_sim's PPC rows (Q-E4)
    "ppc_draws",         # simulated windows per PPC row (Q-E4)
    "cal_glob",          # the calibration bank's shards (Q-E5)
    "cal_n",             # its windows, one per theta
    "cal_shards",        # its shards (cal_n divisible by them)
    "cal_seed",          # its builder seed: distinct from the banks' seeds
    "lc2st_null",        # L-C2ST's permutation trials (Q-E5)
    "lc2st_eval_draws",  # draws per window at which L-C2ST is evaluated
    "bootstrap_n",       # resamples of every bootstrap
    "level",             # the bands' and intervals' level
    "dpi",               # the figures' (Q-E6)
    "formats",           # the figures' file formats (Q-E6), a tuple
    "ncpus",             # worker processes when a command is run by hand
    "jobs",              # block 22: queue, and ncpus/mem/walltime per job kind
)


class EvalConfigError(ValueError):
    """The evaluation's configuration is not usable; the message lists every
    problem found."""


class EvalConfig(namedtuple("EvalConfig", KEYS + ("path",))):
    """The evaluation's values (KEYS) and the file they were read from."""

    __slots__ = ()

    def sub(self, name):
        """A folder under out_root."""
        return os.path.join(self.out_root, name)

    def sample_params(self):
        """The values a sampling depends on (SAMPLE_PARAMS), as a dict."""
        return dict((k, getattr(self, k)) for k in SAMPLE_PARAMS)

    def report_params(self):
        """The values a report depends on beyond its samples (REPORT_PARAMS),
        as a dict of JSON types."""
        return dict((k, list(v) if isinstance(v, tuple) else v)
                    for k, v in ((k, getattr(self, k)) for k in REPORT_PARAMS))


_INTS = ("seed", "n_post", "batch_rows", "ppc_rows", "ppc_draws", "cal_n", "cal_shards",
         "cal_seed", "lc2st_null", "lc2st_eval_draws", "bootstrap_n", "dpi", "ncpus")


def _strip_docs(obj):
    if isinstance(obj, dict):
        return dict((k, _strip_docs(v)) for k, v in obj.items() if not str(k).startswith("_"))
    return obj


def validate(data):
    """The problems of a configuration dict, every one (empty when usable)."""
    names = list(KEYS)
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
    jb = data.get("jobs")
    if "jobs" in data:
        if not (isinstance(jb, dict) and sorted(jb) == sorted(("queue",) + JOB_KINDS)):
            probs.append("jobs must have exactly the keys queue, %s" % ", ".join(JOB_KINDS))
        else:
            if not (isinstance(jb["queue"], str) and jb["queue"]):
                probs.append("jobs.queue must be a queue name, got %r" % (jb["queue"],))
            for kind in JOB_KINDS:
                j = jb[kind]
                if not (isinstance(j, dict) and sorted(j) == ["mem", "ncpus", "walltime"]):
                    probs.append("jobs.%s must have exactly the keys ncpus, mem, walltime" % kind)
                    continue
                if not (isinstance(j["ncpus"], int) and not isinstance(j["ncpus"], bool)
                        and j["ncpus"] >= 1):
                    probs.append("jobs.%s.ncpus must be an integer >= 1, got %r"
                                 % (kind, j["ncpus"]))
                if not (isinstance(j["mem"], str) and _MEM.match(j["mem"])):
                    probs.append("jobs.%s.mem must be like 32gb, got %r" % (kind, j["mem"]))
                if not (isinstance(j["walltime"], str) and _WALL.match(j["walltime"])):
                    probs.append("jobs.%s.walltime must be hh:mm:ss, got %r"
                                 % (kind, j["walltime"]))
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
