"""The values of a recorded project's evaluation (block 33; specs/SPEC.md sec.
2.6 and block 33; decisions D-159..D-164): `<tag>_rec_eval.json`.

    ecfg = ev_rec_config.load(".../<tag>_rec_eval.json")
    ecfg.sub("samples")          # a folder under out_root
    ecfg.sample_params()         # what a sampling depends on
    ecfg.report_params()         # what a report depends on beyond its samples

Standard library only: the launcher (block 38) imports this module on the
login node's python. Keys of the file that start with `_` are comments.

CONTRACT (wave A): the constants below are fixed by specs/SPEC.md block 33
and are shared by blocks 33, 35, 36 and 38.

Pure ASCII, LF only; Python 3.6 syntax.
"""

import json
import math
import os
import re
from collections import namedtuple

FORMATS = ("png", "pdf", "svg")
JOB_KINDS = ("sample", "report", "validate")

# the file's keys, in order (block 33, "The configuration")
KEYS = (
    "out_root",                  # absolute: samples/, metrics/, figures/, jobs/ under it
    "seed",                      # int >= 0: the base of every seed of the evaluation
    "n_post",                    # int > 0: posterior draws per row, a multiple of M (D-163: 1000)
    "batch_rows",                # int > 0: windows per batch through a member
    "oracle_tol",                # nats, in (0, 0.1]: the rebuild oracle's tolerance (2e-3)
    "as_recorded",               # bool: a synthetic project read as recorded (D-162)
    "sim_draw_rows_per_donor",   # null (every row) or int >= 1: k_draw (D-163 (1): 1)
    "edge_eps",                  # in (0, 0.5): the edge width eps of (R1) (D-163 (4): 0.006693)
    "gate",                      # dict with exactly GATE_KEYS
    "bootstrap_n",               # int > 0: resamples of every bootstrap
    "level",                     # in (0, 1): the bands' and intervals' level
    "dpi",                       # int > 0: the figures'
    "formats",                   # non-empty list of distinct FORMATS
    "ncpus",                     # int > 0: worker processes when a command is run by hand
    "validate_against",          # null, or the absolute out_root of the synthetic evaluation
                                 # the validation compares with; not null only with as_recorded
    "jobs",                      # queue, and ncpus/mem/walltime for each of JOB_KINDS
)

# the keys of `gate` (sec. 2.6, "The gate"; D-161, D-163 (4), D-164)
GATE_KEYS = (
    "donor_fracs",               # four positive numbers summing to 1 within 1e-9:
                                 # (f_ref, f_rank, f_null, f_check) (D-164: 0.4, 0.25, 0.25, 0.1)
    "n_ref",                     # int >= 2: the reference's cap (D-163 (4): 800)
    "n_rank",                    # int >= 1: the rank set's cap (D-164: 1000)
    "n_null",                    # int >= 1: the null set's cap (D-164: 1000)
    "n_null_min",                # int >= 1: a smaller null set is a refusal (D-164: 50)
    "rows_per_donor",            # int >= 1: rows of a network in the rank, null, check sets (1)
    "factors",                   # non-empty list of positive numbers (0.25, 0.5, 1, 2, 4)
    "alpha",                     # in (0, 1): alpha_gate (D-163 (4): 0.05)
)

# what a sampling depends on: a sample of other values is stale (block 38)
SAMPLE_PARAMS = ("seed", "n_post", "as_recorded", "sim_draw_rows_per_donor")
SAMPLE_GATE_PARAMS = ("donor_fracs", "n_ref", "n_rank", "n_null", "n_null_min", "rows_per_donor")
# what a report depends on beyond its samples; oracle_tol is in neither list
REPORT_PARAMS = ("seed", "edge_eps", "bootstrap_n", "level", "dpi", "formats")
REPORT_GATE_PARAMS = ("factors", "alpha")


# matched whole (fullmatch): "$" alone would accept a trailing newline
_MEM = re.compile(r"[1-9][0-9]*[kmgt]?b")
_WALL = re.compile(r"[0-9]+:[0-5][0-9]:[0-5][0-9]")
# npe_misspec.bandwidth_grid takes the median distance on a subsample above this
# many reference rows; (R5) is the median over all pairs (block 34)
N_REF_MAX = 1000
_FRACS_TOL = 1e-9


class RecEvalConfigError(ValueError):
    """The configuration is not usable; the message lists every problem."""


def _json_value(v):
    """A value as JSON has it: tuples as lists, dicts copied."""
    if isinstance(v, tuple):
        return [_json_value(x) for x in v]
    if isinstance(v, list):
        return [_json_value(x) for x in v]
    if isinstance(v, dict):
        return dict((k, _json_value(x)) for k, x in v.items())
    return v


class RecEvalConfig(namedtuple("RecEvalConfig", KEYS + ("path",))):
    """The evaluation's values (KEYS; `formats` a tuple, `gate` a dict whose
    `donor_fracs` and `factors` are tuples) and the file they were read from."""

    __slots__ = ()

    def sub(self, name):
        """A folder under out_root."""
        return os.path.join(self.out_root, name)

    def sample_params(self):
        """dict: SAMPLE_PARAMS' values and, under "gate", SAMPLE_GATE_PARAMS'
        (JSON types: lists, not tuples)."""
        out = dict((k, _json_value(getattr(self, k))) for k in SAMPLE_PARAMS)
        out["gate"] = dict((k, _json_value(self.gate[k])) for k in SAMPLE_GATE_PARAMS)
        return out

    def report_params(self):
        """dict: REPORT_PARAMS' values and, under "gate", REPORT_GATE_PARAMS'
        (JSON types)."""
        out = dict((k, _json_value(getattr(self, k))) for k in REPORT_PARAMS)
        out["gate"] = dict((k, _json_value(self.gate[k])) for k in REPORT_GATE_PARAMS)
        return out


def _strip_docs(obj):
    if isinstance(obj, dict):
        return dict((k, _strip_docs(v)) for k, v in obj.items() if not str(k).startswith("_"))
    return obj


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int_at_least(probs, name, v, lo):
    if not (_is_int(v) and v >= lo):
        probs.append("%s must be an integer >= %d, got %r" % (name, lo, v))


def _open_interval(probs, name, v, lo, hi, what="a number"):
    if not (_is_num(v) and lo < v < hi):
        probs.append("%s must be %s in (%g, %g), got %r" % (name, what, lo, hi, v))


def _check_gate(probs, g):
    if not isinstance(g, dict):
        probs.append("gate must be an object with exactly the keys %s, got %r"
                     % (", ".join(GATE_KEYS), g))
        return
    missing = [k for k in GATE_KEYS if k not in g]
    unknown = sorted(k for k in g if k not in GATE_KEYS)
    if missing:
        probs.append("gate: missing key(s): %s" % ", ".join(missing))
    if unknown:
        probs.append("gate: unknown key(s): %s" % ", ".join(unknown))
    if "donor_fracs" in g:
        f = g["donor_fracs"]
        if not (isinstance(f, list) and len(f) == 4 and all(_is_num(x) and x > 0 for x in f)):
            probs.append("gate.donor_fracs must be four positive numbers (f_ref, f_rank, f_null, "
                         "f_check), got %r" % (f,))
        elif abs(sum(float(x) for x in f) - 1.0) > _FRACS_TOL:
            probs.append("gate.donor_fracs must sum to 1 within %g, got %r (sum %.12g)"
                         % (_FRACS_TOL, f, sum(float(x) for x in f)))
    if "n_ref" in g:
        _int_at_least(probs, "gate.n_ref", g["n_ref"], 2)
        if _is_int(g["n_ref"]) and g["n_ref"] > N_REF_MAX:
            probs.append("gate.n_ref must be at most %d, got %r: above it the bandwidths' "
                         "median distance is taken on a subsample of the reference rows, "
                         "which is not (R5)" % (N_REF_MAX, g["n_ref"]))
    for k in ("n_rank", "n_null", "n_null_min", "rows_per_donor"):
        if k in g:
            _int_at_least(probs, "gate." + k, g[k], 1)
    if "factors" in g:
        f = g["factors"]
        if not (isinstance(f, list) and f
                and all(_is_num(x) and x > 0 and math.isfinite(x) for x in f)):
            probs.append("gate.factors must be a non-empty list of positive finite numbers, "
                         "got %r" % (f,))
    if "alpha" in g:
        _open_interval(probs, "gate.alpha", g["alpha"], 0.0, 1.0)


def _check_jobs(probs, jb):
    if not (isinstance(jb, dict) and sorted(jb) == sorted(("queue",) + JOB_KINDS)):
        probs.append("jobs must have exactly the keys queue, %s, got %r"
                     % (", ".join(JOB_KINDS), sorted(jb) if isinstance(jb, dict) else jb))
        return
    if not (isinstance(jb["queue"], str) and jb["queue"]):
        probs.append("jobs.queue must be a queue name, got %r" % (jb["queue"],))
    for kind in JOB_KINDS:
        j = jb[kind]
        if not (isinstance(j, dict) and sorted(j) == ["mem", "ncpus", "walltime"]):
            probs.append("jobs.%s must have exactly the keys ncpus, mem, walltime" % kind)
            continue
        _int_at_least(probs, "jobs.%s.ncpus" % kind, j["ncpus"], 1)
        if not (isinstance(j["mem"], str) and _MEM.fullmatch(j["mem"])):
            probs.append("jobs.%s.mem must be like 32gb, got %r" % (kind, j["mem"]))
        if not (isinstance(j["walltime"], str) and _WALL.fullmatch(j["walltime"])):
            probs.append("jobs.%s.walltime must be hh:mm:ss, got %r" % (kind, j["walltime"]))


def validate(data):
    """The problems of a configuration dict, every one, as strings (empty
    when usable). `data` has no `_` keys."""
    if not isinstance(data, dict):
        return ["the configuration must be a JSON object, got %s" % type(data).__name__]
    probs = []
    missing = [k for k in KEYS if k not in data]
    unknown = sorted(k for k in data if k not in KEYS)
    if missing:
        probs.append("missing key(s): %s" % ", ".join(missing))
    if unknown:
        probs.append("unknown key(s): %s" % ", ".join(unknown))
    has = lambda k: k in data  # noqa: E731
    if has("out_root") and not (isinstance(data["out_root"], str)
                                and os.path.isabs(data["out_root"])):
        probs.append("out_root must be an absolute path, got %r" % (data["out_root"],))
    if has("seed"):
        _int_at_least(probs, "seed", data["seed"], 0)
    for k in ("n_post", "batch_rows", "bootstrap_n", "dpi", "ncpus"):
        if has(k):
            _int_at_least(probs, k, data[k], 1)
    if has("oracle_tol"):
        v = data["oracle_tol"]
        if not (_is_num(v) and 0.0 < v <= 0.1):
            probs.append("oracle_tol must be a number of nats in (0, 0.1], got %r" % (v,))
    if has("as_recorded") and not isinstance(data["as_recorded"], bool):
        probs.append("as_recorded must be true or false, got %r" % (data["as_recorded"],))
    if has("sim_draw_rows_per_donor"):
        v = data["sim_draw_rows_per_donor"]
        if v is not None and not (_is_int(v) and v >= 1):
            probs.append("sim_draw_rows_per_donor must be null (every row) or an integer >= 1, "
                         "got %r" % (v,))
    if has("edge_eps"):
        _open_interval(probs, "edge_eps", data["edge_eps"], 0.0, 0.5)
    if has("gate"):
        _check_gate(probs, data["gate"])
    if has("level"):
        _open_interval(probs, "level", data["level"], 0.0, 1.0)
    if has("formats"):
        fm = data["formats"]
        if not (isinstance(fm, list) and fm and all(f in FORMATS for f in fm)
                and len(set(fm)) == len(fm)):
            probs.append("formats must be a non-empty list of distinct %s, got %r"
                         % ("/".join(FORMATS), fm))
    if has("validate_against"):
        va = data["validate_against"]
        if va is not None:
            if not (isinstance(va, str) and os.path.isabs(va)):
                probs.append("validate_against must be null or the absolute out_root of the "
                             "synthetic evaluation, got %r" % (va,))
            elif data.get("as_recorded") is not True:
                probs.append("validate_against is set but as_recorded is not true: the validation "
                             "compares a synthetic project read as recorded with its synthetic "
                             "evaluation (D-162)")
    if has("jobs"):
        _check_jobs(probs, data["jobs"])
    return probs


def load(path):
    """The RecEvalConfig of a file (ASCII JSON); RecEvalConfigError listing
    every problem, prefixed with the path."""
    with open(path, "rb") as fh:
        blob = fh.read()
    try:
        raw = json.loads(blob.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise RecEvalConfigError("%s: not ASCII (%s)" % (path, exc))
    except ValueError as exc:
        raise RecEvalConfigError("%s: not valid JSON (%s)" % (path, exc))
    data = _strip_docs(raw)
    probs = validate(data)
    if probs:
        raise RecEvalConfigError("%s: %s" % (path, "; ".join(probs)))
    data = dict(data)
    data["formats"] = tuple(data["formats"])
    gate = dict(data["gate"])
    gate["donor_fracs"] = tuple(gate["donor_fracs"])
    gate["factors"] = tuple(gate["factors"])
    data["gate"] = gate
    data["jobs"] = dict(data["jobs"])
    return RecEvalConfig(path=os.path.abspath(path), **data)
