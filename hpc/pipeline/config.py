"""The project configuration: schema, loading, and the capability table.

One JSON file per data set, `projects/<tag>/<tag>.json` (specs/SPEC.md
sec. 3.1). Every key of SCHEMA is required; nothing has a silent default.
A key starting with "_" documents the key after it and is ignored. Every
problem found is reported at once, not the first one only, so one refusal
fixes the whole file.

The capability table (NEEDS_CODE) names the keys that the code of a later
block honours: until that block lands, a run that would use the key is
refused with the block named. This is what keeps a configuration honest: a
value the code cannot carry to the trainer must never look as if it ran.

The pipeline is data-set-agnostic (D-091): nothing here names a project.

A synthetic data set (block 15, `"kind": "synthetic"`) has its own schema,
SCHEMA_SYNTHETIC: no recordings, no simulation campaign, the banks built by
the bench generator from `synthetic.*`, one search per entry of
`synthetic.searches`; `load_config` refuses it unless `synthetic_ok`.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import hashlib
import json
import os
import re

__all__ = ["Config", "ConfigError", "load_config", "unhonoured_keys",
           "SCHEMA", "SCHEMA_SYNTHETIC", "is_synthetic", "NEEDS_CODE", "CONFIG_VERSION"]

CONFIG_VERSION = 10          # v10, 2026-10-19: D-101 (values only: gp_seed, max_cores)

_WALLTIME = re.compile(r"^[0-9]+:[0-5][0-9]:[0-5][0-9]$")
_MEM = re.compile(r"^[0-9]+(kb|mb|gb|tb)$")
_TAG = re.compile(r"^[a-z0-9_]+$")
_CAMPAIGNS = ("S-A1", "S-A2", "S-A5", "S-A25")
_ARMS = ("A0", "A0s", "A1", "A2", "A2s", "A3", "A5", "A_ref", "shuffled")


class ConfigError(ValueError):
    """Every problem of a configuration, as one message with one line each."""

    def __init__(self, problems):
        self.problems = list(problems)
        ValueError.__init__(self, "\n".join(self.problems))


# ---------------------------------------------------------------------------
# The schema: (dotted key, kind). Kinds are checked by _check_value.
# ---------------------------------------------------------------------------

SCHEMA = (
    ("config_version", "version"),
    ("dataset", "tag"),
    ("paths.sbix_dir", "abspath"),
    ("paths.sbi_dir", "abspath"),
    ("paths.work_root", "abspath"),
    ("paths.banks_root", "abspath"),
    ("paths.runs_root", "abspath"),
    ("paths.tune_root", "abspath"),
    ("envs.extract", "str"),
    ("envs.train", "str"),
    ("real.cohort_config", "abspath"),
    ("real.cohort_tag", "str"),
    ("real.extract_root", "abspath"),
    ("sim.sim_main", "abspath"),
    ("sim.campaign_glob", "str"),
    ("sim.out_root", "abspath"),
    ("sim.tools", "abspath"),
    ("sim.plan_args", "str_or_empty"),
    ("sim.test_walltime", "walltime"),
    ("sim.walltime", "walltime"),
    ("sim.ncpus", "int>=1"),
    ("sim.queue", "str"),
    ("sim.concurrency", "int>=1"),
    ("freeze.out", "relpath"),
    ("freeze.campaigns", "str"),
    ("freeze.require_conn_rule", "str"),
    ("freeze.exclude", "map_str_str"),
    ("bank.on_axis", "str"),
    ("bank.on_declared_box", "box2"),
    ("bank.topology_axes", "list_str_nonempty"),
    ("bank.expected_p", "int>=1"),
    ("bank.t_win_s", "float>0"),
    ("bank.n_e", "int>=1"),
    ("bank.floor_hz", "float>0"),
    ("bank.floor_rtol", "float>0"),
    ("bank.sample_seed", "int>=0"),
    ("bank.bootstrap_resamples", "int>=1"),
    ("bank.bootstrap_seed", "int>=0"),
    ("bank.bytes_sample_rows", "int>=1"),
    ("bank.rows_per_shard", "int>=1"),
    ("bank.queue", "str"),
    ("bank.concurrency", "int>=1"),
    ("bank.walltime", "walltime_or_auto"),
    ("bank.verify.ncpus", "int>=1"),
    ("bank.verify.mem", "mem"),
    ("bank.verify.walltime", "walltime"),
    ("bank.max_rerun_rounds", "int>=0"),
    ("split.fractions", "fracs3"),
    ("search.campaign", "campaign"),
    ("search.arm", "arm"),
    ("search.shape_anchors.p", "int>=1"),
    ("search.shape_anchors.d_theta", "int>=1"),
    ("search.shape_anchors.embedding_dim", "int>=1"),
    ("search.space_overrides", "space_overrides"),
    ("search.flow.conditioner_hidden_layers", "int>=1"),
    ("search.flow.num_bins", "int>=1"),
    ("search.n_initial_points", "int>=1"),
    ("search.gp_rounds", "int>=1"),
    ("search.trials_per_round", "int>=1"),
    ("search.max_concurrent_trials", "int>=1"),
    ("search.gp_seed", "int>=0"),
    ("search.train_seed", "int>=0"),
    ("search.gp_noise_sigma", "float>0_or_null"),
    ("search.trial.passes", "float>0"),
    ("search.trial.epochs", "int>=1"),
    ("search.trial.ncpus", "int>=1"),
    ("search.trial.mem", "mem"),
    ("search.trial.walltime", "walltime"),
    ("search.trial.queue", "str"),
    ("search.trial.max_resubmits", "int>=0"),
    ("search.budget_gate.probe_steps", "int>=1"),
    ("search.budget_gate.probe_val_rows", "int>=1"),
    ("search.budget_gate.safety_factor", "float>=1"),
    ("search.max_total_walltime_h", "float>0_or_null"),
    ("search.walltime_reading", "elapsed"),
    ("search.resubmit_killed", "bool"),
    ("search.resubmit_node_faults", "bool"),
    ("best_fit.split_seed", "int>=0"),
    ("best_fit.member_seeds", "list_int_distinct_nonempty"),
    ("best_fit.passes", "float>0"),
    ("best_fit.epochs", "int>=1"),
    ("best_fit.patience_epochs", "int>=1"),
    ("best_fit.n_posterior_draws", "int>=1"),
    ("best_fit.ncpus", "int>=1"),
    ("best_fit.mem", "mem"),
    ("best_fit.walltime", "walltime"),
    ("best_fit.queue", "str"),
    ("best_fit.max_resubmits", "int>=0"),
    ("best_fit.resubmit_node_faults", "bool"),
    ("loader.buffer_shards", "int>=1"),
    ("driver.queue", "str"),
    ("driver.ncpus", "int>=1"),
    ("driver.mem", "mem"),
    ("driver.walltime", "walltime"),
    ("driver.poll_seconds", "int>=1"),
    ("resources.max_cores", "int>=1"),
)

# [block 15, D-118..D-123] A synthetic data set (`"kind": "synthetic"`): no
# recordings and no simulation campaign, so the keys of S1-S4 and the bank's
# go; the banks are the bench builder's (`synthetic.*`), and every search of
# `synthetic.searches` carries its own campaign, so search.campaign and
# search.arm go too. Everything else is the recorded-data schema's.
_NOT_SYNTHETIC = ("real.", "sim.", "freeze.", "bank.")
SCHEMA_SYNTHETIC = tuple(
    (k, kind) for k, kind in SCHEMA
    if not k.startswith(_NOT_SYNTHETIC)
    and k not in ("search.campaign", "search.arm", "paths.sbix_dir")) + (
    ("kind", "synthetic_kind"),
    ("synthetic.builder.provider", "synthetic_provider"),
    ("synthetic.builder.fs", "float>0"),
    ("synthetic.builder.T_win", "float>0"),
    ("synthetic.builder.n_neurons", "int>=1"),
    ("synthetic.builder.wells_per_donor", "int>=1"),
    ("synthetic.builder.donors_per_batch", "int>=1"),
    ("synthetic.builder.n_classes", "int>=1"),
    ("synthetic.builder.tau_ov", "float>0"),
    ("synthetic.builder.class_centres", "synthetic_centres"),
    ("synthetic.builder.centre_scale", "float>0"),
    ("synthetic.sim.seed", "int>=0"),
    ("synthetic.sim.n_theta", "int>=1"),
    ("synthetic.sim.n_windows", "int>=1"),
    ("synthetic.sim.traces_per_shard", "int>=1"),
    ("synthetic.real", "synthetic_real_list"),
    ("synthetic.searches", "synthetic_search_list"),
    ("synthetic.bank_job.ncpus", "int>=1"),
    ("synthetic.bank_job.mem", "mem"),
    ("synthetic.bank_job.walltime", "walltime"),
    ("synthetic.bank_job.queue", "str"),
    ("synthetic.bank_job.max_resubmits", "int>=0"),
)
_GAP_MODES = ("range_shift", "drift", "contamination")
_REAL_KEYS = ("name", "seed", "donors_per_class", "n_windows", "pi", "gap_modes")
_SEARCH_KEYS = ("name", "campaign", "real")

# D-099: the cap must hold the sequencer's job and the largest single job the
# configuration sizes (a smaller one would leave that job waiting for ever)
_SIZED_JOBS = ("search.trial.ncpus", "best_fit.ncpus", "sim.ncpus", "bank.verify.ncpus")
_SIZED_JOBS_SYNTHETIC = ("search.trial.ncpus", "best_fit.ncpus", "synthetic.bank_job.ncpus")


def is_synthetic(data):
    """[block 15] True for a synthetic data set's configuration."""
    return isinstance(data, dict) and data.get("kind") == "synthetic"

# Keys whose values the pipeline can only honour once a later block exists:
# (key, block number, what the block adds, the stages that use the key).
# Checked by unhonoured_keys() for the stages still to run. A later block
# removes its rows here in the same commit that builds them, and the smoke
# test asserts the exact list, so the two cannot drift. Empty since block 7
# (2026-10-16): every key of SCHEMA reaches the code that honours it. The
# mechanism stays for a future key.
NEEDS_CODE = ()


# ---------------------------------------------------------------------------
# Value checks
# ---------------------------------------------------------------------------

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and v == v and v not in (float("inf"), float("-inf")))


def _check_value(key, kind, v):
    """Return a problem string, or None when the value satisfies `kind`."""
    bad = "%s: " % key
    if kind == "version":
        if v != CONFIG_VERSION:
            return bad + "config_version must be %d, got %r" % (CONFIG_VERSION, v)
    elif kind == "tag":
        if not (isinstance(v, str) and _TAG.match(v)):
            return bad + "must match [a-z0-9_]+, got %r" % (v,)
    elif kind == "str":
        if not (isinstance(v, str) and v.strip()):
            return bad + "must be a non-empty string, got %r" % (v,)
    elif kind == "str_or_empty":
        if not isinstance(v, str):
            return bad + "must be a string, got %r" % (v,)
    elif kind == "abspath":
        if not (isinstance(v, str) and v.startswith("/")):
            return bad + "must be an absolute path (no '~'), got %r" % (v,)
    elif kind == "relpath":
        if not (isinstance(v, str) and v and not v.startswith("/")):
            return bad + "must be a relative path, got %r" % (v,)
    elif kind == "walltime":
        if not (isinstance(v, str) and _WALLTIME.match(v)):
            return bad + "must be hh:mm:ss, got %r" % (v,)
    elif kind == "walltime_or_auto":
        if not (isinstance(v, str) and (v == "auto" or _WALLTIME.match(v))):
            return bad + "must be hh:mm:ss or 'auto', got %r" % (v,)
    elif kind == "mem":
        if not (isinstance(v, str) and _MEM.match(v)):
            return bad + "must be a PBS memory such as '160gb', got %r" % (v,)
    elif kind == "int>=1":
        if not (_is_int(v) and v >= 1):
            return bad + "must be an integer >= 1, got %r" % (v,)
    elif kind == "int>=0":
        if not (_is_int(v) and v >= 0):
            return bad + "must be an integer >= 0, got %r" % (v,)
    elif kind == "float>0":
        if not (_is_num(v) and v > 0):
            return bad + "must be a number > 0, got %r" % (v,)
    elif kind == "float>=1":
        if not (_is_num(v) and v >= 1):
            return bad + "must be a number >= 1, got %r" % (v,)
    elif kind == "float>0_or_null":
        if v is not None and not (_is_num(v) and v > 0):
            return bad + "must be a number > 0 or null, got %r" % (v,)
    elif kind == "bool":
        if not isinstance(v, bool):
            return bad + "must be true or false, got %r" % (v,)
    elif kind == "elapsed":
        if v != "elapsed":
            return bad + "only 'elapsed' is implemented, got %r" % (v,)
    elif kind == "campaign":
        if v not in _CAMPAIGNS:
            return bad + "must be one of %s, got %r" % (", ".join(_CAMPAIGNS), v)
    elif kind == "arm":
        if v not in _ARMS:
            return bad + "must be one of %s, got %r" % (", ".join(_ARMS), v)
    elif kind == "map_str_str":
        if not (isinstance(v, dict) and all(isinstance(k, str) and isinstance(x, str)
                                           for k, x in v.items())):
            return bad + "must be an object of string -> string"
    elif kind == "box2":
        if not (isinstance(v, list) and len(v) == 2 and all(_is_num(x) for x in v)
                and v[0] < v[1]):
            return bad + "must be [lo, hi] with lo < hi, got %r" % (v,)
    elif kind == "list_str_nonempty":
        if not (isinstance(v, list) and v and all(isinstance(x, str) and x for x in v)):
            return bad + "must be a non-empty list of strings, got %r" % (v,)
    elif kind == "fracs3":
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)
                and all(0.0 < x < 1.0 for x in v)):
            return bad + "must be three fractions in (0, 1), got %r" % (v,)
        if abs(sum(v) - 1.0) > 1e-9:
            return bad + "the fractions must sum to 1 (within 1e-9), got %r" % (sum(v),)
    elif kind == "list_int_distinct_nonempty":
        if not (isinstance(v, list) and v and all(_is_int(x) and x >= 0 for x in v)):
            return bad + "must be a non-empty list of integers >= 0, got %r" % (v,)
        if len(set(v)) != len(v):
            return bad + "the seeds must be distinct, got %r" % (v,)
    elif kind == "synthetic_kind":
        if v != "synthetic":
            return bad + "must be 'synthetic' (or absent, for a recorded data set), got %r" % (v,)
    elif kind == "synthetic_provider":
        if v != "bench":
            return bad + "must be 'bench' (the bench generator, D-119), got %r" % (v,)
    elif kind == "synthetic_centres":
        if v not in ("simplex", "regular"):
            return bad + "must be 'simplex' or 'regular', got %r" % (v,)
    elif kind == "synthetic_real_list":
        return _check_list(bad, v, _REAL_KEYS, _check_real_item)
    elif kind == "synthetic_search_list":
        return _check_list(bad, v, _SEARCH_KEYS, _check_search_item)
    elif kind == "space_overrides":
        if not isinstance(v, dict):
            return bad + "must be an object axis -> [lo, hi] or [levels]"
        for axis, rng in v.items():
            if not (isinstance(axis, str) and re.match(r"^[a-z0-9_]+$", axis)):
                return bad + "axis name %r is not [a-z0-9_]+" % (axis,)
            if not (isinstance(rng, list) and len(rng) >= 2
                    and all(_is_num(x) or isinstance(x, str) for x in rng)):
                return bad + "%s must be a list of at least two values, got %r" % (axis, rng)
            if len(rng) == 2 and all(_is_num(x) for x in rng) and rng[0] > rng[1]:
                return bad + "%s: lo > hi in %r" % (axis, rng)
    else:
        raise AssertionError("unknown schema kind %r" % kind)
    return None


# [block 15] the lists of a synthetic configuration: objects with exactly the
# listed keys, each item checked by its own function

_REAL_NAME = re.compile(r"^[A-Za-z0-9]+$")
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _check_list(bad, v, keys, item_check):
    if not (isinstance(v, list) and v):
        return bad + "must be a non-empty list of objects, got %r" % (v,)
    probs = []
    for i, item in enumerate(v):
        if not isinstance(item, dict):
            probs.append("[%d] must be an object, got %r" % (i, item))
            continue
        missing = [k for k in keys if k not in item]
        unknown = sorted(k for k in item if k not in keys)
        if missing or unknown:
            probs.append("[%d] must have exactly the keys %s (missing %s, unknown %s)"
                         % (i, ", ".join(keys), missing or "none", unknown or "none"))
            continue
        p = item_check(item)
        if p:
            probs.append("[%d] %s" % (i, p))
    return (bad + "; ".join(probs)) if probs else None


def _check_real_item(r):
    if not (isinstance(r["name"], str) and _REAL_NAME.match(r["name"])):
        return "name must match [A-Za-z0-9]+, got %r" % (r["name"],)
    if not (_is_int(r["seed"]) and r["seed"] >= 0):
        return "%s: seed must be an integer >= 0, got %r" % (r["name"], r["seed"])
    for k in ("donors_per_class", "n_windows"):
        if not (_is_int(r[k]) and r[k] >= 1):
            return "%s: %s must be an integer >= 1, got %r" % (r["name"], k, r[k])
    if not (_is_num(r["pi"]) and 0.0 <= r["pi"] <= 1.0):
        return "%s: pi must be a number in [0, 1], got %r" % (r["name"], r["pi"])
    m = r["gap_modes"]
    if not (isinstance(m, list) and m and all(isinstance(x, str) for x in m)):
        return "%s: gap_modes must be a non-empty list of modes, got %r" % (r["name"], m)
    if any(x not in _GAP_MODES for x in m) or len(set(m)) != len(m):
        return "%s: gap_modes must be distinct modes among %s, got %r" % (
            r["name"], ", ".join(_GAP_MODES), m)
    return None


def _check_search_item(q):
    if not (isinstance(q["name"], str) and _RUN_NAME.match(q["name"])):
        return "name must match [A-Za-z0-9][A-Za-z0-9_-]*, got %r" % (q["name"],)
    if q["campaign"] not in _CAMPAIGNS:
        return "%s: campaign must be one of %s, got %r" % (q["name"], ", ".join(_CAMPAIGNS),
                                                           q["campaign"])
    if not (isinstance(q["real"], str) and q["real"]):
        return "%s: real must name an entry of synthetic.real, got %r" % (q["name"], q["real"])
    return None


def _synthetic_cross_checks(data, problems):
    """[block 15] The rules between keys of a synthetic configuration, on the
    keys that passed their own checks (sec. 3.1, the synthetic kind)."""
    failed = set(p.split(":")[0] for p in problems)

    def ok(*keys):
        return not any(k in failed for k in keys)

    def obj(d, k):
        v = d.get(k) if isinstance(d, dict) else None
        return v if isinstance(v, dict) else {}

    out = []
    syn = obj(data, "synthetic")
    b = obj(syn, "builder")
    sim = obj(syn, "sim")
    if ok("synthetic.builder.wells_per_donor") and b["wells_per_donor"] < 2:
        out.append("synthetic.builder.wells_per_donor: must be >= 2 (the builder's two realisations "
                   "per theta; arm A5's pairs of two wells), got %r" % (b["wells_per_donor"],))
    if ok("synthetic.builder.n_classes") and b["n_classes"] < 2:
        out.append("synthetic.builder.n_classes: must be >= 2 (the DSN loss's classes, F-t), got %r"
                   % (b["n_classes"],))
    if ok("synthetic.builder.class_centres", "synthetic.builder.centre_scale") \
            and b["class_centres"] != "regular" and b["centre_scale"] != 1.0:
        out.append("synthetic.builder.centre_scale: %r needs class_centres 'regular' (the builder "
                   "scales only the regular simplex)" % (b["centre_scale"],))
    if ok("synthetic.builder.wells_per_donor", "synthetic.sim.n_theta",
          "synthetic.sim.traces_per_shard"):
        wpd, tps = b["wells_per_donor"], sim["traces_per_shard"]
        n_w = sim["n_theta"] * wpd
        if tps % wpd or n_w % tps:
            out.append("synthetic.sim.traces_per_shard: %d must be a multiple of wells_per_donor %d "
                       "that divides n_theta * wells_per_donor = %d (one donor's wells in one "
                       "shard; equal shards)" % (tps, wpd, n_w))
    if ok("synthetic.real", "synthetic.searches", "synthetic.sim.seed"):
        names = [r["name"] for r in syn["real"]]
        dup = sorted(set(n for n in names if names.count(n) > 1))
        if dup:
            out.append("synthetic.real: the names must be unique, %s repeated" % ", ".join(dup))
        seeds = [("synthetic.sim", sim["seed"])] + [(r["name"], r["seed"]) for r in syn["real"]]
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                if seeds[i][1] == seeds[j][1]:
                    out.append("synthetic.real: %s and %s share the seed %d; the arms' seeds must "
                               "be pairwise distinct (D-120, D-121; F-aw)"
                               % (seeds[i][0], seeds[j][0], seeds[i][1]))
        runs = [q["name"] for q in syn["searches"]]
        dup = sorted(set(n for n in runs if runs.count(n) > 1))
        if dup:
            out.append("synthetic.searches: the names must be unique, %s repeated" % ", ".join(dup))
        seen = {}
        for q in syn["searches"]:
            if q["real"] not in names:
                out.append("synthetic.searches: %s names the real bank %r, which synthetic.real "
                           "does not list (%s)" % (q["name"], q["real"], ", ".join(names)))
            key = (q["campaign"], q["real"])
            if key in seen:
                out.append("synthetic.searches: %s and %s are the same search (%s on %s)"
                           % (seen[key], q["name"], key[0], key[1]))
            seen.setdefault(key, q["name"])
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _strip_docs(obj):
    """Drop every key starting with '_' recursively (documentation keys),
    into the objects of lists too (block 15: the synthetic lists)."""
    if isinstance(obj, dict):
        return dict((k, _strip_docs(v)) for k, v in obj.items()
                    if not (isinstance(k, str) and k.startswith("_")))
    if isinstance(obj, list):
        return [_strip_docs(x) for x in obj]
    return obj


def _walk(obj, prefix=""):
    """Every dotted leaf path of a nested dict (objects whose children the
    schema does not enumerate are leaves: SCHEMA names them as one key)."""
    for k, v in obj.items():
        path = prefix + k
        if isinstance(v, dict) and path not in _OBJECT_KEYS:
            for p in _walk(v, path + "."):
                yield p
        else:
            yield path


_OBJECT_KEYS = frozenset(("freeze.exclude", "search.space_overrides"))


def _lookup(data, dotted):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


class Config(object):
    """The loaded configuration: `data` (documentation keys removed), the
    file's path and sha256, and dotted access through `get`."""

    def __init__(self, data, path, sha256):
        self.data = data
        self.path = path
        self.sha256 = sha256

    @property
    def tag(self):
        return self.data["dataset"]

    def get(self, dotted):
        return _lookup(self.data, dotted)

    def __getitem__(self, key):
        return self.data[key]

    def section(self, name):
        return self.data[name]


def validate(data):
    """Every problem of a (documentation-stripped) configuration dict: the
    recorded-data schema, or with `"kind": "synthetic"` the synthetic one
    and its cross-checks (block 15)."""
    if "kind" in data and data["kind"] != "synthetic":
        return [_check_value("kind", "synthetic_kind", data["kind"])]
    synthetic = is_synthetic(data)
    schema = SCHEMA_SYNTHETIC if synthetic else SCHEMA
    sized = _SIZED_JOBS_SYNTHETIC if synthetic else _SIZED_JOBS
    problems = []
    present = set(_walk(data))
    expected = set(k for k, _ in schema)
    for key, kind in schema:
        try:
            v = _lookup(data, key)
        except KeyError:
            problems.append("%s: missing (every key is required)" % key)
            continue
        p = _check_value(key, kind, v)
        if p:
            problems.append(p)
    for key in sorted(present - expected):
        # A present path that is a prefix of an expected key is an object
        # the schema enumerates; only true strangers are unknown.
        if any(e.startswith(key + ".") for e in expected):
            continue
        problems.append("%s: unknown key (not in the schema%s)"
                        % (key, " of the synthetic kind" if synthetic else ""))
    named = ("resources.max_cores", "driver.ncpus") + sized
    if not any(p.split(":")[0] in named for p in problems):
        cap = int(_lookup(data, "resources.max_cores"))
        drv = int(_lookup(data, "driver.ncpus"))
        key, size = max(((k, int(_lookup(data, k))) for k in sized), key=lambda kv: kv[1])
        if cap < drv + size:
            problems.append("resources.max_cores: %d cannot hold the sequencer's job (driver.ncpus %d) "
                            "and the largest job (%s %d)" % (cap, drv, key, size))
    if synthetic:
        problems.extend(_synthetic_cross_checks(data, problems))
    return problems


def load_config(path, synthetic_ok=False):
    """Load and validate; raise ConfigError with every problem. A synthetic
    data set's configuration (block 15) is refused unless `synthetic_ok`:
    only the sequencer (pipeline.py) runs one; the tools that read a
    recorded data set's single search or its S1-S6 refuse it."""
    with open(path, "rb") as fh:
        blob = fh.read()
    try:
        raw = json.loads(blob.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise ConfigError(["%s: not ASCII (%s)" % (path, exc)])
    except ValueError as exc:
        raise ConfigError(["%s: not valid JSON (%s)" % (path, exc)])
    if not isinstance(raw, dict):
        raise ConfigError(["%s: the top level must be an object" % path])
    data = _strip_docs(raw)
    problems = validate(data)
    if problems:
        raise ConfigError(problems)
    if is_synthetic(data) and not synthetic_ok:
        raise ConfigError(["%s: a synthetic data set's configuration (\"kind\": \"synthetic\", block "
                           "15); this tool reads a recorded data set (its single search, its "
                           "S1-S6) and does not run on one -- the synthetic kind runs through "
                           "pipeline.py" % path])
    return Config(data, os.path.abspath(path), hashlib.sha256(blob).hexdigest())


def unhonoured_keys(cfg, pending_stages=None):
    """The configuration's keys that no built block honours yet, among
    those the stages in `pending_stages` (names; None = every stage) would
    use: [(key, block, what)], in NEEDS_CODE order. A key is reported only
    when the configuration sets it (the sub-key `search.space_overrides.
    head_pool_ops` only when the axis is in the overrides)."""
    out = []
    for key, block, what, users in NEEDS_CODE:
        if pending_stages is not None and not any(u in pending_stages for u in users):
            continue
        try:
            _lookup(cfg.data, key)
        except KeyError:
            continue
        out.append((key, block, what))
    return out
