"""Block 40: the schemas of a model bundle's manifest, of its extraction
block and of its card's header, and their validators (specs/SPEC.md sec.
2.7 and block 40; D-173..D-176).

CONTRACT (wave A): the constants below are fixed by specs/SPEC.md; the
validators' bodies are the implementer's. Standard library only.

Pure ASCII, LF only.
"""

import datetime
import math
import re

BUNDLE_SCHEMA = "sbitool_bundle/2"
# the first format (block 41): refused, naming both (block 48; no bundle of it exists
# outside the sandbox)
BUNDLE_SCHEMA_1 = "sbitool_bundle/1"

# manifest.json: exactly these top-level keys (sec. 2.7, "The model bundle")
MANIFEST_KEYS = (
    "schema", "name", "created", "card_complete", "arm", "M", "p", "d_z", "W", "members",
    "architecture", "parameters", "extraction", "gate", "sample", "provenance", "software",
    "training", "files")
MEMBER_KEYS = ("file", "sha256", "seed", "state_sha256", "split_seed", "split_hash")
ARCHITECTURE_KEYS = (
    "backbone", "embedding_size", "depth_exponent", "width_multiplier", "block_family",
    "head_fusion", "dropout", "head_pool_ops", "hidden_features", "num_transforms", "num_bins",
    "hidden_layers", "z_score_theta", "z_score_x", "backbone_repr")
BACKBONES = ("dsn", "fixed-stats", "fallback")
PARAMETER_KEYS = ("names", "units", "bounds")
# the parameters that turn a recording into windows ("From a recording to windows")
EXTRACTION_KEYS = (
    "electrodes_per_subset", "n_subsets", "mfr_threshold_hz", "w_size_s", "gaussian_window_s",
    "grid_width", "index_base", "pitch_um", "fs_raw_hz", "detection")
# extraction["detection"]: None, or exactly these (D-176; SpikeInterface's detect_peaks)
DETECTION_KEYS = ("method", "peak_sign", "detect_threshold", "exclude_sweep_ms")
DETECTION_DEFAULT = {"method": "by_channel", "peak_sign": "neg", "detect_threshold": 5.0,
                     "exclude_sweep_ms": 1.0}
GATE_KEYS = ("factors", "alpha", "n_ref", "n_rank", "n_null", "k_gate", "size_mean", "size_lo",
             "size_hi", "level")
SAMPLE_KEYS = ("n_post", "seed", "batch_rows")
PROVENANCE_KEYS = ("project", "ensemble", "search", "campaign", "best_fit_complete_sha256",
                   "sample_json_sha256", "commit", "sources")
SOFTWARE_KEYS = ("python", "torch", "sbi", "zuko", "numpy", "safetensors")
# manifest["training"]: None, or exactly these (block 48): the members' curves
# training/member_<k>.json, k = 0..members-1, and training/search.json when search
TRAINING_KEYS = ("members", "search")
TRAINING_MEMBER_FILE = "training/member_%d.json"
TRAINING_SEARCH_FILE = "training/search.json"
TRAINING_PREFIX = "training/"

# the members of the zip besides members/member_<k>.safetensors
MEMBER_FILE = "members/member_%d.safetensors"
FIXED_MEMBERS = ("manifest.json", "CARD.md", "evaluation/sim.json", "evaluation/gate.safetensors")
GATE_ARRAYS = ("z_ref", "z_rank", "z_null")

# CARD.md's header: exactly these top-level keys, in this order ("The card")
CARD_KEYS = (
    "title", "scientific_question", "simulator", "training", "extraction", "references",
    "authors", "licence", "cite", "data", "software", "limitations")
CARD_SIMULATOR_KEYS = ("model_type", "reference", "fixed_parameters", "searched_parameters")
CARD_TRAINING_KEYS = ("arm", "members", "n_networks", "n_windows", "split_fracs",
                      "hyperparameters", "hyperparameter_search")
CARD_AUTHOR_KEYS = ("name", "contact")
CARD_DATA_KEYS = ("simulated_bank", "recordings")
CARD_BANK_KEYS = ("location", "public", "size_bytes", "sha256")
CARD_RECORDINGS_KEYS = ("location", "public")
TODO = "TODO"                      # the literal an unfilled card field holds
NOT_APPLICABLE = "not applicable"  # card["extraction"] of a bundle whose extraction is null


_HEX64 = re.compile(r"[0-9a-f]{64}")   # the form of a sha256 hex digest (hashlib's hexdigest)
_REPR_MAX = 80                        # characters of a value quoted in a problem


# ---------------------------------------------------------------- helpers
# Each helper appends to `probs` and returns nothing that can raise on a JSON
# or YAML value: no ordering of foreign values, no float() of a foreign value
# outside a guard, no repr() without _r.

def _r(v):
    """A short repr of any value, never raising (an int too long for str(),
    a structure too deep for repr())."""
    try:
        s = repr(v)
    except (ValueError, RecursionError):
        s = "<%s>" % type(v).__name__
    return s if len(s) <= _REPR_MAX else s[:_REPR_MAX - 3] + "..."


def _join(path, key):
    key = key if isinstance(key, str) else _r(key)
    return "%s.%s" % (path, key) if path else key


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    """A finite int or float, never a bool; an int beyond the float range is
    not a number here (json and YAML read a long literal as one)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def _is_str(v, nonempty=True):
    return isinstance(v, str) and (bool(v) or not nonempty)


def _is_hex64(v):
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def _keys(probs, path, d, keys):
    """d must be a dict with exactly `keys`; one problem per missing or
    unknown key. True when d is a dict (the caller then checks the values of
    the keys present)."""
    if not isinstance(d, dict):
        probs.append("%s must be an object with exactly the keys %s, got %s"
                     % (path or "the value", ", ".join(keys), _r(d)))
        return False
    for k in keys:
        if k not in d:
            probs.append("missing key %s" % _join(path, k))
    for k in sorted((k for k in d if k not in keys), key=_r):
        probs.append("unknown key %s" % _join(path, k))
    return True


def _int(probs, name, v, lo=None, allow_none=False):
    if allow_none and v is None:
        return
    if not (_is_int(v) and (lo is None or v >= lo)):
        probs.append("%s must be an integer%s%s, got %s"
                     % (name, "" if lo is None else " >= %d" % lo,
                        " or null" if allow_none else "", _r(v)))


def _num(probs, name, v, lo=None, hi=None, lo_open=True, hi_open=True):
    """A finite number in the interval (lo, hi) (each end open or closed,
    absent when None)."""
    ok = _is_num(v)
    if ok and lo is not None:
        ok = v > lo if lo_open else v >= lo
    if ok and hi is not None:
        ok = v < hi if hi_open else v <= hi
    if not ok:
        iv = ""
        if lo is not None or hi is not None:
            iv = " in %s%s, %s%s" % ("(" if lo_open else "[", "-inf" if lo is None else "%g" % lo,
                                     "inf" if hi is None else "%g" % hi, ")" if hi_open else "]")
        probs.append("%s must be a finite number%s, got %s" % (name, iv, _r(v)))


def _str(probs, name, v):
    if not _is_str(v):
        probs.append("%s must be a non-empty string, got %s" % (name, _r(v)))


def _choice(probs, name, v, choices):
    if not (isinstance(v, (str, int)) and not isinstance(v, bool) and v in choices):
        probs.append("%s must be one of %s, got %s" % (name, ", ".join(_r(c) for c in choices),
                                                       _r(v)))


def _hex(probs, name, v, allow_none=False):
    if allow_none and v is None:
        return
    if not _is_hex64(v):
        probs.append("%s must be a sha256 digest (64 lowercase hex digits)%s, got %s"
                     % (name, " or null" if allow_none else "", _r(v)))


def _list_of_str(probs, name, v, n=None):
    if not (isinstance(v, list) and all(_is_str(x) for x in v)):
        probs.append("%s must be a list of non-empty strings, got %s" % (name, _r(v)))
        return False
    if n is not None and len(v) != n:
        probs.append("%s has %d entries, p = %s" % (name, len(v), _r(n)))
        return False
    return True


def _iso_time(v):
    if not isinstance(v, str):
        return False
    s = v[:-1] + "+00:00" if v.endswith("Z") else v
    try:
        datetime.datetime.fromisoformat(s)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------- extraction

def _extraction(probs, path, d):
    if not _keys(probs, path, d, EXTRACTION_KEYS):
        return
    j = lambda k: _join(path, k)  # noqa: E731
    has = d.__contains__
    for k in ("electrodes_per_subset", "n_subsets", "grid_width"):
        if has(k):
            _int(probs, j(k), d[k], 1)
    if has("mfr_threshold_hz"):
        _num(probs, j("mfr_threshold_hz"), d["mfr_threshold_hz"], 0.0, None, lo_open=False)
    for k in ("w_size_s", "gaussian_window_s", "pitch_um", "fs_raw_hz"):
        if has(k):
            _num(probs, j(k), d[k], 0.0)
    if has("index_base"):
        _choice(probs, j("index_base"), d["index_base"], (0, 1))
    if has("detection") and d["detection"] is not None:
        dp = j("detection")
        det = d["detection"]
        if _keys(probs, dp, det, DETECTION_KEYS):
            if "method" in det:
                _choice(probs, _join(dp, "method"), det["method"], (DETECTION_DEFAULT["method"],))
            if "peak_sign" in det:
                _choice(probs, _join(dp, "peak_sign"), det["peak_sign"],
                        (DETECTION_DEFAULT["peak_sign"],))
            for k in ("detect_threshold", "exclude_sweep_ms"):
                if k in det:
                    _num(probs, _join(dp, k), det[k], 0.0)


def validate_extraction(data):
    """Every problem of an extraction block (a dict with exactly
    EXTRACTION_KEYS; detection None or exactly DETECTION_KEYS), as a list of
    strings naming the key; [] when valid. Never raises on a JSON value."""
    probs = []
    try:
        _extraction(probs, "extraction", data)
    except RecursionError:
        probs.append("extraction: too deeply nested to check")
    return probs


# ---------------------------------------------------------------- manifest

def _members(probs, d):
    mem = d["members"]
    if not isinstance(mem, list):
        probs.append("members must be a list of M objects, got %s" % _r(mem))
        return
    M = d.get("M")
    if _is_int(M) and M >= 1 and len(mem) != M:
        probs.append("members has %d entries but M = %s" % (len(mem), _r(M)))
    seeds = []
    for k, e in enumerate(mem):
        path = "members[%d]" % k
        if not _keys(probs, path, e, MEMBER_KEYS):
            continue
        if "file" in e and e["file"] != MEMBER_FILE % k:
            probs.append("%s.file must be %r (member k in the member order), got %s"
                         % (path, MEMBER_FILE % k, _r(e["file"])))
        for h in ("sha256", "state_sha256", "split_hash"):
            if h in e:
                _hex(probs, _join(path, h), e[h])
        for s in ("seed", "split_seed"):
            if s in e:
                _int(probs, _join(path, s), e[s])
        if _is_int(e.get("seed")):
            seeds.append((k, e["seed"]))
    for (k0, s0), (k1, s1) in zip(seeds, seeds[1:]):
        if s1 <= s0:
            probs.append("members[%d].seed = %s is not above members[%d].seed = %s: the members "
                         "are in the order of their seeds, ascending (sec. 2.5)"
                         % (k1, _r(s1), k0, _r(s0)))


def _architecture(probs, a):
    if not _keys(probs, "architecture", a, ARCHITECTURE_KEYS):
        return
    j = lambda k: "architecture." + k  # noqa: E731
    if "backbone" in a:
        _choice(probs, j("backbone"), a["backbone"], BACKBONES)
    for k in ("embedding_size", "hidden_features", "num_transforms", "num_bins"):
        if k in a:
            _int(probs, j(k), a[k], 1)
    if "hidden_layers" in a:
        _int(probs, j("hidden_layers"), a["hidden_layers"], 1, allow_none=True)
    if "depth_exponent" in a:
        _int(probs, j("depth_exponent"), a["depth_exponent"], 0)
    for k in ("block_family", "head_fusion", "head_pool_ops"):
        if k in a:
            _choice(probs, j(k), a[k], (0, 1))
    if "width_multiplier" in a:
        _num(probs, j("width_multiplier"), a["width_multiplier"], 0.0)
    if "dropout" in a:
        _num(probs, j("dropout"), a["dropout"], 0.0, 1.0, lo_open=False)
    for k in ("z_score_theta", "z_score_x"):
        if k in a and not (a[k] is None or _is_str(a[k])):
            probs.append("%s must be a string or null, got %s" % (j(k), _r(a[k])))
    if "backbone_repr" in a:
        _str(probs, j("backbone_repr"), a["backbone_repr"])


def _parameters(probs, par, p):
    if not _keys(probs, "parameters", par, PARAMETER_KEYS):
        return
    n = p if (_is_int(p) and p >= 1) else None
    if "names" in par:
        _list_of_str(probs, "parameters.names", par["names"], n)
    if "units" in par:
        _list_of_str(probs, "parameters.units", par["units"], n)
    if "bounds" in par and par["bounds"] is not None:
        b = par["bounds"]
        if not isinstance(b, list):
            probs.append("parameters.bounds must be null or a list of p pairs [lo, hi], got %s"
                         % _r(b))
            return
        if n is not None and len(b) != n:
            probs.append("parameters.bounds has %d entries, p = %s" % (len(b), _r(n)))
            return
        for k, pair in enumerate(b):
            name = "parameters.bounds[%d]" % k
            if not (isinstance(pair, list) and len(pair) == 2 and all(_is_num(x) for x in pair)):
                probs.append("%s must be a pair [lo, hi] of finite numbers, got %s"
                             % (name, _r(pair)))
            elif not pair[0] < pair[1]:
                probs.append("%s must have lo < hi, got %s" % (name, _r(pair)))


def _gate(probs, g):
    if not _keys(probs, "gate", g, GATE_KEYS):
        return
    if "factors" in g:
        f = g["factors"]
        if not (isinstance(f, list) and f and all(_is_num(x) and x > 0 for x in f)):
            probs.append("gate.factors must be a non-empty list of positive finite numbers, "
                         "got %s" % _r(f))
    for k in ("alpha", "size_mean", "size_lo", "size_hi", "level"):
        if k in g:
            _num(probs, "gate." + k, g[k], 0.0, 1.0)
    for k in ("n_ref", "n_rank", "n_null", "k_gate"):
        if k in g:
            _int(probs, "gate." + k, g[k], 1)


def _manifest(probs, d):
    if not _keys(probs, "", d, MANIFEST_KEYS):
        return
    has = d.__contains__
    if has("schema") and d["schema"] != BUNDLE_SCHEMA:
        if d["schema"] == BUNDLE_SCHEMA_1:
            probs.append("schema is %r, the first format, which this tool no longer reads: a "
                         "bundle must be %r (export it again)" % (BUNDLE_SCHEMA_1, BUNDLE_SCHEMA))
        else:
            probs.append("schema must be %r, got %s" % (BUNDLE_SCHEMA, _r(d["schema"])))
    if has("name"):
        _str(probs, "name", d["name"])
    if has("created") and not _iso_time(d["created"]):
        probs.append("created must be an ISO 8601 time, got %s" % _r(d["created"]))
    if has("card_complete") and not isinstance(d["card_complete"], bool):
        probs.append("card_complete must be true or false, got %s" % _r(d["card_complete"]))
    if has("arm"):
        _str(probs, "arm", d["arm"])
    for k in ("M", "p", "d_z", "W"):
        if has(k):
            _int(probs, k, d[k], 1)
    if has("members"):
        _members(probs, d)
    if has("architecture"):
        _architecture(probs, d["architecture"])
    if has("parameters"):
        _parameters(probs, d["parameters"], d.get("p"))
    if has("extraction") and d["extraction"] is not None:
        _extraction(probs, "extraction", d["extraction"])
    if has("gate"):
        _gate(probs, d["gate"])
    if has("sample") and _keys(probs, "sample", d["sample"], SAMPLE_KEYS):
        s = d["sample"]
        if "n_post" in s:
            _int(probs, "sample.n_post", s["n_post"], 1)
        if "seed" in s:
            _int(probs, "sample.seed", s["seed"])
        if "batch_rows" in s:
            _int(probs, "sample.batch_rows", s["batch_rows"], 1)
    if has("provenance") and _keys(probs, "provenance", d["provenance"], PROVENANCE_KEYS):
        pv = d["provenance"]
        for k in ("project", "ensemble", "search", "campaign", "commit"):
            if k in pv:
                _str(probs, "provenance." + k, pv[k])
        for k in ("best_fit_complete_sha256", "sample_json_sha256"):
            if k in pv:
                _hex(probs, "provenance." + k, pv[k])
        if "sources" in pv and not isinstance(pv["sources"], dict):
            probs.append("provenance.sources must be an object, got %s" % _r(pv["sources"]))
    if has("software") and _keys(probs, "software", d["software"], SOFTWARE_KEYS):
        for k in SOFTWARE_KEYS:
            if k in d["software"]:
                _str(probs, "software." + k, d["software"][k])
    if has("training") and d["training"] is not None:
        t = d["training"]
        if not isinstance(t, dict):
            probs.append("training must be null or an object with exactly the keys %s, got %s"
                         % (", ".join(TRAINING_KEYS), _r(t)))
        elif _keys(probs, "training", t, TRAINING_KEYS):
            if "members" in t:
                M = d.get("M")
                if not (_is_int(t["members"]) and t["members"] >= 1):
                    probs.append("training.members must be an integer >= 1, got %s"
                                 % _r(t["members"]))
                elif _is_int(M) and t["members"] != M:
                    probs.append("training.members is %s but M = %s" % (_r(t["members"]), _r(M)))
            if "search" in t and not isinstance(t["search"], bool):
                probs.append("training.search must be true or false, got %s" % _r(t["search"]))
    if has("files"):
        f = d["files"]
        if not isinstance(f, dict):
            probs.append("files must be an object member name -> sha256, got %s" % _r(f))
        else:
            for k in sorted(f, key=_r):
                if not _is_str(k):
                    probs.append("files: the member name %s is not a non-empty string" % _r(k))
                else:
                    _hex(probs, "files[%r]" % k, f[k])


def training_files(training):
    """The zip members a manifest's `training` announces: [] for null (or a
    value that is not a valid training object), else training/member_<k>.json
    for k < members, then training/search.json when search is true."""
    if not (isinstance(training, dict) and _is_int(training.get("members"))
            and training["members"] >= 1 and isinstance(training.get("search"), bool)):
        return []
    names = [TRAINING_MEMBER_FILE % k for k in range(training["members"])]
    return names + ([TRAINING_SEARCH_FILE] if training["search"] else [])


def validate_manifest(data):
    """Every problem of a manifest (keys, types, ranges, the cross-conditions
    of sec. 2.7), as a list of strings naming the key; [] when valid. Never
    raises on a JSON value."""
    probs = []
    try:
        _manifest(probs, data)
    except RecursionError:
        probs.append("the manifest is too deeply nested to check")
    return probs


# ---------------------------------------------------------------- card

def _todo(v):
    return isinstance(v, str) and v.strip() == TODO


def _empty(v):
    return (isinstance(v, (str, list, dict)) and len(v) == 0) or (isinstance(v, str)
                                                                  and not v.strip())


def _leaf_keys(keys):
    return dict((k, None) for k in keys)


# the card's named fields: key -> None (a field whose value is free below it),
# a dict of the named fields below it, or a one-item list (each entry's fields)
_CARD_FIELDS = dict(_leaf_keys(CARD_KEYS), **{
    "simulator": _leaf_keys(CARD_SIMULATOR_KEYS),
    "training": _leaf_keys(CARD_TRAINING_KEYS),
    "authors": [_leaf_keys(CARD_AUTHOR_KEYS)],
    "data": {"simulated_bank": _leaf_keys(CARD_BANK_KEYS),
             "recordings": _leaf_keys(CARD_RECORDINGS_KEYS)},
    "software": _leaf_keys(SOFTWARE_KEYS)})


def _todo_or_empty(data):
    """[(dotted name, "TODO" | "empty")], sorted: every value of the header,
    at any depth, that holds TODO, and every named field of the card (sec.
    2.7's keys, the nested ones and each author's included) that is empty
    ("", whitespace, [] or {}). A value below a named field (a
    hyperparameter, a fixed parameter, a reference) is not a field: only a
    TODO in it counts. A null is not empty (the bank's size and digest may be
    null). Iterative, so that no depth makes it raise, and each container is
    visited once, so that a YAML alias that refers to itself ends."""
    out, seen = [], set()
    stack = [("", data, _CARD_FIELDS)]      # (path, value, its fields or None)
    while stack:
        path, v, fields = stack.pop()
        if _todo(v):
            out.append((path, "TODO"))
            continue
        if isinstance(v, (dict, list)):
            if id(v) in seen:
                continue
            seen.add(id(v))
        if isinstance(fields, dict) and isinstance(v, dict):
            for k in v:
                sub = fields.get(k)
                p = _join(path, k)
                if k in fields and _empty(v[k]):
                    out.append((p, "empty"))
                else:
                    stack.append((p, v[k], sub))
        elif isinstance(fields, list) and isinstance(v, list):
            for i, x in enumerate(v):
                stack.append(("%s[%d]" % (path, i), x, fields[0]))
        elif isinstance(v, dict):
            stack.extend((_join(path, k), v[k], None) for k in v)
        elif isinstance(v, list):
            stack.extend(("%s[%d]" % (path, i), x, None) for i, x in enumerate(v))
    return sorted(out)


def _card_sub(probs, path, v, keys):
    """A nested object of the card; a TODO stands for the whole object."""
    if _todo(v):
        return None
    return v if _keys(probs, path, v, keys) else None


def _card_str(probs, name, v):
    if not (_todo(v) or isinstance(v, str)):
        probs.append("%s must be a string, got %s" % (name, _r(v)))


def _card_bool(probs, name, v):
    if not (_todo(v) or isinstance(v, bool)):
        probs.append("%s must be true or false, got %s" % (name, _r(v)))


def _card_list_of_str(probs, name, v):
    if not (_todo(v) or (isinstance(v, list) and all(isinstance(x, str) for x in v))):
        probs.append("%s must be a list of strings, got %s" % (name, _r(v)))


def _card_obj(probs, name, v):
    if not (_todo(v) or isinstance(v, dict)):
        probs.append("%s must be an object, got %s" % (name, _r(v)))


def _card_types(probs, d):
    for k in ("title", "scientific_question", "licence", "cite", "limitations"):
        if k in d:
            _card_str(probs, k, d[k])
    if "references" in d:
        _card_list_of_str(probs, "references", d["references"])
    s = _card_sub(probs, "simulator", d.get("simulator", TODO), CARD_SIMULATOR_KEYS)
    if s is not None:
        for k in ("model_type", "reference"):
            if k in s:
                _card_str(probs, "simulator." + k, s[k])
        if "fixed_parameters" in s:
            _card_obj(probs, "simulator.fixed_parameters", s["fixed_parameters"])
        if "searched_parameters" in s:
            _card_list_of_str(probs, "simulator.searched_parameters", s["searched_parameters"])
    t = _card_sub(probs, "training", d.get("training", TODO), CARD_TRAINING_KEYS)
    if t is not None:
        if "arm" in t:
            _card_str(probs, "training.arm", t["arm"])
        for k in ("members", "n_networks", "n_windows"):
            if k in t and not _todo(t[k]):
                _int(probs, "training." + k, t[k], 1)
        if "split_fracs" in t and not _todo(t["split_fracs"]):
            f = t["split_fracs"]
            if not (isinstance(f, list) and len(f) == 3 and all(_is_num(x) and x >= 0 for x in f)):
                probs.append("training.split_fracs must be three non-negative finite numbers "
                             "(f_tr, f_sel, f_rep), got %s" % _r(f))
        if "hyperparameters" in t:
            _card_obj(probs, "training.hyperparameters", t["hyperparameters"])
        if "hyperparameter_search" in t:
            _card_str(probs, "training.hyperparameter_search", t["hyperparameter_search"])
    if "extraction" in d:
        e = d["extraction"]
        if not (_todo(e) or e == NOT_APPLICABLE):
            if isinstance(e, dict):
                _extraction(probs, "extraction", e)
            else:
                probs.append("extraction must be the manifest's extraction (an object) or %r, "
                             "got %s" % (NOT_APPLICABLE, _r(e)))
    if "authors" in d and not _todo(d["authors"]):
        a = d["authors"]
        if not isinstance(a, list):
            probs.append("authors must be a list of objects (name, contact), got %s" % _r(a))
        else:
            for i, au in enumerate(a):
                au = _card_sub(probs, "authors[%d]" % i, au, CARD_AUTHOR_KEYS)
                if au is not None:
                    for k in CARD_AUTHOR_KEYS:
                        if k in au:
                            _card_str(probs, "authors[%d].%s" % (i, k), au[k])
    dt = _card_sub(probs, "data", d.get("data", TODO), CARD_DATA_KEYS)
    if dt is not None:
        b = _card_sub(probs, "data.simulated_bank", dt.get("simulated_bank", TODO),
                      CARD_BANK_KEYS)
        if b is not None:
            if "location" in b:
                _card_str(probs, "data.simulated_bank.location", b["location"])
            if "public" in b:
                _card_bool(probs, "data.simulated_bank.public", b["public"])
            if "size_bytes" in b and not _todo(b["size_bytes"]):
                _int(probs, "data.simulated_bank.size_bytes", b["size_bytes"], 0, allow_none=True)
            if "sha256" in b and not _todo(b["sha256"]):
                _hex(probs, "data.simulated_bank.sha256", b["sha256"], allow_none=True)
        r = _card_sub(probs, "data.recordings", dt.get("recordings", TODO), CARD_RECORDINGS_KEYS)
        if r is not None:
            if "location" in r:
                _card_str(probs, "data.recordings.location", r["location"])
            if "public" in r:
                _card_bool(probs, "data.recordings.public", r["public"])
    sw = _card_sub(probs, "software", d.get("software", TODO), SOFTWARE_KEYS)
    if sw is not None:
        for k in SOFTWARE_KEYS:
            if k in sw:
                _card_str(probs, "software." + k, sw[k])


_ABSENT = object()


def _get(d, *keys):
    """d[k0][k1]... or _ABSENT, never raising."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return _ABSENT
        d = d[k]
    return d


def _card_agreement(probs, d, manifest):
    """sec. 2.7: training.arm, training.members, simulator.searched_parameters,
    extraction and software agree with the manifest (a TODO is not compared)."""
    if not isinstance(manifest, dict):
        probs.append("the manifest to compare the card with must be an object, got %s"
                     % _r(manifest))
        return
    pairs = ((("training", "arm"), ("arm",)),
             (("training", "members"), ("M",)),
             (("simulator", "searched_parameters"), ("parameters", "names")),
             (("software",), ("software",)))
    for ck, mk in pairs:
        cv, mv = _get(d, *ck), _get(manifest, *mk)
        if cv is _ABSENT or mv is _ABSENT or _todo(cv):
            continue
        _differ(probs, ".".join(ck), cv, ".".join(mk), mv)
    cv, mv = _get(d, "extraction"), _get(manifest, "extraction")
    if cv is not _ABSENT and mv is not _ABSENT and not _todo(cv):
        if mv is None:
            if cv != NOT_APPLICABLE:
                probs.append("extraction must be %r: the manifest's extraction is null, got %s"
                             % (NOT_APPLICABLE, _r(cv)))
        else:
            _differ(probs, "extraction", cv, "extraction", mv)


def _differ(probs, cname, cv, mname, mv):
    """One problem per key at which two objects differ, or one for two other
    values that differ."""
    if isinstance(cv, dict) and isinstance(mv, dict):
        for k in sorted(set(cv) | set(mv), key=_r):
            a, b = cv.get(k, _ABSENT), mv.get(k, _ABSENT)
            if a is _ABSENT or b is _ABSENT or a != b:
                probs.append("%s is %s but the manifest's %s is %s"
                             % (_join(cname, k), "absent" if a is _ABSENT else _r(a),
                                _join(mname, k), "absent" if b is _ABSENT else _r(b)))
    elif cv != mv:
        probs.append("%s is %s but the manifest's %s is %s" % (cname, _r(cv), mname, _r(mv)))


def validate_card_header(data, manifest=None, complete=False):
    """Every problem of a card's header (a dict: exactly CARD_KEYS and the
    nested key sets). With `complete` true an empty value or a TODO anywhere
    is a problem too; with `manifest` given, the agreement sec. 2.7 states
    (arm, members, searched parameters, extraction, software) is checked.
    Returns a list of strings; [] when valid. Never raises on a YAML value."""
    probs = []
    try:
        if not _keys(probs, "", data, CARD_KEYS):
            return probs
        _card_types(probs, data)
        if complete:
            for path, what in _todo_or_empty(data):
                probs.append("%s is %s in a complete card" % (path, what))
        if manifest is not None:
            _card_agreement(probs, data, manifest)
    except RecursionError:
        probs.append("the card's header is too deeply nested to check")
    return probs


def card_todo_fields(data):
    """The dotted names of the header's fields that are empty or hold TODO
    (training.hyperparameter_search, authors[0].contact, ...), sorted."""
    if not isinstance(data, dict):
        return []
    return sorted(set(path for path, _ in _todo_or_empty(data)))
