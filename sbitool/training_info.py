"""Block 48: the model's training, collected for a bundle (specs/SPEC.md
block 48; D-184 (2)).

What the expert layer shows of the training: each member's training curve
(training/member_<k>.json, from the member's record `history`) and a
summary of the hyper-parameter search's trials (training/search.json, from
the search's `trials` folder and its `finalists.json`). Reads JSON only.

    members, search = training_info.collect(P, view, e, records)

The validators member_problems and search_problems are the ones
bundle.check and bundle.write use on these files.

CONTRACT (wave A of S6): the name and the signature below are fixed by
specs/SPEC.md; the body is block 48's.

Pure ASCII, LF only.
"""

import json
import math
import os

MEMBER_KEYS = ("seed", "epoch", "npe", "val_npe", "dsn", "rep")
MEMBER_LISTS = ("epoch", "npe", "val_npe", "dsn", "rep")
SEARCH_KEYS = ("campaign", "n_trials", "objective", "trials", "chosen")
TRIAL_KEYS = ("trial_id", "status", "nll", "config")
MEMBER_FILE = "training/member_%d.json"
SEARCH_FILE = "training/search.json"
PREFIX = "training/"
OBJECTIVE = "nll"
TRIALS_DIR = "trials"
FINALISTS_FILE = "finalists.json"
TRIAL_SUFFIX = ".json"       # a trial's file in the ledger (a ".tmp" is one being written)
STATUS_ABSENT = "absent"      # a trial file without a status
STATUS_UNREADABLE = "unreadable"   # a trial file that is not a JSON object


class TrainingInfoError(ValueError):
    """A member's record holds no training curve that can be packed."""


# ---------------------------------------------------------------- values

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def finite_or_none(v):
    """A curve's entry as written: a finite number as it is, anything else
    (NaN, an infinity, null, an absent key) None."""
    return v if _is_num(v) else None


def _scalar(v):
    """A JSON scalar: null, a bool, a string or a finite number."""
    return v is None or isinstance(v, (bool, str)) or _is_num(v)


def restrict_config(cfg):
    """A trial's config restricted to its JSON scalars whose key does not
    start with "_" (SPEC block 48)."""
    if not isinstance(cfg, dict):
        return {}
    return dict((k, v) for k, v in cfg.items()
                if isinstance(k, str) and not k.startswith("_") and _scalar(v))


def json_bytes(obj):
    """The bytes a training file is written with (as the manifest: indent 1,
    sorted keys, ASCII, no NaN, a final newline)."""
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n").encode("ascii")


# ---------------------------------------------------------------- members

def member_of(record, k=0):
    """The member dict (MEMBER_KEYS) of one record: its seed and the five
    lists of its `history`, one entry per epoch; TrainingInfoError when the
    record has no seed or no history of at least one epoch."""
    where = "member %d" % k
    if not isinstance(record, dict):
        raise TrainingInfoError("%s: the record is not an object" % where)
    seed = record.get("seed")
    if not _is_int(seed):
        raise TrainingInfoError("%s: the record has no integer seed (%r)" % (where, seed))
    hist = record.get("history")
    if not (isinstance(hist, list) and hist and all(isinstance(h, dict) for h in hist)):
        raise TrainingInfoError("%s (seed %d): the record's history is not a non-empty list of "
                                "epochs (export with --no-training to leave the training out)"
                                % (where, seed))
    out = {"seed": int(seed)}
    for key in MEMBER_LISTS:
        out[key] = [finite_or_none(h.get(key)) for h in hist]
    return out


def member_problems(d, k, seed=None):
    """Every problem of training/member_<k>.json's content (exactly
    MEMBER_KEYS; the five lists of one length >= 1, each entry a finite
    number or null; `seed` an integer, the manifest's member's when given)."""
    name = MEMBER_FILE % k
    if not isinstance(d, dict):
        return ["%s must be an object with exactly the keys %s" % (name, ", ".join(MEMBER_KEYS))]
    probs = []
    for key in MEMBER_KEYS:
        if key not in d:
            probs.append("%s: missing key %s" % (name, key))
    for key in sorted((x for x in d if x not in MEMBER_KEYS), key=repr):
        probs.append("%s: unknown key %r" % (name, key))
    if "seed" in d:
        if not _is_int(d["seed"]):
            probs.append("%s: seed must be an integer, got %r" % (name, d["seed"]))
        elif seed is not None and d["seed"] != seed:
            probs.append("%s: seed is %d but the manifest's members[%d].seed is %r"
                         % (name, d["seed"], k, seed))
    lengths = {}
    for key in MEMBER_LISTS:
        if key not in d:
            continue
        v = d[key]
        if not isinstance(v, list) or not v:
            probs.append("%s: %s must be a non-empty list, got %s"
                         % (name, key, "an empty list" if v == [] else type(v).__name__))
            continue
        if not all(x is None or _is_num(x) for x in v):
            probs.append("%s: %s holds an entry that is neither a finite number nor null"
                         % (name, key))
        lengths[key] = len(v)
    if len(set(lengths.values())) > 1:
        probs.append("%s: the lists are not of one length (%s)"
                     % (name, ", ".join("%s %d" % (key, lengths[key]) for key in MEMBER_LISTS
                                        if key in lengths)))
    return probs


# ---------------------------------------------------------------- the search

def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _search_folder(P, view, e):
    """The search's tune folder of ensemble e (<tune_root of the search's
    view>/<campaign>), or (None, why)."""
    try:
        views = P.ev_rec_plan.search_views(view.cfg)
    except Exception as exc:  # noqa: BLE001  (never a refusal: the summary is left out)
        return None, "the project's searches cannot be listed (%s)" % exc
    for name, v in views:
        if name == e.search:
            try:
                return os.path.join(v.data["paths"]["tune_root"], v.data["search"]["campaign"]), ""
            except (KeyError, TypeError) as exc:
                return None, "the search %s has no tune_root or campaign (%s)" % (e.search, exc)
    return None, "the configuration has no search %r" % (e.search,)


def _trial(folder, fname, log):
    """(entry, the file's whole config or None) of one trial file."""
    stem = fname[:-len(TRIAL_SUFFIX)]
    try:
        rec = _read_json(os.path.join(folder, fname))
    except (OSError, ValueError) as exc:
        rec, why = None, str(exc)
    else:
        why = "not a JSON object"
    if not isinstance(rec, dict):
        log("training: trial file %s cannot be read (%s): status %s"
            % (os.path.join(folder, fname), why, STATUS_UNREADABLE))
        return {"trial_id": stem, "status": STATUS_UNREADABLE, "nll": None, "config": {}}, None
    tid = rec.get("trial_id")
    status = rec.get("status")
    cfg = rec.get("config")
    entry = {"trial_id": tid if isinstance(tid, str) and tid else stem,
             "status": status if isinstance(status, str) else STATUS_ABSENT,
             "nll": float(rec["nll"]) if _is_num(rec.get("nll")) else None,
             "config": restrict_config(cfg)}
    return entry, (cfg if isinstance(cfg, dict) else None)


def _chosen(folder, trials, configs, log):
    """The trial_id of the one trial whose config equals the first
    finalist's of finalists.json; None when none or several do, or when
    there is no finalist."""
    path = os.path.join(folder, FINALISTS_FILE)
    try:
        fin = _read_json(path)
        first = fin["finalists"][0]["config"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        log("training: no first finalist in %s (%s): chosen null" % (path, exc))
        return None
    ids = [t["trial_id"] for t, c in zip(trials, configs) if c is not None and c == first]
    if len(ids) != 1:
        log("training: %d trial(s) of %s have the first finalist's config: chosen null"
            % (len(ids), folder))
        return None
    return ids[0]


def search_of(P, view, e, log=print):
    """training/search.json's content for ensemble e, or None (one log line)
    when the search's folder or its trials are absent."""
    folder, why = _search_folder(P, view, e)
    if folder is None:
        log("training: no search summary for %s: %s" % (e.name, why))
        return None
    tdir = os.path.join(folder, TRIALS_DIR)
    if not os.path.isdir(tdir):
        log("training: no search summary for %s: %s absent" % (e.name, tdir))
        return None
    files = sorted(f for f in os.listdir(tdir)
                   if f.endswith(TRIAL_SUFFIX) and os.path.isfile(os.path.join(tdir, f)))
    if not files:
        log("training: no search summary for %s: no trial in %s" % (e.name, tdir))
        return None
    pairs = sorted((_trial(tdir, f, log) for f in files), key=lambda p: p[0]["trial_id"])
    trials = [p[0] for p in pairs]
    configs = [p[1] for p in pairs]
    return {"campaign": str(e.campaign), "n_trials": len(trials), "objective": OBJECTIVE,
            "trials": trials, "chosen": _chosen(folder, trials, configs, log)}


def search_problems(d):
    """Every problem of training/search.json's content (SEARCH_KEYS, each
    trial TRIAL_KEYS, sorted by trial_id, n_trials their number, chosen null
    or one of the ids)."""
    name = SEARCH_FILE
    if not isinstance(d, dict):
        return ["%s must be an object with exactly the keys %s" % (name, ", ".join(SEARCH_KEYS))]
    probs = []
    for key in SEARCH_KEYS:
        if key not in d:
            probs.append("%s: missing key %s" % (name, key))
    for key in sorted((x for x in d if x not in SEARCH_KEYS), key=repr):
        probs.append("%s: unknown key %r" % (name, key))
    if "campaign" in d and not (isinstance(d["campaign"], str) and d["campaign"]):
        probs.append("%s: campaign must be a non-empty string" % name)
    if "objective" in d and d["objective"] != OBJECTIVE:
        probs.append("%s: objective must be %r, got %r" % (name, OBJECTIVE, d["objective"]))
    trials = d.get("trials")
    ids = []
    if "trials" in d:
        if not isinstance(trials, list):
            probs.append("%s: trials must be a list" % name)
            trials = None
        else:
            for i, t in enumerate(trials):
                tp = "%s: trials[%d]" % (name, i)
                if not (isinstance(t, dict) and set(t) == set(TRIAL_KEYS)):
                    probs.append("%s must be an object with exactly the keys %s"
                                 % (tp, ", ".join(TRIAL_KEYS)))
                    continue
                if not (isinstance(t["trial_id"], str) and t["trial_id"]):
                    probs.append("%s.trial_id must be a non-empty string" % tp)
                else:
                    ids.append(t["trial_id"])
                if not isinstance(t["status"], str):
                    probs.append("%s.status must be a string" % tp)
                if not (t["nll"] is None or _is_num(t["nll"])):
                    probs.append("%s.nll must be a finite number or null" % tp)
                if not (isinstance(t["config"], dict) and t["config"] == restrict_config(t["config"])):
                    probs.append("%s.config must hold JSON scalars whose key does not start "
                                 "with \"_\"" % tp)
            if ids != sorted(ids):
                probs.append("%s: the trials are not sorted by trial_id" % name)
    if "n_trials" in d:
        n = d["n_trials"]
        if not _is_int(n):
            probs.append("%s: n_trials must be an integer" % name)
        elif trials is not None and n != len(trials):
            probs.append("%s: n_trials is %d but trials has %d entries" % (name, n, len(trials)))
    if "chosen" in d and d["chosen"] is not None and d["chosen"] not in ids:
        probs.append("%s: chosen %r is not the trial_id of one of the trials"
                     % (name, d["chosen"]))
    return probs


# ---------------------------------------------------------------- the API

def collect(P, view, e, records, log=print):
    """(members, search) for ensemble `e` of the project's view.

    P        the pipeline's modules (sbitool.export._pipeline())
    records  the members' record dicts, in member order
    members  a list of M dicts with exactly MEMBER_KEYS: `seed` the member's
             and the five lists of MEMBER_LISTS, of one length >= 1, from the
             record's `history` (a NaN becomes None)
    search   a dict with exactly SEARCH_KEYS (each trial exactly TRIAL_KEYS,
             sorted by trial_id; `config` restricted to JSON scalars whose
             key does not start with "_"), or None when the search's folder
             or its trials are absent (one log line; never a refusal)
    TrainingInfoError when a record has no history of at least one epoch."""
    members = [member_of(rec, k) for k, rec in enumerate(records)]
    return members, search_of(P, view, e, log=log)
