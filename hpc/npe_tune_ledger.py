#!/usr/bin/env python3
"""
npe_tune_ledger.py -- the campaign's single source of truth.

Scope boundary: persistence only. No training, no scoring, no search. Pure
stdlib plus numpy.

Design decisions locked here, with the reason for each:

  * ONE FILE PER TRIAL, assembled into a ledger by reading a directory --
    not a single append-only file. Several evaluations run as independent
    scheduler jobs on a shared filesystem; concurrent appends to one file
    are only atomic below the pipe buffer, and a ledger row is comfortably
    larger than that. A directory of small files has no such failure mode,
    and a half-written file is detectable (invalid JSON) rather than
    silently interleaved into a neighbour's row.

  * WRITE-THEN-RENAME. Each result is written to a temporary name in the
    same directory and then os.replace()d into place, which is atomic on
    POSIX. A reader therefore never sees a partial file, and a job killed
    mid-write leaves a stray .tmp rather than a corrupt result.

  * TRIAL ID = HASH OF WHAT WAS RUN, not a counter. The id folds in the
    configuration, the split hash, the contract digest, the probe ensemble
    size and the seeds. Re-proposing an already-evaluated configuration is
    then detectable, resuming is idempotent, and a result computed under a
    different split can never be mistaken for a comparable one.

  * NUMPY SCALARS ARE CAST BEFORE SERIALISING. The optimiser returns
    numpy integer types for integer dimensions, and json.dump refuses them
    outright. Found while testing the search loop; cheap to handle here
    once rather than at each call site.

  * THE LEDGER REFUSES TO POOL INCOMPARABLE ROWS. load() filters on the
    split hash and contract digest by default and reports what it dropped,
    so "two trials with different digests are never compared" is mechanical
    rather than remembered.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import hashlib
import json
import os
import glob as _glob
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "TrialRecord",
    "json_safe",
    "trial_id",
    "write_trial",
    "read_trial",
    "load_ledger",
    "ledger_table",
    "pending_ids",
    "write_pending",
    "clear_pending",
]


# ---------------------------------------------------------------------------
# Serialisation safety
# ---------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars and arrays to plain Python types.

    json.dump raises TypeError on numpy.int64, which is exactly what a
    scikit-optimize Integer dimension returns. Rather than casting at every
    call site (and forgetting one), everything written by this module passes
    through here first.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):        # np.int64, np.float32, np.bool_
        return obj.item()
    if isinstance(obj, float):
        # NaN and +-inf are not valid JSON; store as null rather than emit
        # a file that a strict parser will reject.
        return obj if np.isfinite(obj) else None
    return obj


# ---------------------------------------------------------------------------
# Trial identity and records
# ---------------------------------------------------------------------------

def trial_id(config: Dict[str, Any],
             split_hash: str,
             contract_digest: str,
             n_members: int,
             seeds: Sequence[int],
             tag: str = "") -> str:
    """Deterministic short id for one evaluation.

    Two evaluations share an id if and only if they would answer the same
    question: same configuration, same split, same contract, same probe
    size, same seeds. `tag` separates otherwise identical runs that are
    deliberately distinct (for example a learning-curve point at a reduced
    training fraction, or the shuffled-pairs control).
    """
    payload = {
        "config": json_safe(dict(config)),
        "split_hash": str(split_hash),
        "contract_digest": str(contract_digest),
        "n_members": int(n_members),
        "seeds": [int(s) for s in seeds],
        "tag": str(tag),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:12]


@dataclass
class TrialRecord:
    """One evaluated configuration, with everything needed to trust it.

    The provenance block is not optional decoration: without the split hash
    and contract digest a score cannot be compared to any other score, and
    without the convergence fields "converged" is an assumption rather than
    a record (this is the project's open point O3).
    """

    # -- identity ---------------------------------------------------------
    trial_id: str
    tag: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=list)
    n_members: int = 0

    # -- objective --------------------------------------------------------
    nll: float = float("nan")            # eq. (3), on the selection split
    floor: float = float("nan")          # eq. (5)
    delta: float = float("nan")          # eq. (6)
    delta_ci_lo: float = float("nan")
    delta_ci_hi: float = float("nan")
    member_nll: List[float] = field(default_factory=list)
    member_nll_mean: float = float("nan")
    jensen_ok: Optional[bool] = None
    n_eval_rows: int = 0

    # -- convergence provenance (open point O3) ---------------------------
    epochs: List[int] = field(default_factory=list)
    best_val_log_prob: List[float] = field(default_factory=list)
    hit_max_epochs: List[bool] = field(default_factory=list)
    train_seconds: List[float] = field(default_factory=list)

    # -- experiment provenance -------------------------------------------
    split_hash: str = ""
    contract_digest: str = ""
    p: int = 0
    embedding_dim: int = 0
    n_train_rows: int = 0
    git_sha: str = ""
    env_name: str = ""
    library_versions: Dict[str, str] = field(default_factory=dict)
    hostname: str = ""
    started_utc: str = ""
    finished_utc: str = ""

    # -- optional extras --------------------------------------------------
    model_dir: str = ""
    status: str = "ok"
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_trial(record: TrialRecord, results_dir: str) -> str:
    """Atomically write one trial result. Returns the final path."""
    _ensure_dir(results_dir)
    final = os.path.join(results_dir, "trial_%s.json" % record.trial_id)
    tmp = final + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="ascii") as fh:
        json.dump(record.to_dict(), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)                 # atomic on POSIX
    return final


def read_trial(path: str) -> Optional[TrialRecord]:
    """Read one trial file. Returns None (rather than raising) on an
    unreadable or partial file, so one bad file cannot stop a campaign
    from resuming; the caller is told how many were skipped."""
    try:
        with open(path, "r", encoding="ascii") as fh:
            d = json.load(fh)
    except Exception:
        return None
    known = set(TrialRecord.__dataclass_fields__.keys())
    extra = {k: v for k, v in d.items() if k not in known}
    d = {k: v for k, v in d.items() if k in known}
    for key in ("nll", "floor", "delta", "delta_ci_lo", "delta_ci_hi",
                "member_nll_mean"):
        if d.get(key) is None:
            d[key] = float("nan")
    rec = TrialRecord(**d)
    if extra:
        rec.extra.update(extra)
    return rec


def load_ledger(results_dir: str,
                split_hash: Optional[str] = None,
                contract_digest: Optional[str] = None,
                tag: Optional[str] = None,
                require_ok: bool = True,
                verbose: bool = False) -> List[TrialRecord]:
    """Assemble the ledger from a results directory, dropping what is not
    comparable.

    Filtering on split_hash and contract_digest is the mechanical form of
    "two trials with different digests are never compared". Rows dropped for
    that reason are counted and, with verbose=True, reported: a silent drop
    would look identical to a campaign that simply has fewer trials.
    """
    paths = sorted(_glob.glob(os.path.join(results_dir, "trial_*.json")))
    kept: List[TrialRecord] = []
    dropped = {"unreadable": 0, "split": 0, "contract": 0, "tag": 0,
               "status": 0}
    for p in paths:
        rec = read_trial(p)
        if rec is None:
            dropped["unreadable"] += 1
            continue
        if split_hash is not None and rec.split_hash != split_hash:
            dropped["split"] += 1
            continue
        if contract_digest is not None and rec.contract_digest != contract_digest:
            dropped["contract"] += 1
            continue
        if tag is not None and rec.tag != tag:
            dropped["tag"] += 1
            continue
        if require_ok and rec.status != "ok":
            dropped["status"] += 1
            continue
        kept.append(rec)
    if verbose:
        print("[ledger] %d file(s) read, %d kept; dropped: %s"
              % (len(paths), len(kept),
                 ", ".join("%s=%d" % (k, v) for k, v in dropped.items() if v)
                 or "none"), flush=True)
    return kept


def ledger_table(records: Sequence[TrialRecord],
                 columns: Optional[Sequence[str]] = None) -> str:
    """A plain-text table of the ledger, for the job log and the report."""
    if not records:
        return "(ledger is empty)"
    cfg_keys: List[str] = []
    for r in records:
        for k in r.config:
            if k not in cfg_keys:
                cfg_keys.append(k)
    cfg_keys = sorted(cfg_keys)
    head = ["trial", "tag"] + cfg_keys + ["M", "NLL", "gain", "epochs", "sec"]
    rows = [head]
    for r in sorted(records, key=lambda x: (np.inf if np.isnan(x.nll) else x.nll)):
        cfg = [str(r.config.get(k, "-")) for k in cfg_keys]
        ep = ("%d" % int(np.mean(r.epochs))) if r.epochs else "-"
        sec = ("%.0f" % float(np.sum(r.train_seconds))) if r.train_seconds else "-"
        rows.append([r.trial_id[:8], r.tag or "-"] + cfg
                    + ["%d" % r.n_members, "%.4f" % r.nll, "%.4f" % r.delta,
                       ep, sec])
    widths = [max(len(row[i]) for row in rows) for i in range(len(head))]
    out = []
    for j, row in enumerate(rows):
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        if j == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Pending proposals (the coordinator/worker handshake)
# ---------------------------------------------------------------------------

def write_pending(spec: Dict[str, Any], pending_dir: str) -> str:
    """Write one proposed trial specification for a worker job to pick up."""
    _ensure_dir(pending_dir)
    tid = spec["trial_id"]
    final = os.path.join(pending_dir, "trial_%s.json" % tid)
    tmp = final + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="ascii") as fh:
        json.dump(json_safe(spec), fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return final


def pending_ids(pending_dir: str) -> List[str]:
    """Trial ids currently proposed but not yet evaluated."""
    out = []
    for p in sorted(_glob.glob(os.path.join(pending_dir, "trial_*.json"))):
        base = os.path.basename(p)
        out.append(base[len("trial_"):-len(".json")])
    return out


def clear_pending(pending_dir: str, trial_id_: str) -> None:
    """Remove a pending spec once its result exists. Missing file is fine:
    a worker may have been re-run, and failing here would strand a
    successful evaluation."""
    p = os.path.join(pending_dir, "trial_%s.json" % trial_id_)
    if os.path.exists(p):
        os.remove(p)
