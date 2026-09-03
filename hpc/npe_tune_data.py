#!/usr/bin/env python3
"""
npe_tune_data.py -- bank loading, topology grouping, and the frozen split.

Scope boundary: this module loads the (theta, z) bank and decides which row
goes into which split. It does NOT train, score, plot, or search. Nothing
here imports torch or sbi, so it can be exercised on a machine with neither.

Design decisions locked here, with the reason for each:

  * The bank is loaded through gate_data.load_sim, NOT by re-reading parquet.
    That function already implements the activity (MFR) filter and the
    theta-deduplication in the order the rest of the project depends on
    (filter BEFORE dedup, so the activity table stays aligned with raw rows).
    Reimplementing it here would create a second definition of "the bank".

  * Groups are derived from THETA, not from the parquet's topo_idx column.
    load_sim does not return the row masks it applied, so an externally read
    topo_idx column cannot be realigned to its output. Rows from one topology
    draw share the connectivity-kernel axes exactly, so grouping on those
    columns reproduces the topology partition from data that survives every
    filter. If two distinct draws happen to share kernel values they merge
    into one group, which is conservative: fewer, larger groups can only
    reduce leakage, never increase it.

  * The split is frozen to a manifest and hashed. Every trial records the
    hash; two trials with different hashes are never comparable, and the
    ledger enforces that mechanically rather than by memory.

  * Shapes (p, E, n, number of groups, prior bounds) are read from the
    contract and the loaded arrays at run time. No shape is hardcoded.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Bank",
    "SplitManifest",
    "DEFAULT_TOPOLOGY_AXES",
    "load_bank",
    "derive_groups",
    "make_split",
    "save_split",
    "load_split",
    "check_split",
    "subsample_groups",
    "shape_report",
]

# Connectivity-kernel axes: rows sharing a topology draw share these exactly.
DEFAULT_TOPOLOGY_AXES = ("p0_conn", "d0_conn", "beta_conn")


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

@dataclass
class Bank:
    """One loaded, filtered, deduplicated simulated bank.

    Attributes
    ----------
    z : ndarray, shape (n, E), float32
        L2-normalised embeddings; the conditioner.
    theta : ndarray, shape (n, p), float64
        Labels in INFERENCE coordinates.
    groups : ndarray, shape (n,), int64
        Topology-group label per row; see derive_groups.
    contract : npe_contract.Contract
        The data contract. Source of p, E, param_names, coord, bounds.
    meta : dict
        Provenance from load_sim plus the grouping decision.
    """

    z: np.ndarray
    theta: np.ndarray
    groups: np.ndarray
    contract: object
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.z.shape[0])

    @property
    def p(self) -> int:
        return int(self.theta.shape[1])

    @property
    def embedding_dim(self) -> int:
        return int(self.z.shape[1])

    @property
    def n_groups(self) -> int:
        return int(np.unique(self.groups).shape[0])


def derive_groups(theta: np.ndarray,
                  param_names: Sequence[str],
                  axes: Sequence[str] = DEFAULT_TOPOLOGY_AXES,
                  strict: bool = True) -> Tuple[np.ndarray, Dict[str, object]]:
    """Group rows by topology draw, using the connectivity-kernel axes.

    Returns (groups, info). groups[i] is an integer label; rows with equal
    labels share a topology draw and must therefore never be split across
    train and evaluation sides.

    strict=True (default) raises when none of `axes` is present in
    param_names, because falling back to an i.i.d. split silently would
    inflate every held-out score. Pass strict=False to accept the fallback,
    which labels every row uniquely (i.e. an i.i.d. split) and says so in
    the returned info dict.
    """
    theta = np.asarray(theta, dtype=np.float64)
    names = list(param_names)
    idx = [names.index(a) for a in axes if a in names]
    if not idx:
        msg = ("no topology axes %r found in param_names; a grouped split "
               "cannot be derived from theta" % (list(axes),))
        if strict:
            raise ValueError(
                msg + ". Pass strict=False (or --allow-iid-split on the CLI) "
                "only if you accept that rows sharing a topology may then "
                "appear on both sides of the split.")
        groups = np.arange(theta.shape[0], dtype=np.int64)
        return groups, {"grouping": "iid_fallback", "axes_used": [],
                        "n_groups": int(groups.shape[0]), "warning": msg}

    sub = theta[:, idx]
    # np.unique with return_inverse gives a dense integer label per row.
    _, groups = np.unique(sub, axis=0, return_inverse=True)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    counts = np.bincount(groups)
    info = {
        "grouping": "theta_topology_axes",
        "axes_used": [names[i] for i in idx],
        "n_groups": int(counts.shape[0]),
        "group_size_min": int(counts.min()),
        "group_size_median": float(np.median(counts)),
        "group_size_max": int(counts.max()),
    }
    return groups, info


def load_bank(sim_glob: str,
              activity_path: Optional[str] = None,
              min_rate: Optional[float] = None,
              max_rate: Optional[float] = None,
              dedup_theta: bool = True,
              max_rows: Optional[int] = None,
              topology_axes: Sequence[str] = DEFAULT_TOPOLOGY_AXES,
              allow_iid_split: bool = False,
              seed: int = 0) -> Bank:
    """Load the simulated bank via gate_data.load_sim and attach groups.

    Every filter argument is passed straight through, so the bank this
    tuner trains on is the same object the gate would build from the same
    arguments. The activity filter and the deduplication are part of what
    every downstream number is conditional on and are recorded in meta.
    """
    import gate_data  # local import: keeps this module importable standalone

    arm = gate_data.load_sim(
        sim_glob,
        dedup_theta=dedup_theta,
        want_zraw=False,
        max_rows=max_rows,
        seed=seed,
        activity_path=activity_path,
        min_rate=min_rate,
        max_rate=max_rate,
    )
    if arm.theta is None:
        raise ValueError("load_sim returned no theta; the tuner needs labels")

    contract = arm.contract
    groups, ginfo = derive_groups(arm.theta, contract.param_names,
                                  axes=topology_axes,
                                  strict=not allow_iid_split)
    meta = dict(arm.meta)
    meta["grouping"] = ginfo
    meta["sim_glob"] = sim_glob
    return Bank(z=np.asarray(arm.z), theta=np.asarray(arm.theta),
                groups=groups, contract=contract, meta=meta)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass
class SplitManifest:
    """A frozen three-way, group-disjoint partition of one bank.

    train / sel / rep hold ROW INDICES into the bank as loaded under the
    recorded loader arguments. The manifest is only meaningful together with
    those arguments, which is why they are stored alongside and folded into
    the hash.
    """

    train: List[int]
    sel: List[int]
    rep: List[int]
    n_rows: int
    n_groups: int
    p: int
    embedding_dim: int
    param_names: List[str]
    seed: int
    fractions: List[float]
    loader: Dict[str, object] = field(default_factory=dict)
    grouping: Dict[str, object] = field(default_factory=dict)
    contract_digest: str = ""
    version: int = 1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def hash(self) -> str:
        """Stable content hash. Two manifests with the same hash are the
        same experiment; anything else must not be compared."""
        payload = json.dumps(self.to_dict(), sort_keys=True,
                             separators=(",", ":")).encode("ascii")
        return hashlib.sha256(payload).hexdigest()[:16]

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "sel": len(self.sel),
                "rep": len(self.rep)}


def _contract_digest(contract) -> str:
    """Short digest of the contract: axes, coordinates, bounds, E, encoder."""
    meta = getattr(contract, "meta", {}) or {}
    emb = meta.get("embedding", {}) if isinstance(meta, dict) else {}
    payload = {
        "param_names": list(contract.param_names),
        "coord": list(contract.coord),
        "bounds": np.asarray(contract.bounds_theta, dtype=np.float64).tolist(),
        "embedding_dim": int(contract.embedding_dim),
        "dsn_checkpoint_sha256": (emb.get("dsn_checkpoint_sha256")
                                  if isinstance(emb, dict) else None),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:16]


def make_split(bank: Bank,
               fractions: Sequence[float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               min_groups_per_side: int = 5,
               min_rows_per_side: int = 200) -> SplitManifest:
    """Build the three-way group-disjoint split.

    Groups are shuffled once and dealt to the three sides in order until each
    reaches its row quota, so no group is ever divided. The guards are hard
    errors rather than warnings: a side too small to give a meaningful
    standard error makes every later comparison meaningless, and failing at
    freeze time costs seconds where failing later costs a campaign.
    """
    fr = np.asarray(fractions, dtype=np.float64)
    if fr.shape != (3,):
        raise ValueError("fractions must have exactly 3 entries "
                         "(train, sel, rep)")
    if np.any(fr <= 0) or abs(fr.sum() - 1.0) > 1e-9:
        raise ValueError("fractions must be positive and sum to 1, got %r"
                         % (fr.tolist(),))

    rng = np.random.default_rng(seed)
    uniq = np.unique(bank.groups)
    order = rng.permutation(uniq)

    counts = {int(g): int(np.sum(bank.groups == g)) for g in uniq}
    quota = fr * bank.n
    sides: List[List[int]] = [[], [], []]      # group ids per side
    filled = np.zeros(3, dtype=np.float64)     # rows so far per side

    for g in order:
        # Give the group to whichever side is furthest below its quota, in
        # relative terms. This keeps the deal balanced when group sizes vary.
        deficit = (quota - filled) / np.maximum(quota, 1.0)
        side = int(np.argmax(deficit))
        sides[side].append(int(g))
        filled[side] += counts[int(g)]

    idx_by_side = []
    for side_groups in sides:
        mask = np.isin(bank.groups, np.asarray(side_groups, dtype=np.int64))
        idx_by_side.append(np.flatnonzero(mask).astype(np.int64))

    names = ("train", "sel", "rep")
    for name, gl, ix in zip(names, sides, idx_by_side):
        if len(gl) < min_groups_per_side:
            raise ValueError(
                "split side %r got %d groups, below the minimum of %d. The "
                "bank has only %d groups; lower --min-groups-per-side "
                "deliberately, or use a bank with more topology draws."
                % (name, len(gl), min_groups_per_side, bank.n_groups))
        if ix.shape[0] < min_rows_per_side:
            raise ValueError(
                "split side %r got %d rows, below the minimum of %d."
                % (name, int(ix.shape[0]), min_rows_per_side))

    # Disjointness is asserted, not assumed: this is the property the whole
    # evaluation rests on, and it costs microseconds to prove.
    a, b, c = (set(x.tolist()) for x in idx_by_side)
    if a & b or a & c or b & c:
        raise RuntimeError("internal error: split sides overlap")
    if len(a) + len(b) + len(c) != bank.n:
        raise RuntimeError("internal error: split does not cover the bank")

    return SplitManifest(
        train=idx_by_side[0].tolist(),
        sel=idx_by_side[1].tolist(),
        rep=idx_by_side[2].tolist(),
        n_rows=bank.n,
        n_groups=bank.n_groups,
        p=bank.p,
        embedding_dim=bank.embedding_dim,
        param_names=list(bank.contract.param_names),
        seed=int(seed),
        fractions=fr.tolist(),
        loader={k: v for k, v in bank.meta.items() if k != "grouping"},
        grouping=bank.meta.get("grouping", {}),
        contract_digest=_contract_digest(bank.contract),
    )


def save_split(manifest: SplitManifest, path: str) -> str:
    """Write the manifest as JSON, with its own hash recorded inside."""
    d = manifest.to_dict()
    d["_hash"] = manifest.hash()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        json.dump(d, fh, indent=2, sort_keys=True)
    return path


def load_split(path: str) -> SplitManifest:
    """Read a manifest and verify its recorded hash still matches."""
    with open(path, "r", encoding="ascii") as fh:
        d = json.load(fh)
    recorded = d.pop("_hash", None)
    man = SplitManifest(**d)
    if recorded is not None and recorded != man.hash():
        raise ValueError(
            "split manifest %r has been edited: recorded hash %s but content "
            "hashes to %s. Rebuild it rather than repairing it by hand."
            % (path, recorded, man.hash()))
    return man


def check_split(bank: Bank, manifest: SplitManifest) -> None:
    """Refuse a bank/manifest pair that do not describe the same experiment.

    Raises with a specific reason. This is the mechanical form of the rule
    that two runs under different shapes or contracts are never comparable.
    """
    problems = []
    if bank.n != manifest.n_rows:
        problems.append("bank has %d rows, manifest was built on %d"
                        % (bank.n, manifest.n_rows))
    if bank.p != manifest.p:
        problems.append("p differs: bank %d, manifest %d" % (bank.p, manifest.p))
    if bank.embedding_dim != manifest.embedding_dim:
        problems.append("E differs: bank %d, manifest %d"
                        % (bank.embedding_dim, manifest.embedding_dim))
    if list(bank.contract.param_names) != list(manifest.param_names):
        problems.append("param_names differ")
    dig = _contract_digest(bank.contract)
    if manifest.contract_digest and dig != manifest.contract_digest:
        problems.append("contract digest differs: bank %s, manifest %s"
                        % (dig, manifest.contract_digest))
    if problems:
        raise ValueError("split manifest does not match this bank: "
                         + "; ".join(problems))

    # Group disjointness across sides, re-verified against the actual bank.
    gt = set(np.unique(bank.groups[np.asarray(manifest.train, dtype=np.int64)]).tolist())
    gs = set(np.unique(bank.groups[np.asarray(manifest.sel, dtype=np.int64)]).tolist())
    gr = set(np.unique(bank.groups[np.asarray(manifest.rep, dtype=np.int64)]).tolist())
    if gt & gs or gt & gr or gs & gr:
        raise ValueError("split manifest leaks: a topology group appears on "
                         "more than one side")


def subsample_groups(bank: Bank,
                     indices: Sequence[int],
                     fraction: float,
                     seed: int = 0) -> np.ndarray:
    """Take a group-respecting subsample of a set of row indices.

    Used by the learning curve: shrinking the training set by dropping whole
    topology draws, so the subsample keeps the same independence structure as
    the full split. Returns row indices.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1], got %r" % (fraction,))
    idx = np.asarray(indices, dtype=np.int64)
    if fraction == 1.0:
        return idx
    g = bank.groups[idx]
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq)
    target = fraction * idx.shape[0]
    keep_groups, filled = [], 0
    for gg in order:
        if filled >= target and keep_groups:
            break
        keep_groups.append(int(gg))
        filled += int(np.sum(g == gg))
    mask = np.isin(g, np.asarray(keep_groups, dtype=np.int64))
    return idx[mask]


def shape_report(bank: Bank,
                 manifest: Optional[SplitManifest] = None) -> str:
    """One screen of shapes. Printed at the start of every run, and the
    block a cluster run's output is checked against."""
    c = bank.contract
    lines = [
        "---- shape report ----------------------------------------------",
        "  p (parameters)      : %d" % bank.p,
        "  E (embedding dim)   : %d" % bank.embedding_dim,
        "  n (rows)            : %d" % bank.n,
        "  topology groups     : %d" % bank.n_groups,
        "  grouping            : %s" % bank.meta.get("grouping", {}).get("grouping", "?"),
        "  grouping axes       : %s" % (bank.meta.get("grouping", {}).get("axes_used", []),),
        "  contract digest     : %s" % _contract_digest(c),
    ]
    filt = bank.meta.get("activity_filter")
    if isinstance(filt, dict):
        lines.append("  activity filter     : min_rate=%s kept %d/%d (%.1f%%)"
                     % (filt.get("min_rate_hz_per_electrode"),
                        filt.get("n_after", -1), filt.get("n_before", -1),
                        100.0 * float(filt.get("fraction_kept", float("nan")))))
    if "n_rows_dedup" in bank.meta:
        lines.append("  theta dedup         : %d rows kept (dup factor %.3f)"
                     % (bank.meta["n_rows_dedup"],
                        bank.meta.get("duplication_factor", float("nan"))))
    if manifest is not None:
        s = manifest.sizes()
        lines.append("  split (rows)        : train %d / sel %d / rep %d"
                     % (s["train"], s["sel"], s["rep"]))
        lines.append("  split hash          : %s" % manifest.hash())
    lines.append("----------------------------------------------------------------")
    return "\n".join(lines)
