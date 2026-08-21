"""Assemble the two arms of the misspecification gate.

Loading only. No statistics, no plotting -- those live in gate_run.py and
gate_plots.py so that a change to the selection rule cannot silently alter
the analysis, and vice versa.

Two things here are deliberate and worth knowing before changing them.

1. DEDUPLICATION. The campaigns v1..v9 were launched with a hardcoded seed
   base plus the PBS array index, so re-running the launcher replays the same
   pseudo-random stream. Measured over the exported shards: 251,614 rows hold
   only 86,750 distinct theta (factor 2.90), and v4/v7/v8 are wholly contained
   in v3. Duplicated rows are not independent draws from the prior predictive.
   The gate subsamples the reference cloud to n_ref = min(n_sim // 2, 800), so
   duplication biases WHICH rows get drawn rather than inflating the count
   entering the statistic -- a smaller effect than it first appears, but a
   real one. See HANDOFF_seed_collisions.md.

2. COLUMN ORDER. z and theta come from npe_contract.load_shards, which takes
   the order from the sidecar rather than the file, and which enforces A10
   (shards disagreeing on names, coordinates, prior box, embedding dim or DSN
   checkpoint digest are refused rather than pooled). zraw is not returned by
   that loader, so it is read separately and its columns are sorted
   numerically by suffix. The row order of a parquet read is stable, so the
   two reads align; this module asserts that rather than assuming it.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq

_ZRAW_RE = re.compile(r"^zraw_(\d+)$")
_Z_RE = re.compile(r"^z_(\d+)$")


@dataclass
class Arm:
    """One side of the comparison, with everything the gate and plots need."""
    z: np.ndarray                      # (n, E) L2-normalised embedding
    zraw: Optional[np.ndarray]         # (n, E) pre-normalisation, or None
    theta: Optional[np.ndarray]        # (n, p) labels, None for the real arm
    groups: Optional[np.ndarray]       # (n,) group id, real arm only
    classes: Optional[np.ndarray]      # (n,) class id, real arm only
    contract: object = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.z.shape[0])

    @property
    def E(self) -> int:
        return int(self.z.shape[1])


def _ordered_cols(names: Sequence[str], pattern: re.Pattern) -> List[str]:
    """Columns matching pattern, ordered by their NUMERIC suffix.

    Sorting the strings would put z_10 before z_2. The suffix is the axis
    index, so a wrong order silently permutes the embedding axes -- which no
    downstream assertion would catch, because a permuted embedding is still a
    valid-looking array of the right shape.
    """
    hits = []
    for c in names:
        m = pattern.match(c)
        if m:
            hits.append((int(m.group(1)), c))
    return [c for _, c in sorted(hits)]


def _read_columns(path: str, cols: Sequence[str]) -> np.ndarray:
    t = pq.read_table(path, columns=list(cols))
    return np.column_stack([np.asarray(t[c].to_numpy(zero_copy_only=False),
                                       dtype=np.float64) for c in cols])


def _read_ident(path: str, col: str) -> np.ndarray:
    t = pq.read_table(path, columns=[col])
    return np.asarray(t[col].to_pylist())


def load_real(parquet_path: str,
              group_col: str = "culture",
              class_col: str = "condition",
              want_zraw: bool = True) -> Arm:
    """Load the real cohort. No theta: real recordings carry no labels."""
    import npe_contract as C

    z, theta, contract = C.load_shard(parquet_path, require_theta=False)
    if theta is not None:
        raise ValueError(
            "the real shard %r carries theta; that is a simulated shard."
            % (parquet_path,))

    names = pq.read_schema(parquet_path).names
    for c in (group_col, class_col):
        if c not in names:
            raise KeyError("column %r not in %s; available: %s"
                           % (c, parquet_path, names))

    groups = _read_ident(parquet_path, group_col)
    classes = _read_ident(parquet_path, class_col)
    if groups.shape[0] != z.shape[0]:
        raise RuntimeError(
            "ident column has %d rows but the embedding has %d; the two reads "
            "of %r disagree." % (groups.shape[0], z.shape[0], parquet_path))

    zraw = None
    if want_zraw:
        zc = _ordered_cols(names, _ZRAW_RE)
        if zc:
            zraw = _read_columns(parquet_path, zc)
            if zraw.shape != z.shape:
                raise RuntimeError("zraw %r and z %r shapes disagree"
                                   % (zraw.shape, z.shape))

    return Arm(z=z, zraw=zraw, theta=None, groups=groups, classes=classes,
               contract=contract,
               meta={"path": parquet_path, "n_groups": len(set(groups.tolist()))})


def load_sim(pattern: str,
             dedup_theta: bool = True,
             want_zraw: bool = True,
             max_rows: Optional[int] = None,
             seed: int = 0) -> Arm:
    """Load and pool simulated shards matching a glob, optionally deduplicated.

    dedup_theta keeps the FIRST row of each distinct theta. Rows sharing a
    theta differ only in seed_run, i.e. in simulator stochasticity; that is a
    legitimate quantity but it is not prior coverage, and counting it as the
    latter is what the seed-collision defect does.
    """
    import npe_contract as C

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("no shards match %r" % (pattern,))

    z, theta, contract = C.load_shards(paths, require_theta=True)
    if theta is None:
        raise ValueError("simulated shards returned no theta")

    zraw = None
    if want_zraw:
        blocks = []
        total = 0
        for p in paths:
            names = pq.read_schema(p).names
            zc = _ordered_cols(names, _ZRAW_RE)
            if not zc:
                blocks = None
                break
            b = _read_columns(p, zc)
            blocks.append(b)
            total += b.shape[0]
        if blocks:
            zraw = np.vstack(blocks)
            if zraw.shape != z.shape:
                raise RuntimeError(
                    "pooled zraw %r and z %r disagree; the shard order or row "
                    "count differs between the two reads."
                    % (zraw.shape, z.shape))

    meta: Dict[str, object] = {"pattern": pattern, "n_shards": len(paths),
                               "n_rows_raw": int(z.shape[0])}

    if dedup_theta:
        _, keep = np.unique(theta, axis=0, return_index=True)
        keep = np.sort(keep)
        meta["n_rows_dedup"] = int(keep.shape[0])
        meta["duplication_factor"] = float(z.shape[0]) / max(1, keep.shape[0])
        z = z[keep]
        theta = theta[keep]
        if zraw is not None:
            zraw = zraw[keep]

    if max_rows is not None and z.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(z.shape[0], size=max_rows, replace=False))
        meta["n_rows_subsampled"] = int(max_rows)
        z = z[sel]
        theta = theta[sel]
        if zraw is not None:
            zraw = zraw[sel]

    return Arm(z=z, zraw=zraw, theta=theta, groups=None, classes=None,
               contract=contract, meta=meta)


def topology_diversity(theta: np.ndarray,
                       param_names: Sequence[str],
                       axes: Sequence[str] = ("p0_conn", "d0_conn",
                                              "beta_conn")) -> Dict[str, int]:
    """Distinct topology draws behind the rows.

    Rows sharing a topology are not independent: one connectivity realisation
    is reused across every parameter draw made under it. The real arm is
    grouped by culture for exactly this reason; the gate has no equivalent
    grouping for the simulated side, so this number has to be reported rather
    than corrected for.
    """
    idx = [list(param_names).index(a) for a in axes if a in param_names]
    if not idx:
        return {"n_rows": int(theta.shape[0]), "n_topologies": -1}
    sub = theta[:, idx]
    return {"n_rows": int(theta.shape[0]),
            "n_topologies": int(np.unique(sub, axis=0).shape[0])}


def effective_rank(z: np.ndarray) -> Tuple[float, np.ndarray]:
    """Participation-ratio effective rank of an embedding, and its spectrum.

    r_eff = (sum_i s_i)^2 / sum_i s_i^2 on the eigenvalues s_i of the
    covariance. r_eff = 1 means every point lies on one direction; r_eff = E
    means isotropic. Reported because a collapsed embedding makes a gate PASS
    weak evidence (the test loses power in the directions that collapsed)
    while a REJECTION stays decisive.
    """
    x = np.asarray(z, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev[::-1], 0.0, None)
    s = ev.sum()
    if s <= 0:
        return 0.0, ev
    return float(s * s / np.sum(ev * ev)), ev
