#!/usr/bin/env python3
"""
npe_contract.py -- data contract for the SBI / NPE stage.

Single source of truth for:
  * what a training shard looks like on disk (columns, dtypes, sidecar),
  * the parameter coordinate system: natural log on some axes, linear on
    others, plus the inverse map back to biophysical units,
  * the box prior over inference coordinates,
  * the validation assertions that must pass before any training starts.

Scope boundary: this module loads, validates, and does coordinate algebra.
It does NOT build networks, train, sample, or plot. Keeping it that way is
what lets the estimator, the diagnostics, and the data source be swapped
independently.

ASCII-only by policy (HPC transfer safety). Greek letters are spelled out.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# torch is imported lazily inside prior() so that pure-numpy consumers
# (validation, coordinate maths, unit tests) do not pay the import cost.

__all__ = [
    "Contract",
    "load_shard",
    "load_shards",
    "validate_arrays",
    "make_synthetic_shard",
    "ValidationReport",
]

COORD_LN = "ln"
COORD_LINEAR = "linear"
VALID_COORDS = (COORD_LN, COORD_LINEAR)

Z_PREFIX = "z_"
TH_PREFIX = "th_"


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contract:
    """Immutable description of one family of compatible shards.

    Attributes
    ----------
    param_names : list of str, length p
        Column order for the theta block. Authoritative: downstream code
        must key on these names, never on positional index into the
        upstream 36-D registry.
    coord : list of str, length p
        Per-axis coordinate, either "ln" (theta = ln(natural)) or "linear"
        (theta = natural). Mixed coordinates are expected and correct.
    bounds_theta : ndarray, shape (p, 2)
        Prior box in INFERENCE coordinates. Row k is (lo_k, hi_k) with
        lo_k < hi_k strictly.
    embedding_dim : int
        E, the number of z_* columns.
    meta : dict
        Free-form provenance carried through from the sidecar. Not used for
        computation; used for the cross-shard compatibility check and for
        writing an audit trail alongside any trained model.
    """

    param_names: List[str]
    coord: List[str]
    bounds_theta: np.ndarray
    embedding_dim: int
    meta: Dict = field(default_factory=dict)

    # -- basic properties --------------------------------------------------

    @property
    def p(self) -> int:
        """Dimension of the inference parameter vector."""
        return len(self.param_names)

    @property
    def log_mask(self) -> np.ndarray:
        """Boolean mask, True on axes stored as natural log."""
        return np.array([c == COORD_LN for c in self.coord], dtype=bool)

    @property
    def low(self) -> np.ndarray:
        return np.asarray(self.bounds_theta, dtype=np.float64)[:, 0]

    @property
    def high(self) -> np.ndarray:
        return np.asarray(self.bounds_theta, dtype=np.float64)[:, 1]

    @property
    def theta_columns(self) -> List[str]:
        return [TH_PREFIX + n for n in self.param_names]

    @property
    def z_columns(self) -> List[str]:
        return [Z_PREFIX + ("%03d" % i) for i in range(self.embedding_dim)]

    # -- coordinate algebra ------------------------------------------------

    def to_natural(self, theta: np.ndarray) -> np.ndarray:
        """Map inference coordinates to natural (biophysical) units.

        natural_k = exp(theta_k) on log axes, theta_k elsewhere, for each
        fixed axis k in {0, ..., p-1}. Applied row-wise; theta may be
        (p,) or (n, p).
        """
        theta = np.asarray(theta, dtype=np.float64)
        single = theta.ndim == 1
        arr = theta.reshape(1, -1) if single else theta.copy()
        if arr.shape[-1] != self.p:
            raise ValueError(
                "to_natural: expected last axis %d, got %d" % (self.p, arr.shape[-1])
            )
        arr = arr.copy()
        m = self.log_mask
        arr[:, m] = np.exp(arr[:, m])
        return arr[0] if single else arr

    def to_inference(self, natural: np.ndarray) -> np.ndarray:
        """Inverse of to_natural. Raises on non-positive values on log axes."""
        natural = np.asarray(natural, dtype=np.float64)
        single = natural.ndim == 1
        arr = natural.reshape(1, -1) if single else natural.copy()
        if arr.shape[-1] != self.p:
            raise ValueError(
                "to_inference: expected last axis %d, got %d" % (self.p, arr.shape[-1])
            )
        arr = arr.copy()
        m = self.log_mask
        if np.any(arr[:, m] <= 0.0):
            raise ValueError("to_inference: non-positive value on a log axis")
        arr[:, m] = np.log(arr[:, m])
        return arr[0] if single else arr

    def bounds_natural(self) -> np.ndarray:
        """Prior box expressed in natural units, for reporting only."""
        return np.stack(
            [self.to_natural(self.low), self.to_natural(self.high)], axis=1
        )

    # -- prior -------------------------------------------------------------

    def prior(self, device: str = "cpu"):
        """Return the box prior as a torch Distribution.

        The prior is the product of independent uniforms

            p(theta) = prod_k  1{lo_k <= theta_k <= hi_k} / (hi_k - lo_k)

        over the INFERENCE coordinates. It is NOT uniform in natural units:
        on the log axes it is log-uniform, which is exactly how the campaign
        sampler drew them.
        """
        import torch
        from sbi.utils import BoxUniform

        return BoxUniform(
            low=torch.as_tensor(self.low, dtype=torch.float32, device=device),
            high=torch.as_tensor(self.high, dtype=torch.float32, device=device),
            device=device,
        )

    # -- io ----------------------------------------------------------------

    @staticmethod
    def from_sidecar(path: str) -> "Contract":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return Contract.from_dict(d)

    @staticmethod
    def from_dict(d: Dict) -> "Contract":
        try:
            names = list(d["param_names"])
            coord = list(d["coord"])
            bounds = np.asarray(d["bounds_theta"], dtype=np.float64)
            edim = int(d["embedding"]["embedding_dim"])
        except KeyError as exc:
            raise KeyError("sidecar missing required field: %s" % exc) from exc
        meta = {k: v for k, v in d.items() if k not in ("param_names", "coord", "bounds_theta")}
        c = Contract(param_names=names, coord=coord, bounds_theta=bounds,
                     embedding_dim=edim, meta=meta)
        c.validate()
        return c

    def to_dict(self) -> Dict:
        d = dict(self.meta)
        d["param_names"] = list(self.param_names)
        d["coord"] = list(self.coord)
        d["bounds_theta"] = np.asarray(self.bounds_theta).tolist()
        d.setdefault("embedding", {})["embedding_dim"] = int(self.embedding_dim)
        return d

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Assertions A2 and A3 from the export contract.

        A2 -- param_names, coord, bounds_theta all have the same length.
        A3 -- every bound interval is strictly non-degenerate.

        A3 is the guard that catches an accidentally included frozen axis
        (point interval lo == hi), which would give an undefined prior
        density and a divergent flow loss rather than an error message.
        """
        p = len(self.param_names)
        if len(self.coord) != p:
            raise ValueError("A2 failed: len(coord)=%d != len(param_names)=%d"
                             % (len(self.coord), p))
        b = np.asarray(self.bounds_theta, dtype=np.float64)
        if b.shape != (p, 2):
            raise ValueError("A2 failed: bounds_theta shape %s != (%d, 2)"
                             % (b.shape, p))
        bad_coord = [c for c in self.coord if c not in VALID_COORDS]
        if bad_coord:
            raise ValueError("A2 failed: unknown coord labels %s" % sorted(set(bad_coord)))
        if len(set(self.param_names)) != p:
            raise ValueError("A2 failed: duplicate names in param_names")
        width = b[:, 1] - b[:, 0]
        degenerate = [(self.param_names[k], float(b[k, 0]), float(b[k, 1]))
                      for k in range(p) if not width[k] > 0.0]
        if degenerate:
            raise ValueError(
                "A3 failed: non-positive-width prior interval on %s. "
                "A frozen axis (lo == hi) must be excluded from the label."
                % (degenerate,)
            )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive, got %d" % self.embedding_dim)


# ---------------------------------------------------------------------------
# Validation of loaded arrays
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Outcome of validate_arrays. `ok` is True iff `failures` is empty."""
    n_rows: int
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def raise_if_failed(self) -> "ValidationReport":
        if not self.ok:
            raise ValueError("shard validation failed:\n  - " + "\n  - ".join(self.failures))
        return self

    def summary(self) -> str:
        head = "rows=%d  status=%s" % (self.n_rows, "PASS" if self.ok else "FAIL")
        lines = [head]
        for f in self.failures:
            lines.append("  FAIL " + f)
        for w in self.warnings:
            lines.append("  WARN " + w)
        return "\n".join(lines)


def validate_arrays(
    z: np.ndarray,
    theta: Optional[np.ndarray],
    contract: Contract,
    norm_tol: float = 1e-4,
    check_in_box: bool = True,
    const_rtol: float = 1e-9,
) -> ValidationReport:
    """Run assertions A4, A5, A7, A9 over loaded arrays.

    A4 -- no constant theta column (a constant column means a frozen axis
          leaked in, or the shard is degenerate).
    A5 -- every theta row lies inside the prior box. Violations are
          REPORTED, never clipped: clipping would silently redefine the
          prior the estimator is trained under.
    A7 -- every z row has unit L2 norm.
    A9 -- no NaN or Inf anywhere.

    Parameters
    ----------
    theta : ndarray or None
        None for real-data shards, which carry no labels. A4 and A5 are
        skipped in that case.
    """
    rep = ValidationReport(n_rows=int(z.shape[0]))

    # -- shapes ----------------------------------------------------------
    if z.ndim != 2 or z.shape[1] != contract.embedding_dim:
        rep.failures.append(
            "z has shape %s, expected (n, %d)" % (z.shape, contract.embedding_dim))
        return rep

    # -- A9 --------------------------------------------------------------
    if not np.all(np.isfinite(z)):
        n_bad = int(np.sum(~np.isfinite(z)))
        rep.failures.append("A9: %d non-finite entries in z" % n_bad)

    # -- A7 --------------------------------------------------------------
    norms = np.linalg.norm(z, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    rep.stats["max_abs_norm_deviation"] = max_dev
    if max_dev > norm_tol:
        rep.failures.append(
            "A7: max |||z|| - 1| = %.3e exceeds tol %.1e. The embedding is "
            "expected L2-normalised; if this shard is pre-normalisation, load "
            "it as zraw instead." % (max_dev, norm_tol))

    if theta is None:
        return rep

    if theta.ndim != 2 or theta.shape[1] != contract.p:
        rep.failures.append("theta has shape %s, expected (n, %d)" % (theta.shape, contract.p))
        return rep
    if theta.shape[0] != z.shape[0]:
        rep.failures.append("row mismatch: z has %d rows, theta has %d"
                            % (z.shape[0], theta.shape[0]))
        return rep

    if not np.all(np.isfinite(theta)):
        n_bad = int(np.sum(~np.isfinite(theta)))
        rep.failures.append("A9: %d non-finite entries in theta" % n_bad)

    # -- A4 --------------------------------------------------------------
    # A constant column must be judged against the width of its own prior
    # interval, not against exact zero: numpy's std on a genuinely constant
    # column returns O(1e-16) round-off rather than 0.0, so an exact-zero
    # test silently never fires. The prior width is the physically
    # meaningful scale -- a column whose spread is a 1e-9 fraction of the
    # box it was drawn from carries no information about that axis.
    col_std = theta.std(axis=0)
    width = contract.high - contract.low
    rel_std = col_std / np.where(width > 0.0, width, 1.0)
    rep.stats["min_theta_col_std"] = float(np.min(col_std)) if col_std.size else 0.0
    rep.stats["min_theta_col_rel_std"] = float(np.min(rel_std)) if rel_std.size else 0.0
    const_cols = [(contract.param_names[k], float(rel_std[k]))
                  for k in range(contract.p) if rel_std[k] <= const_rtol]
    if const_cols:
        rep.failures.append(
            "A4: effectively constant theta column(s) %s (std as a fraction "
            "of prior width, threshold %.1e). A constant column has zero "
            "spread in the data and will drive the flow log-density to +inf."
            % (const_cols, const_rtol))

    # -- A5 --------------------------------------------------------------
    if check_in_box:
        below = theta < contract.low[None, :]
        above = theta > contract.high[None, :]
        n_out = int(np.sum(np.any(below | above, axis=1)))
        rep.stats["n_rows_out_of_box"] = n_out
        if n_out > 0:
            worst = []
            for k in range(contract.p):
                nk = int(np.sum(below[:, k] | above[:, k]))
                if nk:
                    worst.append((contract.param_names[k], nk))
            worst.sort(key=lambda t: -t[1])
            rep.failures.append(
                "A5: %d/%d rows outside the prior box; worst axes %s. "
                "Do not clip -- reconcile the bounds instead."
                % (n_out, theta.shape[0], worst[:5]))

    return rep


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _sidecar_path_for(parquet_path: str) -> str:
    base, _ = os.path.splitext(parquet_path)
    return base + ".json"


def load_shard(
    data_path: str,
    sidecar_path: Optional[str] = None,
    require_theta: bool = True,
    validate: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Contract]:
    """Load one shard.

    Accepts .parquet (preferred) or .npz. Returns (z, theta, contract),
    with theta None when the shard carries no labels (real recordings).

    Column order is taken from the sidecar, never from the file's own
    column order, so a reordered export cannot silently permute the axes.
    """
    if sidecar_path is None:
        sidecar_path = _sidecar_path_for(data_path)
    contract = Contract.from_sidecar(sidecar_path)

    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(data_path)
        cols = set(df.columns)
        missing_z = [c for c in contract.z_columns if c not in cols]
        if missing_z:
            raise KeyError("shard missing z columns: %s" % missing_z[:5])
        z = df[contract.z_columns].to_numpy(dtype=np.float32)
        has_theta = all(c in cols for c in contract.theta_columns)
        if has_theta:
            theta = df[contract.theta_columns].to_numpy(dtype=np.float64)
        else:
            missing_th = [c for c in contract.theta_columns if c not in cols]
            if require_theta:
                raise KeyError("shard missing theta columns: %s" % missing_th[:5])
            theta = None
    elif ext == ".npz":
        with np.load(data_path, allow_pickle=False) as npz:
            z = np.asarray(npz["z"], dtype=np.float32)
            if "theta" in npz:
                theta = np.asarray(npz["theta"], dtype=np.float64)
            elif require_theta:
                raise KeyError("npz shard has no 'theta' array")
            else:
                theta = None
    else:
        raise ValueError("unsupported shard extension %r (want .parquet or .npz)" % ext)

    if validate:
        validate_arrays(z, theta, contract).raise_if_failed()
    return z, theta, contract


def _contracts_compatible(a: Contract, b: Contract) -> List[str]:
    """Assertion A10: return a list of reasons a and b may not be pooled."""
    reasons = []
    if list(a.param_names) != list(b.param_names):
        reasons.append("param_names differ")
    if list(a.coord) != list(b.coord):
        reasons.append("coord differ")
    if not np.allclose(a.bounds_theta, b.bounds_theta, rtol=0, atol=0):
        reasons.append("bounds_theta differ")
    if a.embedding_dim != b.embedding_dim:
        reasons.append("embedding_dim differ (%d vs %d)" % (a.embedding_dim, b.embedding_dim))
    sha_a = a.meta.get("embedding", {}).get("dsn_checkpoint_sha256")
    sha_b = b.meta.get("embedding", {}).get("dsn_checkpoint_sha256")
    if sha_a is not None and sha_b is not None and sha_a != sha_b:
        reasons.append("dsn_checkpoint_sha256 differ -- embeddings come from "
                       "different encoders and cannot be pooled")
    return reasons


def load_shards(
    data_paths: Sequence[str],
    require_theta: bool = True,
    validate: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Contract]:
    """Load and concatenate several shards, enforcing assertion A10.

    Shards that disagree on the parameter names, coordinates, prior box,
    embedding dimension, or DSN checkpoint digest are refused rather than
    pooled. Pooling incompatible shards is silent and unrecoverable, so
    this is a hard error.
    """
    if not data_paths:
        raise ValueError("load_shards: no paths given")
    zs, ths, contract = [], [], None
    for path in data_paths:
        z, th, c = load_shard(path, require_theta=require_theta, validate=validate)
        if contract is None:
            contract = c
        else:
            reasons = _contracts_compatible(contract, c)
            if reasons:
                raise ValueError(
                    "A10 failed: shard %r is not poolable with the first shard: %s"
                    % (path, "; ".join(reasons)))
        zs.append(z)
        ths.append(th)
    z_all = np.concatenate(zs, axis=0)
    if any(t is None for t in ths):
        th_all = None
    else:
        th_all = np.concatenate(ths, axis=0)
    return z_all, th_all, contract


# ---------------------------------------------------------------------------
# Synthetic shard, for development before the real export lands
# ---------------------------------------------------------------------------

def make_synthetic_shard(
    out_dir: str,
    name: str = "synthetic_000",
    n_rows: int = 2048,
    p: int = 27,
    embedding_dim: int = 12,
    n_log_axes: int = 19,
    seed: int = 0,
    write: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Contract, Optional[str]]:
    """Build a schema-valid shard with a KNOWN, deterministic forward map.

    The generative map is

        u        = (theta - lo) / (hi - lo)   in [0, 1]^p
        raw      = W u + b + noise
        z        = raw / ||raw||_2

    with W fixed by `seed`. This is not the real simulator and makes no
    claim to be: its only job is to give the pipeline a well-posed
    (theta, z) relationship with the right shapes, coordinate mix, and the
    unit-norm constraint, so that everything downstream can be developed
    and tested before the real export arrives.

    Returns (z, theta, contract, parquet_path or None).
    """
    rng = np.random.default_rng(seed)

    names = ["ax%02d" % k for k in range(p)]
    coord = [COORD_LN] * min(n_log_axes, p) + [COORD_LINEAR] * max(0, p - n_log_axes)
    lo = rng.uniform(-3.0, -0.5, size=p)
    hi = lo + rng.uniform(1.0, 4.0, size=p)
    bounds = np.stack([lo, hi], axis=1)

    contract = Contract(
        param_names=names,
        coord=coord,
        bounds_theta=bounds,
        embedding_dim=embedding_dim,
        meta={
            "schema_version": 1,
            "campaign_id": name,
            "synthetic": True,
            "embedding": {
                "embedding_dim": embedding_dim,
                "dsn_checkpoint_sha256": "synthetic-%d" % seed,
                "l2_normalised": True,
            },
        },
    )
    contract.validate()

    theta = rng.uniform(lo[None, :], hi[None, :], size=(n_rows, p))
    u = (theta - lo[None, :]) / (hi - lo)[None, :]
    w_rng = np.random.default_rng(seed + 1000)
    W = w_rng.normal(size=(p, embedding_dim))
    b = w_rng.normal(size=(embedding_dim,))
    raw = u @ W + b[None, :] + 0.05 * rng.normal(size=(n_rows, embedding_dim))
    z = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    path = None
    if write:
        os.makedirs(out_dir, exist_ok=True)
        import pandas as pd
        cols = {}
        for j, c in enumerate(contract.z_columns):
            cols[c] = z[:, j].astype(np.float32)
        for j, c in enumerate(contract.theta_columns):
            cols[c] = theta[:, j].astype(np.float64)
        cols["campaign_id"] = np.array([name] * n_rows, dtype=object)
        cols["iter_idx"] = np.arange(n_rows, dtype=np.int32)
        df = pd.DataFrame(cols)
        path = os.path.join(out_dir, name + ".parquet")
        df.to_parquet(path, index=False)
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(contract.to_dict(), fh, indent=2)

    return z.astype(np.float32), theta, contract, path
