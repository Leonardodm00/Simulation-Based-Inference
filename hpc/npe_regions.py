#!/usr/bin/env python3
"""
npe_regions.py -- per-axis viable parameter regions for the next simulation
campaign.

NOT TSNPE. There is no sequential loop, no rejection sampling, no SIR, no
truncated proposal object. The deliverable is a BOX per parameter axis,
written to disk in a format another program reads to draw its own next round
of simulations however it likes.

WHY NOT PER-AXIS MIN/MAX OF POSTERIOR DRAWS

See docs/hpr_projection_bias.md for the full derivation. Summary: the range
of N posterior draws underestimates the true projection of the joint HPR by
a factor that shrinks only as sqrt(2 log N) -- at p = 27 the gap is a factor
of about two at feasible N, and closing it needs N ~ 10^12. Worse, the true
projection is itself a useless target: at p = 27 the box circumscribing the
99.9% joint ellipsoid has 6*10^11 times its volume and retains posterior
mass indistinguishable from one in double precision. Neither the estimator
nor its estimand is usable.

WHAT THIS MODULE DOES INSTEAD

For each fixed recording, given pooled posterior draws theta^(1..N) ~
q(theta | z), find ONE shared per-axis tail probability alpha such that the
box

    B(alpha) = PROD_j [ Q_j(alpha/2), Q_j(1 - alpha/2) ]                (1)

built from per-axis empirical quantiles Q_j retains a MEASURED target mass:

    m(alpha) = (1/N) SUM_n  1{ theta^(n) in B(alpha) }        =  target   (2)

m(alpha) is exactly weakly monotone non-increasing in alpha (np.quantile is
monotone in its probability argument by construction, so decreasing alpha
can only widen every axis simultaneously, which can only add points to the
box, never remove them). Equation (2) is therefore solved by ordinary
bisection on alpha, with no smoothness or root-uniqueness assumption needed
-- which is why this module hand-rolls a ten-line bisection rather than
reaching for scipy.optimize: the general-purpose solver is built for smooth
functions and buys nothing here, while a monotone bisection is provably
correct for a step function.

The guarantee that ships with the output is therefore MEASURED, not
asserted: "this box contains target% of this recording's posterior mass, at
these N draws." It does not move with the draw budget the way a sample-range
box would (see the theory doc, S4-S5).

WINDOWS AND RECORDINGS

A recording is W windows of one culture with one unknown, shared theta. Per
recording, this module pools (concatenates -- "union") the posterior draws
from every window before calibrating. This is the conservative choice: it
does not attempt to combine the W windows' evidence into a sharper estimate
of that recording's theta (a proper multi-observation update, which the
amortized single-window posterior does not directly give you), and instead
reports the region compatible with ANY window read on its own. Wider than
necessary, safe in the direction that matters for a box that must not
wrongly exclude viable parameter space. See R3 in the validation suite.

ACROSS RECORDINGS

Per-axis intervals are unioned across the R recordings. Projection commutes
with union for the joint region, so this step is exact rather than an
approximation:

    Pi_j( UNION_r M_r )  =  UNION_r Pi_j(M_r)   for each fixed axis j.  (3)

A union of intervals is NOT itself an interval. Two things are therefore
reported per axis: the HULL (min of all lows, max of all highs) as the
primary, simple, always-valid box, and the actual DISJOINT SEGMENTS, so a
downstream sampler can see -- and choose to respect -- any gap that no
recording's data actually supports.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "retained_mass",
    "calibrate_box",
    "RecordingBox",
    "extract_recording_region",
    "AxisUnion",
    "union_intervals",
    "union_across_recordings",
    "require_preconditions",
    "RegionSet",
    "build_region_sets",
    "build_and_write_all",
    "extract_regions_from_real_data",
]


# ---------------------------------------------------------------------------
# Core: one recording, one target mass
# ---------------------------------------------------------------------------

def retained_mass(draws: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Equation (2): the fraction of `draws` inside the axis-aligned box.

    Parameters
    ----------
    draws : (N, p)
    lo, hi : (p,)

    Returns
    -------
    float in [0, 1]. Standard error of this Monte Carlo estimate is
    sqrt(m(1-m)/N) -- at m=0.999 and N=10^4 that is 3e-4, adequate for the
    targets this module uses but not for a target much closer to 1.
    """
    draws = np.atleast_2d(np.asarray(draws, dtype=np.float64))
    inside = np.all((draws >= lo[None, :]) & (draws <= hi[None, :]), axis=1)
    return float(np.mean(inside))


def _box_at_alpha(draws: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return lo, hi


@dataclass
class RecordingBox:
    """Calibrated box for one recording at one target mass."""
    lo: np.ndarray
    hi: np.ndarray
    alpha: float
    achieved_mass: float
    target_mass: float
    n_draws: int
    n_windows: int


def calibrate_box(draws: np.ndarray, target_mass: float,
                  n_iter: int = 40, alpha_floor: float = 1e-6
                  ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Bisect alpha in (1), equation (2), to hit `target_mass`.

    A SINGLE shared alpha across every axis -- one free parameter, not a
    per-axis search. This keeps the box a genuine "same confidence depth on
    every axis" object rather than a p-dimensional optimisation problem, and
    is the direct analogue of equation (13) in the theory note.

    Parameters
    ----------
    draws : (N, p)
        Pooled posterior draws for one recording (already unioned across its
        windows -- see extract_recording_region).
    target_mass : float in (0, 1)
    n_iter : int
        Bisection steps. 40 steps of a monotone bisection on a probability
        in [0, 1] resolves alpha to about 2^-40, far past the resolution the
        N draws can actually support; the loop is cheap (one quantile call
        per step) so there is no reason to cut it short.
    alpha_floor : float
        Smallest alpha considered, avoiding an exactly-zero-width tail cut
        that degenerates to the raw sample range (and reintroduces the bias
        the theory note warns against, at the target_mass = 1 limit only).

    Returns
    -------
    (lo, hi, alpha, achieved_mass)
    """
    draws = np.atleast_2d(np.asarray(draws, dtype=np.float64))
    if not (0.0 < target_mass < 1.0):
        raise ValueError("target_mass must be in (0, 1), got %r" % target_mass)
    if draws.shape[0] < 20:
        raise ValueError("calibrate_box needs at least 20 draws for a "
                         "meaningful quantile; got %d" % draws.shape[0])

    a_lo, a_hi = alpha_floor, 1.0 - alpha_floor
    # m(a_lo) ~= 1 >= target, m(a_hi) ~= 0 <= target: bracket always valid.
    for _ in range(n_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        lo, hi = _box_at_alpha(draws, a_mid)
        m = retained_mass(draws, lo, hi)
        if m >= target_mass:
            a_lo = a_mid          # box still big enough: narrow the search up
        else:
            a_hi = a_mid
    alpha = a_lo
    lo, hi = _box_at_alpha(draws, alpha)
    return lo, hi, float(alpha), retained_mass(draws, lo, hi)


def extract_recording_region(window_draws: Sequence[np.ndarray],
                             target_masses: Sequence[float]
                             ) -> Dict[float, RecordingBox]:
    """One recording: pool its windows' draws, calibrate a box per target.

    Parameters
    ----------
    window_draws : sequence of (n_draws_w, p) arrays
        One array per window of this recording. Pooled by concatenation --
        the "union" described in the WINDOWS AND RECORDINGS section above.
    target_masses : sequence of float

    Returns
    -------
    dict target_mass -> RecordingBox
    """
    if not window_draws:
        raise ValueError("extract_recording_region needs at least one window")
    pooled = np.concatenate([np.atleast_2d(w) for w in window_draws], axis=0)
    out: Dict[float, RecordingBox] = {}
    for m in target_masses:
        lo, hi, alpha, achieved = calibrate_box(pooled, m)
        out[m] = RecordingBox(lo=lo, hi=hi, alpha=alpha,
                              achieved_mass=achieved, target_mass=m,
                              n_draws=pooled.shape[0],
                              n_windows=len(window_draws))
    return out


# ---------------------------------------------------------------------------
# Across recordings: hull + disjoint segments
# ---------------------------------------------------------------------------

def union_intervals(intervals: Sequence[Tuple[float, float]]
                    ) -> List[Tuple[float, float]]:
    """Merge overlapping/touching intervals into disjoint sorted segments.

    Standard sweep: sort by lower endpoint, extend the current segment while
    the next interval overlaps or touches it, else start a new one.
    """
    ivs = sorted((float(lo), float(hi)) for lo, hi in intervals)
    if not ivs:
        return []
    merged = [ivs[0]]
    for lo, hi in ivs[1:]:
        clo, chi = merged[-1]
        if lo <= chi:
            merged[-1] = (clo, max(chi, hi))
        else:
            merged.append((lo, hi))
    return merged


@dataclass
class AxisUnion:
    """Union across recordings, for one axis, at one target mass."""
    name: str
    hull_lo: float
    hull_hi: float
    segments: List[Tuple[float, float]]
    prior_lo: float
    prior_hi: float
    clipped: bool             # True if the hull exceeded the prior box

    @property
    def shrinkage(self) -> float:
        """Fraction of the ORIGINAL prior range the hull retains.

        Near 1: this axis was not constrained by the data at all -- the
        sloppiness result showing up in directly usable form. Near 0: this
        axis was constrained hard.
        """
        span = self.prior_hi - self.prior_lo
        if span <= 0.0:
            return float("nan")
        return (self.hull_hi - self.hull_lo) / span

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def union_across_recordings(per_recording: Mapping[str, RecordingBox],
                            param_names: Sequence[str],
                            prior_low: np.ndarray, prior_high: np.ndarray
                            ) -> List[AxisUnion]:
    """Equation (3): union the calibrated per-recording boxes, per axis.

    Parameters
    ----------
    per_recording : dict recording_id -> RecordingBox
        All at the SAME target mass; call once per target mass.
    param_names : sequence of str, length p
    prior_low, prior_high : (p,)
        Original prior box, for clipping and shrinkage.

    Returns
    -------
    list of AxisUnion, length p, in param_names order.
    """
    if not per_recording:
        raise ValueError("union_across_recordings needs at least one recording")
    p = len(param_names)
    los = np.stack([b.lo for b in per_recording.values()])   # (R, p)
    his = np.stack([b.hi for b in per_recording.values()])
    out = []
    for j in range(p):
        segs_raw = list(zip(los[:, j].tolist(), his[:, j].tolist()))
        segs = union_intervals(segs_raw)
        hull_lo = min(s[0] for s in segs)
        hull_hi = max(s[1] for s in segs)
        clipped = bool(hull_lo < prior_low[j] or hull_hi > prior_high[j])
        hull_lo = max(hull_lo, float(prior_low[j]))
        hull_hi = min(hull_hi, float(prior_high[j]))
        segs = [(max(lo, float(prior_low[j])), min(hi, float(prior_high[j])))
               for lo, hi in segs]
        segs = [(lo, hi) for lo, hi in segs if hi > lo]
        out.append(AxisUnion(name=param_names[j], hull_lo=hull_lo,
                             hull_hi=hull_hi, segments=segs,
                             prior_lo=float(prior_low[j]),
                             prior_hi=float(prior_high[j]), clipped=clipped))
    return out


# ---------------------------------------------------------------------------
# Preconditions -- light, decoupled check, not an orchestration layer
# ---------------------------------------------------------------------------

def require_preconditions(passed: Mapping[str, bool], strict: bool = True
                          ) -> List[str]:
    """Check a caller-supplied map of {check_name: passed} before emitting.

    Deliberately dumb: this module does not read npe_misspec's or
    npe_local's output files itself -- that would couple it to their exact
    schemas for no real benefit. The caller runs whatever diagnostics apply
    (the misspecification gate, expected_coverage, the LCT, ...) and passes
    in the verdicts it already has. A region built on top of a failing gate
    or an overconfident posterior permanently discards true parameter space,
    so the default is to refuse.

    Parameters
    ----------
    passed : dict, e.g. {"misspec_gate": True, "expected_coverage": False}
    strict : bool
        True (default): raise if any value is False. False: return the list
        of failed names without raising, for a caller that wants to warn
        rather than block.

    Returns
    -------
    List of failed check names (empty if all passed).
    """
    failed = [k for k, v in passed.items() if not v]
    if failed and strict:
        raise RuntimeError(
            "refusing to build regions: failed precondition(s) %s. A region "
            "built on an overconfident or misspecified posterior can "
            "permanently exclude viable parameter space. Pass strict=False "
            "to override." % failed)
    return failed


# ---------------------------------------------------------------------------
# Top-level result and IO
# ---------------------------------------------------------------------------

@dataclass
class RegionSet:
    """Everything for one target mass, ready to write out."""
    target_mass: float
    param_names: List[str]
    coord: List[str]
    axes: List[AxisUnion]
    n_recordings: int
    n_windows_total: int
    n_draws_total: int
    achieved_mass_range: Tuple[float, float]   # (min, max) over recordings
    embedding_dim: int = 0
    provenance: Dict = field(default_factory=dict)

    def bounds_theta(self) -> np.ndarray:
        return np.stack([[a.hull_lo, a.hull_hi] for a in self.axes])

    def to_sidecar_dict(self) -> Dict:
        """A drop-in-shaped dict: param_names/coord/bounds_theta/embedding,
        loadable by npe_contract.Contract.from_dict unchanged, plus the
        calibration detail riding in keys that from_dict folds into meta.
        """
        d = {
            "param_names": list(self.param_names),
            "coord": list(self.coord),
            "bounds_theta": self.bounds_theta().tolist(),
            "embedding": {"embedding_dim": int(self.embedding_dim)},
            "region_target_mass": self.target_mass,
            "region_achieved_mass_range": list(self.achieved_mass_range),
            "region_n_recordings": self.n_recordings,
            "region_n_windows_total": self.n_windows_total,
            "region_n_draws_total": self.n_draws_total,
            "region_shrinkage": {a.name: a.shrinkage for a in self.axes},
            "region_n_segments": {a.name: a.n_segments for a in self.axes},
            "region_segments": {a.name: a.segments for a in self.axes},
            "region_clipped_to_prior": {a.name: a.clipped for a in self.axes},
            "region_provenance": dict(self.provenance),
        }
        return d


def build_region_sets(per_window_draws: Mapping[str, Sequence[np.ndarray]],
                      target_masses: Sequence[float],
                      param_names: Sequence[str], coord: Sequence[str],
                      prior_low: np.ndarray, prior_high: np.ndarray,
                      embedding_dim: int = 0,
                      provenance: Optional[Dict] = None
                      ) -> Dict[float, RegionSet]:
    """Full pipeline: per recording -> calibrate -> union -> RegionSet, per
    target mass.

    Parameters
    ----------
    per_window_draws : dict recording_id -> sequence of (n_w, p) arrays
        Posterior draws, one array per window, keyed by recording.
    target_masses : sequence of float
    param_names, coord : as in npe_contract.Contract
    prior_low, prior_high : (p,)
    embedding_dim : int
        Recorded in the output's "embedding" block for schema compatibility
        with npe_contract.Contract.from_dict. An explicit parameter rather
        than a provenance-dict lookup, so a caller who forgets to set it
        gets 0 (visibly wrong) rather than a value that quietly depends on
        which keys happened to be in a free-form dict.
    provenance : dict or None
        Free-form, carried into every RegionSet's provenance field and hence
        into every emitted file (checkpoint sha, group_col used, class
        filter, seed, real-export digest -- whatever the caller has).

    Returns
    -------
    dict target_mass -> RegionSet
    """
    if not per_window_draws:
        raise ValueError("build_region_sets needs at least one recording")
    provenance = dict(provenance or {})
    per_recording_by_mass: Dict[float, Dict[str, RecordingBox]] = {
        m: {} for m in target_masses}
    n_windows_total = 0
    n_draws_total = 0
    for rec_id, windows in per_window_draws.items():
        regions = extract_recording_region(windows, target_masses)
        for m, box in regions.items():
            per_recording_by_mass[m][rec_id] = box
        n_windows_total += len(windows)
        n_draws_total += sum(int(np.atleast_2d(w).shape[0]) for w in windows)

    out: Dict[float, RegionSet] = {}
    for m in target_masses:
        axes = union_across_recordings(per_recording_by_mass[m], param_names,
                                       prior_low, prior_high)
        achieved = [b.achieved_mass for b in per_recording_by_mass[m].values()]
        out[m] = RegionSet(
            target_mass=m, param_names=list(param_names), coord=list(coord),
            axes=axes, n_recordings=len(per_recording_by_mass[m]),
            n_windows_total=n_windows_total, n_draws_total=n_draws_total,
            achieved_mass_range=(float(np.min(achieved)),
                                 float(np.max(achieved))),
            embedding_dim=int(embedding_dim), provenance=provenance)
    return out


def build_and_write_all(per_window_draws: Mapping[str, Sequence[np.ndarray]],
                        target_masses: Sequence[float],
                        param_names: Sequence[str], coord: Sequence[str],
                        prior_low: np.ndarray, prior_high: np.ndarray,
                        out_stem: str, contract=None,
                        provenance: Optional[Dict] = None,
                        preconditions: Optional[Mapping[str, bool]] = None,
                        strict_preconditions: bool = True
                        ) -> Dict[float, Tuple[str, str]]:
    """The single entry point: build every target mass and write every file.

    Writes <out_stem>_m<mass>.json for each target mass (each a self
    contained, drop-in sidecar-shaped file) and ONE <out_stem>.csv with a
    row per (axis, target_mass) -- the "easily readable from another
    algorithm" long-format table.

    Parameters
    ----------
    contract : npe_contract.Contract or None
        If given, natural-unit columns are added to the CSV via
        contract.to_natural(); the CSV then carries lo_natural/hi_natural
        alongside lo_stored/hi_stored. Optional because the core pipeline
        above never needs a Contract object, only its three plain arrays.
    preconditions : dict or None
        Forwarded to require_preconditions before anything is computed.
    """
    if preconditions is not None:
        require_preconditions(preconditions, strict=strict_preconditions)

    sets = build_region_sets(
        per_window_draws, target_masses, param_names, coord, prior_low,
        prior_high,
        embedding_dim=(contract.embedding_dim if contract is not None else 0),
        provenance=provenance)

    all_rows = []
    out: Dict[float, Tuple[str, str]] = {}
    for m in target_masses:
        rs = sets[m]
        json_path = "%s_m%s.json" % (out_stem, str(m).replace(".", "p"))
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(rs.to_sidecar_dict(), fh, indent=2)
        for a in rs.axes:
            row = {
                "axis_name": a.name,
                "target_mass": m,
                "coord": rs.coord[rs.param_names.index(a.name)],
                "lo_stored": a.hull_lo, "hi_stored": a.hull_hi,
                "shrinkage": a.shrinkage, "n_segments": a.n_segments,
                "clipped_to_prior": a.clipped,
                "n_recordings": rs.n_recordings,
            }
            if contract is not None:
                j = rs.param_names.index(a.name)
                is_log = contract.coord[j] == "ln"
                # to_natural expects a full p-length parameter VECTOR (it
                # decides ln-vs-linear per axis across all p at once); a
                # single axis's (lo, hi) pair is not that, so apply the
                # per-axis rule directly instead of slicing into it.
                row["lo_natural"] = float(np.exp(a.hull_lo) if is_log
                                          else a.hull_lo)
                row["hi_natural"] = float(np.exp(a.hull_hi) if is_log
                                          else a.hull_hi)
            all_rows.append(row)
        out[m] = (json_path, "")

    csv_path = out_stem + ".csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    for m in target_masses:
        out[m] = (out[m][0], csv_path)
    return out


# ---------------------------------------------------------------------------
# Real-data loader -- lazy import boundary, NOT YET RUN (see module note)
# ---------------------------------------------------------------------------
#
# Everything above this line is numpy only and is what the validation suite
# exercises. Everything below needs torch, sbi, and pandas, and has never
# been executed against a real trained ensemble or a real shard -- there is
# no sbi/torch install in the environment this module was built in (same
# situation as npe_local.local_c2st). Treat this function as reviewed
# against the sbi/npe_model source, not as tested code.

def _require_ensemble_deps():
    try:
        import torch          # noqa: F401
        import pandas as pd   # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "extract_regions_from_real_data needs torch and pandas, plus "
            "sbi (via npe_model.load_ensemble). The core of this module "
            "(everything above this boundary) needs neither. Original "
            "error: %s" % (exc,)) from exc


def extract_regions_from_real_data(
    ensemble_dir: str,
    real_shard_paths: Sequence[str],
    group_col: str,
    out_stem: str,
    class_col: Optional[str] = None,
    only_class: Optional[str] = None,
    target_masses: Sequence[float] = (0.95, 0.99, 0.999),
    n_draws_per_window: int = 2000,
    seed: int = 0,
    preconditions: Optional[Mapping[str, bool]] = None,
    strict_preconditions: bool = True,
) -> Dict[float, Tuple[str, str]]:
    """Load a trained ensemble and real shards, sample, and write regions.

    Parameters
    ----------
    ensemble_dir : str
        Passed to npe_model.load_ensemble. Its saved contract (written by
        save_ensemble) supplies param_names, coord, and the prior box.
    real_shard_paths : sequence of str
        One or more real .parquet shards, require_theta=False (real
        recordings carry no theta).
    group_col : str
        Recording identifier column, YOUR choice -- see groups_from_table
        in npe_misspec.py, reused here. The real-recording exporter is not
        yet written (Sbi-extractor README section 10), so this is an
        explicit argument, not a guess.
    out_stem : str
    class_col, only_class : as in npe_misspec.misspecification_gate.
        only_class=None pools every class into one set of regions. Unlike
        the misspecification gate, this function does NOT automatically
        also compute per-class regions -- control and pathological cultures
        plausibly warrant separately targeted simulation campaigns, so call
        this once per class with only_class set if that is what you want,
        rather than have the difference computed silently by default.
    target_masses, n_draws_per_window, seed : as documented above.
    preconditions, strict_preconditions : forwarded to
        build_and_write_all -> require_preconditions. Pass the verdicts
        from the misspecification gate and from expected_coverage/the LCT
        here; this function does not read their output files itself, to
        avoid coupling to their exact schemas.

    Returns
    -------
    dict target_mass -> (json_path, csv_path), from build_and_write_all.
    """
    _require_ensemble_deps()
    import torch
    import pandas as pd

    from npe_contract import load_shards
    from npe_misspec import groups_from_table
    from npe_model import load_ensemble

    ensemble, meta = load_ensemble(ensemble_dir)
    if "contract" not in meta:
        raise KeyError("ensemble.json at %r has no 'contract' block -- it "
                       "was saved without save_ensemble(..., contract=...)"
                       % ensemble_dir)
    from npe_contract import Contract
    contract = Contract.from_dict(meta["contract"])

    # z, validated and column-ordered by the contract (load_shards enforces
    # A10 cross-shard compatibility). group/class columns are re-read
    # directly from the same parquet files, in the same path order, so the
    # two loops align row-for-row without needing load_shard to expose its
    # internal DataFrame.
    z_all, _, contract_check = load_shards(list(real_shard_paths),
                                           require_theta=False)
    if contract_check.embedding_dim != contract.embedding_dim:
        raise ValueError(
            "real shard embedding_dim=%d does not match the ensemble's "
            "contract embedding_dim=%d" % (contract_check.embedding_dim,
                                           contract.embedding_dim))
    groups_parts, classes_parts = [], []
    for path in real_shard_paths:
        df = pd.read_parquet(path)
        g, c = groups_from_table(df, group_col, class_col)
        groups_parts.append(g)
        classes_parts.append(c)
    groups = np.concatenate(groups_parts)
    classes = (np.concatenate(classes_parts) if class_col is not None
              else None)
    if groups.shape[0] != z_all.shape[0]:
        raise ValueError(
            "row count mismatch between z (%d, from load_shards) and groups "
            "(%d, read directly): a shard's row order may differ between "
            "the two read paths" % (z_all.shape[0], groups.shape[0]))

    if only_class is not None:
        if classes is None:
            raise ValueError("only_class given but class_col is None")
        keep = classes == only_class
        z_all, groups = z_all[keep], groups[keep]

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    per_window_draws: Dict[str, List[np.ndarray]] = {}
    for i in range(z_all.shape[0]):
        rec = str(groups[i])
        x = torch.as_tensor(z_all[i], dtype=torch.float32)
        draws = ensemble.sample((n_draws_per_window,), x=x,
                                show_progress_bars=False)
        per_window_draws.setdefault(rec, []).append(
            np.asarray(draws.detach().cpu().numpy(), dtype=np.float64))

    provenance = {
        "ensemble_dir": ensemble_dir,
        "real_shard_paths": list(real_shard_paths),
        "group_col": group_col, "class_col": class_col,
        "only_class": only_class, "n_draws_per_window": n_draws_per_window,
        "seed": seed, "embedding_dim": contract.embedding_dim,
        "dsn_checkpoint_sha256": contract.meta.get("dsn_checkpoint_sha256"),
    }
    return build_and_write_all(
        per_window_draws, target_masses, contract.param_names,
        contract.coord, contract.low, contract.high, out_stem,
        contract=contract, provenance=provenance,
        preconditions=preconditions, strict_preconditions=strict_preconditions)
