#!/usr/bin/env python3
"""
npe_misspec.py -- layer 0: is the simulator even the right model?

Every other diagnostic in this repo tests the INFERENCE against the
simulator's own joint. This one tests the SIMULATOR against reality, and it
runs before any of them: if the real embeddings do not lie inside the
simulated distribution, a perfectly calibrated posterior is a posterior for
a world that is not this one.

The statistic is the squared maximum mean discrepancy between the simulated
and real embedding clouds, following Schmitt et al. (2023). For two samples
A and B and a positive-definite kernel k,

    MMD^2(A, B) = mean_{a,a'} k(a,a') + mean_{b,b'} k(b,b')
                  - 2 mean_{a,b} k(a,b)                          (1)

which is zero if and only if the two distributions agree, for a
characteristic kernel.

WHY THIS MODULE EXISTS RATHER THAN JUST embedding_overlap()

npe_diagnostics.embedding_overlap() computes (1) with a sampling-based null
and is correct for INDEPENDENT real embeddings. The real cohort is not
independent: each recording is cut into W_r disjoint windows, and those
windows share one culture and one unknown theta. Feeding all of them to a
null built from i.i.d. simulated draws compares a clustered observation
against an unclustered null. The null is then too tight and the gate fires
on a well-specified model.

The fix here is deliberately the simplest one that is actually valid rather
than the most powerful one:

    ONE WINDOW PER RECORDING, REPEATED.

Draw one window from each of the R recordings, giving R points that are
mutually independent and are each a single window -- exactly comparable to a
single simulated row, which is also one window of one simulation (T_win =
180 s gives one row per simulation). Compare against a null of R-point
simulated subsets. Repeat over several window choices and report the median
p-value and the fraction of choices that reject.

This throws away (W_r - 1)/W_r of the real rows. That is the honest price:
the effective sample size was R all along, and the discarded rows were never
carrying independent information about whether the simulator is right. The
alternative -- averaging each recording's windows into one point -- would
give a real unit with lower variance than a simulated unit and inflate (1)
by construction.

The i.i.d. null is still computed and reported, so the size of the
correction is visible rather than asserted.

WHAT A PASS DOES AND DOES NOT MEAN

Schmitt et al. derive their detection guarantee from an augmented training
objective that maps p(x | M) to a unit Gaussian in summary space. This DSN
is metric-learned onto S^{E-1} instead, so that construction does not apply
and their critical values do not transfer -- which is why the null here is
calibrated by simulation-to-simulation resampling and never from a table.

One direction survives regardless and needs no Gaussianity: h_psi is
DETERMINISTIC, so if the pushforward laws of z differ then the laws of x
differ. A rejection is therefore decisive.

The converse is false and was false for Schmitt et al. too -- they give an
explicit counter-example where the data distributions differ with zero
summary-space MMD. A pass is weak evidence, not a clean bill of health, and
minimum_detectable_shift() exists so that "weak" can be given a number.

Two further limits worth stating plainly:

  - E < p (16 summary dimensions against 27 parameters) is an UNDERCOMPLETE
    summary space. Schmitt et al.'s Experiment 1 found a minimal summary
    space detected prior misspecification but NOT likelihood or noise
    misspecification, while an overcomplete one detected both. Expect
    blindness to simulator and noise misspecification here.
  - z is L2-normalised, so it discards ||zraw||, the amplitude. The
    exporter's own Section 2.1 documents a 9x amplitude discrepancy between
    the two arms (per-electrode mean vs sum over n_e = 9). That class of
    mismatch is invisible in z and visible in zraw. Run both spaces; the
    exporter sets want_zraw=True by default and records zraw_available in
    the sidecar.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from npe_diagnostics import FamilyResult, holm_bonferroni
from npe_local import benjamini_hochberg

__all__ = [
    "bandwidth_grid",
    "mmd2_multiscale",
    "_self_term",
    "GateResult",
    "misspecification_gate",
    "minimum_detectable_shift",
    "groups_from_table",
    "run_spaces",
]


# ---------------------------------------------------------------------------
# Kernel and statistic
# ---------------------------------------------------------------------------

def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(A ** 2, axis=1)[:, None]
                      + np.sum(B ** 2, axis=1)[None, :]
                      - 2.0 * A @ B.T, 0.0)


def bandwidth_grid(A: np.ndarray, B: np.ndarray,
                   factors: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
                   rng: Optional[np.random.Generator] = None,
                   max_n: int = 1000) -> np.ndarray:
    """Median-heuristic bandwidth times a fixed set of factors.

    Schmitt et al. use a SUM of Gaussian kernels at several widths rather
    than one, because a single median-heuristic bandwidth is tuned to the
    bulk of the data and is insensitive to discrepancies at other scales.
    They report that inverse multiquadric kernels give essentially equal
    results, so only the Gaussian family is implemented here.

    Returned bandwidths are in the units of the embedding space. On
    S^{E-1} all pairwise distances lie in [0, 2], so the grid is bounded
    above by 2 * max(factors) whatever the data.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    P = np.concatenate([np.atleast_2d(A), np.atleast_2d(B)], axis=0)
    if P.shape[0] > max_n:
        P = P[rng.choice(P.shape[0], max_n, replace=False)]
    iu = np.triu_indices(P.shape[0], k=1)
    if iu[0].size == 0:
        med = 1.0
    else:
        med = float(np.median(np.sqrt(_sqdist(P, P)[iu])))
    if not np.isfinite(med) or med <= 1e-12:
        med = 1.0
    return np.asarray([med * f for f in factors], dtype=np.float64)


def _self_term(A: np.ndarray, bandwidths: Sequence[float]) -> np.ndarray:
    """mean_{a,a'} k(a,a') per bandwidth, for a fixed sample A.

    Split out because the reference sample is FIXED across every null
    replicate: recomputing an n_ref x n_ref kernel matrix inside the loop
    costs O(B n_ref^2) for a quantity that never changes.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    d = _sqdist(A, A)
    return np.asarray([np.exp(-d / (2.0 * float(bw) ** 2)).mean()
                       for bw in bandwidths], dtype=np.float64)


def mmd2_multiscale(A: np.ndarray, B: np.ndarray,
                    bandwidths: Sequence[float],
                    kaa: Optional[np.ndarray] = None) -> float:
    """Biased squared MMD, equation (1), with a sum of Gaussian kernels.

    The BIASED estimator is used, matching embedding_overlap: the unbiased
    version is undefined for a single observation, and one real recording is
    a case this gate must handle.

    Biasedness matters less than it sounds here because the p-value comes
    from a resampling null computed with the SAME estimator on the SAME
    sample sizes, so the bias is common to observation and null and cancels
    in the comparison. It does mean the reported MMD is not comparable
    across different sample sizes.

    Parameters
    ----------
    kaa : ndarray or None
        Precomputed _self_term(A, bandwidths). Pass it when A is held fixed
        across many calls; the result is identical either way.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    if kaa is None:
        kaa = _self_term(A, bandwidths)
    dbb, dab = _sqdist(B, B), _sqdist(A, B)
    total = 0.0
    for i, bw in enumerate(bandwidths):
        g = 1.0 / (2.0 * float(bw) ** 2)
        total += (kaa[i] + np.exp(-g * dbb).mean()
                  - 2.0 * np.exp(-g * dab).mean())
    return float(total)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Verdict for one embedding space and one subset of the real cohort."""
    space: str                      # "z" or "zraw"
    label: str                      # "pooled", or the class name
    n_real_rows: int
    n_groups: int
    n_sim: int
    mmd_group: float                # median observed MMD over window choices
    p_group: float                  # median p-value over window choices
    reject_fraction: float          # fraction of window choices that reject
    mmd_iid: float                  # all real rows, i.i.d. null
    p_iid: float
    per_group_names: List[str] = field(default_factory=list)
    per_group_p: np.ndarray = field(default_factory=lambda: np.empty(0))
    alpha: float = 0.05
    notes: List[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """The GROUP-aware verdict. p_iid is reported, never used to decide."""
        return self.p_group >= self.alpha

    def family(self) -> Tuple[FamilyResult, FamilyResult]:
        """Per-recording verdicts under FWER (Holm) and FDR (BH)."""
        if self.per_group_p.size == 0:
            raise ValueError("no per-recording p-values were computed")
        return (holm_bonferroni(self.per_group_p, alpha=self.alpha,
                                names=self.per_group_names),
                benjamini_hochberg(self.per_group_p, alpha=self.alpha,
                                   names=self.per_group_names))

    def summary(self, top: int = 5) -> str:
        lines = [
            "  [%s / %s]  %d rows in %d recordings vs %d simulated"
            % (self.space, self.label, self.n_real_rows, self.n_groups,
               self.n_sim),
            "    group-aware : MMD=%.5f  p=%.4f  (%.0f%% of window choices "
            "reject)  -> %s"
            % (self.mmd_group, self.p_group, 100 * self.reject_fraction,
               "PASS" if self.passes else "FAIL"),
            "    i.i.d. null : MMD=%.5f  p=%.4f  (reported only; invalid "
            "under grouping)" % (self.mmd_iid, self.p_iid),
        ]
        if self.per_group_p.size:
            fwer, fdr = self.family()
            lines.append("    per-recording: %d/%d reject (Holm), %d/%d (BH)"
                         % (fwer.n_rejected, len(fwer.names),
                            fdr.n_rejected, len(fdr.names)))
            worst = np.argsort(self.per_group_p)[:top]
            flagged = [self.per_group_names[i] for i in worst
                       if self.per_group_p[i] < self.alpha]
            if flagged:
                lines.append("    worst: " + ", ".join(flagged))
        for n in self.notes:
            lines.append("    " + n)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def _pvalue(observed: float, null: np.ndarray) -> float:
    """(#{null >= observed} + 1) / (B + 1), the convention used repo-wide."""
    return float((np.sum(null >= observed) + 1.0) / (null.size + 1.0))


def _one_gate(z_sim: np.ndarray, z_real: np.ndarray, groups: np.ndarray,
              space: str, label: str, n_null: int, n_window_choices: int,
              alpha: float, rng: np.random.Generator,
              per_group: bool, n_ref_max: int) -> GateResult:
    n_real, n_sim = z_real.shape[0], z_sim.shape[0]
    uniq = list(dict.fromkeys(groups.tolist()))       # stable order
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    n_groups = len(uniq)
    notes: List[str] = []

    bw = bandwidth_grid(z_sim, z_real, rng=rng)

    # Disjoint reference and null pool: reusing the same simulated points on
    # both sides biases the null low and makes the gate anticonservative.
    # The reference is CAPPED: it only has to represent the simulated
    # distribution, and every extra point costs O(n_ref) in each of the
    # B x (choices) kernel evaluations below for no gain in resolution.
    perm = rng.permutation(n_sim)
    n_ref = min(n_sim // 2, n_ref_max)
    reference = z_sim[perm[:n_ref]]
    pool = z_sim[perm[n_ref:]]
    kaa = _self_term(reference, bw)
    if pool.shape[0] < n_groups:
        notes.append("NOTE: simulated pool smaller than the number of "
                     "recordings; the null is drawn with replacement")

    # ---- group-aware: R independent single-window points -----------------
    null_g = np.empty(n_null, dtype=np.float64)
    for b in range(n_null):
        take = rng.choice(pool.shape[0], size=n_groups,
                          replace=pool.shape[0] < n_groups)
        null_g[b] = mmd2_multiscale(reference, pool[take], bw, kaa)

    obs, pvals = [], []
    for _ in range(n_window_choices):
        pick = np.array([g[rng.integers(0, g.size)] for g in idx_by_group])
        m = mmd2_multiscale(reference, z_real[pick], bw, kaa)
        obs.append(m)
        pvals.append(_pvalue(m, null_g))
    obs = np.asarray(obs)
    pvals = np.asarray(pvals)

    # ---- i.i.d. null on all rows, for comparison only --------------------
    null_i = np.empty(n_null, dtype=np.float64)
    for b in range(n_null):
        take = rng.choice(pool.shape[0], size=n_real,
                          replace=pool.shape[0] < n_real)
        null_i[b] = mmd2_multiscale(reference, pool[take], bw, kaa)
    mmd_iid = mmd2_multiscale(reference, z_real, bw, kaa)
    p_iid = _pvalue(mmd_iid, null_i)

    # ---- per-recording scores -------------------------------------------
    # ONE window against the reference, not all W of them. Feeding a
    # recording's whole window set to a null built from W INDEPENDENT
    # simulated draws is the same error the pooled statistic corrects,
    # scaled down: the windows are tightly clustered, their self-term
    # mean_{b,b'} k(b,b') is far larger than for W independent points, and
    # the score fires on every recording including the healthy ones.
    # With a single point the statistic reduces to a monotone function of
    # mean_a k(a, b) -- a distance from that window to the simulated cloud,
    # which is exactly the per-recording quantity of interest.
    names: List[str] = []
    pg = np.empty(0)
    if per_group:
        null_w = np.empty(n_null, dtype=np.float64)
        for b in range(n_null):
            take = rng.integers(0, pool.shape[0], size=1)
            null_w[b] = mmd2_multiscale(reference, pool[take], bw, kaa)
        pg = np.empty(n_groups, dtype=np.float64)
        for i, g in enumerate(idx_by_group):
            ps = [_pvalue(mmd2_multiscale(reference,
                                          z_real[g[rng.integers(0, g.size)]][None, :],
                                          bw, kaa), null_w)
                  for _ in range(n_window_choices)]
            pg[i] = float(np.median(ps))
        names = [str(u) for u in uniq]

    if n_sim < 4 * n_real:
        notes.append("NOTE: n_sim < 4*n_real; the null is coarse")

    return GateResult(
        space=space, label=label, n_real_rows=n_real, n_groups=n_groups,
        n_sim=n_sim, mmd_group=float(np.median(obs)),
        p_group=float(np.median(pvals)),
        reject_fraction=float(np.mean(pvals < alpha)),
        mmd_iid=mmd_iid, p_iid=p_iid, per_group_names=names, per_group_p=pg,
        alpha=alpha, notes=notes)


def misspecification_gate(z_sim: np.ndarray,
                          z_real: np.ndarray,
                          groups: Sequence,
                          classes: Optional[Sequence] = None,
                          space: str = "z",
                          n_null: int = 500,
                          n_window_choices: int = 20,
                          alpha: float = 0.05,
                          seed: int = 0,
                          per_group: bool = True,
                          n_ref_max: int = 800,
                          only_class: Optional[str] = None
                          ) -> Dict[str, GateResult]:
    """Run the gate pooled and, when classes are given, per class.

    Parameters
    ----------
    z_sim : (n_sim, E)
        Simulated embeddings. One row per simulation at T_win = 180 s.
    z_real : (n_real, E)
        Real embeddings, several rows per recording.
    groups : sequence of length n_real
        Recording identifier per row. THE ONE THING THAT MAKES THIS GATE
        VALID -- pass the column your real export uses, whatever you named
        it (see groups_from_table).
    classes : sequence of length n_real, or None
        Condition per row, e.g. control / pathological.
    space : str
        Label only, "z" or "zraw". Nothing branches on it.
    n_null : int
        Resampling replicates for each null.
    n_window_choices : int
        How many one-window-per-recording draws to average the verdict over.
    alpha, seed, per_group : as usual.
    only_class : str or None
        Restrict the whole analysis to one class. The pooled entry is then
        that class alone and is labelled accordingly.

    Returns
    -------
    dict mapping label -> GateResult, always containing "pooled", plus one
    entry per class when classes is given.

    Notes
    -----
    Per-class results are computed EVEN WHEN pooled is requested. A pooled
    gate can pass while one arm sits outside the simulated cloud, and that
    arm is exactly the one worth knowing about.
    """
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    z_real = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    if z_sim.shape[1] != z_real.shape[1]:
        raise ValueError("embedding dims differ: sim %d vs real %d"
                         % (z_sim.shape[1], z_real.shape[1]))
    groups = np.asarray(groups)
    if groups.shape[0] != z_real.shape[0]:
        raise ValueError("groups has length %d but z_real has %d rows"
                         % (groups.shape[0], z_real.shape[0]))
    if classes is not None:
        classes = np.asarray(classes)
        if classes.shape[0] != z_real.shape[0]:
            raise ValueError("classes has length %d but z_real has %d rows"
                             % (classes.shape[0], z_real.shape[0]))
        # A recording must not straddle two classes, or the per-class split
        # would break the group structure the gate depends on.
        for g in np.unique(groups):
            if np.unique(classes[groups == g]).size > 1:
                raise ValueError("recording %r carries more than one class "
                                 "label" % (g,))

    rng = np.random.default_rng(seed)
    out: Dict[str, GateResult] = {}

    if only_class is not None:
        if classes is None:
            raise ValueError("only_class given but classes is None")
        m = classes == only_class
        if not np.any(m):
            raise ValueError("no rows with class %r" % (only_class,))
        out["pooled"] = _one_gate(z_sim, z_real[m], groups[m], space,
                                  "only:%s" % only_class, n_null,
                                  n_window_choices, alpha, rng, per_group,
                                  n_ref_max)
        return out

    out["pooled"] = _one_gate(z_sim, z_real, groups, space, "pooled", n_null,
                              n_window_choices, alpha, rng, per_group,
                              n_ref_max)
    if classes is not None:
        for c in dict.fromkeys(classes.tolist()):
            m = classes == c
            out[str(c)] = _one_gate(z_sim, z_real[m], groups[m], space,
                                    str(c), n_null, n_window_choices, alpha,
                                    rng, per_group, n_ref_max)
    return out


# ---------------------------------------------------------------------------
# How much misspecification would this gate have caught?
# ---------------------------------------------------------------------------

def minimum_detectable_shift(z_sim: np.ndarray,
                             n_groups: int,
                             deltas: Sequence[float] = (0.02, 0.05, 0.1,
                                                        0.2, 0.4),
                             power: float = 0.8,
                             n_null: int = 300,
                             n_rep: int = 40,
                             alpha: float = 0.05,
                             seed: int = 0,
                             normalise: bool = True) -> Tuple[Optional[float],
                                                              np.ndarray]:
    """Smallest rigid shift the gate detects with the requested power.

    A pass is only as meaningful as the gate's power, and power here is set
    by the number of RECORDINGS, not the number of windows. This answers
    "what would we have caught?" in the units of the embedding space, by
    displacing simulated points along a fixed random direction and measuring
    the rejection rate.

    Parameters
    ----------
    z_sim : (n_sim, E)
    n_groups : int
        Number of independent real units, i.e. recordings. Pass the real R.
    deltas : sequence of float
        Shift magnitudes to try, ascending.
    power : float
        Target rejection rate.
    normalise : bool
        Re-project shifted points onto the unit sphere. Leave True when the
        embeddings are L2-normalised (z), False for zraw.

    Returns
    -------
    (mde, rates) : the smallest delta reaching the target power, or None if
    none did, and the rejection rate at each delta.
    """
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    rng = np.random.default_rng(seed)
    n_sim = z_sim.shape[0]

    perm = rng.permutation(n_sim)
    n_ref = min(n_sim // 2, 800)
    reference = z_sim[perm[:n_ref]]
    pool = z_sim[perm[n_ref:]]
    bw = bandwidth_grid(reference, pool, rng=rng)
    kaa = _self_term(reference, bw)

    null = np.empty(n_null, dtype=np.float64)
    for b in range(n_null):
        take = rng.choice(pool.shape[0], size=n_groups,
                          replace=pool.shape[0] < n_groups)
        null[b] = mmd2_multiscale(reference, pool[take], bw, kaa)

    direction = rng.normal(size=z_sim.shape[1])
    direction /= np.linalg.norm(direction)

    rates = np.empty(len(deltas), dtype=np.float64)
    for i, d in enumerate(deltas):
        hits = 0
        for _ in range(n_rep):
            take = rng.choice(pool.shape[0], size=n_groups,
                              replace=pool.shape[0] < n_groups)
            shifted = pool[take] + d * direction[None, :]
            if normalise:
                nrm = np.linalg.norm(shifted, axis=1, keepdims=True)
                shifted = shifted / np.maximum(nrm, 1e-12)
            if _pvalue(mmd2_multiscale(reference, shifted, bw, kaa),
                       null) < alpha:
                hits += 1
        rates[i] = hits / float(n_rep)

    ok = np.flatnonzero(rates >= power)
    return (float(deltas[ok[0]]) if ok.size else None), rates


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def groups_from_table(table, group_col: str,
                      class_col: Optional[str] = None):
    """Pull the recording and class vectors out of a loaded real shard.

    `table` is anything with a `columns` attribute and `[]` access -- a
    pandas DataFrame is the expected case. The column NAMES are yours: the
    exporter carries through whatever ident_columns the real-recording
    trace source passes, so this function validates rather than guesses,
    and names what is actually present when it fails.
    """
    cols = list(getattr(table, "columns", []))
    missing = [c for c in [group_col] + ([class_col] if class_col else [])
               if c not in cols]
    if missing:
        raise KeyError("column(s) %s not in the shard; available: %s"
                       % (missing, cols))
    groups = np.asarray(table[group_col])
    classes = np.asarray(table[class_col]) if class_col else None
    return groups, classes


def run_spaces(spaces: Dict[str, Tuple[np.ndarray, np.ndarray]],
               groups: Sequence, classes: Optional[Sequence] = None,
               **kwargs) -> Dict[str, Dict[str, GateResult]]:
    """Run the gate over several embedding spaces, e.g. {"z":..., "zraw":...}.

    Disagreement between z and zraw is informative and is NOT reconciled
    here: z carries direction only, zraw direction and magnitude, so zraw
    firing alone points at an amplitude mismatch that L2 normalisation hides.
    """
    return {name: misspecification_gate(zs, zr, groups, classes=classes,
                                        space=name, **kwargs)
            for name, (zs, zr) in spaces.items()}
