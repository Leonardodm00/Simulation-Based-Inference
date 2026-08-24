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
    "WitnessResult",
    "witness_function",
    "witness_maps",
    "witness_heatmaps",
    "witness_slices",
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


# ---------------------------------------------------------------------------
# Witness function: from a verdict to a pointer
# ---------------------------------------------------------------------------
#
# The gate above reduces the sim/real comparison to one scalar per space,
# equation (1). That scalar is blind by construction to WHERE the two clouds
# disagree: a bulk displacement of the whole real cohort (a simulation gap,
# Schmitt et al.'s Mode 1) and a handful of real recordings stranded outside
# the simulated support (rare-but-valid events, their Mode 2) can produce
# the same MMD^2. In particular, when the simulated embedding distribution
# is MULTIMODAL, the median-heuristic bandwidth is dominated by
# between-mode distances and the scalar fires even when the real cohort
# sits comfortably inside one of the modes.
#
# The witness function makes the disagreement local. For the sum-of-Gaussian
# kernel k = sum_s k_sigma_s used throughout this module, define, for each
# evaluation point z on S^{E-1},
#
#     g(z) = mu_sim(z) - mu_real(z)
#          = (1/n_fit) sum_i k(z_i_sim, z) - (1/n_real) sum_j k(z_j_real, z)
#
# i.e. the (kernel-smoothed) simulated local density minus the real local
# density at z. g(z) >> 0 where simulations concentrate and real data do
# not; g(z) << 0 where real recordings live and the simulator never goes.
# Evaluating g on the two samples themselves gives two 1-D score sets
#
#     u_i = g(z_i_sim),   v_j = g(z_j_real),
#
# and, when g is estimated from the SAME simulated sample it is evaluated
# on (split=False below), the exact algebraic identity
#
#     mean(u) - mean(v) = MMD^2  of equation (1), biased estimator,       (2)
#
# holds per bandwidth and, by linearity of g in the kernel, for the
# sum over bandwidths -- the exact statistic the gate thresholds. So the
# gate's scalar is literally a difference of the means of two histograms
# this module was already in a position to compute: the row and column
# means of the Gram blocks. The histograms, and the map of g over a 2-D
# projection of the embedding, are what distinguish Mode 1 (bulk of {v_j}
# displaced from {u_i}) from Mode 2 (bulks superimposed, a few v_j
# stranded far negative).
#
# MULTI-BANDWIDTH. Because g is LINEAR in the kernel, the witness of the
# summed kernel is the sum of the per-bandwidth witnesses:
#
#     g_total(z) = sum_s g_sigma_s(z),                                    (3)
#
# so the multi-scale construction of bandwidth_grid()/mmd2_multiscale()
# carries over verbatim: identity (2) applied to (3) recovers exactly
# mmd2_multiscale's output. The sum is what the gate decides on, but it
# COLLAPSES scale information -- a positive small-sigma spike can cancel a
# negative large-sigma trough at the same z. Both per-bandwidth maps and
# the summed map are therefore produced, and a Mode-1/Mode-2 reading
# should only be trusted when it is stable across bandwidths.
#
# TWO CAVEATS, inherited from the estimation setup:
#   (i) With split=True (default), mu_sim is estimated on a simulated
#       half DISJOINT from the evaluated points, so "these points are
#       anomalous" is read off held-out data. The real side is small and
#       is not split: each v_j then contains its own self-term
#       k(z_j, z_j)/n_real = (S bandwidths) x 1/n_real -- a CONSTANT
#       across j at each bandwidth, hence a pure offset that moves the
#       {v_j} histogram bodily and distorts nothing. With split=False the
#       identity (2) is exact and is what the smoke test asserts.
#  (ii) Any post-hoc anomaly claim about specific recordings remains
#       exploratory: the permutation p-value of the gate stays valid, the
#       witness map does not add a test, it adds a pointer.

@dataclass
class WitnessResult:
    """Witness scores of sim/real embedding clouds, per bandwidth and summed.

    u[s, i] is g_sigma_s at the i-th EVALUATED simulated point, v[s, j] at
    the j-th real point; u_sum/v_sum are the bandwidth sums, equation (3).
    """
    space: str
    bandwidths: np.ndarray            # (S,)
    u: np.ndarray                     # (S, n_sim_eval)
    v: np.ndarray                     # (S, n_real)
    u_sum: np.ndarray                 # (n_sim_eval,)
    v_sum: np.ndarray                 # (n_real,)
    fit_idx: np.ndarray               # rows of z_sim defining mu_sim
    eval_idx: np.ndarray              # rows of z_sim that were scored
    split: bool
    notes: List[str] = field(default_factory=list)

    def mmd2_from_witness(self) -> float:
        """mean(u_sum) - mean(v_sum): equals equation (1) when split=False."""
        return float(self.u_sum.mean() - self.v_sum.mean())


def witness_function(z_sim: np.ndarray, z_real: np.ndarray,
                     bandwidths: Optional[Sequence[float]] = None,
                     space: str = "z",
                     split: bool = True,
                     seed: int = 0,
                     n_fit_max: int = 2000,
                     n_eval_max: int = 2000) -> WitnessResult:
    """Evaluate the unnormalised witness g on both samples, per bandwidth.

    Parameters
    ----------
    z_sim : (n_sim, E)
        Simulated embeddings.
    z_real : (n_real, E)
        Real embeddings. All rows are used; with several windows per
        recording the map shows windows, and the caller can aggregate by
        the groups vector afterwards.
    bandwidths : sequence of float, or None
        None -> bandwidth_grid(z_sim, z_real), the same multi-scale grid
        the gate uses, so the summed map matches the gate's statistic.
    split : bool
        True  -> mu_sim estimated on one simulated half, g evaluated on
                 the other half (held-out scores; identity (2) then holds
                 only in expectation).
        False -> fit set = eval set = all of z_sim; identity (2) exact.
    n_fit_max, n_eval_max : int
        Caps on the two simulated sets, for memory: the Gram blocks are
        O(n_fit x n_eval) and O(n_fit x n_real).

    Returns
    -------
    WitnessResult
    """
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    z_real = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    if z_sim.shape[1] != z_real.shape[1]:
        raise ValueError("embedding dims differ: sim %d vs real %d"
                         % (z_sim.shape[1], z_real.shape[1]))
    rng = np.random.default_rng(seed)
    n_sim = z_sim.shape[0]
    notes: List[str] = []

    if bandwidths is None:
        bandwidths = bandwidth_grid(z_sim, z_real, rng=rng)
    bw = np.asarray(bandwidths, dtype=np.float64)

    if split and n_sim < 4:
        raise ValueError("split=True needs n_sim >= 4; got %d" % n_sim)
    if split:
        perm = rng.permutation(n_sim)
        half = n_sim // 2
        fit_idx = np.sort(perm[:min(half, n_fit_max)])
        eval_idx = np.sort(perm[half:half + min(n_sim - half, n_eval_max)])
        notes.append("split=True: mu_sim from %d held-out simulated points; "
                     "identity (2) holds in expectation only"
                     % fit_idx.size)
    else:
        fit_idx = np.arange(n_sim)
        eval_idx = np.arange(n_sim)
        if n_sim > n_fit_max:
            notes.append("NOTE: split=False ignores n_fit_max; the full "
                         "Gram blocks are built")
        notes.append("split=False: identity (2) exact, scores are in-sample")

    fit, ev = z_sim[fit_idx], z_sim[eval_idx]
    d_fe = _sqdist(fit, ev)          # (n_fit, n_eval)
    d_re = _sqdist(z_real, ev)       # (n_real, n_eval)
    d_fr = _sqdist(fit, z_real)      # (n_fit, n_real)
    d_rr = _sqdist(z_real, z_real)   # (n_real, n_real)

    S = bw.size
    u = np.empty((S, ev.shape[0]), dtype=np.float64)
    v = np.empty((S, z_real.shape[0]), dtype=np.float64)
    for s in range(S):
        g = 1.0 / (2.0 * float(bw[s]) ** 2)
        u[s] = np.exp(-g * d_fe).mean(axis=0) - np.exp(-g * d_re).mean(axis=0)
        v[s] = np.exp(-g * d_fr).mean(axis=0) - np.exp(-g * d_rr).mean(axis=0)
    notes.append("each v_j includes its own self-term: a constant offset "
                 "of 1/n_real per bandwidth, shape-preserving")

    return WitnessResult(space=space, bandwidths=bw, u=u, v=v,
                         u_sum=u.sum(axis=0), v_sum=v.sum(axis=0),
                         fit_idx=fit_idx, eval_idx=eval_idx, split=split,
                         notes=notes)


def _project_2d(P: np.ndarray, method: str, seed: int,
                n_sim_pts: Optional[int] = None):
    """2-D layout of the pooled point cloud P.

    Returns (coords, axis_label, lift, var_frac), where

      lift : callable (m, 2) -> (m, E), or None
          The exact right inverse of the projection restricted to the
          2-plane, available for LINEAR layouts only (PCA). Used by
          witness_heatmaps to evaluate the witness at grid locations
          instead of interpolating it. t-SNE is not a linear map and has
          no such inverse, hence None.
      var_frac : float
          Fraction of the total variance of P carried by the 2 plotted
          directions; 1.0 - var_frac is the variance the plane discards
          and is exactly the caveat attached to a lifted heatmap. NaN for
          t-SNE, which has no such decomposition.

    PCA is done by plain SVD (no sklearn needed); t-SNE needs sklearn and
    raises ImportError with a clear message when it is missing. The
    projection provides LAYOUT ONLY: the witness values attached to the
    sampled points are computed in the full embedding space, never in the
    projected one, so t-SNE's distance distortions recolour nothing.

    METHODS

      "pca"       the variance-optimal 2-plane of the POOLED cloud. It
                  minimises reconstruction error, which is a statement
                  about where the points are, NOT about where g varies:
                  a small systematic sim/real offset lying in a
                  low-variance direction is precisely what PCA discards
                  first, and with a multimodal simulator the leading PC
                  is the between-mode axis, i.e. the nuisance structure.
      "contrast"  axis 1 is the normalised difference of sample means
                  (z_sim_bar - z_real_bar), so ANY mean displacement
                  between the two clouds is visible by construction;
                  axis 2 is the leading principal direction of the pooled
                  cloud ORTHOGONAL to axis 1, so the frame stays
                  orthonormal and liftable. Requires n_sim_pts: the
                  number of leading rows of P that are simulated. Falls
                  back to PCA, with a note in the axis label, when the
                  two means coincide to numerical precision.
                  Expect var_frac BELOW the PCA value -- PCA maximises
                  it by definition. That is not a defect: the question a
                  witness map answers is how much DISCREPANCY the plane
                  carries, not how much variance, and those are different
                  optimisation problems.
      "tsne"      non-linear, no lift, no variance decomposition.
    """
    if method == "pca":
        mean = P.mean(axis=0, keepdims=True)
        C = P - mean
        U, sv, Vt = np.linalg.svd(C, full_matrices=False)
        Y = U[:, :2] * sv[:2]
        tot = float(np.sum(sv ** 2))
        ev = (sv[:2] ** 2 / tot * 100.0) if tot > 0 else np.zeros(2)
        basis = Vt[:2]                       # (2, E), orthonormal rows

        def lift(Q: np.ndarray) -> np.ndarray:
            """(m, 2) -> (m, E): the point of the 2-plane through the data
            mean whose projection is Q. The E-2 discarded coordinates are
            set to the mean's, so this is a SLICE, not a marginal."""
            return mean + np.asarray(Q, dtype=np.float64) @ basis

        return (Y, "PCA (%.0f%% + %.0f%% var)" % (ev[0], ev[1]), lift,
                float(np.sum(sv[:2] ** 2) / tot) if tot > 0 else 0.0)
    if method == "contrast":
        if n_sim_pts is None:
            raise ValueError("method='contrast' needs n_sim_pts (how many "
                             "leading rows of P are simulated)")
        n_sim_pts = int(n_sim_pts)
        if not 0 < n_sim_pts < P.shape[0]:
            raise ValueError("n_sim_pts=%d out of range for %d pooled rows"
                             % (n_sim_pts, P.shape[0]))
        mean = P.mean(axis=0, keepdims=True)
        C = P - mean
        tot = float(np.sum(C ** 2))
        delta = P[:n_sim_pts].mean(axis=0) - P[n_sim_pts:].mean(axis=0)
        nd = float(np.linalg.norm(delta))
        if nd <= 1e-12:
            Y, lab, lift, vf = _project_2d(P, "pca", seed)
            return Y, lab + " [contrast fell back: means coincide]", lift, vf
        a1 = delta / nd
        # Deflate a1 out of the cloud, then take the leading direction of
        # what is left: guarantees an orthonormal frame, hence an exact
        # lift, while spending axis 1 on the discrepancy.
        C_perp = C - np.outer(C @ a1, a1)
        _U2, sv2, Vt2 = np.linalg.svd(C_perp, full_matrices=False)
        a2 = Vt2[0]
        basis = np.stack([a1, a2], axis=0)          # (2, E), orthonormal
        Y = C @ basis.T
        vf = float(np.sum(Y ** 2) / tot) if tot > 0 else 0.0

        def lift(Q: np.ndarray) -> np.ndarray:
            """(m, 2) -> (m, E) on the contrast plane through the mean."""
            return mean + np.asarray(Q, dtype=np.float64) @ basis

        return (Y, "contrast: mean-diff axis + top orth. PC (%.0f%% var)"
                % (100.0 * vf), lift, vf)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError("t-SNE needs scikit-learn; pip install "
                              "scikit-learn or drop 'tsne' from methods"
                              ) from e
        perp = max(5.0, min(30.0, (P.shape[0] - 1) / 3.0))
        Y = TSNE(n_components=2, perplexity=perp, init="pca",
                 random_state=seed).fit_transform(P)
        return (np.asarray(Y, dtype=np.float64),
                "t-SNE (perplexity %.0f)" % perp, None, float("nan"))
    raise ValueError("unknown method %r; use 'pca', 'contrast' or 'tsne'"
                     % (method,))


def witness_maps(z_sim: np.ndarray, z_real: np.ndarray,
                 outdir: str,
                 space: str = "z",
                 bandwidths: Optional[Sequence[float]] = None,
                 split: bool = True,
                 methods: Sequence[str] = ("pca", "tsne"),
                 groups: Optional[Sequence] = None,
                 seed: int = 0,
                 max_points: int = 2000,
                 n_label: int = 3,
                 dpi: int = 150) -> Dict[str, str]:
    """Witness maps (PCA and t-SNE layouts) and witness histograms, saved.

    Produces, in `outdir`:
      witness_map_<space>_<method>.png   one panel per bandwidth + 'sum',
                                         points coloured by g (diverging
                                         map, red = sim-dense, blue =
                                         real-dense), sims as dots, real
                                         recordings as triangles;
      witness_hist_<space>.png           the {u_i} vs {v_j} histograms per
                                         bandwidth + summed: the Mode-1
                                         (bulk shift) vs Mode-2 (tail
                                         excess) readout;
      witness_<space>.npz                coords, scores and indices, so
                                         plots can be regenerated without
                                         recomputing kernels.

    Returns a dict name -> saved path. Matplotlib is imported lazily with
    the Agg backend, so this runs on a display-less cluster node.

    Parameters beyond witness_function's: `groups` (length n_real) labels
    the `n_label` most negative real points -- the recordings sitting
    deepest in the simulation gap; `max_points` caps the simulated points
    that are PLOTTED (scores are computed on all evaluated points; the cap
    only thins the scatter and the t-SNE input).
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    res = witness_function(z_sim, z_real, bandwidths=bandwidths, space=space,
                           split=split, seed=seed)
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    z_real = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    ev = z_sim[res.eval_idx]

    keep = np.arange(ev.shape[0])
    if ev.shape[0] > max_points:
        keep = np.sort(rng.choice(ev.shape[0], max_points, replace=False))
    P = np.concatenate([ev[keep], z_real], axis=0)
    n_s = keep.size

    S = res.bandwidths.size
    panels = [("sigma=%.3g" % res.bandwidths[s], res.u[s][keep], res.v[s])
              for s in range(S)] + [("sum over bandwidths",
                                     res.u_sum[keep], res.v_sum)]

    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != z_real.shape[0]:
            raise ValueError("groups has length %d but z_real has %d rows"
                             % (groups.shape[0], z_real.shape[0]))
    worst = np.argsort(res.v_sum)[:max(0, int(n_label))]

    saved: Dict[str, str] = {}
    coords: Dict[str, np.ndarray] = {}
    for method in methods:
        try:
            Y, axlabel, _lift, _vf = _project_2d(P, method, seed,
                                                 n_sim_pts=n_s)
        except ImportError as e:
            res.notes.append("SKIPPED %s: %s" % (method, e))
            continue
        coords[method] = Y
        ncol = len(panels)
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.6),
                                 squeeze=False)
        for ax, (title, us, vs) in zip(axes[0], panels):
            allv = np.concatenate([us, vs])
            vmax = float(np.percentile(np.abs(allv), 99.5))
            vmax = vmax if vmax > 0 else 1e-12
            sc = ax.scatter(Y[:n_s, 0], Y[:n_s, 1], c=us, cmap="coolwarm",
                            vmin=-vmax, vmax=vmax, s=8, alpha=0.55,
                            linewidths=0, label="sim")
            ax.scatter(Y[n_s:, 0], Y[n_s:, 1], c=vs, cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, s=52, marker="^",
                       edgecolors="black", linewidths=0.6, label="real")
            if groups is not None:
                for j in worst:
                    ax.annotate(str(groups[j]), (Y[n_s + j, 0], Y[n_s + j, 1]),
                                fontsize=7, xytext=(3, 3),
                                textcoords="offset points")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(axlabel, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cb.set_label("g(z) = mu_sim(z) - mu_real(z)")
        axes[0][0].legend(loc="upper left", fontsize=8)
        fig.suptitle("Witness map [%s, %s]  (colour scale per panel; "
                     "red: sim-dense, blue: real-dense)" % (space, method),
                     y=1.04)
        path = os.path.join(outdir, "witness_map_%s_%s.png" % (space, method))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["map_%s" % method] = path

    # ---- histograms: the Mode-1 vs Mode-2 readout -------------------------
    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6),
                             squeeze=False)
    for ax, (title, us, vs) in zip(axes[0], panels):
        lo = float(min(us.min(), vs.min()))
        hi = float(max(us.max(), vs.max()))
        if hi <= lo:
            hi = lo + 1e-12
        bins = np.linspace(lo, hi, 40)
        ax.hist(us, bins=bins, density=True, alpha=0.55, label="u (sim)")
        ax.hist(vs, bins=bins, density=True, alpha=0.55, label="v (real)")
        ax.axvline(float(us.mean()), color="C0", ls="--", lw=1)
        ax.axvline(float(vs.mean()), color="C1", ls="--", lw=1)
        ax.set_title("%s\nmean(u)-mean(v)=%.3e"
                     % (title, float(us.mean() - vs.mean())), fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Witness histograms [%s]: bulk shift -> Mode 1 "
                 "(simulation gap); superposed bulks + stranded v_j -> "
                 "Mode 2 (rare-but-valid)" % space, y=1.06)
    hpath = os.path.join(outdir, "witness_hist_%s.png" % space)
    fig.savefig(hpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved["hist"] = hpath

    npz = {"bandwidths": res.bandwidths, "u": res.u, "v": res.v,
           "u_sum": res.u_sum, "v_sum": res.v_sum, "fit_idx": res.fit_idx,
           "eval_idx": res.eval_idx, "plotted_sim_idx": res.eval_idx[keep],
           "split": np.asarray(res.split)}
    for m, Y in coords.items():
        npz["coords_%s" % m] = Y
    apath = os.path.join(outdir, "witness_%s.npz" % space)
    np.savez_compressed(apath, **npz)
    saved["arrays"] = apath
    return saved


# ---------------------------------------------------------------------------
# Witness HEATMAPS: the field, not just the samples
# ---------------------------------------------------------------------------
#
# witness_maps() colours the SAMPLED points by g. A heatmap instead shows g
# as a field over the whole 2-D panel. The scientific content of that step
# is entirely in how the field at a grid location y in R^2 is obtained,
# because g is defined on the E-dimensional embedding space and a 2-D grid
# location is not a point of that space. Three constructions are provided
# and they answer three DIFFERENT questions; the mode is always written
# into the file name and the panel title so a saved figure can never be
# misread later.
#
#   field="lift"  (linear layouts only, i.e. PCA)
#       The 2-D grid is lifted back into the embedding space by the exact
#       right inverse of the projection: y -> mean + y B, with B the (2, E)
#       orthonormal PCA basis. The witness is then EVALUATED there, with
#       real kernel evaluations against the same fit set the scatter uses,
#       so the background and the overlaid points are on one scale and no
#       interpolation happens anywhere.
#       What it shows: g restricted to a 2-PLANE SLICE through the data
#       mean. The E-2 discarded coordinates are held at the mean's values,
#       so if the plane carries little variance the slice can be far from
#       where the data actually live. The variance fraction is printed on
#       the figure for exactly this reason. This is a slice, NOT a
#       marginal and NOT an average over the discarded directions.
#
#   field="proj"  (linear layouts only)
#       The witness of the PUSHFORWARD distributions: project both clouds,
#       then build g in 2-D from scratch with its own median-heuristic
#       bandwidth grid. Well defined, needs no lifting, and is the only
#       mode in which background and point positions are consistent by
#       construction. It answers a strictly weaker question -- two clouds
#       can differ in E dimensions and coincide after projection -- and
#       its value is NOT the gate's statistic.
#
#   field="nw"  (any layout, and the ONLY option for t-SNE)
#       Nadaraya-Watson smoothing, in the plane, of the witness values
#       already attached to the plotted points, pooling {u_i} and {v_j}
#       since both are values of the SAME function g at different
#       locations. Grid cells further than mask_radius * h from every
#       sample are masked rather than extrapolated. This is a DISPLAY
#       SMOOTHER: it interpolates true values of g, it does not evaluate
#       g, and its resolution is set by h, not by the kernel bandwidths.
#
# For t-SNE, "lift" is impossible (no inverse map) and "proj" is refused:
# t-SNE coordinates carry no metric, so a Gaussian kernel built on them
# has no interpretation. Both raise ValueError rather than producing a
# plausible-looking figure.
#
# In every mode the ZERO LEVEL SET of g is drawn: the boundary between
# the sim-dense (g > 0) and real-dense (g < 0) regions of the panel.

def _witness_on_points(Q: np.ndarray, fit: np.ndarray, real: np.ndarray,
                       bandwidths: np.ndarray,
                       chunk: int = 2048) -> np.ndarray:
    """g evaluated at arbitrary points Q, per bandwidth. Returns (S, m).

    Identical formula to witness_function's u and v -- same fit set, same
    kernels -- so field and scatter share one scale. Chunked over Q
    because the Gram block is O(n_fit x m) and m is a grid.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    bandwidths = np.asarray(bandwidths, dtype=np.float64)
    out = np.empty((bandwidths.size, Q.shape[0]), dtype=np.float64)
    for a in range(0, Q.shape[0], int(chunk)):
        q = Q[a:a + int(chunk)]
        d_f = _sqdist(fit, q)
        d_r = _sqdist(real, q)
        for s in range(bandwidths.size):
            gam = 1.0 / (2.0 * float(bandwidths[s]) ** 2)
            out[s, a:a + q.shape[0]] = (np.exp(-gam * d_f).mean(axis=0)
                                        - np.exp(-gam * d_r).mean(axis=0))
    return out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning NaN instead of raising on a
    degenerate (zero-variance) input."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 1e-300 or sb <= 1e-300:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _grid_2d(Y: np.ndarray, grid: int, pad: float = 0.06):
    """Regular grid covering the layout Y with a small margin."""
    x0, x1 = float(Y[:, 0].min()), float(Y[:, 0].max())
    y0, y1 = float(Y[:, 1].min()), float(Y[:, 1].max())
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    gx = np.linspace(x0 - pad * dx, x1 + pad * dx, int(grid))
    gy = np.linspace(y0 - pad * dy, y1 + pad * dy, int(grid))
    XX, YY = np.meshgrid(gx, gy)
    return XX, YY, np.column_stack([XX.ravel(), YY.ravel()])


def _field_nw(Qgrid: np.ndarray, Ydata: np.ndarray, vals: np.ndarray,
              h: float, mask_radius: float,
              chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Nadaraya-Watson smoothing of vals (S, n_data) onto Qgrid (m, 2).

    Returns (field (S, m), mask (m,) True where no sample is within
    mask_radius * h and the value must not be shown).
    """
    S = vals.shape[0]
    out = np.empty((S, Qgrid.shape[0]), dtype=np.float64)
    mask = np.empty(Qgrid.shape[0], dtype=bool)
    for a in range(0, Qgrid.shape[0], int(chunk)):
        q = Qgrid[a:a + int(chunk)]
        d = _sqdist(q, Ydata)                      # (m_chunk, n_data)
        W = np.exp(-d / (2.0 * float(h) ** 2))
        tot = np.maximum(W.sum(axis=1), 1e-300)
        out[:, a:a + q.shape[0]] = (W @ vals.T).T / tot[None, :]
        mask[a:a + q.shape[0]] = np.sqrt(d.min(axis=1)) > mask_radius * h
    return out, mask


def witness_heatmaps(z_sim: np.ndarray, z_real: np.ndarray,
                     outdir: str,
                     space: str = "z",
                     bandwidths: Optional[Sequence[float]] = None,
                     split: bool = True,
                     methods: Sequence[str] = ("pca", "contrast", "tsne"),
                     field: str = "auto",
                     grid: int = 120,
                     groups: Optional[Sequence] = None,
                     seed: int = 0,
                     max_points: int = 2000,
                     sphere: Optional[bool] = None,
                     nw_scale: float = 0.04,
                     mask_radius: float = 2.5,
                     n_label: int = 3,
                     dpi: int = 150) -> Dict[str, str]:
    """Witness g as a filled field over each 2-D layout, saved as PNG.

    Produces, in `outdir`, for each method:
      witness_heat_<space>_<method>_<field>.png
          one panel per bandwidth plus the summed panel; filled colour is
          g, black contour is the zero level set g = 0, sampled points are
          overlaid coloured by THEIR OWN g (in the full embedding space)
          on the same scale -- so in "lift" mode a visible mismatch
          between a point and its background is the plane's discarded
          variance made legible, not a bug;
      witness_heat_<space>.npz
          grid coordinates, the field per bandwidth, the mask, and the
          bandwidths, so figures can be restyled without recomputing.

    Parameters (beyond witness_function's)
    --------------------------------------
    methods : sequence of {"pca", "contrast", "tsne"}
        "pca" is the variance-optimal plane; "contrast" spends its first
        axis on the sim/real mean difference and is the one to read when
        the PCA plane is dominated by a nuisance axis such as the
        between-mode direction of a multimodal simulator; "tsne" is
        non-linear and therefore restricted to field="nw".
    field : {"auto", "lift", "proj", "nw"}
        "auto" -> "lift" for the linear layouts (pca, contrast), "nw" for
        tsne. See the block comment above this function: the three modes
        answer different questions. "lift" and "proj" raise ValueError for
        t-SNE.

    Slice diagnostics (printed, and stored in the npz)
    --------------------------------------------------
    var_frac
        Fraction of the POOLED PLOTTED cloud's total variance carried by
        the two plotted directions. "Plotted" matters: the cloud is the
        held-out evaluation half of z_sim (thinned to max_points) stacked
        on all of z_real, not the full simulated set, so this number is
        not comparable across different max_points or split settings. Answers "where are the points", not "does the
        picture represent g". Reported for completeness, not as the test.
    resid, resid_over_sigma
        Off-plane distance of each datum from its own lifted shadow, and
        the median of that distance divided by each bandwidth. This is
        the diagnostic with teeth: g resolves structure at the scale
        sigma, so a residual well above sigma means the narrow-sigma
        panels are evaluating g in a region no datum occupies.
    rho, rho_sum
        Correlation, across the data, between g at the lifted shadow and
        g at the datum itself, per bandwidth and for the summed kernel.
        rho near 1: the slice tracks the landscape. rho near 0: it is a
        decorative cut and the verdict must come from "proj"/"nw" or from
        a different plane. A rho below 0.5 raises a WARNING note.
    grid : int
        Grid points per axis. Cost is O(grid^2 * n_fit) kernel entries per
        bandwidth, chunked; 120 is cheap, 240 is still fine for n_fit
        ~2000.
    sphere : bool or None
        "lift" only. Renormalise lifted grid points onto S^{E-1} before
        evaluating g. None -> auto: True when the embeddings are unit
        norm (the z space), False otherwise (zraw). Without it the lifted
        plane sits INSIDE the sphere, strictly further from every datum
        than the data are from each other, which compresses the field
        toward zero.
    nw_scale : float
        "nw" only. Smoothing bandwidth h as a fraction of the layout
        diagonal.
    mask_radius : float
        "nw" only. Cells further than mask_radius * h from every sample
        are left blank instead of extrapolated.

    Returns
    -------
    dict name -> saved path.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if field not in ("auto", "lift", "proj", "nw"):
        raise ValueError("field must be auto, lift, proj or nw; got %r"
                         % (field,))
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    res = witness_function(z_sim, z_real, bandwidths=bandwidths, space=space,
                           split=split, seed=seed)
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    z_real = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    ev = z_sim[res.eval_idx]
    fit = z_sim[res.fit_idx]

    if sphere is None:
        nrm = np.linalg.norm(z_sim, axis=1)
        sphere = bool(np.allclose(nrm, 1.0, atol=1e-6))

    keep = np.arange(ev.shape[0])
    if ev.shape[0] > max_points:
        keep = np.sort(rng.choice(ev.shape[0], max_points, replace=False))
    P = np.concatenate([ev[keep], z_real], axis=0)
    n_s = keep.size

    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != z_real.shape[0]:
            raise ValueError("groups has length %d but z_real has %d rows"
                             % (groups.shape[0], z_real.shape[0]))
    worst = np.argsort(res.v_sum)[:max(0, int(n_label))]

    saved: Dict[str, str] = {}
    store: Dict[str, np.ndarray] = {"bandwidths": res.bandwidths}
    for method in methods:
        try:
            Y, axlabel, lift, var_frac = _project_2d(P, method, seed,
                                                     n_sim_pts=n_s)
        except ImportError as e:
            res.notes.append("SKIPPED %s: %s" % (method, e))
            continue

        fld = field
        if fld == "auto":
            fld = "lift" if lift is not None else "nw"
        if fld in ("lift", "proj") and lift is None:
            raise ValueError(
                "field=%r needs a linear layout; %s has no inverse map. "
                "Use field='nw' for %s." % (fld, method, method))

        XX, YY, Q = _grid_2d(Y, grid)
        pt_vals = np.concatenate([res.u[:, keep], res.v], axis=1)   # (S, n)
        diag: Dict[str, np.ndarray] = {"var_frac": np.asarray(var_frac)}

        if fld == "lift":
            Ql = lift(Q)
            if sphere:
                Ql = Ql / np.maximum(np.linalg.norm(Ql, axis=1,
                                                    keepdims=True), 1e-12)
            F = _witness_on_points(Ql, fit, z_real, res.bandwidths)
            bws = res.bandwidths
            mask = np.zeros(Q.shape[0], dtype=bool)
            # --- how faithful is this slice, in the units that matter? ----
            # Variance explained answers "where are the points". It does
            # NOT answer "does the slice reproduce g". These two do:
            #   resid   : off-plane distance of each datum, to be compared
            #             against the BANDWIDTHS, since g only resolves
            #             structure at the scale sigma. resid >> sigma
            #             means the slice is kernel-invisible to the data.
            #   rho     : correlation, over the data, between g at the
            #             datum's lifted shadow and g at the datum itself.
            #             rho near 1 means the picture tells the truth
            #             about g; rho near 0 means the plane is a
            #             decorative cut.
            Ydat = lift(Y)
            resid = np.linalg.norm(P - Ydat, axis=1)
            shadow = Ydat
            if sphere:
                shadow = shadow / np.maximum(
                    np.linalg.norm(shadow, axis=1, keepdims=True), 1e-12)
            F_at = _witness_on_points(shadow, fit, z_real, res.bandwidths)
            diag["resid"] = resid
            diag["resid_over_sigma"] = (float(np.median(resid))
                                        / res.bandwidths)
            diag["rho"] = np.asarray(
                [_safe_corr(F_at[s], pt_vals[s]) for s in range(bws.size)])
            diag["rho_sum"] = np.asarray(
                _safe_corr(F_at.sum(axis=0), pt_vals.sum(axis=0)))
            sub = ("exact evaluation on the lifted 2-plane%s | var %.0f%% | "
                   "median off-plane resid %.3g (%.2g-%.2g x sigma) | "
                   "slice fidelity rho=%.2f"
                   % (" (re-normalised onto the sphere)" if sphere else "",
                      100.0 * var_frac, float(np.median(resid)),
                      float(diag["resid_over_sigma"].min()),
                      float(diag["resid_over_sigma"].max()),
                      float(diag["rho_sum"])))
            if float(diag["rho_sum"]) < 0.5:
                res.notes.append(
                    "WARNING [%s]: slice fidelity rho=%.2f -- the lifted "
                    "plane does not track g at the data. Read field='proj' "
                    "or field='nw', or method='contrast'."
                    % (method, float(diag["rho_sum"])))
            if float(np.median(resid)) > float(res.bandwidths.min()):
                res.notes.append(
                    "WARNING [%s]: median off-plane residual %.3g exceeds "
                    "the smallest bandwidth %.3g -- the narrow-sigma panels "
                    "are cutting through empty space."
                    % (method, float(np.median(resid)),
                       float(res.bandwidths.min())))
        elif fld == "proj":
            bws = bandwidth_grid(Y[:n_s], Y[n_s:], rng=rng)
            F = _witness_on_points(Q, Y[:n_s], Y[n_s:], bws)
            pt_vals_2d = _witness_on_points(Y, Y[:n_s], Y[n_s:], bws)
            mask = np.zeros(Q.shape[0], dtype=bool)
            # Same fidelity question, different comparison: does the
            # projected witness rank the data the way the E-dim one does?
            diag["rho_sum"] = np.asarray(
                _safe_corr(pt_vals_2d.sum(axis=0), pt_vals.sum(axis=0)))
            pt_vals = pt_vals_2d
            sub = ("witness of the PROJECTED clouds, 2-D bandwidths | var "
                   "%.0f%% | rank agreement with the E-dim witness "
                   "rho=%.2f | weaker than the E-dim statistic"
                   % (100.0 * var_frac, float(diag["rho_sum"])))
        else:
            span = float(np.hypot(np.ptp(Y[:, 0]),
                                  np.ptp(Y[:, 1]))) or 1.0
            h = nw_scale * span
            F, mask = _field_nw(Q, Y, pt_vals, h, mask_radius)
            bws = res.bandwidths
            # Exact at the samples by construction, so a fidelity
            # correlation would be a tautology: recorded as NaN, not 1.
            diag["rho_sum"] = np.asarray(float("nan"))
            sub = ("Nadaraya-Watson smoothing of the sampled g, h=%.3g; "
                   "interpolated, not evaluated (no slice: fidelity "
                   "undefined)" % h)

        F_sum = F.sum(axis=0)
        panels = [("sigma=%.3g" % bws[s], F[s], pt_vals[s])
                  for s in range(bws.size)]
        panels.append(("sum over bandwidths", F_sum, pt_vals.sum(axis=0)))

        ncol = len(panels)
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.8),
                                 squeeze=False)
        ext = [XX.min(), XX.max(), YY.min(), YY.max()]
        for ax, (title, f, pv) in zip(axes[0], panels):
            G = np.where(mask, np.nan, f).reshape(XX.shape)
            vmax = float(np.nanpercentile(
                np.abs(np.concatenate([f[~mask], pv])), 99.5))
            vmax = vmax if vmax > 0 else 1e-12
            im = ax.imshow(G, extent=ext, origin="lower", cmap="coolwarm",
                           vmin=-vmax, vmax=vmax, aspect="auto",
                           interpolation="bilinear")
            # Masked cells stay masked for the contour: filling them with
            # zeros would make the g = 0 line trace the mask boundary
            # instead of the sim-dense / real-dense frontier.
            with np.errstate(invalid="ignore"):
                if np.nanmin(G) < 0.0 < np.nanmax(G):
                    ax.contour(XX, YY, np.ma.masked_invalid(G),
                               levels=[0.0], colors="k", linewidths=0.8)
            ax.scatter(Y[:n_s, 0], Y[:n_s, 1], c=pv[:n_s], cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, s=5, alpha=0.5, linewidths=0.2,
                       edgecolors="k", label="sim")
            ax.scatter(Y[n_s:, 0], Y[n_s:, 1], c=pv[n_s:], cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, s=58, marker="^",
                       edgecolors="k", linewidths=0.7, label="real")
            if groups is not None:
                for j in worst:
                    ax.annotate(str(groups[j]), (Y[n_s + j, 0],
                                                 Y[n_s + j, 1]),
                                fontsize=7, xytext=(3, 3),
                                textcoords="offset points")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(axlabel, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cb.set_label("g(z) = mu_sim(z) - mu_real(z)")
        axes[0][0].legend(loc="upper left", fontsize=8)
        fig.suptitle("Witness heatmap [%s, %s, field=%s]\n%s  |  black line: "
                     "g = 0" % (space, method, fld, sub), fontsize=11,
                     y=1.10)
        path = os.path.join(outdir,
                            "witness_heat_%s_%s_%s.png" % (space, method, fld))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["heat_%s" % method] = path

        store["%s_XX" % method] = XX
        store["%s_YY" % method] = YY
        store["%s_field" % method] = F
        store["%s_field_sum" % method] = F_sum
        store["%s_mask" % method] = mask
        store["%s_bandwidths" % method] = np.asarray(bws)
        store["%s_coords" % method] = Y
        store["%s_field_mode" % method] = np.asarray(fld)
        for k, val in diag.items():
            store["%s_%s" % (method, k)] = np.asarray(val)
        print("[witness_heatmaps] %s / %s / field=%s" % (space, method, fld))
        print("    variance carried by the plotted plane : %s"
              % ("%.1f%%" % (100.0 * var_frac)
                 if np.isfinite(var_frac) else "n/a (non-linear layout)"))
        if "resid" in diag:
            print("    median off-plane residual            : %.4g"
                  % float(np.median(diag["resid"])))
            print("    that residual in units of sigma      : %s"
                  % np.array2string(diag["resid_over_sigma"],
                                    precision=2, separator=", "))
            print("    slice fidelity rho, per bandwidth    : %s"
                  % np.array2string(diag["rho"], precision=2,
                                    separator=", "))
        if "rho_sum" in diag:
            print("    slice fidelity rho, summed kernel    : %.3f"
                  % float(diag["rho_sum"]))
        for n in res.notes:
            if n.startswith("WARNING [%s]" % method):
                print("    " + n)

    apath = os.path.join(outdir, "witness_heat_%s.npz" % space)
    np.savez_compressed(apath, **store)
    saved["arrays"] = apath
    return saved


# ---------------------------------------------------------------------------
# Parallel slices: is one cut representative, or is the plane an artefact?
# ---------------------------------------------------------------------------
#
# field="lift" shows g on ONE affine 2-plane, the one through the mean of
# the plotted cloud. The honest question is whether that single cut stands
# in for the E-dimensional landscape, and the honest answer is NOT the
# variance fraction. Variance says where the points are; it says nothing
# about whether g keeps the same shape as you move off the plane. Two
# quantities decide it, both already reported by witness_heatmaps:
#
#     resid_over_sigma   how far the data sit off the plane, measured in
#                        the only unit g can resolve, the bandwidth;
#     rho                whether g at a datum's shadow tracks g at the
#                        datum itself.
#
# The decision rule:
#
#     rho_sum >= ~0.9 AND median resid <~ sigma_min
#         -> the single cut already reproduces g at the data. Extra
#            slices are decoration; refer to the diagnostics and move on.
#     otherwise
#         -> the flat cut is not established as representative. Either
#            read field="proj" (which marginalises over the discarded
#            directions instead of fixing them) or run this function,
#            which shows whether the structure survives translation.
#
# What this function does: it translates the SAME 2-plane along a
# discarded direction w (orthogonal to the plane, by default the leading
# principal direction of the off-plane residuals), evaluating g exactly on
# each translate,
#
#     z(y, t) = mean + y B + t w,     y in R^2,  t in R,  w perp rows(B),
#
# at offsets t placed at QUANTILES of the data's own w-coordinate, so
# every slice is one the data actually populate. Empty-space slices are
# the trap this whole module keeps walking into and quantiles avoid it by
# construction.
#
# The colour scale is SHARED across slices -- unlike witness_heatmaps,
# where each panel is a different bandwidth and therefore a different
# magnitude. Here every panel is the same statistic at a different depth,
# so a common scale is the whole point: it is what makes "the structure
# is stable" or "the sign flips at depth" visible rather than normalised
# away.
#
# The quantitative readout is per-slice agreement with the central slice:
# a correlation and a sign-disagreement fraction. Stable correlation near
# 1 across depth means the 2-plane picture generalises and one cut was
# enough after all; a correlation that decays, or sign flips, means the
# discrepancy has structure in the discarded directions and no single
# flat cut can represent it.

def witness_slices(z_sim: np.ndarray, z_real: np.ndarray,
                   outdir: str,
                   space: str = "z",
                   bandwidths: Optional[Sequence[float]] = None,
                   split: bool = True,
                   method: str = "pca",
                   quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
                   grid: int = 100,
                   groups: Optional[Sequence] = None,
                   seed: int = 0,
                   max_points: int = 2000,
                   sphere: Optional[bool] = None,
                   bandwidth_index: Optional[int] = None,
                   dpi: int = 150) -> Dict[str, str]:
    """A stack of parallel lifted slices, to test whether one cut suffices.

    Produces, in `outdir`:
      witness_slices_<space>_<method>.png
          one panel per offset, SHARED colour scale, zero contour drawn,
          and only the data lying within that slice's slab overlaid;
      witness_slices_<space>_<method>.npz
          offsets, the field per slice, the slab membership counts, and
          the stability diagnostics.

    Parameters
    ----------
    method : {"pca", "contrast"}
        Must be a linear layout: a slice needs an exact lift. "tsne"
        raises ValueError.
    quantiles : sequence of float in (0, 1)
        Where to place the slices along the discarded direction, as
        quantiles of the DATA's coordinate along that direction. Using
        quantiles rather than fixed offsets guarantees every slice is
        populated. The slice nearest the median is the reference slice
        for the stability diagnostics.
    bandwidth_index : int or None
        None -> the field is the sum over bandwidths, i.e. the gate's own
        kernel. An integer selects a single bandwidth of the grid, which
        is the honest thing to do when the summed panels of
        witness_heatmaps disagreed across scales.

    Returns
    -------
    dict name -> saved path.

    Notes
    -----
    Cost is len(quantiles) * grid^2 * n_fit kernel entries per bandwidth,
    chunked. grid=100 with 5 slices and n_fit ~2000 is a few seconds.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    res = witness_function(z_sim, z_real, bandwidths=bandwidths, space=space,
                           split=split, seed=seed)
    z_sim = np.atleast_2d(np.asarray(z_sim, dtype=np.float64))
    z_real = np.atleast_2d(np.asarray(z_real, dtype=np.float64))
    ev, fit = z_sim[res.eval_idx], z_sim[res.fit_idx]

    if sphere is None:
        sphere = bool(np.allclose(np.linalg.norm(z_sim, axis=1), 1.0,
                                  atol=1e-6))

    keep = np.arange(ev.shape[0])
    if ev.shape[0] > max_points:
        keep = np.sort(rng.choice(ev.shape[0], max_points, replace=False))
    P = np.concatenate([ev[keep], z_real], axis=0)
    n_s = keep.size

    Y, axlabel, lift, var_frac = _project_2d(P, method, seed, n_sim_pts=n_s)
    if lift is None:
        raise ValueError("witness_slices needs a linear layout with an exact "
                         "lift; %r has none. Use method='pca' or "
                         "'contrast'." % (method,))

    # Recover the frame from the lift itself, so the two can never drift
    # apart: lift(Q) = mean + Q @ basis, hence lift(0) = mean and
    # lift(I_2) - mean = basis.
    mean = lift(np.zeros((1, 2)))                       # (1, E)
    basis = lift(np.eye(2)) - mean                      # (2, E)

    # Discarded direction: leading principal direction of the off-plane
    # residuals. Orthogonal to the plane by construction, since the
    # residuals are.
    R = P - lift(Y)
    _U, _sv, Vt = np.linalg.svd(R, full_matrices=False)
    w = Vt[0]
    w = w - basis.T @ (basis @ w)                       # re-orthogonalise
    nw_ = float(np.linalg.norm(w))
    if nw_ <= 1e-12:
        raise ValueError("the cloud is exactly 2-dimensional: there is no "
                         "off-plane direction to slice along, and the single "
                         "lifted plane is already the whole space")
    w = w / nw_

    t_data = R @ w                                       # (n_pooled,)
    qs = np.asarray(sorted(float(q) for q in quantiles), dtype=np.float64)
    if np.any((qs <= 0.0) | (qs >= 1.0)):
        raise ValueError("quantiles must lie strictly inside (0, 1)")
    offsets = np.quantile(t_data, qs)
    ref = int(np.argmin(np.abs(qs - 0.5)))               # reference slice
    slab = float(np.median(np.diff(offsets))) / 2.0 if offsets.size > 1 \
        else float(np.std(t_data))
    slab = slab if slab > 0 else float(np.std(t_data)) or 1.0

    XX, YY, Q = _grid_2d(Y, grid)
    if bandwidth_index is None:
        which = slice(None)
        kernel_label = "sum over bandwidths"
    else:
        bi = int(bandwidth_index)
        if not 0 <= bi < res.bandwidths.size:
            raise ValueError("bandwidth_index %d outside 0..%d"
                             % (bi, res.bandwidths.size - 1))
        which = slice(bi, bi + 1)
        kernel_label = "sigma=%.3g" % res.bandwidths[bi]

    fields = np.empty((offsets.size, Q.shape[0]), dtype=np.float64)
    for i, t in enumerate(offsets):
        Ql = lift(Q) + float(t) * w[None, :]
        if sphere:
            Ql = Ql / np.maximum(np.linalg.norm(Ql, axis=1, keepdims=True),
                                 1e-12)
        fields[i] = _witness_on_points(Ql, fit, z_real,
                                       res.bandwidths[which]).sum(axis=0)

    # --- does the picture survive translation? ----------------------------
    stab = np.asarray([_safe_corr(fields[i], fields[ref])
                       for i in range(offsets.size)])
    sign_dis = np.asarray([float(np.mean(np.sign(fields[i])
                                         != np.sign(fields[ref])))
                           for i in range(offsets.size)])
    pt_vals = np.concatenate([res.u[:, keep], res.v], axis=1)[which].sum(0)

    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != z_real.shape[0]:
            raise ValueError("groups has length %d but z_real has %d rows"
                             % (groups.shape[0], z_real.shape[0]))

    # SHARED scale across slices: the comparison is the point.
    vmax = float(np.percentile(np.abs(np.concatenate(
        [fields.ravel(), pt_vals])), 99.5)) or 1e-12
    ncol = offsets.size
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.8),
                             squeeze=False)
    ext = [XX.min(), XX.max(), YY.min(), YY.max()]
    for i, ax in enumerate(axes[0]):
        G = fields[i].reshape(XX.shape)
        im = ax.imshow(G, extent=ext, origin="lower", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="bilinear")
        if G.min() < 0.0 < G.max():
            ax.contour(XX, YY, G, levels=[0.0], colors="k", linewidths=0.8)
        inslab = np.abs(t_data - offsets[i]) <= slab
        ms = inslab[:n_s]
        mr = inslab[n_s:]
        ax.scatter(Y[:n_s][ms, 0], Y[:n_s][ms, 1], c=pt_vals[:n_s][ms],
                   cmap="coolwarm", vmin=-vmax, vmax=vmax, s=6, alpha=0.6,
                   linewidths=0.2, edgecolors="k", label="sim in slab")
        ax.scatter(Y[n_s:][mr, 0], Y[n_s:][mr, 1], c=pt_vals[n_s:][mr],
                   cmap="coolwarm", vmin=-vmax, vmax=vmax, s=58, marker="^",
                   edgecolors="k", linewidths=0.7, label="real in slab")
        if groups is not None:
            for j in np.flatnonzero(mr):
                ax.annotate(str(groups[j]), (Y[n_s + j, 0], Y[n_s + j, 1]),
                            fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        ax.set_title("q=%.2f  t=%+.3g%s\ncorr to ref %.2f | sign diff %.0f%%"
                     % (qs[i], offsets[i], "  (ref)" if i == ref else "",
                        stab[i], 100 * sign_dis[i]), fontsize=9)
        ax.set_xlabel("%d sim, %d real in slab" % (int(ms.sum()),
                                                   int(mr.sum())), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
    cb.set_label("g(z) = mu_sim(z) - mu_real(z)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Witness slices [%s, %s, %s] along the leading discarded "
                 "direction\n%s | plane var %.0f%% | SHARED colour scale | "
                 "black line: g = 0"
                 % (space, method, kernel_label, axlabel, 100.0 * var_frac),
                 fontsize=11, y=1.10)
    path = os.path.join(outdir, "witness_slices_%s_%s.png" % (space, method))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    apath = os.path.join(outdir,
                         "witness_slices_%s_%s.npz" % (space, method))
    np.savez_compressed(apath, XX=XX, YY=YY, fields=fields, offsets=offsets,
                        quantiles=qs, ref=np.asarray(ref), direction=w,
                        stability=stab, sign_disagreement=sign_dis,
                        t_data=t_data, coords=Y, slab=np.asarray(slab),
                        var_frac=np.asarray(var_frac),
                        bandwidths=res.bandwidths)

    print("[witness_slices] %s / %s / %s" % (space, method, kernel_label))
    print("    offsets t (data quantiles) : %s"
          % np.array2string(offsets, precision=3, separator=", "))
    print("    corr to reference slice    : %s"
          % np.array2string(stab, precision=3, separator=", "))
    print("    sign disagreement          : %s"
          % np.array2string(sign_dis, precision=3, separator=", "))
    worst = float(np.nanmin(stab))
    if worst < 0.8 or float(np.nanmax(sign_dis)) > 0.1:
        print("    VERDICT: the structure does NOT survive translation "
              "(min corr %.2f, max sign disagreement %.0f%%). One flat cut "
              "is not representative -- read field='proj', or treat the "
              "discarded directions as carrying part of the discrepancy."
              % (worst, 100 * float(np.nanmax(sign_dis))))
    else:
        print("    VERDICT: the structure is stable across depth "
              "(min corr %.2f). The single lifted plane of "
              "witness_heatmaps is representative; extra slices add "
              "nothing." % worst)
    return {"slices": path, "arrays": apath}
