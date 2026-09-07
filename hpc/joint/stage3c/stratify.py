"""Mechanistic stratification, eq. (5) (plan v0.6, S2.7).

Three covariances of culture-level posterior means, all in parameter
coordinates:

    Sigma_rep  = Cov(m_g - m_g') over SAME-DONOR pairs -- the empirical
                 nuisance floor, covering differentiation, plating, recording,
                 detection, encoding and inference, because it is measured
                 through the whole chain rather than modelled;
    Sigma_pat  = Cov of donor means WITHIN one diagnosis;
    Sigma_dx   = Cov of diagnosis means.

and the generalised eigenproblem

    Sigma_pat v_j = mu_j Sigma_rep v_j                                   (5)

Directions with large mu_j are those along which donors differ more than
replicates of one donor do: mechanistic heterogeneity above the measured noise
floor.

The null value of mu_j is NOT 1
-------------------------------
This is the same trap as the replicate statistic's target (S2.5b), in a
different place, and the plan's wording invites it. Under the null that donors
do NOT differ mechanistically, a donor mean estimated from k wells still
varies, because each m_g carries estimation noise:

    Var(m_g | donor) = Sigma_rep / 2         (a difference of two independent
                                              wells has twice one well's
                                              variance)
    Var(donor mean of k wells) = Sigma_rep / (2k)

so under the null

    mu_j = Sigma_pat / Sigma_rep = 1 / (2k).                             (5a)

With one well per donor the null is 1/2, not 1; with two wells it is 1/4.
Comparing mu_j against 1 is therefore CONSERVATIVE by a factor 2k -- it
under-detects heterogeneity rather than inventing it, which is the safe
direction, but the threshold should be stated for what it is.
`stratify` returns `null_mu` computed from the actual mean wells per donor and
flags directions against BOTH, so the choice is visible rather than baked in.

Pure ASCII, LF only.
"""

import numpy as np
from scipy.linalg import eigh

__all__ = ["replicate_covariance", "patient_covariance", "diagnosis_covariance",
           "stratify", "validate_on_bench"]


def replicate_covariance(means, donors, ddof=1):
    """Sigma_rep from all within-donor pairs of culture-level means.

    All pairs, not one per donor: a donor with k wells contributes k(k-1)/2
    differences and they are not independent, but using one pair per donor
    throws away most of the information for a variance that is already
    estimated from very few units (D12).
    """
    means = np.atleast_2d(np.asarray(means, dtype=np.float64))
    donors = np.asarray(donors).ravel()
    diffs = []
    n_singleton = 0
    for d in np.unique(donors):
        idx = np.flatnonzero(donors == d)
        if idx.size < 2:
            n_singleton += 1
            continue
        for a in range(idx.size):
            for b in range(a + 1, idx.size):
                diffs.append(means[idx[a]] - means[idx[b]])
    if len(diffs) < 2:
        raise ValueError("need at least 2 same-donor pairs; got %d "
                         "(%d singleton donors). This is decision D12."
                         % (len(diffs), n_singleton))
    diffs = np.asarray(diffs)
    # Mean zero by construction under the null, so the covariance is taken
    # about zero rather than about the sample mean: subtracting a sample mean
    # here would remove a real common shift and shrink the floor.
    cov = diffs.T @ diffs / max(len(diffs) - ddof, 1)
    return np.atleast_2d(cov), {"n_pairs": len(diffs),
                                "n_singleton_donors": int(n_singleton)}


def _donor_means(means, donors):
    means = np.atleast_2d(np.asarray(means, dtype=np.float64))
    donors = np.asarray(donors).ravel()
    uniq = np.unique(donors)
    dm = np.stack([means[donors == d].mean(axis=0) for d in uniq])
    counts = np.array([int((donors == d).sum()) for d in uniq])
    return dm, uniq, counts


def patient_covariance(means, donors, labels):
    """Sigma_pat: covariance of donor means, pooled WITHIN diagnosis.

    Pooled within, not across: a between-diagnosis difference would otherwise
    enter Sigma_pat and the stratification would rediscover the label it is
    supposed to be independent of.
    """
    dm, uniq, counts = _donor_means(means, donors)
    labels = np.asarray(labels).ravel()
    donor_label = np.array([labels[np.flatnonzero(
        np.asarray(donors).ravel() == d)[0]] for d in uniq])
    parts, n = [], 0
    for c in np.unique(donor_label):
        rows = dm[donor_label == c]
        if rows.shape[0] < 2:
            continue
        centred = rows - rows.mean(axis=0, keepdims=True)
        parts.append(centred.T @ centred)
        n += rows.shape[0] - 1
    if not parts or n == 0:
        raise ValueError("need at least 2 donors in some diagnosis")
    return np.atleast_2d(sum(parts) / n), {"n_donors": int(dm.shape[0]),
                                           "mean_wells_per_donor":
                                               float(counts.mean())}


def diagnosis_covariance(means, donors, labels):
    """Sigma_dx: covariance of the diagnosis means."""
    dm, uniq, _ = _donor_means(means, donors)
    labels = np.asarray(labels).ravel()
    donor_label = np.array([labels[np.flatnonzero(
        np.asarray(donors).ravel() == d)[0]] for d in uniq])
    cls_means = np.stack([dm[donor_label == c].mean(axis=0)
                          for c in np.unique(donor_label)])
    if cls_means.shape[0] < 2:
        raise ValueError("need at least 2 diagnoses")
    centred = cls_means - cls_means.mean(axis=0, keepdims=True)
    return np.atleast_2d(centred.T @ centred / (cls_means.shape[0] - 1))


def stratify(means, donors, labels, param_names=None, jitter=1e-10):
    """Solve eq. (5) and report the spectrum against BOTH thresholds."""
    means = np.atleast_2d(np.asarray(means, dtype=np.float64))
    d = means.shape[1]
    S_rep, rep_info = replicate_covariance(means, donors)
    S_pat, pat_info = patient_covariance(means, donors, labels)
    S_dx = diagnosis_covariance(means, donors, labels)

    scale = float(np.trace(S_rep)) / max(d, 1)
    mu, vec = eigh(S_pat, S_rep + jitter * scale * np.eye(d))
    order = np.argsort(-mu)
    mu = mu[order]
    vec = vec[:, order]

    k = pat_info["mean_wells_per_donor"]
    null_mu = 1.0 / (2.0 * k)
    names = list(param_names) if param_names else ["axis%d" % j
                                                   for j in range(d)]

    dirs = []
    for j in range(d):
        v = vec[:, j]
        heavy = np.argsort(-np.abs(v))[:3]
        dirs.append({
            "index": j, "mu": float(mu[j]),
            "above_null": bool(mu[j] > null_mu),
            "above_one": bool(mu[j] > 1.0),
            "vector": v.tolist(),
            "dominant_axes": [names[t] for t in heavy],
            "dominant_loadings": [float(v[t]) for t in heavy],
        })
    return {
        "mu": mu.tolist(),
        "null_mu": null_mu,
        "null_mu_derivation": ("1/(2k) with k = %.2f wells per donor; a "
                               "difference of two wells has twice one well's "
                               "variance, and a donor mean of k wells has "
                               "1/k of it" % k),
        "n_above_null": int(np.sum(mu > null_mu)),
        "n_above_one": int(np.sum(mu > 1.0)),
        "Sigma_rep": S_rep.tolist(), "Sigma_pat": S_pat.tolist(),
        "Sigma_dx": S_dx.tolist(),
        "axes": names, "directions": dirs,
        "rep_info": rep_info, "pat_info": pat_info,
    }


def validate_on_bench(result, spread_axes, gap_ratio=3.0, top_load=1):
    """The Stage 3c gate: does the spectrum SEPARATE the axes given spread?

    Two conditions, and neither is a count:

      1. **Separation.** mu[n_expected - 1] / mu[n_expected] >= gap_ratio.
         The informative directions must stand clearly apart from the rest,
         not merely clear a threshold.
      2. **Identification.** Each of the leading n_expected directions loads
         most heavily on one of the axes that was actually given spread.

    [CORRECTION] The first version required `n_above_null == n_expected`
    exactly. mu_j is estimated from a few dozen donors, so the d - n_expected
    null directions fluctuate around 1/(2k) and roughly half of them land
    above it by chance: a perfectly correct run with mu = [8.30, 0.29, 0.22,
    0.15] against a null of 0.25 was FAILED because one null direction sat at
    0.29. An exact-count test on a noisy statistic is not a gate, it is a coin
    flip with extra steps. The count is still reported, as information.
    """
    spread_axes = sorted(int(a) for a in spread_axes)
    n_expected = len(spread_axes)
    mu = np.asarray(result["mu"], dtype=np.float64)
    names = result["axes"]
    expected_names = {names[a] for a in spread_axes}

    if n_expected >= mu.size:
        ratio = float("inf")
    elif mu[n_expected] <= 0:
        ratio = float("inf")
    else:
        ratio = float(mu[n_expected - 1] / mu[n_expected])

    leading = [d["dominant_axes"][:top_load]
               for d in result["directions"][:n_expected]]
    identified = [bool(set(l) & expected_names) for l in leading]

    passed = bool(ratio >= gap_ratio and all(identified))
    return {
        "passed": passed,
        "separation_ratio": ratio,
        "gap_ratio_required": float(gap_ratio),
        "leading_direction_axes": leading,
        "expected_axes": sorted(expected_names),
        "identified": identified,
        "n_above_null": result["n_above_null"],
        "null_mu": result["null_mu"],
        "note": ("mu[%d]/mu[%d] = %.2f (need >= %.1f); leading directions load "
                 "on %s, expected %s. The count above the null (%d) is "
                 "reported but NOT used as the criterion: null directions "
                 "fluctuate around 1/(2k) and about half exceed it by chance."
                 % (n_expected - 1, n_expected, ratio, gap_ratio, leading,
                    sorted(expected_names), result["n_above_null"])),
    }
