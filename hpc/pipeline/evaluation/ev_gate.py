"""The gate of a recorded project's evaluation (block 34; specs/SPEC.md sec.
2.6, "The gate", equations (R5)-(R8), and block 34; decisions D-161, D-163
(4), D-164).

    role = ev_gate.assign_roles(donor, fracs, n_ref, n_rank, n_null,
                                rows_per_donor, seed)
    out = ev_gate.gate(z_ref, z_rank, z_null, z_test, factors)
    out["p"]            # (n,) the p-value of each tested window (R8)

Pure functions of arrays: numpy and `npe_misspec` (the kernel's bandwidths,
the reference's self term, the p-value's convention). No torch; no file is
read or written.

CONTRACT (wave A): the constants and the signatures are fixed by
specs/SPEC.md block 34.

Pure ASCII, LF only.
"""

import _evalpath  # noqa: F401

import numpy as np

import npe_misspec

# a row's role in the gate (`gate_role` of block 35's <stem>_sim.npz)
ROLE_NONE, ROLE_REF, ROLE_RANK, ROLE_NULL, ROLE_CHECK = 0, 1, 2, 3, 4

# points per kernel matrix in member_statistic: bounds the memory of one
# (N_ref, chunk) float64 matrix (800 x 4096 x 8 B = 26 MB), not a parameter
# of the science
_CHUNK = 4096

# guard of floor(f G) against binary roundoff (0.29 * 100 = 28.999...96):
# far below the resolution 1 / G of any real f G, far above float64 roundoff
_FLOOR_EPS = 1e-9


def _occurrence(donor):
    """(n,) int64: each row's position among its own network's rows, in the
    given order (0 for the first)."""
    n = donor.shape[0]
    order = np.argsort(donor, kind="stable")   # by network, then position
    d_sorted = donor[order]
    start = np.ones(n, dtype=bool)
    start[1:] = d_sorted[1:] != d_sorted[:-1]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    occ = np.empty(n, dtype=np.int64)
    occ[order] = np.arange(n) - first
    return occ


def assign_roles(donor, fracs, n_ref, n_rank, n_null, rows_per_donor, seed):
    """(n,) int8: the gate's role of each row of D_sim.

    donor (n,) int: the simulated network of each row, in global row order.
    fracs: (f_ref, f_rank, f_null, f_check), four positive numbers summing
    to 1. n_ref, n_rank, n_null: the caps of the reference, rank and null
    sets. rows_per_donor: the most rows of one network in the rank, null
    and check sets. seed: of the permutation of the networks.

    The rule is specs/SPEC.md block 34's: the sorted distinct networks are
    permuted by numpy.random.default_rng(seed).permutation and dealt, in
    that order, floor(f_ref G), floor(f_rank G), floor(f_null G) and the
    rest; within a group the rows are ordered by (occurrence within their
    network, position in `donor`). ValueError when a group has no network.
    """
    donor = np.asarray(donor)
    if donor.ndim != 1 or not np.issubdtype(donor.dtype, np.integer):
        raise ValueError("donor must be a 1-D integer array, got shape %s dtype %s"
                         % (donor.shape, donor.dtype))
    fr = np.asarray(fracs, dtype=np.float64)
    if (fr.shape != (4,) or not np.all(np.isfinite(fr)) or np.any(fr <= 0)
            or abs(float(fr.sum()) - 1.0) > 1e-9):
        raise ValueError("fracs must be four positive numbers summing to 1, got %r"
                         % (fracs,))
    for name, v in (("n_ref", n_ref), ("n_rank", n_rank), ("n_null", n_null),
                    ("rows_per_donor", rows_per_donor)):
        if int(v) != v or v < (1 if name == "rows_per_donor" else 0):
            raise ValueError("%s must be a %s integer, got %r"
                             % (name, "positive" if name == "rows_per_donor"
                                else "non-negative", v))
    n = donor.shape[0]
    nets = np.unique(donor)                      # sorted distinct networks
    G = int(nets.size)
    perm = np.random.default_rng(seed).permutation(nets)
    sizes = [int(np.floor(f * G + _FLOOR_EPS)) for f in fr[:3]]
    sizes.append(G - sum(sizes))
    if min(sizes) < 1:
        raise ValueError("the gate's groups need one network each: G = %d networks "
                         "with fracs %r give group sizes %r"
                         % (G, tuple(float(f) for f in fr), sizes))
    bounds = np.cumsum([0] + sizes)
    group_of_net = {}
    for k in range(4):
        for net in perm[bounds[k]:bounds[k + 1]].tolist():
            group_of_net[net] = k
    group = np.fromiter((group_of_net[d] for d in donor.tolist()), dtype=np.int64,
                        count=n)
    occ = _occurrence(donor)
    pos = np.arange(n)
    rr = np.lexsort((pos, occ))                  # round-robin: (occurrence, position)

    role = np.full(n, ROLE_NONE, dtype=np.int8)
    caps = (int(n_ref), int(n_rank), int(n_null), None)
    codes = (ROLE_REF, ROLE_RANK, ROLE_NULL, ROLE_CHECK)
    for k in range(4):
        rows = rr[group[rr] == k]
        if k > 0:                                # the reference takes any occurrence
            rows = rows[occ[rows] < int(rows_per_donor)]
        if caps[k] is not None:
            rows = rows[:caps[k]]
        role[rows] = codes[k]
    return role


def _check_finite(name, a):
    if not np.all(np.isfinite(a)):
        raise ValueError("%s has %d non-finite values" % (name, int(np.sum(~np.isfinite(a)))))


# Custom: the biased squared MMD (R6) between the reference and each of many
# single points, from one kernel matrix per chunk of points;
# npe_misspec.mmd2_multiscale takes one second sample per call and a call per
# window and member would dominate the report's time. Checked:
# npe_misspec.mmd2_multiscale (the reference, oracle R34a), npe_misspec._self_term
# (used for the first term), npe_misspec._sqdist (used for the distances),
# scipy.spatial.distance.cdist (no kernel sums).
def member_statistic(z_ref, z, factors):
    """(t, bandwidths, median) for one member: t (n,) float64, the biased
    squared MMD (R6) between each point of z (n, d_z) and the reference
    z_ref (N_ref, d_z), with the kernel (R5); bandwidths (n_h,) float64 =
    factors * median; median the median pairwise distance among the
    reference embeddings. n may be 0. ValueError on a non-finite input."""
    z_ref = np.asarray(z_ref, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64)
    if z_ref.ndim != 2 or z_ref.shape[0] < 1:
        raise ValueError("z_ref must be (N_ref, d_z) with N_ref >= 1, got %s"
                         % (z_ref.shape,))
    if z.ndim != 2 or z.shape[1] != z_ref.shape[1]:
        raise ValueError("z must be (n, d_z) with d_z = %d, got %s"
                         % (z_ref.shape[1], z.shape))
    if fac.ndim != 1 or fac.size < 1 or not np.all(np.isfinite(fac)) or np.any(fac <= 0):
        raise ValueError("factors must be positive finite numbers, got %r" % (factors,))
    _check_finite("z_ref", z_ref)
    _check_finite("z", z)
    # (R5): the bandwidths of the reference alone (npe_misspec's median
    # heuristic; its subsampling starts above 1,000 points)
    bandwidths = npe_misspec.bandwidth_grid(z_ref, z_ref[:0], factors=fac.tolist())
    median = float(npe_misspec.bandwidth_grid(z_ref, z_ref[:0], factors=(1.0,))[0])
    # first term of (R6): constant in x; second: k_m(z, z) = n_h
    kaa = npe_misspec._self_term(z_ref, bandwidths)
    const = float(np.sum(kaa)) + float(bandwidths.size)
    gam = 1.0 / (2.0 * bandwidths ** 2)
    n = z.shape[0]
    t = np.empty(n, dtype=np.float64)
    for s in range(0, n, _CHUNK):
        d2 = npe_misspec._sqdist(z_ref, z[s:s + _CHUNK])        # (N_ref, chunk)
        cross = np.zeros(d2.shape[1], dtype=np.float64)
        for g in gam:
            cross += np.exp(-g * d2).mean(axis=0)
        t[s:s + _CHUNK] = const - 2.0 * cross
    return t, np.asarray(bandwidths, dtype=np.float64), median


def _count_below(sorted_ref, x):
    """#{v in sorted_ref : v < x} for every x."""
    return np.searchsorted(sorted_ref, x, side="left")


def _count_at_least(sorted_ref, x):
    """#{v in sorted_ref : v >= x} for every x."""
    return sorted_ref.size - np.searchsorted(sorted_ref, x, side="left")


def gate(z_ref, z_rank, z_null, z_test, factors):
    """The gate of one ensemble on n tested windows.

    z_ref (M, N_ref, d_z), z_rank (M, N_rank, d_z), z_null (M, N_null, d_z),
    z_test (M, n, d_z), n >= 0: each member's embeddings of the reference,
    rank, null and tested rows. factors: the kernel's bandwidth factors.

    Returns a dict of float64 arrays: `t`, `r`, `p_member` (M, n); `T`, `p`
    (n,); `t_rank` (M, N_rank); `t_null`, `r_null` (M, N_null); `T_null`
    (N_null,); `bandwidths` (M, n_h); `median` (M,). (R6)-(R8): r by the
    count of rank values strictly below, T the mean over the members, the
    p-values by (1 + the count of null values at least as large) / (1 +
    N_null). The null rows are ranked by the expression that ranks a tested
    window. ValueError on a non-finite input.
    """
    arrs = {}
    for name, a in (("z_ref", z_ref), ("z_rank", z_rank), ("z_null", z_null),
                    ("z_test", z_test)):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 3:
            raise ValueError("%s must be (M, rows, d_z), got shape %s" % (name, a.shape))
        _check_finite(name, a)
        arrs[name] = a
    M, N_ref, d_z = arrs["z_ref"].shape
    for name in ("z_rank", "z_null", "z_test"):
        if arrs[name].shape[0] != M or arrs[name].shape[2] != d_z:
            raise ValueError("%s has shape %s; z_ref has M = %d, d_z = %d"
                             % (name, arrs[name].shape, M, d_z))
    N_rank = arrs["z_rank"].shape[1]
    N_null = arrs["z_null"].shape[1]
    n = arrs["z_test"].shape[1]
    if M < 1 or N_ref < 1 or N_rank < 1 or N_null < 1:
        raise ValueError("the gate needs M, N_ref, N_rank, N_null >= 1, got %d, %d, %d, %d"
                         % (M, N_ref, N_rank, N_null))
    n_h = len(factors)
    t = np.empty((M, n))
    t_rank = np.empty((M, N_rank))
    t_null = np.empty((M, N_null))
    c_test = np.empty((M, n), dtype=np.int64)    # #{a in I_rank : t_m(x_a) < t_m(x)}
    c_null = np.empty((M, N_null), dtype=np.int64)
    p_member = np.empty((M, n))
    bandwidths = np.empty((M, n_h))
    median = np.empty(M)
    for m in range(M):
        pts = np.concatenate([arrs["z_rank"][m], arrs["z_null"][m], arrs["z_test"][m]])
        tm, bw, med = member_statistic(arrs["z_ref"][m], pts, factors)
        t_rank[m] = tm[:N_rank]
        t_null[m] = tm[N_rank:N_rank + N_null]
        t[m] = tm[N_rank + N_null:]
        bandwidths[m] = bw
        median[m] = med
        srt = np.sort(t_rank[m])
        c_test[m] = _count_below(srt, t[m])
        c_null[m] = _count_below(srt, t_null[m])
        # (R8) per member, npe_misspec._pvalue's convention
        p_member[m] = (1.0 + _count_at_least(np.sort(t_null[m]), t[m])) / (1.0 + N_null)
    r = c_test / float(N_rank)
    r_null = c_null / float(N_rank)
    # (R7): T = (1/M) sum_m r_m = (sum_m count_m) / (M N_rank), formed from the
    # integer counts so that equal values of T are equal floats whatever the
    # members' split (ties between a window and a null row decide (R8))
    T = c_test.sum(axis=0) / float(M * N_rank)
    T_null = c_null.sum(axis=0) / float(M * N_rank)
    p = (1.0 + _count_at_least(np.sort(T_null), T)) / (1.0 + N_null)
    return {"t": t, "r": r, "p_member": p_member, "T": np.asarray(T, dtype=np.float64),
            "p": np.asarray(p, dtype=np.float64), "t_rank": t_rank, "t_null": t_null,
            "r_null": r_null, "T_null": np.asarray(T_null, dtype=np.float64),
            "bandwidths": bandwidths, "median": median}
