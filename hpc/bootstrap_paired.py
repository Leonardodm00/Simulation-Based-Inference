#!/usr/bin/env python3
"""
bootstrap_paired.py -- paired bootstrap for a per-row difference under a
grouped split (JOINT_DSN_NPE_PLAN v0.6, S2.4 and S2.4a).

The object
----------
Two arms are scored on the SAME held-out rows, so the per-row difference

    d_i = l_i^(A) - l_i^(B),      l_i = -log q(theta_i | z_i),

is formed and D = E[d] is estimated. Row difficulty cancels in d_i, so the
covariance between arms is never needed. The rows are NOT exchangeable: the
split is grouped (a topology draw on the simulated bank, a culture on the real
one), and rows of one group share 3 of 26 theta axes by construction, so
rho(d) > 0 is expected -- small, but with n_bar ~ 77 even rho = 0.05 gives a
design effect near 5. A row bootstrap then understates the standard error by
1/sqrt(DEFF) and its nominal 95% interval covers the truth far less often, with
nothing in its output saying so. That is why the resampling UNIT is the
decision here, not the interval recipe.

The four schemes (S2.4a) -- all first-class; this module chooses none
-------------------------------------------------------------------------
    none   rows with replacement                    estimand: row-weighted mean
    all    groups; EVERY row of each drawn group,    estimand: row-weighted mean
           ratio estimator sum(d_i) / sum(n_g)                  (PRIMARY)
    mean   groups; the group mean dbar_g             estimand: group-weighted mean
    one    groups; one random row per drawn group    estimand: group-weighted mean,
                                                     variance larger by ~n_g

`all` and `mean` differ in ESTIMAND, not only in variance, whenever group
sizes are unequal -- and they are: the MFR filter strips rows from quiet
networks, so group size correlates with the biology. Report both; agreement
is a sentence, disagreement is a finding.

What this module does NOT do
----------------------------
No verdicts. `compare_schemes` reports all four schemes, rho(d), n0, DEFF and
the tail share side by side; the diagnostic stage decides on the measured
numbers. No I/O beyond the optional CLI. No torch, no bank dependency: it runs
on any stored pair of per-row arrays.

API
---
    paired_bootstrap(d, groups=None, within="all", n_boot=2000, level=0.95,
                     seed=0)                          -> Result
    intraclass_correlation(d, groups)                 -> Result (rho, n0, deff)
    tail_share(d, top_frac=0.01)                      -> Result
    compare_schemes(d, groups, label=None, ...)       -> Result
    format_comparison(result)                         -> str

`Result` is a dict with attribute access, so both `res["ci"]` / `res.get("ci")`
and `res.ci` / `getattr(icc, "rho")` work -- the two access patterns the
callers use.

Interval: percentile by default (`ci`), with the basic interval (`ci_basic`)
and the bootstrap standard error (`se`) alongside. Group labels may be any
array of hashables (ints from a split manifest, strings from a specs file);
they are re-coded internally.

Pure ASCII, LF only.
"""

import argparse
import math
import sys

import numpy as np

__all__ = ["SCHEMES", "Result", "paired_bootstrap", "intraclass_correlation",
           "tail_share", "compare_schemes", "format_comparison"]

SCHEMES = ("none", "all", "mean", "one")

_UNIT = {"none": "row", "all": "group", "mean": "group", "one": "group"}
_WEIGHT = {"none": "row", "all": "row", "mean": "group", "one": "group"}

# Row draws for the `none` scheme are generated in chunks of at most this many
# indices at a time, so a large evaluation split does not allocate n_boot x N
# integers at once.
_NONE_CHUNK_ELEMENTS = 4_000_000


class Result(dict):
    """A dict whose keys are also attributes.

    `res.ci`, `res["ci"]` and `res.get("ci")` all work. Callers written
    against either convention keep working, which is the point.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


# ---------------------------------------------------------------------------
# Input preparation and guards
# ---------------------------------------------------------------------------

def _as_finite_vector(d, name="d"):
    a = np.asarray(d, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError("%s: need at least 2 rows, got %d" % (name, a.size))
    bad = int(np.sum(~np.isfinite(a)))
    if bad:
        raise ValueError(
            "%s: %d of %d values are not finite. A NaN or infinite per-row "
            "difference is a failed fit or a mismatched split, not a score "
            "to average; drop or fix upstream, never here." % (name, bad,
                                                               a.size))
    return a


def _groups_struct(groups, n):
    """Re-code group labels to 0..G-1 and precompute per-group sums.

    Returns a dict with codes, G, sizes n_g, sums S_g, means dbar_g, and the
    row order sorted by group with per-group start offsets (for `one`).
    """
    g = np.asarray(groups).ravel()
    if g.size != n:
        raise ValueError("groups has %d entries, d has %d rows" % (g.size, n))
    uniq, codes = np.unique(g, return_inverse=True)
    codes = codes.astype(np.int64).ravel()
    G = int(uniq.size)
    sizes = np.bincount(codes, minlength=G).astype(np.int64)
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    return dict(labels=uniq, codes=codes, G=G, sizes=sizes, order=order,
                starts=starts)


def _check_common(within, n_boot, level):
    if within not in SCHEMES:
        raise ValueError("within must be one of %s, got %r"
                         % (", ".join(SCHEMES), within))
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1, got %d" % n_boot)
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError("level must lie in (0, 1), got %r" % level)
    return n_boot, level


# ---------------------------------------------------------------------------
# The paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(d, groups=None, within="all", n_boot=2000, level=0.95,
                     seed=0, return_samples=False):
    """Bootstrap interval for D = E[d] under one resampling scheme.

    Parameters
    ----------
    d : array-like, shape (N,)
        Per-row paired difference, l_i^(A) - l_i^(B). Must be finite.
    groups : array-like, shape (N,), or None
        Group label per row, taken from the split manifest (simulated bank:
        the topology-draw group; real cohort: the specs-file `culture`).
        Required for the grouped schemes; ignored by `none`.
    within : {"none", "all", "mean", "one"}
        The resampling scheme (module docstring). `all` is the primary.
    n_boot : int
        Number of resamples.
    level : float in (0, 1)
        Interval level, e.g. 0.95.
    seed : int
        Seed of the resampling generator. Same seed, same interval.
    return_samples : bool
        Also return the n_boot bootstrap statistics under key `samples`.

    Returns
    -------
    Result with keys: scheme, unit, weighting, estimate, se, ci (percentile,
    a 2-tuple), ci_basic, level, n_boot, n_rows, n_groups, seed.
    """
    n_boot, level = _check_common(within, n_boot, level)
    d = _as_finite_vector(d)
    N = int(d.size)
    rng = np.random.default_rng(int(seed))

    gs = None
    if groups is not None:
        gs = _groups_struct(groups, N)
    if within != "none":
        if gs is None:
            raise ValueError("scheme %r resamples groups; pass `groups` "
                             "(from the split manifest)" % within)
        if gs["G"] < 2:
            raise ValueError("scheme %r needs at least 2 groups, got %d"
                             % (within, gs["G"]))

    if within == "none":
        estimate = float(d.mean())
        stats = np.empty(n_boot, dtype=np.float64)
        chunk = max(1, _NONE_CHUNK_ELEMENTS // N)
        done = 0
        while done < n_boot:
            b = min(chunk, n_boot - done)
            idx = rng.integers(0, N, size=(b, N))
            stats[done:done + b] = d[idx].mean(axis=1)
            done += b
    else:
        G, codes, sizes = gs["G"], gs["codes"], gs["sizes"]
        S = np.bincount(codes, weights=d, minlength=G)      # per-group sums
        n_g = sizes.astype(np.float64)
        dbar = S / n_g                                         # group means
        if within == "all":
            estimate = float(d.mean())                         # = S.sum()/N
            # Multiplicity of each group in a resample of G groups with
            # replacement: Multinomial(G, 1/G). Every row of a drawn group is
            # kept, so the statistic is the ratio sum(m_g S_g)/sum(m_g n_g).
            M = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)
            stats = (M @ S) / (M @ n_g)
        elif within == "mean":
            estimate = float(dbar.mean())
            M = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)
            stats = (M @ dbar) / float(G)
        else:  # "one"
            estimate = float(dbar.mean())
            d_sorted = d[gs["order"]]
            starts = gs["starts"]
            gidx = rng.integers(0, G, size=(n_boot, G))
            u = rng.random((n_boot, G))
            row = starts[gidx] + np.floor(u * sizes[gidx]).astype(np.int64)
            stats = d_sorted[row].mean(axis=1)

    stats = np.asarray(stats, dtype=np.float64)
    se = float(stats.std(ddof=1)) if n_boot > 1 else float("nan")
    alpha = 1.0 - level
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    lo, hi = float(lo), float(hi)
    res = Result(scheme=within, unit=_UNIT[within], weighting=_WEIGHT[within],
                 estimate=estimate, se=se, ci=(lo, hi),
                 ci_basic=(2.0 * estimate - hi, 2.0 * estimate - lo),
                 level=level, n_boot=n_boot, n_rows=N,
                 n_groups=(int(gs["G"]) if gs is not None else None),
                 seed=int(seed))
    if return_samples:
        res["samples"] = stats
    return res


# ---------------------------------------------------------------------------
# Intraclass correlation and design effect
# ---------------------------------------------------------------------------

def intraclass_correlation(d, groups):
    """One-way random-effects ANOVA estimate of rho(d), with n0 and DEFF.

    Model: d_{gi} = mu + a_g + e_{gi}, Var[a_g] = sigma_b^2, Var[e] = sigma_w^2,
    rho = sigma_b^2 / (sigma_b^2 + sigma_w^2). With G groups of sizes n_g and
    N = sum n_g:

        MSB = sum_g n_g (dbar_g - dbar)^2 / (G - 1)
        MSW = sum_g sum_i (d_gi - dbar_g)^2 / (N - G)
        n0  = (N - sum_g n_g^2 / N) / (G - 1)      (unequal-size correction)
        rho = (MSB - MSW) / (MSB + (n0 - 1) MSW)   (clipped to [0, 1])
        DEFF = 1 + (n0 - 1) rho

    n0 equals the common group size under balance. rho_raw (unclipped) is
    returned as well, since a raw value well below zero is itself informative
    about the ANOVA assumptions.

    Degenerate inputs do not raise: N == G (every group one row) leaves MSW
    undefined, and constant d leaves rho as 0/0. Both return rho = nan or 0
    with `degenerate=True` and a `note`, so a caller that computes the
    interval and the ICC in one block keeps the interval.
    """
    d = _as_finite_vector(d)
    N = int(d.size)
    gs = _groups_struct(groups, N)
    G, codes, sizes = gs["G"], gs["codes"], gs["sizes"]
    if G < 2:
        raise ValueError("intraclass_correlation needs at least 2 groups, "
                         "got %d" % G)
    n_g = sizes.astype(np.float64)
    S = np.bincount(codes, weights=d, minlength=G)
    dbar = S / n_g
    grand = float(d.mean())
    n0 = (N - float(np.sum(n_g ** 2)) / N) / (G - 1)
    n_bar = N / float(G)

    base = Result(n_rows=N, n_groups=G, n_bar=n_bar, n0=float(n0))
    if N == G:
        base.update(rho=float("nan"), rho_raw=float("nan"), deff=float("nan"),
                    deff_raw=float("nan"), msb=float("nan"),
                    msw=float("nan"), degenerate=True,
                    note="every group has exactly one row; the within-group "
                         "variance is not estimable and rho is undefined")
        return base

    msb = float(np.sum(n_g * (dbar - grand) ** 2) / (G - 1))
    msw = float(np.sum((d - dbar[codes]) ** 2) / (N - G))
    denom = msb + (n0 - 1.0) * msw
    # Constant d gives MSB and MSW of order 1e-32, not exactly 0, so the 0/0
    # case is detected on the variance SCALE of d rather than by equality.
    var_d = float(np.mean((d - grand) ** 2))
    if not np.isfinite(denom) or denom <= 0.0 \
            or var_d <= 1e-22 * (1.0 + grand * grand):
        base.update(rho=0.0, rho_raw=float("nan"), deff=1.0,
                    deff_raw=float("nan"), msb=msb, msw=msw, degenerate=True,
                    note="d has no variance between or within groups; rho "
                         "is 0/0 and is reported as 0 with DEFF = 1")
        return base
    rho_raw = (msb - msw) / denom
    rho = float(min(1.0, max(0.0, rho_raw)))
    base.update(rho=rho, rho_raw=float(rho_raw),
                deff=float(1.0 + (n0 - 1.0) * rho),
                deff_raw=float(1.0 + (n0 - 1.0) * rho_raw),
                msb=msb, msw=msw, degenerate=False, note="")
    return base


# ---------------------------------------------------------------------------
# Tail diagnostic
# ---------------------------------------------------------------------------

def tail_share(d, top_frac=0.01):
    """How much of sum |d_i| is carried by the largest few rows.

    A mean whose sum is dominated by a handful of rows has no trustworthy
    interval under ANY scheme: the bootstrap distribution is then set by
    whether those rows are drawn, not by the bulk. This reports the share of
    sum |d_i| carried by the top ceil(top_frac * N) rows (`share_top`), the
    share of the single largest row (`share_max`), and the ratio of
    `share_top` to what an equal-weight top set would carry (`ratio`, equal
    to 1 when all |d_i| are equal). No threshold is applied here.
    """
    d = _as_finite_vector(d)
    N = int(d.size)
    top_frac = float(top_frac)
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must lie in (0, 1], got %r" % top_frac)
    a = np.abs(d)
    total = float(a.sum())
    k = int(max(1, math.ceil(top_frac * N)))
    uniform = k / float(N)
    if total <= 0.0:
        return Result(n_rows=N, n_top=k, top_frac=top_frac, share_top=uniform,
                      share_max=1.0 / N, uniform_share=uniform, ratio=1.0,
                      sum_abs=0.0, note="all d_i are zero")
    srt = np.sort(a)[::-1]
    share_top = float(srt[:k].sum() / total)
    share_max = float(srt[0] / total)
    return Result(n_rows=N, n_top=k, top_frac=top_frac, share_top=share_top,
                  share_max=share_max, uniform_share=uniform,
                  ratio=share_top / uniform, sum_abs=total, note="")


# ---------------------------------------------------------------------------
# The comparison across schemes
# ---------------------------------------------------------------------------

def compare_schemes(d, groups, label=None, n_boot=2000, level=0.95, seed=0,
                    top_frac=0.01):
    """All four schemes, the ICC, the design effect and the tail share.

    The `none` and `all` standard errors are related by
    se(none)/se(all) ~= 1/sqrt(DEFF_raw) (S2.4a; exact to O(1/G) with the
    unclipped rho). Both sides of that identity are returned
    (`se_ratio_none_all`, `predicted_ratio`) so a report can show the
    measured design effect and the interval it implies side by side.
    """
    d = _as_finite_vector(d)
    schemes = {}
    for k, s in enumerate(SCHEMES):
        schemes[s] = paired_bootstrap(d, groups=groups, within=s,
                                      n_boot=n_boot, level=level,
                                      seed=int(seed) + k)
    icc = intraclass_correlation(d, groups)
    tail = tail_share(d, top_frac=top_frac)
    se_none, se_all = schemes["none"]["se"], schemes["all"]["se"]
    ratio = (se_none / se_all) if se_all > 0 else float("nan")
    # The identity se(none)/se(all) = 1/sqrt(DEFF) is exact to O(1/G) with
    # the UNCLIPPED rho: when the sample's between-group dispersion falls
    # below the iid value, rho_raw < 0 and the group bootstrap is correctly
    # narrower than the row bootstrap. The clipped DEFF (>= 1) is the design
    # effect for reporting; the raw one is what the interval widths obey.
    deff = icc["deff_raw"]
    pred = (1.0 / math.sqrt(deff)) if (np.isfinite(deff) and deff > 0) \
        else float("nan")
    return Result(label=(label or ""), n_rows=int(d.size),
                  n_groups=icc["n_groups"], schemes=schemes, icc=icc,
                  tail=tail, se_ratio_none_all=float(ratio),
                  predicted_ratio=float(pred), primary="all", level=level,
                  n_boot=int(n_boot), seed=int(seed))


def _f(v, prec=4):
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "nan"
    return ("%." + str(prec) + "f") % v


def format_comparison(res):
    """ASCII table of a `compare_schemes` result."""
    icc, tail = res["icc"], res["tail"]
    pct = int(round(100 * res["level"]))
    lines = []
    if res.get("label"):
        lines.append(str(res["label"]))
    lines.append("rows %d, groups %d, n_bar %s, n0 %s; rho(d) = %s (raw %s), "
                 "DEFF = %s (raw %s)%s"
                 % (res["n_rows"], res["n_groups"], _f(icc["n_bar"], 1),
                    _f(icc["n0"], 1), _f(icc["rho"], 3), _f(icc["rho_raw"], 3),
                    _f(icc["deff"], 2), _f(icc["deff_raw"], 2),
                    ("  [degenerate: %s]" % icc["note"]) if icc["degenerate"]
                    else ""))
    lines.append("tail: top %d row(s) (%.1f%%) carry %.1f%% of sum|d| "
                 "(uniform would be %.1f%%, ratio %s); largest row %.1f%%"
                 % (tail["n_top"], 100 * tail["top_frac"],
                    100 * tail["share_top"], 100 * tail["uniform_share"],
                    _f(tail["ratio"], 2), 100 * tail["share_max"]))
    lines.append("")
    lines.append("| scheme | unit  | weight | estimate | se       | %d%% CI "
                 "(percentile)     | note |" % pct)
    lines.append("|---|---|---|---|---|---|---|")
    for s in SCHEMES:
        r = res["schemes"][s]
        note = "PRIMARY" if s == res.get("primary") else ""
        lines.append("| %-6s | %-5s | %-6s | %8s | %8s | [%8s, %8s] | %s |"
                     % (s, r["unit"], r["weighting"], _f(r["estimate"]),
                        _f(r["se"]), _f(r["ci"][0]), _f(r["ci"][1]), note))
    lines.append("")
    lines.append("se(none)/se(all) = %s   vs   1/sqrt(DEFF_raw) = %s"
                 % (_f(res["se_ratio_none_all"], 3),
                    _f(res["predicted_ratio"], 3)))
    lines.append("`all` and `mean` estimates differ by %s (they share an "
                 "estimand only under equal group sizes)"
                 % _f(res["schemes"]["all"]["estimate"]
                      - res["schemes"]["mean"]["estimate"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: compare two stored per-row arrays
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Paired cluster bootstrap of D = mean(a - b) from two "
                    "stored per-row arrays (e.g. Stage 3 *_perrow.npz).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help=".npz with the first arm's rows")
    p.add_argument("--b", required=True, help=".npz with the second arm's rows")
    p.add_argument("--key", default="nll", help="array key for the per-row "
                   "score in both files (default: nll)")
    p.add_argument("--group-key", default="group",
                   help="array key for the group label, read from --a "
                        "(default: group)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    with np.load(args.a) as za, np.load(args.b) as zb:
        for z, path in ((za, args.a), (zb, args.b)):
            if args.key not in z.files:
                raise SystemExit("%s: no array %r (has %s)"
                                 % (path, args.key, ", ".join(z.files)))
        if args.group_key not in za.files:
            raise SystemExit("%s: no array %r (has %s)"
                             % (args.a, args.group_key, ", ".join(za.files)))
        a, b, g = za[args.key], zb[args.key], za[args.group_key]
        if args.group_key in zb.files and not np.array_equal(g, zb[args.group_key]):
            raise SystemExit("group labels differ between the two files; the "
                             "rows are not paired")
    if a.shape != b.shape:
        raise SystemExit("shape mismatch: %s vs %s; the rows are not paired"
                         % (a.shape, b.shape))
    res = compare_schemes(np.asarray(a, dtype=np.float64)
                          - np.asarray(b, dtype=np.float64), g,
                          label=args.label or ("%s minus %s (%s)"
                                               % (args.a, args.b, args.key)),
                          n_boot=args.n_boot, level=args.level,
                          seed=args.seed)
    print(format_comparison(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
