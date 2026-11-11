#!/usr/bin/env python3
"""Block 20: the evaluation's quantities, from block 19's samples
(specs/SPEC.md sec. 2.5 and block 20; D-126, D-127).

    python3 hpc/pipeline/evaluation/ev_metrics.py --config hpc/pipeline/projects/synth/synth.json \\
        --eval-config hpc/pipeline/projects/synth/synth_eval.json [--ensemble <q>:<arm> ...] [--ncpus 8]

Reads <out_root>/samples/ and writes under <out_root>/metrics/:
    <stem>_<D>.json, .npz   one ensemble on one data set D (sim, R0, R3, R3dag):
                            the NLL, the HPD expected coverage with its band, SBC,
                            TARP (x-dependent and random references), the
                            recovery, the posterior-predictive check, L-C2ST,
                            and on a cohort the members' embeddings against
                            D_sim's (MMD^2, PCA); the .npz holds the per-row
                            arrays and curves the figures draw
    paired_<q>_<D>.json, .npz  a search with both arms' ensembles on a data set
                            they share: E3, the donor-paired difference, its
                            donor-cluster bootstrap and G.2's verdict
    search_<q>.json         the search: its trials in proposal order with their
                            arm, nll and L_pseudo_real and the running minimum
                            of nll per arm; every member's learning curve; the
                            corner plots' example donors per cohort (Q-E6)
    METRICS.json            the index: the files, the ensembles reported, the
                            ones left out and why
Pass line: EVAL METRICS OK: ...; exit 2 on the arguments, 3 on a refusal.

The estimators are sec. 2.5's; the library calls are hpc/npe_diagnostics.py
(SBC, the Holm family, TARP, the expected coverage's KS test, the embedding
overlap), hpc/bootstrap_paired.py (the paired bootstrap), sbi 0.27's LC2ST,
scikit-learn's PCA and scipy's binomial and Kolmogorov distributions. The
ensembles are independent, so each one's quantities are computed in a
worker process (`--ncpus`), one thread each; the paired comparisons and the
searches after them.

Pure ASCII, LF only.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from multiprocessing import get_context

import _evalpath  # noqa: F401

import numpy as np
from scipy.stats import binom, kstwo

import ev_config
import ev_plan
from stages.synthetic import RunView

import bootstrap_paired as BP
import npe_diagnostics as ND
from bench_burst_provider import BENCH_AXES

EXIT_ARGS, EXIT_REFUSE = 2, 3
AXES = tuple(a[0] for a in BENCH_AXES)
STATS = ("mean", "sd", "max", "q90", "burst_frac", "acf1", "skew", "roughness")
BETAS = np.round(np.linspace(0.0, 1.0, 101), 10)       # the credibility grid of (E2)
SIGMA0_SQ = 1.0 / 12.0                                 # the box prior's marginal variance
RANK_BINS = 20                                         # SBC's rank histogram
HIST_LEVEL = 0.99                                      # its per-bin band (Talts et al.'s)
MMD_NULL = 500                                         # embedding_overlap's null draws
SCHEMA = "eval_metrics/1"


# =============================================================== the quantities
# Pure functions of arrays; no file is read or written here.

def hpd_ranks(logq_draws, logq_ref):
    """gamma_i = (1/n_post) #{s : log q(theta^(s)_i | x_i) > log q(theta^ref_i | x_i)},
    (n,) float64 (sec. 2.5); rows with logq_ref = -inf give 1."""
    lq = np.asarray(logq_draws, dtype=np.float64)
    ref = np.asarray(logq_ref, dtype=np.float64)
    return np.mean(lq > ref[:, None], axis=1)


def coverage_curve(gamma, betas=BETAS):
    """C(beta) = (1/n) #{i : gamma_i <= beta} on the grid (E2)."""
    g = np.asarray(gamma, dtype=np.float64)
    return (g[None, :] <= np.asarray(betas)[:, None]).mean(axis=1)


def binomial_band(n, betas, level):
    """The pointwise band of C(beta) for a calibrated estimator on n
    independent rows: n C(beta) ~ Binomial(n, beta)."""
    a = (1.0 - level) / 2.0
    return binom.ppf(a, n, betas) / n, binom.ppf(1.0 - a, n, betas) / n


def donor_bootstrap_band(gamma, donor, betas, level, n_boot, seed):
    """The pointwise band of C(beta) from n_boot resamples of the donors
    (every row of a drawn donor; the ratio estimator over the rows)."""
    g = np.asarray(gamma, dtype=np.float64)
    _u, codes = np.unique(np.asarray(donor), return_inverse=True)
    G = int(codes.max()) + 1
    hit = (g[:, None] <= np.asarray(betas)[None, :]).astype(np.float64)
    counts = np.zeros((G, len(betas)))
    np.add.at(counts, codes, hit)
    sizes = np.bincount(codes, minlength=G).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    M = rng.multinomial(G, np.full(G, 1.0 / G), size=int(n_boot)).astype(np.float64)
    curves = (M @ counts) / (M @ sizes)[:, None]
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(curves, [a, 1.0 - a], axis=0)
    return lo, hi


def coverage_summary(curve, betas=BETAS):
    """The signed area between C(beta) and the diagonal (negative:
    overconfident), its largest deviation, and C at four levels."""
    dev = np.asarray(curve) - np.asarray(betas)
    area = float(np.sum((dev[1:] + dev[:-1]) * np.diff(betas)) / 2.0)
    at = dict(("%.2f" % b, float(curve[int(round(b * 100))])) for b in (0.5, 0.68, 0.9, 0.95))
    return {"signed_area": area, "max_abs_dev": float(np.max(np.abs(dev))), "at": at}


def sbc_block(theta_ref, draws, level, seed):
    """SBC per axis (npe_diagnostics), the Holm family at 1 - level, the
    ECDF differences of the normalised ranks with the KS band, and the rank
    histogram with its per-bin binomial band."""
    th = np.asarray(theta_ref, dtype=np.float64)
    ps = np.asarray(draws, dtype=np.float64)
    n, n_post = ps.shape[0], ps.shape[1]
    res = ND.simulation_based_calibration(th, ps, param_names=AXES, seed=int(seed))
    alpha = round(1.0 - level, 10)
    fam = ND.holm_bonferroni([r.ks_pvalue for r in res], alpha=alpha, names=AXES)
    ranks = np.stack([r.ranks for r in res], axis=1).astype(np.int64)        # (n, 10)
    u = ranks / float(n_post)
    ecdf = (u[:, :, None] <= BETAS[None, None, :]).mean(axis=0)            # (10, 101)
    edges = np.linspace(0.0, n_post + 1.0, RANK_BINS + 1)
    counts = np.stack([np.histogram(ranks[:, k], bins=edges)[0] for k in range(ranks.shape[1])])
    per_bin = np.histogram(np.arange(n_post + 1), bins=edges)[0] / float(n_post + 1)
    a = (1.0 - HIST_LEVEL) / 2.0
    arrays = {"sbc_ranks": ranks, "sbc_ecdf_diff": (ecdf - BETAS[None, :]).astype(np.float32),
              "sbc_hist_counts": counts.astype(np.int64), "sbc_hist_edges": edges,
              "sbc_hist_expected": n * per_bin, "sbc_hist_lo": binom.ppf(a, n, per_bin),
              "sbc_hist_hi": binom.ppf(1.0 - a, n, per_bin)}
    summary = {"n_rows": int(n), "n_post": int(n_post),
               "ks_p": [float(r.ks_pvalue) for r in res],
               "mean_rank_frac": [float(r.mean_rank_frac) for r in res],
               "verdict": [r.verdict for r in res],
               "holm_adjusted_p": [float(x) for x in fam.adjusted],
               "holm_rejected": [bool(x) for x in fam.rejected], "holm_alpha": alpha,
               "n_rejected": int(fam.n_rejected),
               "ks_band_halfwidth": float(kstwo.ppf(level, n)) if n > 0 else None,
               "hist_level": HIST_LEVEL}
    return summary, arrays


def tarp_block(theta_ref, draws, stats, seed):
    """TARP (npe_diagnostics.tarp) with reference points read out linearly
    from the windows' eight statistics (primary) and drawn uniformly."""
    out, arrays = {}, {}
    for mode in ("x", "random"):
        r = ND.tarp(theta_ref, draws, Z=(stats if mode == "x" else None), reference_mode=mode,
                    seed=int(seed))
        out[mode] = {"ks_p": float(r.ks_pvalue), "max_dev": float(r.max_deviation),
                     "reference_mode": r.reference_mode, "notes": list(r.notes)}
        arrays["tarp_%s_f" % mode] = np.asarray(r.ecdf_x, dtype=np.float64)
    return out, arrays


def recovery_arrays(theta_ref, draws):
    """Per row and axis: the posterior mean, SD, 5 % and 95 % quantiles, the
    z-score (mhat - theta^ref) / sigma and the contraction 1 - sigma^2 /
    sigma_0^2."""
    d = np.asarray(draws, dtype=np.float64)
    mhat = d.mean(axis=1)
    sigma = d.std(axis=1, ddof=1)
    q05, q95 = np.quantile(d, [0.05, 0.95], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = (mhat - np.asarray(theta_ref, dtype=np.float64)) / sigma
    return {"mhat": mhat, "sigma": sigma, "q05": q05, "q95": q95, "zeta": zeta,
            "contraction": 1.0 - sigma ** 2 / SIGMA0_SQ}


def recovery_summary(theta_ref, rec, rows):
    """Per axis over `rows`: the contraction 1 - mean sigma^2 / (1/12), the
    90 % interval's coverage, the z-scores' mean and SD, the RMSE of the
    posterior mean, the mean posterior SD."""
    t = np.asarray(theta_ref, dtype=np.float64)[rows]
    s = rec["sigma"][rows]
    z = rec["zeta"][rows]
    inside = (rec["q05"][rows] <= t) & (t <= rec["q95"][rows])
    return {"n_rows": int(np.sum(rows)), "axes": list(AXES),
            "contraction": [float(x) for x in 1.0 - np.mean(s ** 2, axis=0) / SIGMA0_SQ],
            "interval90_coverage": [float(x) for x in inside.mean(axis=0)],
            "zeta_mean": [float(x) for x in np.nanmean(z, axis=0)],
            "zeta_sd": [float(x) for x in np.nanstd(z, axis=0, ddof=1)],
            "rmse": [float(x) for x in np.sqrt(np.mean((rec["mhat"][rows] - t) ** 2, axis=0))],
            "sigma_mean": [float(x) for x in s.mean(axis=0)], "sigma0": math.sqrt(SIGMA0_SQ)}


def nll_block(logq_ref, logq_ref_members, in_box, donor):
    """The NLL over the in-box rows, its donor means and each member's."""
    l = -np.asarray(logq_ref, dtype=np.float64)
    ok = np.asarray(in_box, dtype=bool) & np.isfinite(l)
    lm = -np.asarray(logq_ref_members, dtype=np.float64)
    donors = np.unique(donor)
    lbar = np.array([np.mean(l[ok & (donor == d)]) if np.any(ok & (donor == d)) else np.nan
                     for d in donors])
    members = [float(np.mean(lm[ok, m])) for m in range(lm.shape[1])] if ok.any() else []
    summary = {"n_rows": int(ok.sum()), "n_out_of_box": int((~ok).sum()),
               "mean": float(np.mean(l[ok])) if ok.any() else None,
               "members": members,
               "members_sd": float(np.std(members, ddof=1)) if len(members) > 1 else None,
               "n_donors": int(np.sum(np.isfinite(lbar)))}
    return summary, {"nll_rows": np.where(ok, l, np.nan), "donor_ids": donors.astype(np.int64),
                     "donor_nll": lbar}


def ppc_block(obs, sim):
    """Per row and statistic: the rank of the observed value among the
    simulated ones, the simulated 5-95 % range and whether the observed value
    lies outside it; per statistic the fraction outside. Rows with a
    non-finite statistic are left out and counted."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    ok = np.all(np.isfinite(obs), axis=1) & np.all(np.isfinite(sim), axis=(1, 2))
    rank = (sim < obs[:, None, :]).sum(axis=1)
    lo, hi = np.quantile(sim, [0.05, 0.95], axis=1)
    outside = (obs < lo) | (obs > hi)
    frac = outside[ok].mean(axis=0) if ok.any() else np.full(obs.shape[1], np.nan)
    summary = {"n_rows": int(ok.sum()), "n_nonfinite": int((~ok).sum()),
               "n_sim_per_row": int(sim.shape[1]), "stats": list(STATS),
               "frac_outside_90": [float(x) for x in frac]}
    return summary, {"ppc_obs": obs, "ppc_rank": rank.astype(np.int64), "ppc_lo": lo,
                     "ppc_hi": hi, "ppc_outside": outside, "ppc_ok": ok}


def embedding_block(z_sim, z_coh, cls_sim, cls_coh, seed):
    """Per member: the MMD^2 of its embeddings of the cohort against D_sim's
    (embedding_overlap: Gaussian kernel at the median bandwidth, its null
    from D_sim alone) and the first two principal components of the pooled
    embeddings."""
    from sklearn.decomposition import PCA
    M = z_sim.shape[0]
    rows = []
    coords, evr = [], []
    for m in range(M):
        r = ND.embedding_overlap(z_sim[m], z_coh[m], n_null=MMD_NULL, seed=int(seed) + m)
        rows.append({"mmd": float(r.mmd), "mmd2": float(r.mmd) ** 2, "p": float(r.p_value),
                     "null_q95": float(r.null_quantile_95), "bandwidth": float(r.bandwidth),
                     "notes": list(r.notes)})
        pooled = np.concatenate([z_sim[m], z_coh[m]]).astype(np.float64)
        pca = PCA(n_components=2, random_state=int(seed)).fit(pooled)
        coords.append(pca.transform(pooled))
        evr.append(pca.explained_variance_ratio_)
    source = np.r_[np.zeros(z_sim.shape[1], dtype=np.int64), np.ones(z_coh.shape[1], dtype=np.int64)]
    return {"members": rows, "n_null": MMD_NULL}, {
        "pca_coords": np.stack(coords).astype(np.float32), "pca_evr": np.stack(evr),
        "pca_source": source, "pca_cls": np.r_[np.asarray(cls_sim), np.asarray(cls_coh)]}


def paired_block(l_term, l_a1, donor, ok, sd_term, sd_a1, n_boot, level, seed):
    """E3 on the rows `ok`: d_i = l_i(E_term) - l_i(E_A1), the donor-cluster
    bootstrap (scheme `all`, one copy of each donor), sigma_seed = the larger
    of the two arms' SDs over members, and G.2's two clauses."""
    d = np.asarray(l_term, dtype=np.float64)[ok] - np.asarray(l_a1, dtype=np.float64)[ok]
    g = np.asarray(donor)[ok]
    res = BP.paired_bootstrap(d, groups=g, within="all", n_boot=int(n_boot), level=level,
                              seed=int(seed))
    sigma_seed = max(float(sd_term), float(sd_a1))
    beyond = abs(res.estimate) > 2.0 * sigma_seed
    excludes = res.ci[0] > 0.0 or res.ci[1] < 0.0
    donors = np.unique(g)
    dd = np.array([d[g == x].mean() for x in donors])
    se = np.array([d[g == x].std(ddof=1) / math.sqrt(np.sum(g == x)) if np.sum(g == x) > 1
                   else np.nan for x in donors])
    summary = {"n_rows": int(d.size), "n_donors": int(donors.size), "Dbar": float(res.estimate),
               "ci": [float(res.ci[0]), float(res.ci[1])], "se": float(res.se),
               "level": level, "n_boot": int(n_boot), "scheme": "all",
               "sigma_seed": sigma_seed, "two_sigma_seed": 2.0 * sigma_seed,
               "beyond_2_sigma_seed": bool(beyond), "ci_excludes_0": bool(excludes),
               "resolved": bool(beyond and excludes),
               "lower_nll": ("term arm" if res.estimate < 0 else "A1") if beyond and excludes
               else None}
    return summary, {"donor_ids": donors.astype(np.int64), "delta_d": dd, "se_d": se}


def example_donors(donor_ids, donor_nll, donor_cls):
    """Per condition, the donor of median lbar_d (Q-E6): the lower median of
    an even count, ties broken by the donor id. [(cls, donor, lbar)]."""
    donor_ids = np.asarray(donor_ids)
    donor_nll = np.asarray(donor_nll, dtype=np.float64)
    donor_cls = np.asarray(donor_cls)
    ok = np.isfinite(donor_nll)
    out = []
    for c in sorted(set(donor_cls[ok].tolist())):
        sel = np.flatnonzero(ok & (donor_cls == c))
        order = sel[np.lexsort((donor_ids[sel], donor_nll[sel]))]
        k = order[(order.size - 1) // 2]
        out.append({"cls": int(c), "donor": int(donor_ids[k]), "lbar": float(donor_nll[k])})
    return out


def running_min(values):
    """The running minimum of a sequence, NaN kept out."""
    out, cur = [], float("inf")
    for v in values:
        if v is not None and np.isfinite(v):
            cur = min(cur, float(v))
        out.append(cur if np.isfinite(cur) else None)
    return out


# =============================================================== reading, writing

def _clean(obj):
    """JSON-ready: numpy scalars to Python, NaN and inf to None."""
    if isinstance(obj, dict):
        return dict((str(k), _clean(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_clean(obj), fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def write_npz(path, arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def datasets_of(prov):
    """The data sets of a sample provenance, in its order (sim first)."""
    out = []
    stem = prov["stem"]
    for f in prov["files"]:
        b = os.path.basename(f)
        if b.endswith("_cal.npz"):
            continue
        out.append(b[len(stem) + 1:-len(".npz")])
    return out


def _limit_threads():
    """One thread per worker: torch's and the BLAS libraries' (sklearn's MLP)."""
    import torch
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
        _limit_threads.keep = threadpool_limits(limits=1)
    except ImportError:
        pass


# =============================================================== one ensemble

def lc2st_fit(cal, n_null, seed):
    """sbi's LC2ST on the calibration file: theta of the calibration bank
    against one ensemble draw per window, the windows' statistics as x
    (z-scored), `n_null` permutation trials. The seed reaches the
    classifiers only through train_on_observed_data(seed=...), which sets
    their random_state for the null trials too (sbi 0.27: the constructor's
    seed seeds the folds alone, and the classifiers were unseeded)."""
    import torch
    from sbi.diagnostics.lc2st import LC2ST
    lc = LC2ST(prior_samples=torch.as_tensor(cal["theta"], dtype=torch.float32),
               xs=torch.as_tensor(cal["stats"], dtype=torch.float32),
               posterior_samples=torch.as_tensor(cal["draws"], dtype=torch.float32),
               seed=int(seed), num_trials_null=int(n_null), z_score=True)
    lc.train_on_observed_data(seed=int(seed), verbosity=0)
    lc.train_under_null_hypothesis(verbosity=0)
    return lc


def lc2st_eval(lc, draws, stats, rows, n_eval):
    """The local statistic and its p-value at each row of `rows`, with
    n_eval of its draws spread over the members."""
    import torch
    n_post = draws.shape[1]
    idx = np.arange(n_eval) * (n_post // n_eval)
    t, p = [], []
    for i in rows:
        th = torch.as_tensor(np.ascontiguousarray(draws[i, idx]), dtype=torch.float32)
        xo = torch.as_tensor(stats[i][None, :], dtype=torch.float32)
        t.append(lc.get_statistic_on_observed_data(theta_o=th, x_o=xo))
        p.append(lc.p_value(theta_o=th, x_o=xo))
    return np.asarray(t, dtype=np.float64), np.asarray(p, dtype=np.float64)


def ensemble_metrics(task):
    """Every quantity of one ensemble on each of its data sets; writes its
    files and returns (stem, summary per data set)."""
    cfg_path, ecfg_path, name, mdir = task
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    e = ev_plan.find(cfg, name)
    prov = json.load(open(ev_plan.sample_json(ecfg, e.stem)))
    sdir = os.path.dirname(ev_plan.sample_json(ecfg, e.stem))
    level, nb = ecfg.level, ecfg.bootstrap_n
    S = lambda *p: ev_plan.seed_of(ecfg.seed, "metrics", e.stem, *p)      # noqa: E731
    t0 = time.time()
    cal = np.load(os.path.join(sdir, "%s_cal.npz" % e.stem))
    lc = lc2st_fit(cal, ecfg.lc2st_null, S("lc2st"))
    data = dict((D, dict(np.load(os.path.join(sdir, "%s_%s.npz" % (e.stem, D)))))
                for D in datasets_of(prov))           # every array read once
    out = {}
    for D, z in data.items():
        n = z["theta_ref"].shape[0]
        in_box = z["in_box"]
        calib = z["calib"]
        donor = z["donor"]
        hpd_rows = calib & in_box
        js = {"schema": SCHEMA, "ensemble": e.name, "stem": e.stem, "arm": e.arm,
              "search": e.search["name"], "kind": e.kind, "dataset": D, "n_rows": int(n),
              "n_calib": int(calib.sum()), "n_in_box": int(in_box.sum()),
              "n_donors": int(np.unique(donor).size), "M": int(z["logq_ref_members"].shape[1]),
              "n_post": int(z["draws"].shape[1]), "level": level,
              "sample": ev_plan.sample_json(ecfg, e.stem)}
        arr = {"rows": z["rows"], "donor": donor, "cls": z["cls"], "in_box": in_box,
               "calib": calib, "theta_ref": z["theta_ref"], "betas": BETAS}
        # the NLL
        s, a = nll_block(z["logq_ref"], z["logq_ref_members"], in_box, donor)
        js["nll"] = s
        arr.update(a)
        # the HPD expected coverage, the ensemble's and each member's
        gamma = np.where(in_box, hpd_ranks(z["logq_draws"], z["logq_ref"]), np.nan)
        g = gamma[hpd_rows]
        curve = coverage_curve(g)
        if D == "sim":
            lo, hi = binomial_band(int(g.size), BETAS, level)
            band = "binomial: where a calibrated estimator's curve lies, one row per donor"
        else:
            lo, hi = donor_bootstrap_band(g, donor[hpd_rows], BETAS, level, nb, S(D, "band"))
            band = "donor-cluster bootstrap of the curve (%d resamples)" % nb
        dm = z["draw_member"]
        mem_curves = []
        for m in range(js["M"]):
            own = z["logq_draws_own"][:, dm == m]
            gm = hpd_ranks(own[hpd_rows], z["logq_ref_members"][hpd_rows, m])
            mem_curves.append(coverage_curve(gm))
        ec = ND.expected_coverage(z["logq_ref"][hpd_rows], z["logq_draws"][hpd_rows],
                                  seed=S(D, "ec")) if g.size else None
        js["coverage"] = dict(coverage_summary(curve), n_rows=int(g.size), band=band,
                              ks_p=(float(ec.ks_pvalue) if ec else None),
                              verdict=(ec.verdict if ec else None),
                              members=[coverage_summary(c) for c in mem_curves])
        arr.update({"gamma": gamma, "coverage": curve, "coverage_lo": lo, "coverage_hi": hi,
                    "coverage_members": np.stack(mem_curves)})
        # SBC and TARP over the calibration rows (every row, theta-dagger outside the box too)
        s, a = sbc_block(z["theta_ref"][calib], z["draws"][calib], level, S(D, "sbc"))
        js["sbc"] = s
        arr.update(a)
        s, a = tarp_block(z["theta_ref"][calib], z["draws"][calib], z["stats"][calib], S(D, "tarp"))
        js["tarp"] = s
        arr.update(a)
        # the recovery
        rec = recovery_arrays(z["theta_ref"], z["draws"])
        js["recovery"] = recovery_summary(z["theta_ref"], rec, calib)
        arr.update(dict((k, v.astype(np.float32)) for k, v in rec.items()))
        # the posterior-predictive check
        if z["ppc_rows"].size:
            s, a = ppc_block(z["stats"][z["ppc_rows"]], z["ppc_stats"])
            js["ppc"] = s
            arr.update(a)
            arr["ppc_rows"] = z["ppc_rows"]
        # L-C2ST at the cohort's windows, at D_sim's PPC rows as the reference
        if not D.endswith("dag"):
            ev_rows = z["ppc_rows"] if D == "sim" else np.arange(n)
            t, p = lc2st_eval(lc, z["draws"], z["stats"], ev_rows, ecfg.lc2st_eval_draws)
            dn = donor[ev_rows]
            dids = np.unique(dn)
            js["lc2st"] = {"n_rows": int(ev_rows.size), "n_null": int(ecfg.lc2st_null),
                           "n_eval_draws": int(ecfg.lc2st_eval_draws), "z_score": True,
                           "stat_mean": float(np.mean(t)) if t.size else None,
                           "frac_p_below_alpha": (float(np.mean(p < round(1.0 - level, 10)))
                                                  if p.size else None),
                           "alpha": round(1.0 - level, 10), "n_cal": int(cal["theta"].shape[0]),
                           "stat_q95": float(np.quantile(t, 0.95)) if t.size else None}
            arr.update({"lc2st_rows": ev_rows.astype(np.int64), "lc2st_stat": t, "lc2st_p": p,
                        "lc2st_donor_ids": dids.astype(np.int64),
                        "lc2st_donor_stat": np.array([t[dn == d].mean() for d in dids])})
        # the embeddings: a cohort's against D_sim's
        if D != "sim" and not D.endswith("dag"):
            zs = data["sim"]
            s, a = embedding_block(zs["z"], z["z"], zs["cls"], z["cls"], S(D, "mmd"))
            js["embedding"] = s
            arr.update(a)
        write_json(os.path.join(mdir, "%s_%s.json" % (e.stem, D)), js)
        write_npz(os.path.join(mdir, "%s_%s.npz" % (e.stem, D)), arr)
        out[D] = js
    return e.stem, out, round(time.time() - t0, 1)


# =============================================================== across ensembles

def member_sd(z):
    """The SD over members of the members' mean NLL on the in-box rows."""
    ok = z["in_box"] & np.isfinite(z["logq_ref"])
    m = -np.mean(z["logq_ref_members"][ok], axis=0)
    return float(np.std(m, ddof=1)) if m.size > 1 else 0.0


def paired_all(cfg, ecfg, reported, mdir, log):
    """E3 for every search with both arms' ensembles reported, on each data
    set the two share."""
    files = []
    by = {}
    for e in reported:
        by.setdefault(e.search["name"], []).append(e)
    for q in cfg.data["synthetic"]["searches"]:
        ens = by.get(q["name"], [])
        arms = dict((e.arm, e) for e in ens)
        if len(arms) < 2 or "A1" not in arms:
            continue
        term = [a for a in arms if a != "A1"][0]
        eT, eA = arms[term], arms["A1"]
        sd = ev_plan.sample_json
        dsT = datasets_of(json.load(open(sd(ecfg, eT.stem))))
        dsA = datasets_of(json.load(open(sd(ecfg, eA.stem))))
        sdir = os.path.dirname(sd(ecfg, eT.stem))
        for D in [d for d in dsT if d in dsA]:
            zT = np.load(os.path.join(sdir, "%s_%s.npz" % (eT.stem, D)))
            zA = np.load(os.path.join(sdir, "%s_%s.npz" % (eA.stem, D)))
            if not (np.array_equal(zT["rows"], zA["rows"])
                    and np.array_equal(zT["theta_ref"], zA["theta_ref"])):
                raise ev_plan.RefuseError("%s and %s do not share the rows of %s"
                                          % (eT.name, eA.name, D))
            ok = zT["in_box"] & np.isfinite(zT["logq_ref"]) & np.isfinite(zA["logq_ref"])
            sT, sA = member_sd(zT), member_sd(zA)
            s, a = paired_block(-zT["logq_ref"], -zA["logq_ref"], zT["donor"], ok, sT, sA,
                                ecfg.bootstrap_n, ecfg.level,
                                ev_plan.seed_of(ecfg.seed, "paired", q["name"], D))
            s.update({"schema": SCHEMA, "search": q["name"], "dataset": D, "term_arm": term,
                      "ensembles": {"term": eT.name, "A1": eA.name},
                      "sd_members": {term: sT, "A1": sA}})
            base = os.path.join(mdir, "paired_%s_%s" % (q["name"], D))
            write_json(base + ".json", s)
            write_npz(base + ".npz", a)
            files += [base + ".json", base + ".npz"]
            log("  paired %s on %s: Dbar %+.4f [%+.4f, %+.4f], 2 sigma_seed %.4f -> %s"
                % (q["name"], D, s["Dbar"], s["ci"][0], s["ci"][1], s["two_sigma_seed"],
                   "resolved" if s["resolved"] else "not resolved"))
    return files


def examples_of_search(cfg, q, reported, mdir):
    """The corner plots' example donors of search q on each cohort its
    ensembles are scored on (Q-E6): per condition the donor of median lbar_d
    under the search's A1 ensemble -- or, while that one is not reported,
    under S-A1's -- so that the choice is not tuned to an arm; with each
    donor's first row in the data set."""
    mine = [e for e in reported if e.search["name"] == q["name"]]
    ref = [e for e in mine if e.arm == "A1"]
    ref = ref[0] if ref else next((e for e in reported if e.search["campaign"] == "S-A1"), None)
    out = {}
    if ref is None or not mine:
        return out
    for c in mine[0].cohorts:
        p = os.path.join(mdir, "%s_%s.npz" % (ref.stem, c))
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        first = dict((int(d), i) for i, d in reversed(list(enumerate(z["donor"]))))
        cls_of = np.array([z["cls"][first[int(d)]] for d in z["donor_ids"]])
        ex = example_donors(z["donor_ids"], z["donor_nll"], cls_of)
        for x in ex:
            x["row"] = int(first[x["donor"]])
        out[c] = {"chosen_by": ref.name, "donors": ex}
    return out


def search_summary(cfg, q, reported):
    """The search's trials in proposal order (pipeline_rounds.json), with
    their arm, nll and L_pseudo_real and the running minimum of nll per arm;
    its finalist; every reported member's learning curve."""
    v = RunView(cfg, q)
    sdir = os.path.join(v.data["paths"]["tune_root"], q["campaign"])
    rounds = json.load(open(os.path.join(sdir, "pipeline_rounds.json"))).get("rounds") or []
    trials = []
    for r in rounds:
        for tid in r.get("trial_ids") or []:
            p = os.path.join(sdir, "trials", "%s.json" % tid)
            rec = json.load(open(p)) if os.path.isfile(p) else {}
            trials.append({"trial_id": tid, "round": r.get("round"), "arm": rec.get("arm"),
                           "status": rec.get("status", "absent"), "nll": rec.get("nll"),
                           "L_pseudo_real": rec.get("L_pseudo_real")})
    for arm in sorted(set(t["arm"] for t in trials if t["arm"])):
        rm = running_min([t["nll"] if (t["arm"] == arm and t["status"] == "ok") else None
                          for t in trials])
        for t, x in zip(trials, rm):
            t.setdefault("running_min_nll", {})[arm] = x
    sc = os.path.join(sdir, "SEARCH_COMPLETE.json")
    fin = (json.load(open(sc)).get("finalist") or {}).get("trial_id") if os.path.isfile(sc) else None
    ens = {}
    for e in reported:
        if e.search["name"] != q["name"]:
            continue
        mem = []
        for rp in e.records:
            rec = json.load(open(rp))
            hist = rec.get("history") or []
            mem.append({"seed": rec.get("seed"), "L": rec.get("L"),
                        "L_pseudo_real": rec.get("L_pseudo_real"),
                        "epoch": [h.get("epoch") for h in hist],
                        "npe": [h.get("npe") for h in hist],
                        "val_npe": [h.get("val_npe") for h in hist]})
        ens[e.name] = {"arm": e.arm, "kind": e.kind, "members": mem}
    return {"schema": SCHEMA, "search": q["name"], "campaign": q["campaign"], "real": q["real"],
            "folder": sdir, "trials": trials, "finalist": fin, "ensembles": ens}


def compute(cfg_path, ecfg_path, names=None, ncpus=None, log=print):
    """Every quantity of every sampled ensemble (or of `names`), then the
    paired comparisons and the searches; returns the index."""
    t0 = time.time()
    cfg = ev_plan.load_project(cfg_path)
    ecfg = ev_config.load(ecfg_path)
    mdir = ecfg.sub("metrics")
    os.makedirs(mdir, exist_ok=True)
    found, missing = ev_plan.ensembles(cfg, with_missing=True)
    left_out = [{"ensemble": "%s (%s)" % (q, os.path.basename(f)), "why": w} for q, f, w in missing]
    todo = []
    for e in found:
        if names and e.name not in names:
            continue
        pj = ev_plan.sample_json(ecfg, e.stem)
        if not os.path.isfile(pj):
            left_out.append({"ensemble": e.name, "why": "not sampled (%s absent)" % pj})
            continue
        prov = json.load(open(pj))
        if prov.get("sample_params") != ecfg.sample_params():
            left_out.append({"ensemble": e.name, "why": "sampled with other values: %s"
                             % prov.get("sample_params")})
            continue
        todo.append(e)
    if names:
        unknown = sorted(set(names) - set(e.name for e in found))
        if unknown:
            raise ev_plan.RefuseError("no complete ensemble %s" % ", ".join(unknown))
    if not todo:
        raise ev_plan.RefuseError("no ensemble has a current sample under %s"
                                  % ecfg.sub("samples"))
    n_w = max(1, min(int(ncpus or ecfg.ncpus), len(todo)))
    log("metrics: %d ensemble(s) in %d worker(s): %s" % (len(todo), n_w,
                                                          ", ".join(e.name for e in todo)))
    tasks = [(cfg.path, ecfg.path, e.name, mdir) for e in todo]
    if n_w == 1:
        _limit_threads()
        results = [ensemble_metrics(t) for t in tasks]
    else:
        with get_context("fork").Pool(n_w, initializer=_limit_threads) as pool:
            results = pool.map(ensemble_metrics, tasks, chunksize=1)
    per = {}
    for stem, out, secs in results:
        per[stem] = sorted(out)
        log("  %s: %s (%.0f s)" % (stem, ", ".join(
            "%s NLL %.3f" % (D, js["nll"]["mean"]) if js["nll"]["mean"] is not None
            else "%s NLL -" % D for D, js in sorted(out.items())), secs))
    files = []
    for stem, ds in per.items():
        for D in ds:
            files += [os.path.join(mdir, "%s_%s.json" % (stem, D)),
                      os.path.join(mdir, "%s_%s.npz" % (stem, D))]
    files += paired_all(cfg, ecfg, todo, mdir, log)
    searches = []
    for q in cfg.data["synthetic"]["searches"]:
        if not any(e.search["name"] == q["name"] for e in todo):
            continue
        p = os.path.join(mdir, "search_%s.json" % q["name"])
        summ = search_summary(cfg, q, todo)
        summ["example_donors"] = examples_of_search(cfg, q, todo, mdir)
        write_json(p, summ)
        files.append(p)
        searches.append(q["name"])
    index = {"schema": SCHEMA, "config": cfg.path, "eval_config": ecfg.path,
             "ensembles": dict((e.name, {"stem": e.stem, "arm": e.arm, "kind": e.kind,
                                         "search": e.search["name"], "datasets": per[e.stem]})
                               for e in todo),
             "left_out": left_out, "searches": searches, "files": files,
             "report_inputs": ev_plan.report_inputs(cfg, ecfg),
             "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(mdir, "METRICS.json"), index)
    n_pair = sum(1 for f in files if os.path.basename(f).startswith("paired_")) // 2
    log("EVAL METRICS OK: %d ensemble(s), %d data-set file(s), %d paired comparison(s), "
        "%d search(es), %d left out (%.0f s)"
        % (len(todo), sum(len(v) for v in per.values()), n_pair, len(searches), len(left_out),
           time.time() - t0))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the pipeline's project file (<tag>.json)")
    ap.add_argument("--eval-config", required=True, help="the evaluation's file (<tag>_eval.json)")
    ap.add_argument("--ensemble", action="append", default=None,
                    help="<search>:<arm>; repeat for several (default: every sampled ensemble)")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the eval config's ncpus")
    args = ap.parse_args(argv)
    try:
        compute(args.config, args.eval_config, names=args.ensemble, ncpus=args.ncpus,
                log=lambda t: print(t, flush=True))
    except (ev_plan.RefuseError, ev_config.EvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
