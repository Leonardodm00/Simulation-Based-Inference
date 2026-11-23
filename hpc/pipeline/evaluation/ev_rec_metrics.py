#!/usr/bin/env python3
"""Block 36: the quantities of a recorded project's evaluation, from the
samples of block 35 (specs/SPEC.md sec. 2.6 and block 36; decisions
D-160..D-165).

    python3 hpc/pipeline/evaluation/ev_rec_metrics.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json [--ensemble <name> ...] [--ncpus N]

Writes under <out_root>/metrics/: per ensemble <stem>_sim.json/.npz and
<stem>_<r>.json/.npz for each recording bank; per comparison
paired_<name>_sim.json/.npz; METRICS.json, the index. Pass line: REC METRICS
OK ...; exit 2 on the arguments, 3 on a refusal.

The quantities of D_sim are block 20's functions (ev_metrics), imported,
never copied, with block 20's seeds (seed_of(seed, "metrics", <stem>, <data
set>, <what>); the paired bootstrap's seed_of(seed, "paired", <name>, "sim"),
block 27's), so that a synthetic project read as recorded with every row a
draw row gives block 20's numbers (D-162). The functions below are the ones
no library has: the edge split (R1), (R2), the well's mixture (R4), the
flagged shares and the gate's size (R9), (R10).

Every output calls mix_g "the mixture over the well's windows", never the
well's posterior (D-160); a window with p < alpha_gate is "flagged by the
gate".

Pure ASCII, LF only.
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context

import _evalpath  # noqa: F401

import numpy as np

import ev_gate
import ev_rec_config
import ev_rec_plan
from ev_plan import RefuseError, seed_of

# block 20's quantities, imported, never copied (block 36, "Rule")
import ev_metrics as EM
from ev_metrics import (BETAS, binomial_band, coverage_curve, coverage_summary, hpd_ranks,
                        member_sd, nll_block, paired_block, recovery_arrays, recovery_summary,
                        sbc_block, tarp_block, write_json, write_npz)
import npe_diagnostics as ND

EXIT_ARGS, EXIT_REFUSE = 2, 3
SCHEMA = "rec_eval_metrics/1"
INDEX_SCHEMA = "rec_eval_metrics_index/1"

# the guard of ceil(alpha_gate (N_null + 1)) against binary roundoff (block 36,
# the contract of wave B: 0.05 x 100 is a hair above 5 in some products)
_CEIL_EPS = 1e-9
# rows per chunk of a recording bank's per-window quantities: bounds the
# float64 copy of the draws (256 x 1000 x 14 x 8 B = 29 MB); not a parameter
# of the science (each row's quantities are computed from that row alone)
_CHUNK = 256
# the words every output uses for (R4) (D-160)
MIXTURE_WORDS = "the mixture over the well's windows"
MIXTURE_NOTE = ("the mixture over the well's windows (R4): where the windows of the well put "
                "their mass, every window weighing the same; it is not the posterior of the "
                "well's parameter given its windows, and its width does not shrink with the "
                "number of windows (D-160)")
SHARE_NOTE = ("a share of windows flagged by the gate is read against the size band (the range "
              "of the flagged share of independent windows of the simulator's law, for a null "
              "set of this size, D-165), not against alpha_gate alone; no test is made on it "
              "(sec. 8, Q-R1)")
CAUTION = ("C_E^edge and C_E^int need not equal beta for a calibrated estimator: the expected "
           "coverage averages over the prior, not over a part of it; the split shows where a "
           "deficit of C_E sits, read through (R2) (sec. 2.6)")
BAND = "binomial: where a calibrated estimator's curve lies, one row per donor"


# =============================================================== pure functions
# No file is read or written here.

# Custom: (R1), the edge rows of the unit box; no library computes it.
# Checked: numpy.minimum, numpy.ndarray.min (used).
def edge_rows(theta, eps):
    """(n,) bool: (R1). theta (n, p) float in the unit box; row i is an edge
    row iff min_k min(theta_ik, 1 - theta_ik) < eps (strict)."""
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim != 2 or th.shape[1] < 1:
        raise ValueError("theta must be (n, p) with p >= 1, got shape %s" % (th.shape,))
    if not 0.0 < float(eps) < 0.5:
        raise ValueError("eps must be in (0, 1/2), got %r" % (eps,))
    if th.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return np.minimum(th, 1.0 - th).min(axis=1) < float(eps)


# Custom: the edge split (R2) of the expected coverage; no library computes
# it. Checked: ev_metrics.coverage_curve and coverage_summary (used for each
# curve).
def edge_split(gamma, edge, eps, p, betas=None):
    """(summary, arrays): the edge split of the expected coverage over S_cal.

    gamma (n_cal,) float64: the HPD ranks of the rows of S_cal. edge (n_cal,)
    bool: their edge flags. eps, p: the edge width and the number of axes
    (for the expected share 1 - (1 - 2 eps)^p). betas: the grid (default
    ev_metrics.BETAS).

    summary: n_cal, n_edge, n_int, pi_edge, pi_edge_expected, eps, `edge`
    and `interior` (ev_metrics.coverage_summary of each curve, None for an
    empty set), `r2_max_abs_dev` (the largest absolute difference between
    the two sides of (R2) over the grid; None when a set is empty).
    arrays: coverage_edge, coverage_int (len(betas),) float64, NaN-filled
    for an empty set.
    """
    b = BETAS if betas is None else np.asarray(betas, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64).ravel()
    e = np.asarray(edge, dtype=bool).ravel()
    if g.shape != e.shape:
        raise ValueError("gamma and edge must have one entry per row of S_cal, got %s and %s"
                         % (g.shape, e.shape))
    n_cal, n_edge = int(g.size), int(e.sum())
    n_int = n_cal - n_edge
    nan = np.full(b.size, np.nan)
    c_edge = coverage_curve(g[e], b) if n_edge else nan.copy()
    c_int = coverage_curve(g[~e], b) if n_int else nan.copy()
    pi = (n_edge / float(n_cal)) if n_cal else None
    r2 = None
    if n_edge and n_int:
        whole = coverage_curve(g, b)
        r2 = float(np.max(np.abs(whole - (pi * c_edge + (1.0 - pi) * c_int))))
    summary = {"n_cal": n_cal, "n_edge": n_edge, "n_int": n_int, "pi_edge": pi,
               "pi_edge_expected": float(1.0 - (1.0 - 2.0 * float(eps)) ** int(p)),
               "eps": float(eps), "p": int(p),
               "edge": coverage_summary(c_edge, b) if n_edge else None,
               "interior": coverage_summary(c_int, b) if n_int else None,
               "r2_max_abs_dev": r2, "caution": CAUTION}
    return summary, {"coverage_edge": c_edge, "coverage_int": c_int}


# Custom: (R4), the mixture over a well's windows from their pooled draws; no
# library pools grouped Monte Carlo draws. Checked: numpy.unique, numpy.mean,
# numpy.std, numpy.quantile (used).
def well_summary(draws, well):
    """dict: (R4) from the pooled draws of each well's windows. draws (n,
    n_post, p); well (n,) int. Returns well_ids (G,) int64 sorted, well_n
    (G,) int64, and well_mean, well_sd (ddof 1), well_q05, well_q50,
    well_q95, each (G, p) float64."""
    d = np.asarray(draws)
    w = np.asarray(well).ravel()
    if d.ndim != 3 or d.shape[0] != w.size:
        raise ValueError("draws must be (n, n_post, p) with n = len(well), got %s and %s"
                         % (d.shape, w.shape))
    ids, inv = np.unique(w, return_inverse=True)
    G, p = int(ids.size), int(d.shape[2])
    out = {"well_ids": ids.astype(np.int64),
           "well_n": np.bincount(inv, minlength=G).astype(np.int64)}
    for k in ("well_mean", "well_sd", "well_q05", "well_q50", "well_q95"):
        out[k] = np.empty((G, p), dtype=np.float64)
    for gi in range(G):
        # every window has the same n_post draws, so pooling weighs each equally
        pooled = d[inv == gi].reshape(-1, p).astype(np.float64)
        out["well_mean"][gi] = pooled.mean(axis=0)
        out["well_sd"][gi] = pooled.std(axis=0, ddof=1)
        q = np.quantile(pooled, [0.05, 0.5, 0.95], axis=0)
        out["well_q05"][gi], out["well_q50"][gi], out["well_q95"][gi] = q[0], q[1], q[2]
    return out


# Custom: the flagged count per group; no library function returns (ids, n,
# k). Checked: numpy.unique, numpy.bincount (used).
def flag_shares(flag, group):
    """(ids, n, k): for each distinct value of group (n,) int, sorted: the
    number of rows and the number of them with flag (n,) bool true; each
    (G,) int64."""
    f = np.asarray(flag, dtype=bool).ravel()
    g = np.asarray(group).ravel()
    if f.shape != g.shape:
        raise ValueError("flag and group must have the same length, got %s and %s"
                         % (f.shape, g.shape))
    ids, inv = np.unique(g, return_inverse=True)
    G = int(ids.size)
    n = np.bincount(inv, minlength=G).astype(np.int64)
    k = np.bincount(inv, weights=f.astype(np.int64), minlength=G).astype(np.int64)
    return ids.astype(np.int64), n, k


def gate_size(alpha, n_null, level):
    """dict: k_gate = ceil(alpha (n_null + 1) - 1e-9) - 1 (R9) and, when
    k_gate >= 1, the size's law (R10) Beta(k_gate, n_null + 1 - k_gate):
    size_mean = k_gate / (n_null + 1), size_lo and size_hi the ends of its
    central interval at `level`. With k_gate = 0 the three are None."""
    from scipy.stats import beta
    n_null = int(n_null)
    k = max(int(math.ceil(float(alpha) * (n_null + 1) - _CEIL_EPS)) - 1, 0)
    out = {"k_gate": k, "n_null": n_null, "alpha": float(alpha), "level": float(level),
           "size_mean": None, "size_lo": None, "size_hi": None}
    if k >= 1:
        a = (1.0 - float(level)) / 2.0
        lo, hi = beta.ppf([a, 1.0 - a], k, n_null + 1 - k)
        out.update(size_mean=k / float(n_null + 1), size_lo=float(lo), size_hi=float(hi))
    return out


def window_summary(draws):
    """dict of (n, p) float64 arrays: per window and axis the draws' mean
    `mhat`, SD `sigma` (ddof 1), `q05`, `q50`, `q95` and the contraction chi
    (R3) `contraction`. block 20's recovery_arrays with a NaN reference (its
    z-score dropped), the median beside it; computed in chunks of rows, each
    row's quantities from that row's draws alone."""
    d = np.asarray(draws)
    if d.ndim != 3:
        raise ValueError("draws must be (n, n_post, p), got %s" % (d.shape,))
    n, p = int(d.shape[0]), int(d.shape[2])
    per = dict((k, np.empty((n, p), dtype=np.float64))
               for k in ("mhat", "sigma", "q05", "q50", "q95", "contraction"))
    for s0 in range(0, n, _CHUNK):
        block = d[s0:s0 + _CHUNK]
        ra = recovery_arrays(np.full((block.shape[0], p), np.nan), block)
        for k in ("mhat", "sigma", "q05", "q95", "contraction"):
            per[k][s0:s0 + _CHUNK] = ra[k]
        per["q50"][s0:s0 + _CHUNK] = np.quantile(block.astype(np.float64), 0.5, axis=1)
    return per


def check_share(flag, level):
    """The check rows' flagged share with its exact (Clopper-Pearson)
    interval at `level` (scipy.stats.binomtest)."""
    from scipy.stats import binomtest
    f = np.asarray(flag, dtype=bool)
    n, k = int(f.size), int(f.sum())
    if n == 0:
        return {"n": 0, "n_flagged": 0, "share": None, "ci": [None, None], "level": level}
    ci = binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    return {"n": n, "n_flagged": k, "share": k / float(n), "ci": [float(ci.low), float(ci.high)],
            "level": level}


def example_wells(T, well, cls):
    """Per class, the example well (sec. 8, Q-R4): the well whose mean T over
    its windows is the median over the class's wells (the lower middle one
    for an even number; ties by the well id), with its first window (bank
    row index). ev_metrics.example_donors makes the choice."""
    T = np.asarray(T, dtype=np.float64)
    well = np.asarray(well)
    cls = np.asarray(cls)
    ids, inv = np.unique(well, return_inverse=True)
    if ids.size == 0:
        return []
    mean_T = np.bincount(inv, weights=T, minlength=ids.size) / np.bincount(inv, minlength=ids.size)
    first = np.array([int(np.flatnonzero(inv == gi)[0]) for gi in range(ids.size)])
    out = []
    for x in EM.example_donors(ids, mean_T, cls[first]):
        gi = int(np.flatnonzero(ids == x["donor"])[0])
        out.append({"cls": x["cls"], "well": x["donor"], "mean_T": x["lbar"],
                    "first_window": int(first[gi])})
    return out


def _shares(ids, n, k, key):
    return [{key: int(i), "n": int(a), "n_flagged": int(b), "share": b / float(a)}
            for i, a, b in zip(ids, n, k)]


# =============================================================== one ensemble

def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _load(path):
    with np.load(path) as z:
        return dict((k, z[k]) for k in z.files)


def sample_left_out(ecfg, e):
    """Why ensemble e is not reported: its sample absent, or made with other
    sample_params (block 36, "Rule"); None when its sample is current."""
    pj = ev_rec_plan.sample_json(ecfg, e.stem)
    if not os.path.isfile(pj):
        return "not sampled (%s absent)" % pj
    prov = _read_json(pj)
    if prov.get("sample_params") != ecfg.sample_params():
        return "sampled with other values: %s" % prov.get("sample_params")
    return None


def _roles(role):
    return dict((k, np.flatnonzero(role == k)) for k in
                (ev_gate.ROLE_REF, ev_gate.ROLE_RANK, ev_gate.ROLE_NULL, ev_gate.ROLE_CHECK))


def n_null_of(ecfg, e):
    """N_null of ensemble e's sample (its <stem>_sim.npz `gate_role`)."""
    path = os.path.join(os.path.dirname(ev_rec_plan.sample_json(ecfg, e.stem)),
                        "%s_sim.npz" % e.stem)
    with np.load(path) as z:
        return int(np.sum(z["gate_role"] == ev_gate.ROLE_NULL))


def _sim_metrics(ecfg, e, prov, z, S):
    """The quantities of D_sim: (json, npz arrays, the gate's reference, rank
    and null embeddings)."""
    level, eps = ecfg.level, float(ecfg.edge_eps)
    th = z["theta_ref"]
    n, p = th.shape
    M = int(z["logq_ref_members"].shape[1])
    in_box, calib, draw, donor = z["in_box"], z["calib"], z["draw"], z["donor"]
    didx = z["draw_idx"]
    if not np.array_equal(np.flatnonzero(draw), didx):
        raise RefuseError("%s: draw_idx is not the draw rows of its sample" % e.name)
    if np.any(calib & ~draw):
        raise RefuseError("%s: a calibration row is not a draw row (D-163 (1))" % e.name)
    pos = np.full(n, -1, dtype=np.int64)
    pos[didx] = np.arange(didx.size)
    axes = prov.get("axes") or {}
    names = list(axes.get("names") or ["theta_%d" % k for k in range(p)])
    s_cal = calib & in_box
    js = {"schema": SCHEMA, "ensemble": e.name, "stem": e.stem, "search": e.search,
          "campaign": e.campaign, "arm": e.arm, "dataset": "sim", "n": int(n),
          "n_cal": int(s_cal.sum()), "n_calib": int(calib.sum()), "n_d": int(didx.size),
          "n_in_box": int(in_box.sum()), "n_networks": int(np.unique(donor).size), "M": M,
          "n_post": int(z["draws"].shape[1]), "p": int(p),
          "axes": {"names": names, "bounds": axes.get("bounds")}, "level": level,
          "sample": ev_rec_plan.sample_json(ecfg, e.stem)}
    arr = {"rows": z["rows"], "donor": donor, "cls": z["cls"], "well": z["well"],
           "in_box": in_box, "calib": calib, "draw": draw, "theta_ref": th, "betas": BETAS}
    # the NLL over every row in the box, the networks as donors
    s, a = nll_block(z["logq_ref"], z["logq_ref_members"], in_box, donor)
    js["nll"] = s
    arr.update(a)
    # gamma on the draw rows; C_E, its band and the members' curves over S_cal
    gamma = np.full(n, np.nan)
    gamma[didx] = np.where(in_box[didx], hpd_ranks(z["logq_draws"], z["logq_ref"][didx]), np.nan)
    g = gamma[s_cal]
    curve = coverage_curve(g)
    lo, hi = binomial_band(int(g.size), BETAS, level)
    dm = z["draw_member"]
    mem_curves = []
    for m in range(M):
        own = z["logq_draws_own"][:, dm == m]
        gm = hpd_ranks(own[pos[s_cal]], z["logq_ref_members"][s_cal, m])
        mem_curves.append(coverage_curve(gm))
    ec = ND.expected_coverage(z["logq_ref"][s_cal], z["logq_draws"][pos[s_cal]],
                              seed=S("sim", "ec")) if g.size else None
    js["coverage"] = dict(coverage_summary(curve), n_rows=int(g.size), band=BAND,
                          ks_p=(float(ec.ks_pvalue) if ec else None),
                          verdict=(ec.verdict if ec else None),
                          members=[coverage_summary(c) for c in mem_curves])
    arr.update({"gamma": gamma, "coverage": curve, "coverage_lo": lo, "coverage_hi": hi,
                "coverage_members": np.stack(mem_curves)})
    # SBC and TARP over the calibration rows, with the project's axis names
    pc = pos[calib]
    s, a = sbc_block(th[calib], z["draws"][pc], level, S("sim", "sbc"), axes=names)
    js["sbc"] = s
    arr.update(a)
    s, a = tarp_block(th[calib], z["draws"][pc], z["stats"][pc], S("sim", "tarp"))
    js["tarp"] = s
    arr.update(a)
    # the recovery on the draw rows (NaN elsewhere), summarised over the calibration rows
    rd = recovery_arrays(th[didx], z["draws"])
    rec = {}
    for k, v in rd.items():
        full = np.full((n, p), np.nan)
        full[didx] = v
        rec[k] = full
    js["recovery"] = recovery_summary(th, rec, calib, axes=names)
    arr.update(dict((k, v.astype(np.float32)) for k, v in rec.items()))
    # the edge split (R1), (R2) over S_cal; the NLL over the edge and interior rows
    edge = edge_rows(th, eps)
    es, ea = edge_split(g, edge[s_cal], eps, p)
    l = arr["nll_rows"]
    ok = np.isfinite(l)
    for side, sel in (("edge", ok & edge), ("int", ok & ~edge)):
        es["nll_%s_n" % side] = int(sel.sum())
        es["nll_%s_mean" % side] = float(np.mean(l[sel])) if sel.any() else None
    js["edge"] = es
    arr.update(ea)
    arr["edge"] = edge
    # the gate: the check rows as test rows
    role = z["gate_role"]
    R = _roles(role)
    zpos = np.full(n, -1, dtype=np.int64)
    zpos[z["z_idx"]] = np.arange(z["z_idx"].size)
    gate_rows = np.flatnonzero(role != ev_gate.ROLE_NONE)
    if np.any(zpos[gate_rows] < 0):
        raise RefuseError("%s: a gate row has no embedding in its sample (z_idx)" % e.name)
    Z = z["z"]
    emb = lambda rows: Z[:, zpos[rows], :]                     # noqa: E731
    ref, rank, null = (emb(R[ev_gate.ROLE_REF]), emb(R[ev_gate.ROLE_RANK]),
                       emb(R[ev_gate.ROLE_NULL]))
    factors = list(ecfg.gate["factors"])
    alpha = float(ecfg.gate["alpha"])
    gt = ev_gate.gate(ref, rank, null, emb(R[ev_gate.ROLE_CHECK]), factors)
    size = gate_size(alpha, null.shape[1], level)
    if size["k_gate"] < 1:
        raise RefuseError(_k_gate_refusal(alpha, null.shape[1]))
    flag = gt["p"] < alpha
    js["gate"] = dict(size, n_ref=int(ref.shape[1]), n_rank=int(rank.shape[1]),
                      n_null=int(null.shape[1]), n_check=int(R[ev_gate.ROLE_CHECK].size),
                      factors=factors, bandwidths=gt["bandwidths"].tolist(),
                      median=gt["median"].tolist(), check=check_share(flag, level),
                      note=SHARE_NOTE)
    arr.update({"gate_role": role, "gate_check_rows": R[ev_gate.ROLE_CHECK].astype(np.int64),
                "gate_T_null": gt["T_null"], "gate_T_check": gt["T"], "gate_p_check": gt["p"]})
    return js, arr, (ref, rank, null)


def _bank_metrics(ecfg, e, prov, r, zr, gref, S):
    """The quantities of recording bank r: (json, npz arrays)."""
    level = ecfg.level
    ref, rank, null = gref
    d = zr["draws"]
    n, p = int(d.shape[0]), int(d.shape[2])
    M = int(zr["z"].shape[0])
    axes = prov.get("axes") or {}
    names = list(axes.get("names") or ["theta_%d" % k for k in range(p)])
    # per window and axis: block 20's recovery_arrays with a NaN reference
    # (its z-score is not written), the median beside it, chi (R3)
    per = window_summary(d)
    chi = per["contraction"]
    posterior = {"contraction_mean": chi.mean(axis=0).tolist(),
                 "contraction_q05": np.quantile(chi, 0.05, axis=0).tolist(),
                 "contraction_q50": np.quantile(chi, 0.50, axis=0).tolist(),
                 "contraction_q95": np.quantile(chi, 0.95, axis=0).tolist(),
                 "what": "per axis: chi = 1 - sigma^2 / (1/12) of each window's draws (R3), its "
                         "mean and its 5 %, 50 % and 95 % quantiles over the windows"}
    # the gate on the bank's windows
    factors = list(ecfg.gate["factors"])
    alpha = float(ecfg.gate["alpha"])
    gt = ev_gate.gate(ref, rank, null, zr["z"], factors)
    flag = gt["p"] < alpha
    size = gate_size(alpha, null.shape[1], level)
    # the wells: (R4) from the pooled draws, the flagged shares, the ids
    well, cls = zr["well"], zr["cls"]
    ws = well_summary(d, well)
    wid, wn, wk = flag_shares(flag, well)
    if not np.array_equal(wid, ws["well_ids"]):
        raise AssertionError("well_summary and flag_shares disagree on the wells")
    cid, cn, ck = flag_shares(flag, cls)
    wells = []
    for gi, w in enumerate(ws["well_ids"]):
        rows = np.flatnonzero(well == w)
        j = int(rows[0])
        wells.append({"well": int(w), "n": int(ws["well_n"][gi]), "cls": int(cls[j]),
                      "donor": int(zr["donor"][j]), "batch": int(zr["batch"][j]),
                      "first_window": j, "what": MIXTURE_WORDS,
                      "mean": ws["well_mean"][gi].tolist(), "sd": ws["well_sd"][gi].tolist(),
                      "q05": ws["well_q05"][gi].tolist(), "q50": ws["well_q50"][gi].tolist(),
                      "q95": ws["well_q95"][gi].tolist(),
                      "n_flagged": int(wk[gi]), "share_flagged": wk[gi] / float(wn[gi]),
                      "mean_T": float(np.mean(gt["T"][rows]))})
    gate = dict(size, n=n, n_flagged=int(flag.sum()), share=(float(flag.mean()) if n else None),
                per_class=_shares(cid, cn, ck, "cls"), per_well=_shares(wid, wn, wk, "well"),
                note=SHARE_NOTE)
    # the embeddings: each member's first two principal components of the
    # reference rows' and the bank's embeddings pooled (for the figure)
    from sklearn.decomposition import PCA
    coords, evr = [], []
    for m in range(M):
        pooled = np.concatenate([ref[m], zr["z"][m]]).astype(np.float64)
        pca = PCA(n_components=2, random_state=S(r, "pca")).fit(pooled)
        coords.append(pca.transform(pooled))
        evr.append(pca.explained_variance_ratio_)
    source = np.r_[np.zeros(ref.shape[1], dtype=np.int64), np.ones(n, dtype=np.int64)]
    js = {"schema": SCHEMA, "ensemble": e.name, "stem": e.stem, "dataset": r, "bank": r,
          "n_r": n, "n_post": int(d.shape[1]), "M": M, "p": p,
          "axes": {"names": names, "bounds": axes.get("bounds")}, "level": level,
          "n_wells": int(ws["well_ids"].size),
          "well_ids": ws["well_ids"].tolist(),
          "classes": np.unique(cls).astype(np.int64).tolist(),
          "batches": np.unique(zr["batch"]).astype(np.int64).tolist(),
          "posterior": posterior, "wells": wells, "mixture_note": MIXTURE_NOTE, "gate": gate,
          "example_wells": example_wells(gt["T"], well, cls),
          "pca": {"source": "0: the gate's reference rows of D_sim, 1: the bank's windows",
                  "n_ref": int(ref.shape[1])},
          "sample": ev_rec_plan.sample_json(ecfg, e.stem)}
    arr = dict((k, zr[k]) for k in ("rows", "cls", "donor", "well", "batch", "subregion",
                                    "window_idx", "rep_group", "contaminated"))
    arr.update(dict((k, v.astype(np.float32)) for k, v in per.items()))
    arr.update(ws)
    arr.update({"gate_t": gt["t"], "gate_r": gt["r"], "gate_p_member": gt["p_member"],
                "gate_T": gt["T"], "gate_p": gt["p"], "gate_flag": flag,
                "pca_coords": np.stack(coords).astype(np.float32), "pca_evr": np.stack(evr),
                "pca_source": source})
    return js, arr


def ensemble_metrics(task):
    """Every quantity of one ensemble on D_sim and on each of its recording
    banks; writes its files and returns (stem, [data sets], seconds,
    summary)."""
    ecfg_path, e, mdir = task
    t0 = time.time()
    ecfg = ev_rec_config.load(ecfg_path)
    pj = ev_rec_plan.sample_json(ecfg, e.stem)
    prov = _read_json(pj)
    sdir = os.path.dirname(pj)
    S = lambda *p: seed_of(ecfg.seed, "metrics", e.stem, *p)      # noqa: E731  block 20's
    z = _load(os.path.join(sdir, "%s_sim.npz" % e.stem))
    js, arr, gref = _sim_metrics(ecfg, e, prov, z, S)
    del z
    write_json(os.path.join(mdir, "%s_sim.json" % e.stem), js)
    write_npz(os.path.join(mdir, "%s_sim.npz" % e.stem), arr)
    out = {"sim": {"nll": js["nll"]["mean"], "check_share": js["gate"]["check"]["share"]}}
    for r in e.banks:
        zr = _load(os.path.join(sdir, "%s_%s.npz" % (e.stem, r)))
        jr, ar = _bank_metrics(ecfg, e, prov, r, zr, gref, S)
        write_json(os.path.join(mdir, "%s_%s.json" % (e.stem, r)), jr)
        write_npz(os.path.join(mdir, "%s_%s.npz" % (e.stem, r)), ar)
        out[r] = {"n_r": jr["n_r"], "share": jr["gate"]["share"]}
    return e.stem, ["sim"] + list(e.banks), round(time.time() - t0, 1), out


# =============================================================== the comparisons

# what must be equal in the two samples of a comparison on D_sim (block 27's)
_SHARED_ROWS = ("rows", "theta_ref", "donor")
_PAIRED_ARRAYS = ("rows", "theta_ref", "donor", "in_box", "logq_ref", "logq_ref_members")


def _paired_arrays(ecfg, e):
    path = os.path.join(os.path.dirname(ev_rec_plan.sample_json(ecfg, e.stem)),
                        "%s_sim.npz" % e.stem)
    with np.load(path) as z:
        return dict((k, z[k]) for k in _PAIRED_ARRAYS)


def paired_all(view, ecfg, reported, made, mdir, log):
    """Block 27's E3 on D_sim for each comparison of ev_rec_plan.comparisons
    (`made`, its order): block 20's paired_block on the rows in the box under
    both ensembles, the networks as the bootstrap's groups, sigma_seed the
    larger of the two ensembles' member_sd, the seed seed_of(seed, "paired",
    name, "sim") (block 27's). A comparison whose ensembles are not both
    reported, or whose two samples' rows differ, is not made and is named.
    Returns (files, made, left_out)."""
    by = dict((e.name, e) for e in reported)
    complete = dict((e.name, e) for e in view.ensembles)

    def not_reported(name):
        e = complete.get(name)
        if e is None:
            return "%s is not reported (not complete)" % name
        why = sample_left_out(ecfg, e)
        if why is not None:
            return "%s is not reported (%s)" % (name, why)
        return "%s was not requested (--ensemble)" % name

    files, done, left_out = [], [], []
    for c in made:
        why = None
        for name in (c.E, c.E_ref):
            if name not in by:
                why = not_reported(name)
                break
        if why is None:
            zE, zR = _paired_arrays(ecfg, by[c.E]), _paired_arrays(ecfg, by[c.E_ref])
            if not all(zE[k].shape == zR[k].shape and np.array_equal(zE[k], zR[k])
                       for k in _SHARED_ROWS):
                why = "%s and %s do not share the rows of D_sim" % (c.E, c.E_ref)
        if why is not None:
            left_out.append({"comparison": c.name, "E": c.E, "E_ref": c.E_ref, "why": why})
            log("  comparison %s left out: %s" % (c.name, why))
            continue
        D = "sim"
        ok = zE["in_box"] & np.isfinite(zE["logq_ref"]) & np.isfinite(zR["logq_ref"])
        sE, sR = member_sd(zE), member_sd(zR)
        seed = seed_of(ecfg.seed, "paired", c.name, D)
        s, a = paired_block(-zE["logq_ref"], -zR["logq_ref"], zE["donor"], ok, sE, sR,
                            ecfg.bootstrap_n, ecfg.level, seed)
        lower = s.pop("lower_nll")
        s.update({"schema": SCHEMA, "dataset": D, "comparison": c.name, "kind": c.kind,
                  "E": c.E, "E_ref": c.E_ref, "search": c.search, "search_ref": c.search_ref,
                  "arm": c.arm, "arm_ref": c.arm_ref, "sd_members_E": sE, "sd_members_ref": sR,
                  "lower_nll_ensemble": ((c.E if s["Dbar"] < 0 else c.E_ref)
                                         if s["resolved"] else None),
                  "bootstrap_seed": seed, "groups": "the simulated networks of D_sim"})
        if c.kind == "vs_A1":                   # block 27's legacy keys
            s.update({"term_arm": c.arm, "ensembles": {"term": c.E, "A1": c.E_ref},
                      "sd_members": {c.arm: sE, "A1": sR}, "lower_nll": lower})
        base = os.path.join(mdir, "paired_%s_%s" % (c.name, D))
        write_json(base + ".json", s)
        write_npz(base + ".npz", a)
        files += [base + ".json", base + ".npz"]
        log("  paired %s on %s: Dbar %+.4f [%+.4f, %+.4f], 2 sigma_seed %.4f -> %s"
            % (c.name, D, s["Dbar"], s["ci"][0], s["ci"][1], s["two_sigma_seed"],
               "resolved" if s["resolved"] else "not resolved"))
        done.append({"name": c.name, "kind": c.kind, "E": c.E, "E_ref": c.E_ref,
                     "search": c.search, "search_ref": c.search_ref, "arm": c.arm,
                     "arm_ref": c.arm_ref, "datasets": [D], "paired": {D: base + ".json"}})
    return files, done, left_out


# =============================================================== compute, main

def _k_gate_refusal(alpha, n_null):
    return ("k_gate = 0 (R9): with gate.alpha %g and a null set of N_null = %d rows, "
            "alpha_gate (N_null + 1) = %g <= 1, so no window can be flagged; raise gate.alpha "
            "or the null set's size (gate.n_null, the networks of D_sim)"
            % (alpha, n_null, alpha * (n_null + 1)))


def _limit_threads():
    """One thread per worker (block 20's rule): the BLAS libraries'."""
    try:
        from threadpoolctl import threadpool_limits
        _limit_threads.keep = threadpool_limits(limits=1)
    except ImportError:
        pass


def compute(cfg_path, ecfg_path, names=None, ncpus=None, log=print):
    """Every quantity of every ensemble with a current sample (or of those
    named), the comparisons, and METRICS.json; returns the index dict.
    ev_plan.RefuseError when no ensemble has a current sample, or for
    k_gate = 0."""
    t0 = time.time()
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(cfg_path, ecfg)
    made, not_made = ev_rec_plan.comparisons(view)       # its RefuseError before any work
    mdir = os.path.abspath(ecfg.sub("metrics"))
    left_out = [{"ensemble": "%s (%s)" % (q, os.path.basename(f)), "why": w}
                for q, f, w in view.missing]
    if names:
        unknown = sorted(set(names) - set(e.name for e in view.ensembles))
        if unknown:
            raise RefuseError("no complete ensemble %s (complete: %s)"
                              % (", ".join(unknown),
                                 ", ".join(e.name for e in view.ensembles) or "none"))
    todo = []
    for e in view.ensembles:
        if names and e.name not in names:
            continue
        why = sample_left_out(ecfg, e)
        if why is not None:
            left_out.append({"ensemble": e.name, "why": why})
            log("  %s left out: %s" % (e.name, why))
            continue
        todo.append(e)
    if not todo:
        raise RefuseError("no ensemble has a current sample under %s" % ecfg.sub("samples"))
    alpha = float(ecfg.gate["alpha"])
    for e in todo:                                         # k_gate (R9) before any work
        nn = n_null_of(ecfg, e)
        if gate_size(alpha, nn, ecfg.level)["k_gate"] < 1:
            raise RefuseError("%s: %s" % (e.name, _k_gate_refusal(alpha, nn)))
    os.makedirs(mdir, exist_ok=True)
    n_w = max(1, min(int(ncpus or ecfg.ncpus), len(todo)))
    log("rec metrics: %d ensemble(s) in %d worker(s): %s"
        % (len(todo), n_w, ", ".join(e.name for e in todo)))
    tasks = [(ecfg.path, e, mdir) for e in todo]
    if n_w == 1:
        _limit_threads()
        results = [ensemble_metrics(t) for t in tasks]
    else:
        with get_context("fork").Pool(n_w, initializer=_limit_threads) as pool:
            results = pool.map(ensemble_metrics, tasks, chunksize=1)
    files, per = [], {}
    for stem, ds, secs, out in results:
        per[stem] = ds
        for D in ds:
            files += [os.path.join(mdir, "%s_%s.json" % (stem, D)),
                      os.path.join(mdir, "%s_%s.npz" % (stem, D))]
        sim = out["sim"]
        log("  %s: sim NLL %s, check rows flagged %s; %s (%.0f s)" % (
            stem, "-" if sim["nll"] is None else "%.3f" % sim["nll"],
            "-" if sim["check_share"] is None else "%.3f" % sim["check_share"],
            ", ".join("%s %d windows, flagged %s" % (
                r, out[r]["n_r"], "-" if out[r]["share"] is None else "%.3f" % out[r]["share"])
                for r in ds[1:]) or "no recording bank", secs))
    paired_files, done, cmp_left = paired_all(view, ecfg, todo, made, mdir, log)
    files += paired_files
    cmp_left += [{"comparison": q, "E": None, "E_ref": None, "why": w} for q, w in not_made]
    for q, w in not_made:
        log("  comparison %s not made: %s" % (q, w))
    index = {"schema": INDEX_SCHEMA, "config": os.path.abspath(cfg_path),
             "eval_config": ecfg.path, "view_kind": view.kind,
             "ensembles": dict((e.name, {"stem": e.stem, "arm": e.arm, "search": e.search,
                                         "campaign": e.campaign, "banks": list(e.banks),
                                         "datasets": per[e.stem]}) for e in todo),
             "left_out": left_out, "files": files, "comparisons": done,
             "comparisons_left_out": cmp_left,
             "report_inputs": ev_rec_plan.report_inputs(view, ecfg),
             "seconds": round(time.time() - t0, 1)}
    write_json(os.path.join(mdir, "METRICS.json"), index)
    log("comparisons: %d made, %d not made" % (len(done), len(cmp_left)))
    log("REC METRICS OK: %d ensemble(s), %d data-set file(s), %d comparison(s), %d left out "
        "(%.0f s)" % (len(todo), sum(len(v) for v in per.values()), len(done), len(left_out),
                      time.time() - t0))
    return index


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_REFUSE."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the project's configuration (<tag>.json)")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's configuration (<tag>_rec_eval.json)")
    ap.add_argument("--ensemble", action="append", default=None,
                    help="an ensemble's name; repeat for several (default: every one sampled)")
    ap.add_argument("--ncpus", type=int, default=None, help="default: the configuration's ncpus")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    if args.ncpus is not None and args.ncpus < 1:
        print("--ncpus must be >= 1, got %d" % args.ncpus, flush=True)
        return EXIT_ARGS
    try:
        compute(args.config, args.eval_config, names=args.ensemble, ncpus=args.ncpus,
                log=lambda t: print(t, flush=True))
    except (RefuseError, ev_rec_config.RecEvalConfigError) as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
