#!/usr/bin/env python3
"""How much of each trial's report-split NLL is the flow's edge term (SPEC
block 7, added 2026-10-20: the user's "Write it" after the comments on the
tail NLLs). Read-only: it opens files and writes none.

    python3 hpc/pipeline/tools/perrow_check.py --config <project>.json
        [--campaign-dir DIR]... [--trial ID]... [--bound 5.0]

Every trial's flow (joint_model.build_joint_model: zuko's NSF behind sbi's
transform to the unconstrained space of the unit box) models
u = logit(theta_k) with splines on [-B, B] and the identity outside, over a
standard normal base. A coordinate with |u| > B therefore adds

    tau(theta) = u^2 / 2 + ln(2 pi) / 2 + ln(theta (1 - theta))

nats to the row's -log q(theta | z), whatever the trial learned (checked by
computation with sbi 0.27.0 and zuko 1.6.0, whose bound is 5.0). Over the
n report rows

    T = (1/n) sum_i sum_k tau(theta_ik) 1{|u_ik| > B},   L = T + R,

T the same for every trial on the same rows, R = L - T the part the trial
controls. The report: per row set, T and a table per coordinate; per trial,
L, T, R and the ledger's nll; then the pass line

    PERROW CHECK OK: <k> trial(s) read, <s> skipped, <m> without a per-row
    file; <r> row set(s); T = <t> nats/row; nothing was written

Exit 0; 2 the configuration does not load; 3 no per-row file was read.
Needs numpy (the sbi_export environment has it). Pure ASCII, LF only;
Python 3.6 syntax.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import zipfile
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

import config as C                      # noqa: E402

BOUND = 5.0             # zuko 1.6.0 MonotonicRQSTransform's default (sbi 0.27.0 passes none)
LEDGER_TOL = 1e-6       # nats/row: the runner writes L = float(mean(nll)) of the same array
HALF_LN_2PI = 0.5 * math.log(2.0 * math.pi)
KEYS = ("nll", "theta", "group")


# ------------------------------------------------------------ loading ---

def campaign_dirs(cfg, extra):
    """The configuration's campaign folder, then every --campaign-dir."""
    d = cfg.data
    out = [os.path.abspath(os.path.join(d["paths"]["tune_root"], d["search"]["campaign"]))]
    for e in extra:
        e = os.path.abspath(e)
        if e not in out:
            out.append(e)
    return out


def trial_ids(camp, only):
    """The trial ids of a campaign folder: its ledger records and run folders."""
    ids = set()
    tdir = os.path.join(camp, "trials")
    if os.path.isdir(tdir):
        ids.update(f[:-len(".json")] for f in os.listdir(tdir) if f.endswith(".json"))
    rdir = os.path.join(camp, "runs")
    if os.path.isdir(rdir):
        ids.update(f for f in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, f)))
    if only:
        ids = set(i for i in ids if any(i.startswith(o) for o in only))
    return sorted(ids)


def read_ledger(camp, tid):
    """The trial's ledger record (a dict), {'status': 'unreadable'}, or None."""
    p = os.path.join(camp, "trials", tid + ".json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return {"status": "unreadable"}
    return rec if isinstance(rec, dict) else {"status": "unreadable"}


def perrow_paths(camp, tid):
    return sorted(glob.glob(os.path.join(camp, "runs", tid, "*_perrow.npz")))


def _member(z, path, name):
    """One array of an .npz; an object array (pickled donor ids) is read
    again with pickles allowed -- the file is the runner's own."""
    try:
        return z[name]
    except ValueError:
        with np.load(path, allow_pickle=True) as z2:
            return z2[name]


def load_perrow(path):
    """(arrays, None) or (None, problem). arrays: nll (n,) float64, theta
    (n, d) float64, theta_stored (as in the file), group (n,)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in KEYS if k not in z.files]
            if missing:
                return None, "no %s in the file" % ", ".join(missing)
            raw = dict((k, _member(z, path, k)) for k in KEYS)
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile, zlib.error) as exc:
        return None, "cannot be read (%s: %s)" % (type(exc).__name__, exc)
    nll, theta, group = raw["nll"], raw["theta"], raw["group"]
    if nll.ndim != 1 or theta.ndim != 2 or group.ndim != 1:
        return None, ("shapes nll %s, theta %s, group %s: want (n,), (n, d), (n,)"
                      % (nll.shape, theta.shape, group.shape))
    n = nll.shape[0]
    if n == 0 or theta.shape[0] != n or group.shape[0] != n:
        return None, ("row counts differ or are zero: nll %d, theta %d, group %d"
                      % (n, theta.shape[0], group.shape[0]))
    nll64 = np.asarray(nll, dtype=np.float64)
    th64 = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(nll64)):
        return None, "%d non-finite nll value(s)" % int(np.sum(~np.isfinite(nll64)))
    inside = np.isfinite(th64) & (th64 > 0.0) & (th64 < 1.0)
    if not np.all(inside):
        return None, ("%d theta value(s) outside the open unit box (or non-finite)"
                      % int(np.sum(~inside)))
    return {"nll": nll64, "theta": th64, "theta_stored": theta, "group": group}, None


def row_set_id(theta_stored, group):
    """12 hex digits of a sha256 of theta as stored and of the donors."""
    h = hashlib.sha256()
    h.update(("%s %r\n" % (theta_stored.dtype.str, tuple(theta_stored.shape))).encode("ascii"))
    h.update(np.ascontiguousarray(theta_stored).tobytes())
    h.update("\n".join(str(g) for g in group.tolist()).encode("utf-8"))
    return h.hexdigest()[:12]


def bank_axes(cfg, d):
    """(names, physical): names of the d coordinates, and per coordinate
    (lo, hi, coord, unit) from the bank's sidecar, or None where unknown."""
    names = ["theta[%d]" % k for k in range(d)]
    phys = [None] * d
    root = os.path.join(cfg.data["paths"]["banks_root"], cfg.tag + "_sim")
    paths = [os.path.join(root, "shard_0000.json")] + sorted(glob.glob(os.path.join(root, "shard_*.json")))
    side = None
    for p in paths:
        if os.path.isfile(p):
            try:
                with open(p, "r") as fh:
                    side = json.load(fh)
                break
            except (OSError, ValueError):
                continue
    if not isinstance(side, dict):
        return names, phys
    pn = side.get("param_names")
    if isinstance(pn, list) and len(pn) == d:
        names = [str(x) for x in pn]
    bounds = side.get("bounds_theta")
    lat = side.get("latent_spec") or {}
    coords = lat.get("coord_physical")
    units = lat.get("units")
    if isinstance(bounds, list) and len(bounds) == d and isinstance(coords, list) and len(coords) == d:
        for k in range(d):
            try:
                lo, hi = float(bounds[k][0]), float(bounds[k][1])
            except (TypeError, ValueError, IndexError):
                continue
            unit = str(units[k]) if isinstance(units, list) and len(units) == d else ""
            phys[k] = (lo, hi, str(coords[k]), unit)
    return names, phys


# ----------------------------------------------------------- analysis ---

def logit(theta):
    return np.log(theta) - np.log1p(-theta)


def edge_terms(theta, bound):
    """(tau, u, mask), each (n, d): the edge term of every coordinate past
    the bound (0 elsewhere, float64 nats), u = logit(theta), |u| > bound."""
    u = logit(theta)
    mask = np.abs(u) > bound
    tau = np.where(mask, 0.5 * u * u + HALF_LN_2PI + np.log(theta) + np.log1p(-theta), 0.0)
    return tau, u, mask


def decompose(nll, tau_row, edge_row):
    """L = T + R and the row summaries; nll, tau_row (n,) float64, edge_row (n,) bool."""
    L = float(np.mean(nll))
    T = float(np.mean(tau_row))
    rest = nll[~edge_row]
    return {"n": int(nll.size), "L": L, "T": T, "R": L - T,
            "median_l": float(np.median(nll)),
            "frac_l_neg": float(np.mean(nll < 0.0)),
            "frac_edge_rows": float(np.mean(edge_row)),
            "L_no_edge": float(np.mean(rest)) if rest.size else float("nan")}


def ledger_check(L, rec, tol=LEDGER_TOL):
    """(verdict, text): 'equal', 'MISMATCH' or 'none' against the ledger's nll."""
    if rec is None:
        return "none", "no ledger record"
    v = rec.get("nll")
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
        return "none", "ledger %s, no nll" % rec.get("status", "?")
    diff = abs(L - float(v))
    if diff <= tol:
        return "equal", "ledger %s nll %.6f = L" % (rec.get("status", "?"), float(v))
    return "MISMATCH", ("ledger %s nll %.6f MISMATCH (L differs by %.3g nats)"
                        % (rec.get("status", "?"), float(v), diff))


def coordinate_table(theta, tau, u, bound):
    """Per coordinate: shares past the bound, part of T, distinct values, range."""
    rows = []
    for k in range(theta.shape[1]):
        col = theta[:, k]
        rows.append({"k": k, "low": float(np.mean(u[:, k] < -bound)),
                     "high": float(np.mean(u[:, k] > bound)),
                     "T_k": float(np.mean(tau[:, k])), "distinct": int(np.unique(col).size),
                     "min": float(col.min()), "max": float(col.max())})
    return rows


def physical_edges(p, bound):
    """The edge zone of one coordinate in physical units, or None."""
    if p is None:
        return None
    lo, hi, coord, unit = p
    s_lo = 1.0 / (1.0 + math.exp(bound))
    s_hi = 1.0 - s_lo
    a, b = lo + s_lo * (hi - lo), lo + s_hi * (hi - lo)
    if coord == "ln":
        a, b = math.exp(a), math.exp(b)
    elif coord != "linear":
        return None
    return a, b, unit


def analyse(cfg, camps, only, bound):
    """Everything the report prints, as data. No file is written."""
    sets, trials, skipped, without = {}, [], [], []
    for ci, camp in enumerate(camps):
        for tid in trial_ids(camp, only):
            rec = read_ledger(camp, tid)
            paths = perrow_paths(camp, tid)
            if not paths:
                without.append({"campaign": ci, "trial": tid,
                                "status": (rec or {}).get("status", "no record")})
                continue
            for path in paths:
                arr, problem = load_perrow(path)
                if arr is None:
                    skipped.append({"campaign": ci, "trial": tid, "path": path, "why": problem})
                    continue
                sid = row_set_id(arr["theta_stored"], arr["group"])
                if sid not in sets:
                    tau, u, mask = edge_terms(arr["theta"], bound)
                    tau_row, edge_row = tau.sum(axis=1), mask.any(axis=1)
                    sets[sid] = {"id": sid, "n": arr["theta"].shape[0], "d": arr["theta"].shape[1],
                                 "donors": int(np.unique(arr["group"].astype(str)).size),
                                 "T": float(np.mean(tau_row)), "tau_row": tau_row,
                                 "edge_row": edge_row,
                                 "frac_edge_rows": float(np.mean(edge_row)),
                                 "frac_edge_coords": float(np.mean(mask)),
                                 "table": coordinate_table(arr["theta"], tau, u, bound),
                                 "trials": []}
                s = sets[sid]
                dec = decompose(arr["nll"], s["tau_row"], s["edge_row"])
                verdict, text = ledger_check(dec["L"], rec)
                s["trials"].append(tid)
                trials.append(dict(dec, campaign=ci, trial=tid, path=path, set=sid,
                                   ledger=verdict, ledger_text=text))
    d = max([s["d"] for s in sets.values()] or [0])
    names, phys = bank_axes(cfg, d) if d else ([], [])
    return {"bound": bound, "camps": camps, "sets": sets, "trials": trials,
            "skipped": skipped, "without": without, "names": names, "phys": phys}


# ----------------------------------------------------------- reporting ---

def _pct(x):
    return "%6.2f%%" % (100.0 * x)


def pass_line(res):
    ts = [s["T"] for s in sorted(res["sets"].values(), key=lambda s: -len(s["trials"]))]
    if len(ts) == 1:
        tt = "T = %.4f nats/row" % ts[0]
    else:
        tt = "T = %s nats/row (one per row set)" % ", ".join("%.4f" % t for t in ts)
    return ("PERROW CHECK OK: %d trial(s) read, %d skipped, %d without a per-row file; "
            "%d row set(s); %s; nothing was written"
            % (len(res["trials"]), len(res["skipped"]), len(res["without"]), len(res["sets"]), tt))


def print_report(res, out=sys.stdout):
    w = out.write
    b = res["bound"]
    w("PERROW CHECK: the flow's edge term in the report-split NLL (read-only)\n")
    w("  bound B = %g: a coordinate with |logit theta| > B (theta < %.5f or > %.5f) adds\n"
      % (b, 1.0 / (1.0 + math.exp(b)), 1.0 - 1.0 / (1.0 + math.exp(b))))
    w("  tau = u^2/2 + ln(2 pi)/2 + ln(theta (1 - theta)) nats to -log q in every trial;\n")
    w("  L = T + R: T the edge part (the same for every trial on the same rows), R the rest\n")
    for ci, camp in enumerate(res["camps"]):
        n_read = sum(1 for t in res["trials"] if t["campaign"] == ci)
        w("campaign [%d] %s: %d per-row file(s) read\n" % (ci + 1, camp, n_read))
    names, phys = res["names"], res["phys"]
    for s in sorted(res["sets"].values(), key=lambda s: -len(s["trials"])):
        w("row set %s: n = %d rows, d = %d, %d donors; %d trial(s)\n"
          % (s["id"], s["n"], s["d"], s["donors"], len(s["trials"])))
        w("  T = %.4f nats/row; rows with an edge coordinate %s; coordinates past the bound %s\n"
          % (s["T"], _pct(s["frac_edge_rows"]).strip(), _pct(s["frac_edge_coords"]).strip()))
        w("  %-24s %8s %8s %10s %9s %9s %9s  %s\n"
          % ("coordinate", "u < -B", "u > B", "part of T", "distinct", "min", "max",
             "edge zone (physical units)"))
        for r in sorted(s["table"], key=lambda r: -r["T_k"]):
            k = r["k"]
            nm = names[k] if k < len(names) else "theta[%d]" % k
            pe = physical_edges(phys[k] if k < len(phys) else None, b)
            zone = ("below %.4g, above %.4g %s" % pe) if pe else "-"
            w("  %-24s %8s %8s %10.4f %9d %9.5f %9.5f  %s\n"
              % (nm[:24], _pct(r["low"]), _pct(r["high"]), r["T_k"], r["distinct"],
                 r["min"], r["max"], zone.rstrip()))
    w("trials, sorted by L (nats/row; l_i the per-row NLL)\n")
    for t in sorted(res["trials"], key=lambda t: t["L"]):
        w("  [%d] %-14s L %8.4f = T %7.4f + R %8.4f | median l %8.4f | l < 0 in %s | "
          "L without edge rows %8.4f | set %s | %s\n"
          % (t["campaign"] + 1, t["trial"][:14], t["L"], t["T"], t["R"], t["median_l"],
             _pct(t["frac_l_neg"]).strip(), t["L_no_edge"], t["set"], t["ledger_text"]))
    for k in res["skipped"]:
        w("  SKIPPED [%d] %s: %s (%s)\n" % (k["campaign"] + 1, k["trial"], k["why"], k["path"]))
    if res["without"]:
        w("  without a per-row file: %d trial(s) (%s)\n"
          % (len(res["without"]), ", ".join("[%d] %s %s" % (x["campaign"] + 1, x["trial"][:12], x["status"])
                                            for x in res["without"][:12])
             + (", ..." if len(res["without"]) > 12 else "")))
    rs = np.array([t["R"] for t in res["trials"]], dtype=np.float64)
    if rs.size:
        w("R over %d trial(s): min %.4f, median %.4f, max %.4f; below 0 in %d\n"
          % (rs.size, rs.min(), float(np.median(rs)), rs.max(), int(np.sum(rs < 0))))
    mism = [t for t in res["trials"] if t["ledger"] == "MISMATCH"]
    if mism:
        w("  %d trial(s) whose per-row file does not give the ledger's nll: %s\n"
          % (len(mism), ", ".join(t["trial"][:12] for t in mism)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--campaign-dir", action="append", default=[], metavar="DIR")
    ap.add_argument("--trial", action="append", default=[], metavar="ID")
    ap.add_argument("--bound", type=float, default=BOUND)
    args = ap.parse_args(argv)
    try:
        cfg = C.load_config(args.config)
    except C.ConfigError as exc:
        print("PERROW CHECK: the configuration %s does not load: %s" % (args.config, exc.problems))
        return 2
    res = analyse(cfg, campaign_dirs(cfg, args.campaign_dir), args.trial, args.bound)
    print_report(res)
    if not res["trials"]:
        print("PERROW CHECK: no per-row file was read (%d skipped, %d trial(s) without one)"
              % (len(res["skipped"]), len(res["without"])))
        return 3
    print(pass_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
