#!/usr/bin/env python3
"""Per-electrode firing rate for both arms, on the SAME analysis window.

Why this exists
---------------
46.9% of the simulated prior predictive collapses onto one embedding. The
embedding norm does NOT separate silent from active simulations: sampled
rows below the norm floor include one with 1059 spikes, and rows above it
include some with 6. So any "drop the degenerate ones" rule has to be
defined on the OBSERVABLE, not on the embedding.

This script computes one quantity, in the same units, for both arms:

    rate = spikes / (n_e * T_win)          [Hz per electrode]

For the real arm the traces hold no spike times, only the smoothed IFR, and
the sidecar defines its units as "spikes per bin per electrode (multiply by
fs_ifr for Hz)". So the matching quantity is mean(ifr_trace) * fs_ifr over
the same window. The two are then directly comparable, which is what lets
the validity cutoff be anchored to the lowest activity real cultures
actually show rather than to a number chosen by hand.

The simulated window is the EXPORTED one: spikes with
trim_head_s <= t < trim_head_s + T_win. Counting over the full simulation
would include the burn-in transient that --trim_head_s exists to remove,
and would call a network active on the strength of spikes that never
entered its embedding.

Output
------
<out>.npz with
    sim_rate      (n_sim,)  Hz/electrode, ALIGNED to the row order that
                            gate_data.load_sim produces (sorted glob, shards
                            concatenated in that order)
    sim_spikes    (n_sim,)  raw in-window spike counts
    sim_found     (n_sim,)  bool, False where the mea file could not be located
    real_rate     (n_real,) Hz/electrode, one entry per exported real row
    meta          json blob

Both arms' row orders match what gate_run.py loads, so the arrays can be
used as masks directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def _index_mea_tree(mea_root: str) -> dict:
    """(campaign, sweep, topo, iter) -> path, built with ONE directory walk.

    86k individual globs would dominate the runtime; one walk is seconds.
    """
    idx = {}
    for dirpath, dirnames, filenames in os.walk(mea_root):
        base = os.path.basename(dirpath)
        if not base.startswith("topo_"):
            continue
        parts = dirpath.split(os.sep)
        if len(parts) < 3:
            continue
        sweep = parts[-2]
        camp = parts[-3]
        try:
            topo = int(base.split("_")[-1])
        except ValueError:
            continue
        for fn in filenames:
            if not (fn.startswith("mea_iter_") and fn.endswith(".npz")):
                continue
            try:
                it = int(os.path.splitext(fn)[0].split("_")[-1])
            except ValueError:
                continue
            idx[(camp, sweep, topo, it)] = os.path.join(dirpath, fn)
    return idx


def sim_rates(pattern: str, mea_root: str, trim_head_s: float,
              window_s: float, n_e: int, verbose: bool = True):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("no shards match %r" % (pattern,))
    if verbose:
        print("indexing the MEA tree ...", flush=True)
    idx = _index_mea_tree(mea_root)
    if verbose:
        print("  %d mea_iter files indexed" % len(idx), flush=True)

    t0, t1 = float(trim_head_s), float(trim_head_s) + float(window_s)
    spikes, found = [], []
    for si, p in enumerate(paths):
        t = pq.read_table(p, columns=["campaign_id", "topo_idx", "iter_idx"])
        cid = t["campaign_id"].to_pylist()
        ti = t["topo_idx"].to_pylist()
        ii = t["iter_idx"].to_pylist()
        for c, tt, it in zip(cid, ti, ii):
            camp, sweep = str(c).split("__", 1)
            f = idx.get((camp, sweep, int(tt), int(it)))
            if f is None:
                spikes.append(-1)
                found.append(False)
                continue
            with np.load(f, allow_pickle=False) as z:
                dt = np.asarray(z["det_t"], dtype=np.float64).ravel()
            spikes.append(int(np.count_nonzero((dt >= t0) & (dt < t1))))
            found.append(True)
        if verbose:
            print("  shard %2d/%d  %-46s rows=%d"
                  % (si + 1, len(paths), os.path.basename(p)[:46], len(cid)),
                  flush=True)

    spikes = np.asarray(spikes, dtype=np.int64)
    found = np.asarray(found, dtype=bool)
    rate = np.where(found, spikes / (float(n_e) * float(window_s)), np.nan)
    return rate, spikes, found


def real_rates(real_parquet: str, specs_path: str, window_s: float,
               fs_ifr: float, verbose: bool = True):
    """mean(ifr_trace) * fs_ifr per exported window, in row order."""
    t = pq.read_table(real_parquet, columns=["name", "window_idx"])
    names = [str(x) for x in t["name"].to_pylist()]
    wins = [int(x) for x in t["window_idx"].to_pylist()]

    with open(specs_path) as fh:
        specs = json.load(fh)
    by_name = {str(s["name"]): s["path"] for s in specs}

    W = int(round(window_s * fs_ifr))
    cache = {}
    out = np.full(len(names), np.nan)
    for i, (nm, w) in enumerate(zip(names, wins)):
        # the exported 'name' may carry a suffix; match the longest spec name
        # that is a prefix of it, rather than assuming an exact key
        p = by_name.get(nm)
        if p is None:
            cand = [k for k in by_name if nm.startswith(k)]
            if not cand:
                continue
            p = by_name[max(cand, key=len)]
        if p not in cache:
            with np.load(p, allow_pickle=False) as d:
                k = "ifr_trace" if "ifr_trace" in d.files else "X"
                cache[p] = np.asarray(d[k], dtype=np.float64).ravel()
        tr = cache[p]
        a, b = w * W, (w + 1) * W
        if b <= tr.size:
            out[i] = float(np.mean(tr[a:b])) * float(fs_ifr)
    if verbose:
        print("  real rows resolved: %d / %d"
              % (int(np.isfinite(out).sum()), out.size))
    return out


def describe(name, x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print("%-6s (no finite values)" % name)
        return
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("%-6s n=%-7d  min=%.4g  p1=%.4g  p5=%.4g  p25=%.4g  med=%.4g  "
          "p75=%.4g  p95=%.4g  p99=%.4g  max=%.4g"
          % (name, x.size, *q))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", required=True, help="glob for simulated shards")
    ap.add_argument("--mea_root", required=True)
    ap.add_argument("--real", required=True, help="real cohort parquet")
    ap.add_argument("--specs", required=True, help="specs_real.json")
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--trim_head_s", type=float, default=20.0)
    ap.add_argument("--window_s", type=float, default=180.0)
    ap.add_argument("--fs_ifr", type=float, default=100.0)
    ap.add_argument("--n_electrodes", type=int, default=9)
    args = ap.parse_args()

    print("[1/2] simulated arm")
    sr, ss, sf = sim_rates(args.sim, args.mea_root, args.trim_head_s,
                           args.window_s, args.n_electrodes)
    print("[2/2] real arm")
    rr = real_rates(args.real, args.specs, args.window_s, args.fs_ifr)

    print("")
    print("firing rate, Hz per electrode, on the SAME %g s window:"
          % args.window_s)
    describe("sim", sr)
    describe("real", rr)

    finite_r = rr[np.isfinite(rr)]
    if finite_r.size:
        lo = float(finite_r.min())
        print("")
        print("lowest real rate observed : %.5g Hz/electrode" % lo)
        for frac, tag in ((1.0, "at the real minimum"),
                          (0.5, "half the real minimum"),
                          (0.1, "a tenth of the real minimum")):
            thr = lo * frac
            keep = int(np.count_nonzero(sr >= thr))
            print("  cutoff %-26s %.5g Hz -> keeps %d / %d sim rows (%.1f%%)"
                  % (tag, thr, keep, sr.size, 100.0 * keep / sr.size))
        z = int(np.count_nonzero(sr == 0.0))
        print("  simulations with EXACTLY zero in-window spikes: %d (%.1f%%)"
              % (z, 100.0 * z / sr.size))
        nf = int(np.count_nonzero(~sf))
        if nf:
            print("  WARNING: %d sim rows had no locatable mea file" % nf)

    np.savez_compressed(
        args.out + ".npz", sim_rate=sr, sim_spikes=ss, sim_found=sf,
        real_rate=rr,
        meta=json.dumps({"sim_pattern": args.sim, "mea_root": args.mea_root,
                         "real": args.real, "specs": args.specs,
                         "trim_head_s": args.trim_head_s,
                         "window_s": args.window_s, "fs_ifr": args.fs_ifr,
                         "n_electrodes": args.n_electrodes,
                         "units": "Hz per electrode"}))
    print("")
    print("wrote %s.npz" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
