#!/usr/bin/env python3
"""The ensemble's report-split NLL from the members' per-row files
(block 7; D-089; specs/SPEC.md sec. 2.3).

With l_{m,i} = -log q_{omega_m}(theta_i | h_{psi_m}(x_i)) member m's NLL on
report row i (the runner's `<stem>_perrow.npz`, array `nll`, float64), the
uniform mixture q_bar = (1/M) sum_m q_{omega_m} has

    L_bar = -(1/n) sum_{i=1}^{n} [ logsumexp_m( -l_{m,i} ) - log M ].

The members must score the SAME report rows in the SAME order, which the
shared donor split guarantees and which this tool checks: the per-row
`group` (donor) and `theta` arrays of every member must equal the first
member's exactly; otherwise the files are not an ensemble and the tool
refuses. `scipy.special.logsumexp` does the sum.

    python3 ensemble_nll.py --out <file.json> --members <m1_perrow.npz> <m2_perrow.npz> ...

Prints the pass line

    ENSEMBLE OK: M=<M> n=<n> L_bar=<L_bar> members=<L_1>,...,<L_M>

and writes `--out` (L_bar, the members' L_m = mean_i l_{m,i}, n, M, the
files). Property used by the smoke test: M copies of one member give
L_bar = L_1 (logsumexp of M equal values -l is -l + log M). Runs in the
training environment (numpy, scipy). Not sequencer code.

Pure ASCII, LF only.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.special import logsumexp


def ensemble_nll(nll_rows):
    """nll_rows: (M, n) float64 of l_{m,i}. Returns (L_bar, [L_m])."""
    a = np.asarray(nll_rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("need an (M, n) array of per-row NLLs, got shape %r" % (a.shape,))
    m = a.shape[0]
    mix = logsumexp(-a, axis=0) - np.log(float(m))      # log q_bar(theta_i | x_i)
    return float(-np.mean(mix)), [float(x) for x in np.mean(a, axis=1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True, help="the members' _perrow.npz files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows, groups, thetas = [], [], []
    for p in args.members:
        with np.load(p) as z:
            if "nll" not in z.files:
                raise SystemExit("%s: no 'nll' array (not a runner per-row file)" % p)
            rows.append(np.asarray(z["nll"], dtype=np.float64).ravel())
            groups.append(np.asarray(z["group"]).ravel() if "group" in z.files else None)
            thetas.append(np.asarray(z["theta"]) if "theta" in z.files else None)
    n = rows[0].size
    for p, r, g, th in zip(args.members, rows, groups, thetas):
        if r.size != n:
            raise SystemExit("%s: %d report rows, the first member has %d: not the same split"
                             % (p, r.size, n))
        if not np.all(np.isfinite(r)):
            raise SystemExit("%s: non-finite per-row NLL" % p)
        if groups[0] is not None and g is not None and not np.array_equal(g, groups[0]):
            raise SystemExit("%s: the report rows' donors differ from the first member's: "
                             "not the same split" % p)
        if thetas[0] is not None and th is not None and not np.array_equal(th, thetas[0]):
            raise SystemExit("%s: the report rows' theta differ from the first member's: "
                             "not the same split or order" % p)
    L_bar, members = ensemble_nll(np.stack(rows))
    rec = {"schema": "ensemble_nll/1", "M": len(rows), "n_report": int(n),
           "L_bar": L_bar, "L_members": members, "L_members_mean": float(np.mean(members)),
           "members": [os.path.abspath(p) for p in args.members],
           "formula": "L_bar = -(1/n) sum_i [logsumexp_m(-l_{m,i}) - log M]"}
    d = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, args.out)
    print("ENSEMBLE OK: M=%d n=%d L_bar=%.12g members=%s"
          % (len(rows), n, L_bar, ",".join("%.12g" % x for x in members)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
