#!/usr/bin/env python3
"""The bank's identity for the search (block 7): the grouped split's hash,
the contract digest and the row counts, read from the simulated bank's
small arrays without opening a window, and computed by the SAME function
the runner uses (`run_joint_arms.grouped_split`), so the hash the pipeline
folds into every trial id is the hash every trial's record carries.

    python3 bank_split_hash.py --sim-shards '<banks_root>/<tag>_sim/shard_*.npz' \\
        --split-fracs 0.85 0.10 0.05 --split-seed 0 --out <file.json>

Prints one line, the pass line:

    SPLIT OK: hash=<64 hex> contract=<64 hex> rows=<n> train/sel/report=<a>/<b>/<c>
    donors=<d> (<a'>/<b'>/<c'>) shards=<k> W=<W> d_theta=<p>

and writes the same numbers to `--out`. Runs in the training environment
(numpy, torch: the runner's module imports torch). Not sequencer code:
the sequencer runs it through `envs.run_in_env`.

Pure ASCII, LF only.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JOINT = os.path.abspath(os.path.join(_HERE, "..", "..", "joint"))
for _p in (os.path.join(_JOINT, "stage3"), os.path.join(_JOINT, "stage1"),
           os.path.join(_JOINT, "stage2"), _JOINT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latent_bank import ShardIndex          # noqa: E402
from run_joint_arms import grouped_split    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-shards", required=True, help="glob of the simulated shards")
    ap.add_argument("--split-fracs", type=float, nargs=3, required=True)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="the JSON file to write")
    args = ap.parse_args(argv)

    fracs = tuple(float(f) for f in args.split_fracs)
    if abs(sum(fracs) - 1.0) > 1e-9 or any(f <= 0.0 for f in fracs):
        raise SystemExit("--split-fracs must be three positive fractions summing to 1, got %r"
                         % (fracs,))
    paths = sorted(glob.glob(args.sim_shards))
    if not paths:
        raise SystemExit("no shards matched %r" % args.sim_shards)
    index = ShardIndex(paths)
    donor = np.asarray(index.donor).ravel()
    split, split_hash = grouped_split(donor, fracs=fracs, seed=int(args.split_seed))
    n_rows = [int((split == k).sum()) for k in (0, 1, 2)]
    n_donors = [int(len(np.unique(donor[split == k]))) for k in (0, 1, 2)]
    if n_rows[0] == 0 or n_rows[2] == 0:
        raise SystemExit("split has an empty train or report part (%d/%d/%d rows): the bank "
                         "has too few donors" % tuple(n_rows))
    rec = {"schema": "bank_split_hash/1",
           "sim_shards": args.sim_shards, "n_shards": len(paths),
           "split_fracs": list(fracs), "split_seed": int(args.split_seed),
           "split_hash": split_hash,
           "contract_digest": index.sidecar["contract_digest"],
           "latent_kind": (index.sidecar.get("latent_spec") or {}).get("kind"),
           "n_rows": int(index.n_rows), "n_train": n_rows[0], "n_sel": n_rows[1],
           "n_report": n_rows[2], "n_donors": int(len(np.unique(donor))),
           "n_donors_split": n_donors, "W": int(index.W),
           "d_theta": int(np.asarray(index.theta).shape[1])}
    d = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, args.out)
    print("SPLIT OK: hash=%s contract=%s rows=%d train/sel/report=%d/%d/%d donors=%d (%d/%d/%d) "
          "shards=%d W=%d d_theta=%d"
          % (split_hash, rec["contract_digest"], rec["n_rows"], n_rows[0], n_rows[1], n_rows[2],
             rec["n_donors"], n_donors[0], n_donors[1], n_donors[2], len(paths), rec["W"],
             rec["d_theta"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
