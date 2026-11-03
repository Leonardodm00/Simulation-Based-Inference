#!/usr/bin/env python3
"""Check one bank of a synthetic data set and write its record (block 16 of
specs/SPEC.md; the pipeline's stages sim_bank and real_bank_<r> run it as a
job once every shard is built).

    python3 synth_bank_record.py --bank-dir <dir> --expect <expect.json> \\
        --out <dir>/BANK_RECORD.json

<expect.json> is the stage's (stages/synthetic.py, SynthBankStage.expect):
the build (per shard, the builder's arguments without --out-dir; the rows),
the arm, W, theta's width, the wells per donor, the windows per well, the
classes and the donors per class (arm R), the latent kind (`box` on arm S).

The checks, every shard k < K of the build:
    exactly the K pairs shard_%04d.{npz,json} (no other index)
    every zip member's CRC-32 (ZipFile.testzip)
    every sidecar's digest recomputed (latent_bank.contract_digest), and one
    digest for the whole bank
    the sidecar's arm, theta_withheld (true on arm R only), W, latent_spec
    kind (`box` on arm S, absent on arm R), and its builder arguments
    (provenance.argv without --out-dir) equal to the build's
    the rows of x, its width W, theta's width; every donor with exactly
    wells_per_donor wells
    arm S: every cls is -1; arm R: for c = 0, ..., n_classes - 1 exactly
    donors_per_class * wells_per_donor * n_windows rows of class c, no other
    class
On success: <out> (schema synth_bank_record/1) then <out>.sha256, both
written atomically, and the pass line

    SYNTH BANK RECORD OK: <out> (<K> shard(s), <n> rows, digest <16 hex>)

On a problem: one `REFUSE: <problem>` line each, nothing written, exit 4.

Runs in the training environment (numpy). Pure ASCII, LF only.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.abspath(os.path.join(_HERE, "..", "..", "joint", "stage1"))
if _STAGE1 not in sys.path:
    sys.path.insert(0, _STAGE1)

import latent_bank as LB    # noqa: E402

SCHEMA = "synth_bank_record/1"
_SHARD = re.compile(r"^shard_([0-9]+)\.(npz|json)$")
EXIT_REFUSE = 4


def normalise(argv):
    """A builder's argument list without `--out-dir` and its value (the
    stage's rule, stages/synthetic.py _normalise)."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--out-dir":
            skip = True
            continue
        out.append(str(a))
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def check_shard(bank_dir, k, exp, problems):
    """The checks of shard k; returns its record entry or None."""
    build = exp["build"]
    npz = os.path.join(bank_dir, "shard_%04d.npz" % k)
    js = os.path.join(bank_dir, "shard_%04d.json" % k)
    missing = [os.path.basename(p) for p in (npz, js) if not os.path.isfile(p)]
    if missing:
        problems.append("shard %d: %s missing" % (k, " and ".join(missing)))
        return None
    try:
        with open(js, "r") as fh:
            side = json.load(fh)
    except (OSError, ValueError) as exc:
        problems.append("shard %d: its sidecar cannot be read (%s)" % (k, exc))
        return None
    try:
        digest = LB.contract_digest(side)
    except LB.ContractError as exc:
        problems.append("shard %d: %s" % (k, exc))
        return None
    if side.get("contract_digest") != digest:
        problems.append("shard %d: its sidecar's digest does not match its contract fields" % k)
    got = normalise((side.get("provenance") or {}).get("argv") or [])
    if got != list(build["argv"][k]):
        problems.append("shard %d: built with the arguments `%s`, the build's are `%s`"
                        % (k, " ".join(got), " ".join(build["argv"][k])))
    if side.get("arm") != exp["arm"]:
        problems.append("shard %d: arm %r, expected %r" % (k, side.get("arm"), exp["arm"]))
    if bool(side.get("theta_withheld")) != bool(exp["theta_withheld"]):
        problems.append("shard %d: theta_withheld %r, expected %r"
                        % (k, side.get("theta_withheld"), exp["theta_withheld"]))
    if side.get("W") != int(exp["W"]):
        problems.append("shard %d: W %r, expected %d (round(T_win * fs))" % (k, side.get("W"),
                                                                             int(exp["W"])))
    kind = (side.get("latent_spec") or {}).get("kind")
    if kind != exp["latent_kind"]:
        problems.append("shard %d: latent_spec kind %r, expected %r" % (k, kind, exp["latent_kind"]))
    try:
        with zipfile.ZipFile(npz) as z:
            bad = z.testzip()
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError, ValueError, NotImplementedError) as exc:
        problems.append("shard %d: %s is not a readable zip archive (%s)" % (k, os.path.basename(npz),
                                                                             exc))
        return None
    if bad is not None:
        problems.append("shard %d: member %s of %s fails its CRC-32" % (k, bad, os.path.basename(npz)))
        return None
    with np.load(npz) as z:
        x_shape = LB.npz_member_shape(z, "x")
        th_shape = LB.npz_member_shape(z, "theta")
        cls = np.asarray(z["cls"])
        donor = np.asarray(z["donor"])
        well = np.asarray(z["well"])
    rows = int(x_shape[0])
    if rows != int(build["rows_per_shard"][k]):
        problems.append("shard %d: %d rows, the build's %d" % (k, rows, int(build["rows_per_shard"][k])))
    if len(x_shape) != 2 or int(x_shape[1]) != int(exp["W"]):
        problems.append("shard %d: x of shape %r, windows of W = %d expected" % (k, x_shape,
                                                                                int(exp["W"])))
    if len(th_shape) != 2 or int(th_shape[1]) != int(exp["d_theta"]):
        problems.append("shard %d: theta of shape %r: %s columns, search.shape_anchors.d_theta is %d"
                        % (k, th_shape, th_shape[1] if len(th_shape) == 2 else "?",
                           int(exp["d_theta"])))
    wells = {}
    for d, w in set(zip(donor.tolist(), well.tolist())):
        wells[d] = wells.get(d, 0) + 1
    off = sorted(d for d, n in wells.items() if n != int(exp["wells_per_donor"]))
    if off:
        problems.append("shard %d: %d donor(s) without exactly %d wells (e.g. donor %d: %d)"
                        % (k, len(off), int(exp["wells_per_donor"]), off[0], wells[off[0]]))
    if exp["arm"] == "S":
        if cls.size and not bool(np.all(cls == -1)):
            problems.append("shard %d: arm S rows with a class other than -1: %s"
                            % (k, sorted(set(np.unique(cls).tolist()) - set([-1]))[:6]))
    else:
        want = int(exp["donors_per_class"]) * int(exp["wells_per_donor"]) * int(exp["n_windows"])
        for c in range(int(exp["n_classes"])):
            n_c = int((cls == c).sum())
            if n_c != want:
                problems.append("shard %d: %d rows of class %d, expected %d (donors_per_class x "
                                "wells_per_donor x n_windows)" % (k, n_c, c, want))
        other = sorted(set(np.unique(cls).tolist()) - set(range(int(exp["n_classes"]))))
        if other:
            problems.append("shard %d: classes outside 0..%d: %s" % (k, int(exp["n_classes"]) - 1,
                                                                    other[:6]))
    return {"file": os.path.basename(npz), "sha256": sha256_of(npz), "bytes": os.path.getsize(npz),
            "rows": rows, "sidecar": os.path.basename(js), "sidecar_sha256": sha256_of(js),
            "contract_digest": side.get("contract_digest")}


def check_bank(bank_dir, exp):
    """(problems, shard entries)."""
    problems = []
    if not os.path.isdir(bank_dir):
        return ["no bank folder %s" % bank_dir], []
    if os.path.abspath(bank_dir) != os.path.abspath(exp["dir"]):
        problems.append("--bank-dir %s is not the expectation's folder %s" % (bank_dir, exp["dir"]))
    n = int(exp["build"]["n_shards"])
    extra = []
    for p in sorted(os.listdir(bank_dir)):
        m = _SHARD.match(p)
        if m and int(m.group(1)) >= n:
            extra.append(p)
    if extra:
        problems.append("shard files of an index >= %d (the build's shards): %s%s"
                        % (n, ", ".join(extra[:6]), " ..." if len(extra) > 6 else ""))
    entries = []
    for k in range(n):
        e = check_shard(bank_dir, k, exp, problems)
        if e is not None:
            entries.append(e)
    digests = sorted(set(e["contract_digest"] for e in entries))
    if len(digests) > 1:
        problems.append("the shards carry %d contract digests (one bank has one): %s"
                        % (len(digests), ", ".join(d[:16] for d in digests)))
    return problems, entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--expect", required=True, help="the stage's expectation (JSON)")
    ap.add_argument("--out", required=True, help="the record to write (BANK_RECORD.json)")
    args = ap.parse_args(argv)
    with open(args.expect, "r") as fh:
        exp = json.load(fh)
    problems, entries = check_bank(args.bank_dir, exp)
    if problems:
        for p in problems:
            print("REFUSE: %s" % p)
        print("SYNTH BANK RECORD REFUSED: %d problem(s); nothing written" % len(problems))
        return EXIT_REFUSE
    rec = {"schema": SCHEMA, "bank": exp["bank"], "dir": os.path.abspath(args.bank_dir),
           "n_shards": len(entries), "n_rows": int(sum(e["rows"] for e in entries)),
           "W": int(exp["W"]), "d_theta": int(exp["d_theta"]),
           "contract_digest": entries[0]["contract_digest"], "arm": exp["arm"],
           "latent_kind": exp["latent_kind"], "theta_withheld": bool(exp["theta_withheld"]),
           "shards": entries, "build": exp["build"],
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "tool": "hpc/pipeline/tools/synth_bank_record.py"}
    blob = json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    _write_atomic(args.out, blob)
    _write_atomic(args.out + ".sha256", "%s  %s\n" % (hashlib.sha256(blob.encode("ascii")).hexdigest(),
                                                      os.path.basename(args.out)))
    print("SYNTH BANK RECORD OK: %s (%d shard(s), %d rows, digest %s)"
          % (args.out, rec["n_shards"], rec["n_rows"], rec["contract_digest"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
