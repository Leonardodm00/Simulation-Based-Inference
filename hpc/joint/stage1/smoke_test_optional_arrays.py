#!/usr/bin/env python3
"""Smoke test of the shard contract's optional row arrays (the contract commit
of wave W3, 2026-11-17; SBI D-148 (3), (4); specs/SPEC.md block 31 and
Sbi-extractor's Block 9): `latent_bank.OPTIONAL_ROW_ARRAYS` = {"rep_group"}.

Checks: a shard without rep_group passes as before; with an int64 (n,)
rep_group it passes, is written and read back unchanged and concatenates
with a shard that also holds it; a float, a 2-d or a short rep_group, and an
unknown array, are refused with the reason; two shards that disagree on
holding it do not concatenate.

Run: python3 hpc/joint/stage1/smoke_test_optional_arrays.py   (numpy; < 1 s)
Last line on success: OPTIONAL ARRAYS OK

Pure ASCII, LF only.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latent_bank as LB    # noqa: E402


def base(n=5, W=4):
    return {"x": np.zeros((n, W), np.float32), "theta": np.zeros((n, 0), np.float32),
            "cls": np.zeros(n, np.int64), "nu": np.zeros((n, 5), np.float32),
            "realisation_id": np.arange(n, dtype=np.uint64), "donor": np.arange(n),
            "well": np.arange(n), "batch": np.zeros(n, np.int64),
            "subregion": np.zeros(n, np.int64), "window_idx": np.zeros(n, np.int64),
            "contaminated": np.zeros(n, bool)}


def main():
    n, W = 5, 4
    assert LB.check_arrays(base(n, W)) == n
    a = dict(base(n, W), rep_group=np.array([0, 0, 1, 1, 2]))
    assert LB.check_arrays(a) == n
    for bad, why in ((dict(base(n, W), rep_group=np.zeros(n, np.float32)), "dtype kind"),
                     (dict(base(n, W), rep_group=np.zeros((n, 1), np.int64)), "ndim"),
                     (dict(base(n, W), rep_group=np.zeros(n - 1, np.int64)), "rows"),
                     (dict(base(n, W), foo=np.zeros(n)), "unknown")):
        try:
            LB.check_arrays(bad)
        except LB.ContractError as exc:
            assert why in str(exc), (why, exc)
        else:
            raise AssertionError("not refused: %s" % why)
    side = LB.build_sidecar([], [], "unit_box", 100.0, 0.01, 0.04, W, "per_unit_mean", {}, {}, {},
                            {}, "0" * 64, False, "R")
    d = tempfile.mkdtemp()
    try:
        p1, p2 = os.path.join(d, "shard_0000.npz"), os.path.join(d, "shard_0001.npz")
        LB.write_shard(p1, a, side)
        LB.write_shard(p2, base(n, W), side)
        arr, _ = LB.read_shard(p1)
        assert arr["rep_group"].tolist() == [0, 0, 1, 1, 2]
        merged, _ = LB.concat_shards([p1, p1])
        assert merged["rep_group"].shape == (2 * n,)
        try:
            LB.concat_shards([p1, p2])
        except LB.ContractError as exc:
            assert "holds the arrays" in str(exc), exc
        else:
            raise AssertionError("shards that disagree on rep_group were concatenated")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("OPTIONAL ARRAYS OK")


if __name__ == "__main__":
    main()
