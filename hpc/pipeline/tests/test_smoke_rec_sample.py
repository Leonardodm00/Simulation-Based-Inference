#!/usr/bin/env python3
"""Smoke test of block 35: the sampler on a recorded project (specs/SPEC.md block 35, "Test
oracles", checks R35*).

CONTRACT SKELETON (wave B): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 35's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r35a_as_recorded    D-162's first target: the arrays equal block 19's, the draws bit for bit
    test_r35b_recorded       the recorded-kind project with k_draw = 1: draw rows, z_idx, roles, the bank's file
    test_r35c_contract       every key, shape and dtype; the same seed, identical files
    test_r35d_refusals       five refusals that write nothing
    test_r35e_mixture        the mixture density at the draws, from the members in float64

Run
    python3 hpc/pipeline/tests/test_smoke_rec_sample.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built
        before (rec_fixture.py); REC_KEEP=1 keeps the one this run builds.

Pure ASCII, LF only.
"""

import os
import sys
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
_EVAL = os.path.join(_PIPE, "evaluation")
for _p in (_HERE, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rec_fixture  # noqa: E402

SEED = 20261123  # every random number of this file derives from it

def test_r35a_as_recorded():
    """D-162's first target: the arrays equal block 19's, the draws bit for bit (specs/SPEC.md block 35)."""
    raise NotImplementedError("test_r35a_as_recorded: not written yet")


def test_r35b_recorded():
    """the recorded-kind project with k_draw = 1: draw rows, z_idx, roles, the bank's file (specs/SPEC.md block 35)."""
    raise NotImplementedError("test_r35b_recorded: not written yet")


def test_r35c_contract():
    """every key, shape and dtype; the same seed, identical files (specs/SPEC.md block 35)."""
    raise NotImplementedError("test_r35c_contract: not written yet")


def test_r35d_refusals():
    """five refusals that write nothing (specs/SPEC.md block 35)."""
    raise NotImplementedError("test_r35d_refusals: not written yet")


def test_r35e_mixture():
    """the mixture density at the draws, from the members in float64 (specs/SPEC.md block 35)."""
    raise NotImplementedError("test_r35e_mixture: not written yet")


CHECKS = [("test_r35a_as_recorded", test_r35a_as_recorded),
          ("test_r35b_recorded", test_r35b_recorded),
          ("test_r35c_contract", test_r35c_contract),
          ("test_r35d_refusals", test_r35d_refusals),
          ("test_r35e_mixture", test_r35e_mixture)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    try:
        for name, fn in CHECKS:
            if only and name not in only:
                continue
            t1 = time.time()
            try:
                fn()
                rows.append((name, "PASS", "", time.time() - t1))
            except NotImplementedError as exc:
                bad += 1
                rows.append((name, "TODO", str(exc), time.time() - t1))
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        rec_fixture.cleanup()
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
