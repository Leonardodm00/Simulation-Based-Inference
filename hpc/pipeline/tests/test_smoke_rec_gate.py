#!/usr/bin/env python3
"""Smoke test of block 34: the gate (specs/SPEC.md block 34, "Test
oracles", checks R34*).

CONTRACT SKELETON (wave A): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 34's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r34a_reference    member_statistic equals npe_misspec.mmd2_multiscale point by point
    test_r34b_by_hand      t, r, T and p from loops over (R5)-(R8) on a tiny case
    test_r34c_null_as_test the null rows as test rows: T = T_null, p by hand
    test_r34d_size         the p-value's size on toy embeddings of one law
    test_r34e_power        the flagged share against the displacement
    test_r34f_roles        assign_roles: groups, caps, order, determinism, refusal
    test_r34g_edges        non-finite input refused; no test row

Run
    python3 hpc/pipeline/tests/test_smoke_rec_gate.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

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

SEED = 20261123  # every random number of this file derives from it

def test_r34a_reference():
    """member_statistic equals npe_misspec.mmd2_multiscale point by point (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34a_reference: not written yet")


def test_r34b_by_hand():
    """t, r, T and p from loops over (R5)-(R8) on a tiny case (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34b_by_hand: not written yet")


def test_r34c_null_as_test():
    """the null rows as test rows: T = T_null, p by hand (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34c_null_as_test: not written yet")


def test_r34d_size():
    """the p-value's size on toy embeddings of one law (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34d_size: not written yet")


def test_r34e_power():
    """the flagged share against the displacement (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34e_power: not written yet")


def test_r34f_roles():
    """assign_roles: groups, caps, order, determinism, refusal (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34f_roles: not written yet")


def test_r34g_edges():
    """non-finite input refused; no test row (specs/SPEC.md block 34)."""
    raise NotImplementedError("test_r34g_edges: not written yet")


CHECKS = [("test_r34a_reference", test_r34a_reference), ("test_r34b_by_hand", test_r34b_by_hand), ("test_r34c_null_as_test", test_r34c_null_as_test), ("test_r34d_size", test_r34d_size), ("test_r34e_power", test_r34e_power), ("test_r34f_roles", test_r34f_roles), ("test_r34g_edges", test_r34g_edges)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
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
