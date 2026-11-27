#!/usr/bin/env python3
"""Smoke test of block 45: inference from a bundle (specs/SPEC.md block 45, "Test
oracles" and "Contract of wave C", checks T45*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t45a_evaluation     the evaluation's numbers: block 35's file bit for bit, block 36's
    test_t45b_mixture        the mixture over the windows, by numpy on the pooled draws
    test_t45c_contract       keys, shapes, dtypes; one seed, one result; seed and batch matter
    test_t45d_end_to_end     a ptrain folder through windows and infer by the command line
    test_t45e_refusals       n_post not a multiple of M, another W, a broken bundle

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_infer.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

Pure ASCII, LF only.
"""

import os
import shutil
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SEED = 20261127  # every random number of this file derives from it
_TMP = []


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


def test_t45a_evaluation():
    """T45a: on the fixture's bank, with the sample's seed and the bundle's n_post and batch, draws,
    draw_member, logq_draws, logq_draws_own and z equal block 35's file bit for bit; the
    summaries, the gate's arrays and the flags equal block 36's."""
    raise NotImplementedError("TODO test_t45a_evaluation")


def test_t45b_mixture():
    """T45b: mix_* equal numpy's mean, SD (ddof 1) and quantiles of the result's own draws pooled
    over the windows as float64, within 1e-12."""
    raise NotImplementedError("TODO test_t45b_mixture")


def test_t45c_contract():
    """T45c: every key, shape and dtype of ARRAYS and JSON_KEYS; two runs with one seed give
    identical files; another seed gives other draws and the same z; another batch gives other
    draws when n exceeds it."""
    raise NotImplementedError("TODO test_t45c_contract")


def test_t45d_end_to_end():
    """T45d: a ptrain folder written by the test, through `windows` and `infer` by the command
    line with a bundle whose extraction fits it: exit 0, n windows as T44c counts them; the
    same through `infer --recording`."""
    raise NotImplementedError("TODO test_t45d_end_to_end")


def test_t45e_refusals():
    """T45e: n_post not a multiple of M, windows of another length than W, a bundle that fails
    its check, --option detection: refused, nothing written."""
    raise NotImplementedError("TODO test_t45e_refusals")


CHECKS = [("test_t45a_evaluation", test_t45a_evaluation),
          ("test_t45b_mixture", test_t45b_mixture),
          ("test_t45c_contract", test_t45c_contract),
          ("test_t45d_end_to_end", test_t45d_end_to_end),
          ("test_t45e_refusals", test_t45e_refusals)]


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
        _cleanup()
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
