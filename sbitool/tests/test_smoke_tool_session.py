#!/usr/bin/env python3
"""Smoke test of block 49: the session engine (specs/SPEC.md block 49, "Test
oracles", checks T49*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t49a_describe       T49a: the electrodes and their spikes equal the folder's; spans
    test_t49b_statistics     T49b: counts, rates and the CV against numpy on planted trains
    test_t49c_exclusion      T49c: excluded electrodes as silent ones for the partition
    test_t49d_windows        T49d: the windows of a span; infer equals block 45's on them
    test_t49e_record         T49e: refusals; the result's keys; JSON-serialisable

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_session.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<a kept fixture> reuses it.

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

SEED = 20261128  # every random number of this file derives from it
_TMP = []


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


def test_t49a_describe():
    """T49a: the electrodes and their spikes equal the folder's; spans (SPEC block 49)."""
    raise NotImplementedError("TODO test_t49a_describe")


def test_t49b_statistics():
    """T49b: counts, rates and the CV against numpy on planted trains (SPEC block 49)."""
    raise NotImplementedError("TODO test_t49b_statistics")


def test_t49c_exclusion():
    """T49c: excluded electrodes as silent ones for the partition (SPEC block 49)."""
    raise NotImplementedError("TODO test_t49c_exclusion")


def test_t49d_windows():
    """T49d: the windows of a span; infer equals block 45's on them (SPEC block 49)."""
    raise NotImplementedError("TODO test_t49d_windows")


def test_t49e_record():
    """T49e: refusals; the result's keys; JSON-serialisable (SPEC block 49)."""
    raise NotImplementedError("TODO test_t49e_record")


CHECKS = [("test_t49a_describe", test_t49a_describe),
          ("test_t49b_statistics", test_t49b_statistics),
          ("test_t49c_exclusion", test_t49c_exclusion),
          ("test_t49d_windows", test_t49d_windows),
          ("test_t49e_record", test_t49e_record)]


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
