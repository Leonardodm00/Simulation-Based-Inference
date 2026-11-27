#!/usr/bin/env python3
"""Smoke test of block 44: from a recording to windows (specs/SPEC.md block 44, "Test
oracles", checks T44*).

CONTRACT SKELETON (wave B): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 44's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t44a_ifr            the pooled IFR by its definition, computed independently
    test_t44b_partition      the subregions equal the extractor's command-line result
    test_t44c_windows        the windows' count, order, start times and dtype
    test_t44d_refusals       another rate, grid or pitch; a bundle without extraction

Run
    python3 sbitool/tests/test_smoke_tool_windows.py [check ...]
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


def test_t44a_ifr():
    """the pooled IFR by its definition, computed independently (specs/SPEC.md block 44)."""
    raise NotImplementedError("test_t44a_ifr: not written yet")


def test_t44b_partition():
    """the subregions equal the extractor's command-line result (specs/SPEC.md block 44)."""
    raise NotImplementedError("test_t44b_partition: not written yet")


def test_t44c_windows():
    """the windows' count, order, start times and dtype (specs/SPEC.md block 44)."""
    raise NotImplementedError("test_t44c_windows: not written yet")


def test_t44d_refusals():
    """another rate, grid or pitch; a bundle without extraction (specs/SPEC.md block 44)."""
    raise NotImplementedError("test_t44d_refusals: not written yet")


CHECKS = [("test_t44a_ifr", test_t44a_ifr),
          ("test_t44b_partition", test_t44b_partition),
          ("test_t44c_windows", test_t44c_windows),
          ("test_t44d_refusals", test_t44d_refusals)]


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
