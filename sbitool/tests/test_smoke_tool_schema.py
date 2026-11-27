#!/usr/bin/env python3
"""Smoke test of block 40: the tool's package, its schemas and its interfaces (specs/SPEC.md block 40, "Test
oracles", checks T40*).

CONTRACT SKELETON (wave A): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 40's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t40a_validators   a valid manifest and card; a table of single faults, each named
    test_t40b_recording    Recording.check and duration_s
    test_t40c_sources      the source groups: the modules imported, their digests; SourceError
    test_t40d_cli_static   python3 -m sbitool without a command; the package's imports

Run
    python3 sbitool/tests/test_smoke_tool_schema.py [check ...]
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


def test_t40a_validators():
    """a valid manifest and card; a table of single faults, each named (specs/SPEC.md block 40)."""
    raise NotImplementedError("test_t40a_validators: not written yet")


def test_t40b_recording():
    """Recording.check and duration_s (specs/SPEC.md block 40)."""
    raise NotImplementedError("test_t40b_recording: not written yet")


def test_t40c_sources():
    """the source groups: the modules imported, their digests; SourceError (specs/SPEC.md block 40)."""
    raise NotImplementedError("test_t40c_sources: not written yet")


def test_t40d_cli_static():
    """python3 -m sbitool without a command; the package's imports (specs/SPEC.md block 40)."""
    raise NotImplementedError("test_t40d_cli_static: not written yet")


CHECKS = [("test_t40a_validators", test_t40a_validators),
          ("test_t40b_recording", test_t40b_recording),
          ("test_t40c_sources", test_t40c_sources),
          ("test_t40d_cli_static", test_t40d_cli_static)]


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
