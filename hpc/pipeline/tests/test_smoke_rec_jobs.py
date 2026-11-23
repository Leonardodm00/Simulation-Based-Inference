#!/usr/bin/env python3
"""Smoke test of block 38: the jobs and the run sheet of a recorded project's evaluation (specs/SPEC.md block 38, "Test
oracles", checks R38*).

CONTRACT SKELETON (wave C): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 38's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r38a_dry_run          the dry run on the committed synth.json and synth_rec_eval.json; the wrapper
    test_r38b_as_recorded      under the fake PBS, as recorded: samplings, report, validation; done, stale, live
    test_r38c_recorded         under the fake PBS, the recorded kind: no validate unit; an incomplete best fit
    test_r38d_static           ASCII and LF, bash -n, the launcher's imports

Run
    python3 hpc/pipeline/tests/test_smoke_rec_jobs.py [check ...]
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

def test_r38a_dry_run():
    """the dry run on the committed synth.json and synth_rec_eval.json; the wrapper (specs/SPEC.md block 38)."""
    raise NotImplementedError("test_r38a_dry_run: not written yet")


def test_r38b_as_recorded():
    """under the fake PBS, as recorded: samplings, report, validation; done, stale, live (specs/SPEC.md block 38)."""
    raise NotImplementedError("test_r38b_as_recorded: not written yet")


def test_r38c_recorded():
    """under the fake PBS, the recorded kind: no validate unit; an incomplete best fit (specs/SPEC.md block 38)."""
    raise NotImplementedError("test_r38c_recorded: not written yet")


def test_r38d_static():
    """ASCII and LF, bash -n, the launcher's imports (specs/SPEC.md block 38)."""
    raise NotImplementedError("test_r38d_static: not written yet")


CHECKS = [("test_r38a_dry_run", test_r38a_dry_run),
          ("test_r38b_as_recorded", test_r38b_as_recorded),
          ("test_r38c_recorded", test_r38c_recorded),
          ("test_r38d_static", test_r38d_static)]


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
