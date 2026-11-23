#!/usr/bin/env python3
"""Smoke test of block 33: the view of a recorded project, its data and the evaluation's configuration (specs/SPEC.md block 33, "Test
oracles", checks R33*).

CONTRACT SKELETON (wave A): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 33's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r33a_config       the configuration: every key rule, sample_params, report_params
    test_r33b_as_recorded  the as-recorded view equals ev_plan's ensembles and comparisons
    test_r33c_recorded     the recorded view: names, bank, missing, refusals
    test_r33d_comparisons  the comparisons of a recorded view, by hand
    test_r33e_draw_rows    draw_rows by hand; k = 1 equals calibration_rows
    test_r33f_recording    recording(): keys, dtypes, order, no theta; axes_of

Run
    python3 hpc/pipeline/tests/test_smoke_rec_plan.py [check ...]
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

def test_r33a_config():
    """the configuration: every key rule, sample_params, report_params (specs/SPEC.md block 33)."""
    raise NotImplementedError("test_r33a_config: not written yet")


def test_r33b_as_recorded():
    """the as-recorded view equals ev_plan's ensembles and comparisons (specs/SPEC.md block 33)."""
    raise NotImplementedError("test_r33b_as_recorded: not written yet")


def test_r33c_recorded():
    """the recorded view: names, bank, missing, refusals (specs/SPEC.md block 33)."""
    raise NotImplementedError("test_r33c_recorded: not written yet")


def test_r33d_comparisons():
    """the comparisons of a recorded view, by hand (specs/SPEC.md block 33)."""
    raise NotImplementedError("test_r33d_comparisons: not written yet")


def test_r33e_draw_rows():
    """draw_rows by hand; k = 1 equals calibration_rows (specs/SPEC.md block 33)."""
    raise NotImplementedError("test_r33e_draw_rows: not written yet")


def test_r33f_recording():
    """recording(): keys, dtypes, order, no theta; axes_of (specs/SPEC.md block 33)."""
    raise NotImplementedError("test_r33f_recording: not written yet")


CHECKS = [("test_r33a_config", test_r33a_config), ("test_r33b_as_recorded", test_r33b_as_recorded), ("test_r33c_recorded", test_r33c_recorded), ("test_r33d_comparisons", test_r33d_comparisons), ("test_r33e_draw_rows", test_r33e_draw_rows), ("test_r33f_recording", test_r33f_recording)]


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
