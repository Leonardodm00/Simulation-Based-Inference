#!/usr/bin/env python3
"""Smoke test of block 36: the quantities of a recorded project and the validation tool (specs/SPEC.md block 36, "Test
oracles", checks R36*).

CONTRACT SKELETON (wave B): every check below raises NotImplementedError
and is reported as TODO, so this file cannot pass until block 36's
implementer has written each check from the oracle of the same name in
specs/SPEC.md. A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_r36a_edge           the edge split: (R1) by hand, the toy's two posteriors, (R2)
    test_r36b_recording      a recording bank's quantities and the well's mixture, by definition
    test_r36c_gate           the gate's wiring, the shares, k_gate and the size band, the example well
    test_r36d_as_recorded    D-162's second target: the metrics on D_sim equal block 20's
    test_r36e_validate       ev_rec_validate: identical trees pass, four perturbations fail
    test_r36f_axes           ev_metrics' optional axes: unchanged without, named with
    test_r36g_end_to_end     the recorded-kind project through the sampler and the metrics

Run
    python3 hpc/pipeline/tests/test_smoke_rec_metrics.py [check ...]
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

def test_r36a_edge():
    """the edge split: (R1) by hand, the toy's two posteriors, (R2) (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36a_edge: not written yet")


def test_r36b_recording():
    """a recording bank's quantities and the well's mixture, by definition (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36b_recording: not written yet")


def test_r36c_gate():
    """the gate's wiring, the shares, k_gate and the size band, the example well (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36c_gate: not written yet")


def test_r36d_as_recorded():
    """D-162's second target: the metrics on D_sim equal block 20's (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36d_as_recorded: not written yet")


def test_r36e_validate():
    """ev_rec_validate: identical trees pass, four perturbations fail (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36e_validate: not written yet")


def test_r36f_axes():
    """ev_metrics' optional axes: unchanged without, named with (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36f_axes: not written yet")


def test_r36g_end_to_end():
    """the recorded-kind project through the sampler and the metrics (specs/SPEC.md block 36)."""
    raise NotImplementedError("test_r36g_end_to_end: not written yet")


CHECKS = [("test_r36a_edge", test_r36a_edge),
          ("test_r36b_recording", test_r36b_recording),
          ("test_r36c_gate", test_r36c_gate),
          ("test_r36d_as_recorded", test_r36d_as_recorded),
          ("test_r36e_validate", test_r36e_validate),
          ("test_r36f_axes", test_r36f_axes),
          ("test_r36g_end_to_end", test_r36g_end_to_end)]


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
