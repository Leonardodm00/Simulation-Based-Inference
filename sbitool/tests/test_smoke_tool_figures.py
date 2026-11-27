#!/usr/bin/env python3
"""Smoke test of block 46: the device and the window as figures (specs/SPEC.md block 46, "Test
oracles" and "Contract of wave C", checks T46*).

A check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t46a_device         one marker per electrode at its position, the subregions' colours
    test_t46b_window         the raster holds the span's spikes; the lower panel's data are x
    test_t46c_command        PNG files with the PNG signature; no statistic in the module

Run
    SBIX_CLONE=<the Sbi-extractor's checkout> PYTHONDONTWRITEBYTECODE=1 \
        python3 sbitool/tests/test_smoke_tool_figures.py [check ...]
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


def test_t46a_device():
    """T46a: on a 6 x 6 recording with two subregions, one marker per electrode at its position
    and the subregions' electrodes in their colours, read back from the artists; hollow
    markers for the electrodes without a spike; the highlight's rings."""
    raise NotImplementedError("TODO test_t46a_device")


def test_t46b_window():
    """T46b: the raster holds exactly the spikes of the subregion's electrodes inside the
    window's span; the lower panel's y data are the window's x and its x data the bins' starts."""
    raise NotImplementedError("TODO test_t46b_window")


def test_t46c_command():
    """T46c: the command writes PNG files with the PNG signature; device.py imports no statistics
    library and calls no mean, std, var, median, quantile, percentile, average, histogram or
    bincount (static)."""
    raise NotImplementedError("TODO test_t46c_command")


CHECKS = [("test_t46a_device", test_t46a_device),
          ("test_t46b_window", test_t46b_window),
          ("test_t46c_command", test_t46c_command)]


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
