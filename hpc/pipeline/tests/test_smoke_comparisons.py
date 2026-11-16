#!/usr/bin/env python3
"""Smoke test for block 27 of specs/SPEC.md: the evaluation's comparisons
across searches (sec. 2.5, "The comparisons"; D-132 (5), D-136; the roadmap
synth, step S6; wave W2).

[contract: wave W2] Skeleton written with the contract. Every check below
raises NotImplementedError until block 27's implementer adapts it, and the
runner counts that as an error, so an unadapted copy cannot pass. A check
that does not apply is not deleted: it raises unittest.SkipTest("<reason>").

Checks (specs/SPEC.md block 27, "Test oracles")
    test_e27a_plan       ev_plan.comparisons on configurations alone: the
                         committed synth.json v4's six comparisons in order;
                         the v3-like fixture's within-search pair and no
                         A2_vs_A5; the refusals (no S-A1 search, two);
                         S-A25 uncompared; different cohorts, no A2_vs_A5
    test_e27b_metrics    a reduced v4-like project (S-A1 on R0, S-A2-only and
                         S-A5-only on R3) through ev_metrics.compute: the nine
                         paired files, their keys per kind, Dbar, sigma_seed
                         and the interval recomputed; METRICS.json's
                         comparisons; a reference not sampled: left out and
                         named; rows altered: refused
    test_e27c_launcher   report_inputs carries the comparisons; a report made
                         without them is stale; the dry run lists them; no
                         S-A1 search: --what report refused, --what
                         calbank,sample not
    test_convergence     not a discretisation: skipped

Run
    python3 hpc/pipeline/tests/test_smoke_comparisons.py [check ...]

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


# ---------------------------------------------------------------- checks ---

def test_e27a_plan():
    raise NotImplementedError("block 27: E27a, ev_plan.comparisons on configurations")


def test_e27b_metrics():
    raise NotImplementedError("block 27: E27b, the comparisons on a reduced v4-like project")


def test_e27c_launcher():
    raise NotImplementedError("block 27: E27c, report_inputs and the launcher")


def test_convergence():
    raise unittest.SkipTest("not a discretisation: no convergence order")


# ------------------------------------------------------------------ main ---

CHECKS = [("test_e27a_plan", test_e27a_plan), ("test_e27b_metrics", test_e27b_metrics),
          ("test_e27c_launcher", test_e27c_launcher), ("test_convergence", test_convergence)]


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
