#!/usr/bin/env python3
"""
smoke_test_probe_dsn_runtime.py -- tests for probe_dsn_runtime.py.

The probe is a diagnostic, so its FAILURE paths matter more than its success
path: a probe that reports PASS when the DSN is missing is worse than no probe
at all. Every negative case below therefore asserts both the exit code and the
specific check id that failed, not just "it did not pass".

Most cases build a STUB DSN tree -- a directory holding minimal
condition_space.py, backbone.py and train.py with the same call signatures the
real ones expose -- and hand it to the probe through --dsn-main-dir. That
tests the probe's resolution and reporting logic. Since migration step 2 the
real DSN is the in-repo hpc/dsn, so two cases (T2, T3) run the probe with NO
flag and assert that it resolves that tree and that a stale DSN_MAIN_DIR is
reported as ignored rather than followed. Whether the real DSN's modules then
import (P4-P7) depends on torch being present, so those cases assert on P3
only.

Run:
    python3 smoke_test_probe_dsn_runtime.py

Pure ASCII, LF only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(_HERE, "probe_dsn_runtime.py")

STUB_CONDITION_SPACE = '''\
"""Stub condition_space: legality projection with one known illegal pair."""


def project_condition(mining_strategy, loss_type, strict_semihard):
    # The real one forbids certain (mining, loss) pairs. This stub forbids
    # exactly one, so the test can exercise both the "already legal" and the
    # "was projected" branches of the probe's reporting.
    if mining_strategy == "hard_neg" and loss_type == "joint_sep":
        return ("easy_pos_semihard_neg", loss_type, bool(strict_semihard))
    return (mining_strategy, loss_type, bool(strict_semihard))
'''

STUB_BACKBONE = '''\
"""Stub backbone: same construction surface as the real DSN backbone."""

import torch
import torch.nn as nn


class BackboneConfig(object):
    def __init__(self, depth_exponent=3, width_multiplier=2.0, block_family=0,
                 head_fusion=False, dropout=0.0, stem_width=16,
                 embedding_size=12):
        self.depth_exponent = int(depth_exponent)
        self.width_multiplier = float(width_multiplier)
        self.block_family = int(block_family)
        self.head_fusion = bool(head_fusion)
        self.dropout = float(dropout)
        self.stem_width = int(stem_width)
        self.embedding_size = int(embedding_size)


class _Stub(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c1 = nn.Conv1d(1, cfg.stem_width, 5, stride=4, padding=2)
        self.fc = nn.Linear(cfg.stem_width, cfg.embedding_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = torch.relu(self.c1(x)).mean(-1)
        return torch.nn.functional.normalize(self.fc(h), dim=-1)


def build_backbone(cfg):
    return _Stub(cfg)
'''

STUB_TRAIN = '''\
"""Stub train.py: only build_loss_and_miner, which is all the adapter calls."""


class _Loss(object):
    def __call__(self, z, c, mined):
        return 0.0

    def stats(self):
        return {}


class _Miner(object):
    def __call__(self, z, c):
        return ()


def build_loss_and_miner(cfg, n_classes=None, total_steps=None):
    return _Loss(), _Miner()
'''


def make_stub_dsn(root, omit=()):
    """A stub DSN tree. `omit` drops files, for the missing-file cases."""
    os.makedirs(root, exist_ok=True)
    files = {"condition_space.py": STUB_CONDITION_SPACE,
             "backbone.py": STUB_BACKBONE,
             "train.py": STUB_TRAIN}
    for name, body in files.items():
        if name in omit:
            continue
        with open(os.path.join(root, name), "w", encoding="ascii") as fh:
            fh.write(body)
    return root


def run_probe(dsn_dir_flag, extra=(), env_dsn=None):
    """Run the probe in a subprocess. Returns (returncode, combined output).

    dsn_dir_flag -> passed as --dsn-main-dir (None: no flag, the probe must
    resolve the in-repo hpc/dsn). env_dsn -> DSN_MAIN_DIR in the child's
    environment (None: unset), to prove it is ignored.
    """
    env = dict(os.environ)
    env.pop("DSN_MAIN_DIR", None)
    if env_dsn is not None:
        env["DSN_MAIN_DIR"] = env_dsn
    argv = [sys.executable, PROBE]
    if dsn_dir_flag is not None:
        argv += ["--dsn-main-dir", dsn_dir_flag]
    proc = subprocess.run(argv + list(extra),
                          capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout + proc.stderr


def in_repo_dsn():
    """What the probe must resolve when given no flag: <hpc>/dsn."""
    return os.path.realpath(os.path.join(_HERE, "..", "..", "dsn"))


def check(name, cond, out=""):
    if cond:
        print("  PASS  %s" % name)
        return True
    print("  FAIL  %s" % name)
    if out:
        print("        ---- probe output ----")
        for line in out.strip().split("\n"):
            print("        %s" % line)
    return False


def failed_ids(out):
    """The check ids the probe reported as FAIL."""
    ids = []
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "[probe]" and parts[2] == "FAIL":
            ids.append(parts[1])
    return ids


def main():
    if not os.path.isfile(PROBE):
        print("cannot find %s" % PROBE)
        return 1

    have_torch = True
    try:
        import torch  # noqa: F401
    except Exception:
        have_torch = False

    have_pml = True
    try:
        import pytorch_metric_learning  # noqa: F401
    except Exception:
        have_pml = False

    tmp = tempfile.mkdtemp(prefix="probe_smoke_")
    ok = True
    try:
        # ---- T1: everything present -------------------------------------
        # P1 (torch) and P2 (pytorch_metric_learning) depend on the sandbox,
        # so assert on the DSN-specific checks and on their absence from the
        # failure list rather than on the overall verdict.
        print("T1  all stubs present")
        good = make_stub_dsn(os.path.join(tmp, "dsn_good"))
        rc, out = run_probe(good)
        bad = failed_ids(out)
        ok &= check("P3 (dir resolves) did not fail", "P3" not in bad, out)
        ok &= check("P4 (files present) did not fail", "P4" not in bad, out)
        ok &= check("P5 (condition_space) did not fail", "P5" not in bad, out)
        ok &= check("project_condition result is reported",
                    "project_condition(" in out, out)
        if have_torch:
            ok &= check("P6 (backbone) did not fail", "P6" not in bad, out)
            ok &= check("forward shape (2, 12) is reported",
                        "(2, 12)" in out, out)
            ok &= check("P7 (dsn loss) did not fail", "P7" not in bad, out)
            ok &= check("[dsn-loss] line is emitted",
                        "[dsn-loss] " in out, out)
        else:
            print("  SKIP  P6/P7: no torch in this environment")
        if have_torch and have_pml:
            ok &= check("exit 0 and VERDICT: PASS",
                        rc == 0 and "VERDICT: PASS (7/7 checks)" in out, out)
        else:
            print("  SKIP  overall verdict: torch/pml not both present")

        # ---- T2: no flag, no variable -> the in-repo hpc/dsn ------------
        # Migration step 2: an unset DSN_MAIN_DIR is not a failure state any
        # more, because the tree is a property of the checkout.
        print("T2  no flag and no DSN_MAIN_DIR -> resolves the in-repo hpc/dsn")
        rc, out = run_probe(None)
        ok &= check("P3 passes", "P3" not in failed_ids(out), out)
        ok &= check("P3 names the in-repo tree", in_repo_dsn() in out, out)
        ok &= check("does not claim UNSET", "is UNSET" not in out, out)
        ok &= check("P4 (files present) passes on the real tree",
                    "P4" not in failed_ids(out), out)

        # ---- T3: a stale DSN_MAIN_DIR is IGNORED, and said so -----------
        # The old fallback would have followed the variable to a dead path
        # and failed P3; the new resolver must ignore it, resolve hpc/dsn,
        # and put the fact that it ignored something in the log.
        print("T3  stale DSN_MAIN_DIR (dead path) is ignored and reported")
        rc, out = run_probe(None, env_dsn=os.path.join(tmp, "does_not_exist"))
        ok &= check("P3 passes despite the variable",
                    "P3" not in failed_ids(out), out)
        ok &= check("says IGNORED", "IGNORED" in out, out)
        ok &= check("still names the in-repo tree", in_repo_dsn() in out, out)

        # ---- T3b: an EXPLICIT dead path is still a failure --------------
        print("T3b --dsn-main-dir pointing at a nonexistent path")
        rc, out = run_probe(os.path.join(tmp, "does_not_exist"))
        ok &= check("exit 1", rc == 1, out)
        ok &= check("P3 fails", "P3" in failed_ids(out), out)
        ok &= check("says DOES NOT EXIST", "DOES NOT EXIST" in out, out)

        # ---- T4: directory present, condition_space.py missing ----------
        print("T4  condition_space.py missing")
        part = make_stub_dsn(os.path.join(tmp, "dsn_no_cs"),
                             omit=("condition_space.py",))
        rc, out = run_probe(part)
        bad = failed_ids(out)
        ok &= check("exit 1", rc == 1, out)
        ok &= check("P4 fails", "P4" in bad, out)
        ok &= check("names the missing file",
                    "condition_space.py" in out, out)
        ok &= check("P5 also fails (import cannot succeed)",
                    "P5" in bad, out)
        ok &= check("P3 still passes (the dir itself is fine)",
                    "P3" not in bad, out)

        # ---- T5: backbone.py missing -> fallback would be silent --------
        print("T5  backbone.py missing")
        nb = make_stub_dsn(os.path.join(tmp, "dsn_no_bb"),
                           omit=("backbone.py",))
        rc, out = run_probe(nb)
        bad = failed_ids(out)
        ok &= check("exit 1", rc == 1, out)
        ok &= check("P6 fails", "P6" in bad, out)
        ok &= check("warns about the silent fallback",
                    "FALL BACK" in out, out)

        # ---- T6: an illegal triple is reported as projected -------------
        print("T6  illegal (mining, loss) triple is projected and reported")
        rc, out = run_probe(good, extra=["--mining-strategy", "hard_neg",
                                         "--loss-type", "joint_sep"])
        ok &= check("P5 does not fail", "P5" not in failed_ids(out), out)
        ok &= check("reports that the input was projected",
                    "PROJECTED, input was illegal" in out, out)

        # ---- T7: a legal triple is reported as unchanged ----------------
        print("T7  legal triple is reported as already legal")
        rc, out = run_probe(good)
        ok &= check("reports input already legal",
                    "input already legal" in out, out)

        # ---- T8: --dsn-main-dir overrides the in-repo default -----------
        print("T8  --dsn-main-dir overrides the in-repo tree")
        rc, out = run_probe(good, env_dsn=os.path.join(tmp, "does_not_exist"))
        ok &= check("P3 passes via the flag",
                    "P3" not in failed_ids(out), out)
        ok &= check("P3 names the stub, not hpc/dsn",
                    os.path.realpath(good) in out
                    and in_repo_dsn() not in out, out)

        # ---- T9: the pass condition is a positive line ------------------
        print("T9  pass condition is a line that must appear")
        ok &= check("VERDICT line is always emitted",
                    "[probe] VERDICT:" in out, out)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("SMOKE TEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
