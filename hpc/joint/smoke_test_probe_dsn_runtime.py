#!/usr/bin/env python3
"""
smoke_test_probe_dsn_runtime.py -- tests for probe_dsn_runtime.py.

The probe is a diagnostic, so its FAILURE paths matter more than its success
path: a probe that reports PASS when the DSN is missing is worse than no probe
at all. Every negative case below therefore asserts both the exit code and the
specific check id that failed, not just "it did not pass".

The DSN repo is not present in a sandbox, so these tests build a STUB
DSN_MAIN_DIR: a directory holding minimal condition_space.py, backbone.py and
train.py modules with the same call signatures the real ones expose. That
tests the probe's resolution and reporting logic. It does NOT test the real
DSN's behaviour, which only the cluster can do -- which is the whole point of
shipping the probe.

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
    """A stub DSN_MAIN_DIR. `omit` drops files, for the missing-file cases."""
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


def run_probe(dsn_dir_env, extra=(), unset=False):
    """Run the probe in a subprocess. Returns (returncode, combined output)."""
    env = dict(os.environ)
    env.pop("DSN_MAIN_DIR", None)
    if not unset and dsn_dir_env is not None:
        env["DSN_MAIN_DIR"] = dsn_dir_env
    proc = subprocess.run([sys.executable, PROBE] + list(extra),
                          capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout + proc.stderr


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

        # ---- T2: DSN_MAIN_DIR unset -------------------------------------
        print("T2  DSN_MAIN_DIR unset")
        rc, out = run_probe(None, unset=True)
        ok &= check("exit 1", rc == 1, out)
        ok &= check("P3 fails", "P3" in failed_ids(out), out)
        ok &= check("says UNSET, not 'missing'", "is UNSET" in out, out)
        ok &= check("VERDICT is FAIL", "VERDICT: FAIL" in out, out)

        # ---- T3: DSN_MAIN_DIR points at a dead path ---------------------
        # This is the distinction that cost a session: a deleted symlink and
        # an unset variable must NOT produce the same message.
        print("T3  DSN_MAIN_DIR points at a nonexistent path")
        rc, out = run_probe(os.path.join(tmp, "does_not_exist"))
        ok &= check("exit 1", rc == 1, out)
        ok &= check("P3 fails", "P3" in failed_ids(out), out)
        ok &= check("says DOES NOT EXIST", "DOES NOT EXIST" in out, out)
        ok &= check("does NOT say UNSET", "is UNSET" not in out, out)

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

        # ---- T8: --dsn-main-dir overrides the environment ---------------
        print("T8  --dsn-main-dir overrides DSN_MAIN_DIR")
        rc, out = run_probe(os.path.join(tmp, "does_not_exist"),
                            extra=["--dsn-main-dir", good])
        ok &= check("P3 passes via the flag",
                    "P3" not in failed_ids(out), out)

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
