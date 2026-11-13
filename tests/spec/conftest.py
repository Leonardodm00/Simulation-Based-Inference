"""Shared paths for the spec-derived tests (independent tester; first
written 2026-10-20, extended 2026-11-13 by the t1-A run of the split audit,
D-131).

The pipeline modules import each other as top-level modules (`import pbs`),
so hpc/pipeline goes on sys.path; hpc/joint's stage folders likewise for the
block 4-6 tests. SBIX_CLONE names the read-only Sbi-extractor checkout.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
JOINT = os.path.join(REPO, "hpc", "joint")
GIULIA = os.path.join(PIPE, "projects", "giulia", "giulia.json")
LINDDUP = os.path.join(PIPE, "projects", "linddup", "linddup.json")
SYNTH = os.path.join(PIPE, "projects", "synth", "synth.json")
FAKE_PBS = os.path.join(PIPE, "tests", "fake_pbs")
SBIX = os.environ.get("SBIX_CLONE", "/home/claude/sbix")

for p in (PIPE,):
    if p not in sys.path:
        sys.path.insert(0, p)
