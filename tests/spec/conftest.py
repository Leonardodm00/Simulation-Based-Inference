"""Shared paths for the spec-derived tests (independent tester, 2026-10-20).

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
SBIX = os.environ.get("SBIX_CLONE", "/home/claude/sbix")

for p in (PIPE,):
    if p not in sys.path:
        sys.path.insert(0, p)
