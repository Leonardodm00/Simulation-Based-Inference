"""Spec-derived static checks (SPEC secs. 1, 4; block 1 'agnostic').

Oracles O-G1, O-G2 of the 2026-10-20 report. Expected values come from the
spec text, not from the code under test.
"""
import ast
import glob
import os
import sys

import pytest

from conftest import PIPE

# sec. 1 / D-091: the code files of the sequencer and the stages
SEQ_FILES = sorted(glob.glob(os.path.join(PIPE, "*.py")) + glob.glob(os.path.join(PIPE, "stages", "*.py")))
CODE_FILES = sorted(
    p for p in glob.glob(os.path.join(PIPE, "**", "*"), recursive=True)
    if p.endswith((".py", ".sh"))
    and os.sep + "projects" + os.sep not in p and os.sep + "tests" + os.sep not in p)
FORBIDDEN = ("giulia", "ldellamea", "davinci-1/home", "cadex", "hhgap")   # block 1 oracle 'agnostic'

# everything a .py, .sh or .pbs under hpc/pipeline/ (pure ASCII, LF: sec. 4, hpc-python-compat)
ALL_SHIPPED = sorted(p for p in glob.glob(os.path.join(PIPE, "**", "*"), recursive=True)
                     if os.path.isfile(p) and p.endswith((".py", ".sh", ".pbs", ".json", ".md")))


def test_sequencer_files_found():
    assert len(SEQ_FILES) >= 10
    assert any(p.endswith("run_pipeline.sh") for p in CODE_FILES)


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: os.path.relpath(p, PIPE))
def test_agnostic_no_project_value(path):
    text = open(path, "rb").read().decode("ascii", "replace").lower()
    hits = [s for s in FORBIDDEN if s in text]
    assert not hits, "%s names %s" % (path, hits)


@pytest.mark.parametrize("path", ALL_SHIPPED, ids=lambda p: os.path.relpath(p, PIPE))
def test_ascii_lf(path):
    blob = open(path, "rb").read()
    assert all(b < 128 for b in blob), "non-ASCII byte in %s" % path
    assert b"\r" not in blob, "CR in %s" % path


@pytest.mark.parametrize("path", SEQ_FILES, ids=lambda p: os.path.relpath(p, PIPE))
def test_python36_syntax(path):
    """sec. 4: no f-strings, no walrus, no postponed annotations; parses
    under the 3.6 grammar (ast feature_version)."""
    src = open(path).read()
    tree = ast.parse(src, feature_version=(3, 6))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.JoinedStr), "f-string in %s line %d" % (path, node.lineno)
        assert not isinstance(node, ast.NamedExpr), "walrus in %s line %d" % (path, node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            assert "annotations" not in [a.name for a in node.names], path


@pytest.mark.parametrize("path", SEQ_FILES, ids=lambda p: os.path.relpath(p, PIPE))
def test_stdlib_only(path):
    """sec. 4: the stdlib only in the sequencer's process."""
    local = {"config", "state", "pbs", "envs", "pipeline", "stages"}
    std = set(sys.stdlib_module_names)
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module.split(".")[0]]
        for n in names:
            assert n in std or n in local, "%s imports %s" % (path, n)


def test_no_nohup_in_code():
    """D-098: no `nohup` in the pipeline code (block 1b invariant)."""
    for p in CODE_FILES:
        txt = open(p).read()
        code = "\n".join(l for l in txt.split("\n") if not l.lstrip().startswith("#"))
        assert "nohup " not in code, p
