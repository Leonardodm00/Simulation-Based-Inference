"""Shared helpers for the spec-derived tests of the joint-docs number scripts.

Written by the integration tester from specs/SPEC.md (Stage 5 spec). Expected
values are computed here from the spec, the governing documents, the FREEZE
commit's files (read with `git show`) and library implementations -- never by
calling the script under test, except to obtain the output being checked.

The scripts are run as subprocesses with torch BLOCKED (a stub `torch` module
that raises ImportError placed first on PYTHONPATH), because SPEC section 4
says they are torch-free by design and must run without it.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(REPO, "hpc", "joint", "docs", "tools")
DOCS = os.path.join(REPO, "hpc", "joint", "docs")
FREEZE = "834eb41"


def git_show(path: str, commit: str = FREEZE, repo: str = REPO) -> str:
    out = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (commit, path)],
                         check=True, stdout=subprocess.PIPE)
    return out.stdout.decode("utf-8")


def load_functions(src: str, names, ns=None) -> dict:
    """Execute the named top-level functions / classes of `src` alone (AST)."""
    import numpy as np
    tree = ast.parse(src)
    ns = dict(ns or {})
    ns.setdefault("np", np)
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(compile(ast.get_source_segment(src, node), "<freeze>", "exec"), ns)
            found.add(node.name)
    missing = set(names) - found
    assert not missing, "not found at FREEZE: %r" % missing
    return ns


def _notorch_dir(tmp_root: str) -> str:
    d = os.path.join(tmp_root, "notorch")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "torch.py"), "w") as fh:
        fh.write('raise ImportError("torch blocked by the spec tests")\n')
    return d


def run_script(name: str, cwd: str = TOOLS, tmp_root: str = None, extra_env=None,
               timeout: int = 900) -> subprocess.CompletedProcess:
    """Run `python <name>` from the tools directory with torch blocked."""
    import tempfile
    tmp_root = tmp_root or tempfile.mkdtemp(prefix="spec_notorch_")
    env = dict(os.environ)
    env["PYTHONPATH"] = _notorch_dir(tmp_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra_env or {})
    return subprocess.run([sys.executable, name], cwd=cwd, env=env, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


_CACHE: dict = {}


def script_output(name: str) -> str:
    """stdout of one torch-blocked run of the script, cached per session."""
    if name not in _CACHE:
        res = run_script(name)
        assert res.returncode == 0, res.stderr[-2000:]
        _CACHE[name] = res.stdout
    return _CACHE[name]


def floats(line: str):
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", line)]


def line_with(text: str, needle: str) -> str:
    hits = [ln for ln in text.splitlines() if needle in ln]
    assert hits, "no output line contains %r" % needle
    return hits[0]


def doc_text(name: str) -> str:
    with open(os.path.join(DOCS, name), encoding="ascii") as fh:
        return fh.read()


def scratch_clone(tmp_path) -> str:
    """A full-history local clone of the repository at the commit under test,
    for tests that must modify the working tree (never the real checkout)."""
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], check=True,
                          stdout=subprocess.PIPE, text=True).stdout.strip()
    dst = str(tmp_path / "clone")
    subprocess.run(["git", "clone", "-q", REPO, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", head], check=True)
    return dst


@pytest.fixture
def clone(tmp_path):
    d = scratch_clone(tmp_path)
    yield d
    shutil.rmtree(d, ignore_errors=True)
