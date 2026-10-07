"""Block 6 (check_notation.py, smoke_test_notation.py): spec-derived tests.

Oracles (SPEC Block 6): canonical form (whitespace removed, subscript before
superscript, accent braces removed); declared symbols with substituted
index-like scripts (`\\chi^2_{26}` for a declared `\\chi^2_d`); operators,
relations, numbers and type tokens accepted with their scripts still checked;
fenced code and inline code not scanned; an undeclared symbol exits 1 and is
named with its line of first use; E1-E4 and P0-P7 exit 0; the smoke test
passes (43 PASS).
"""
import glob
import os
import subprocess
import sys

import pytest

from conftest import DOCS, TOOLS

CHK = os.path.join(TOOLS, "check_notation.py")

MASTER = """# E0 (test master)

## 1. Notation and symbols

| Symbol | Name | Type | Units | First used in |
|---|---|---|---|---|
| $x$ | a trace | vector | Hz | E1 |
| $x^2_i$ | squared entry | scalar | Hz^2 | E1 |
| $\\hat \\theta$ | estimate | vector | -- | E1 |
| $\\chi^2_d$ | chi-square with d dof | law | -- | E1 |
| $d$, $j$ | dimension, index | int | -- | E1 |
| $B_{\\rm sim}, B_{\\rm met}$ | batch sizes | int | rows | E1 |

## 2. Next section

| $z$ | NOT in the master section | -- | -- | -- |
"""


def _run(tmp_path, body, name="doc.md"):
    m = tmp_path / "E0.md"
    m.write_text(MASTER)
    d = tmp_path / name
    d.write_text(body)
    return subprocess.run([sys.executable, CHK, "--master", str(m), "--docs", str(d)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.mark.parametrize("body", [
    "The entry $x_i^2$ is used.\n",              # subscript/superscript order
    "An estimate $\\hat{\\theta}$.\n",           # accent braces removed
    "A law $\\chi^2_{26}$ and $\\chi^2_{d}$.\n",  # index-like substitution
    "Sizes $B_{\\rm sim} \\le B_{\\rm met}$.\n",  # two symbols from one cell
    "Sum $\\sum_{j=1}^{d} x_j \\le 10^{-3}$, $x \\in \\mathbb{R}^d$.\n",
    "Code `$q$` inline and\n\n```\n$q + w$\n```\n\nnot scanned; $x$ is.\n",
])
def test_accepted(tmp_path, body):
    res = _run(tmp_path, body)
    assert res.returncode == 0, res.stdout
    assert res.stdout.startswith("OK   notation:")


@pytest.mark.parametrize("body,sym,line", [
    ("line one\n$x + q$ uses q\n", "q", 2),
    ("$x$\n\n$\\sum_{k=1}^{d} x$ has an undeclared index k\n", "k", 3),
    ("$x$ ok\n$z$ is declared only outside section 1\n", "z", 2),
    ("$\\hat w$ is not declared\n", "\\hat{w}", 1),
])
def test_residual_named_with_first_line(tmp_path, body, sym, line):
    res = _run(tmp_path, body)
    assert res.returncode == 1, res.stdout
    assert res.stdout.startswith("FAIL notation:")
    rows = [l.split() for l in res.stdout.splitlines()[1:]]
    # the residual is named as its whole token (e.g. `\sum_{k=1}^{d}` for an undeclared k)
    hits = [r for r in rows if sym in r[0] or sym.replace("{", "").replace("}", "") in r[0]]
    assert hits, res.stdout
    assert hits[0][1] == "line" and int(hits[0][2]) == line


def test_whitespace_between_scripts_removed(tmp_path):
    """Canonical form removes whitespace: `x ^ 2 _ i` is TeX for the declared `x^2_i`."""
    res = _run(tmp_path, "Here $x ^ 2 _ i$.\n")
    assert res.returncode == 0, res.stdout


def test_inline_span_broken_across_lines_is_scanned(tmp_path):
    """An undeclared symbol must exit 1 wherever it sits in prose math. Markdown
    renders `$x +` / `q$` on two lines as one inline span (P6 lines 317, 319 and
    1067 have such spans); the checker must scan it."""
    res = _run(tmp_path, "The sum $x +\nq$ is split by the line wrap.\n")
    assert res.returncode == 1, res.stdout


def test_any_failing_document_makes_exit_1(tmp_path):
    m = tmp_path / "E0.md"
    m.write_text(MASTER)
    good, bad = tmp_path / "good.md", tmp_path / "bad.md"
    good.write_text("$x$\n")
    bad.write_text("$q$\n")
    res = subprocess.run([sys.executable, CHK, "--master", str(m), "--docs", str(good), str(bad)],
                         stdout=subprocess.PIPE, text=True)
    assert res.returncode == 1 and "OK   notation" in res.stdout and "FAIL notation" in res.stdout


def test_real_documents_pass():
    docs = sorted(glob.glob(os.path.join(DOCS, "E[1-4]_*.md")) + glob.glob(os.path.join(DOCS, "P[0-7]_*.md")))
    assert len(docs) == 12
    res = subprocess.run([sys.executable, CHK, "--master",
                          os.path.join(DOCS, "E0_READERS_GUIDE_NOTATION.md"), "--docs"] + docs,
                         stdout=subprocess.PIPE, text=True)
    assert res.returncode == 0, res.stdout
    assert res.stdout.count("OK   notation") == 12


def test_master_self_list():
    res = subprocess.run([sys.executable, CHK, "--master",
                          os.path.join(DOCS, "E0_READERS_GUIDE_NOTATION.md"), "--self", "--list"],
                         stdout=subprocess.PIPE, text=True)
    assert res.returncode == 0, res.stdout[-2000:]
    assert "declared symbols" in res.stdout


def test_smoke_test_passes():
    res = subprocess.run([sys.executable, "smoke_test_notation.py", "--docs-dir", ".."], cwd=TOOLS,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert "43/43 passed" in res.stdout


def test_ascii_lf():
    for f in ("check_notation.py", "smoke_test_notation.py", "e1_numbers.py", "e2_numbers.py",
              "e3_numbers.py", "e4_numbers.py", "p5_numbers.py"):
        raw = open(os.path.join(TOOLS, f), "rb").read()
        raw.decode("ascii")
        assert b"\r" not in raw, f
