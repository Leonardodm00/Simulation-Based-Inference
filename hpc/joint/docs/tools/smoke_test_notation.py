#!/usr/bin/env python3
"""Smoke test for check_notation.py: tokenizer, matcher, document scan, CLI.

Run:
    cd hpc/joint/docs/tools
    python3 smoke_test_notation.py                 # fixture tests only
    python3 smoke_test_notation.py --docs-dir ..   # + the real E0 and P0

Every test prints PASS or FAIL with the observed value; the exit code is 1
if any failed. No network, no torch, standard library only.
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import check_notation as N  # noqa: E402

RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(info)) if info else ""))


MASTER = r"""# fixture master

## 1. Notation and symbols

| Symbol | Name / Meaning | Type & domain | Units | First used in S |
|---|---|---|---|---|
| $\theta$ | parameters | $\theta \in \Theta$ | mixed | E1 |
| $\Theta$ | prior box | set | -- | E1 |
| $d_\theta$ | dimension | $\mathbb{N}$ | -- | E1 |
| $x$ | window | $\mathbb{R}^{W}_{\ge 0}$ | Hz | E1 |
| $W$ | samples | $\mathbb{N}$ | samples | E1 |
| $z$ | embedding | $S^{E-1}$ | -- | E1 |
| $S^{E-1}$ | unit sphere | set | -- | E1 |
| $E$ | embedding dimension | $\mathbb{N}$ | -- | E1 |
| $g, g'$ | two wells | indices | -- | E5 |
| $k$ | axis index | index | -- | E1 |
| $j$ | direction index | index | -- | E5 |
| $m$ | nuisance direction index | index | -- | E7 |
| $m_g$ | posterior mean (analytic level) | $\mathbb{R}^{d_\theta}$ | param | E5 |
| $\hat m_g$ | its Monte Carlo estimate (computed level) | $\mathbb{R}^{d_\theta}$ | param | E5 |
| $T_{gg'}$ | replicate statistic | $\mathbb{R}_{\ge 0}$ | -- | E5 |
| $\Delta_{gg'}$ | disagreement | $\mathbb{R}^{d_\theta}$ | param | E5 |
| $M$ | the metric | SPD | -- | E5 |
| $\chi^2_d$ | chi-square with $d$ degrees of freedom | law | -- | E5 |
| $d$ | degrees of freedom | $\mathbb{N}$ | -- | E5 |
| $\Sigma_0$ | prior covariance | PD | param$^2$ | E5 |
| $\lambda_j$ | precision ratio | $\mathbb{R}_{\ge 0}$ | -- | E5 |
| $\lambda_{\rm dsn}, \lambda_{\rm rep}$ | term weights | $\mathbb{R}_{\ge 0}$ | -- | E4 |
| $q_\omega$ | the flow, $q_\omega(\theta \mid z)$ | density | -- | E2 |
| $\Delta t$ | bin width | s | s | E1 |
| $\delta\theta_m$ | aliasing shift | $\mathbb{R}^{d_\theta}$ | param | E7 |
| $\mathcal{L}^{\rm sim}_{\rm NPE}$ | NPE loss | $\mathbb{R}$ | nats | E4 |
| $a_k, b_k$ | box bounds on axis $k$ | reals | mixed | E1 |
| $\hat\Delta$ | information gain | $\mathbb{R}$ | nats | E7 |

### 1.1 Conventions

Prose here uses $\theta$ and may not declare anything.

## 2. Glossary

| $zzz$ | a row outside the master table does not declare |
"""

GOOD_DOC = r"""
Inline code `$notmath$` and a fenced block are ignored:

```
$ignored \omega$
```

Here $x_g$, $x_{g'}$, $T_{gg'}^2$, $\chi^2_{26}$, $\Sigma_0^{-1}$, $\hat m_{g'}$,
$\sum_{j=1}^{d_\theta}\lambda_j$, $\mathbb{E}_{q_\omega}[\theta\mid x_g]$,
$q_\omega(\theta \mid z)$, $\mathcal{L}_{\rm NPE}^{\rm sim}$, $\Delta t$,
$\delta\theta_m$, $S^{E-1} \subset \mathbb{R}^{E}$, $\mathrm{diag}((b_k-a_k)^2/12)$,
$10^{-3}$, $4 d_\theta = 104$, $\theta^{(k)}$, $x \in \mathbb{R}^{W}_{\ge 0}$,
$\log_{10} \lambda_{\rm dsn}$, $\hat\Delta = L_0 - L$ is bad only for $L$.
$$ T_{gg'} = \Delta_{gg'}^\top M \Delta_{gg'} \tag{3} $$
"""

BAD_DOC = r"""
Undeclared: $\lambda$, $\hat C_g$, $\theta^{\rm ns}$, $\beta_1$, $\omega$, $L$.
"""


def test_tokenizer():
    cases = [
        (r"\hat m_g", ["\\hatm_g"]),
        (r"\hat{m}_g", ["\\hatm_g"]),
        (r"T_{gg'} = \Delta_{gg'}^\top M \Delta_{gg'}",
         ["T_{gg'}", "\\Delta_{gg'}^\\top", "M", "\\Delta_{gg'}"]),
        (r"\mathcal{L}_{\rm NPE}^{\rm sim}", ["\\mathcal{L}_{\\rmNPE}^{\\rmsim}"]),
        (r"\sum_{j=1}^{d_\theta} \lambda_j/(1+\lambda_j)",
         ["\\sum_{j=1}^{d_\\theta}", "\\lambda_j", "\\lambda_j"]),
        (r"10^{-3} \le x \in \mathbb{R}^{W}_{\ge 0}", ["x", "\\mathbb{R}_{\\ge0}^{W}"]),
        (r"\delta\theta_m, \Delta t, \Delta y_g, \hat\Delta = L_0 - L",
         ["\\delta\\theta_m", "\\Deltat", "\\Deltay_g", "\\hat\\Delta", "L_0", "L"]),
        (r"\chi^2_{26}", ["\\chi_{26}^2"]),
        (r"g, g'", ["g", "g'"]),
        (r"\bar C_{\rm true}^{-1}", ["\\barC_{\\rmtrue}^{-1}"]),
        (r"\theta^{(k)}", ["\\theta^{(k)}"]),
    ]
    for i, (src, want) in enumerate(cases, 1):
        got = N.tokenize(src)
        check("T1.%d tokenize %r" % (i, src[:40]), got == want, got)


def test_master_and_matcher():
    declared = N.load_master(MASTER)
    check("T2.1 master declares the first-cell symbols only",
          "zzz" not in declared and "\\theta" in declared and "g'" in declared, len(declared))
    check("T2.2 a comma cell declares each symbol",
          "\\lambda_{\\rmdsn}" in declared and "\\lambda_{\\rmrep}" in declared and
          "a_k" in declared and "b_k" in declared)
    check("T2.3 canonical forms: accent, script order, compound",
          "\\hatm_g" in declared and "\\mathcal{L}_{\\rmNPE}^{\\rmsim}" in declared and
          "\\Deltat" in declared and "\\delta\\theta_m" in declared)
    res = N.check_document(GOOD_DOC, declared)
    toks = [t for t, _, _ in res]
    check("T2.4 good document: only the deliberately undeclared L remains",
          toks == ["L", "L_0"], toks)
    res = N.check_document(BAD_DOC, declared)
    toks = [t for t, _, _ in res]
    want = ["L", "\\beta_1", "\\hatC_g", "\\lambda", "\\omega", "\\theta^{\\rmns}"]
    check("T2.5 bad document: every undeclared symbol reported once, sorted", toks == want, toks)
    check("T2.6 residual carries the first line and its span",
          res and res[0][1] == 2 and "L" in res[0][2], res[:1])
    try:
        N.load_master("# no heading here\n| $x$ | y |\n")
        check("T2.7 missing master heading raises", False)
    except ValueError:
        check("T2.7 missing master heading raises", True)
    check("T2.8 code and fences are not scanned",
          all(t != "\\omega" for t, _, _ in N.check_document(GOOD_DOC, declared)))
    spans = N.math_spans(GOOD_DOC)
    disp = [(line, sp) for line, sp in spans if "\\tag{3}" in sp]
    check("T2.9 display math is scanned, with the line of its opening $$",
          len(disp) == 1 and disp[0][0] == GOOD_DOC.split("\n").index("$$ T_{gg'} = \\Delta_{gg'}^\\top M \\Delta_{gg'} \\tag{3} $$") + 1,
          disp[:1])
    # P2 forms: a primed index of a declared index, and a relation inside a
    # subscript (the index symbols are the fixture's g, j, k).
    doc = r"$x_{g'}$, $\lambda_{j'}$, $\sum_{j \ne j'} \lambda_j$, $\sum_{k : a_k \ge 1} b_k$, $x_{g''}$"
    toks = [t for t, _, _ in N.check_document(doc, declared)]
    check("T2.10 a primed declared index and a relation inside a subscript pass",
          toks == [], toks)
    doc = r"$x_{q'}$, $\sum_{j \ne q} \lambda_j$"
    toks = [t for t, _, _ in N.check_document(doc, declared)]
    check("T2.11 an undeclared index stays undeclared, primed or in a relation",
          toks == ["\\sum_{j\\neq}", "x_{q'}"], toks)
    # a parenthesised index list before \in (the recursion guard), a primed
    # indexed symbol inside a script, and a comma list of relations
    doc = r"$\sum_{(g, g', k) \in \Theta} a_k$, $\sum_{k : a_k \ge 1, b_{k'} \ge 1} b_k$, $x_{g, g'}$"
    toks = [t for t, _, _ in N.check_document(doc, declared)]
    check("T2.12 a parenthesised list before \\in, a primed indexed symbol in a "
          "script, and a comma list of relations pass", toks == [], toks)


def test_cli(tmp):
    master = os.path.join(tmp, "E0.md")
    good = os.path.join(tmp, "good.md")
    bad = os.path.join(tmp, "bad.md")
    with open(master, "w", encoding="ascii", newline="\n") as fh:
        fh.write(MASTER)
    with open(good, "w", encoding="ascii", newline="\n") as fh:
        fh.write(GOOD_DOC.replace("$\\hat\\Delta = L_0 - L$ is bad only for $L$.", "$\\hat\\Delta$."))
    with open(bad, "w", encoding="ascii", newline="\n") as fh:
        fh.write(BAD_DOC)
    script = os.path.join(_HERE, "check_notation.py")
    r = subprocess.run([sys.executable, script, "--master", master, "--docs", good],
                       capture_output=True, text=True)
    check("T3.1 CLI passes a clean document (rc 0)", r.returncode == 0 and "OK   notation" in r.stdout, r.stdout.strip()[:80])
    r = subprocess.run([sys.executable, script, "--master", master, "--docs", bad],
                       capture_output=True, text=True)
    check("T3.2 CLI fails a document with residuals (rc 1) and names them",
          r.returncode == 1 and "\\beta_1" in r.stdout and "line" in r.stdout, r.stdout.strip()[:80])
    r = subprocess.run([sys.executable, script, "--master", master, "--list"],
                       capture_output=True, text=True)
    check("T3.3 --list prints the declared symbols with a count",
          r.returncode == 0 and "declared symbols" in r.stdout and "\\theta" in r.stdout)
    r = subprocess.run([sys.executable, script, "--master", master, "--self"],
                       capture_output=True, text=True)
    check("T3.4 --self checks the master's own prose", r.returncode == 0 and "OK   notation" in r.stdout, r.stdout.strip()[:80])
    with open(bad, "wb") as fh:
        fh.write(b"non-ascii \xc2\xb5 here\n")
    r = subprocess.run([sys.executable, script, "--master", master, "--docs", bad],
                       capture_output=True, text=True)
    check("T3.5 a non-ASCII document is refused", r.returncode != 0)
    # determinism
    r1 = subprocess.run([sys.executable, script, "--master", master, "--docs", good, "--list"],
                        capture_output=True, text=True).stdout
    r2 = subprocess.run([sys.executable, script, "--master", master, "--docs", good, "--list"],
                        capture_output=True, text=True).stdout
    check("T3.6 output is identical across two runs", r1 == r2)


def test_real(docs_dir):
    master = os.path.join(docs_dir, "E0_READERS_GUIDE_NOTATION.md")
    if not os.path.isfile(master):
        print("SKIP  T4 real documents (no E0 at %s)" % master)
        return
    with open(master, encoding="ascii") as fh:
        text = fh.read()
    declared = N.load_master(text)
    check("T4.1 E0 declares at least 100 symbols", len(declared) >= 100, len(declared))
    res = N.check_document(text, declared)
    check("T4.2 E0's own prose uses only declared symbols", not res, [t for t, _, _ in res][:8])
    # Every written document of the set is checked here; a document that does
    # not exist yet is skipped, so the suite grows with the set.
    docs = ("P0_PARAMETERS_OVERVIEW.md", "P1_ENCODER_AXES.md", "P2_DSN_LOSS_AXES.md",
            "P3_REPLICATE_AXES.md", "P4_FLOW_AXES.md", "P5_OPTIMISER_AND_SCHEDULE.md",
            "P6_SEARCH_DRIVER.md", "P7_UPSTREAM_AND_JOBS.md", "E1_THE_PROBLEM.md",
            "E2_NPE_AND_FLOWS.md", "E3_THE_SUMMARY_NETWORK.md",
            "E4_JOINT_OBJECTIVE_AND_LOOP.md")
    for n, name in enumerate(docs, 3):
        path = os.path.join(docs_dir, name)
        if not os.path.isfile(path):
            print("SKIP  T4.%d %s (absent)" % (n, name))
            continue
        with open(path, encoding="ascii") as fh:
            res = N.check_document(fh.read(), declared)
        check("T4.%d %s uses only declared symbols" % (n, name), not res, [t for t, _, _ in res][:8])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default=None, help="hpc/joint/docs, for the real-document tests")
    args = ap.parse_args(argv)
    test_tokenizer()
    test_master_and_matcher()
    tmp = tempfile.mkdtemp(prefix="notation_")
    test_cli(tmp)
    if args.docs_dir:
        test_real(args.docs_dir)
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("%d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
