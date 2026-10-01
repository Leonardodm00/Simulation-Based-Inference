#!/usr/bin/env python3
"""
smoke_test_inventory.py -- the smoke test of inventory_joint_knobs.py.

What it checks, in order (each test prints PASS/FAIL and the count at the end):

  T1  fixture tree: a synthetic mini "hpc/" with every surface present; the
      extractor returns exactly the rows the fixture was built to produce
      (flags, dest, types, defaults, choices, subparser contexts, keyword
      defaults, dataclass fields with a name filter, module and class
      constants, PBS directives, -v defaults, env defaults, arrays,
      positional defaults).
  T2  negative paths: a missing file is reported, not skipped silently; an
      unowned row is reported and `--strict` exits 2; a file with a non-ASCII
      byte raises (the extractor refuses to read what the cluster would refuse
      to compile).
  T3  determinism: two extractions of the fixture are byte-identical (JSON and
      markdown).
  T4  index check: a synthetic 00_INDEX.md whose block equals the rendering
      passes; one whose block differs by one character fails and names the
      line; missing markers fail.
  T5  the real tree, when --hpc-dir points at one: no missing file, no
      unowned row, two extractions byte-identical, every `.py` under
      docs/tools is pure ASCII and LF-only, and the rendered markdown is pure
      ASCII.

Run from anywhere:

    python3 smoke_test_inventory.py                       # T1-T4 only
    python3 smoke_test_inventory.py --hpc-dir <repo>/hpc  # T1-T5

Exit status 0 iff every test passed. Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import inventory_joint_knobs as INV  # noqa: E402

RESULTS = []


def _report(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                        ("  -- " + detail) if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# The fixture: a mini hpc/ tree built to exercise every code path
# ---------------------------------------------------------------------------

FIXTURE_FILES = {
    "joint/stage9/run_thing.py": textwrap.dedent('''\
        """A CLI with a plain parser."""
        import argparse

        LIMIT = 5
        _PRIVATE = (1, 2)
        __all__ = ["main"]
        not_a_const = 3


        class Thing(object):
            N_STATS = 8
            lower = 1


        def build_parser():
            p = argparse.ArgumentParser()
            p.add_argument("--alpha", type=float, default=0.5, help="first\\nline")
            p.add_argument("-b", "--beta", type=int, default=3, choices=(1, 3))
            p.add_argument("--gamma", dest="g_val", default="x")
            p.add_argument("--on", action="store_true")
            p.add_argument("--need", required=True)
            p.add_argument("--lr", type=float, default=0.01)
            p.add_argument("positional")
            return p


        def helper(a, b=2, *, c=None, d=float("nan")):
            return a


        class Cfg(object):
            def __init__(self, lr=1e-3, patience=5, name="n"):
                self.lr = lr
    '''),
    "joint/stage9/tune_thing.py": textwrap.dedent('''\
        """A CLI with subparsers and a shared helper."""
        import argparse


        def build_parser():
            ap = argparse.ArgumentParser()
            sub = ap.add_subparsers(dest="command")

            def common(q, need_shards=False):
                q.add_argument("--campaign", default="S-A1")
                if need_shards:
                    q.add_argument("--sim-shards", required=True)
                return q

            common(sub.add_parser("space"))
            q = common(sub.add_parser("propose"))
            q.add_argument("--n-points", type=int, default=8)
            q = common(sub.add_parser("argv"), need_shards=True)
            q.add_argument("--tag", default="")
            return ap
    '''),
    "dsn/cfg.py": textwrap.dedent('''\
        from dataclasses import dataclass, field
        from typing import Tuple, Dict, Any


        @dataclass
        class SpaceCfg:
            a_range: Tuple[int, int] = (3, 6)
            b_choices: Tuple[int, ...] = (0, 1)
            not_wanted: int = 7
            meta: Dict[str, Any] = field(default_factory=dict)
    '''),
    "joint/stage9/jobs/thing.pbs": textwrap.dedent('''\
        #!/bin/bash
        #PBS -N thing
        #PBS -l select=1:ncpus=4:mem=16gb
        #PBS -l walltime=06:00:00
        #PBS -j oe
        : "${OUT_DIR:=}"
        : "${EPOCHS:=10}"
        ENV_NAME="${ENV_NAME:-sbi_env}"
        ARMS=(A1 A0  A5)
        lower=(x y)
        echo "done"
    '''),
    "joint/stage9/jobs/launch.sh": textwrap.dedent('''\
        #!/bin/bash
        CAMPAIGN="${1:-}"
        PYBIN="${PYBIN:-python}"
    '''),
}

FIXTURE_TARGETS = {
    "cli": ["joint/stage9/run_thing.py", "joint/stage9/tune_thing.py"],
    "signature": {"joint/stage9/run_thing.py": ["helper", "Cfg.__init__",
                                                "does_not_exist"]},
    "dataclass": {"dsn/cfg.py": {"SpaceCfg": "_range|_choices|^meta$",
                                 "Missing": None}},
    "constant": ["joint/stage9/run_thing.py"],
    "job": ["joint/stage9/jobs/thing.pbs", "joint/stage9/jobs/launch.sh"],
}

FIXTURE_RULES = [
    (r"\|--(alpha|beta|gamma|on|need|lr|campaign|sim-shards|n-points|tag)$", "P9"),
    (r"\|(helper|Cfg\.__init__)\.", "P9"),
    (r"\|SpaceCfg\.", "P9"),
    (r"\|(LIMIT|_PRIVATE|Thing\.N_STATS)$", "P9"),
    (r"\|(PBS_-N|PBS_-l|PBS_-j|OUT_DIR|EPOCHS|ENV_NAME|ARMS|CAMPAIGN|PYBIN)$",
     "P9"),
]


def _write_fixture(root):
    for rel, text in FIXTURE_FILES.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)


def _rows_by(rows, surface):
    return [r for r in rows if r["surface"] == surface]


def _find(rows, surface, name, **extra):
    for r in rows:
        if r["surface"] == surface and r["name"] == name \
                and all(r.get(k) == v for k, v in extra.items()):
            return r
    return None


# ---------------------------------------------------------------------------

def test_fixture(root):
    rows, missing = INV.extract_all(root, FIXTURE_TARGETS)
    ok = True

    # --- cli -----------------------------------------------------------------
    cli = _rows_by(rows, "cli")
    ok &= _report("T1.1 cli row count (positional excluded)", len(cli) == 10,
                  "got %d" % len(cli))
    r = _find(rows, "cli", "--alpha")
    ok &= _report("T1.2 --alpha type/default/help first line",
                  r is not None and r["type"] == "float" and r["default"] == "0.5"
                  and r["help"] == "first" and r["dest"] == "alpha")
    r = _find(rows, "cli", "--beta")
    ok &= _report("T1.3 --beta long form, int, choices",
                  r is not None and r["type"] == "int" and r["default"] == "3"
                  and r["choices"] == "(1, 3)")
    r = _find(rows, "cli", "--gamma")
    ok &= _report("T1.4 --gamma dest honoured",
                  r is not None and r["dest"] == "g_val" and r["type"] == "str"
                  and r["default"] == "'x'")
    r = _find(rows, "cli", "--on")
    ok &= _report("T1.5 store_true is a flag defaulting False",
                  r is not None and r["type"] == "flag" and r["default"] == "False")
    r = _find(rows, "cli", "--need")
    ok &= _report("T1.6 required flag", r is not None and r["required"] is True)
    ctx = {(r["name"], r["context"]) for r in cli if "tune_thing" in r["file"]}
    want = {("--campaign", "common"), ("--sim-shards", "common[need_shards]"),
            ("--n-points", "propose"), ("--tag", "argv")}
    ok &= _report("T1.7 subparser contexts", ctx == want, "got %s" % sorted(ctx))

    # --- signatures -----------------------------------------------------------
    sig = _rows_by(rows, "signature")
    r = _find(rows, "signature", "helper.a")
    ok &= _report("T1.8 required positional kept in rows",
                  r is not None and r["required"] is True and r["default"] == "")
    r = _find(rows, "signature", "helper.d")
    ok &= _report("T1.9 keyword-only default rendered",
                  r is not None and r["default"] == "float('nan')"
                  and r["required"] is False)
    r = _find(rows, "signature", "Cfg.__init__.lr")
    ok &= _report("T1.10 __init__ default, self skipped",
                  r is not None and r["default"] == "0.001"
                  and _find(rows, "signature", "Cfg.__init__.self") is None)
    r = _find(rows, "signature", "does_not_exist")
    ok &= _report("T1.11 missing function reported in-row",
                  r is not None and r["default"] == "<MISSING FUNCTION>")

    # --- dataclasses ----------------------------------------------------------
    dc = _rows_by(rows, "dataclass")
    names = sorted(r["name"] for r in dc)
    ok &= _report("T1.12 dataclass filter and missing class",
                  names == ["Missing", "SpaceCfg.a_range", "SpaceCfg.b_choices",
                            "SpaceCfg.meta"], "got %s" % names)
    r = _find(rows, "dataclass", "SpaceCfg.meta")
    ok &= _report("T1.13 field(default_factory) rendered",
                  r is not None and r["default"] == "field(default_factory=dict)"
                  and r["dataclass"] is True)

    # --- constants ------------------------------------------------------------
    cn = sorted(r["name"] for r in _rows_by(rows, "constant"))
    ok &= _report("T1.14 ALL_CAPS module and class constants only",
                  cn == ["LIMIT", "Thing.N_STATS", "_PRIVATE"], "got %s" % cn)

    # --- job scripts ----------------------------------------------------------
    jv = _rows_by(rows, "job_var")
    kinds = sorted((r["name"], r["kind"], r["default"]) for r in jv)
    want = sorted([
        ("PBS_-N", "directive", "thing"),
        ("PBS_-l", "directive", "select=1:ncpus=4:mem=16gb"),
        ("PBS_-l", "directive", "walltime=06:00:00"),
        ("PBS_-j", "directive", "oe"),
        ("OUT_DIR", "-v default", ""),
        ("EPOCHS", "-v default", "10"),
        ("ENV_NAME", "env default", "sbi_env"),
        ("ARMS", "array", "A1 A0 A5"),
        ("CAMPAIGN", "positional $1", ""),
        ("PYBIN", "env default", "python"),
    ])
    ok &= _report("T1.15 job-script rows (lower-case array excluded)",
                  kinds == want, "got %s" % kinds)
    ok &= _report("T1.16 no missing file on the fixture", not missing,
                  "missing %s" % missing)

    # --- owners ---------------------------------------------------------------
    # The two sentinel rows (a missing function, a missing class) stay
    # unowned ON PURPOSE: a target that vanished from the code must surface as
    # an error under --strict, not be absorbed by a rule.
    unowned = INV.assign_owners(rows, FIXTURE_RULES)
    ok &= _report("T1.17 every real fixture row owned; the two sentinels not",
                  sorted(unowned) == ["dataclass|dsn/cfg.py|Missing",
                                      "signature|joint/stage9/run_thing.py|does_not_exist"],
                  "unowned %s" % unowned)
    # --- same knob, different defaults --------------------------------------
    var = dict(INV.knob_variants(rows))
    # `b` is a deliberate name collision (helper.b = 2 against the dataclass
    # field b_choices = (0, 1)): the grouping is by normalised NAME alone, so
    # the table is a prompt for a reader, not a verdict, and this asserts
    # that the collision is shown rather than hidden.
    ok &= _report("T1.18 knob_variants finds lr (0.001 vs 0.01) and the b collision",
                  sorted(var) == ["b", "lr"] and
                  sorted(d for _, _, d in var.get("lr", [])) == ["0.001", "0.01"],
                  "got %s" % sorted(var))
    return ok, rows, missing


def test_negative(root):
    ok = True
    # missing file
    rows, missing = INV.extract_all(root, {"cli": ["joint/stage9/nope.py"]})
    ok &= _report("T2.1 missing file reported", missing == ["joint/stage9/nope.py"]
                  and rows == [])
    # unowned row + --strict exit code through the CLI
    rc = subprocess.call([sys.executable,
                          os.path.join(_HERE, "inventory_joint_knobs.py"),
                          "--hpc-dir", root, "--strict"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok &= _report("T2.2 --strict exits 2 on a tree with missing targets", rc == 2,
                  "rc=%d" % rc)
    # non-ASCII byte refused
    bad = os.path.join(root, "joint/stage9/bad.py")
    with open(bad, "wb") as fh:
        fh.write(b"X = 1  # \xe2\x80\x94 em dash\n")
    raised = False
    try:
        INV.extract_constants(bad, "joint/stage9/bad.py")
    except UnicodeDecodeError:
        raised = True
    ok &= _report("T2.3 non-ASCII source refused", raised)
    os.remove(bad)
    return ok


def test_determinism(root):
    r1, m1 = INV.extract_all(root, FIXTURE_TARGETS)
    INV.assign_owners(r1, FIXTURE_RULES)
    r2, m2 = INV.extract_all(root, FIXTURE_TARGETS)
    INV.assign_owners(r2, FIXTURE_RULES)
    j_same = INV.render_json(r1, m1, "c") == INV.render_json(r2, m2, "c")
    md_same = INV.render_markdown(r1, m1, "c") == INV.render_markdown(r2, m2, "c")
    return _report("T3.1 two extractions byte-identical", j_same and md_same)


def test_index_check(root):
    rows, missing = INV.extract_all(root, FIXTURE_TARGETS)
    INV.assign_owners(rows, FIXTURE_RULES)
    md = INV.render_markdown(rows, missing, "c")
    index = "# title\n\nprose\n\n%s\n%s%s\n\ntail\n" % (INV.INDEX_BEGIN, md,
                                                       INV.INDEX_END)
    ok1, _ = INV.check_index(index, md)
    broken = index.replace("| `--alpha` |", "| `--alpha2` |", 1)
    ok2, msg2 = INV.check_index(broken, md)
    ok3, _ = INV.check_index("no markers here", md)
    stamped = INV.render_markdown(rows, missing, "deadbeef")
    ok4, _ = INV.check_index(index, stamped)
    return (_report("T4.1 matching block passes", ok1)
            & _report("T4.4 a different commit stamp still passes", ok4)
            & _report("T4.2 one-character change fails and names the line",
                      (not ok2) and "first difference" in msg2, msg2.split("\n")[0])
            & _report("T4.3 missing markers fail", not ok3))


def _ascii_lf(path):
    data = open(path, "rb").read()
    return all(b < 128 for b in data) and b"\r" not in data


def test_real_tree(hpc_dir):
    ok = True
    rows, missing = INV.extract_all(hpc_dir)
    unowned = INV.assign_owners(rows)
    ok &= _report("T5.1 real tree: no missing target file", not missing,
                  "missing %s" % missing)
    ok &= _report("T5.2 real tree: every row owned", not unowned,
                  "unowned %s" % unowned[:5])
    rows2, missing2 = INV.extract_all(hpc_dir)
    INV.assign_owners(rows2)
    ok &= _report("T5.3 real tree: two extractions byte-identical",
                  INV.render_json(rows, missing, "c") == INV.render_json(rows2, missing2, "c")
                  and INV.render_markdown(rows, missing, "c") == INV.render_markdown(rows2, missing2, "c"))
    md = INV.render_markdown(rows, missing, "c")
    ok &= _report("T5.4 rendered markdown is pure ASCII",
                  all(ord(ch) < 128 for ch in md))
    tools = [os.path.join(_HERE, f) for f in os.listdir(_HERE) if f.endswith(".py")]
    bad = [p for p in tools if not _ascii_lf(p)]
    ok &= _report("T5.5 every tools/*.py is pure ASCII and LF-only", not bad,
                  "bad %s" % bad)
    counts = {s: sum(1 for r in rows if r["surface"] == s)
              for s in ("cli", "signature", "dataclass", "constant", "job_var")}
    ok &= _report("T5.6 every surface non-empty on the real tree",
                  all(v > 0 for v in counts.values()), "%s" % counts)
    return ok


def test_p0_tables(hpc_dir):
    """T6: the generated tables of P0 (p0_tables.py), real tree only."""
    import p0_tables as P
    ok = True
    inv_path = os.path.join(_HERE, "inventory.json")
    if not os.path.isfile(inv_path):
        return _report("T6 p0 tables", False, "inventory.json missing; run the extractor first")
    blocks = P.render(hpc_dir, inv_path)
    a_rows = [l for l in blocks["A"].split("\n") if l.startswith("| ") and l[2].isdigit()]
    ok &= _report("T6.1 table A has one row per axis of JOINT_KNOB_ORDER",
                  len(a_rows) == 23, "got %d" % len(a_rows))
    JS, NTJ = P._load_space(hpc_dir)
    pri = P.priors_from_source(JS)
    cats = {k for k, v in pri.items() if v == "categorical"}
    ok &= _report("T6.2 priors parsed for every axis; categorical set == _NO_BOUNDARY",
                  set(pri) == set(JS.JOINT_KNOB_ORDER) and cats == set(JS._NO_BOUNDARY),
                  "categorical %s" % sorted(cats))
    reach = [l for l in a_rows if "NOT PASSED" in l]
    ok &= _report("T6.3 every searched axis reaches the runner (no NOT PASSED row)",
                  not reach, "%d rows" % len(reach))
    doc = "x\n" + "".join(P.MARK[k][0] + "\n" + blocks[k] + P.MARK[k][1] + "\n" for k in ("A", "F", "K")) + "y\n"
    res = P.check_doc(doc, blocks)
    ok &= _report("T6.4 check_doc passes on a doc built from the blocks",
                  all(r[1] for r in res), "%s" % [(r[0], r[1]) for r in res])
    broken = doc.replace("| `lr` |", "| `lr2` |", 1)
    res2 = P.check_doc(broken, blocks)
    ok &= _report("T6.5 check_doc fails when one cell changes",
                  not all(r[1] for r in res2))
    with open(inv_path, "r", encoding="ascii") as fh:
        inv = json.load(fh)
    owned = [r for r in inv["rows"] if r["owner"] not in ("plumbing", "UNOWNED")
             and not (r["surface"] == "signature" and r.get("required"))]
    n_knobs = len({INV.knob_name(r) for r in owned})
    k_rows = [l for l in blocks["K"].split("\n") if l.startswith("| `")]
    ok &= _report("T6.6 table K lists every owned knob exactly once",
                  len(k_rows) == n_knobs, "rows %d, knobs %d" % (len(k_rows), n_knobs))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpc-dir", default=None,
                    help="the repository's hpc/ directory, for T5")
    args = ap.parse_args(argv)

    root = tempfile.mkdtemp(prefix="inv_fixture_")
    try:
        _write_fixture(root)
        ok = test_fixture(root)[0]
        ok &= test_negative(root)
        ok &= test_determinism(root)
        ok &= test_index_check(root)
        if args.hpc_dir:
            if os.path.isdir(os.path.join(args.hpc_dir, "joint")):
                ok &= test_real_tree(args.hpc_dir)
                ok &= test_p0_tables(args.hpc_dir)
            else:
                ok &= _report("T5 real tree", False,
                              "%s has no joint/ subdirectory" % args.hpc_dir)
        else:
            print("SKIP  T5 real tree (no --hpc-dir given)")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    n_pass = sum(1 for _, v in RESULTS if v)
    print("%d/%d passed" % (n_pass, len(RESULTS)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
