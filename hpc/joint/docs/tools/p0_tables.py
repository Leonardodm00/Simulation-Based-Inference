#!/usr/bin/env python3
"""
p0_tables.py -- the generated tables of P0_PARAMETERS_OVERVIEW.md.

P0 is written by hand around three tables that must never drift from the
code, so those three are generated here and embedded between markers, and
`--check-doc` compares them with a fresh rendering (the same contract as
inventory_joint_knobs.py --check-index):

  A  the searched axes of JOINT_KNOB_ORDER: block, type, the range resolved
     at the DUP15HD shapes (p, E, d_theta) = (26, 12, 26), the prior the
     optimiser samples with, the campaign that pins the axis and at what,
     the runner flag the axis reaches, the runner's own default, the owner
     document  -- read from joint_space.py and npe_tune_joint.py, imported,
     plus inventory.json for the runner defaults
  F  the same knob under different names and defaults, restricted to rows
     with an owner document (plumbing excluded) -- from inventory.json
  K  the alphabetical index of every non-plumbing knob: where it is set, its
     owner document -- from inventory.json

Needs the repository checkout (joint_space imports the DSN's condition_space
through dsn_locate); needs no torch, skopt or sbi.

Usage
-----
  python3 p0_tables.py --hpc-dir <repo>/hpc --inventory inventory.json --out-md /tmp/p0_tables.md
  python3 p0_tables.py --hpc-dir <repo>/hpc --inventory inventory.json --check-doc ../P0_PARAMETERS_OVERVIEW.md

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import inventory_joint_knobs as INV  # noqa: E402

MARK = {"A": ("<!-- p0:A:begin -->", "<!-- p0:A:end -->"),
        "F": ("<!-- p0:F:begin -->", "<!-- p0:F:end -->"),
        "K": ("<!-- p0:K:begin -->", "<!-- p0:K:end -->")}

# The DUP15HD shapes the ranges are resolved at (plan S2.3; usage v1.3 S7).
SHAPES = {"p": 26, "embedding_dim": 12, "d_theta": 26}


def priors_from_source(JS) -> Dict[str, str]:
    """The prior each free axis is sampled with, READ from the branches of
    joint_space.space_dimensions() rather than copied: the function needs
    skopt to run, so its source is parsed instead. Each `if/elif axis in
    (...)` branch names its axes and the dimension built in the next
    `dims.append(...)` line; the trailing `else` is a uniform Real."""
    import inspect
    src = inspect.getsource(JS.space_dimensions).split("\n")
    branches: List[Tuple[Optional[List[str]], str]] = []
    axes: Optional[List[str]] = None
    pending = False
    for line in src:
        t = line.strip()
        m = re.match(r"(?:el)?if axis (?:in \((.*)\)|== \"(\w+)\"|in (_\w+)):", t)
        if m:
            if m.group(1) is not None:
                axes = [a.strip().strip("\"'") for a in m.group(1).split(",") if a.strip()]
            elif m.group(2) is not None:
                axes = [m.group(2)]
            else:
                axes = sorted(getattr(JS, m.group(3)))
            pending = True
            continue
        if t.startswith("else:"):
            axes = None
            pending = True
            continue
        if pending and "dims.append(" in t:
            if "Categorical" in t:
                kind = "categorical"
            elif "Integer" in t:
                kind = "integer, " + ("log-uniform" if "log-uniform" in t else "uniform")
            else:
                kind = "real, " + ("log-uniform" if "log-uniform" in t else "uniform")
            branches.append((axes, kind))
            pending = False
    out: Dict[str, str] = {}
    for axis in JS.JOINT_KNOB_ORDER:
        for ax, kind in branches:
            if ax is None or axis in ax:
                out[axis] = kind
                break
    return out


def _load_space(hpc_dir: str):
    stage4 = os.path.join(hpc_dir, "joint", "stage4")
    for p in (stage4, os.path.join(hpc_dir, "joint", "stage3"), hpc_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import joint_space as JS  # noqa: E402
    import npe_tune_joint as NTJ  # noqa: E402
    return JS, NTJ


def _fmt_range(axis: str, r: Any) -> str:
    if axis in ("dsn_on", "rep_on"):
        return "{0, 1}"
    vals = list(r)
    if axis in ("block_family", "head_fusion", "batch_size_npe", "loss_type",
                "mining_strategy"):
        return "{" + ", ".join(str(v) for v in vals) + "}"
    return "[%s, %s]" % (vals[0], vals[1])


def table_A(JS, NTJ, inventory: Dict[str, Any]) -> List[str]:
    spec = JS.default_joint_space(p=SHAPES["p"],
                                  embedding_dim=SHAPES["embedding_dim"],
                                  d_theta=SHAPES["d_theta"])
    pins = {c: JS.resolved_pins(c, spec) for c in ("S-A1", "S-A2", "S-A5")}
    runner = {r["name"]: r for r in inventory["rows"]
              if r["surface"] == "cli" and r["file"] == "joint/stage3/run_joint_arms.py"}
    owner_of = {}
    for r in inventory["rows"]:
        if r["surface"] == "dataclass" and r["name"].startswith("JointSpaceSpec."):
            owner_of[r["name"].split(".", 1)[1]] = r["owner"]
    owner_of.setdefault("dsn_on", "P2")
    owner_of.setdefault("rep_on", "P3")

    priors = priors_from_source(JS)
    lines = ["| # | block | axis | type | range at (26, 12, 26) | prior | S-A1 | S-A2 | S-A5 | reaches | runner default | doc |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    n = 0
    for block, axes in JS.BLOCKS.items():
        for axis in axes:
            n += 1
            r = spec.range_of(axis)
            if axis in JS._STR_AXES:
                typ = "str"
            elif axis in JS._INT_AXES:
                typ = "int"
            else:
                typ = "float"
            prior = priors.get(axis, "?")
            cells = []
            for c in ("S-A1", "S-A2", "S-A5"):
                cells.append("free" if axis not in pins[c]
                             else "pin " + repr(pins[c][axis]))
            if axis in NTJ.AXIS_TO_FLAG:
                reach = "`%s`" % NTJ.AXIS_TO_FLAG[axis]
            elif axis in NTJ.DERIVED:
                # the two switches and the two log-weights reach the runner as
                # one weight each: 0 when the switch is off, 10**x when on
                reach = ("`--lambda-dsn` (derived)" if "dsn" in axis
                         else "`--lambda-rep` (derived)")
            else:
                reach = "NOT PASSED"
            flag = NTJ.AXIS_TO_FLAG.get(axis)
            if axis in ("dsn_on", "log10_lambda_dsn"):
                flag = "--lambda-dsn"
            if axis in ("rep_on", "log10_lambda_rep"):
                flag = "--lambda-rep"
            rd = runner.get(flag, {}).get("default", "") if flag else ""
            lines.append("| %d | %s | `%s` | %s | `%s` | %s | %s | %s | %s | %s | %s | %s |"
                         % (n, block, axis, typ, _fmt_range(axis, r), prior,
                            cells[0], cells[1], cells[2], reach,
                            ("`%s`" % rd) if rd else "", owner_of.get(axis, "?")))
    fixed = ", ".join("`%s=%r`" % (k, v) for k, v in sorted(spec.fixed.items()))
    lines.append("")
    lines.append("Fixed by plan S5.1 and absent from the space, as the resolved spec "
                 "reports them: %s. Resolved anchors: `hidden_features` range from "
                 "the width rule `%s`; `n_posterior_draws` floor from `%s`."
                 % (fixed, spec.anchored_to["width_rule"], spec.anchored_to["draws_rule"]))
    return lines


def table_F(inventory: Dict[str, Any]) -> List[str]:
    rows = inventory["rows"]
    owner = {(r["file"], r["name"]): r["owner"] for r in rows}
    variants = INV.knob_variants(rows)
    # A searched RANGE is not a default: the range-bearing dataclasses are
    # left out, so the table compares values against values.
    range_classes = ("JointSpaceSpec.", "SpaceSpec.", "SearchConfig.",
                     "RegularizationConfig.")
    lines = ["| knob | where | name | default | doc |", "|---|---|---|---|---|"]
    kept = 0
    for k, items in variants:
        items = [(f, n, d) for f, n, d in items
                 if owner.get((f, n), "plumbing") != "plumbing"
                 and not n.startswith(range_classes)]
        if len({d for _, _, d in items}) < 2:
            continue
        kept += 1
        for i, (f, n, d) in enumerate(items):
            lines.append("| %s | %s | `%s` | `%s` | %s |"
                         % ("`" + k + "`" if i == 0 else "", f.split("/")[-1], n,
                            INV._short(d, 48), owner.get((f, n), "")))
    lines.insert(0, "%d knobs with more than one default among the owned rows "
                    "(plumbing excluded)." % kept)
    lines.insert(1, "")
    return lines


def table_K(inventory: Dict[str, Any]) -> List[str]:
    rows = [r for r in inventory["rows"] if r["owner"] not in ("plumbing", "UNOWNED")
            and not (r["surface"] == "signature" and r.get("required"))]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(INV.knob_name(r), []).append(r)
    lines = ["| knob | set where (surface: file, name) | doc |", "|---|---|---|"]
    for k in sorted(groups, key=lambda s: s.lower()):
        rs = sorted(groups[k], key=lambda r: (r["surface"], r["file"], r["name"]))
        docs = sorted({r["owner"] for r in rs})
        where = "; ".join("%s: %s, `%s`" % (r["surface"], r["file"].split("/")[-1], r["name"])
                          for r in rs)
        lines.append("| `%s` | %s | %s |" % (k, where, ", ".join(docs)))
    lines.insert(0, "%d knobs, %d owned rows. A knob is one normalised name; "
                    "its rows are every place that name carries a value."
                    % (len(groups), len(rows)))
    lines.insert(1, "")
    return lines


def render(hpc_dir: str, inventory_path: str) -> Dict[str, str]:
    JS, NTJ = _load_space(hpc_dir)
    with open(inventory_path, "r", encoding="ascii") as fh:
        inventory = json.load(fh)
    return {"A": "\n".join(table_A(JS, NTJ, inventory)) + "\n",
            "F": "\n".join(table_F(inventory)) + "\n",
            "K": "\n".join(table_K(inventory)) + "\n"}


def check_doc(doc_text: str, blocks: Dict[str, str]) -> List[Tuple[str, bool, str]]:
    out = []
    for key, (b, e) in MARK.items():
        a, z = doc_text.find(b), doc_text.find(e)
        if a < 0 or z < 0 or z < a:
            out.append((key, False, "markers not found in order"))
            continue
        got = doc_text[a + len(b):z].strip("\n")
        want = blocks[key].strip("\n")
        if got == want:
            out.append((key, True, "matches"))
            continue
        gl, wl = got.split("\n"), want.split("\n")
        msg = "block lengths %d vs %d" % (len(gl), len(wl))
        for i, (x, y) in enumerate(zip(gl, wl), 1):
            if x != y:
                msg = "first difference at line %d:\n    doc  : %s\n    fresh: %s" % (i, x[:120], y[:120])
                break
        out.append((key, False, msg))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpc-dir", required=True)
    ap.add_argument("--inventory", default=os.path.join(_HERE, "inventory.json"))
    ap.add_argument("--out-md", default=None, help="write the three blocks here")
    ap.add_argument("--check-doc", default=None, help="P0 file to check")
    args = ap.parse_args(argv)
    blocks = render(args.hpc_dir, args.inventory)
    if args.out_md:
        with open(args.out_md, "w", encoding="ascii") as fh:
            for key in ("A", "F", "K"):
                fh.write(MARK[key][0] + "\n" + blocks[key] + MARK[key][1] + "\n\n")
    rc = 0
    if args.check_doc:
        with open(args.check_doc, "r", encoding="ascii") as fh:
            results = check_doc(fh.read(), blocks)
        for key, ok, msg in results:
            print("%s  table %s: %s" % ("OK  " if ok else "FAIL", key, msg))
            rc = rc if ok else 1
    print("tables rendered: A %d lines, F %d lines, K %d lines"
          % tuple(blocks[k].count("\n") for k in ("A", "F", "K")))
    return rc


if __name__ == "__main__":
    sys.exit(main())
