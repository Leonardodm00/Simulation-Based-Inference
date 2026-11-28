"""Block 31, amendment 5: the runner's parser (specs/SPEC.md block 31, "Flags").

Oracles (from the spec, before the code was read; tester run 2026-11-17_1730):
  O31p.1 `build_parser` lifted out of run_joint_arms.py by AST with ARMS alone
         (what the stage-4 suites J28 and J35 do) builds the parser: the two
         choices are literal tuples, {cross_well, within_group} and {donor,
         rep_group, batch, well}, defaults cross_well and donor.
  O31p.2 the runner checks at import that these choices equal
         joint_batches.REP_PAIRS and its REP_GROUP_KEYS: a copy whose literal
         or whose constant differs fails to import (AssertionError), also
         under `python -O` (the check is not an assert statement).
  O31p.3 the parser is otherwise the base commit's (a915950): every action's
         attributes, the help and the usage are equal.
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

import _spec_util as U

RUNNER = os.path.join(U.JOINT, "stage3", "run_joint_arms.py")
BASE_STAGE3 = "/tmp/base_a915950/hpc/joint/stage3"


def _lift(path):
    tree = ast.parse(open(path, encoding="ascii").read())
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_parser"]
    arms = [n for n in tree.body if isinstance(n, ast.Assign)
            and getattr(n.targets[0], "id", "") == "ARMS"]
    assert len(fn) == 1 and len(arms) == 1
    ns = {"argparse": argparse, "__doc__": ""}
    exec(compile(ast.Module(body=arms + fn, type_ignores=[]), "<runner>", "exec"), ns)
    return ns["build_parser"]()


def _action(parser, flag):
    acts = [a for a in parser._actions if flag in a.option_strings]
    assert len(acts) == 1, flag
    return acts[0]


def test_lifted_parser_has_literal_choices_and_defaults():
    p = _lift(RUNNER)
    rp = _action(p, "--rep-pairs")
    rk = _action(p, "--rep-group-key")
    assert set(rp.choices) == {"cross_well", "within_group"} and rp.default == "cross_well"
    assert set(rk.choices) == {"donor", "rep_group", "batch", "well"} and rk.default == "donor"
    req = ["--arm", "A5", "--sim-shards", "s/*.npz", "--out-dir", "o"]
    ns = p.parse_args(req + ["--rep-pairs", "within_group", "--rep-group-key", "rep_group"])
    assert (ns.rep_pairs, ns.rep_group_key) == ("within_group", "rep_group")
    ns = p.parse_args(req)
    assert (ns.rep_pairs, ns.rep_group_key) == ("cross_well", "donor")
    with pytest.raises(SystemExit):
        p.parse_args(req + ["--rep-group-key", "subregion"])


_IMPORT = textwrap.dedent(r'''
    import importlib.util, os, sys
    joint, path = sys.argv[1], sys.argv[2]
    for d in (os.path.join(joint, "stage3"), joint, os.path.join(joint, "stage2"),
              os.path.join(joint, "stage1")):
        sys.path.insert(0, d)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("rja_copy", path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except AssertionError as exc:
        print("ASSERTION: %s" % exc)
        sys.exit(7)
    print("IMPORTED")
''')


def _import_copy(tmp_path, text, optimize=False):
    mod = tmp_path / "rja_copy.py"
    mod.write_text(text)
    script = tmp_path / "imp.py"
    script.write_text(_IMPORT)
    argv = [sys.executable] + (["-O"] if optimize else []) + [str(script), U.JOINT, str(mod)]
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return r.returncode, r.stdout


@pytest.mark.parametrize("old, new, optimize", [
    (None, None, False),                                          # the control: imports
    ('choices=("donor", "rep_group", "batch", "well")', 'choices=("donor", "rep_group", "batch")', False),
    ('choices=("donor", "rep_group", "batch", "well")', 'choices=("donor", "rep_group", "batch")', True),
    ('choices=("cross_well", "within_group")', 'choices=("cross_well", "within_group", "any")', False),
    ('REP_GROUP_KEYS = ("donor", "rep_group", "batch", "well")',
     'REP_GROUP_KEYS = ("donor", "rep_group", "batch", "well", "subregion")', True),
])
def test_import_time_check(tmp_path, old, new, optimize):
    text = open(RUNNER, encoding="ascii").read()
    if old is not None:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    rc, out = _import_copy(tmp_path, text, optimize)
    if old is None:
        assert rc == 0 and "IMPORTED" in out, out[-3000:]
    else:
        assert rc == 7 and "ASSERTION:" in out, out[-3000:]


_DUMP = textwrap.dedent(r'''
    import json, os, sys
    stage3, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, stage3)
    sys.dont_write_bytecode = True
    import run_joint_arms as R
    p = R.build_parser()
    acts = []
    for a in p._actions:
        acts.append({"type_": type(a).__name__, "option_strings": a.option_strings, "dest": a.dest,
                     "nargs": a.nargs, "const": repr(a.const), "default": repr(a.default),
                     "type": getattr(a.type, "__name__", repr(a.type)),
                     "choices": None if a.choices is None else list(a.choices),
                     "required": a.required, "help": a.help, "metavar": repr(a.metavar)})
    json.dump({"actions": acts, "help": p.format_help(), "usage": p.format_usage()},
              open(out, "w"), indent=1)
''')


def test_parser_is_the_base_commits(tmp_path):
    if not os.path.isdir(BASE_STAGE3):
        pytest.skip("no base tree (git archive a915950 hpc | tar -x -C /tmp/base_a915950)")
    script = tmp_path / "dump.py"
    script.write_text(_DUMP)
    got = {}
    for name, stage3 in (("head", os.path.join(U.JOINT, "stage3")), ("base", BASE_STAGE3)):
        o = str(tmp_path / (name + ".json"))
        r = subprocess.run([sys.executable, str(script), stage3, o], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=300,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        assert r.returncode == 0, r.stdout[-3000:]
        got[name] = json.load(open(o))
    h, b = got["head"], got["base"]
    assert len(h["actions"]) == len(b["actions"])
    for x, y in zip(h["actions"], b["actions"]):
        assert x == y, (x, y)
    assert h["help"] == b["help"] and h["usage"] == b["usage"]
