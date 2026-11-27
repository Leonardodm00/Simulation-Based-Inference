#!/usr/bin/env python3
"""Smoke test of block 42: the card (specs/SPEC.md block 42, "Test
oracles", checks T42*).

The card is text and its checks are exact: every comparison below is an
equality (of strings, integers, lists or dicts as JSON and YAML read them),
so no check has a numerical tolerance; a float pre-filled from a record must
come back from the card bit for bit (YAML writes a float by its repr).

Checks
    test_t42a_template       the template parses, its pre-filled values, the draft's fields
    test_t42b_complete       a complete card; a table of single faults
    test_t42c_malformed      malformed cards and unsafe YAML are refused
    test_t42d_round_trip     parse then dump gives the same header
    test_t42e_template_command card-template --bundle on a stand-in for block 41's reader
    test_t42f_bundle_commands  card-template --bundle and card-check --bundle on a real
                             bundle (SKIP while block 41's bundle.write/read are stubs)

Inputs: the records of ensemble S-A1:A1 of the recorded-kind fixture
(REC_FIXTURE, default /tmp/rec_fixture_fitsdtk2: runs_root/S-A1/synth_best/
member_seed<k>/A1_*seed<k>.json, read only); build one with
REC_KEEP=1 python3 hpc/pipeline/tests/rec_fixture.py when it is absent.

Run
    PYTHONDONTWRITEBYTECODE=1 python3 sbitool/tests/test_smoke_tool_card.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

Pure ASCII, LF only.
"""

import ast
import contextlib
import copy
import glob
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SEED = 20261127  # every random number of this file derives from it
_TMP = []


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)


import yaml  # noqa: E402

from sbitool import card as C  # noqa: E402
from sbitool import schema as S  # noqa: E402
from sbitool.tests import test_smoke_tool_schema as T40  # noqa: E402

FIXTURE = os.environ.get("REC_FIXTURE", "/tmp/rec_fixture_fitsdtk2")
BEST = os.path.join(FIXTURE, "runs_root", "S-A1", "synth_best")
RUNNER = os.path.join(_REPO, "hpc", "joint", "stage3", "run_joint_arms.py")

# the 12 fields block 42 lists as TODO, and training.n_networks: the records
# count rows (windows), never networks, so it is TODO too (never a guess)
WANT_TODO = sorted(["title", "scientific_question", "references", "authors", "licence", "cite",
                    "data", "limitations", "simulator.model_type", "simulator.reference",
                    "simulator.fixed_parameters", "training.hyperparameter_search",
                    "training.n_networks"])

# This test's own reading of run_joint_arms.py's build_parser (written here
# from the parser's help, not copied from card.py): the config keys that are
# NOT the model or its optimisation -- paths, switches of a run, the seeds and
# the fields the card holds on their own.
_NOT_HYPER = {"arm", "seed", "split_seed", "split_fracs",                       # own fields
              "sim_shards", "real_shards", "out_dir", "warm_start_ckpt",       # paths
              "dsn_main_dir", "sbi_hpc_dir",
              "dry_run", "resume", "stop_after_epoch", "probe_steps",          # switches
              "probe_val_rows", "threads", "buffer_shards"}


# ------------------------------------------------------------ fixtures

def _records():
    """The records of S-A1:A1 in member order (the seeds ascending)."""
    if not os.path.isdir(BEST):
        raise unittest.SkipTest("no fixture at %s: build it with REC_KEEP=1 python3 "
                                "hpc/pipeline/tests/rec_fixture.py" % FIXTURE)
    seeds = sorted(int(m.group(1)) for m in
                   (re.fullmatch(r"member_seed(\d+)", d) for d in os.listdir(BEST)) if m)
    recs = []
    for k in seeds:
        files = [f for f in glob.glob(os.path.join(BEST, "member_seed%d" % k, "A1_*.json"))
                 if re.search(r"seed%d\.json$" % k, f)]
        assert len(files) == 1, files
        with open(files[0]) as f:
            recs.append(json.load(f))
    assert len(recs) >= 2, "the fixture's ensemble has %d members" % len(recs)
    return recs


def _manifest(recs):
    """A manifest written from sec. 2.7 (T40a's), made consistent with the
    records: arm, M, the members' seeds and split, the bank's parameter
    names and box, W."""
    man = T40._valid_manifest()
    r0 = recs[0]
    shard_dir = os.path.dirname(r0["config"]["sim_shards"])
    with open(os.path.join(shard_dir, "shard_0000.json")) as f:
        side = json.load(f)
    M, names = len(recs), list(side["param_names"])
    man.update({"arm": r0["arm"], "M": M, "p": len(names), "W": int(side["W"]),
                "card_complete": False})
    man["members"] = [{"file": S.MEMBER_FILE % k, "sha256": T40._hex("w%d" % k),
                       "seed": r["seed"], "state_sha256": T40._hex("s%d" % k),
                       "split_seed": r["split_seed"], "split_hash": r["split_hash"]}
                      for k, r in enumerate(recs)]
    man["parameters"] = {"names": names, "units": ["1"] * len(names),
                         "bounds": [list(b) for b in side["bounds_theta"]]}
    man["software"] = dict(man["software"], python=platform.python_version())
    man["provenance"] = dict(man["provenance"], project="rec_fixture", ensemble="S-A1:A1",
                             campaign="S-A1")
    probs = S.validate_manifest(man)
    assert probs == [], probs
    return man


def _filled(header, man):
    """header with every TODO replaced by T40a's complete card's value
    (n_networks: a value the test writes, as an author would)."""
    full = T40._valid_card(man)
    h = copy.deepcopy(header)
    for name in S.card_todo_fields(h):
        path = name.split(".")
        src = full
        for k in path:
            src = src[k]
        T40._set(h, path, copy.deepcopy(src))
    return h


def _parser_dests():
    """The destinations of run_joint_arms.py's build_parser, read from its
    source (static: the runner imports torch)."""
    with open(RUNNER) as f:
        tree = ast.parse(f.read())
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and n.name == "build_parser"][0]
    out = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)):
            out.add(n.args[0].value.lstrip("-").replace("-", "_"))
    return out


def _run(fn, argv):
    """(exit code, stdout lines, stderr lines) of a command's main, in process."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(argv)
    return code, out.getvalue().splitlines(), err.getvalue().splitlines()


def _cli(*args):
    """(exit code, stdout lines, stderr lines) of python3 -m sbitool ..."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "sbitool"] + list(args), cwd=_REPO, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       timeout=120)
    return p.returncode, p.stdout.splitlines(), p.stderr.splitlines()


def _write(d, name, text):
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def _ascii(text):
    b = text.encode("utf-8")
    return all(c < 128 for c in b) and b"\r" not in b and b"\t" not in b


# ------------------------------------------------------------ checks

def test_t42a_template():
    """the template parses, its pre-filled values, the draft's fields (specs/SPEC.md block 42)."""
    recs = _records()
    man = _manifest(recs)
    text = C.template(man, recs, project="rec_fixture")
    assert _ascii(text), "the template is not pure ASCII / LF / no tabs"
    h, body = C.parse(text)
    # the header: CARD_KEYS in order, the nested sets in their order
    assert tuple(h) == S.CARD_KEYS, list(h)
    assert tuple(h["simulator"]) == S.CARD_SIMULATOR_KEYS, list(h["simulator"])
    assert tuple(h["training"]) == S.CARD_TRAINING_KEYS, list(h["training"])
    assert tuple(h["extraction"]) == S.EXTRACTION_KEYS, list(h["extraction"])
    assert tuple(h["software"]) == S.SOFTWARE_KEYS, list(h["software"])
    # the pre-filled values, computed here from the records and the manifest
    t = h["training"]
    assert t["arm"] == man["arm"] == recs[0]["arm"] == "A1", t["arm"]
    assert t["members"] == man["M"] == len(recs) and isinstance(t["members"], int), t["members"]
    n_train = {r["n_train"] for r in recs}
    assert len(n_train) == 1, n_train              # one shared split (D-089)
    assert t["n_windows"] == n_train.pop(), t["n_windows"]
    with open(os.path.join(BEST, "BEST_FIT_COMPLETE.json")) as f:
        ident = json.load(f)["finalist"]["identity"]
    assert t["n_windows"] == ident["n_train"], (t["n_windows"], ident["n_train"])  # 2nd source
    assert t["split_fracs"] == recs[0]["split_fracs"] == recs[0]["config"]["split_fracs"]
    assert t["n_networks"] == S.TODO, t["n_networks"]
    # n_networks is no record's: the export counts it on the bank and passes it (block 42)
    h_n, _ = C.parse(C.template(man, recs, project="rec_fixture", n_networks=136))
    assert h_n["training"]["n_networks"] == 136 and h_n["training"]["n_windows"] == t["n_windows"]
    for bad in (0, -1, True, 1.5, "136"):
        try:
            C.template(man, recs, n_networks=bad)
            raise AssertionError("n_networks %r accepted" % (bad,))
        except ValueError:
            pass
    want_h = dict((k, v) for k, v in recs[0]["config"].items() if k not in _NOT_HYPER)
    assert t["hyperparameters"] == want_h, (sorted(t["hyperparameters"]), sorted(want_h))
    for r in recs[1:]:                              # every member trained with the same values
        assert dict((k, v) for k, v in r["config"].items() if k not in _NOT_HYPER) == want_h
    assert h["simulator"]["searched_parameters"] == man["parameters"]["names"]
    assert h["extraction"] == man["extraction"]
    assert h["software"] == man["software"]
    # the runner's parser is partitioned exactly into hyperparameters and the rest,
    # in card.py and in this test's own reading
    dests = _parser_dests()
    assert set(C.HYPERPARAMETER_KEYS) | set(C.RUN_KEYS) == dests, \
        (sorted(dests - set(C.HYPERPARAMETER_KEYS) - set(C.RUN_KEYS)),
         sorted(set(C.HYPERPARAMETER_KEYS) | set(C.RUN_KEYS) - dests))
    assert not set(C.HYPERPARAMETER_KEYS) & set(C.RUN_KEYS)
    assert set(C.RUN_KEYS) == _NOT_HYPER, sorted(set(C.RUN_KEYS) ^ _NOT_HYPER)
    # the draft: valid against the manifest, exactly the TODO fields
    assert S.validate_card_header(h, man) == [], S.validate_card_header(h, man)
    assert S.card_todo_fields(h) == WANT_TODO, S.card_todo_fields(h)
    # T40a's draft (block 42's list, by block 40's author) is TODO where this one is
    draft40 = T40._draft_card(man)
    for k in ("title", "scientific_question", "references", "authors", "licence", "cite",
              "data", "limitations"):
        assert h[k] == draft40[k] == S.TODO, k
    # the body: a headed section for every field to fill
    for f in WANT_TODO:
        assert "### `%s`\n" % f in body, "no section for %s" % f
    assert "rec_fixture" in body and "S-A1:A1" in body
    # card-check (the command line, through the dispatcher): draft, the fields named
    d = _tmpdir("t42a")
    path = _write(d, "CARD.md", text)
    code, out, err = _cli("card-check", path)
    assert code == 0, (code, out, err)
    assert out[0] == "CARD OK: draft, %d field(s) to fill: %s" % (len(WANT_TODO),
                                                                   ", ".join(WANT_TODO)), out
    # a manifest without extraction: "not applicable"
    man0 = copy.deepcopy(man)
    man0["extraction"] = None
    h0, _ = C.parse(C.template(man0, recs))
    assert h0["extraction"] == S.NOT_APPLICABLE and S.validate_card_header(h0, man0) == []
    # no value guessed: no records -> the record fields are TODO; inconsistent
    # records are refused
    hn, _ = C.parse(C.template(man, None))
    assert [hn["training"][k] for k in ("n_windows", "split_fracs", "hyperparameters")] \
        == [S.TODO] * 3
    for label, mut in (("one record", lambda rs: rs[:1]),
                       ("arm differs", lambda rs: [dict(rs[0], arm="A2")] + rs[1:]),
                       ("n_train differs", lambda rs: rs[:1] + [dict(rs[1], n_train=1)]),
                       ("seeds out of order", lambda rs: rs[::-1]),
                       ("lr differs", lambda rs: rs[:1] + [dict(rs[1], config=dict(
                           rs[1]["config"], lr=0.5))])):
        try:
            C.template(man, mut(copy.deepcopy(recs)))
        except ValueError:
            continue
        raise AssertionError("template accepted records with %s" % label)


def _fault_table():
    """(label, mutate(header), the key the one problem names)."""
    def drop(path):
        return lambda h: T40._drop(h, path)

    def put(path, v):
        return lambda h: T40._set(h, path, v)
    return [
        ("a key removed", drop(["licence"]), "licence"),
        ("a nested key removed", drop(["training", "n_windows"]), "training.n_windows"),
        ("an unknown key added", put(["notes"], "hi"), "notes"),
        ("an unknown nested key", put(["simulator", "version"], "1"), "simulator.version"),
        ("an empty list", put(["references"], []), "references"),
        ("an empty string", put(["authors", 0, "contact"], ""), "authors[0].contact"),
        ("arm changed", put(["training", "arm"], "A2"), "training.arm"),
        ("members changed", put(["training", "members"], 3), "training.members"),
        ("a searched parameter renamed", put(["simulator", "searched_parameters", 0], "zeta"),
         "simulator.searched_parameters"),
        ("a TODO left in a nested value", put(["training", "hyperparameters", "lr"], "TODO"),
         "training.hyperparameters.lr"),
        ("a TODO left in an author", put(["authors", 0, "name"], "TODO"), "authors[0].name"),
        ("software changed", put(["software", "torch"], "0.0.1"), "software.torch"),
    ]


def test_t42b_complete():
    """a complete card; a table of single faults (specs/SPEC.md block 42)."""
    recs = _records()
    man = _manifest(recs)
    h, body = C.parse(C.template(man, recs))
    full = _filled(h, man)
    assert S.card_todo_fields(full) == []
    text = C.dump(full, body)
    assert C.check(text, man, complete=True) == [], C.check(text, man, complete=True)
    # the pre-filled values are untouched by the filling
    for k in ("arm", "members", "n_windows", "split_fracs", "hyperparameters"):
        assert full["training"][k] == h["training"][k], k
    d = _tmpdir("t42b")
    code, out, err = _run(C.main_check, [_write(d, "CARD.md", text)])
    assert code == 0 and out[0] == "CARD OK: complete", (code, out, err)
    code, out, err = _cli("card-check", os.path.join(d, "CARD.md"))
    assert code == 0 and out[0] == "CARD OK: complete", (code, out, err)
    # the faults: exactly one problem naming the key, against the manifest
    for label, mutate, key in _fault_table():
        f = copy.deepcopy(full)
        mutate(f)
        probs = C.check(C.dump(f, body), man, complete=True)
        assert len(probs) == 1 and key in probs[0], (label, probs)
    # through the command line (no manifest): a missing or unknown key exits 3
    # naming it; an empty value or a TODO makes a draft naming it
    for label, mutate, key, want in (
            ("a key removed", lambda x: T40._drop(x, ["licence"]), "licence", 3),
            ("an unknown key", lambda x: T40._set(x, ["notes"], "hi"), "notes", 3),
            ("an empty list", lambda x: T40._set(x, ["references"], []), "references", 0),
            ("a nested TODO", lambda x: T40._set(x, ["training", "hyperparameters", "lr"],
                                                 "TODO"), "training.hyperparameters.lr", 0)):
        f = copy.deepcopy(full)
        mutate(f)
        code, out, err = _run(C.main_check, [_write(d, "F.md", C.dump(f, body))])
        assert code == want, (label, code, out, err)
        if want == 3:
            assert out == [] and len(err) == 1 and key in err[0], (label, out, err)
        else:
            assert out[0] == "CARD OK: draft, 1 field(s) to fill: %s" % key, (label, out)


def test_t42c_malformed():
    """malformed cards and unsafe YAML are refused (specs/SPEC.md block 42)."""
    d = _tmpdir("t42c")
    sentinel = os.path.join(d, "pwned")
    cases = [
        ("no header", "# A card\n\ntitle: x\n"),
        ("one ---", "---\ntitle: x\nlicence: MIT\n"),
        ("a tab error", "---\ntitle: x\n\tlicence: MIT\n---\nbody\n"),
        ("a header that is a list", "---\n- title\n- licence\n---\nbody\n"),
        ("an empty header", "---\n---\nbody\n"),
        ("a python object tag",
         "---\ntitle: !!python/object/apply:os.system ['touch %s']\n---\n" % sentinel),
        ("a python name tag", "---\ntitle: !!python/name:os.system ''\n---\n"),
        ("a python object/new tag",
         "---\ntitle: !!python/object/new:subprocess.Popen [['touch', '%s']]\n---\n"
         % sentinel),
    ]
    for label, text in cases:
        try:
            C.parse(text)
        except C.CardError:
            pass
        else:
            raise AssertionError("parse accepted %s" % label)
        try:
            C.check(text)
        except C.CardError:
            pass
        else:
            raise AssertionError("check accepted %s" % label)
        code, out, err = _run(C.main_check, [_write(d, "BAD.md", text)])
        assert code == 3 and out == [] and len(err) == 1 and err[0].startswith("CARD REFUSED:"), \
            (label, code, out, err)
    assert not os.path.exists(sentinel), "a YAML tag ran a command"
    # the unsafe loader would have built the object: the refusal is safe_load's doing
    try:
        yaml.load("x: !!python/name:os.system ''", Loader=yaml.UnsafeLoader)["x"]
    except Exception as exc:  # noqa: BLE001 -- only to show what the full loader does
        raise AssertionError("the probe of the unsafe loader failed: %r" % exc)
    # one through the dispatcher, the process's own exit code
    code, out, err = _cli("card-check", os.path.join(d, "BAD.md"))
    assert code == 3 and len(err) == 1, (code, out, err)
    # the arguments: no card file, no argument -> 2
    assert _run(C.main_check, [os.path.join(d, "absent.md")])[0] == 2
    assert _run(C.main_check, [])[0] == 2
    assert _run(C.main_template, ["--out", os.path.join(d, "x.md")])[0] == 2


def test_t42d_round_trip():
    """parse then dump gives the same header (specs/SPEC.md block 42)."""
    recs = _records()
    man = _manifest(recs)
    text = C.template(man, recs)
    h, b = C.parse(text)
    assert C.dump(h, b) == text                       # the text itself, byte for byte
    assert C.parse(C.dump(h, b)) == (h, b)
    # the complete card, with a non-ASCII value, a body with --- lines and
    # one sub-object shared by two keys (an alias would appear)
    full = _filled(h, man)
    full["authors"] = [{"name": "Jos\u00e9 \u00c5ngstr\u00f6m", "contact": "j@example.org"}]
    full["simulator"]["fixed_parameters"] = {"note": "a\n---\nb", "N": "1000 neurons"}
    shared = {"location": "zenodo:1", "public": True}
    full["data"] = {"simulated_bank": dict(shared, size_bytes=10, sha256=None),
                    "recordings": shared}
    full["references"] = ["one", "two"]
    body = "\n# Notes\n\n---\n\ntext\n---\n"
    t2 = C.dump(full, body)
    assert _ascii(t2.split("---\n", 2)[1]), "the header is not pure ASCII"
    assert "&id" not in t2 and "*id" not in t2, "the header has YAML anchors"
    assert C.parse(t2) == (full, body)
    assert C.check(t2, man, complete=True) == [], C.check(t2, man, complete=True)
    # the key order is the schema's whatever the dict's order
    rev = dict(reversed(list(copy.deepcopy(full).items())))
    rev["training"] = dict(reversed(list(rev["training"].items())))
    assert C.dump(rev, body) == t2
    # a header safe_dump cannot write is refused, not written
    try:
        C.dump(dict(full, title=("a", "b")), body)
    except C.CardError:
        pass
    else:
        raise AssertionError("dump wrote a tuple")


def test_t42e_template_command():
    """card-template --bundle on a stand-in for block 41's reader (specs/SPEC.md block 42)."""
    from sbitool import bundle
    recs = _records()
    man = _manifest(recs)
    h, body = C.parse(C.template(man, recs))
    full = _filled(h, man)

    class _Stand:            # bundle.Bundle's attributes that card.py reads
        manifest, card = man, C.dump(full, body)
    d = _tmpdir("t42e")
    bpath = _write(d, "toy.sbibundle", "stand-in")
    saved = bundle.read
    bundle.read = lambda path: _Stand()
    try:
        out_md = os.path.join(d, "CARD.md")
        code, out, err = _run(C.main_template, ["--bundle", bpath, "--out", out_md])
        assert code == 0, (code, out, err)
        with open(out_md) as f:
            text = f.read()
        assert _ascii(text)
        ht, _ = C.parse(text)
        want = copy.deepcopy(h)       # the training fields come from the bundle's card
        for k in ("n_networks", "n_windows", "split_fracs", "hyperparameters"):
            want["training"][k] = full["training"][k]
        assert ht == want, (ht, want)
        left = sorted(set(WANT_TODO) - {"training.n_networks"})
        assert out == ["CARD TEMPLATE OK: %s, %d field(s) to fill: %s"
                       % (out_md, len(left), ", ".join(left))], out
        # never overwrites
        code, out, err = _run(C.main_template, ["--bundle", bpath, "--out", out_md])
        assert code == 3 and len(err) == 1 and "exists" in err[0], (code, out, err)
        # a bundle whose card is not a card: the record fields stay TODO
        _Stand.card = "not a card"
        out2 = os.path.join(d, "CARD2.md")
        assert _run(C.main_template, ["--bundle", bpath, "--out", out2])[0] == 0
        with open(out2) as f:
            h2, _ = C.parse(f.read())
        assert [h2["training"][k] for k in ("n_networks", "n_windows", "split_fracs",
                                            "hyperparameters")] == [S.TODO] * 4
        assert h2["training"]["arm"] == man["arm"] and h2["software"] == man["software"]
    finally:
        bundle.read = saved


def test_t42f_bundle_commands():
    """card-template --bundle and card-check --bundle on a bundle (specs/SPEC.md block 42)."""
    from sbitool import bundle
    d = _tmpdir("t42f")
    try:
        bundle.read(os.path.join(d, "absent.sbibundle"))
    except NotImplementedError:
        raise unittest.SkipTest("block 41: bundle.read is a stub in this tree; the "
                                "orchestrator runs this check on the merged tree")
    except Exception:  # noqa: BLE001 -- any refusal of an absent file: block 41 is written
        pass
    import numpy as np
    import torch
    recs = _records()
    man = _manifest(recs)
    h, body = C.parse(C.template(man, recs))
    full = _filled(h, man)
    states = [{"w": torch.zeros(3, dtype=torch.float32) + k} for k in range(man["M"])]
    rng = np.random.default_rng(SEED)
    # the gate's sets of the sizes the manifest declares (bundle.check compares them)
    gate = dict((k, rng.standard_normal((man["M"], man["gate"][n], man["d_z"])).astype(np.float32))
                for k, n in (("z_ref", "n_ref"), ("z_rank", "n_rank"), ("z_null", "n_null")))
    bpath = os.path.join(d, "toy" + bundle.EXTENSION)
    man_w = copy.deepcopy(man)
    man_w["card_complete"] = True
    bundle.write(bpath, man_w, C.dump(full, body), states, gate, b"{}")
    # card-template --bundle: the manifest's fields, the training fields of
    # the bundle's own card, TODO elsewhere
    out_md = os.path.join(d, "CARD.md")
    code, out, err = _run(C.main_template, ["--bundle", bpath, "--out", out_md])
    assert code == 0 and out[0].startswith("CARD TEMPLATE OK: "), (code, out, err)
    with open(out_md) as f:
        ht, _ = C.parse(f.read())
    want = copy.deepcopy(h)
    for k in ("n_networks", "n_windows", "split_fracs", "hyperparameters"):
        want["training"][k] = full["training"][k]
    assert ht == want, (ht, want)
    assert _run(C.main_template, ["--bundle", bpath, "--out", out_md])[0] == 3  # no overwrite
    # card-check --bundle: complete; arm changed -> exit 3 naming it
    code, out, err = _run(C.main_check, [_write(d, "FULL.md", C.dump(full, body)),
                                         "--bundle", bpath])
    assert code == 0 and out == ["CARD OK: complete"], (code, out, err)
    bad = copy.deepcopy(full)
    bad["training"]["arm"] = "A2"
    code, out, err = _run(C.main_check, [_write(d, "BAD.md", C.dump(bad, body)),
                                         "--bundle", bpath])
    assert code == 3 and len(err) == 1 and "training.arm" in err[0], (code, out, err)


CHECKS = [("test_t42a_template", test_t42a_template),
          ("test_t42b_complete", test_t42b_complete),
          ("test_t42c_malformed", test_t42c_malformed),
          ("test_t42d_round_trip", test_t42d_round_trip),
          ("test_t42e_template_command", test_t42e_template_command),
          ("test_t42f_bundle_commands", test_t42f_bundle_commands)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    try:
        for name, fn in CHECKS:
            if only and name not in only:
                continue
            t1 = time.time()
            try:
                fn()
                rows.append((name, "PASS", "", time.time() - t1))
            except NotImplementedError as exc:
                bad += 1
                rows.append((name, "TODO", str(exc), time.time() - t1))
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        _cleanup()
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
