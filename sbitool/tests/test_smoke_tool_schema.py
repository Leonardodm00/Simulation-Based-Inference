#!/usr/bin/env python3
"""Smoke test of block 40: the tool's package, its schemas and its
interfaces (specs/SPEC.md block 40, "Test oracles", checks T40*).

Every check is written from the oracle of the same name in specs/SPEC.md. A
check that cannot run here is not deleted: it raises
unittest.SkipTest("<reason>"), so the summary shows why. No comparison here
has a tolerance: the validators' results are lists of strings, the digests
hex strings, and duration_s is the same float division as the test's.

Checks
    test_t40a_validators   a valid manifest and card; a table of single faults, each named
    test_t40b_recording    Recording.check and duration_s
    test_t40c_sources      the source groups: the modules imported, their digests; SourceError
    test_t40d_cli_static   python3 -m sbitool without a command; the package's imports

Run
    OMP_NUM_THREADS=2 SBIX_CLONE=<Sbi-extractor checkout> \
        python3 sbitool/tests/test_smoke_tool_schema.py [check ...]
        (T40c imports the estimator's modules: torch, sbi, zuko; without
        SBIX_CLONE or SBIX_DIR it checks only the extractor's SourceError)
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO.

Pure ASCII, LF only.
"""

import ast
import copy
import datetime
import hashlib
import importlib
import json
import os
import random
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


import numpy as np  # noqa: E402

from sbitool import schema as S  # noqa: E402
from sbitool import _sources  # noqa: E402
from sbitool.interfaces import Recording  # noqa: E402


# ------------------------------------------------------------ T40a fixtures
# A manifest and a card header written from specs/SPEC.md sec. 2.7, by hand:
# M = 3 members, p = 2 parameters, d_z = 4, W = 50.

def _hex(tag):
    return hashlib.sha256(tag.encode("ascii")).hexdigest()


def _valid_manifest():
    M, p = 3, 2
    return {
        "schema": "sbitool_bundle/2", "name": "toy_bundle", "created": "2026-11-27T10:00:00Z",
        "card_complete": True, "arm": "A1", "M": M, "p": p, "d_z": 4, "W": 50,
        "members": [{"file": "members/member_%d.safetensors" % k, "sha256": _hex("w%d" % k),
                     "seed": 10 + k, "state_sha256": _hex("s%d" % k), "split_seed": 7,
                     "split_hash": _hex("split")} for k in range(M)],
        "architecture": {"backbone": "dsn", "embedding_size": 4, "depth_exponent": 3,
                         "width_multiplier": 2.0, "block_family": 0, "head_fusion": 1,
                         "dropout": 0.1, "head_pool_ops": 1, "hidden_features": 48,
                         "num_transforms": 3, "num_bins": 8, "hidden_layers": None,
                         "z_score_theta": "transform_to_unconstrained", "z_score_x": "none",
                         "backbone_repr": "DSNBackbone(...)"},
        "parameters": {"names": ["g_syn", "tau_m"], "units": ["nS", "ms"],
                       "bounds": [[0.1, 2.0], [5.0, 40.0]]},
        "extraction": {"electrodes_per_subset": 9, "n_subsets": 9, "mfr_threshold_hz": 0.1,
                       "w_size_s": 0.02, "gaussian_window_s": 0.04, "grid_width": 64,
                       "index_base": 0, "pitch_um": 60.0, "fs_raw_hz": 10000.0,
                       "detection": dict(S.DETECTION_DEFAULT)},
        "gate": {"factors": [0.5, 1.0, 2.0], "alpha": 0.05, "n_ref": 200, "n_rank": 100,
                 "n_null": 199, "k_gate": 9, "size_mean": 0.045, "size_lo": 0.02,
                 "size_hi": 0.08, "level": 0.95},
        "sample": {"n_post": 300, "seed": 0, "batch_rows": 256},
        "provenance": {"project": "toy", "ensemble": "S-A1:A1", "search": "S-A1",
                       "campaign": "toy_campaign", "best_fit_complete_sha256": _hex("bf"),
                       "sample_json_sha256": _hex("sj"), "commit": "1314931",
                       "sources": {"hpc/dsn/backbone.py": _hex("bb")}},
        "software": {"python": "3.13.0", "torch": "2.5.1", "sbi": "0.27.0", "zuko": "1.2.0",
                     "numpy": "2.1.0", "safetensors": "0.4.5"},
        "training": None,       # block 48: no training information
        "files": {"CARD.md": _hex("card"), "evaluation/sim.json": _hex("sim"),
                  "evaluation/gate.safetensors": _hex("gate")},
    }


def _valid_card(man):
    """A complete card whose fields agree with `man`."""
    return {
        "title": "A toy ensemble", "scientific_question": "Which g_syn fits the bursts?",
        "simulator": {"model_type": "spiking network", "reference": "doi:10/toy",
                      "fixed_parameters": {"N": "1000 neurons"},
                      "searched_parameters": list(man["parameters"]["names"])},
        "training": {"arm": man["arm"], "members": man["M"], "n_networks": 120,
                     "n_windows": 36000, "split_fracs": [0.7, 0.15, 0.15],
                     "hyperparameters": {"lr": 0.001, "b_sim": 128},
                     "hyperparameter_search": "a GP search of 60 trials"},
        "extraction": copy.deepcopy(man["extraction"]),
        "references": ["Doe 2026"],
        "authors": [{"name": "A. Author", "contact": "a@example.org"}],
        "licence": "CC-BY-4.0", "cite": "Author 2026",
        "data": {"simulated_bank": {"location": "zenodo:1", "public": True, "size_bytes": 10,
                                    "sha256": None},
                 "recordings": {"location": "on request", "public": False}},
        "software": dict(man["software"]),
        "limitations": "trained on one network model",
    }


def _draft_card(man):
    """The card with TODO where the export cannot fill (block 42's list)."""
    c = _valid_card(man)
    for k in ("title", "scientific_question", "references", "authors", "licence", "cite",
              "data", "limitations"):
        c[k] = S.TODO
    for k in ("model_type", "reference", "fixed_parameters"):
        c["simulator"][k] = S.TODO
    c["training"]["hyperparameter_search"] = S.TODO
    return c


def _set(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def _drop(d, path):
    for k in path[:-1]:
        d = d[k]
    del d[path[-1]]


def _paths(v, prefix=()):
    """Every path (tuple of keys and indices) into a JSON value."""
    out = []
    if isinstance(v, dict):
        for k in v:
            out.append(prefix + (k,))
            out.extend(_paths(v[k], prefix + (k,)))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            out.append(prefix + (i,))
            out.extend(_paths(x, prefix + (i,)))
    return out


# values that are valid JSON (json.loads reads NaN, Infinity and long
# integers) or valid YAML (dates, bytes, integer keys), none of which may make
# a validator raise
_ODD = [None, True, False, 0, -1, 1, 10 ** 400, -10 ** 400, float("nan"), float("inf"),
        float("-inf"), 1e308, "", "TODO", " ", "x", [], {}, [None], [[1, 2]], {"a": None},
        {1: 2}, datetime.date(2026, 11, 27), datetime.datetime(2026, 11, 27, 1, 2, 3), b"\x00",
        json.loads('[NaN, Infinity, 1' + '0' * 400 + ']'),
        # not JSON but a Python value a caller may pass: an int too long for str()
        10 ** 5000, {10 ** 5000: 1}]


def _random_json(rng, depth):
    r = rng.random()
    if depth <= 0 or r < 0.4:
        return rng.choice(_ODD[:20])
    if r < 0.7:
        return [_random_json(rng, depth - 1) for _ in range(rng.randint(0, 3))]
    keys = ["a", "M", "members", "TODO", "1"]
    return dict((rng.choice(keys) + str(i), _random_json(rng, depth - 1))
                for i in range(rng.randint(0, 3)))


def _one(probs, key, label):
    """exactly one problem, and it names the key"""
    assert len(probs) == 1, "%s: expected exactly one problem, got %d: %r" % (label, len(probs),
                                                                            probs)
    assert key in probs[0], "%s: the problem does not name %r: %r" % (label, key, probs[0])


def _fault_table():
    """(label, target, mutate(manifest, card), the key the one problem names).
    target: 'manifest', 'extraction', 'card' (complete, against the manifest)."""
    nan, big = float("nan"), 10 ** 400
    m = lambda f: (lambda man, card: f(man))  # noqa: E731
    c = lambda f: (lambda man, card: f(card))  # noqa: E731
    return [
        # a missing key, at three depths
        ("missing top key", "manifest", m(lambda d: _drop(d, ["W"])), "W"),
        ("missing gate key", "manifest", m(lambda d: _drop(d, ["gate", "alpha"])), "gate.alpha"),
        ("missing member key", "manifest", m(lambda d: _drop(d, ["members", 1, "split_hash"])),
         "members[1].split_hash"),
        ("missing detection key", "extraction",
         m(lambda d: _drop(d, ["extraction", "detection", "exclude_sweep_ms"])),
         "extraction.detection.exclude_sweep_ms"),
        ("missing card key", "card", c(lambda d: _drop(d, ["limitations"])), "limitations"),
        ("missing card nested key", "card", c(lambda d: _drop(d, ["data", "recordings", "public"])),
         "data.recordings.public"),
        # an unknown key
        ("unknown top key", "manifest", m(lambda d: _set(d, ["colour"], "red")), "colour"),
        ("unknown architecture key", "manifest",
         m(lambda d: _set(d, ["architecture", "n_layers"], 3)), "architecture.n_layers"),
        ("unknown extraction key", "extraction", m(lambda d: _set(d, ["extraction", "x"], 1)),
         "extraction.x"),
        ("unknown card key", "card", c(lambda d: _set(d, ["notes"], "hi")), "notes"),
        ("unknown author key", "card", c(lambda d: _set(d, ["authors", 0, "orcid"], "0000")),
         "authors[0].orcid"),
        # a wrong type
        ("M a string", "manifest", m(lambda d: _set(d, ["M"], "3")), "M"),
        ("card_complete a string", "manifest", m(lambda d: _set(d, ["card_complete"], "yes")),
         "card_complete"),
        ("bool where a number", "manifest", m(lambda d: _set(d, ["gate", "alpha"], True)),
         "gate.alpha"),
        ("bool where an int", "extraction", m(lambda d: _set(d, ["extraction", "n_subsets"], True)),
         "extraction.n_subsets"),
        ("list where a dict", "manifest", m(lambda d: _set(d, ["architecture"], [])),
         "architecture"),
        ("list where a dict (card)", "card", c(lambda d: _set(d, ["training"], ["A1"])),
         "training"),
        ("float where an int", "manifest", m(lambda d: _set(d, ["W"], 50.0)), "W"),
        ("unknown backbone", "manifest", m(lambda d: _set(d, ["architecture", "backbone"], "cnn")),
         "architecture.backbone"),
        ("index_base 2", "extraction", m(lambda d: _set(d, ["extraction", "index_base"], 2)),
         "extraction.index_base"),
        ("detection method", "extraction",
         m(lambda d: _set(d, ["extraction", "detection", "method"], "locally_exclusive")),
         "extraction.detection.method"),
        ("public a string", "card", c(lambda d: _set(d, ["data", "recordings", "public"], "no")),
         "data.recordings.public"),
        # the cross-conditions of sec. 2.7
        ("M not the members' number", "manifest", m(lambda d: _set(d, ["M"], 4)), "members"),
        ("a member dropped", "manifest", m(lambda d: d["members"].pop()), "members"),
        ("p names", "manifest", m(lambda d: d["parameters"]["names"].pop()), "parameters.names"),
        ("p units", "manifest", m(lambda d: d["parameters"]["units"].append("s")),
         "parameters.units"),
        ("p bounds", "manifest", m(lambda d: d["parameters"]["bounds"].pop()),
         "parameters.bounds"),
        ("lo = hi", "manifest", m(lambda d: _set(d, ["parameters", "bounds", 1], [5.0, 5.0])),
         "parameters.bounds[1]"),
        ("lo > hi", "manifest", m(lambda d: _set(d, ["parameters", "bounds", 0], [2.0, 0.1])),
         "parameters.bounds[0]"),
        ("member file out of order", "manifest",
         m(lambda d: _set(d, ["members", 2, "file"], "members/member_1.safetensors")),
         "members[2].file"),
        ("seeds not ascending", "manifest", m(lambda d: _set(d, ["members", 2, "seed"], 10)),
         "members[2].seed"),
        ("a member digest's form", "manifest",
         m(lambda d: _set(d, ["members", 0, "sha256"], _hex("x").upper())), "members[0].sha256"),
        ("a file digest's form", "manifest",
         m(lambda d: _set(d, ["files", "CARD.md"], _hex("x")[:63])), "files['CARD.md']"),
        ("a provenance digest's form", "manifest",
         m(lambda d: _set(d, ["provenance", "sample_json_sha256"], 12)),
         "provenance.sample_json_sha256"),
        ("schema", "manifest", m(lambda d: _set(d, ["schema"], "sbitool_bundle/1")), "schema"),
        ("created not ISO", "manifest", m(lambda d: _set(d, ["created"], "yesterday")), "created"),
        # non-finite numbers and an integer beyond float range
        ("NaN", "manifest", m(lambda d: _set(d, ["gate", "level"], nan)), "gate.level"),
        ("inf", "extraction", m(lambda d: _set(d, ["extraction", "pitch_um"], float("inf"))),
         "extraction.pitch_um"),
        ("NaN bound", "manifest", m(lambda d: _set(d, ["parameters", "bounds", 0, 1], nan)),
         "parameters.bounds[0]"),
        ("int beyond float range", "manifest", m(lambda d: _set(d, ["gate", "size_mean"], big)),
         "gate.size_mean"),
        ("int beyond float range (extraction)", "extraction",
         m(lambda d: _set(d, ["extraction", "fs_raw_hz"], big)), "extraction.fs_raw_hz"),
        ("negative rate", "extraction", m(lambda d: _set(d, ["extraction", "fs_raw_hz"], -1.0)),
         "extraction.fs_raw_hz"),
        ("alpha out of (0, 1)", "manifest", m(lambda d: _set(d, ["gate", "alpha"], 1.0)),
         "gate.alpha"),
        # a complete card: TODO, empty values, agreement with the manifest
        ("TODO in a complete card", "card",
         c(lambda d: _set(d, ["training", "hyperparameter_search"], "TODO")),
         "training.hyperparameter_search"),
        ("TODO nested in a complete card", "card",
         c(lambda d: _set(d, ["simulator", "fixed_parameters", "N"], "TODO")),
         "simulator.fixed_parameters.N"),
        ("TODO for a whole object", "card", c(lambda d: _set(d, ["data"], "TODO")), "data"),
        ("an empty list", "card", c(lambda d: _set(d, ["references"], [])), "references"),
        ("an empty string", "card", c(lambda d: _set(d, ["authors", 0, "contact"], "")),
         "authors[0].contact"),
        ("arm differs", "card", c(lambda d: _set(d, ["training", "arm"], "A2")), "training.arm"),
        ("members differ", "card", c(lambda d: _set(d, ["training", "members"], 2)),
         "training.members"),
        ("searched parameter renamed", "card",
         c(lambda d: _set(d, ["simulator", "searched_parameters", 1], "tau_s")),
         "simulator.searched_parameters"),
        ("software differs", "card", c(lambda d: _set(d, ["software", "torch"], "2.6.0")),
         "software.torch"),
        ("extraction differs", "card", c(lambda d: _set(d, ["extraction", "n_subsets"], 4)),
         "extraction.n_subsets"),
        ("extraction not applicable", "card", c(lambda d: _set(d, ["extraction"],
                                                               S.NOT_APPLICABLE)), "extraction"),
    ]


def test_t40a_validators():
    """a valid manifest and card; a table of single faults, each named (specs/SPEC.md block 40)."""
    man = _valid_manifest()
    card = _valid_card(man)
    # the valid objects give [] (exact: a validator either finds a problem or not)
    assert S.validate_manifest(man) == [], S.validate_manifest(man)
    assert S.validate_extraction(man["extraction"]) == [], S.validate_extraction(man["extraction"])
    assert S.validate_card_header(card, man, complete=True) == [], \
        S.validate_card_header(card, man, complete=True)
    assert S.card_todo_fields(card) == []
    # a value below a named field (a hyperparameter) may be empty or null; a TODO in it may not
    c = copy.deepcopy(card)
    c["training"]["hyperparameters"].update(head_pool_ops=[], warm_start_ckpt=None, tag="")
    assert S.validate_card_header(c, man, complete=True) == [], \
        S.validate_card_header(c, man, complete=True)
    # the manifest's key set is the module's constant, in sec. 2.7's order
    assert tuple(man) == S.MANIFEST_KEYS and tuple(card) == S.CARD_KEYS
    # a JSON round trip changes nothing
    assert S.validate_manifest(json.loads(json.dumps(man))) == []
    # a manifest without extraction, and a card that says so
    man0, card0 = copy.deepcopy(man), copy.deepcopy(card)
    man0["extraction"] = None
    card0["extraction"] = S.NOT_APPLICABLE
    assert S.validate_manifest(man0) == [] and S.validate_card_header(card0, man0, True) == []
    _one(S.validate_card_header(card, man0, True), "extraction", "extraction vs null")
    # detection null (spike trains), bounds null, hidden_layers an int
    man1 = copy.deepcopy(man)
    man1["extraction"]["detection"] = None
    man1["parameters"]["bounds"] = None
    man1["architecture"]["hidden_layers"] = 2
    assert S.validate_manifest(man1) == [], S.validate_manifest(man1)
    # the draft: valid as a draft, TODO fields listed exactly, not complete
    draft = _draft_card(man)
    assert S.validate_card_header(draft, man) == [], S.validate_card_header(draft, man)
    want = sorted(["title", "scientific_question", "references", "authors", "licence", "cite",
                   "data", "limitations", "simulator.model_type", "simulator.reference",
                   "simulator.fixed_parameters", "training.hyperparameter_search"])
    assert S.card_todo_fields(draft) == want, S.card_todo_fields(draft)
    assert len(S.validate_card_header(draft, man, complete=True)) == len(want)
    # the table of single faults: exactly one problem each, naming the key
    table = _fault_table()
    for label, target, mutate, key in table:
        m, c = copy.deepcopy(man), copy.deepcopy(card)
        mutate(m, c)
        if target == "manifest":
            probs = S.validate_manifest(m)
        elif target == "extraction":
            probs = S.validate_extraction(m["extraction"])
            _one(S.validate_manifest(m), key, label + " (in the manifest)")
        else:
            probs = S.validate_card_header(c, man, complete=True)
        _one(probs, key, label)
    # non-objects at the top: one problem, no exception
    for v in (None, [], "x", 3.5, True, 10 ** 400, float("nan")):
        for f in (S.validate_manifest, S.validate_extraction, S.validate_card_header):
            assert len(f(v)) == 1, (f.__name__, v, f(v))
        assert S.card_todo_fields(v) == []
    # never an exception: every value of the valid objects replaced by each odd value
    n_calls = 0
    for base, calls in ((man, (S.validate_manifest,)),
                        (man["extraction"], (S.validate_extraction,)),
                        (card, (lambda d: S.validate_card_header(d, man, True),
                                lambda d: S.validate_card_header(d, None, False),
                                S.card_todo_fields))):
        for path in _paths(base):
            for odd in _ODD:
                d = copy.deepcopy(base)
                _set(d, list(path), copy.deepcopy(odd))
                for f in calls:
                    out = f(d)
                    n_calls += 1
                    assert isinstance(out, list) and all(isinstance(s, str) for s in out), \
                        (path, odd, out)
    # and random nested values, from the file's seed, as whole objects and as the manifest
    # given to the card's check
    rng = random.Random(SEED)
    for _ in range(300):
        v = _random_json(rng, 4)
        for out in (S.validate_manifest(v), S.validate_extraction(v), S.validate_card_header(v),
                    S.validate_card_header(card, v, True), S.card_todo_fields(v)):
            n_calls += 1
            assert isinstance(out, list) and all(isinstance(s, str) for s in out)
    # a structure deeper than the interpreter's recursion limit
    deep = []
    for _ in range(sys.getrecursionlimit() + 100):
        deep = [deep]
    c = copy.deepcopy(card)
    c["training"]["hyperparameters"] = {"deep": deep}
    out = S.validate_card_header(c, man, True) + S.card_todo_fields(c)
    m = copy.deepcopy(man)
    m["provenance"]["sources"] = {"deep": deep}
    out += S.validate_manifest(m)
    assert all(isinstance(s, str) for s in out)
    # a value that contains itself (a YAML alias: safe_load builds one from "&a [*a]")
    loop = []
    loop.append(loop)
    c = copy.deepcopy(card)
    c["references"] = loop
    c["training"]["hyperparameters"] = {"self": loop}
    out = S.validate_card_header(c, man, True) + S.card_todo_fields(c)
    assert all(isinstance(s, str) for s in out)
    assert n_calls > 5000, n_calls


# ------------------------------------------------------------ T40b

def _rec(**kw):
    base = dict(name="r", spikes={0: np.array([1, 5, 9], dtype=np.int64),
                                  3: np.array([0, 99], dtype=np.int64)},
                fs_raw_hz=1000.0, n_samples=100,
                positions_um={0: (0.0, 0.0), 3: (180.0, 0.0)}, grid=(64, 0, 60.0))
    base.update(kw)
    return Recording(**base)


def test_t40b_recording():
    """Recording.check and duration_s (specs/SPEC.md block 40)."""
    assert _rec().check() == [], _rec().check()
    assert _rec(positions_um=None, grid=None).check() == []
    i64 = lambda *v: np.array(v, dtype=np.int64)  # noqa: E731
    faults = [
        ("unsorted", dict(spikes={0: i64(5, 1, 9), 3: i64(0, 99)}), "spikes[0]"),
        ("repeated", dict(spikes={0: i64(1, 5, 5), 3: i64(0, 99)}), "spikes[0]"),
        ("negative index", dict(spikes={0: i64(-1, 5), 3: i64(0, 99)}), "spikes[0]"),
        ("index = n_samples", dict(spikes={0: i64(1, 5), 3: i64(0, 100)}), "spikes[3]"),
        ("rate zero", dict(fs_raw_hz=0.0), "fs_raw_hz"),
        ("rate negative", dict(fs_raw_hz=-1000.0), "fs_raw_hz"),
        ("rate NaN", dict(fs_raw_hz=float("nan")), "fs_raw_hz"),
        ("positions of another electrode", dict(positions_um={0: (0.0, 0.0), 4: (1.0, 0.0)}),
         "positions_um"),
        ("positions of fewer electrodes", dict(positions_um={0: (0.0, 0.0)}), "positions_um"),
        ("float train", dict(spikes={0: np.array([1.0, 5.0]), 3: i64(0, 99)}), "spikes[0]"),
    ]
    for label, kw, key in faults:
        _one(_rec(**kw).check(), key, label)
    # duration_s = n_samples / fs_raw_hz: the same float division, so equal exactly
    for n, fs in ((100, 1000.0), (1234567, 20000.0), (7, 3.0)):
        assert _rec(n_samples=n, fs_raw_hz=fs, spikes={0: i64(1, 5), 3: i64(0, 6)}).duration_s \
            == n / fs


# ------------------------------------------------------------ T40c

_SEC27_MODULES = {   # sec. 2.7, "The sources the package imports": module -> its file
    "joint_model": "hpc/joint/stage2/joint_model.py",
    "run_joint_arms": "hpc/joint/stage3/run_joint_arms.py",
    "backbone": "hpc/dsn/backbone.py",
    "condition_space": "hpc/dsn/condition_space.py",
    "generate_burst_data": "hpc/dsn/generate_burst_data.py",
    "ev_sample": "hpc/pipeline/evaluation/ev_sample.py",
    "ev_gate": "hpc/pipeline/evaluation/ev_gate.py",
    "ev_rec_metrics": "hpc/pipeline/evaluation/ev_rec_metrics.py",
    "ev_models": "hpc/pipeline/evaluation/ev_models.py",
}
_SEC27_EXTRACTOR = {"channel_subset_extraction": "extractor/channel_subset_extraction.py",
                    "bank_real": "export_windows/bank_real.py"}


def _file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_t40c_sources():
    """the source groups: the modules imported, their digests; SourceError (specs/SPEC.md
    block 40)."""
    saved_env = dict((v, os.environ.get(v)) for v in _sources.EXTRACTOR_ENV + ("SBI_HPC_DIR",))
    saved_path = list(sys.path)
    try:
        try:
            _sources.use("model", "nonsense")
            raise AssertionError("an unknown group did not raise SourceError")
        except _sources.SourceError as exc:
            assert "nonsense" in str(exc)
        got = _sources.use("ifr", "evaluation", "model")       # any order of the arguments
        want = sorted(_SEC27_MODULES.values())
        assert sorted(got) == want, sorted(got)
        # the folders on sys.path, in GROUPS' order, each once
        folders = [os.path.join(_REPO, d) for g in ("model", "evaluation")
                   for d in _sources.GROUPS[g][1]]
        idx = [sys.path.index(os.path.normpath(f)) for f in folders]
        assert idx == sorted(idx), idx
        assert all(sys.path.count(os.path.normpath(f)) == 1 for f in folders)
        _sources.use("model")                                   # again: nothing added
        assert all(sys.path.count(os.path.normpath(f)) == 1 for f in folders)
        # the digests, recomputed here; the modules imported are those very files
        for mod, rel in sorted(_SEC27_MODULES.items()):
            path = os.path.join(_REPO, rel)
            assert got[rel]["path"] == path, (rel, got[rel]["path"])
            assert got[rel]["sha256"] == _file_sha256(path), rel
            m = importlib.import_module(mod)
            assert os.path.abspath(m.__file__) == path, (mod, m.__file__)
        for name in ("build_joint_model", "state_sha256"):
            assert callable(getattr(sys.modules["joint_model"], name))
        for name in ("make_backbone", "FixedStatsSummary"):
            assert hasattr(sys.modules["run_joint_arms"], name)
        assert callable(sys.modules["generate_burst_data"].compute_ifr_trace)
        # the extractor's group without its folder: SourceError naming SBIX_DIR
        clone = saved_env.get("SBIX_CLONE") or saved_env.get("SBIX_DIR")
        for v in _sources.EXTRACTOR_ENV:
            os.environ.pop(v, None)
        for call in (_sources.extractor_dir, lambda: _sources.use("extractor")):
            try:
                call()
                raise AssertionError("no extractor folder and no SourceError")
            except _sources.SourceError as exc:
                assert "SBIX_DIR" in str(exc), str(exc)
        os.environ["SBIX_DIR"] = os.path.join(_tmpdir("nosbix"), "absent")
        try:
            _sources.use("extractor")
            raise AssertionError("an absent SBIX_DIR folder and no SourceError")
        except _sources.SourceError as exc:
            assert "SBIX_DIR" in str(exc), str(exc)
        # with the checkout (SBIX_CLONE): its two files, their digests, the modules imported
        if clone and os.path.isdir(clone):
            os.environ.pop("SBIX_DIR")
            os.environ["SBIX_CLONE"] = clone
            got = _sources.use("extractor")
            assert sorted(got) == sorted(_SEC27_EXTRACTOR.values()), sorted(got)
            # the extractor's modules put <SBI_HPC_DIR>/dsn on sys.path at import (its
            # dsn_tree); here that is this repository's hpc/, when the caller has not set it
            os.environ.setdefault("SBI_HPC_DIR", os.path.join(_REPO, "hpc"))
            for mod, rel in sorted(_SEC27_EXTRACTOR.items()):
                path = os.path.join(os.path.abspath(clone), rel)
                assert got[rel]["sha256"] == _file_sha256(path), rel
                m = importlib.import_module(mod)
                assert os.path.abspath(m.__file__) == path, (mod, m.__file__)
            assert callable(sys.modules["bank_real"].windows_of)
            assert callable(sys.modules["channel_subset_extraction"].partition_subregions)
    finally:
        for v, val in saved_env.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
        sys.path[:] = saved_path


# ------------------------------------------------------------ T40d

# block 40's modules and the libraries outside the standard library each may
# import (specs/SPEC.md block 40, "Modules": standard library, plus numpy for
# Recording)
_BLOCK40_MODULES = {"__init__.py": (), "__main__.py": (), "_sources.py": (), "schema.py": (),
                    "interfaces.py": ("numpy",)}


def _imports(path):
    tree = ast.parse(open(path, "rb").read().decode("ascii"), filename=path)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def test_t40d_cli_static():
    """python3 -m sbitool without a command; the package's imports (specs/SPEC.md block 40)."""
    from sbitool import __main__ as CLI
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for argv, code in (([], CLI.EXIT_ARGS), (["no-such-command"], CLI.EXIT_ARGS)):
        r = subprocess.run([sys.executable, "-m", "sbitool"] + argv, cwd=_REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == code, (argv, r.returncode, r.stdout, r.stderr)
        for name in CLI.COMMANDS:
            assert name in r.stderr, (argv, name, r.stderr)
    # a command whose module is not written: exit 3 and a line saying so; one whose function
    # is absent: exit 3; a written command: its main's code
    saved = dict(CLI.COMMANDS)
    try:
        CLI.COMMANDS["fake"] = "sbitool._no_such_module_t40d:main"
        CLI.COMMANDS["nofunc"] = "sbitool.schema:no_such_main"
        CLI.COMMANDS["echo"] = "sbitool.tests.test_smoke_tool_schema:_fake_main"
        assert CLI.main(["fake"]) == CLI.EXIT_REFUSE
        assert CLI.main(["nofunc"]) == CLI.EXIT_REFUSE
        assert CLI.main(["echo", "a", "b"]) == 7
        seen = sys.modules["sbitool.tests.test_smoke_tool_schema"]._FAKE_ARGV
        assert seen == [["a", "b"]], seen
    finally:
        CLI.COMMANDS.clear()
        CLI.COMMANDS.update(saved)
    # static: block 40's modules import the standard library, numpy where named, and sbitool
    pkg = os.path.join(_REPO, "sbitool")
    for fname, extra in sorted(_BLOCK40_MODULES.items()):
        bad = _imports(os.path.join(pkg, fname)) - set(sys.stdlib_module_names) - set(extra) \
            - {"sbitool"}
        assert not bad, (fname, sorted(bad))


_FAKE_ARGV = []


def _fake_main(argv):
    _FAKE_ARGV.append(list(argv))
    return 7


CHECKS = [("test_t40a_validators", test_t40a_validators),
          ("test_t40b_recording", test_t40b_recording),
          ("test_t40c_sources", test_t40c_sources),
          ("test_t40d_cli_static", test_t40d_cli_static)]


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
