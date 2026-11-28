"""Spec tests of block 40 (the tool's package, its schemas and its
interfaces; specs/SPEC.md sec. 2.7 and block 40). Integration tester,
2026-11-28, incremental run on 39e6f66.

Every expected value comes from the spec's text: the key lists are written
here from sec. 2.7, the valid manifest and card are built here from sec. 2.7,
and digests are recomputed with hashlib. Needs nothing but the standard
library and numpy (SBIX_CLONE for the extractor group).
"""

import ast
import copy
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.dont_write_bytecode = True

from sbitool import schema as S  # noqa: E402
from sbitool.interfaces import Recording  # noqa: E402

SBIX = os.environ.get("SBIX_CLONE", "/home/claude/sbix")
H = "a" * 64            # a well-formed sha256 hex digest


# ------------------------------------------------------------ the key lists, from sec. 2.7

def test_key_lists_are_the_spec_s():
    assert S.BUNDLE_SCHEMA == "sbitool_bundle/1"
    assert tuple(S.MANIFEST_KEYS) == (
        "schema", "name", "created", "card_complete", "arm", "M", "p", "d_z", "W", "members",
        "architecture", "parameters", "extraction", "gate", "sample", "provenance", "software",
        "files")
    assert set(S.MEMBER_KEYS) == {"file", "sha256", "seed", "state_sha256", "split_seed",
                                  "split_hash"}
    assert set(S.ARCHITECTURE_KEYS) == {
        "backbone", "embedding_size", "depth_exponent", "width_multiplier", "block_family",
        "head_fusion", "dropout", "head_pool_ops", "hidden_features", "num_transforms",
        "num_bins", "hidden_layers", "z_score_theta", "z_score_x", "backbone_repr"}
    assert set(S.PARAMETER_KEYS) == {"names", "units", "bounds"}
    assert set(S.EXTRACTION_KEYS) == {
        "electrodes_per_subset", "n_subsets", "mfr_threshold_hz", "w_size_s",
        "gaussian_window_s", "grid_width", "index_base", "pitch_um", "fs_raw_hz", "detection"}
    assert set(S.DETECTION_KEYS) == {"method", "peak_sign", "detect_threshold",
                                     "exclude_sweep_ms"}
    assert set(S.GATE_KEYS) == {"factors", "alpha", "n_ref", "n_rank", "n_null", "k_gate",
                                "size_mean", "size_lo", "size_hi", "level"}
    # "the user's eight (D-158) and the sharing fields (D-175)", in sec. 2.7's order
    assert tuple(S.CARD_KEYS) == (
        "title", "scientific_question", "simulator", "training", "extraction", "references",
        "authors", "licence", "cite", "data", "software", "limitations")
    assert set(S.CARD_SIMULATOR_KEYS) == {"model_type", "reference", "fixed_parameters",
                                          "searched_parameters"}
    assert set(S.CARD_TRAINING_KEYS) == {"arm", "members", "n_networks", "n_windows",
                                         "split_fracs", "hyperparameters",
                                         "hyperparameter_search"}
    assert set(S.SOFTWARE_KEYS) == {"python", "torch", "sbi", "zuko", "numpy", "safetensors"}


# ------------------------------------------------------------ a valid manifest and card

def extraction():
    return {"electrodes_per_subset": 9, "n_subsets": 2, "mfr_threshold_hz": 0.1,
            "w_size_s": 0.01, "gaussian_window_s": 0.02, "grid_width": 6, "index_base": 0,
            "pitch_um": 60.0, "fs_raw_hz": 10000.0, "detection": None}


def manifest(M=2, p=3):
    return {
        "schema": "sbitool_bundle/1", "name": "toy", "created": "2026-11-28T00:00:00+00:00",
        "card_complete": False, "arm": "A1", "M": M, "p": p, "d_z": 4, "W": 50,
        "members": [{"file": "members/member_%d.safetensors" % k, "sha256": H, "seed": 1 + k,
                     "state_sha256": H, "split_seed": 1, "split_hash": H} for k in range(M)],
        "architecture": {"backbone": "fixed-stats", "embedding_size": 8, "depth_exponent": 0,
                         "width_multiplier": 1.0, "block_family": 0, "head_fusion": 0,
                         "dropout": 0.0, "head_pool_ops": 0, "hidden_features": 16,
                         "num_transforms": 2, "num_bins": 4, "hidden_layers": None,
                         "z_score_theta": "transform_to_unconstrained", "z_score_x": "none",
                         "backbone_repr": "FixedStatsSummary()"},
        "parameters": {"names": ["a%d" % k for k in range(p)], "units": ["1"] * p,
                       "bounds": [[0.0, 1.0]] * p},
        "extraction": extraction(),
        "gate": {"factors": [0.25, 0.5, 1, 2, 4], "alpha": 0.05, "n_ref": 10, "n_rank": 10,
                 "n_null": 99, "k_gate": 4, "size_mean": 0.04, "size_lo": 0.01,
                 "size_hi": 0.09, "level": 0.95},
        "sample": {"n_post": 40, "seed": 7, "batch_rows": 16},
        "provenance": {"project": "toy", "ensemble": "S-A1:A1", "search": "S-A1",
                       "campaign": "S-A1", "best_fit_complete_sha256": H,
                       "sample_json_sha256": H, "commit": "abc", "sources": {}},
        "software": {"python": "3.13", "torch": "2", "sbi": "0.27", "zuko": "1.6",
                     "numpy": "2.5", "safetensors": "0.8"},
        "files": {"CARD.md": H, "evaluation/sim.json": H, "evaluation/gate.safetensors": H},
    }


def card(man):
    return {
        "title": "t", "scientific_question": "q",
        "simulator": {"model_type": "m", "reference": "r",
                      "fixed_parameters": {"rho": "1300 cells/mm2"},
                      "searched_parameters": list(man["parameters"]["names"])},
        "training": {"arm": man["arm"], "members": man["M"], "n_networks": 3, "n_windows": 6,
                     "split_fracs": [0.8, 0.1, 0.1], "hyperparameters": {"lr": 0.001},
                     "hyperparameter_search": "GP"},
        "extraction": copy.deepcopy(man["extraction"]),
        "references": ["ref"], "authors": [{"name": "n", "contact": "c"}], "licence": "MIT",
        "cite": "c", "data": {"simulated_bank": {"location": "l", "public": False,
                                                  "size_bytes": None, "sha256": None},
                               "recordings": {"location": "l", "public": False}},
        "software": dict(man["software"]), "limitations": "l",
    }


def test_t40a_valid_gives_no_problem():
    m = manifest()
    assert S.validate_manifest(m) == []
    assert S.validate_extraction(m["extraction"]) == []
    assert S.validate_card_header(card(m), m) == []
    assert S.validate_card_header(card(m), m, complete=True) == []
    # the extraction null is valid (a bundle that accepts windows only), and the card then
    # says "not applicable"
    m2 = manifest()
    m2["extraction"] = None
    c2 = card(m2)
    c2["extraction"] = "not applicable"
    assert S.validate_manifest(m2) == []
    assert S.validate_card_header(c2, m2, complete=True) == []
    # a detection of exactly DETECTION_KEYS (D-176's values) is valid
    e = extraction()
    e["detection"] = {"method": "by_channel", "peak_sign": "neg", "detect_threshold": 5,
                      "exclude_sweep_ms": 1.0}
    assert S.validate_extraction(e) == []


def _set(d, path, value):
    for k in path[:-1]:
        d = d[k]
    if value is _DEL:
        del d[path[-1]]
    else:
        d[path[-1]] = value


_DEL = object()

# (path, value, a word the one problem must contain): single faults of sec. 2.7
MANIFEST_FAULTS = [
    (("arm",), _DEL, "arm"),
    (("foo",), 1, "foo"),
    (("M",), "2", "M"),
    (("M",), 3, "M"),                                  # M not the members' number
    (("parameters", "bounds", 1), [1.0, 1.0], "bounds[1]"),   # lo >= hi
    (("parameters", "bounds", 0), [0.0, float("nan")], "bounds[0]"),
    (("parameters", "bounds", 2), [0, 10 ** 400], "bounds[2]"),  # an int beyond float range
    (("gate", "alpha"), float("nan"), "alpha"),
    (("gate", "alpha"), float("inf"), "alpha"),
    (("members", 0, "sha256"), "A" * 64, "sha256"),    # the digest's form
    (("members", 1, "seed"), 0, "seed"),               # seeds ascending
    (("members", 1, "file"), "members/member_0.safetensors", "file"),
    (("schema",), "sbitool_bundle/2", "schema"),
    (("created",), "yesterday", "created"),
    (("card_complete",), 0, "card_complete"),
    (("extraction", "detection"), {"method": "by_channel", "peak_sign": "pos",
                                   "detect_threshold": 5, "exclude_sweep_ms": 1.0},
     "peak_sign"),
    (("extraction", "index_base"), 2, "index_base"),
    (("extraction", "w_size_s"), 0.0, "w_size_s"),
    (("architecture", "backbone"), "cnn", "backbone"),
    (("sample", "n_post"), 0, "n_post"),
    (("files",), [], "files"),
]


@pytest.mark.parametrize("path,value,word", MANIFEST_FAULTS,
                         ids=[".".join(map(str, f[0])) + "=" + repr(f[1])[:20]
                              for f in MANIFEST_FAULTS])
def test_t40a_manifest_single_fault(path, value, word):
    m = manifest()
    _set(m, path, value)
    probs = S.validate_manifest(m)
    assert len(probs) == 1 and word in probs[0], probs


def test_t40a_p_disagrees_with_the_parameters():
    # p is one fault that three cross-conditions see: names, units and bounds each of p
    m = manifest()
    m["p"] = 2
    probs = S.validate_manifest(m)
    assert len(probs) == 3 and all("p = 2" in q for q in probs), probs


CARD_FAULTS = [
    (("title",), "TODO", "title"),
    (("training", "arm"), "A5", "arm"),
    (("simulator", "searched_parameters"), ["a0", "a1", "zz"], "searched_parameters"),
    (("training", "members"), 3, "members"),
    (("licence",), _DEL, "licence"),
    (("colour",), "red", "colour"),
    (("references",), [], "references"),
    (("training", "hyperparameters", "lr"), "TODO", "lr"),
]


@pytest.mark.parametrize("path,value,word", CARD_FAULTS,
                         ids=[".".join(map(str, f[0])) for f in CARD_FAULTS])
def test_t40a_card_single_fault(path, value, word):
    m = manifest()
    c = card(m)
    _set(c, path, value)
    probs = S.validate_card_header(c, m, complete=True)
    assert len(probs) == 1 and word in probs[0], probs


ODD = [None, True, 0, -1, 10 ** 400, 1e308, float("nan"), float("inf"), "", "x", [], {},
       [[[[[]]]]], {"a": {"b": {"c": None}}}, {1: 2}, [None] * 3, "TODO"]


def _every_path(d, path=()):
    yield path
    if isinstance(d, dict):
        for k, v in d.items():
            for p in _every_path(v, path + (k,)):
                yield p
    elif isinstance(d, list):
        for i, v in enumerate(d):
            for p in _every_path(v, path + (i,)):
                yield p


def test_t40a_never_raises_on_any_json_value():
    # every odd value at every path of a valid manifest and card: a list back, never raises
    base_m = manifest()
    n = 0
    for path in list(_every_path(base_m))[1:]:
        for v in ODD:
            m = manifest()
            _set(m, path, v)
            out = S.validate_manifest(m)
            assert isinstance(out, list)
            n += 1
    for path in list(_every_path(card(base_m)))[1:]:
        for v in ODD:
            c = card(manifest())
            _set(c, path, v)
            assert isinstance(S.validate_card_header(c, manifest(), complete=True), list)
            n += 1
    for v in ODD:
        assert isinstance(S.validate_manifest(v), list)
        assert isinstance(S.validate_extraction(v), list)
        assert isinstance(S.validate_card_header(v, v), list)
    # a structure deeper than the recursion limit
    deep = []
    for _ in range(5000):
        deep = [deep]
    m = manifest()
    m["provenance"]["sources"] = {"x": deep}
    assert isinstance(S.validate_manifest(m), list)
    assert n > 1000


# ------------------------------------------------------------ T40b: Recording.check

def _rec(**kw):
    d = dict(name="r", spikes={0: np.array([1, 5, 9], dtype=np.int64),
                               1: np.array([], dtype=np.int64)},
             fs_raw_hz=10000.0, n_samples=10, positions_um=None, grid=None)
    d.update(kw)
    return Recording(**d)


def test_t40b_recording_check():
    assert _rec().check() == []
    # duration_s = n_samples / fs_raw_hz [s]
    assert _rec(n_samples=12345, fs_raw_hz=10110.09).duration_s == 12345 / 10110.09
    cases = {
        "unsorted": dict(spikes={0: np.array([5, 1], dtype=np.int64)}),
        "duplicate": dict(spikes={0: np.array([1, 1], dtype=np.int64)}),
        "index n_samples (open end)": dict(spikes={0: np.array([1, 10], dtype=np.int64)}),
        "negative": dict(spikes={0: np.array([-1, 3], dtype=np.int64)}),
        "rate 0": dict(fs_raw_hz=0.0),
        "rate < 0": dict(fs_raw_hz=-5.0),
        "rate nan": dict(fs_raw_hz=float("nan")),
        "rate inf": dict(fs_raw_hz=float("inf")),
        "positions of other electrodes": dict(positions_um={0: (0.0, 0.0), 7: (1.0, 0.0)}),
        "float spikes": dict(spikes={0: np.array([1.0, 2.0])}),
    }
    for what, kw in cases.items():
        probs = _rec(**kw).check()
        assert len(probs) >= 1, what
    # the last legal index n_samples - 1 is accepted, index 0 too
    assert _rec(spikes={0: np.array([0, 9], dtype=np.int64)}).check() == []


# ------------------------------------------------------------ T40c: _sources

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_t40c_sources_digests():
    from sbitool import _sources
    files = _sources.use("model", "evaluation", "ifr")
    must = ["hpc/joint/stage2/joint_model.py", "hpc/dsn/backbone.py",
            "hpc/dsn/condition_space.py", "hpc/joint/stage3/run_joint_arms.py",
            "hpc/dsn/generate_burst_data.py", "hpc/pipeline/evaluation/ev_sample.py",
            "hpc/pipeline/evaluation/ev_gate.py", "hpc/pipeline/evaluation/ev_rec_metrics.py"]
    for rel in must:
        assert rel in files, rel
        assert files[rel]["sha256"] == _sha(os.path.join(REPO, rel)), rel
        assert os.path.samefile(files[rel]["path"], os.path.join(REPO, rel))


def _py(code, env):
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def test_t40c_extractor_absent_and_sbi_hpc_dir():
    env = dict(os.environ)
    for k in ("SBIX_DIR", "SBIX_CLONE", "SBI_HPC_DIR"):
        env.pop(k, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    rc, out = _py("from sbitool import _sources\n"
                  "try:\n _sources.use('extractor')\nexcept _sources.SourceError as e:\n"
                  " print('SE', e)\n", env)
    assert rc == 0 and "SE" in out and "SBIX_DIR" in out, out
    if not os.path.isdir(SBIX):
        pytest.skip("no extractor checkout")
    env["SBIX_CLONE"] = SBIX
    rc, out = _py("import os\nfrom sbitool import _sources\nf=_sources.use('extractor')\n"
                  "print('HPC', os.environ['SBI_HPC_DIR'])\nprint(sorted(f))\n", env)
    assert rc == 0 and ("HPC %s" % os.path.join(REPO, "hpc")) in out, out
    env["SBI_HPC_DIR"] = "/elsewhere"
    rc, out = _py("import os\nfrom sbitool import _sources\n_sources.use('extractor')\n"
                  "print('HPC', os.environ['SBI_HPC_DIR'])\n", env)
    assert rc == 0 and "HPC /elsewhere" in out, out     # "when it is unset" only


# ------------------------------------------------------------ T40d: the command line

COMMANDS = ("export", "bundle-check", "card-template", "card-check", "windows", "infer",
            "device-figure")


def test_t40d_command_line():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "sbitool"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2
    for c in COMMANDS:
        assert c in r.stdout + r.stderr, c
    r = subprocess.run([sys.executable, "-m", "sbitool", "frobnicate"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run([sys.executable, "-m", "sbitool", "-h"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0 and all(c in r.stdout for c in COMMANDS)
    # every command: an argument error exits 2 (block 40: "2 on the arguments")
    for c in COMMANDS:
        r = subprocess.run([sys.executable, "-m", "sbitool", c, "--no-such-flag"], cwd=REPO,
                           env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 2, (c, r.returncode, r.stdout[-300:], r.stderr[-300:])


# the libraries each module's block names (block 40: stdlib and numpy; 41: torch,
# safetensors, sbi through the builders; 42: PyYAML; 43: numpy, scipy, spikeinterface,
# probeinterface; 44, 45: numpy; 46: matplotlib, numpy)
ALLOWED = {
    "__init__.py": set(), "__main__.py": set(), "_sources.py": set(), "schema.py": set(),
    "interfaces.py": {"numpy"},
    "bundle.py": {"safetensors", "torch", "numpy"}, "model.py": {"torch", "sbi"},
    "export.py": {"torch", "safetensors", "sbi", "zuko", "numpy"},
    "card.py": {"yaml"},
    "recordings/__init__.py": set(), "recordings/ptrain.py": {"numpy"},
    "recordings/si.py": {"numpy", "spikeinterface", "probeinterface"},
    "recordings/layout.py": {"numpy", "scipy"},
    "windows.py": {"numpy"},
    "infer.py": {"numpy", "torch", "safetensors", "sbi", "zuko"},
    "plotting/__init__.py": set(), "plotting/device.py": {"matplotlib", "numpy"},
}
# the sources' modules, reached through _sources (sec. 2.7)
SOURCES = {"_evalpath", "config", "ev_models", "ev_plan", "ev_rec_config", "ev_rec_data",
           "ev_rec_metrics", "ev_rec_plan", "ev_gate", "ev_sample", "joint_model",
           "run_joint_arms", "channel_subset_extraction", "bank_real", "ev_figures"}


def test_t40d_static_imports():
    pkg = os.path.join(REPO, "sbitool")
    std = set(sys.stdlib_module_names)
    bad = []
    for rel, allowed in ALLOWED.items():
        tree = ast.parse(open(os.path.join(pkg, rel)).read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            for nm in names:
                top = nm.split(".")[0]
                if top in std or top == "sbitool" or top in SOURCES or top in allowed:
                    continue
                bad.append((rel, nm))
    assert not bad, bad
