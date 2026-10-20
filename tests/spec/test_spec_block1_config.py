"""Spec-derived tests of the configuration schema (SPEC sec. 3.1; block 1).

Oracle O1.1 / O1.2 of the 2026-10-20 report: every key required, unknown /
missing / wrong-typed keys each a named problem, every problem listed at
once, `_` keys ignored, the D-099 lower bound on resources.max_cores.
"""
import copy
import json
import os

import pytest

from conftest import GIULIA

import config as C


def _raw():
    with open(GIULIA) as fh:
        return json.load(fh)


def _write(tmp_path, data, name="cfg.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data, indent=1))
    return str(p)


def _problems(tmp_path, data):
    try:
        C.load_config(_write(tmp_path, data))
    except C.ConfigError as exc:
        return exc.problems
    return []


def _set(d, dotted, value):
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def _del(d, dotted):
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    del cur[parts[-1]]


def test_committed_giulia_loads():
    cfg = C.load_config(GIULIA)
    assert cfg.tag == "giulia"
    assert cfg.get("config_version") == 10          # sec. 3.1: v10 (D-101)
    assert C.unhonoured_keys(cfg) == []             # block 7: NEEDS_CODE empty


def test_committed_values_of_the_decisions():
    cfg = C.load_config(GIULIA)
    g = cfg.get
    assert g("search.campaign") == "S-A1" and g("search.arm") == "A1"          # D-039
    assert g("split.fractions") == [0.85, 0.10, 0.05]                          # D-088
    assert g("search.max_total_walltime_h") == 400                             # D-090
    assert g("search.walltime_reading") == "elapsed"                           # D-090
    assert len(g("best_fit.member_seeds")) == 10                               # D-089
    assert g("best_fit.patience_epochs") == 10                                 # sec. 2.3
    assert g("search.space_overrides")["head_pool_ops"] == [0, 1]              # D-086
    assert g("search.gp_seed") == 1 and g("resources.max_cores") == 400        # D-101 (v10)
    assert g("driver.ncpus") == 4 and g("driver.walltime") == "24:00:00"       # v8 driver block


def test_sha256_of_bytes():
    import hashlib
    cfg = C.load_config(GIULIA)
    assert cfg.sha256 == hashlib.sha256(open(GIULIA, "rb").read()).hexdigest()


@pytest.mark.parametrize("key", [k for k, _ in C.SCHEMA][::7])
def test_missing_key_named(tmp_path, key):
    d = _raw()
    _del(d, key)
    probs = _problems(tmp_path, d)
    assert any(p.startswith(key + ":") and "missing" in p for p in probs), probs


def test_unknown_key_named(tmp_path):
    d = _raw()
    d["search"]["trial"]["bogus_knob"] = 3
    probs = _problems(tmp_path, d)
    assert probs == ["search.trial.bogus_knob: unknown key (not in the schema)"], probs


def test_unknown_top_level_key(tmp_path):
    d = _raw()
    d["extra_block"] = {"a": 1}
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith("extra_block.a"), probs


def test_doc_keys_ignored(tmp_path):
    d = _raw()
    d["_note"] = "documentation"
    d["search"]["_trial"] = "documents trial"
    assert _problems(tmp_path, d) == []


@pytest.mark.parametrize("key,value", [
    ("search.trial.epochs", "20"),          # int as str
    ("search.trial.epochs", 2.5),           # float for int
    ("search.trial.epochs", True),          # bool is not an int
    ("search.trial.epochs", 0),             # range >= 1
    ("search.trial.max_resubmits", -1),     # >= 0
    ("search.resubmit_killed", 1),          # int is not bool
    ("search.trial.walltime", "120:00"),    # hh:mm:ss
    ("search.trial.walltime", "1:60:00"),   # minutes < 60
    ("search.trial.mem", "160"),            # PBS memory
    ("bank.walltime", "soon"),              # hh:mm:ss or auto
    ("bank.on_declared_box", [3.0, 0.01]),  # lo < hi
    ("bank.topology_axes", []),             # non-empty
    ("bank.t_win_s", 0.0),                  # > 0
    ("bank.t_win_s", float("nan")),         # NaN is not > 0
    ("paths.work_root", "~/pipeline"),      # absolute
    ("paths.work_root", "pipeline/giulia"), # absolute
    ("freeze.out", "/abs/path.json"),       # relative to sbix_dir
    ("dataset", "Giulia"),                  # [a-z0-9_]+
    ("search.campaign", "S-A3"),            # one of four
    ("search.walltime_reading", "cpu"),     # only elapsed
    ("search.gp_noise_sigma", 0.0),         # > 0 when given
    ("search.max_total_walltime_h", -1),    # > 0 or null
    ("search.budget_gate.safety_factor", 0.9),  # >= 1
    ("best_fit.member_seeds", []),          # non-empty
    ("best_fit.member_seeds", [1, 1]),      # distinct
    ("config_version", 9),                  # == 10
    ("split.fractions", [0.85, 0.10]),      # three
    ("split.fractions", [0.85, 0.10, 0.06]),  # sum 1
    ("split.fractions", [1.0, 0.0, 0.0]),   # each in (0, 1)
    ("freeze.exclude", {"a": 1}),           # str -> str
])
def test_one_named_problem(tmp_path, key, value):
    d = _raw()
    _set(d, key, value)
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith(key + ":"), probs


def test_nullables_accept_null(tmp_path):
    d = _raw()
    d["search"]["gp_noise_sigma"] = None
    d["search"]["max_total_walltime_h"] = None
    assert _problems(tmp_path, d) == []


def test_fraction_tolerance_1e9(tmp_path):
    """sum 1 within 1e-9: 1 + 5e-10 passes, 1 + 5e-9 fails."""
    d = _raw()
    d["split"]["fractions"] = [0.85, 0.10, 0.05 + 5e-10]
    assert _problems(tmp_path, d) == []
    d["split"]["fractions"] = [0.85, 0.10, 0.05 + 5e-9]
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith("split.fractions:")


def test_every_problem_listed(tmp_path):
    d = _raw()
    _del(d, "loader.buffer_shards")
    d["search"]["walltime_reading"] = "cpu"
    d["bank"]["unknown"] = 1
    probs = _problems(tmp_path, d)
    keys = sorted(p.split(":")[0] for p in probs)
    assert keys == ["bank.unknown", "loader.buffer_shards", "search.walltime_reading"], probs


def test_max_cores_lower_bound(tmp_path):
    """v9 (D-099): max_cores >= driver.ncpus + the largest sized job."""
    d = _raw()
    d["resources"]["max_cores"] = 4 + 48 - 1
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith("resources.max_cores:"), probs
    assert "driver.ncpus 4" in probs[0] and "48" in probs[0]
    d["resources"]["max_cores"] = 52
    assert _problems(tmp_path, d) == []


def test_max_cores_counts_bank_verify(tmp_path):
    d = _raw()
    d["bank"]["verify"]["ncpus"] = 100
    d["resources"]["max_cores"] = 103
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and "bank.verify.ncpus 100" in probs[0], probs


def test_not_ascii_refused(tmp_path):
    p = tmp_path / "c.json"
    p.write_bytes(open(GIULIA, "rb").read().replace(b'"giulia"', '"giuli\xc3\xa0"'.encode("latin-1"), 1))
    with pytest.raises(C.ConfigError):
        C.load_config(str(p))


def test_config_version_must_be_int(tmp_path):
    """sec. 3.1: `config_version` is an int. A float 10.0 is a value of the
    wrong type (finding F-config-version if it passes)."""
    d = _raw()
    d["config_version"] = 10.0
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith("config_version:"), probs


def test_space_override_unknown_axis_refused_at_load(tmp_path):
    """sec. 3.1: `search.space_overrides` axes must exist in the joint space
    (block 7); a value of the wrong range is a refusal before anything runs
    (exit 2). An axis the joint space does not have should be named at load."""
    d = _raw()
    d["search"]["space_overrides"]["no_such_axis"] = [1, 2]
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and "no_such_axis" in probs[0], probs
