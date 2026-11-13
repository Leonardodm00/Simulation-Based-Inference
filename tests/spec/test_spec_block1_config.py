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
    assert g("search.gp_seed") == 1                                            # D-101 (v10)
    assert g("resources.max_cores") == 300     # D-115 (2026-10-30): 300, a value under v10 (was 400, D-101)
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


# ---------------------------------------------------------------------------
# t1-A (2026-11-13, D-131's split audit): extensions. Oracles O1.1-O1.6 of
# the t1-A report, from sec. 3.1 and the silent-failure list.
# ---------------------------------------------------------------------------

from conftest import LINDDUP, SYNTH   # noqa: E402

NAN = float("nan")


def test_committed_linddup_loads():
    """sec. 0 / 3.1: linddup.json (config v10, D-104..D-109) loads with no
    problem; the D-099 bound holds there too (LindDUP: 200, D-109)."""
    cfg = C.load_config(LINDDUP)
    assert cfg.get("config_version") == 10
    assert cfg.get("resources.max_cores") == 200
    assert C.unhonoured_keys(cfg) == []


def test_synthetic_file_needs_synthetic_ok():
    """sec. 3.1: load_config(path) refuses a synthetic configuration unless
    called with synthetic_ok=True, as pipeline.py calls it (boundary with
    block 15, from block 1's side)."""
    with pytest.raises(C.ConfigError) as ei:
        C.load_config(SYNTH)
    assert len(ei.value.problems) == 1 and "synthetic" in ei.value.problems[0]
    cfg = C.load_config(SYNTH, synthetic_ok=True)
    assert cfg.get("kind") == "synthetic"


@pytest.mark.parametrize("campaign", ["S-A2-only", "S-A5-only"])
def test_recorded_campaign_set(tmp_path, campaign):
    """sec. 3.1, the recorded data set's table: `search.campaign` in {S-A1,
    S-A2, S-A5, S-A25}. Block 26 adds the term-only campaigns for the
    synthetic searches (synth.json v4) and says 'S-A1, S-A2, S-A5 and S-A25
    are unchanged'; nothing amends the recorded table."""
    d = _raw()
    d["search"]["campaign"] = campaign
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith("search.campaign:"), probs


@pytest.mark.parametrize("key,value", [
    # NaN (json.loads accepts it) must not slip past a comparison-based check
    ("bank.floor_hz", NAN),
    ("bank.floor_rtol", NAN),
    ("search.trial.passes", NAN),
    ("best_fit.passes", NAN),
    ("search.budget_gate.safety_factor", NAN),
    ("search.gp_noise_sigma", NAN),
    ("search.max_total_walltime_h", NAN),
    ("split.fractions", [NAN, 0.10, 0.05]),
    ("bank.on_declared_box", [NAN, 3.0]),
    # a JSON bool where sec. 3.1 says int (bool is a subclass of int in Python)
    ("driver.ncpus", True),
    ("search.gp_seed", False),
    ("best_fit.member_seeds", [True, 2]),
    ("search.trial.max_resubmits", False),
    # an integral float where sec. 3.1 says int
    ("sim.concurrency", 1.0),
    ("loader.buffer_shards", 4.0),
    # PBS memory and walltime forms
    ("driver.mem", "16 gb"),
    ("best_fit.walltime", "300:00"),
    ("bank.verify.walltime", "auto"),          # only bank.walltime takes auto
    # strings that must be non-empty / absolute / a known arm
    ("envs.train", ""),
    ("search.arm", "A9"),
    ("real.cohort_config", "relative/cohort.json"),
])
def test_one_named_problem_t1a(tmp_path, key, value):
    d = _raw()
    _set(d, key, value)
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith(key + ":"), probs


def test_resources_true_is_one_problem(tmp_path):
    """resources.max_cores true: one problem at the key, and the D-099
    bound is not evaluated on a value that failed its own check."""
    d = _raw()
    d["resources"]["max_cores"] = True
    probs = _problems(tmp_path, d)
    assert len(probs) == 1 and probs[0].startswith("resources.max_cores:"), probs


def test_kind_other_than_synthetic_refused(tmp_path):
    """sec. 3.1: `kind` is `synthetic` or absent; any other value is refused."""
    d = _raw()
    d["kind"] = "recorded"
    probs = _problems(tmp_path, d)
    assert probs and all(p.startswith("kind:") for p in probs), probs


def test_max_cores_message_names_largest_job(tmp_path):
    """Block 1 (d): `resources.max_cores: <n> cannot hold the sequencer's job
    (driver.ncpus <d>) and the largest job (<key> <v>)`: here sim.ncpus is
    raised to be the largest."""
    d = _raw()
    d["sim"]["ncpus"] = 100
    d["resources"]["max_cores"] = 103
    probs = _problems(tmp_path, d)
    assert probs == ["resources.max_cores: 103 cannot hold the sequencer's job (driver.ncpus 4) "
                     "and the largest job (sim.ncpus 100)"], probs


def test_documentation_keys_inside_nested_objects_ignored(tmp_path):
    """sec. 3.1: a key starting with `_` documents the key after it and is
    ignored -- at any depth, in the schema's leaf objects too."""
    d = _raw()
    d["search"]["trial"]["_epochs"] = "doc"
    d["bank"]["verify"]["_x"] = {"nested": 1}
    d["search"]["space_overrides"]["_head_pool_ops"] = "doc"
    assert _problems(tmp_path, d) == []


def test_categorical_axis_pinned_by_one_level(tmp_path):
    """sec. 3.1: `search.space_overrides` axis -> [lo, hi] (numeric) or
    [levels] (categorical); block 7: apply_overrides gives 'a range per axis
    from the configuration, an axis freed by two levels or pinned by one'.
    A categorical axis pinned by one level ([1]) is a valid configuration."""
    d = _raw()
    d["search"]["space_overrides"]["head_pool_ops"] = [1]
    assert _problems(tmp_path, d) == []
