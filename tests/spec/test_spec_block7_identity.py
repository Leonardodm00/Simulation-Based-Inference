"""Spec-derived tests: block 7 amendment (j) (D-134 (1)), specs/SPEC.md.

Oracles (from the spec):
- K_cfg = {split_fracs <- split.fractions, split_seed <- search.train_seed,
  campaign <- search.campaign, shape_anchors, space_overrides, flow}; each
  compared with the configuration's value: numbers as floats, lists element
  by element in order, dicts key by key.
- REFUSE text: "REFUSE: the configuration no longer matches this search's
  identity (<key>: search <old>, configuration <new>; ...); a search is not
  continued under another identity -- restore the value(s), or reset the
  search first".
- at the top of every poll: on a difference the stage fails with that
  text, nothing proposed or submitted, pipeline_rounds.json not rewritten,
  the jobs neither deleted nor forgotten; verify() returns the text.
- a key outside K_cfg (walltime, poll_seconds, max_concurrent_trials,
  resources.*) does not refuse.
- the dry run prints the REFUSE line.
- contract: the K_cfg part of an identity written from each committed
  project file compares equal to that file (no refusal on running chains).

Run: python3 -m pytest -q tests/spec/test_spec_block7_identity.py
"""

import copy
import json
import os
import shutil
import sys
import tempfile

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
TESTS = os.path.join(PIPE, "tests")
for p in (PIPE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PIPELINE_SKIP_CONDA", "1")

import config as C                          # noqa: E402
import stages as S                          # noqa: E402
from stages import search as SE             # noqa: E402
from stages.search import SearchStage       # noqa: E402

PROJ = os.path.join(PIPE, "projects")
EXPECTED_PREFIX = "REFUSE: the configuration no longer matches this search's identity ("
EXPECTED_SUFFIX = ("); a search is not continued under another identity -- restore the value(s), "
                   "or reset the search first")


class Cfg(object):
    def __init__(self, data):
        self.data = data


def base_data():
    return {"split": {"fractions": [0.85, 0.1, 0.05]},
            "search": {"train_seed": 0, "campaign": "S-A1",
                       "shape_anchors": {"p": 10, "d_theta": 10, "embedding_dim": 64},
                       "space_overrides": {"depth_exponent": [3, 5], "head_pool_ops": [0, 1]},
                       "flow": {"conditioner_hidden_layers": 2, "num_bins": 10}}}


def ident_from(data):
    """The K_cfg part of the identity as the spec says it is written:
    split_fracs through the split tool's repr(float(f)) and JSON, the rest
    copied from the configuration."""
    s = data["search"]
    return json.loads(json.dumps({
        "split_fracs": [float(repr(float(f))) for f in data["split"]["fractions"]],
        "split_seed": int(s["train_seed"]), "campaign": s["campaign"],
        "shape_anchors": dict(s["shape_anchors"]), "space_overrides": dict(s["space_overrides"]),
        "flow": dict(s["flow"]), "split_hash": "h", "contract_digest": "c", "n_train": 5}))


CHANGES = {
    "split_fracs": lambda d: d["split"].__setitem__("fractions", [0.8, 0.15, 0.05]),
    "split_seed": lambda d: d["search"].__setitem__("train_seed", 1),
    "campaign": lambda d: d["search"].__setitem__("campaign", "S-A2"),
    "shape_anchors": lambda d: d["search"]["shape_anchors"].__setitem__("embedding_dim", 32),
    "space_overrides": lambda d: d["search"]["space_overrides"].__setitem__("depth_exponent", [3, 4]),
    "flow": lambda d: d["search"]["flow"].__setitem__("conditioner_hidden_layers", 3),
}


def test_none_changed_no_refusal():
    d = base_data()
    assert SE.identity_refusal(ident_from(d), Cfg(d)) is None


@pytest.mark.parametrize("key", sorted(CHANGES))
def test_each_field_alone(key):
    d = base_data()
    ident = ident_from(d)
    d2 = copy.deepcopy(d)
    CHANGES[key](d2)
    text = SE.identity_refusal(ident, Cfg(d2))
    assert text is not None and text.startswith(EXPECTED_PREFIX) and text.endswith(EXPECTED_SUFFIX), text
    body = text[len(EXPECTED_PREFIX):-len(EXPECTED_SUFFIX)]
    parts = body.split("; ")
    assert len(parts) == 1 and parts[0].startswith(key + ": search "), parts
    assert ", configuration " in parts[0]


def test_two_changed_both_named():
    d = base_data()
    ident = ident_from(d)
    d2 = copy.deepcopy(d)
    CHANGES["split_seed"](d2)
    CHANGES["flow"](d2)
    text = SE.identity_refusal(ident, Cfg(d2))
    body = text[len(EXPECTED_PREFIX):-len(EXPECTED_SUFFIX)]
    keys = [p.split(":")[0] for p in body.split("; ")]
    assert sorted(keys) == ["flow", "split_seed"], body


@pytest.mark.parametrize("cfgval,identval,refused", [
    ([0.8500, 0.10, 0.05], [0.85, 0.1, 0.05], False),     # 0.85 written 0.8500
    ([0.10, 0.85, 0.05], [0.85, 0.1, 0.05], True),        # another order
    ([0.85, 0.1, 0.05, 0.0], [0.85, 0.1, 0.05], True),    # another length
])
def test_fraction_forms(cfgval, identval, refused):
    d = base_data()
    ident = ident_from(d)
    ident["split_fracs"] = identval
    d["split"]["fractions"] = cfgval
    assert (SE.identity_refusal(ident, Cfg(d)) is not None) is refused


def test_numbers_as_floats_dicts_key_by_key():
    d = base_data()
    ident = ident_from(d)
    d2 = copy.deepcopy(d)
    d2["search"]["space_overrides"] = {"head_pool_ops": [0.0, 1.0], "depth_exponent": [3.0, 5]}
    d2["search"]["flow"] = {"num_bins": 10.0, "conditioner_hidden_layers": 2}
    d2["search"]["train_seed"] = 0.0
    assert SE.identity_refusal(ident, Cfg(d2)) is None
    d3 = copy.deepcopy(d)
    d3["search"]["flow"]["extra"] = 1                       # a key on one side only differs
    assert SE.identity_refusal(ident, Cfg(d3)) is not None
    d4 = copy.deepcopy(d)
    d4["search"]["space_overrides"]["head_pool_ops"] = [1, 0]     # list order
    assert SE.identity_refusal(ident, Cfg(d4)) is not None


def test_keys_outside_kcfg_ignored():
    d = base_data()
    ident = ident_from(d)
    d2 = copy.deepcopy(d)
    d2["search"]["trial"] = {"walltime": "99:00:00"}
    d2["search"]["max_concurrent_trials"] = 99
    d2["resources"] = {"max_cores": 7}
    d2["driver"] = {"poll_seconds": 5}
    assert SE.identity_refusal(ident, Cfg(d2)) is None
    ident["split_hash"] = "other"            # bank fields are not compared at a poll
    assert SE.identity_refusal(ident, Cfg(d2)) is None


# ------------------------------------------- contract: committed projects ---

def _committed_cfgs():
    from stages import synthetic as SY
    out = []
    for tag in ("giulia", "linddup"):
        out.append((tag, C.load_config(os.path.join(PROJ, tag, tag + ".json"))))
    synth = C.load_config(os.path.join(PROJ, "synth", "synth.json"), synthetic_ok=True)
    for g in S.default_graph(synth):
        if isinstance(g, SY.RunSearchStage):
            out.append((g.name, g._view(synth)))
    return out


def test_committed_projects_compare_equal():
    cfgs = _committed_cfgs()
    assert len(cfgs) == 7, [n for n, _ in cfgs]   # 2 recorded + synth v4's five searches
    for name, cfg in cfgs:
        assert SE.identity_refusal(ident_from(cfg.data), cfg) is None, name


# --------------------------------------------------- the poll and the dry run ---

class _NoSubmitPBS(object):
    def __init__(self):
        self.lines = []

    def log(self, text):
        self.lines.append(text)

    def __getattr__(self, name):
        raise AssertionError("the scheduler was used (%s) after the identity refused" % name)


def test_poll_refuses_before_anything(tmp_path):
    with open(os.path.join(PROJ, "giulia", "giulia.json"), "rb") as fh:
        data = C._strip_docs(json.loads(fh.read().decode("ascii")))
    for k in ("work_root", "runs_root", "tune_root"):
        data["paths"][k] = str(tmp_path / k)
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))
    cfg = C.load_config(str(path))
    stg = SearchStage()
    ident = ident_from(cfg.data)
    ident["split_fracs"] = [0.8, 0.15, 0.05]
    rounds = stg._dirs(cfg)["rounds_file"]
    os.makedirs(os.path.dirname(rounds))
    with open(rounds, "w") as fh:
        fh.write('{"identity": "untouched"}')
    before = open(rounds, "rb").read()
    jobs = [{"id": "9.x", "role": "trial", "cmd": [], "log": None}]
    st = {"sub": {"identity": ident, "phase": "rounds", "rounds": []}, "jobs": copy.deepcopy(jobs)}
    pbs = _NoSubmitPBS()
    assert stg.poll(cfg, st, pbs) == "finished"
    assert st["sub"]["phase"] == "failed"
    assert st["sub"]["fail"].startswith(EXPECTED_PREFIX + "split_fracs: search [0.8, 0.15, 0.05], "
                                        "configuration [0.85, 0.1, 0.05]"), st["sub"]["fail"]
    assert st["jobs"] == jobs                                   # not forgotten, nothing added
    assert open(rounds, "rb").read() == before                  # not rewritten
    ok, why = stg.verify(cfg, st)
    assert ok is False and why == st["sub"]["fail"]
    lines = stg.dry_run(cfg, st)
    assert lines[0] == st["sub"]["fail"], lines[:2]
    # restored: the poll goes on past the check (it then needs the scheduler)
    st2 = {"sub": {"identity": ident_from(cfg.data), "phase": "rounds", "rounds": []}, "jobs": []}
    lines = stg.dry_run(cfg, st2)
    assert not any(ln.startswith("REFUSE") for ln in lines)
