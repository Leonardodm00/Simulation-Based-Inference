"""Spec tests of block 42 (the card; specs/SPEC.md sec. 2.7, "The card", and
block 42). Integration tester, 2026-11-28, incremental run on 39e6f66.

The manifest and the members' records are built here; the expected
pre-filled values are read from them by the test; the TODO list is the
spec's (block 42's Rule).
"""

import os
import subprocess
import sys
import tempfile

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import _tool_util as U  # noqa: E402
from test_spec_block40_schema import manifest  # noqa: E402

from sbitool import card as C  # noqa: E402
from sbitool import schema as S  # noqa: E402

# block 42's Rule: TODO in these, and n_networks when the export gives none
TODO_FIELDS = {"title", "scientific_question", "simulator.model_type", "simulator.reference",
               "simulator.fixed_parameters", "training.hyperparameter_search", "references",
               "authors", "licence", "cite", "data", "limitations"}


def records(man):
    cfg = {"embedding_size": 8, "depth_exponent": 1, "width_multiplier": 1.0, "block_family": 0,
           "head_fusion": 0, "dropout": 0.1, "head_pool_ops": 0, "hidden_features": 16,
           "num_transforms": 2, "num_bins": 4, "flow_hidden_layers": None, "epochs": 3,
           "lr": 0.001, "b_sim": 64, "sim_shards": "/x/y", "out_dir": "/o", "seed": 0,
           "threads": 2}
    out = []
    for m in man["members"]:
        c = dict(cfg, seed=m["seed"])
        out.append({"arm": man["arm"], "seed": m["seed"], "n_train": 640,
                    "split_fracs": [0.8, 0.1, 0.1], "split_seed": 1, "config": c})
    return out


def test_t42a_template_prefilled_and_todo():
    man = manifest()
    recs = records(man)
    text = C.template(man, recs, project="toy", n_networks=4)
    h, body = C.parse(text)
    assert list(h) == list(S.CARD_KEYS)                         # in CARD_KEYS' order
    t = h["training"]
    assert t["arm"] == man["arm"] and t["members"] == man["M"]
    assert t["n_networks"] == 4 and t["n_windows"] == 640
    assert t["split_fracs"] == [0.8, 0.1, 0.1]
    hp = t["hyperparameters"]
    # the keys that describe the model and its optimisation; no path, switch or seed
    for k in ("embedding_size", "hidden_features", "lr", "epochs", "b_sim", "dropout"):
        assert hp[k] == recs[0]["config"][k], k
    for k in ("sim_shards", "out_dir", "seed", "threads"):
        assert k not in hp, k
    assert h["simulator"]["searched_parameters"] == man["parameters"]["names"]
    assert h["extraction"] == man["extraction"]
    assert h["software"] == man["software"]
    todo = set(S.card_todo_fields(h))
    assert todo == TODO_FIELDS, todo ^ TODO_FIELDS
    # without n_networks the field is TODO too; a value is never guessed
    h2, _ = C.parse(C.template(man, recs))
    assert h2["training"]["n_networks"] == "TODO"
    # records that disagree on a pre-filled value: refused, not guessed
    bad = records(man)
    bad[1]["split_fracs"] = [0.7, 0.2, 0.1]
    with pytest.raises(ValueError):
        C.template(man, bad)
    with pytest.raises(ValueError):
        C.template(man, recs, n_networks=0)
    # a manifest with extraction null: "not applicable"
    m0 = manifest()
    m0["extraction"] = None
    assert C.parse(C.template(m0, records(m0)))[0]["extraction"] == "not applicable"


def _filled(man):
    h, body = C.parse(C.template(man, records(man), n_networks=4))
    h.update(title="T", scientific_question="Q", references=["R"], licence="MIT", cite="C",
             limitations="L", authors=[{"name": "N", "contact": "c@x"}],
             data={"simulated_bank": {"location": "l", "public": False, "size_bytes": None,
                                      "sha256": None},
                   "recordings": {"location": "l", "public": True}})
    h["simulator"].update(model_type="m", reference="r", fixed_parameters={"rho": "1 /mm2"})
    h["training"]["hyperparameter_search"] = "GP over 50 trials"
    return h, body


def _cli(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "sbitool"] + list(args), cwd=U.REPO, env=env,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def _write(text):
    fd, p = tempfile.mkstemp(suffix=".md", prefix="card_")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return p


def test_t42a_card_check_names_exactly_the_todo_fields():
    man = manifest()
    p = _write(C.template(man, records(man), n_networks=4))
    rc, out, err = _cli("card-check", p)
    assert rc == 0 and out.startswith("CARD OK: draft, %d field(s) to fill: " % len(TODO_FIELDS))
    names = set(out.splitlines()[0].split(": ", 2)[2].split(", "))
    assert names == TODO_FIELDS


def test_t42b_complete_and_single_faults():
    man = manifest()
    h, body = _filled(man)
    assert C.check(C.dump(h, body), man, complete=True) == []
    rc, out, err = _cli("card-check", _write(C.dump(h, body)))
    assert rc == 0 and "CARD OK: complete" in out
    faults = [
        ("a key removed", lambda d: d.pop("licence"), "licence"),
        ("an unknown key", lambda d: d.update(colour="red"), "colour"),
        ("an empty list", lambda d: d.update(references=[]), "references"),
        ("arm changed", lambda d: d["training"].update(arm="A5"), "arm"),
        ("a parameter renamed", lambda d: d["simulator"]["searched_parameters"].__setitem__(
            0, "zz"), "searched_parameters"),
        ("a nested TODO", lambda d: d["simulator"]["fixed_parameters"].update(rho="TODO"),
         "rho"),
    ]
    for what, f, word in faults:
        hh, _ = _filled(man)
        f(hh)
        probs = C.check(C.dump(hh, body), man, complete=True)
        assert len(probs) == 1 and word in probs[0], (what, probs)


def test_t42c_malformed_cards():
    bad = {
        "no header": "# a card\n",
        "one ---": "---\ntitle: x\n",
        "tab error": "---\ntitle: x\n\tlicence: y\n---\n",
        "a list": "---\n- a\n- b\n---\n",
        "python tag": "---\ntitle: !!python/object/apply:os.system ['echo hi']\n---\n",
    }
    for what, text in bad.items():
        with pytest.raises(C.CardError):
            C.parse(text)
        rc, out, err = _cli("card-check", _write(text))
        assert rc == 3 and len([ln for ln in err.splitlines() if ln.strip()]) == 1, (what, err)
        assert "hi\n" not in out                                 # nothing executed


def test_t42d_round_trip():
    man = manifest()
    text = C.template(man, records(man), n_networks=4)
    h, body = C.parse(text)
    assert C.dump(h, body) == text
    # and through PyYAML's own reader
    assert yaml.safe_load(text.split("---\n")[1]) == h


def test_card_template_command_takes_export_arguments():
    """Block 42's Public API: `card-template <bundle or export arguments> --out CARD.md`."""
    rc, out, err = _cli("card-template", "--config", "x.json", "--eval-config", "y.json",
                        "--ensemble", "S-A1:A1", "--out", "/tmp/never_written_CARD.md")
    assert rc != 2, ("card-template does not take the export's arguments (exit %d): %s"
                     % (rc, (out + err)[-300:]))
