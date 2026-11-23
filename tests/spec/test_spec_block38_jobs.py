"""Spec tests of block 38, the launcher of a recorded project's evaluation
(`ev_rec_submit.py`, `submit_rec_eval.sh`, `synth_rec_eval.json`).
Integration tester, 2026-11-23, incremental run on dc096e5.

The committed configuration is checked against block 38's values and
`synth_eval.json`; the launcher's dry runs (nothing submitted) are checked
on the as-recorded tree REC_SPEC_OUT/asA (sampled and reported by
test_spec_block36_validation.py) and the recorded tree runA, under copies of
their evaluation configurations whose `out_root` is copied to a temporary
folder so that the trees under test are not changed.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
EVAL = os.path.join(PIPE, "evaluation")
SUBMIT = os.path.join(EVAL, "ev_rec_submit.py")
WRAP = os.path.join(EVAL, "submit_rec_eval.sh")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

FIX = os.environ.get("REC_FIXTURE", "").strip()
OUT = os.environ.get("REC_SPEC_OUT", "/tmp/tst/e2e")
needs_fixture = pytest.mark.skipif(not (FIX and os.path.isdir(FIX)), reason="REC_FIXTURE not set")


def _strip(o):
    if isinstance(o, dict):
        return dict((k, _strip(v)) for k, v in o.items() if not k.startswith("_"))
    return o


def test_committed_synth_rec_eval_values():
    r = _strip(json.load(open(os.path.join(PIPE, "projects", "synth", "synth_rec_eval.json"))))
    s = _strip(json.load(open(os.path.join(PIPE, "projects", "synth", "synth_eval.json"))))
    assert r["as_recorded"] is True and r["sim_draw_rows_per_donor"] is None
    for k in ("seed", "n_post", "batch_rows", "oracle_tol", "bootstrap_n", "level", "dpi",
              "formats", "ncpus"):
        assert r[k] == s[k], k
    assert r["validate_against"] == s["out_root"]
    assert r["out_root"] == "/davinci-1/home/ldellamea/runs/synth/evaluation_rec"
    assert r["edge_eps"] == 0.006693
    assert r["gate"] == {"donor_fracs": [0.4, 0.25, 0.25, 0.1], "n_ref": 800, "n_rank": 1000,
                         "n_null": 1000, "n_null_min": 50, "rows_per_donor": 1,
                         "factors": [0.25, 0.5, 1, 2, 4], "alpha": 0.05}
    assert r["jobs"] == {"queue": s["jobs"]["queue"],
                         "sample": {"ncpus": 8, "mem": "64gb", "walltime": "24:00:00"},
                         "report": {"ncpus": 8, "mem": "64gb", "walltime": "24:00:00"},
                         "validate": {"ncpus": 1, "mem": "8gb", "walltime": "01:00:00"}}
    assert s["jobs"]["queue"] == "intel"
    raw = open(os.path.join(PIPE, "projects", "synth", "synth_rec_eval.json"), "rb").read()
    assert all(b < 128 for b in raw) and b"\r" not in raw


def test_ascii_lf_and_bash_syntax():
    for p in (SUBMIT, WRAP):
        raw = open(p, "rb").read()
        assert all(b < 128 for b in raw) and b"\r" not in raw, p
    assert subprocess.run(["bash", "-n", WRAP]).returncode == 0
    # the launcher imports the standard library and the pipeline's modules only
    src = open(SUBMIT).read()
    mods = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", src, re.M))
    third = {m.split(".")[0] for m in mods} & {"numpy", "scipy", "torch", "sklearn",
                                                "matplotlib", "sbi", "zuko"}
    assert not third, third


def _tree(tmp, run, cfg_name, ecfg_name, **change):
    """A copy of the run's out_root and of its evaluation configuration with
    out_root there (and `change` applied)."""
    src = os.path.join(OUT, run)
    dst = os.path.join(str(tmp), run)
    shutil.copytree(src, dst, symlinks=True)
    d = json.load(open(os.path.join(OUT, "%s_%s" % (run, ecfg_name))))
    d["out_root"] = dst
    for k, v in change.items():
        d[k] = v
    p = os.path.join(str(tmp), "%s_%s" % (run, ecfg_name))
    json.dump(d, open(p, "w"), indent=1)
    return os.path.join(FIX, cfg_name), p, dst


def _dry(cfg, ecfg, *extra):
    env = dict(os.environ, PIPELINE_SKIP_CONDA="1", ROADMAP_DISABLE="0")
    env.pop("ROADMAP_DISABLE")
    env["PATH"] = os.path.join(PIPE, "tests", "fake_pbs") + os.pathsep + env["PATH"]
    r = subprocess.run([sys.executable, SUBMIT, "--config", cfg, "--eval-config", ecfg,
                        "--dry-run"] + list(extra), capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def _would(out):
    """{unit: qsub line} of the units a dry run would submit."""
    res = {}
    for m in re.finditer(r"REC EVAL SUBMIT: (\S+) would be submitted[^\n]*\n\s+(qsub [^\n]+)", out):
        res[m.group(1)] = m.group(2)
    return res


def _snapshot(root):
    out = {}
    for dp, _d, fns in os.walk(root):
        for f in fns:
            p = os.path.join(dp, f)
            out[p] = os.stat(p).st_mtime_ns
    return out


@needs_fixture
def test_dry_run_states_as_recorded(tmp_path):
    from test_spec_block36_validation import _as_tree
    _as_tree()
    cfg, ecfg, dst = _tree(tmp_path / "a", "asA", "synth.json", "rec_eval_as.json")
    before = _snapshot(dst)
    rc, out = _dry(cfg, ecfg)
    assert rc == 0, out
    w = _would(out)
    # samplings done, report done, the validation absent: only the validation, at once
    assert list(w) == ["validate"], out
    assert "depend" not in w["validate"]
    assert "ROADMAP_STAGE=I36" in w["validate"] and "-N er_valid" in w["validate"]
    assert _snapshot(dst) == before, "a dry run wrote under out_root"
    # once the record exists for the current metrics: every unit done
    rec = os.path.join(dst, "VALIDATION.json")
    r = subprocess.run([sys.executable, os.path.join(EVAL, "ev_rec_validate.py"), "--old",
                        json.load(open(ecfg))["validate_against"], "--new", dst, "--record", rec],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    rc, out = _dry(cfg, ecfg)
    assert rc == 0 and _would(out) == {} and "validated: OK, 3 ensemble(s)" in out, out
    # a report value changed (dpi): the report stale and the validation after it
    cfg2, ecfg2, _ = _tree(tmp_path / "b", "asA", "synth.json", "rec_eval_as.json", dpi=61)
    shutil.copy(rec, os.path.join(str(tmp_path / "b"), "asA", "VALIDATION.json"))
    rc, out = _dry(cfg2, ecfg2)
    w = _would(out)
    assert rc == 0 and list(w) == ["report", "validate"], out
    assert "depend=afterok:<report>" in w["validate"]
    assert "ROADMAP_STAGE=I37" in w["report"] and "select=1:ncpus=2" in w["report"]
    # n_post changed: the samplings stale, the report after all of them
    cfg3, ecfg3, _ = _tree(tmp_path / "c", "asA", "synth.json", "rec_eval_as.json", n_post=60)
    rc, out = _dry(cfg3, ecfg3)
    w = _would(out)
    samples = [u for u in w if u.startswith("sample:")]
    assert rc == 0 and len(samples) == 3 and "report" in w, out
    dep = re.search(r"depend=afterok:(\S+)", w["report"]).group(1).split(":")
    assert sorted(dep) == sorted("<%s>" % u.replace(":", "_") for u in samples)
    for u in samples:
        assert "ROADMAP_STAGE=I35" in w[u]
        name = re.search(r"-N (\S+)", w[u]).group(1)
        assert len(name) <= 15 and re.fullmatch(r"er_s\d+", name)
    # --resample one: only that sampling and what follows
    rc, out = _dry(cfg, ecfg, "--resample", "S-A1:A1")
    w = _would(out)
    assert rc == 0 and [u for u in w if u.startswith("sample:")] == ["sample:S-A1:A1"], out


@needs_fixture
def test_recorded_view_has_no_validation(tmp_path):
    from test_spec_block36_37_e2e import _report
    _report("runA")
    cfg, ecfg, dst = _tree(tmp_path, "runA", "synth_rec.json", "rec_eval.json")
    rc, out = _dry(cfg, ecfg)
    assert rc == 0 and "validate" not in _would(out), out
    rc, out = _dry(cfg, ecfg, "--what", "validate")
    assert rc == 3 and "REFUSE" in out, out
    rc, out = _dry(cfg, ecfg, "--what", "nonsense")
    assert rc in (2, 3), out
