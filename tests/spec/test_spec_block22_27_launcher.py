"""Spec-derived tests of the evaluation's launcher as block 27 amends block 22
(the comparisons listed after the report's unit; a configuration whose
comparisons cannot be listed refused only when `report` is in --what),
written by the integration tester from specs/SPEC.md alone. The real entry
point, `submit_eval.sh`, is run with SKIP_CONDA=1 on copies of the committed
configurations whose paths lie in a temporary folder.

Run: python3 -m pytest -q tests/spec/test_spec_block22_27_launcher.py
Pure ASCII, LF only.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _spec_util as U  # noqa: E402

SUBMIT = os.path.join(U.EVAL, "submit_eval.sh")
V4_LINES = [
    "EVAL SUBMIT: comparison S-A2-R0: S-A2-R0:A2 against S-A1:A1 (vs_A1)",
    "EVAL SUBMIT: comparison S-A5-R0: S-A5-R0:A5 against S-A1:A1 (vs_A1)",
    "EVAL SUBMIT: comparison S-A2-R3: S-A2-R3:A2 against S-A1:A1 (vs_A1)",
    "EVAL SUBMIT: comparison S-A5-R3: S-A5-R3:A5 against S-A1:A1 (vs_A1)",
    "EVAL SUBMIT: comparison S-A2-R0_vs_S-A5-R0: S-A2-R0:A2 against S-A5-R0:A5 (A2_vs_A5)",
    "EVAL SUBMIT: comparison S-A2-R3_vs_S-A5-R3: S-A2-R3:A2 against S-A5-R3:A5 (A2_vs_A5)",
]


def _run(cp, ep, *args):
    env = dict(os.environ, SKIP_CONDA="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PIPELINE_SKIP_CONDA", None)
    r = subprocess.run(["bash", SUBMIT, "--config", cp, "--eval-config", ep] + list(args),
                       capture_output=True, text=True, env=env, timeout=300)
    return r.returncode, r.stdout.splitlines(), r.stderr


def _tree(root):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


def test_dry_run_on_v4_lists_the_six_comparisons_after_the_report(tmp_path):
    cp, ep = U.write_project(str(tmp_path / "p"))
    before = _tree(str(tmp_path))
    rc, out, err = _run(cp, ep, "--dry-run")
    assert rc == 0, out + [err]
    assert out[-1].startswith("EVAL DRY RUN: 5 job(s) would be submitted, 0 done, 0 live;"), out[-1]
    i = [k for k, l in enumerate(out) if l.startswith("EVAL SUBMIT: report:")]
    assert len(i) == 1, out
    assert out[i[0] + 1:i[0] + 7] == V4_LINES
    assert sum(1 for l in out if "not complete:" in l) == 5
    assert sum(1 for l in out if l.strip().startswith("qsub ")) == 5
    assert _tree(str(tmp_path)) == before                    # nothing written


@pytest.mark.parametrize("what", ["calbank,sample", "calbank", "sample", "data"])
def test_without_report_no_comparison_is_consulted(tmp_path, what):
    # term-only searches with no S-A1 search: the comparisons would refuse
    searches = [U.search("T2", "S-A2-only", "R0"), U.search("T5", "S-A5-only", "R0")]
    cp, ep = U.write_project(str(tmp_path / "p"), searches=searches)
    rc, out, err = _run(cp, ep, "--dry-run", "--what", what)
    assert rc == 0, out + [err]
    assert not [l for l in out if "comparison" in l or l.startswith("REFUSE")], out


@pytest.mark.parametrize("what", ["report", "calbank,sample,report", None])
def test_with_report_unlistable_comparisons_refuse_exit_3_nothing_written(tmp_path, what):
    searches = [U.search("T2", "S-A2-only", "R0"), U.search("T5", "S-A5-only", "R0")]
    cp, ep = U.write_project(str(tmp_path / "p"), searches=searches)
    before = _tree(str(tmp_path))
    args = ["--dry-run"] + (["--what", what] if what else [])
    rc, out, err = _run(cp, ep, *args)
    assert rc == 3, out + [err]
    ref = [l for l in out if l.startswith("REFUSE: ")]
    assert len(ref) == 1 and "T2" in ref[0], out
    assert not [l for l in out if "would be submitted" in l], out
    assert _tree(str(tmp_path)) == before
    # the same refusal without --dry-run: nothing submitted, nothing written
    rc, out, err = _run(cp, ep, *([] if what is None else ["--what", what]))
    assert rc == 3 and [l for l in out if l.startswith("REFUSE: ")], out
    assert _tree(str(tmp_path)) == before


def test_uncompared_search_line(tmp_path):
    searches = [U.search("S-A1", "S-A1", "R0"), U.search("X25", "S-A25", "R3"),
                U.search("T2", "S-A2-only", "R0")]
    cp, ep = U.write_project(str(tmp_path / "p"), searches=searches)
    rc, out, err = _run(cp, ep, "--dry-run", "--what", "report")
    assert rc == 0, out + [err]
    i = [k for k, l in enumerate(out) if l.startswith("EVAL SUBMIT: report:")][0]
    assert out[i + 1:i + 3] == [
        "EVAL SUBMIT: comparison T2: T2:A2 against S-A1:A1 (vs_A1)",
        "EVAL SUBMIT: no comparison for X25: campaign S-A25: its arms are not defined against "
        "A1 (D-136)"]


def test_report_recorded_without_comparisons_is_stale(tmp_path):
    sys.path.insert(0, U.EVAL)
    import ev_config
    import ev_plan
    import ev_submit
    cp, ep = U.write_project(str(tmp_path / "p"))
    cfg, ecfg = ev_plan.load_project(cp), ev_config.load(ep)
    now = ev_plan.report_inputs(cfg, ecfg)
    os.makedirs(ecfg.out_root, exist_ok=True)
    rc_path = os.path.join(ecfg.out_root, "REPORT_COMPLETE.json")
    old = dict(now)
    old.pop("comparisons")
    for inputs, want in ((old, "stale"), (dict(now, comparisons=now["comparisons"][:-1]), "stale"),
                         (now, "done")):
        with open(rc_path, "w") as fh:
            json.dump({"schema": "eval_report/1", "inputs": inputs, "finished": "x"}, fh)
        state, why = ev_submit.report_state(cfg, ecfg)
        assert state == want, (state, why)


@pytest.mark.parametrize("which", ["project", "evaluation"])
def test_a_refused_configuration_exits_2(tmp_path, which):
    """Block 22: exit "2 on the arguments or the configurations". A project file the
    pipeline's config.py refuses (alt_best_fit true on an S-A1 search, block 18) and an
    evaluation file ev_config refuses (n_post -5) each end with exit 2 and no traceback."""
    if which == "project":
        cp, ep = U.write_project(str(tmp_path / "p"), searches=[U.search("S-A1", "S-A1", "R0", True)])
    else:
        cp, ep = U.write_project(str(tmp_path / "p"))
        d = json.load(open(ep))
        d["n_post"] = -5
        json.dump(d, open(ep, "w"))
    rc, out, err = _run(cp, ep, "--dry-run")
    assert rc == 2, (rc, out[-3:], err[-1500:])
    assert "Traceback" not in err, err[-1500:]
