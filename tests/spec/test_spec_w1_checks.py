"""Spec-derived tests of `hpc/pipeline/tests/run_w1_checks.pbs` (SPEC sec. 6:
the cluster-side check of wave W1, pass line `W1CHECK: ALL 6 SUITES PASSED`;
sec. 4: a pass condition is a line that must APPEAR), added to this run's
scope by the tester because the script changed and no block claims it. Run
here as sec. 6 says: `W1_SKIP_CONDA=1 bash <script>`, with W1_SUITES chosen
so that each case takes seconds.

Run: python3 -m pytest -q tests/spec/test_spec_w1_checks.py
Pure ASCII, LF only.
"""

import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO, "hpc", "pipeline", "tests", "run_w1_checks.pbs")


def _run(tmp_path, suites, extra_env=None, repo=REPO):
    env = dict(os.environ, W1_SKIP_CONDA="1", W1_REPO=repo, W1_OUT=str(tmp_path / "out"),
               TMPDIR=str(tmp_path), W1_SUITES=suites, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PIPELINE_SKIP_CONDA", None)
    env.update(extra_env or {})
    r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=env, timeout=900,
                       cwd=str(tmp_path))
    return r.returncode, r.stdout.strip().splitlines()


def test_script_is_ascii_lf_and_parses():
    data = open(SCRIPT, "rb").read()
    assert all(b < 128 for b in data) and b"\r" not in data
    assert data.startswith(b"#!/bin/bash\n")
    assert subprocess.run(["bash", "-n", SCRIPT]).returncode == 0


def test_a_failing_suite_never_gives_the_pass_line(tmp_path):
    rc, out = _run(tmp_path, "test_smoke_does_not_exist.py")
    assert rc == 1, out
    assert out[-1].startswith("W1CHECK: FAILED: test_smoke_does_not_exist.py"), out[-1]
    assert not [l for l in out if "SUITES PASSED" in l]


def test_not_a_checkout_is_refused(tmp_path):
    rc, out = _run(tmp_path, "test_smoke_pipeline.py", repo=str(tmp_path))
    assert rc == 2 and out[-1].startswith("W1CHECK: REFUSE: "), out


def test_suites_that_skip_for_want_of_torch_do_not_pass(tmp_path):
    """A training environment in which torch cannot be imported (a node or an
    environment that is not the one the suites need): test_smoke_block7.py
    skips every check and test_smoke_identity.py its end-to-end ones, each
    with exit 0. The header of the script documents the summaries that must
    appear (block7 "7 of 8 checks pass", identity "5 of 7"); the last line
    must not declare every suite passed when they do not."""
    shim = tmp_path / "shim" / "torch"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("raise ImportError('torch withheld by the test')\n")
    rc, out = _run(tmp_path, "test_smoke_identity.py test_smoke_block7.py",
                   {"PYTHONPATH": str(tmp_path / "shim")})
    assert any("torch MISSING" in l for l in out), out
    assert not out[-1].startswith("W1CHECK: ALL 2 SUITES PASSED"), out
