"""Chain C1 (the real-data project): the checks that need no chain run --
integration tester, audit D-131, run t2-C1. Oracles C1-X1, C1-X3, C1-X4,
C1-Q1/Q2 on the committed configurations (giulia.json, linddup.json)
through the entry point's dry run, and the sequencer's job name across
projects (block 1 (c): `pl_<tag>`).

    python3 -m pytest -q tests/spec/test_spec_chain_C1_static.py

Pure ASCII, LF only.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
FAKE = os.path.join(PIPE, "tests", "fake_pbs")
sys.path.insert(0, PIPE)
import config as C                 # noqa: E402

PROJECTS = ("giulia", "linddup")
SEQUENCER_CODE = sorted(glob.glob(os.path.join(PIPE, "*.py")) +
                        glob.glob(os.path.join(PIPE, "stages", "*.py")) +
                        [os.path.join(PIPE, "run_pipeline.sh")])


def _cfg(tag):
    with open(os.path.join(PIPE, "projects", tag, "%s.json" % tag)) as fh:
        return C._strip_docs(json.load(fh))


def _env(spool, **extra):
    env = dict(os.environ)
    env["PATH"] = FAKE + os.pathsep + env.get("PATH", "")
    env["FAKE_PBS_SPOOL"] = spool
    env["SKIP_CONDA"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for k in ("PBS_JOBID", "WORK_ROOT", "CHAIN_NAME"):
        env.pop(k, None)
    env.update(extra)
    return env


def _sh(args, spool, timeout=300, **extra):
    p = subprocess.run(["bash", os.path.join(PIPE, "run_pipeline.sh")] + list(args),
                       env=_env(spool, **extra), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout


# ---------------------------------------------------------------------------
# C1-X1 (D-091): no project value in the sequencer's code
# ---------------------------------------------------------------------------

def _project_tokens(tag):
    d = _cfg(tag)
    vals = [d["dataset"], d["real"]["cohort_tag"], d["sim"]["campaign_glob"],
            d["freeze"]["campaigns"], d["freeze"]["out"]]
    for k in ("sbix_dir", "sbi_dir", "work_root", "banks_root", "runs_root", "tune_root"):
        vals.append(d["paths"][k])
    for k in ("cohort_config", "extract_root"):
        vals.append(d["real"][k])
    for k in ("sim_main", "out_root", "tools"):
        vals.append(d["sim"][k])
    toks = set()
    for v in vals:
        toks.add(v)
        for part in v.split("/"):
            if len(part) >= 4 and part not in ("home", "davinci-1", "repos", "banks", "runs",
                                                "SBI", "pipeline", "Sbi-extractor", "artifacts",
                                                "Main", "Config", "hpc", "dsn"):
                toks.add(part)
    return sorted(toks)


@pytest.mark.parametrize("tag", PROJECTS)
def test_no_project_value_in_the_sequencer_code(tag):
    hits = []
    for f in SEQUENCER_CODE:
        t = open(f).read()
        for k in _project_tokens(tag):
            if k in t:
                hits.append((os.path.relpath(f, REPO), k))
    assert not hits, hits


# ---------------------------------------------------------------------------
# C1-X3 (sec. 4): stdlib only, Python 3.6 syntax, ASCII, LF
# ---------------------------------------------------------------------------

def test_sequencer_imports_stdlib_only():
    std = set(sys.stdlib_module_names)
    local = set(os.path.splitext(os.path.basename(f))[0] for f in SEQUENCER_CODE) | {"stages"}
    bad = []
    for f in SEQUENCER_CODE:
        if not f.endswith(".py"):
            continue
        for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))",
                             open(f).read(), re.M):
            names = [m.group(1)] if m.group(1) else [x.strip().split(" ")[0]
                                                     for x in m.group(2).split(",")]
            for n in names:
                top = n.split(".")[0]
                if top and top not in std and top not in local:
                    bad.append((os.path.relpath(f, REPO), n))
    assert not bad, bad


def test_sequencer_python36_syntax():
    vermin = shutil.which("vermin")
    if vermin is None:
        pytest.skip("vermin is not installed")
    files = [f for f in SEQUENCER_CODE if f.endswith(".py")]
    p = subprocess.run([vermin, "--no-tips", "--violations", "-t=3.6-"] + files,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert p.returncode == 0, p.stdout[-3000:]


def test_pipeline_files_ascii_lf():
    bad = []
    for f in SEQUENCER_CODE + glob.glob(os.path.join(PIPE, "projects", "*", "*.json")) + \
            glob.glob(os.path.join(PIPE, "jobs", "*")) + glob.glob(os.path.join(PIPE, "tools", "*")):
        b = open(f, "rb").read()
        if b"\r" in b or any(c > 127 for c in b):
            bad.append(os.path.relpath(f, REPO))
    assert not bad, bad


# ---------------------------------------------------------------------------
# C1-X4 (sec. 3.1): the recorded tools refuse a synthetic configuration
# ---------------------------------------------------------------------------

def test_load_config_refuses_synthetic_without_the_flag():
    p = os.path.join(PIPE, "projects", "synth", "synth.json")
    with pytest.raises(C.ConfigError):
        C.load_config(p)
    assert C.load_config(p, synthetic_ok=True).data["kind"] == "synthetic"
    for tag in PROJECTS:
        assert "kind" not in C.load_config(os.path.join(PIPE, "projects", tag, "%s.json" % tag)).data


@pytest.mark.parametrize("tool,args", [
    ("tools/inputs_check.py", ["--no-qstat"]),
    ("tools/perrow_check.py", []),
    ("tools/search_report.py", []),
    ("tools/search_reset.py", []),
    ("tools/purge_outputs.py", ["--what", "search"]),
])
def test_recorded_tools_refuse_a_synthetic_configuration(tool, args):
    spool = tempfile.mkdtemp(prefix="c1_static_")
    try:
        cmd = [sys.executable, os.path.join(PIPE, tool), "--config",
               os.path.join(PIPE, "projects", "synth", "synth.json")] + args
        p = subprocess.run(cmd, env=_env(spool), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=120)
        assert p.returncode == 2, (tool, p.returncode, p.stdout[-2000:])
    finally:
        shutil.rmtree(spool, ignore_errors=True)


# ---------------------------------------------------------------------------
# C1-Q1, C1-Q2 on the committed projects: the dry run's lines
# ---------------------------------------------------------------------------

def _argv_line(out, marker):
    for line in out.splitlines():
        if marker in line and " argv: " in line:
            return line.split(" argv: ", 1)[1]
    return None


@pytest.mark.parametrize("tag", PROJECTS)
def test_committed_dry_run_carries_the_configuration(tag):
    d = _cfg(tag)
    spool = tempfile.mkdtemp(prefix="c1_static_")
    try:
        rc, out = _sh(["--project", tag, "--dry-run"], spool)
        assert rc == 0 and "REFUSE" not in out, out[-3000:]
        assert not glob.glob(os.path.join(spool, "*.json")), "the dry run submitted a job"
        se, b = d["search"], d["best_fit"]
        tri = _argv_line(out, "npe_tune_joint.py evaluate")
        assert tri, out[-3000:]
        want_t = ["--epochs %d" % se["trial"]["epochs"], "--passes %g" % se["trial"]["passes"],
                  "--split-fracs %s" % " ".join(str(f) for f in d["split"]["fractions"]),
                  "--flow-hidden-layers %d" % se["flow"]["conditioner_hidden_layers"],
                  "--buffer-shards %d" % d["loader"]["buffer_shards"],
                  "--threads %d" % se["trial"]["ncpus"], "--patience 99",
                  "--campaign %s" % se["campaign"]]
        miss = [w for w in want_t if w not in tri]
        assert not miss, (miss, tri)
        mem = _argv_line(out, "run_joint_arms.py --arm <finalist's arm>")
        assert mem, out[-3000:]
        want_m = ["--split-seed %d" % b["split_seed"], "--epochs %d" % b["epochs"],
                  "--passes %g" % b["passes"], "--patience %d" % b["patience_epochs"],
                  "--threads %d" % b["ncpus"], "--n-posterior-draws %d" % b["n_posterior_draws"],
                  "--buffer-shards %d" % d["loader"]["buffer_shards"],
                  "--split-fracs %s" % " ".join(str(f) for f in d["split"]["fractions"])]
        miss = [w for w in want_m if w not in mem]
        assert not miss, (miss, mem)
        q = [l for l in out.splitlines() if "qsub -N t_<trial>" in l]
        assert q and ("walltime=%s" % se["trial"]["walltime"]) in q[0] and \
            ("ncpus=%d:mem=%s" % (se["trial"]["ncpus"], se["trial"]["mem"])) in q[0], q
        assert ("cores: at most %d at once" % d["resources"]["max_cores"]) in out
        assert ("PBS jobs named pl_%s" % tag) in out
    finally:
        shutil.rmtree(spool, ignore_errors=True)


@pytest.mark.parametrize("tag", PROJECTS)
def test_dry_run_trial_line_states_the_resume_rule_of_d111(tag):
    """D-111 (e): every trial's and member's argv carries --resume; the dry
    run's own line for them should not say otherwise."""
    spool = tempfile.mkdtemp(prefix="c1_static_")
    try:
        rc, out = _sh(["--project", tag, "--dry-run"], spool)
        tri = _argv_line(out, "npe_tune_joint.py evaluate")
        mem = _argv_line(out, "run_joint_arms.py --arm <finalist's arm>")
        assert "--resume on a resubmission" not in (tri or "") + (mem or ""), (tri, mem)
    finally:
        shutil.rmtree(spool, ignore_errors=True)


# ---------------------------------------------------------------------------
# The sequencer's job name across projects (block 1 (c): pl_<tag>)
# ---------------------------------------------------------------------------

def test_chain_name_is_pl_tag_for_any_valid_tag():
    sys.path.insert(0, os.path.join(PIPE, "stages"))
    from stages import common as CM

    class _C(object):
        def __init__(self, tag):
            self.data = {"dataset": tag}
    a, b = "cohort_alpha_1", "cohort_alpha_2"       # valid tags ([a-z0-9_]+)
    names = (CM.chain_name(_C(a)), CM.chain_name(_C(b)))
    assert names == ("pl_" + a, "pl_" + b), names


def test_two_projects_with_long_tags_do_not_share_a_sequencer():
    """Two recorded projects whose tags share their first 12 characters, on
    one scheduler: the second project's submission must not be refused for
    the first project's sequencer job (block 1 (c): the refusal is for a job
    of THIS project's sequencer, pl_<tag>)."""
    root = tempfile.mkdtemp(prefix="c1_names_")
    spool = os.path.join(root, "spool")
    try:
        paths = []
        for tag in ("cohort_alpha_1", "cohort_alpha_2"):
            d = json.load(open(os.path.join(PIPE, "projects", "giulia", "giulia.json")))
            d["dataset"] = tag
            base = os.path.join(root, tag)
            for k in ("work_root", "banks_root", "runs_root", "tune_root"):
                d["paths"][k] = os.path.join(base, k)
            d["paths"]["sbi_dir"] = REPO        # this checkout's job scripts
            p = os.path.join(root, "%s.json" % tag)
            with open(p, "w") as fh:
                json.dump(d, fh)
            paths.append(p)
        # the first project's sequencer job stays queued 30 s (FAKE_PBS_DELAY)
        rc1, out1 = _sh(["--config", paths[0]], spool, FAKE_PBS_DELAY="30")
        rc2, out2 = _sh(["--config", paths[1]], spool, FAKE_PBS_DELAY="30")
        # the second project's --stop, while the first one's job is queued
        rc3, out3 = _sh(["--config", paths[1], "--stop"], spool, FAKE_PBS_DELAY="30")
        jobs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(spool, "*.json")))
                if os.path.basename(f)[:-5].isdigit()]
        names = [j["name"] for j in jobs]
        first = jobs[0] if jobs else {}
        first_deleted_by_other = "deleted" in (first.get("comment") or "").lower()
        # end both chains: delete what is queued (the fake qdel)
        for j in jobs:
            subprocess.run([os.path.join(FAKE, "qdel"), j["id"]], env=_env(spool),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert rc1 == 0, out1[-2000:]
        assert rc2 == 0 and not first_deleted_by_other, (
            "submission of the second project: exit %d, %s; its --stop: exit %d, %s; the first "
            "project's job %s: %s" % (rc2, [l for l in out2.splitlines() if "REFUSE" in l], rc3,
                                     [l for l in out3.splitlines() if "stop:" in l],
                                     first.get("id"), first.get("comment")), names)
    finally:
        time.sleep(1)
        shutil.rmtree(root, ignore_errors=True)
