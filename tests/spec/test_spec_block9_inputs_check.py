"""Spec-derived tests of block 9, the inputs check `tools/inputs_check.py`
(SPEC block 9 and its amendment (b), D-112 / D-113). t1-A run of
2026-11-13 (D-131).

The repository's smoke test of this block skips here: the Sbi-extractor
checkout on GitHub (e9aeb55) predates D-107 / D-113 (no
`sim_reextract/profiles/linddup.sh`, no `KNOWN_TOPOLOGY_AXES`, no job
protocol). These tests build the inputs themselves: the MEAMA-like tree of
the smoke test's builders (`write_task`, `write_manifest`,
`_write_registry`: fixture inputs, not the code under test) and a stand-in
Sbi-extractor surface holding exactly what the spec says the tool reads --
`sbi_labels` with `KNOWN_TOPOLOGY_AXES`, `JOB_ARGS_TOPOLOGY_BOUNDS` and
`load_registry`, the S2 profile, `sim_reextract_plan.py` taking
`--quiet-hours`, env.sh's `sbix_pipeline_v` and the six job scripts calling
`pipeline_stage_job_start`. Expected values are the spec's: which check
FAILs, the last line, the exit codes, the read-only rule.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import pytest

from conftest import PIPE, SYNTH

np = pytest.importorskip("numpy")
sys.path.insert(0, os.path.join(PIPE, "tests"))
import test_smoke_inputs_check as FX     # noqa: E402  (the fixture builders only)

TOOL = os.path.join(PIPE, "tools", "inputs_check.py")
SIX = ("extractor/run_extractor_array_mea.pbs", "extractor/run_cohort_manifest.pbs",
       "sim_reextract/run_sim_reextract_gate.pbs", "export_windows/run_bank_task.pbs",
       "export_windows/run_bank_gate.pbs", "sim_reextract/run_mea_task.pbs")   # block 3 (b), D-113

SBI_LABELS = '''
import importlib.util, os
KNOWN_TOPOLOGY_AXES = ("conn_prob", "stoa_gate_p", "p0_conn", "d0_conn", "beta_conn")
JOB_ARGS_TOPOLOGY_BOUNDS = {"conn_prob": ("conn_prob_lo", "conn_prob_hi"),
                            "stoa_gate_p": ("stoa_gate_lo", "stoa_gate_hi")}
class _Reg(object):
    def __init__(self, names):
        self.param_names = list(names)
def load_registry(sim_dir=None):
    spec = importlib.util.spec_from_file_location("_hsr", os.path.join(sim_dir, "HPC_single_run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return _Reg(m.PARAM_NAMES)
'''


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def build(root, campaigns=(FX.CAMP,), simtime=None, w_size=0.01, eps=9, manifest=True):
    sbix = os.path.join(root, "sbix")
    _w(os.path.join(sbix, "sbi_labels.py"), SBI_LABELS)
    _w(os.path.join(sbix, "sim_reextract", "profiles", "linddup.sh"), "# the S2 profile\n")
    _w(os.path.join(sbix, "sim_reextract", "sim_reextract_plan.py"),
       "# argparse: p.add_argument('--quiet-hours', type=float)\n")
    _w(os.path.join(sbix, "env.sh"), "sbix_pipeline_v() {\n  :\n}\n")
    for rel in SIX:
        _w(os.path.join(sbix, rel), "#!/bin/bash\npipeline_stage_job_start \"$ENV_NAME\"\n")
    jobs = os.path.join(root, "sbi", "hpc", "pipeline", "jobs")
    os.makedirs(jobs)
    for f in ("stage_job.sh", "activate_env.sh", "chain.sh"):
        shutil.copy2(os.path.join(PIPE, "jobs", f), jobs)
    sim_main = os.path.join(root, "MEAMA")
    os.makedirs(sim_main)
    names, bounds = FX._write_registry(sim_main)
    for camp in campaigns:
        for n, (sw, nt, seed) in enumerate(FX.TASKS):
            td = FX.write_task(sim_main, camp, sw, nt, seed, n, names, bounds)
            if simtime is not None:
                p = os.path.join(td, "job_args.json")
                ja = json.load(open(p))
                ja["simtime"] = simtime
                json.dump(ja, open(p, "w"), indent=1)
    if manifest:
        FX.write_manifest(os.path.join(root, "extracted", "cohort_manifest.json"), w_size=w_size,
                          eps=eps)
    cfg = json.load(open(os.path.join(PIPE, "projects", "linddup", "linddup.json")))
    cfg["paths"].update({"sbix_dir": sbix, "sbi_dir": os.path.join(root, "sbi"),
                         "work_root": os.path.join(root, "work"),
                         "banks_root": os.path.join(root, "banks"),
                         "runs_root": os.path.join(root, "runs"), "tune_root": os.path.join(root, "tune")})
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort_config.json")
    cfg["sim"].update({"sim_main": sim_main, "out_root": os.path.join(root, "mea_out"),
                       "tools": os.path.join(root, "tools")})
    if len(campaigns) > 1:
        cfg["sim"]["campaign_glob"] = "campaign_meama_*"
    if simtime is not None:
        cfg["bank"]["t_win_s"] = simtime
    path = os.path.join(root, "linddup.json")
    json.dump(cfg, open(path, "w"), indent=1)
    old = time.time() - 30 * 86400.0            # a campaign that ended a month ago
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            os.utime(os.path.join(dp, n), (old, old))
    return path


def run(cfg_path, *extra):
    rep = cfg_path + ".report.json"
    r = subprocess.run([sys.executable, TOOL, "--config", cfg_path, "--no-qstat", "--quiet-hours", "0",
                        "--json", rep] + list(extra), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=300)
    doc = json.load(open(rep)) if os.path.isfile(rep) else None
    return r.returncode, r.stdout.rstrip("\n").split("\n"), doc


def fails(doc):
    return sorted(set(r["check"] for r in doc["checks"] if r["status"] == "FAIL"))


def _snapshot(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = os.path.join(dp, n)
            st = os.lstat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime)
    return out


# ----------------------------------------------------------------- tests ----

def test_consistent_tree(tmp_path):
    """The consistent tree: no FAIL, exit 0, the last line `INPUTS CHECK:
    <f> FAIL, <w> WARN`; the counts 1 campaign, 2 tasks, 5 topologies, 10
    iteration files; the tree is unchanged by the run (read-only)."""
    cfg = build(str(tmp_path / "t"))
    rc, out, doc = run(cfg)
    assert rc == 0, out[-25:]
    assert out[-1].startswith("INPUTS CHECK: 0 FAIL, ") and out[-1].endswith(" WARN")
    assert fails(doc) == []
    s = doc["summary"]
    assert (s["n_tasks"], s["n_topologies"], s["n_iterations"]) == (2, 5, 10)


def test_read_only(tmp_path):
    """Block 9: 'Reads, never writes (but --json)'; the oracle: the tree's
    listing and modification times are unchanged by a run. Every path under
    the project's folders -- the simulation tree `sim.sim_main` and the
    Sbi-extractor checkout `paths.sbix_dir` included -- is compared. (The
    stand-in's load_registry imports HPC_single_run.py as Sbi-extractor's
    does: the import is what a real checkout does too.)"""
    root = str(tmp_path / "t")
    cfg = build(root)
    before = _snapshot(root)
    run(cfg)
    after = _snapshot(root)
    after.pop("linddup.json.report.json", None)
    new = sorted(set(after) - set(before))
    assert new == [], new
    assert after == before


def test_window_floor_rule(tmp_path):
    """Block 9 `window`: W = round(T / dt), |W dt - T| <= 1e-9 s AND
    floor(T / dt) = W, else FAIL. T = 0.3 s, dt = 0.025 s: T / dt is
    11.999999999999998 in float64, so round gives 12 with |12 dt - T| =
    0 < 1e-9 while floor gives 11: FAIL (the bank plan's own rule)."""
    T, dt = 0.3, 0.025
    assert round(T / dt) == 12 and math.floor(T / dt) == 11 and abs(12 * dt - T) <= 1e-9
    cfg = build(str(tmp_path / "t"), simtime=T, w_size=dt)
    rc, out, doc = run(cfg)
    assert "window" in fails(doc), out[-25:]
    assert rc == 1


def test_window_exact(tmp_path):
    """The same rule passes T = 200 s at dt = 0.01 s (W 20,000)."""
    cfg = build(str(tmp_path / "t"))
    rc, out, doc = run(cfg)
    row = [r for r in doc["checks"] if r["check"] == "window"][0]
    assert row["status"] == "OK" and "20000" in row["text"]


def test_n_e_fails_alone(tmp_path):
    cfg = build(str(tmp_path / "t"), eps=1)
    rc, out, doc = run(cfg)
    assert fails(doc) == ["n_e"] and rc == 1


def test_batches_rule(tmp_path):
    """`batches`: N of a `_v<N>` suffix, else 0 -- two campaigns with one
    number FAIL (here two campaigns without a suffix: 0 and 0)."""
    cfg = build(str(tmp_path / "t"), campaigns=("campaign_meama_rho1600_n01", "campaign_meama_rho1600_n02"))
    rc, out, doc = run(cfg)
    assert fails(doc) == ["batches"], fails(doc)
    cfg2 = build(str(tmp_path / "u"), campaigns=("campaign_meama_a_v1", "campaign_meama_b_v2"))
    rc2, out2, doc2 = run(cfg2)
    assert "batches" not in fails(doc2)


def test_cohort_manifest_states(tmp_path):
    """`cohort manifest`: absent -> NOTE (S1 runs, a first extraction);
    present without a matching .sha256 -> FAIL."""
    cfg = build(str(tmp_path / "t"), manifest=False)
    rc, out, doc = run(cfg)
    row = [r for r in doc["checks"] if r["check"] == "cohort manifest"][0]
    assert row["status"] == "NOTE"
    cfg2 = build(str(tmp_path / "u"))
    side = os.path.join(str(tmp_path / "u"), "extracted", "cohort_manifest.json.sha256")
    open(side, "w").write("0" * 64 + "  cohort_manifest.json\n")
    rc2, out2, doc2 = run(cfg2)
    assert "cohort manifest" in fails(doc2) and rc2 == 1


def test_refused_configuration_exit2(tmp_path):
    """Exit 2 on a refused configuration: a schema problem, and a synthetic
    configuration (sec. 3.1: inputs_check refuses one)."""
    cfg = build(str(tmp_path / "t"))
    d = json.load(open(cfg))
    del d["loader"]
    bad = str(tmp_path / "bad.json")
    json.dump(d, open(bad, "w"))
    assert run(bad)[0] == 2
    assert run(SYNTH)[0] == 2


def test_profile_and_protocol_rows(tmp_path):
    """`S2 profile` FAIL when the profile is absent (no S2 record);
    `protocol (Sbi-extractor)` FAIL when a job script does not call
    pipeline_stage_job_start; `plan option` FAIL without --quiet-hours."""
    root = str(tmp_path / "t")
    cfg = build(root)
    os.remove(os.path.join(root, "sbix", "sim_reextract", "profiles", "linddup.sh"))
    assert fails(run(cfg)[2]) == ["S2 profile"]
    root2 = str(tmp_path / "u")
    cfg2 = build(root2)
    open(os.path.join(root2, "sbix", SIX[3]), "w").write("#!/bin/bash\nconda activate x\n")
    assert fails(run(cfg2)[2]) == ["protocol (Sbi-extractor)"]
    root3 = str(tmp_path / "v")
    cfg3 = build(root3)
    open(os.path.join(root3, "sbix", "sim_reextract", "sim_reextract_plan.py"), "w").write("# old\n")
    assert fails(run(cfg3)[2]) == ["plan option"]
