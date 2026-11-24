"""The README's LindDUP sections (hpc/pipeline/README.md; specs/SPEC.md sec. 0
covers the README; the run sheet "LindDUP: lcdup15 and its three arms -- the
first submission", N1-N9, and the v11 description).

Oracles (tester run 2026-11-17_1730; from the spec and the committed files, read
with json and csv here, not through the pipeline):
  OR.1 the description's numbers are linddup.json v11's and the cohort map's.
  OR.2 "Where things land" names the folders the configuration and block 29's
       views give (tune_root/<q>/<campaign>, runs_root/<q>/<dataset>_best).
  OR.3 every pass line the run sheet states appears when its command runs here:
       N1's cluster-side check, N6's rows, N7's dry-run lines, N9's stop line and
       status line; N4 and N5 behave as stated on a stand-in campaign.
  OR.4 the README is pure ASCII with LF line ends (it ships to the cluster).
"""

import csv
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import _spec_util as U

README = os.path.join(U.PIPE, "README.md")


def _readme():
    return open(README, "rb").read().decode("ascii")


def _section(title_start, end_marker="\n### "):
    t = _readme()
    i = t.index(title_start)
    j = t.find(end_marker, i + len(title_start))
    return t[i:j if j >= 0 else len(t)]


def _sheet():
    return _section("### LindDUP: lcdup15 and its three arms")


def _cmd(label):
    """The command line of run-sheet step `label` (the first line starting
    with `cd ` or `bash ` after its comment)."""
    s = _sheet()
    i = s.index("# %s." % label)
    for line in s[i:].splitlines()[1:]:
        if line.startswith(("cd ", "bash ")):
            return line
    raise AssertionError(label)


def test_readme_ascii_lf():
    blob = open(README, "rb").read()
    blob.decode("ascii")
    assert b"\r" not in blob


def test_description_numbers_are_the_files():
    d = U.strip_docs(U.raw_json(U.LINDDUP))
    rows = list(csv.DictReader(open(U.COHORT_MAP, newline="")))
    inc = [r for r in rows if r["include"].strip().lower() in ("yes", "true", "1")]
    desc = _readme()[_readme().index("`projects/linddup/linddup.json` [since 2026-11-17"):]
    desc = desc[:desc.index("What follows is v10's")]
    assert "campaign_lcdup15_rho1300_neur_n01" in desc and d["sim"]["campaign_glob"] == \
        "campaign_lcdup15_rho1300_neur_n01"
    assert "p = 14" in desc and d["bank"]["expected_p"] == 14 == d["search"]["shape_anchors"]["p"]
    assert "%d wells" % len(inc) in desc and len(inc) == 30 and len(rows) - len(inc) == 5
    assert [(q["name"], q["campaign"], q["max_cores"]) for q in d["searches"]] == \
        [("A1", "S-A1", 148), ("A2", "S-A2-only", 148), ("A5", "S-A5-only", 148)]
    assert "`resources.max_cores` 444" in desc and d["resources"]["max_cores"] == 444
    assert "3 jobs of 48 at once" in desc and 148 // d["search"]["trial"]["ncpus"] == 3
    assert d["replicate"] == {"pairs": "within_group", "group_key": "rep_group"}
    assert d["bank"]["drop_partial_topologies"] is True
    assert d["search"]["max_total_walltime_h"] is None
    queues = {d["sim"]["queue"], d["bank"]["queue"], d["search"]["trial"]["queue"],
              d["best_fit"]["queue"], d["driver"]["queue"]}
    assert "every job on `intel`" in desc and queues == {"intel"}
    sheet = _sheet()
    per = {}
    for r in inc:
        per[r["donor"]] = per.get(r["donor"], 0) + 1
    assert "30 wells, 17 control and 13 DUP" in sheet and per == {"control": 17, "pathological": 13}
    excluded = [r["out_dir"] for r in rows if r not in inc]
    assert "five copied `DATA_P/Batch4` wells" in sheet
    assert len(excluded) == 5 and all("DATA_P_Batch4" in x for x in excluded), excluded


def test_where_things_land():
    d = U.strip_docs(U.raw_json(U.LINDDUP))
    s = _sheet()
    p = d["paths"]
    assert "the work root `%s`" % p["work_root"] in s
    assert "re-detections `%s`" % d["sim"]["out_root"] in s
    assert "`<Sbi-extractor>/%s`" % d["freeze"]["out"] in s
    assert "`%s`" % os.path.join(p["banks_root"], "linddup_sim") in s and "`linddup_real`" in s
    assert "`%s/<A1|A2|A5>/<S-A1|S-A2-only|S-A5-only>`" % p["tune_root"] in s
    assert "`%s/<A1|A2|A5>/linddup_best`" % p["runs_root"] in s
    # the views give these folders (block 29): tune_root/<q>/<campaign>, runs_root/<q>/<ds>_best
    import config as C
    import stages.runs as R
    import stages.search as SS
    cfg = C.load_config(U.LINDDUP)
    for name, view in R.search_views(cfg):
        q = [x for x in d["searches"] if x["name"] == name][0]
        assert SS.SearchStage()._dirs(view)["campaign"] == os.path.join(p["tune_root"], name,
                                                                         q["campaign"])
        assert view.data["paths"]["runs_root"] == os.path.join(p["runs_root"], name)


def test_n1_cluster_side_check_on_a_copy(tmp_path):
    cmd = _cmd("N1")
    k = cmd.index("bash -c '")
    root = str(tmp_path / "SBI")
    shutil.copytree(os.path.join(U.REPO, "hpc"), os.path.join(root, "hpc"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run(["bash", "-c", "cd %s && %s" % (root, cmd[k:])], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "PIPELINE CODE OK", r.stdout


@pytest.fixture(scope="module")
def dry():
    """The LindDUP dry run on a copy of linddup.json with its paths moved and
    S1's artefact of record present (as on davinci: extracted_v2)."""
    root = U.tmpdir("spec_readme_")
    path, d = U.linddup_copy(root)
    m = os.path.join(d["real"]["extract_root"], "cohort_manifest.json")
    os.makedirs(os.path.dirname(m))
    with open(m, "w") as fh:
        fh.write("{}\n")
    with open(m + ".sha256", "w") as fh:
        fh.write("%s  cohort_manifest.json\n" % hashlib.sha256(open(m, "rb").read()).hexdigest())
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    env = U.env_with_fake_pbs(spool, {"SKIP_CONDA": "1"})
    rc, out = U.run(["bash", os.path.join(U.PIPE, "run_pipeline.sh"), "--config", path, "--dry-run"],
                    env=env, cwd=root, timeout=300)
    yield rc, out, root, path, d, env
    shutil.rmtree(root, ignore_errors=True)


def test_n7_dry_run_lines(dry):
    rc, out, root, path, d, env = dry
    lines = out.splitlines()
    assert rc == 0 and not re.search(r"REFUSE|STOP|FAILED", out), out[-3000:]
    assert any(l.endswith("real_extraction: would be skipped (artefact of record exists)") for l in lines)
    assert any(l.endswith("sim_redetection: would run") for l in lines)      # N8's grep
    assert re.search(r"the array at CONCURRENCY 9 = min\(sim\.concurrency 9, the cap's room 9\) of "
                     r"48 cores", out)
    six = [l for l in lines if "cores (block 29, D-141): the jobs of search_" in l
           and "share at most 148 cores" in l and l.endswith("at most 3 job(s) of 48 cores at once")]
    assert len(six) == 6, six
    assert len([l for l in lines if "at most 3 trial(s) at once" in l]) == 3
    assert len([l for l in lines if "at most 3 member(s) at once" in l]) == 3
    flags = "--rep-pairs within_group --rep-group-key rep_group"
    argv = [l for l in lines if "argv:" in l and flags in l]
    assert len(argv) == 9, argv
    assert any("cores: at most 444 at once" in l for l in lines)
    for q in ("A1", "A2", "A5"):
        assert any(("cores of search %s: at most 148" % q) in l for l in lines), q
    assert any("the sequencer: PBS jobs named pl_linddup on queue intel" in l for l in lines)


def test_n7_argv_lines_end_with_the_flags(dry):
    """N7 says "nine argv lines ending in" the replicate flags."""
    rc, out, root, path, d, env = dry
    flags = "--rep-pairs within_group --rep-group-key rep_group"
    argv = [l for l in out.splitlines() if "argv:" in l and flags in l]
    ending = [l for l in argv if l.endswith(flags)]
    assert len(ending) == 9, ["..." + l[-80:] for l in argv]


def test_n9_stop_and_status_lines(dry):
    rc, out, root, path, d, env = dry
    trials = os.path.join(d["paths"]["tune_root"], "A1", "S-A1", "trials")
    U.write_json(os.path.join(trials, "a" * 64 + ".json"),
                 {"trial_id": "a" * 64, "status": "ok", "nll": 1.0, "config": {}})
    rc, out = U.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "stop-search", "--config", path,
                     "--search", "A1"], env=env, cwd=root)
    last = out.strip().splitlines()[-1]
    assert rc == 0 and last.startswith("STOP REQUESTED: search A1 (S-A1); kill running: no; the "
                                       "sequencer acts at its next tick"), out
    rc, out = U.run(["bash", os.path.join(U.PIPE, "run_pipeline.sh"), "--config", path, "--status"],
                    env=env, cwd=root, timeout=300)
    assert re.search(r"search A1: stop requested \S+ \(kill running: no\)", out), out[-3000:]
    assert "cores: at most 444" in out and "cores of search A1: at most 148" in out
    rc, out = U.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "stop-search", "--config", path,
                     "--search", "A1", "--cancel"], env=env, cwd=root)
    assert rc == 0 and "STOP CANCELLED" in out, out


def test_n6_rows_on_the_lcdup15_like_tree():
    sys.path.insert(0, U.PTESTS)
    import test_smoke_inputs_check as TIC
    root = TIC.fresh()
    try:
        rc, lines, doc = TIC.run(root)
        text = "\n".join(lines)
        sheet = _sheet()
        n6 = sheet[sheet.index("# N6."):sheet.index("# N7.")]
        assert ("cohort map (OK: 35 wells, 5 excluded; 2 donors; 4 pairing groups; included per class "
                "control:17,\n#     pathological:13)") in n6
        assert re.search(r"cohort map\s*:?\s*OK.*35 wells, 5 excluded; 2 donors; 4 pairing groups; "
                         r"included per class control:17, pathological:13", text) or \
            ("35 wells, 5 excluded; 2 donors; 4 pairing groups; included per class control:17, "
             "pathological:13") in text, text
        assert re.search(r"every topology folder \(\d+\) holds topology_meta\.json", text), text
        m = re.search(r"(\d+) search\(es\): A1: S-A1, max_cores 148; A2: S-A2-only, max_cores 148; "
                      r"A5: S-A5-only, max_cores 148; the caps add up to 444 cores against "
                      r"resources\.max_cores 444 \(the sequencer's job takes 4 of them\)(.*)$", text, re.M)
        assert m and m.group(1) == "3", text
        # N6 quotes the row with "-- each arm runs 3 jobs of 48, 432 cores in all" inside it
        assert m.group(2) == "", m.group(2)
        assert re.search(r"INPUTS CHECK: 0 FAIL, \d+ WARN", text), text
    finally:
        TIC.cleanup(root)


def test_n6_searches_row_as_quoted(tmp_path):
    """N6: "searches (NOTE: 3 search(es): ...; the caps add up to 444 cores against
    resources.max_cores 444 (the sequencer's job takes 4 of them) -- each arm runs 3
    jobs of 48, 432 cores in all)": the tool prints no such last clause."""
    sys.path.insert(0, U.PTESTS)
    import test_smoke_inputs_check as TIC
    root = TIC.fresh()
    try:
        rc, lines, doc = TIC.run(root)
        row = [l for l in lines if "search(es):" in l]
        assert len(row) == 1 and row[0].rstrip().endswith("432 cores in all"), row
    finally:
        TIC.cleanup(root)


def _campaign_tree(base):
    """A stand-in campaign: two tasks; task0 topo_00000 complete with meta and one
    npz without its json; task0 topo_00001 without meta; task1 topo_00000 complete."""
    C = "campaign_lcdup15_rho1300_neur_n01"
    t0 = os.path.join(base, C, "sweep_cfd_task0000")
    t1 = os.path.join(base, C, "sweep_cpu_task0001")
    for topo, meta, iters in ((os.path.join(t0, "topo_00000"), True, [(0, True), (1, True), (2, False)]),
                              (os.path.join(t0, "topo_00001"), False, [(0, True), (1, False)]),
                              (os.path.join(t1, "topo_00000"), True, [(0, True)])):
        os.makedirs(topo)
        if meta:
            open(os.path.join(topo, "topology_meta.json"), "w").write("{}")
        for i, js in iters:
            open(os.path.join(topo, "iter_%05d.npz" % i), "wb").write(b"x" * (i + 1))
            if js:
                open(os.path.join(topo, "iter_%05d.json" % i), "w").write("{}")
    old = 1700000000
    for dp, dn, fn in os.walk(os.path.join(base, C)):
        os.utime(dp, (old, old))
    return C, t0, t1


def test_n4_n5_on_a_stand_in_campaign(tmp_path):
    base = str(tmp_path / "LindaColloDUP")
    home = str(tmp_path / "home")
    os.makedirs(home)
    C, t0, t1 = _campaign_tree(base)
    sim_main_marker = "/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP"
    n4 = _cmd("N4").replace("cd " + sim_main_marker, "cd " + base)
    n5 = _cmd("N5").replace("cd " + sim_main_marker, "cd " + base)
    assert n4.startswith("cd " + base) and n5.startswith("cd " + base)
    # no scheduler here: N4's `qstat -u $USER | grep ...` finds nothing (its pass);
    # stderr kept apart from the lines the sheet states
    env = dict(os.environ, HOME=home, USER="tester")
    r = subprocess.run(["bash", "-c", n4], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=120)
    assert r.stdout.strip().splitlines()[-1] == ("topology folders 3, without topology_meta.json 1; "
                                                 "simulation files 6, without their json 2"), r.stdout
    m_t0 = os.stat(t0).st_mtime
    m_topo = os.stat(os.path.join(t0, "topo_00000")).st_mtime
    r = subprocess.run(["bash", "-c", n5], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=120)
    A = os.path.join(base, C + "_set_aside")
    # b' = b less the files inside the moved folders: 2 - 1 = 1
    assert r.stdout.strip().splitlines()[-1].startswith(
        "MOVED 1 topology folder(s) and 1 simulation file(s) to %s" % A), r.stdout
    assert os.path.isdir(os.path.join(A, "sweep_cfd_task0000", "topo_00001"))
    assert os.path.isfile(os.path.join(A, "sweep_cfd_task0000", "topo_00000", "iter_00002.npz"))
    assert os.stat(t0).st_mtime == m_t0 and os.stat(os.path.join(t0, "topo_00000")).st_mtime == m_topo
    # outside the campaign glob (sim.campaign_glob of linddup.json)
    glob_ = U.strip_docs(U.raw_json(U.LINDDUP))["sim"]["campaign_glob"]
    assert not fnmatch.fnmatch(os.path.basename(A), glob_)
    assert [os.path.basename(p) for p in glob.glob(os.path.join(base, glob_))] == [C]
    r = subprocess.run(["bash", "-c", n4], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=120)
    assert r.stdout.strip().splitlines()[-1] == ("topology folders 2, without topology_meta.json 0; "
                                                 "simulation files 3, without their json 0"), r.stdout   # 6 - 2 (the moved folder) - 1
