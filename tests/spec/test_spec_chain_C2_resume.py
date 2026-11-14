"""Chain t2-C2, step 6: the chain interrupted at stage boundaries and
resumed through its real entry point (C-O11; sec. 2.2, block 1 (c), D-098):

  1. submitted; once the three banks are done, `--stop --wait` (the stop is
     honoured at a tick boundary; the jobs keep running);
  2. submitted again; once search_S-A1 has trial jobs, its driver job is
     deleted with qdel (PBS ending the job: the successor continues);
  3. once best_fit_S-A1 has member jobs, `--stop --wait` again;
  4. submitted again, to the end.

Expected: the chain ends `done`, every stage job submitted once (the same
jobs as the uninterrupted run1), the artefacts' numbers those of run1.
"""

import json
import os
import shutil
import time

import pytest

import c2_helpers as H

ROOT_NAME = "run_int"


def _state(cfg):
    try:
        return H.state(cfg)
    except (OSError, ValueError):
        return None


def _stage(cfg, name):
    st = _state(cfg)
    return (st or {}).get("stages", {}).get(name, {})


def _jobs_named(spool, prefix):
    return [j for j in H.jobs(spool) if j["name"].startswith(prefix)]


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory):
    base = os.environ.get("C2_ROOT")
    root = os.path.join(os.path.abspath(base), ROOT_NAME) if base else \
        str(tmp_path_factory.mktemp(ROOT_NAME))
    marker = os.path.join(root, "steps.json")
    if os.path.isfile(marker):
        with open(marker) as fh:
            steps = json.load(fh)
        cfg_path = os.path.join(root, "project", "synth.json")
        with open(cfg_path) as fh:
            cfg = H._strip(json.load(fh))
        spool = os.path.join(root, "spool")
        if H.chain_end(cfg) and not H.live_jobs(spool):
            return cfg_path, cfg, spool, H.fake_env(spool), steps
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    cfg_path, cfg = H.reduced(os.path.join(root, "project"))
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    steps = {}

    # 1. the banks, then a stop at the boundary
    rc, out = H.entry(cfg_path, env)
    steps["submit1"] = [rc, out[-400:]]
    ok = H.wait_until(lambda: all(_stage(cfg, s).get("status") == "done"
                                  for s in ("sim_bank", "real_bank_R0", "real_bank_R3")),
                      timeout=900)
    steps["banks_done"] = bool(ok)
    rc, out = H.entry(cfg_path, env, "--stop", "--wait", "300")
    steps["stop1"] = [rc, out[-600:]]
    steps["chain_end_after_stop1"] = H.chain_end(cfg)
    steps["jobs_after_stop1"] = len(H.jobs(spool))
    time.sleep(5)

    # 2. again; the driver deleted while S-A1's trials run
    rc, out = H.entry(cfg_path, env)
    steps["submit2"] = [rc, out[-400:]]
    ok = H.wait_until(lambda: any(j["name"].startswith("t_") and j["state"] == "R"
                                  for j in H.jobs(spool)), timeout=900)
    steps["trial_running"] = bool(ok)
    drivers = [j for j in H.jobs(spool) if j["name"] == "pl_synth" and j["state"] == "R"]
    steps["qdel_driver"] = drivers[-1]["id"] if drivers else None
    if drivers:
        rc, out = H.run(["qdel", drivers[-1]["id"]], env)
        steps["qdel"] = [rc, out]

    # 3. a stop while best_fit_S-A1's members run
    ok = H.wait_until(lambda: any(j["name"].startswith("bf_") for j in H.jobs(spool)),
                      timeout=1800)
    steps["members_started"] = bool(ok)
    rc, out = H.entry(cfg_path, env, "--stop", "--wait", "300")
    steps["stop2"] = [rc, out[-600:]]
    steps["chain_end_after_stop2"] = H.chain_end(cfg)
    time.sleep(5)

    # 4. again, to the end
    rc, out = H.entry(cfg_path, env)
    steps["submit3"] = [rc, out[-400:]]
    ce = H.wait_chain(cfg, spool, timeout=2400)
    steps["chain_end"] = ce
    with open(marker, "w") as fh:
        json.dump(steps, fh, indent=1)
    return cfg_path, cfg, spool, env, steps


def test_each_interruption_happened(interrupted):
    _, cfg, spool, env, steps = interrupted
    assert steps["submit1"][0] == 0 and steps["banks_done"]
    assert steps["stop1"][0] == 0, steps["stop1"]
    assert "stopped" in (steps["chain_end_after_stop1"] or ""), steps["chain_end_after_stop1"]
    assert steps["submit2"][0] == 0 and steps["trial_running"] and steps["qdel_driver"]
    assert steps["members_started"]
    assert steps["stop2"][0] == 0, steps["stop2"]
    assert "stopped" in (steps["chain_end_after_stop2"] or ""), steps["chain_end_after_stop2"]
    assert steps["submit3"][0] == 0


def test_resumed_chain_ends_done_with_every_job_once(interrupted, run1):
    """C-O11: no stage job twice; the same set of stage jobs as run1."""
    _, cfg, spool, env, steps = interrupted
    assert steps["chain_end"] and "done (exit 0)" in steps["chain_end"], steps["chain_end"]
    assert "PIPELINE DONE" in H.log_text(cfg)
    jobs = [j for j in H.jobs(spool) if j["name"] != "pl_synth"]
    outs = [j["out"] for j in jobs]
    assert len(outs) == len(set(outs)), sorted(o for o in outs if outs.count(o) > 1)
    base = sorted(j["name"] for j in H.jobs(run1[2]) if j["name"] != "pl_synth")
    assert sorted(j["name"] for j in jobs) == base
    for j in jobs:
        assert j["exit_status"] == 0, (j["name"], j["exit_status"])


def test_resumed_chain_same_numbers_as_run1(interrupted, run1):
    """The resumed run's results are the uninterrupted run's (one thread,
    the same seeds): the finalists and the ensembles."""
    _, cfg, spool, env, steps = interrupted
    c1 = run1[1]
    for q in H.SEARCHES:
        s_a = H.read_json(os.path.join(H.campaign_dir(cfg, q), "SEARCH_COMPLETE.json"))
        s_b = H.read_json(os.path.join(H.campaign_dir(c1, q), "SEARCH_COMPLETE.json"))
        assert s_a["finalist"] == s_b["finalist"], q["name"]
        for alt in ((False, True) if q["alt_best_fit"] else (False,)):
            e_a = H.read_json(os.path.join(H.best_dir(cfg, q, alt), "BEST_FIT_COMPLETE.json"))
            e_b = H.read_json(os.path.join(H.best_dir(c1, q, alt), "BEST_FIT_COMPLETE.json"))
            assert e_a["ensemble"]["L_bar"] == e_b["ensemble"]["L_bar"], (q["name"], alt)
