"""Chain C1 (the real-data project): where the chain must stop -- integration
tester, audit D-131, run t2-C1. Oracles C1-F1..F5, C1-R3..R5 and C1-H1 of
the report (SPEC.md secs. 2.1, 2.2, 3.1, 3.2; blocks 1-3).

Each case builds the stand-in of `chain_c1_harness.py` (a recorded data set,
S1-S8 to run), makes one input missing or wrong, and drives the chain through
`hpc/pipeline/run_pipeline.sh` (the D-098 driver jobs, fake PBS).

    SBIX_CLONE=<Sbi-extractor checkout> python3 -m pytest -q tests/spec/test_spec_chain_C1_refusals.py

About 5 minutes. Pure ASCII, LF only.
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import chain_c1_harness as H      # noqa: E402

if H.ST.sbix_clone() is None:
    pytest.skip("no Sbi-extractor checkout (set SBIX_CLONE)", allow_module_level=True)

NOTES = {}


@pytest.fixture
def S():
    s = H.ChainStandin()
    yield s
    if os.environ.get("C1_KEEP") != "1":
        s.close()


def _stage_jobs(s):
    return [j for j in s.jobs() if not j["name"].startswith("pl_")]


def test_s1_artefact_with_a_wrong_sidecar_stops_the_chain_at_s1(S):
    """C1-H1 + C1-F2 (sec. 2.1): a manifest whose sidecar does not match is
    not done; S1 runs, its launcher refuses a root that holds a manifest,
    and the chain stops there with that text -- nothing else submitted."""
    s = S
    man = os.path.join(s.ds["extract_root"], "cohort_manifest.json")
    os.makedirs(os.path.dirname(man), exist_ok=True)
    with open(os.path.join(s.golden, "cohort_manifest.json"), "rb") as fh:
        body = fh.read()
    with open(man, "wb") as fh:
        fh.write(body)
    with open(man + ".sha256", "w") as fh:
        fh.write("%s  cohort_manifest.json\n" % ("0" * 64))
    rc, out = s.sh([], timeout=300)
    assert rc == 0, out[-2000:]
    end = s.wait_chain_end(900)
    log = s.log_text()
    NOTES["s1_wrong_sidecar"] = {"end": end, "statuses": s.statuses(),
                                 "stage_jobs": [j["name"] for j in _stage_jobs(s)],
                                 "stop_lines": [l for l in log.splitlines() if "STOP" in l]}
    assert end and "failed (exit 3)" in end, end
    assert s.statuses()["real_extraction"] == "failed"
    assert any("STOP: stage real_extraction" in l and "already holds a cohort_manifest.json" in l
               for l in log.splitlines()), log[-3000:]
    assert not _stage_jobs(s), [j["name"] for j in _stage_jobs(s)]
    # sec. 2.2 (4): a re-run stops again with the same report (no --retry)
    n0 = len(s.jobs())
    rc, out = s.sh(["--dry-run"], timeout=300)
    assert rc == 3 and "FAILED in the state file" in out and "--retry real_extraction" in out, \
        out[-2000:]
    assert len(s.jobs()) == n0


def test_missing_s2_profile_is_refused_before_anything_runs(S):
    """C1-F3 (block 3): S2 without sim_reextract/profiles/<tag>.sh is refused
    before anything runs."""
    s = S
    os.remove(os.path.join(s.sbix, "sim_reextract", "profiles", "%s.sh" % H.TAG))
    rc_dry, out_dry = s.sh(["--dry-run"], timeout=300)
    rc, out = s.sh([], timeout=300)
    end = s.wait_chain_end(900) if rc == 0 else None
    log = s.log_text()
    NOTES["s2_no_profile"] = {"dry_rc": rc_dry, "dry_refuse": [l for l in out_dry.splitlines()
                                                              if "REFUSE" in l or "profile" in l],
                              "submit_rc": rc, "submit_tail": out[-1500:], "end": end,
                              "statuses": s.statuses(),
                              "stage_jobs": [j["name"] for j in _stage_jobs(s)],
                              "stop_lines": [l for l in log.splitlines() if "STOP" in l
                                             or "REFUSE" in l]}
    assert not _stage_jobs(s), NOTES["s2_no_profile"]
    assert "profiles/%s.sh" % H.TAG in (out + log), NOTES["s2_no_profile"]


def test_bank_plan_altered_after_s4_stops_the_bank(S):
    """C1-F5 (sec. 2.1: "never a silent reuse"): the plan of record altered
    (its sidecar no longer matches) after S4 is done and while S5 runs; the
    chain resumed. The simulated bank must not be built from it: S5 stops
    with the refusal of the tool that reads it."""
    s = S
    s.restore_inputs()
    rc, out = s.sh([], timeout=300)
    assert rc == 0, out[-2000:]
    assert s.wait_for(lambda: s.statuses().get("bank_plan") == "done", 900)
    rc, out = s.sh(["--stop", "--wait", "300"], timeout=400)
    plan = os.path.join(s.sbix, "export_windows", H.TAG, "bank_plan.json")
    with open(plan, "a") as fh:
        fh.write("\n")
    st_before = s.statuses()
    rc, out = s.sh([], timeout=300)
    end = s.wait_chain_end(1800)
    log = s.log_text()
    rec = os.path.join(s.cfg["paths"]["banks_root"], "synth_sim", "BANK_RECORD.json")
    NOTES["plan_altered"] = {"statuses_before_resume": st_before, "end": end,
                             "statuses": s.statuses(), "record_exists": os.path.isfile(rec),
                             "stop_lines": [l for l in log.splitlines() if "STOP" in l
                                            or "ABORT" in l or "REFUSED" in l][-12:]}
    assert end and "failed (exit 3)" in end, NOTES["plan_altered"]
    assert not os.path.isfile(rec), NOTES["plan_altered"]


def test_existing_plan_without_matching_sidecar_on_a_fresh_state(S):
    """C1-F5: a plan whose sidecar does not match, on a fresh state, is not
    done: S4 runs and its tool refuses to overwrite it -- a stop at
    bank_plan, nothing of S5-S8 submitted."""
    s = S
    s.restore_inputs()
    plan = os.path.join(s.sbix, "export_windows", H.TAG, "bank_plan.json")
    os.makedirs(os.path.dirname(plan), exist_ok=True)
    with open(plan, "w") as fh:
        fh.write('{"half": "written"}\n')
    with open(plan + ".sha256", "w") as fh:
        fh.write("%s  bank_plan.json\n" % ("1" * 64))
    rc, out = s.sh([], timeout=300)
    assert rc == 0, out[-2000:]
    end = s.wait_chain_end(900)
    log = s.log_text()
    NOTES["plan_bad_sidecar"] = {"end": end, "statuses": s.statuses(),
                                 "stage_jobs": [j["name"] for j in _stage_jobs(s)],
                                 "stop_lines": [l for l in log.splitlines() if "STOP" in l][-5:]}
    st = s.statuses()
    assert end and "failed (exit 3)" in end, NOTES["plan_bad_sidecar"]
    assert st["bank_plan"] == "failed" and all(st[n] == "skipped" for n in
                                               ("real_extraction", "sim_redetection",
                                                "label_freeze"))
    assert all(st[n] == "pending" for n in ("sim_bank", "real_bank", "search", "best_fit")), st
    assert not _stage_jobs(s)


def test_configuration_refused_at_the_entry_point(S):
    """C1-F1 (sec. 3.1): a missing key, an unknown key and a wrong type, each
    listed, exit 2, nothing submitted."""
    s = S
    del s.cfg["bank"]["floor_hz"]
    s.cfg["search"]["unknown_knob"] = 1
    s.cfg["best_fit"]["epochs"] = "40"
    s.write_cfg()
    rc, out = s.sh([], timeout=300)
    assert rc == 2, out[-2000:]
    assert "bank.floor_hz" in out and "unknown_knob" in out and "best_fit.epochs" in out, out
    assert "3 problem(s)" in out, out
    assert not s.jobs()


def test_configuration_change_needs_the_flag_and_a_second_sequencer_is_refused(S):
    """C1-R5 and C1-R4 at the entry point: a second submission while the
    sequencer's job is queued or running is refused (exit 4); once it has
    stopped, a changed configuration is refused (exit 2) without
    --accept-config-change and accepted with it."""
    s = S
    s.restore_inputs()
    rc, out = s.sh([], timeout=300)
    assert rc == 0 and "SUBMITTED" in out, out
    rc2, out2 = s.sh([], timeout=300)
    assert rc2 == 4 and "REFUSE: the sequencer is already queued or running" in out2, out2
    rc, out = s.sh(["--stop", "--wait", "300"], timeout=400)
    assert rc == 0, out
    s.wait_for(lambda: not [j for j in s.live_jobs() if j["name"].startswith("pl_")], 120)
    s.cfg["driver"]["poll_seconds"] = 2
    s.write_cfg()
    n0 = len(s.jobs())
    rc3, out3 = s.sh([], timeout=300)
    assert rc3 == 2 and "REFUSE: the configuration changed" in out3, out3
    assert len(s.jobs()) == n0
    rc4, out4 = s.sh(["--accept-config-change"], timeout=300)
    assert rc4 == 0 and "SUBMITTED" in out4, out4
    # the acceptance reaches the state when the first driver runs (block 1 (e))
    want = H.sha256_file(s.path)
    ok = s.wait_for(lambda: (s.state() or {}).get("config_sha256") == want, 300, every=0.5)
    rc, out = s.sh(["--stop", "--wait", "300"], timeout=400)
    s.wait_for(lambda: not s.live_jobs(), 600)
    assert ok, (s.state() or {}).get("config_sha256")


def teardown_module(module):
    out = os.environ.get("C1_REFUSALS_OUT")
    if out:
        with open(out, "w") as fh:
            json.dump(NOTES, fh, indent=1, default=str)
