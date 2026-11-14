"""Chain C3 (the evaluation), run t2-C3 of D-131: an upstream artefact
missing or wrong, and what the chain does with it; the configuration of the
committed synth.json v4 (term-only campaigns) through the evaluation. The
scenarios are run by tests/spec/chain_c3_support.py (W1-W5) on the reduced
project; these tests read what they recorded.

    python3 -m pytest tests/spec/test_spec_chain_C3_upstream.py -q
    C3_FIXTURE=<dir> python3 -m pytest tests/spec/test_spec_chain_C3_upstream.py -q

Pure ASCII, LF only.
"""

import glob
import json
import os
import re

from chain_c3_support import fixture


def _cfg(F):
    with open(F["cfg"]) as fh:
        return json.load(fh)


def _need(F, key):
    rec = F["rec"]
    assert key + "_error" not in rec, rec[key + "_error"]
    assert key in rec, "the fixture has no %s" % key
    return rec[key]


def _summary(out):
    m = re.search(r"EVAL (SUBMIT OK|DRY RUN): (\d+) job\(s\) (?:would be )?submitted, (\d+) done, "
                  r"(\d+) live", out)
    assert m, out[-2000:]
    return int(m.group(2)), int(m.group(3)), int(m.group(4))


def _qsub_lines(out, name_prefix):
    return [l.strip() for l in out.splitlines() if l.strip().startswith("qsub -N %s" % name_prefix)]


# ------------------------------------------------------------------ W1: a best fit missing

def test_W1_incomplete_best_fit_is_named_not_sampled():
    """Block 22: an ensemble whose best fit is not complete is named
    (`not complete: <q> (<folder>: ...)`) and not sampled; the plan goes on
    with the others (exit 0)."""
    F = fixture()
    s = _need(F, "W1")
    assert s["rc"] == 0, s["out"][-2000:]
    assert re.search(r"not complete: S-A5-R3 \(.*S-A5-R3/\w+_best: ", s["out"]), s["out"][-3000:]
    assert "sample:S-A5-R3:" not in s["out"], s["out"][-3000:]
    assert "REFUSE" not in s["out"]


# ------------------------------------------------------------------ W2: a wrong member

def test_W2_wrong_member_refused_and_report_not_run():
    """Block 19's oracle and block 22's afterok: a member whose checkpoint is
    another member's does not reproduce its per-row NLL, so its sampling
    refuses (exit 3, REFUSE naming the member, the differences and
    oracle_tol) and writes nothing; the report queued after it does not run,
    so REPORT_COMPLETE.json is unchanged."""
    F = fixture()
    W = _need(F, "W2")
    s = W["submission"]
    assert s["rc"] == 0, s["out"][-2000:]
    jobs = dict((j["name"], j) for j in W["jobs"])
    smp = [j for n, j in jobs.items() if n.startswith("ev_s")]
    assert len(smp) == 1, sorted(jobs)
    assert smp[0]["exit_status"] == 3, (smp[0]["exit_status"], W["logs"][str(smp[0]["num"])][-2000:])
    log = W["logs"][str(smp[0]["num"])]
    assert "REFUSE: member seed 2 does not reproduce" in log and "oracle_tol" in log, log[-2000:]
    rep = jobs.get("ev_report")
    assert rep is not None and rep["depend"] == smp[0]["id"], (rep, smp[0]["id"])
    assert rep["exit_status"] != 0 and rep["started"] == rep["finished"], rep
    assert W["report_sha_after"] == W["report_sha_before"]
    assert W["samples_after"] == W["samples_before"], "the refused sampling wrote files"


def test_W2_after_the_fix_the_plan_is_consistent():
    """With the checkpoint put back, the plan exits 0 and refuses nothing."""
    F = fixture()
    W = _need(F, "W2")
    p = W["plan_after_restore"]
    assert p["rc"] == 0 and "REFUSE" not in p["out"], p["out"][-2000:]


# ------------------------------------------------------------------ W3: --what sample, bank still queued

def test_W3_samplings_wait_for_live_calibration_jobs():
    """Block 22: `sample:<q>:<arm>` ... "after every calibration job
    submitted now or live (-W depend=afterok:<id>:...)". With the
    calibration bank's jobs queued, the samplings must wait for them whether
    or not --what names calbank."""
    F = fixture()
    W = _need(F, "W3")
    assert W["calbank"]["rc"] == 0, W["calbank"]["out"][-2000:]
    cal_ids = [j["id"] for j in W["calbank_jobs"]]
    both = _qsub_lines(W["calbank_sample"]["out"], "ev_s")
    assert both and all(("depend=afterok:" + ":".join(cal_ids)) in l for l in both), both
    only = _qsub_lines(W["sample_only"]["out"], "ev_s")
    assert only, W["sample_only"]["out"][-2000:]
    missing = [l for l in only if "depend=afterok:" not in l]
    assert not missing, ("with --what sample, %d sampling(s) would start before the calibration "
                         "bank's queued jobs %s:\n%s" % (len(missing), cal_ids, missing[0]))


# ------------------------------------------------------------------ W4: a cohort bank rebuilt

def test_W4_rebuilt_cohort_seen_by_the_data_overview():
    """Block 23: the data overview is stale once a cohort's sidecar changed."""
    F = fixture()
    W = _need(F, "W4")
    assert W["rebuild"]["rc"] == 0, W["rebuild"]["tail"]
    out = W["plan"]["out"]
    assert W["plan"]["rc"] == 0, out[-2000:]
    assert "data would be submitted (stale: its banks or its values changed" in out, out[-3000:]
    assert _summary(W["plan_restored"]["out"])[0] == 0, W["plan_restored"]["out"][-2000:]


def test_W4_rebuilt_cohort_and_the_samplings():
    """The samplings that read the rebuilt cohort (S-A1's, scored on every
    cohort; S-A5-R3's) were drawn on its old windows. Block 22's done
    predicate for a sampling (its files, SAMPLE_PARAMS, its best fit's
    BEST_FIT_COMPLETE.json) does not name the banks; this records what the
    launcher says of them, against the data overview, which does."""
    F = fixture()
    W = _need(F, "W4")
    out = W["plan"]["out"]
    stale = [u for u in ("sample:S-A1:A1", "sample:S-A5-R3:A5")
             if ("%s would be submitted" % u) in out]
    assert stale == ["sample:S-A1:A1", "sample:S-A5-R3:A5"], (
        "after R3 was rebuilt, the launcher counts %s done (sampled on the old R3) while the data "
        "overview is stale:\n%s" % (sorted(set(("sample:S-A1:A1", "sample:S-A5-R3:A5")) - set(stale)),
                                    "\n".join(l for l in out.splitlines() if "sample:" in l)))


# ------------------------------------------------------------------ W5: a calibration bank of another build

def test_W5_calibration_bank_of_another_build():
    """Block 22: a calibration-bank folder holding a sidecar of other
    arguments is a refusal, never used; block 19: the sampling checks the
    bank before any sampling. A bank built with n_neurons 50 (the project's
    builder: 100) must not reach a sampling, whichever units --what names."""
    F = fixture()
    W = _need(F, "W5")
    assert all(b["rc"] == 0 for b in W["builds"]), W["builds"]
    full = W["plan_calbank_sample"]
    assert full["rc"] == 3 and "holds a calibration bank of another build" in full["out"], \
        full["out"][-2000:]
    only = W["plan_sample_only"]
    assert only["rc"] == 3, ("--what sample: exit %d, the bank of another build not seen:\n%s"
                             % (only["rc"], only["out"][-1500:]))


def test_W5_sampler_refuses_a_calibration_bank_of_another_build():
    F = fixture()
    W = _need(F, "W5")
    s = W["sampler"]
    assert s["rc"] == 3 and "REFUSE" in s["tail"], (
        "ev_sample.py with a calibration bank of n_neurons 50 (the simulated bank's 100): exit %d\n%s"
        % (s["rc"], s["tail"][-1200:]))


# ------------------------------------------------------------------ v4: the term-only campaigns

def _paired_files(F):
    out = F["rec"]["A"]["out_root"]
    with open(os.path.join(out, "metrics", "METRICS.json")) as fh:
        idx = json.load(fh)
    return sorted(os.path.basename(f) for f in idx["files"]
                  if os.path.basename(f).startswith("paired_") and f.endswith(".json"))


def test_v4_mixed_search_with_alt_best_fit_is_paired():
    """Sec. 2.5 (E3), block 20: a search with a term whose two arms'
    ensembles are reported (here S-A2x-R0, campaign S-A2, alt_best_fit true)
    gets the donor-paired comparison on every data set the two share."""
    F = fixture()
    got = _paired_files(F)
    assert "paired_S-A2x-R0_R0.json" in got and "paired_S-A2x-R0_sim.json" in got, got
    figs = glob.glob(os.path.join(F["rec"]["A"]["out_root"], "figures", "searches", "S-A2x-R0",
                                  "arms_forest_R0.*"))
    assert figs, "no forest plot for the paired search"


def test_v4_term_only_searches_have_a_paired_comparison():
    """Sec. 2.5 (D-121, D-126): the evaluation compares, on L_pseudo_real
    paired by donor, the term arm's ensemble with arm A1's (E3: Dbar,
    its bootstrap interval, 2 sigma_seed, the G.2 verdict; block 21's forest
    plot and across/dbar). Under the committed synth.json v4 (block 26, D-132
    (4)) the term arms A2 and A5 are optimised by their own searches
    (S-A2-only, S-A5-only) and A1 by S-A1, with no fit of another arm. The
    reduced project carries that shape (S-A2-R0 on S-A2-only, S-A5-R3 on
    S-A5-only, S-A1 scored on R0 and R3): their term arms must still be
    compared with A1 on their cohort, or the evaluation's headline result is
    absent for every v4 search."""
    F = fixture()
    got = _paired_files(F)
    term_only = [q["name"] for q in _cfg(F)["synthetic"]["searches"]
                 if q["campaign"] in ("S-A2-only", "S-A5-only")]
    covered = [q for q in term_only if any(f.startswith("paired_%s_" % q) for f in got)]
    assert covered == term_only, (
        "E3 computed for %s only; none for the term-only searches %s (the committed v4's four "
        "term searches)" % (sorted(set(f.split("_")[1] for f in got)), sorted(set(term_only) - set(covered))))


# ------------------------------------------------------------------ the ledger's bytes lost

def test_W3_unreadable_ledger_does_not_resubmit_queued_jobs():
    """Block 22: a unit is live while its job in <out_root>/jobs/ledger.json
    is queued, held or running, and a live unit is not submitted twice
    (E22c). With the calibration bank's two jobs queued and the ledger's
    bytes lost (a write a node fault left empty; t1-D's F701 met it from the
    purge's side), the plan must not submit those units again: it either
    still sees them live or refuses."""
    F = fixture()
    W = _need(F, "W3")
    p = W["plan_empty_ledger"]
    again = [l for l in p["out"].splitlines() if re.search(r"calbank:\d+ would be submitted", l)]
    assert p["rc"] == 3 or not again, (
        "with the ledger unreadable the plan exits %d and would submit %d calibration job(s) "
        "again while %s are queued:\n%s" % (p["rc"], len(again),
                                            [j["id"] for j in W["calbank_jobs"]], "\n".join(again)))
