"""Chain C1 (the real-data project): a search run again after the chain has
finished -- integration tester, audit D-131, run t2-C1. Step 9's "an
artefact reused after its upstream changed": S8's BEST_FIT_COMPLETE.json
against S7's SEARCH_COMPLETE.json once the search is reset and run again
(block 7, `tools/search_reset.py`, whose pass line says "next: submit with
--retry search --accept-config-change"; sec. 2.2's resume rule).

The chain runs end to end on the stand-in of `chain_c1_harness.py` through
`hpc/pipeline/run_pipeline.sh`; then the search is reset with the tool's
CODE, `search.gp_seed` is changed (the reset's documented use, D-101: "the
restarted search draws 50 new configurations"), and the submission is made
as the tool says. Observed: the finalist of the new search, and whether the
best fit that the run reports is that finalist's.

    SBIX_CLONE=<Sbi-extractor checkout> python3 -m pytest -q tests/spec/test_spec_chain_C1_retry.py

About 4 minutes. Pure ASCII, LF only.
"""

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import chain_c1_harness as H      # noqa: E402

pytest.importorskip("torch")
pytest.importorskip("sbi")
pytest.importorskip("skopt")
if H.ST.sbix_clone() is None:
    pytest.skip("no Sbi-extractor checkout (set SBIX_CLONE)", allow_module_level=True)


def test_best_fit_after_the_search_is_run_again():
    s = H.ChainStandin()
    try:
        s.restore_inputs()
        rc, out = s.sh([], timeout=300)
        assert rc == 0, out[-2000:]
        end = s.wait_chain_end(2400)
        assert end and " done (exit 0)" in end, end
        d = s.cfg
        camp = os.path.join(d["paths"]["tune_root"], "S-A1")
        best = os.path.join(d["paths"]["runs_root"], "synth_best")
        bf0 = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
        sc0 = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        assert bf0["finalist"]["trial_id"] == sc0["finalist"]["trial_id"]
        rc, lst = s.tool("tools/search_reset.py", ["--config", s.path])
        code = re.search(r"CODE ([0-9a-f]{8})", lst).group(1)
        rc, out = s.tool("tools/search_reset.py", ["--config", s.path, "--confirm", code])
        assert rc == 0 and "SEARCH RESET DONE" in out, out[-2000:]
        s.cfg["search"]["gp_seed"] = 17
        s.write_cfg()
        n0 = len(s.jobs())
        rc, out = s.sh(["--retry", "search", "--accept-config-change"], timeout=300)
        assert rc == 0, out[-2000:]
        s.wait_for(lambda: len(s.jobs()) > n0 + 1 and s.chain_end(), 2400, every=2.0)
        end = s.wait_chain_end(2400)
        sc1 = json.load(open(os.path.join(camp, "SEARCH_COMPLETE.json")))
        bf1 = json.load(open(os.path.join(best, "BEST_FIT_COMPLETE.json")))
        new_names = [j["name"] for j in s.jobs()[n0:]]
        rc, st = s.sh(["--status"], timeout=300)
        obs = {"end": end, "statuses": s.statuses(),
               "search_finalist_before": sc0["finalist"]["trial_id"],
               "search_finalist_after": sc1["finalist"]["trial_id"],
               "best_fit_finalist": bf1["finalist"]["trial_id"],
               "member_jobs_after_retry": [n for n in new_names if n.startswith("bf_")],
               "status_best_fit": [l for l in st.splitlines() if "| best_fit" in l]}
        print(json.dumps(obs, indent=1))
        assert end and " done (exit 0)" in end, obs
        if sc1["finalist"]["trial_id"] == sc0["finalist"]["trial_id"]:
            pytest.skip("the new search found the same finalist: nothing to compare (%s)" % obs)
        # the run reports PIPELINE DONE: its best fit should be the new finalist's
        assert bf1["finalist"]["trial_id"] == sc1["finalist"]["trial_id"], obs
    finally:
        if os.environ.get("C1_KEEP") != "1":
            s.close()
