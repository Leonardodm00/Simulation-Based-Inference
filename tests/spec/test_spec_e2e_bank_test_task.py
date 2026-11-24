"""Step 6 for block 2's amendment 4: the bank stages through the real entry
point (`pipeline.py run`) on the stand-in davinci (the Sbi-extractor tools of
SBIX_CLONE copied into a temporary tree, the fake PBS), task 0's topologies
partial with drop_partial_topologies true, so that the plan's task 0 holds no
simulation.

Oracles (tester run 2026-11-17_1730):
  E2.1 the sequencer stopped while the test task is queued and run again
       re-attaches to it (the state keeps its index): the test task is task 1,
       submitted once (bank_test, TASK_INDEX 1, log bank_test_1.log, `TASK 1
       OK`), then the array and its gate and the verification job: 4 jobs in
       all; WALLTIME=auto read from task 1's marker; PIPELINE DONE; the
       certificate.
  E2.2 two runs from scratch in the same folder build the same bank: every
       shard's sha256, the plan's sha256 and the record's rows equal.
"""

import glob
import hashlib
import json
import os
import shutil
import sys

import pytest

import _spec_util as U

sys.path.insert(0, U.PTESTS)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _bank_digest(s):
    shards = sorted(glob.glob(os.path.join(s.bank, "shard_*.npz")))
    rec = json.load(open(os.path.join(s.bank, "BANK_RECORD.json")))
    plan = json.load(open(s.plan))
    # the plan's own metadata (Sbi-extractor's: its creation time, the free space
    # it measured) changes between runs; its tasks and rows must not
    for k in ("created", "free_space"):
        plan.pop(k, None)
    return {"plan": json.dumps(plan, sort_keys=True), "shards": [(os.path.basename(p), _sha(p)) for p in shards],
            "n_rows": rec.get("n_rows"), "n_shards": len(shards)}


@pytest.fixture
def standin():
    if not os.path.isdir(U.SBIX_CLONE):
        pytest.skip("no Sbi-extractor checkout (SBIX_CLONE)")
    import test_smoke_block2 as T2
    s = T2.Standin(bank_walltime="auto")
    T2._partial_task0(s)
    try:
        yield s, T2
    finally:
        s.close()


def test_task0_empty_reattach_and_determinism(standin):
    s, T2 = standin
    # E2.1: three ticks while the fake PBS holds the test task 4 s before it starts
    rc1, out1 = s.run(["--max-ticks", "3"], timeout=600, delay="4.0")
    assert rc1 == 0 and "tick limit 3 reached" in out1, out1[-3000:]
    assert "sim_bank: recorded bank test task 1 -> " in out1, out1[-3000:]
    assert s.n_submitted() == 1
    st = json.load(open(os.path.join(s.cfg["paths"]["work_root"], "state.json")))
    assert st["stages"]["sim_bank"]["sub"]["test_index"] == 1
    plan = json.load(open(s.plan))
    n_iters = [t["n_iters"] for t in plan["tasks"]]
    assert n_iters[0] == 0 and all(n > 0 for n in n_iters[1:]), n_iters
    rc2, out2 = s.run([], timeout=900)
    assert rc2 == 0 and "PIPELINE DONE" in out2 and "sim_bank: VERIFIED:" in out2, out2[-3000:]
    jobs = s.jobs()
    assert [r["name"] for r in jobs] == ["bank_test", "bank_task", "bank_gate", "bank_verify"], jobs
    assert jobs[0]["vars"]["TASK_INDEX"] == "1"
    assert jobs[0]["out"].endswith(os.path.join("logs", "bank_test_1.log"))
    assert "TASK 1 OK" in open(jobs[0]["out"]).read()
    assert (out1 + out2).count("recorded bank test task") == 1
    assert ("WALLTIME=auto: test task 1 read %d simulations" % n_iters[1]) in out2, out2[-3000:]
    assert "sim_bank: test task passed: TASK 1 OK" in out2
    assert os.path.isfile(os.path.join(s.bank, "BANK_VERIFIED.json"))
    first = _bank_digest(s)
    # E2.2: the same command from scratch in the same folder (state, plan and banks
    # moved out of the way, nothing else changed)
    trash = os.path.join(s.root, "previous_run")
    os.makedirs(trash)
    for p in (os.path.join(s.cfg["paths"]["work_root"]), os.path.dirname(s.plan),
              os.path.join(s.cfg["paths"]["banks_root"])):
        shutil.move(p, os.path.join(trash, os.path.basename(p.rstrip("/")) + "_%d" % len(os.listdir(trash))))
    # the pre-made S7/S8 artefacts of the stand-in live outside these folders
    rc3, out3 = s.run([], timeout=900)
    assert rc3 == 0 and "PIPELINE DONE" in out3, out3[-3000:]
    second = _bank_digest(s)
    assert second == first, (first, second)
