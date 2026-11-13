"""Block 14's validation target at the decided shape (SPEC.md block 14, "End
to end on the bench generator at the decided shape"): 180-s windows at 100
Hz (W 18,000); arm S 120 theta x 2 wells x 1 window; R0 and R3 of 3 x 9
donors x 2 wells x 6 windows, regular centres x 0.6, R3 at pi = 0.3 with the
three modes; A1, A2 and A5 on R0 and A5 on R3 exit 0 at 2 epochs of 3 steps,
with 972 cross-well pairs (27 donors x 6 x 6) and the diagnostic's 64 over
all 27 donors.
"""
import json
import os
import subprocess
import time

import pytest

import t1b_helpers as H

pytest.importorskip("torch")

COMMON = ["--provider", "bench", "--fs", "100.0", "--T-win", "180.0", "--n-neurons", "100",
          "--wells-per-donor", "2", "--donors-per-batch", "8", "--n-classes", "3", "--tau-ov", "0.1",
          "--class-centres", "regular", "--centre-scale", "0.6"]


@pytest.fixture(scope="module")
def decided():
    root = H.tmpdir("t1b_decided_")
    out = {"root": root, "build_s": {}}
    specs = {
        "S": ["--arm", "S", "--prior", "box", "--n-traces", "240", "--n-windows", "1", "--seed", "1000"],
        "R0": ["--arm", "R", "--donors-per-class", "9", "--n-traces", "54", "--n-windows", "6",
               "--pi", "0.0", "--gap-modes", "range_shift,drift,contamination", "--seed", "2000"],
        "R3": ["--arm", "R", "--donors-per-class", "9", "--n-traces", "54", "--n-windows", "6",
               "--pi", "0.3", "--gap-modes", "range_shift,drift,contamination", "--seed", "3000"],
    }
    for name, args in specs.items():
        t0 = time.time()
        rc, txt = H.run_builder(os.path.join(root, name), args + COMMON, timeout=3600)
        out["build_s"][name] = time.time() - t0
        assert rc == 0, txt[-2000:]
        out[name] = os.path.join(root, name, "shard_*.npz")
    yield out
    H.rmtree(root)


def _run(arm, sim, real, out_dir):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [H.PYTHON, H.RUNNER, "--arm", arm, "--seed", "0", "--sim-shards", sim, "--real-shards", real,
           "--out-dir", out_dir, "--threads", "1", "--epochs", "2", "--steps-per-epoch", "3",
           "--b-sim", "24", "--b-met", "12", "--b-rep", "4", "--n-posterior-draws", "16",
           "--embedding-size", "8", "--hidden-features", "24", "--num-transforms", "3",
           "--depth-exponent", "3", "--width-multiplier", "1.5"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=H.STAGE3, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=3600)
    return r.returncode, r.stdout, time.time() - t0


@pytest.mark.parametrize("arm,real", [("A1", "R0"), ("A2", "R0"), ("A5", "R0"), ("A5", "R3")])
def test_b14_decided_shape(decided, arm, real):
    out_dir = os.path.join(decided["root"], "runs_%s_%s" % (arm, real))
    rc, txt, secs = _run(arm, decided["S"], decided[real], out_dir)
    print("%s on %s: exit %d in %.1f s (builds %s)" % (arm, real, rc, secs,
                                                    {k: round(v, 1) for k, v in decided["build_s"].items()}))
    assert rc == 0, txt[-3000:]
    assert "replicate pairs  : 972  (two wells of one donor)" in txt
    rec = json.load(open(os.path.join(out_dir, "%s_seed0.json" % arm)))
    assert rec["prior"] == "box_unit_box" and rec["L0"] == 0.0
    assert rec["replicate"]["n_pairs"] == 64 and rec["replicate"]["n_donors_scored"] == 27
    if arm == "A2":
        assert "metric classes   : 3 (on the real arm)" in txt
