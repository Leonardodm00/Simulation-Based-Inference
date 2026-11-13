"""Reproducer of F206 (t1-B report): the shard jobs of one bank start
together into a folder nobody creates beforehand; `latent_bank.write_shard`
creates it with isdir-then-makedirs (no exist_ok), after the shard's whole
simulation. With the window between the two widened (a 0.5 s sleep inside
os.makedirs, in each builder process), the second of two builds dies with
FileExistsError and its shard is lost -- the stage would count that job as a
resubmission (block 16). Expected by block 16: each shard job builds its
shard.
"""
import os
import subprocess
import sys
import tempfile

import t1b_helpers as H

WRAPPER = r'''
import os, sys, time
sys.path.insert(0, %(stage1)r)
os.chdir(%(stage1)r)
_real = os.makedirs
def _slow(*a, **k):
    time.sleep(0.5)
    return _real(*a, **k)
os.makedirs = _slow
import build_latent_bank as B
sys.argv = ["build_latent_bank.py"] + sys.argv[1:]
sys.exit(B.main())
'''


def test_two_shard_jobs_into_one_fresh_folder():
    root = H.tmpdir("t1b_race_")
    try:
        script = os.path.join(root, "wrapper.py")
        with open(script, "w") as fh:
            fh.write(WRAPPER % {"stage1": H.STAGE1})
        bank = os.path.join(root, "bank")                 # absent: the builders create it
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        procs = [subprocess.Popen([sys.executable, script, "--out-dir", bank, "--arm", "S", "--prior",
                                   "box", "--provider", "reference", "--shard-index", str(k),
                                   "--n-traces", "2", "--n-windows", "1", "--T-win", "1.0", "--fs",
                                   "10.0", "--seed", "1"], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, universal_newlines=True, env=env)
                 for k in (0, 1)]
        outs = [p.communicate()[0] for p in procs]
        codes = [p.returncode for p in procs]
        built = sorted(f for f in os.listdir(bank) if f.endswith(".npz"))
        assert codes == [0, 0] and built == ["shard_0000.npz", "shard_0001.npz"], (
            codes, built, [o.strip().splitlines()[-1] for o in outs])
    finally:
        H.rmtree(root)
