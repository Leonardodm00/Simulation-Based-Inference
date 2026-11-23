#!/usr/bin/env python3
"""The fixture of the recorded evaluation's tests (block 33; specs/SPEC.md
block 33, "The fixture"): built once per process, a few minutes in the
sandbox.

    F = rec_fixture.fixture()
    F["root"]       the tree
    F["cfg"]        the reduced synthetic project's configuration (synth.json)
    F["ecfg_old"]   the synthetic evaluation's configuration (blocks 19-22)
    F["old_out"]    its out_root: the samples and metrics of blocks 19-20
    F["ecfg_as"]    rec_eval_as.json: the project read as recorded (D-162)
    F["cfg_rec"]    a recorded-kind configuration on the same banks and runs
    F["ecfg_rec"]   rec_eval.json: its evaluation's configuration

REC_FIXTURE=<dir> reuses a tree built before; REC_KEEP=1 keeps the tree
this process builds and prints where. Needs the training environment
(torch, sbi, zuko, scikit-optimize); `need_training_env()` raises
unittest.SkipTest without it. Not pipeline code: a test fixture.

CONTRACT (wave A): the keys above are fixed by specs/SPEC.md block 33. The
functions are stubs.

Pure ASCII, LF only.
"""

FIXTURE_KEYS = ("root", "cfg", "ecfg_old", "old_out", "ecfg_as", "cfg_rec", "ecfg_rec")


def need_training_env():
    """unittest.SkipTest when torch, sbi, zuko or skopt cannot be imported."""
    raise NotImplementedError("block 33")


def fixture():
    """dict with FIXTURE_KEYS (and `built`, True when this process built the
    tree): the fixture of specs/SPEC.md block 33, built once per process."""
    raise NotImplementedError("block 33")


def cleanup():
    """Removes the tree this process built, unless REC_KEEP=1 (then prints
    where it is). A tree given by REC_FIXTURE is never removed."""
    raise NotImplementedError("block 33")
