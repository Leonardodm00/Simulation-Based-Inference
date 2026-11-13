"""Boundaries of blocks 18-23 with their neighbours (blocks 1, 7, 15, 16, 17),
from this run's side: what the producers actually wrote on the reduced
project against what blocks 18-23 read (SPEC.md blocks 7, 16, 19, 20, 23 and
the notation table).

Pure ASCII, LF only.
"""

import glob
import math
import os

import numpy as np
import pytest

from conftest import load_json

RECORD_KEYS = ("arm", "seed", "split_seed", "split_fracs", "split_hash", "config", "L",
               "L_pseudo_real", "history")
CONFIG_KEYS = ("depth_exponent", "width_multiplier", "block_family", "head_fusion", "dropout",
               "hidden_features", "num_transforms", "num_bins", "seed")


def _folders(project):
    return sorted(glob.glob(os.path.join(project["root"], "runs_root", "*", "synth_best*")))


def _shards(project, bank):
    return sorted(glob.glob(os.path.join(project["root"], "banks_root", "synth_%s" % bank,
                                         "shard_*.npz")))


def test_best_fit_folders_and_member_files(project):
    """block 7 -> 19: BEST_FIT_COMPLETE.json with member_seeds; per member
    `<stem>.json`, `<stem>_ckpt.pt`, `<stem>_perrow.npz` in member_seed<k>/,
    the stem `<arm>_seed<k>` (k = split seed) or `<arm>_split<s>_seed<k>`."""
    for f in _folders(project):
        b = load_json(os.path.join(f, "BEST_FIT_COMPLETE.json"))
        assert b["member_seeds"]
        for k in b["member_seeds"]:
            recs = glob.glob(os.path.join(f, "member_seed%d" % k, "*_seed%d.json" % k))
            assert len(recs) == 1, recs
            r = load_json(recs[0])
            stem = os.path.basename(recs[0])[:-5]
            want = ("%s_seed%d" % (r["arm"], k) if int(r["split_seed"]) == int(k)
                    else "%s_split%d_seed%d" % (r["arm"], int(r["split_seed"]), k))
            assert stem == want, (stem, want)
            for suffix in ("_ckpt.pt", "_perrow.npz"):
                assert os.path.isfile(recs[0][:-5] + suffix), suffix
            for key in RECORD_KEYS:
                assert key in r, (recs[0], key)
            for key in CONFIG_KEYS:
                assert key in r["config"], (recs[0], key)
            assert len(r["history"]) >= 1 and all("val_npe" in h for h in r["history"])


def test_perrow_contract_and_alignment(project):
    """block 7's per-row file: nll (n,) float64, theta (n, d) float32 in the
    open box, group (n,) int64 -- the report rows of the member's split in
    global row order: theta and group equal the bank's at those rows (what
    block 19's D_sim reads); nll_pseudo_real one value per cohort window."""
    import run_joint_arms as R
    sim = {}
    for p in _shards(project, "sim"):
        z = np.load(p)
        for k in ("theta", "donor"):
            sim.setdefault(k, []).append(np.asarray(z[k]))
    sim = dict((k, np.concatenate(v)) for k, v in sim.items())
    cfg = load_json(project["cfg"])
    for f in _folders(project):
        q = os.path.basename(os.path.dirname(f))
        real = [x for x in cfg["synthetic"]["searches"] if x["name"] == q][0]["real"]
        n_real = sum(np.load(p)["theta"].shape[0] for p in _shards(project, "real_" + real))
        for rp in glob.glob(os.path.join(f, "member_seed*", "*_perrow.npz")):
            z = np.load(rp)
            r = load_json(rp.replace("_perrow.npz", ".json"))
            assert z["nll"].dtype == np.float64 and z["nll"].ndim == 1
            assert z["theta"].dtype == np.float32 and z["theta"].shape == (z["nll"].size, 10)
            assert np.issubdtype(z["group"].dtype, np.integer) and z["group"].shape == z["nll"].shape
            assert np.all((z["theta"] > 0) & (z["theta"] < 1))
            split, h = R.grouped_split(sim["donor"], fracs=tuple(r["split_fracs"]),
                                       seed=int(r["split_seed"]))
            assert h == r["split_hash"]
            rows = np.flatnonzero(split == 2)
            np.testing.assert_array_equal(z["theta"], sim["theta"][rows])
            np.testing.assert_array_equal(z["group"], sim["donor"][rows])
            assert abs(float(np.mean(z["nll"])) - float(r["L"])) <= 1e-6
            assert "nll_pseudo_real" in z.files and z["nll_pseudo_real"].shape == (n_real,)
            assert abs(float(np.mean(z["nll_pseudo_real"])) - float(r["L_pseudo_real"])) <= 1e-6


def test_ledger_and_rounds(project):
    """block 7 -> 20: every trial of the rounds has a ledger record with its
    arm, status, nll and L_pseudo_real (block 17); the arm is the one its
    switches give (sec. 2.5)."""
    cfg = load_json(project["cfg"])
    for q in cfg["synthetic"]["searches"]:
        sdir = os.path.join(project["root"], "tune_root", q["name"], q["campaign"])
        rounds = load_json(os.path.join(sdir, "pipeline_rounds.json"))["rounds"]
        assert [r["round"] for r in rounds] == list(range(len(rounds)))
        for r in rounds:
            for tid in r["trial_ids"]:
                rec = load_json(os.path.join(sdir, "trials", "%s.json" % tid))
                c = rec["config"]
                arm = "A2" if int(c["dsn_on"]) else ("A5" if int(c["rep_on"]) else "A1")
                assert rec["arm"] == arm and rec["status"] == "ok"
                assert math.isfinite(rec["nll"]) and "L_pseudo_real" in rec
        sc = load_json(os.path.join(sdir, "SEARCH_COMPLETE.json"))
        assert sc["finalist"]["trial_id"] in [t for r in rounds for t in r["trial_ids"]]


@pytest.mark.parametrize("bank", ["sim", "real_R0", "real_R3"])
def test_bank_contract(project, bank):
    """block 16 -> 19/23: the shards' arrays and sidecars the evaluation
    reads; x (n, W) float32 with W = round(T_win f_s); the unit bin w_size =
    1/f_s (per_unit_mean); bank-wide donor ids (block 12)."""
    paths = _shards(project, bank)
    assert paths
    seen = set()
    for k, p in enumerate(paths):
        z = np.load(p)
        s = load_json(p[:-4] + ".json")
        n = z["theta"].shape[0]
        W = int(round(float(s["T_win"]) * float(s["fs"])))
        assert int(s["W"]) == W and z["x"].shape == (n, W) and z["x"].dtype == np.float32
        assert z["theta"].shape == (n, 10) and np.all((z["theta"] > 0) & (z["theta"] < 1))
        for key in ("donor", "well", "cls", "window_idx"):
            assert np.issubdtype(z[key].dtype, np.integer) and z[key].shape == (n,), key
        assert z["contaminated"].dtype == bool and z["contaminated"].shape == (n,)
        assert s["scale_convention"] == "per_unit_mean"
        assert abs(float(s["w_size"]) - 1.0 / float(s["fs"])) < 1e-12
        lat = s["latent_spec"]
        for key in ("free_idx", "label_idx", "class_centres", "seed", "T_win", "fs", "n_neurons",
                    "n_latent", "tau_ov"):
            assert key in lat, key
        d = set(np.unique(z["donor"]).tolist())
        assert not (d & seen)
        seen |= d
        assert all(k * 1000000 <= x < (k + 1) * 1000000 for x in d)
        if bank == "sim":
            assert lat.get("kind") == "box" and np.all(z["cls"] == -1) and not s["theta_withheld"]
            assert not z["contaminated"].any()
        else:
            assert s["theta_withheld"] is True and "kind" not in lat
            g = s["gap_spec"]
            for key in ("modes", "pi", "shift_max", "drift_amp_max", "drift_period_s", "contam_frac_max"):
                assert key in g, key
            if float(g["pi"]) == 0.0:
                assert not z["contaminated"].any()
            for dd in d:
                sel = z["donor"] == dd
                assert np.unique(z["well"][sel]).size == 2
                assert np.unique(z["cls"][sel]).size == 1
                assert np.all(z["theta"][sel] == z["theta"][sel][0])
