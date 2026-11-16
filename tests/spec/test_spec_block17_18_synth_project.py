"""Spec-derived tests of blocks 17 and 18 (the committed synthetic project and
the other arm's best fit's configuration rule), of the README's run sheets
(the change that put both blocks in this run's scope), and of sec. 7's
dependency pin; written by the integration tester from specs/SPEC.md alone.

Run: python3 -m pytest -q tests/spec/test_spec_block17_18_synth_project.py
Pure ASCII, LF only.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _spec_util as U  # noqa: E402

import config as C  # noqa: E402  (the pipeline's, put first by _spec_util)


def _data():
    return C.load_config(U.SYNTH, synthetic_ok=True).data


def test_block17_values_as_amended():
    d = _data()
    b = d["synthetic"]["builder"]
    # block 17 (D-118..D-123), block 25 (v4: label axes 0-7, centre scale 0.562743)
    assert (b["provider"], b["fs"], b["T_win"], b["n_neurons"]) == ("bench", 100.0, 180.0, 100)
    assert (b["wells_per_donor"], b["donors_per_batch"], b["n_classes"]) == (2, 8, 3)
    assert b["tau_ov"] == 0.10 and b["class_centres"] == "regular"
    assert b["centre_scale"] == 0.562743 and b["label_axes"] == list(range(8))
    s = d["synthetic"]["sim"]
    assert (s["n_theta"], s["n_windows"], s["seed"], s["traces_per_shard"]) == (20000, 1, 1000, 2000)
    assert s["n_theta"] * b["wells_per_donor"] // s["traces_per_shard"] == 20      # K = 20
    real = dict((r["name"], r) for r in d["synthetic"]["real"])
    assert (real["R0"]["seed"], real["R0"]["donors_per_class"], real["R0"]["n_windows"],
            real["R0"]["pi"]) == (2000, 9, 6, 0.0)
    assert (real["R3"]["seed"], real["R3"]["donors_per_class"], real["R3"]["n_windows"],
            real["R3"]["pi"]) == (3000, 9, 6, 0.3)
    for r in real.values():
        assert sorted(r["gap_modes"]) == ["contamination", "drift", "range_shift"]
    # block 26: the five searches of v4, in order, every alt_best_fit false
    assert [(q["name"], q["campaign"], q["real"], q["alt_best_fit"])
            for q in d["synthetic"]["searches"]] == [
        ("S-A1", "S-A1", "R0", False), ("S-A2-R0", "S-A2-only", "R0", False),
        ("S-A5-R0", "S-A5-only", "R0", False), ("S-A2-R3", "S-A2-only", "R3", False),
        ("S-A5-R3", "S-A5-only", "R3", False)]
    se = d["search"]
    assert (se["n_initial_points"], se["gp_rounds"], se["trials_per_round"]) == (20, 4, 10)
    t = se["trial"]
    assert (t["ncpus"], t["mem"], t["walltime"], t["passes"], t["epochs"]) == (
        8, "96gb", "24:00:00", 6.0, 20)
    assert se["space_overrides"] == {"depth_exponent": [3, 5], "embedding_size": [16, 64],
                                     "head_pool_ops": [0, 1]}
    assert se["shape_anchors"] == {"p": 10, "d_theta": 10, "embedding_dim": 64}
    bf = d["best_fit"]
    assert (len(bf["member_seeds"]), bf["ncpus"], bf["passes"], bf["epochs"],
            bf["patience_epochs"], bf["mem"], bf["walltime"]) == (5, 8, 40.0, 40, 10, "96gb",
                                                                   "120:00:00")
    assert d["loader"]["buffer_shards"] == 20
    jb = d["synthetic"]["bank_job"]
    assert (jb["ncpus"], jb["mem"], jb["walltime"]) == (1, "8gb", "02:00:00")
    assert d["driver"]["ncpus"] == 4 and d["resources"]["max_cores"] == 120
    # sec. 2.5: M = 5 members trained with seeds m = 1..5 on split seed 1
    assert bf["member_seeds"] == [1, 2, 3, 4, 5] and bf["split_seed"] == 1


def test_block18_alt_arms():
    assert C.ALT_ARMS == {"S-A2": ("A1", "A2"), "S-A5": ("A1", "A5")}


@pytest.mark.parametrize("campaign", ["S-A1", "S-A25", "S-A2-only", "S-A5-only"])
def test_block18_alt_best_fit_true_refused_outside_ALT_ARMS(tmp_path, campaign):
    searches = [U.search("S-A1", "S-A1", "R0"), U.search("Q", campaign, "R3", True)]
    if campaign == "S-A1":
        searches = [U.search("S-A1", "S-A1", "R0"), U.search("Q", "S-A1", "R3", True)]
    cp, _ = U.write_project(str(tmp_path), searches=searches)
    with pytest.raises(C.ConfigError) as ei:
        C.load_config(cp, synthetic_ok=True)
    assert "alt_best_fit" in str(ei.value)


@pytest.mark.parametrize("campaign", ["S-A2", "S-A5"])
def test_block18_alt_best_fit_true_accepted_in_ALT_ARMS(tmp_path, campaign):
    cp, _ = U.write_project(str(tmp_path), searches=[U.search("S-A1", "S-A1", "R0"),
                                                     U.search("Q", campaign, "R0", True)])
    C.load_config(cp, synthetic_ok=True)


def test_readme_names_only_paths_that_exist():
    text = open(os.path.join(U.PIPE, "README.md")).read()
    paths = sorted(set(re.findall(r"\bhpc/(?:pipeline|joint)/[A-Za-z0-9_./-]+\.(?:py|sh|pbs|json)\b",
                                  text)))
    assert paths, "the README names no path"
    missing = [p for p in paths if not os.path.exists(os.path.join(U.REPO, p))]
    assert not missing, missing


def test_readme_paragraph_of_block7_i_j_agrees_with_the_spec():
    # the paragraph added since the base (D-134): rule 5, the done predicate, the set-aside,
    # the identity refused even under --accept-config-change
    text = " ".join(open(os.path.join(U.PIPE, "README.md")).read().split())
    for frag in ("returns to pending every stage downstream of it that is done or skipped",
                 "names the finalist of its search's `SEARCH_COMPLETE.json`",
                 "`<folder>.set_aside<k>` (never deleted)",
                 "even under `--accept-config-change`"):
        assert frag in text, frag


def test_sec7_metric_learning_pin():
    req = open(os.path.join(U.REPO, "hpc", "requirements.txt")).read()
    env = open(os.path.join(U.REPO, "hpc", "environment.yml")).read()
    pat = re.compile(r"^\s*-?\s*pytorch-metric-learning\s*>=\s*1\.6\.3\s*,\s*<\s*3\b", re.M)
    assert pat.search(req), "requirements.txt"
    assert pat.search(env), "environment.yml"
    try:
        import pytorch_metric_learning as P
        from pytorch_metric_learning.utils import loss_and_miner_utils  # noqa: F401,E501
    except ImportError:
        pytest.skip("pytorch-metric-learning not installed")
    major = int(P.__version__.split(".")[0])
    assert major < 3
