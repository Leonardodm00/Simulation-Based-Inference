"""Boundaries between blocks 11-17 and their neighbours, from this group's
side (step 7 of the tester's procedure; B5, B18, B22 of the t1-B report):

- B5/12: block 5's ShardIndex over shards with block 12's bank-wide ids --
  the small arrays and offsets are the shards' own, one contract, and the
  grouped split by donor (block 6) never puts one donor in two parts;
- B18/15: block 18's other-arm stage on block 15's view (its arms from the
  view's campaign, ALT_ARMS);
- B22/16: block 22's calibration bank is built with block 16's builder
  options (the same provider, f_s, T_win, neurons, centres, label axes).
"""
import os

import numpy as np
import pytest

import t1b_helpers as H

H.pipeline_path()
import config as C                     # noqa: E402
from stages import synthetic as SY     # noqa: E402


@pytest.fixture(scope="module")
def three_shards():
    d = H.tmpdir("t1b_b5_")
    paths = []
    for k in range(3):
        rc, out = H.run_builder(d, ["--arm", "S", "--prior", "box", "--provider", "bench",
                                    "--fs", "10.0", "--T-win", "12.8", "--n-traces", str(6 + 2 * k),
                                    "--wells-per-donor", "2", "--n-windows", "2", "--seed", "77",
                                    "--shard-index", str(k)])
        assert rc == 0, out
        paths.append(os.path.join(d, "shard_%04d.npz" % k))
    yield paths
    H.rmtree(d)


def test_b5_shard_index_over_bank_wide_ids(three_shards):
    pytest.importorskip("torch")
    H.joint_paths()
    import latent_bank as LB
    idx = LB.ShardIndex(three_shards)
    loaded = [H.load_shard(p) for p in three_shards]
    sizes = [a["x"].shape[0] for a, _s in loaded]
    assert idx.n_shards == 3 and idx.n_rows == sum(sizes)
    np.testing.assert_array_equal(np.asarray(idx.offsets), np.concatenate([[0], np.cumsum(sizes)]))
    for key in ("donor", "well", "batch", "cls"):
        np.testing.assert_array_equal(getattr(idx, key), np.concatenate([a[key] for a, _s in loaded]))
    np.testing.assert_array_equal(idx.theta, np.concatenate([a["theta"] for a, _s in loaded]))
    assert len(set(s["contract_digest"] for _a, s in loaded)) == 1
    # one donor per shard id range, none shared: 3 + 4 + 5 donors
    assert np.unique(idx.donor).size == 3 + 4 + 5
    import run_joint_arms as RJA
    split, _h = RJA.grouped_split(idx.donor, fracs=(0.5, 0.25, 0.25), seed=1)
    for dnr in np.unique(idx.donor):
        assert np.unique(split[idx.donor == dnr]).size == 1, dnr


def test_b18_alt_stage_reads_the_views_campaign(tmp_path):
    d = H.committed_synth()
    d["synthetic"]["searches"][1].update(campaign="S-A2", alt_best_fit=True)
    p = H.write_json(os.path.join(str(tmp_path), "synth.json"), d)
    cfg = C.load_config(p, synthetic_ok=True)
    q = d["synthetic"]["searches"][1]
    st = SY.RunAltBestFitStage(q)
    v = st._view(cfg)
    assert st._arms(v) == C.ALT_ARMS["S-A2"] == ("A1", "A2")
    assert st._dirs(v)["root"] == os.path.join(d["paths"]["runs_root"], q["name"],
                                               d["dataset"] + "_best_alt")
    q2 = dict(d["synthetic"]["searches"][2])                # S-A5-only: no other arm
    with pytest.raises(RuntimeError):
        SY.RunAltBestFitStage(q2)._arms(SY.RunView(cfg, q2))


def test_b22_calibration_bank_uses_the_sim_banks_options():
    H.add_paths(os.path.join(H.PIPE, "evaluation"))
    import ev_config
    import ev_plan
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    ecfg = ev_config.load(os.path.join(H.PIPE, "projects", "synth", "synth_eval.json"))
    cal = ev_plan.calbank_build(cfg, ecfg)
    sim = SY.build_of(cfg, "sim")

    def opts(argv):
        out = {}
        for i, a in enumerate(argv):
            if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[a] = argv[i + 1]
        return out

    o_sim, o_cal = opts(sim["argv"][0]), opts(cal["argv"][0])
    for key in ("--provider", "--fs", "--T-win", "--n-neurons", "--donors-per-batch", "--n-classes",
                "--tau-ov", "--class-centres", "--centre-scale", "--label-axes", "--prior", "--arm"):
        assert o_cal[key] == o_sim[key], key
    assert o_cal["--wells-per-donor"] == "1" and o_cal["--n-windows"] == "1"
    assert int(o_cal["--seed"]) not in [cfg.data["synthetic"]["sim"]["seed"]] + \
        [r["seed"] for r in cfg.data["synthetic"]["real"]]


def test_step8_swap_a_method_through_the_configuration():
    """Directive audit, swappability: the class centres (a method choice:
    `simplex` at scale 1 instead of `regular`), the label axes and the
    windows per well changed through the configuration alone; the bank
    stages still build and record every bank under the fake PBS."""
    import pipeline as P
    import stages as S
    root = H.tmpdir("t1b_swap_")
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    old = H.fake_env(spool)
    try:
        path, d = H.reduced_synth(root, **{"synthetic.builder.class_centres": "simplex",
                                           "synthetic.builder.centre_scale": 1.0,
                                           "synthetic.builder.label_axes": [0, 1, 2, 3, 4, 5, 6]})
        d["synthetic"]["real"][0]["n_windows"] = 3
        H.write_json(path, d)
        cfg = C.load_config(path, synthetic_ok=True)
        graph = [s for s in S.default_graph(cfg) if isinstance(s, SY.SynthBankStage)]
        lines = []
        rc = P.run(cfg, graph=graph, poll_seconds=0.2, log=lines.append, max_ticks=3000)
        assert rc == 0, "\n".join(map(str, lines))[-3000:]
        for name in ("sim", "R0", "R3"):
            ok, why = SY.SynthBankStage(name).is_done(cfg)
            assert ok, why
        a, side = H.load_shard(os.path.join(SY.bank_dir(cfg, "R0"), "shard_0000.npz"))
        assert a["x"].shape[0] == 3 * 2 * 2 * 3
        assert side["latent_spec"]["label_idx"] == [0, 1, 2, 3, 4, 5, 6]
        assert "class_centres_mode" not in side["latent_spec"]
    finally:
        H.restore_env(old)
        H.rmtree(root)


def test_o16_8_record_kept_shard_lost():
    """done(b) is the record, its .sha256 and the build (sec. 2.1, block 16).
    A shard deleted after the record: on a fresh state the bank is skipped as
    done, and what a search would read (the view's glob) is the remaining
    shards -- the record still lists the lost one. Expected by the spec's
    rule; reported for the project to judge (the record holds every shard's
    sha256, which done(b) does not consult)."""
    import glob as _glob
    import pipeline as P
    import stages as S
    from stages import common as CM
    root = H.tmpdir("t1b_lost_")
    spool = os.path.join(root, "spool")
    os.makedirs(spool)
    old = H.fake_env(spool)
    try:
        path, d = H.reduced_synth(root)
        cfg = C.load_config(path, synthetic_ok=True)
        banks = [s for s in S.default_graph(cfg) if isinstance(s, SY.SynthBankStage)]
        lines = []
        assert P.run(cfg, graph=banks, poll_seconds=0.2, log=lines.append, max_ticks=3000) == 0
        sim = SY.bank_dir(cfg, "sim")
        for f in ("shard_0002.npz", "shard_0002.json"):
            os.remove(os.path.join(sim, f))
        os.remove(os.path.join(d["paths"]["work_root"], "state.json"))
        lines = []
        assert P.run(cfg, graph=banks, poll_seconds=0.2, log=lines.append, max_ticks=3000) == 0
        assert SY.SynthBankStage("sim").is_done(cfg)[0]
        rec = __import__("json").load(open(os.path.join(sim, "BANK_RECORD.json")))
        listed = sorted(e["file"] for e in rec["shards"])
        g = CM.shard_globs(SY.RunView(cfg, d["synthetic"]["searches"][0]))[0]
        present = sorted(os.path.basename(p) for p in _glob.glob(g))
        assert present == listed, "done, but the search would read %s of the record's %s" % (
            present, listed)
    finally:
        H.restore_env(old)
        H.rmtree(root)
