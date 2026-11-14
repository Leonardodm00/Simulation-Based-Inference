"""Chain t2-C2, step 8 (swappability across the chain): components changed
through the configuration alone, the rest of the chain unchanged, run end to
end through the real entry point:

- `loader.buffer_shards` 2 on a simulated bank of 3 shards (block 5's
  rolling buffer instead of the whole bank, for the trials and the members);
- `search.flow.num_bins` 8 (sec. 3.1: `search.flow.{conditioner_hidden_layers,
  num_bins}`; D-082: every training value from the configuration);
- the builder's centres `simplex` at scale 1 (block 12's default design).

Expected: PIPELINE DONE; the runner of every trial and member given the
configuration's values (`--num-bins 8`, `--buffer-shards 2`), its record's
loader `{buffer_shards 2, n_shards 3}` and its flow's num_bins 8.
"""

import os
import shutil

import pytest

import c2_helpers as H


def _flag(argv, name):
    idx = [i for i, a in enumerate(argv) if a == name]
    return argv[idx[-1] + 1] if idx else None


@pytest.fixture(scope="module")
def p_swap(tmp_path_factory):
    base = os.environ.get("C2_ROOT")
    root = os.path.join(os.path.abspath(base), "p_swap") if base else \
        str(tmp_path_factory.mktemp("p_swap"))
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    q = {"name": "S-A2-R0", "campaign": "S-A2-only", "real": "R0", "alt_best_fit": False}
    cfg_path, cfg = H.reduced(
        os.path.join(root, "project"), [q],
        **{"synthetic.sim": {"seed": 11, "n_theta": 30, "n_windows": 1, "traces_per_shard": 20},
           "loader.buffer_shards": 2, "search.flow.num_bins": 8,
           "synthetic.builder.class_centres": "simplex", "synthetic.builder.centre_scale": 1.0})
    spool = os.path.join(root, "spool")
    env = H.fake_env(spool)
    rc, out = H.entry(cfg_path, env)
    assert rc == 0, out
    ce = H.wait_chain(cfg, spool, timeout=1800)
    return {"cfg": cfg, "spool": spool, "ce": ce, "q": q}


def test_swapped_chain_runs_to_the_end(p_swap):
    assert p_swap["ce"] and "done (exit 0)" in p_swap["ce"], p_swap["ce"]
    assert "PIPELINE DONE" in H.log_text(p_swap["cfg"])


def test_buffer_shards_reach_every_run(p_swap):
    cfg, q = p_swap["cfg"], p_swap["q"]
    runs = [rec["runner_record"] for rec in H.ledger(cfg, q).values()]
    import glob
    runs += glob.glob(os.path.join(H.best_dir(cfg, q), "member_seed*", "*_seed*.json"))
    assert len(runs) >= 4
    for rp in runs:
        run = H.read_json(rp)
        assert run["loader"]["buffer_shards"] == 2 and run["loader"]["n_shards"] == 3, rp


def test_num_bins_reaches_every_run(p_swap):
    """sec. 3.1's `search.flow.num_bins` and D-082: the runner of every trial
    and member is given the configuration's 8 bins."""
    cfg, q = p_swap["cfg"], p_swap["q"]
    import glob
    got = []
    for tid, rec in H.ledger(cfg, q).items():
        got.append(("trial " + tid[:12], _flag([str(a) for a in rec["argv"]], "--num-bins"),
                    H.read_json(rec["runner_record"])["config"].get("num_bins")))
    for p in sorted(glob.glob(os.path.join(H.best_dir(cfg, q), "jobs", "member_seed*.0.argv"))):
        a = H.read_argv(p)
        got.append((os.path.basename(p), _flag(a, "--num-bins"), None))
    print(got)
    bad = [g for g in got if g[1] != "8" or (g[2] is not None and g[2] != 8)]
    assert not bad, "search.flow.num_bins 8 in the configuration; the runs got: %s" % bad
