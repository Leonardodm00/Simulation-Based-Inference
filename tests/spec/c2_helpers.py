"""Helpers of the tier-2 chain run t2-C2 (the synthetic experiment) of the
split audit D-131: a reduced copy of the committed synth.json, the fake PBS
of hpc/pipeline/tests/fake_pbs on PATH, the chain driven through its real
entry point `hpc/pipeline/run_pipeline.sh --config <file>` (the sequencer as
a chain of PBS jobs, D-098), and readers of what the chain wrote.

This module computes no expected value: the tests compute theirs from
specs/SPEC.md with numpy / scipy. It writes only under the temporary folder
it is given.
"""

import copy
import glob
import json
import os
import re
import subprocess
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
FAKE = os.path.join(PIPE, "tests", "fake_pbs")
RUN_PIPELINE = os.path.join(PIPE, "run_pipeline.sh")
COMMITTED = os.path.join(PIPE, "projects", "synth", "synth.json")

# The searches of the reduced chain: the three kinds of the committed v4
# (S-A1 on R0, a term-only A2 search on R0, a term-only A5 search on R3), and
# a mixed S-A2 search with alt_best_fit true (v3's shape) for G_q (block 18).
SEARCHES = [
    {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
    {"name": "S-A2-R0", "campaign": "S-A2-only", "real": "R0", "alt_best_fit": False},
    {"name": "S-A5-R3", "campaign": "S-A5-only", "real": "R3", "alt_best_fit": False},
    {"name": "S-A2m-R3", "campaign": "S-A2", "real": "R3", "alt_best_fit": True},
]


def committed():
    with open(COMMITTED) as fh:
        return json.load(fh)


def _strip(d):
    """The configuration without its `_` documentation keys (lists too)."""
    if isinstance(d, dict):
        return dict((k, _strip(v)) for k, v in d.items() if not k.startswith("_"))
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def reduced(root, searches=None, **over):
    """A reduced copy of the committed synth.json under `root`, every value
    within sec. 3.1's constraints: the bench generator at f_s 10 Hz and
    T_win 12.8 s (W 128); arm S 40 theta x 2 wells x 1 window in 4 shards of
    20 wells; R0 and R3 of 3 conditions x 2 donors x 2 wells x 2 windows (R3
    at pi 0.3, the three modes); a small search space; 2 + 1 x 2 trials per
    search; 2 members; the sequencer and every job 1 core under a cap of 3.
    `over` sets dotted keys. Returns (path, data)."""
    d = copy.deepcopy(committed())
    for k in list(d["paths"]):
        d["paths"][k] = os.path.join(root, k)
    d["paths"]["sbi_dir"] = REPO
    syn = d["synthetic"]
    syn["builder"].update({"fs": 10.0, "T_win": 12.8})
    syn["sim"] = {"seed": 11, "n_theta": 40, "n_windows": 1, "traces_per_shard": 20}
    syn["real"] = [
        {"name": "R0", "seed": 22, "donors_per_class": 2, "n_windows": 2, "pi": 0.0,
         "gap_modes": ["range_shift", "drift", "contamination"]},
        {"name": "R3", "seed": 33, "donors_per_class": 2, "n_windows": 2, "pi": 0.3,
         "gap_modes": ["range_shift", "drift", "contamination"]}]
    syn["searches"] = copy.deepcopy(searches if searches is not None else SEARCHES)
    syn["bank_job"] = {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00", "queue": "test",
                       "max_resubmits": 1}
    s = d["search"]
    s["shape_anchors"] = {"p": 10, "d_theta": 10, "embedding_dim": 8}
    s["space_overrides"] = {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                            "embedding_size": [8, 8], "hidden_features": [24, 24],
                            "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                            "head_pool_ops": [0, 1]}
    s.update({"n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 2,
              "max_concurrent_trials": 2, "gp_seed": 3, "train_seed": 0})
    s["budget_gate"] = {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5}
    s["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "4gb",
                       "walltime": "00:10:00", "queue": "test", "max_resubmits": 1})
    d["best_fit"].update({"member_seeds": [1, 2], "passes": 1.0, "epochs": 2,
                          "n_posterior_draws": 16, "ncpus": 1, "mem": "4gb",
                          "walltime": "00:10:00", "queue": "test", "max_resubmits": 1})
    d["loader"] = {"buffer_shards": 4}
    d["driver"] = {"queue": "test", "ncpus": 1, "mem": "1gb", "walltime": "01:00:00",
                   "poll_seconds": 1}
    d["resources"] = {"max_cores": 3}
    for k, v in over.items():
        cur = d
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = v
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "synth.json")
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path, _strip(d)


def fake_env(spool, **extra):
    """The environment of the commands and, through the fake PBS, of every job
    (the fake hands its own environment to its jobs)."""
    os.makedirs(spool, exist_ok=True)
    env = dict(os.environ)
    path = env.get("PATH", "")
    env["PATH"] = FAKE + os.pathsep + path.replace(FAKE + os.pathsep, "")
    env.update({"FAKE_PBS_SPOOL": spool, "FAKE_PBS_DELAY": "0.2",
                "FAKE_PBS_ENFORCE_WALLTIME": "0", "SKIP_CONDA": "1",
                "PIPELINE_SKIP_CONDA": "1", "OMP_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    for k in list(env):
        if k.startswith("FAKE_PBS_") and k not in ("FAKE_PBS_SPOOL", "FAKE_PBS_DELAY",
                                                   "FAKE_PBS_ENFORCE_WALLTIME"):
            env.pop(k)
    env.update(dict((k, str(v)) for k, v in extra.items()))
    return env


def run(argv, env, timeout=600, cwd=None):
    p = subprocess.run(argv, env=env, cwd=cwd or REPO, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout


def entry(cfg_path, env, *flags, timeout=600):
    """`bash run_pipeline.sh --config <cfg> <flags>` (rc, output)."""
    return run(["bash", RUN_PIPELINE, "--config", cfg_path] + list(flags), env, timeout=timeout)


def work_root(cfg):
    return cfg["paths"]["work_root"]


def chain_end(cfg):
    p = os.path.join(work_root(cfg), "driver", "CHAIN_END")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return fh.read().strip()


def jobs(spool):
    """Every job record of the fake PBS spool, by job number."""
    out = []
    for p in glob.glob(os.path.join(spool, "*.json")):
        b = os.path.basename(p)[:-5]
        if not b.isdigit():
            continue
        for _ in range(5):
            try:
                with open(p) as fh:
                    out.append(json.load(fh))
                break
            except ValueError:
                time.sleep(0.05)
    return sorted(out, key=lambda r: r["num"])


def live_jobs(spool):
    return [r for r in jobs(spool) if r.get("state") != "F"]


def wait_chain(cfg, spool, timeout=1800, settle=True):
    """Wait for driver/CHAIN_END and (settle) for every job of the spool to
    finish; return the CHAIN_END text (None on timeout)."""
    t0 = time.time()
    ce = None
    while time.time() - t0 < timeout:
        ce = chain_end(cfg)
        if ce is not None and (not settle or not live_jobs(spool)):
            return ce
        time.sleep(1.0)
    return ce


def wait_until(pred, timeout=900, interval=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    return None


def ncpus(rec):
    """K x N of `-l select=K:ncpus=N...` (K = 1 when absent)."""
    for r in rec.get("resources") or []:
        m = re.search(r"select=(?:([0-9]+):)?ncpus=([0-9]+)", r)
        if m:
            return int(m.group(1) or 1) * int(m.group(2))
    return None


def cores_profile(recs, count_dependent=True):
    """The largest sum of cores of the jobs submitted and not finished at
    one instant (queued, held or running), over the submission and finish
    instants. With count_dependent False, a job that waits on a dependency
    (the successor driver, `afterany`) counts only once it starts."""
    events = set()
    for r in recs:
        events.add(r["submitted"])
        if r.get("started"):
            events.add(r["started"])
    worst, when, who = 0, None, []
    for t in sorted(events):
        tot, names = 0, []
        for r in recs:
            begin = r["submitted"]
            if not count_dependent and r.get("depend"):
                begin = r.get("started") or float("inf")
            end = r.get("finished") or float("inf")
            if begin <= t < end:
                tot += ncpus(r) or 0
                names.append(r["name"])
        if tot > worst:
            worst, when, who = tot, t, names
    return worst, when, who


def state(cfg):
    p = os.path.join(work_root(cfg), "state.json")
    with open(p) as fh:
        return json.load(fh)


def log_text(cfg):
    p = os.path.join(work_root(cfg), "pipeline.log")
    if not os.path.isfile(p):
        return ""
    with open(p, errors="replace") as fh:
        return fh.read()


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_argv(path):
    with open(path) as fh:
        return [l.rstrip("\n") for l in fh if l.rstrip("\n") != ""]


def campaign_dir(cfg, q):
    return os.path.join(cfg["paths"]["tune_root"], q["name"], q["campaign"])


def best_dir(cfg, q, alt=False):
    return os.path.join(cfg["paths"]["runs_root"], q["name"],
                        cfg["dataset"] + ("_best_alt" if alt else "_best"))


def bank_dir(cfg, bank):
    tag = cfg["dataset"]
    return os.path.join(cfg["paths"]["banks_root"],
                        tag + ("_sim" if bank == "sim" else "_real_" + bank))


def ledger(cfg, q):
    """{trial id: record} of the search's ledger (`trials/*.json`)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(campaign_dir(cfg, q), "trials", "*.json"))):
        out[os.path.basename(p)[:-5]] = read_json(p)
    return out


def full_run(root, searches=None, timeout=2400, **over):
    """The whole chain once under `root`: (cfg_path, cfg, spool, env, submit
    output, CHAIN_END text)."""
    cfg_path, cfg = reduced(os.path.join(root, "project"), searches, **over)
    spool = os.path.join(root, "spool")
    env = fake_env(spool)
    rc, out = entry(cfg_path, env)
    if rc != 0:
        return cfg_path, cfg, spool, env, out, None
    ce = wait_chain(cfg, spool, timeout=timeout)
    return cfg_path, cfg, spool, env, out, ce
