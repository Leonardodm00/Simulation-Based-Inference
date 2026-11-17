#!/usr/bin/env python3
"""Smoke test for blocks 9 and 32 of specs/SPEC.md: the inputs check
(`tools/inputs_check.py`), on an lcdup15-like campaign tree, a cohort
manifest of the committed cohort map's wells and a configuration derived
from `projects/linddup/linddup.json` (v11), written to a temporary folder.

The tree (block 32's fixture): the campaign campaign_lcdup15_rho1300_neur_n01
with the tasks sweep_cfd_task0000 (3 topologies, K = 2 workers x 2 per worker
= 4) and sweep_cpu_task0001 (2 topologies, K = 3 x 1 = 3); a 37-axis registry
of which 13 axes are active (the group neuron_synapse), ln coordinates but
Sigma, the on-axis gbarA with box [0.01, 10] in every task; conn_rule flat;
conn_prob the only topology axis, one value per topology; each topology
holds topology.npz, topology_meta.json and its K iteration files
iter_<i>.npz, each with its iter_<i>.json. The cohort manifest lists the 35
wells of the committed linddup_cohort_map.csv, which real.cohort_map names
(a copy in the tree). The cohort map's reader is the Sbi-extractor
checkout's own export_windows/cohort_map.py once its Block 9 is built; while
the checkout's raises NotImplementedError (the contract's stub), the
stand-in FAKE_COHORT_MAP below, implementing Block 9's contract, takes its
place in the test's Sbi-extractor tree (block 32 allows it). The run prints
which one it used.

Checks (specs/SPEC.md blocks 9 and 32, "Test oracles")
    test_known_answer   T32a: the consistent tree gives 0 FAIL, 0 WARN at
                        quiet hours 0; 1 campaign, 2 tasks, 5 topologies, 18
                        iteration files; conn_prob's 5 distinct sampled
                        values (topology axis conn_prob OK); the cohort map
                        OK with 35 wells, 5 excluded, 2 donors, 4 pairing
                        groups, control 17 and pathological 13 included
                        (D-144..D-146); topology metadata OK; partial
                        topologies NOTE with 0; the searches NOTE (A1, A2,
                        A5 at 148 cores, 444 against 444); the plan's
                        --quiet-hours 6 leaves nothing out (D-112); the job
                        protocol's files on both sides (D-113); exit 0 and
                        the last line
    test_partial        T32c: a partial topology (2 of K files) is a NOTE
                        counting it, no FAIL; Block 10's count besides: 3
                        json and 1 failure line (blank lines not counted) is
                        complete, an iteration file without its json is not
                        a finished run (f counts the json, the simulations
                        the iteration files), 0 json beside
                        topology_meta.json is partial
    test_being_written  D-112: one task changed just now is named as left
                        out, not sampled (3 files, its conn_prob values
                        absent), no WARN; iteration files 10 h apart give
                        the pace WARN against --quiet-hours 6
    test_reference      read_scalars equals numpy.load's own 0-d members on a
                        file holding arrays too; the counts equal an os.walk
                        count of iter_*.npz; the scan's iter_*.json and
                        iter_*.npz counts and its metadata and failures
                        flags equal a glob of each topology (one npz
                        without its json, one failures file);
                        failure_lines equals a count by str.splitlines; the
                        cohort map row's counts equal the csv module's count
                        of the map
    test_convergence    skipped (not a discretisation)
    test_invariants     the WARNs: a quiet-hours window covering the tree's
                        age gives the campaign WARN (0 h none); NaN-valued
                        p0_conn in every file, not excluded, gives the
                        freeze-exclude WARN and no FAIL; stoa_gate_p is read
                        as the configuration names it (bank.topology_axes),
                        not as a name in the code: the gate variant of the
                        tree with stoa_gate_p in the configuration passes,
                        with its job_args box
    test_contract       read-only: every file's and folder's size and
                        modification time unchanged by a run, in this
                        process and in a fresh one (every import from the
                        tree a first one: no bytecode written); the --json
                        report's keys; exit 1 with a FAIL, 2 on a refused
                        configuration
    test_determinism    two runs give the same rows (but the age text) and
                        the same sample; the sample is round-robin
    test_edge_cases     each mutation of the blocks' lists makes its named
                        check FAIL and no other (T32b: a manifest well
                        missing from the map, a map row of no well, a
                        topology folder without topology_meta.json, a
                        job_args without n_params_per_worker, f(t, k) >
                        K(t); besides: the cohort map's module absent or the
                        stub, each "pull it"; a bad include value; the map
                        file absent); each v11 variant gives its NOTE and no
                        FAIL (no map, no manifest, drop_partial_topologies
                        false, searches null, a search without a cap);
                        dispatch_size's refusals

Run
    SBIX_CLONE=<Sbi-extractor checkout> python3 hpc/pipeline/tests/test_smoke_inputs_check.py [check ...]
        About 5 s. Needs the Sbi-extractor checkout (SBIX_CLONE or a known
        place: its sbi_labels.py, sim_reextract/profiles/linddup.sh, the
        D-113 job scripts, export_windows/cohort_map.py) and numpy.

Pure ASCII, LF only.
"""

import contextlib
import csv
import glob
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_PIPE, _HERE, os.path.join(_PIPE, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import standin as ST            # noqa: E402
import inputs_check as IC       # noqa: E402

PROJECT = os.path.join(_PIPE, "projects", "linddup")
MAP_NAME = "linddup_cohort_map.csv"
CAMP = "campaign_lcdup15_rho1300_neur_n01"
SIM = "LindaColloDUP"
N_REG = 37
ACTIVE = [0, 1, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17]     # neuron_synapse's 13 axes (D-138)
# the registry's natural boxes of the active axes (the brief's); every other axis frozen
REG_BOXES = {0: ("Sigma", [3.0, 15.0]), 1: ("gbarA", [0.01, 10.0]), 4: ("tauA", [10.0, 2000.0]),
             5: ("U_0_ar", [1e-4, 0.05]), 6: ("U_max", [0.1, 1.0]), 7: ("U_0_sr", [0.01, 0.7]),
             8: ("Omega_f_sr", [0.1, 4.5]), 9: ("Omega_f_ar", [0.1, 4.5]), 10: ("Omega_d", [0.1, 4.5]),
             14: ("g_ampa", [0.01, 5.0]), 15: ("g_nmda", [0.01, 5.0]), 16: ("delta_gA", [0.005, 10.0]),
             17: ("x0", [0.05, 0.5])}
LAUNCHED = {7: [0.1, 1.0], 14: [0.01, 10.0], 15: [0.01, 10.0]}   # the tasks' boxes where the launch differs
FROZEN = [0.5, 2.0]                                                # linear under rule (1)
# sweep, topologies, seed, (_resolved_n_workers, n_params_per_worker): K = 4 and 3
TASKS = (("sweep_cfd_task0000", 3, 7001, (2, 2)), ("sweep_cpu_task0001", 2, 7002, (3, 1)))
CONN = {(0, 0): 0.05, (0, 1): 0.12, (0, 2): 0.27, (1, 0): 0.33, (1, 1): 0.39}   # conn_prob per topology
GATES = {(0, 0): 0.31, (0, 1): 0.47, (0, 2): 0.66, (1, 0): 0.25, (1, 1): 0.92}  # the gate variant's stoa_gate_p
N_NEURONS = 159
_TEMPLATE = {}

# Block 9's contract (Sbi-extractor specs/SPEC.md), as a stand-in module the test
# writes into its temporary Sbi-extractor tree while the checkout's is the stub
FAKE_COHORT_MAP = '''"""A stand-in for Sbi-extractor's export_windows/cohort_map.py, written by the
SBI repository's hpc/pipeline/tests/test_smoke_inputs_check.py into its
temporary Sbi-extractor tree while the checkout's module is the contract's
stub. It implements that repository's specs/SPEC.md Block 9 (the file, the
refusals, the ids) and returns its CohortMap's fields. Not shipped.

Pure ASCII, LF only.
"""
import collections
import csv
import hashlib
import io
import os

REQUIRED_COLUMNS = ("out_dir", "donor", "rep_group", "include")
INCLUDE = {"yes": True, "true": True, "1": True, "no": False, "false": False, "0": False}
CohortMap = collections.namedtuple("CohortMap", ["path", "sha256", "donor_labels", "rep_group_labels",
                                                 "donor_of", "rep_group_of", "excluded"])


class CohortMapError(ValueError):
    """A cohort map the real bank cannot be built on."""


def read_cohort_map(path, manifest):
    with open(path, "rb") as fh:
        blob = fh.read()
    try:
        text = blob.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CohortMapError("%s: not ASCII (%s)" % (path, exc))
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise CohortMapError("%s: no header row" % path)
    head = [h.strip() for h in rows[0]]
    missing = [c for c in REQUIRED_COLUMNS if c not in head]
    if missing:
        raise CohortMapError("%s: required column(s) missing: %s" % (path, ", ".join(missing)))
    col = dict((c, head.index(c)) for c in REQUIRED_COLUMNS)
    wells = [str(w["out_dir"]).replace(os.sep, "/") for w in manifest["wells"]]
    index = dict((o, i) for i, o in enumerate(wells))
    got = {}
    for line, r in enumerate(rows[1:], 2):
        if not r:
            continue
        if len(r) < len(head):
            raise CohortMapError("%s line %d: %d fields, the header has %d"
                                 % (path, line, len(r), len(head)))
        out_dir = r[col["out_dir"]].strip()
        if out_dir not in index:
            raise CohortMapError("%s line %d: %r names no well of the manifest" % (path, line, out_dir))
        i = index[out_dir]
        if i in got:
            raise CohortMapError("%s line %d: a second row for %s" % (path, line, out_dir))
        donor, group = r[col["donor"]].strip(), r[col["rep_group"]].strip()
        if not donor or not group:
            raise CohortMapError("%s line %d: an empty donor or rep_group label" % (path, line))
        inc = r[col["include"]].strip().lower()
        if inc not in INCLUDE:
            raise CohortMapError("%s line %d: include %r is not one of yes, no, true, false, 1, 0"
                                 % (path, line, r[col["include"]]))
        got[i] = (donor, group, INCLUDE[inc])
    lacking = [wells[i] for i in range(len(wells)) if i not in got]
    if lacking:
        raise CohortMapError("%s: %d well(s) of the manifest without a row: %s"
                             % (path, len(lacking), ", ".join(lacking[:5])))
    kept = sorted(i for i in got if got[i][2])
    if not kept:
        raise CohortMapError("%s: every well excluded" % path)
    donors = tuple(sorted(set(got[i][0] for i in kept)))
    groups = tuple(sorted(set(got[i][1] for i in kept)))
    return CohortMap(path=path, sha256=hashlib.sha256(blob).hexdigest(), donor_labels=donors,
                     rep_group_labels=groups,
                     donor_of=dict((i, donors.index(got[i][0])) for i in kept),
                     rep_group_of=dict((i, groups.index(got[i][1])) for i in kept),
                     excluded=tuple(str(manifest["wells"][i]["out_dir"])
                                    for i in range(len(wells)) if not got[i][2]))
'''

# the contract's stub of that module (Sbi-extractor ae71f33), for the mutation
STUB_COHORT_MAP = '''"""The contract's stub (Block 9 not built)."""


class CohortMapError(ValueError):
    pass


def read_cohort_map(path, manifest):
    raise NotImplementedError("Block 9")
'''


# --------------------------------------------------------------- the tree ---

def is_log(box):
    """Rule (1) of sbi_labels.load_registry: a positive box of at least one decade."""
    return box[0] > 0 and box[1] > 0 and np.log10(box[1] / box[0]) >= 1.0


def registry(launched=False):
    """(names, natural boxes) of the 37 axes; the tasks' boxes with `launched`."""
    names = ["p%02d" % j for j in range(N_REG)]
    bounds = [list(FROZEN) for _ in range(N_REG)]
    for j, (name, box) in REG_BOXES.items():
        names[j], bounds[j] = name, list(box)
    if launched:
        for j, box in LAUNCHED.items():
            bounds[j] = list(box)
    return names, bounds


def _write_registry(sim_main):
    names, bounds = registry()
    log = [j for j in range(N_REG) if is_log(bounds[j])]
    theta = [[float(np.log(lo)), float(np.log(hi))] if j in log else [lo, hi]
             for j, (lo, hi) in enumerate(bounds)]
    with open(os.path.join(sim_main, "HPC_main_sweep.py"), "w") as fh:
        fh.write("PARAM_BOUNDS = %r\nPARAM_BOUNDS_THETA = %r\nLOG_PARAMS = %r\nLOG_BASE = 'natural'\n"
                 "KERNEL_BOUNDS = [[0.1, 1.0], [20.0, 200.0], [0.5, 3.0]]\n" % (bounds, theta, log))
    with open(os.path.join(sim_main, "HPC_single_run.py"), "w") as fh:
        fh.write("PARAM_NAMES = %r\nPARAM_UNITS = %r\nSWEEP_GROUPS = %r\n"
                 % (names, ["a.u."] * N_REG, {"neuron_synapse": ACTIVE}))


def write_task(sim_main, camp, sweep, n_topo, seed, n_task, dispatch, gates=False, kernel_nan=False):
    """One lcdup15-like sweep task: job_args.json, manifest.json and n_topo
    complete topologies of K = workers x per-worker simulations each."""
    names, bounds = registry(launched=True)
    log = [j for j in range(N_REG) if is_log(bounds[j])]
    td = os.path.join(sim_main, camp, sweep)
    os.makedirs(td)
    workers, per = dispatch
    K = workers * per
    ja = {"simtime": 180.0, "conn_rule": "flat", "conn_prob_lo": 0.01, "conn_prob_hi": 0.4,
          "_active_indices": ACTIVE, "_active_param_names": [names[j] for j in ACTIVE],
          "_resolved_n_workers": workers, "n_params_per_worker": per,
          "_resolved_seed_master": seed, "seed_master": None, "mode": "Full",
          "sweep_group": "neuron_synapse"}
    if gates:
        ja.update(stoa_gate_lo=0.2, stoa_gate_hi=1.0)
    mf = {"active_indices": ACTIVE, "sweep_group": "neuron_synapse", "param_names": names,
          "log_params": [names[j] for j in log], "log_transform": "natural", "conn_rule": "flat",
          "param_bounds": bounds, "manifest_version": 4}
    for name, doc in (("job_args.json", ja), ("manifest.json", mf)):
        with open(os.path.join(td, name), "w") as fh:
            json.dump(doc, fh, indent=1)
    rng = np.random.default_rng(seed)
    for k in range(n_topo):
        top = os.path.join(td, "topo_%05d" % k)
        os.makedirs(top)
        cp = CONN.get((n_task, k), float(rng.uniform(0.01, 0.4)))
        np.savez_compressed(os.path.join(top, "topology.npz"),
                            N_pos=rng.uniform(0.0, 350.0, (N_NEURONS, 2)), conn_prob=np.float64(cp))
        with open(os.path.join(top, "topology_meta.json"), "w") as fh:
            json.dump({"topo_idx": k, "conn_prob": cp, "simtime": 180.0, "n_neurons": N_NEURONS}, fh)
        for i in range(K):
            theta, params = np.ones(N_REG), np.ones(N_REG)       # the frozen axes: constant columns
            for j in ACTIVE:                                     # uniform in each axis's coordinate
                lo, hi = (np.log(bounds[j][0]), np.log(bounds[j][1])) if j in log else bounds[j]
                theta[j] = lo + rng.uniform() * (hi - lo)
                params[j] = np.exp(theta[j]) if j in log else theta[j]
            extra = {"stoa_gate_p": np.float64(GATES.get((n_task, k), 0.5))} if gates else {}
            if kernel_nan:
                extra.update(p0_conn=np.float64("nan"), d0_conn=np.float64("nan"),
                             beta_conn=np.float64("nan"))
            seed_run = seed * 1000 + 100 * k + i
            np.savez_compressed(os.path.join(top, "iter_%05d.npz" % i), params=params, theta=theta,
                                conn_rule=np.array("flat"), conn_prob=np.float64(cp),
                                topo_idx=np.int32(k), seed_run=np.int64(seed_run),
                                spk_N_t=np.sort(rng.uniform(0, 180, 50)).astype(np.float32),
                                spk_N_i=rng.integers(0, N_NEURONS, 50).astype(np.int32), **extra)
            with open(os.path.join(top, "iter_%05d.json" % i), "w") as fh:
                json.dump({"iter_idx": i, "topo_idx": k, "conn_prob": cp, "seed_run": seed_run}, fh)
    return td


def _sign(path):
    import hashlib
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as fh:
        fh.write("%s  %s\n" % (h, os.path.basename(path)))


def map_rows(path=None):
    """The rows of a cohort map (the committed one by default), by the csv module."""
    with open(path or os.path.join(PROJECT, MAP_NAME), newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path, w_size=0.01, eps=9):
    """A cohort manifest of the committed map's 35 wells, signed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wells = [{"culture": r["out_dir"].replace("/", "_"), "out_dir": r["out_dir"], "n_archives": 9,
              "T_rec": 1200.0} for r in map_rows()]
    doc = {"manifest_version": 3, "n_wells": len(wells), "n_subsets_per_well": 9,
           "classes": ["control", "pathological"],
           "preprocessing": {"w_size": w_size, "gaussian_window": 0.02,
                             "electrodes_per_subset": eps, "fs_raw": 10110.09},
           "wells": wells}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    _sign(path)


@contextlib.contextmanager
def _no_bytecode():
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = old


def probe_cohort_map(path):
    """None when the module at `path` reads a two-well map (its Block 9
    built), else why the stand-in is used: absent, or the stub's refusal.
    Nothing is written into the checkout; sys.path is restored."""
    if not os.path.isfile(path):
        return "is absent"
    tmp = tempfile.mkdtemp(prefix="inputs_check_probe_")
    saved = list(sys.path)
    name = "inputs_check_probe_cohort_map"
    try:
        mp = os.path.join(tmp, "map.csv")
        with open(mp, "w") as fh:
            fh.write("out_dir,donor,rep_group,include\nc/r/w1,c,c_r,yes\np/r/w2,p,p_r,yes\n")
        man = {"wells": [{"out_dir": "c/r/w1"}, {"out_dir": "p/r/w2"}], "classes": ["c", "p"]}
        sys.path[:0] = [os.path.dirname(path), os.path.dirname(os.path.dirname(path))]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        with _no_bytecode():
            spec.loader.exec_module(mod)
            try:
                mod.read_cohort_map(mp, man)
            except NotImplementedError:
                return "raises NotImplementedError (the contract's stub)"
            except Exception:      # noqa: BLE001 -- built: it refused by its own rules
                pass
        return None
    finally:
        sys.path[:] = saved
        sys.modules.pop(name, None)
        shutil.rmtree(tmp, ignore_errors=True)


def install_cohort_map(clone, sbix):
    """Block 9's reader in the test's Sbi-extractor tree: the checkout's own
    export_windows/cohort_map.py when it reads a map (with its siblings and
    the root modules, should it import one), else FAKE_COHORT_MAP. Returns
    which one, for the run's printout."""
    dst = os.path.join(sbix, *IC.COHORT_MAP_REL)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = os.path.join(clone, *IC.COHORT_MAP_REL)
    why = probe_cohort_map(src)
    if why is None:
        for d_src, d_dst in ((os.path.dirname(src), os.path.dirname(dst)), (clone, sbix)):
            for f in os.listdir(d_src):
                if f.endswith(".py") and not os.path.exists(os.path.join(d_dst, f)):
                    shutil.copy2(os.path.join(d_src, f), d_dst)
        return "the checkout's own %s (Block 9 built)" % src
    with open(dst, "w") as fh:
        fh.write(FAKE_COHORT_MAP)
    return "the test's stand-in FAKE_COHORT_MAP (the checkout's %s %s)" % (src, why)


def _age(top):
    """Every folder and file under `top` (and itself) a month old, as the template."""
    old = _TEMPLATE["old"]
    for dp, dns, fns in os.walk(top):
        for n in dns + fns:
            os.utime(os.path.join(dp, n), (old, old))
    os.utime(top, (old, old))


def build_template():
    """The consistent tree, written once; each check copies it."""
    if "root" in _TEMPLATE:
        return _TEMPLATE["root"]
    clone = ST.sbix_clone()
    if clone is None or not os.path.isfile(os.path.join(clone, "sim_reextract", "profiles",
                                                        "linddup.sh")):
        raise unittest.SkipTest("no Sbi-extractor checkout with profiles/linddup.sh (SBIX_CLONE)")
    if not os.path.isfile(os.path.join(clone, "sim_reextract", "run_mea_task.pbs")):
        raise unittest.SkipTest("the Sbi-extractor checkout %s is before D-113 (no "
                                "sim_reextract/run_mea_task.pbs)" % clone)
    root = tempfile.mkdtemp(prefix="inputs_check_tpl_")
    _TEMPLATE["old"] = time.time() - 30 * 86400.0          # a campaign that ended a month ago
    sbix = os.path.join(root, "sbix")
    os.makedirs(os.path.join(sbix, "sim_reextract", "profiles"))
    shutil.copy2(os.path.join(clone, "sbi_labels.py"), sbix)
    shutil.copy2(os.path.join(clone, "sim_reextract", "profiles", "linddup.sh"),
                 os.path.join(sbix, "sim_reextract", "profiles"))
    # D-112's plan option and D-113's side of the job protocol, as the checkout has them
    for rel in ("env.sh", "sim_reextract/sim_reextract_plan.py") + IC.SBIX_PROTOCOL_SCRIPTS:
        dst = os.path.join(sbix, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(clone, rel), dst)
    _TEMPLATE["cohort_map"] = install_cohort_map(clone, sbix)       # block 32
    jobs = os.path.join(root, "sbi", "hpc", "pipeline", "jobs")
    os.makedirs(jobs)
    for f in ("stage_job.sh", "activate_env.sh", "chain.sh"):
        shutil.copy2(os.path.join(_PIPE, "jobs", f), jobs)
    sim_main = os.path.join(root, SIM)
    os.makedirs(sim_main)
    _write_registry(sim_main)
    for n, (sw, nt, seed, disp) in enumerate(TASKS):
        write_task(sim_main, CAMP, sw, nt, seed, n, disp)
    write_manifest(os.path.join(root, "extracted", "cohort_manifest.json"))
    shutil.copy2(os.path.join(PROJECT, MAP_NAME), os.path.join(root, MAP_NAME))
    with open(os.path.join(PROJECT, "linddup.json"), "rb") as fh:
        cfg = json.loads(fh.read().decode("ascii"))
    cfg["paths"].update({"sbix_dir": sbix, "sbi_dir": os.path.join(root, "sbi"),
                         "work_root": os.path.join(root, "work"), "banks_root": os.path.join(root, "banks"),
                         "runs_root": os.path.join(root, "runs"), "tune_root": os.path.join(root, "tune")})
    cfg["real"].update({"extract_root": os.path.join(root, "extracted"),
                        "cohort_config": os.path.join(root, "cohort_config.json"),
                        "cohort_map": os.path.join(root, MAP_NAME)})
    cfg["sim"].update({"sim_main": sim_main, "out_root": os.path.join(root, "mea_out_linddup"),
                       "tools": os.path.join(root, "tools")})
    with open(os.path.join(root, "linddup.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    _age(root)
    _TEMPLATE["root"] = root
    return root


def fresh():
    """A copy of the template (times kept); returns its root."""
    tpl = build_template()
    root = tempfile.mkdtemp(prefix="inputs_check_")
    dst = os.path.join(root, "t")
    shutil.copytree(tpl, dst)
    # the configuration names the template's paths: point them at the copy
    path = os.path.join(dst, "linddup.json")
    text = open(path).read().replace(tpl, dst)
    with open(path, "w") as fh:
        fh.write(text)
    return dst


def edit_json(path, change):
    with open(path) as fh:
        d = json.load(fh)
    change(d)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1)


def run(root, *extra):
    """(exit code, printed lines, the --json report)."""
    rep = os.path.join(os.path.dirname(root), "report.json")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = IC.main(["--config", os.path.join(root, "linddup.json"), "--no-qstat",
                      "--quiet-hours", "0", "--json", rep] + list(extra))
    doc = json.load(open(rep)) if os.path.isfile(rep) else None
    return rc, buf.getvalue().splitlines(), doc


def fails(doc):
    return sorted(set(r["check"] for r in doc["checks"] if r["status"] == "FAIL"))


def warns(doc):
    return sorted(set(r["check"] for r in doc["checks"] if r["status"] == "WARN"))


def by_check(doc):
    return dict((r["check"], r) for r in doc["checks"])


def cleanup(root):
    shutil.rmtree(os.path.dirname(root), ignore_errors=True)


def _cfg(root):
    return os.path.join(root, "linddup.json")


def _task(root, n):
    return os.path.join(root, SIM, CAMP, TASKS[n][0])


def _topo(root, n, k):
    return os.path.join(_task(root, n), "topo_%05d" % k)


def _old(*paths):
    """Back to the template's age: a mutation that adds or removes a file
    changes its folder's time, which the being-written rule reads (D-112)."""
    for p in paths:
        os.utime(p, (_TEMPLATE["old"], _TEMPLATE["old"]))


# ----------------------------------------------------------------- checks ---

def test_known_answer():
    root = fresh()
    try:
        rc, out, doc = run(root)
        assert rc == 0 and out[-1] == "INPUTS CHECK: 0 FAIL, 0 WARN", (rc, out[-12:])
        printed = [ln for ln in out if ln[:4] in ("OK  ", "FAIL", "WARN", "NOTE")]
        assert len(printed) == len(doc["checks"]), (len(printed), len(doc["checks"]))   # one line each
        assert any(ln.startswith("OK    topology axis conn_prob") for ln in printed), printed
        s = doc["summary"]
        assert (s["campaigns"], s["n_tasks"], s["n_topologies"], s["n_iterations"], s["sampled"]) \
            == ([CAMP], 2, 5, 18, 5), s
        assert s["members"]["conn_prob"] == {"files": 5, "nan": 0, "distinct": 5}, s["members"]
        rows = by_check(doc)
        assert rows["topology axis conn_prob"]["text"] == "5 distinct sampled values in [0.05, 0.39]", \
            rows["topology axis conn_prob"]
        assert rows["label width"]["text"].startswith("13 active axes + 1 topology axes = 14"), \
            rows["label width"]
        assert rows["on axis"]["status"] == "OK" and \
            rows["on axis"]["text"] == "gbarA boxes [[0.01, 10.0]] inside the declared [0.01, 10.0]", \
            rows["on axis"]
        assert rows["simtime"]["status"] == "OK" and rows["conn_rule"]["status"] == "OK"
        assert rows["batches"]["text"] == "%s 0" % CAMP
        assert rows["registry"]["status"] == "OK" and rows["window"]["text"].endswith("W 18000")
        assert rows["cohort manifest"]["status"] == "OK" and rows["S2 profile"]["status"] == "OK"
        # D-112: linddup.json's plan_args --quiet-hours 6; the month-old tree: nothing left out
        assert s["plan_quiet_hours"] == 6.0 and s["being_written"] == [], s
        assert rows["being written"]["status"] == "OK" and "S2 plans all 2" in rows["being written"]["text"]
        assert rows["plan option"]["status"] == "OK", rows["plan option"]
        assert rows["pace"]["status"] == "NOTE" and "95% under 0.00 h" in rows["pace"]["text"], rows["pace"]
        # D-113: both sides of the job protocol
        assert rows["protocol (SBI)"]["status"] == "OK", rows["protocol (SBI)"]
        assert "none yet (the submission records them" in rows["protocol (SBI)"]["text"]
        assert rows["protocol (Sbi-extractor)"]["status"] == "OK", rows["protocol (Sbi-extractor)"]
        # T32a, block 32's rows: the committed map (D-144: donor = condition; D-146: pairing
        # group = condition x batch; D-145: the five copied wells out) against its 35 wells
        cm = rows["cohort map"]
        assert cm["status"] == "OK" and cm["text"].startswith(
            "35 wells, 5 excluded; 2 donors; 4 pairing groups; included per class control:17, "
            "pathological:13 (%s, sha256 " % os.path.join(root, MAP_NAME)), cm
        assert rows["topology metadata"] == {"status": "OK", "check": "topology metadata",
                                             "text": "every topology folder (5) holds topology_meta.json"}
        assert rows["partial topologies"] == {
            "status": "NOTE", "check": "partial topologies",
            "text": "0 partial topologies in 0 task(s), 0 simulations: the bank plan drops them whole"}, \
            rows["partial topologies"]
        sq = rows["searches"]
        assert sq["status"] == "NOTE" and sq["text"] == (
            "3 search(es): A1: S-A1, max_cores 148; A2: S-A2-only, max_cores 148; A5: S-A5-only, "
            "max_cores 148; the caps add up to 444 cores against resources.max_cores 444 (the "
            "sequencer's job takes 4 of them)"), sq
    finally:
        cleanup(root)


def test_partial():
    root = fresh()
    try:
        top = _topo(root, 0, 1)                      # K = 4: two of its runs kept
        for i in (2, 3):
            for ext in ("npz", "json"):
                os.remove(os.path.join(top, "iter_%05d.%s" % (i, ext)))
        _old(top)
        rc, out, doc = run(root)
        rows = by_check(doc)
        assert rc == 0 and not fails(doc) and not warns(doc), (fails(doc), warns(doc))
        pt = rows["partial topologies"]
        assert pt["status"] == "NOTE" and pt["text"] == (
            "1 partial topologies in 1 task(s), 2 simulations: the bank plan drops them whole -- "
            "%s/%s topo 1: 2 of 4" % (CAMP, TASKS[0][0])), pt
        assert doc["summary"]["n_iterations"] == 16 and rows["topology metadata"]["status"] == "OK"
        # Block 10's count: 3 json and 1 failure line is complete (blank lines not counted);
        # an iteration file without its iter_*.json is not a finished run (f counts the json,
        # the simulations count the iteration files); 0 json beside topology_meta.json is
        # partial, with no simulation
        top = _topo(root, 0, 2)
        for ext in ("npz", "json"):
            os.remove(os.path.join(top, "iter_00003.%s" % ext))
        with open(os.path.join(top, IC.FAILURES), "w") as fh:
            fh.write('{"iter": 3, "error": "diverged"}\n\n   \n')
        _old(top)
        top = _topo(root, 1, 0)
        os.remove(os.path.join(top, "iter_00002.json"))       # its npz kept: 2 of 3, 3 files
        _old(top)
        top = _topo(root, 1, 1)
        for f in glob.glob(os.path.join(top, "iter_*")):
            os.remove(f)
        _old(top)
        rc, out, doc = run(root)
        rows = by_check(doc)
        assert rc == 0 and not fails(doc) and not warns(doc), (fails(doc), warns(doc))
        pt = rows["partial topologies"]
        assert pt["text"] == (
            "3 partial topologies in 2 task(s), 5 simulations: the bank plan drops them whole -- "
            "%s/%s topo 1: 2 of 4, %s/%s topo 0: 2 of 3, %s/%s topo 1: 0 of 3"
            % (CAMP, TASKS[0][0], CAMP, TASKS[1][0], CAMP, TASKS[1][0])), pt
        assert doc["summary"]["n_iterations"] == 12 and doc["summary"]["sampled"] == 4, doc["summary"]
    finally:
        cleanup(root)


def test_being_written():
    root = fresh()
    try:
        young = _task(root, 1)
        os.utime(os.path.join(young, "topo_00001"), None)        # a file written there just now
        rc, out, doc = run(root)
        rows = by_check(doc)
        s = doc["summary"]
        name = "%s/%s" % (CAMP, TASKS[1][0])
        assert rc == 0 and not fails(doc) and not warns(doc), doc
        assert s["being_written"] == [name] and s["sampled"] == 3, s
        bw = rows["being written"]
        assert bw["status"] == "NOTE" and bw["text"].startswith("1 of 2 task(s) changed less than 6 h ago") \
            and name in bw["text"], bw
        assert s["members"]["conn_prob"] == {"files": 3, "nan": 0, "distinct": 3}, s["members"]
        # the pace: task 0's files written 10 h apart (a month ago), against the plan's 6 h
        base = time.time() - 30 * 86400.0
        files = sorted(os.path.join(dp, f) for dp, _d, fns in os.walk(_task(root, 0))
                       for f in fns if f.startswith("iter_") and f.endswith(".npz"))
        for j, f in enumerate(files):
            os.utime(f, (base + j * 36000.0, base + j * 36000.0))
        rc, out, doc = run(root)
        warn = [r for r in doc["checks"] if r["status"] == "WARN"]
        assert rc == 0 and [r["check"] for r in warn] == ["pace"], warn
        assert "the longest 10.00 h (in %s/%s)" % (CAMP, TASKS[0][0]) in warn[0]["text"], warn
        assert "--quiet-hours 6 is not longer than most gaps" in warn[0]["text"], warn
    finally:
        cleanup(root)


def test_reference():
    root = fresh()
    try:
        path = os.path.join(_topo(root, 0, 1), "iter_00001.npz")
        got = IC.read_scalars(path)
        with np.load(path, allow_pickle=False) as z:
            want = {}
            for k in z.files:
                a = z[k]
                if a.ndim == 0:
                    v = a[()]
                    want[k] = str(v) if isinstance(v, str) else float(v)
        assert got == want, (got, want)
        assert "spk_N_t" not in got and "theta" not in got and got["conn_prob"] == CONN[(0, 1)]
        n = sum(1 for dp, _d, fns in os.walk(os.path.join(root, SIM))
                for f in fns if f.startswith("iter_") and f.endswith(".npz"))
        rc, out, doc = run(root)
        assert doc["summary"]["n_iterations"] == n == 18
        # block 32: the scan's raw counts against a glob of each topology folder, on a
        # tree whose json and npz differ (one npz without its json, one failures file)
        os.remove(os.path.join(_topo(root, 1, 0), "iter_00001.json"))
        with open(os.path.join(_topo(root, 0, 0), IC.FAILURES), "w") as fh:
            fh.write("{}\n")
        cfg = IC.C.load_config(_cfg(root))
        _c, tasks = IC.scan_campaign(cfg.data["sim"]["sim_main"], cfg.data["sim"]["campaign_glob"])
        seen = []
        for t in tasks:
            for k, r in t["raw"].items():
                top = os.path.join(t["dir"], "topo_%05d" % k)
                assert r["json"] == len(glob.glob(os.path.join(top, "iter_*.json"))), (top, r)
                assert len(t["topo"][k]) == len(glob.glob(os.path.join(top, "iter_*.npz"))), (top, t)
                assert r["meta"] == os.path.isfile(os.path.join(top, "topology_meta.json")), (top, r)
                assert r["failures"] == os.path.isfile(os.path.join(top, IC.FAILURES)), (top, r)
                seen.append((r["json"], len(t["topo"][k]), r["failures"]))
        assert seen == [(4, 4, True), (4, 4, False), (4, 4, False), (2, 3, False), (3, 3, False)], seen
        # failure_lines against str.splitlines: blank and blank-only lines not counted
        fp = os.path.join(os.path.dirname(root), "failures.jsonl")
        body = '{"a": 1}\n\n  \t \n{"b": 2}\r\n{"c": 3}'
        with open(fp, "w", newline="") as fh:
            fh.write(body)
        assert IC.failure_lines(fp) == sum(1 for ln in body.splitlines() if ln.strip()) == 3
        # the cohort map row's counts against the csv module's reading of the map
        rows = map_rows(os.path.join(root, MAP_NAME))
        inc = [r for r in rows if r["include"].strip().lower() in ("yes", "true", "1")]
        per = {}
        for r in inc:
            c = r["out_dir"].split("/")[0]
            per[c] = per.get(c, 0) + 1
        want = ("%d wells, %d excluded; %d donors; %d pairing groups; included per class %s"
                % (len(rows), len(rows) - len(inc), len(set(r["donor"].strip() for r in inc)),
                   len(set(r["rep_group"].strip() for r in inc)),
                   ", ".join("%s:%d" % (c, per.get(c, 0)) for c in ("control", "pathological"))))
        assert by_check(doc)["cohort map"]["text"].startswith(want + " ("), (want, by_check(doc)["cohort map"])
    finally:
        cleanup(root)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def _gate_tree(root):
    """The gate variant (D-107's contact gate, as MEAMA had it): stoa_gate_p
    per topology in every iteration file, stoa_gate_lo / stoa_gate_hi in
    every job_args.json, and the configuration's topology axes [conn_prob,
    stoa_gate_p] with the widths 15."""
    sim = os.path.join(root, SIM)
    shutil.rmtree(os.path.join(sim, CAMP))
    for n, (sw, nt, seed, disp) in enumerate(TASKS):
        write_task(sim, CAMP, sw, nt, seed, n, disp, gates=True)
    _age(os.path.join(sim, CAMP))

    def ch(d):
        d["bank"].update(topology_axes=["conn_prob", "stoa_gate_p"], expected_p=15)
        d["search"]["shape_anchors"].update(p=15, d_theta=15)
    edit_json(_cfg(root), ch)


def test_invariants():
    root = fresh()
    try:
        rc, out, doc = run(root, "--quiet-hours", "1000000")
        warn = [r for r in doc["checks"] if r["status"] == "WARN"]
        assert rc == 0 and [r["check"] for r in warn] == ["campaign"], warn
        assert "a job may still be writing it" in warn[0]["text"]
        rc, out, doc = run(root)
        assert not warns(doc)
    finally:
        cleanup(root)
    root = fresh()
    try:                                  # NaN-valued kernel axes in every file, not excluded
        sim = os.path.join(root, SIM)
        shutil.rmtree(os.path.join(sim, CAMP))
        for n, (sw, nt, seed, disp) in enumerate(TASKS):
            write_task(sim, CAMP, sw, nt, seed, n, disp, kernel_nan=True)
        _age(os.path.join(sim, CAMP))     # written a month ago, as the template (D-112)
        rc, out, doc = run(root)
        assert rc == 0 and not fails(doc), fails(doc)
        assert warns(doc) == ["freeze exclude beta_conn", "freeze exclude d0_conn",
                              "freeze exclude p0_conn"], warns(doc)
    finally:
        cleanup(root)
    root = fresh()
    try:                                  # stoa_gate_p as the configuration names it (block 32)
        _gate_tree(root)
        rc, out, doc = run(root)
        rows = by_check(doc)
        assert rc == 0 and not fails(doc) and not warns(doc), (fails(doc), warns(doc))
        g = rows["topology axis stoa_gate_p"]
        assert g["status"] == "OK" and g["text"] == ("5 distinct sampled values in [0.25, 0.92]; box "
                                                     "[0.2, 1.0] from job_args"), g
        assert rows["label width"]["text"].startswith("13 active axes + 2 topology axes = 15")
        assert doc["summary"]["members"]["stoa_gate_p"] == {"files": 5, "nan": 0, "distinct": 5}
    finally:
        cleanup(root)


def _snapshot(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = os.path.join(dp, n)
            st = os.lstat(p)
            out[os.path.relpath(p, root)] = (st.st_size if n in fns else None, st.st_mtime_ns)
    return out


def test_contract():
    root = fresh()
    try:
        before = _snapshot(root)
        rc, out, doc = run(root)
        assert rc == 0 and _snapshot(root) == before, "the tree changed"
        # a fresh process: every import from the tree is a first one (sbi_labels, the
        # registry's modules, the cohort map's reader); bytecode allowed by the environment
        env = dict(os.environ)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        r = subprocess.run([sys.executable, os.path.join(_PIPE, "tools", "inputs_check.py"), "--config",
                            _cfg(root), "--no-qstat", "--quiet-hours", "0"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, env=env,
                           cwd=os.path.dirname(root), timeout=300)
        lines = r.stdout.splitlines()
        assert r.returncode == 0 and lines[-1] == "INPUTS CHECK: 0 FAIL, 0 WARN", lines[-5:]
        after = _snapshot(root)
        assert after == before, sorted(set(after) ^ set(before))[:10] or "a time or a size changed"
        assert sorted(doc) == ["checks", "config", "config_sha256", "fail", "summary", "warn"], sorted(doc)
        assert all(sorted(r) == ["check", "status", "text"] for r in doc["checks"])
        edit_json(_cfg(root), lambda d: d["bank"].update(n_e=1))
        rc, out, doc = run(root)
        assert rc == 1 and out[-1] == "INPUTS CHECK: 1 FAIL, 0 WARN", out[-3:]
        edit_json(_cfg(root), lambda d: d["bank"].update(n_e="nine"))
        rc, out, doc = run(root)
        assert rc == 2 and "is refused" in out[0], out[:3]
    finally:
        cleanup(root)


def test_determinism():
    root = fresh()
    try:
        _r1, _o1, d1 = run(root)
        _r2, _o2, d2 = run(root)
        strip = lambda d: [(r["status"], r["check"], r["text"]) for r in d["checks"]
                           if not r["text"].startswith("last change")]
        assert strip(d1) == strip(d2) and d1["summary"] == d2["summary"]
        cfg = IC.C.load_config(_cfg(root))
        _c, tasks = IC.scan_campaign(cfg.data["sim"]["sim_main"], cfg.data["sim"]["campaign_glob"])
        picks = IC.sample_files(tasks, 3)
        assert [(n, k) for n, k, _p in picks] == [(0, 0), (1, 0), (0, 1)], picks   # round-robin
        assert [(n, k) for n, k, _p in IC.sample_files(tasks, 200)] == \
            [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    finally:
        cleanup(root)


# ------------------------------------------------- the mutations (each one FAIL) ---

def _m_simtime(root):
    edit_json(os.path.join(_task(root, 1), "job_args.json"), lambda d: d.update(simtime=200.0))


def _m_wsize(root):                       # 180 s is not a whole number of 0.07 s bins
    write_manifest(os.path.join(root, "extracted", "cohort_manifest.json"), w_size=0.07)


def _m_ne(root):
    edit_json(_cfg(root), lambda d: d["bank"].update(n_e=1))


def _m_width(root):
    edit_json(_cfg(root), lambda d: d["bank"].update(expected_p=13))


def _m_onbox(root):
    def ch(d):
        d["param_bounds"][1] = [0.005, 10.0]          # gbarA wider than the declared [0.01, 10]
    edit_json(os.path.join(_task(root, 0), "manifest.json"), ch)


def _m_rule(root):
    edit_json(os.path.join(_task(root, 1), "job_args.json"), lambda d: d.update(conn_rule="weibull"))


def _m_axis_absent(root):
    for i in range(4):
        p = os.path.join(_topo(root, 0, 1), "iter_%05d.npz" % i)
        with np.load(p) as z:
            d = dict((k, z[k]) for k in z.files if k != "conn_prob")
        np.savez_compressed(p, **d)


def _m_gate_hi(root):
    _gate_tree(root)
    edit_json(os.path.join(_task(root, 1), "job_args.json"), lambda d: d.pop("stoa_gate_hi"))


def _m_unknown_axis(root):
    def ch(d):
        d["bank"].update(topology_axes=["conn_prob", "gate_x"], expected_p=15)
        d["search"]["shape_anchors"].update(p=15, d_theta=15)
    edit_json(_cfg(root), ch)


def _m_exclude(root):
    edit_json(_cfg(root), lambda d: d["freeze"]["exclude"].update(p0_conn="not in the data"))


def _m_batches(root):
    camp2 = "campaign_lcdup15_rho1300_neur_n02"
    write_task(os.path.join(root, SIM), camp2, "sweep_cpu_task0000", 1, 7003, 2, (3, 1))
    edit_json(_cfg(root), lambda d: d["sim"].update(campaign_glob="campaign_lcdup15_rho1300_neur_n0*"))


def _m_profile(root):
    os.remove(os.path.join(root, "sbix", "sim_reextract", "profiles", "linddup.sh"))


def _m_manifest_sha(root):
    with open(os.path.join(root, "extracted", "cohort_manifest.json"), "a") as fh:
        fh.write("\n")


def _m_all_written(root):
    for sw, _n, _s, _d in TASKS:
        os.utime(os.path.join(root, SIM, CAMP, sw), None)


def _m_plan_option(root):
    with open(os.path.join(root, "sbix", "sim_reextract", "sim_reextract_plan.py"), "w") as fh:
        fh.write("# a plan before D-112\n")


def _m_plan_args(root):
    edit_json(_cfg(root), lambda d: d["sim"].update(plan_args="--quiet-hours soon"))


def _m_protocol_sbi(root):
    os.remove(os.path.join(root, "sbi", "hpc", "pipeline", "jobs", "stage_job.sh"))


def _m_protocol_sbix(root):
    with open(os.path.join(root, "sbix", "sim_reextract", "run_mea_task.pbs"), "w") as fh:
        fh.write("#!/bin/bash\n# a job script before D-113\n")


def _edit_map(root, change):
    p = os.path.join(root, MAP_NAME)
    with open(p) as fh:
        lines = fh.read().splitlines()
    with open(p, "w") as fh:
        fh.write("\n".join(change(lines)) + "\n")


def _m_map_missing_well(root):            # T32b: a manifest well missing from the map
    _edit_map(root, lambda ls: ls[:1] + ls[2:])


def _m_map_extra_row(root):               # T32b: a map row of no well
    _edit_map(root, lambda ls: ls + ["control/DATA_C_Batch9/ptrain_Z9,control,control_Batch9,yes,"
                                     "no such well"])


def _m_map_bad_value(root):               # a bad value: include outside the set
    _edit_map(root, lambda ls: ls[:1] + [ls[1].replace(",yes,", ",maybe,", 1)] + ls[2:])


def _m_map_file_absent(root):
    os.remove(os.path.join(root, MAP_NAME))


def _m_map_module_absent(root):
    os.remove(os.path.join(root, "sbix", *IC.COHORT_MAP_REL))


def _m_map_module_stub(root):
    with open(os.path.join(root, "sbix", *IC.COHORT_MAP_REL), "w") as fh:
        fh.write(STUB_COHORT_MAP)


def _m_topology_meta(root):               # T32b: a topology folder without topology_meta.json
    top = _topo(root, 1, 0)
    os.remove(os.path.join(top, "topology_meta.json"))
    _old(top)


def _m_dispatch_key(root):                # T32b: a job_args without n_params_per_worker
    p = os.path.join(_task(root, 1), "job_args.json")
    edit_json(p, lambda d: d.pop("n_params_per_worker"))
    _old(p)


def _m_too_many(root):                    # T32b: f(t, k) > K(t), a failure beside K finished runs
    top = _topo(root, 0, 2)
    with open(os.path.join(top, IC.FAILURES), "w") as fh:
        fh.write('{"iter": 4, "error": "killed"}\n')
    _old(top)


# (mutation, the one row that FAILs, a text its line must hold or None)
MUTATIONS = ((_m_simtime, "simtime", None), (_m_wsize, "window", None), (_m_ne, "n_e", None),
             (_m_width, "label width", None), (_m_onbox, "on axis", None),
             (_m_rule, "conn_rule", None), (_m_axis_absent, "topology axis conn_prob", None),
             (_m_gate_hi, "topology axis stoa_gate_p", "stoa_gate_lo / stoa_gate_hi"),
             (_m_unknown_axis, "topology axis gate_x", None),
             (_m_exclude, "freeze exclude p0_conn", None), (_m_batches, "batches", None),
             (_m_profile, "S2 profile", None), (_m_manifest_sha, "cohort manifest", None),
             (_m_all_written, "being written", None), (_m_plan_option, "plan option", None),
             (_m_plan_args, "plan args", None), (_m_protocol_sbi, "protocol (SBI)", None),
             (_m_protocol_sbix, "protocol (Sbi-extractor)", None),
             # block 32
             (_m_map_missing_well, "cohort map", "the real bank (S6) refuses this map"),
             (_m_map_extra_row, "cohort map", "the real bank (S6) refuses this map"),
             (_m_map_bad_value, "cohort map", "the real bank (S6) refuses this map"),
             (_m_map_file_absent, "cohort map", "absent"),
             (_m_map_module_absent, "cohort map", IC.PULL_BLOCK9),
             (_m_map_module_stub, "cohort map", IC.PULL_BLOCK9 + " ("),
             (_m_topology_meta, "topology metadata",
              "1 topology folder(s) without topology_meta.json: %s/%s/topo_00000 -- move them out of "
              "their sweep folder before S2: the virtual MEA stops on the whole task (ANN reply "
              "2026-11-17, sec. 2.2)" % (CAMP, TASKS[1][0])),
             (_m_dispatch_key, "partial topologies",
              "%s/%s: job_args.json lacks a positive integer (n_params_per_worker absent)"
              % (CAMP, TASKS[1][0])),
             (_m_too_many, "partial topologies", "%s/%s topo 2: 5 finished of 4 dispatched"
              % (CAMP, TASKS[0][0])))


# ---------------------------------------- the v11 variants (a NOTE, no FAIL) ---

def _v_no_map(root):
    edit_json(_cfg(root), lambda d: d["real"].update(cohort_map=None))


def _v_no_manifest(root):
    for f in ("cohort_manifest.json", "cohort_manifest.json.sha256"):
        os.remove(os.path.join(root, "extracted", f))


def _v_no_drop(root):
    edit_json(_cfg(root), lambda d: d["bank"].update(drop_partial_topologies=False))


def _v_single_search(root):
    def ch(d):
        d["searches"] = None
        d["search"].update(campaign="S-A1", arm="A1")
    edit_json(_cfg(root), ch)


def _v_uncapped(root):
    edit_json(_cfg(root), lambda d: d["searches"][0].update(max_cores=None))


# (variant, its row, the row's status (None: no such row), a text its line must hold)
VARIANTS = ((_v_no_map, "cohort map", "NOTE", "no cohort map: donor = well"),
            (_v_no_manifest, "cohort map", "NOTE", "absent: nothing to check the map"),
            (_v_no_drop, "partial topologies", "NOTE", "not checked (drop_partial_topologies false)"),
            (_v_single_search, "searches", None, None),
            (_v_uncapped, "searches", "NOTE",
             "A1: S-A1, no cap of its own; A2: S-A2-only, max_cores 148; A5: S-A5-only, max_cores "
             "148; the caps add up to 296 cores against resources.max_cores 444 (the sequencer's job "
             "takes 4 of them); 1 search(es) without a cap of their own: the project's cap only"))


def test_edge_cases():
    bad = []
    for mutate, label, text in MUTATIONS:
        root = fresh()
        try:
            mutate(root)
            rc, out, doc = run(root)
            line = [r["text"] for r in doc["checks"] if r["check"] == label and r["status"] == "FAIL"]
            if rc != 1 or fails(doc) != [label]:
                bad.append("%s: exit %s, FAIL %r (want [%r])" % (mutate.__name__, rc, fails(doc), label))
            elif text is not None and not any(text in x for x in line):
                bad.append("%s: the %s line %r lacks %r" % (mutate.__name__, label, line, text))
        finally:
            cleanup(root)
    for vary, label, status, text in VARIANTS:
        root = fresh()
        try:
            vary(root)
            rc, out, doc = run(root)
            row = by_check(doc).get(label)
            if rc != 0 or fails(doc):
                bad.append("%s: exit %s, FAIL %r (want none)" % (vary.__name__, rc, fails(doc)))
            elif status is None and row is not None:
                bad.append("%s: a %s row %r (want none)" % (vary.__name__, label, row))
            elif status is not None and (row is None or row["status"] != status or text not in row["text"]):
                bad.append("%s: the %s row %r (want %s with %r)" % (vary.__name__, label, row, status, text))
        finally:
            cleanup(root)
    # K(t)'s reading (Block 10): both keys positive integers, nothing else
    assert IC.dispatch_size({"_resolved_n_workers": 48, "n_params_per_worker": 4}) == (192, None)
    for ja in ({"_resolved_n_workers": 48}, {"_resolved_n_workers": 48, "n_params_per_worker": 0},
               {"_resolved_n_workers": 48, "n_params_per_worker": 4.0},
               {"_resolved_n_workers": True, "n_params_per_worker": 4}, None):
        k, why = IC.dispatch_size(ja)
        if k is not None or not why.startswith("job_args.json lacks a positive integer"):
            bad.append("dispatch_size(%r) = %r" % (ja, (k, why)))
    assert not bad, "\n".join(bad)


CHECKS = [("test_known_answer", test_known_answer), ("test_partial", test_partial),
          ("test_being_written", test_being_written), ("test_reference", test_reference),
          ("test_convergence", test_convergence), ("test_invariants", test_invariants),
          ("test_contract", test_contract), ("test_determinism", test_determinism),
          ("test_edge_cases", test_edge_cases)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn()
            rows.append((name, "PASS", "", time.time() - t1))
        except unittest.SkipTest as exc:
            rows.append((name, "SKIP", str(exc), time.time() - t1))
        except AssertionError as exc:
            bad += 1
            rows.append((name, "FAIL", str(exc)[:2500], time.time() - t1))
        except Exception:
            bad += 1
            rows.append((name, "ERROR", traceback.format_exc()[-2500:], time.time() - t1))
    if _TEMPLATE.get("root"):
        shutil.rmtree(_TEMPLATE["root"], ignore_errors=True)
    print("cohort map reader: %s" % _TEMPLATE.get("cohort_map", "not installed (no template)"))
    print("| check | result | seconds |")
    print("|---|---|---|")
    for name, res, msg, dt in rows:
        print("| %s | %s | %.1f |" % (name, res, dt))
    for name, res, msg, dt in rows:
        if msg:
            print("\n== %s: %s\n%s" % (name, res, msg))
    print("\n%d of %d checks pass in %.0f s" % (sum(1 for r in rows if r[1] == "PASS"), len(rows),
                                               time.time() - t0))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
