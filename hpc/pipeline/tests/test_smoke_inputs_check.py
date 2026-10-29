#!/usr/bin/env python3
"""Smoke test for block 9 of specs/SPEC.md: the inputs check
(`tools/inputs_check.py`), on a MEAMA-like campaign tree, a cohort manifest
and a configuration derived from `projects/linddup/linddup.json`, written to
a temporary folder.

Checks (specs/SPEC.md block 9, "Test oracles")
    test_known_answer   the consistent tree: 0 FAIL, 0 WARN at quiet hours 0;
                        1 campaign, 2 tasks, 5 topologies, 10 iteration
                        files; stoa_gate_p's 5 distinct sampled values;
                        exit 0 and the last line
    test_reference      read_scalars equals numpy.load's own 0-d members on a
                        file holding arrays too; the counts equal an os.walk
                        count of iter_*.npz
    test_convergence    skipped (not a discretisation)
    test_invariants     the WARNs: a quiet-hours window covering the tree's
                        age gives the campaign WARN (0 h none); NaN-valued
                        p0_conn in every file, not excluded, gives the
                        freeze-exclude WARN and no FAIL
    test_contract       read-only: every file's and folder's size and
                        modification time unchanged by a run; the --json
                        report's keys; exit 1 with a FAIL, 2 on a refused
                        configuration
    test_determinism    two runs give the same rows (but the age text) and
                        the same sample; the sample is round-robin
    test_edge_cases     each mutation of the block's list makes its named
                        check FAIL and no other

Run
    python3 hpc/pipeline/tests/test_smoke_inputs_check.py
        About 10 s. Needs the Sbi-extractor checkout (SBIX_CLONE or a known
        place: its sbi_labels.py and sim_reextract/profiles/linddup.sh) and
        numpy.

Pure ASCII, LF only.
"""

import contextlib
import io
import json
import os
import shutil
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

CAMP = "campaign_meama_rho1600_n01"
N_REG = 37                        # the MEAMA registry's width, O_N at 36
ACTIVE = list(range(5, 37))       # 32 active axes, as the tripartite group
TASKS = (("sweep_cpu_task0000", 3, 7001), ("sweep_intel_task0001", 2, 7002))   # sweep, topologies, seed
GATES = {(0, 0): 0.31, (0, 1): 0.47, (0, 2): 0.66, (1, 0): 0.25, (1, 1): 0.92}  # stoa_gate_p per topology
_TEMPLATE = {}


# --------------------------------------------------------------- the tree ---

def _write_registry(sim_main):
    bounds = [[0.5, 2.0]] * (N_REG - 1) + [[0.03, 3.0]]
    theta = [list(b) for b in bounds]
    theta[-1] = [float(np.log(0.03)), float(np.log(3.0))]
    with open(os.path.join(sim_main, "HPC_main_sweep.py"), "w") as fh:
        fh.write("PARAM_BOUNDS = %r\nPARAM_BOUNDS_THETA = %r\nLOG_PARAMS = [%d]\nLOG_BASE = 'natural'\n"
                 "KERNEL_BOUNDS = [[0.1, 1.0], [20.0, 200.0], [0.5, 3.0]]\n"
                 % (bounds, theta, N_REG - 1))
    names = ["p%02d" % j for j in range(N_REG - 1)] + ["O_N"]
    with open(os.path.join(sim_main, "HPC_single_run.py"), "w") as fh:
        fh.write("PARAM_NAMES = %r\nPARAM_UNITS = %r\nSWEEP_GROUPS = %r\n"
                 % (names, ["a.u."] * N_REG, {"tripartite": ACTIVE}))
    return names, bounds


def write_task(sim_main, camp, sweep, n_topo, seed, n_task, names, bounds, gates=True,
               kernel_nan=False):
    td = os.path.join(sim_main, camp, sweep)
    os.makedirs(td)
    ja = {"simtime": 200.0, "conn_rule": "flat", "conn_prob_lo": 0.1, "conn_prob_hi": 0.6,
          "stoa_gate_lo": 0.2, "stoa_gate_hi": 1.0, "_resolved_seed_master": seed,
          "seed_master": None, "seed_stoa_gate": 3, "mode": "Full", "sweep_group": "tripartite"}
    mf = {"active_indices": ACTIVE, "sweep_group": "tripartite", "param_names": names,
          "log_params": ["O_N"], "log_transform": "natural", "conn_rule": "flat",
          "param_bounds": bounds, "manifest_version": 4}
    for name, doc in (("job_args.json", ja), ("manifest.json", mf)):
        with open(os.path.join(td, name), "w") as fh:
            json.dump(doc, fh, indent=1)
    rng = np.random.default_rng(seed)
    for k in range(n_topo):
        os.makedirs(os.path.join(td, "topo_%05d" % k))
        cp = float(rng.uniform(0.1, 0.6))
        for i in range(2):
            extra = {"stoa_gate_p": np.float64(GATES.get((n_task, k), 0.5))} if gates else {}
            if kernel_nan:
                extra.update(p0_conn=np.float64("nan"), d0_conn=np.float64("nan"),
                             beta_conn=np.float64("nan"))
            np.savez_compressed(os.path.join(td, "topo_%05d" % k, "iter_%05d.npz" % i),
                                params=np.ones(N_REG), theta=np.zeros(N_REG),
                                conn_rule=np.array("flat"), conn_prob=np.float64(cp),
                                topo_idx=np.int32(k), seed_run=np.int64(seed * 100 + 10 * k + i),
                                spk_N_t=np.sort(rng.uniform(0, 200, 50)).astype(np.float32),
                                spk_N_i=rng.integers(0, 40, 50).astype(np.int32), **extra)
    return td


def _sign(path):
    import hashlib
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as fh:
        fh.write("%s  %s\n" % (h, os.path.basename(path)))


def write_manifest(path, w_size=0.01, eps=9):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {"manifest_version": 3, "n_wells": 2, "n_subsets_per_well": 9,
           "classes": ["control", "pathological"],
           "preprocessing": {"w_size": w_size, "gaussian_window": 0.02,
                             "electrodes_per_subset": eps, "fs_raw": 10110.09},
           "wells": [{"out_dir": "control/a/w1", "T_rec": 1200.0},
                     {"out_dir": "pathological/b/w2", "T_rec": 1200.0}]}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    _sign(path)


def build_template():
    """The consistent tree, written once; each check copies it."""
    if _TEMPLATE:
        return _TEMPLATE["root"]
    clone = ST.sbix_clone()
    if clone is None or not os.path.isfile(os.path.join(clone, "sim_reextract", "profiles",
                                                        "linddup.sh")):
        raise unittest.SkipTest("no Sbi-extractor checkout with profiles/linddup.sh (SBIX_CLONE)")
    root = tempfile.mkdtemp(prefix="inputs_check_tpl_")
    sbix = os.path.join(root, "sbix")
    os.makedirs(os.path.join(sbix, "sim_reextract", "profiles"))
    shutil.copy2(os.path.join(clone, "sbi_labels.py"), sbix)
    shutil.copy2(os.path.join(clone, "sim_reextract", "profiles", "linddup.sh"),
                 os.path.join(sbix, "sim_reextract", "profiles"))
    sim_main = os.path.join(root, "MEAMA")
    os.makedirs(sim_main)
    names, bounds = _write_registry(sim_main)
    for n, (sw, nt, seed) in enumerate(TASKS):
        write_task(sim_main, CAMP, sw, nt, seed, n, names, bounds)
    write_manifest(os.path.join(root, "extracted", "cohort_manifest.json"))
    with open(os.path.join(_PIPE, "projects", "linddup", "linddup.json"), "rb") as fh:
        cfg = json.loads(fh.read().decode("ascii"))
    cfg["paths"].update({"sbix_dir": sbix, "sbi_dir": os.path.join(root, "sbi"),
                         "work_root": os.path.join(root, "work"), "banks_root": os.path.join(root, "banks"),
                         "runs_root": os.path.join(root, "runs"), "tune_root": os.path.join(root, "tune")})
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort_config.json")
    cfg["sim"].update({"sim_main": sim_main, "out_root": os.path.join(root, "mea_out_linddup"),
                       "tools": os.path.join(root, "tools")})
    with open(os.path.join(root, "linddup.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    old = time.time() - 30 * 86400.0          # a campaign that ended a month ago
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            os.utime(os.path.join(dp, n), (old, old))
    _TEMPLATE.update(root=root, names=names, bounds=bounds)
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


def cleanup(root):
    shutil.rmtree(os.path.dirname(root), ignore_errors=True)


# ----------------------------------------------------------------- checks ---

def test_known_answer():
    root = fresh()
    try:
        rc, out, doc = run(root)
        assert rc == 0 and out[-1] == "INPUTS CHECK: 0 FAIL, 0 WARN", (rc, out[-12:])
        printed = [ln for ln in out if ln[:4] in ("OK  ", "FAIL", "WARN", "NOTE")]
        assert len(printed) == len(doc["checks"]), (len(printed), len(doc["checks"]))   # one line each
        assert any(ln.startswith("OK    topology axis stoa_gate_p") for ln in printed), printed
        s = doc["summary"]
        assert (s["campaigns"], s["n_tasks"], s["n_topologies"], s["n_iterations"], s["sampled"]) \
            == ([CAMP], 2, 5, 10, 5), s
        assert s["members"]["stoa_gate_p"] == {"files": 5, "nan": 0, "distinct": 5}, s["members"]
        rows = dict((r["check"], r) for r in doc["checks"])
        assert rows["topology axis stoa_gate_p"]["status"] == "OK"
        assert "5 distinct sampled values in [0.25, 0.92]" in rows["topology axis stoa_gate_p"]["text"]
        assert "box [0.2, 1.0] from job_args" in rows["topology axis stoa_gate_p"]["text"]
        assert rows["label width"]["text"].startswith("32 active axes + 2 topology axes = 34")
        assert rows["batches"]["text"] == "%s 0" % CAMP
        assert rows["registry"]["status"] == "OK" and rows["window"]["text"].endswith("W 20000")
        assert rows["cohort manifest"]["status"] == "OK" and rows["S2 profile"]["status"] == "OK"
    finally:
        cleanup(root)


def test_reference():
    root = fresh()
    try:
        path = os.path.join(root, "MEAMA", CAMP, TASKS[0][0], "topo_00001", "iter_00001.npz")
        got = IC.read_scalars(path)
        with np.load(path, allow_pickle=False) as z:
            want = {}
            for k in z.files:
                a = z[k]
                if a.ndim == 0:
                    v = a[()]
                    want[k] = str(v) if isinstance(v, str) else float(v)
        assert got == want, (got, want)
        assert "spk_N_t" not in got and got["stoa_gate_p"] == GATES[(0, 1)]
        n = sum(1 for dp, _d, fns in os.walk(os.path.join(root, "MEAMA"))
                for f in fns if f.startswith("iter_") and f.endswith(".npz"))
        rc, out, doc = run(root)
        assert doc["summary"]["n_iterations"] == n == 10
    finally:
        cleanup(root)


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    root = fresh()
    try:
        rc, out, doc = run(root, "--quiet-hours", "1000000")
        warn = [r for r in doc["checks"] if r["status"] == "WARN"]
        assert rc == 0 and [r["check"] for r in warn] == ["campaign"], warn
        assert "a job may still be writing it" in warn[0]["text"]
        rc, out, doc = run(root)
        assert not [r for r in doc["checks"] if r["status"] == "WARN"]
    finally:
        cleanup(root)
    root = fresh()
    try:                                  # NaN-valued kernel axes in every file, not excluded
        sim = os.path.join(root, "MEAMA")
        shutil.rmtree(os.path.join(sim, CAMP))
        for n, (sw, nt, seed) in enumerate(TASKS):
            write_task(sim, CAMP, sw, nt, seed, n, _TEMPLATE["names"], _TEMPLATE["bounds"],
                       kernel_nan=True)
        rc, out, doc = run(root)
        warn = sorted(r["check"] for r in doc["checks"] if r["status"] == "WARN")
        assert rc == 0 and not fails(doc), fails(doc)
        assert warn == ["freeze exclude beta_conn", "freeze exclude d0_conn",
                        "freeze exclude p0_conn"], warn
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
        assert _snapshot(root) == before, "the tree changed"
        assert sorted(doc) == ["checks", "config", "config_sha256", "fail", "summary", "warn"], sorted(doc)
        assert all(sorted(r) == ["check", "status", "text"] for r in doc["checks"])
        edit_json(os.path.join(root, "linddup.json"), lambda d: d["bank"].update(n_e=1))
        rc, out, doc = run(root)
        assert rc == 1 and out[-1] == "INPUTS CHECK: 1 FAIL, 0 WARN", out[-3:]
        edit_json(os.path.join(root, "linddup.json"), lambda d: d["bank"].update(n_e="nine"))
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
        cfg = IC.C.load_config(os.path.join(root, "linddup.json"))
        _c, tasks = IC.scan_campaign(cfg.data["sim"]["sim_main"], cfg.data["sim"]["campaign_glob"])
        picks = IC.sample_files(tasks, 3)
        assert [(n, k) for n, k, _p in picks] == [(0, 0), (1, 0), (0, 1)], picks   # round-robin
        assert [(n, k) for n, k, _p in IC.sample_files(tasks, 200)] == \
            [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    finally:
        cleanup(root)


def _m_simtime(root):
    edit_json(os.path.join(root, "MEAMA", CAMP, TASKS[1][0], "job_args.json"),
              lambda d: d.update(simtime=180.0))


def _m_wsize(root):
    write_manifest(os.path.join(root, "extracted", "cohort_manifest.json"), w_size=0.03)


def _m_ne(root):
    edit_json(os.path.join(root, "linddup.json"), lambda d: d["bank"].update(n_e=1))


def _m_width(root):
    edit_json(os.path.join(root, "linddup.json"), lambda d: d["bank"].update(expected_p=33))


def _m_onbox(root):
    def ch(d):
        d["param_bounds"][36] = [0.01, 3.0]
    edit_json(os.path.join(root, "MEAMA", CAMP, TASKS[0][0], "manifest.json"), ch)


def _m_rule(root):
    edit_json(os.path.join(root, "MEAMA", CAMP, TASKS[1][0], "job_args.json"),
              lambda d: d.update(conn_rule="weibull"))


def _m_gate_absent(root):
    for i in range(2):
        p = os.path.join(root, "MEAMA", CAMP, TASKS[0][0], "topo_00001", "iter_%05d.npz" % i)
        with np.load(p) as z:
            d = dict((k, z[k]) for k in z.files if k != "stoa_gate_p")
        np.savez_compressed(p, **d)


def _m_gate_hi(root):
    edit_json(os.path.join(root, "MEAMA", CAMP, TASKS[1][0], "job_args.json"),
              lambda d: d.pop("stoa_gate_hi"))


def _m_unknown_axis(root):
    def ch(d):
        d["bank"].update(topology_axes=["conn_prob", "stoa_gate_p", "gate_x"], expected_p=35)
        d["search"]["shape_anchors"].update(p=35, d_theta=35)
    edit_json(os.path.join(root, "linddup.json"), ch)


def _m_exclude(root):
    edit_json(os.path.join(root, "linddup.json"),
              lambda d: d["freeze"]["exclude"].update(p0_conn="not in the data"))


def _m_batches(root):
    camp2 = "campaign_meama_rho1600_n02"
    write_task(os.path.join(root, "MEAMA"), camp2, "sweep_cpu_task0000", 1, 7003, 0,
               _TEMPLATE["names"], _TEMPLATE["bounds"])
    edit_json(os.path.join(root, "linddup.json"),
              lambda d: d["sim"].update(campaign_glob="campaign_meama_rho1600_n0*"))


def _m_profile(root):
    os.remove(os.path.join(root, "sbix", "sim_reextract", "profiles", "linddup.sh"))


def _m_manifest_sha(root):
    with open(os.path.join(root, "extracted", "cohort_manifest.json"), "a") as fh:
        fh.write("\n")


MUTATIONS = ((_m_simtime, "simtime"), (_m_wsize, "window"), (_m_ne, "n_e"),
             (_m_width, "label width"), (_m_onbox, "on axis"), (_m_rule, "conn_rule"),
             (_m_gate_absent, "topology axis stoa_gate_p"), (_m_gate_hi, "topology axis stoa_gate_p"),
             (_m_unknown_axis, "topology axis gate_x"), (_m_exclude, "freeze exclude p0_conn"),
             (_m_batches, "batches"), (_m_profile, "S2 profile"),
             (_m_manifest_sha, "cohort manifest"))


def test_edge_cases():
    bad = []
    for mutate, label in MUTATIONS:
        root = fresh()
        try:
            mutate(root)
            rc, out, doc = run(root)
            if rc != 1 or fails(doc) != [label]:
                bad.append("%s: exit %s, FAIL %r (want [%r])" % (mutate.__name__, rc, fails(doc), label))
        finally:
            cleanup(root)
    assert not bad, "\n".join(bad)


CHECKS = [("test_known_answer", test_known_answer), ("test_reference", test_reference),
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
    if _TEMPLATE:
        shutil.rmtree(_TEMPLATE["root"], ignore_errors=True)
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
