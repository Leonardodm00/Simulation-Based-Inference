#!/usr/bin/env python3
"""Smoke test for `tools/perrow_check.py` (specs/SPEC.md block 7, added
2026-10-20): the read-only split of each trial's report-split NLL into the
flow's edge term T and the rest R.

A hand-made campaign folder (a copy of the Giulia configuration with every
path under a temporary folder) holds 200 report rows on 20 donors, d = 3,
theta stored as float32 as the runner stores it, with edge coordinates
planted (|logit theta| > 5): rows 0-9 theta_0 = 0.003, rows 10-14
theta_2 = 0.999, rows 15-16 theta_1 = 1e-4 and theta_0 = 0.006; rows 17 and
18 hold theta_1 = 0.012 and theta_2 = 0.9925, just INSIDE the bound
(|logit| 4.41 and 4.89). Every trial's per-row NLL is T_i + r_i with r_i
known:

    campaign 1 (the configured S-A1)
      good      r = -0.5                          ledger nll = L
      bad       r = 0.3 + 0.1 ((i mod 3) - 1)     ledger nll = L
      mism      r = -0.2                          ledger nll = L + 0.01 (MISMATCH)
      other     r = 0, another row set (row 50: theta_1 = 0.0005)
      pickled   r = 0.1, the donors as an object array (pickled)
      running   a run folder without a per-row file, no ledger record
      corrupt   theta = 1.0 in one row (outside the open box): SKIPPED
    campaign 2 (--campaign-dir, a set-aside run)
      old       r = -1.0                          ledger nll = L
      trunc     a per-row file that is not a zip: SKIPPED

Checks
    test_known_answer   T and its table by a scalar transcription, every
                        trial's L and R, the two row sets, the ledger
                        verdicts, the skipped and missing files, the shares
                        of edge rows, the pass line, the physical edge zone
                        from a hand-made bank sidecar
    test_reference      tau against a scalar math transcription; symmetric
                        in theta and 1 - theta; 0 inside the bound; with
                        torch, sbi and zuko importable, the tool's T_i
                        against the real flow's -log q (both coordinates
                        past the bound; skipped otherwise)
    test_convergence    not a discretisation: skipped
    test_invariants     the CLI writes nothing: every file's bytes and
                        mtime unchanged, no file added
    test_contract       --trial keeps one trial; exit 2 for a configuration
                        that does not load, 3 when no per-row file is read
    test_edge_cases     load_perrow's refusals (a missing key, shapes, a
                        non-finite nll, theta at 0, no rows); ledger_check
                        without a record or an nll
    MUTATION=sign       the ln(theta (1 - theta)) term negated   } each must make
    MUTATION=bound      the edge mask |u| >= 4 instead of > 5   } test_known_answer
    MUTATION=crosscheck the ledger comparison always equal      } FAIL

Run
    python3 hpc/pipeline/tests/test_smoke_perrow.py
        A few seconds without torch; numpy needed (sbi_export has it).
    MUTATION=sign python3 hpc/pipeline/tests/test_smoke_perrow.py
        must report test_known_answer FAIL (likewise bound, crosscheck).

Pure ASCII, LF only; Python 3.6 syntax.
"""

import hashlib
import json
import math
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
_TOOLS = os.path.join(_PIPE, "tools")
for _p in (_PIPE, _TOOLS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C              # noqa: E402
import perrow_check as PC       # noqa: E402

GIULIA = os.path.join(_PIPE, "projects", "giulia", "giulia.json")
TOOL = os.path.join(_TOOLS, "perrow_check.py")
N, D, DONORS = 200, 3, 20

MUTATION = os.environ.get("MUTATION", "")
_REAL_EDGE_TERMS = PC.edge_terms
if MUTATION == "sign":
    def _sign(theta, bound):
        tau, u, mask = _REAL_EDGE_TERMS(theta, bound)
        wrong = np.where(mask, 0.5 * u * u + PC.HALF_LN_2PI - np.log(theta) - np.log1p(-theta), 0.0)
        return wrong, u, mask
    PC.edge_terms = _sign
elif MUTATION == "bound":
    def _bound(theta, bound):
        u = PC.logit(theta)
        mask = np.abs(u) >= 4.0
        tau = np.where(mask, 0.5 * u * u + PC.HALF_LN_2PI + np.log(theta) + np.log1p(-theta), 0.0)
        return tau, u, mask
    PC.edge_terms = _bound
elif MUTATION == "crosscheck":
    PC.ledger_check = lambda L, rec, tol=PC.LEDGER_TOL: ("equal", "ledger (not compared)")
elif MUTATION:
    raise SystemExit("unknown MUTATION %r; known: sign, bound, crosscheck" % MUTATION)


# --------------------------------------------------------------- oracle ---

def tau_scalar(t):
    """The edge term of one coordinate, by hand, in float64 (0 inside the bound)."""
    t = float(t)
    u = math.log(t) - math.log(1.0 - t)
    if abs(u) <= 5.0:
        return 0.0
    return 0.5 * u * u + 0.5 * math.log(2.0 * math.pi) + math.log(t) + math.log(1.0 - t)


def tau_rows(theta32):
    return np.array([sum(tau_scalar(v) for v in row) for row in theta32.astype(np.float64)])


# -------------------------------------------------------------- fixture ---

def _tid(name):
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def _project(root):
    """The committed Giulia file with every path under `root`."""
    with open(GIULIA, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    for k in list(cfg["paths"]):
        cfg["paths"][k] = os.path.join(root, k)
    cfg["real"]["cohort_config"] = os.path.join(root, "cohort.json")
    cfg["real"]["extract_root"] = os.path.join(root, "extracted")
    for k in ("sim_main", "out_root", "tools"):
        cfg["sim"][k] = os.path.join(root, k)
    path = os.path.join(root, "giulia.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    return path


def theta_main():
    th = np.random.RandomState(7).uniform(0.2, 0.8, size=(N, D))
    th[0:10, 0] = 0.003
    th[10:15, 2] = 0.999
    th[15:17, 1] = 1e-4
    th[15:17, 0] = 0.006
    th[17, 1] = 0.012
    th[18, 2] = 0.9925
    return th.astype(np.float32)


def theta_other():
    th = theta_main()
    th[50, 1] = np.float32(0.0005)
    return th


def donors():
    return np.array(["donor_%02d" % (i // (N // DONORS)) for i in range(N)])


R_OF = {
    "good": np.full(N, -0.5),
    "bad": 0.3 + 0.1 * ((np.arange(N) % 3) - 1.0),
    "mism": np.full(N, -0.2),
    "other": np.zeros(N),
    "pickled": np.full(N, 0.1),
    "old": np.full(N, -1.0),
}


def _write_trial(camp, name, theta32, group, r, ledger_shift=0.0, object_group=False):
    tid = _tid(name)
    run = os.path.join(camp, "runs", tid)
    os.makedirs(run)
    nll = tau_rows(theta32) + r
    g = np.array(group.tolist(), dtype=object) if object_group else group
    np.savez_compressed(os.path.join(run, "A1_seed0_perrow.npz"), nll=nll, group=g, theta=theta32)
    with open(os.path.join(camp, "trials", tid + ".json"), "w") as fh:
        json.dump({"trial_id": tid, "status": "ok", "nll": float(np.mean(nll)) + ledger_shift}, fh)
    return tid


def build(root):
    """The folders of the module docstring; returns (config path, ids, campaign 1, campaign 2)."""
    path = _project(root)
    cfg = C.load_config(path)
    camp1 = os.path.join(cfg.data["paths"]["tune_root"], cfg.data["search"]["campaign"])
    camp2 = os.path.join(cfg.data["paths"]["tune_root"], "S-A1_set_aside_2026-10-19T161735Z")
    for c in (camp1, camp2):
        os.makedirs(os.path.join(c, "trials"))
        os.makedirs(os.path.join(c, "runs"))
    th, g = theta_main(), donors()
    ids = {}
    ids["good"] = _write_trial(camp1, "good", th, g, R_OF["good"])
    ids["bad"] = _write_trial(camp1, "bad", th, g, R_OF["bad"])
    ids["mism"] = _write_trial(camp1, "mism", th, g, R_OF["mism"], ledger_shift=0.01)
    ids["other"] = _write_trial(camp1, "other", theta_other(), g, R_OF["other"])
    ids["pickled"] = _write_trial(camp1, "pickled", th, g, R_OF["pickled"], object_group=True)
    ids["running"] = _tid("running")
    os.makedirs(os.path.join(camp1, "runs", ids["running"]))
    with open(os.path.join(camp1, "runs", ids["running"], "A1_seed0_train_ckpt.pt"), "wb") as fh:
        fh.write(b"not a real checkpoint")
    ids["corrupt"] = _tid("corrupt")
    bad_th = th.copy()
    bad_th[3, 1] = 1.0
    os.makedirs(os.path.join(camp1, "runs", ids["corrupt"]))
    np.savez_compressed(os.path.join(camp1, "runs", ids["corrupt"], "A1_seed0_perrow.npz"),
                        nll=np.zeros(N), group=g, theta=bad_th)
    ids["old"] = _write_trial(camp2, "old", th, g, R_OF["old"])
    ids["trunc"] = _tid("trunc")
    os.makedirs(os.path.join(camp2, "runs", ids["trunc"]))
    with open(os.path.join(camp2, "runs", ids["trunc"], "A1_seed0_perrow.npz"), "wb") as fh:
        fh.write(b"PK\x03\x04 not a zip file at all")
    bank = os.path.join(cfg.data["paths"]["banks_root"], cfg.tag + "_sim")
    os.makedirs(bank)
    with open(os.path.join(bank, "shard_0000.json"), "w") as fh:
        json.dump({"param_names": ["O_N", "g_syn", "v_rest"],
                   "bounds_theta": [[math.log(0.01), math.log(3.0)], [0.0, 2.0], [-70.0, -50.0]],
                   "coord": "unit_box",
                   "latent_spec": {"coord_physical": ["ln", "linear", "linear"],
                                   "units": ["1", "nS", "mV"]}}, fh)
    return path, ids, camp1, camp2


def run_cli(args, timeout=120):
    p = subprocess.run([sys.executable, TOOL] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def snapshot(root):
    out = {}
    for d, _subs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                out[p] = (hashlib.sha256(fh.read()).hexdigest(), os.stat(p).st_mtime_ns)
    return out


# --------------------------------------------------------------- checks ---

def test_known_answer():
    root = tempfile.mkdtemp(prefix="perrow_ka_")
    try:
        path, ids, camp1, camp2 = build(root)
        cfg = C.load_config(path)
        res = PC.analyse(cfg, [os.path.abspath(camp1), os.path.abspath(camp2)], [], 5.0)
        by = dict((t["trial"], t) for t in res["trials"])
        # the rows, the edge term by hand
        th = theta_main()
        t_rows = tau_rows(th)
        T_main = float(np.mean(t_rows))
        T_other = float(np.mean(tau_rows(theta_other())))
        assert len(res["sets"]) == 2, sorted(res["sets"])
        main = max(res["sets"].values(), key=lambda s: len(s["trials"]))
        other = min(res["sets"].values(), key=lambda s: len(s["trials"]))
        assert sorted(main["trials"]) == sorted(ids[k] for k in ("good", "bad", "mism", "pickled", "old")), \
            main["trials"]
        assert other["trials"] == [ids["other"]], other["trials"]
        # float64 sums in another order: 1e-10 nats is far above roundoff, far below any tau
        assert abs(main["T"] - T_main) < 1e-10, (main["T"], T_main)
        assert abs(other["T"] - T_other) < 1e-10, (other["T"], T_other)
        assert abs(T_other - T_main - tau_scalar(np.float32(0.0005)) / N) < 1e-10
        assert abs(main["frac_edge_rows"] - 17.0 / N) < 1e-15, main["frac_edge_rows"]
        assert main["n"] == N and main["d"] == D and main["donors"] == DONORS, main
        tab = dict((r["k"], r) for r in main["table"])
        assert (tab[0]["low"], tab[0]["high"]) == (12.0 / N, 0.0), tab[0]
        assert (tab[1]["low"], tab[1]["high"]) == (2.0 / N, 0.0), tab[1]
        assert (tab[2]["low"], tab[2]["high"]) == (0.0, 5.0 / N), tab[2]
        t32 = th.astype(np.float64)
        want_k = [sum(tau_scalar(v) for v in t32[:, k]) / N for k in range(D)]
        for k in range(D):
            assert abs(tab[k]["T_k"] - want_k[k]) < 1e-10, (k, tab[k]["T_k"], want_k[k])
        # every trial: L the per-row mean, R the planted rest
        for name in ("good", "bad", "mism", "pickled", "old", "other"):
            t = by[ids[name]]
            rr = float(np.mean(R_OF[name]))
            assert abs(t["R"] - rr) < 1e-10, (name, t["R"], rr)
            assert abs(t["L"] - t["T"] - t["R"]) < 1e-12, name
        assert abs(by[ids["good"]]["L_no_edge"] - (-0.5)) < 1e-12, by[ids["good"]]["L_no_edge"]
        # the ledger verdicts
        verdicts = dict((k, by[ids[k]]["ledger"]) for k in ("good", "bad", "mism", "pickled", "old", "other"))
        assert verdicts == {"good": "equal", "bad": "equal", "mism": "MISMATCH", "pickled": "equal",
                            "old": "equal", "other": "equal"}, verdicts
        # what was skipped and what had no file
        sk = dict((k["trial"], k["why"]) for k in res["skipped"])
        assert sorted(sk) == sorted([ids["corrupt"], ids["trunc"]]), sk
        assert "outside the open unit box" in sk[ids["corrupt"]], sk[ids["corrupt"]]
        assert "cannot be read" in sk[ids["trunc"]], sk[ids["trunc"]]
        assert [w["trial"] for w in res["without"]] == [ids["running"]], res["without"]
        # the names and the physical edge zone from the sidecar
        assert res["names"] == ["O_N", "g_syn", "v_rest"], res["names"]
        s = 1.0 / (1.0 + math.exp(5.0))
        lo, hi, unit = PC.physical_edges(res["phys"][0], 5.0)
        span = math.log(3.0) - math.log(0.01)
        assert abs(lo - math.exp(math.log(0.01) + s * span)) < 1e-12 and unit == "1", (lo, unit)
        assert abs(hi - math.exp(math.log(0.01) + (1 - s) * span)) < 1e-12, hi
        lo2, hi2, _u = PC.physical_edges(res["phys"][2], 5.0)
        assert abs(lo2 - (-70.0 + 20.0 * s)) < 1e-12 and abs(hi2 - (-70.0 + 20.0 * (1 - s))) < 1e-12
        # the CLI's report and pass line
        rc, out = run_cli(["--config", path, "--campaign-dir", camp2])
        assert rc == 0, out[-3000:]
        want = ("PERROW CHECK OK: 6 trial(s) read, 2 skipped, 1 without a per-row file; 2 row set(s); "
                "T = %.4f, %.4f nats/row (one per row set); nothing was written" % (T_main, T_other))
        assert want in out, (want, out[-3000:])
        assert "MISMATCH" in out and "SKIPPED" in out, out[-3000:]
        assert "below 0.01039, above 2.888 1" in out, out[-3000:]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_reference():
    # by hand: theta = 0.003 -> u = ln(0.003/0.997)
    u = math.log(0.003 / 0.997)
    want = 0.5 * u * u + 0.5 * math.log(2 * math.pi) + math.log(0.003 * 0.997)
    assert abs(want - 11.962413) < 1e-6, want
    pts = np.array([[0.003, 0.997, 1e-4, 0.5, 0.01, 0.9925, 0.006, 0.994]])
    tau, u_, mask = PC.edge_terms(pts, 5.0)
    exp = [tau_scalar(v) for v in pts[0]]
    assert np.allclose(tau[0], exp, rtol=0, atol=1e-12), (tau[0], exp)
    assert abs(tau[0, 0] - tau[0, 1]) < 1e-9, "symmetric in theta and 1 - theta"
    assert tau[0, 3] == 0.0 and tau[0, 4] == 0.0 and tau[0, 5] == 0.0, "inside the bound: 0"
    assert mask[0].tolist() == [True, True, True, False, False, False, True, True], mask
    try:
        import torch
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform
    except Exception as exc:  # noqa: BLE001 - any import failure means: not here
        raise unittest.SkipTest("the real-flow oracle needs torch, sbi and zuko (%s)" % type(exc).__name__)
    torch.manual_seed(0)
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    build = posterior_nn(model="zuko_nsf", z_score_theta="transform_to_unconstrained", z_score_x="none",
                         hidden_features=20, num_transforms=3, num_bins=10, x_dist=prior)
    est = build(prior.sample((64,)), torch.randn(64, 2))
    with torch.no_grad():
        for p in est.parameters():
            p.add_(0.3 * torch.randn(p.shape))
        th = torch.tensor([[0.003, 0.999], [5e-4, 0.006], [1e-4, 0.995]], dtype=torch.float32)
        got = est.loss(th, condition=torch.randn(3, 2)).double().numpy()
    tau_f, _u, _m = PC.edge_terms(th.double().numpy(), 5.0)
    # the flow computes in float32: 1e-4 nats covers its rounding at these points
    assert np.allclose(got, tau_f.sum(axis=1), rtol=0, atol=1e-4), (got, tau_f.sum(axis=1))


def test_convergence():
    raise unittest.SkipTest("not a discretisation: nothing converges")


def test_invariants():
    root = tempfile.mkdtemp(prefix="perrow_inv_")
    try:
        path, ids, camp1, camp2 = build(root)
        before = snapshot(root)
        rc, out = run_cli(["--config", path, "--campaign-dir", camp2])
        assert rc == 0, out[-2000:]
        after = snapshot(root)
        assert sorted(after) == sorted(before), "files added or removed: %s" % (
            sorted(set(after) ^ set(before)),)
        changed = [p for p in before if before[p] != after[p]]
        assert not changed, "files changed: %s" % changed
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_contract():
    root = tempfile.mkdtemp(prefix="perrow_ct_")
    try:
        path, ids, camp1, camp2 = build(root)
        rc, out = run_cli(["--config", path, "--trial", ids["good"][:10]])
        assert rc == 0, out[-2000:]
        assert ("PERROW CHECK OK: 1 trial(s) read, 0 skipped, 0 without a per-row file; 1 row set(s); "
                in out), out[-2000:]
        rc, out = run_cli(["--config", path, "--trial", "zzzz"])
        assert rc == 3 and "no per-row file was read" in out, (rc, out[-2000:])
        broken = os.path.join(root, "broken.json")
        with open(broken, "w") as fh:
            fh.write('{"config_version": 10}')
        rc, out = run_cli(["--config", broken])
        assert rc == 2 and "does not load" in out, (rc, out[-2000:])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edge_cases():
    root = tempfile.mkdtemp(prefix="perrow_ec_")
    try:
        g = donors()
        th = theta_main()

        def case(name, **arrays):
            p = os.path.join(root, name + ".npz")
            np.savez(p, **arrays)
            return PC.load_perrow(p)

        a, why = case("nokey", nll=np.zeros(N), group=g)
        assert a is None and "no theta" in why, why
        a, why = case("shapes", nll=np.zeros(N), group=g, theta=th[:, 0])
        assert a is None and "shapes" in why, why
        a, why = case("rows", nll=np.zeros(N - 1), group=g, theta=th)
        assert a is None and "row counts" in why, why
        nan = np.zeros(N)
        nan[5] = np.nan
        a, why = case("nan", nll=nan, group=g, theta=th)
        assert a is None and "non-finite nll" in why, why
        zero = th.copy()
        zero[0, 0] = 0.0
        a, why = case("zero", nll=np.zeros(N), group=g, theta=zero)
        assert a is None and "outside the open unit box" in why, why
        a, why = case("empty", nll=np.zeros(0), group=g[:0], theta=th[:0])
        assert a is None and "zero" in why, why
        a, why = case("fine", nll=np.zeros(N), group=g, theta=th)
        assert a is not None and why is None and a["theta"].dtype == np.float64, why
        assert PC.ledger_check(0.5, None)[0] == "none"
        assert PC.ledger_check(0.5, {"status": "failed"})[0] == "none"
        assert PC.ledger_check(0.5, {"status": "ok", "nll": True})[0] == "none"
        assert PC.ledger_check(0.5, {"status": "ok", "nll": 0.5 + 5e-7})[0] == "equal"
        assert PC.ledger_check(0.5, {"status": "ok", "nll": 0.5 + 5e-6})[0] == "MISMATCH"
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- runner ---

def main():
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    print("== %s" % os.path.basename(__file__))
    print("python %s, numpy %s" % (sys.version.split()[0], np.__version__))
    if MUTATION:
        print("MUTATION=%s: the named piece is broken; test_known_answer must FAIL" % MUTATION)
    results = []
    for name, func in checks:
        start = time.time()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.time() - start, detail))
    width = max(len(r[0]) for r in results)
    for name, status, seconds, detail in results:
        head = (detail.strip().splitlines() or [""])[0]
        print("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, head[:200]))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()[:6000]))
    counts = dict((s, sum(1 for r in results if r[1] == s))
                  for s in ("PASS", "FAIL", "ERROR", "SKIP"))
    print("\n-- %d pass, %d fail, %d error, %d skip"
          % (counts["PASS"], counts["FAIL"], counts["ERROR"], counts["SKIP"]))
    return 1 if counts["FAIL"] or counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
