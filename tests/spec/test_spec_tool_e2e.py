"""End to end through the tool's real entry points (specs/SPEC.md sec. 6, the
rows of blocks 40-46), twice with the same seed. Integration tester,
2026-11-28, incremental run on 39e6f66.

export -> bundle-check --rebuild -> card-template -> card-check -> windows
-> infer (--windows and --recording) -> device-figure, each from a clean
folder, the whole chain run twice. Checks: every command exits 0 with its
pass line; every output exists; arrays finite with the contract's shapes and
dtypes; the two chains give identical numbers (the npz members' arrays, the
JSON's bytes but for the time and the bundle's digest, which holds the
creation time). Needs REC_FIXTURE, REC_SPEC_OUT and SBIX_CLONE; reuses the
evaluated project of test_spec_block41_45_bundle.ctx().
"""

import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import _tool_util as U  # noqa: E402
import test_spec_block41_45_bundle as T  # noqa: E402

pytestmark = T.pytestmark


def _chain(root):
    c = T.ctx()
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    lines = {}

    def ok(key, rc_txt, line):
        rc, txt = rc_txt
        assert rc == 0 and line in txt, (key, rc, txt[-1500:])
        lines[key] = [ln for ln in txt.splitlines() if line in ln][-1]

    b = os.path.join(root, "m.sbibundle")
    ok("export", T._tool("export", "--config", c["cfg"], "--eval-config", c["ecfg"],
                         "--ensemble", "S-A2-R0:A2", "--out", b, "--extraction", c["ext"]),
       "BUNDLE EXPORT OK")
    ok("check", T._tool("bundle-check", b, "--rebuild"), "BUNDLE CHECK OK")
    card = os.path.join(root, "CARD.md")
    ok("card-template", T._tool("card-template", "--bundle", b, "--out", card),
       "CARD TEMPLATE OK")
    ok("card-check", T._tool("card-check", card, "--bundle", b), "CARD OK: draft")
    rng = np.random.default_rng(4600)
    n = 40000
    sp = {e: np.sort(rng.choice(n, size=int(rng.integers(40, 160)), replace=False))
          for e in range(36)}
    rec = U.write_ptrain(os.path.join(root, "rec"), sp, n)
    opts = ["--option", "fs_raw_hz=10000.0", "--option", "grid_width=6", "--option",
            "index_base=0", "--option", "pitch_um=60.0"]
    w = os.path.join(root, "w.npz")
    ok("windows", T._tool("windows", "--bundle", b, "--recording", rec, "--adapter", "ptrain",
                          "--out", w, *opts), "WINDOWS OK")
    ok("infer-w", T._tool("infer", "--bundle", b, "--windows", w, "--out",
                          os.path.join(root, "post_w")), "INFER OK")
    ok("infer-r", T._tool("infer", "--bundle", b, "--recording", rec, "--adapter", "ptrain",
                          "--out", os.path.join(root, "post_r"), *opts), "INFER OK")
    ok("figures", T._tool("device-figure", "--bundle", b, "--recording", rec, "--adapter",
                          "ptrain", "--out", os.path.join(root, "fig"), "--subregion", "0",
                          "--window", "1", *opts), "DEVICE FIGURES OK: 2 file(s)")
    return lines


def _arrays(path):
    with np.load(path) as z:
        return dict((k, z[k]) for k in z.files)


def test_tool_chain_twice_identical():
    a, b = os.path.join(T.OUT, "chainA"), os.path.join(T.OUT, "chainB")
    la, lb = _chain(a), _chain(b)
    for k in ("windows", "infer-w", "infer-r", "figures"):
        assert la[k].replace(a, "") == lb[k].replace(b, ""), (la[k], lb[k])
    for f in ("fig/device.png", "fig/window_s0_w1.png", "post_w/posterior.json",
              "post_r/posterior.npz", "w.npz", "CARD.md"):
        assert os.path.isfile(os.path.join(a, f)), f
    wa, wb = _arrays(os.path.join(a, "w.npz")), _arrays(os.path.join(b, "w.npz"))
    for k in ("windows", "subregion", "window_idx", "t_start_s", "fs_ifr_hz"):
        np.testing.assert_array_equal(wa[k], wb[k])
    for post in ("post_w", "post_r"):
        pa = _arrays(os.path.join(a, post, "posterior.npz"))
        pb = _arrays(os.path.join(b, post, "posterior.npz"))
        assert set(pa) == set(pb)
        for k in pa:
            np.testing.assert_array_equal(pa[k], pb[k], err_msg=k)
            if pa[k].dtype.kind == "f":
                assert np.all(np.isfinite(pa[k])), k
        n = pa["draws"].shape[0]
        assert n == wa["windows"].shape[0] and pa["draws"].dtype == np.float32
        ja = json.load(open(os.path.join(a, post, "posterior.json")))
        jb = json.load(open(os.path.join(b, post, "posterior.json")))
        ja["bundle"].pop("sha256")
        jb["bundle"].pop("sha256")
        assert ja == jb
    # the bundles: the same members and evaluation files (the manifest holds the time)
    import zipfile
    za, zb = (zipfile.ZipFile(os.path.join(x, "m.sbibundle")) for x in (a, b))
    for n in za.namelist():
        if n != "manifest.json":
            assert za.read(n) == zb.read(n), n


def test_swap_the_gates_kernel_through_the_configuration():
    """Directive audit, swappability: the gate's bandwidth factors changed in the
    evaluation's configuration alone ((0.5, 1, 2) for D-163's five) go through the
    sampler, the metrics, the export and the inference; nothing else is edited."""
    c = T.ctx()
    root = os.path.join(T.OUT, "swap")
    ecfg = T._ecfg_file(root, lambda d: d["gate"].update(factors=[0.5, 1, 2]))
    if not os.path.isfile(os.path.join(root, "metrics", "METRICS.json")):
        rc, out = T._run([sys.executable, os.path.join(T.EVAL, "ev_rec_sample.py"), "--config",
                          c["cfg"], "--eval-config", ecfg, "--ensemble", "S-A1:A1"])
        assert rc == 0, out[-1500:]
        rc, out = T._run([sys.executable, os.path.join(T.EVAL, "ev_rec_metrics.py"), "--config",
                          c["cfg"], "--eval-config", ecfg, "--ncpus", "1"])
        assert rc == 0, out[-1500:]
    sim = json.load(open(os.path.join(root, "metrics", "S-A1_A1_sim.json")))
    assert sim["gate"]["factors"] == [0.5, 1, 2]
    assert all(len(bw) == 3 for bw in sim["gate"]["bandwidths"])
    b = os.path.join(root, "s.sbibundle")
    if os.path.exists(b):
        os.remove(b)
    rc, out = T._tool("export", "--config", c["cfg"], "--eval-config", ecfg, "--ensemble",
                      "S-A1:A1", "--out", b)
    assert rc == 0, out[-1500:]
    import zipfile
    man = json.loads(zipfile.ZipFile(b).read("manifest.json"))
    assert man["gate"]["factors"] == [0.5, 1, 2]
    v, ec = T._view()
    import ev_rec_data
    x = np.asarray(ev_rec_data.recording(v, "real")["x"], dtype=np.float32)
    prov = json.load(open(os.path.join(root, "samples", "S-A1_A1_sample.json")))
    res = T._infer(b, x, seed=int(prov["seeds"]["real"]))
    met = T._npz(os.path.join(root, "metrics", "S-A1_A1_real.npz"))
    np.testing.assert_allclose(res["arrays"]["gate_p"], met["gate_p"], rtol=0, atol=1e-12)
