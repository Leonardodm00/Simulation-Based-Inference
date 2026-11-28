"""Spec tests of blocks 41 (the model bundle) and 45 (inference from a
bundle), with block 42's template inside the export; specs/SPEC.md sec. 2.7,
blocks 41, 42, 45. Integration tester, 2026-11-28, incremental run on
39e6f66.

Needs REC_FIXTURE (a built rec_fixture tree), REC_SPEC_OUT (a folder of this
run) and SBIX_CLONE. The recorded project of the fixture is sampled (block
35) and evaluated (block 36) by their command lines under REC_SPEC_OUT/tool,
then exported by `python3 -m sbitool export`. Expected values: the
pipeline's own files (block 35's samples, block 36's metrics, the
checkpoints' model_sha256, the bank's donors), read with numpy, json,
hashlib, zipfile and torch; the mixture's summaries by numpy.
"""

import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import _tool_util as U  # noqa: E402

REPO = U.REPO
EVAL = os.path.join(REPO, "hpc", "pipeline", "evaluation")
FIX = os.environ.get("REC_FIXTURE", "").strip()
OUT = os.path.join(os.environ.get("REC_SPEC_OUT", "/tmp/tst/e2e"), "tool")
pytestmark = pytest.mark.skipif(not (FIX and os.path.isdir(FIX)), reason="REC_FIXTURE not set")
ENSEMBLES = ("S-A1:A1", "S-A2-R0:A2")
# 6 x 6 at 60 um. [2026-11-28_0151, D-180 (4)] the export refuses an --extraction whose
# w_size_s is not the simulated bank's sidecar w_size (the fixture's: 0.1 s), so the
# extraction is at 0.1 s bins with sigma_sm 0.2 s (2 bins, as 0.02 / 0.01 before) and a raw
# rate of 1 kHz; the test recordings are 40,000 samples (40 s, K = 400, W = 128: 3 windows
# per subregion, as before at 10 kHz and 0.01 s).
EXT = U.extraction(n_subsets=2, electrodes_per_subset=9, w_size_s=0.1, gaussian_window_s=0.2,
                   fs_raw_hz=1000.0)
FS_OPT = "fs_raw_hz=1000.0"
_C = {}


def _env():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SBIX_CLONE=U.SBIX)


def _run(argv, timeout=1200):
    r = subprocess.run(argv, cwd=REPO, env=_env(), capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def _tool(*args, timeout=1200):
    return _run([sys.executable, "-m", "sbitool"] + list(args), timeout)


def _ecfg_file(root, mutate=None, name="rec_eval.json"):
    d = json.load(open(os.path.join(FIX, "rec_eval.json")))
    d["out_root"] = root
    if mutate:
        mutate(d)
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, name)
    json.dump(d, open(p, "w"), indent=1)
    return p


def _stem(name):
    return name.replace(":", "_")


def ctx():  # noqa: D401
    """The evaluated project and its bundles, made once (kept under OUT)."""
    if _C:
        return _C
    cfg = os.path.join(FIX, "synth_rec.json")
    ecfg = _ecfg_file(OUT)
    done = os.path.join(OUT, "metrics", "METRICS.json")
    if not os.path.isfile(done):
        for name in ENSEMBLES:
            rc, out = _run([sys.executable, os.path.join(EVAL, "ev_rec_sample.py"), "--config",
                            cfg, "--eval-config", ecfg, "--ensemble", name, "--ncpus", "1"])
            assert rc == 0 and "REC SAMPLE OK" in out, out[-2000:]
        rc, out = _run([sys.executable, os.path.join(EVAL, "ev_rec_metrics.py"), "--config", cfg,
                        "--eval-config", ecfg, "--ncpus", "1"])
        assert rc == 0 and "REC METRICS OK" in out, out[-2000:]
    bdir = os.path.join(OUT, "bundles")
    os.makedirs(bdir, exist_ok=True)
    ext = os.path.join(OUT, "extraction.json")
    json.dump(EXT, open(ext, "w"))
    bundles = {}
    for name in ENSEMBLES:
        b = os.path.join(bdir, _stem(name) + ".sbibundle")
        if not os.path.isfile(b):
            rc, out = _tool("export", "--config", cfg, "--eval-config", ecfg, "--ensemble", name,
                            "--out", b, "--extraction", ext)
            assert rc == 0 and "BUNDLE EXPORT OK" in out and "draft" in out, out[-2000:]
        bundles[name] = b
    _C.update(cfg=cfg, ecfg=ecfg, bundles=bundles, ext=ext, bdir=bdir)
    return _C


T_ctx = ctx


def _npz(path):
    with np.load(path) as z:
        return dict((k, z[k]) for k in z.files)


def _zip(path):
    with zipfile.ZipFile(path) as zf:
        return [(i.filename, zf.read(i.filename)) for i in zf.infolist()]


def _rezip(path, entries):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, d in entries:
            zf.writestr(n, d)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _view():
    sys.path.insert(0, EVAL)
    import _evalpath  # noqa: F401
    import ev_rec_config as RC
    import ev_rec_plan as RP
    c = ctx()
    ec = RC.load(c["ecfg"])
    return RP.load_view(c["cfg"], ec), ec


# ===================================================================== block 41

def test_t41b_bundle_content():
    c = ctx()
    v, ec = _view()
    for name in ENSEMBLES:
        stem = _stem(name)
        ents = dict(_zip(c["bundles"][name]))
        man = json.loads(ents["manifest.json"])
        M = man["M"]
        want = ({"manifest.json", "CARD.md", "evaluation/sim.json",
                 "evaluation/gate.safetensors"} |
                {"members/member_%d.safetensors" % k for k in range(M)})
        assert set(ents) == want
        from sbitool import schema
        assert schema.validate_manifest(man) == []
        assert set(man) == set(schema.MANIFEST_KEYS)
        prov = json.load(open(os.path.join(OUT, "samples", stem + "_sample.json")))
        simj = open(os.path.join(OUT, "metrics", stem + "_sim.json"), "rb").read()
        sim = json.loads(simj)
        # evaluation/sim.json is the metrics' file byte for byte
        assert ents["evaluation/sim.json"] == simj
        # the parameters: the sample's axes; units "1" (the fixture's unit box)
        assert man["parameters"]["names"] == prov["axes"]["names"]
        assert man["parameters"]["bounds"] == prov["axes"]["bounds"]
        assert man["parameters"]["units"] == ["1"] * man["p"]
        # the gate: the metrics' gate
        for k in ("factors", "alpha", "n_ref", "n_rank", "n_null", "k_gate", "size_mean",
                  "size_lo", "size_hi", "level"):
            assert man["gate"][k] == sim["gate"][k], k
        # the sample's defaults: the evaluation's n_post, seed, batch_rows (D-179)
        assert man["sample"] == {"n_post": ec.n_post, "seed": ec.seed,
                                 "batch_rows": ec.batch_rows}
        # the members: the records, in seed order; state_sha256 the checkpoint's model_sha256
        e = [x for x in v.ensembles if x.name == name][0]
        import torch
        seeds = []
        for k, rp in enumerate(e.records):
            rec = json.load(open(rp))
            ck = torch.load(rp[:-5] + "_ckpt.pt", map_location="cpu", weights_only=False)
            m = man["members"][k]
            assert m["seed"] == rec["seed"] and m["split_hash"] == rec["split_hash"]
            assert m["split_seed"] == rec["split_seed"]
            assert m["state_sha256"] == ck["model_sha256"]
            assert m["sha256"] == _sha(ents[m["file"]])
            seeds.append(m["seed"])
        assert seeds == sorted(seeds) and len(set(seeds)) == M
        # every other member's digest
        for n in ("CARD.md", "evaluation/sim.json", "evaluation/gate.safetensors"):
            assert man["files"][n] == _sha(ents[n]), n
        # the gate's sets: z at the roles 1, 2, 3 of gate_role, in row order (numpy here)
        from safetensors.numpy import load as st_load
        g = st_load(ents["evaluation/gate.safetensors"])
        s = _npz(os.path.join(OUT, "samples", stem + "_sim.npz"))
        pos = {int(r): i for i, r in enumerate(s["z_idx"])}
        for role, key in ((1, "z_ref"), (2, "z_rank"), (3, "z_null")):
            rows = np.flatnonzero(s["gate_role"] == role)
            assert_array_equal(g[key], s["z"][:, [pos[int(r)] for r in rows], :])
            assert g[key].dtype == np.float32
        assert man["d_z"] == s["z"].shape[2] and man["M"] == s["z"].shape[0]
        assert man["provenance"]["sample_json_sha256"] == _sha(
            open(os.path.join(OUT, "samples", stem + "_sample.json"), "rb").read())
        assert man["provenance"]["best_fit_complete_sha256"] == _sha(
            open(os.path.join(e.folder, "BEST_FIT_COMPLETE.json"), "rb").read())
        assert man["extraction"] == EXT and man["card_complete"] is False


def test_t41a_rebuild_is_the_pipelines():
    c = ctx()
    v, ec = _view()
    import torch
    import ev_models
    import ev_rec_data
    import ev_sample
    sys.path.insert(0, os.path.join(REPO, "hpc", "joint", "stage3"))
    import run_joint_arms as R
    from sbitool import bundle, model
    index, _ = ev_rec_data.sim_index(v)
    s = _npz(os.path.join(OUT, "samples", "S-A1_A1_sim.npz"))
    rows = s["rows"][:16]
    xr = R._rows_x(index, rows)              # the bank's windows of the rows (the runner's)
    x = np.asarray(xr.numpy() if hasattr(xr, "numpy") else xr, dtype=np.float32)
    theta = np.asarray(index.theta)[rows].astype(np.float64)
    inb = np.all((theta > 0) & (theta < 1), axis=1)
    for name in ENSEMBLES:
        e = [y for y in v.ensembles if y.name == name][0]
        mods = model.load(bundle.read(c["bundles"][name]))
        for k, rp in enumerate(e.records):
            ref, _info = ev_models.rebuild(rp, index)
            a, b = ref.state_dict(), mods[k].state_dict()
            assert list(a) == list(b)
            for kk in a:
                assert torch.equal(a[kk], b[kk]), kk
            # log q, 8 draws per row under one seed, the embeddings: bit for bit
            assert_array_equal(ev_sample.logq_at(mods[k], theta, x, inb, 16),
                               ev_sample.logq_at(ref, theta, x, inb, 16))
            assert_array_equal(ev_sample.draws_of(mods[k], x, 8, 123 + k, 16),
                               ev_sample.draws_of(ref, x, 8, 123 + k, 16))
            assert_array_equal(ev_sample.embed(mods[k], x, 16), ev_sample.embed(ref, x, 16))


def _check(path):
    return _tool("bundle-check", path)


def test_t41c_integrity():
    c = ctx()
    src = c["bundles"]["S-A1:A1"]
    d = os.path.join(OUT, "tamper")
    os.makedirs(d, exist_ok=True)
    rc, out = _check(src)
    assert rc == 0 and "BUNDLE CHECK OK" in out, out
    rc, out = _tool("bundle-check", src, "--rebuild")
    assert rc == 0 and "BUNDLE CHECK OK" in out, out
    ents = _zip(src)

    def tampered(fname, member, edit):
        p = os.path.join(d, fname)
        _rezip(p, [(n, edit(b) if n == member else b) for n, b in ents])
        return p

    def flip(b, at=None):
        at = len(b) // 2 if at is None else at
        return b[:at] + bytes([b[at] ^ 1]) + b[at + 1:]

    for member in ("members/member_0.safetensors", "evaluation/gate.safetensors",
                   "CARD.md", "evaluation/sim.json"):
        p = tampered("t_%s.sbibundle" % member.replace("/", "_"), member, flip)
        rc, out = _check(p)
        assert rc == 3 and member in out, (member, out)
    # a zip with an extra member, and one with a .npz member
    p = os.path.join(d, "extra.sbibundle")
    _rezip(p, ents + [("notes.txt", b"x")])
    rc, out = _check(p)
    assert rc == 3 and "notes.txt" in out
    p = os.path.join(d, "npz.sbibundle")
    _rezip(p, ents + [("evaluation/x.npz", b"x")])
    rc, out = _check(p)
    assert rc == 3 and "x.npz" in out


def _redigest(entries, edits):
    """The bundle's members with `edits` (name -> function of the bytes) applied, and the
    manifest's `files` digests of the fixed members recomputed by hashlib, so that a
    member's own digest holds and only the manifest's agreement with it is in question.
    A manifest edit is a function of the parsed manifest (key "manifest")."""
    d = dict(entries)
    for n, f in edits.items():
        if n != "manifest":
            d[n] = f(d[n])
    m = json.loads(d["manifest.json"])
    for n in m["files"]:
        m["files"][n] = _sha(d[n])
    if "manifest" in edits:
        edits["manifest"](m)
    d["manifest.json"] = json.dumps(m, indent=1, sort_keys=True).encode("ascii") + b"\n"
    return [(n, d[n]) for n, _b in entries]


def test_t41c_manifest_byte_changed():
    """T41c as amended (fix round of 2026-11-28, D-180): a manifest edited into another
    valid one fails bundle-check (exit 3, naming what disagrees) where it breaks its agreement
    with the members it describes: the gate's alpha (sim.json's gate), a parameter's name
    (sim.json's axes, where it holds axes) and card_complete true on a draft (the card)."""
    c = T_ctx()
    ents = _zip(c["bundles"]["S-A1:A1"])
    d = os.path.join(OUT, "tamper")
    os.makedirs(d, exist_ok=True)
    man = json.loads(dict(ents)["manifest.json"])
    sim = json.loads(dict(ents)["evaluation/sim.json"])
    assert man["card_complete"] is False                 # the fixture's bundles are drafts
    edits = {
        "gate.alpha": (lambda m: m["gate"].__setitem__(
            "alpha", 0.6 if m["gate"]["alpha"] != 0.6 else 0.5), "alpha"),
        "parameters.names[0]": (lambda m: m["parameters"]["names"].__setitem__(
            0, m["parameters"]["names"][0] + "x"), "names"),
        "card_complete": (lambda m: m.__setitem__("card_complete", True), "card_complete"),
    }
    if "axes" not in sim:
        edits.pop("parameters.names[0]")
    passed = []
    for what, (f, word) in edits.items():
        m = json.loads(json.dumps(man))
        f(m)
        text = json.dumps(m, indent=1, sort_keys=True).encode("ascii") + b"\n"
        p = os.path.join(d, "man_%s.sbibundle" % what.replace(".", "_").replace("[", "_")
                         .replace("]", ""))
        _rezip(p, [(n, text if n == "manifest.json" else b) for n, b in ents])
        rc, out = _check(p)
        if rc != 3 or word not in out or "BUNDLE CHECK OK" in out:
            passed.append((what, rc, out[-300:]))
    assert not passed, ("bundle-check does not refuse, naming the key, a bundle whose "
                        "manifest was changed at: %s" % passed)
    assert "axes" in sim, "the fixture's sim.json holds no axes: parameters are compared " \
        "with nothing"


def test_t41c_agreement_each_member():
    """Each comparison of the fix round's bundle.check, one at a time, with every member's
    own digest holding (_redigest): the gate's other keys; sim.json without gate; the
    bounds; d_z against the gate's sets; card_complete false on a complete card; a card
    that does not parse with card_complete true. And a valid, consistent manifest edit (the
    bundle's name) passes, as the amended T41c states."""
    from sbitool import schema as S
    c = T_ctx()
    ents = _zip(c["bundles"]["S-A1:A1"])
    d = os.path.join(OUT, "tamper2")
    os.makedirs(d, exist_ok=True)
    man = json.loads(dict(ents)["manifest.json"])
    sim = json.loads(dict(ents)["evaluation/sim.json"])

    def sim_edit(f):
        def g(b):
            o = json.loads(b)
            f(o)
            return json.dumps(o).encode("ascii")
        return g

    cases = {}
    for k in S.GATE_KEYS:
        v = man["gate"][k]
        nv = ([x * 2 for x in v] if isinstance(v, list) else
              (v + 1 if isinstance(v, int) and not isinstance(v, bool) else v * 0.5))
        cases["manifest gate.%s" % k] = ({"manifest": (lambda m, k=k, nv=nv:
                                                       m["gate"].__setitem__(k, nv))}, k)
    cases["sim.json gate.k_gate"] = ({"evaluation/sim.json": sim_edit(
        lambda o: o["gate"].__setitem__("k_gate", o["gate"]["k_gate"] + 1))}, "k_gate")
    cases["sim.json without gate"] = ({"evaluation/sim.json": sim_edit(
        lambda o: o.pop("gate"))}, "gate")
    if "axes" in sim:
        cases["sim.json axes.bounds"] = ({"evaluation/sim.json": sim_edit(
            lambda o: o["axes"].__setitem__("bounds", [[0.0, 2.0]] * len(o["axes"]["names"])))},
            "bounds")
    cases["manifest d_z"] = ({"manifest": lambda m: m.__setitem__("d_z", m["d_z"] + 1)}, "d_z")
    cases["unparsable card, card_complete true"] = (
        {"CARD.md": lambda b: b"no header at all\n",
         "manifest": lambda m: m.__setitem__("card_complete", True)}, "card_complete")
    bad = []
    for what, (edits, word) in cases.items():
        p = os.path.join(d, "a_%d.sbibundle" % len(bad + [0]) + what.replace(" ", "_")[:20])
        _rezip(p, _redigest(ents, dict(edits)))
        rc, out = _check(p)
        if rc != 3 or word not in out:
            bad.append((what, rc, out[-400:]))
    # a valid, consistent edit (another name, digests re-made): it passes, and the pass line
    # ends with the file's sha256 in full (hashlib), which differs from the original's
    p = os.path.join(d, "renamed.sbibundle")
    _rezip(p, _redigest(ents, {"manifest": lambda m: m.__setitem__("name", "another")}))
    rc, out = _check(p)
    line = [ln for ln in out.splitlines() if "BUNDLE CHECK OK" in ln]
    if not (rc == 0 and line and line[-1].rstrip().endswith(
            "sha256 " + _sha(open(p, "rb").read()))):
        bad.append(("a consistent rename passes with the sha256", rc, out[-400:]))
    assert _sha(open(p, "rb").read()) != _sha(open(c["bundles"]["S-A1:A1"], "rb").read())
    assert not bad, bad


def test_t41d_extraction_against_the_banks_sidecar():
    """Fix round of 2026-11-28, D-180 (4): the export refuses (exit 3, nothing written) an
    --extraction whose w_size_s is not the simulated bank's sidecar w_size (math.isclose,
    rel_tol 1e-12). A relative difference of 1e-13 is accepted, one of 1e-11 refused."""
    c = T_ctx()
    d = os.path.join(OUT, "refuse_ext")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    side = json.load(open(sorted(glob.glob(os.path.join(FIX, "banks_root", "synth_sim",
                                                        "shard_*.json")))[0]))
    w = float(side["w_size"])
    for rel, want_rc in ((0.0, 0), (1e-13, 0), (1e-11, 3), (-1e-11, 3), (1.0, 3)):
        e = dict(EXT, w_size_s=w * (1.0 + rel))
        ep = os.path.join(d, "ext.json")
        json.dump(e, open(ep, "w"))
        out = os.path.join(d, "x.sbibundle")
        if os.path.exists(out):
            os.remove(out)
        rc, txt = _tool("export", "--config", c["cfg"], "--eval-config", c["ecfg"],
                        "--ensemble", "S-A1:A1", "--out", out, "--extraction", ep)
        assert rc == want_rc, (rel, rc, txt[-600:])
        if want_rc == 3:
            assert "w_size" in txt, txt[-600:]
            left = [f for f in os.listdir(d) if f != "ext.json"]
            assert left == [], (rel, left)


def test_d180_extraction_vs_sidecar_hand_written():
    """The comparison on a hand-written sidecar (the fixture's holds neither
    gaussian_window nor electrodes_per_subset): each compared where the sidecar holds it at
    its top level; a sidecar without w_size is a refusal."""
    from sbitool import export as X
    e = dict(EXT)
    X._extraction_vs_bank(e, {"w_size": 0.1}, "f")
    X._extraction_vs_bank(e, {"w_size": 0.1, "gaussian_window": 0.2,
                              "electrodes_per_subset": 9}, "f")
    for side, word in (({}, "w_size"), ({"w_size": 0.1, "gaussian_window": 0.02},
                                        "gaussian_window"),
                       ({"w_size": 0.1, "electrodes_per_subset": 4}, "electrodes_per_subset")):
        with pytest.raises(X.ExportRefused) as ei:
            X._extraction_vs_bank(e, side, "f")
        assert word in str(ei.value)


def test_t41d_refusals_write_nothing():
    c = ctx()
    d = os.path.join(OUT, "refuse")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    out = os.path.join(d, "x.sbibundle")
    cases = {"an unknown ensemble": ([], "S-A9:A1"),
             "no sample": (["no_sample"], "S-A1:A1"),
             "a sample of other values": (["n_post"], "S-A1:A1"),
             "an invalid extraction": (["bad_ext"], "S-A1:A1"),
             "units of p - 1": (["units"], "S-A1:A1")}
    for what, (flags, name) in cases.items():
        ecfg = c["ecfg"]
        extra = ["--extraction", c["ext"]]
        if "no_sample" in flags:
            ecfg = _ecfg_file(os.path.join(d, "empty_root"))
        if "n_post" in flags:
            ecfg = _ecfg_file(OUT, lambda e: e.update(n_post=20), name="rec_eval_np20.json")
        if "bad_ext" in flags:
            bad = dict(EXT, w_size_s=-1)
            json.dump(bad, open(os.path.join(d, "bad.json"), "w"))
            extra = ["--extraction", os.path.join(d, "bad.json")]
        if "units" in flags:
            json.dump(["Hz"] * 9, open(os.path.join(d, "u.json"), "w"))
            extra += ["--units", os.path.join(d, "u.json")]
        rc, txt = _tool("export", "--config", c["cfg"], "--eval-config", ecfg, "--ensemble", name,
                        "--out", out, *extra)
        assert rc == 3, (what, rc, txt[-800:])
        if "units" in flags:
            assert "--units" in txt
        left = [f for f in os.listdir(d) if f.endswith(".sbibundle") or ".tmp" in f
                or ".part" in f]
        assert left == [], (what, left)
    # the same export twice: members with the same digests (and the gate's file)
    b1 = c["bundles"]["S-A1:A1"]
    b2 = os.path.join(d, "again.sbibundle")
    rc, txt = _tool("export", "--config", c["cfg"], "--eval-config", c["ecfg"], "--ensemble",
                    "S-A1:A1", "--out", b2, "--extraction", c["ext"])
    assert rc == 0, txt
    m1, m2 = (json.loads(dict(_zip(b))["manifest.json"]) for b in (b1, b2))
    assert [m["sha256"] for m in m1["members"]] == [m["sha256"] for m in m2["members"]]
    # the gate's file and sim.json too (the card's body names the bundle, which differs here)
    for k in ("evaluation/sim.json", "evaluation/gate.safetensors"):
        assert m1["files"][k] == m2["files"][k], k
    # --units with p strings replaces the units
    json.dump(["u%d" % k for k in range(10)], open(os.path.join(d, "u10.json"), "w"))
    b3 = os.path.join(d, "units.sbibundle")
    rc, txt = _tool("export", "--config", c["cfg"], "--eval-config", c["ecfg"], "--ensemble",
                    "S-A1:A1", "--out", b3, "--units", os.path.join(d, "u10.json"))
    assert rc == 0, txt
    m3 = json.loads(dict(_zip(b3))["manifest.json"])
    assert m3["parameters"]["units"] == ["u%d" % k for k in range(10)]
    assert m3["extraction"] is None


def test_t41e_card_and_template():
    c = ctx()
    v, ec = _view()
    import ev_rec_data
    from sbitool import card as C
    ents = dict(_zip(c["bundles"]["S-A1:A1"]))
    man = json.loads(ents["manifest.json"])
    h, body = C.parse(ents["CARD.md"].decode("utf-8"))
    e = [y for y in v.ensembles if y.name == "S-A1:A1"][0]
    rec = json.load(open(e.records[0]))
    index, _ = ev_rec_data.sim_index(v)
    donors, counts = np.unique(np.asarray(index.donor), return_counts=True)
    assert np.all(counts == counts[0])              # the fixture: one size per network
    nn = h["training"]["n_networks"]
    assert isinstance(nn, int) and not isinstance(nn, bool)
    assert nn * int(counts[0]) == h["training"]["n_windows"] == rec["n_train"]
    assert h["training"]["arm"] == man["arm"] and h["training"]["members"] == man["M"]
    # a complete card given to the export: card_complete true; one TODO: a draft
    h.update(title="T", scientific_question="Q", references=["R"], licence="MIT", cite="C",
             limitations="L", authors=[{"name": "N", "contact": "c"}],
             data={"simulated_bank": {"location": "l", "public": False, "size_bytes": None,
                                      "sha256": None},
                   "recordings": {"location": "l", "public": False}})
    h["simulator"].update(model_type="m", reference="r", fixed_parameters={"rho": "1"})
    h["training"]["hyperparameter_search"] = "GP"
    d = os.path.join(OUT, "cards")
    os.makedirs(d, exist_ok=True)
    full = os.path.join(d, "CARD.md")
    open(full, "w").write(C.dump(h, body))
    h2 = json.loads(json.dumps(h))
    h2["licence"] = "TODO"
    one = os.path.join(d, "CARD_todo.md")
    open(one, "w").write(C.dump(h2, body))
    for card, want in ((full, True), (one, False)):
        b = os.path.join(d, os.path.basename(card) + ".sbibundle")
        rc, txt = _tool("export", "--config", c["cfg"], "--eval-config", c["ecfg"], "--ensemble",
                        "S-A1:A1", "--out", b, "--extraction", c["ext"], "--card", card)
        assert rc == 0, txt[-800:]
        mm = json.loads(dict(_zip(b))["manifest.json"])
        assert mm["card_complete"] is want, (card, mm["card_complete"])
        if want:
            c["complete_bundle"] = b
        rc, txt = _tool("card-check", card, "--bundle", b)
        assert rc == 0 and (("CARD OK: complete" in txt) == want), txt


# ===================================================================== block 45

def _infer(bundle_path, windows, **kw):
    from sbitool import bundle, infer
    return infer.run(bundle.read(bundle_path), windows, log=lambda t: None, **kw)


def test_t45a_the_same_numbers_as_the_evaluation():
    c = ctx()
    v, ec = _view()
    import ev_rec_data
    for name in ENSEMBLES:
        stem = _stem(name)
        prov = json.load(open(os.path.join(OUT, "samples", stem + "_sample.json")))
        for r in prov["banks"]:
            bank = ev_rec_data.recording(v, r)
            res = _infer(c["bundles"][name], np.asarray(bank["x"], dtype=np.float32),
                         seed=int(prov["seeds"][r]))
            A = res["arrays"]
            smp = _npz(os.path.join(OUT, "samples", "%s_%s.npz" % (stem, r)))
            met = _npz(os.path.join(OUT, "metrics", "%s_%s.npz" % (stem, r)))
            for k in ("draws", "draw_member", "logq_draws", "logq_draws_own", "z"):
                assert A[k].dtype == smp[k].dtype, k
                assert_array_equal(A[k], smp[k], err_msg=k)
            for k in ("mhat", "sigma", "q05", "q50", "q95", "contraction"):
                assert A[k].dtype == np.float32
                assert_array_equal(A[k], met[k], err_msg=k)
            for k in ("gate_t", "gate_r", "gate_p_member", "gate_T", "gate_p"):
                # 1e-12: the same functions on the same arrays (T45a)
                assert_allclose(A[k], met[k], rtol=0, atol=1e-12, err_msg=k)
            assert_array_equal(A["gate_flag"], met["gate_flag"])
            mj = json.load(open(os.path.join(OUT, "metrics", "%s_%s.json" % (stem, r))))
            assert res["json"]["gate"]["n_flagged"] == mj["gate"]["n_flagged"]
            assert res["json"]["gate"]["k_gate"] == mj["gate"]["k_gate"]


def test_t45b_mixture_and_json():
    c = ctx()
    v, ec = _view()
    import ev_rec_data
    bank = ev_rec_data.recording(v, "real")
    x = np.asarray(bank["x"], dtype=np.float32)
    res = _infer(c["bundles"]["S-A1:A1"], x)
    A, J = res["arrays"], res["json"]
    pooled = A["draws"].reshape(-1, A["draws"].shape[2]).astype(np.float64)
    # 1e-12: numpy's mean, SD (ddof 1) and linear quantiles of the same float64 numbers
    assert_allclose(A["mix_mean"], pooled.mean(axis=0), rtol=0, atol=1e-12)
    assert_allclose(A["mix_sd"], pooled.std(axis=0, ddof=1), rtol=0, atol=1e-12)
    for q, k in ((0.05, "mix_q05"), (0.5, "mix_q50"), (0.95, "mix_q95")):
        assert_allclose(A[k], np.quantile(pooled, q, axis=0), rtol=0, atol=1e-12)
    from sbitool import infer
    assert tuple(J) == tuple(infer.JSON_KEYS)
    assert set(J) == {"schema", "bundle", "n", "n_post", "M", "p", "seed", "batch_rows",
                      "parameters", "gate", "ids", "mixture_note", "software"}
    man = json.loads(dict(_zip(c["bundles"]["S-A1:A1"]))["manifest.json"])
    assert J["schema"] == "sbitool_posterior/1" and J["n"] == x.shape[0]
    assert J["bundle"]["sha256"] == _sha(open(c["bundles"]["S-A1:A1"], "rb").read())
    assert J["n_post"] == man["sample"]["n_post"] and J["seed"] == man["sample"]["seed"]
    assert J["batch_rows"] == man["sample"]["batch_rows"]
    for k in ("alpha", "k_gate", "n_null", "size_mean", "size_lo", "size_hi", "level"):
        assert J["gate"][k] == man["gate"][k], k
    assert J["gate"]["n_flagged"] == int(A["gate_flag"].sum())
    assert J["gate"]["share_flagged"] == int(A["gate_flag"].sum()) / x.shape[0]
    assert_array_equal(A["gate_flag"], A["gate_p"] < man["gate"]["alpha"])
    note = J["mixture_note"].lower()
    assert "mixture" in note and "not the posterior" in note
    assert "posterior of the recording" not in note.replace("not the posterior of the "
                                                            "recording", "")


def test_t45c_contract_and_seeds():
    c = ctx()
    v, ec = _view()
    import ev_rec_data
    from sbitool import infer
    x = np.asarray(ev_rec_data.recording(v, "real")["x"], dtype=np.float32)
    n = x.shape[0]
    b = c["bundles"]["S-A2-R0:A2"]
    man = json.loads(dict(_zip(b))["manifest.json"])
    M, p, n_post, d_z = man["M"], man["p"], man["sample"]["n_post"], man["d_z"]
    ids = {"subregion": np.arange(n) % 3, "tag": np.arange(n) * 2.5}
    r1 = _infer(b, x, ids=ids)
    A = r1["arrays"]
    shapes = {"draws": (n, n_post, p), "draw_member": (n_post,), "logq_draws": (n, n_post),
              "logq_draws_own": (n, n_post), "z": (M, n, d_z), "mhat": (n, p),
              "sigma": (n, p), "q05": (n, p), "q50": (n, p), "q95": (n, p),
              "contraction": (n, p), "gate_t": (M, n), "gate_r": (M, n),
              "gate_p_member": (M, n), "gate_T": (n,), "gate_p": (n,), "gate_flag": (n,),
              "mix_mean": (p,), "mix_sd": (p,), "mix_q05": (p,), "mix_q50": (p,),
              "mix_q95": (p,), "rows": (n,)}
    dtypes = dict(draws="float32", draw_member="int64", logq_draws="float32",
                  logq_draws_own="float32", z="float32", gate_flag="bool", rows="int64",
                  gate_t="float64", gate_r="float64", gate_p_member="float64",
                  gate_T="float64", gate_p="float64", mix_mean="float64")
    for k, sh in shapes.items():
        assert A[k].shape == sh, (k, A[k].shape)
        if k in dtypes:
            assert str(A[k].dtype) == dtypes[k], (k, A[k].dtype)
    # n_post / M per member in member order; contraction = 1 - 12 sigma^2 (R3)
    assert_array_equal(A["draw_member"], np.repeat(np.arange(M), n_post // M))
    assert_allclose(A["contraction"], 1.0 - 12.0 * A["sigma"].astype(np.float64) ** 2,
                    rtol=0, atol=1e-6)        # float32 storage of both
    assert_array_equal(A["subregion"], ids["subregion"])
    assert_array_equal(A["tag"], ids["tag"])
    for k in A:
        if k not in ("draw_member", "rows", "subregion", "tag", "gate_flag"):
            assert np.all(np.isfinite(A[k])), k
    # write: the two files; two runs with one seed identical
    d1, d2 = os.path.join(OUT, "inf1"), os.path.join(OUT, "inf2")
    infer.write(r1, d1)
    infer.write(_infer(b, x, ids=ids), d2)
    z1, z2 = _npz(os.path.join(d1, "posterior.npz")), _npz(os.path.join(d2, "posterior.npz"))
    assert set(z1) == set(z2)
    for k in z1:
        assert_array_equal(z1[k], z2[k], err_msg=k)
    assert open(os.path.join(d1, "posterior.json"), "rb").read() == \
        open(os.path.join(d2, "posterior.json"), "rb").read()
    # another seed: other draws, the same z; another batch (n > batch): other draws
    r3 = _infer(b, x, seed=man["sample"]["seed"] + 1)
    assert not np.array_equal(r3["arrays"]["draws"], A["draws"])
    assert_array_equal(r3["arrays"]["z"], A["z"])
    assert n > man["sample"]["batch_rows"]
    r4 = _infer(b, x, batch=n)
    assert not np.array_equal(r4["arrays"]["draws"], A["draws"])


def test_t45d_end_to_end_from_files():
    c = ctx()
    d = os.path.join(OUT, "e2e45")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    rng = np.random.default_rng(4504)
    n = 40000                                          # 40 s at 1 kHz: K = 400, W = 128: 3
    sp = {e: np.sort(rng.choice(n, size=int(rng.integers(40, 160)), replace=False))
          for e in range(36)}
    U.write_ptrain(os.path.join(d, "rec"), sp, n)
    b = c["bundles"]["S-A1:A1"]
    opts = ["--option", FS_OPT, "--option", "grid_width=6", "--option",
            "index_base=0", "--option", "pitch_um=60.0"]
    rc, txt = _tool("windows", "--bundle", b, "--recording", os.path.join(d, "rec"), "--adapter",
                    "ptrain", "--out", os.path.join(d, "w.npz"), *opts)
    # 2 subregions x floor(400 / 128) = 6 windows
    assert rc == 0 and "WINDOWS OK: 6 window(s) of 2 subregion(s)" in txt, txt
    rc, txt = _tool("infer", "--bundle", b, "--windows", os.path.join(d, "w.npz"), "--out",
                    os.path.join(d, "o1"))
    assert rc == 0 and "INFER OK: 6 window(s)" in txt, txt
    rc, txt = _tool("infer", "--bundle", b, "--recording", os.path.join(d, "rec"), "--adapter",
                    "ptrain", "--out", os.path.join(d, "o2"), *opts)
    assert rc == 0 and "INFER OK: 6 window(s)" in txt, txt
    z1, z2 = (_npz(os.path.join(d, o, "posterior.npz")) for o in ("o1", "o2"))
    for k in ("draws", "z", "gate_p", "subregion", "window_idx", "t_start_s"):
        assert_array_equal(z1[k], z2[k], err_msg=k)
    # the same windows through the module: the inference's numbers are the windows'
    with np.load(os.path.join(d, "w.npz")) as w:
        res = _infer(b, w["windows"])
        assert_array_equal(res["arrays"]["draws"], z1["draws"])
        assert_array_equal(w["t_start_s"], z1["t_start_s"])


def test_t45e_refusals():
    c = ctx()
    v, ec = _view()
    import ev_rec_data
    from sbitool import infer
    b = c["bundles"]["S-A1:A1"]
    man = json.loads(dict(_zip(b))["manifest.json"])
    x = np.asarray(ev_rec_data.recording(v, "real")["x"], dtype=np.float32)
    with pytest.raises(infer.InferRefused):
        _infer(b, x, n_post=man["M"] * 3 + 1)
    with pytest.raises(infer.InferRefused):
        _infer(b, np.zeros((3, man["W"] + 1), dtype=np.float32))
    bad = x.copy()
    bad[2, 5] = np.nan
    with pytest.raises(infer.InferRefused):
        _infer(b, bad)
    with pytest.raises(infer.InferRefused):
        _infer(b, x.astype(np.float64))
    d = os.path.join(OUT, "refuse45")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    np.savez(os.path.join(d, "w.npz"), windows=x)
    rc, txt = _tool("infer", "--bundle", b, "--windows", os.path.join(d, "w.npz"), "--out",
                    os.path.join(d, "o"), "--n-post", str(man["M"] * 3 + 1))
    assert rc == 3 and "REFUSED" in txt and not os.path.exists(os.path.join(d, "o")), txt
    np.savez(os.path.join(d, "w2.npz"), windows=np.zeros((2, man["W"] - 1), np.float32))
    rc, txt = _tool("infer", "--bundle", b, "--windows", os.path.join(d, "w2.npz"), "--out",
                    os.path.join(d, "o"))
    assert rc == 3 and not os.path.exists(os.path.join(d, "o")), txt
    ents = _zip(b)
    tb = os.path.join(d, "t.sbibundle")
    _rezip(tb, [(n, (bb[:-1] + bytes([bb[-1] ^ 1])) if n.startswith("members/") else bb)
                for n, bb in ents])
    rc, txt = _tool("infer", "--bundle", tb, "--windows", os.path.join(d, "w.npz"), "--out",
                    os.path.join(d, "o"))
    assert rc == 3 and "REFUSED" in txt and not os.path.exists(os.path.join(d, "o")), txt


def test_d178_units_sources_in_order():
    """Block 41, Inputs (D-178): --units when given; else the sidecar's latent_spec.units
    when it holds p non-empty strings; else "1" when the declared coordinate is the unit
    box (no bounds, or every bound [0, 1]); else a refusal naming --units."""
    from sbitool import export as X
    p = 3
    side = {"latent_spec": {"units": ["Hz", "1", "mV"]}}
    assert X._units(side, p, [[0, 5], [1, 2], [3, 4]]) == ["Hz", "1", "mV"]
    assert X._units(side, p, None, ["a", "b", "c"]) == ["a", "b", "c"]
    assert X._units({}, p, None) == ["1"] * p
    assert X._units({}, p, [[0.0, 1.0]] * p) == ["1"] * p
    for bad_side in ({}, {"latent_spec": {"units": ["Hz", ""]}},
                     {"latent_spec": {"units": ["Hz", "", "mV"]}}):
        with pytest.raises(X.ExportRefused) as ei:
            X._units(bad_side, p, [[0, 5], [1, 2], [3, 4]])
        assert "--units" in str(ei.value)
    with pytest.raises(X.ExportRefused):
        X._units(side, p, None, ["a", "b"])


def test_t41c_complete_card_with_card_complete_false():
    """card_complete is compared with the card both ways: a bundle exported with a complete
    card whose manifest says false is refused, naming card_complete."""
    c = T_ctx()
    b = c.get("complete_bundle")
    assert b and os.path.isfile(b), "test_t41e makes the complete bundle"
    ents = _zip(b)
    assert json.loads(dict(ents)["manifest.json"])["card_complete"] is True
    p = os.path.join(OUT, "tamper2", "complete_false.sbibundle")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _rezip(p, _redigest(ents, {"manifest": lambda m: m.__setitem__("card_complete", False)}))
    rc, out = _check(p)
    assert rc == 3 and "card_complete" in out, out[-600:]
    rc, out = _check(b)
    assert rc == 0 and out.rstrip().endswith("sha256 " + _sha(open(b, "rb").read())), out
