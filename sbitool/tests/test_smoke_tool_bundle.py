#!/usr/bin/env python3
"""Smoke test of block 41: the model bundle -- export, check, rebuild (specs/SPEC.md block 41, "Test
oracles", checks T41*).

On rec_fixture's recorded-kind project (cfg_rec, ecfg_rec): its ensembles
are sampled (block 35, ev_rec_sample.run) and evaluated (block 36,
ev_rec_metrics.compute) into this test's own folder, with a copy of the
evaluation's configuration whose out_root is that folder, then exported. The
fixture's tree is never written. A check that cannot run here raises
unittest.SkipTest("<reason>"), so the summary shows why.

Checks
    test_t41a_rebuild        the rebuild is the pipeline's: digests, densities, draws, embeddings bit for bit
    test_t41b_content        the bundle's members, manifest, gate sets and metrics
    test_t41c_safety         a changed byte is refused; the manifest agrees with its members;
                             the pass line's sha256; no pickle on the reading path
    test_t41d_refusals       the refusals write nothing (an --extraction of another bin
                             width included); the same export twice
    test_t41e_card           a complete card, a draft, the template

Run
    python3 sbitool/tests/test_smoke_tool_bundle.py [check ...]
        Prints a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit 1 if any check failed, errored or
        is still TODO. REC_FIXTURE=<dir> reuses a fixture tree built before
        (hpc/pipeline/tests/rec_fixture.py); REC_KEEP=1 keeps the one this
        run builds.

Pure ASCII, LF only.
"""

import ast
import contextlib
import copy
import hashlib
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
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_PIPE_TESTS = os.path.join(_REPO, "hpc", "pipeline", "tests")
_EVAL = os.path.join(_REPO, "hpc", "pipeline", "evaluation")
for _p in (_PIPE_TESTS, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rec_fixture  # noqa: E402

from sbitool import schema as S  # noqa: E402

SEED = 20261127  # every random number of this file derives from it
_TMP = []
ENSEMBLES = ("S-A1:A1", "S-A2-R0:A2")   # the fixture's recorded-kind ensembles
N_ROWS = 16        # T41a: rows of D_sim
N_DRAWS = 8        # T41a: draws per row under one seed
_CTX = {}          # the evaluated project and its exports, made once per process


def _tmpdir(tag):
    d = tempfile.mkdtemp(prefix="sbitool_%s_" % tag)
    _TMP.append(d)
    return d


def _cleanup():
    if os.environ.get("REC_KEEP", "").strip() == "1":
        for d in _TMP:
            print("[test_smoke_tool_bundle] kept: %s" % d, flush=True)
    else:
        for d in _TMP:
            shutil.rmtree(d, ignore_errors=True)
    rec_fixture.cleanup()


# ------------------------------------------------------------------ helpers

def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_ecfg(src, out_root, name, **over):
    """A copy of the evaluation configuration `src` with another out_root (and
    `over`), written into out_root's parent (test_smoke_rec_metrics')."""
    d = _read_json(src)
    d["out_root"] = out_root
    d.update(over)
    path = os.path.join(os.path.dirname(out_root), name)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path


def _software():
    import platform
    import numpy
    import safetensors
    import sbi
    import torch
    import zuko
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "sbi": str(sbi.__version__), "zuko": str(zuko.__version__),
            "numpy": str(numpy.__version__), "safetensors": str(safetensors.__version__)}


def _card_text(arm, M, names, extraction=None, todo=None):
    """A card written by the test from schema.CARD_KEYS, in agreement with a
    manifest of this arm, M, searched parameters and extraction; `todo` a
    top-level key whose value is replaced by TODO."""
    import yaml
    h = {"title": "The fixture's ensemble, a test bundle",
         "scientific_question": "Does the bundle rebuild the pipeline's members?",
         "simulator": {"model_type": "bench burst provider",
                       "reference": "hpc/pipeline/tests/rec_fixture.py",
                       "fixed_parameters": {"n_neurons": "100 (1)"},
                       "searched_parameters": list(names)},
         "training": {"arm": arm, "members": int(M), "n_networks": 8, "n_windows": 16,
                      "split_fracs": [0.85, 0.1, 0.05], "hyperparameters": {"epochs": 2},
                      "hyperparameter_search": "the fixture's reduced search"},
         "extraction": S.NOT_APPLICABLE if extraction is None else extraction,
         "references": ["specs/SPEC.md block 41"],
         "authors": [{"name": "Test Author", "contact": "test@example.org"}],
         "licence": "CC-BY-4.0", "cite": "the repository",
         "data": {"simulated_bank": {"location": "the fixture's tree", "public": False,
                                     "size_bytes": None, "sha256": None},
                  "recordings": {"location": "the fixture's tree", "public": False}},
         "software": _software(), "limitations": "a test fixture of two members"}
    assert tuple(h) == S.CARD_KEYS
    if todo:
        h[todo] = S.TODO
    return "---\n%s---\n\n# A test card\n\nWritten by test_smoke_tool_bundle.py.\n" % (
        yaml.safe_dump(h, sort_keys=False))


def _card_for(e, prov, tag, todo=None):
    path = os.path.join(_tmpdir("card"), "CARD.md")
    with open(path, "w") as fh:
        fh.write(_card_text(e.arm, len(e.records), prov["axes"]["names"], todo=todo))
    return path


def _export_argv(name, out, card=None, extraction=None, ecfg=None, ctx=None):
    c = ctx or _CTX
    argv = ["--config", c["cfg"], "--eval-config", ecfg or c["ecfg"], "--ensemble", name,
            "--out", out]
    if card:
        argv += ["--card", card]
    if extraction:
        argv += ["--extraction", extraction]
    return argv


def _main(module_main, argv):
    """(exit code, printed text) of a command's main, in this process."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = module_main(argv)
    return code, buf.getvalue()


def _ctx():
    """The recorded-kind project sampled and evaluated into this test's
    folder, and one bundle per ensemble exported with a complete card."""
    if _CTX:
        return _CTX
    rec_fixture.need_training_env()
    import ev_rec_config
    import ev_rec_metrics
    import ev_rec_plan
    import ev_rec_sample
    from sbitool import export
    F = rec_fixture.fixture()
    root = _tmpdir("eval")
    out_root = os.path.join(root, "rec_eval_out")
    ecfg_path = _write_ecfg(F["ecfg_rec"], out_root, "rec_eval.json")
    ecfg = ev_rec_config.load(ecfg_path)
    view = ev_rec_plan.load_view(F["cfg_rec"], ecfg)
    assert tuple(e.name for e in view.ensembles) == ENSEMBLES, [e.name for e in view.ensembles]
    for name in ENSEMBLES:
        ev_rec_sample.run(F["cfg_rec"], ecfg_path, name, log=lambda t: None)
    lines = []
    ev_rec_metrics.compute(F["cfg_rec"], ecfg_path, ncpus=1, log=lines.append)
    assert lines[-1].startswith("REC METRICS OK: "), lines[-5:]
    ctx = dict(F=F, cfg=F["cfg_rec"], ecfg=ecfg_path, ecfg_obj=ecfg, view=view,
               out_root=out_root, tag=view.cfg.data["dataset"], bundles={}, logs={})
    bdir = _tmpdir("bundles")
    for name in ENSEMBLES:
        e = ev_rec_plan.find(view, name)
        prov = _read_json(ev_rec_plan.sample_json(ecfg, e.stem))
        card = _card_for(e, prov, ctx["tag"])
        out = os.path.join(bdir, "%s.sbibundle" % e.stem)
        code, text = _main(export.main, _export_argv(name, out, card=card, ctx=ctx))
        assert code == 0, (name, text[-3000:])
        assert "BUNDLE EXPORT OK: %s (%d members, " % (out, len(e.records)) in text, text[-2000:]
        ctx["bundles"][name] = (out, card)
        ctx["logs"][name] = text
        with zipfile.ZipFile(out) as zf:
            listing = ", ".join("%s (%d B)" % (i.filename, i.file_size) for i in zf.infolist())
        print("[T41] %s exported: %d bytes; members: %s" % (name, os.path.getsize(out), listing),
              flush=True)
    _CTX.update(ctx)                 # only once every export passed
    return _CTX


def _ensemble(name):
    import ev_rec_plan
    c = _ctx()
    e = ev_rec_plan.find(c["view"], name)
    prov = _read_json(ev_rec_plan.sample_json(c["ecfg_obj"], e.stem))
    return e, prov


def _ckpt(record):
    """The member's checkpoint, read by the test (the reference): its meta and
    model_sha256."""
    import torch
    return torch.load(record[:-len(".json")] + "_ckpt.pt", map_location="cpu",
                      weights_only=False)


def _bits_equal(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _max_abs(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    return float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0


# the package's side of T41a, in a process of its own: no bank and no checkpoint
# file is reachable, and torch.load, pickle and numpy.load raise if called
_CHILD = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from sbitool import _sources, bundle, model
_sources.use("model", "evaluation")
model._builders()
import ev_sample
import numpy, pickle, torch
from safetensors.numpy import load_file, save_file
def _forbidden(*a, **k):
    raise AssertionError("the package's reading path called a forbidden loader")
torch.load = numpy.load = pickle.load = pickle.loads = _forbidden
inp = load_file(sys.argv[3])
b = bundle.read(sys.argv[2])
models = model.load(b)
out = {}
seed, nd = int(sys.argv[5]), int(sys.argv[6])
for k, m in enumerate(models):
    box = inp["in_box"].astype(bool)
    out["logq_%d" % k] = ev_sample.logq_at(m, inp["theta"], inp["x"], box, 16)
    out["draws_%d" % k] = ev_sample.draws_of(m, inp["x"], nd, seed, 16)
    out["lqd_%d" % k] = ev_sample.logq_draws_of(m, out["draws_%d" % k], inp["x"], 16)
    out["z_%d" % k] = ev_sample.embed(m, inp["x"], 16)
# safetensors writes an array's buffer as it lies in memory: the draws are a strided
# view (sbi's (S, B, d) transposed), so each array is made C-contiguous first
save_file(dict((k, numpy.ascontiguousarray(v)) for k, v in out.items()), sys.argv[4])
print("CHILD OK %d" % len(models))
"""


# ------------------------------------------------------------------ checks

def test_t41a_rebuild():
    """the rebuild is the pipeline's: digests, densities, draws, embeddings bit for bit (specs/SPEC.md block 41)."""
    # Tolerance 0 (bit for bit): the bundle's member and ev_models.rebuild's are the same
    # code, the same weights and the same inputs on the same threads; any difference at all
    # means the bundle does not hold the pipeline's member (SPEC sec. 2.7: "bit for bit").
    import numpy as np
    import ev_rec_data
    import ev_sample
    import run_joint_arms as R
    from joint_model import state_sha256
    from safetensors.numpy import load_file, save_file
    from sbitool import bundle, model
    c = _ctx()
    index, _ = ev_rec_data.sim_index(c["view"])
    seed = SEED % (2 ** 31)
    report = []
    for name in ENSEMBLES:
        e, prov = _ensemble(name)
        path = c["bundles"][name][0]
        b = bundle.read(path)
        models = model.load(b)
        rec0 = _read_json(e.records[0])
        rows = np.asarray(ev_rec_data.report_rows(index, rec0["split_fracs"],
                                                  rec0["split_seed"], rec0["split_hash"]))
        assert rows.size >= N_ROWS, rows.size
        rows = np.sort(rows[:N_ROWS])
        x = R._rows_x(index, rows).numpy()
        theta = np.asarray(index.theta, dtype=np.float64)[rows]
        in_box = np.all((theta > 0) & (theta < 1), axis=1)
        work = _tmpdir("child")
        inp, outp = os.path.join(work, "in.safetensors"), os.path.join(work, "out.safetensors")
        save_file({"x": np.ascontiguousarray(x), "theta": theta,
                   "in_box": in_box.astype(np.uint8)}, inp)
        # the bundle copied where no record, checkpoint or bank lies beside it
        lone = os.path.join(work, "lone.sbibundle")
        shutil.copyfile(path, lone)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
        r = subprocess.run([sys.executable, "-c", _CHILD, _REPO, lone, inp, outp, str(seed),
                            str(N_DRAWS)], cwd=work, env=env, capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0 and "CHILD OK %d" % len(e.records) in r.stdout, (
            r.stdout[-2000:], r.stderr[-3000:])
        child = load_file(outp)
        for k, rpath in enumerate(e.records):
            ck = _ckpt(rpath)
            got = state_sha256(models[k].state_dict())
            assert got == ck["model_sha256"], (name, k, got, ck["model_sha256"])
            assert b.manifest["members"][k]["state_sha256"] == ck["model_sha256"], (name, k)
            ref, info = __import__("ev_models").rebuild(rpath, index)
            assert info["seed"] == b.manifest["members"][k]["seed"], (name, k, info["seed"])
            cmp = {
                "logq": (ev_sample.logq_at(ref, theta, x, in_box, 16),
                         ev_sample.logq_at(models[k], theta, x, in_box, 16), child["logq_%d" % k]),
                "draws": (ev_sample.draws_of(ref, x, N_DRAWS, seed, 16),
                          ev_sample.draws_of(models[k], x, N_DRAWS, seed, 16),
                          child["draws_%d" % k]),
                "z": (ev_sample.embed(ref, x, 16), ev_sample.embed(models[k], x, 16),
                      child["z_%d" % k])}
            lqd_ref = ev_sample.logq_draws_of(ref, cmp["draws"][0], x, 16)
            lqd = ev_sample.logq_draws_of(models[k], cmp["draws"][0], x, 16)
            cmp["logq_draws"] = (lqd_ref, lqd, child["lqd_%d" % k])
            for what, (a, b_in, b_child) in sorted(cmp.items()):
                d1, d2 = _max_abs(a, b_in), _max_abs(a, b_child)
                report.append("%s member %d %s: max |diff| %g in process, %g in the lone process"
                              % (name, k, what, d1, d2))
                assert _bits_equal(a, b_in), (name, k, what, "in process", d1)
                assert _bits_equal(a, b_child), (name, k, what, "lone process", d2)
            assert np.isfinite(cmp["logq"][0][in_box]).all() and in_box.any(), (name, k)
    _CTX["t41a"] = report
    print("\n".join("[T41a] " + line for line in report), flush=True)


def _zip_members(path):
    with zipfile.ZipFile(path) as zf:
        return dict((n, zf.read(n)) for n in zf.namelist())


def test_t41b_content():
    """the bundle's members, manifest, gate sets and metrics (specs/SPEC.md block 41)."""
    import numpy as np
    import ev_gate
    from sbitool import bundle
    c = _ctx()
    mdir = c["ecfg_obj"].sub("metrics")
    sdir = c["ecfg_obj"].sub("samples")
    for name in ENSEMBLES:
        e, prov = _ensemble(name)
        path, card = c["bundles"][name]
        zm = _zip_members(path)
        M = len(e.records)
        man = json.loads(zm["manifest.json"].decode("ascii"))
        # [block 48] and the training's files the manifest announces (block 48's test checks them)
        want = sorted(list(S.FIXED_MEMBERS) + [S.MEMBER_FILE % k for k in range(M)]
                      + S.training_files(man["training"]))
        assert sorted(zm) == want, (name, sorted(zm))
        assert S.validate_manifest(man) == [], S.validate_manifest(man)
        assert bundle.check(path) == [], bundle.check(path)
        # parameters: the sample's axes
        assert man["parameters"]["names"] == prov["axes"]["names"], man["parameters"]
        assert man["parameters"]["bounds"] == prov["axes"]["bounds"], man["parameters"]
        assert len(man["parameters"]["units"]) == man["p"] == prov["p"]
        # gate: the metrics' gate
        sim_bytes = open(os.path.join(mdir, "%s_sim.json" % e.stem), "rb").read()
        sim = json.loads(sim_bytes.decode("utf-8"))
        assert man["gate"] == dict((k, sim["gate"][k]) for k in S.GATE_KEYS), (man["gate"],)
        # members: the records, in the members' order (the seeds ascending)
        assert [m["seed"] for m in man["members"]] == sorted(e.seeds) == list(e.seeds)
        for k, rpath in enumerate(e.records):
            rec, ck = _read_json(rpath), _ckpt(rpath)
            m = man["members"][k]
            assert m["file"] == S.MEMBER_FILE % k
            assert m["seed"] == rec["seed"] and m["split_seed"] == rec["split_seed"], (m, k)
            assert m["split_hash"] == rec["split_hash"], (m, k)
            assert m["state_sha256"] == ck["model_sha256"], (name, k)
            assert m["sha256"] == _sha(zm[m["file"]]), (name, k)
            if k == 0:
                meta = ck["meta"]
                a = man["architecture"]
                assert a["backbone"] == ("fixed-stats" if meta.get("fixed_summary")
                                         else meta["backbone"]), (a["backbone"], meta["backbone"])
                assert a["backbone_repr"] == meta["backbone_repr"]
                assert (man["W"], man["p"]) == (meta["W"], meta["d_theta"])
                assert a["embedding_size"] == meta["embedding_size"]
                for key in ("hidden_features", "num_transforms", "num_bins", "depth_exponent",
                            "dropout"):
                    assert a[key] == rec["config"][key], (key, a[key], rec["config"][key])
                assert a["hidden_layers"] == rec["config"]["flow_hidden_layers"]
        assert man["arm"] == e.arm and man["M"] == M
        assert man["sample"] == {"n_post": prov["n_post"], "seed": c["ecfg_obj"].seed,
                                 "batch_rows": c["ecfg_obj"].batch_rows}
        assert man["provenance"]["ensemble"] == name and man["provenance"]["project"] == c["tag"]
        assert man["provenance"]["best_fit_complete_sha256"] == prov["best_fit_complete_sha256"]
        assert man["provenance"]["sample_json_sha256"] == _sha(
            open(os.path.join(sdir, "%s_sample.json" % e.stem), "rb").read())
        assert man["software"] == _software(), man["software"]
        # the gate's sets: the sample's z at the roles 1, 2, 3, in row order
        b = bundle.read(path)
        g = b.gate_sets()
        with np.load(os.path.join(sdir, "%s_sim.npz" % e.stem)) as z:
            role, z_idx, Z = z["gate_role"], z["z_idx"], z["z"]
        for key, r in (("z_ref", ev_gate.ROLE_REF), ("z_rank", ev_gate.ROLE_RANK),
                       ("z_null", ev_gate.ROLE_NULL)):
            rows = np.flatnonzero(role == r)
            want_z = Z[:, np.searchsorted(z_idx, rows), :]
            assert np.array_equal(z_idx[np.searchsorted(z_idx, rows)], rows)
            assert g[key].dtype == np.float32 and _bits_equal(g[key], want_z), (name, key)
            assert g[key].shape == (M, man["gate"]["n_" + key[2:]], man["d_z"]), g[key].shape
        # evaluation/sim.json: the metrics' file byte for byte; CARD.md: the card given
        assert zm["evaluation/sim.json"] == sim_bytes
        assert b.sim_metrics() == sim
        assert zm["CARD.md"] == open(card, "rb").read()
        assert b.card == open(card, encoding="utf-8").read()
        assert b.manifest == man and b.sha256 == _sha(open(path, "rb").read())


def _rezip(src, dst, edit=None, add=None):
    """A copy of the zip `src` at `dst` with member contents changed by
    edit(name, data) -> data (the zip's CRCs recomputed, so that only the
    bundle's own digests can see it) and members `add` appended."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in zin.namelist():
            data = zin.read(n)
            zout.writestr(n, edit(n, data) if edit else data)
        for n, data in (add or {}).items():
            zout.writestr(n, data)
    return dst


def _flip_last(target):
    def edit(n, data):
        if n != target:
            return data
        return data[:-1] + bytes([data[-1] ^ 0x01])
    return edit


def _manifest_text(m):
    """A manifest as the export writes it (indent 1, sorted keys, ASCII, a final
    newline)."""
    return (json.dumps(m, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _rebundle(src, dst, man_edit=None, members=None):
    """A copy of the bundle `src` at `dst` whose members `members` (name -> bytes)
    replace the originals, with the manifest's files digests made to match them
    (so that only the manifest's agreement with its members can see the change),
    and then man_edit(manifest) applied: a valid manifest whose every digest
    holds."""
    ents = _zip_members(src)
    man = json.loads(ents["manifest.json"].decode("ascii"))
    for n, data in (members or {}).items():
        ents[n] = data
        man["files"][n] = _sha(data)
    if man_edit:
        man_edit(man)
    ents["manifest.json"] = _manifest_text(man)
    with zipfile.ZipFile(src) as zin:
        order = zin.namelist()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in order:
            zout.writestr(n, ents[n])
    assert S.validate_manifest(man) == [], S.validate_manifest(man)
    return dst


def _refused_naming(bundle, path, what, names):
    """bundle-check exits 3 with REFUSE lines naming each of `names`; read()
    refuses too."""
    code, text = _main(bundle.main, [path])
    assert code == bundle.EXIT_REFUSE, (what, code, text)
    lines = [l for l in text.splitlines() if l.strip()]
    assert lines and all(l.startswith("REFUSE: ") for l in lines), (what, text)
    for n in names:
        assert n in text, (what, n, text)
    try:
        bundle.read(path)
        raise AssertionError("%s: read() did not refuse" % what)
    except bundle.BundleError:
        pass
    return text


def _calls(path):
    """The dotted names of every call and every imported module of a file."""
    tree = ast.parse(open(path, "rb").read().decode("ascii"), filename=path)
    calls, imports = set(), set()

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = dotted(node.value)
            return None if base is None else base + "." + node.attr
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            d = dotted(node.func)
            if d:
                calls.add(d)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.update("%s.%s" % (node.module, a.name) for a in node.names)
            imports.add(node.module)
    return calls, imports


def test_t41c_safety():
    """a changed byte is refused; no pickle on the reading path (specs/SPEC.md block 41)."""
    from sbitool import bundle
    c = _ctx()
    good = c["bundles"][ENSEMBLES[0]][0]
    work = _tmpdir("safety")
    code, text = _main(bundle.main, [good])
    assert code == 0 and text.startswith("BUNDLE CHECK OK: %s" % good), (code, text)
    # [fix round of 2026-11-28] the pass line ends with the file's sha256 in full
    good_sha = _sha(open(good, "rb").read())
    assert text.rstrip("\n").endswith(", sha256 %s" % good_sha), (text, good_sha)
    # one byte changed: in a member's weights, in the manifest (a digest's hex digit; and a
    # byte that breaks its JSON), in the gate's file: exit 3 naming the member
    man = _zip_members(good)["manifest.json"]
    want0 = json.loads(man.decode("ascii"))["members"][0]["sha256"]
    i = man.index(want0.encode("ascii"))
    swap = b"0" if man[i:i + 1] != b"0" else b"1"
    cases = [
        ("member_0 weights", _flip_last(S.MEMBER_FILE % 0), S.MEMBER_FILE % 0),
        ("manifest digest", lambda n, d: d[:i] + swap + d[i + 1:] if n == "manifest.json" else d,
         S.MEMBER_FILE % 0),
        ("manifest JSON", lambda n, d: d.replace(b"{", b"[", 1) if n == "manifest.json" else d,
         "manifest.json"),
        ("gate", _flip_last("evaluation/gate.safetensors"), "evaluation/gate.safetensors"),
        ("sim.json", lambda n, d: d.replace(b"0", b"1", 1) if n == "evaluation/sim.json" else d,
         "evaluation/sim.json"),
        ("card", lambda n, d: d + b" " if n == "CARD.md" else d, "CARD.md")]
    for what, edit, named in cases:
        bad = _rezip(good, os.path.join(work, "bad.sbibundle"), edit=edit)
        code, text = _main(bundle.main, [bad])
        assert code == bundle.EXIT_REFUSE, (what, code, text)
        lines = [l for l in text.splitlines() if l.strip()]
        assert lines and all(l.startswith("REFUSE: ") for l in lines), (what, text)
        assert named in text, (what, named, text)
        try:
            bundle.read(bad)
            raise AssertionError("%s: read() did not refuse" % what)
        except bundle.BundleError as exc:
            assert named in str(exc), (what, str(exc))
    # [fix round of 2026-11-28, D-180, the tester's F2] the manifest compared with the
    # members it describes: an edit that leaves it valid JSON of a valid form, every
    # digest holding, is refused where it disagrees with a member, naming what disagrees
    ents = _zip_members(good)
    man0 = json.loads(ents["manifest.json"].decode("ascii"))
    sim0 = json.loads(ents["evaluation/sim.json"].decode("utf-8"))
    assert man0["card_complete"] is True and sim0["gate"]["alpha"] == man0["gate"]["alpha"]
    assert "axes" in sim0 and sim0["axes"]["names"] == man0["parameters"]["names"], sim0.get("axes")
    new_alpha = 0.6 if man0["gate"]["alpha"] != 0.6 else 0.5

    def setp(*path_value):
        *path, value = path_value

        def f(m):
            d = m
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
        return f
    p_tamper = os.path.join(work, "tamper.sbibundle")
    tampered = [
        ("the gate's alpha", setp("gate", "alpha", new_alpha), None, ["gate.alpha"]),
        ("a parameter's name", setp("parameters", "names",
                                    [man0["parameters"]["names"][0] + "x"]
                                    + man0["parameters"]["names"][1:]), None,
         ["parameters.names"]),
        ("a parameter's bound", setp("parameters", "bounds",
                                     None if man0["parameters"]["bounds"] is not None
                                     else [[0.0, 2.0]] * man0["p"]), None, ["parameters.bounds"]),
        ("card_complete false on a complete card", setp("card_complete", False), None,
         ["card_complete", "CARD.md"]),
        # sim.json without its gate (its digest made to match): a problem
        ("sim.json without gate", None,
         {"evaluation/sim.json": json.dumps(dict((k, v) for k, v in sim0.items()
                                                 if k != "gate"), allow_nan=False).encode("ascii")},
         ["evaluation/sim.json has no gate"]),
        # sim.json's gate of another value (its digest made to match)
        ("sim.json's gate.k_gate", None,
         {"evaluation/sim.json": json.dumps(dict(sim0, gate=dict(
             sim0["gate"], k_gate=sim0["gate"]["k_gate"] + 1)), allow_nan=False).encode("ascii")},
         ["gate.k_gate"]),
        # sim.json's axes of other names (its digest made to match)
        ("sim.json's axes", None,
         {"evaluation/sim.json": json.dumps(dict(sim0, axes=dict(
             sim0["axes"], names=["y" + n for n in sim0["axes"]["names"]])), allow_nan=False).encode("ascii")},
         ["parameters.names", "axes.names"])]
    for what, man_edit, members, names in tampered:
        _rebundle(good, p_tamper, man_edit=man_edit, members=members)
        _refused_naming(bundle, p_tamper, what, names)
    # a draft: the card replaced by one with a TODO and card_complete false passes;
    # card_complete true on that draft is refused (the hub would take it as complete)
    e0, prov0 = _ensemble(ENSEMBLES[0])
    todo_card = _card_text(e0.arm, len(e0.records), prov0["axes"]["names"],
                           todo="licence").encode("utf-8")
    draft = _rebundle(good, os.path.join(work, "draft.sbibundle"),
                      man_edit=setp("card_complete", False), members={"CARD.md": todo_card})
    code, text = _main(bundle.main, [draft])
    assert code == 0 and " bytes, draft), sha256 " in text, (code, text)
    _rebundle(good, p_tamper, man_edit=setp("card_complete", True),
              members={"CARD.md": todo_card})
    _refused_naming(bundle, p_tamper, "card_complete true on a draft", ["card_complete"])
    # a card that does not parse is not complete
    _rebundle(good, p_tamper, members={"CARD.md": b"no header\n"})
    _refused_naming(bundle, p_tamper, "card_complete true on a card that does not parse",
                    ["card_complete", "does not parse"])
    # a manifest edited into another valid one that agrees with every member passes (the
    # file cannot vouch for itself): its sha256 on the pass line is another one
    _rebundle(good, p_tamper, man_edit=setp("name", man0["name"] + "-renamed"))
    code, text = _main(bundle.main, [p_tamper])
    other_sha = _sha(open(p_tamper, "rb").read())
    assert code == 0 and text.rstrip("\n").endswith(", sha256 %s" % other_sha), (code, text)
    assert other_sha != good_sha
    # an extra member, a .pt member: refused
    for extra in ("extra.txt", "members/member_0.pt", "evaluation/z.npy"):
        bad = _rezip(good, os.path.join(work, "extra.sbibundle"), add={extra: b"x"})
        code, text = _main(bundle.main, [bad])
        assert code == bundle.EXIT_REFUSE and extra in text, (extra, code, text)
    # not a zip; a missing member
    junk = os.path.join(work, "junk.sbibundle")
    with open(junk, "wb") as fh:
        fh.write(b"not a zip")
    assert _main(bundle.main, [junk])[0] == bundle.EXIT_REFUSE
    miss = os.path.join(work, "miss.sbibundle")
    with zipfile.ZipFile(good) as zin, zipfile.ZipFile(miss, "w") as zout:
        for n in zin.namelist():
            if n != "evaluation/sim.json":
                zout.writestr(n, zin.read(n))
    code, text = _main(bundle.main, [miss])
    assert code == bundle.EXIT_REFUSE and "missing member evaluation/sim.json" in text, text
    # verified when read: a bundle changed after it was opened is refused at state(k)
    live = os.path.join(work, "live.sbibundle")
    shutil.copyfile(good, live)
    b = bundle.read(live)
    _rezip(good, live + ".x", edit=_flip_last(S.MEMBER_FILE % 0))
    os.replace(live + ".x", live)
    b.state(1)                                               # unchanged: read
    try:
        b.state(0)
        raise AssertionError("a member changed after opening was read")
    except bundle.BundleError as exc:
        assert S.MEMBER_FILE % 0 in str(exc), str(exc)
    # the command line itself (a process of its own): exit 0 with --rebuild, 3 on a change
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, "-m", "sbitool", "bundle-check", good, "--rebuild"],
                       cwd=_REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "BUNDLE CHECK OK: " in r.stdout, (r.stdout, r.stderr[-2000:])
    assert r.stdout.rstrip("\n").endswith(", sha256 %s" % good_sha), r.stdout
    bad = _rezip(good, os.path.join(work, "bad.sbibundle"), edit=_flip_last(S.MEMBER_FILE % 1))
    r = subprocess.run([sys.executable, "-m", "sbitool", "bundle-check", bad], cwd=_REPO,
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 3 and S.MEMBER_FILE % 1 in r.stdout, (r.returncode, r.stdout)
    # static: the reading path calls no torch.load, pickle or numpy.load
    forbidden_calls = {"torch.load", "pickle.load", "pickle.loads", "numpy.load", "np.load",
                       "joblib.load", "dill.load"}
    for mod in ("bundle.py", "model.py"):
        calls, imports = _calls(os.path.join(_REPO, "sbitool", mod))
        assert not (calls & forbidden_calls), (mod, calls & forbidden_calls)
        assert not any(i.split(".")[0] in ("pickle", "dill", "joblib", "cloudpickle")
                       for i in imports), (mod, imports)
        assert not any(i in ("torch.load", "numpy.load") for i in imports), (mod, imports)
    # export.py reads a checkpoint only through ev_models.rebuild
    calls, imports = _calls(os.path.join(_REPO, "sbitool", "export.py"))
    # (numpy.load reads block 35's npz there, with allow_pickle at its default, False)
    allowed = {"np.load", "numpy.load"}
    assert not (calls & (forbidden_calls - allowed)) and "pickle" not in imports, calls
    assert "allow_pickle" not in open(os.path.join(_REPO, "sbitool", "export.py")).read()
    assert "P.ev_models.rebuild" in calls, sorted(c for c in calls if "rebuild" in c)
    src = open(os.path.join(_REPO, "sbitool", "export.py")).read()
    assert "_ckpt.pt" not in src and ".pt\"" not in src


def _listing(d):
    out = []
    for root, _dirs, files in os.walk(d):
        out += [os.path.relpath(os.path.join(root, f), d) for f in files]
    return sorted(out)


def test_t41d_refusals():
    """the refusals write nothing; the same export twice (specs/SPEC.md block 41)."""
    from sbitool import export
    c = _ctx()
    name = ENSEMBLES[0]
    e, prov = _ensemble(name)
    card = c["bundles"][name][1]
    work = _tmpdir("refuse")
    outd = os.path.join(work, "out")
    os.makedirs(outd)
    out = os.path.join(outd, "x.sbibundle")
    # a copy of the evaluation without its samples' folder; one with samples and no metrics
    nos_root = os.path.join(work, "nosample", "rec_eval_out")
    os.makedirs(nos_root)
    nos = _write_ecfg(c["ecfg"], nos_root, "rec_eval.json")
    nom_root = os.path.join(work, "nometrics", "rec_eval_out")
    shutil.copytree(c["ecfg_obj"].sub("samples"), os.path.join(nom_root, "samples"))
    nom = _write_ecfg(c["ecfg"], nom_root, "rec_eval.json")
    other = _write_ecfg(c["ecfg"], c["out_root"], "rec_eval_other.json",
                        seed=c["ecfg_obj"].seed + 1)
    bad_ext = os.path.join(work, "extraction.json")
    with open(bad_ext, "w") as fh:
        json.dump({"electrodes_per_subset": 4}, fh)             # most keys missing
    # [fix round of 2026-11-28, D-180 (4)] a valid --extraction whose w_size_s is not the
    # simulated bank's sidecar w_size (the bin width the networks were trained on)
    import ev_rec_data
    index, _ = ev_rec_data.sim_index(c["view"])
    w_size = float(index.sidecar["w_size"])
    ext_ok = {"electrodes_per_subset": 9, "n_subsets": 9, "mfr_threshold_hz": 0.1,
              "w_size_s": w_size, "gaussian_window_s": 2.0 * w_size, "grid_width": 64,
              "index_base": 0, "pitch_um": 60.0, "fs_raw_hz": 10000.0,
              "detection": dict(S.DETECTION_DEFAULT)}
    assert S.validate_extraction(ext_ok) == [], S.validate_extraction(ext_ok)
    other_w = os.path.join(work, "extraction_other_w.json")
    with open(other_w, "w") as fh:
        json.dump(dict(ext_ok, w_size_s=w_size * (1.0 + 1e-9)), fh)
    cases = [("unknown ensemble", _export_argv("S-XX:A1", out, card=card), "no complete ensemble"),
             ("no sample", _export_argv(name, out, card=card, ecfg=nos), "not sampled"),
             ("no metrics", _export_argv(name, out, card=card, ecfg=nom), "no metrics"),
             ("a sample of other values", _export_argv(name, out, card=card, ecfg=other),
              "sampled with other values"),
             ("an invalid --extraction", _export_argv(name, out, card=card, extraction=bad_ext),
              "missing key extraction.n_subsets"),
             ("an --extraction of another w_size_s",
              _export_argv(name, out, card=card, extraction=other_w),
              "w_size_s is %r but the simulated bank's sidecar w_size is %r"
              % (w_size * (1.0 + 1e-9), w_size))]
    before = _listing(work)
    for what, argv, says in cases:
        code, text = _main(export.main, argv)
        assert code == export.EXIT_REFUSE, (what, code, text[-2000:])
        assert "REFUSE: " in text and says in text, (what, says, text[-2000:])
        assert _listing(work) == before, (what, sorted(set(_listing(work)) - set(before)))
    # the sidecar's value itself is taken (no --card: the template carries the extraction)
    ok_path = os.path.join(work, "extraction_ok.json")
    with open(ok_path, "w") as fh:
        json.dump(ext_ok, fh)
    okb = os.path.join(work, "ok_ext.sbibundle")
    code, text = _main(export.main, _export_argv(name, okb, extraction=ok_path))
    assert code == 0 and "BUNDLE EXPORT OK: " in text, text[-2000:]
    from sbitool import bundle
    assert bundle.read(okb).manifest["extraction"] == ext_ok
    os.remove(okb)
    # [D-182] where the sidecar's latent_spec holds n_e (the Sbi-extractor's banks; the
    # fixture's does not), electrodes_per_subset is compared with it: the comparison itself,
    # on a sidecar written here as export_windows/bank_export.py's contract_kwargs lays it out
    side = {"w_size": w_size, "latent_spec": {"kind": "campaign", "n_e": 9}}
    export._extraction_vs_bank(ext_ok, side, "x.json")
    export._extraction_vs_bank(dict(ext_ok, w_size_s=w_size * (1.0 + 1e-14)), side, "x.json")
    for key, skey, val in (("electrodes_per_subset", "latent_spec.n_e", 4),
                           ("w_size_s", "w_size", 2.0 * w_size)):
        try:
            export._extraction_vs_bank(dict(ext_ok, **{key: val}), side, "x.json")
            raise AssertionError("%s = %r against the sidecar's %s was taken" % (key, val, skey))
        except export.ExportRefused as exc:
            assert "%s is %r" % (key, val) in str(exc) and skey in str(exc), str(exc)
    # the smoothing width is in no sidecar: another value is the author's declaration, taken
    export._extraction_vs_bank(dict(ext_ok, gaussian_window_s=30.0 * w_size), side, "x.json")
    # top-level keys of those names are not the bank's contract and are not read
    export._extraction_vs_bank(dict(ext_ok, electrodes_per_subset=4),
                               {"w_size": w_size, "electrodes_per_subset": 9}, "x.json")
    # a sidecar without w_size is refused
    try:
        export._extraction_vs_bank(ext_ok, {"latent_spec": {"n_e": 9}}, "x.json")
        raise AssertionError("a sidecar without w_size was taken")
    except export.ExportRefused as exc:
        assert "no w_size" in str(exc), str(exc)
    # the same export twice: the members' digests are the same
    a, b = os.path.join(outd, "a.sbibundle"), os.path.join(outd, "b.sbibundle")
    for path in (a, b):
        code, text = _main(export.main, _export_argv(name, path, card=card))
        assert code == 0, text[-2000:]
    ma, mb = (json.loads(_zip_members(p)["manifest.json"]) for p in (a, b))
    assert [m["sha256"] for m in ma["members"]] == [m["sha256"] for m in mb["members"]]
    assert ma["files"] == mb["files"], (ma["files"], mb["files"])
    za, zb = _zip_members(a), _zip_members(b)
    assert all(za[n] == zb[n] for n in za if n != "manifest.json")
    assert sorted(os.listdir(outd)) == ["a.sbibundle", "b.sbibundle"], os.listdir(outd)


def test_t41e_card():
    """a complete card, a draft, the template (specs/SPEC.md block 41)."""
    from sbitool import bundle, card as CARD, export
    c = _ctx()
    name = ENSEMBLES[0]
    e, prov = _ensemble(name)
    # a complete card (the one of the export made once): card_complete true
    good = c["bundles"][name][0]
    b = bundle.read(good)
    assert b.manifest["card_complete"] is True, b.manifest["card_complete"]
    assert ", card complete)" in c["logs"][name]
    # one TODO: a draft
    work = _tmpdir("card")
    draft = os.path.join(work, "draft.sbibundle")
    todo = _card_for(e, prov, c["tag"], todo="licence")
    code, text = _main(export.main, _export_argv(name, draft, card=todo))
    assert code == 0 and " bytes, draft)" in text, text[-2000:]
    bd = bundle.read(draft)
    assert bd.manifest["card_complete"] is False
    assert bd.card == open(todo, encoding="utf-8").read()
    code, text = _main(bundle.main, [draft])
    assert code == 0 and " bytes, draft)" in text, text
    # a card that disagrees with the manifest (another arm) is refused, nothing written
    wrong = os.path.join(work, "wrong.md")
    with open(wrong, "w") as fh:
        fh.write(_card_text("A9", len(e.records), prov["axes"]["names"]))
    code, text = _main(export.main, _export_argv(name, os.path.join(work, "w.sbibundle"),
                                                 card=wrong))
    assert code == export.EXIT_REFUSE and "training.arm" in text, text[-2000:]
    assert not os.path.exists(os.path.join(work, "w.sbibundle"))
    # without --card: block 42's template, a draft
    try:
        CARD.template(b.manifest, [_read_json(r) for r in e.records], project=c["tag"])
    except NotImplementedError:
        raise unittest.SkipTest("the complete card and the draft PASS; the template part is "
                                "skipped: sbitool.card.template is a stub until block 42 is "
                                "merged (blocks 41 and 42 are written in parallel)")
    tpl = os.path.join(work, "template.sbibundle")
    code, text = _main(export.main, _export_argv(name, tpl))
    assert code == 0 and " bytes, draft)" in text, text[-2000:]
    bt = bundle.read(tpl)
    assert bt.manifest["card_complete"] is False
    header, _body = CARD.parse(bt.card)
    assert tuple(header) == S.CARD_KEYS, tuple(header)
    assert S.validate_card_header(header, bt.manifest) == [], S.validate_card_header(
        header, bt.manifest)
    # [2026-11-27, D-178] the template's n_networks: the networks of the training split,
    # counted here on the bank's donors with the record's own split (numpy only)
    import glob
    import numpy as np
    rec0 = _read_json(e.records[0])
    shards = sorted(glob.glob(rec0["config"]["sim_shards"]))
    donors = np.concatenate([np.load(q)["donor"] for q in shards])
    n_networks = header["training"]["n_networks"]
    assert isinstance(n_networks, int) and 1 <= n_networks < np.unique(donors).size, n_networks
    # every network has the same number of windows in the fixture, so the training
    # windows are n_networks times that number
    per = donors.size // np.unique(donors).size
    assert n_networks * per == header["training"]["n_windows"] == rec0["n_train"], (
        n_networks, per, rec0["n_train"])
    # the units: "1" on the unit box; --units replaces them; a wrong list is refused
    assert bt.manifest["parameters"]["units"] == ["1"] * bt.manifest["p"]
    uf = os.path.join(work, "units.json")
    want_u = ["Hz"] + ["1"] * (bt.manifest["p"] - 1)
    with open(uf, "w") as fh:
        json.dump(want_u, fh)
    wu = os.path.join(work, "units.sbibundle")
    code, text = _main(export.main, _export_argv(name, wu) + ["--units", uf])
    assert code == 0 and bundle.read(wu).manifest["parameters"]["units"] == want_u, text[-800:]
    with open(uf, "w") as fh:
        json.dump(want_u[:-1], fh)
    code, text = _main(export.main, _export_argv(name, os.path.join(work, "u2.sbibundle"))
                       + ["--units", uf])
    assert code == export.EXIT_REFUSE and "--units" in text, text[-800:]
    assert not os.path.exists(os.path.join(work, "u2.sbibundle"))


CHECKS = [("test_t41a_rebuild", test_t41a_rebuild),
          ("test_t41b_content", test_t41b_content),
          ("test_t41c_safety", test_t41c_safety),
          ("test_t41d_refusals", test_t41d_refusals),
          ("test_t41e_card", test_t41e_card)]


def main():
    rows, bad = [], 0
    t0 = time.time()
    only = set(sys.argv[1:])
    try:
        for name, fn in CHECKS:
            if only and name not in only:
                continue
            t1 = time.time()
            try:
                fn()
                rows.append((name, "PASS", "", time.time() - t1))
            except NotImplementedError as exc:
                bad += 1
                rows.append((name, "TODO", str(exc), time.time() - t1))
            except unittest.SkipTest as exc:
                rows.append((name, "SKIP", str(exc), time.time() - t1))
            except AssertionError as exc:
                bad += 1
                rows.append((name, "FAIL", str(exc)[:3000], time.time() - t1))
            except Exception:
                bad += 1
                rows.append((name, "ERROR", traceback.format_exc()[-3000:], time.time() - t1))
    finally:
        _cleanup()
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
