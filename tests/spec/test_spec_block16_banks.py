"""Spec-derived tests of block 16 (the synthetic banks; SPEC.md secs. 2.1,
2.4, block 16). Oracles O16.1-O16.8 of the t1-B report.

The expected builds are written here from block 16's rule (argument by
argument); the banks are built by the builder as a job runs it; the record
tool and the stage are driven through their public entries (the command
line; the Stage interface of block 1 with a recorder written here; the
sequencer under the fake PBS of sec. 3.3).
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

import t1b_helpers as H

H.pipeline_path()
import config as C                     # noqa: E402  (code under test)
from stages import synthetic as SY     # noqa: E402  (code under test)

H.add_paths(H.STAGE1)
import latent_bank as LB               # noqa: E402  (the contract's digest: the fixture's tool)


# ------------------------------------------------------------- the builds ---

def _opts(b):
    """Block 16's builder options from synthetic.builder, in the spec's order,
    floats as repr, plus block 25's --label-axes."""
    return ["--provider", b["provider"], "--fs", repr(float(b["fs"])),
            "--T-win", repr(float(b["T_win"])), "--n-neurons", str(b["n_neurons"]),
            "--wells-per-donor", str(b["wells_per_donor"]),
            "--donors-per-batch", str(b["donors_per_batch"]), "--n-classes", str(b["n_classes"]),
            "--tau-ov", repr(float(b["tau_ov"])), "--class-centres", b["class_centres"],
            "--centre-scale", repr(float(b["centre_scale"])),
            "--label-axes", ",".join(str(k) for k in b["label_axes"])]


def _expected_build(d, bank):
    syn = d["synthetic"]
    b = syn["builder"]
    wpd = b["wells_per_donor"]
    if bank == "sim":
        s = syn["sim"]
        K = s["n_theta"] * wpd // s["traces_per_shard"]
        argv = [["--arm", "S", "--prior", "box", "--shard-index", str(k),
                 "--n-traces", str(s["traces_per_shard"]), "--n-windows", str(s["n_windows"]),
                 "--seed", str(s["seed"])] + _opts(b) for k in range(K)]
        rows = [s["traces_per_shard"] * s["n_windows"]] * K
        arm = "S"
    else:
        r = [x for x in syn["real"] if x["name"] == bank][0]
        n_tr = b["n_classes"] * r["donors_per_class"] * wpd
        argv = [["--arm", "R", "--shard-index", "0", "--n-traces", str(n_tr),
                 "--donors-per-class", str(r["donors_per_class"]), "--n-windows", str(r["n_windows"]),
                 "--pi", repr(float(r["pi"])), "--gap-modes", ",".join(r["gap_modes"]),
                 "--seed", str(r["seed"])] + _opts(b)]
        rows = [n_tr * r["n_windows"]]
        arm = "R"
    return {"schema": "synth_bank_build/1", "bank": bank, "arm": arm, "n_shards": len(argv),
            "rows_per_shard": rows, "argv": argv}


@pytest.mark.parametrize("bank", ["sim", "R0", "R3"])
def test_o16_1_build_of_committed(bank):
    cfg = C.load_config(H.SYNTH, synthetic_ok=True)
    got = SY.build_of(cfg, bank)
    want = _expected_build(H.committed_synth(), bank)
    assert got == want
    assert json.loads(json.dumps(got)) == got                  # strings and integers only
    if bank == "sim":
        assert got["n_shards"] == 20 and set(got["rows_per_shard"]) == {2000}
    else:
        assert got["n_shards"] == 1 and got["rows_per_shard"] == [324]


# --------------------------------------------------- hand-built banks ---

@pytest.fixture(scope="module")
def project():
    """A reduced project (helpers.reduced_synth) whose three banks are built
    by hand from the expected builds, into a pristine copy kept aside."""
    root = H.tmpdir("t1b_b16_")
    path, d = H.reduced_synth(root)
    pristine = os.path.join(root, "pristine")
    for bank in ("sim", "R0", "R3"):
        bdir = os.path.join(pristine, d["dataset"] + ("_sim" if bank == "sim" else "_real_" + bank))
        for argv in _expected_build(d, bank)["argv"]:
            rc, out = H.run_builder(bdir, argv)
            assert rc == 0, out
    yield {"root": root, "path": path, "d": d, "pristine": pristine}
    H.rmtree(root)


def _fresh_banks(project, sub):
    """A copy of the pristine banks under <root>/<sub>/banks_root; the
    configuration pointed at it. Returns (cfg, banks_root)."""
    root = os.path.join(project["root"], sub)
    H.rmtree(root)
    banks = os.path.join(root, "banks_root")
    shutil.copytree(project["pristine"], banks)
    d = H.deepcopy(project["d"])
    for k in ("work_root", "runs_root", "tune_root"):
        d["paths"][k] = os.path.join(root, k)
    d["paths"]["banks_root"] = banks
    p = H.write_json(os.path.join(root, "synth.json"), d)
    return C.load_config(p, synthetic_ok=True), banks


class Rec(object):
    """A recorder with the interface the stages use (block 1's Submitter):
    log, check, fits, submit."""

    def __init__(self):
        self.lines, self.subs = [], []

    def log(self, t):
        self.lines.append(str(t))

    def check(self, n):
        return None

    def fits(self, n):
        return True

    def submit(self, line, role=None, log_path=None, **kw):
        self.subs.append((list(line), role))
        return "%d.rec" % len(self.subs)


def _tree_digest(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        out[os.path.basename(p)] = H.sha256_file(p) if os.path.isfile(p) else "dir"
    return out


def test_o16_3_built_shards_not_rebuilt(project):
    cfg, banks = _fresh_banks(project, "built")
    for bank in ("sim", "R0"):
        stage = SY.SynthBankStage(bank)
        st, rec = {"sub": {}, "jobs": []}, Rec()
        stage.start(cfg, st, rec)
        assert rec.subs == [], (bank, rec.subs)                # nothing built again
        assert stage.poll(cfg, st, rec) == "running"
        assert [r for _l, r in rec.subs] == ["record"]           # only the record job
        line = rec.subs[0][0]
        assert line[line.index("-N") + 1] == "sr_" + bank


def test_o16_4_foreign_argv_refused(project):
    cfg, banks = _fresh_banks(project, "foreign")
    bdir = os.path.join(banks, "synth_real_R0")
    js = os.path.join(bdir, "shard_0000.json")
    side = json.load(open(js))
    argv = side["provenance"]["argv"]
    argv[argv.index("--seed") + 1] = "999"
    side["provenance"]["argv"] = argv
    with open(js, "w") as fh:
        json.dump(side, fh)
    before = _tree_digest(bdir)
    with pytest.raises(RuntimeError) as e:
        SY.SynthBankStage("R0").start(cfg, {"sub": {}, "jobs": []}, Rec())
    assert "holds a bank of another build" in str(e.value)
    assert "--seed 999" in str(e.value)
    assert _tree_digest(bdir) == before


def test_o16_4_extra_index_refused(project):
    cfg, banks = _fresh_banks(project, "extra")
    bdir = os.path.join(banks, "synth_real_R0")
    for ext in ("npz", "json"):
        shutil.copy(os.path.join(bdir, "shard_0000." + ext), os.path.join(bdir, "shard_0001." + ext))
    before = _tree_digest(bdir)
    with pytest.raises(RuntimeError) as e:
        SY.SynthBankStage("R0").start(cfg, {"sub": {}, "jobs": []}, Rec())
    assert "shard_0001" in str(e.value)
    assert _tree_digest(bdir) == before


def test_o16_4_record_of_another_build_refused(project):
    cfg, banks = _fresh_banks(project, "otherrec")
    bdir = os.path.join(banks, "synth_real_R0")
    other = _expected_build(project["d"], "R0")
    other["rows_per_shard"] = [1]
    H.write_json(os.path.join(bdir, "BANK_RECORD.json"), {"build": other})
    before = _tree_digest(bdir)
    with pytest.raises(RuntimeError):
        SY.SynthBankStage("R0").start(cfg, {"sub": {}, "jobs": []}, Rec())
    assert _tree_digest(bdir) == before


def test_o16_4_npz_without_json_rebuilt(project):
    cfg, banks = _fresh_banks(project, "partial")
    os.remove(os.path.join(banks, "synth_sim", "shard_0001.json"))
    rec = Rec()
    SY.SynthBankStage("sim").start(cfg, {"sub": {}, "jobs": []}, rec)
    names = [l[l.index("-N") + 1] for l, _r in rec.subs]
    assert names == ["sb_sim_1"]


def test_o16_4_unreadable_json_set_aside(project):
    cfg, banks = _fresh_banks(project, "unreadable")
    js = os.path.join(banks, "synth_sim", "shard_0002.json")
    with open(js, "w") as fh:
        fh.write("{not json")
    rec = Rec()
    SY.SynthBankStage("sim").start(cfg, {"sub": {}, "jobs": []}, rec)
    assert os.path.isfile(js + ".unreadable1") and not os.path.exists(js)
    assert [l[l.index("-N") + 1] for l, _r in rec.subs] == ["sb_sim_2"]


# ------------------------------------------------------- the record tool ---

PASS_LINE = re.compile(r"^SYNTH BANK RECORD OK: (\S+) \((\d+) shard\(s\), (\d+) rows, digest ([0-9a-f]{16})\)$",
                       re.M)


def _tool(cfg, bank, banks_dir=None):
    stage = SY.SynthBankStage(bank)
    exp = stage.expect(cfg)
    bdir = banks_dir or exp["dir"]
    tmp = H.tmpdir("t1b_exp_")
    ep = H.write_json(os.path.join(tmp, "expect.json"), exp)
    out = os.path.join(bdir, "BANK_RECORD.json")
    r = subprocess.run([H.PYTHON, H.RECORD_TOOL, "--bank-dir", bdir, "--expect", ep, "--out", out],
                       cwd=H.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=600,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    H.rmtree(tmp)
    return r.returncode, r.stdout, out


def _assert_refused(rc, out, path):
    assert rc == 4, out
    assert re.search(r"^REFUSE: ", out, re.M), out
    assert not os.path.exists(path) and not os.path.exists(path + ".sha256")


@pytest.mark.parametrize("bank", ["sim", "R0", "R3"])
def test_o16_7_clean_bank_recorded(project, bank):
    cfg, banks = _fresh_banks(project, "clean_" + bank)
    rc, out, rpath = _tool(cfg, bank)
    assert rc == 0, out
    m = PASS_LINE.search(out)
    assert m and m.group(1) == rpath, out
    rec = json.load(open(rpath))
    want_keys = {"schema", "bank", "dir", "n_shards", "n_rows", "W", "d_theta", "contract_digest",
                 "arm", "latent_kind", "theta_withheld", "shards", "build", "created_utc", "tool"}
    assert set(rec) == want_keys
    build = _expected_build(project["d"], bank)
    assert rec["schema"] == "synth_bank_record/1" and rec["bank"] == bank
    assert rec["build"] == build
    assert rec["n_shards"] == build["n_shards"] == int(m.group(2))
    assert rec["n_rows"] == sum(build["rows_per_shard"]) == int(m.group(3))
    assert rec["W"] == 128 and rec["d_theta"] == 10
    assert rec["arm"] == build["arm"]
    assert rec["latent_kind"] == ("box" if bank == "sim" else None)
    assert rec["theta_withheld"] is (bank != "sim")
    assert rec["contract_digest"][:16] == m.group(4)
    for e in rec["shards"]:
        assert set(e) == {"file", "sha256", "bytes", "rows", "sidecar", "sidecar_sha256",
                          "contract_digest"}
        f = os.path.join(rec["dir"], e["file"])
        assert e["sha256"] == H.sha256_file(f) and e["bytes"] == os.path.getsize(f)
        assert e["sidecar_sha256"] == H.sha256_file(os.path.join(rec["dir"], e["sidecar"]))
        assert e["contract_digest"] == rec["contract_digest"]
    tok = open(rpath + ".sha256").read().split()[0]
    assert tok == hashlib.sha256(open(rpath, "rb").read()).hexdigest()
    # done(b): the record, its matching .sha256 and the configuration's build
    ok, why = SY.SynthBankStage(bank).is_done(cfg)
    assert ok, why


def _rewrite_npz(path, **changes):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(changes)
    tmp = path + ".t1b.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _rewrite_sidecar(path, mutate, redigest=True):
    side = json.load(open(path))
    mutate(side)
    if redigest:
        side["contract_digest"] = LB.contract_digest(side)
    with open(path, "w") as fh:
        json.dump(side, fh)


def test_o16_7_refuse_class_row_moved(project):
    """block 16's case: one row of class 0 moved to class 1 (here 7 and 9
    rows where 8 are expected: 2 donors x 2 wells x 2 windows)."""
    cfg, banks = _fresh_banks(project, "moved")
    npz = os.path.join(banks, "synth_real_R0", "shard_0000.npz")
    with np.load(npz) as z:
        cls = z["cls"].copy()
    assert np.bincount(cls).tolist() == [8, 8, 8]
    cls[np.flatnonzero(cls == 0)[0]] = 1
    _rewrite_npz(npz, cls=cls)
    rc, out, rpath = _tool(cfg, "R0")
    _assert_refused(rc, out, rpath)
    assert "7 rows of class 0" in out and "9 rows of class 1" in out


def test_o16_7_refuse_extra_index(project):
    cfg, banks = _fresh_banks(project, "extra_tool")
    bdir = os.path.join(banks, "synth_sim")
    for ext in ("npz", "json"):
        shutil.copy(os.path.join(bdir, "shard_0000." + ext), os.path.join(bdir, "shard_0003." + ext))
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)


@pytest.mark.parametrize("where", ["crc", "deflate"])
def test_o16_7_refuse_flipped_bytes(project, where):
    cfg, banks = _fresh_banks(project, "flip_" + where)
    npz = os.path.join(banks, "synth_sim", "shard_0001.npz")
    blob = bytearray(open(npz, "rb").read())
    # a byte in the middle of the archive (inside the compressed x member):
    # either the deflate stream breaks or the CRC-32 no longer matches
    i = len(blob) // 2 if where == "crc" else len(blob) // 3
    blob[i] ^= 0xFF
    open(npz, "wb").write(bytes(blob))
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)


def test_o16_7_refuse_two_digests(project):
    cfg, banks = _fresh_banks(project, "twodig")
    js = os.path.join(banks, "synth_sim", "shard_0002.json")
    _rewrite_sidecar(js, lambda s: s["nuisance_spec"].update(t1b="another contract"))
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)
    assert "contract digests" in out


def test_o16_7_refuse_stale_digest(project):
    cfg, banks = _fresh_banks(project, "staledig")
    js = os.path.join(banks, "synth_sim", "shard_0000.json")
    _rewrite_sidecar(js, lambda s: s.update(T_win=99.0), redigest=False)
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)


@pytest.mark.parametrize("bank,mutate", [
    ("R0", lambda s: s.update(theta_withheld=False)),
    ("sim", lambda s: s.update(theta_withheld=True)),
    ("sim", lambda s: s["latent_spec"].pop("kind")),
    ("R0", lambda s: s["latent_spec"].update(kind="box")),
    ("R0", lambda s: s.update(arm="S")),
], ids=["R_not_withheld", "S_withheld", "S_no_box_kind", "R_box_kind", "R_arm_S"])
def test_o16_7_refuse_sidecar_fields(project, bank, mutate):
    cfg, banks = _fresh_banks(project, "side_%s_%d" % (bank, abs(hash(str(mutate))) % 1000))
    bdir = os.path.join(banks, "synth_sim" if bank == "sim" else "synth_real_" + bank)
    _rewrite_sidecar(os.path.join(bdir, "shard_0000.json"), mutate)
    rc, out, rpath = _tool(cfg, bank)
    _assert_refused(rc, out, rpath)


def test_o16_7_refuse_argv(project):
    cfg, banks = _fresh_banks(project, "argv_tool")
    js = os.path.join(banks, "synth_real_R3", "shard_0000.json")

    def mut(s):
        a = s["provenance"]["argv"]
        a[a.index("--pi") + 1] = "0.2"
    _rewrite_sidecar(js, mut)
    rc, out, rpath = _tool(cfg, "R3")
    _assert_refused(rc, out, rpath)


def test_o16_7_refuse_wells_per_donor(project):
    cfg, banks = _fresh_banks(project, "wells")
    npz = os.path.join(banks, "synth_sim", "shard_0000.npz")
    with np.load(npz) as z:
        well = z["well"].copy()
        donor = z["donor"].copy()
    first = np.flatnonzero(donor == donor[0])
    well[first] = well[first[0]]                     # the first donor now has one well
    _rewrite_npz(npz, well=well)
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)


def test_o16_7_refuse_rows_and_width(project):
    cfg, banks = _fresh_banks(project, "rows")
    npz = os.path.join(banks, "synth_sim", "shard_0001.npz")
    with np.load(npz) as z:
        arrays = {k: z[k][:-1] for k in z.files}
    _rewrite_npz(npz, **arrays)
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)
    cfg2, banks2 = _fresh_banks(project, "width")
    exp_cfg = C.load_config(cfg2.path, synthetic_ok=True)
    exp_cfg.data["search"]["shape_anchors"]["d_theta"] = 9        # theta's width disagrees
    rc, out, rpath = _tool(exp_cfg, "sim")
    _assert_refused(rc, out, rpath)


def test_o16_7_refuse_class_on_arm_S(project):
    cfg, banks = _fresh_banks(project, "scls")
    npz = os.path.join(banks, "synth_sim", "shard_0000.npz")
    with np.load(npz) as z:
        cls = z["cls"].copy()
    cls[0] = 0
    _rewrite_npz(npz, cls=cls)
    rc, out, rpath = _tool(cfg, "sim")
    _assert_refused(rc, out, rpath)


def test_o16_8_done_needs_the_build_and_the_digest(project):
    cfg, banks = _fresh_banks(project, "done")
    rc, out, rpath = _tool(cfg, "R0")
    assert rc == 0, out
    stage = SY.SynthBankStage("R0")
    assert stage.is_done(cfg)[0]
    rec = json.load(open(rpath))
    rec["build"]["rows_per_shard"] = [999]
    blob = json.dumps(rec)
    open(rpath, "w").write(blob)
    open(rpath + ".sha256", "w").write(hashlib.sha256(blob.encode()).hexdigest() + "  x\n")
    assert not stage.is_done(cfg)[0]                     # another build
    open(rpath + ".sha256", "w").write("0" * 64 + "\n")
    assert not stage.is_done(cfg)[0]                     # sidecar mismatch
