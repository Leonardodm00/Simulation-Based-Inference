"""Block 24 -- the purge of a synthetic project, with the archive (SPEC.md
sec. 5, block 24; D-132). Spec-derived tests of the integration tester, run
t1-D of 2026-11-13. Oracles O24.1-O24.15 of the report.

Every expected value is computed here from the spec with the standard
library (os.walk, hashlib, tarfile, gzip's magic bytes) or read from the
tree the test made; the tool is run as the user runs it (a subprocess under
the fake PBS), except where a failure must be injected in-process.

    python3 -m pytest tests/spec/test_spec_block24_purge.py -v
"""

import contextlib
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

import pytest

import spec_helpers as H

H.add_paths(H.PIPE, H.TOOLS, H.PTESTS)

import config as C                  # noqa: E402
import state as ST                  # noqa: E402

KINDS5 = "evaluation,best_fits,searches,banks,state"
SEARCHES = [{"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False},
            {"name": "S-A2-R0", "campaign": "S-A2-only", "real": "R0", "alt_best_fit": False},
            {"name": "S-A5-R3", "campaign": "S-A5-only", "real": "R3", "alt_best_fit": False}]
BIG = 30 * 1000 * 1000      # a sparse file's apparent size, so the GB columns read 0.03


# ------------------------------------------------------------------ fixture ---

def _project(root, eval_out=None):
    """A reduced synthetic project (test_smoke_synthetic._small) with three
    searches on the v4 campaigns, its trees, an evaluation configuration at
    the default path and decoys. Returns a dict of paths."""
    import test_smoke_synthetic as TS
    path, _d = TS._small(root, searches=[dict(q) for q in SEARCHES])
    cfg = C.load_config(path, synthetic_ok=True)
    p = cfg.data["paths"]
    ev = eval_out or os.path.join(p["runs_root"], "evaluation")
    with open(H.SYNTH_EVAL) as fh:
        e = json.load(fh)
    e["out_root"] = ev
    e["cal_glob"] = os.path.join(ev, "calbank", "shard_*.npz")
    epath = os.path.join(root, "synth_eval.json")          # <config folder>/<dataset>_eval.json
    H.write(epath, json.dumps(e, indent=1, sort_keys=True))
    w = H.write
    # the evaluation
    w(os.path.join(ev, "samples", "S-A1_A1_sim.npz"), os.urandom(900))
    w(os.path.join(ev, "samples", "S-A1_A1_sample.json"), '{"s": 1}')
    w(os.path.join(ev, "calbank", "shard_0000.npz"), os.urandom(800))
    w(os.path.join(ev, "calbank", "shard_0000.json"), '{"rows": 4}')
    w(os.path.join(ev, "metrics", "METRICS.json"), '{"ensembles": {}}')
    w(os.path.join(ev, "metrics", "S-A1_A1_R0.npz"), os.urandom(321))
    w(os.path.join(ev, "metrics", "samples", "deep.json"), '{"a nested samples/": "archived"}')
    w(os.path.join(ev, "figures", "FIGURES.md"), "# figures\n")
    w(os.path.join(ev, "figures", "ensembles", "S-A1_A1", "cal_coverage.png"), os.urandom(250))
    big = os.path.join(ev, "figures", "big.pdf")
    w(big, b"")
    os.truncate(big, BIG)
    w(os.path.join(ev, "data", "DATA_VIEW.json"), '{"n": 1}')
    w(os.path.join(ev, "REPORT_COMPLETE.json"), '{"done": true}')
    w(os.path.join(ev, "jobs", "logs", "report.log"), "EVAL REPORT OK\n")
    w(os.path.join(ev, "jobs", "ledger.json"),
      json.dumps({"report": {"job_id": "999990.cluster"},
                  "sample:S-A1:A1": {"job_id": "999991.cluster"}}))
    # the best fits, the searches, the banks
    for q in SEARCHES:
        w(os.path.join(p["runs_root"], q["name"], "synth_best", "member_seed1", "m.json"), '{"nll": 1}')
        c = os.path.join(p["tune_root"], q["name"], q["campaign"])
        w(os.path.join(c, "trials", "t0.json"), '{"status": "ok"}')
        w(os.path.join(c, "runs", "t0", "ckpt.pt"), os.urandom(400))
    w(os.path.join(p["runs_root"], "S-A2-R0", "synth_best_alt", "member_seed1", "m.json"), '{"nll": 3}')
    w(os.path.join(p["tune_root"], "S-A1", "S-A1_set_aside_2026-11-01T000000Z", "x.json"), "{}")
    banks = []
    for b in ("sim", "R0", "R3"):
        bd = os.path.join(p["banks_root"], "synth_sim" if b == "sim" else "synth_real_" + b)
        w(os.path.join(bd, "shard_0000.npz"), os.urandom(700))
        w(os.path.join(bd, "BANK_RECORD.json"), '{"bank": "%s"}' % b)
        banks.append(bd)
    bigbank = os.path.join(p["banks_root"], "synth_sim", "shard_0001.npz")
    w(bigbank, b"")
    os.truncate(bigbank, BIG)
    # the work root
    wr = p["work_root"]
    st = ST.State.new(wr, "synth", path, cfg.sha256)
    st.save()
    st.save()                                               # state.json and its .prev
    w(os.path.join(wr, "pipeline.log"), "tick\n")
    w(os.path.join(wr, "driver", "driver.1.argv"), "drive\n")
    w(os.path.join(wr, "bank_jobs", "sim_bank", "sb_sim_0.log"), "ok\n")
    w(os.path.join(wr, "avoid_nodes.txt"), "# avoid\n")
    w(os.path.join(wr, "purge", "PURGE_2026-11-01T000000Z.json"), '{"schema": "purge_outputs/1"}')
    # decoys: never targets
    decoys = [os.path.join(p["banks_root"], "giulia_sim", "shard_0000.npz"),
              os.path.join(p["banks_root"], "synth_real_R9", "shard_0000.npz"),   # not configured
              os.path.join(p["banks_root"], "notes.txt"),
              os.path.join(p["runs_root"], "S-OLD", "m.json"),
              os.path.join(p["tune_root"], "S-OLD", "x.json")]
    for dpath in decoys:
        w(dpath, "decoy\n")
    targets = ([("evaluation", ev)]
               + [("best_fits", os.path.join(p["runs_root"], q["name"])) for q in SEARCHES]
               + [("searches", os.path.join(p["tune_root"], q["name"])) for q in SEARCHES]
               + [("banks", b) for b in banks]
               + [("state", os.path.join(wr, n)) for n in sorted(os.listdir(wr)) if n != "purge"])
    return {"root": root, "cfg": path, "eval": epath, "ev": ev, "wr": wr, "p": p,
            "targets": targets, "decoys": decoys}


def _members(ev):
    """[(arcname, size, mtime)] of the archive by the spec: every regular
    file under out_root but its TOP-LEVEL samples/ and calbank/, names
    relative to out_root's parent, sorted."""
    base = os.path.dirname(os.path.normpath(ev))
    out = []
    for d, subs, files in os.walk(ev):
        if os.path.normpath(d) == os.path.normpath(ev):
            subs[:] = [s for s in subs if s not in ("samples", "calbank")]
        for f in files:
            q = os.path.join(d, f)
            st = os.lstat(q)
            out.append((os.path.relpath(q, base), st.st_size, int(st.st_mtime)))
    return sorted(out)


def _purge(fx, args, env=None):
    env = env or H.fake_env(fx["root"])
    return H.run([H.PYTHON, H.PURGE, "--config", fx["cfg"]] + args, env)


def _code(out):
    m = re.search(r"^CODE ([0-9a-f]+) ", out, re.M)
    assert m, out
    return m.group(1)


def _listed(out):
    """[(kind, path, files)] of the listing's DELETE lines, in order."""
    return [(m.group(1), m.group(2), int(m.group(3)))
            for m in re.finditer(r"^  would DELETE \[(\w+)\] (.+?): (\d+) file\(s\), ", out, re.M)]


@pytest.fixture
def fx():
    root = tempfile.mkdtemp(prefix="t1d_b24_")
    try:
        yield _project(root)
    finally:
        _qdel_all(root)
        shutil.rmtree(root, ignore_errors=True)


def _qdel_all(root):
    spool = os.path.join(root, "spool")
    if not os.path.isdir(spool):
        return
    env = H.fake_env(root)
    rc, out = H.run(["qselect"], env)
    for j in out.split():
        H.run(["qdel", j], env)


def _qsub(fx, name, seconds):
    env = H.fake_env(fx["root"])
    sdir = os.path.join(fx["root"], "spool", "scripts")
    os.makedirs(sdir, exist_ok=True)
    script = os.path.join(sdir, "%s.sh" % name)
    with open(script, "w") as fh:
        fh.write("#!/bin/bash\nsleep %s\n" % seconds)
    rc, out = H.run(["qsub", "-N", name, "-q", "cpu", "-l", "select=1:ncpus=1", "-o",
                     script + ".log", script], env)
    assert rc == 0, out
    return out.split()[0]


def _live(fx, job):
    rc, out = H.run(["qstat", "-x", "-f", job], H.fake_env(fx["root"]))
    m = re.search(r"job_state = (\w)", out)
    return bool(m) and m.group(1) in ("Q", "R", "H", "W")


# ------------------------------------------------------------------- checks ---

def test_o24_3_order_whatever_the_kinds_order_and_counts(fx):
    """O24.3, O24.4: the documented order whatever the order of --what; the
    file counts are os.walk's; nothing written by a listing."""
    before = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")])
    arch = os.path.join(fx["root"], "a.tar.gz")
    rc, out = _purge(fx, ["--what", "state,banks,searches,best_fits,evaluation", "--archive", arch])
    assert rc == 0, out
    got = _listed(out)
    want = [(k, t, H.walk_counts(t)[0]) for k, t in fx["targets"]]
    assert got == want, "\n".join(map(str, got)) + "\n--- want\n" + "\n".join(map(str, want))
    assert "  kept: %s" % os.path.join(fx["wr"], "purge") in out, out
    assert H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")]) == before
    # the same CODE whatever the order of the kinds (the same targets in the same order)
    rc2, out2 = _purge(fx, ["--what", KINDS5, "--archive", arch])
    assert rc2 == 0 and _code(out2) == _code(out), (out, out2)


def test_o24_5_archive_listing(fx):
    """O24.5: the archive's member count and bytes, and the two skipped
    subtrees' counts and bytes, as os.walk gives them."""
    arch = os.path.join(fx["root"], "a.tar.gz")
    rc, out = _purge(fx, ["--what", "evaluation", "--archive", arch])
    assert rc == 0, out
    mem = _members(fx["ev"])
    assert any(a.endswith(os.path.join("metrics", "samples", "deep.json")) for a, _s, _t in mem)
    assert not any(a.split(os.sep)[1] in ("samples", "calbank") for a, _s, _t in mem)
    b = sum(s for _a, s, _t in mem)
    assert "  would ARCHIVE %d file(s), %.2f GB uncompressed, into %s" % (len(mem), b / 1e9, arch) \
        in out, out
    for s in ("samples", "calbank"):
        n_s, b_s = H.walk_counts(os.path.join(fx["ev"], s))
        assert "    not archived: %s/ (%d file(s), %.2f GB)" % (s, n_s, b_s / 1e9) in out, out


def test_o24_7_8_confirm_archive_and_record(fx):
    """O24.7, O24.8: the archive (gzip, names, sizes, bytes, .sha256), the
    pass line, the targets gone, the decoys and the older record unchanged,
    the new record; then a listing that finds nothing and asks no archive."""
    arch = os.path.join(fx["root"], "arch", "eval.tar.gz")
    os.makedirs(os.path.dirname(arch))
    mem = _members(fx["ev"])
    base = os.path.dirname(fx["ev"])
    content = {}
    for a, _s, _t in mem:
        with open(os.path.join(base, a), "rb") as fh:
            content[a] = hashlib.sha256(fh.read()).hexdigest()
    counts = [H.walk_counts(t) for _k, t in fx["targets"]]
    keep = [os.path.join(fx["wr"], "purge")] + fx["decoys"] + [fx["cfg"], fx["eval"]]
    before = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")])
    rc, out = _purge(fx, ["--what", KINDS5, "--archive", arch])
    rc, out = _purge(fx, ["--what", KINDS5, "--archive", arch, "--confirm", _code(out)])
    assert rc == 0, out
    with open(arch, "rb") as fh:
        raw = fh.read()
    assert raw[:2] == b"\x1f\x8b", "not a gzip file"
    sha = hashlib.sha256(raw).hexdigest()
    with tarfile.open(arch, "r:gz") as tf:
        infos = tf.getmembers()
        got = [(ti.name, ti.size) for ti in infos]
        data = dict((ti.name, hashlib.sha256(tf.extractfile(ti).read()).hexdigest()) for ti in infos)
    assert got == [(a, s) for a, s, _t in mem], (got, mem)
    assert all(n.startswith(os.path.basename(fx["ev"]) + "/") for n, _s in got)
    assert data == content
    with open(arch + ".sha256") as fh:
        assert fh.read() == "%s  %s\n" % (sha, os.path.basename(arch))
    rc_c, out_c = H.run(["sha256sum", "-c", os.path.basename(arch) + ".sha256"], dict(os.environ),
                        cwd=os.path.dirname(arch))
    assert rc_c == 0 and ": OK" in out_c, out_c
    assert not os.path.lexists(arch + ".tmp")
    n = sum(c[0] for c in counts)
    b = sum(c[1] for c in counts)
    line = ("PURGE DONE: %d target(s) deleted, %d file(s), %.2f GB; archive %s (%d file(s), sha256 %s)"
            "; the work root keeps purge/ only: the next submission starts a new run"
            % (len(fx["targets"]), n, b / 1e9, arch, len(mem), sha[:16]))
    assert line in out, (line, out)
    for _k, t in fx["targets"]:
        assert not os.path.lexists(t), t
    after = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")])
    for k, v in before.items():
        full = os.path.join(fx["root"], k)
        if any(full == x or full.startswith(x + os.sep) for x in keep):
            assert after.get(k) == v, "changed: %s" % k
    assert sorted(os.listdir(fx["wr"])) == ["purge"]
    recs = sorted(glob.glob(os.path.join(fx["wr"], "purge", "PURGE_*.json")))
    assert len(recs) == 2, recs
    with open(recs[-1]) as fh:
        rec = json.load(fh)
    assert rec["schema"] == "purge_outputs/1" and rec["status"] == "done", rec
    a = rec["archive"]
    assert (a["path"], a["members"], a["bytes"], a["result"], a["sha256"]) == \
        (arch, len(mem), sum(s for _a, s, _t in mem), "written", sha), a
    # a second listing: every target absent, no archive asked (exit 0 without --archive)
    rc2, out2 = _purge(fx, ["--what", KINDS5])
    assert rc2 == 0 and "would DELETE" not in out2 and "would ARCHIVE" not in out2, out2


def test_o24_7_no_archive_pass_line(fx):
    """O24.7: --no-archive deletes the evaluation without packing it; the
    pass line's archive clause is `no archive`; without `state` no suffix."""
    rc, out = _purge(fx, ["--what", "evaluation,banks", "--no-archive"])
    rc, out = _purge(fx, ["--what", "evaluation,banks", "--no-archive", "--confirm", _code(out)])
    assert rc == 0, out
    m = re.search(r"^PURGE DONE: (\d+) target\(s\) deleted, (\d+) file\(s\), ([0-9.]+) GB; (.*)$",
                  out, re.M)
    assert m and m.group(4) == "no archive", out
    assert int(m.group(1)) == 4, out
    assert not os.path.exists(fx["ev"])
    assert os.path.exists(os.path.join(fx["wr"], "state.json"))


def test_o24_6_argument_refusals_exit_2_write_nothing(fx):
    """O24.1, O24.2, O24.6: every exit-2 case of the spec, nothing written."""
    p = fx["p"]
    adir = os.path.join(fx["root"], "arch")
    os.makedirs(adir)
    exists = os.path.join(adir, "old.tar.gz")
    H.write(exists, "x")
    cases = [
        ["--what", KINDS5],                                                   # no --archive
        ["--what", KINDS5, "--archive", exists],                              # it exists
        ["--what", KINDS5, "--archive", os.path.join(fx["ev"], "a.tar.gz")],  # under out_root
        ["--what", KINDS5, "--archive", os.path.join(p["tune_root"], "S-A1", "a.tar.gz")],
        ["--what", KINDS5, "--archive", os.path.join(fx["wr"], "driver", "a.tar.gz")],
        ["--what", KINDS5, "--archive", os.path.join(adir, "nope", "a.tar.gz")],
        ["--what", "banks", "--archive", os.path.join(adir, "a.tar.gz")],     # without evaluation
        ["--what", KINDS5, "--archive", os.path.join(adir, "a.tar.gz"), "--no-archive"],
        ["--what", "search,best_fit"],                                        # block 10's kinds
        ["--what", ""],
        ["--what", "banks,sideways"],
    ]
    before = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")])
    for args in cases:
        rc, out = _purge(fx, args)
        assert rc == 2, (args, rc, out)
        assert H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")]) == before, args


def test_o24_1_recorded_configuration_refuses_the_new_options():
    """O24.1: the five kinds, --eval-config, --archive and --no-archive on a
    recorded configuration: exit 2."""
    import test_smoke_purge as TP
    root = tempfile.mkdtemp(prefix="t1d_b24_rec_")
    try:
        path, _cfg, _t, _sha = TP._fixture(root)
        env = H.fake_env(root)
        for args in (["--what", "banks"], ["--what", "state"],
                     ["--what", "search", "--eval-config", path],
                     ["--what", "search", "--archive", os.path.join(root, "a.tar.gz")],
                     ["--what", "search", "--no-archive"]):
            rc, out = H.run([H.PYTHON, H.PURGE, "--config", path] + args, env)
            assert rc == 2, (args, out)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_o24_2_default_eval_config_must_load_when_present(fx):
    """O24.2: the default evaluation configuration exists but does not load
    -> exit 2 even when `evaluation` is not named."""
    H.write(fx["eval"], "{ not json")
    rc, out = _purge(fx, ["--what", "banks"])
    assert rc == 2, out


def test_o24_2_ledger_checked_whatever_the_kinds(fx):
    """O24.2, O24.11: a live job of the evaluation's ledger refuses a purge
    that names neither the evaluation nor anything under it."""
    jid = _qsub(fx, "ev_s0a", 60)
    led = os.path.join(fx["ev"], "jobs", "ledger.json")
    H.write(led, json.dumps({"sample:S-A1:A1": {"job_id": jid}}))
    before = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")])
    rc, out = _purge(fx, ["--what", "banks"])
    assert rc == 4 and jid.split(".")[0] in out, out
    rc, out = _purge(fx, ["--what", "banks", "--confirm", "00000000"])
    assert rc == 4, out
    assert H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")]) == before


def test_o24_6_members_folders_hold_a_fifo_or_a_linked_folder(fx):
    """O24.6: a non-regular file, or a symbolic link to a folder, among the
    members' folders: exit 4, nothing deleted."""
    arch = os.path.join(fx["root"], "a.tar.gz")
    fifo = os.path.join(fx["ev"], "metrics", "pipe")
    os.mkfifo(fifo)
    rc, out = _purge(fx, ["--what", "evaluation", "--archive", arch])
    assert rc == 4, out
    os.remove(fifo)
    link = os.path.join(fx["ev"], "figures", "elsewhere")
    os.symlink(fx["wr"], link)
    rc, out = _purge(fx, ["--what", "evaluation", "--archive", arch])
    assert rc == 4, out
    os.remove(link)
    assert not os.path.exists(arch)


def test_o24_11_refusals_exit_4(fx):
    """O24.11: the lock; a target that is a symbolic link (a bank folder, a
    work-root entry, the evaluation); a wrong CODE -- each exit 4, nothing
    deleted, no record."""
    p = fx["p"]
    before = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")])
    lock = os.path.join(fx["wr"], "pipeline.lock")
    H.write(lock, "pid:1 elsewhere-host 2026-11-13T00:00:00Z\n")
    rc, out = _purge(fx, ["--what", "banks"])
    assert rc == 4, out
    os.remove(lock)
    rc, out = _purge(fx, ["--what", "banks", "--confirm", "0badc0de"])
    assert rc == 4 and re.search(r"^CODE [0-9a-f]{8} ", out, re.M), out
    assert H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool")]) == before
    # a bank folder that is a link
    real = os.path.join(p["banks_root"], "synth_real_R3")
    moved = os.path.join(fx["root"], "elsewhere_R3")
    os.rename(real, moved)
    os.symlink(moved, real)
    rc, out = _purge(fx, ["--what", "banks"])
    assert rc == 4, out
    os.remove(real)
    os.rename(moved, real)
    # a work-root entry that is a link
    ln = os.path.join(fx["wr"], "to_banks")
    os.symlink(p["banks_root"], ln)
    rc, out = _purge(fx, ["--what", "state"])
    assert rc == 4, out
    os.remove(ln)
    assert not glob.glob(os.path.join(fx["wr"], "purge", "PURGE_2026-11-1[3-9]*.json"))


def test_o24_10_code_archive_part(fx):
    """O24.10: with an archive asked the CODE changes with the archive's path
    and with a member's mtime that leaves block 10's terms (count, bytes,
    newest time) as they were; without an archive it does not change."""
    a1 = os.path.join(fx["root"], "a1.tar.gz")
    a2 = os.path.join(fx["root"], "a2.tar.gz")
    c1 = _code(_purge(fx, ["--what", "evaluation", "--archive", a1])[1])
    c2 = _code(_purge(fx, ["--what", "evaluation", "--archive", a2])[1])
    n1 = _code(_purge(fx, ["--what", "evaluation", "--no-archive"])[1])
    assert c1 != c2, "the archive's path is not in the CODE"
    assert n1 not in (c1, c2)
    f = os.path.join(fx["ev"], "metrics", "METRICS.json")
    t = os.lstat(f).st_mtime
    os.utime(f, (t - 100, t - 100))        # older: the newest time of the target is unchanged
    c1b = _code(_purge(fx, ["--what", "evaluation", "--archive", a1])[1])
    n1b = _code(_purge(fx, ["--what", "evaluation", "--no-archive"])[1])
    assert c1b != c1, "a member's mtime is not in the archive's part of the CODE"
    assert n1b == n1, "block 10's canonical text holds more than count, bytes and newest time"
    b0 = _code(_purge(fx, ["--what", "banks"])[1])
    H.write(os.path.join(fx["p"]["banks_root"], "synth_sim", "extra.json"), "{}")
    assert _code(_purge(fx, ["--what", "banks"])[1]) != b0, "a file added to a bank left the CODE"


def test_o24_14_block10_code_unchanged_on_a_recorded_project():
    """O24.10, O24.14: on a recorded project the CODE equals the one block
    10's tool (at 4d10214, before block 24) prints for the same tree."""
    import test_smoke_purge as TP
    root = tempfile.mkdtemp(prefix="t1d_b24_b10_")
    old = H.tree_at(H.BEFORE_B24, "hpc/pipeline")
    try:
        path, _cfg, _t, _sha = TP._fixture(root)
        env = H.fake_env(root)
        for what in ("search,best_fit,sim_bank", "sim_bank", "best_fit,search"):
            rc_n, out_n = H.run([H.PYTHON, H.PURGE, "--config", path, "--what", what], env)
            rc_o, out_o = H.run([H.PYTHON, os.path.join(old, "hpc", "pipeline", "tools",
                                                        "purge_outputs.py"),
                                 "--config", path, "--what", what], env)
            assert rc_n == 0 and rc_o == 0, (out_n, out_o)
            assert _code(out_n) == _code(out_o), (what, out_n, out_o)
            assert out_n == out_o, (out_n, out_o)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)


def _inproc(fx, args):
    """The tool in this process (to inject a failure); (exit, output)."""
    import purge_outputs as PO
    env = H.fake_env(fx["root"])
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = PO.main(["--config", fx["cfg"]] + args)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return rc, buf.getvalue()


def test_o24_9_archive_compared_with_the_listing(fx):
    """O24.7, O24.9: a member that grows between the listing and the archive
    is caught by the re-read (sizes compared with the listing): PURGE ARCHIVE
    FAILED, exit 5, no .tmp, no archive, nothing deleted."""
    import purge_outputs as PO
    arch = os.path.join(fx["root"], "a.tar.gz")
    rc, out = _purge(fx, ["--what", KINDS5, "--archive", arch])
    code = _code(out)
    before = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool"),
                                          os.path.join(fx["wr"], "purge")])
    real = PO.write_archive
    victim = os.path.join(fx["ev"], "data", "DATA_VIEW.json")

    def grow_then_write(archive):
        with open(victim, "ab") as fh:
            fh.write(b" ")
        return real(archive)
    PO.write_archive = grow_then_write
    try:
        rc, out = _inproc(fx, ["--what", KINDS5, "--archive", arch, "--confirm", code])
    finally:
        PO.write_archive = real
    assert rc == 5 and "PURGE ARCHIVE FAILED" in out and "nothing deleted" in out, out
    assert not os.path.lexists(arch) and not os.path.lexists(arch + ".tmp")
    after = H.snapshot(fx["root"], skip=[os.path.join(fx["root"], "spool"),
                                         os.path.join(fx["wr"], "purge")])
    changed = [k for k in before if before[k] != after.get(k)]
    assert changed == [os.path.relpath(victim, fx["root"])], changed
    recs = sorted(glob.glob(os.path.join(fx["wr"], "purge", "PURGE_*.json")))
    with open(recs[-1]) as fh:
        rec = json.load(fh)
    assert rec["status"] == "archive failed" and rec["archive"]["result"].startswith("failed"), rec


def test_o24_8_a_deletion_that_fails_is_incomplete(fx):
    """O24.8 (block 10's rule kept): a target whose removal fails -> FAILED
    line, the others deleted, PURGE INCOMPLETE, exit 5, the record failed."""
    import purge_outputs as PO
    rc, out = _purge(fx, ["--what", "searches,banks"])
    code = _code(out)
    real = PO._remove
    bad = os.path.join(fx["p"]["tune_root"], "S-A2-R0")

    def remove(path):
        if path == bad:
            raise OSError("Device or resource busy (the test)")
        return real(path)
    PO._remove = remove
    try:
        rc, out = _inproc(fx, ["--what", "searches,banks", "--confirm", code])
    finally:
        PO._remove = real
    assert rc == 5 and "PURGE INCOMPLETE" in out and "FAILED [searches] %s" % bad in out, out
    assert os.path.isdir(bad)
    assert not os.path.exists(os.path.join(fx["p"]["tune_root"], "S-A1"))
    assert not os.path.exists(os.path.join(fx["p"]["banks_root"], "synth_sim"))
    recs = sorted(glob.glob(os.path.join(fx["wr"], "purge", "PURGE_*.json")))
    with open(recs[-1]) as fh:
        assert json.load(fh)["status"] == "failed"


def test_o24_3_archive_in_the_work_root_survives_state(fx):
    """O24.3, O24.6 edge: an archive placed directly in the work root is not
    under a target (the work root is not one; its entries are), so it is
    accepted with `state` named -- and must survive the purge it was made
    by: the deletion follows the listing, made before the archive."""
    arch = os.path.join(fx["wr"], "eval_archive.tar.gz")
    rc, out = _purge(fx, ["--what", KINDS5, "--archive", arch])
    if rc == 2:
        pytest.skip("refused up front (exit 2): conservative, recorded in the report")
    rc, out = _purge(fx, ["--what", KINDS5, "--archive", arch, "--confirm", _code(out)])
    assert rc == 0, out
    assert os.path.isfile(arch) and os.path.isfile(arch + ".sha256"), os.listdir(fx["wr"])


def test_o24_3_overlap_out_root_inside_the_work_root():
    """Edge (the spec is silent): the evaluation's out_root inside the work
    root, `evaluation` and `state` named. Every byte goes, but is the run
    reported complete? Recorded as found."""
    root = tempfile.mkdtemp(prefix="t1d_b24_ov_")
    try:
        fx = _project(root, eval_out=None)
        ev_in = os.path.join(fx["wr"], "evaluation")
        shutil.move(fx["ev"], ev_in)
        with open(fx["eval"]) as fh:
            e = json.load(fh)
        e["out_root"] = ev_in
        e["cal_glob"] = os.path.join(ev_in, "calbank", "shard_*.npz")
        H.write(fx["eval"], json.dumps(e))
        rc, out = H.run([H.PYTHON, H.PURGE, "--config", fx["cfg"], "--what", "evaluation,state",
                         "--no-archive"], H.fake_env(root))
        assert rc == 0, out
        listed = [t for _k, t, _n in _listed(out)]
        rc2, out2 = H.run([H.PYTHON, H.PURGE, "--config", fx["cfg"], "--what", "evaluation,state",
                           "--no-archive", "--confirm", _code(out)], H.fake_env(root))
        gone = not os.path.exists(ev_in)
        assert listed.count(ev_in) == 1 and rc2 == 0, (
            "listed %d time(s); the confirm exit %d; out_root gone: %s\n%s"
            % (listed.count(ev_in), rc2, gone, out2))
    finally:
        _qdel_all(root)
        shutil.rmtree(root, ignore_errors=True)


def test_o24_2_unreadable_ledger_is_not_read_as_empty(fx):
    """O24.11 edge: the evaluation's ledger cannot be read while one of its
    jobs is live. The spec refuses while a job of the ledger is not
    finished; a ledger that cannot be read hides them. Expected (the safe
    reading): a refusal, not a deletion."""
    jid = _qsub(fx, "ev_s1b", 60)
    led = os.path.join(fx["ev"], "jobs", "ledger.json")
    H.write(led, '{"sample:S-A1:A1": {"job_id": "%s"}' % jid)      # truncated JSON
    rc, out = _purge(fx, ["--what", "banks"])
    assert rc == 4, ("an unreadable ledger read as no job; the live job %s is not seen:\n%s"
                     % (jid, out))


def test_o24_2_explicit_eval_config_that_is_missing(fx):
    """Edge (the spec is silent): --eval-config names a file that does not
    exist and `evaluation` is not named. Recorded as found (the safe reading
    is exit 2: an explicit file that cannot be read)."""
    rc, out = _purge(fx, ["--what", "banks", "--eval-config", os.path.join(fx["root"], "nope.json")])
    assert rc == 2, ("an explicit --eval-config that does not exist is ignored (exit %d); the "
                     "ledger check is skipped:\n%s" % (rc, out))


def test_o24_6_tilde_expanded(fx):
    """O24.6: `~` in --archive is the user's home."""
    home = os.path.join(fx["root"], "home")
    os.makedirs(home)
    env = H.fake_env(fx["root"], extra={"HOME": home})
    rc, out = _purge(fx, ["--what", "evaluation", "--archive", "~/e.tar.gz"], env=env)
    assert rc == 0 and "into %s" % os.path.join(home, "e.tar.gz") in out, out


def test_o24_13_ledger_written_by_the_evaluations_launcher(fx):
    """O24.13, the boundary with block 22: the ledger ev_submit.py writes,
    with its calibration jobs queued, refuses the purge; once they have
    ended, the purge lists."""
    env = H.fake_env(fx["root"], delay="600")          # the jobs stay queued
    led = os.path.join(fx["ev"], "jobs", "ledger.json")
    os.remove(led)
    shutil.rmtree(os.path.join(fx["ev"], "calbank"))      # the fixture's stand-in, not a real build
    rc, out = H.run([H.PYTHON, os.path.join(H.EVAL, "ev_submit.py"), "--config", fx["cfg"],
                     "--eval-config", fx["eval"], "--what", "calbank"], env)
    assert rc == 0 and os.path.isfile(led), out
    with open(led) as fh:
        jobs = [r["job_id"] for r in json.load(fh).values()]
    assert jobs, "the launcher recorded no job"
    rc, out = _purge(fx, ["--what", "searches"], env=env)
    assert rc == 4 and all(j.split(".")[0] in out for j in jobs), out
    for j in jobs:
        H.run(["qdel", j], env)
    end = time.time() + 60
    while any(_live(fx, j) for j in jobs):
        assert time.time() < end
        time.sleep(0.3)
    rc, out = _purge(fx, ["--what", "searches"], env=env)
    assert rc == 0, out


def _training_env():
    try:
        import torch  # noqa: F401
        import sbi    # noqa: F401
        import skopt  # noqa: F401
        import zuko   # noqa: F401
        from pytorch_metric_learning.utils import loss_and_miner_utils  # noqa: F401
    except ImportError as exc:
        pytest.skip("the training environment is not installed here (%s)" % exc)


def test_o24_12_after_a_real_run_nothing_is_left():
    """O24.12, the boundary with blocks 1, 15-18: a reduced project run to
    PIPELINE DONE under the fake PBS, then the five kinds purged: nothing the
    run wrote under the banks, tune, runs and work roots is left but purge/;
    the next dry run starts a new run (no state, every stage to run)."""
    _training_env()
    import test_smoke_synthetic as TS
    root = tempfile.mkdtemp(prefix="t1d_b24_e2e_")
    try:
        path, d = TS._small(root, searches=[
            {"name": "S-A1", "campaign": "S-A1", "real": "R0", "alt_best_fit": False}])
        spool = os.path.join(root, "spool")
        env = H.fake_env(root, extra={"FAKE_PBS_ENFORCE_WALLTIME": "0", "OMP_NUM_THREADS": "1",
                                      "PYTHONPATH": H.PIPE})
        env["FAKE_PBS_SPOOL"] = spool
        rc, out = H.run([H.PYTHON, os.path.join(H.PIPE, "pipeline.py"), "run", "--config", path,
                         "--poll-seconds", "0.5"], env, cwd=root, timeout=1800)
        assert rc == 0 and "PIPELINE DONE" in out, out[-3000:]
        p = d["paths"]
        made = {}
        for key in ("banks_root", "tune_root", "runs_root", "work_root"):
            made[key] = sorted(os.listdir(p[key]))
        assert made["banks_root"] == ["synth_real_R0", "synth_real_R3", "synth_sim"], made
        ev = os.path.join(p["runs_root"], "evaluation")
        H.write(os.path.join(ev, "metrics", "METRICS.json"), "{}")
        with open(H.SYNTH_EVAL) as fh:
            e = json.load(fh)
        e["out_root"] = ev
        e["cal_glob"] = os.path.join(ev, "calbank", "shard_*.npz")
        H.write(os.path.join(root, "synth_eval.json"), json.dumps(e))
        H.write(os.path.join(p["banks_root"], "giulia_sim", "keep.npz"), "decoy")
        arch = os.path.join(root, "eval.tar.gz")
        rc, out = H.run([H.PYTHON, H.PURGE, "--config", path, "--what", KINDS5, "--archive", arch],
                        env)
        assert rc == 0, out
        rc, out = H.run([H.PYTHON, H.PURGE, "--config", path, "--what", KINDS5, "--archive", arch,
                         "--confirm", _code(out)], env)
        assert rc == 0 and "PURGE DONE" in out, out
        assert sorted(os.listdir(p["banks_root"])) == ["giulia_sim"], os.listdir(p["banks_root"])
        assert os.listdir(p["tune_root"]) == [], os.listdir(p["tune_root"])
        assert os.listdir(p["runs_root"]) == [], os.listdir(p["runs_root"])
        assert os.listdir(p["work_root"]) == ["purge"], os.listdir(p["work_root"])
        rc, out = H.run([H.PYTHON, os.path.join(H.PIPE, "pipeline.py"), "run", "--config", path,
                         "--dry-run"], env, cwd=root)
        assert rc == 0 and "REFUSE" not in out and "STOP" not in out, out
        assert not os.path.exists(os.path.join(p["work_root"], "state.json"))
    finally:
        _qdel_all(root)
        shutil.rmtree(root, ignore_errors=True)


def test_o24_15_stdlib_only_python36_ascii_lf():
    """O24.15, OG.1: the purge tool imports with the standard library alone
    (no site-packages: -S -I), parses under Python 3.6 (vermin), and is
    ASCII with LF line ends."""
    rc, out = H.run([H.PYTHON, "-S", "-I", "-c",
                     "import sys; sys.path[:0] = [%r, %r, %r]; "
                     "import purge_outputs, ev_config, ev_submit, stages.synthetic; "
                     "print('imports ok')" % (H.PIPE, H.TOOLS, H.EVAL)], dict(os.environ),
                    cwd=H.TOOLS)
    assert rc == 0 and "imports ok" in out, out
    with open(H.PURGE, "rb") as fh:
        raw = fh.read()
    assert b"\r" not in raw and all(c < 128 for c in raw)
    vermin = shutil.which("vermin")
    if vermin is None:
        pytest.skip("vermin not installed")
    rc, out = H.run([vermin, "-t=3.6-", "--no-tips", "--violations", H.PURGE], dict(os.environ))
    assert rc == 0, out


def test_o24_7_an_existing_sha256_beside_the_archive(fx):
    """Edge (the spec refuses an existing archive path, and is silent on its
    .sha256): a `<path>.sha256` left by another archive. Recorded as found:
    the safe reading is a refusal (exit 2) rather than an overwrite."""
    arch = os.path.join(fx["root"], "e.tar.gz")
    H.write(arch + ".sha256", "0" * 64 + "  e.tar.gz\n")
    rc, out = _purge(fx, ["--what", "evaluation", "--archive", arch])
    assert rc == 2, ("an existing %s.sha256 is not refused (exit %d): the confirm overwrites it"
                     % (os.path.basename(arch), rc))


def test_o24_3_out_root_under_another_target_is_deleted_unarchived():
    """Edge (the spec is silent): the evaluation's out_root under another
    target -- here the work root -- and a purge that names `state` alone.
    The archive is the spec's guarantee for the evaluation's results
    ("written and verified before anything is deleted"); expected (the safe
    reading): a refusal, or the archive asked for. Recorded as found."""
    root = tempfile.mkdtemp(prefix="t1d_b24_ov2_")
    try:
        fx = _project(root)
        ev_in = os.path.join(fx["wr"], "evaluation")
        shutil.move(fx["ev"], ev_in)
        with open(fx["eval"]) as fh:
            e = json.load(fh)
        e["out_root"] = ev_in
        e["cal_glob"] = os.path.join(ev_in, "calbank", "shard_*.npz")
        H.write(fx["eval"], json.dumps(e))
        env = H.fake_env(root)
        rc, out = H.run([H.PYTHON, H.PURGE, "--config", fx["cfg"], "--what", "state"], env)
        if rc != 0:
            return                                   # refused: the safe reading
        rc2, out2 = H.run([H.PYTHON, H.PURGE, "--config", fx["cfg"], "--what", "state", "--confirm",
                           _code(out)], env)
        assert os.path.isdir(ev_in), ("the evaluation's results deleted with no archive asked "
                                      "(exit %d):\n%s" % (rc2, out2))
    finally:
        _qdel_all(root)
        shutil.rmtree(root, ignore_errors=True)
