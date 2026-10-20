"""Spec-derived tests of block 7's shared helpers and the ensemble tool
(SPEC sec. 2.3; block 7: node faults D-096, checkpoints D-097, an
unreadable record, the ensemble formula D-089).

Oracles O7.1-O7.4, O7.7, O7.9, O-G8 of the 2026-10-20 report. Expected
values are hand-written or computed with math / numpy loops, never with the
code under test.
"""
import errno
import json
import math
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from conftest import GIULIA, PIPE

import config as C
from stages import common as CM


# ------------------------------------------------------------ node_fault ----

# the three davinci lines of 2026-10-18, as the spec quotes them
DAVINCI = [
    ("runner exited -7", None),
    (None, "BrokenPipeError: [Errno 108] Cannot send after transport endpoint shutdown"),
    (None, "OSError: [Errno 5] Input/output error: '/davinci-1/home/u/banks/giulia_sim/shard_0012.npz'"),
]


@pytest.mark.parametrize("err,log", DAVINCI + [
    (None, "RUN_ARGV EXIT: 135\n"),
    (None, "line\nBus error (core dumped)\n"),
    (None, "OSError: [Errno 107] Transport endpoint is not connected"),
    (None, "OSError: [Errno 116] Stale file handle"),
    (None, "OSError: [Errno 121] Remote I/O error"),
    (None, "Could not chdir to home directory /home/u: No such file or directory"),
    (None, "something: Input/output error"),
])
def test_node_fault_named(err, log):
    assert CM.node_fault(err, log) is not None


@pytest.mark.parametrize("err,log", [
    (None, 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad shape\n'),
    (None, "FileNotFoundError: [Errno 2] No such file or directory: 'x'"),
    (None, "OSError: [Errno 28] No space left on device"),
    (None, "OSError: [Errno 122] Disk quota exceeded"),
    ("runner exited -9", None),
    ("runner exited 1", None),
    (None, "RUN_ARGV EXIT: 137\n"),        # SIGKILL: memory or an operator
    (None, "RUN_ARGV EXIT: 1\n"),
    (None, "[Errno 50] something"),        # an errno number that merely starts with 5
])
def test_node_fault_not(err, log):
    assert CM.node_fault(err, log) is None


def test_errno_numbers_as_spec():
    """The spec names 5, 107, 108, 116, 121 (Linux)."""
    assert (errno.EIO, errno.ENOTCONN, errno.ESHUTDOWN, errno.ESTALE, errno.EREMOTEIO) == \
        (5, 107, 108, 116, 121)


# --------------------------------------------------------- try_read_json ----

def test_try_read_json(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text('{"a": 1}')
    assert CM.try_read_json(str(ok)) == ({"a": 1}, None)

    empty = tmp_path / "e.json"
    empty.write_bytes(b"")
    obj, prob = CM.try_read_json(str(empty))
    assert obj is None and "(0 bytes" in prob

    nul = tmp_path / "n.json"
    nul.write_bytes(b"\0" * 37)
    obj, prob = CM.try_read_json(str(nul))
    assert obj is None and "37 bytes, all NUL" in prob

    lst = tmp_path / "l.json"
    lst.write_text("[1, 2]")
    obj, prob = CM.try_read_json(str(lst))
    assert obj is None and prob.startswith("not a JSON object (list")

    obj, prob = CM.try_read_json(str(tmp_path / "absent.json"))
    assert obj is None and "Error" in prob           # FileNotFoundError: an OSError

    obj, prob = CM.try_read_json(str(tmp_path))      # a directory: cannot be opened as a file
    assert obj is None and prob


# ------------------------------------------------------------- set_aside ----

def test_set_aside_numbering(tmp_path):
    p = tmp_path / "t.json"
    for k in (1, 2, 3):
        p.write_text(str(k))
        new = CM.set_aside(str(p), "killed")
        assert new == str(p) + ".killed%d" % k
        assert open(new).read() == str(k)            # never overwritten
    # a gap: .killed1 removed -> the first k >= 1 not taken is 1
    os.remove(str(p) + ".killed1")
    p.write_text("4")
    assert CM.set_aside(str(p), "killed") == str(p) + ".killed1"
    assert open(str(p) + ".killed2").read() == "2"


# ------------------------------------------------------------ write_json ----

def test_write_json_fsyncs_before_rename(tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(CM.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(CM.os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1])
    path = tmp_path / "r" / "x.json"
    CM.write_json(str(path), {"b": 2, "a": 1})
    assert events == ["fsync", "replace"]
    assert json.load(open(str(path))) == {"a": 1, "b": 2}


# --------------------------------------------------------------- memory -----

@pytest.mark.parametrize("text,gb", [("160gb", 160.0), ("1tb", 1024.0), ("512mb", 0.5),
                                     ("1048576kb", 1.0)])
def test_mem_to_gb(text, gb):
    assert CM.mem_to_gb(text) == pytest.approx(gb, rel=0, abs=1e-12)


def test_mem_to_gb_refuses():
    with pytest.raises(ValueError):
        CM.mem_to_gb("160GB")


def test_elapsed_hours():
    assert CM.elapsed_hours("2026-10-01T00:00:00Z", "2026-10-02T06:00:00Z") == 30.0


# -------------------------------------------------------------- qsub line ----

def test_qsub_line_shape():
    cfg = C.load_config(GIULIA)
    t = cfg.data["search"]["trial"]
    argv = CM.qsub_line(cfg, "trial_0123456789abcdef", t["queue"], t["ncpus"], t["mem"], t["walltime"],
                        "/w/jobs/a.1.log", "/w/jobs/a.1.argv", "/w")
    assert argv[0] == "qsub"
    o = dict(zip(argv[1:-1:2], argv[2:-1:2]))
    assert o["-q"] == "cpu"
    assert argv[argv.index("-o") + 1] == "/w/jobs/a.1.log"
    assert "select=1:ncpus=48:mem=160gb" in argv
    assert "walltime=120:00:00" in argv
    name = argv[argv.index("-N") + 1]
    assert len(name) <= 15 and name[0].isalpha()          # PBS Pro's name rule
    assert argv[-1].endswith(os.path.join("hpc", "pipeline", "jobs", "run_argv.pbs"))
    v = argv[argv.index("-v") + 1]
    assert "ARGV_FILE=/w/jobs/a.1.argv" in v and "ENV_NAME=sbi_env" in v


def test_qsub_line_refuses_comma():
    cfg = C.load_config(GIULIA)
    with pytest.raises(ValueError):
        CM.qsub_line(cfg, "x", "cpu", 1, "1gb", "01:00:00", "/l", "/a,b", "/w")


def test_argv_file_roundtrip_with_empty_token(tmp_path):
    """One token per line (block 7 `jobs/run_argv.pbs`). The job script
    reads every line, empty ones included; `read_argv_file` should agree."""
    argv = ["python", "x.py", "--plan-args", "", "--z"]
    p = CM.write_argv_file(str(tmp_path / "a.argv"), argv)
    out = subprocess.run(["bash", "-c", 'A=(); while IFS= read -r l || [ -n "$l" ]; do A+=("$l"); done < "$1";'
                          ' printf "%s\\n" "${#A[@]}"', "_", p], stdout=subprocess.PIPE,
                         universal_newlines=True).stdout
    assert int(out.strip()) == len(argv)                  # what the job runs
    assert CM.read_argv_file(p) == argv                   # what the pipeline reads back


# ------------------------------------------------ launcher / header reads ----

def test_shell_default_and_header(tmp_path):
    sh = tmp_path / "launch.sh"
    sh.write_text('#!/bin/bash\nNCPUS="${NCPUS:-4}"\n  GATE_NCPUS="${GATE_NCPUS:-16}"\n')
    assert CM.shell_default(str(sh), "NCPUS") == "4"
    assert CM.shell_default(str(sh), "GATE_NCPUS") == "16"
    with pytest.raises(RuntimeError):
        CM.shell_default(str(sh), "WALLTIME")
    pbs = tmp_path / "job.pbs"
    pbs.write_text("#!/bin/bash\n#PBS -N x\n#PBS -l select=2:ncpus=24:mem=100gb\n")
    assert CM.pbs_header_ncpus(str(pbs)) == 48
    pbs.write_text("#!/bin/bash\n")
    with pytest.raises(RuntimeError):
        CM.pbs_header_ncpus(str(pbs))


# ------------------------------------------------------- checkpoint (D-097) --

def test_checkpoint_problem(tmp_path):
    good = tmp_path / "a_train_ckpt.pt"
    with zipfile.ZipFile(str(good), "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", b"x" * 4096)
    assert CM.checkpoint_problem(str(good)) == (None, "")
    blob = bytearray(good.read_bytes())
    i = blob.index(b"x" * 64) + 10
    blob[i] ^= 0xFF
    bad = tmp_path / "b_train_ckpt.pt"
    bad.write_bytes(bytes(blob))
    kind, why = CM.checkpoint_problem(str(bad))
    assert kind == "content" and "archive/data.pkl" in why
    trunc = tmp_path / "c_train_ckpt.pt"
    trunc.write_bytes(good.read_bytes()[:100])
    assert CM.checkpoint_problem(str(trunc))[0] == "content"
    moved = CM.set_aside_bad_checkpoints(str(tmp_path), lambda t: None, "search", "trial x")
    assert sorted(os.path.basename(m) for m in moved) == ["b_train_ckpt.pt.corrupt1",
                                                         "c_train_ckpt.pt.corrupt1"]
    assert good.exists()


# --------------------------------------------------- ensemble NLL (D-089) ----

def _perrow(path, nll, group=None, theta=None):
    d = {"nll": np.asarray(nll, dtype=np.float64)}
    if group is not None:
        d["group"] = np.asarray(group)
    if theta is not None:
        d["theta"] = np.asarray(theta, dtype=np.float32)
    np.savez(str(path), **d)
    return str(path)


def _tool(args):
    return subprocess.run([sys.executable, os.path.join(PIPE, "tools", "ensemble_nll.py")] + args,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def _by_hand(rows):
    """L_bar = -(1/n) sum_i [log sum_m exp(-l_mi) - log M], with math only."""
    M, n = len(rows), len(rows[0])
    tot = 0.0
    for i in range(n):
        vals = [-rows[m][i] for m in range(M)]
        mx = max(vals)
        tot += mx + math.log(sum(math.exp(v - mx) for v in vals)) - math.log(M)
    return -tot / n


def test_ensemble_copies_equal_member(tmp_path):
    rng = np.random.default_rng(0)
    l = rng.normal(2.0, 1.0, 50)
    files = [_perrow(tmp_path / ("m%d.npz" % k), l, group=np.arange(50) % 7) for k in range(3)]
    out = tmp_path / "e.json"
    r = _tool(["--members"] + files + ["--out", str(out)])
    assert r.returncode == 0, r.stdout
    assert r.stdout.startswith("ENSEMBLE OK: M=3 n=50 L_bar=")
    rec = json.load(open(str(out)))
    assert abs(rec["L_bar"] - float(np.mean(l))) <= 1e-12      # M copies: L_bar = L_1 (spec, 1e-12)


def test_ensemble_formula_and_jensen(tmp_path):
    rng = np.random.default_rng(1)
    rows = [rng.normal(1.0, 2.0, 40), rng.normal(3.0, 0.5, 40), rng.normal(0.0, 1.0, 40)]
    files = [_perrow(tmp_path / ("m%d.npz" % k), r) for k, r in enumerate(rows)]
    out = tmp_path / "e.json"
    r = _tool(["--members"] + files + ["--out", str(out)])
    assert r.returncode == 0, r.stdout
    rec = json.load(open(str(out)))
    # float64 sums of 40 terms: 1e-10 is far above roundoff, far below any formula error
    assert rec["L_bar"] == pytest.approx(_by_hand([list(x) for x in rows]), abs=1e-10)
    assert rec["L_bar"] <= np.mean([np.mean(x) for x in rows]) + 1e-12   # Jensen
    assert rec["M"] == 3 and rec["n_report"] == 40


def test_ensemble_refuses_row_counts(tmp_path):
    a = _perrow(tmp_path / "a.npz", np.ones(5))
    b = _perrow(tmp_path / "b.npz", np.ones(6))
    r = _tool(["--members", a, b, "--out", str(tmp_path / "e.json")])
    assert r.returncode != 0 and not (tmp_path / "e.json").exists()


def test_ensemble_refuses_other_donors(tmp_path):
    a = _perrow(tmp_path / "a.npz", np.ones(5), group=[1, 1, 2, 2, 3])
    b = _perrow(tmp_path / "b.npz", np.ones(5), group=[1, 1, 2, 3, 3])
    r = _tool(["--members", a, b, "--out", str(tmp_path / "e.json")])
    assert r.returncode != 0


def test_ensemble_refuses_nonfinite(tmp_path):
    a = _perrow(tmp_path / "a.npz", [1.0, np.inf, 2.0])
    b = _perrow(tmp_path / "b.npz", [1.0, 1.0, 2.0])
    r = _tool(["--members", a, b, "--out", str(tmp_path / "e.json")])
    assert r.returncode != 0
