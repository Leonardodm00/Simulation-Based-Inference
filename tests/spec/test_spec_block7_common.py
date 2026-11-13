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


# ---------------------------------------------------------------------------
# t1-A (2026-11-13, D-131): block 7's amendments of 2026-10-21 (D-102, the
# avoid list) and 2026-10-29 (D-111 environment faults; D-113 the job
# protocol). Expected values are the spec's known answers.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    (94, True), ("94", True), (0, False), (1, False), (93, False), (135, False), (271, False),
    (None, False)])
def test_held_off_node(code, expected):
    """D-102: held_off_node on 94 (int or text) and on 0, 1, 93, 135, 271, None."""
    assert CM.held_off_node({"exit_status": code}) is expected


@pytest.mark.parametrize("code", [96, 97, 98, -2, -11, "96", -1, -3, -4, -5, -6, -7, -10, -14, -16,
                                  -18, -20])
def test_env_fault_named(code):
    """D-111 (c): exit 96, 97 / 98, a code of JOB_EXEC_FAULTS ({-1, -2, -3,
    -4, -5, -6, -7, -10, -11, -14, -16, -18, -20}), 96 as text."""
    assert CM.env_fault({"exit_status": code}, None) is not None


@pytest.mark.parametrize("code", [0, 1, 2, 3, 94, 135, 271, -29, None, -24, -17, -19])
def test_env_fault_not(code):
    """D-111 (c): None on 0, 1, 2, 3, 94, 135, 271, -29, None (and the
    limits' kills -24..-29 and a hook's deletion -17, -19, which the
    constants' comment of the spec leaves out of JOB_EXEC_FAULTS)."""
    assert CM.env_fault({"exit_status": code}, None) is None


def test_env_fault_marker_without_exit_status():
    """D-111 (c): the log's `RUN_ARGV ENV FAULT:` line with no exit status."""
    log = "x\nRUN_ARGV ENV FAULT: host broken: the environment e cannot start (conda: no)\n"
    why = CM.env_fault({"exit_status": None}, log)
    assert why is not None and "host broken" in why


def test_constants_d102_d111():
    """The constants the spec names (block 7): AVOID_FAULTS 3, AVOID_HOLD_S
    600, AVOID_EXIT 94, AVOID_FILE_NAME avoid_nodes.txt, ENV_FAULT_EXIT 96,
    ENV_FAULT_STREAK 12, ENV_FAULT_HOSTS 3, LOG_GRACE_S 300, RECORD_GRACE_S
    300 (the environment overrides are unset here)."""
    for v in ("PIPELINE_AVOID_NODE_FAULTS", "PIPELINE_AVOID_HOLD_S", "PIPELINE_ENV_FAULT_STREAK",
              "PIPELINE_ENV_FAULT_HOSTS", "PIPELINE_RECORD_GRACE_S"):
        if os.environ.get(v):
            pytest.skip("%s overridden here" % v)
    assert (CM.AVOID_FAULTS, CM.AVOID_HOLD_S, CM.AVOID_EXIT, CM.AVOID_FILE_NAME) == \
        (3, 600, 94, "avoid_nodes.txt")
    assert (CM.ENV_FAULT_EXIT, CM.ENV_FAULT_STREAK, CM.ENV_FAULT_HOSTS) == (96, 12, 3)
    assert CM.JOB_EXEC_FAULTS == frozenset((-1, -2, -3, -4, -5, -6, -7, -10, -11, -14, -16, -18, -20))
    assert CM.LOG_GRACE_S == 300 and CM.RECORD_GRACE_S == 300


def test_host_of_log_lines():
    """D-102 / D-111 / D-113: the host from `RUN_ARGV: host <h>`, `RUN_ARGV
    AVOID: host <h>`, `RUN_ARGV ENV: host <h>:` and `RUN_ARGV ENV FAULT:
    host <h>:`; the protocol's `RUN_ARGV: host <h>, job <id>, env <e>`."""
    assert CM.host_of_log("RUN_ARGV: host dvnode012\n") == "dvnode012"
    assert CM.host_of_log("RUN_ARGV AVOID: host nodeX is on the avoid list f (D-102); ...") == "nodeX"
    assert CM.host_of_log("RUN_ARGV ENV: host noconda: conda unavailable (...)") == "noconda"
    assert CM.host_of_log("RUN_ARGV ENV FAULT: host broken: the environment") == "broken"
    assert CM.host_of_log("RUN_ARGV: host gn1, job 7.s, env e (the pipeline's job protocol, D-113)") \
        == "gn1"
    assert CM.host_of_log("no such line") is None


def test_avoid_list_read_add(tmp_path):
    """D-102 known answers: an absent file reads []; a header, blanks and two
    hosts read the two; add appends once, never rewrites a line."""
    p = str(tmp_path / "w" / "avoid_nodes.txt")
    assert CM.read_avoid_list(p) == []
    assert CM.add_to_avoid_list(p, "nodeA", "a note") is True
    first = open(p, "rb").read()
    assert CM.read_avoid_list(p) == ["nodeA"]
    assert CM.add_to_avoid_list(p, "nodeA", "again") is False
    assert open(p, "rb").read() == first
    assert CM.add_to_avoid_list(p, "nodeB", "b") is True
    assert open(p, "rb").read().startswith(first)
    with open(p, "a") as fh:
        fh.write("\n   \n# a comment\n")
    assert CM.read_avoid_list(p) == ["nodeA", "nodeB"]
    assert CM.add_to_avoid_list(p, "?", "never") is False


def test_note_node_fault_and_sync(tmp_path):
    """D-102: listed at its third fault and not before; `?` never listed; a
    fourth fault counted, not listed again; after the line is deleted by
    hand, sync sets the base to 4: the fifth and sixth faults do not list it
    again, the seventh does."""
    p = str(tmp_path / "avoid_nodes.txt")
    sub, logs = {}, []
    assert CM.note_node_fault(sub, "h", "w", "search", logs.append, p) == (1, False)
    assert CM.note_node_fault(sub, "h", "w", "search", logs.append, p) == (2, False)
    assert CM.read_avoid_list(p) == []
    assert CM.note_node_fault(sub, "h", "w", "search", logs.append, p) == (3, True)
    assert CM.read_avoid_list(p) == ["h"]
    n, listed = CM.note_node_fault(sub, "h", "w", "search", logs.append, p)
    assert n == 4 and listed is False
    for _ in range(5):
        CM.note_node_fault(sub, "?", "w", "search", logs.append, p)
    assert "?" not in CM.read_avoid_list(p)
    # the user deletes the host's line
    lines = [l for l in open(p).read().split("\n") if l.strip() != "h"]
    open(p, "w").write("\n".join(lines))
    assert CM.sync_avoid_list(sub, p, logs.append, "search") == ["h"]
    assert sub["avoid_base"]["h"] == 4
    assert CM.note_node_fault(sub, "h", "w", "search", logs.append, p)[1] is False   # 5th
    assert CM.note_node_fault(sub, "h", "w", "search", logs.append, p)[1] is False   # 6th
    assert CM.note_node_fault(sub, "h", "w", "search", logs.append, p)[1] is True    # 7th
    assert CM.read_avoid_list(p) == ["h"]


def test_env_fault_streak_guard(tmp_path):
    """D-111 (d): 11 faults on 3 hosts do not stop, the 12th does (naming the
    hosts this run listed); 12 on 2 hosts do not stop; a job that ran clears
    the streak."""
    p = str(tmp_path / "avoid_nodes.txt")
    sub = {}
    hosts = ["a", "b", "c"]
    for i in range(11):
        assert CM.note_env_fault(sub, hosts[i % 3], "x", "search", lambda t: None, p)[1] is None
    n, why = CM.note_env_fault(sub, "a", "x", "search", lambda t: None, p)
    assert why is not None and "12 job(s) in a row" in why and "--retry search" in why
    sub2 = {}
    for i in range(12):
        assert CM.note_env_fault(sub2, ["a", "b"][i % 2], "x", "s", lambda t: None,
                                 str(tmp_path / "f2"))[1] is None
    CM.clear_env_fault_streak(sub2)
    assert sub2["env_fault_streak"]["n"] == 0


def test_qsub_line_roles_and_avoid(tmp_path):
    """D-111 (b): JOBS_DIR, WORK_ROOT, CHAIN_NAME (pl_<tag>), ROLE through
    -v; a role other than stage / driver refused; D-102: AVOID_FILE and
    AVOID_HOLD_S only with avoid_file."""
    cfg = C.load_config(GIULIA)
    a = CM.qsub_line(cfg, "t", "cpu", 2, "1gb", "01:00:00", "/l", "/a", "/w")
    v = a[a.index("-v") + 1]
    assert "ROLE=stage" in v and "CHAIN_NAME=pl_giulia" in v and "AVOID_FILE" not in v
    assert "JOBS_DIR=%s" % os.path.join(cfg.data["paths"]["sbi_dir"], "hpc", "pipeline", "jobs") in v
    assert "WORK_ROOT=%s" % cfg.data["paths"]["work_root"] in v
    b = CM.qsub_line(cfg, "t", "cpu", 2, "1gb", "01:00:00", "/l", "/a", "/w", avoid_file="/w/avoid",
                     role="driver")
    vb = b[b.index("-v") + 1]
    assert "ROLE=driver" in vb and "AVOID_FILE=/w/avoid" in vb and "AVOID_HOLD_S=" in vb
    with pytest.raises(ValueError):
        CM.qsub_line(cfg, "t", "cpu", 2, "1gb", "01:00:00", "/l", "/a", "/w", role="x")


def test_stage_job_protocol_refuses_comma_or_blank(tmp_path):
    """D-113 (a): the five variables; a value with a comma or a blank is
    refused (ValueError)."""
    raw = json.load(open(GIULIA))
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw))
    v = CM.stage_job_protocol(C.load_config(str(p)))
    assert sorted(v) == sorted(["PIPELINE_JOBS_DIR", "WORK_ROOT", "CHAIN_NAME", "AVOID_FILE",
                                "AVOID_HOLD_S"])
    for bad in ("/w/a,b", "/w/a b"):
        raw["paths"]["work_root"] = bad
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            CM.stage_job_protocol(C.load_config(str(p)))


def test_free_round_stall():
    """D-113 (blocks 2 (d), 3 (c)): FREE_ROUND_STALL (3) free rounds in a row
    that leave at least as many unstarted stop the stage: three free rounds
    run (rounds 1-3), the decision after the third is the STOP; fewer
    unstarted resets it; a counted round resets it."""
    if os.environ.get("PIPELINE_FREE_ROUND_STALL"):
        pytest.skip("overridden here")
    sub = {}
    assert CM.free_round(sub, 2, "sim_bank") is None      # -> free round 1
    assert CM.free_round(sub, 2, "sim_bank") is None      # -> free round 2
    assert CM.free_round(sub, 2, "sim_bank") is None      # -> free round 3
    why = CM.free_round(sub, 2, "sim_bank")
    assert why is not None and "--retry sim_bank" in why
    sub = {}
    for n in (3, 3, 2, 2, 2):                               # fewer once: the stall restarts
        assert CM.free_round(sub, n, "s") is None
    sub = {}
    CM.free_round(sub, 1, "s")
    CM.free_round(sub, 1, "s")
    CM.counted_round(sub)
    assert CM.free_round(sub, 1, "s") is None and CM.free_round(sub, 1, "s") is None


def test_unrun_dependent():
    """D-113 (b): a finished job that waited on another (`after`) and that
    PBS ended with a code of JOB_EXEC_FAULTS before a log of its own."""
    assert CM.unrun_dependent({"exit_status": -1, "after": "7[].s"}, False) is True
    assert CM.unrun_dependent({"exit_status": -1, "after": "7[].s"}, True) is False
    assert CM.unrun_dependent({"exit_status": -1}, False) is False
    assert CM.unrun_dependent({"exit_status": 1, "after": "7"}, False) is False
    assert CM.unrun_dependent({"exit_status": None, "after": "7"}, False) is False


# ------------------------------------------------- perrow_check (block 7) ---

def _perrow_module():
    sys.path.insert(0, os.path.join(PIPE, "tools"))
    import perrow_check as PC
    return PC


def _tau_math(t):
    """tau(theta) = u^2/2 + ln(2 pi)/2 + ln(theta (1 - theta)), u = logit(theta)."""
    u = math.log(t) - math.log(1.0 - t)
    return u * u / 2.0 + math.log(2.0 * math.pi) / 2.0 + math.log(t * (1.0 - t))


def test_perrow_tau_known_answers():
    """Block 7 (perrow_check): tau(0.003) = 11.962413 nats; the two edges
    symmetric; 0 inside the bound B = 5 (|u| <= 5)."""
    PC = _perrow_module()
    assert PC.BOUND == 5.0
    th = np.array([[0.003, 0.997, 0.5, 1e-4, 0.006]])
    tau, u, mask = PC.edge_terms(th, 5.0)
    assert abs(_tau_math(0.003) - 11.962413) < 5e-7          # the spec's 6 decimals
    assert tau[0, 0] == pytest.approx(_tau_math(0.003), abs=1e-12)
    assert tau[0, 1] == pytest.approx(tau[0, 0], abs=1e-9)    # symmetric edges
    assert tau[0, 2] == 0.0 and not mask[0, 2]
    assert tau[0, 3] == pytest.approx(_tau_math(1e-4), abs=1e-9)
    # 0.006: |logit| = 5.11 > 5, past the bound
    assert mask[0, 4] and tau[0, 4] == pytest.approx(_tau_math(0.006), abs=1e-12)


@pytest.mark.parametrize("vals,best_e,gain", [
    ([5.0 - 0.05 * e for e in range(20)], 19, 0.25),                       # a steady fall
    ([abs(e - 9) * 0.1 + 1.0 for e in range(20)], 9, 0.0),                # a minimum at 9, then a rise
    ([3.0, 2.0, 1.5, 1.0] + [1.0 - 9e-7] * 16, 3, 0.0),                   # a fall of 9e-7 after 3
    ([3.0, 2.0, 1.5, 1.0] + [1.0 - 9e-7 * k for k in range(1, 17)], 19, 4.5e-6),   # a creep
    ([2.0, 1.9, 1.8, 1.7], 3, None),                                      # 4 epochs: g none
    ([1.0] * 6, 0, 0.0),                                                  # flat: e* = 0; E = 6
    ([1.0] * 5, 0, None),                                                 # E = 5: g none
    ([1.0], 0, None),                                                     # 1 epoch
])
def test_perrow_curve_rule(vals, best_e, gain):
    """Block 7 (perrow_check, the training curves): e* by the runner's rule
    (best = inf, e* = -1; epoch e becomes e* when val(e) < best - 1e-6);
    g = min_{e < E-5} val(e) - min_e val(e), none when E <= 5."""
    PC = _perrow_module()
    c = PC.curve_summary([{"val_npe": v} for v in vals])
    assert c["best_epoch"] == best_e and c["epochs"] == len(vals)
    if gain is None:
        assert c["gain_tail"] is None
    else:
        assert c["gain_tail"] == pytest.approx(gain, abs=1e-12)


@pytest.mark.parametrize("hist", ["not a list", [{"val_npe": 1.0}, {"val_npe": float("nan")}],
                                  [{"val_npe": True}], [{"val_npe": 10 ** 400}], [], [{"x": 1}]])
def test_perrow_curve_refusals(hist):
    """A history without a finite val_npe in every entry is no curve."""
    PC = _perrow_module()
    assert PC.curve_summary(hist) is None
