"""Spec-derived tests of the job scripts of block 7: the activation
`jobs/activate_env.sh` (D-111 (a)), the job `jobs/run_argv.pbs` (D-102's
avoid list, D-111 (b), its exit codes) and, as a boundary, the evaluation's
`eval_run.pbs` (block 22) through the same activation. t1-A run of
2026-11-13 (D-131).

Run by hand with bash: no conda on PATH (the sandbox has none), a `hostname`
stub first on PATH where a node name matters, fake environment folders
(`conda-meta/`, `bin/python`) for the folder fallback. Expected values are
the spec's lines and exit codes.
"""
import os
import shutil
import stat
import subprocess
import sys

import pytest

from conftest import PIPE, REPO

JOBS = os.path.join(PIPE, "jobs")
ACTIVATE = os.path.join(JOBS, "activate_env.sh")
RUN_ARGV = os.path.join(JOBS, "run_argv.pbs")
EVAL_RUN = os.path.join(PIPE, "evaluation", "jobs", "eval_run.pbs")
BASE_PATH = "/usr/local/bin:/usr/bin:/bin"


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _stub_bin(tmp_path, host):
    d = tmp_path / "stub_bin"
    d.mkdir(exist_ok=True)
    _exe(str(d / "hostname"), "#!/bin/bash\necho %s\n" % host)
    return str(d)


def _env_folder(root, name, works=True, meta=True, activate_d=()):
    """A fake environment folder: conda-meta/ and an executable bin/python
    (the system python when `works`, else a script exiting 1)."""
    p = os.path.join(root, name)
    os.makedirs(os.path.join(p, "bin"))
    if meta:
        os.makedirs(os.path.join(p, "conda-meta"))
    if works:
        _exe(os.path.join(p, "bin", "python"), '#!/bin/bash\nexec %s "$@"\n' % sys.executable)
    else:
        _exe(os.path.join(p, "bin", "python"), "#!/bin/bash\nexit 1\n")
    if activate_d:
        ad = os.path.join(p, "etc", "conda", "activate.d")
        os.makedirs(ad)
        for fname, body in activate_d:
            with open(os.path.join(ad, fname), "w") as fh:
                fh.write(body)
    return p


def _bash(script, env, cwd=None):
    r = subprocess.run(["bash", "-c", script], env=env, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
    return r.returncode, r.stdout


def _base_env(tmp_path, host="gn1", **extra):
    env = {"PATH": _stub_bin(tmp_path, host) + ":" + BASE_PATH, "HOME": str(tmp_path / "home"),
           "LANG": "C"}
    os.makedirs(env["HOME"], exist_ok=True)
    env.update(extra)
    return env


# ------------------------------------------------------- activate_env.sh ---

def test_no_conda_in_this_sandbox():
    assert shutil.which("conda", path=BASE_PATH) is None


def test_activate_skip_conda(tmp_path):
    """(a): with SKIP_CONDA=1 nothing: 0, nothing printed, PATH unchanged."""
    env = _base_env(tmp_path, SKIP_CONDA="1")
    rc, out = _bash('. "%s"; set -u; P0="$PATH"; pipeline_activate myenv; r=$?; '
                    '[ "$PATH" = "$P0" ] && echo SAMEPATH; exit $r' % ACTIVATE, env)
    assert rc == 0 and out.strip() == "SAMEPATH"


def test_activate_folder_fallback(tmp_path):
    """(a): conda absent -> the first usable of the recorded prefix,
    $HOME/.conda/envs/<env>, <d>/<env> for each d of CONDA_ENVS_PATH (an
    absent folder first here); activated as conda activates one: CONDA_
    PREFIX, CONDA_DEFAULT_ENV, CONDA_SHLVL=1, PATH=<p>/bin:$PATH, every
    activate.d script sourced in name order under set +u, the caller's -u
    restored; printed `RUN_ARGV ENV: host <h>: conda unavailable (<its
    first error line>); <env> activated from its folder <p>`."""
    envs = str(tmp_path / "envs")
    p = _env_folder(envs, "myenv", activate_d=[("b.sh", 'ORDER="${ORDER}b"; X=$UNSET_VAR\n'),
                                               ("a.sh", 'ORDER="${ORDER}a"\n')])
    env = _base_env(tmp_path, host="noconda", SKIP_CONDA="0",
                    CONDA_ENVS_PATH=str(tmp_path / "absent") + ":" + envs)
    rc, out = _bash('. "%s"; set -u; pipeline_activate myenv; r=$?; '
                    'echo "PREFIX=$CONDA_PREFIX DEF=$CONDA_DEFAULT_ENV LVL=$CONDA_SHLVL ORDER=$ORDER"; '
                    'echo "FIRST=${PATH%%%%:*}"; case $- in *u*) echo U_ON;; esac; exit $r' % ACTIVATE,
                    env)
    assert rc == 0, out
    lines = out.strip().split("\n")
    env_line = [l for l in lines if l.startswith("RUN_ARGV ENV: ")]
    assert len(env_line) == 1
    assert env_line[0].startswith("RUN_ARGV ENV: host noconda: conda unavailable (")
    assert env_line[0].endswith("; myenv activated from its folder %s" % p)
    assert "conda" in env_line[0].split("(", 1)[1]                     # its first error line
    assert "PREFIX=%s DEF=myenv LVL=1 ORDER=ab" % p in out
    assert "FIRST=%s/bin" % p in out
    assert "U_ON" in out


def test_activate_recorded_prefix_first(tmp_path):
    """(a): the recorded prefix `<WORK_ROOT>/envs/<env>.prefix` comes before
    $HOME/.conda/envs/<env>."""
    rec = _env_folder(str(tmp_path / "recorded"), "myenv")
    home_env = _env_folder(str(tmp_path / "home" / ".conda" / "envs"), "myenv")
    wr = tmp_path / "work"
    (wr / "envs").mkdir(parents=True)
    (wr / "envs" / "myenv.prefix").write_text(rec + "\n")
    env = _base_env(tmp_path, SKIP_CONDA="0", WORK_ROOT=str(wr))
    rc, out = _bash('. "%s"; pipeline_activate myenv; r=$?; echo "P=$CONDA_PREFIX"; exit $r'
                    % ACTIVATE, env)
    assert rc == 0 and "P=%s" % rec in out and home_env not in out.split("P=")[-1]


def test_activate_nothing_usable_96(tmp_path):
    """(a): none usable -> `RUN_ARGV ENV FAULT: host <h>: the environment
    <env> cannot start (conda: <error>; folders: <p> (<why>), ...)`, status
    96, naming `no conda-meta` and `its python does not start on this node`."""
    envs = str(tmp_path / "envs")
    p1 = _env_folder(str(tmp_path / "home" / ".conda" / "envs"), "myenv", meta=False)
    p2 = _env_folder(envs, "myenv", works=False)
    env = _base_env(tmp_path, host="broken", SKIP_CONDA="0", CONDA_ENVS_PATH=envs)
    rc, out = _bash('. "%s"; pipeline_activate myenv' % ACTIVATE, env)
    assert rc == 96, out
    line = [l for l in out.split("\n") if l.startswith("RUN_ARGV ENV FAULT: ")]
    assert len(line) == 1
    assert line[0].startswith("RUN_ARGV ENV FAULT: host broken: the environment myenv cannot start "
                              "(conda: ")
    assert "%s (no conda-meta)" % p1 in line[0]
    assert "%s (its python does not start on this node)" % p2 in line[0]


# --------------------------------------------------------- run_argv.pbs ----

def _job_env(tmp_path, argv, host="gn1", **extra):
    af = tmp_path / "job.argv"
    af.write_text("\n".join(argv) + "\n")
    env = _base_env(tmp_path, host=host, ARGV_FILE=str(af), WORKDIR=str(tmp_path),
                    ENV_NAME="myenv", SKIP_CONDA="1", AVOID_HOLD_S="0")
    env.update(extra)
    return env


def _run_job(env, script=RUN_ARGV):
    r = subprocess.run(["bash", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=120)
    return r.returncode, r.stdout


def test_run_argv_command_exit_and_spaces(tmp_path):
    """Block 7: one job = one command line read from a file, one argument
    per line (an argument with a space kept); the last line `RUN_ARGV EXIT:
    <n>` with the command's own status."""
    marker = tmp_path / "ran"
    code = "import sys; open(%r, 'w').write(repr(sys.argv[1:])); sys.exit(7)" % str(marker)
    env = _job_env(tmp_path, [sys.executable, "-c", code, "a b", "c"])
    rc, out = _run_job(env)
    assert rc == 7
    assert out.strip().split("\n")[-1] == "RUN_ARGV EXIT: 7"
    assert marker.read_text() == "['a b', 'c']"
    assert "RUN_ARGV: host gn1" in out


@pytest.mark.parametrize("listed,held", [
    ("gn1", True),
    ("  gn1  ", True),          # surrounding blanks: the line is the host
    ("gn", False),              # a prefix of the host
    ("gn12", False),            # a longer name
    ("# gn1", False),           # a comment
])
def test_run_argv_avoid_list(tmp_path, listed, held):
    """D-102: on a host the avoid list names exactly, the job prints
    `RUN_ARGV AVOID: host <h> is on the avoid list <path> (D-102); holding
    this slot for <s> s, then exiting 94`, holds, prints `RUN_ARGV EXIT: 94`
    and exits 94 -- no activation, no command."""
    marker = tmp_path / "ran"
    avoid = tmp_path / "avoid_nodes.txt"
    avoid.write_text("# header\n%s\n" % listed)
    env = _job_env(tmp_path, [sys.executable, "-c", "open(%r, 'w').write('x')" % str(marker)],
                   AVOID_FILE=str(avoid))
    rc, out = _run_job(env)
    if held:
        assert rc == 94 and not marker.exists()
        assert ("RUN_ARGV AVOID: host gn1 is on the avoid list %s (D-102); holding this slot for 0 s, "
                "then exiting 94" % avoid) in out
        assert out.strip().split("\n")[-1] == "RUN_ARGV EXIT: 94"
    else:
        assert rc == 0 and marker.exists() and "RUN_ARGV AVOID" not in out


def test_run_argv_absent_avoid_list_runs(tmp_path):
    marker = tmp_path / "ran"
    env = _job_env(tmp_path, [sys.executable, "-c", "open(%r, 'w').write('x')" % str(marker)],
                   AVOID_FILE=str(tmp_path / "nope.txt"))
    rc, out = _run_job(env)
    assert rc == 0 and marker.exists()


def test_run_argv_environment_fault_96(tmp_path):
    """D-111 (b): a stage job whose environment cannot start prints the
    FAULT line, holds AVOID_HOLD_S s, prints `RUN_ARGV EXIT: 96` last and
    exits 96; the command does not run."""
    marker = tmp_path / "ran"
    wr = tmp_path / "work"
    wr.mkdir()
    env = _job_env(tmp_path, [sys.executable, "-c", "open(%r, 'w').write('x')" % str(marker)],
                   host="broken", SKIP_CONDA="0", JOBS_DIR=JOBS, WORK_ROOT=str(wr),
                   CHAIN_NAME="pl_toy", ROLE="stage")
    rc, out = _run_job(env)
    assert rc == 96, out
    assert not marker.exists()
    assert "RUN_ARGV ENV FAULT: host broken" in out
    assert out.strip().split("\n")[-1] == "RUN_ARGV EXIT: 96"


def test_run_argv_without_jobs_dir_old_path_98(tmp_path):
    """D-111 (b): without JOBS_DIR (a job of the code before the amendment)
    on a node without conda: the two lines of before -- exit 98 (block 7c's
    oracle: 'A job without JOBS_DIR on a node without conda exits 98')."""
    env = _job_env(tmp_path, ["true"], SKIP_CONDA="0")
    rc, out = _run_job(env)
    assert rc == 98, out


def test_run_argv_workdir_and_argv_file(tmp_path):
    """Block 7 / D-111: exit 90 when WORKDIR cannot be entered, 91 when the
    argv file is absent."""
    env = _job_env(tmp_path, ["true"], WORKDIR=str(tmp_path / "absent"))
    assert _run_job(env)[0] == 90
    env = _job_env(tmp_path, ["true"], ARGV_FILE=str(tmp_path / "absent.argv"))
    assert _run_job(env)[0] == 91


# -------------------------------------- boundary: block 22's eval_run.pbs ---

def test_eval_run_uses_the_activation_contract(tmp_path):
    """Boundary block 7 -> block 22: eval_run.pbs sources jobs/activate_env.sh
    and calls pipeline_activate; the activation's 96 (nothing usable) is the
    job's 96, before the command (block 22: 'exits 90, 91, 95, 96 or the
    command's; last line `EVAL JOB EXIT: <n>`')."""
    if not os.path.isfile(EVAL_RUN):
        pytest.skip("no eval_run.pbs at this commit")
    marker = tmp_path / "ran"
    env = _job_env(tmp_path, [sys.executable, "-c", "open(%r, 'w').write('x')" % str(marker)],
                   host="broken", SKIP_CONDA="0", JOBS_DIR=JOBS, WORK_ROOT=str(tmp_path),
                   ROADMAP_DISABLE="1", WORKDIR=REPO)
    rc, out = _run_job(env, EVAL_RUN)
    assert rc == 96, out
    assert not marker.exists()
    assert "RUN_ARGV ENV FAULT: host broken" in out
    assert out.strip().split("\n")[-1] == "EVAL JOB EXIT: 96"
