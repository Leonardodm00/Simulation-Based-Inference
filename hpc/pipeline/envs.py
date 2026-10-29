"""Running a command in one of the project's conda environments on the
login node (block 7): the GP proposal, the finalists, the bank's split hash
and the ensemble NLL run in `envs.train`, the sequencer itself in
`envs.extract`.

The activation is the job scripts' own, `jobs/activate_env.sh` (D-111): conda
first -- `set +u` across the hook and the activate (third-party activate.d
scripts are not -u clean), `set -u` after --, and when conda is absent from
the node or fails there, the environment's own folder (its recorded prefix,
`$HOME/.conda/envs/<env>`, CONDA_ENVS_PATH). Neither: EnvError. With
PIPELINE_SKIP_CONDA=1 (the tests, or a shell already in the right
environment) the command runs as it is.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import os
import subprocess

__all__ = ["run_in_env", "python_in_env", "EnvError"]

_HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVATE = os.path.join(_HERE, "jobs", "activate_env.sh")
# `bash -c` with the environment's name as $1 and the command after it; the
# activation's own status (96: the environment cannot start here) is the exit
_HOOK = ('. "$PIPELINE_ACTIVATE" || exit 97; pipeline_activate "$1" || exit $?; '
         'shift; exec "$@"')
# the exits that mean the command never ran: the activation file (97), the
# environment (96: activate_env.sh), the hook / activation before D-111 (97, 98)
ENV_EXITS = (96, 97, 98)


class EnvError(RuntimeError):
    """The command could not be run in the environment (the hook or the
    activation failed), as opposed to the command's own failure."""


def skip_conda():
    return os.environ.get("PIPELINE_SKIP_CONDA", "0") == "1"


def python_in_env():
    """The interpreter name the command lines use inside an environment."""
    return os.environ.get("PIPELINE_PYTHON", "python") if not skip_conda() \
        else os.environ.get("PIPELINE_PYTHON", "python3")


def run_in_env(env_name, argv, cwd=None, log_path=None, timeout=None, env=None):
    """Run `argv` inside `env_name`; return (returncode, output text).
    stdout and stderr are merged, appended to `log_path` when given, and
    returned. `env` replaces the process environment (a launcher's
    variables). The environment that cannot start (96, or 97 / 98: the
    activation's file, or conda before D-111) raises EnvError. WORK_ROOT,
    from `env` or this process's environment, is where the activation
    records the prefix conda gave."""
    argv = [str(a) for a in argv]
    if skip_conda():
        cmd = argv
    else:
        cmd = ["bash", "-c", _HOOK, "_", str(env_name)] + argv
        env = dict(os.environ if env is None else env)
        env["PIPELINE_ACTIVATE"] = ACTIVATE
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=timeout)
    out = proc.stdout or ""
    if log_path:
        d = os.path.dirname(log_path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(log_path, "a") as fh:
            fh.write("+ %s\n" % " ".join(argv))
            fh.write(out)
            if not out.endswith("\n"):
                fh.write("\n")
            fh.write("= exit %d\n" % proc.returncode)
    if proc.returncode in ENV_EXITS and not skip_conda():
        raise EnvError("the environment %s cannot start on this node (exit %d): %s"
                       % (env_name, proc.returncode, out.strip()[-500:]))
    return proc.returncode, out
