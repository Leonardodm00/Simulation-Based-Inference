"""Running a command in one of the project's conda environments on the
login node (block 7): the GP proposal, the finalists, the bank's split hash
and the ensemble NLL run in `envs.train`, the sequencer itself in
`envs.extract`.

The activation is the proven block of the job scripts: `set +u` across the
hook and the activate (third-party activate.d scripts are not -u clean),
`set -u` after. With PIPELINE_SKIP_CONDA=1 (the tests, or a shell already
in the right environment) the command runs as it is.

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

import os
import subprocess

__all__ = ["run_in_env", "python_in_env", "EnvError"]

_HOOK = ('set +u; eval "$(conda shell.bash hook)" || exit 97; '
         'conda activate "$1" || exit 98; set -u; shift; exec "$@"')


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
    variables). Exit 97 / 98 (the hook / the activation failed) raise
    EnvError."""
    argv = [str(a) for a in argv]
    if skip_conda():
        cmd = argv
    else:
        cmd = ["bash", "-c", _HOOK, "_", str(env_name)] + argv
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
    if proc.returncode in (97, 98) and not skip_conda():
        raise EnvError("conda %s failed for environment %s: %s"
                       % ("hook" if proc.returncode == 97 else "activate", env_name,
                          out.strip()[-500:]))
    return proc.returncode, out
