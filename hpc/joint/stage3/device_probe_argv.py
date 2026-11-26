#!/usr/bin/env python3
"""Turn an existing runner command into a probe of the same configuration on
another device (block 39, D-169).

The pipeline writes every trial's and every gate probe's command as an argv
file, one argument per line (hpc/pipeline/jobs/run_argv.pbs). This tool
copies one, changing only where it runs and what it writes:

    --out-dir         the folder given here (never the search's)
    --device          the device given here
    --threads         the cores of the new job
    --probe-steps, --probe-val-rows   the probe's sizes (added when absent,
                      so a trial's command becomes a probe of that trial's
                      configuration)
    --probe-full-loss added with --full-loss: the steps timed carry the arm's
                      whole objective (DSN and replicate terms), not the NPE
                      term alone as the budget gate's probes do
    --resume, --stop-after-epoch      removed (a probe trains nothing durable)
    the interpreter   `python3`, i.e. the one of the environment the new job
                      activates (the source names its own environment's)
    the runner        the run_joint_arms.py beside this tool (the checkout
                      that has --device), not the source's

Every other argument is kept, in order: the configuration timed is the one
the source command names.

    python3 device_probe_argv.py --source <file.argv> --out-dir <dir> \
        --argv-file <new.argv> [--device cuda] [--threads 8] \
        [--probe-steps 200] [--probe-val-rows 2000]

It prints the changes and `PROBE ARGV OK: <new file>`. It refuses (exit 2) a
source that is not a run_joint_arms.py command, an --out-dir equal to the
source's, and an existing --argv-file.

Pure ASCII, LF only.
"""

import argparse
import os
import sys

# flags of run_joint_arms.py that take no value (argparse store_true)
BOOLEAN_FLAGS = ("--resume", "--dry-run")


def read_argv(path):
    with open(path, "r") as fh:
        return [line.rstrip("\n").rstrip("\r") for line in fh if line.strip() != ""]


def flag_value(argv, flag):
    """The value after `flag`, or None when the flag is absent."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def set_flag(argv, flag, value):
    """A copy of argv with `flag value` set: replaced where present (the
    first occurrence; a second one is an error), appended otherwise."""
    if argv.count(flag) > 1:
        raise ValueError("%s appears %d times in the source" % (flag, argv.count(flag)))
    out = list(argv)
    if flag in out:
        out[out.index(flag) + 1] = str(value)
    else:
        out += [flag, str(value)]
    return out


def drop_flag(argv, flag, takes_value):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if takes_value else 1
            continue
        out.append(argv[i])
        i += 1
    return out


def convert(argv, out_dir, device, threads, probe_steps, probe_val_rows,
            python="python3", runner=None, full_loss=False):
    where = [i for i, a in enumerate(argv) if os.path.basename(a) == "run_joint_arms.py"]
    if len(where) != 1 or where[0] > 1:
        raise ValueError("the source is not a run_joint_arms.py command")
    argv = list(argv)
    if where[0] == 1:
        if not os.path.basename(argv[0]).startswith("python"):
            raise ValueError("the source's first argument is not a python interpreter: %r" % argv[0])
        argv[0] = python
    if runner is not None:
        argv[where[0]] = runner
    src_out = flag_value(argv, "--out-dir")
    if src_out is None:
        raise ValueError("the source has no --out-dir")
    if os.path.abspath(src_out) == os.path.abspath(out_dir):
        raise ValueError("--out-dir is the source's own folder (%s)" % src_out)
    out = drop_flag(argv, "--resume", takes_value=False)
    out = drop_flag(out, "--stop-after-epoch", takes_value=True)
    out = set_flag(out, "--out-dir", out_dir)
    out = set_flag(out, "--device", device)
    out = set_flag(out, "--threads", int(threads))
    out = set_flag(out, "--probe-steps", int(probe_steps))
    out = set_flag(out, "--probe-val-rows", int(probe_val_rows))
    out = drop_flag(out, "--probe-full-loss", takes_value=False)
    if full_loss:
        out.append("--probe-full-loss")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--argv-file", required=True)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--probe-steps", type=int, default=200)
    p.add_argument("--probe-val-rows", type=int, default=2000)
    p.add_argument("--full-loss", action="store_true")
    p.add_argument("--python", default="python3")
    p.add_argument("--runner", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_joint_arms.py"))
    a = p.parse_args(argv)
    if os.path.exists(a.argv_file):
        print("REFUSE: %s exists" % a.argv_file)
        return 2
    try:
        src = read_argv(a.source)
        new = convert(src, a.out_dir, a.device, a.threads, a.probe_steps, a.probe_val_rows,
                      python=a.python, runner=a.runner, full_loss=a.full_loss)
    except (OSError, ValueError) as exc:
        print("REFUSE: %s" % exc)
        return 2
    print("%-18s %s -> %s" % ("interpreter", src[0], new[0]))
    print("%-18s %s -> %s" % ("runner", src[1] if len(src) > 1 else None, new[1] if len(new) > 1 else None))
    for flag in ("--out-dir", "--device", "--threads", "--probe-steps", "--probe-val-rows", "--b-sim", "--arm"):
        print("%-18s %s -> %s" % (flag, flag_value(src, flag), flag_value(new, flag)))
    print("%-18s %s" % ("--probe-full-loss", "--probe-full-loss" in new))
    d = os.path.dirname(os.path.abspath(a.argv_file))
    os.makedirs(d, exist_ok=True)
    with open(a.argv_file, "w") as fh:
        fh.write("\n".join(new) + "\n")
    print("PROBE ARGV OK: %s (%d arguments)" % (a.argv_file, len(new)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
