#!/usr/bin/env python3
"""A fault-injecting stand-in for `hpc/joint/stage3/run_joint_arms.py`, for
the pipeline's block-7b tests only (not pipeline code).

`test_smoke_block7b.py` copies `hpc/` into a temporary tree and installs this
file there as `joint/stage3/run_joint_arms.py`, with `REAL_RUNNER` set to the
repository's runner. Imported as a module (the split tool imports
`grouped_split`), it re-exports the real runner. Run as a script (a trial's
or a member's runner), it reads `fault_plan.json` beside itself and acts by
the plan of the run folder (`--out-dir`):

    trials   {"trials": [[a0, a1, ...], ...]}: the i-th trial run folder seen
             (the order of first calls; the tests throttle at 1) takes its
             actions in turn, one per call, the last repeated
    members  {"members": {"member_seed<k>": [a0, ...]}}
    probes   (`--probe-steps` in the argv): always the real runner

Actions: ok (the real runner); sigbus (the process kills itself with
SIGBUS); eio, eshutdown (an uncaught OSError [Errno 5] / BrokenPipeError
[Errno 108], as davinci printed them on 2026-10-18); value (an uncaught
ValueError: not a node fault); stop1_flip / stop1_trunc (the real runner
with --stop-after-epoch 1, then the checkpoint corrupted: one byte flipped
inside a member, or the file cut in half); sleep<N>_ok (sleep N s, then ok).

Pure ASCII, LF only; stdlib only.
"""

import errno
import glob
import importlib.util
import json
import os
import runpy
import signal
import sys
import time
import zipfile

REAL_RUNNER = "@@REAL_RUNNER@@"

if __name__ != "__main__":
    _spec = importlib.util.spec_from_file_location("_real_run_joint_arms", REAL_RUNNER)
    _real = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_real)
    globals().update(dict((k, v) for k, v in _real.__dict__.items() if not k.startswith("__")))


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _out_dir(argv):
    if "--out-dir" in argv:
        return argv[argv.index("--out-dir") + 1]
    return ""


def _next_action(out_dir):
    plan = json.load(open(os.path.join(_here(), "fault_plan.json")))
    state = os.path.join(_here(), "fault_state")
    if not os.path.isdir(state):
        os.makedirs(state)
    name = os.path.basename(os.path.normpath(out_dir))
    if name.startswith("member_seed"):
        actions = plan.get("members", {}).get(name, ["ok"])
    else:
        order_path = os.path.join(state, "order.json")
        order = json.load(open(order_path)) if os.path.isfile(order_path) else []
        if name not in order:
            order.append(name)
            with open(order_path, "w") as fh:
                json.dump(order, fh)
        i = order.index(name)
        trials = plan.get("trials", [])
        actions = trials[i] if i < len(trials) else ["ok"]
    count_path = os.path.join(state, "attempts_" + name)
    n = int(open(count_path).read()) if os.path.isfile(count_path) else 0
    with open(count_path, "w") as fh:
        fh.write(str(n + 1))
    return actions[min(n, len(actions) - 1)], n


def _run_real(extra=()):
    sys.argv = [REAL_RUNNER] + sys.argv[1:] + list(extra)
    try:
        runpy.run_path(REAL_RUNNER, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    return 0


def _corrupt(out_dir, how):
    ck = sorted(glob.glob(os.path.join(out_dir, "*_train_ckpt.pt")))
    if not ck:
        sys.stderr.write("fault_runner: no checkpoint to corrupt in %s\n" % out_dir)
        sys.exit(9)
    p = ck[0]
    with open(p, "rb") as fh:
        raw = bytearray(fh.read())
    if how == "trunc":
        raw = raw[:len(raw) // 2]
    else:
        with zipfile.ZipFile(p) as z:
            info = max(z.infolist(), key=lambda i: i.file_size)
        off = info.header_offset
        n_name = int.from_bytes(raw[off + 26:off + 28], "little")
        n_extra = int.from_bytes(raw[off + 28:off + 30], "little")
        at = off + 30 + n_name + n_extra + min(1000, info.file_size // 2)
        raw[at] ^= 0xFF
    with open(p, "wb") as fh:
        fh.write(bytes(raw))
    print("fault_runner: checkpoint %s corrupted (%s)" % (p, how), flush=True)


def main():
    argv = sys.argv[1:]
    if "--probe-steps" in argv:
        return _run_real()
    out_dir = _out_dir(argv)
    action, n = _next_action(out_dir)
    print("fault_runner: %s attempt %d -> %s" % (os.path.basename(out_dir), n, action), flush=True)
    if action == "ok":
        return _run_real()
    if action.startswith("sleep") and action.endswith("_ok"):
        time.sleep(float(action[len("sleep"):-len("_ok")]))
        return _run_real()
    if action == "sigbus":
        os.kill(os.getpid(), signal.SIGBUS)
        time.sleep(30)
        return 99
    if action == "eio":
        raise OSError(errno.EIO, os.strerror(errno.EIO), "/x/banks/synth_sim/shard_0064.npz")
    if action == "eshutdown":
        raise BrokenPipeError(errno.ESHUTDOWN, "Cannot send after transport endpoint shutdown",
                              "/x/SBI/hpc/dsn")
    if action == "value":
        raise ValueError("Expected parameter loc (Tensor of shape (16, 23)) to satisfy the "
                         "constraint Real(), but found invalid values")
    if action in ("stop1_flip", "stop1_trunc"):
        rc = _run_real(["--stop-after-epoch", "1"])
        if rc != 0:
            return rc
        _corrupt(out_dir, "flip" if action == "stop1_flip" else "trunc")
        return 0
    sys.stderr.write("fault_runner: unknown action %r\n" % action)
    return 98


if __name__ == "__main__":
    sys.exit(main())
