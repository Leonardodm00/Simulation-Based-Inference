#!/usr/bin/env python3
"""Smoke test of block 39 (D-169): the runner's --device option.

Run (CPU node or login node; the GPU checks are SKIPPED and say so):
    cd hpc/joint/stage3 && python3 smoke_test_device.py
Run inside a GPU job (every check runs):
    cd hpc/joint/stage3 && python3 smoke_test_device.py
The last line is `DEVICE SUITE: <p> pass, <f> fail, <s> skip`; exit 1 on a
failure. A run on a GPU must end with 0 skip.

Oracles
-------
V1  the option: default cpu, cuda accepted, anything else refused (argparse).
V2  the refusal: --device cuda without a visible GPU exits with a message
    naming torch's version; nothing is trained (no fallback to the CPU).
V3  the CPU path is the path of before: `_on_device` returns its argument
    itself, and a model built with device=None and one built with
    device="cpu" under one seed have the same state digest.
V4  the checkpoint holds CPU tensors and its digests are those of the
    model's state.
V5  end to end on the CPU: the runner with `--device cpu` and without the
    option writes the same record, per-row arrays and model digest (arms A1,
    A2, A5); the probe's record carries device cpu and None GPU fields.
V6  [GPU] a model built on the GPU takes CPU batches; its loss on a batch
    agrees with a CPU model of the same weights within 1e-3 nats per row
    (float32, different convolution and spline kernels on the two devices;
    the losses are of order 1 to 10).
V7  [GPU] the runner trains arms A1, A2 and A5 on the GPU end to end; the
    checkpoint it writes holds CPU tensors and loads into a CPU-built model
    with the recorded digest; a run stopped after an epoch resumes.
V8  [GPU] the probe on the GPU records the device's name and a positive
    peak of allocated memory.
V9  device_probe_argv.py: a trial's command (with --resume) becomes a probe
    of the same configuration that writes only into the new folder; the
    flags it must not touch are unchanged; its three refusals refuse. The
    converted command is executed (on the GPU when there is one).
V10 --probe-full-loss (arm A5; on the GPU when there is one): the probe's
    steps call the replicate term once each; without the flag never.

Pure ASCII, LF only.
"""

import contextlib
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ("", os.path.join("..", "stage2"), os.path.join("..", "stage1"), ".."):
    _p = os.path.abspath(os.path.join(_HERE, _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_joint_arms as RJ                                 # noqa: E402
from joint_model import build_joint_model, state_sha256    # noqa: E402

CONFIG = {
    "seed": 0,
    "d_theta": 5, "W": 64, "E": 8, "n_rows": 48,
    "loss_tol_nats": 1e-3,          # V6, justified in the docstring
    "arms": ("A1", "A2", "A5"),
    "runner_args": ["--epochs", "2", "--steps-per-epoch", "3", "--b-sim", "24",
                    "--n-posterior-draws", "32", "--threads", "1"],
}

RESULTS = []
HAS_GPU = torch.cuda.is_available()


def ok(name, cond, detail=""):
    RESULTS.append(("PASS" if cond else "FAIL", name, detail))
    print("[%s] %s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)


def skip(name, why):
    RESULTS.append(("SKIP", name, why))
    print("[SKIP] %s -- %s" % (name, why), flush=True)


class _Backbone(torch.nn.Module):
    """A small GroupNorm encoder (no DSN tree needed)."""

    def __init__(self, E):
        super().__init__()
        self.c1 = torch.nn.Conv1d(1, 8, 5, stride=4, padding=2)
        self.n1 = torch.nn.GroupNorm(4, 8)
        self.fc = torch.nn.Linear(8, E)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = torch.relu(self.n1(self.c1(x))).mean(-1)
        return torch.nn.functional.normalize(self.fc(h), dim=-1)


def _model(device=None, seed=0):
    from sbi.utils import BoxUniform
    c = CONFIG
    g = torch.Generator().manual_seed(seed + 100)
    theta = torch.rand(c["n_rows"], c["d_theta"], generator=g)
    x = torch.rand(c["n_rows"], c["W"], generator=g)
    lo, hi = torch.zeros(c["d_theta"]), torch.ones(c["d_theta"])
    if device is not None and torch.device(device).type != "cpu":
        lo, hi = lo.to(device), hi.to(device)
    torch.manual_seed(seed)
    bb = _Backbone(c["E"])
    m = build_joint_model(bb, BoxUniform(low=lo, high=hi), theta, x,
                          hidden_features=16, num_transforms=2, device=device)
    return m, theta, x


def check_v1():
    p = RJ.build_parser()
    base = ["--arm", "A1", "--sim-shards", "x", "--out-dir", "y"]
    ok("V1 default is cpu", p.parse_args(base).device == "cpu")
    ok("V1 cuda accepted", p.parse_args(base + ["--device", "cuda"]).device == "cuda")
    try:
        with open(os.devnull, "w") as dn:
            old = sys.stderr
            sys.stderr = dn
            try:
                p.parse_args(base + ["--device", "gpu0"])
            finally:
                sys.stderr = old
        ok("V1 another value refused", False, "accepted gpu0")
    except SystemExit as exc:
        ok("V1 another value refused", exc.code == 2)


def check_v2():
    ok("V2 cpu resolves", RJ.resolve_device("cpu") == torch.device("cpu"))
    if HAS_GPU:
        ok("V2 cuda resolves on a GPU node", RJ.resolve_device("cuda").type == "cuda")
        return
    try:
        RJ.resolve_device("cuda")
        ok("V2 cuda refused without a GPU", False, "no refusal")
    except SystemExit as exc:
        msg = str(exc)
        ok("V2 cuda refused without a GPU",
           "torch sees no GPU" in msg and torch.__version__ in msg, msg[:60])


def check_v3():
    m, theta, x = _model(None)
    ok("V3 _on_device is the identity on the CPU", m._on_device(x) is x and m._on_device(theta) is theta)
    m2, _, _ = _model("cpu")
    ok("V3 device=None and device='cpu' build one model",
       state_sha256(m.state_dict()) == state_sha256(m2.state_dict()))
    ok("V3 the model reports its device", m.device == torch.device("cpu"))


def check_v4():
    m, _, _ = _model("cuda" if HAS_GPU else None)
    ck = m.checkpoint()
    on_cpu = all(v.device.type == "cpu" for v in ck["model_state"].values()) and \
        all(v.device.type == "cpu" for v in ck["encoder_state"].values())
    ok("V4 the checkpoint's tensors are on the CPU", on_cpu,
       "(model on %s)" % m.device.type)
    ok("V4 its digests are the state's",
       ck["model_sha256"] == state_sha256(m.state_dict())
       and ck["encoder_sha256"] == state_sha256(m.encoder.state_dict()))
    m_cpu, _, _ = _model(None, seed=7)       # other weights, CPU-built
    m_cpu.load_checkpoint(ck)
    ok("V4 it loads into a CPU-built model with the recorded digest",
       state_sha256(m_cpu.state_dict()) == ck["model_sha256"])


def _banks(tmp):
    s1 = os.path.abspath(os.path.join(_HERE, "..", "stage1"))
    for arm, n, extra in (("S", 40, ["--seed", "1"]),
                          ("R", 16, ["--seed", "2", "--pi", "0.4",
                                     "--gap-modes", "range_shift,drift"])):
        cmd = [sys.executable, os.path.join(s1, "build_latent_bank.py"),
               "--out-dir", os.path.join(tmp, arm), "--arm", arm,
               "--n-traces", str(n), "--wells-per-donor", "2",
               "--n-windows", "4", "--T-win", "1.28", "--fs", "50",
               "--n-neurons", "20", "--n-classes", "3",
               "--provider", "reference"] + extra
        r = subprocess.run(cmd, cwd=s1, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("bank %s: %s" % (arm, r.stderr.strip()[-300:]))


def _run(tmp, arm, out, extra):
    cmd = [sys.executable, os.path.join(_HERE, "run_joint_arms.py"),
           "--arm", arm, "--seed", str(CONFIG["seed"]),
           "--sim-shards", os.path.join(tmp, "S", "*.npz"),
           "--real-shards", os.path.join(tmp, "R", "*.npz"),
           "--out-dir", out] + CONFIG["runner_args"] + list(extra)
    return subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)


def _result(out, arm):
    stem = os.path.join(out, "%s_seed%d" % (arm, CONFIG["seed"]))
    rec = json.load(open(stem + ".json"))
    rec.pop("config")
    pr = np.load(stem + "_perrow.npz")
    ck = torch.load(stem + "_ckpt.pt", map_location="cpu", weights_only=False)
    return (json.dumps(rec, sort_keys=True, default=str),
            {k: pr[k].tobytes() for k in pr.files}, ck)


def check_v5(tmp):
    for arm in CONFIG["arms"]:
        ra = _run(tmp, arm, os.path.join(tmp, "plain"), [])
        rb = _run(tmp, arm, os.path.join(tmp, "cpu"), ["--device", "cpu"])
        if ra.returncode or rb.returncode:
            ok("V5 %s runs on the CPU" % arm, False, (ra.stderr + rb.stderr).strip()[-200:])
            continue
        a, b = _result(os.path.join(tmp, "plain"), arm), _result(os.path.join(tmp, "cpu"), arm)
        ok("V5 %s: --device cpu is the run without the option" % arm,
           a[0] == b[0] and a[1] == b[1] and a[2]["model_sha256"] == b[2]["model_sha256"])
        ok("V5 %s: the log names the device" % arm, "device           : cpu" in rb.stdout)
    r = _run(tmp, "A1", os.path.join(tmp, "probe_cpu"), ["--probe-steps", "3", "--probe-val-rows", "8"])
    files = glob.glob(os.path.join(tmp, "probe_cpu", "*_probe.json"))
    if r.returncode or len(files) != 1:
        ok("V5 the probe runs on the CPU", False, r.stderr.strip()[-200:])
        return
    rec = json.load(open(files[0]))
    ok("V5 the CPU probe's device fields",
       rec["device"] == "cpu" and rec["gpu_name"] is None
       and rec["gpu_peak_allocated_gb"] is None and rec["step_seconds"] > 0)


def check_v10(tmp):
    """--probe-full-loss: the probe's steps call the arm's replicate term;
    without the flag they never do (the budget gate's probe, unchanged)."""
    calls = {"n": 0}
    orig = RJ.ReplicateConsistencyLoss.forward

    def counted(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    dev = "cuda" if HAS_GPU else "cpu"
    seen = {}
    RJ.ReplicateConsistencyLoss.forward = counted
    try:
        for name, extra in (("gate", []), ("full", ["--probe-full-loss"])):
            calls["n"] = 0
            out = os.path.join(tmp, "v10_" + name)
            argv = ["--arm", "A5", "--seed", str(CONFIG["seed"]),
                    "--sim-shards", os.path.join(tmp, "S", "*.npz"),
                    "--real-shards", os.path.join(tmp, "R", "*.npz"),
                    "--out-dir", out, "--device", dev] + CONFIG["runner_args"] \
                + ["--probe-steps", "3", "--probe-val-rows", "8"] + extra
            cwd = os.getcwd()
            os.chdir(_HERE)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = RJ.main(argv)
            finally:
                os.chdir(cwd)
            files = glob.glob(os.path.join(out, "*_probe.json"))
            rec = json.load(open(files[0])) if len(files) == 1 else {}
            seen[name] = (rc, calls["n"], rec.get("probe_full_loss"))
    except BaseException as exc:                      # SystemExit included
        ok("V10 the two probes of A5 run", False, repr(exc)[:200])
        return
    finally:
        RJ.ReplicateConsistencyLoss.forward = orig
    ok("V10 the gate's probe never calls the replicate term",
       seen["gate"] == (0, 0, False) or seen["gate"] == (None, 0, False), str(seen["gate"]))
    ok("V10 --probe-full-loss calls it at every one of the 3 steps",
       seen["full"][1] == 3 and seen["full"][2] is True and seen["full"][0] in (0, None),
       str(seen["full"]))


def check_v6():
    if not HAS_GPU:
        skip("V6 the GPU model against a CPU model of the same weights", "torch sees no GPU")
        return
    mg, theta, x = _model("cuda")
    mc, _, _ = _model(None, seed=3)
    mc.load_checkpoint(mg.checkpoint())
    mg.eval()
    mc.eval()
    with torch.no_grad():
        lg = mg.npe_loss(theta, x)
        lc = mc.npe_loss(theta, x)
        z = mg.encode(x)
        s = mg.sample_posterior(4, x[:3])
    rs = mg.rsample_posterior(4, x[:3])
    ok("V6 CPU batches are consumed on the GPU",
       lg.device.type == "cuda" and z.device.type == "cuda"
       and s.device.type == "cuda" and rs.device.type == "cuda"
       and tuple(s.shape) == (3, 4, CONFIG["d_theta"]))
    gap = float((lg.cpu() - lc).abs().max())
    ok("V6 the loss agrees with the CPU model's within %.0e" % CONFIG["loss_tol_nats"],
       bool(torch.isfinite(lg).all()) and gap < CONFIG["loss_tol_nats"], "max gap %.2e" % gap)


def check_v7(tmp):
    if not HAS_GPU:
        skip("V7 arms A1, A2, A5 train on the GPU end to end", "torch sees no GPU")
        return
    out = os.path.join(tmp, "gpu")
    for arm in CONFIG["arms"]:
        r = _run(tmp, arm, out, ["--device", "cuda"])
        if r.returncode:
            ok("V7 %s trains on the GPU" % arm, False, r.stderr.strip()[-300:])
            continue
        rec, _pr, ck = _result(out, arm)
        fin = np.isfinite(json.loads(rec)["L"])
        ok("V7 %s trains on the GPU" % arm, bool(fin) and "device           : cuda" in r.stdout)
        ok("V7 %s: its checkpoint holds CPU tensors" % arm,
           all(v.device.type == "cpu" for v in ck["model_state"].values()))
    out2 = os.path.join(tmp, "gpu_resume")
    r1 = _run(tmp, "A5", out2, ["--device", "cuda", "--stop-after-epoch", "0"])
    r2 = _run(tmp, "A5", out2, ["--device", "cuda", "--resume"])
    ok("V7 a stopped GPU run resumes to a record",
       r1.returncode == 0 and r2.returncode == 0 and "resumed from" in r2.stdout
       and os.path.isfile(os.path.join(out2, "A5_seed%d.json" % CONFIG["seed"])),
       (r1.stderr + r2.stderr).strip()[-200:])


def check_v8(tmp):
    if not HAS_GPU:
        skip("V8 the probe on the GPU", "torch sees no GPU")
        return
    out = os.path.join(tmp, "probe_gpu")
    r = _run(tmp, "A1", out, ["--device", "cuda", "--probe-steps", "3", "--probe-val-rows", "8"])
    files = glob.glob(os.path.join(out, "*_probe.json"))
    if r.returncode or len(files) != 1:
        ok("V8 the probe runs on the GPU", False, r.stderr.strip()[-300:])
        return
    rec = json.load(open(files[0]))
    ok("V8 the GPU probe's device fields",
       rec["device"] == "cuda" and bool(rec["gpu_name"])
       and rec["gpu_peak_allocated_gb"] > 0 and "PROBE GPU:" in r.stdout,
       "%s, %.3f GiB" % (rec["gpu_name"], rec["gpu_peak_allocated_gb"]))


def check_v9(tmp):
    import device_probe_argv as DP
    src_out = os.path.join(tmp, "v9_source")
    src = ["/some/env/bin/python3", "/old/checkout/hpc/joint/stage3/run_joint_arms.py",
           "--arm", "A5", "--seed", str(CONFIG["seed"]),
           "--sim-shards", os.path.join(tmp, "S", "*.npz"),
           "--real-shards", os.path.join(tmp, "R", "*.npz"),
           "--out-dir", src_out] + CONFIG["runner_args"] + ["--resume"]
    src_file = os.path.join(tmp, "v9_trial.argv")
    with open(src_file, "w") as fh:
        fh.write("\n".join(src) + "\n")
    new_out, new_file = os.path.join(tmp, "v9_probe"), os.path.join(tmp, "v9_probe.argv")
    dev = "cuda" if HAS_GPU else "cpu"
    cmd = [sys.executable, os.path.join(_HERE, "device_probe_argv.py"), "--source", src_file,
           "--out-dir", new_out, "--argv-file", new_file, "--device", dev, "--threads", "2",
           "--probe-steps", "3", "--probe-val-rows", "8", "--python", sys.executable]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not os.path.isfile(new_file):
        ok("V9 the conversion runs", False, (r.stdout + r.stderr).strip()[-200:])
        return
    new = DP.read_argv(new_file)
    kept = all(DP.flag_value(new, f) == DP.flag_value(src, f)
               for f in ("--arm", "--seed", "--sim-shards", "--real-shards", "--epochs",
                         "--steps-per-epoch", "--b-sim", "--n-posterior-draws"))
    ok("V9 the configuration's flags are kept", kept)
    ok("V9 what must change has changed",
       DP.flag_value(new, "--out-dir") == new_out and DP.flag_value(new, "--device") == dev
       and DP.flag_value(new, "--threads") == "2" and DP.flag_value(new, "--probe-steps") == "3"
       and "--resume" not in new and new[0] == sys.executable
       and new[1] == os.path.join(_HERE, "run_joint_arms.py"))
    r2 = subprocess.run(new, cwd=_HERE, capture_output=True, text=True)
    files = glob.glob(os.path.join(new_out, "*_probe.json"))
    ok("V9 the converted command runs and writes only into the new folder",
       r2.returncode == 0 and len(files) == 1 and not os.path.exists(src_out),
       r2.stderr.strip()[-200:])
    refusals = []
    r = subprocess.run(cmd, capture_output=True, text=True)             # the argv file exists now
    refusals.append(r.returncode == 2 and "REFUSE" in r.stdout)
    for bad_src, bad_out in ((src_file, src_out), (None, new_out + "2")):
        f = src_file
        if bad_src is None:
            f = os.path.join(tmp, "v9_other.argv")
            with open(f, "w") as fh:
                fh.write("python3\nsomething_else.py\n--out-dir\nx\n")
        c = [sys.executable, os.path.join(_HERE, "device_probe_argv.py"), "--source", f,
             "--out-dir", bad_out, "--argv-file", os.path.join(tmp, "v9_never.argv")]
        r = subprocess.run(c, capture_output=True, text=True)
        refusals.append(r.returncode == 2 and "REFUSE" in r.stdout
                        and not os.path.exists(os.path.join(tmp, "v9_never.argv")))
    ok("V9 an existing file, the source's own folder and a foreign command are refused",
       all(refusals), str(refusals))


def main():
    print("torch %s, CUDA build %s, GPU visible: %s"
          % (torch.__version__, torch.version.cuda, HAS_GPU), flush=True)
    check_v1()
    check_v2()
    check_v3()
    check_v4()
    tmp = tempfile.mkdtemp()
    try:
        _banks(tmp)
        check_v5(tmp)
        check_v6()
        check_v7(tmp)
        check_v8(tmp)
        check_v9(tmp)
        check_v10(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = dict((k, sum(1 for r in RESULTS if r[0] == k)) for k in ("PASS", "FAIL", "SKIP"))
    print("DEVICE SUITE: %d pass, %d fail, %d skip" % (n["PASS"], n["FAIL"], n["SKIP"]), flush=True)
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
