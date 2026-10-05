#!/usr/bin/env python3
"""e4_numbers.py -- every [RAN] number of E4_JOINT_OBJECTIVE_AND_LOOP.md, recomputed.

Torch-free: numpy and scipy only. E4 states what the three terms of the
joint objective estimate, which weights each term reaches, what a loss
weight does under AdamW and the gradient clip, what the per-epoch
gradient-cosine record can resolve, and what each arm isolates. Every
number it quotes is printed here.

Sources, kept apart as in e3_numbers.py:

  * [REPO] read at the freeze commit with `git show` and executed alone (AST
    extraction; nothing else of the module runs, because the modules import
    torch, which the sandbox does not have): `run_joint_arms.grouped_split`
    and `run_joint_arms.arm_config`. The runner's defaults are read from its
    argument parser with `ast` (no import); the arms job's arm order and
    variables from its text.
  * [REPO] hpc/joint is checked byte-identical to the freeze (`git diff
    --quiet`), then the Stage 1 command line builds one default bench shard
    (`build_latent_bank.py --provider bench`) in a temporary directory, and
    `latent_bank.concat_shards` reads it back.
  * a numpy TRANSCRIPTION of `FixedStatsSummary.forward`
    (run_joint_arms.py:96-111) with torch's semantics for `std` (Bessel's
    correction) and `quantile` (linear interpolation), in float64 where the
    runner computes in float32.
  * the AdamW step and the clip coefficient of p5_numbers.py (a replica of
    torch 2.10.0's `_single_tensor_adam`, decoupled branch, and of
    `clip_grad_norm_`), imported, not copied.
  * a TOY for B2: fixed gradient sequences that do not depend on the weights
    (open loop). Its numbers describe the optimiser's arithmetic on given
    gradients, not a training run.

Blocks:
  B0  the freeze: hpc/joint byte-identical to 834eb41; the code read
  B1  the streams' laws: metric-batch composition, class shares against the
      bank's, repeated rows inside a batch, draws per row over a run; the
      rows the A0s pre-training draws from, against the grouped split (F-be)
  B2  what a loss weight reaches under AdamW: omega without the clip, the
      clip's coupling, psi's two limits and the transition between them (F-bd)
  B3  what the rho_grad record can resolve: exact binomial intervals for the
      fraction of negative probes, and the chance of seeing none
  B4  A_ref's eight fixed features on a default bench shard (F-bg)
  B5  the arms as configured: arm_config, the streams each arm draws, the
      job's index map, and what one submission fixes (F-bf)

Run from hpc/joint/docs/tools (needs git and the repository's history):

    python e4_numbers.py

Every block prints only machine-independent quantities (no paths, no
timings), so two runs give identical output.

Pure ASCII, LF only.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True      # importing the stage1 modules must not leave caches

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # repo root
_STAGE1 = os.path.join(_ROOT, "hpc", "joint", "stage1")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from p5_numbers import adamw_step, clip_coef          # noqa: E402  (torch 2.10.0 replica)

FREEZE = "834eb41"
RUNNER = "hpc/joint/stage3/run_joint_arms.py"
TRAIN = "hpc/joint/stage2/joint_train.py"
ARMS_PBS = "hpc/joint/stage3/jobs/joint_arms.pbs"

# [KB] class counts of the two cohorts (E4 S3.2 gives the sources)
COUNTS_DUP15HD = (918, 972)     # HPC_PATHS.md: 17 control and 18 pathological wells x 54 windows
COUNTS_GIULIA = (72, 108, 108)  # decision log D-056: by the tiling rule (reasoning there)
N_SEED = 5                      # plan Stage 3: n_seed = 5 per arm
N_BANK_SHARDS = 32              # the bank job's documented array (P7)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", _ROOT] + list(args), check=False,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _show(path: str) -> str:
    out = _git("show", "%s:%s" % (FREEZE, path))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode())
    return out.stdout.decode("utf-8")


def _load(path: str, names: tuple, extra_ns=None):
    """Execute the named top-level functions of `path` as read at the freeze."""
    src = _show(path)
    tree = ast.parse(src)
    ns = {"np": np, "hashlib": hashlib, "json": json}
    ns.update(extra_ns or {})
    lines = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            seg = ast.get_source_segment(src, node)
            exec(compile(seg, path, "exec"), ns)      # noqa: S102 -- the repository's own code
            lines[node.name] = (node.lineno, node.end_lineno)
    missing = [n for n in names if n not in lines]
    if missing:
        raise RuntimeError("not found in %s: %r" % (path, missing))
    return ns, lines


def _parser_defaults(src: str) -> dict:
    """{flag: default} of every add_argument call inside build_parser."""
    out = {}
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_parser":
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "add_argument" and call.args
                        and isinstance(call.args[0], ast.Constant)):
                    for kw in call.keywords:
                        if kw.arg == "default":
                            out[call.args[0].value] = ast.literal_eval(kw.value)
    return out


def _init_defaults(src: str, cls: str) -> dict:
    """{arg: default} of the __init__ of class `cls`."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    a = item.args
                    names = [x.arg for x in a.args][-len(a.defaults):]
                    return {n: ast.literal_eval(d) for n, d in zip(names, a.defaults)}
    raise RuntimeError("no %s.__init__" % cls)


def hr(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# B0  the freeze
# --------------------------------------------------------------------------- #
def block_b0() -> dict:
    hr("B0  the freeze: hpc/joint at %s, and the code read" % FREEZE)
    dif = _git("diff", "--quiet", FREEZE, "--", "hpc/joint/stage1", "hpc/joint/stage2",
               "hpc/joint/stage3", "hpc/joint/stage4")
    if dif.returncode != 0:
        raise SystemExit("hpc/joint differs from %s: the numbers would not describe the freeze"
                         % FREEZE)
    print("hpc/joint/stage1..stage4 byte-identical to %s: True" % FREEZE)
    rsrc = _show(RUNNER)
    d = _parser_defaults(rsrc)
    keys = ("--epochs", "--steps-per-epoch", "--b-sim", "--b-met", "--b-rep", "--lr",
            "--weight-decay", "--lambda-dsn", "--lambda-rep", "--warmup-frac-rep",
            "--n-posterior-draws", "--encoder-steps", "--embedding-size", "--one-minus-beta1")
    print("runner defaults (%s, build_parser): %s"
          % (RUNNER, ", ".join("%s %s" % (k, d[k]) for k in keys)))
    pat = re.search(r"patience=(\d+)", rsrc)
    print("runner's TrainConfig call: patience=%s" % pat.group(1))
    tc = _init_defaults(_show(TRAIN), "TrainConfig")
    print("TrainConfig.__init__ defaults (%s): grad_clip %s, beta2 %s, patience %s"
          % (TRAIN, tc["grad_clip"], tc["beta2"], tc["patience"]))
    ns, lines = _load(RUNNER, ("grouped_split", "arm_config"))
    print("executed from the freeze: grouped_split (lines %d-%d), arm_config (lines %d-%d)"
          % (lines["grouped_split"] + lines["arm_config"]))
    return {"defaults": d, "grad_clip": float(tc["grad_clip"]), "ns": ns, "rsrc": rsrc}


# --------------------------------------------------------------------------- #
# the bench shard (shared by B1 and B4)
# --------------------------------------------------------------------------- #
def build_bench_shard(tmp: str):
    cmd = [sys.executable, "build_latent_bank.py", "--out-dir", tmp, "--provider", "bench"]
    res = subprocess.run(cmd, cwd=_STAGE1, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError((res.stdout + res.stderr)[-2000:])
    if _STAGE1 not in sys.path:
        sys.path.insert(0, _STAGE1)
    import latent_bank as LB                           # noqa: E402
    arrays, side = LB.concat_shards(sorted(glob.glob(os.path.join(tmp, "*.npz"))))
    return arrays, side


# --------------------------------------------------------------------------- #
# B1  the streams' laws
# --------------------------------------------------------------------------- #
def p_repeat(n_rows: int, n_draws: int) -> float:
    """P(at least one row drawn twice) for n_draws uniform draws with replacement."""
    p_distinct = 1.0
    for k in range(n_draws):
        p_distinct *= 1.0 - k / float(n_rows)
    return 1.0 - p_distinct


def block_b1(env: dict, bench) -> None:
    hr("B1  the three streams' laws (joint_batches.py:168-199) and the A0s pre-training's rows")
    d = env["defaults"]
    b_met, n_ep, n_step = d["--b-met"], d["--epochs"], d["--steps-per-epoch"]
    n_plan = n_ep * n_step
    print("n_plan = epochs * steps_per_epoch = %d * %d = %d; with the rho_grad probe (lambda_dsn > 0)"
          " one more batch per epoch: %d batches drawn per stream" % (n_ep, n_step, n_plan,
                                                                     n_plan + n_ep))
    print("")
    print("metric batch: per = max(1, B_met // C) rows per class (:181), C * per rows in all")
    for B in (b_met, 64):
        for C in (2, 3):
            per = max(1, B // C)
            print("   B_met %2d, C = %d: %2d rows per class, %2d rows per batch" % (B, C, per, C * per))
    arrays, side = bench
    cls = np.asarray(arrays["cls"])
    bench_counts = tuple(int((cls == c).sum()) for c in np.unique(cls))
    print("")
    print("class shares: in the bank, in every metric batch, and the factor between them")
    print("   (each class gets the same per rows whatever its size; factor = (1/C) / (N_c / N))")
    for name, counts in (("DUP15HD (real cohort)", COUNTS_DUP15HD),
                         ("Giulia (real cohort)", COUNTS_GIULIA),
                         ("bench shard, arm S (one default shard)", bench_counts)):
        N, C = sum(counts), len(counts)
        per = max(1, b_met // C)
        parts = []
        for c, n_c in enumerate(counts):
            parts.append("class %d: %d rows, share %.4f, factor %.4f, P(repeat) %.4f, draws/row %.3f (%.3f with probes)"
                         % (c, n_c, n_c / N, (1.0 / C) / (n_c / N), p_repeat(n_c, per),
                            n_plan * per / n_c, (n_plan + n_ep) * per / n_c))
        print("   %s, N = %d, C = %d, per = %d:" % (name, N, C, per))
        for p in parts:
            print("      " + p)
    print("   (P(repeat): a class's per draws contain one row twice; draws/row: expected draws of one")
    print("    row of the class over the run's n_plan steps, metric stream)")

    print("")
    print("A0s pre-training (run_joint_arms.py:452-459): encoder_steps * B_met = %d * %d = %d draws,"
          % (d["--encoder-steps"], b_met, d["--encoder-steps"] * b_met))
    print("uniform with replacement over EVERY row of the source bank (:158); A2s's metric stream reads")
    print("the training split only (:498-502). The grouped split of the bench's donor ids (:118-143):")
    donors = np.asarray(arrays["donor"])
    gs = env["ns"]["grouped_split"]
    n_draw = d["--encoder-steps"] * b_met
    for label, dd in (("1 shard", donors),
                      ("%d shards (ids restart per shard, F-at)" % N_BANK_SHARDS,
                       np.tile(donors, N_BANK_SHARDS))):
        N = dd.size
        hit = 1.0 - (1.0 - 1.0 / N) ** n_draw
        print("   %s: N = %d rows; draws per row %.4f; P(a given row is drawn) %.6f" %
              (label, N, n_draw / float(N), hit))
        for seed in range(N_SEED):
            idx, dig = gs(dd, seed=seed)
            rows = [int((idx == p).sum()) for p in (0, 1, 2)]
            print("      seed %d: train / selection / report rows %s; report rows the pre-training"
                  " is expected to draw: %.2f of %d; A2s's metric stream: 0" %
                  (seed, rows, rows[2] * hit, rows[2]))
    print("   (the same draws over the real arm R are A0's: the pseudo-real endpoint scores every arm-R")
    print("    row, so with N_R = 512 the expected share of scored rows the pre-training saw is %.6f)"
          % (1.0 - (1.0 - 1.0 / 512) ** n_draw))


# --------------------------------------------------------------------------- #
# B2  what a loss weight reaches under AdamW (toy, open loop)
# --------------------------------------------------------------------------- #
def toy_gradients(seed: int = 4, d_psi: int = 40, d_om: int = 20, n: int = 250,
                  scale: float = 0.3):
    """Fixed gradient sequences: a (NPE on psi), b (metric term on psi),
    c (NPE on omega). Per-coordinate mean plus unit noise; b's coordinate
    scales log-uniform over two decades."""
    rng = np.random.default_rng(seed)
    mu_a, mu_b, mu_c = (rng.standard_normal(d_psi), rng.standard_normal(d_psi),
                        rng.standard_normal(d_om))
    s_b = scale * 10.0 ** rng.uniform(-1.0, 1.0, size=d_psi)
    a = scale * (mu_a[None, :] + rng.standard_normal((n, d_psi)))
    b = s_b[None, :] * (mu_b[None, :] + rng.standard_normal((n, d_psi)))
    c = scale * (mu_c[None, :] + rng.standard_normal((n, d_om)))
    return a, b, c


def run_toy(a, b, c, lam: float, clip, lr: float, beta1: float, beta2: float,
            use_a: bool = True):
    n, d_psi = a.shape
    d_om = c.shape[1]
    psi, m_p, v_p = np.zeros(d_psi), np.zeros(d_psi), np.zeros(d_psi)
    om, m_o, v_o = np.zeros(d_om), np.zeros(d_om), np.zeros(d_om)
    coefs = []
    for t in range(1, n + 1):
        g_p = (a[t - 1] if use_a else 0.0) + lam * b[t - 1]
        g_o = c[t - 1].copy()
        if clip is not None:
            tot = math.sqrt(float(g_p @ g_p) + float(g_o @ g_o))
            k = clip_coef(tot, clip)
            g_p, g_o = k * g_p, k * g_o
            coefs.append(k)
        psi, m_p, v_p = adamw_step(psi, g_p, m_p, v_p, t, lr, beta1, beta2, 1e-8, 0.0)
        om, m_o, v_o = adamw_step(om, g_o, m_o, v_o, t, lr, beta1, beta2, 1e-8, 0.0)
    return psi, om, np.asarray(coefs)


def _cos(u, v) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def block_b2(env: dict) -> None:
    hr("B2  what a loss weight reaches under AdamW and the clip (toy, open loop)")
    d = env["defaults"]
    lr, beta1, beta2 = d["--lr"], 1.0 - d["--one-minus-beta1"], 0.999
    gclip = env["grad_clip"]
    a, b, c = toy_gradients()
    n, d_psi = a.shape
    print("toy: %d steps; psi %d coordinates (gradient a + lambda b), omega %d (gradient c, no lambda);"
          % (n, d_psi, c.shape[1]))
    print("AdamW at the runner's lr %g, beta1 %.1f, beta2 %.3f, eps 1e-8, no decay; clip at %.1f"
          % (lr, beta1, beta2, gclip))
    na = np.median(np.linalg.norm(a, axis=1))
    nb = np.median(np.linalg.norm(b, axis=1))
    nc = np.median(np.linalg.norm(c, axis=1))
    print("median per-step norms: ||a|| %.3f, ||b|| %.3f, ||c|| %.3f; ||a|| / ||b|| = %.4f"
          % (na, nb, nc, na / nb))
    r_k = np.sqrt((a ** 2).mean(0)) / np.sqrt((b ** 2).mean(0))
    print("per-coordinate rms ratio r_k = rms(a_k) / rms(b_k): min %.4f, median %.4f, max %.4f"
          % (r_k.min(), np.median(r_k), r_k.max()))

    print("")
    print("(i) omega without the clip: its gradient carries no lambda, and Adam acts per coordinate")
    _, om0, _ = run_toy(a, b, c, 0.0, None, lr, beta1, beta2)
    for lam in (0.1, 1.0, 10.0, 1000.0):
        _, om, _ = run_toy(a, b, c, lam, None, lr, beta1, beta2)
        diff = float(np.max(np.abs(om - om0)))
        assert diff == 0.0
        print("   lambda %7g: max |omega(lambda) - omega(0)| = %.1f (bitwise equal: %s)"
              % (lam, diff, bool(np.array_equal(om, om0))))

    print("")
    print("(ii) with the global clip at %.1f over (psi, omega) (clip_grad_norm_ over model.parameters())"
          % gclip)
    for lam in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        _, om, k = run_toy(a, b, c, lam, gclip, lr, beta1, beta2)
        rel = float(np.linalg.norm(om - om0) / np.linalg.norm(om0))
        cv = float(k.std() / k.mean())
        print("   lambda %6g: clipped on %5.3f of steps, mean coefficient %.4f (min %.4f, sd/mean %.4f);"
              " ||omega - omega_free|| / ||omega_free|| = %.4f, cosine %.6f"
              % (lam, float((k < 1.0).mean()), float(k.mean()), float(k.min()), cv, rel,
                 _cos(om, om0)))

    print("")
    print("(iii) psi without the clip: the update's direction against the two limits")
    psi_npe, _, _ = run_toy(a, b, c, 0.0, None, lr, beta1, beta2)
    psi_dsn, _, _ = run_toy(a, b, c, 1.0, None, lr, beta1, beta2, use_a=False)
    for e10 in np.arange(-3.0, 3.01, 0.5):
        lam = 10.0 ** e10
        psi, _, _ = run_toy(a, b, c, lam, None, lr, beta1, beta2)
        near_dsn = np.abs(psi - psi_dsn) < np.abs(psi - psi_npe)
        print("   lambda 10^%+.1f: cos(dpsi, dpsi_NPE-only) %+.4f, cos(dpsi, dpsi_metric-only) %+.4f;"
              " coordinates nearer the metric-only update %2d of %d"
              % (e10, _cos(psi, psi_npe), _cos(psi, psi_dsn), int(near_dsn.sum()), d_psi))
    # the two crossovers, by bisection on log10(lambda)
    def gap_cos(e10):
        psi, _, _ = run_toy(a, b, c, 10.0 ** e10, None, lr, beta1, beta2)
        return _cos(psi, psi_npe) - _cos(psi, psi_dsn)

    def gap_half(e10):
        psi, _, _ = run_toy(a, b, c, 10.0 ** e10, None, lr, beta1, beta2)
        return d_psi / 2.0 - float((np.abs(psi - psi_dsn) < np.abs(psi - psi_npe)).sum())

    for name, fn in (("the two cosines are equal", gap_cos),
                     ("half the coordinates are nearer the metric-only update", gap_half)):
        lo, hi = -3.0, 3.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fn(mid) > 0:
                lo = mid
            else:
                hi = mid
        print("   %s at lambda = %.4f (10^%+.3f)" % (name, 10.0 ** hi, hi))
    print("   against: median r_k %.4f, global ||a|| / ||b|| %.4f; r_k spans %.4f to %.4f"
          % (np.median(r_k), na / nb, r_k.min(), r_k.max()))
    big, _, _ = run_toy(a, b, c, 1e6, None, lr, beta1, beta2)
    print("   lambda 1e6: max |psi - psi_metric-only| = %.3e (the scale-free limit)"
          % float(np.max(np.abs(big - psi_dsn))))
    print("   cos(dpsi_NPE-only, dpsi_metric-only) = %+.4f" % _cos(psi_npe, psi_dsn))


# --------------------------------------------------------------------------- #
# B3  the resolution of the rho_grad record
# --------------------------------------------------------------------------- #
def cp_interval(k: int, n: int, level: float = 0.95):
    """Clopper-Pearson by scipy (test inversion) and by the beta quantiles."""
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=level, method="exact")
    al = (1.0 - level) / 2.0
    lo_b = 0.0 if k == 0 else float(stats.beta.ppf(al, k, n - k + 1))
    hi_b = 1.0 if k == n else float(stats.beta.ppf(1.0 - al, k + 1, n - k))
    assert abs(ci.low - lo_b) < 1e-9 and abs(ci.high - hi_b) < 1e-9
    return float(ci.low), float(ci.high)


def block_b3(env: dict) -> None:
    hr("B3  what the rho_grad record can resolve (one probe batch per epoch, joint_train.py:234-249)")
    n_ep = env["defaults"]["--epochs"]
    print("probes per run at the runner's %d epochs: %d; per arm over n_seed = %d seeds: %d"
          % (n_ep, n_ep, N_SEED, n_ep * N_SEED))
    print("95 percent Clopper-Pearson intervals for the fraction of probes with rho_grad < 0"
          " (scipy test inversion = beta quantiles, to 1e-9):")
    for n, ks in ((n_ep, range(0, n_ep + 1)), (n_ep * N_SEED, (0, 1, 2, 5, 10, 25))):
        for k in ks:
            lo, hi = cp_interval(k, n)
            print("   n = %2d, k = %2d: [%.4f, %.4f], width %.4f" % (n, k, lo, hi, hi - lo))
    print("chance that a run shows NO negative probe when a fraction f of probe batches is negative:")
    for f in (0.05, 0.10, 0.20, 0.30):
        print("   f = %.2f: (1 - f)^%d = %.4f; (1 - f)^%d = %.4f"
              % (f, n_ep, (1 - f) ** n_ep, n_ep * N_SEED, (1 - f) ** (n_ep * N_SEED)))
    for n in (n_ep, n_ep * N_SEED):
        print("   smallest f seen at least once with probability 0.95 in %d probes: 1 - 0.05^(1/%d) = %.4f"
              % (n, n, 1.0 - 0.05 ** (1.0 / n)))


# --------------------------------------------------------------------------- #
# B4  A_ref's fixed features
# --------------------------------------------------------------------------- #
FEATURES = ("mean", "sd", "max", "q90", "burst_frac", "acf1", "skew", "rough")


def fixed_stats(x: np.ndarray) -> np.ndarray:
    """numpy transcription of FixedStatsSummary.forward (run_joint_arms.py:96-111)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(-1, keepdims=True)
    sd = x.std(-1, ddof=1, keepdims=True) + 1e-8          # torch .std: correction = 1
    xc = (x - mu) / sd
    feats = [
        mu, sd,
        x.max(-1, keepdims=True),
        np.quantile(x, 0.9, axis=-1, keepdims=True),      # torch.quantile: 'linear'
        (x > mu + sd).mean(-1, keepdims=True),
        (xc[:, 1:] * xc[:, :-1]).mean(-1, keepdims=True),
        (xc ** 3).mean(-1, keepdims=True),
        np.abs(np.diff(x, axis=-1)).mean(-1, keepdims=True),
    ]
    return np.concatenate(feats, axis=-1)


def block_b4(env: dict, bench) -> None:
    hr("B4  A_ref's eight fixed features on one default bench shard (--provider bench)")
    arrays, side = bench
    x = np.asarray(arrays["x"])
    print("shard: %d rows, W = %d, scale convention %s, d_theta %d"
          % (x.shape[0], x.shape[1], side["scale_convention"], np.asarray(arrays["theta"]).shape[1]))
    F = fixed_stats(x)
    sds = F.std(0, ddof=1)
    print("%-11s %11s %11s %11s %11s" % ("feature", "mean", "sd", "min", "max"))
    for j, name in enumerate(FEATURES):
        print("%-11s %11.5f %11.5f %11.5f %11.5f"
              % (name, F[:, j].mean(), sds[j], F[:, j].min(), F[:, j].max()))
    j_hi, j_lo = int(np.argmax(sds)), int(np.argmin(sds))
    print("largest sd / smallest sd across rows: %s / %s = %.1f"
          % (FEATURES[j_hi], FEATURES[j_lo], sds[j_hi] / sds[j_lo]))
    m = np.abs(F.mean(0))
    print("largest |mean| / smallest |mean|: %s / %s = %.1f"
          % (FEATURES[int(np.argmax(m))], FEATURES[int(np.argmin(m))], m.max() / m.min()))
    print("feature with the largest mean-to-sd ratio (offset in sd units): %s, %.1f"
          % (FEATURES[int(np.argmax(m / sds))], float(np.max(m / sds))))
    E = env["defaults"]["--embedding-size"]
    print("a learned arm's z is on the unit sphere of R^%d: every coordinate in [-1, 1], squared"
          " coordinates summing to 1 (the r2 encoder's L2 normalisation; E1)" % E)


# --------------------------------------------------------------------------- #
# B5  the arms as configured
# --------------------------------------------------------------------------- #
def block_b5(env: dict) -> None:
    hr("B5  the arms as configured (run_joint_arms.py:175-202) and the arms job (joint_arms.pbs)")
    d = env["defaults"]
    args = SimpleNamespace(lambda_dsn=d["--lambda-dsn"], lambda_rep=d["--lambda-rep"])
    arm_config = env["ns"]["arm_config"]
    pbs = _show(ARMS_PBS)
    order = re.search(r"^ARMS=\(([^)]*)\)", pbs, re.M).group(1).split()
    print("job order %s (joint_arms.pbs); arm = ARMS[IDX %% %d], seed = IDX / %d"
          % (" ".join(order), len(order), len(order)))
    print("%-9s %-6s %-5s %-6s %-6s %-5s %-38s %-15s %-5s %-6s %s"
          % ("arm", "frozen", "dom", "l_dsn", "l_rep", "warm", "metric stream (source)",
             "replicate", "probe", "real?", "indices in -J 0-44"))
    for arm in order:
        cfg = arm_config(arm, args)
        dom = cfg["dsn_domain"]
        if dom is None:
            met = "none"
        else:
            src = "sim, training split" if dom == "sim" else "real, every row"
            used = "used" if cfg["lambda_dsn"] > 0 else "drawn, unused"
            met = "%s (%s)" % (used, src)
        rep = "pairs, all real" if cfg["lambda_rep"] > 0 else "none"
        needs_real = dom == "real" or cfg["lambda_rep"] > 0
        idx = [i for i in range(45) if order[i % len(order)] == arm]
        tag = arm
        if cfg["freeze_encoder"] and dom is not None:
            tag += "*"
        print("%-9s %-6s %-5s %-6g %-6g %-5s %-38s %-15s %-5s %-6s %s"
              % (tag, cfg["freeze_encoder"], dom, cfg["lambda_dsn"], cfg["lambda_rep"],
                 cfg["warm_start"], met, rep, cfg["lambda_dsn"] > 0, needs_real, idx))
    print("(* the encoder is pre-trained by l_DSN alone on every row of its domain's bank, then frozen)")
    var = dict(re.findall(r'^: "\$\{(\w+):=([^}]*)\}"', pbs, re.M))
    print("job variables: LAMBDA_DSN=%s, LAMBDA_REP=%s, WARM_START_CKPT=%r (one value per submission)"
          % (var["LAMBDA_DSN"], var["LAMBDA_REP"], var["WARM_START_CKPT"]))
    reads_l = [a for a in order if arm_config(a, args)["lambda_dsn"] > 0]
    print("arms whose run reads LAMBDA_DSN: %s; LAMBDA_REP: %s; WARM_START_CKPT: A3"
          % (", ".join(reads_l), ", ".join(a for a in order if arm_config(a, args)["lambda_rep"] > 0)))
    stem = re.search(r'"%s_seed%d" % \(args\.arm, args\.seed\)', env["rsrc"]) is not None
    print("record stem '<arm>_seed<k>' carries neither lambda nor the checkpoint: %s" % stem)
    print("a submission at LAMBDA_DSN=1 into the same OUT_DIR rewrites the files of: %s"
          % ", ".join("%s_seed%d" % (a, s) for a in reads_l for s in (0,)) + ", ... (all seeds)")
    qline = [i for i, ln in enumerate(pbs.splitlines(), 1) if "qsub -J 0-44" in ln]
    print("A3 without WARM_START_CKPT exits before training (run_joint_arms.py:477-478); the job's"
          " documented array command (joint_arms.pbs:%d) sets WARM_START_CKPT: %s"
          % (qline[0], "WARM_START_CKPT" in pbs.splitlines()[qline[0] - 1]))


def main() -> None:
    env = block_b0()
    with tempfile.TemporaryDirectory() as tmp:
        bench = build_bench_shard(tmp)
        block_b1(env, bench)
        block_b2(env)
        block_b3(env)
        block_b4(env, bench)
        block_b5(env)
    print("")
    print("done.")


if __name__ == "__main__":
    main()
