#!/usr/bin/env python3
"""p5_numbers.py -- every [RAN] number of P5_OPTIMISER_AND_SCHEDULE.md, recomputed.

Torch-free (numpy only), so it runs in any environment that can run the
notation checker. Every block prints the numbers the document quotes, in the
order the document quotes them. Nothing here trains anything: the AdamW
update is replicated from torch 2.10.0's `torch/optim/adam.py`
(`_single_tensor_adam`, the `decoupled_weight_decay=True` branch that
`torch/optim/adamw.py` selects), the gradient clip from
`torch/nn/utils/clip_grad.py` (`clip_grad_norm_`), the schedule arithmetic
from `hpc/joint/stage2/joint_train.py` and `hpc/joint/stage3/run_joint_arms.py`,
the batch-size trim rule from `hpc/joint/stage4/joint_space.py`, the grouped
split from `run_joint_arms.py::grouped_split` (copied verbatim: it is
torch-free), and sbi 0.27.0's stopping rule from
`sbi/inference/trainers/base.py::_converged`.

Run from hpc/joint/docs/tools:

    python p5_numbers.py            # all blocks

Pure ASCII, LF only.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

# ---------------------------------------------------------------------------
# The configured values the document quotes (each with its [REPO] line in P5)
# ---------------------------------------------------------------------------

RUNNER = dict(epochs=10, steps_per_epoch=25, b_sim=128, b_met=32, b_rep=4,
              lr=1e-3, weight_decay=0.0, one_minus_beta1=0.1,
              encoder_steps=200)                      # run_joint_arms.py:215-226, 258
TRAINCONFIG = dict(epochs=20, steps_per_epoch=50, lr=5e-4, weight_decay=0.0,
                   grad_clip=5.0, patience=5, beta1=0.9, beta2=0.999)  # joint_train.py:41-44
RUNNER_PATIENCE = 99                                  # run_joint_arms.py:522
SPACE = dict(lr=(1e-4, 2e-3), one_minus_beta1=(1e-2, 1e-1),
             weight_decay=(1e-5, 1e-2), batch_size_npe=(256, 512, 1024),
             one_minus_beta2=1e-3)                    # joint_space.py:238-241, 324
DSN_SEARCH = dict(lr=(1e-4, 0.2), one_minus_beta1=(1e-2, 1e-1),
                  one_minus_beta2=(1e-4, 1e-2), weight_decay=(1e-4, 1e-2))  # dsn/config.py:931-934
DSN_TRAIN = dict(lr=3e-4, weight_decay=1e-4, max_epochs=100, patience=10)   # dsn/config.py:738-745
TORCH_ADAMW = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)  # torch adamw.py:24-27
SBI_TRAIN = dict(training_batch_size=200, learning_rate=5e-4,
                 validation_fraction=0.1, stop_after_epochs=20, clip_max_norm=5.0)  # sbi npe_base.py:254-259
N_SIM_ROWS = 29616        # [KB] plan S5.1: the MFR-filtered DUP15HD bank
N_TOPOLOGY = 383          # [KB] plan S5.1: distinct topology draws
BENCH = dict(n_traces=64, wells_per_donor=2, n_windows=8)   # build_latent_bank.pbs:44-46


def hr(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# B1  the AdamW step: timescales, bias corrections, the decoupled decay, the
#     loss-scale invariance, the clip coefficient
# ---------------------------------------------------------------------------

def adamw_step(param, grad, m, v, t, lr, beta1, beta2, eps, weight_decay):
    """One AdamW step on a numpy vector, in the order torch 2.10.0 applies it
    (`_single_tensor_adam`, decoupled branch, no amsgrad, no capturable):
    decay the parameter, update the moments, bias-correct, divide, subtract."""
    if weight_decay != 0.0:
        param = param * (1.0 - lr * weight_decay)
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad * grad
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    step_size = lr / bc1
    denom = np.sqrt(v) / math.sqrt(bc2) + eps
    param = param - step_size * m / denom
    return param, m, v


def clip_coef(total_norm: float, max_norm: float = 5.0) -> float:
    """torch clip_grad_norm_: min(1, max_norm / (total_norm + 1e-6))."""
    return min(1.0, max_norm / (total_norm + 1e-6))


def b1() -> None:
    hr("B1  the AdamW step (torch 2.10.0 adam.py:417-419, 528-547; adamw.py:20-49)")
    print("averaging timescale 1/(1 - beta) and half-life ln 2 / (-ln beta), in steps:")
    for name, u in (("u1 = 0.1 (runner default, beta1 = 0.9)", 0.1),
                    ("u1 = 0.01 (space lower end, beta1 = 0.99)", 0.01),
                    ("u2 = 1e-3 (space FIXED, beta2 = 0.999)", 1e-3),
                    ("u2 = 1e-4 (DSN range end, beta2 = 0.9999)", 1e-4),
                    ("u2 = 1e-2 (DSN range end, beta2 = 0.99)", 1e-2)):
        beta = 1.0 - u
        print("   %-44s tau = %8.1f   half-life = %8.2f"
              % (name, 1.0 / u, math.log(2.0) / (-math.log(beta))))
    print("planned steps T = epochs * steps_per_epoch: runner %d, TrainConfig %d, encoder pre-training %d"
          % (RUNNER["epochs"] * RUNNER["steps_per_epoch"],
             TRAINCONFIG["epochs"] * TRAINCONFIG["steps_per_epoch"], RUNNER["encoder_steps"]))
    print("bias corrections 1 - beta^t at beta1 = 0.9 / 0.99 and beta2 = 0.999:")
    for t in (1, 10, 25, 75, 250, 1000):
        print("   t = %5d   1-0.9^t = %.4f   1-0.99^t = %.4f   1-0.999^t = %.4f"
              % (t, 1 - 0.9 ** t, 1 - 0.99 ** t, 1 - 0.999 ** t))
    print("weight of the gradient k steps old inside m_hat_t at t = 250, beta1 = 0.9:")
    t = 250
    for k in (0, 1, 5, 10, 20, 50):
        print("   k = %3d   (1-beta1) beta1^k / (1-beta1^t) = %.4e" % (k, 0.1 * 0.9 ** k / (1 - 0.9 ** t)))
    print("same for the second moment at beta2 = 0.999, t = 250 (nearly flat over the run):")
    for k in (0, 100, 249):
        print("   k = %3d   (1-beta2) beta2^k / (1-beta2^t) = %.4e" % (k, 1e-3 * 0.999 ** k / (1 - 0.999 ** t)))

    print("")
    print("decoupled decay: per-step factor (1 - lr wd), cumulative (1 - lr wd)^T, horizon 1/(lr wd) steps")
    cases = (("runner default        lr 1e-3  wd 0", 1e-3, 0.0),
             ("space weakest corner  lr 1e-4  wd 1e-5", 1e-4, 1e-5),
             ("space strongest corner lr 2e-3  wd 1e-2", 2e-3, 1e-2),
             ("space, lr 1e-3        wd 1e-2", 1e-3, 1e-2),
             ("torch AdamW defaults  lr 1e-3  wd 1e-2 (F-q)", 1e-3, 1e-2),
             ("DSN TrainConfig       lr 3e-4  wd 1e-4", 3e-4, 1e-4))
    for name, lr, wd in cases:
        per = 1.0 - lr * wd
        hz = (1.0 / (lr * wd)) if wd > 0 else float("inf")
        print("   %-48s per-step %.8f   T=200: %.5f   T=250: %.5f   T=1000: %.5f   horizon %s"
              % (name, per, per ** 200, per ** 250, per ** 1000,
                 ("%.0f" % hz) if wd > 0 else "inf"))
    # the shrink that the decay alone contributes over 250 steps at the strongest corner
    lr, wd = 2e-3, 1e-2
    print("   strongest-corner shrink over 250 steps: 1 - (1 - lr wd)^250 = %.4f  (= %.2f %%)"
          % (1 - (1 - lr * wd) ** 250, 100 * (1 - (1 - lr * wd) ** 250)))
    # and at the weakest corner, where %.5f above rounds to 1: in scientific notation
    lr, wd = 1e-4, 1e-5
    print("   weakest-corner shrink over 250 steps:   1 - (1 - lr wd)^250 = %.3e; over 1000 steps %.3e"
          % (1 - (1 - lr * wd) ** 250, 1 - (1 - lr * wd) ** 1000))
    lr, wd = 2e-3, 1e-2
    print("   strongest-corner shrink over the TrainConfig's 1000 steps: %.4f  (= %.2f %%)"
          % (1 - (1 - lr * wd) ** 1000, 100 * (1 - (1 - lr * wd) ** 1000)))

    print("")
    print("loss-scale invariance of the Adam direction (numpy replica, eps = 1e-8):")
    rng = np.random.default_rng(0)
    d = 50
    grads = rng.normal(size=(250, d))
    for scale in (1.0, 1e3, 1e-3):
        p = np.zeros(d)
        m = np.zeros(d)
        v = np.zeros(d)
        for t in range(1, 251):
            p, m, v = adamw_step(p, scale * grads[t - 1], m, v, t, lr=1e-3,
                                 beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
        if scale == 1.0:
            ref = p.copy()
        print("   scale %7.0e : max |param - param(scale 1)| = %.3e   (||param|| = %.4f)"
              % (scale, float(np.max(np.abs(p - ref))), float(np.linalg.norm(p))))
    print("   (a per-step rescaling -- what the clip does -- is NOT cancelled: the EMAs mix steps)")
    p = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    for t in range(1, 251):
        c = 1.0 if t % 2 else 0.1
        p, m, v = adamw_step(p, c * grads[t - 1], m, v, t, lr=1e-3,
                             beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
    print("   alternating scale 1 / 0.1: max |param - param(scale 1)| = %.3e"
          % float(np.max(np.abs(p - ref))))

    print("")
    print("clip coefficient min(1, 5 / (||g|| + 1e-6)) at ||g|| = 1, 5, 50, 500: %s"
          % ", ".join("%.4f" % clip_coef(g) for g in (1.0, 5.0, 50.0, 500.0)))
    print("a stale gradient of norm s added to the clipped set raises the total norm from g to sqrt(g^2 + s^2):")
    for g, s in ((3.0, 4.0), (5.0, 5.0), (1.0, 10.0)):
        tot = math.hypot(g, s)
        print("   g = %4.1f, s = %4.1f -> total %.3f, coef %.4f (was %.4f without the stale part)"
              % (g, s, tot, clip_coef(tot), clip_coef(g)))


# ---------------------------------------------------------------------------
# B2  the schedule: steps, draws, passes, coverage, the trim rule, early stopping
# ---------------------------------------------------------------------------

def coverage(n_rows: int, n_draws: int) -> float:
    """Expected fraction of rows seen at least once by n_draws i.i.d. draws
    with replacement from n_rows rows: 1 - (1 - 1/N)^n."""
    return 1.0 - (1.0 - 1.0 / n_rows) ** n_draws


def grouped_split_counts(n_groups: int, fracs=(0.7, 0.15, 0.15), seed: int = 0):
    """Group counts of `run_joint_arms.grouped_split` (its rounding, verbatim)."""
    groups = np.arange(n_groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n = len(perm)
    n_tr = int(round(fracs[0] * n))
    n_se = int(round(fracs[1] * n))
    assign = {}
    for g in perm[:n_tr]:
        assign[g] = 0
    for g in perm[n_tr:n_tr + n_se]:
        assign[g] = 1
    for g in perm[n_tr + n_se:]:
        assign[g] = 2
    idx = np.array([assign[g] for g in groups])
    digest = hashlib.sha256(
        json.dumps({str(k): int(v) for k, v in sorted(assign.items())},
                   sort_keys=True).encode("ascii")).hexdigest()
    return int((idx == 0).sum()), int((idx == 1).sum()), int((idx == 2).sum()), digest


def b2() -> None:
    hr("B2  the schedule (joint_train.py:148-213; joint_batches.py:172; joint_space.py:304-306)")
    T_run = RUNNER["epochs"] * RUNNER["steps_per_epoch"]
    T_lib = TRAINCONFIG["epochs"] * TRAINCONFIG["steps_per_epoch"]
    print("planned steps: runner %d x %d = %d; TrainConfig %d x %d = %d; the loop never stops early below patience"
          % (RUNNER["epochs"], RUNNER["steps_per_epoch"], T_run,
             TRAINCONFIG["epochs"], TRAINCONFIG["steps_per_epoch"], T_lib))
    # the bench training split: 64 traces, 2 wells per donor -> 32 donors; 8 windows per trace
    n_donors = BENCH["n_traces"] // BENCH["wells_per_donor"]
    rows_per_donor = BENCH["wells_per_donor"] * BENCH["n_windows"]
    tr, se, rp, digest = grouped_split_counts(n_donors)
    print("bench split at seed 0: %d donors -> train/selection/report %d/%d/%d donors = %d/%d/%d rows; split hash %s..."
          % (n_donors, tr, se, rp, tr * rows_per_donor, se * rows_per_donor, rp * rows_per_donor, digest[:12]))
    n_bench_train = tr * rows_per_donor
    tr3, se3, rp3, _ = grouped_split_counts(N_TOPOLOGY)
    print("if the DUP15HD bank were grouped by its %d topology draws [KB]: %d/%d/%d groups at seed 0"
          % (N_TOPOLOGY, tr3, se3, rp3))
    n_dup_train = int(round(0.7 * N_SIM_ROWS))
    print("DUP15HD training rows, taken as 0.7 x %d = %d (a row fraction the group split only approximates)"
          % (N_SIM_ROWS, n_dup_train))
    print("")
    print("| b_sim | draws/epoch (25 steps) | draws/run (250) | pass-equivalents of %d rows | coverage per epoch | coverage per run | visits per row per run | bench visits per row per epoch (%d rows) |"
          % (n_dup_train, n_bench_train))
    for b in (128, 256, 512, 1024):
        per_epoch = b * RUNNER["steps_per_epoch"]
        per_run = per_epoch * RUNNER["epochs"]
        print("| %4d | %6d | %7d | %.2f | %.3f | %.3f | %.2f | %.1f |"
              % (b, per_epoch, per_run, per_run / n_dup_train, coverage(n_dup_train, per_epoch),
                 coverage(n_dup_train, per_run), per_run / n_dup_train, per_epoch / n_bench_train))
    print("the plan's pass-based epoch at B_sim = 512 over the whole bank: %d / 512 = %.1f steps (plan S5.1 says 'about 58')"
          % (N_SIM_ROWS, N_SIM_ROWS / 512))
    print("the plan's pass over the TRAINING rows at 512: %d / 512 = %.1f steps; the code's epoch is %d steps whatever b_sim"
          % (n_dup_train, n_dup_train / 512, RUNNER["steps_per_epoch"]))
    print("")
    print("the trim rule `n_train >= 20 * b` (joint_space.py:306; npe_tune_search.py:138): thresholds 256/512/1024 -> %d/%d/%d rows"
          % (20 * 256, 20 * 512, 20 * 1024))
    for n in (n_bench_train, 5119, 5120, 10240, 20479, 20480, n_dup_train):
        kept = [b for b in (256, 512, 1024) if n >= 20 * b] or [256]
        print("   n_train = %6d -> batch_size_npe %s   ('steps per pass' the rule imagines: %s)"
              % (n, kept, ", ".join("%.1f" % (n / b) for b in kept)))
    print("   margins: the DUP15HD training rows clear the 1024 threshold by %d rows; on the bench a batch of 256 is %.0f %% of the %d training rows"
          % (n_dup_train - 20 * 1024, 100.0 * 256 / n_bench_train, n_bench_train))
    print("")
    print("early stopping: the loop breaks when epoch - best_epoch >= patience (joint_train.py:262-263)")
    for pat, ep in ((RUNNER_PATIENCE, RUNNER["epochs"]), (TRAINCONFIG["patience"], TRAINCONFIG["epochs"]),
                    (SBI_TRAIN["stop_after_epochs"], 2 ** 31 - 1)):
        print("   patience %3d, epochs %10d: earliest break after epoch index %d, i.e. %d epochs run; can fire: %s"
              % (pat, ep, pat, pat + 1, "yes" if ep >= pat + 1 else "NO"))
    # sbi's rule against the joint loop's rule on random validation curves
    rng = np.random.default_rng(1)
    agree = 0
    trials = 2000
    for _ in range(trials):
        curve = np.cumsum(rng.normal(-0.02, 0.1, size=60))      # a noisy, slowly improving validation loss
        # joint loop
        best, best_ep, stop_j = math.inf, -1, None
        for ep, val in enumerate(curve):
            if val < best - 1e-6:
                best, best_ep = val, ep
            if ep - best_ep >= 5:
                stop_j = ep
                break
        # sbi: _converged is called with the loss of the PREVIOUS epoch before each new one
        best_s, since, stop_s = math.inf, 0, None
        for ep, val in enumerate(curve):
            if ep == 0 or val < best_s:
                best_s, since = val, 0
            else:
                since += 1
            if since > 5 - 1:
                stop_s = ep
                break
        agree += int(stop_j == stop_s)
    print("   joint rule (patience 5, threshold 1e-6) and sbi's rule (stop_after_epochs 5, strict <) stop at the same epoch on %d of %d random curves"
          % (agree, trials))
    print("")
    print("the rho_grad probe takes one extra batcher.next() per epoch when lambda_dsn > 0 (joint_train.py:236):")
    print("   after %d epochs the simulated stream of an A2 run is %d batches (%d rows at b_sim 128) ahead of an A1 run at the same seed"
          % (RUNNER["epochs"], RUNNER["epochs"], RUNNER["epochs"] * RUNNER["b_sim"]))


# ---------------------------------------------------------------------------
# B3  the surfaces: the defaults side by side and the runner's position in the space
# ---------------------------------------------------------------------------

def b3() -> None:
    hr("B3  the surfaces (joint_space.py:238-241; run_joint_arms.py:215-226, 258; joint_train.py:41-44; dsn/config.py; sbi; torch)")
    lo, hi = SPACE["lr"]
    print("lr: space [%g, %g] log-uniform; runner %g (inside); TrainConfig %g (inside); DSN TrainConfig %g (inside); DSN search [%g, %g]; sbi %g; torch %g"
          % (lo, hi, RUNNER["lr"], TRAINCONFIG["lr"], DSN_TRAIN["lr"], DSN_SEARCH["lr"][0], DSN_SEARCH["lr"][1],
             SBI_TRAIN["learning_rate"], TORCH_ADAMW["lr"]))
    print("   log-uniform mass of the space below the runner's 1e-3: %.3f; below 5e-4: %.3f"
          % (math.log(1e-3 / lo) / math.log(hi / lo), math.log(5e-4 / lo) / math.log(hi / lo)))
    print("   the DSN search's upper end 0.2 is %.0f times the joint space's 2e-3" % (0.2 / 2e-3))
    lo, hi = SPACE["weight_decay"]
    print("weight_decay: space [%g, %g] log-uniform; runner %g (BELOW the range, F-r); TrainConfig %g; torch AdamW %g (F-q); DSN TrainConfig %g; DSN search [%g, %g]"
          % (lo, hi, RUNNER["weight_decay"], TRAINCONFIG["weight_decay"], TORCH_ADAMW["weight_decay"],
             DSN_TRAIN["weight_decay"], DSN_SEARCH["weight_decay"][0], DSN_SEARCH["weight_decay"][1]))
    lo, hi = SPACE["one_minus_beta1"]
    print("one_minus_beta1: space [%g, %g] log-uniform; runner %g (the upper edge); torch/sbi beta1 0.9 = u1 0.1"
          % (lo, hi, RUNNER["one_minus_beta1"]))
    print("   beta1 range [%.2f, %.2f]; the runner default sits on the boundary (boundary_axes would flag it)"
          % (1 - hi, 1 - lo))
    print("one_minus_beta2: FIXED %g (beta2 = %.3f); the DSN searches [%g, %g]; torch and sbi default beta2 0.999"
          % (SPACE["one_minus_beta2"], 1 - SPACE["one_minus_beta2"], DSN_SEARCH["one_minus_beta2"][0],
             DSN_SEARCH["one_minus_beta2"][1]))
    print("batch_size_npe: space %s; runner --b-sim %d (not in the set, F-r); BatchSpec %d; sbi %d; standalone NPEConfig 512"
          % (SPACE["batch_size_npe"], RUNNER["b_sim"], 512, SBI_TRAIN["training_batch_size"]))
    print("   ratios b_sim / b_met / b_rep at the runner: %d / %d / %d = %.0f : %.0f : 1; BatchSpec 512/64/8 = %.0f : %.0f : 1"
          % (RUNNER["b_sim"], RUNNER["b_met"], RUNNER["b_rep"], RUNNER["b_sim"] / RUNNER["b_rep"],
             RUNNER["b_met"] / RUNNER["b_rep"], 512 / 8, 64 / 8))
    print("   the space's 1024 with the runner's b_met 32 / b_rep 4 (not searched, F-l) = %.0f : %.0f : 1"
          % (1024 / RUNNER["b_rep"], RUNNER["b_met"] / RUNNER["b_rep"]))
    print("grad_clip: TrainConfig %g, no flag; sbi clip_max_norm %g; the DSN TrainConfig has no clip field"
          % (TRAINCONFIG["grad_clip"], SBI_TRAIN["clip_max_norm"]))
    print("patience: runner %d (no flag); TrainConfig %d; DSN TrainConfig %d of %d epochs; sbi stop_after_epochs %d"
          % (RUNNER_PATIENCE, TRAINCONFIG["patience"], DSN_TRAIN["patience"], DSN_TRAIN["max_epochs"],
             SBI_TRAIN["stop_after_epochs"]))
    print("the split: joint (0.7, 0.15, 0.15) by group; sbi validation_fraction %g by row, no grouping" % SBI_TRAIN["validation_fraction"])


# ---------------------------------------------------------------------------
# B4  the encoder pre-training's optimiser (F-q), in numbers
# ---------------------------------------------------------------------------

def b4() -> None:
    hr("B4  F-q: AdamW(backbone.parameters(), lr=lr) in train_encoder_only (run_joint_arms.py:154)")
    lr = RUNNER["lr"]
    wd = TORCH_ADAMW["weight_decay"]
    T = RUNNER["encoder_steps"]
    print("torch defaults reach it: betas (%.1f, %.3f) = the runner's own defaults (u1 %.1f); weight_decay %g against the runner's %g"
          % (TORCH_ADAMW["betas"][0], TORCH_ADAMW["betas"][1], RUNNER["one_minus_beta1"], wd, RUNNER["weight_decay"]))
    print("decay over the %d pre-training steps at lr %g: (1 - lr wd)^T = %.5f, a %.2f %% shrink of every encoder weight"
          % (T, lr, (1 - lr * wd) ** T, 100 * (1 - (1 - lr * wd) ** T)))
    print("the joint loop at the runner's defaults decays nothing (wd 0); so by default the FROZEN arms' encoder is the only")
    print("tensor in Stage 3 that weight decay ever touches")
    print("the pre-training's batch generator and the sim stream's share the seed value seed+1 (run_joint_arms.py:155; joint_batches.py:126)")


def main() -> None:
    b1()
    b2()
    b3()
    b4()
    print("")
    print("done.")


if __name__ == "__main__":
    main()
