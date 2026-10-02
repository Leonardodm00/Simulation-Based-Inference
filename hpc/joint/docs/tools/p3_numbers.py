#!/usr/bin/env python3
"""p3_numbers.py -- every [RAN] number of P3_REPLICATE_AXES.md, recomputed.

Torch-free (numpy + scipy only), so it runs in any environment that can run
the notation checker. Every block prints the numbers the document quotes, in
the order the document quotes them. Nothing here trains anything: the
replicate term's constants, the draw floor, the finite-draw correction, the
inverse-covariance inflation kappa_S, the warm-up ramp at the runner's
defaults, the clamp dead zone, the invalid-target pull and the bench's pair count are all closed-form
or small Monte Carlo properties of the loss as built.

Run from hpc/joint/docs/tools:

    python p3_numbers.py            # all blocks
    python p3_numbers.py --quick    # skip the Monte Carlo block (B6)

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy import stats

D_THETA = 26          # DUP15HD banks
D_BENCH = 10          # the bench shape (p, E, d_theta) = (10, 10, 10)


def kappa_s(s_mc: int, d: int = D_THETA) -> float:
    """Inverse-covariance inflation, metric doc eq. (72): n_W / (n_W - d - 1)."""
    n_w = 2 * (s_mc - 1)
    if n_w <= d + 1:
        return math.inf
    return n_w / (n_w - d - 1)


def residual(s_mc: int, p_eff: float, d: int = D_THETA) -> float:
    """res_S = (kappa_S - 1)(p_eff + d/S), FINITE_DRAW eq. (88)."""
    return (kappa_s(s_mc, d) - 1.0) * (p_eff + d / s_mc)


def p_cross(s_mc: int, d: int = D_THETA) -> float:
    """p_x(S): the p_eff above which res_S > d/S, FINITE_DRAW eq. (89)."""
    return 2.0 * d / (d + 1.0) * (1.0 - (d + 2.0) / s_mc)


def equilibrium(s_mc: int, p_eff: float, d: int = D_THETA) -> float:
    """True disagreement at which E[L_rep] is minimised under (a)-(c),
    FINITE_DRAW S3.9.8 item 2: p_eff/kappa - (1 - 1/kappa) d/S."""
    k = kappa_s(s_mc, d)
    return p_eff / k - (1.0 - 1.0 / k) * d / s_mc


def ramp(progress: float, t_warm: float) -> float:
    """ReplicateConsistencyLoss.ramp, joint_losses.py:284-290."""
    if t_warm <= 0.0:
        return 1.0
    if progress >= t_warm:
        return 1.0
    return max(0.0, progress / t_warm)


def block1_space() -> None:
    print("== B1  ranges, pins and the draw floor")
    for d in (D_THETA, D_BENCH):
        print("   d_theta = %2d : floor 4 d_theta = %3d ; range [%d, 400]"
              % (d, 4 * d, 4 * d))
    print("   JointSpaceSpec() default n_posterior_draws = (100, 400):"
          " 100 < 104 = floor at d_theta = 26 (F-o)")
    print("   log10_lambda_rep in [-3, 1]  ->  lambda_rep in [%g, %g]"
          % (10.0 ** -3, 10.0 ** 1))
    print("   warmup_frac_rep in [0.0, 0.5]; inactive pin 0.0; runner default 0.3 (F-p)")
    print("   Sigma_0 on the unit cube: diag(1/12) = diag(%.6f)" % (1.0 / 12.0))
    print("   smallest S_mc with finite kappa_S at d_theta = 26: %d"
          % min(s for s in range(2, 100) if math.isfinite(kappa_s(s))))
    print("   C_hat_g positive definite a.s. from S_mc = %d; C_bar from S_mc = %d"
          % (D_THETA + 1, math.ceil(D_THETA / 2) + 1))


def block2_kappa() -> None:
    print("== B2  d/S, kappa_S, the residual at p_eff = 3, and p_x")
    print("   %6s %8s %8s %8s %8s %8s %8s" % ("S_mc", "d/S", "kappa_S",
                                            "res_S", "ratio", "p_x", "equil"))
    for s in (100, 104, 128, 256, 285, 400, 1000):
        r = residual(s, 3.0)
        print("   %6d %8.3f %8.3f %8.3f %8.2f %8.3f %8.3f"
              % (s, D_THETA / s, kappa_s(s), r, r / (D_THETA / s),
                 p_cross(s), equilibrium(s, 3.0)))
    print("   floor identity (8 d - 2)/(7 d - 3): %.3f at d = 26, %.3f at d = 10"
          % ((8 * D_THETA - 2) / (7 * D_THETA - 3),
             (8 * D_BENCH - 2) / (7 * D_BENCH - 3)))
    print("   kappa_S at the bench floor S_mc = 40, d = 10: %.3f" % kappa_s(40, D_BENCH))
    s_min = min(s for s in range(15, 2000) if kappa_s(s) <= 1.05)
    print("   smallest S_mc with kappa_S <= 1.05 at d = 26: %d" % s_min)
    print("   relative Monte Carlo excess d/(S p_eff) at p_eff = 3: %.3f (S=104), %.3f (S=128)"
          % (D_THETA / (104 * 3.0), D_THETA / (128 * 3.0)))
    print("   log kappa_S at S_mc = 128: %.3f" % math.log(kappa_s(128)))


def block3_ramp() -> None:
    print("== B3  the warm-up ramp at the runner's defaults (10 epochs x 25 steps, t_warm = 0.3)")
    n_ep, n_step, t_warm = 10, 25, 0.3
    total = n_ep * n_step
    first_full = min(k for k in range(total) if ramp(k / total, t_warm) >= 1.0)
    print("   planned steps %d; ramp reaches 1 at step index %d (epoch %d)"
          % (total, first_full, first_full // n_step))
    for ep in range(4):
        last = (ep + 1) * n_step - 1
        print("   epoch %d: ramp at its first step %.3f, at its last step %.3f (logged value)"
              % (ep, ramp(ep * n_step / total, t_warm), ramp(last / total, t_warm)))
    print("   fraction of the planned steps with a partial ramp: %.1f%%"
          % (100.0 * first_full / total))
    print("   search range [0, 0.5]: at 0.5 the ramp spans %d of %d steps" % (total // 2, total))


def block4_draws_per_step() -> None:
    print("== B4  draws per optimiser step of arm A5 at the defaults")
    b_rep, s_mc = 4, 128
    print("   2 wells x B_rep = %d pairs x S_mc = %d draws = %d draws of d_theta = 26 per step"
          % (b_rep, s_mc, 2 * b_rep * s_mc))
    print("   covariances per step: %d of size 26 x 26 (one per well), each from %d draws"
          % (2 * b_rep, s_mc))
    print("   diagnostics: at most 64 pairs, %d draws each, non-reparameterised" % s_mc)


def block5_clamps() -> None:
    print("== B5  the two clamps: dead zone and invalid-target pull")
    t_floor, p_min = 1e-8, 1e-3
    for p_eff in (3.0, 13.0):
        print("   loss plateau at T_hat <= t_floor, p_eff = %4.1f: (ln 1e-8 - ln p_eff)^2 = %.1f"
              % (p_eff, (math.log(t_floor) - math.log(p_eff)) ** 2))
    print("   P(chi2_26 <= 26) = %.3f : exact-metric chance a collapsed pair (Delta = 0)"
          " has T_hat <= 0" % stats.chi2.cdf(26, df=26))
    print("   gradient weight 2|ln(T/p)|/T at p_eff = 3: T = 0.01 -> %.0f ; T = 0.1 -> %.1f ;"
          " T = 1 -> %.2f ; T = 3 -> 0"
          % (2 * abs(math.log(0.01 / 3.0)) / 0.01, 2 * abs(math.log(0.1 / 3.0)) / 0.1,
             2 * abs(math.log(1.0 / 3.0)) / 1.0))
    for T in (0.1, 1.0, 3.0):
        print("   invalid target (p_eff_raw <= 0 -> p_min = 1e-3): loss at T_hat = %.1f is %.1f,"
              " slope sign %s" % (T, (math.log(T) - math.log(p_min)) ** 2,
                                  "+" if T > p_min else "-"))


def block6_montecarlo(seed: int = 0, n_pairs: int = 4000) -> None:
    print("== B6  Monte Carlo: a collapsed pair under the realised metric (Gaussian draws)")
    rng = np.random.default_rng(seed)
    d = D_THETA
    for s_mc in (104, 128, 256):
        n_neg = 0
        t_hat = np.empty(n_pairs)
        for k in range(n_pairs):
            a = rng.standard_normal((s_mc, d))
            b = rng.standard_normal((s_mc, d))
            m_a, m_b = a.mean(0), b.mean(0)
            c_bar = 0.5 * (np.cov(a, rowvar=False, ddof=1) + np.cov(b, rowvar=False, ddof=1))
            delta = m_a - m_b
            x = np.linalg.solve(2.0 * c_bar, delta)
            t_hat[k] = float(delta @ x) - d / s_mc
            n_neg += int(t_hat[k] <= 0.0)
        print("   S_mc = %4d : mean T_hat = %+.4f (eq. (72) with p_eff = 0 predicts %+.4f);"
              " fraction with T_hat <= 0: %.3f"
              % (s_mc, t_hat.mean(), (kappa_s(s_mc) - 1.0) * d / s_mc, n_neg / n_pairs))


def block7_bench_pairs(n_traces: int = 64, wells_per_donor: int = 2, n_windows: int = 8) -> None:
    """Stage 1 job defaults (build_latent_bank.pbs:44-46): rows are windows, pairs are by donor."""
    print("== B7  bench replicate pairs at the Stage 1 defaults (F-ab)")
    n_donors = n_traces // wells_per_donor
    rows = wells_per_donor * n_windows
    pairs = rows * (rows - 1) // 2
    within = wells_per_donor * (n_windows * (n_windows - 1) // 2)
    across = pairs - within
    print("   %d traces, %d wells per donor, %d windows: %d donors, %d rows per donor"
          % (n_traces, wells_per_donor, n_windows, n_donors, rows))
    print("   pairs per donor %d: within one well %d (%.1f%%), across wells %d"
          % (pairs, within, 100.0 * within / pairs, across))
    print("   N_pair as the code counts it: %d ; pairs of WELLS: %d"
          % (n_donors * pairs, n_donors * (wells_per_donor * (wells_per_donor - 1) // 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the Monte Carlo block")
    args = ap.parse_args()
    block1_space()
    block2_kappa()
    block3_ramp()
    block4_draws_per_step()
    block5_clamps()
    if not args.quick:
        block6_montecarlo()
    block7_bench_pairs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
