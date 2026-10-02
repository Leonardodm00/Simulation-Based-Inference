#!/usr/bin/env python3
"""p4_numbers.py -- every [RAN] number of P4_FLOW_AXES.md, recomputed.

Torch-free (numpy + scipy only), so it runs in any environment that can run
the notation checker. Every block prints the numbers the document quotes, in
the order the document quotes them. Nothing here trains anything: the width
rule, the spline's parameter count and clipping bounds, the hyper-network's
shapes and weight counts (nominal and unmasked), the cost asymmetry between
density evaluation and sampling, the box-to-unconstrained transform's
Jacobian anchor and the layered defaults are all closed-form properties of
the flow as `sbi` 0.27.0 and `zuko` 1.6.0 build it, replicated here from
the two wheels' source (`sbi/neural_nets/net_builders/flow.py`,
`zuko/flows/spline.py`, `zuko/flows/autoregressive.py`, `zuko/nn.py`,
`zuko/transforms.py`) and from `hpc/joint/stage4/joint_space.py` and
`hpc/npe_tune_search.py`.

Run from hpc/joint/docs/tools:

    python p4_numbers.py            # all blocks
    python p4_numbers.py --quick    # skip the Monte Carlo check of B5

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy import integrate, stats

D_THETA = 26          # DUP15HD banks
E_DUP = 12            # the search anchor --embedding-dim
E_RUNNER = 10         # run_joint_arms.py --embedding-size default
D_BENCH = 10          # the bench shape (p, E, d_theta) = (10, 10, 10)
BOUND = 5.0           # zuko MonotonicRQSTransform bound (transforms.py:473)
SLOPE = 1e-3          # zuko NSF slope (spline.py:52)


# ---------------------------------------------------------------------------
# B1  the width rule, as coded (joint_space.py:299-302; npe_tune_search.py:131-134)
# ---------------------------------------------------------------------------

def width_rule(p: int, embedding_dim: int):
    """Exactly `default_joint_space` / `default_space`: returns (lo, hi)."""
    scale = max(int(p), int(embedding_dim))
    lo = max(32, int(2 * scale))
    hi = max(lo * 2, int(8 * scale))
    lo, hi = min(lo, 64), max(hi, 256)
    return lo, hi


def width_rule_closed(scale: int):
    """The same function written out: lo = clamp(2 s, 32, 64), hi = max(8 s, 256)."""
    return min(max(32, 2 * scale), 64), max(8 * scale, 256)


def recorded_string(scale: int):
    """What the ledger string 'clamp([2,8] * max(p,E), [64,256])' would give."""
    return min(max(2 * scale, 64), 256), min(max(8 * scale, 64), 256)


def block1_width_rule() -> None:
    print("== B1  the hidden_features width rule, as coded, and the string the ledger records")
    for (p, e) in ((D_THETA, E_DUP), (D_THETA, E_RUNNER), (D_BENCH, D_BENCH)):
        lo, hi = width_rule(p, e)
        assert (lo, hi) == width_rule_closed(max(p, e))
        print("   (p, E) = (%2d, %2d): hidden_features in [%d, %d]" % (p, e, lo, hi))
    print("   closed form: lo = clamp(2 max(p,E), 32, 64), hi = max(8 max(p,E), 256)")
    print("   | s = max(p,E) | as coded | the recorded string, read as a clamp |")
    for s in (6, 10, 16, 26, 32, 33, 40, 64):
        print("   | %2d | [%d, %d] | [%d, %d] |"
              % ((s,) + width_rule_closed(s) + recorded_string(s)))
    n_diff = sum(width_rule_closed(s) != recorded_string(s) for s in range(1, 129))
    print("   the two functions differ at %d of the 128 values s = 1..128 (F-ad)" % n_diff)
    # log-uniform integer prior over [52, 256] and [32, 256]: mass below 64 and 128
    for lo, hi in ((52, 256), (32, 256), (64, 256)):
        f64 = (math.log(64) - math.log(lo)) / (math.log(hi) - math.log(lo)) if lo < 64 else 0.0
        f128 = (math.log(128) - math.log(lo)) / (math.log(hi) - math.log(lo))
        print("   log-uniform on [%d, %d]: mass below 64 = %.3f, below 128 = %.3f"
              % (lo, hi, f64, f128))
    print("   runner default 48 against the lower bound: 48 < 52 at (26, 12); 48 >= 32 on the bench (F-r)")
    print("   num_transforms: uniform integer on [4, 12]; runner default 3 < 4 (F-r)")


# ---------------------------------------------------------------------------
# B2  the rational-quadratic spline of zuko 1.6.0 (transforms.py:449-567)
# ---------------------------------------------------------------------------

def rqs_knots(widths, heights, derivatives, bound=BOUND, slope=SLOPE):
    """zuko MonotonicRQSTransform.__init__ in numpy: returns (horizontal, vertical, derivs)."""
    widths = np.asarray(widths, float)
    heights = np.asarray(heights, float)
    derivatives = np.asarray(derivatives, float)
    ls = math.log(slope)
    widths = widths / (1 + np.abs(2 * widths / ls))
    heights = heights / (1 + np.abs(2 * heights / ls))
    derivatives = derivatives / (1 + np.abs(derivatives / ls))

    def softmax(v):
        v = v - v.max()
        e = np.exp(v)
        return e / e.sum()
    w = np.concatenate([[0.0], softmax(widths)])
    h = np.concatenate([[0.0], softmax(heights)])
    d = np.concatenate([[0.0], derivatives, [0.0]])
    horizontal = bound * (2 * np.cumsum(w) - 1)
    vertical = bound * (2 * np.cumsum(h) - 1)
    return horizontal, vertical, np.exp(d)


def rqs_forward(x, horizontal, vertical, derivs):
    """zuko MonotonicRQSTransform._call in numpy, one scalar x."""
    bins = len(horizontal) - 1
    k = int(np.sum(horizontal < x)) - 1
    if k < 0 or k >= bins:
        return x
    x0, x1 = horizontal[k], horizontal[k + 1]
    y0, y1 = vertical[k], vertical[k + 1]
    d0, d1 = derivs[k], derivs[k + 1]
    s = (y1 - y0) / (x1 - x0)
    z = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * (s * z ** 2 + d0 * z * (1 - z)) / (s + (d0 + d1 - 2 * s) * z * (1 - z))


def block2_spline() -> None:
    print("== B2  the spline: parameters per axis, clipping bounds, domain, the identity at zero")
    for k in (6, 8, 10, 16):
        print("   K_bins = %2d : 3 K - 1 = %2d spline parameters per axis per transform" % (k, 3 * k - 1))
    ls = -math.log(SLOPE)
    print("   |ln slope| = %.4f : widths and heights clipped to +-%.4f, knot log-slopes to +-%.4f"
          % (ls, ls / 2, ls))
    print("   -> any two bin widths (heights) differ by at most a factor exp(%.4f) = %.0f;"
          % (ls, math.exp(ls)))
    print("      knot slopes in [%g, %g]; the two boundary slopes are exp(0) = 1" % (SLOPE, 1 / SLOPE))
    for k in (8, 10):
        w_min = 2 * BOUND / (1 + (k - 1) * math.exp(ls))
        print("   K_bins = %2d : the narrowest bin a conditioner can ask for is %.5f (of a domain of %g)"
              % (k, w_min, 2 * BOUND))
    lo, hi = stats.logistic.cdf(-BOUND), stats.logistic.cdf(BOUND)
    print("   domain [-%g, %g] in u = logit(theta) <-> theta in [%.5f, %.5f]" % (BOUND, BOUND, lo, hi))
    out = 2 * stats.logistic.cdf(-BOUND)
    print("   uniform theta -> u standard logistic: mass outside the domain per axis %.5f;"
          " P(at least one of %d axes outside) = %.3f; one of %d: %.3f"
          % (out, D_THETA, 1 - (1 - out) ** D_THETA, D_BENCH, 1 - (1 - out) ** D_BENCH))
    print("   standard logistic: sd = pi/sqrt(3) = %.4f, so the domain edge is %.2f sd away"
          % (math.pi / math.sqrt(3), BOUND / (math.pi / math.sqrt(3))))
    # identity at zero hyper-network output
    for k in (8, 10):
        hz, vz, dz = rqs_knots(np.zeros(k), np.zeros(k), np.zeros(k - 1))
        grid = np.linspace(-6, 6, 241)
        y = np.array([rqs_forward(x, hz, vz, dz) for x in grid])
        print("   K_bins = %2d, all spline parameters 0: knots evenly spaced (%.3f apart), slopes 1,"
              " max |y - x| on [-6, 6] = %.1e (the identity)" % (k, hz[1] - hz[0], np.abs(y - grid).max()))
    # monotone and bounded: a random spline maps [-5,5] onto [-5,5]
    rng = np.random.default_rng(0)
    hz, vz, dz = rqs_knots(rng.normal(size=10), rng.normal(size=10), rng.normal(size=9))
    grid = np.linspace(-BOUND, BOUND, 2001)
    y = np.array([rqs_forward(x, hz, vz, dz) for x in grid])
    print("   a random 10-bin spline: image of [-5, 5] is [%.3f, %.3f], monotone: %s"
          % (y.min(), y.max(), bool(np.all(np.diff(y) >= 0))))


# ---------------------------------------------------------------------------
# B3  the hyper-network: shapes, nominal and unmasked weights (zuko nn.py:218-293)
# ---------------------------------------------------------------------------

def masked_mlp_masks(d: int, context: int, hidden, total: int, order=None):
    """Replicates MaskedAutoregressiveTransform.__init__ + MaskedMLP.__init__
    for a fully autoregressive transform (passes = d) with a context of
    `context` features and `total` univariate parameters per feature.
    Returns the list of boolean masks, one per MaskedLinear, shaped (out, in)."""
    if order is None:
        order = np.arange(d)
    order = np.asarray(order)
    adjacency = order[:, None] > order                         # (d, d)
    if context > 0:
        adjacency = np.concatenate([adjacency, np.ones((d, context), bool)], axis=1)
    adjacency = np.repeat(adjacency, total, axis=0)            # (d*total, d+context)
    out_features, in_features = adjacency.shape
    uniq, inverse = np.unique(adjacency, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    precedence = (uniq.astype(float) @ uniq.astype(float).T) == uniq.sum(axis=1)[None, :]
    masks = []
    indices = None
    for i, features in enumerate(tuple(hidden) + (out_features,)):
        if i > 0:
            mask = precedence[:, indices]
        else:
            mask = uniq
        if (~mask).all():
            raise ValueError("null Jacobian")
        if i < len(hidden):
            reachable = np.nonzero(mask.sum(axis=1))[0]
            indices = reachable[np.arange(features) % len(reachable)]
            mask = mask[indices]
        else:
            mask = mask[inverse]
        masks.append(mask)
    return masks


def flow_counts(d: int, e: int, hidden: int, n_tf: int, k_bins: int, hidden_layers=None):
    """Nominal (every tensor element) and unmasked weight counts of the NSF
    `sbi` builds: n_tf transforms, each a MaskedMLP with hidden layers
    [hidden] * n_tf (sbi flow.py:1142-1143), output d * (3 k_bins - 1).
    `hidden_layers` overrides the sbi replication (zuko's own default is (64, 64))."""
    total = 3 * k_bins - 1
    layers = list(hidden_layers) if hidden_layers is not None else [hidden] * n_tf
    masks = masked_mlp_masks(d, e, layers, total)
    shapes = [m.shape for m in masks]
    nominal_w = sum(o * i for (o, i) in shapes)
    biases = sum(o for (o, _) in shapes)
    unmasked_w = sum(int(m.sum()) for m in masks)
    per_tf_nominal = nominal_w + biases
    per_tf_unmasked = unmasked_w + biases
    if hidden_layers is None:
        # closed form of the nominal count, eq. (P4.9), checked against the shapes
        closed = ((d + e) * hidden + hidden
                  + (n_tf - 1) * (hidden * hidden + hidden)
                  + hidden * d * total + d * total)
        assert closed == per_tf_nominal, (closed, per_tf_nominal)
    return {"shapes": shapes, "per_tf_nominal": per_tf_nominal,
            "per_tf_unmasked": per_tf_unmasked,
            "nominal": n_tf * per_tf_nominal, "unmasked": n_tf * per_tf_unmasked,
            "out": d * total}


def block3_hypernet() -> None:
    print("== B3  the conditioner of one transform: shapes and weight counts")
    c = flow_counts(D_THETA, E_DUP, 48, 3, 8)
    print("   runner defaults (48/3/8) at (d_theta, E) = (26, 12): MaskedLinear shapes (out, in) = %s"
          % (c["shapes"],))
    print("   -> each conditioner has n_tf = 3 hidden layers of 48 (sbi passes [hidden_features] * num_transforms)")
    rows = [
        ("runner defaults 48/3/8", D_THETA, E_DUP, 48, 3, 8),
        ("runner defaults 48/3/8 at E = 10", D_THETA, E_RUNNER, 48, 3, 8),
        ("library build_joint_model 64/5/10", D_THETA, E_DUP, 64, 5, 10),
        ("standalone NPEConfig 128/8/10", D_THETA, E_DUP, 128, 8, 10),
        ("space lower corner 52/4/10", D_THETA, E_DUP, 52, 4, 10),
        ("space lower corner, trained at 8 bins (F-a)", D_THETA, E_DUP, 52, 4, 8),
        ("space upper corner 256/12/10", D_THETA, E_DUP, 256, 12, 10),
        ("space geometric middle 115/8/10", D_THETA, E_DUP, 115, 8, 10),
        ("bench, runner defaults 48/3/8", D_BENCH, D_BENCH, 48, 3, 8),
        ("bench, space lower corner 32/4/10", D_BENCH, D_BENCH, 32, 4, 10),
        ("bench, space upper corner 256/12/10", D_BENCH, D_BENCH, 256, 12, 10),
    ]
    print("   | configuration | (d, E) | out per transform | per transform nominal | unmasked | total nominal | total unmasked | unmasked share |")
    for name, d, e, h, t, k in rows:
        c = flow_counts(d, e, h, t, k)
        print("   | %s | (%d, %d) | %d | %d | %d | %d | %d | %.2f |"
              % (name, d, e, c["out"], c["per_tf_nominal"], c["per_tf_unmasked"],
                 c["nominal"], c["unmasked"], c["unmasked"] / c["nominal"]))
    print("   linear layers per conditioner = n_tf + 1; in the whole chain n_tf (n_tf + 1): %s"
          % ", ".join("%d at n_tf = %d" % (t * (t + 1), t) for t in (3, 4, 5, 8, 12)))
    # the encoder for scale: P1's count at the runner's defaults
    print("   for scale, the encoder at the runner's defaults is 359450 weights (P1 eq. (P1.10), E = 10)")


# ---------------------------------------------------------------------------
# B4  density evaluation against sampling (zuko transforms.py:966-1007)
# ---------------------------------------------------------------------------

def block4_cost() -> None:
    print("== B4  conditioner calls: one per transform for log_prob, d_theta per transform for a sample")
    for d in (D_THETA, D_BENCH):
        for t in (3, 4, 12):
            print("   d_theta = %2d, n_tf = %2d : log_prob %2d calls ; sample / rsample %3d calls (x%d)"
                  % (d, t, t, t * d, d))
    print("   the calls are batched over draws and rows: S_mc = 128 draws for 2 B_rep = 8 wells is one"
          " batch of 1024 rows through the same %d calls (P3)" % (3 * D_THETA))


# ---------------------------------------------------------------------------
# B5  the box transform: Jacobian anchor and the initial loss
# ---------------------------------------------------------------------------

def block5_box(seed: int = 0, n: int = 200000, quick: bool = False) -> None:
    print("== B5  the unconstrained transform: the prior floor anchor and the identity-initialised flow")
    # (i) exactness: the standard logistic density at u = logit(theta) equals theta (1 - theta)
    th = np.array([1e-4, 0.01, 0.3, 0.5, 0.9, 0.9999])
    u = np.log(th / (1 - th))
    res = np.abs(stats.logistic.pdf(u) - th * (1 - th)).max()
    print("   logistic density at logit(theta) minus theta(1-theta): max |.| = %.1e over a test set" % res)
    print("   -> a flow whose unconstrained density is the product logistic has box-coordinate NLL"
          " exactly 0 nats/row: L_0 = 0 for the uniform prior on the unit cube")
    # (ii) the exact logit-normal: KL(logistic || N(0,1)) per axis
    kl = 0.5 * (math.pi ** 2 / 3) + 0.5 * math.log(2 * math.pi) - 2.0
    print("   identity-initialised flow (standard normal in u): NLL per axis = pi^2/6 + ln(2 pi)/2 - 2 = %.4f nats"
          % kl)
    f = lambda x: stats.logistic.pdf(x) * (-stats.norm.logpdf(x) + stats.logistic.logpdf(x))
    kl_num, _ = integrate.quad(f, -40, 40)
    print("   checked by quadrature: %.4f ; per row: %.2f nats at d_theta = 26, %.2f at 10"
          % (kl_num, kl * D_THETA, kl * D_BENCH))
    print("   the box Jacobian term sum_k log(theta_k (1 - theta_k)) is at most %.3f d_theta = %.2f"
          " and averages -2 d_theta = %d under the uniform prior" % (-math.log(4), -math.log(4) * D_THETA, -2 * D_THETA))
    if quick:
        return
    rng = np.random.default_rng(seed)
    th = rng.uniform(size=(n, D_THETA))
    u = np.log(th / (1 - th))
    nll_box = (-stats.norm.logpdf(u) + np.log(th * (1 - th))).sum(axis=1)
    print("   Monte Carlo (%d rows, seed %d): mean box NLL of the identity-initialised flow %.3f nats/row"
          " (sd of the mean %.3f)" % (n, seed, nll_box.mean(), nll_box.std() / math.sqrt(n)))


# ---------------------------------------------------------------------------
# B6  the layered defaults
# ---------------------------------------------------------------------------

def block6_defaults() -> None:
    print("== B6  the same three knobs on six surfaces, with the nominal weight count at (26, 12)")
    rows = [
        ("zuko NSF / MAF / MaskedMLP own defaults", 64, 3, 8, "two hidden layers of 64 whatever the transform count; never reached through sbi"),
        ("sbi build_zuko_nsf", 50, 5, 8, ""),
        ("sbi posterior_nn", 50, 5, 10, ""),
        ("joint build_joint_model", 64, 5, 10, "library default, no cluster run reaches it (F-m)"),
        ("run_joint_arms.py flags", 48, 3, 8, "what a Stage 3 arm and a Stage 4 trial without --num-bins train (F-a)"),
        ("standalone npe_model.NPEConfig", 128, 8, 10, "the other stack; z_score_x = independent"),
        ("joint space at (26, 12, 26)", None, None, 10, "hidden [52, 256] log-uniform, transforms [4, 12], bins fixed"),
    ]
    print("   | surface | hidden | transforms | bins | nominal weights | note |")
    for name, h, t, k, note in rows:
        if h is None:
            print("   | %s | [52, 256] | [4, 12] | %d | 201032 to 11129688 | %s |" % (name, k, note))
            continue
        if name.startswith("zuko"):
            c = flow_counts(D_THETA, E_DUP, h, t, k, hidden_layers=(64, 64))
        else:
            c = flow_counts(D_THETA, E_DUP, h, t, k)
        print("   | %s | %d | %d | %d | %d | %s |" % (name, h, t, k, c["nominal"], note))
    lo = flow_counts(D_THETA, E_DUP, 52, 4, 10)["nominal"]
    hi = flow_counts(D_THETA, E_DUP, 256, 12, 10)["nominal"]
    print("   space corners at (26, 12, 26): %d to %d nominal weights, a factor %.1f" % (lo, hi, hi / lo))
    a8 = flow_counts(D_THETA, E_DUP, 52, 4, 8)
    a10 = flow_counts(D_THETA, E_DUP, 52, 4, 10)
    print("   F-a at the lower corner: 8 bins gives %d weights and %d outputs per transform; the ledger's 10 bins"
          " would give %d and %d" % (a8["nominal"], a8["out"], a10["nominal"], a10["out"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the Monte Carlo check of B5")
    args = ap.parse_args()
    block1_width_rule()
    block2_spline()
    block3_hypernet()
    block4_cost()
    block5_box(quick=args.quick)
    block6_defaults()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
