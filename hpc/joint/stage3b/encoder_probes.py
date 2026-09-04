"""Two direct probes of the DOMAIN effect, needing no new training.

Plan v0.6, Stage 3b. The decomposition of eq. (9) says HOW LARGE the domain
effect is; these two say WHAT IT IS. Both run on an existing checkpoint and
the two banks, in seconds.

Probe 1 -- per-layer activation statistics, simulated vs real input
-------------------------------------------------------------------
An encoder fitted on real windows and then applied to simulated ones may be
operating off its training support. The visible signature is dead or saturated
units: a ReLU stage whose input distribution has shifted downward outputs zero
for most simulated rows, and every layer after it sees a degenerate input. The
probe records, per layer and per domain, the fraction of exactly-zero
activations, the fraction beyond a saturation threshold, and the standardised
mean shift between domains.

This is a DIAGNOSTIC, not a test. A large dead fraction on simulated input is
strong evidence of off-support operation; a small one does not prove the
domains match, because a linear layer can shift its output distribution
without any unit dying.

One reading trap, measured rather than assumed: the DSN convolutions are
BIAS-FREE, so an exactly-zero output means the receptive field was entirely
zero. On IFR traces with silent stretches that happens for a third of the
entries at the first stage, and it is a property of the DATA, not of the unit.
`dead_frac` is therefore only interpretable as a sim-vs-real DIFFERENCE, which
is what `flag_degenerate_layers` requires (dead on simulated input AND not
dead on real). The absolute number on its own means very little.

Probe 2 -- the marginal along the real cloud's surviving direction
------------------------------------------------------------------
The r2 encoder's real-arm embedding cloud has r_eff = 1.000 of E = 10 [KB], so
it has essentially ONE direction. Project both clouds onto it. Simulated
embeddings PILING AT ONE END is off-support: the flow is then asked to
evaluate where it has no training mass. Simulated embeddings SPREAD ACROSS the
real range is not off-support, and points instead at O2 -- the prior
predictive genuinely varying less than the cohort in the features this encoder
responds to, which is simulator work rather than encoder work.

The two outcomes call for different fixes, which is why the probe reports the
tail mass on each side separately rather than a single distance.

Pure ASCII, LF only.
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["layer_activation_stats", "embedding_direction_probe",
           "effective_rank"]

# Layer types worth hooking. Normalisations are included because a shifted
# input shows up in their output before it shows up as a dead ReLU.
_HOOKABLE = (nn.Conv1d, nn.Linear, nn.ReLU, nn.GELU, nn.SiLU, nn.GroupNorm,
             nn.LayerNorm)


def _stats(a, sat_z=4.0):
    a = a.detach().reshape(-1).cpu().numpy().astype(np.float64)
    if a.size == 0:
        return None
    sd = float(a.std())
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": sd,
        "dead_frac": float(np.mean(a == 0.0)),
        "sat_frac": (0.0 if sd == 0.0
                     else float(np.mean(np.abs(a - a.mean()) > sat_z * sd))),
        "p01": float(np.percentile(a, 1)),
        "p99": float(np.percentile(a, 99)),
    }


def layer_activation_stats(encoder, x_sim, x_real, max_rows=512, sat_z=4.0):
    """Forward hooks on every hookable layer, once per domain.

    Returns
    -------
    dict layer_name -> {"sim": stats, "real": stats, "mean_shift_z": float}

    `mean_shift_z` is (mean_sim - mean_real) / std_real, the standardised shift
    of that layer's output distribution between domains. It is reported with
    std_real in the same record: a large z against a tiny std_real is a
    degenerate layer, not a large shift, and the two must not be confused.
    """
    encoder.eval()
    captured = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                # CLONE, do not keep a reference.
                #
                # [CORRECTION] The DSN backbone uses nn.ReLU(inplace=True), so
                # a stored reference to a conv output is REWRITTEN by the
                # activation that follows it. Probe 1 was reporting 38% dead
                # units on convolution layers -- which is implausible for a
                # convolution and is in fact the ReLU's own output, attributed
                # to the wrong layer. Every "dead conv" reading before this
                # fix was an artefact.
                captured[name] = out.detach().clone()
        return hook

    handles = []
    for name, mod in encoder.named_modules():
        if isinstance(mod, _HOOKABLE):
            handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        per_domain = {}
        for tag, x in (("sim", x_sim), ("real", x_real)):
            captured.clear()
            with torch.no_grad():
                encoder(x[:max_rows])
            per_domain[tag] = {k: _stats(v, sat_z) for k, v in captured.items()}
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name in sorted(set(per_domain["sim"]) & set(per_domain["real"])):
        s, r = per_domain["sim"][name], per_domain["real"][name]
        if s is None or r is None:
            continue
        z = (float("nan") if r["std"] == 0.0
             else (s["mean"] - r["mean"]) / r["std"])
        out[name] = {"sim": s, "real": r, "mean_shift_z": z}
    return out


def flag_degenerate_layers(stats, dead_threshold=0.90, z_threshold=3.0):
    """Layers where the simulated domain looks degenerate. Ordered, not scored.

    Two separate flags because they mean different things: `dead` says the
    layer stopped responding on simulated input; `shifted` says its output
    distribution moved but is still alive.
    """
    flags = []
    for name, rec in stats.items():
        dead = rec["sim"]["dead_frac"] >= dead_threshold \
            and rec["real"]["dead_frac"] < dead_threshold
        shifted = np.isfinite(rec["mean_shift_z"]) \
            and abs(rec["mean_shift_z"]) >= z_threshold
        if dead or shifted:
            flags.append({"layer": name, "dead": bool(dead),
                          "shifted": bool(shifted),
                          "dead_frac_sim": rec["sim"]["dead_frac"],
                          "dead_frac_real": rec["real"]["dead_frac"],
                          "mean_shift_z": rec["mean_shift_z"],
                          "std_real": rec["real"]["std"]})
    return sorted(flags, key=lambda f: -abs(f["mean_shift_z"] or 0.0))


def effective_rank(z):
    """Participation-ratio effective rank. Duplicated deliberately: this
    module must run without the Stage 3 package on the path."""
    z = np.asarray(z, dtype=np.float64)
    c = np.atleast_2d(np.cov(z - z.mean(0, keepdims=True), rowvar=False))
    ev = np.clip(np.linalg.eigvalsh(c), 0.0, None)
    denom = float((ev ** 2).sum())
    return 1.0 if denom <= 0.0 else float((ev.sum() ** 2) / denom)


def embedding_direction_probe(z_sim, z_real, n_dirs=1):
    """Project both clouds onto the REAL cloud's leading direction(s).

    The direction comes from the real cloud because the question is whether
    the simulated rows land where the encoder has real training mass. Using
    the pooled cloud would let the simulated rows define the direction they
    are then measured against.

    Returns a dict per direction with the two clouds' quantiles, the fraction
    of simulated projections below/above the real range, and a one-dimensional
    two-sample statistic.
    """
    z_sim = np.asarray(z_sim, dtype=np.float64)
    z_real = np.asarray(z_real, dtype=np.float64)
    if z_sim.shape[1] != z_real.shape[1]:
        raise ValueError("embedding dimensions differ: %d vs %d"
                         % (z_sim.shape[1], z_real.shape[1]))

    mu = z_real.mean(0, keepdims=True)
    c = np.atleast_2d(np.cov(z_real - mu, rowvar=False))
    ev, evec = np.linalg.eigh(c)
    order = np.argsort(-ev)
    out = {
        "r_eff_real": effective_rank(z_real),
        "r_eff_sim": effective_rank(z_sim),
        "eigenvalue_share_real": (ev[order] / ev[order].sum()).tolist()
        if ev.sum() > 0 else None,
        "directions": [],
    }

    for k in range(min(int(n_dirs), z_real.shape[1])):
        v = evec[:, order[k]]
        ps = (z_sim - mu) @ v
        pr = (z_real - mu) @ v
        lo, hi = float(pr.min()), float(pr.max())
        below = float(np.mean(ps < lo))
        above = float(np.mean(ps > hi))
        # Where in the real distribution do the simulated points sit? A value
        # near 0 or 1 means they pile at one end -- off-support. Near 0.5 with
        # a small spread means they sit in the middle of the real range but do
        # not span it, which is the O2 reading, not off-support.
        rank = float(np.mean(np.searchsorted(np.sort(pr), ps)
                             / max(len(pr), 1)))
        # Kolmogorov-Smirnov, computed here to avoid a scipy import in a
        # module that is otherwise numpy-only.
        allv = np.sort(np.concatenate([ps, pr]))
        cdf_s = np.searchsorted(np.sort(ps), allv, side="right") / len(ps)
        cdf_r = np.searchsorted(np.sort(pr), allv, side="right") / len(pr)
        ks = float(np.max(np.abs(cdf_s - cdf_r)))

        out["directions"].append({
            "index": int(k),
            "variance_share": float(ev[order[k]] / ev[order].sum())
            if ev.sum() > 0 else None,
            "real_range": [lo, hi],
            "sim_quantiles": np.percentile(ps, [1, 25, 50, 75, 99]).tolist(),
            "real_quantiles": np.percentile(pr, [1, 25, 50, 75, 99]).tolist(),
            "sim_frac_below_real_range": below,
            "sim_frac_above_real_range": above,
            "sim_mean_rank_in_real": rank,
            "ks": ks,
            "reading": _read_direction(below, above, rank, ps, pr),
        })
    return out


def _read_direction(below, above, rank, ps, pr):
    """One sentence naming which of the three explanations this looks like."""
    outside = below + above
    if outside > 0.25 and (below > 4 * above or above > 4 * below):
        return ("off-support: %.0f%% of simulated embeddings fall outside the "
                "real range, piled at one end. Fit the encoder on simulated "
                "windows (the domain fix)." % (100 * outside))
    if outside > 0.25:
        return ("off-support on BOTH sides (%.0f%%): the simulated cloud is "
                "wider than the real one, not shifted off it."
                % (100 * outside))
    spread = (float(np.std(ps)) / float(np.std(pr))) if np.std(pr) > 0 else \
        float("nan")
    if np.isfinite(spread) and spread < 0.5:
        return ("inside the real range but %.2fx its spread: consistent with "
                "O2 -- the prior predictive varying LESS than the cohort in "
                "the features this encoder responds to. Simulator work, not "
                "encoder work." % spread)
    return ("simulated embeddings span the real range (%.0f%% outside, spread "
            "ratio %.2f): no off-support signature on this direction."
            % (100 * outside, spread))
