"""The realisation floor: dispersion from graph sampling alone.

New in plan v0.6 (S2.6 Lever 3, extended). Fix theta* AND nu, redraw ONLY the
connectivity realisation G, and measure the dispersion of the posterior means.

This is the quantity decision D17 turns on. The Weibull kernel axes describe a
DISTRIBUTION over connections; the realised graph is a latent nuisance
marginalised inside p(x | theta). If the bank contains one realisation per
kernel-parameter value -- which it does, 383 topology draws with the kernel
axes drawn per directory -- then what the flow learns about those axes is
partly the identity of one graph. The realisation floor measures how much of
the apparent information sits there.

The validation the plan asks for: the floor must CONCENTRATE on the kernel
axes. If it spreads across the neuron/synapse axes instead, the realisation is
not acting as a nuisance on the connectivity block and D17's framing is wrong.

Pure ASCII, LF only.
"""

import numpy as np

from floor_core import concentration_on, floor_summary, posterior_means_over

__all__ = ["realisation_floor", "kernel_axis_concentration"]


def realisation_floor(model, spec, provider, theta_star, realisation_spec,
                      n_draws=64, base_seed=0, nuisance_spec=None,
                      n_draws_post=128, param_names=None):
    """Cov of posterior means over realisation draws at fixed theta* and nu.

    nu is held at ZERO throughout, which is the identity map after the
    centring fix, so nothing of the nuisance layer leaks into this floor.
    Each draw gets its own well identity, which is what makes the realisation
    seed differ; everything else is held.
    """
    keys = [("FLOOR", "W%d" % i, 0) for i in range(int(n_draws))]
    means = posterior_means_over(
        model, spec, provider, theta_star, n_draws, realisation_spec,
        base_seed=base_seed, nuisance_spec=nuisance_spec,
        nu_rows=None, realisation_keys=keys, n_draws_post=n_draws_post)

    out = floor_summary(means, param_names=param_names)
    out["kind"] = "realisation"
    out["means"] = means.tolist()
    return out


def kernel_axis_concentration(floor, kernel_idx):
    """Fraction of the realisation floor's variance on the kernel axes.

    The D17 validation. A value near 1 means the realisation acts as a
    nuisance on the connectivity block, as S2.5c assumes. A value near
    len(kernel_idx)/d means it is spread evenly, i.e. the realisation is
    moving the whole parameter vector and the kernel axes are not special --
    in which case masking them out of the loss does not address the confound.
    """
    d = len(floor["axes"])
    k = len(list(kernel_idx))
    share = concentration_on(kernel_idx, floor)
    uniform = k / float(d)
    # Threshold: halfway between the uniform reference and 1.
    #
    # [CORRECTION] The first version used "twice uniform", which is
    # UNREACHABLE whenever the subset is half the axes or more: with 3 kernel
    # axes of 6 the uniform reference is 0.50 and the threshold 1.00, so a
    # floor carrying 100.0% of its variance on exactly those axes was reported
    # as "not concentrated". On the real bank (3 of 26) the old rule happened
    # to be usable, which is why it survived until a 6-axis bench case was
    # tried. A midpoint rule is attainable at every k and reduces to a
    # sensible 0.558 at k/d = 3/26.
    threshold = 0.5 * (1.0 + uniform)
    return {
        "concentration": share,
        "uniform_reference": uniform,
        "threshold": threshold,
        "excess_over_uniform": share - uniform,
        "concentrated": bool(share > threshold),
    }
