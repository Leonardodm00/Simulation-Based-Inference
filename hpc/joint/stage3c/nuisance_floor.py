"""The nuisance floor: how much heterogeneity does pure nuisance manufacture?

Plan v0.6, S2.6 Lever 3. Fix one theta*, vary ONLY the observation-level
nuisance nu -- gain, baseline, detection threshold, electrode dropout, slow
drift -- push each draw through the whole pipeline (simulator, encoder, flow),
and measure the dispersion of the resulting posterior means.

That dispersion is the floor: apparent mechanistic variation that no biology
produced. It is the exact analogue of measuring delta_min from a shuffled
control rather than guessing it, and it needs no new biology and no new data.

Two properties make the result readable, and both are enforced here rather
than assumed:

  * nu = 0 is the identity map, so the floor is centred on the no-nuisance
    observable and its mean is the noise-free posterior mean;
  * the graph realisation and the spike seed are HELD across draws (common
    random numbers, see floor_core), so their variance is not counted as
    nuisance.

Pure ASCII, LF only.
"""

import numpy as np

from floor_core import concentration_on, floor_summary, posterior_means_over

__all__ = ["nuisance_floor", "validate_against_injected"]


def nuisance_floor(model, spec, provider, theta_star, nuisance_spec,
                   realisation_spec, n_draws=64, base_seed=0,
                   n_draws_post=128, param_names=None, components=None):
    """Cov of posterior means over nu draws at fixed theta*.

    `components` restricts the variation to a subset of nu's components (by
    index into NU_COMPONENTS), which is how the per-factor attribution in
    S2.7's "closing the loop" is built: run the floor once per component and
    compare each one's leading direction against the aliasing prediction
    delta_theta_m of eq. (4).
    """
    from latent_nuisance import N_COMPONENTS, sample_nuisance

    nu, _ = sample_nuisance(nuisance_spec, ["B"] * n_draws,
                            ["D%d" % i for i in range(n_draws)],
                            ["W%d" % i for i in range(n_draws)],
                            base_seed=base_seed)
    if components is not None:
        keep = np.zeros(N_COMPONENTS, dtype=bool)
        keep[np.asarray(list(components), dtype=int)] = True
        nu = nu * keep[None, :]

    means = posterior_means_over(
        model, spec, provider, theta_star, n_draws, realisation_spec,
        base_seed=base_seed, nuisance_spec=nuisance_spec, nu_rows=nu,
        realisation_keys=None, n_draws_post=n_draws_post)

    out = floor_summary(means, param_names=param_names)
    out["kind"] = "nuisance"
    out["components_varied"] = ("all" if components is None
                                else list(components))
    out["nu_used"] = nu.tolist()
    out["means"] = means.tolist()
    return out


def validate_against_injected(floor, injected_directions, top_k=1):
    """Does the recovered subspace match the nu directions that were injected?

    Plan Stage 3c: "on the bench the recovered nuisance subspace must match
    the injected nu directions". Reported as the principal angle between the
    leading floor direction(s) and the span of the injected ones -- an angle,
    not a correlation, because the comparison is between SUBSPACES and a sign
    flip of an eigenvector is meaningless.

    Returns cos of the smallest principal angle in [0, 1]; 1 is exact
    agreement. The machinery is not applied to real data until this passes.
    """
    A = np.atleast_2d(np.asarray(
        [d["vector"] for d in floor["directions"][:top_k]], dtype=np.float64)).T
    B = np.atleast_2d(np.asarray(injected_directions, dtype=np.float64))
    if B.shape[0] != A.shape[0]:
        B = B.T
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.clip(s.max(), 0.0, 1.0))
