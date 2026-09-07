"""Realisation latent: the per-well draw that stands for the connectivity
graph G, at fixed theta.

Why this module exists (plan v0.6, S2.5c and S4.0)
--------------------------------------------------
The Weibull kernel axes (p0_conn, d0_conn, beta_conn) parameterise a
DISTRIBUTION over connections. Two wells of one donor share that distribution;
they do not share the realised graph drawn from it. The realisation is a latent
nuisance marginalised inside p(x | theta):

    p(x | theta) = integral p(x | theta, G) p(G | theta) dG.              (R1)

Two consequences the bench must reproduce, and the current ANN bank does not:

1. Two wells of one donor must share theta EXACTLY and draw INDEPENDENT
   realisations. Without this, smoke test S10 cannot run and neither can J13b,
   which is the check behind decision D17.
2. The marginal law of x at fixed theta must be invariant to the realisation
   seed -- (R1) says the seed is integrated out, so it must not carry
   information. If it does, the "realisation" is really a parameter and the
   whole S2.5c argument fails.

Distinction from nu, which is easy to lose: nu acts AFTER the dynamics and can
be inverted (latent_nuisance.py); G acts ON the dynamics and cannot. That is
why G is inside the likelihood and nu is outside it.

Pure ASCII, LF only. numpy only.
"""

import hashlib

import numpy as np


class RealisationSpec(object):
    """How realisations are assigned.

    Parameters
    ----------
    share_within : tuple of str
        Levels WITHIN WHICH the realisation is constant. The default
        ("subregion",) means the graph is constant within a subregion, so all
        windows of a subregion see the same graph -- which is why they are
        exchangeable but NOT independent (S2.7) -- while different subregions
        of one well, and different wells of one donor, each draw their own.

        [CORRECTION] The condition was inverted: naming "subregion" removed it
        from the key, so every subregion of a well shared one graph and the
        plan's Lever 2 (S2.6: "subregions of one well share every nuisance
        factor but differ in local graph realisation") was reversed. The
        nesting that breaks the nuisance/mechanism confound was therefore
        absent from the bench while appearing to be configured.
    n_per_theta : int
        How many independent realisations to generate per theta value when
        building a bank designed for J13b. 1 reproduces the current ANN bank's
        structure (and its D17 confound); >= 2 is what the check needs.
    """

    def __init__(self, share_within=("subregion",), n_per_theta=2):
        self.share_within = tuple(share_within)
        self.n_per_theta = int(n_per_theta)
        if self.n_per_theta < 1:
            raise ValueError("n_per_theta must be >= 1")

    def to_dict(self):
        return {
            "share_within": list(self.share_within),
            "n_per_theta": self.n_per_theta,
        }


def realisation_key(spec, donor, well, subregion):
    """The identity of a realisation: everything NOT in share_within varies.

    A well is always part of the key -- two wells of one donor must differ, by
    S2.5c. Naming a level below the well in `share_within` adds it to the key,
    so the realisation varies across that level and is constant within it.
    """
    parts = ["donor=%s" % donor, "well=%s" % well]
    if "subregion" in spec.share_within:
        # Constant WITHIN a subregion, hence the subregion identifies it.
        parts.append("subregion=%s" % subregion)
    return "|".join(parts)


def realisation_seed(base_seed, key):
    """A 64-bit seed derived from the key by hashing.

    Hashing rather than arithmetic, so that two different keys cannot alias
    onto one stream by an unlucky choice of base_seed.
    """
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("ascii"))
    h.update(b"|realisation|")
    h.update(str(key).encode("ascii"))
    return int.from_bytes(h.digest()[:8], "big")


def realisation_id(spec, base_seed, donor, well, subregion):
    """Stable integer id written into the bank as `realisation_id`."""
    return realisation_seed(base_seed, realisation_key(spec, donor, well,
                                                       subregion))


def assign_realisations(spec, base_seed, donors, wells, subregions):
    """Vectorised assignment over rows.

    Returns
    -------
    ids : (n,) uint64 array of realisation ids
    seeds : (n,) int list of the same values, for passing to a provider
    """
    donors = list(donors)
    wells = list(wells)
    subregions = list(subregions)
    n = len(wells)
    if not (len(donors) == len(subregions) == n):
        raise ValueError("donors, wells, subregions must be the same length")
    ids = np.empty(n, dtype=np.uint64)
    for i in range(n):
        ids[i] = realisation_id(spec, base_seed, donors[i], wells[i],
                                subregions[i])
    return ids, [int(v) for v in ids]


def distinct_realisations_per_theta(theta, realisation_ids, decimals=12):
    """Count distinct realisations per distinct theta row.

    This is the D17 audit in one function: run it on the bench bank (must be
    >= 2 for the designated subset) and, at Stage 6, on the REAL bank keyed on
    the three Weibull axes (currently expected to be 1, which is the whole
    problem).

    Returns
    -------
    counts : dict theta-key -> number of distinct realisation ids
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    realisation_ids = np.asarray(realisation_ids).ravel()
    if theta.shape[0] != realisation_ids.size:
        raise ValueError("theta and realisation_ids disagree in length")
    counts = {}
    for i in range(theta.shape[0]):
        key = tuple(np.round(theta[i], decimals))
        counts.setdefault(key, set()).add(int(realisation_ids[i]))
    return {k: len(v) for k, v in counts.items()}
