"""Three-stream batches for the joint objective, eq. (2) (plan v0.6, Stage 2).

One optimiser step consumes three batches drawn from two different banks:

    sim stream  : i.i.d. prior-faithful (theta, x) from the SIMULATED bank.
                  Feeds L_NPE. theta exists only here.
    met stream  : class-balanced (x, c) from the REAL bank.
                  Feeds l_DSN. c exists only here.
    rep stream  : same-donor well PAIRS from the REAL bank.
                  Feeds L_rep. The replicate structure exists only here.

Three properties this module is responsible for, each of which has a test:

  * **The simulated stream never sees a real or surrogate row** (J4). Surrogate
    (augmented) rows carry no valid (theta, x) pair; they are real-arm rows and
    are excluded from the NPE term by construction, not by a filter that could
    be forgotten.
  * **The streams are independent given psi** (J9). Each stream draws from its
    OWN torch.Generator, so re-ordering or resizing one stream cannot perturb
    another's draws. Without this, a change to the metric batch size silently
    changes which simulated rows are seen, and any A/B comparison across
    configurations is confounded.
  * **Replicate pairs are exhaustive and exchangeable.** All within-donor pairs
    are enumerated once; a donor contributing k wells supplies k(k-1)/2 pairs.
    Donors with a single well contribute none and are counted, not silently
    dropped -- decision D12 turns on that count.

Pure ASCII, LF only. torch + numpy.
"""

import numpy as np
import torch

__all__ = ["enumerate_donor_pairs", "ThreeStreamBatcher", "BatchSpec"]


def enumerate_donor_pairs(donor_ids):
    """All unordered within-donor index pairs.

    Returns
    -------
    pairs : (P, 2) int64 array, i < j within each row
    n_singleton_donors : int, donors contributing no pair (the D12 number)
    """
    donor_ids = np.asarray(donor_ids).ravel()
    pairs = []
    n_singleton = 0
    for d in np.unique(donor_ids):
        idx = np.flatnonzero(donor_ids == d)
        if idx.size < 2:
            n_singleton += 1
            continue
        for a in range(idx.size):
            for b in range(a + 1, idx.size):
                pairs.append((int(idx[a]), int(idx[b])))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), n_singleton
    return np.asarray(pairs, dtype=np.int64), n_singleton


class BatchSpec(object):
    """Sizes of the three streams per optimiser step."""

    def __init__(self, b_sim=512, b_met=64, b_rep=8):
        self.b_sim = int(b_sim)
        self.b_met = int(b_met)
        self.b_rep = int(b_rep)
        for name, v in (("b_sim", self.b_sim), ("b_met", self.b_met),
                        ("b_rep", self.b_rep)):
            if v < 0:
                raise ValueError("%s must be >= 0" % name)

    def to_dict(self):
        return {"b_sim": self.b_sim, "b_met": self.b_met, "b_rep": self.b_rep}


class ThreeStreamBatcher(object):
    """Draws one (sim, met, rep) batch triple per call.

    Parameters
    ----------
    sim_theta : (N_sim, d) tensor
    sim_x     : (N_sim, W) tensor
    real_x    : (N_real, W) tensor or None
    real_cls  : (N_real,) int tensor or None
    real_donor: (N_real,) int array or None
    surrogate : (N_real,) bool array or None
        Augmented rows. Excluded from the replicate stream, since a surrogate
        has no well of its own, and excluded from class statistics by the DSN
        loss itself.
    spec      : BatchSpec
    seed      : int
    met_x, met_cls : tensors or None
        The SOURCE of the metric stream. Defaults to the real arm. Arms A0s
        and A2s put the metric term on the SIMULATED arm with generator labels
        (plan S2.2), and must pass the simulated x and cls here.

        [CORRECTION] Before this argument existed the metric stream always
        read `real_x`, so A2s -- whose whole purpose is to remove the domain
        asymmetry from A2 -- was training l_DSN on real rows exactly as A2
        does. The two arms differed only in name, and eq. (9)'s decomposition
        into a domain effect and an objective effect would have been computed
        from two copies of the same run.
    """

    def __init__(self, sim_theta, sim_x, real_x=None, real_cls=None,
                 real_donor=None, surrogate=None, spec=None, seed=0,
                 met_x=None, met_cls=None):
        self.sim_theta = sim_theta
        self.sim_x = sim_x
        if sim_theta.shape[0] != sim_x.shape[0]:
            raise ValueError("sim_theta and sim_x disagree in length")
        self.n_sim = int(sim_theta.shape[0])

        self.real_x = real_x
        self.real_cls = real_cls
        self.spec = spec or BatchSpec()

        # Metric-stream source. When it is the real arm, the surrogate mask
        # applies; a simulated source has no surrogates.
        self._met_is_real = met_x is None
        self.met_x = real_x if met_x is None else met_x
        self.met_cls = real_cls if met_cls is None else met_cls

        # One generator per stream: this is what makes J9 true.
        self.g_sim = torch.Generator().manual_seed(int(seed) + 1)
        self.g_met = torch.Generator().manual_seed(int(seed) + 2)
        self.g_rep = torch.Generator().manual_seed(int(seed) + 3)

        self.surrogate = (np.zeros(0, dtype=bool) if real_x is None
                          else (np.zeros(real_x.shape[0], dtype=bool)
                                if surrogate is None
                                else np.asarray(surrogate, dtype=bool)))

        self.class_index = {}
        if self.met_cls is not None:
            cls_np = np.asarray(self.met_cls).ravel()
            met_mask = (self.surrogate if self._met_is_real
                        else np.zeros(cls_np.size, dtype=bool))
            if met_mask.size != cls_np.size:
                raise ValueError("surrogate mask length %d != metric rows %d"
                                 % (met_mask.size, cls_np.size))
            for c in np.unique(cls_np):
                keep = np.flatnonzero((cls_np == c) & (~met_mask))
                if keep.size:
                    self.class_index[int(c)] = keep

        self.pairs = np.zeros((0, 2), dtype=np.int64)
        self.n_singleton_donors = 0
        if real_donor is not None:
            donors = np.asarray(real_donor).ravel()
            # A surrogate row has no well identity, so it cannot be a
            # replicate. Exclude it by MASK and map the surviving indices back.
            #
            # [CORRECTION] Relabelling surrogates with negative ids silently
            # collided when the donor array had a narrow string dtype: on a
            # '<U2' array the tenth surrogate's id '-10' truncates to '-1', so
            # two surrogates became one donor and formed a bogus replicate
            # pair. Masking has no dtype dependence at all.
            keep = np.flatnonzero(~self.surrogate)
            sub_pairs, self.n_singleton_donors = enumerate_donor_pairs(
                donors[keep])
            self.pairs = (keep[sub_pairs] if sub_pairs.size
                          else np.zeros((0, 2), dtype=np.int64))

    # -- the three streams ---------------------------------------------------

    def sim_batch(self):
        """i.i.d. rows of the SIMULATED bank. Never touches the real arm."""
        if self.spec.b_sim == 0:
            return None
        idx = torch.randint(0, self.n_sim, (self.spec.b_sim,),
                            generator=self.g_sim)
        return self.sim_theta[idx], self.sim_x[idx], idx

    def met_batch(self):
        """Class-balanced real rows, surrogates excluded from the index."""
        if self.spec.b_met == 0 or not self.class_index:
            return None
        classes = sorted(self.class_index)
        per = max(1, self.spec.b_met // len(classes))
        picks = []
        for c in classes:
            pool = self.class_index[c]
            sel = torch.randint(0, len(pool), (per,), generator=self.g_met)
            picks.append(torch.as_tensor(pool[sel.numpy()], dtype=torch.long))
        idx = torch.cat(picks)
        return self.met_x[idx], self.met_cls[idx], idx

    def rep_batch(self):
        """Same-donor well pairs."""
        if self.spec.b_rep == 0 or self.pairs.shape[0] == 0:
            return None
        n = min(self.spec.b_rep, self.pairs.shape[0])
        sel = torch.randperm(self.pairs.shape[0], generator=self.g_rep)[:n]
        p = self.pairs[sel.numpy()]
        ig = torch.as_tensor(p[:, 0], dtype=torch.long)
        igp = torch.as_tensor(p[:, 1], dtype=torch.long)
        return self.real_x[ig], self.real_x[igp], (ig, igp)

    def next(self):
        return {"sim": self.sim_batch(), "met": self.met_batch(),
                "rep": self.rep_batch()}

    def report(self):
        return "\n".join([
            "simulated rows   : %d" % self.n_sim,
            "real rows        : %d" % (0 if self.real_x is None
                                       else self.real_x.shape[0]),
            "surrogate rows   : %d" % int(self.surrogate.sum()),
            "metric source    : %s" % ("real" if self._met_is_real else "sim"),
            "classes          : %s" % sorted(self.class_index),
            "replicate pairs  : %d" % self.pairs.shape[0],
            "singleton donors : %d   <- D12: pairs are impossible for these"
            % self.n_singleton_donors,
        ])
