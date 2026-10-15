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

__all__ = ["enumerate_donor_pairs", "ThreeStreamBatcher", "BatchSpec", "ShardStream"]


class ShardStream(object):
    """The simulated stream from a rolling buffer of shards (D-084;
    specs/SPEC.md block 5).

    The stream walks the bank in passes. In each pass the shard order is a
    permutation drawn from this stream's generator; the buffer holds the
    masked rows of at most K = `buffer_shards` shards; every resident shard
    carries a QUOTA, initialised to its number of masked rows when it is
    loaded (plus any overshoot carried from its previous pass). Each row of a
    batch picks a resident shard with probability proportional to its
    remaining quota, then a row uniformly within that shard, and the shard's
    quota goes down by one. A shard whose quota is spent is evicted and the
    next shard of the pass loaded. So every row's expected number of draws
    per pass is exactly 1 whatever the shards' sizes, a pass is n_rows / B
    steps (the overshoot of a batch is carried into the shard's next quota),
    and no more than K shards' windows are ever resident.

    When K >= n_shards the buffer holds every shard in the natural order,
    never evicts and never permutes, and a batch is `torch.randint(0,
    n_rows, (B,), generator)` over the masked rows in the concatenated
    order: the in-memory batcher's draws at the same seed, bit for bit (the
    equivalence oracle of the bench test).

    Parameters
    ----------
    index : latent_bank.ShardIndex
    row_mask : (n_rows,) bool  -- the rows this stream may draw (the split)
    batch_size : int
    buffer_shards : int >= 1
    seed : int  -- the generator is seeded with seed + 1, as the in-memory
        simulated stream's is (J9: one generator per stream)
    """

    def __init__(self, index, row_mask, batch_size, buffer_shards, seed=0):
        self.index = index
        self.row_mask = np.asarray(row_mask, dtype=bool).ravel()
        if self.row_mask.size != index.n_rows:
            raise ValueError("row_mask has %d entries, the bank %d rows"
                             % (self.row_mask.size, index.n_rows))
        self.batch_size = int(batch_size)
        self.K = int(buffer_shards)
        if self.K < 1 or self.batch_size < 1:
            raise ValueError("buffer_shards and batch_size must be >= 1")
        self.seed = int(seed)
        self.g = torch.Generator().manual_seed(self.seed + 1)
        # shards that hold at least one masked row, in natural order
        self.shards = [k for k in range(index.n_shards)
                       if self.row_mask[index.offsets[k]:index.offsets[k + 1]].any()]
        if not self.shards:
            raise ValueError("no shard holds a masked row")
        self.n_rows = int(self.row_mask.sum())
        self.whole = self.K >= len(self.shards)
        self.theta = torch.as_tensor(np.asarray(index.theta), dtype=torch.float32)
        # schedule state
        self.pass_no = 0
        self.order = []          # the shard order of the current pass
        self.pos = 0             # the next shard of `order` to load
        self.buffer = []         # [(shard, rows (global ids), x tensor, quota)]
        self.carry = {}          # shard -> overshoot owed to its next quota
        self.steps = 0
        self._loads = 0
        if self.whole:
            for k in self.shards:
                self._load(k)
            self._rows = torch.as_tensor(np.concatenate([b[1] for b in self.buffer]),
                                         dtype=torch.long)
            self._x = torch.cat([b[2] for b in self.buffer], dim=0)
        else:
            self._fill()

    # -- the schedule --------------------------------------------------------

    def _new_pass(self):
        """A fresh permutation of the shards; a shard still resident is
        moved behind the first later non-resident one, so it is not loaded
        twice (its rows would then be drawn twice as often); with nearly
        every shard resident no partner exists and the duplicate is
        accepted."""
        self.pass_no += 1
        perm = torch.randperm(len(self.shards), generator=self.g).tolist()
        order = [self.shards[i] for i in perm]
        resident = set(b[0] for b in self.buffer)
        for i in range(min(self.K - 1, len(order))):
            if order[i] in resident:
                for j in range(i + 1, len(order)):
                    if order[j] not in resident:
                        order[i], order[j] = order[j], order[i]
                        break
        self.order = order
        self.pos = 0

    def _load(self, k):
        rows = self.index.rows_of(k)
        keep = self.row_mask[rows]
        rows = rows[keep]
        x = torch.as_tensor(self.index.load_x(k)[keep], dtype=torch.float32)
        quota = float(rows.size) + float(self.carry.pop(k, 0.0))
        self.buffer.append([k, rows, x, quota])
        self._loads += 1

    def _evict_spent(self):
        kept = []
        for b in self.buffer:
            if b[3] > 0.0:
                kept.append(b)
            elif b[3] < 0.0:
                # the overshoot (a negative quota) is owed to the next pass
                self.carry[b[0]] = self.carry.get(b[0], 0.0) + b[3]
        self.buffer = kept

    def _fill(self):
        """Evict the spent shards and load until K are resident. A shard
        loaded with a carried overshoot larger than its row count (a small
        shard after a batch of B > n_k picks) starts spent: it is evicted
        at once and skips this pass, which is how its overshoot is repaid."""
        self._evict_spent()
        guard = 0
        while len(self.buffer) < self.K:
            if self.pos >= len(self.order):
                self._new_pass()
            k = self.order[self.pos]
            self.pos += 1
            if any(b[0] == k for b in self.buffer):
                continue          # the duplicate case of _new_pass
            self._load(k)
            self._evict_spent()
            guard += 1
            if guard > 4 * len(self.shards) + 4 * self.K:
                raise RuntimeError("ShardStream cannot fill its buffer: every "
                                   "shard's quota is spent (carry %r)" % self.carry)

    @property
    def resident_shards(self):
        return [b[0] for b in self.buffer]

    @property
    def loads(self):
        """How many shard loads so far (a memory-boundedness witness)."""
        return self._loads

    # -- drawing -------------------------------------------------------------

    def next(self):
        """(theta (B, p), x (B, W), rows (B,) global row ids)."""
        B = self.batch_size
        if self.whole:
            idx = torch.randint(0, self.n_rows, (B,), generator=self.g)
            rows = self._rows[idx]
            self.steps += 1
            return self.theta[rows], self._x[idx], rows
        self._fill()
        # after _fill every resident quota is positive; clamp anyway, so a
        # rounding residue can never reach multinomial as a negative weight
        w = torch.tensor([max(b[3], 0.0) for b in self.buffer], dtype=torch.float64)
        pick = torch.multinomial(w, B, replacement=True, generator=self.g)
        xs, rows = [], []
        for j, b in enumerate(self.buffer):
            m = pick == j
            n_j = int(m.sum())
            if n_j == 0:
                continue
            local = torch.randint(0, int(b[1].size), (n_j,), generator=self.g)
            xs.append(b[2][local])
            rows.append(torch.as_tensor(b[1], dtype=torch.long)[local])
            b[3] -= float(n_j)
        # the picks are grouped by shard above; restore the drawn order so a
        # batch is not sorted by shard (an unsorted batch is what the
        # in-memory batcher gives too)
        x = torch.cat(xs, dim=0)
        r = torch.cat(rows, dim=0)
        order = torch.argsort(torch.argsort(pick, stable=True), stable=True)
        x, r = x[order], r[order]
        self.steps += 1
        return self.theta[r], x, r

    # -- state (block 6's checkpoint) ------------------------------------------

    def state_dict(self):
        return {"seed": self.seed, "batch_size": self.batch_size, "K": self.K,
                "pass_no": self.pass_no, "order": list(self.order), "pos": self.pos,
                "buffer": [[int(b[0]), float(b[3])] for b in self.buffer],
                "carry": dict((int(k), float(v)) for k, v in self.carry.items()),
                "steps": self.steps, "generator": self.g.get_state()}

    def load_state_dict(self, st):
        if (int(st["batch_size"]) != self.batch_size or int(st["K"]) != self.K
                or int(st["seed"]) != self.seed):
            raise ValueError("ShardStream state is for another (seed, batch, K)")
        self.pass_no = int(st["pass_no"])
        self.order = list(st["order"])
        self.pos = int(st["pos"])
        self.steps = int(st["steps"])
        self.g.set_state(st["generator"])
        if not self.whole:
            self.buffer, self.carry = [], {}
            for k, quota in st["buffer"]:
                self._load(int(k))
                self.buffer[-1][3] = float(quota)
        self.carry = dict((int(k), float(v)) for k, v in st["carry"].items())
        return self


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
                 met_x=None, met_cls=None, sim_stream=None):
        # [block 5, D-084] `sim_stream` (a ShardStream) replaces the in-memory
        # simulated tensors: the simulated batches then come from the rolling
        # buffer of shards and sim_theta / sim_x are None. Everything else is
        # unchanged, and a caller that passes tensors gets exactly the old
        # batcher.
        self.sim_stream = sim_stream
        self.sim_theta = sim_theta
        self.sim_x = sim_x
        if sim_stream is not None:
            if sim_theta is not None or sim_x is not None:
                raise ValueError("give either sim_stream or the simulated tensors")
            self.n_sim = int(sim_stream.n_rows)
        else:
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
        if self.sim_stream is not None:
            if self.sim_stream.batch_size != self.spec.b_sim:
                raise ValueError("the stream's batch size %d is not b_sim %d"
                                 % (self.sim_stream.batch_size, self.spec.b_sim))
            return self.sim_stream.next()
        idx = torch.randint(0, self.n_sim, (self.spec.b_sim,),
                            generator=self.g_sim)
        return self.sim_theta[idx], self.sim_x[idx], idx

    # -- state (block 6's checkpoint) ------------------------------------------

    def state_dict(self):
        st = {"g_sim": self.g_sim.get_state(), "g_met": self.g_met.get_state(),
              "g_rep": self.g_rep.get_state()}
        if self.sim_stream is not None:
            st["sim_stream"] = self.sim_stream.state_dict()
        return st

    def load_state_dict(self, st):
        self.g_sim.set_state(st["g_sim"])
        self.g_met.set_state(st["g_met"])
        self.g_rep.set_state(st["g_rep"])
        if self.sim_stream is not None:
            self.sim_stream.load_state_dict(st["sim_stream"])
        return self

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
            "simulated rows   : %d%s" % (self.n_sim, (
                "   (streamed: buffer %d shard(s) of %d)"
                % (self.sim_stream.K, len(self.sim_stream.shards))
                if self.sim_stream is not None else "")),
            "real rows        : %d" % (0 if self.real_x is None
                                       else self.real_x.shape[0]),
            "surrogate rows   : %d" % int(self.surrogate.sum()),
            "metric source    : %s" % ("real" if self._met_is_real else "sim"),
            "classes          : %s" % sorted(self.class_index),
            "replicate pairs  : %d" % self.pairs.shape[0],
            "singleton donors : %d   <- D12: pairs are impossible for these"
            % self.n_singleton_donors,
        ])
