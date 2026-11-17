"""Three-stream batches for the joint objective, eq. (2) (plan v0.6, Stage 2).

One optimiser step consumes three batches drawn from two different banks:

    sim stream  : i.i.d. prior-faithful (theta, x) from the SIMULATED bank.
                  Feeds L_NPE. theta exists only here.
    met stream  : class-balanced (x, c) from the REAL bank.
                  Feeds l_DSN. c exists only here.
    rep stream  : replicate PAIRS of real windows from the REAL bank.
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
  * **Replicate pairs follow one stated rule** [block 31, D-146..D-148]. A
    row's group is the value of a real-bank id array (the KEY: the donor by
    default). Under `cross_well` (block 14's F-ab, the default) every pair of
    rows of one group in two different wells is enumerated once and a step
    draws b_rep distinct pairs of that set; groups without such a pair are
    counted, not silently dropped -- decision D12 turns on that count. Under
    `within_group` the pair set is every pair of two different rows of one
    group, any well, P = {(i, j) : i < j, g_i = g_j}; it is never enumerated
    (its size grows as the square of the groups' sizes) and a step draws
    b_rep independent pairs uniform on P (GroupPairSampler).

Pure ASCII, LF only. torch + numpy.
"""

import numpy as np
import torch

__all__ = ["enumerate_donor_pairs", "ThreeStreamBatcher", "BatchSpec", "ShardStream",
           "GroupPairSampler", "REP_PAIRS"]

# [block 31, D-148 (3)] the replicate pair rules (run_joint_arms.py --rep-pairs)
REP_PAIRS = ("cross_well", "within_group")


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


def enumerate_donor_pairs(donor_ids, well_ids=None):
    """All unordered within-donor index pairs.

    With `well_ids` [block 14, F-ab], only pairs of rows of one donor in two
    DIFFERENT wells: the replicate pair of plan S2.5 is two wells of one
    donor, and two windows of one well (one realisation, one nuisance draw)
    are not two conditionally independent recordings. Without it, every pair
    of rows of one donor (the behaviour before block 14).

    Returns
    -------
    pairs : (P, 2) int64 array, i < j within each row
    n_singleton_donors : int, donors contributing no pair (the D12 number):
        with `well_ids`, the donors with rows in fewer than two wells
    """
    donor_ids = np.asarray(donor_ids).ravel()
    wells = None
    if well_ids is not None:
        wells = np.asarray(well_ids).ravel()
        if wells.size != donor_ids.size:
            raise ValueError("well_ids has %d entries, donor_ids %d"
                             % (wells.size, donor_ids.size))
    pairs = []
    n_singleton = 0
    for d in np.unique(donor_ids):
        idx = np.flatnonzero(donor_ids == d)
        if idx.size < 2 or (wells is not None
                            and np.unique(wells[idx]).size < 2):
            n_singleton += 1
            continue
        for a in range(idx.size):
            for b in range(a + 1, idx.size):
                if wells is not None and wells[idx[a]] == wells[idx[b]]:
                    continue
                pairs.append((int(idx[a]), int(idx[b])))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64), n_singleton
    return np.asarray(pairs, dtype=np.int64), n_singleton


class GroupPairSampler(object):
    """[block 31, D-147] Uniform draws from the within-group pairs
    P = {(i, j) : i < j, g_i = g_j}, without enumerating P (specs/SPEC.md
    block 31, `within_group`).

    With n_h the rows of group h and C_h = n_h (n_h - 1) / 2 its unordered
    pairs, |P| = sum_h C_h. One draw: a group h with probability
    w_h = C_h / |P|, then two distinct rows of h uniform among its C_h
    unordered pairs; so every pair of P has probability w_h / C_h = 1 / |P|.
    Two rows of one well are in P (D-146, D-147).

    Parameters
    ----------
    group_ids : (n,) array -- the group of each entry (any dtype that
        numpy.unique sorts: int ids, or strings)
    rows : (n,) int array or None -- the row index each entry stands for
        (default 0..n-1); a draw returns these indices

    Attributes
    ----------
    groups : (G,) the distinct group ids, increasing (numpy.unique)
    sizes : (G,) int64 -- n_h; pairs_per_group : (G,) int64 -- C_h
    n_pairs : int -- |P|; n_groups : G; n_groups_paired : #{h : n_h >= 2}
    """

    def __init__(self, group_ids, rows=None):
        g = np.asarray(group_ids).ravel()
        if rows is None:
            rows = np.arange(g.size, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64).ravel()
        if rows.size != g.size:
            raise ValueError("rows has %d entries, group_ids %d" % (rows.size, g.size))
        self.groups, inv = np.unique(g, return_inverse=True)
        inv = np.asarray(inv).ravel()
        self.sizes = np.bincount(inv, minlength=self.groups.size).astype(np.int64)
        # rows grouped by group (stable: row order kept within a group);
        # group h's rows are rows_by_group[offsets[h]:offsets[h + 1]]
        self.rows_by_group = rows[np.argsort(inv, kind="stable")]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.pairs_per_group = self.sizes * (self.sizes - 1) // 2
        self.n_pairs = int(self.pairs_per_group.sum())
        self.paired = np.flatnonzero(self.sizes >= 2)
        # w_h over the groups with a pair (float64; a group of one row has
        # w_h = 0 and is left out of the draw)
        self.weights = (self.pairs_per_group[self.paired].astype(np.float64)
                        / float(self.n_pairs) if self.n_pairs
                        else np.zeros(0, dtype=np.float64))

    @property
    def n_groups(self):
        return int(self.groups.size)

    @property
    def n_groups_paired(self):
        return int(self.paired.size)

    def members(self, h):
        """The rows of group index h (0..G-1), in row order."""
        return self.rows_by_group[self.offsets[h]:self.offsets[h + 1]]

    def draw(self, n, rng):
        """n independent pairs, each uniform on P.

        rng : numpy.random.Generator
        Returns (n, 2) int64 rows, i < j within each pair.

        Custom: numpy has no draw of a uniform pair from a union of groups
        without enumerating the pairs. Checked: numpy.random.Generator.choice
        over the groups (p = w) then over the rows, which is what this uses:
        the group by Generator.choice, then a uniform row a of the group and
        a uniform other row b (b drawn on n_h - 1 values and shifted past a),
        by Generator.integers, which is choice over the rows vectorised with
        a per-pick bound n_h. (a, b) is uniform on the n_h (n_h - 1) ordered
        pairs of distinct rows, so {a, b} is uniform on the C_h unordered."""
        n = int(n)
        if self.n_pairs == 0:
            raise ValueError("no within-group pair: every group has fewer than two rows")
        if n < 0:
            raise ValueError("n must be >= 0")
        k = self.paired[rng.choice(self.paired.size, size=n, p=self.weights)]
        nk = self.sizes[k]
        a = rng.integers(0, nk)
        b = rng.integers(0, nk - 1)
        b = b + (b >= a)
        i = self.rows_by_group[self.offsets[k] + a]
        j = self.rows_by_group[self.offsets[k] + b]
        return np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1).astype(np.int64)


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
    rep_pairs : "cross_well" (default) or "within_group"  [block 31]
        The replicate pair rule. cross_well: block 14's -- two rows of one
        group in two different wells (with `real_well`; without it any two
        rows of one group, as before block 14), enumerated, b_rep distinct
        pairs per step. within_group: two different rows of one group, any
        well, b_rep independent draws uniform on P per step, P never
        enumerated (GroupPairSampler); the draws use the replicate generator
        g_rep, so the checkpoint's state resumes the same sequence.
    rep_group : (N_real,) array or None  [block 31]
        The KEY array: the group of each real row. None: `real_donor`, so
        the default call is block 14's exactly.
    rep_group_key : str  [block 31]
        The KEY's name (the real bank's id array: donor, rep_group, batch or
        well), for the report lines; default "donor".
    """

    def __init__(self, sim_theta, sim_x, real_x=None, real_cls=None,
                 real_donor=None, surrogate=None, spec=None, seed=0,
                 met_x=None, met_cls=None, sim_stream=None, real_well=None,
                 rep_pairs="cross_well", rep_group=None, rep_group_key="donor"):
        # [block 14, F-ab] `real_well`: the real rows' well ids; when given,
        # a replicate pair is two rows of one donor in two different wells
        # (enumerate_donor_pairs). None keeps the pairs by donor alone.
        # [block 31] `rep_group` (the KEY array) takes the donor's place in
        # that rule, and `rep_pairs` chooses the rule; the defaults are
        # block 14's call, unchanged.
        if rep_pairs not in REP_PAIRS:
            raise ValueError("rep_pairs must be one of %s, got %r" % (REP_PAIRS, rep_pairs))
        self.rep_pairs = rep_pairs
        self.rep_group_key = str(rep_group_key)
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
        self.pairs_across_wells = real_well is not None
        # [block 31] within_group: the sampler over P (None without real
        # rows); `pairs` is None, since P is never enumerated -- a reader of
        # `pairs` fails loudly instead of seeing no pair; `n_pairs` is |P|
        # under either rule.
        self.group_sampler = None
        if self.rep_pairs == "within_group":
            self.pairs = None
        if rep_group is not None:
            rep_group = np.asarray(rep_group).ravel()
            if rep_group.size != self.surrogate.size:
                raise ValueError("rep_group has %d entries, the real arm %d rows"
                                 % (rep_group.size, self.surrogate.size))
        groups = real_donor if rep_group is None else rep_group
        if groups is not None:
            donors = np.asarray(groups).ravel()
            # A surrogate row has no well identity, so it cannot be a
            # replicate. Exclude it by MASK and map the surviving indices back.
            #
            # [CORRECTION] Relabelling surrogates with negative ids silently
            # collided when the donor array had a narrow string dtype: on a
            # '<U2' array the tenth surrogate's id '-10' truncates to '-1', so
            # two surrogates became one donor and formed a bogus replicate
            # pair. Masking has no dtype dependence at all.
            keep = np.flatnonzero(~self.surrogate)
            if self.rep_pairs == "within_group":
                # [block 31] R = the real rows that are not surrogates; the
                # rows of one well are not told apart (`real_well` unused)
                self.group_sampler = GroupPairSampler(donors[keep], rows=keep)
                # the groups contributing no pair (the D12 number of the rule)
                self.n_singleton_donors = (self.group_sampler.n_groups
                                           - self.group_sampler.n_groups_paired)
            else:
                wells = (None if real_well is None
                         else np.asarray(real_well).ravel()[keep])
                sub_pairs, self.n_singleton_donors = enumerate_donor_pairs(
                    donors[keep], wells)
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

    @property
    def n_pairs(self):
        """[block 31] |P|, the replicate pairs of the rule: the enumerated
        pairs under cross_well, sum_h n_h (n_h - 1) / 2 under within_group."""
        if self.rep_pairs == "within_group":
            return 0 if self.group_sampler is None else self.group_sampler.n_pairs
        return int(self.pairs.shape[0])

    @property
    def pair_rule_text(self):
        """[block 31] The rule in words, as the report line and the
        replicate diagnostic print it: `two rows of one <KEY> group, any
        well` (within_group); `two wells of one <KEY> group` (cross_well).
        With KEY donor under cross_well the text is block 14's, unchanged:
        `two wells of one donor` (`any two rows of one donor` without well
        ids), so that a call without the new flags prints what it printed
        before (sec. 4; B14's runner check reads that line)."""
        key = self.rep_group_key
        if self.rep_pairs == "within_group":
            return "two rows of one %s group, any well" % key
        if key == "donor":
            return ("two wells of one donor" if self.pairs_across_wells
                    else "any two rows of one donor")
        return ("two wells of one %s group" % key if self.pairs_across_wells
                else "any two rows of one %s group" % key)

    def draw_within_pairs(self, n):
        """[block 31] n independent pairs uniform on P (within_group), as an
        (n, 2) int64 array of real-row indices, i < j. One call is one
        step's draw: a 62-bit seed taken from the replicate generator g_rep
        (whose state the batcher's state_dict, and so block 6's checkpoint,
        carries) seeds the numpy Generator that GroupPairSampler.draw uses;
        so the same seed gives the same pairs, and a restored state the same
        continuation."""
        if self.group_sampler is None or self.group_sampler.n_pairs == 0:
            raise ValueError("no within-group pair to draw")
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=self.g_rep).item())
        return self.group_sampler.draw(int(n), np.random.default_rng(seed))

    def rep_batch(self):
        """Replicate pairs: under cross_well (block 14) b_rep distinct pairs
        of the enumerated set; under within_group (block 31) b_rep
        independent pairs uniform on P, so a step may repeat a pair."""
        if self.spec.b_rep == 0 or self.n_pairs == 0:
            return None
        if self.rep_pairs == "within_group":
            p = self.draw_within_pairs(self.spec.b_rep)
        else:
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
        lines = [
            "simulated rows   : %d%s" % (self.n_sim, (
                "   (streamed: buffer %d shard(s) of %d)"
                % (self.sim_stream.K, len(self.sim_stream.shards))
                if self.sim_stream is not None else "")),
            "real rows        : %d" % (0 if self.real_x is None
                                       else self.real_x.shape[0]),
            "surrogate rows   : %d" % int(self.surrogate.sum()),
            "metric source    : %s" % ("real" if self._met_is_real else "sim"),
            "classes          : %s" % sorted(self.class_index),
            "replicate pairs  : %d  (%s)" % (self.n_pairs, self.pair_rule_text),
        ]
        if self.rep_pairs == "within_group":
            # [block 31] the groups with a pair, of the groups of the
            # non-surrogate real rows
            gs = self.group_sampler
            lines.append("replicate groups : %d of %d"
                         % ((gs.n_groups_paired, gs.n_groups) if gs is not None
                            else (0, 0)))
        elif self.rep_group_key == "donor":
            lines.append("singleton donors : %d   <- D12: pairs are impossible for these"
                         % self.n_singleton_donors)
        else:
            lines.append("singleton groups : %d   <- D12: pairs are impossible for "
                         "these (%s groups)" % (self.n_singleton_donors,
                                                self.rep_group_key))
        return "\n".join(lines)
