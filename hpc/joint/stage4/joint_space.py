#!/usr/bin/env python3
"""
joint_space.py -- the Stage 4 joint search space (plan S5.1) and its
campaigns (S5.2).

What this module is for
-----------------------
One space unions the DSN's searched axes, the NPE stack's, and the two loss
weights, so a single ledger can rank configurations that differ in the
encoder, the flow, the optimiser and the loss composition at once. Everything
here is pure bookkeeping: no torch, no sbi, no training. It converts between

    a skopt POINT (a positional list over the FREE axes of one campaign)
    <-> a CONFIG (a flat dict over ALL axes, frozen ones filled in)

and it canonicalises the coordinates that the configuration does not read, so
that two points differing only in inactive coordinates build BYTE-IDENTICAL
configurations. That last property is what makes the ledger's deduplication
honest, and it is the precondition for the per-finalist control test: a
control is only a measurement of the candidate's own noise floor if the two
configurations differ in EXACTLY one respect, the shuffled pairing.

Three things this module deliberately does not decide
-----------------------------------------------------
1. WHICH campaigns to run (D4). All four of S5.2 are defined here as data;
   the choice is a runtime argument. Changing it must not require an edit.
2. The budget per campaign. That belongs to the launcher.
3. Whether an axis helps. Gates gate, NLL ranks; nothing here optimises.

Axis inventory (S5.1)
---------------------
  encoder    depth_exponent, width_multiplier, block_family, embedding_size,
             head_fusion, dropout
  DSN loss   dsn_on, log10_lambda_dsn, loss_type, mining_strategy, margin,
             angular_alpha_deg, lambda_sep
  replicate  rep_on, log10_lambda_rep, warmup_frac_rep, n_posterior_draws
  flow       hidden_features, num_transforms
  optimiser  lr, one_minus_beta1, weight_decay, batch_size_npe

Fixed by S5.1 and therefore ABSENT from this space: stem_width, group_width,
the GroupNorm settings, kernels and strides, head_pool_ops, strict_semihard,
sep_warmup_frac, num_bins, one_minus_beta2. `strict_semihard` is fixed but
still passes through the legality projection, because the projection may move
it and a silently unprojected value would make two nominally different cells
the same experiment.

Ranges are taken from the DSN's own `config.SearchConfig` and the NPE stack's
`npe_tune_search.default_space`, not invented here; see RANGE_PROVENANCE.

Pure ASCII, LF only.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "JOINT_KNOB_ORDER", "BLOCKS", "CAMPAIGNS", "RANGE_PROVENANCE",
    "JointSpaceSpec", "default_joint_space", "Campaign",
    "campaign", "free_axes", "space_dimensions",
    "config_from_point", "point_from_config", "canonicalise_config",
    "config_key", "configs_identical", "diff_configs",
    "control_config_from", "assert_control_identity",
    "apply_control_shuffle", "provenance_report",
    "adapter", "boundary_axes", "resolved_pins",
]


# ---------------------------------------------------------------------------
# The axes
# ---------------------------------------------------------------------------

# Order is fixed and load-bearing: a skopt point is a positional list, so the
# mapping between a point and a configuration is this tuple and nothing else.
# Grouped by block, blocks in the order S5.1 lists them. NEVER reorder; append
# only, and only at the end of the block concerned.
JOINT_KNOB_ORDER: Tuple[str, ...] = (
    # encoder
    "depth_exponent", "width_multiplier", "block_family", "embedding_size",
    "head_fusion", "dropout",
    # DSN loss
    "dsn_on", "log10_lambda_dsn", "loss_type", "mining_strategy",
    "margin", "angular_alpha_deg", "lambda_sep",
    # replicate term
    "rep_on", "log10_lambda_rep", "warmup_frac_rep", "n_posterior_draws",
    # flow
    "hidden_features", "num_transforms",
    # optimiser
    "lr", "one_minus_beta1", "weight_decay", "batch_size_npe",
)

BLOCKS: Dict[str, Tuple[str, ...]] = {
    "encoder": ("depth_exponent", "width_multiplier", "block_family",
                "embedding_size", "head_fusion", "dropout"),
    "dsn_loss": ("dsn_on", "log10_lambda_dsn", "loss_type",
                 "mining_strategy", "margin", "angular_alpha_deg",
                 "lambda_sep"),
    "replicate": ("rep_on", "log10_lambda_rep", "warmup_frac_rep",
                  "n_posterior_draws"),
    "flow": ("hidden_features", "num_transforms"),
    "optimiser": ("lr", "one_minus_beta1", "weight_decay", "batch_size_npe"),
}

# Where each range came from, so a reader can check it against the source
# rather than trust this file. Reported by `provenance_report()`.
RANGE_PROVENANCE: Dict[str, str] = {
    "depth_exponent": "DSN config.SearchConfig.depth_exponent_range",
    "width_multiplier": "DSN config.SearchConfig.width_multiplier_range",
    "block_family": "DSN config.SearchConfig.block_family_choices",
    "embedding_size": "DSN config.SearchConfig.embedding_size_range",
    "head_fusion": "DSN config.SearchConfig.head_fusion_choices",
    "dropout": "DSN config.SearchConfig.dropout_range",
    "dsn_on": "campaign switch (plan S5.2)",
    "log10_lambda_dsn": "plan S5.1, [-3, 1]",
    "loss_type": "DSN condition_space.LOSS_TYPES",
    "mining_strategy": "DSN condition_space.MINING_STRATEGIES",
    "margin": "DSN config.SearchConfig.margin_range",
    "angular_alpha_deg": "DSN config.SearchConfig.angular_alpha_deg_range",
    "lambda_sep": "DSN config.SearchConfig.lambda_sep_range",
    "rep_on": "campaign switch (plan S5.2)",
    "log10_lambda_rep": "plan S5.1, same treatment as lambda_dsn",
    "warmup_frac_rep": "DSN config.SearchConfig.sep_warmup_frac_range",
    "n_posterior_draws": "plan S5.1: lower bound >= 4 d_theta",
    "hidden_features": "npe_tune_search.default_space, shape-resolved",
    "num_transforms": "npe_tune_search.default_space",
    "lr": "npe_tune_search.default_space.learning_rate",
    "one_minus_beta1": "DSN config.SearchConfig.one_minus_beta1_range",
    "weight_decay": "DSN config.SearchConfig.weight_decay_range",
    "batch_size_npe": "npe_tune_search.default_space.batch_sizes",
}

# The value an inactive axis is pinned to. These are the DSN's own base
# defaults (config.TrainConfig), NOT zeros: an inactive axis is still READ by
# some code paths -- under "joint"/"joint_sep" the margin is read as
# 2 * m_cos even though it is not searched -- so pinning it to 0 would change
# the experiment rather than neutralise the coordinate.
# INVARIANT, and it is not decoration: every canonical value must lie INSIDE
# that axis's searched range. A canonicalised config is replayed into the GP
# through point_from_config, and skopt rejects a point outside the space --
# so a canonical value out of range makes the whole ledger unreplayable the
# first time an inactive axis appears in it. Asserted by J27.
#
# `None` means "resolve from the spec at canonicalisation time": the value is
# the axis's own lower bound, used where a fixed constant cannot satisfy the
# invariant because the bound itself depends on the problem.
INACTIVE_CANONICAL: Dict[str, Any] = {
    "log10_lambda_dsn": 0.0,
    "loss_type": "triplet",
    "mining_strategy": "hard",
    # [CORRECTION] These three were taken from the DSN's config.TrainConfig
    # (margin 0.3). That is the wrong base config for THIS stack: the loss
    # is built by stage2/dsn_loss_adapter.DSNLossConfig, whose margin default
    # is 0.2. Under "joint"/"joint_sep" the margin is still READ, so a
    # canonical 0.3 here against a 0.2 in the runner meant the tuner and the
    # bench would train different losses while recording the same cell.
    # The base config that BUILDS the loss defines the clamp constants.
    # Agreement is asserted by J35; if DSNLossConfig's defaults move, these
    # must move with them.
    "margin": 0.2,                 # DSNLossConfig.margin
    "angular_alpha_deg": 18.0,     # DSNLossConfig.angular_alpha_deg
    "lambda_sep": 0.1,             # DSNLossConfig.lambda_sep
    "log10_lambda_rep": 0.0,
    "warmup_frac_rep": 0.0,
    # 4 * d_theta, resolved from the spec. Zero would read as "no draws",
    # which is true when rep_on = 0 -- and is exactly the trap: it is
    # OUTSIDE [4 d_theta, max], so a config carrying it cannot be told to
    # the optimiser. The term is off, so the value is inert either way; the
    # floor is the inert value that is also a legal coordinate.
    "n_posterior_draws": None,
}


def _inactive_value(axis: str, spec: Optional["JointSpaceSpec"]) -> Any:
    """The canonical value of an inactive axis, resolved against the spec."""
    v = INACTIVE_CANONICAL[axis]
    if v is not None:
        return v
    if spec is None:
        raise ValueError(
            "axis %r has a spec-dependent canonical value (its lower bound) "
            "and no spec was given. Pass spec= to canonicalise_config / "
            "config_from_point." % axis)
    return spec.range_of(axis)[0]

_INT_AXES = frozenset((
    "depth_exponent", "block_family", "embedding_size", "head_fusion",
    "dsn_on", "rep_on", "n_posterior_draws", "hidden_features",
    "num_transforms", "batch_size_npe",
))
_STR_AXES = frozenset(("loss_type", "mining_strategy"))


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass
class JointSpaceSpec:
    """Searched ranges, resolved against the problem's shapes."""

    depth_exponent: Tuple[int, int] = (3, 6)
    width_multiplier: Tuple[float, float] = (1.5, 3.0)
    block_family: Tuple[int, ...] = (0, 1)
    embedding_size: Tuple[int, int] = (8, 16)
    head_fusion: Tuple[int, ...] = (0, 1)
    dropout: Tuple[float, float] = (0.0, 0.3)

    log10_lambda_dsn: Tuple[float, float] = (-3.0, 1.0)
    loss_type: Tuple[str, ...] = ("triplet", "joint", "joint_sep")
    mining_strategy: Tuple[str, ...] = ("hard", "easy_positive",
                                        "easy_pos_semihard_neg")
    margin: Tuple[float, float] = (0.1, 1.0)
    angular_alpha_deg: Tuple[float, float] = (2.0, 20.0)
    lambda_sep: Tuple[float, float] = (1e-3, 1.0)

    log10_lambda_rep: Tuple[float, float] = (-3.0, 1.0)
    warmup_frac_rep: Tuple[float, float] = (0.0, 0.5)
    n_posterior_draws: Tuple[int, int] = (100, 400)

    hidden_features: Tuple[int, int] = (64, 256)
    num_transforms: Tuple[int, int] = (4, 12)

    lr: Tuple[float, float] = (1e-4, 2e-3)
    one_minus_beta1: Tuple[float, float] = (1e-2, 1e-1)
    weight_decay: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_npe: Tuple[int, ...] = (256, 512, 1024)

    # Values held fixed by S5.1, carried so a config is self-describing.
    fixed: Dict[str, Any] = field(default_factory=dict)
    anchored_to: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def range_of(self, axis: str) -> Any:
        if axis in ("dsn_on", "rep_on"):
            return (0, 1)
        if not hasattr(self, axis):
            raise KeyError("no range for axis %r" % axis)
        return getattr(self, axis)


def default_joint_space(p: int,
                        embedding_dim: int,
                        d_theta: int,
                        n_train: Optional[int] = None,
                        strict_semihard: int = 1,
                        n_posterior_draws_max: int = 400) -> JointSpaceSpec:
    """Ranges for a problem of this shape.

    Two ranges are resolved rather than constant:

    `hidden_features` follows the NPE stack's own rule -- roughly
    [2, 8] * max(p, E), widened so it always contains [64, 256] -- so a bank
    with more parameters searches wider flows with no edit.

    `n_posterior_draws` has a HARD lower bound of 4 * d_theta. Below it the
    per-culture covariance estimate C_hat_g of eq. (3h) is rank-deficient and
    the Cholesky solve fails outright; the axis trades compute against the
    d_theta / S_mc variance inflation, and the correction is applied
    regardless, so the axis controls variance and never bias. At
    d_theta = 26 the bound is 104.

    `batch_size_npe` is trimmed when the training split is too small for the
    largest option to give a sensible number of steps per epoch: a batch that
    swallows the epoch makes early stopping meaningless.

    `strict_semihard` is FIXED by S5.1, which does not state at what value.
    The default here is 1, matching DSNLossConfig and therefore what Stage 3
    has actually been running; it is exposed as an argument because that is a
    decision rather than a fact. The value still passes through the legality
    projection, which zeroes it under "triplet" and under "hard" mining.
    """
    p = int(p)
    embedding_dim = int(embedding_dim)
    d_theta = int(d_theta)
    if d_theta < 1:
        raise ValueError("d_theta must be positive, got %d" % d_theta)

    scale = max(p, embedding_dim)
    lo = max(32, 2 * scale)
    hi = max(lo * 2, 8 * scale)
    lo, hi = min(lo, 64), max(hi, 256)

    batches = [256, 512, 1024]
    if n_train is not None:
        batches = [b for b in batches if int(n_train) >= 20 * b] or [min(batches)]

    draws_lo = 4 * d_theta
    draws_hi = int(n_posterior_draws_max)
    if draws_hi < draws_lo:
        raise ValueError(
            "n_posterior_draws upper bound %d is below the hard lower bound "
            "4 * d_theta = %d. Below that bound C_hat_g is rank-deficient and "
            "the Cholesky solve fails, so this is not a range to widen "
            "downwards -- raise the upper bound instead."
            % (draws_hi, draws_lo))

    return JointSpaceSpec(
        hidden_features=(int(lo), int(hi)),
        n_posterior_draws=(int(draws_lo), int(draws_hi)),
        batch_size_npe=tuple(int(b) for b in batches),
        fixed={"strict_semihard": int(strict_semihard),
               "head_pool_ops": 1, "sep_warmup_frac": 0.0,
               "num_bins": 10, "one_minus_beta2": 1e-3},
        anchored_to={"p": p, "embedding_dim": embedding_dim,
                     "d_theta": d_theta,
                     "n_train": (int(n_train) if n_train is not None else None),
                     "width_rule": "clamp([2,8] * max(p,E), [64,256])",
                     "draws_rule": "lower bound = 4 * d_theta"},
    )


# ---------------------------------------------------------------------------
# Campaigns (S5.2). D4 is a runtime choice over this table, not a code edit.
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    """One campaign: which axes are free, and what the frozen ones are set to.

    `pinned` fixes an axis at a value for the whole campaign. Every axis of
    JOINT_KNOB_ORDER is either free or pinned; there is no third state, and
    `campaign()` verifies that partition rather than assuming it.
    """

    name: str
    pinned: Dict[str, Any]
    note: str = ""

    def free(self) -> Tuple[str, ...]:
        return tuple(k for k in JOINT_KNOB_ORDER if k not in self.pinned)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "pinned": dict(self.pinned),
                "free": list(self.free()), "note": self.note}


_DSN_AXES_OFF = {
    "dsn_on": 0,
    "log10_lambda_dsn": INACTIVE_CANONICAL["log10_lambda_dsn"],
    "loss_type": INACTIVE_CANONICAL["loss_type"],
    "mining_strategy": INACTIVE_CANONICAL["mining_strategy"],
    "margin": INACTIVE_CANONICAL["margin"],
    "angular_alpha_deg": INACTIVE_CANONICAL["angular_alpha_deg"],
    "lambda_sep": INACTIVE_CANONICAL["lambda_sep"],
}

_REP_AXES_OFF = {
    "rep_on": 0,
    "log10_lambda_rep": INACTIVE_CANONICAL["log10_lambda_rep"],
    "warmup_frac_rep": INACTIVE_CANONICAL["warmup_frac_rep"],
    "n_posterior_draws": INACTIVE_CANONICAL["n_posterior_draws"],
}

CAMPAIGNS: Dict[str, Campaign] = {
    "S-A1": Campaign(
        name="S-A1",
        pinned=dict(**_DSN_AXES_OFF, **_REP_AXES_OFF),
        note="NPE only: dsn_on = rep_on = 0. The nested baseline that A2 "
             "and A5 must beat at the optimum, not on average."),
    "S-A2": Campaign(
        name="S-A2",
        pinned=dict(**_REP_AXES_OFF),
        note="DSN term free (dsn_on searched), replicate term off. Nests "
             "S-A1 at dsn_on = 0."),
    "S-A5": Campaign(
        name="S-A5",
        pinned=dict(**_DSN_AXES_OFF),
        note="Replicate term free (rep_on searched), DSN term off. Nests "
             "S-A1 at rep_on = 0."),
    "S-A25": Campaign(
        name="S-A25",
        pinned={},
        note="Both terms free. Not one of the four campaigns of S5.2; "
             "provided because the union space is the natural object and a "
             "combined campaign may be wanted once the partial dependences "
             "on dsn_on and rep_on are in hand. Budget it separately."),
}


def campaign(name: str) -> Campaign:
    """Look up a campaign, verifying its pinned/free partition covers the space."""
    if name not in CAMPAIGNS:
        raise KeyError("unknown campaign %r; known: %s"
                       % (name, ", ".join(sorted(CAMPAIGNS))))
    c = CAMPAIGNS[name]
    unknown = [k for k in c.pinned if k not in JOINT_KNOB_ORDER]
    if unknown:
        raise RuntimeError("campaign %r pins axes outside JOINT_KNOB_ORDER: %s"
                           % (name, unknown))
    if set(c.free()) | set(c.pinned) != set(JOINT_KNOB_ORDER):
        raise RuntimeError("campaign %r does not partition the space" % name)
    return c


def free_axes(name: str) -> Tuple[str, ...]:
    """The FREE axes of a campaign, in JOINT_KNOB_ORDER."""
    return campaign(name).free()


def resolved_pins(campaign_name: str,
                  spec: Optional["JointSpaceSpec"] = None) -> Dict[str, Any]:
    """A campaign's pinned axes with spec-dependent values resolved and cast.

    `Campaign.pinned` holds the raw table, which may carry None for an axis
    whose canonical value is its own lower bound. This is what a caller
    should compare against when asking "did the pin hold".
    """
    out: Dict[str, Any] = {}
    for axis, value in campaign(campaign_name).pinned.items():
        if value is None:
            value = _inactive_value(axis, spec)
        out[axis] = _cast(axis, value)
    return out


# ---------------------------------------------------------------------------
# skopt dimensions
# ---------------------------------------------------------------------------

def space_dimensions(spec: JointSpaceSpec, campaign_name: str):
    """Build the scikit-optimize dimension list over the campaign's FREE axes.

    The list is in JOINT_KNOB_ORDER restricted to the free axes, which is the
    same order `config_from_point` reads a point in. skopt is imported lazily
    so this module stays importable (and testable) without it.
    """
    from skopt.space import Categorical, Integer, Real

    c = campaign(campaign_name)
    dims = []
    for axis in c.free():
        r = spec.range_of(axis)
        if axis in ("block_family", "head_fusion", "dsn_on", "rep_on"):
            dims.append(Categorical([int(v) for v in r], name=axis))
        elif axis == "batch_size_npe":
            dims.append(Categorical([int(v) for v in r], name=axis))
        elif axis in _STR_AXES:
            dims.append(Categorical(list(r), name=axis))
        elif axis in ("hidden_features",):
            dims.append(Integer(int(r[0]), int(r[1]), prior="log-uniform",
                                name=axis))
        elif axis in _INT_AXES:
            dims.append(Integer(int(r[0]), int(r[1]), name=axis))
        elif axis in ("lr", "lambda_sep", "one_minus_beta1", "weight_decay"):
            # Searched in LOG space: these span decades and a uniform prior
            # would spend almost every draw in the top decade.
            dims.append(Real(float(r[0]), float(r[1]), prior="log-uniform",
                             name=axis))
        else:
            dims.append(Real(float(r[0]), float(r[1]), name=axis))
    return dims


# ---------------------------------------------------------------------------
# Point <-> config, with canonicalisation
# ---------------------------------------------------------------------------

def _cast(axis: str, value: Any) -> Any:
    """Cast to a plain Python type. Not cosmetic: skopt returns numpy scalars,
    which json.dump refuses and which would otherwise reach the ledger and
    make two identical configs compare unequal."""
    if axis in _STR_AXES:
        return str(value)
    if axis in _INT_AXES:
        return int(value)
    return float(value)


def _active_loss_hps(loss_type: str) -> Tuple[str, ...]:
    """A(l) from the DSN's condition_space, imported not duplicated.

    The DSN repo is reached through DSN_MAIN_DIR. Duplicating the table here
    would let the two drift apart silently, which is exactly the failure this
    module exists to prevent, so an unavailable DSN repo is an error and not
    a fallback.
    """
    main = os.environ.get("DSN_MAIN_DIR", "")
    if main and main not in sys.path:
        sys.path.insert(0, main)
    try:
        import condition_space as CS
    except ImportError as exc:
        raise ImportError(
            "joint_space needs the DSN's condition_space for the legality "
            "projection and the active-loss-hyperparameter mask A(l). Set "
            "DSN_MAIN_DIR to the DSN repo's Main/ directory. Duplicating the "
            "table here is not an option: the two copies would drift and the "
            "canonicalisation would stop matching the trainer. (%s)" % exc)
    return tuple(CS.active_loss_hps(str(loss_type)))


def _project_condition(mining_strategy: str, loss_type: str,
                       strict_semihard: int) -> Tuple[str, str, bool]:
    main = os.environ.get("DSN_MAIN_DIR", "")
    if main and main not in sys.path:
        sys.path.insert(0, main)
    import condition_space as CS
    return CS.project_condition(mining_strategy, loss_type,
                                bool(strict_semihard))


def canonicalise_config(config: Dict[str, Any],
                        spec: Optional[JointSpaceSpec] = None) -> Dict[str, Any]:
    """Pin every coordinate the configuration does not read, and project.

    Three clamps, applied in this order:

      (a) dsn_on == 0  -> every DSN-loss axis takes its canonical value.
      (b) rep_on == 0  -> every replicate axis takes its canonical value.
      (c) dsn_on == 1  -> the loss hyper-parameters NOT in A(loss_type) take
          their canonical values, and (mining_strategy, loss_type,
          strict_semihard) is replaced by its legality projection.

    Clause (c) is where a naive implementation goes wrong, and the trap is
    the canonical VALUE rather than the mask. "Inactive" is not "unused":
    under "joint" and "joint_sep" the margin is still READ -- the DSN's
    JointTripletLoss takes margin = 2 * m_cos -- it is simply FIXED at the
    base config's value instead of searched. So an inactive axis must be
    pinned to the DSN's TrainConfig default (margin 0.3, angular_alpha_deg
    18.0, lambda_sep 0.1), never to zero: pinning to zero would neutralise
    the coordinate for deduplication while silently changing the experiment.
    The base config DEFINES these clamp constants; INACTIVE_CANONICAL copies
    them, and if the DSN's defaults move, these must move with them.

    A(l) itself is imported from condition_space rather than reproduced, so
    the mask cannot drift; the values above are the part that must be kept
    in sync by hand, which is why they are named with their source.

    Idempotent by construction: canonicalise(canonicalise(x)) == canonicalise(x).
    Returns a NEW dict; the input is not modified.
    """
    cfg = copy.deepcopy(dict(config))
    missing = [k for k in JOINT_KNOB_ORDER if k not in cfg]
    if missing:
        raise KeyError("config is missing axes: %s" % ", ".join(missing))

    for axis in JOINT_KNOB_ORDER:
        cfg[axis] = _cast(axis, cfg[axis])

    if int(cfg["dsn_on"]) == 0:
        for axis in _DSN_AXES_OFF:
            cfg[axis] = _cast(axis, 0 if axis == "dsn_on"
                              else _inactive_value(axis, spec))
    else:
        strict = int((spec.fixed if spec is not None else {}).get(
            "strict_semihard", 0))
        m, l, s = _project_condition(cfg["mining_strategy"],
                                     cfg["loss_type"], strict)
        cfg["mining_strategy"], cfg["loss_type"] = str(m), str(l)
        cfg["_strict_semihard_projected"] = int(s)
        active = _active_loss_hps(l)
        for axis in ("margin", "angular_alpha_deg", "lambda_sep"):
            if axis not in active:
                cfg[axis] = _cast(axis, INACTIVE_CANONICAL[axis])

    if int(cfg["rep_on"]) == 0:
        for axis in _REP_AXES_OFF:
            cfg[axis] = _cast(axis, 0 if axis == "rep_on"
                              else _inactive_value(axis, spec))

    return cfg


def config_from_point(point: Sequence[Any], campaign_name: str,
                      spec: Optional[JointSpaceSpec] = None) -> Dict[str, Any]:
    """Turn a skopt point over a campaign's free axes into a full config.

    The pinned axes are filled from the campaign, then the whole thing is
    canonicalised, so two points differing only in inactive coordinates
    return byte-identical dicts.
    """
    c = campaign(campaign_name)
    free = c.free()
    if len(point) != len(free):
        raise ValueError("point has %d entries, expected %d for campaign %r "
                         "(free axes: %s)"
                         % (len(point), len(free), campaign_name,
                            ", ".join(free)))
    cfg: Dict[str, Any] = {}
    for axis, value in zip(free, point):
        cfg[axis] = _cast(axis, value)
    for axis, value in resolved_pins(campaign_name, spec).items():
        cfg[axis] = value
    cfg["_campaign"] = c.name
    return canonicalise_config(cfg, spec=spec)


def point_from_config(config: Dict[str, Any],
                      campaign_name: str) -> List[Any]:
    """Inverse of `config_from_point`, for replaying a ledger.

    Only the FREE axes are returned, in JOINT_KNOB_ORDER. Note the round trip
    is not the identity on inactive coordinates -- it cannot be, since
    canonicalisation deliberately destroys them. It IS the identity on
    configs that are already canonical, which is the property the ledger
    needs, and that is what the smoke test asserts.
    """
    c = campaign(campaign_name)
    out: List[Any] = []
    for axis in c.free():
        if axis not in config:
            raise KeyError("config is missing free axis %r of campaign %r"
                           % (axis, campaign_name))
        out.append(_cast(axis, config[axis]))
    return out


def config_key(config: Dict[str, Any]) -> str:
    """A stable string key over the searched axes and nothing else.

    Keys on JOINT_KNOB_ORDER only, so bookkeeping fields (`_campaign`, a
    trial id, a timestamp) never make two identical recipes look different.
    """
    payload = {k: config[k] for k in JOINT_KNOB_ORDER if k in config}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def configs_identical(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return config_key(a) == config_key(b)


def diff_configs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """Every axis on which two configs differ, as {axis: (a_value, b_value)}.

    Compares the union of all keys, not only JOINT_KNOB_ORDER, so a stray
    training-recipe field is visible rather than silently equal.
    """
    keys = sorted(set(a) | set(b))
    return {k: (a.get(k, "<missing>"), b.get(k, "<missing>"))
            for k in keys if a.get(k, "<missing>") != b.get(k, "<missing>")}


# ---------------------------------------------------------------------------
# The control recipe (HANDOFF_DELTA_MIN_PER_CONFIG_v1 S4.3)
# ---------------------------------------------------------------------------

SHUFFLE_FIELDS = ("shuffle_pairs", "shuffle_seed")


def apply_control_shuffle(theta: np.ndarray, z: np.ndarray,
                          seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Permute the (theta, z) PAIRING, keeping both marginals exactly.

    Returns (theta_permuted, z) with `z` untouched: permuting one side is
    what destroys the association, and permuting both would leave the
    pairing intact for a permutation applied to both. Both marginals are
    preserved exactly because the operation is a permutation of whole rows,
    not a resampling -- so the control's achievable optimum IS the prior, and
    its measured gain is the "learned nothing" floor for this recipe.

    Draws a permutation of {0, ..., N-1} and rejects the identity when N > 1,
    since an identity permutation would leave the pairing intact and the
    control would silently measure the candidate instead of its floor. The
    probability of drawing it is 1/N! and therefore negligible, but "so
    unlikely it will never happen" is the class of assumption that produces
    an unreproducible result once.

    One permutation per control seed: the seed is the only thing that varies
    across the n_ctrl control runs of a finalist.
    """
    theta = np.asarray(theta)
    z = np.asarray(z)
    if theta.shape[0] != z.shape[0]:
        raise ValueError("theta has %d rows, z has %d: not a paired bank"
                         % (theta.shape[0], z.shape[0]))
    n = int(theta.shape[0])
    if n < 2:
        raise ValueError("need at least 2 rows to permute a pairing, got %d" % n)
    rng = np.random.default_rng(int(seed))
    ident = np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if not np.array_equal(perm, ident):
            return theta[perm], z
    raise RuntimeError("drew the identity permutation 100 times; the RNG is "
                       "not behaving")


def control_config_from(config: Dict[str, Any],
                        permutation_seed: int) -> Dict[str, Any]:
    """Build a finalist's control config: the SAME recipe, shuffled pairing.

    A per-configuration floor is only meaningful if the control differs from
    the candidate in exactly one respect. Same architecture, same optimiser,
    same schedule, same batch size, same epoch ceiling, same early-stopping
    rule, same split hash, same seeds policy -- only the (theta, z) pairing
    is permuted. If the candidate early-stops on validation loss and the
    control does not, the two are not comparable and the floor is
    meaningless.

    So the control is constructed by COPYING the candidate and setting only
    the two shuffle fields, rather than by rebuilding a config from scratch:
    a rebuilt config can silently pick up a changed default.

    Expect early stopping to fire almost immediately on shuffled data. That
    is correct behaviour and not a bug -- it is what the recipe does when
    there is no signal, which is exactly the quantity being measured. Record
    epochs-to-stop for every control run so this is visible rather than
    inferred.
    """
    ctrl = copy.deepcopy(dict(config))
    ctrl["shuffle_pairs"] = 1
    ctrl["shuffle_seed"] = int(permutation_seed)
    return ctrl


def assert_control_identity(candidate: Dict[str, Any],
                            control: Dict[str, Any]) -> None:
    """Raise unless `control` differs from `candidate` ONLY in the shuffle.

    This is the enforcement S4.3 asks for. It compares the union of keys, so
    an extra field on either side is a difference; and it requires the
    control to actually BE a control (`shuffle_pairs == 1`) and the candidate
    not to be (`shuffle_pairs` absent or 0), because a control compared
    against another control measures nothing.
    """
    if int(control.get("shuffle_pairs", 0)) != 1:
        raise ValueError("control config does not have shuffle_pairs = 1; it "
                         "is not a control")
    if int(candidate.get("shuffle_pairs", 0)) != 0:
        raise ValueError("candidate config has shuffle_pairs = 1; comparing a "
                         "control against a control measures nothing")
    offending = {k: v for k, v in diff_configs(candidate, control).items()
                 if k not in SHUFFLE_FIELDS}
    if offending:
        bits = ", ".join("%s: %r vs %r" % (k, a, b)
                         for k, (a, b) in sorted(offending.items()))
        raise ValueError(
            "control config differs from its candidate in %d field(s) other "
            "than the shuffle: %s. The per-finalist floor is only meaningful "
            "when the two differ in exactly one respect."
            % (len(offending), bits))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The optimiser adapter: reuse npe_tune_search's mechanics, do not copy them
# ---------------------------------------------------------------------------

# Axes with no meaningful "edge". A Categorical dimension has no boundary to
# widen: the best configuration choosing "joint_sep" is not evidence that the
# range was too narrow, it is the answer. Switches likewise.
_NO_BOUNDARY = frozenset(("block_family", "head_fusion", "dsn_on", "rep_on",
                          "loss_type", "mining_strategy", "batch_size_npe"))


def boundary_axes(config: Dict[str, Any],
                  spec: JointSpaceSpec,
                  campaign_name: Optional[str] = None,
                  rel_tol: float = 1e-6) -> List[str]:
    """Which FREE numeric axes of the best configuration sit on their edge.

    A configuration on a boundary says the RANGE is binding, not the budget,
    so the response is to widen the range rather than buy more evaluations.
    Only free axes are checked: a pinned axis is on its "edge" by
    construction and reporting it would fire the escalation rule on every
    campaign that pins anything.

    Inactive axes are skipped for the same reason -- under dsn_on = 0 the
    canonicalisation has already pinned `margin` to 0.3, which is inside its
    range but is not a search result.
    """
    c = campaign(campaign_name) if campaign_name else None
    free = set(c.free()) if c is not None else set(JOINT_KNOB_ORDER)
    inactive: set = set()
    if int(config.get("dsn_on", 1)) == 0:
        inactive |= set(_DSN_AXES_OFF) - {"dsn_on"}
    if int(config.get("rep_on", 1)) == 0:
        inactive |= set(_REP_AXES_OFF) - {"rep_on"}

    on_edge: List[str] = []
    for axis in JOINT_KNOB_ORDER:
        if axis not in free or axis in _NO_BOUNDARY or axis in inactive:
            continue
        if axis not in config:
            continue
        try:
            lo, hi = spec.range_of(axis)
        except (KeyError, TypeError, ValueError):
            continue
        v = config[axis]
        if axis in _INT_AXES:
            if int(v) <= int(lo) or int(v) >= int(hi):
                on_edge.append(axis)
        else:
            lo, hi = float(lo), float(hi)
            # Relative tolerance on the wider side, absolute near zero: a
            # range starting at 0.0 (dropout, warmup_frac_rep) has no
            # relative tolerance to speak of.
            eps = rel_tol * max(1.0, abs(lo), abs(hi))
            if float(v) <= lo + eps or float(v) >= hi - eps:
                on_edge.append(axis)
    return on_edge


def adapter(campaign_name: str, spec: JointSpaceSpec):
    """A `npe_tune_search.SpaceAdapter` bound to one campaign.

    This is how Stage 4 gets the stateless-Optimizer-from-the-ledger,
    constant-liar batching and three-condition escalation of
    `npe_tune_search` without a second copy of any of it. The campaign is
    baked in because a point's LENGTH depends on it: an S-A1 point has 12
    entries and an S-A2 point 19, so an adapter that did not know which
    campaign it served could not read a point at all.
    """
    import npe_tune_search as TSR

    c = campaign(campaign_name)

    def _dims(sp):
        return space_dimensions(sp, c.name)

    def _cfp(point):
        return config_from_point(point, c.name, spec=spec)

    def _pfc(cfg):
        return point_from_config(cfg, c.name)

    def _key(cfg):
        return config_key(cfg)

    def _edges(cfg, sp):
        return boundary_axes(cfg, sp, campaign_name=c.name)

    return TSR.SpaceAdapter(name="joint:%s" % c.name, dimensions=_dims,
                            config_from_point=_cfp, point_from_config=_pfc,
                            config_key=_key, boundary_axes=_edges)


def provenance_report(spec: JointSpaceSpec,
                      campaign_name: Optional[str] = None) -> str:
    """The space, its ranges and where each came from. ASCII only."""
    lines = ["joint search space: %d axes over %d blocks"
             % (len(JOINT_KNOB_ORDER), len(BLOCKS))]
    pinned: Dict[str, Any] = {}
    if campaign_name is not None:
        c = campaign(campaign_name)
        pinned = resolved_pins(campaign_name, spec)
        lines.append("campaign %s: %d free, %d pinned -- %s"
                     % (c.name, len(c.free()), len(c.pinned), c.note))
    lines.append("")
    lines.append("| block | axis | range | status | provenance |")
    lines.append("|---|---|---|---|---|")
    for block, axes in BLOCKS.items():
        for axis in axes:
            try:
                r = spec.range_of(axis)
                rs = ("[%s]" % ", ".join(str(v) for v in r)) if r else "-"
            except KeyError:
                rs = "-"
            status = ("pinned at %r" % (pinned[axis],)) if axis in pinned \
                else "free"
            lines.append("| %s | %s | %s | %s | %s |"
                         % (block, axis, rs, status,
                            RANGE_PROVENANCE.get(axis, "?")))
    lines.append("")
    lines.append("fixed by S5.1 (absent from the space): %s"
                 % ", ".join("%s=%r" % (k, v)
                             for k, v in sorted(spec.fixed.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default=None,
                    help="one of: %s" % ", ".join(sorted(CAMPAIGNS)))
    ap.add_argument("--p", type=int, default=26)
    ap.add_argument("--embedding-dim", type=int, default=12)
    ap.add_argument("--d-theta", type=int, default=26)
    ap.add_argument("--n-train", type=int, default=None)
    args = ap.parse_args()
    sp = default_joint_space(p=args.p, embedding_dim=args.embedding_dim,
                             d_theta=args.d_theta, n_train=args.n_train)
    print(provenance_report(sp, args.campaign))
