#!/usr/bin/env python3
"""
smoke_test_joint_space.py -- the Stage 4 joint search space (plan S5.1/S5.2)
and the control-recipe identity of HANDOFF_DELTA_MIN_PER_CONFIG_v1 S4.3.

Run:
    python smoke_test_joint_space.py                  # all
    python smoke_test_joint_space.py -k J23           # one
    DSN_MAIN_DIR=/path/to/DSN/Main python smoke_test_joint_space.py

J-numbering continues the joint stack's: J1-J18 are taken, J19 is the
per-finalist p-value calibration (implemented as S20/S21 in
hpc/smoke_test_tune.py, where the statistic lives), J20 is the recipe
identity test reserved by that handoff, and J21-J25 are new here.

  J20  RECIPE IDENTITY. A control config built through the production path
       differs from its candidate in exactly the two shuffle fields and
       nothing else; a tampered control is caught; and the shuffle destroys
       the pairing while preserving both marginals exactly.
  J21  Point <-> config round trip is the identity on canonical configs,
       for every campaign.
  J22  Every campaign partitions the space; pinned axes are really pinned;
       S-A2 and S-A5 nest S-A1.
  J23  THE LOAD-BEARING ONE. Points differing only in INACTIVE coordinates
       build byte-identical configs -- including the case a naive mask gets
       wrong, where `margin` is still read under a joint loss.
  J24  Guards: the 4*d_theta floor on n_posterior_draws, wrong point length,
       unknown campaign, missing axes, a control compared to a control.
  J25  [needs skopt] The dimension list matches the campaign's free axes in
       name, order and type.
  J26  [needs skopt + SBI_HPC_DIR] The GP proposes joint configurations
       through npe_tune_search's SpaceAdapter: distinct, canonical, pins
       respected, warm start reproducible, and the boundary rule sees free
       axes only.
  J27  Every canonical value lies inside its axis's range, so a
       canonicalised config can be replayed into the optimiser.

Tests that need the DSN repo (through DSN_MAIN_DIR) SKIP without it rather
than fail, but the parts that do not touch the DSN loss still run.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# stage4 sits at <hpc>/joint/stage4, so the tuning stack is two levels up.
# Resolving it here rather than relying on SBI_HPC_DIR being exported keeps
# J26 from SKIPPING silently -- it is the test that proves the GP can
# propose joint configurations at all, and a skip reads like a pass in the
# summary line.
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "..")),
           os.environ.get("SBI_HPC_DIR", "")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import joint_space as JS  # noqa: E402

RESULTS: List[Tuple[str, str, str]] = []


class Skip(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _dsn_available() -> bool:
    main = os.environ.get("DSN_MAIN_DIR", "")
    if not main:
        return False
    if main not in sys.path:
        sys.path.insert(0, main)
    try:
        import condition_space  # noqa: F401
        return True
    except ImportError:
        return False


def _spec():
    return JS.default_joint_space(p=26, embedding_dim=12, d_theta=26,
                                  n_train=20000)


def _point(campaign_name: str, spec, rng, **overrides):
    """A random legal point over a campaign's free axes, in JOINT_KNOB_ORDER."""
    pt = []
    for axis in JS.free_axes(campaign_name):
        if axis in overrides:
            pt.append(overrides[axis])
            continue
        r = spec.range_of(axis)
        if axis in JS._STR_AXES:
            pt.append(str(rng.choice(list(r))))
        elif axis in ("block_family", "head_fusion", "dsn_on", "rep_on",
                      "batch_size_npe"):
            pt.append(int(rng.choice(list(r))))
        elif axis in JS._INT_AXES:
            pt.append(int(rng.integers(int(r[0]), int(r[1]) + 1)))
        else:
            pt.append(float(rng.uniform(float(r[0]), float(r[1]))))
    return pt


# ---------------------------------------------------------------------------

def j20_recipe_identity() -> str:
    """J20: the control differs in the shuffle and in nothing else."""
    spec = _spec()
    rng = np.random.default_rng(0)
    campaigns = ["S-A1", "S-A5"] + (["S-A2", "S-A25"] if _dsn_available() else [])

    for name in campaigns:
        cfg = JS.config_from_point(_point(name, spec, rng), name, spec=spec)
        cfg["shuffle_pairs"] = 0
        # Fields a trainer would attach: the identity check must cover them,
        # not just the searched axes.
        cfg.update({"max_epochs": 200, "patience": 20, "split_hash": "abc123",
                    "early_stop_on": "val_nll", "seed_policy": "per_member"})
        ctrl = JS.control_config_from(cfg, permutation_seed=7)
        JS.assert_control_identity(cfg, ctrl)          # must not raise
        d = JS.diff_configs(cfg, ctrl)
        check(set(d) == {"shuffle_pairs", "shuffle_seed"},
              "%s: control differs in %s" % (name, sorted(d)))

        # Every way a control can silently stop being a control.
        for field, value in (("max_epochs", 5), ("patience", 3),
                             ("early_stop_on", "none"), ("lr", 0.5),
                             ("split_hash", "def456"), ("hidden_features", 9)):
            bad = JS.control_config_from(cfg, permutation_seed=7)
            bad[field] = value
            try:
                JS.assert_control_identity(cfg, bad)
                raise AssertionError("%s: a control differing in %r must be "
                                     "refused" % (name, field))
            except ValueError as exc:
                check(field in str(exc), "the error must name the field: %s" % exc)
        # An extra field on either side is a difference, not a tie.
        bad = JS.control_config_from(cfg, permutation_seed=7)
        bad["extra_knob"] = 1
        try:
            JS.assert_control_identity(cfg, bad)
            raise AssertionError("an extra field must be refused")
        except ValueError:
            pass

    # A control compared against another control measures nothing.
    cfg = JS.config_from_point(_point("S-A1", spec, rng), "S-A1", spec=spec)
    cfg["shuffle_pairs"] = 0
    c1 = JS.control_config_from(cfg, 1)
    c2 = JS.control_config_from(cfg, 2)
    for a, b in ((c1, c2), (cfg, cfg)):
        try:
            JS.assert_control_identity(a, b)
            raise AssertionError("must refuse control-vs-control and "
                                 "candidate-vs-candidate")
        except ValueError:
            pass

    # The shuffle: pairing destroyed, both marginals preserved EXACTLY.
    rng2 = np.random.default_rng(3)
    n, p, E = 400, 26, 12
    theta = rng2.normal(size=(n, p))
    z = rng2.normal(size=(n, E))
    th_s, z_s = JS.apply_control_shuffle(theta, z, seed=11)
    check(np.array_equal(np.sort(theta, axis=0), np.sort(th_s, axis=0)),
          "the theta marginal must be preserved exactly")
    check(np.array_equal(z, z_s), "z must be untouched")
    matched = int(np.sum(np.all(np.isclose(theta, th_s), axis=1)))
    check(matched < n // 10, "the pairing must be destroyed, %d/%d rows still "
                             "aligned" % (matched, n))
    a1, _ = JS.apply_control_shuffle(theta, z, seed=11)
    a2, _ = JS.apply_control_shuffle(theta, z, seed=12)
    check(np.array_equal(th_s, a1), "same seed must give the same permutation")
    check(not np.array_equal(th_s, a2), "a different seed must differ")
    for bad in ((theta[:1], z[:1]), (theta, z[:10])):
        try:
            JS.apply_control_shuffle(bad[0], bad[1], seed=0)
            raise AssertionError("degenerate shuffle input must be refused")
        except ValueError:
            pass
    return ("%d campaign(s); control differs only in the shuffle; marginals "
            "exact, %d/%d rows still aligned" % (len(campaigns), matched, n))


def j21_point_config_round_trip() -> str:
    spec = _spec()
    rng = np.random.default_rng(1)
    names = ["S-A1", "S-A5"] + (["S-A2", "S-A25"] if _dsn_available() else [])
    counts = []
    for name in names:
        for _ in range(25):
            pt = _point(name, spec, rng)
            cfg = JS.config_from_point(pt, name, spec=spec)
            back = JS.point_from_config(cfg, name)
            again = JS.config_from_point(back, name, spec=spec)
            check(JS.configs_identical(cfg, again),
                  "%s: round trip is not the identity on a canonical config: "
                  "%s" % (name, JS.diff_configs(cfg, again)))
            check(JS.config_key(cfg) == JS.config_key(again),
                  "%s: config_key must agree too" % name)
        counts.append("%s(%d free)" % (name, len(JS.free_axes(name))))
    # Canonicalisation is idempotent.
    cfg = JS.config_from_point(_point(names[0], spec, rng), names[0], spec=spec)
    once = JS.canonicalise_config(cfg, spec=spec)
    twice = JS.canonicalise_config(once, spec=spec)
    check(JS.config_key(once) == JS.config_key(twice),
          "canonicalise must be idempotent")
    return "; ".join(counts)


def j22_campaign_partition() -> str:
    out = []
    for name in sorted(JS.CAMPAIGNS):
        c = JS.campaign(name)
        check(set(c.free()) | set(c.pinned) == set(JS.JOINT_KNOB_ORDER),
              "%s does not partition the space" % name)
        check(not (set(c.free()) & set(c.pinned)),
              "%s has an axis both free and pinned" % name)
        check(list(c.free()) == [k for k in JS.JOINT_KNOB_ORDER
                                 if k not in c.pinned],
              "%s free axes are out of JOINT_KNOB_ORDER" % name)
        out.append("%s %d/%d free" % (name, len(c.free()),
                                      len(JS.JOINT_KNOB_ORDER)))
    check(len(set(JS.JOINT_KNOB_ORDER)) == len(JS.JOINT_KNOB_ORDER),
          "JOINT_KNOB_ORDER has a duplicate")
    flat = [a for axes in JS.BLOCKS.values() for a in axes]
    check(flat == list(JS.JOINT_KNOB_ORDER),
          "BLOCKS must flatten to JOINT_KNOB_ORDER exactly, got %s" % flat)

    # S-A1 is nested inside S-A2 and S-A5: a point of the larger campaign
    # with its switch at 0 must build the same config as the S-A1 point.
    spec = _spec()
    rng = np.random.default_rng(2)
    pairs = [("S-A5", "rep_on")]
    if _dsn_available():
        pairs.append(("S-A2", "dsn_on"))
    for big, switch in pairs:
        pt = _point(big, spec, rng, **{switch: 0})
        cfg_big = JS.config_from_point(pt, big, spec=spec)
        shared = {k: cfg_big[k] for k in JS.free_axes("S-A1")}
        cfg_small = JS.config_from_point(
            [shared[k] for k in JS.free_axes("S-A1")], "S-A1", spec=spec)
        d = {k: v for k, v in JS.diff_configs(cfg_big, cfg_small).items()
             if not k.startswith("_")}
        check(not d, "%s at %s=0 must equal an S-A1 config, differs in %s"
                     % (big, switch, d))
        out.append("%s nests S-A1 at %s=0" % (big, switch))
    return "; ".join(out)


def j23_inactive_coordinates_are_canonical() -> str:
    """J23: points differing ONLY in inactive coordinates build identical
    configs. This is what makes ledger deduplication and the control-recipe
    comparison mean anything."""
    spec = _spec()
    rng = np.random.default_rng(4)

    # (a) rep_on = 0: the three replicate axes must not survive.
    base = _point("S-A25" if _dsn_available() else "S-A5", spec, rng,
                  rep_on=0, **({"dsn_on": 0} if _dsn_available() else {}))
    name = "S-A25" if _dsn_available() else "S-A5"
    free = list(JS.free_axes(name))
    cfg1 = JS.config_from_point(base, name, spec=spec)
    alt = list(base)
    for axis, value in (("log10_lambda_rep", 0.87), ("warmup_frac_rep", 0.42),
                        ("n_posterior_draws", 377)):
        if axis in free:
            alt[free.index(axis)] = value
    cfg2 = JS.config_from_point(alt, name, spec=spec)
    check(JS.configs_identical(cfg1, cfg2),
          "rep_on=0: inactive replicate axes leaked: %s"
          % JS.diff_configs(cfg1, cfg2))
    detail = ["rep_on=0 canonical"]

    if not _dsn_available():
        raise Skip("DSN_MAIN_DIR not set; the dsn_on clauses need "
                   "condition_space")

    # (b) dsn_on = 0: every DSN-loss axis must not survive.
    base = _point("S-A25", spec, rng, dsn_on=0)
    free = list(JS.free_axes("S-A25"))
    cfg1 = JS.config_from_point(base, "S-A25", spec=spec)
    alt = list(base)
    for axis, value in (("log10_lambda_dsn", 0.9), ("loss_type", "joint_sep"),
                        ("mining_strategy", "easy_positive"),
                        ("margin", 0.77), ("angular_alpha_deg", 19.5),
                        ("lambda_sep", 0.42)):
        alt[free.index(axis)] = value
    cfg2 = JS.config_from_point(alt, "S-A25", spec=spec)
    check(JS.configs_identical(cfg1, cfg2),
          "dsn_on=0: inactive DSN axes leaked: %s" % JS.diff_configs(cfg1, cfg2))
    detail.append("dsn_on=0 canonical")

    # (c) dsn_on = 1 under "joint": margin and lambda_sep are both INACTIVE
    # by A(l), so neither may survive -- but the VALUE they are pinned to is
    # the trap. "Inactive" is not "unused": the DSN still reads the margin
    # under "joint" as 2 * m_cos, fixed at the base config's value. An
    # implementation that zeroed inactive axes would pass the identity check
    # while silently training a different loss.
    base = _point("S-A25", spec, rng, dsn_on=1, loss_type="joint",
                  margin=0.55, lambda_sep=0.02)
    cfg1 = JS.config_from_point(base, "S-A25", spec=spec)
    for axis, value in (("lambda_sep", 0.99), ("margin", 0.95)):
        alt = list(base)
        alt[free.index(axis)] = value
        cfg2 = JS.config_from_point(alt, "S-A25", spec=spec)
        check(JS.configs_identical(cfg1, cfg2),
              "loss_type=joint: %s is inactive by A(l) and must not survive: "
              "%s" % (axis, JS.diff_configs(cfg1, cfg2)))
    check(abs(float(cfg1["margin"]) - 0.2) < 1e-12,
          "inactive margin must be pinned to DSNLossConfig's default 0.2 -- "
          "the base config that BUILDS the loss, not the DSN's TrainConfig -- "
          "and not to 0, since it is still READ as 2 * m_cos. Got %r"
          % cfg1["margin"])
    check(abs(float(cfg1["lambda_sep"]) - 0.1) < 1e-12,
          "inactive lambda_sep must be pinned to the TrainConfig default 0.1 "
          "or every non-joint_sep trial fires the INERT warning. Got %r"
          % cfg1["lambda_sep"])
    check(abs(float(cfg1["angular_alpha_deg"]) - float(base[free.index(
        "angular_alpha_deg")])) < 1e-12,
        "angular_alpha_deg IS active under joint and must survive")
    detail.append("loss=joint: margin/lambda_sep inert at base defaults "
                  "0.2/0.1, alpha active")

    # (d) under "joint_sep" both angular_alpha_deg and lambda_sep are active.
    base = _point("S-A25", spec, rng, dsn_on=1, loss_type="joint_sep")
    cfg1 = JS.config_from_point(base, "S-A25", spec=spec)
    alt = list(base)
    alt[free.index("lambda_sep")] = float(cfg1["lambda_sep"]) * 0.5 + 1e-3
    cfg2 = JS.config_from_point(alt, "S-A25", spec=spec)
    check(not JS.configs_identical(cfg1, cfg2),
          "loss_type=joint_sep: lambda_sep is active and must survive")
    detail.append("loss=joint_sep: lambda_sep active")

    # (e) the legality projection ran: strict_semihard is recorded projected.
    check("_strict_semihard_projected" in cfg1,
          "the legality projection must be applied and recorded when dsn_on=1")
    return "; ".join(detail)


def j24_guards() -> str:
    hits = []

    def expect(exc_types, fn, what):
        try:
            fn()
        except exc_types:
            hits.append(what)
            return
        raise AssertionError("%s did not raise" % what)

    # The 4*d_theta floor is not a range to widen downwards.
    expect(ValueError,
           lambda: JS.default_joint_space(p=26, embedding_dim=12, d_theta=26,
                                          n_posterior_draws_max=50),
           "n_posterior_draws upper bound below 4*d_theta")
    expect(ValueError,
           lambda: JS.default_joint_space(p=26, embedding_dim=12, d_theta=0),
           "d_theta = 0")
    sp = JS.default_joint_space(p=26, embedding_dim=12, d_theta=26)
    check(sp.n_posterior_draws[0] == 104,
          "the floor must be 4*d_theta = 104, got %d" % sp.n_posterior_draws[0])
    sp2 = JS.default_joint_space(p=26, embedding_dim=12, d_theta=40,
                                 n_posterior_draws_max=400)
    check(sp2.n_posterior_draws[0] == 160, "the floor must track d_theta")

    spec = _spec()
    rng = np.random.default_rng(5)
    pt = _point("S-A1", spec, rng)
    expect(ValueError, lambda: JS.config_from_point(pt[:-1], "S-A1", spec=spec),
           "short point")
    expect(ValueError, lambda: JS.config_from_point(pt + [1], "S-A1", spec=spec),
           "long point")
    expect(KeyError, lambda: JS.campaign("S-A9"), "unknown campaign")
    expect(KeyError, lambda: JS.free_axes("nope"), "unknown campaign in free_axes")
    expect(KeyError, lambda: JS.canonicalise_config({"dsn_on": 0}),
           "config missing axes")
    cfg = JS.config_from_point(pt, "S-A1", spec=spec)
    expect(KeyError, lambda: JS.point_from_config({}, "S-A1"),
           "config missing a free axis")
    expect(KeyError, lambda: spec.range_of("no_such_axis"), "unknown axis range")
    # config_key ignores bookkeeping fields but diff_configs does not.
    a = dict(cfg); b = dict(cfg); b["_trial_id"] = 99
    check(JS.configs_identical(a, b),
          "config_key must ignore bookkeeping fields")
    check("_trial_id" in JS.diff_configs(a, b),
          "diff_configs must NOT ignore them")
    return "%d guards fire; floor 104 at d_theta=26, 160 at 40" % len(hits)


def j25_skopt_dimensions() -> str:
    try:
        import skopt  # noqa: F401
    except ImportError as exc:
        raise Skip("skopt not installed (%s)" % exc)
    from skopt.space import Categorical, Integer, Real

    spec = _spec()
    out = []
    for name in sorted(JS.CAMPAIGNS):
        dims = JS.space_dimensions(spec, name)
        free = JS.free_axes(name)
        check(len(dims) == len(free),
              "%s: %d dims for %d free axes" % (name, len(dims), len(free)))
        check([d.name for d in dims] == list(free),
              "%s: dimension names must match the free axes IN ORDER" % name)
        for d, axis in zip(dims, free):
            if axis in JS._STR_AXES or axis in ("block_family", "head_fusion",
                                                "dsn_on", "rep_on",
                                                "batch_size_npe"):
                check(isinstance(d, Categorical),
                      "%s must be Categorical, got %s" % (axis, type(d)))
            elif axis in JS._INT_AXES:
                check(isinstance(d, Integer),
                      "%s must be Integer, got %s" % (axis, type(d)))
            else:
                check(isinstance(d, Real),
                      "%s must be Real, got %s" % (axis, type(d)))
        # A sampled point must build a config without complaint.
        from skopt.space import Space
        pt = Space(dims).rvs(n_samples=1, random_state=0)[0]
        cfg = JS.config_from_point(pt, name, spec=spec)
        check(all(k in cfg for k in JS.JOINT_KNOB_ORDER),
              "%s: a sampled point must build a complete config" % name)
        out.append("%s %d dims" % (name, len(dims)))
    return "; ".join(out)


# ---------------------------------------------------------------------------

def j26_optimiser_adapter() -> str:
    """J26: the GP proposes JOINT configurations through the shared
    mechanics of npe_tune_search, with no second copy of the optimiser."""
    try:
        import skopt  # noqa: F401
    except ImportError as exc:
        raise Skip("skopt not installed (%s)" % exc)
    try:
        import npe_tune_search as TSR
    except ImportError as exc:
        raise Skip("npe_tune_search not importable from %s or SBI_HPC_DIR (%s)"
                   % (os.path.abspath(os.path.join(_HERE, "..", "..")), exc))

    spec = _spec()
    names = ["S-A1", "S-A5"] + (["S-A2"] if _dsn_available() else [])
    out = []
    for name in names:
        ad = JS.adapter(name, spec)
        check(ad.name == "joint:%s" % name, ad.name)
        check(len(ad.dimensions(spec)) == len(JS.free_axes(name)),
              "%s: dimension count" % name)

        # Cold start: distinct proposals, each a complete legal config.
        props = TSR.propose(spec, observations=[], n_points=4,
                            n_initial_points=4, seed=0, adapter=ad)
        check(len(props) == 4, "%s: got %d proposals" % (name, len(props)))
        keys = {JS.config_key(c) for c in props}
        check(len(keys) == 4, "%s: proposals must be distinct, got %d unique"
                              % (name, len(keys)))
        for cfg in props:
            check(all(k in cfg for k in JS.JOINT_KNOB_ORDER),
                  "%s: a proposal is missing axes" % name)
            check(JS.config_key(JS.canonicalise_config(cfg, spec=spec))
                  == JS.config_key(cfg),
                  "%s: a proposal must already be canonical" % name)
            for axis, value in JS.resolved_pins(name, spec).items():
                check(cfg[axis] == value,
                      "%s: proposal violates the pin on %s (%r != %r)"
                      % (name, axis, cfg[axis], value))

        # Warm start from a ledger: reproducible, and excludes what it has seen.
        obs = [(c, float(i)) for i, c in enumerate(props)]
        a = TSR.propose(spec, observations=obs, n_points=3,
                        n_initial_points=4, seed=1, adapter=ad)
        b = TSR.propose(spec, observations=obs, n_points=3,
                        n_initial_points=4, seed=1, adapter=ad)
        check([JS.config_key(x) for x in a] == [JS.config_key(x) for x in b],
              "%s: same seed must replay identically" % name)
        seen = {JS.config_key(c) for c, _ in obs}
        check(not (seen & {JS.config_key(x) for x in a}),
              "%s: a proposal repeated an observed config" % name)
        out.append("%s %d dims" % (name, len(ad.dimensions(spec))))

    # Escalation reads the joint boundary rule: a pinned axis must never
    # fire it, an interior best must not escalate on boundary grounds, and a
    # best sitting on a real edge must.
    name = names[0]
    ad = JS.adapter(name, spec)
    rng = np.random.default_rng(7)
    interior = JS.config_from_point(_point(name, spec, rng), name, spec=spec)
    # Put EVERY free numeric axis at its midpoint, not a hand-picked list:
    # a random draw lands on an integer range's endpoint often enough
    # (depth_exponent spans just 3..6) that a partial override is flaky.
    for axis in JS.free_axes(name):
        if axis in JS._NO_BOUNDARY:
            continue
        lo, hi = spec.range_of(axis)
        interior[axis] = JS._cast(axis, (float(lo) + float(hi)) / 2.0)
    edges = JS.boundary_axes(interior, spec, campaign_name=name)
    check(not edges, "an interior config must not be on any boundary: %s" % edges)

    edged = dict(interior)
    edged["num_transforms"] = int(spec.num_transforms[1])
    edges = JS.boundary_axes(edged, spec, campaign_name=name)
    check(edges == ["num_transforms"],
          "a config at the top of num_transforms must report exactly that: %s"
          % edges)

    # S-A1 pins the whole DSN block at values that are range endpoints; those
    # must be invisible to the boundary rule or every S-A1 run escalates.
    check(not any(a in JS.campaign("S-A1").pinned
                  for a in JS.boundary_axes(
                      JS.config_from_point(_point("S-A1", spec, rng), "S-A1",
                                           spec=spec),
                      spec, campaign_name="S-A1")),
          "a pinned axis must never fire the boundary rule")

    v = TSR.escalation_verdict([(edged, 1.0)] * 20, spec, tau_stop=1e9,
                               adapter=ad)
    check("num_transforms" in v.on_boundary,
          "escalation must see the joint boundary: %s" % v.on_boundary)
    out.append("boundary rule: interior clean, edge caught, pins invisible")
    return "; ".join(out)


def j27_canonical_values_are_in_range() -> str:
    """J27: every canonical value lies INSIDE its axis's searched range.

    This is the invariant that makes a canonicalised config replayable. A
    canonical value outside the range produces a point skopt refuses, so the
    first ledger entry with an inactive axis makes the whole warm start fail
    -- and it fails at `opt.tell`, far from the canonicalisation that caused
    it. Regression: `n_posterior_draws` was pinned to 0 while its range
    starts at 4 * d_theta = 104, which is exactly this failure.

    Checked across several d_theta, because the offending axis's bound moves
    with the problem and a single d_theta could pass by luck.
    """
    rows = []
    for d_theta in (10, 26, 40, 90):
        spec = JS.default_joint_space(p=26, embedding_dim=12, d_theta=d_theta,
                                      n_posterior_draws_max=400)
        for axis in JS.INACTIVE_CANONICAL:
            v = JS._inactive_value(axis, spec)
            r = spec.range_of(axis)
            if axis in JS._STR_AXES:
                check(v in r, "canonical %r for %s is not one of %r"
                              % (v, axis, r))
                continue
            lo, hi = float(r[0]), float(r[1])
            check(lo <= float(v) <= hi,
                  "canonical value %r for %s is OUTSIDE its range [%g, %g] "
                  "at d_theta=%d; a config carrying it cannot be told to the "
                  "optimiser" % (v, axis, lo, hi, d_theta))
        # And end to end: a canonicalised config replays as a legal point.
        for name in ("S-A1", "S-A5"):
            for axis, v in JS.resolved_pins(name, spec).items():
                if axis in JS._STR_AXES:
                    continue
                lo, hi = spec.range_of(axis)
                check(float(lo) <= float(v) <= float(hi),
                      "%s pins %s at %r, outside [%s, %s]"
                      % (name, axis, v, lo, hi))
        rows.append("d_theta=%d ok (draws floor %d)"
                    % (d_theta, spec.n_posterior_draws[0]))

    # The error message when a spec is genuinely needed and absent.
    try:
        JS._inactive_value("n_posterior_draws", None)
        raise AssertionError("a spec-dependent canonical value without a "
                             "spec must raise")
    except ValueError as exc:
        check("spec" in str(exc), exc)
    return "; ".join(rows)


def j35_canonical_values_match_the_loss_builder() -> str:
    """J35: the canonical values agree with the config that BUILDS the loss.

    Under "joint" and "joint_sep" the margin is inactive by A(l) but still
    READ, so its canonical value is not bookkeeping -- it is a training
    hyper-parameter. If joint_space pins it to one number and
    dsn_loss_adapter.DSNLossConfig defaults to another, the tuner and the
    Stage 3 bench train different losses while recording the same cell, and
    nothing anywhere reports a disagreement.

    Regression for exactly that: margin was 0.3 here (the DSN's TrainConfig)
    against 0.2 there (DSNLossConfig). Also checks that the runner's flag
    defaults agree, since a third copy lives in its argparse.
    """
    import ast as _ast

    adapter_path = os.path.abspath(os.path.join(_HERE, "..", "stage2",
                                                "dsn_loss_adapter.py"))
    if not os.path.isfile(adapter_path):
        raise Skip("dsn_loss_adapter.py not found")
    tree = _ast.parse(open(adapter_path, encoding="ascii").read())
    init = next((n for n in _ast.walk(tree)
                 if isinstance(n, _ast.FunctionDef) and n.name == "__init__"
                 and any(a.arg == "loss_type" for a in n.args.args)), None)
    check(init is not None, "could not find DSNLossConfig.__init__")
    names = [a.arg for a in init.args.args][1:]
    defaults = [_ast.literal_eval(d) for d in init.args.defaults]
    base = dict(zip(names[len(names) - len(defaults):], defaults))

    for axis in ("margin", "angular_alpha_deg", "lambda_sep"):
        want = base[axis]
        got = JS.INACTIVE_CANONICAL[axis]
        check(abs(float(got) - float(want)) < 1e-12,
              "INACTIVE_CANONICAL[%r] = %r but DSNLossConfig defaults to %r. "
              "These MUST agree: the axis is inactive but still read, so a "
              "mismatch trains a different loss than the ledger records."
              % (axis, got, want))

    runner = os.path.abspath(os.path.join(_HERE, "..", "stage3",
                                          "run_joint_arms.py"))
    if os.path.isfile(runner):
        import argparse as _ap
        rt = _ast.parse(open(runner, encoding="ascii").read())
        fn = next((n for n in _ast.walk(rt) if isinstance(n, _ast.FunctionDef)
                   and n.name == "build_parser"), None)
        arms = next((n for n in rt.body if isinstance(n, _ast.Assign)
                     and getattr(n.targets[0], "id", "") == "ARMS"), None)
        if fn is not None and arms is not None:
            ns = {"argparse": _ap, "__doc__": ""}
            exec(compile(_ast.Module(body=[arms, fn], type_ignores=[]),
                         "<r>", "exec"), ns)
            d = vars(ns["build_parser"]().parse_args(
                ["--arm", "A2", "--sim-shards", "s", "--out-dir", "o"]))
            for axis in ("margin", "angular_alpha_deg", "lambda_sep",
                         "loss_type", "mining_strategy"):
                check(d[axis] == base[axis],
                      "run_joint_arms --%s defaults to %r but DSNLossConfig "
                      "to %r; a run passing no flag would build a different "
                      "loss than Stage 3 did"
                      % (axis.replace("_", "-"), d[axis], base[axis]))
            spec = _spec()
            check(int(spec.fixed["strict_semihard"])
                  == int(bool(d["strict_semihard"])),
                  "the space fixes strict_semihard at %r while the runner "
                  "defaults to %r; the search and the bench would train "
                  "different cells"
                  % (spec.fixed["strict_semihard"], d["strict_semihard"]))
    return ("margin/alpha/lambda_sep = %.3g/%.3g/%.3g agree across "
            "joint_space, DSNLossConfig and the runner"
            % (base["margin"], base["angular_alpha_deg"], base["lambda_sep"]))


TESTS: Dict[str, Tuple[str, Callable[[], str]]] = {
    "J20": ("control recipe identity + shuffle marginals", j20_recipe_identity),
    "J21": ("point <-> config round trip", j21_point_config_round_trip),
    "J22": ("campaigns partition the space; A2/A5 nest A1", j22_campaign_partition),
    "J23": ("inactive coordinates canonicalised", j23_inactive_coordinates_are_canonical),
    "J24": ("guards", j24_guards),
    "J25": ("[needs skopt] dimensions match free axes", j25_skopt_dimensions),
    "J26": ("[needs skopt+SBI_HPC_DIR] GP proposes joint configs", j26_optimiser_adapter),
    "J27": ("canonical values lie inside their searched ranges", j27_canonical_values_are_in_range),
    "J35": ("canonical values match the loss builder and the runner", j35_canonical_values_match_the_loss_builder),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="pattern", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for tid, (desc, _) in TESTS.items():
            print("  %-5s %s" % (tid, desc))
        return 0

    ids = [t for t in TESTS if args.pattern is None or args.pattern in t]
    print("=" * 72)
    print("smoke_test_joint_space.py -- %d test(s)%s" %
          (len(ids), "" if _dsn_available() else
           "   [DSN_MAIN_DIR unset: DSN-loss clauses will skip]"))
    print("=" * 72)

    n_pass = n_fail = n_skip = 0
    for tid in ids:
        desc, fn = TESTS[tid]
        try:
            detail = fn() or ""
            n_pass += 1
            print("  %-5s PASS  %s" % (tid, detail), flush=True)
        except Skip as exc:
            n_skip += 1
            print("  %-5s SKIP  %s" % (tid, exc), flush=True)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print("  %-5s FAIL  %s: %s" % (tid, type(exc).__name__, exc),
                  flush=True)
            RESULTS.append((tid, "fail", traceback.format_exc()))

    print("-" * 72)
    print("%d passed, %d failed, %d skipped, %d total"
          % (n_pass, n_fail, n_skip, len(ids)))
    if n_fail:
        for tid, _, tb in RESULTS:
            print("\n--- %s ---\n%s" % (tid, tb))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
