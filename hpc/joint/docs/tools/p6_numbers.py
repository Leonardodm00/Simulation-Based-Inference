#!/usr/bin/env python3
"""p6_numbers.py -- every [RAN] number of P6_SEARCH_DRIVER.md, recomputed.

Torch-free. Unlike p4/p5 this script IMPORTS the search driver's own modules
(`stage4/joint_space.py`, `stage4/npe_tune_joint.py`, `npe_tune_search.py`,
`npe_tune_ledger.py`, `npe_tune_gates.py`, `npe_diagnostics.py`) and, where
present, scikit-optimize itself, because the objects P6 describes -- the
campaigns, the canonicalisation, the trial id, the proposal loop, the
control test -- are functions and the cheapest faithful replica of a
function is the function. Nothing here trains anything: the one piece of
the trainer that is read, `run_joint_arms.arm_config`, is extracted from
the runner's source with `ast` and executed on its own, since the runner
module imports torch at the top.

Blocks that need scikit-optimize (B2, B3 and the proposal half of B4) print
"SKIPPED" when it is not installed; the others run on numpy + scipy alone.

Run from hpc/joint/docs/tools:

    python p6_numbers.py            # all blocks

Pure ASCII, LF only.
"""

from __future__ import annotations

import ast
import glob
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
from contextlib import redirect_stdout

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # repo root
for _p in ("hpc/joint/stage4", "hpc/joint", "hpc"):
    _q = os.path.join(_ROOT, _p)
    if _q not in sys.path:
        sys.path.insert(0, _q)

import joint_space as JS          # noqa: E402
import npe_tune_joint as NTJ      # noqa: E402
import npe_tune_search as TSR     # noqa: E402
import npe_tune_ledger as TL      # noqa: E402
import npe_tune_gates as TG       # noqa: E402
import npe_diagnostics as D       # noqa: E402

try:
    import skopt                  # noqa: E402
    import sklearn                # noqa: E402
    HAVE_SKOPT = True
except ImportError:               # pragma: no cover
    skopt = sklearn = None
    HAVE_SKOPT = False
import scipy                      # noqa: E402

# ---------------------------------------------------------------------------
# The configured values the document quotes (each with its [REPO] line in P6)
# ---------------------------------------------------------------------------

SHAPES = dict(p=26, embedding_dim=12, d_theta=26)      # npe_tune_joint.py:753-755
N_TRAIN_DUP15HD = 20731                                 # P5 S3.3.4 (0.7 of 29616 rows by donor)
N_TRAIN_BENCH = 352                                     # P5 S3.3.4
PROPOSE = dict(n_points=8, n_initial_points=12, seed=0)  # npe_tune_joint.py:759, 777-778
FINALISTS = dict(top_k=3, rank_split="sel", gate_split="gate")  # :797-799
CONTROLS = dict(n_control=5, n_seeds=1, alpha=0.05, floor=0.0)  # :805-808
DSN_SEARCH = dict(l3c=(300, 100), mea=(300, 150))      # config_l3c_joint_search.json:217-218; config_mea_joint_full.davinci.json:226-227
NPE_TUNER = dict(n_seed_reps=2, n_control=2, tau_stop_fallback=0.01, k_sigma=3.0)  # npe_tune.py (P6 S3.5)
SKOPT_EI = dict(xi=0.01, kappa=1.96)                    # skopt acquisition.py:40-41
SKOPT_LBFGS = dict(n_points=10000, n_restarts_optimizer=5)  # skopt optimizer.py:298-299
GP_KERNEL = dict(const=(1.0, (0.01, 1000.0)), matern_nu=2.5, ls_bounds=(0.01, 100.0),
                 n_restarts_optimizer=2)                 # skopt utils.py:364-392


def hr(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def spec_default():
    return JS.default_joint_space(p=SHAPES["p"], embedding_dim=SHAPES["embedding_dim"],
                                  d_theta=SHAPES["d_theta"])


# ---------------------------------------------------------------------------
# B1  the space, the campaigns, the pins, the canonicalisation
# ---------------------------------------------------------------------------

def b1_space():
    hr("B1  space and campaigns at (p, E, d_theta) = (26, 12, 26)")
    spec = spec_default()
    print("axes: %d over %d blocks, sizes %s"
          % (len(JS.JOINT_KNOB_ORDER), len(JS.BLOCKS),
             [len(v) for v in JS.BLOCKS.values()]))
    print("resolved ranges: hidden_features %s, n_posterior_draws %s, batch_size_npe %s"
          % (spec.hidden_features, spec.n_posterior_draws, spec.batch_size_npe))
    print("fixed by the spec (passed to the runner only as --strict-semihard): %s"
          % json.dumps(spec.fixed, sort_keys=True))
    print("anchored_to: %s" % json.dumps(spec.anchored_to, sort_keys=True))

    # the width rule, step by step
    p, E = SHAPES["p"], SHAPES["embedding_dim"]
    scale = max(p, E)
    lo = max(32, 2 * scale)
    hi = max(lo * 2, 8 * scale)
    print("width rule: scale %d, lo max(32, 2 scale) = %d, hi max(2 lo, 8 scale) = %d, "
          "then clamp -> [%d, %d]" % (scale, lo, hi, min(lo, 64), max(hi, 256)))

    print("")
    print("| campaign | free | pinned | skopt dims (transformed) | Categorical / Integer / Real |")
    print("|---|---|---|---|---|")
    for name in ("S-A1", "S-A2", "S-A5", "S-A25"):
        c = JS.campaign(name)
        free = c.free()
        if HAVE_SKOPT:
            from skopt.space import Space
            from skopt.utils import normalize_dimensions
            dims = JS.space_dimensions(spec, name)
            tdims = Space(normalize_dimensions(dims)).transformed_n_dims
            kinds = [type(d).__name__ for d in dims]
            counts = "%d / %d / %d" % (kinds.count("Categorical"), kinds.count("Integer"),
                                       kinds.count("Real"))
            tds = str(tdims)
        else:
            counts, tds = "n/a", "n/a (skopt absent)"
        print("| %s | %d | %d | %s | %s |" % (name, len(free), len(c.pinned), tds, counts))

    print("")
    print("log-uniform axes: %s" % ", ".join(
        a for a in JS.JOINT_KNOB_ORDER
        if a in ("hidden_features", "lr", "lambda_sep", "one_minus_beta1", "weight_decay")))
    print("S-A1 resolved pins: %s" % json.dumps(JS.resolved_pins("S-A1", spec), sort_keys=True))
    print("S-A2 resolved pins: %s" % json.dumps(JS.resolved_pins("S-A2", spec), sort_keys=True))
    print("S-A5 resolved pins: %s" % json.dumps(JS.resolved_pins("S-A5", spec), sort_keys=True))

    # J27 invariant: every canonical value inside its range
    print("")
    print("canonical values inside their ranges (J27):")
    ok_all = True
    for axis in JS.INACTIVE_CANONICAL:
        v = JS._inactive_value(axis, spec)
        r = spec.range_of(axis)
        if axis in JS._STR_AXES:
            ok = v in r
            print("  %-18s %-10r in %s : %s" % (axis, v, list(r), ok))
        else:
            ok = float(r[0]) <= float(v) <= float(r[1])
            print("  %-18s %-10r in [%s, %s] : %s" % (axis, v, r[0], r[1], ok))
        ok_all &= ok
    print("  all inside: %s" % ok_all)


def b1_canonicalise():
    hr("B1b  canonicalisation: the three clamps, the projection, idempotence, the key")
    spec = spec_default()
    base = dict(depth_exponent=4, width_multiplier=2.0, block_family=1, embedding_size=12,
                head_fusion=0, dropout=0.1,
                dsn_on=1, log10_lambda_dsn=-1.0, loss_type="joint_sep",
                mining_strategy="easy_pos_semihard_neg", margin=0.7, angular_alpha_deg=10.0,
                lambda_sep=0.05,
                rep_on=0, log10_lambda_rep=0.5, warmup_frac_rep=0.3, n_posterior_draws=300,
                hidden_features=128, num_transforms=8,
                lr=5e-4, one_minus_beta1=0.05, weight_decay=1e-4, batch_size_npe=512)

    c1 = JS.canonicalise_config(base, spec=spec)
    print("(c)+(b) dsn_on=1 joint_sep/easy_pos_semihard_neg, rep_on=0, spec strict 1:")
    print("   margin %r -> %r (inactive under joint_sep: pinned); angular %r, lambda_sep %r kept"
          % (base["margin"], c1["margin"], c1["angular_alpha_deg"], c1["lambda_sep"]))
    print("   replicate block -> log10_lambda_rep %r, warmup_frac_rep %r, n_posterior_draws %r"
          % (c1["log10_lambda_rep"], c1["warmup_frac_rep"], c1["n_posterior_draws"]))
    print("   _strict_semihard_projected = %r (spec.fixed strict_semihard = %r)"
          % (c1["_strict_semihard_projected"], spec.fixed["strict_semihard"]))

    b2 = dict(base, loss_type="triplet", mining_strategy="hard")
    c2 = JS.canonicalise_config(b2, spec=spec)
    print("(c) dsn_on=1 triplet/hard: margin %r kept; angular %r -> %r; lambda_sep %r -> %r; "
          "strict projected = %r"
          % (c2["margin"], b2["angular_alpha_deg"], c2["angular_alpha_deg"],
             b2["lambda_sep"], c2["lambda_sep"], c2["_strict_semihard_projected"]))

    b3 = dict(base, dsn_on=0)
    c3 = JS.canonicalise_config(b3, spec=spec)
    print("(a) dsn_on=0: loss axes -> %s"
          % json.dumps({k: c3[k] for k in ("log10_lambda_dsn", "loss_type", "mining_strategy",
                                            "margin", "angular_alpha_deg", "lambda_sep")},
                       sort_keys=True))
    print("   '_strict_semihard_projected' present: %s" % ("_strict_semihard_projected" in c3))

    # without a spec
    try:
        JS.canonicalise_config(base)
        print("canonicalise_config(base) without spec: returned")
    except ValueError as exc:
        print("canonicalise_config(base, spec=None) with rep_on=0 raises ValueError: %s"
              % str(exc).split(".")[0])
    b4 = dict(base, rep_on=1)
    c4 = JS.canonicalise_config(b4)            # no spec: strict read as 0
    c4s = JS.canonicalise_config(b4, spec=spec)
    print("rep_on=1, no spec: _strict_semihard_projected = %r; with spec: %r; "
          "config_key equal: %s"
          % (c4["_strict_semihard_projected"], c4s["_strict_semihard_projected"],
             JS.config_key(c4) == JS.config_key(c4s)))

    # idempotence and the round trip
    print("idempotent: %s" % (JS.canonicalise_config(c1, spec=spec) == c1))
    pt = JS.point_from_config(c1, "S-A2")
    back = JS.config_from_point(pt, "S-A2", spec=spec)
    print("S-A2 point length %d; config_from_point(point_from_config(c)) == c on the "
          "searched axes: %s" % (len(pt), JS.config_key(back) == JS.config_key(c1)))
    k1 = JS.config_key(c1)
    k1b = JS.config_key(dict(c1, _campaign="S-A2", trial_id="abc", shuffle_pairs=1,
                             shuffle_seed=1000))
    print("config_key ignores bookkeeping and shuffle fields: %s (key has %d chars)"
          % (k1 == k1b, len(k1)))
    print("key fields: %d (= JOINT_KNOB_ORDER)" % len(json.loads(k1)))


# ---------------------------------------------------------------------------
# B2  the optimiser as skopt builds it; the fixed noise on standardised y
# ---------------------------------------------------------------------------

def _synthetic_nll(cfg) -> float:
    """A deterministic stand-in objective (nats/row), smooth in the knobs."""
    v = 2.0
    v += 0.30 * (math.log10(float(cfg["lr"]) / 1e-3)) ** 2
    v += 0.02 * (int(cfg["num_transforms"]) - 8) ** 2 / 4.0
    v += 0.15 * (math.log10(float(cfg["hidden_features"]) / 128.0)) ** 2
    v += 0.05 * float(cfg["dropout"])
    v -= 0.10 * int(cfg["dsn_on"]) * (1.0 - min(1.0, abs(float(cfg["log10_lambda_dsn"]) + 1.0)))
    return round(v, 6)


def b2_optimizer():
    hr("B2  the GP as built: kernel, acquisition, the fixed noise in standardised units")
    if not HAVE_SKOPT:
        print("SKIPPED: scikit-optimize not installed")
        return
    print("library versions: skopt %s, sklearn %s, scipy %s, numpy %s"
          % (skopt.__version__, sklearn.__version__, scipy.__version__, np.__version__))
    spec = spec_default()
    ad = JS.adapter("S-A1", spec)
    sigma_seed = 0.02
    opt = TSR.build_optimizer(spec, n_initial_points=PROPOSE["n_initial_points"],
                              seed=PROPOSE["seed"], noise=sigma_seed ** 2, adapter=ad)
    est = opt.base_estimator_
    print("base estimator: %s" % type(est).__name__)
    print("  kernel (before fit): %s" % est.kernel)
    print("  normalize_y=%s, noise=%r, n_restarts_optimizer=%d, alpha=%r"
          % (est.normalize_y, est.noise, est.n_restarts_optimizer, est.alpha))
    akw = opt.acq_func_kwargs or {}
    print("acq_func=%s, acq_optimizer=%s (resolved from 'auto'), xi=%r, kappa=%r "
          "(skopt defaults; acq_func_kwargs=%r), n_initial_points=%d, "
          "n_points(candidates)=%d, n_restarts=%d"
          % (opt.acq_func, opt.acq_optimizer, akw.get("xi", SKOPT_EI["xi"]),
             akw.get("kappa", SKOPT_EI["kappa"]), opt.acq_func_kwargs, opt.n_initial_points_,
             opt.n_points, opt.n_restarts_optimizer))
    print("space: %d dims, %d transformed, is_real=%s, is_categorical=%s"
          % (opt.space.n_dims, opt.space.transformed_n_dims, opt.space.is_real,
             opt.space.is_categorical))
    print("initial point generator: %r -> _initial_samples is %r: the design points are "
          "drawn one at a time by space.rvs(random_state=rng) at each ask, not "
          "generated up front"
          % (opt._initial_point_generator, opt._initial_samples))

    # tell 12 synthetic observations with a known spread and inspect the fit
    rng = np.random.default_rng(0)
    pts = [JS.config_from_point(x, "S-A1", spec=spec)
           for x in opt.space.rvs(n_samples=12, random_state=0)]
    ys = [_synthetic_nll(c) + float(rng.normal(0.0, 0.3)) for c in pts]
    obs = list(zip(pts, ys))
    opt2 = TSR.build_optimizer(spec, n_initial_points=PROPOSE["n_initial_points"],
                               seed=PROPOSE["seed"], noise=sigma_seed ** 2, adapter=ad,
                               observations=obs)
    gp = opt2.models[-1]
    ystd = float(np.ravel(gp._y_train_std)[0])
    ymean = float(np.ravel(gp._y_train_mean)[0])
    print("")
    print("after 12 observations (synthetic y, sample sd %.4f, mean %.4f):"
          % (float(np.std(ys)), float(np.mean(ys))))
    print("  kernel handed to the fit (gp.kernel): %s" % gp.kernel)
    print("  fitted kernel_ (skopt zeroes the white term here after the fit and keeps "
          "the level in noise_): %s" % gp.kernel_)
    print("  gp.noise_ = %r (fixed: bounds 'fixed'); gp._y_train_mean = %.4f, "
          "gp._y_train_std = %.4f" % (gp.noise_, ymean, ystd))
    print("  --sigma-seed %.3f -> noise %.1e in STANDARDISED units; in raw units "
          "%.1e x %.4f^2 = %.2e, i.e. an assumed seed sd of %.4f nats/row, not %.3f"
          % (sigma_seed, sigma_seed ** 2, sigma_seed ** 2, ystd,
             sigma_seed ** 2 * ystd ** 2, sigma_seed * ystd, sigma_seed))
    for s in (0.05, 0.3, 1.0):
        print("    if the ledger's sd were %.2f: assumed raw sd %.4f (ratio %.2f)"
              % (s, sigma_seed * s, s))
    # the EI incumbent
    print("  EI incumbent y_opt = min(y_i) = %.4f (the minimum OBSERVED noisy value)"
          % float(np.min(opt2.yi)))


# ---------------------------------------------------------------------------
# B3  the initial design and the replay: same seed, same points
# ---------------------------------------------------------------------------

def b3_replay():
    hr("B3  proposal rounds: the positional initial design and the seed replay")
    if not HAVE_SKOPT:
        print("SKIPPED: scikit-optimize not installed")
        return
    spec = spec_default()
    ad = JS.adapter("S-A1", spec)
    kw = dict(n_points=PROPOSE["n_points"], n_initial_points=PROPOSE["n_initial_points"],
              seed=PROPOSE["seed"], adapter=ad)

    r1 = TSR.propose(spec, observations=[], **kw)
    r1_again = TSR.propose(spec, observations=[], **kw)
    print("round 1, no observations: %d proposals; a second call with the same seed "
          "returns the same %d: %s"
          % (len(r1), len(r1_again),
             [JS.config_key(a) for a in r1] == [JS.config_key(a) for a in r1_again]))
    r1_excl = TSR.propose(spec, observations=[], exclude=r1, **kw)
    print("round 1 repeated with the 8 pending EXCLUDED (same seed, nothing evaluated): "
          "%d proposals (the design is replayed, every point is a duplicate, no "
          "observation exists to nudge with)" % len(r1_excl))
    r1_seed1 = TSR.propose(spec, observations=[], exclude=r1, **dict(kw, seed=1))
    print("  same, with --seed 1: %d proposals" % len(r1_seed1))

    # round 2: 8 observations, nothing pending. The optimiser has 12 - 8 = 4
    # initial points left, so the first 4 asks of the batch are random draws
    # and the last 4 come from the GP (fitted on 8 observations + 4 lies).
    obs8 = [(c, _synthetic_nll(c)) for c in r1]
    opt8 = TSR.build_optimizer(spec, n_initial_points=PROPOSE["n_initial_points"],
                               seed=PROPOSE["seed"], adapter=ad, observations=obs8)
    raw = opt8.ask(n_points=PROPOSE["n_points"], strategy="cl_min")
    keys1 = {JS.config_key(c) for c in r1}
    dup = [JS.config_key(JS.config_from_point(x, "S-A1", spec=spec)) in keys1 for x in raw]
    print("round 2 (8 observed), the raw ask of 8: _n_initial_points = %d; points already "
          "in the ledger: %s -- the 4 remaining random draws REPLAY round 1's first 4 "
          "points (same seed, same rng path), the 4 GP points are new"
          % (opt8._n_initial_points, dup))
    r2 = TSR.propose(spec, observations=obs8, **kw)
    obs8_other = [(c, y + 0.37 * ((i % 3) - 1)) for i, (c, y) in enumerate(obs8)]
    r2_other = TSR.propose(spec, observations=obs8_other, **kw)
    k2 = [JS.config_key(a) for a in r2]
    k2o = [JS.config_key(a) for a in r2_other]
    same = [a == b for a, b in zip(k2, k2o)]
    print("  propose() drops the 4 duplicates, keeps the 4 GP points and re-asks for 4: "
          "%d proposals; against a ledger with the SAME 8 configs and different y the "
          "proposals agree position by position as %s -- the first 4 are the GP's "
          "(y-dependent), the last 4 are the re-ask's random draws"
          % (len(r2), same))
    r2_again = TSR.propose(spec, observations=obs8, **kw)
    print("  replayed with the same ledger and seed: identical: %s"
          % (k2 == [JS.config_key(a) for a in r2_again]))
    r2_excl = TSR.propose(spec, observations=obs8, exclude=r2, **kw)
    print("  with those 8 pending and excluded: %d NEW proposals (the nudge tells "
          "max(y) = %.4f at the duplicate and re-asks)"
          % (len(r2_excl), max(y for _, y in obs8)))

    # round 3: 16 observations -> pure GP
    obs16 = obs8 + [(c, _synthetic_nll(c)) for c in r2]
    r3 = TSR.propose(spec, observations=obs16, **kw)
    obs16_other = obs8_other + [(c, _synthetic_nll(c) + 0.2) for c in r2]
    r3_other = TSR.propose(spec, observations=obs16_other, **kw)
    k3 = [JS.config_key(a) for a in r3]
    k3o = [JS.config_key(a) for a in r3_other]
    print("round 3 (16 observed): %d proposals, %d of them shared with a run on the same "
          "configs and different y (all GP now)"
          % (len(r3), len(set(k3) & set(k3o))))
    y16 = [y for _, y in obs16]
    print("  cl_min lie value = min(y) = %.4f; the worst observed value used by the "
          "nudge = %.4f" % (min(y16), max(y16)))
    print("with --n-points 8 and --n-initial-points 12 the GP is first consulted in "
          "round 2, after 8 observations and with 4 lies standing in for the undrawn "
          "design points; the first 12 ledger entries count as the design whatever "
          "proposed them; after round 2 the ledger holds 8 + 4 = 12 random points and "
          "4 GP points")


# ---------------------------------------------------------------------------
# B4  trial ids, argv, arms, the control path, exercised through the CLI
# ---------------------------------------------------------------------------

def _arm_config_from_runner():
    """Extract run_joint_arms.arm_config without importing torch."""
    path = os.path.join(_ROOT, "hpc", "joint", "stage3", "run_joint_arms.py")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "arm_config":
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {}
            exec(compile(mod, path, "exec"), ns)
            return ns["arm_config"]
    raise RuntimeError("arm_config not found in %s" % path)


def _normalise_argv(argv):
    out = []
    for a in argv:
        if a == sys.executable:
            out.append("python")
        elif a.endswith("run_joint_arms.py"):
            out.append("run_joint_arms.py")
        else:
            out.append(a)
    return out


def b4_ids_and_argv():
    hr("B4  trial ids, argv, arms: direct calls")
    spec = spec_default()
    cfg = JS.config_from_point([4, 2.0, 1, 12, 0, 0.1, 128, 8, 5e-4, 0.05, 1e-4, 512],
                               "S-A1", spec=spec)
    tid0 = NTJ._trial_id(cfg, "", "", [0])
    tid_h = NTJ._trial_id(cfg, "abc123", "", [0])
    tid_s = NTJ._trial_id(cfg, "", "", [1])
    tid_t = NTJ._trial_id(cfg, "", "", [0], tag="control")
    cfg_nc = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tid_nc = NTJ._trial_id(cfg_nc, "", "", [0])
    print("S-A1 config, defaults ('' split hash, '' digest, seeds [0], no tag): %s" % tid0)
    print("  --split-hash abc123 -> %s; --seed 1 -> %s; --tag control -> %s"
          % (tid_h, tid_s, tid_t))
    print("  the id covers the WHOLE config dict: dropping the bookkeeping field "
          "_campaign changes it -> %s (config_key would not change)" % tid_nc)
    print("  id length %d hex chars = sha256[:12]" % len(tid0))

    argv = _normalise_argv(NTJ.build_argv(cfg, arm=NTJ.arm_for_config(cfg),
                                          sim_shards="S/*.npz", out_dir="runs/x", seed=0,
                                          spec_fixed=spec.fixed))
    print("")
    print("build_argv for that S-A1 config (%d tokens, %d flags):" % (len(argv), (len(argv) - 2) // 2))
    print("  " + " ".join(argv))
    flags = [a for a in argv if a.startswith("--")]
    print("  flags: %s" % " ".join(flags))
    print("  no --sep-warmup-frac, no --num-bins, no --one-minus-beta2, no shuffle flag: %s"
          % all(f not in flags for f in ("--sep-warmup-frac", "--num-bins",
                                           "--one-minus-beta2", "--shuffle-seed",
                                           "--shuffle-pairs")))

    # arms
    a2 = dict(cfg, dsn_on=1, log10_lambda_dsn=-1.0)
    a5 = dict(cfg, rep_on=1, log10_lambda_rep=-0.5, n_posterior_draws=200)
    a25 = dict(cfg, dsn_on=1, rep_on=1)
    print("")
    print("arm_for_config: dsn 0/rep 0 -> %s; dsn 1 -> %s; rep 1 -> %s"
          % (NTJ.arm_for_config(cfg), NTJ.arm_for_config(a2), NTJ.arm_for_config(a5)))
    try:
        NTJ.arm_for_config(a25)
    except ValueError as exc:
        print("  dsn 1 / rep 1 -> ValueError: %s" % str(exc)[:90])
    ctrl = JS.control_config_from(JS.canonicalise_config(a2, spec=spec), permutation_seed=1000)
    print("  control of the dsn_on=1 config -> arm %r" % NTJ.arm_for_config(ctrl))
    argv_c = _normalise_argv(NTJ.build_argv(ctrl, arm=NTJ.arm_for_config(ctrl),
                                            sim_shards="S/*.npz", out_dir="runs/c", seed=0,
                                            spec_fixed=spec.fixed))
    lam = argv_c[argv_c.index("--lambda-dsn") + 1]
    print("  its argv passes --lambda-dsn %s and --arm shuffled; shuffle_seed appears in "
          "the argv: %s" % (lam, any("1000" == a for a in argv_c)))

    arm_config = _arm_config_from_runner()
    args = types.SimpleNamespace(lambda_dsn=float(lam), lambda_rep=0.0)
    for arm in ("A1", "A2", "A5", "shuffled"):
        print("  runner arm_config(%r): %s" % (arm, json.dumps(arm_config(arm, args),
                                                               sort_keys=True)))
    print("  so on the shuffled arm lambda_dsn and lambda_rep are 0.0 whatever the argv "
          "says, and the permutation seed is args.seed + 777 = %d for every control of "
          "a job run at SEED=0" % (0 + 777))

    # control ids differ, argvs do not
    ids, argvs = [], []
    for s in range(CONTROLS["n_control"]):
        c = JS.control_config_from(JS.canonicalise_config(a2, spec=spec),
                                   permutation_seed=1000 + s)
        ids.append(NTJ._trial_id(c, "", "", [0], tag="control"))
        argvs.append(" ".join(_normalise_argv(NTJ.build_argv(
            c, arm=NTJ.arm_for_config(c), sim_shards="S/*.npz", out_dir="runs/c",
            seed=0, spec_fixed=spec.fixed))))
    print("  %d controls of one finalist: %d distinct trial ids, %d distinct argv strings"
          % (len(ids), len(set(ids)), len(set(argvs))))

    # the shape anchors at propose time and at evaluate time
    print("")
    bench = JS.default_joint_space(10, 10, 10)
    cfg_b = JS.config_from_point([4, 2.0, 1, 12, 0, 0.1, 40, 8, 5e-4, 0.05, 1e-4, 512],
                                 "S-A1", spec=bench)
    tid_b = NTJ._trial_id(cfg_b, "", "", [0])
    cfg_b26 = JS.canonicalise_config(cfg_b, spec=spec)
    tid_b26 = NTJ._trial_id(cfg_b26, "", "", [0])
    print("an S-A1 config proposed at the bench anchors (10, 10, 10): n_posterior_draws "
          "pinned to %d, hidden_features %d, id %s" % (cfg_b["n_posterior_draws"],
                                                      cfg_b["hidden_features"], tid_b))
    print("  re-canonicalised by `evaluate` under the job's default anchors (26, 12, 26): "
          "n_posterior_draws %d, id %s; ids equal: %s; the pending file "
          "pending/%s.json is then never removed"
          % (cfg_b26["n_posterior_draws"], tid_b26, tid_b == tid_b26, tid_b))
    try:
        TSR.build_optimizer(spec, n_initial_points=12, seed=0, adapter=JS.adapter("S-A1", spec),
                            observations=[(cfg_b, 2.0)])
        print("  replaying that record into a (26, 12, 26) optimiser: accepted")
    except ValueError as exc:
        print("  replaying that record into a (26, 12, 26) optimiser: ValueError (%s...)"
              % str(exc)[:60])
    print("  launch_joint_tune.sh forwards CAMPAIGN, RESULTS_DIR, SIM_SHARDS, OUT_DIR and, "
          "when exported, REAL_SHARDS, SPLIT_HASH, CONTRACT_DIGEST, EPOCHS, STEPS_PER_EPOCH, "
          "SEED, SBI_HPC_DIR; not P, EMBEDDING_DIM, D_THETA, TAG")


def _fake_record(tid, campaign, cfg, nll, tag="", seed=0, extra=None):
    L0 = 3.0        # a synthetic prior floor, so that delta = L0 - nll is positive
    rec = {"trial_id": tid, "tag": tag, "config": cfg, "campaign": campaign,
           "arm": NTJ.arm_for_config(cfg), "seeds": [seed], "split_hash": "fake",
           "contract_digest": "", "status": "ok", "error": "",
           "L": nll, "L0": L0, "delta_hat": L0 - nll, "nll": nll, "delta": L0 - nll}
    if extra:
        rec.update(extra)
    return rec


def b4_cli():
    hr("B4b  the subcommands end to end on a temporary ledger (no runner)")
    if not HAVE_SKOPT:
        print("SKIPPED: scikit-optimize not installed")
        return
    spec = spec_default()
    root = tempfile.mkdtemp(prefix="p6_ledger_")
    camp = "S-A2"
    common = ["--campaign", camp, "--results-dir", root]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            NTJ.main(["propose"] + common + ["--n-points", "8", "--n-initial-points", "12",
                                              "--seed", "0"])
        head = buf.getvalue().splitlines()[0]
        print("propose round 1: %s" % head)
        pend = sorted(glob.glob(os.path.join(root, camp, "pending", "*.json")))
        print("  pending files: %d" % len(pend))

        # the job's index map: sorted listing
        ids = [os.path.basename(p)[:-5] for p in pend]
        print("  sorted index map, element 0 -> %s, element 7 -> %s" % (ids[0], ids[-1]))

        # "evaluate" them with fake records
        for p in pend:
            with open(p, "r", encoding="ascii") as fh:
                s = json.load(fh)
            cfg = JS.canonicalise_config(s["config"], spec=spec)
            tid = NTJ._trial_id(cfg, "", "", [0])
            assert tid == s["trial_id"], (tid, s["trial_id"])
            NTJ.write_record(root, camp, _fake_record(tid, camp, cfg, _synthetic_nll(cfg)))
            os.remove(p)
        buf = io.StringIO()
        with redirect_stdout(buf):
            NTJ.main(["propose"] + common + ["--n-points", "8", "--n-initial-points", "12",
                                              "--seed", "0"])
        print("propose round 2: %s" % buf.getvalue().splitlines()[0])
        for p in sorted(glob.glob(os.path.join(root, camp, "pending", "*.json"))):
            with open(p, "r", encoding="ascii") as fh:
                s = json.load(fh)
            cfg = JS.canonicalise_config(s["config"], spec=spec)
            NTJ.write_record(root, camp, _fake_record(s["trial_id"], camp, cfg,
                                                      _synthetic_nll(cfg)))
            os.remove(p)
        # a failed and a non-finite record, which must be dropped
        bad = JS.canonicalise_config(dict(cfg, lr=2e-3), spec=spec)
        NTJ.write_record(root, camp, _fake_record("f" * 12, camp, bad, float("nan")))
        NTJ.write_record(root, camp, dict(_fake_record("e" * 12, camp, bad, 1.0),
                                          status="failed"))
        obs = NTJ.load_observations(root, camp, spec)
        n_files = len(glob.glob(os.path.join(root, camp, "trials", "*.json")))
        print("ledger: %d record files, %d enter the surrogate (one nan, one failed dropped)"
              % (n_files, len(obs)))
        dsn_on = sum(int(c["dsn_on"]) for c, _ in obs)
        print("  dsn_on = 1 in %d of %d observations (S-A2 searches the switch)"
              % (dsn_on, len(obs)))

        # status with and without --sigma-seed
        for extra, label in (([], "no --sigma-seed (tau_stop 0.0)"),
                             (["--sigma-seed", "0.05"], "--sigma-seed 0.05")):
            buf = io.StringIO()
            with redirect_stdout(buf):
                NTJ.main(["status"] + common + ["--top", "3"] + extra)
            lines = buf.getvalue().splitlines()
            print("status, %s: %s" % (label, lines[-1]))

        # finalists: default labels, then equal labels
        buf = io.StringIO()
        with redirect_stdout(buf):
            NTJ.main(["finalists"] + common + ["--top-k", "3"])
        out = buf.getvalue().splitlines()
        print("finalists (defaults): %s; warning printed: %s"
              % (out[0], any(l.startswith("[warn]") for l in out)))
        buf = io.StringIO()
        with redirect_stdout(buf):
            NTJ.main(["finalists"] + common + ["--top-k", "3", "--rank-split", "sel",
                                               "--gate-split", "sel"])
        out = buf.getvalue().splitlines()
        print("finalists (--gate-split sel): warning printed: %s"
              % any(l.startswith("[warn]") for l in out))
        with open(os.path.join(root, camp, "finalists.json"), "r", encoding="ascii") as fh:
            fin = json.load(fh)
        print("  finalists.json keys per entry: %s; 'delta_gate_split' present: %s"
              % (sorted(fin["finalists"][0].keys()),
                 any("delta_gate_split" in f for f in fin["finalists"])))
        arms = [NTJ.arm_for_config(f["config"]) for f in fin["finalists"]]
        print("  finalist arms: %s" % arms)

        # controls --plan
        buf = io.StringIO()
        with redirect_stdout(buf):
            NTJ.main(["controls"] + common + ["--plan", "--n-control", "5"])
        out = buf.getvalue().splitlines()
        print("controls --plan: %s" % out[0])
        print("  last line: %s" % out[-1])
        specs = sorted(glob.glob(os.path.join(root, camp, "control_specs", "*.json")))
        print("  control_specs/: %d files; pending/: %d files"
              % (len(specs), len(glob.glob(os.path.join(root, camp, "pending", "*.json")))))
        with open(specs[0], "r", encoding="ascii") as fh:
            s0 = json.load(fh)
        # evaluate --pending-id on a control id, as the plan message instructs
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                NTJ.main(["argv"] + common + ["--sim-shards", "S/*.npz", "--out-dir", "runs",
                                              "--pending-id", s0["trial_id"]])
            print("argv --pending-id <control id>: returned")
        except FileNotFoundError as exc:
            print("argv --pending-id <control id>: FileNotFoundError on .../%s"
                  % os.path.relpath(exc.filename, root))
        buf = io.StringIO()
        with redirect_stdout(buf):
            NTJ.main(["argv"] + common + ["--sim-shards", "S/*.npz", "--out-dir", "runs",
                                          "--config-json", specs[0]])
        argv = _normalise_argv(buf.getvalue().split())
        print("argv --config-json <control spec>: --arm %s, --lambda-dsn %s"
              % (argv[argv.index("--arm") + 1], argv[argv.index("--lambda-dsn") + 1]))

        # score: identical controls (what identical runs give), then jittered
        by_fin = {}
        for p in specs:
            with open(p, "r", encoding="ascii") as fh:
                s = json.load(fh)
            by_fin.setdefault(s["finalist_trial_id"], []).append(s)
        for mode, jitter in (("identical", 0.0), ("jitter 1e-9", 1e-9)):
            for fid, ss in by_fin.items():
                for j, s in enumerate(ss):
                    cfg = JS.canonicalise_config(s["config"], spec=spec)
                    floor_delta = 0.05 + jitter * j          # the control's gain
                    NTJ.write_record(root, camp, _fake_record(
                        s["trial_id"], camp, cfg, 3.0 - floor_delta, tag="control"))
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    NTJ.main(["controls"] + common + ["--score", "--n-control", "5"])
            except SystemExit as exc:
                print("controls --score (%s): SystemExit: %s" % (mode, exc))
                continue
            out = buf.getvalue().splitlines()
            verdict_lines = [l for l in out if l.startswith("NO finalist") or
                             l.startswith("shipped configuration")]
            with open(os.path.join(root, camp, "controls.json"), "r", encoding="ascii") as fh:
                cj = json.load(fh)
            ps = ["%.3g" % v["pvalue"] for v in cj["verdicts"]]
            sds = ["%.2g" % v["control_sd"] for v in cj["verdicts"]]
            rej = [v["rejected"] for v in cj["verdicts"]]
            print("controls --score, controls %s: control sd %s, raw p %s, rejected %s"
                  % (mode, sds, ps, rej))
            print("  %s" % (verdict_lines[0] if verdict_lines else "(no verdict line)"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# B5  the control test: t, p, Holm, the cost
# ---------------------------------------------------------------------------

def b5_control_test():
    hr("B5  control_pvalue / Holm: degenerate inputs, thresholds, a worked case")
    from scipy import stats
    n_ctrl, n_s, alpha = CONTROLS["n_control"], CONTROLS["n_seeds"], CONTROLS["alpha"]
    same = [0.05] * n_ctrl
    print("identical controls %s -> (t, p) = %s" % (same, TG.control_pvalue(0.12, same)))
    print("one control -> %s" % (TG.control_pvalue(0.12, [0.05]),))
    jit = [0.05 + 1e-9 * j for j in range(n_ctrl)]
    t, p = TG.control_pvalue(0.12, jit)
    print("controls jittered by 1e-9: sd %.2e, t = %.3g, p = %.3g" % (np.std(jit, ddof=1), t, p))
    s = 0.02
    ctrl = [0.05 - 0.02, 0.05 - 0.01, 0.05, 0.05 + 0.01, 0.05 + 0.02]
    sd = float(np.std(ctrl, ddof=1))
    for gap in (0.03, 0.05, 0.08):
        t, p = TG.control_pvalue(0.05 + gap, ctrl, n_seeds=n_s)
        print("worked case: controls sd %.4f, gap %.2f nats/row -> t = %.3f, p = %.4f"
              % (sd, gap, t, p))
    se_pred = sd * math.sqrt(1.0 / n_s + 1.0 / n_ctrl)
    se_ci = sd * math.sqrt(1.0 / n_ctrl)
    print("se factor sqrt(1/n_s + 1/n_ctrl) = %.4f vs sqrt(1/n_ctrl) = %.4f (ratio %.3f)"
          % (math.sqrt(1.0 / n_s + 1.0 / n_ctrl), math.sqrt(1.0 / n_ctrl),
             se_pred / se_ci))
    K = FINALISTS["top_k"]
    thr = [alpha / (K - r) for r in range(K)]
    print("Holm thresholds for K = %d at alpha = %.2f: %s (smallest p first)"
          % (K, alpha, ["%.4f" % x for x in thr]))
    nu = n_ctrl - 1
    for a in (alpha / K, alpha):
        tcrit = float(stats.t.isf(a, df=nu))
        print("  one-sided t critical at %.4f with nu = %d: %.3f -> minimal gap %.2f x s_j"
              % (a, nu, tcrit, tcrit * math.sqrt(1.0 / n_s + 1.0 / n_ctrl)))
    fam = D.holm_bonferroni([0.004, 0.030, 0.020], alpha=alpha)
    print("holm_bonferroni([0.004, 0.030, 0.020]) -> adjusted %s, rejected %s"
          % (["%.3f" % x for x in fam.adjusted], list(map(bool, fam.rejected))))
    verdicts = TG.per_finalist_control_verdicts(
        [dict(name="r1", delta=0.10, control_deltas=ctrl),
         dict(name="r2", delta=0.12, control_deltas=same),
         dict(name="r3", delta=0.13, control_deltas=ctrl)], alpha=alpha)
    print("per_finalist_control_verdicts with one untestable finalist: family size %d, "
          "adjusted %s, rejected %s"
          % (sum(np.isfinite(v.pvalue) for v in verdicts),
             ["%.3g" % v.adjusted_pvalue for v in verdicts], [v.rejected for v in verdicts]))
    print("cost: K x n_ctrl = %d x %d = %d control runs; plus K finalist re-scores if a "
          "gate split existed" % (K, n_ctrl, K * n_ctrl))


# ---------------------------------------------------------------------------
# B6  escalation verdict on synthetic traces; the boundary band per axis
# ---------------------------------------------------------------------------

def b6_escalation():
    hr("B6  escalation_verdict: window, tau_stop, boundary; the boundary tolerance per axis")
    spec = spec_default()
    ad = JS.adapter("S-A1", spec)

    def mk(n, improving, edge=False, slope=0.5):
        rng = np.random.default_rng(1)
        obs = []
        for i in range(n):
            pt = [4, 2.0, 1, 12, 0, 0.1, 128, 8, 5e-4, 0.05, 1e-4, 512]
            pt[8] = float(1e-4 * 10 ** (1.3 * rng.random()))   # lr
            cfg = JS.config_from_point(pt, "S-A1", spec=spec)
            y = 2.5 - (slope * i / n if improving else 0.0) + 0.01 * rng.random()
            obs.append((cfg, y))
        if edge:
            best = min(range(n), key=lambda i: obs[i][1])
            c = dict(obs[best][0], weight_decay=1e-5)
            obs[best] = (c, obs[best][1])
        return obs

    print("window w = max(5, n // 3) capped at n - 1:")
    for n in (2, 5, 8, 12, 20, 24, 36, 60):
        w = max(5, n // 3)
        w = min(w, n - 1) if n > 1 else 0
        print("  n = %2d -> w = %d" % (n, w))
    cases = [("n = 8, flat", mk(8, False), 0.02),
             ("n = 20, flat, tau 0.02", mk(20, False), 0.02),
             ("n = 20, flat, tau 0.0 (no --sigma-seed)", mk(20, False), 0.0),
             ("n = 20, descending, tau 0.02", mk(20, True), 0.02),
             ("n = 20, descending 1e-3 per step, tau 0.02", mk(20, True, slope=0.02), 0.02),
             ("n = 20, descending 1e-3 per step, tau 0.0", mk(20, True, slope=0.02), 0.0),
             ("n = 20, flat, best on weight_decay edge", mk(20, False, edge=True), 0.02)]
    for label, obs, tau in cases:
        v = TSR.escalation_verdict(obs, spec, tau_stop=tau, adapter=ad)
        print("%s -> %s; recent improvement %.4f; boundary %s"
              % (label, "ESCALATE" if v.escalate else "STOP", v.recent_improvement,
                 v.on_boundary))
        for r in v.reasons:
            print("    - %s" % r)

    print("")
    print("boundary band eps = rel_tol * max(1, |lo|, |hi|), rel_tol = 1e-6, on the free "
          "numeric axes of S-A25 (integers compare exactly to lo and hi):")
    print("| axis | lo | hi | eps | eps / lo | band at lo in decades |")
    print("|---|---|---|---|---|---|")
    for axis in JS.JOINT_KNOB_ORDER:
        if axis in JS._NO_BOUNDARY or axis in JS._INT_AXES:
            continue
        lo, hi = spec.range_of(axis)
        lo, hi = float(lo), float(hi)
        eps = 1e-6 * max(1.0, abs(lo), abs(hi))
        rel = (eps / lo) if lo > 0 else float("nan")
        dec = math.log10((lo + eps) / lo) if lo > 0 else float("nan")
        print("| %s | %g | %g | %.1e | %s | %s |"
              % (axis, lo, hi, eps, ("%.1e" % rel) if lo > 0 else "-- (lo = 0: absolute)",
                 ("%.4f" % dec) if lo > 0 else "--"))


# ---------------------------------------------------------------------------
# B7  the n_train trim (P5 S3.3.4) and the DSN / NPE-tuner constants
# ---------------------------------------------------------------------------

def b7_trim_and_constants():
    hr("B7  the batch trim at the two n_train values; the sibling drivers' constants")
    for n in (N_TRAIN_DUP15HD, N_TRAIN_BENCH):
        sp = JS.default_joint_space(26, 12, 26, n_train=n)
        print("n_train = %d -> batch_size_npe %s (thresholds 20 b: %s)"
              % (n, sp.batch_size_npe, [20 * b for b in (256, 512, 1024)]))
    for n in (26, 10):
        sp = JS.default_joint_space(n, 12, n)
        print("d_theta = p = %d -> n_posterior_draws lower bound %d, hidden_features %s"
              % (n, sp.n_posterior_draws[0], sp.hidden_features))
    print("DSN standalone search (gp_minimize): n_calls / n_initial_points = %s (l3c), %s (mea); "
          "legacy initial rule min(10, max(1, n_calls // 2)) at 300 -> %d"
          % (DSN_SEARCH["l3c"], DSN_SEARCH["mea"], min(10, max(1, 300 // 2))))
    print("joint driver: %d initial points, batches of %d -> the design spans %.1f rounds; "
          "12 / 300 = %.1f%% of the DSN budget, 12 / 23 = %.2f points per axis on S-A25, "
          "1.0 per axis on S-A1"
          % (PROPOSE["n_initial_points"], PROPOSE["n_points"],
             PROPOSE["n_initial_points"] / PROPOSE["n_points"], 100 * 12 / 300, 12 / 23))
    print("standalone NPE tuner baseline: --n-seed-reps %d, --n-control %d; status fallback "
          "--tau-stop %.2f, --k-sigma %.1f"
          % (NPE_TUNER["n_seed_reps"], NPE_TUNER["n_control"], NPE_TUNER["tau_stop_fallback"],
             NPE_TUNER["k_sigma"]))
    print("joint_tune.pbs: one element per pending spec, 8 h walltime, 4 CPUs, 16 GB; a "
          "round of 8 is `qsub -J 0-7`")


if __name__ == "__main__":
    b1_space()
    b1_canonicalise()
    b2_optimizer()
    b3_replay()
    b4_ids_and_argv()
    b4_cli()
    b5_control_test()
    b6_escalation()
    b7_trim_and_constants()
