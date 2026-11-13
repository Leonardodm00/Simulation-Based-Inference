#!/usr/bin/env python3
"""Build one bench bank shard (plan v0.6, S6 Stage 1).

One shard per call, so the whole bank is a PBS array over --shard-index.

Design points that are not obvious from the flags
-------------------------------------------------
* **Theta is drawn per DONOR, not per well.** Two wells of one donor share
  theta exactly and draw independent connectivity realisations. That is the
  premise of the replicate constraint (S2.5c) and the bank is where it has to
  be made true.
* **The arm decides what is withheld, not what is drawn.** Arm R records theta
  in the shard and sets `theta_withheld` in the sidecar; the loader is what
  must honour it. Withholding at write time would make the pseudo-real
  endpoint (S4.0) impossible to score.
* **--provider reference is for plumbing only.** It uses the smoke test's
  fixture generator, which is not the science. Any shard built with it is
  stamped `generator: reference-fixture` in the sidecar, so a bank built by
  accident cannot be mistaken for a real one downstream.

Expected output of a correct probe run (state this before running it):

    arm              : S
    rows             : n_traces * n_windows
    W                : round(T_win * fs)
    p (latent dim)   : n_latent
    distinct donors  : n_traces / wells_per_donor
    distinct realis. : n_traces        (one per well; NOT one per donor)

Pure ASCII, LF only.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

from latent_bank import build_sidecar, shape_report, write_shard
from latent_gap import GAP_MODES, GapSpec
from latent_nuisance import NuisanceSpec, sample_nuisance
from latent_realisation import RealisationSpec
from latent_sbi_simulator import (LatentSBISpec, provider_sha256, sample_prior,
                                  sample_box_prior, sample_prior_given_classes,
                                  simulate_windows, simplex_centres)

# [block 12, F-at] The stride between the id ranges of consecutive shards: a
# shard holds fewer than this many wells, so shard k's ids lie in
# [k * ID_STRIDE, (k + 1) * ID_STRIDE).
ID_STRIDE = 1000000


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True,
                   help="directory for the shard and its sidecar")
    p.add_argument("--arm", choices=("S", "R"), default="S",
                   help="S = simulated arm, R = pseudo-real arm (theta withheld)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--n-traces", type=int, default=64,
                   help="wells in this shard; must be a multiple of --wells-per-donor")
    p.add_argument("--wells-per-donor", type=int, default=2)
    p.add_argument("--donors-per-batch", type=int, default=8)
    p.add_argument("--n-windows", type=int, default=8)
    p.add_argument("--T-win", type=float, default=60.0)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--n-neurons", type=int, default=100)
    p.add_argument("--n-latent", type=int, default=6)
    p.add_argument("--n-label-axes", type=int, default=3)
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--tau-ov", type=float, default=0.10)
    p.add_argument("--prior", choices=("mixture", "box"), default="mixture",
                   help="theta's prior: the bench's class mixture (eq. 7; the "
                        "default), or the unit box (arm S of a synthetic data "
                        "set, D-118: theta i.i.d. uniform on (0, 1)^p, cls -1, "
                        "latent_spec kind 'box'); box is refused on arm R")
    p.add_argument("--class-centres", choices=("simplex", "regular"),
                   default="simplex",
                   help="the class centres of eq. (7): the bench's own "
                        "simplex_centres (the default; not a regular simplex "
                        "for C >= 3, F-ba), or the DSN generator's regular "
                        "simplex scaled about 0.5 by --centre-scale (D-120); "
                        "providers reference and bench")
    p.add_argument("--centre-scale", type=float, default=1.0,
                   help="s in mu_c = 0.5 + s (m_c - 0.5), with "
                        "--class-centres regular only (D-120: 0.6)")
    p.add_argument("--label-axes", default=None,
                   help="[block 25, D-132] --provider bench only: the "
                        "class-bearing axes, comma-separated increasing "
                        "indices (default the bench's 0-6); the others are "
                        "free -- uniform whatever the condition, and the "
                        "axes gap mode (a) displaces")
    p.add_argument("--donors-per-class", type=int, default=0,
                   help="a cohort with exactly this many donors per class, the "
                        "classes assigned to donor indices by a permutation "
                        "seeded like the rest of the shard, so batches mix "
                        "them (D-120); requires --n-traces = C * N * "
                        "--wells-per-donor; 0 (the default) draws each donor's "
                        "class uniformly, as eq. (7) does")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pi", type=float, default=0.0,
                   help="gap severity; must be 0 on arm S")
    p.add_argument("--gap-modes", default="range_shift",
                   help="comma-separated subset of " + ",".join(GAP_MODES))
    p.add_argument("--n-per-theta", type=int, default=2,
                   help="realisations per theta; >= 2 is what J13b/D17 need")
    p.add_argument("--provider", choices=("reference", "dsn", "bench"),
                   default="reference")
    p.add_argument("--dsn-main-dir", default=None,
                   help="explicit DSN tree for --provider dsn AND bench "
                        "(bench imports compute_ifr_trace from there); "
                        "default is this repo's hpc/dsn. DSN_MAIN_DIR is "
                        "ignored (migration step 2)")
    p.add_argument("--max-records", type=int, default=0,
                   help="cap traces, for a probe run; 0 = no cap")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and the expected output, write nothing")
    return p


def class_centres_for(args, n_label_axes):
    """[block 12] The class centres of eq. (7) for providers reference and
    bench: the bench's `simplex_centres` (the default, unchanged), or the DSN
    generator's regular simplex scaled about 0.5 (D-120)."""
    if args.class_centres == "regular":
        from latent_sbi_simulator import load_dsn_modules, regular_centres
        lbg, _gbd = load_dsn_modules(args.dsn_main_dir)
        return regular_centres(lbg, args.n_classes, n_label_axes,
                               args.centre_scale)
    if args.centre_scale != 1.0:
        raise SystemExit("--centre-scale needs --class-centres regular")
    return simplex_centres(args.n_classes, n_label_axes)


def make_provider_and_spec(args):
    """Return (provider, tag, spec, scale_convention).

    With --provider dsn the bench spec is DERIVED from the DSN LatentSpec
    rather than restated, so the class-centre convention cannot drift between
    the generator and the prior. With --provider bench the axis set is
    STRUCTURAL (10 axes, 7 class-bearing by default -- STAGE_A_BENCH_
    GENERATOR_SPEC_v1 S4; --label-axes names others, block 25, D-132), so
    --n-latent and --n-label-axes are ignored, as the dsn branch
    already ignores them. With --provider reference it is built from the CLI
    flags, since there is no generator to agree with.

    `scale_convention` is what the provider ACTUALLY emits, per row of x
    (O-2, resolved in bench_burst_provider):
        bench            per_unit_mean   -- trace / n_neurons, the analogue
                                            of EXTRACTOR_USAGE S4.1 eq. (3)
        dsn, reference   sum_over_units  -- the undivided population trace
    [CORRECTION] every earlier sidecar draft declared "per_electrode_mean"
    unconditionally, which no provider implemented -- the same class of
    false record as the extractor's "sum_over_electrodes" trap. Truthful
    per-provider values change the contract digest; no bank exists, so
    nothing is invalidated.
    """
    if args.provider == "bench":
        from bench_burst_provider import (BENCH_AXES, BENCH_LABEL_IDX,
                                          bench_free_idx, load_bench_provider)
        label = BENCH_LABEL_IDX
        if args.label_axes is not None:
            # [block 25, D-132] the class-bearing axes from the command line
            try:
                label = tuple(int(t) for t in str(args.label_axes).split(",") if t.strip())
                bench_free_idx(label)
            except ValueError as exc:
                raise SystemExit("--label-axes %r: %s" % (args.label_axes, exc))
        provider = load_bench_provider(args.dsn_main_dir, label_idx=label)
        spec = LatentSBISpec(
            n_latent=len(BENCH_AXES),
            label_idx=label,
            class_centres=class_centres_for(args, len(label)),
            tau_ov=args.tau_ov, n_windows_per_trace=args.n_windows,
            T_win=args.T_win, fs=args.fs, n_neurons=args.n_neurons,
            seed=args.seed)
        return provider, "bench", spec, "per_unit_mean"

    if args.label_axes is not None:
        raise SystemExit("--label-axes is for --provider bench (block 25): the dsn "
                         "provider's label axes are the DSN spec's, the reference "
                         "fixture's the first --n-label-axes")

    if args.provider == "dsn":
        from latent_sbi_simulator import (latent_spec_from_dsn,
                                          load_dsn_modules, DSNBurstProvider)
        if args.class_centres != "simplex" or args.centre_scale != 1.0:
            raise SystemExit(
                "--class-centres / --centre-scale are for providers reference "
                "and bench: the dsn provider's centres are the DSN "
                "generator's own (latent_spec_from_dsn)")
        lbg, gbd = load_dsn_modules(args.dsn_main_dir)
        dsn_spec = lbg.LatentSpec(n_classes=args.n_classes,
                                  class_overlap=args.tau_ov,
                                  n_per_class=tuple([1] * args.n_classes),
                                  n_neurons=args.n_neurons)
        spec = latent_spec_from_dsn(dsn_spec, lbg,
                                    n_windows_per_trace=args.n_windows,
                                    T_win=args.T_win, seed=args.seed)
        if abs(spec.fs - args.fs) > 1e-9:
            raise SystemExit(
                "--fs %.6f disagrees with the DSN spec's 1/w_size = %.6f. The "
                "generator's bin width is authoritative; drop --fs."
                % (args.fs, spec.fs))
        return (DSNBurstProvider(lbg, gbd, dsn_spec), "dsn", spec,
                "sum_over_units")

    from smoke_test_latent_sbi import ReferenceBurstProvider
    spec = LatentSBISpec(
        n_latent=args.n_latent,
        label_idx=tuple(range(args.n_label_axes)),
        class_centres=class_centres_for(args, args.n_label_axes),
        tau_ov=args.tau_ov, n_windows_per_trace=args.n_windows,
        T_win=args.T_win, fs=args.fs, n_neurons=args.n_neurons,
        seed=args.seed)
    return (ReferenceBurstProvider(), "reference-fixture", spec,
            "sum_over_units")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.arm == "S" and args.pi != 0.0:
        raise SystemExit("--pi must be 0 on arm S; the gap defines arm R")
    if args.prior == "box" and args.arm != "S":
        raise SystemExit("--prior box is arm S's (the simulated bank over the "
                         "whole box, D-118); arm R draws its theta from the "
                         "conditions' prior")
    if args.n_traces % args.wells_per_donor:
        raise SystemExit("--n-traces must be a multiple of --wells-per-donor")
    # theta is drawn per DONOR and each WELL draws its own realisation, so the
    # number of realisations per theta value is exactly wells_per_donor.
    # --n-per-theta previously recorded an intention and enforced nothing,
    # which is worse than not having it: a bank built with --n-per-theta 2 and
    # --wells-per-donor 1 would have looked configured for the D17 check while
    # containing one realisation per theta, exactly the confound the check
    # exists to detect.
    if args.wells_per_donor < args.n_per_theta:
        raise SystemExit(
            "--n-per-theta %d needs at least that many wells per donor "
            "(--wells-per-donor is %d): theta is per donor and the realisation "
            "is per well, so wells_per_donor IS the realisation count per "
            "theta." % (args.n_per_theta, args.wells_per_donor))
    # [block 12] a cohort of a set number of donors per class
    if args.donors_per_class < 0:
        raise SystemExit("--donors-per-class must be >= 0")
    if args.donors_per_class:
        if args.prior == "box":
            raise SystemExit("--donors-per-class needs the classes of eq. (7); "
                             "--prior box has none")
        need = args.n_classes * args.donors_per_class * args.wells_per_donor
        if args.n_traces != need:
            raise SystemExit(
                "--donors-per-class %d with %d classes and %d wells per donor "
                "is %d wells, but --n-traces is %d"
                % (args.donors_per_class, args.n_classes,
                   args.wells_per_donor, need, args.n_traces))
        if args.max_records:
            raise SystemExit("--max-records would unbalance the cohort of "
                             "--donors-per-class; build the cohort whole")

    n_traces = args.n_traces
    if args.max_records:
        n_traces = min(n_traces, args.max_records)
        n_traces -= n_traces % args.wells_per_donor
        if n_traces == 0:
            raise SystemExit("--max-records is below one donor's worth of wells")

    provider, provider_tag, spec, scale_conv = make_provider_and_spec(args)
    nspec = NuisanceSpec()
    rspec = RealisationSpec(n_per_theta=args.n_per_theta)
    gspec = GapSpec(modes=tuple(m for m in args.gap_modes.split(",") if m),
                    pi=args.pi)

    n_donors = n_traces // args.wells_per_donor
    base_seed = args.seed * 1000003 + args.shard_index

    if args.dry_run:
        print("DRY RUN -- nothing written")
        print("  arm                 : %s" % args.arm)
        print("  shard               : %d" % args.shard_index)
        print("  wells (traces)      : %d" % n_traces)
        print("  donors              : %d" % n_donors)
        print("  windows per trace   : %d" % spec.n_windows_per_trace)
        print("  rows                : %d" % (n_traces * spec.n_windows_per_trace))
        print("  W                   : %d" % spec.W)
        print("  provider            : %s" % args.provider)
        print("  prior               : %s" % (
            "box (theta uniform on (0, 1)^%d, cls -1)" % spec.n_latent
            if args.prior == "box" else "mixture (eq. 7)"))
        if args.prior != "box":
            print("  classes             : %d, centres %s%s, tau_ov %.3f, %s"
                  % (spec.n_classes, args.class_centres,
                     (" x %.3f" % args.centre_scale
                      if args.class_centres == "regular" else ""),
                     spec.tau_ov,
                     ("%d donors per class" % args.donors_per_class
                      if args.donors_per_class else "drawn per donor")))
        print("  ids                 : donor, well, batch + %d (shard %d x "
              "ID_STRIDE)" % (int(args.shard_index) * ID_STRIDE,
                              args.shard_index))
        print("  gap                 : modes=%s pi=%.3f"
              % (list(gspec.modes), gspec.pi))
        print("  theta_withheld      : %s" % (args.arm == "R"))
        print("  out                 : %s"
              % os.path.join(args.out_dir, "shard_%04d.npz" % args.shard_index))
        return 0

    rng = np.random.default_rng(base_seed)

    # One theta per DONOR. This is the line the whole construction rests on.
    # [block 11] --prior box draws it from the unit box instead of eq. (7);
    # the default path is unchanged, draw for draw.
    if args.prior == "box":
        cls_donor, phi_donor = sample_box_prior(spec, n_donors, rng)
    elif args.donors_per_class:
        # [block 12, D-120] each class on exactly N donor indices, in the
        # order of a permutation drawn from this shard's generator, so that
        # the batches of consecutive donors mix the classes.
        cls_donor = rng.permutation(np.repeat(np.arange(spec.n_classes),
                                              args.donors_per_class))
        phi_donor = sample_prior_given_classes(spec, cls_donor, rng)
    else:
        cls_donor, phi_donor = sample_prior(spec, n_donors, rng)

    # The shard-local ids key the nuisance and the realisation seeds, as they
    # always have (so a shard's windows are unchanged byte for byte); the
    # arrays carry them offset by the shard (below).
    donors = np.repeat(np.arange(n_donors), args.wells_per_donor)
    wells = np.arange(n_traces)
    batches = donors // max(1, args.donors_per_batch)
    # [block 12, F-at] bank-wide ids: shard k's donor, well and batch ids are
    # k * ID_STRIDE + the local id, so the shards of one bank never share an
    # id and concat_shards needs no offset; shard 0's ids are the local ones.
    if n_traces >= ID_STRIDE:
        raise SystemExit("a shard of %d wells is at or above ID_STRIDE %d"
                         % (n_traces, ID_STRIDE))
    id_off = int(args.shard_index) * ID_STRIDE

    nu, _ = sample_nuisance(nspec, ["b%d" % b for b in batches],
                            ["d%d" % d for d in donors],
                            ["w%d" % w for w in wells], base_seed=base_seed)

    J = spec.n_windows_per_trace
    n_rows = n_traces * J
    X = np.empty((n_rows, spec.W), dtype=np.float32)
    TH = np.empty((n_rows, spec.n_latent), dtype=np.float32)
    CL = np.empty(n_rows, dtype=np.int64)
    NU = np.empty((n_rows, nu.shape[1]), dtype=np.float32)
    RID = np.empty(n_rows, dtype=np.uint64)
    DON = np.empty(n_rows, dtype=np.int64)
    WEL = np.empty(n_rows, dtype=np.int64)
    BAT = np.empty(n_rows, dtype=np.int64)
    SUB = np.zeros(n_rows, dtype=np.int64)
    WIN = np.empty(n_rows, dtype=np.int64)
    CON = np.zeros(n_rows, dtype=bool)

    for i in range(n_traces):
        d = int(donors[i])
        x, info = simulate_windows(
            spec, phi_donor[d], provider, donor=int(d), well=int(wells[i]),
            subregion=0, realisation_spec=rspec, base_seed=base_seed,
            nuisance_spec=nspec, nu_row=nu[i], gap_spec=gspec,
            rng=np.random.default_rng(base_seed + 7919 * (i + 1)))
        sl = slice(i * J, (i + 1) * J)
        X[sl] = x.astype(np.float32)
        TH[sl] = info["phi_eff"].astype(np.float32)
        CL[sl] = int(cls_donor[d])
        NU[sl] = nu[i].astype(np.float32)
        RID[sl] = info["realisation_id"]
        DON[sl] = id_off + d
        WEL[sl] = id_off + int(wells[i])
        BAT[sl] = id_off + int(batches[i])
        WIN[sl] = np.arange(J)
        CON[sl] = info["contaminated"]

    arrays = {"x": X, "theta": TH, "cls": CL, "nu": NU, "realisation_id": RID,
              "donor": DON, "well": WEL, "batch": BAT, "subregion": SUB,
              "window_idx": WIN, "contaminated": CON}

    # [block 11] a box bank says so in its latent_spec, which is what the
    # runner reads (run_joint_arms.unit_box_kind); a mixture bank's dict is
    # unchanged, so its contract digest is too.
    latent = spec.to_dict()
    if args.prior == "box":
        latent["kind"] = "box"
    # [block 12] the cohort's design, recorded only when it departs from the
    # defaults, so every earlier bank's digest is unchanged.
    if args.class_centres != "simplex":
        latent["class_centres_mode"] = args.class_centres
        latent["centre_scale"] = float(args.centre_scale)
    if args.donors_per_class:
        latent["donors_per_class"] = int(args.donors_per_class)

    sidecar = build_sidecar(
        param_names=["phi%d" % k for k in range(spec.n_latent)],
        bounds_theta=[(0.0, 1.0)] * spec.n_latent,
        coord="unit_box", fs=spec.fs, w_size=1.0 / spec.fs, T_win=spec.T_win,
        W=spec.W, scale_convention=scale_conv,
        latent_spec=latent, nuisance_spec=nspec.to_dict(),
        realisation_spec=rspec.to_dict(), gap_spec=gspec.to_dict(),
        generator_sha256=provider_tag + ":" + provider_sha256(provider),
        theta_withheld=(args.arm == "R"), arm=args.arm,
        extra={"shard_index": args.shard_index,
               "base_seed": base_seed,
               "built_utc": datetime.datetime.now(
                   datetime.timezone.utc).isoformat(),
               "argv": sys.argv[1:]})

    # Verify the D17 property on the bytes actually written, not on intent.
    from latent_realisation import distinct_realisations_per_theta
    counts = distinct_realisations_per_theta(TH, RID)
    worst = min(counts.values()) if counts else 0
    if worst < args.n_per_theta:
        raise SystemExit(
            "built shard has only %d realisation(s) for some theta value but "
            "--n-per-theta is %d. J13b and decision D17 cannot be evaluated on "
            "this bank." % (worst, args.n_per_theta))

    path = os.path.join(args.out_dir, "shard_%04d.npz" % args.shard_index)
    written = write_shard(path, arrays, sidecar)
    print(shape_report(arrays, sidecar))
    print("realis./theta    : %d (>= --n-per-theta %d)" % (worst,
                                                           args.n_per_theta))
    print("rows written     : %d" % written)
    print("path             : %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
