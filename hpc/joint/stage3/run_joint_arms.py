#!/usr/bin/env python3
"""Run ONE (arm, seed) of the Stage 3 hypothesis test (plan v0.6, Stage 3).

One process per (arm, seed) so the whole comparison is a PBS array and a
failed arm costs one job rather than the campaign.

The arms, and what distinguishes them
-------------------------------------
    A0       encoder fitted by l_DSN on the REAL arm, frozen, then NPE on sim
             at frozen z. The status quo: two stages, two objectives, and the
             encoder used out of domain.
    A0s      the same but the encoder is fitted on SIM with generator labels.
             The control that separates the OBJECTIVE effect from the DOMAIN
             effect (eq. 9, Stage 3b). Without it, A0 - A1 is one number
             answering two questions.
    A1       NPE only, joint. The sbi embedding-net / BayesFlow regime.
    A2       NPE + lambda_dsn * l_DSN, joint. Nests A1 at lambda_dsn = 0.
    A2s      A2 with the metric term on SIM.
    A3       A2 warm-started at A0's psi. Asks whether A0's encoder is a good
             basin rather than a bad objective.
    A5       NPE + lambda_rep * L_rep, joint. The label never enters training.
    A_ref    NPE on FIXED hand-crafted summaries, no trainable encoder. If A1
             cannot beat this, the encoder is the problem, not the objective.
    shuffled theta permuted against x in TRAINING. Gives delta_min for G1: an
             arm whose gain does not exceed this has learned nothing.

Everything the report needs is written per run, including the PER-ROW held-out
NLL and the group label per row. Without those two arrays the paired bootstrap
of S2.4a cannot be built after the fact, only re-run -- and re-running is what
makes a comparison quietly non-reproducible.

Pure ASCII, LF only.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           os.path.join(_HERE, ".."), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latent_bank import concat_shards                      # noqa: E402
from latent_sbi_simulator import LatentSBISpec, prior_log_prob  # noqa: E402
from joint_batches import BatchSpec, ThreeStreamBatcher    # noqa: E402
from joint_losses import ReplicateConsistencyLoss, box_prior_covariance  # noqa: E402
from joint_model import build_joint_model                  # noqa: E402
from joint_train import TrainConfig, evaluate_npe, train_joint  # noqa: E402
from joint_diagnostics import (cluster_scores, effective_rank,  # noqa: E402
                               information_gain, local_information_spectrum,
                               load_repo_diagnostics, mc_prior_floor,
                               p_eff_from_spectrum_result, per_axis_contraction,
                               per_row_nll, replicate_report)

ARMS = ("A0", "A0s", "A1", "A2", "A2s", "A3", "A5", "A_ref", "shuffled")


# ---------------------------------------------------------------------------
# Fixed hand-crafted summary, for A_ref
# ---------------------------------------------------------------------------

class FixedStatsSummary(torch.nn.Module):
    """Burst-like statistics of an IFR window. NO trainable parameters.

    Deliberately crude and deliberately fixed: A_ref exists to answer "is the
    learned encoder worth its cost", so a tuned reference would beg the
    question.
    """

    N_STATS = 8

    def __init__(self):
        super().__init__()
        # sbi's `check_net_device` does `next(net.parameters())` on the
        # embedding net and raises StopIteration on a parameter-free module.
        # One frozen zero parameter satisfies it without touching the output,
        # and requires_grad=False keeps the summary genuinely FIXED -- which is
        # the whole point of A_ref: a tuned reference would beg the question it
        # is there to answer.
        self._device_anchor = torch.nn.Parameter(torch.zeros(1),
                                                 requires_grad=False)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True) + 1e-8
        xc = (x - mu) / sd
        feats = [
            mu, sd,
            x.amax(-1, keepdim=True),
            torch.quantile(x, 0.9, dim=-1, keepdim=True),
            (x > mu + sd).float().mean(-1, keepdim=True),          # burst frac
            (xc[:, 1:] * xc[:, :-1]).mean(-1, keepdim=True),       # lag-1 acf
            (xc ** 3).mean(-1, keepdim=True),                      # skew
            (x.diff(dim=-1).abs()).mean(-1, keepdim=True),         # roughness
        ]
        return torch.cat(feats, dim=-1)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def grouped_split(groups, fracs=(0.7, 0.15, 0.15), seed=0):
    """Frozen 3-way split BY GROUP, with a hash of the assignment.

    Grouping is by donor, not by row: windows of one culture never split
    across train / selection / report, or the held-out score is contaminated
    by the same preparation it is scoring.
    """
    groups = np.asarray(groups).ravel()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n = len(perm)
    n_tr = int(round(fracs[0] * n))
    n_se = int(round(fracs[1] * n))
    assign = {}
    for g in perm[:n_tr]:
        assign[g] = 0
    for g in perm[n_tr:n_tr + n_se]:
        assign[g] = 1
    for g in perm[n_tr + n_se:]:
        assign[g] = 2
    idx = np.array([assign[g] for g in groups])
    digest = hashlib.sha256(
        json.dumps({str(k): int(v) for k, v in sorted(assign.items())},
                   sort_keys=True).encode("ascii")).hexdigest()
    return idx, digest


# ---------------------------------------------------------------------------
# Encoder-only pre-training, for A0 / A0s / A3
# ---------------------------------------------------------------------------

def train_encoder_only(backbone, x, cls, dsn_loss_fn, steps, batch_size, lr,
                       seed=0, log_fn=None):
    """Fit psi by l_DSN alone. This is stage one of the two-stage arms."""
    torch.manual_seed(int(seed))
    opt = torch.optim.AdamW(backbone.parameters(), lr=lr)
    g = torch.Generator().manual_seed(int(seed) + 1)
    n = x.shape[0]
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=g)
        z = backbone(x[idx])
        loss = dsn_loss_fn(z, cls[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_fn is not None and step % max(1, steps // 4) == 0:
            log_fn("  encoder step %4d  l_DSN %.5f" % (step, float(loss)))
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


# ---------------------------------------------------------------------------
# Arm dispatch
# ---------------------------------------------------------------------------

def arm_config(arm, args):
    """(freeze_encoder, dsn_domain, lambda_dsn, lambda_rep, warm_start,
    fixed_summary, shuffle_theta) for each arm. One place, not scattered."""
    base = dict(freeze_encoder=False, dsn_domain=None, lambda_dsn=0.0,
                lambda_rep=0.0, warm_start=None, fixed_summary=False,
                shuffle_theta=False)
    if arm == "A0":
        base.update(freeze_encoder=True, dsn_domain="real")
    elif arm == "A0s":
        base.update(freeze_encoder=True, dsn_domain="sim")
    elif arm == "A1":
        pass
    elif arm == "A2":
        base.update(dsn_domain="real", lambda_dsn=args.lambda_dsn)
    elif arm == "A2s":
        base.update(dsn_domain="sim", lambda_dsn=args.lambda_dsn)
    elif arm == "A3":
        base.update(dsn_domain="real", lambda_dsn=args.lambda_dsn,
                    warm_start="A0")
    elif arm == "A5":
        base.update(lambda_rep=args.lambda_rep)
    elif arm == "A_ref":
        base.update(fixed_summary=True, freeze_encoder=True)
    elif arm == "shuffled":
        base.update(shuffle_theta=True)
    else:
        raise ValueError("unknown arm %r" % arm)
    return base


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-shards", required=True,
                   help="glob for arm-S shards")
    p.add_argument("--real-shards", default=None,
                   help="glob for arm-R shards; required for A0/A2/A3/A5")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=25)
    p.add_argument("--b-sim", type=int, default=128)
    p.add_argument("--b-met", type=int, default=32)
    p.add_argument("--b-rep", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lambda-dsn", type=float, default=0.1)
    p.add_argument("--lambda-rep", type=float, default=0.05)
    p.add_argument("--warmup-frac-rep", type=float, default=0.3)
    p.add_argument("--n-posterior-draws", type=int, default=128)
    p.add_argument("--encoder-steps", type=int, default=200)
    p.add_argument("--hidden-features", type=int, default=48)
    p.add_argument("--num-transforms", type=int, default=3)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--embedding-size", type=int, default=10)
    p.add_argument("--warm-start-ckpt", default=None,
                   help="A0 checkpoint, for arm A3")
    p.add_argument("--dsn-main-dir", default=None)
    p.add_argument("--sbi-hpc-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def load_arm(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no shards matched %r" % pattern)
    arrays, sidecar = concat_shards(paths)
    return arrays, sidecar, paths


def make_backbone(W, E, dsn_main_dir, seed):
    """The real DSN backbone if available, else a small GroupNorm CNN."""
    torch.manual_seed(int(seed))
    d = dsn_main_dir or os.environ.get("DSN_MAIN_DIR")
    if d and os.path.isfile(os.path.join(d, "backbone.py")):
        if d not in sys.path:
            sys.path.insert(0, d)
        from backbone import BackboneConfig, build_backbone
        return build_backbone(BackboneConfig(depth_exponent=3,
                                             width_multiplier=2.0,
                                             stem_width=16,
                                             embedding_size=E)), "dsn"
    import torch.nn as nn

    class SmallBackbone(nn.Module):
        def __init__(self, E):
            super().__init__()
            self.c1 = nn.Conv1d(1, 16, 5, stride=4, padding=2)
            self.n1 = nn.GroupNorm(4, 16)
            self.c2 = nn.Conv1d(16, 32, 3, stride=2, padding=1)
            self.n2 = nn.GroupNorm(4, 32)
            self.fc = nn.Linear(32, E)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = torch.relu(self.n1(self.c1(x)))
            h = torch.relu(self.n2(self.c2(h))).mean(-1)
            return torch.nn.functional.normalize(self.fc(h), dim=-1)

    return SmallBackbone(E), "fallback"


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg_arm = arm_config(args.arm, args)

    sim, sim_side, sim_paths = load_arm(args.sim_shards)
    real = real_side = None
    if args.real_shards:
        real, real_side, _ = load_arm(args.real_shards)
    needs_real = cfg_arm["dsn_domain"] == "real" or cfg_arm["lambda_rep"] > 0
    if needs_real and real is None:
        raise SystemExit("arm %s needs --real-shards" % args.arm)

    W = int(sim_side["W"])
    d_theta = int(np.asarray(sim["theta"]).shape[1])
    E = FixedStatsSummary.N_STATS if cfg_arm["fixed_summary"] \
        else args.embedding_size

    if args.dry_run:
        print("DRY RUN -- nothing trained")
        print("  arm              : %s (seed %d)" % (args.arm, args.seed))
        print("  config           : %s" % cfg_arm)
        print("  sim shards       : %d, rows %d, W %d, d_theta %d"
              % (len(sim_paths), sim["x"].shape[0], W, d_theta))
        print("  real rows        : %s"
              % (real["x"].shape[0] if real is not None else "-"))
        print("  embedding size   : %d%s"
              % (E, " (fixed summary)" if cfg_arm["fixed_summary"] else ""))
        print("  out              : %s"
              % os.path.join(args.out_dir, "%s_seed%d.json"
                             % (args.arm, args.seed)))
        return 0

    # ---- tensors and split -------------------------------------------------
    theta = torch.as_tensor(np.asarray(sim["theta"]), dtype=torch.float32)
    x = torch.as_tensor(np.asarray(sim["x"]), dtype=torch.float32)
    split, split_hash = grouped_split(sim["donor"], seed=args.seed)
    tr, se, rp = (split == 0), (split == 1), (split == 2)
    if rp.sum() == 0 or tr.sum() == 0:
        raise SystemExit("split has an empty train or report part (%d/%d/%d "
                         "rows); the bank has too few donors"
                         % (tr.sum(), se.sum(), rp.sum()))
    if se.sum() == 0:
        print("WARNING: empty selection split -- early stopping disabled; "
              "the LAST epoch's weights will be scored, not the best.")

    theta_tr, x_tr = theta[tr], x[tr]
    if cfg_arm["shuffle_theta"]:
        g = torch.Generator().manual_seed(args.seed + 777)
        theta_tr = theta_tr[torch.randperm(theta_tr.shape[0], generator=g)]

    # ---- prior -------------------------------------------------------------
    from sbi.utils import BoxUniform
    lo = torch.zeros(d_theta)
    hi = torch.ones(d_theta)
    prior = BoxUniform(low=lo, high=hi)
    latent = sim_side["latent_spec"]
    spec = LatentSBISpec(n_latent=latent["n_latent"],
                         label_idx=tuple(latent["label_idx"]),
                         class_centres=np.asarray(latent["class_centres"]),
                         tau_ov=latent["tau_ov"],
                         n_windows_per_trace=latent["n_windows_per_trace"],
                         T_win=latent["T_win"], fs=latent["fs"],
                         n_neurons=latent["n_neurons"])

    # ---- encoder -----------------------------------------------------------
    dsn_loss_fn = None
    if cfg_arm["dsn_domain"] is not None:
        from dsn_loss_adapter import DSNLossConfig, build_dsn_loss
        n_classes = int(len(np.unique(sim["cls"])))
        # The separation ramp is driven by how many times l_DSN is EVALUATED.
        # For the two-stage arms that is the encoder pre-training, so the
        # horizon is --encoder-steps; for the joint arms it is the joint loop.
        # [CORRECTION] Both used epochs * steps_per_epoch, so A0's ramp had
        # barely started when its encoder was frozen.
        total = (args.encoder_steps if cfg_arm["freeze_encoder"]
                 else args.epochs * args.steps_per_epoch)
        dsn_loss_fn = build_dsn_loss(n_classes, total_steps=total,
                                     cfg=DSNLossConfig(),
                                     dsn_main_dir=args.dsn_main_dir)

    if cfg_arm["fixed_summary"]:
        backbone, bb_kind = FixedStatsSummary(), "fixed-stats"
    else:
        backbone, bb_kind = make_backbone(W, E, args.dsn_main_dir, args.seed)

    if cfg_arm["freeze_encoder"] and cfg_arm["dsn_domain"] is not None:
        src = real if cfg_arm["dsn_domain"] == "real" else sim
        xs = torch.as_tensor(np.asarray(src["x"]), dtype=torch.float32)
        cs = torch.as_tensor(np.asarray(src["cls"]), dtype=torch.long)
        print("pre-training the encoder on the %s arm (%d rows)"
              % (cfg_arm["dsn_domain"], xs.shape[0]))
        train_encoder_only(backbone, xs, cs, dsn_loss_fn, args.encoder_steps,
                           args.b_met, args.lr, seed=args.seed, log_fn=print)
    elif cfg_arm["fixed_summary"]:
        pass

    model = build_joint_model(backbone, prior, theta_tr[:64], x_tr[:64],
                              hidden_features=args.hidden_features,
                              num_transforms=args.num_transforms,
                              num_bins=args.num_bins,
                              meta={"arm": args.arm, "seed": args.seed,
                                    "backbone": bb_kind,
                                    "split_hash": split_hash,
                                    # Needed to rebuild the encoder from a
                                    # checkpoint alone (Stage 3b probes).
                                    "W": int(W), "embedding_size": int(E),
                                    "d_theta": int(d_theta),
                                    "fixed_summary": bool(cfg_arm["fixed_summary"])})

    if cfg_arm["warm_start"] == "A0":
        if not args.warm_start_ckpt:
            raise SystemExit("arm A3 needs --warm-start-ckpt (an A0 run)")
        ck = torch.load(args.warm_start_ckpt, map_location="cpu",
                        weights_only=False)
        model.encoder.load_state_dict(ck["encoder_state"])
        for p in model.encoder.parameters():
            p.requires_grad_(True)
        print("warm-started psi from %s" % args.warm_start_ckpt)

    if cfg_arm["freeze_encoder"]:
        model.freeze_encoder()          # weights AND eval mode, see joint_model

    # ---- batches and criteria ---------------------------------------------
    real_x = real_cls = real_donor = None
    if real is not None:
        real_x = torch.as_tensor(np.asarray(real["x"]), dtype=torch.float32)
        real_cls = torch.as_tensor(np.asarray(real["cls"]), dtype=torch.long)
        real_donor = np.asarray(real["donor"])

    # The metric stream reads the SIMULATED arm for the *s arms (A0s, A2s),
    # with the generator's labels, and the real arm otherwise.
    met_kw = {}
    if cfg_arm["dsn_domain"] == "sim":
        met_kw = {"met_x": x_tr,
                  "met_cls": torch.as_tensor(np.asarray(sim["cls"])[tr],
                                             dtype=torch.long)}
    batcher = ThreeStreamBatcher(
        theta_tr, x_tr, real_x, real_cls, real_donor, None,
        BatchSpec(args.b_sim,
                  args.b_met if cfg_arm["dsn_domain"] is not None else 0,
                  args.b_rep if cfg_arm["lambda_rep"] > 0 else 0),
        seed=args.seed, **met_kw)
    print(batcher.report())

    Sigma0 = box_prior_covariance(lo.numpy(), hi.numpy(),
                                  dtype=torch.get_default_dtype())
    rep_crit = ReplicateConsistencyLoss(Sigma0, n_draws=args.n_posterior_draws,
                                        warmup=args.warmup_frac_rep)

    tcfg = TrainConfig(epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                       lr=args.lr, weight_decay=args.weight_decay,
                       lambda_dsn=cfg_arm["lambda_dsn"],
                       lambda_rep=cfg_arm["lambda_rep"], patience=99,
                       rho_grad_probe=cfg_arm["lambda_dsn"] > 0)

    history = train_joint(model, batcher, tcfg,
                          val_theta=theta[se] if se.sum() else None,
                          val_x=x[se] if se.sum() else None,
                          dsn_loss_fn=dsn_loss_fn if cfg_arm["lambda_dsn"] > 0
                          else None,
                          rep_criterion=rep_crit if cfg_arm["lambda_rep"] > 0
                          else None,
                          n_posterior_draws=args.n_posterior_draws,
                          seed=args.seed)

    # ---- diagnostics on the REPORT split ----------------------------------
    theta_rp, x_rp = theta[rp], x[rp]
    nll = per_row_nll(model, theta_rp, x_rp)
    floor = mc_prior_floor(lambda t: prior_log_prob(spec, t.numpy()), theta_rp)

    # PRIMARY ENDPOINT (decision D10): the PSEUDO-REAL held-out NLL. theta is
    # recorded on arm R and withheld from every training loss, so scoring
    # against it is possible on the bench and impossible on the cohort -- which
    # is the entire reason the bench earns its cost (S4.0).
    #
    # [CORRECTION] The runner previously scored only the simulated report
    # split and reported that as L. That is the SECONDARY endpoint. Ranking
    # arms on it answers a different question, and P6 predicts explicitly that
    # the two rankings need not agree -- so reporting one of them under the
    # other's name would have made P6 untestable.
    nll_pr = None
    if real is not None and real_side.get("theta_withheld", False):
        theta_pr = torch.as_tensor(np.asarray(real["theta"]),
                                   dtype=torch.float32)
        x_pr = torch.as_tensor(np.asarray(real["x"]), dtype=torch.float32)
        nll_pr = per_row_nll(model, theta_pr, x_pr)
        floor_pr = mc_prior_floor(lambda t: prior_log_prob(spec, t.numpy()),
                                  theta_pr)
    with torch.no_grad():
        z_rp = model.encode(x_rp).cpu().numpy()
        samples = model.sample_posterior(args.n_posterior_draws,
                                          x_rp[:256]).cpu().numpy()

    n_spec = samples.shape[0]
    repo = load_repo_diagnostics(args.sbi_hpc_dir)
    if repo is not None and hasattr(repo, "information_spectrum"):
        spec_res = repo.information_spectrum(theta_rp[:n_spec].numpy(), samples)
        spectrum_source = "repo:npe_diagnostics.information_spectrum"
    else:
        spec_res = local_information_spectrum(theta_rp[:n_spec].numpy(), samples)
        spectrum_source = "local-fallback"
    p_eff_spec, spectrum_note = p_eff_from_spectrum_result(
        spec_res, n_rows=n_spec, d=d_theta)
    if p_eff_spec is None:
        print("WARNING: p_eff spectrum %s -- widen the report split or "
              "lower --n-report-min before reading P15/P16" % spectrum_note)

    rec = {
        "arm": args.arm,
        "seed": args.seed,
        "endpoint_primary": ("pseudo_real" if nll_pr is not None
                             else "simulated (arm R absent)"),
        "split_hash": split_hash,
        "backbone": bb_kind,
        "n_train": int(tr.sum()), "n_sel": int(se.sum()),
        "n_report": int(rp.sum()),
        "L": float(np.mean(nll)),
        "L0": floor,
        "delta_hat": information_gain(nll, floor),
        "L_pseudo_real": (None if nll_pr is None else float(np.mean(nll_pr))),
        "L0_pseudo_real": (None if nll_pr is None else floor_pr),
        "delta_hat_pseudo_real": (None if nll_pr is None
                                  else information_gain(nll_pr, floor_pr)),
        "val_npe": history[-1].get("val_npe"),
        "r_eff": effective_rank(z_rp),
        "contraction": per_axis_contraction(
            samples, np.sqrt(np.diag(Sigma0.numpy()))).tolist(),
        "p_eff_spectrum": p_eff_spec,
        "spectrum_source": spectrum_source,
        "spectrum_note": spectrum_note,
        "spectrum_n_rows": int(n_spec),
        "history": history,
        "config": vars(args),
        "arm_config": cfg_arm,
    }
    rec.update({"cluster_" + k: v
                for k, v in cluster_scores(z_rp, sim["cls"][rp],
                                           seed=args.seed).items()})

    if real is not None:
        from joint_batches import enumerate_donor_pairs
        pairs, n_single = enumerate_donor_pairs(real_donor)
        if pairs.shape[0]:
            take = pairs[:min(64, pairs.shape[0])]
            rr = replicate_report(model, real_x[take[:, 0]],
                                  real_x[take[:, 1]], Sigma0,
                                  n_draws=args.n_posterior_draws)
            rr["n_pairs"] = int(take.shape[0])
            rr["n_singleton_donors"] = int(n_single)
            rr["p_eff_gap"] = (None if p_eff_spec is None
                               else float(rr["p_eff_mean"] - p_eff_spec))
            rec["replicate"] = rr

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "%s_seed%d" % (args.arm, args.seed))
    with open(stem + ".json", "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    per_row = {"nll": nll, "group": np.asarray(sim["donor"])[rp],
               "theta": theta_rp.numpy()}
    if nll_pr is not None:
        per_row["nll_pseudo_real"] = nll_pr
        per_row["group_pseudo_real"] = np.asarray(real["donor"])
    np.savez_compressed(stem + "_perrow.npz", **per_row)
    torch.save(model.checkpoint(), stem + "_ckpt.pt")

    print("arm              : %s (seed %d)" % (args.arm, args.seed))
    if nll_pr is not None:
        print("L pseudo-real    : %.4f nats/row   <- PRIMARY (D10)"
              % rec["L_pseudo_real"])
        print("delta_hat (p-r)  : %.4f" % rec["delta_hat_pseudo_real"])
    else:
        print("L pseudo-real    : not scored (no arm-R shards with theta)")
    print("L simulated      : %.4f nats/row   (secondary)" % rec["L"])
    print("L0 (MC floor)    : %.4f" % rec["L0"])
    print("delta_hat (sim)  : %.4f" % rec["delta_hat"])
    print("r_eff            : %.4f of E = %d" % (rec["r_eff"], E))
    print("p_eff (spectrum) : %s  [%s; %s]"
          % ("n/a" if p_eff_spec is None else "%.4f" % p_eff_spec,
             spectrum_source, spectrum_note))
    if "replicate" in rec:
        print("T vs p_eff       : %.3f vs %.3f (log ratio %.3f)"
              % (rec["replicate"]["T_mean"], rec["replicate"]["p_eff_mean"],
                 rec["replicate"]["log_ratio_mean"]))
    print("written          : %s.json, _perrow.npz, _ckpt.pt" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
