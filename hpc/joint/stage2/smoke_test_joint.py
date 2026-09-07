"""Smoke test for joint_model.py / joint_batches.py / joint_train.py (J1-J9).

Run:  python3 smoke_test_joint.py

J10-J16, the replicate statistic itself, live in smoke_test_joint_losses.py.
The two files are separate because this one needs sbi and torch while that one
needs only torch, so the loss can be verified on a machine where sbi is not
installed.

Pure ASCII, LF only. Requires torch, numpy, sbi.
"""

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sbi.utils import BoxUniform  # noqa: E402

from joint_batches import BatchSpec, ThreeStreamBatcher, enumerate_donor_pairs  # noqa: E402
from joint_losses import ReplicateConsistencyLoss, box_prior_covariance  # noqa: E402
from joint_model import (JointDSNNPE, assert_no_batchnorm, build_joint_model,  # noqa: E402
                         state_sha256)
from joint_train import TrainConfig, evaluate_npe, gradient_cosine, train_joint  # noqa: E402

RESULTS = []
W, E_DIM, D_THETA = 64, 6, 4


def report(name, status, detail):
    RESULTS.append(status)
    print("[%s] %-46s %s" % (status, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


class TinyBackbone(nn.Module):
    """Stand-in encoder: GroupNorm only, L2-normalised output, like the DSN's.

    The real `OneDCNNBackbone` is imported when DSN_MAIN_DIR is set (J0); this
    exists so the loop and the model can be tested without the DSN repo.
    """

    def __init__(self, W, E, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=5, stride=4, padding=2)
        self.norm = nn.GroupNorm(2, 8)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(8, E)
        self.cfg = {"W": W, "E": E}

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = torch.relu(self.norm(self.conv(x)))
        h = self.drop(h.mean(dim=-1))
        return torch.nn.functional.normalize(self.fc(h), dim=-1)


def make_bank(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, D_THETA, generator=g)
    t = torch.linspace(0, 1, W).unsqueeze(0)
    # x depends on theta, so the NPE term has something to learn.
    x = (theta[:, :1] * torch.sin(6.28 * (2 + 4 * theta[:, 1:2]) * t)
         + theta[:, 2:3] + 0.05 * torch.randn(n, W, generator=g))
    return theta, x


def make_model(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    prior = BoxUniform(low=torch.zeros(D_THETA), high=torch.ones(D_THETA))
    theta, x = make_bank(64, seed=seed)
    bb = TinyBackbone(W, E_DIM, dropout=dropout)
    model = build_joint_model(bb, prior, theta, x, hidden_features=24,
                              num_transforms=2, num_bins=6)
    return model, prior


# ---------------------------------------------------------------------------
# J0 -- the real DSN backbone plugs in unchanged
# ---------------------------------------------------------------------------

def test_j0():
    if not os.environ.get("DSN_MAIN_DIR"):
        report("J0 real OneDCNNBackbone builds and back-propagates", "SKIP",
               "set DSN_MAIN_DIR to run this")
        return
    sys.path.insert(0, os.environ["DSN_MAIN_DIR"])
    from backbone import BackboneConfig, build_backbone
    cfg = BackboneConfig(depth_exponent=2, width_multiplier=2.0,
                         stem_width=8, embedding_size=E_DIM)
    bb = build_backbone(cfg)
    assert_no_batchnorm(bb)
    prior = BoxUniform(low=torch.zeros(D_THETA), high=torch.ones(D_THETA))
    theta, x = make_bank(32)
    model = build_joint_model(bb, prior, theta, x, hidden_features=16,
                              num_transforms=2, num_bins=5)
    loss = model.npe_loss(theta, x).mean()
    loss.backward()
    gnorm = sum(float(p.grad.abs().sum()) for p in model.encoder_parameters()
                if p.grad is not None)
    ok("J0 real OneDCNNBackbone builds and back-propagates", gnorm > 0,
       "sum |dL/dpsi| = %.4f, E = %d" % (gnorm, E_DIM))


# ---------------------------------------------------------------------------
# J1 / J2 -- the loop learns, and the NPE term alone matches a plain loop
# ---------------------------------------------------------------------------

def test_j1_j2():
    theta, x = make_bank(512, seed=1)
    vtheta, vx = make_bank(256, seed=99)
    model, _ = make_model(seed=3)
    batcher = ThreeStreamBatcher(theta, x, spec=BatchSpec(b_sim=64, b_met=0,
                                                          b_rep=0), seed=5)
    cfg = TrainConfig(epochs=6, steps_per_epoch=8, lr=1e-3, patience=99,
                      rho_grad_probe=False)
    before = evaluate_npe(model, vtheta, vx)
    train_joint(model, batcher, cfg, val_theta=vtheta, val_x=vx, log_fn=None,
                seed=3)
    after = evaluate_npe(model, vtheta, vx)
    try:
        evaluate_npe(model, vtheta[:0], vx[:0])
        empty_ok = False
    except ValueError:
        empty_ok = True
    ok("J1b an empty split raises instead of scoring 0.0", empty_ok,
       "a constant 0.0 'best' would restore epoch-0 weights at the end")

    ok("J1 the loop reduces held-out NPE loss", after < before,
       "val NLL %.4f -> %.4f (nats/row)" % (before, after))

    # Shuffled control: theta permuted against x destroys the dependence, so a
    # model that has learned anything must beat it. This is delta_min for G1.
    perm = torch.randperm(vtheta.shape[0])
    shuffled = evaluate_npe(model, vtheta[perm], vx)
    ok("J2 and beats the shuffled control", after < shuffled,
       "learned %.4f vs shuffled %.4f" % (after, shuffled))


# ---------------------------------------------------------------------------
# J4 -- the simulated stream never sees a real or surrogate row
# ---------------------------------------------------------------------------

def test_j4():
    theta, x = make_bank(200, seed=7)
    real_x = torch.rand(40, W)
    cls = torch.tensor(([0] * 20) + ([1] * 20))
    donor = np.repeat(np.arange(20), 2)
    surrogate = np.zeros(40, dtype=bool)
    surrogate[:6] = True

    b = ThreeStreamBatcher(theta, x, real_x, cls, donor, surrogate,
                           BatchSpec(b_sim=32, b_met=8, b_rep=4), seed=1)
    seen_sim = set()
    seen_met = set()
    seen_rep = set()
    for _ in range(50):
        batch = b.next()
        seen_sim.update(batch["sim"][2].tolist())
        seen_met.update(batch["met"][2].tolist())
        ig, igp = batch["rep"][2]
        seen_rep.update(ig.tolist() + igp.tolist())

    ok("J4a sim indices stay inside the simulated bank",
       max(seen_sim) < theta.shape[0],
       "%d distinct sim rows, max index %d of %d"
       % (len(seen_sim), max(seen_sim), theta.shape[0]))
    ok("J4b surrogate rows never enter the metric stream",
       not any(surrogate[i] for i in seen_met),
       "%d distinct metric rows, %d surrogates in the bank"
       % (len(seen_met), int(surrogate.sum())))
    ok("J4c surrogate rows never enter the replicate stream",
       not any(surrogate[i] for i in seen_rep),
       "a surrogate has no well identity, so it cannot be a replicate")

    # Surrogates are excluded by MASK, so the donor dtype cannot matter. A
    # narrow string dtype used to truncate the relabelled ids ('-10' -> '-1')
    # and fuse two surrogates into one donor.
    n_sur = 12
    donors_s = np.array(["d%d" % (i // 2) for i in range(n_sur + 8)])
    sur_s = np.zeros(donors_s.size, dtype=bool)
    sur_s[:n_sur] = True
    b2 = ThreeStreamBatcher(theta, x, torch.rand(donors_s.size, W),
                            torch.zeros(donors_s.size, dtype=torch.long),
                            donors_s, sur_s, BatchSpec(4, 0, 2), seed=2)
    touched = set(b2.pairs.ravel().tolist()) if b2.pairs.size else set()
    ok("J4e surrogates cannot pair even with a narrow string donor dtype",
       not any(sur_s[i] for i in touched),
       "%d surrogates of dtype %s, %d pairs, none touching a surrogate"
       % (n_sur, donors_s.dtype, b2.pairs.shape[0]))

    # The *s arms put the metric stream on the SIMULATED arm. The batch must
    # then come from sim rows, not real ones, or A2s == A2.
    sim_cls = torch.tensor([i % 3 for i in range(theta.shape[0])])
    b3 = ThreeStreamBatcher(theta, x, real_x, cls, donor, None,
                            BatchSpec(4, 6, 0), seed=3, met_x=x,
                            met_cls=sim_cls)
    xm, cm, im = b3.met_batch()
    from_sim = all(torch.equal(xm[k], x[im[k]]) for k in range(xm.shape[0]))
    ok("J4f met_x/met_cls redirect the metric stream to the sim arm",
       from_sim and "sim" in b3.report(),
       "%d metric rows, all identical to sim rows at their indices"
       % xm.shape[0])

    pairs, n_single = enumerate_donor_pairs([0, 0, 1, 1, 1, 2])
    ok("J4d donor pairs are enumerated exhaustively",
       pairs.shape[0] == 1 + 3 and n_single == 1,
       "donors of size 2,3,1 give %d pairs and %d singleton"
       % (pairs.shape[0], n_single))


# ---------------------------------------------------------------------------
# J5 -- SKIP with a reason
# ---------------------------------------------------------------------------

def test_j3():
    """J3: the real l_DSN, and the ramp the probe must not disturb."""
    if not os.environ.get("DSN_MAIN_DIR"):
        report("J3 real CompositeDSNLoss + miner", "SKIP",
               "set DSN_MAIN_DIR to run this")
        return
    from dsn_loss_adapter import DSNLossConfig, build_dsn_loss

    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(24, E_DIM), dim=-1)
    c = torch.tensor(([0] * 8) + ([1] * 8) + ([2] * 8))

    ad = build_dsn_loss(3, total_steps=100,
                        cfg=DSNLossConfig(loss_type="joint_sep",
                                          sep_warmup_frac=0.2))
    v = ad(z, c)
    ok("J3a the real CompositeDSNLoss evaluates", torch.isfinite(v) and v >= 0,
       "l_DSN = %.6f, mined %d triplets"
       % (float(v), int(ad.stats().get("n_mined", -1))))

    # Parity with a direct call: the adapter must add nothing but the mining.
    # TWO separate frozen_warmup blocks, not one. The context manager restores
    # the counter on EXIT; it does not hold it still during the block, so two
    # calls inside one block would see consecutive ramp factors and the
    # comparison would measure the ramp rather than the adapter. Each block
    # therefore starts from the same step.
    with ad.frozen_warmup():
        direct = float(ad.loss_fn(z, c, ad.miner(z, c)))
    with ad.frozen_warmup():
        through = float(ad(z, c))
    ok("J3b the adapter is exactly loss_fn(z, c, miner(z, c))",
       abs(direct - through) < 1e-12,
       "direct %.12f vs adapter %.12f" % (direct, through))

    for _ in range(5):
        ad(z, c)
    before = ad.warmup_state()
    with ad.frozen_warmup():
        ad(z, c)
        ad(z, c)
    after = ad.warmup_state()
    ok("J3c frozen_warmup leaves the separation ramp untouched",
       before == after,
       "step/scale %s before, %s after two frozen evaluations"
       % (before, after))

    ad(z, c)
    ok("J3d while a normal call still advances it",
       ad.warmup_state()[0] == before[0] + 1,
       "step %d -> %d" % (before[0], ad.warmup_state()[0]))

    # The gradient must reach z, or the term teaches the encoder nothing.
    z2 = z.clone().requires_grad_(True)
    ad(z2, c).backward()
    ok("J3e l_DSN back-propagates into the embedding",
       z2.grad is not None and float(z2.grad.abs().sum()) > 1e-8,
       "sum |dl/dz| = %.6f" % float(z2.grad.abs().sum()))


def test_j5():
    report("J5 one host sync per epoch on the training path", "SKIP",
           "needs a CUDA device; assert with torch.cuda.set_sync_debug_mode "
           "on the cluster")


# ---------------------------------------------------------------------------
# J6 -- checkpoint round-trip, encoder reloadable alone
# ---------------------------------------------------------------------------

def test_j6():
    model, prior = make_model(seed=11)
    theta, x = make_bank(32, seed=12)
    before = model.npe_loss(theta, x).detach().clone()

    ckpt = model.checkpoint()
    model2, _ = make_model(seed=999)          # different init on purpose
    diff_before = float((model2.npe_loss(theta, x) - before).abs().max())
    model2.load_checkpoint(ckpt)
    after = model2.npe_loss(theta, x).detach()

    ok("J6a checkpoint round-trip reproduces the loss exactly",
       torch.allclose(before, after, atol=1e-12),
       "max |diff| = %.2e (a different init differed by %.3f)"
       % (float((before - after).abs().max()), diff_before))

    ok("J6b the encoder is stored separately and hashed",
       ckpt["encoder_sha256"] == state_sha256(model.encoder.state_dict()),
       "encoder sha256 = %s..." % ckpt["encoder_sha256"][:16])

    # The encoder must be loadable ALONE, which is what keeps
    # dsn_frozen.load_frozen_dsn working against a joint checkpoint.
    fresh = type(model.encoder)(W, E_DIM)
    fresh.load_state_dict(ckpt["encoder_state"])
    z_a = model.encode(x)
    z_b = fresh(x)
    ok("J6c the encoder alone reproduces its embeddings",
       torch.allclose(z_a, z_b, atol=1e-12),
       "max |dz| = %.2e" % float((z_a - z_b).abs().max()))

    bad = dict(ckpt)
    bad["format"] = "something/else"
    try:
        model2.load_checkpoint(bad)
        caught = False
    except ValueError:
        caught = True
    ok("J6d an unknown checkpoint format is refused", caught,
       "load_checkpoint raises rather than half-loading")


# ---------------------------------------------------------------------------
# J7 -- rho_grad against a finite-difference cosine
# ---------------------------------------------------------------------------

def test_j6b_frozen():
    """A frozen encoder must stay in eval mode across train()/eval() flips."""
    model, _ = make_model(seed=13, dropout=0.5)
    x = torch.rand(8, W)
    model.freeze_encoder()
    model.train()                      # what evaluate_npe does on exit
    z1 = model.encode(x)
    z2 = model.encode(x)
    ok("J6e a frozen encoder is deterministic even after model.train()",
       torch.equal(z1, z2) and not model.encoder.training,
       "dropout=0.5 encoder, two passes identical, encoder.training=%s"
       % model.encoder.training)
    model.unfreeze_encoder()
    model.train()
    z3 = model.encode(x)
    z4 = model.encode(x)
    ok("J6f and stochastic again once unfrozen",
       not torch.equal(z3, z4), "dropout active after unfreeze")


def test_j7():
    torch.manual_seed(0)
    lin = nn.Linear(3, 1, bias=False)

    class Shim(object):
        def encoder_parameters(self):
            return lin.parameters()

        def zero_grad(self, set_to_none=True):
            lin.zero_grad(set_to_none=set_to_none)

    a_vec = torch.tensor([[1.0, 0.0, 0.0]])
    b_vec = torch.tensor([[0.0, 1.0, 0.0]])
    rho = gradient_cosine(Shim(), lambda: (lin(a_vec) ** 2).sum(),
                          lambda: (lin(b_vec) ** 2).sum())
    # d/dW (Wa)^2 = 2(Wa) a^T and likewise for b; a and b are orthogonal, so
    # the two gradients are orthogonal and the cosine is exactly 0.
    ok("J7a rho_grad is 0 for orthogonal-gradient terms", abs(rho) < 1e-9,
       "cosine = %.3e (analytic 0)" % rho)

    rho2 = gradient_cosine(Shim(), lambda: (lin(a_vec) ** 2).sum(),
                           lambda: 3.0 * (lin(a_vec) ** 2).sum())
    ok("J7b and +1 for a positively scaled copy", abs(rho2 - 1.0) < 1e-9,
       "cosine = %.9f (analytic 1)" % rho2)

    rho3 = gradient_cosine(Shim(), lambda: (lin(a_vec) ** 2).sum(),
                           lambda: -(lin(a_vec) ** 2).sum())
    ok("J7c and -1 for an opposed copy -- P4's signature",
       abs(rho3 + 1.0) < 1e-9, "cosine = %.9f (analytic -1)" % rho3)


# ---------------------------------------------------------------------------
# J8 -- the metric stream gives gradient to psi and none to omega
# ---------------------------------------------------------------------------

def test_j8():
    model, _ = make_model(seed=21)
    xm = torch.rand(16, W)
    z = model.encode(xm)
    # NOTE the loss must depend on the DIRECTION of z, not its norm. The
    # backbone L2-normalises its output, so ||z|| == 1 identically and any
    # loss that is a function of the norm alone has gradient exactly zero --
    # an easy way to write a test that passes while measuring nothing. Here
    # the loss pulls the batch together, which is direction-dependent.
    l_met = ((z - z.mean(0, keepdim=True)) ** 2).sum(-1).mean()
    l_met.backward()

    psi = sum(float(p.grad.abs().sum()) for p in model.encoder_parameters()
              if p.grad is not None)
    omega = [p.grad for p in model.flow_parameters()]
    none_or_zero = all(g is None or float(g.abs().sum()) == 0.0
                       for g in omega)
    ok("J8a the metric stream reaches psi", psi > 1e-6,
       "sum |dl/dpsi| = %.6f (threshold 1e-6, not >0: a norm-only loss "
       "would give ~1e-9 here and look like a pass)" % psi)
    ok("J8b and gives exactly nothing to omega", none_or_zero,
       "%d flow tensors, all grads None or 0" % len(omega))

    n_enc = len(list(model.encoder_parameters()))
    n_flow = len(model.flow_parameters())
    n_all = len(list(model.parameters()))
    ok("J8c psi and omega partition the parameters exactly",
       n_enc + n_flow == n_all,
       "%d encoder + %d flow = %d total (no double counting)"
       % (n_enc, n_flow, n_all))


# ---------------------------------------------------------------------------
# J9 -- the streams interact only through psi
# ---------------------------------------------------------------------------

def test_j14_real():
    """The replicate term on the REAL model: psi yes, omega no.

    J14 in smoke_test_joint_losses.py uses a toy where the encoder sits OUTSIDE
    the flow. In JointDSNNPE the encoder sits INSIDE the sbi estimator, so the
    two are not the same test, and it was the difference between them that hid
    a disconnected replicate term until Stage 3 (arm A5 scoring identically to
    A1). This checks the arrangement that actually runs.
    """
    from joint_losses import stop_grad_params
    model, _ = make_model(seed=71)
    Sigma0 = box_prior_covariance(np.zeros(D_THETA), np.ones(D_THETA),
                                  dtype=torch.get_default_dtype())
    crit = ReplicateConsistencyLoss(Sigma0, n_draws=48)
    xg, xgp = torch.rand(3, W), torch.rand(3, W)

    model.zero_grad(set_to_none=True)
    with stop_grad_params(model.flow_parameters()):
        loss, _ = crit(model.rsample_posterior(48, xg),
                       model.rsample_posterior(48, xgp))
    loss.backward()

    psi = sum(float(p.grad.abs().sum()) for p in model.encoder_parameters()
              if p.grad is not None)
    omega = [p.grad for p in model.flow_parameters()]
    ok("J14r the replicate term reaches psi in the REAL model", psi > 1e-9,
       "sum |dL_rep/dpsi| = %.6e" % psi)
    ok("J14r-b and gives omega nothing",
       all(g is None or float(g.abs().sum()) == 0.0 for g in omega),
       "%d flow tensors" % len(omega))

    # Two ways the term silently disconnects, both of which happened.
    model.zero_grad(set_to_none=True)
    with stop_grad_params(model.estimator):
        loss2, _ = crit(model.rsample_posterior(48, xg),
                        model.rsample_posterior(48, xgp))
    ok("J14r-c freezing the whole estimator would disconnect it",
       not loss2.requires_grad,
       "requires_grad = %s -- hence flow_parameters(), not estimator"
       % loss2.requires_grad)

    with stop_grad_params(model.flow_parameters()):
        loss3, _ = crit(model.sample_posterior(48, xg),
                        model.sample_posterior(48, xgp))
    ok("J14r-d and so would sbi's non-reparameterised sample()",
       not loss3.requires_grad,
       "requires_grad = %s -- hence rsample_posterior" % loss3.requires_grad)


def test_j9():
    theta, x = make_bank(300, seed=31)
    real_x = torch.rand(40, W)
    cls = torch.tensor(([0] * 20) + ([1] * 20))
    donor = np.repeat(np.arange(20), 2)

    def sim_draws(b_met, b_rep, n=20):
        b = ThreeStreamBatcher(theta, x, real_x, cls, donor, None,
                               BatchSpec(b_sim=16, b_met=b_met, b_rep=b_rep),
                               seed=4)
        return [b.next()["sim"][2].tolist() for _ in range(n)]

    base = sim_draws(8, 4)
    bigger_met = sim_draws(24, 4)
    no_rep = sim_draws(8, 0)
    ok("J9a resizing the metric stream leaves the sim draws identical",
       base == bigger_met, "20 consecutive batches compared, byte-identical")
    ok("J9b switching the replicate stream off does too",
       base == no_rep, "the streams share no generator")

    # And the sim term's VALUE must not depend on the other batches, which is
    # only true because the backbone has no BatchNorm. Run at dropout 0.
    model, _ = make_model(seed=41, dropout=0.0)
    model.eval()
    th, xs = theta[:16], x[:16]
    v1 = float(model.npe_loss(th, xs).mean())
    _ = model.encode(real_x)                       # another stream, same psi
    v2 = float(model.npe_loss(th, xs).mean())
    ok("J9c the sim term's value is unaffected by the other streams",
       abs(v1 - v2) < 1e-12, "%.12f vs %.12f" % (v1, v2))
    ok("J9d and the model carries no BatchNorm", assert_no_batchnorm(model),
       "asserted at construction as well")


# ---------------------------------------------------------------------------
# Integration: all three terms together for a few steps
# ---------------------------------------------------------------------------

def test_integration():
    theta, x = make_bank(400, seed=51)
    vtheta, vx = make_bank(128, seed=52)
    real_x = torch.rand(24, W)
    cls = torch.tensor(([0] * 12) + ([1] * 12))
    donor = np.repeat(np.arange(12), 2)

    model, _ = make_model(seed=61)
    batcher = ThreeStreamBatcher(theta, x, real_x, cls, donor, None,
                                 BatchSpec(b_sim=32, b_met=8, b_rep=3), seed=7)
    Sigma0 = box_prior_covariance(np.zeros(D_THETA), np.ones(D_THETA),
                                  dtype=torch.get_default_dtype())
    crit = ReplicateConsistencyLoss(Sigma0, n_draws=64, warmup=0.3)

    def dsn_loss_fn(z, c):
        """Stand-in for l_DSN: pull same-class embeddings together."""
        loss = torch.zeros((), dtype=z.dtype)
        for k in torch.unique(c):
            zk = z[c == k]
            if zk.shape[0] > 1:
                loss = loss + ((zk - zk.mean(0, keepdim=True)) ** 2).sum(-1).mean()
        return loss

    cfg = TrainConfig(epochs=3, steps_per_epoch=4, lr=1e-3, lambda_dsn=0.1,
                      lambda_rep=0.05, patience=99)
    hist = train_joint(model, batcher, cfg, val_theta=vtheta, val_x=vx,
                       dsn_loss_fn=dsn_loss_fn, rep_criterion=crit,
                       n_posterior_draws=64, log_fn=None, seed=61)

    ok("INT1 all three terms run together", len(hist) == 3,
       "%d epochs, final npe %.4f dsn %.4f rep %.4f"
       % (len(hist), hist[-1]["npe"], hist[-1]["dsn"], hist[-1]["rep"]))
    ok("INT2 the warm-up ramp advances across epochs",
       hist[0]["ramp"] < hist[-1]["ramp"] or hist[-1]["ramp"] == 1.0,
       "ramp %.2f -> %.2f" % (hist[0]["ramp"], hist[-1]["ramp"]))
    ok("INT3 rho_grad is measured and finite",
       "rho_grad" in hist[-1] and np.isfinite(hist[-1]["rho_grad"]),
       "rho_grad = %.4f" % hist[-1].get("rho_grad", float("nan")))
    ok("INT4 T and p_eff are logged per epoch",
       np.isfinite(hist[-1]["T"]) and np.isfinite(hist[-1]["p_eff"]),
       "T = %.3f, p_eff = %.3f, invalid p_eff = %d"
       % (hist[-1]["T"], hist[-1]["p_eff"], hist[-1]["n_p_eff_invalid"]))

    # The flow must still have moved -- if the stop-gradient were applied to
    # the whole step rather than only the replicate term, omega would be frozen.
    ok("INT5 omega still trains despite the replicate stop-gradient",
       any(p.requires_grad for p in model.flow_parameters()),
       "requires_grad restored after every replicate forward pass")


def main():
    print("=" * 74)
    print("Smoke test: joint model, batches and training loop (Stage 2, J1-J9)")
    print("torch %s" % torch.__version__)
    print("=" * 74)
    test_j0()
    test_j1_j2()
    test_j3()
    test_j4()
    test_j5()
    test_j6()
    test_j6b_frozen()
    test_j7()
    test_j8()
    test_j9()
    test_j14_real()
    test_integration()
    print("-" * 74)
    n_fail = RESULTS.count("FAIL")
    print("%d passed, %d failed, %d skipped"
          % (RESULTS.count("PASS"), n_fail, RESULTS.count("SKIP")))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
