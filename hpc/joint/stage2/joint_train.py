"""The explicit joint training loop, eq. (2) (plan v0.6, Stage 2).

Why an explicit loop rather than sbi's: sbi's `train()` takes one loss and no
labels, and weight decay and LR schedules are unreachable through it [REPO].
The flow itself is still built by `posterior_nn`, so nothing downstream
changes.

What the loop is responsible for, beyond the arithmetic
-------------------------------------------------------
* **Driving the warm-up.** `ReplicateConsistencyLoss.set_progress` is called
  every step with the fraction of planned training elapsed. Forgetting this
  leaves the ramp at its default of 1.0 and the replicate term fights the NPE
  term from step one, which is the failure S2.5e exists to prevent.
* **Logging every quantity in S3 per epoch**, including the ones that only
  make sense as a series: the gradient cosine rho_grad, the replicate
  statistic against its target, and the count of pairs whose p_eff was
  invalid.
* **Early stopping on the NPE validation score only.** Stopping on the total
  loss would let a shrinking replicate term buy patience for a worsening
  posterior.

Pure ASCII, LF only.
"""

import contextlib
import math

import torch


@contextlib.contextmanager
def _nullcontext():
    yield

from joint_losses import stop_grad_params

__all__ = ["TrainConfig", "gradient_cosine", "train_joint", "evaluate_npe"]


class TrainConfig(object):
    def __init__(self, epochs=20, steps_per_epoch=50, lr=5e-4,
                 weight_decay=0.0, grad_clip=5.0, lambda_dsn=0.0,
                 lambda_rep=0.0, patience=5, rho_grad_probe=True,
                 beta1=0.9, beta2=0.999):
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.lambda_dsn = float(lambda_dsn)
        self.lambda_rep = float(lambda_rep)
        self.patience = int(patience)
        self.rho_grad_probe = bool(rho_grad_probe)
        # [CHANGE Stage 4] The betas were left at AdamW's defaults, so
        # one_minus_beta1 -- a searched axis of JOINT_KNOB_ORDER -- could not
        # reach the optimiser. A search over an axis that never reaches the
        # trainer reads as "this axis does not matter" in the partial
        # dependence, which is the worst kind of null result: confidently
        # wrong. Defaults reproduce AdamW's, so nothing already run moves.
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("betas must lie in [0, 1), got (%g, %g)"
                             % (self.beta1, self.beta2))

    def to_dict(self):
        return dict(self.__dict__)


def _flat_grad(params):
    gs = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    if not gs:
        return None
    return torch.cat(gs)


def gradient_cosine(model, loss_a_fn, loss_b_fn):
    """rho_grad: cosine between the two terms' gradients w.r.t. psi.

    Computed on ONE probe batch, twice, with the graph rebuilt each time --
    the terms must not share a backward pass or the second gradient would be
    the sum of both. P4 predicts this goes negative on a non-trivial fraction
    of steps under A2.
    """
    params = list(model.encoder_parameters())

    model.zero_grad(set_to_none=True)
    loss_a_fn().backward()
    ga = _flat_grad(params)

    model.zero_grad(set_to_none=True)
    loss_b_fn().backward()
    gb = _flat_grad(params)

    model.zero_grad(set_to_none=True)
    if ga is None or gb is None:
        return float("nan")
    na, nb = ga.norm(), gb.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((ga @ gb) / (na * nb))


def evaluate_npe(model, theta, x, batch_size=512):
    """Mean held-out -log q_omega(theta | h_psi(x)), in nats/row."""
    if theta.shape[0] == 0:
        # An empty split must not score 0.0: with early stopping, a constant
        # "best" of 0.0 at epoch 0 restores the UNTRAINED weights at the end
        # and the run reports a model that never learned, silently.
        raise ValueError("evaluate_npe called on an empty split")
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, theta.shape[0], batch_size):
            t = theta[i:i + batch_size]
            xx = x[i:i + batch_size]
            l = model.npe_loss(t, xx)
            total += float(l.sum())
            n += t.shape[0]
    model.train()
    return total / max(n, 1)


def train_joint(model, batcher, cfg, val_theta=None, val_x=None,
                dsn_loss_fn=None, rep_criterion=None, n_posterior_draws=256,
                log_fn=print, seed=0):
    """Run the loop. Returns a history list, one dict per epoch.

    dsn_loss_fn : callable(z, c) -> scalar, or None. Injected rather than
        imported, so this module does not depend on the DSN's loss internals
        and can be tested without the DSN repo present.
    rep_criterion : ReplicateConsistencyLoss or None.
    """
    torch.manual_seed(int(seed))
    # Only parameters that can move. A frozen encoder (arms A0, A0s, A_ref)
    # would otherwise get AdamW state and, more importantly, weight decay --
    # which decays a frozen tensor toward zero on every step even though its
    # gradient is None, quietly changing an arm that is supposed to be held
    # fixed.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; every arm must train "
                           "something (the flow at minimum)")
    opt = torch.optim.AdamW(trainable, lr=cfg.lr,
                            weight_decay=cfg.weight_decay,
                            betas=(getattr(cfg, "beta1", 0.9),
                                   getattr(cfg, "beta2", 0.999)))
    total_steps = max(1, cfg.epochs * cfg.steps_per_epoch)
    step = 0
    best, best_epoch, best_state = math.inf, -1, None
    history = []

    for epoch in range(cfg.epochs):
        acc = {"npe": 0.0, "dsn": 0.0, "rep": 0.0, "n": 0,
               "T": [], "p_eff": [], "n_p_eff_invalid": 0}

        for _ in range(cfg.steps_per_epoch):
            batch = batcher.next()
            opt.zero_grad(set_to_none=True)
            # Accumulate from the first term rather than from a CPU zero
            # tensor, so the sum lives on whatever device the losses do.
            total = None

            if batch["sim"] is not None:
                theta, x, _ = batch["sim"]
                l_npe = model.npe_loss(theta, x).mean()
                total = l_npe if total is None else total + l_npe
                acc["npe"] += float(l_npe)

            if cfg.lambda_dsn > 0.0 and dsn_loss_fn is not None \
                    and batch["met"] is not None:
                xm, cm, _ = batch["met"]
                l_dsn = dsn_loss_fn(model.encode(xm), cm)
                total = (cfg.lambda_dsn * l_dsn if total is None
                         else total + cfg.lambda_dsn * l_dsn)
                acc["dsn"] += float(l_dsn)

            if cfg.lambda_rep > 0.0 and rep_criterion is not None \
                    and batch["rep"] is not None:
                xg, xgp, _ = batch["rep"]
                rep_criterion.set_progress(step / float(total_steps))
                # Stop-gradient on OMEGA ONLY. Not on the estimator: sbi
                # keeps the embedding net inside it, so freezing the estimator
                # would freeze psi too and disconnect the term entirely.
                with stop_grad_params(model.flow_parameters()):
                    sg = model.rsample_posterior(n_posterior_draws, xg)
                    sgp = model.rsample_posterior(n_posterior_draws, xgp)
                    l_rep, info = rep_criterion(sg, sgp)
                if not l_rep.requires_grad:
                    raise RuntimeError(
                        "the replicate term carries no gradient. Either the "
                        "sampler detached (use rsample_posterior) or psi was "
                        "frozen along with omega (pass flow_parameters(), not "
                        "the estimator). Failing loudly: a disconnected term "
                        "trains silently and looks like a weak effect.")
                total = (cfg.lambda_rep * l_rep if total is None
                         else total + cfg.lambda_rep * l_rep)
                acc["rep"] += float(l_rep)
                acc["T"].append(info["T"].mean().item())
                acc["p_eff"].append(info["p_eff"].mean().item())
                acc["n_p_eff_invalid"] += int(info["n_p_eff_invalid"])

            if total is None:
                raise RuntimeError("no active term this step: every stream "
                                   "returned None. Check BatchSpec and the "
                                   "lambdas.")
            total.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
            opt.step()
            acc["n"] += 1
            step += 1

        rec = {
            "epoch": epoch,
            "npe": acc["npe"] / max(acc["n"], 1),
            "dsn": acc["dsn"] / max(acc["n"], 1),
            "rep": acc["rep"] / max(acc["n"], 1),
            "T": (sum(acc["T"]) / len(acc["T"])) if acc["T"] else float("nan"),
            "p_eff": (sum(acc["p_eff"]) / len(acc["p_eff"]))
                     if acc["p_eff"] else float("nan"),
            "n_p_eff_invalid": acc["n_p_eff_invalid"],
            "ramp": rep_criterion.ramp if rep_criterion is not None else None,
        }

        if val_theta is not None and val_x is not None:
            rec["val_npe"] = evaluate_npe(model, val_theta, val_x)
            if rec["val_npe"] < best - 1e-6:
                best, best_epoch = rec["val_npe"], epoch
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}

        if cfg.rho_grad_probe and cfg.lambda_dsn > 0.0 \
                and dsn_loss_fn is not None:
            probe = batcher.next()
            if probe["sim"] is not None and probe["met"] is not None:
                th, xs, _ = probe["sim"]
                xm, cm, _ = probe["met"]
                # The probe must not advance the DSN separation ramp: it is a
                # measurement, not a training step. See
                # DSNLossAdapter.frozen_warmup.
                freeze = getattr(dsn_loss_fn, "frozen_warmup", None)
                ctx = freeze() if freeze is not None else _nullcontext()
                with ctx:
                    rec["rho_grad"] = gradient_cosine(
                        model,
                        lambda: model.npe_loss(th, xs).mean(),
                        lambda: dsn_loss_fn(model.encode(xm), cm))
        ws = getattr(dsn_loss_fn, "warmup_state", None)
        if ws is not None:
            rec["sep_warmup"] = ws()

        history.append(rec)
        if log_fn is not None:
            log_fn("epoch %2d | npe %.4f | dsn %.4f | rep %.4f | T %.3f vs "
                   "p_eff %.3f | ramp %s | val %s"
                   % (rec["epoch"], rec["npe"], rec["dsn"], rec["rep"],
                      rec["T"], rec["p_eff"], rec["ramp"],
                      ("%.4f" % rec["val_npe"]) if "val_npe" in rec else "-"))

        if val_theta is not None and cfg.patience > 0 \
                and epoch - best_epoch >= cfg.patience:
            if log_fn is not None:
                log_fn("early stop: no NPE validation improvement for %d epochs"
                       % cfg.patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
