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
import os

import torch


@contextlib.contextmanager
def _nullcontext():
    yield

from joint_losses import stop_grad_params

__all__ = ["TrainConfig", "gradient_cosine", "train_joint", "evaluate_npe",
           "evaluate_npe_shards", "CKPT_FORMAT", "load_train_checkpoint"]

CKPT_FORMAT = "joint_train_ckpt/1"


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


def evaluate_npe_shards(model, index, row_mask, batch_size=512):
    """`evaluate_npe` over the masked rows of a ShardIndex with at most two
    shards' windows in memory (block 5, D-084). The batches are the SAME
    `batch_size` runs of consecutive masked rows that `evaluate_npe` forms
    on the concatenated split (`ShardIndex.iter_batches`), so the float32
    batch sums and the float64 total agree with it bit for bit."""
    import numpy as np
    row_mask = np.asarray(row_mask, dtype=bool).ravel()
    if not row_mask.any():
        raise ValueError("evaluate_npe_shards called on an empty split")
    model.eval()
    total, n = 0.0, 0
    theta_all = np.asarray(index.theta)
    with torch.no_grad():
        for rows, x in index.iter_batches(row_mask, batch_size):
            t = torch.as_tensor(theta_all[rows], dtype=torch.float32)
            xx = torch.as_tensor(x, dtype=torch.float32)
            l = model.npe_loss(t, xx)
            total += float(l.sum())
            n += int(t.shape[0])
    model.train()
    return total / max(n, 1)


def _save_checkpoint(path, payload):
    """[D-111] Durable: the temporary file is flushed and fsynced before its
    rename, and the folder after it (best effort), so a node that goes down
    leaves the old or the new checkpoint, not an empty one."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def _set_aside_checkpoint(path):
    """[D-111] `<path>.corrupt<k>`, the first free k: a checkpoint that could
    not be loaded is kept, never deleted (as the pipeline's D-097)."""
    k = 1
    while os.path.exists("%s.corrupt%d" % (path, k)):
        k += 1
    dst = "%s.corrupt%d" % (path, k)
    os.replace(path, dst)
    return dst


def load_train_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise ValueError("%s: unknown training checkpoint format %r"
                         % (path, ck.get("format")))
    return ck


def train_joint(model, batcher, cfg, val_theta=None, val_x=None,
                dsn_loss_fn=None, rep_criterion=None, n_posterior_draws=256,
                log_fn=print, seed=0, val_eval=None, ckpt_path=None,
                resume=False, stop_after_epoch=None):
    """Run the loop. Returns a history list, one dict per epoch.

    dsn_loss_fn : callable(z, c) -> scalar, or None. Injected rather than
        imported, so this module does not depend on the DSN's loss internals
        and can be tested without the DSN repo present.
    rep_criterion : ReplicateConsistencyLoss or None.
    val_eval : callable() -> float, or None [block 5]. When given it replaces
        evaluate_npe(model, val_theta, val_x) as the selection-split NLL
        (the streaming runner evaluates shard by shard).
    ckpt_path : str or None [block 6]. When given, a checkpoint is written
        at the end of every epoch (model, optimizer, epoch, history, the
        early-stopping state, the torch RNG state, the batcher's generators
        and stream), atomically. With `resume` and the file present, the
        loop restores all of it and continues from the next epoch; absent,
        it starts fresh. `stop_after_epoch` ends the loop after that epoch's
        checkpoint by raising StoppedAfterEpoch (carrying the partial
        history), so a caller cannot mistake a partial history for a result.
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
    first_epoch = 0

    # [block 6] resume from the per-epoch checkpoint when asked and present.
    # [D-111] the pipeline asks for --resume on every job, so a checkpoint that
    # cannot be loaded (torn by a node that went down) starts the run fresh
    # instead of failing it; an OSError is the file system's (a node fault).
    ck = None
    if ckpt_path and resume and os.path.isfile(ckpt_path):
        try:
            ck = load_train_checkpoint(ckpt_path)
        except OSError:
            raise
        except Exception as exc:
            dst = _set_aside_checkpoint(ckpt_path)
            if log_fn is not None:
                log_fn("checkpoint %s could not be loaded (%s: %s); moved aside as %s; starting "
                       "from epoch 0" % (os.path.basename(ckpt_path), type(exc).__name__,
                                         str(exc)[:200], os.path.basename(dst)))
            ck = None
    if ck is not None:
        if ck["config"] != cfg.to_dict():
            raise ValueError("%s was written under another TrainConfig: %r vs %r"
                             % (ckpt_path, ck["config"], cfg.to_dict()))
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        history = list(ck["history"])
        step = int(ck["step"])
        best, best_epoch = float(ck["best"]), int(ck["best_epoch"])
        best_state = ck["best_state"]
        torch.set_rng_state(ck["torch_rng"])
        if ck.get("batcher_state") is not None and hasattr(batcher, "load_state_dict"):
            batcher.load_state_dict(ck["batcher_state"])
        first_epoch = int(ck["epoch"]) + 1
        if log_fn is not None:
            log_fn("resumed from %s: %d epoch(s) done, best val %s at epoch %d"
                   % (ckpt_path, first_epoch, ("%.4f" % best) if best < math.inf else "-",
                      best_epoch))
        # The stop test of the checkpointed epoch, re-applied: a run killed
        # between that epoch's checkpoint and its early stop must not train
        # one epoch more than the uninterrupted run.
        has_val0 = val_eval is not None or val_theta is not None
        if has_val0 and cfg.patience > 0 and (first_epoch - 1) - best_epoch >= cfg.patience:
            if log_fn is not None:
                log_fn("early stop: no NPE validation improvement for %d epochs"
                       % cfg.patience)
            first_epoch = cfg.epochs

    for epoch in range(first_epoch, cfg.epochs):
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

        if val_eval is not None or (val_theta is not None and val_x is not None):
            rec["val_npe"] = (val_eval() if val_eval is not None
                              else evaluate_npe(model, val_theta, val_x))
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

        # [block 6] the per-epoch checkpoint, before the stop test, so a run
        # that stops here resumes to the same decision.
        if ckpt_path:
            _save_checkpoint(ckpt_path, {
                "format": CKPT_FORMAT, "epoch": epoch, "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "history": list(history), "best": float(best),
                "best_epoch": int(best_epoch), "best_state": best_state,
                "torch_rng": torch.get_rng_state(),
                "batcher_state": (batcher.state_dict()
                                  if hasattr(batcher, "state_dict") else None),
                "config": cfg.to_dict()})

        has_val = val_eval is not None or val_theta is not None
        if has_val and cfg.patience > 0 and epoch - best_epoch >= cfg.patience:
            if log_fn is not None:
                log_fn("early stop: no NPE validation improvement for %d epochs"
                       % cfg.patience)
            break

        if stop_after_epoch is not None and epoch >= int(stop_after_epoch):
            if log_fn is not None:
                log_fn("stopped after epoch %d (checkpoint %s); resume to continue"
                       % (epoch, ckpt_path))
            raise StoppedAfterEpoch(epoch, history)

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


class StoppedAfterEpoch(Exception):
    """train_joint ended at `stop_after_epoch`; `history` is partial."""

    def __init__(self, epoch, history):
        Exception.__init__(self, "stopped after epoch %d" % epoch)
        self.epoch = int(epoch)
        self.history = list(history)
