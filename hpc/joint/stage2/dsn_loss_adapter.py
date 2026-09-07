"""The real l_DSN, adapted to the joint loop's (z, c) -> scalar signature.

Plan v0.6, Stage 2. `joint_train.train_joint` takes `dsn_loss_fn` as an
INJECTED callable so the loop can be tested without the DSN repo present. This
module supplies the real one.

Nothing about the loss is reimplemented. `build_loss_and_miner` from the DSN's
own `train.py` is called, which means three things are inherited rather than
re-derived, and all three are easy to get wrong:

  1. **The margin conversion.** The config states the margin in COSINE
     distance; `CompositeDSNLoss` works in SQUARED EUCLIDEAN, so the DSN
     converts once with ||u - v||^2 = 2 d_cos(u, v). The MINER keeps the cosine
     margin, because it scores under CosineSimilarity. Re-deriving this here
     would eventually put the factor 2 in one place and not the other.
  2. **The distance family.** `distances.CosineSimilarity()` must be passed
     EXPLICITLY to the miner as well as to the loss -- the miner's default is
     LpDistance, so omitting it would mine triplets in one geometry and score
     them in another, with no error anywhere.
  3. **The loss_type dispatch**, including which knobs are inert on the current
     path (the silhouette gate) and which raise.

The one thing this module owns is the WARM-UP HORIZON. `SepWarmup` needs
`total_steps`, and in the joint loop that is `epochs * steps_per_epoch`, not
the DSN's own schedule. Passing None with a nonzero `sep_warmup_frac` is not an
error but is not a warm-up either; `build_dsn_loss` therefore requires
total_steps whenever the fraction is nonzero, rather than letting it degrade
into a constant weight.

Pure ASCII, LF only.
"""

import contextlib
import os
import sys

__all__ = ["DSNLossConfig", "build_dsn_loss", "load_dsn_train_module"]


class DSNLossConfig(object):
    """The fields `build_loss_and_miner` reads, and nothing else.

    A plain object rather than the DSN's `TrainConfig`, so the joint stack does
    not inherit the whole training configuration surface just to build one
    loss. Field names match exactly, because the DSN reads them by name.
    """

    def __init__(self, loss_type="joint_sep", margin=0.2, swap=True,
                 angular_alpha_deg=18.0, strict_semihard=True,
                 lambda_sep=0.1, sep_warmup_frac=0.0,
                 mining_strategy="easy_pos_semihard_neg",
                 sep_centre_means=None, sep_gate_threshold=None):
        self.loss_type = str(loss_type)
        self.margin = float(margin)
        self.swap = bool(swap)
        self.angular_alpha_deg = float(angular_alpha_deg)
        self.strict_semihard = bool(strict_semihard)
        self.lambda_sep = float(lambda_sep)
        self.sep_warmup_frac = float(sep_warmup_frac)
        self.mining_strategy = str(mining_strategy)
        self.sep_centre_means = sep_centre_means
        self.sep_gate_threshold = sep_gate_threshold

    def to_dict(self):
        return dict(self.__dict__)


def load_dsn_train_module(dsn_main_dir=None):
    """Import the DSN's train.py through DSN_MAIN_DIR."""
    dsn_main_dir = dsn_main_dir or os.environ.get("DSN_MAIN_DIR")
    if not dsn_main_dir:
        raise RuntimeError(
            "DSN_MAIN_DIR is not set. Point it at the DSN repo's Main/ "
            "directory (use the dsn_main symlink -- the real path contains a "
            "space).")
    if not os.path.isfile(os.path.join(dsn_main_dir, "train.py")):
        raise RuntimeError("no train.py under %r" % dsn_main_dir)
    if dsn_main_dir not in sys.path:
        sys.path.insert(0, dsn_main_dir)
    import train as dsn_train
    return dsn_train


class DSNLossAdapter(object):
    """Callable (z, c) -> scalar, mining the triplets itself.

    Kept as a plain object rather than an nn.Module: `CompositeDSNLoss` has no
    trainable parameters of its own (SepWarmup holds a step counter, which is a
    buffer), and registering it as a child of the joint model would put a
    non-parameter counter into the model's state_dict and hence into the
    checkpoint digest, making two runs that differ only in step count look like
    different models.
    """

    def __init__(self, loss_fn, miner, cfg):
        self.loss_fn = loss_fn
        self.miner = miner
        self.cfg = cfg

    def __call__(self, z, c):
        mined = self.miner(z, c)
        return self.loss_fn(z, c, mined)

    def stats(self):
        return (self.loss_fn.stats() if hasattr(self.loss_fn, "stats")
                else {})

    @contextlib.contextmanager
    def frozen_warmup(self):
        """Restore the separation ramp to its entry state on exit.

        NOTE the exact semantics, which are easy to misread from the name: the
        counter still advances DURING the block and is rewound at the end. Two
        calls inside one block therefore see CONSECUTIVE ramp factors. To
        compare two evaluations at the same ramp factor, use two separate
        blocks (smoke test J3b does).


        `SepWarmup.forward` advances its own counter on every call, so the ramp
        is driven by how many times l_DSN is EVALUATED, not by how many
        optimiser steps have run. Those coincide in the DSN's own loop but not
        in ours: the per-epoch rho_grad probe (P4) evaluates l_DSN one extra
        time per epoch, which would advance the ramp faster than the schedule
        says and make `sep_lambda_t` disagree with the epoch count in the
        history. The drift is small, silent, and would only ever be noticed by
        someone auditing the ramp after the fact -- so the probe wraps its call
        in this.

        `SepWarmup.step` is a BUFFER holding the counter, not a method; there
        is no public way to rewind it, hence the snapshot.
        """
        w = getattr(self.loss_fn, "warmup", None)
        if w is None:
            yield self
            return
        saved_step = w.step.detach().clone()
        saved_scale = w.scale.detach().clone()
        try:
            yield self
        finally:
            w.step.copy_(saved_step)
            w.scale.copy_(saved_scale)

    def warmup_state(self):
        """(step, scale) of the separation ramp, for the per-epoch log."""
        w = getattr(self.loss_fn, "warmup", None)
        if w is None:
            return None
        return int(w.step), float(w.scale)


def build_dsn_loss(n_classes, total_steps=None, cfg=None, dsn_main_dir=None):
    """Build the real l_DSN plus its miner.

    Parameters
    ----------
    n_classes : int
        C. Required for 'joint_sep': the separation target is -1/(C-1).
    total_steps : int or None
        The warm-up horizon, epochs * steps_per_epoch. REQUIRED whenever
        cfg.sep_warmup_frac > 0 -- see the module docstring.
    """
    cfg = cfg or DSNLossConfig()
    if cfg.sep_warmup_frac > 0.0 and not total_steps:
        raise ValueError(
            "sep_warmup_frac = %.3f needs total_steps (epochs * "
            "steps_per_epoch); a ramp without a horizon is not well posed."
            % cfg.sep_warmup_frac)
    dsn_train = load_dsn_train_module(dsn_main_dir)
    loss_fn, miner = dsn_train.build_loss_and_miner(
        cfg, n_classes=n_classes, total_steps=total_steps)
    return DSNLossAdapter(loss_fn, miner, cfg)
