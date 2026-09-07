"""JointDSNNPE: the DSN encoder and the sbi flow as one trainable model.

Plan v0.6, Stage 2. The flow is still built by `posterior_nn`, so posterior
objects, gates and ensembles downstream are untouched; only the training loop
is ours (sbi's loop cannot take a second or third loss with labels).

A gotcha found while writing this, which belongs in Stage 0
------------------------------------------------------------
`npe_model.py` builds the estimator with
`z_score_theta="transform_to_unconstrained"` [REPO]. Outside sbi's `NPE`
class that setting FAILS unless the prior is threaded through, because the
transform needs to know the support:

    ValueError: Transformation to unconstrained space requires a distribution
                provided through `x_dist`.

(sbi names the flow's own variable `x` internally, so `x_dist` is the prior
over THETA -- the naming is confusing and the error message does not say so.)
`build_joint_model` therefore requires `prior` and passes it as `x_dist`.
Silently falling back to `z_score_theta="none"` would change the parameter
space the flow works in, which is not a defensible default and would make the
joint arms incomparable with A0.

Pure ASCII, LF only.
"""

import hashlib
import io
import json

import torch
import torch.nn as nn

from sbi.neural_nets import posterior_nn

__all__ = ["JointDSNNPE", "build_joint_model", "assert_no_batchnorm",
           "state_sha256"]


def assert_no_batchnorm(module):
    """Fail loudly if any BatchNorm is present.

    Not pedantry: with BatchNorm the three streams of eq. (2) would also mix
    through running statistics -- a stateful channel invisible to any analysis
    of psi, and one that would make J9 false without anything in the code
    saying so.
    """
    bad = [name for name, m in module.named_modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.SyncBatchNorm))]
    if bad:
        raise RuntimeError("BatchNorm modules present, which breaks the "
                           "stream independence of eq. (2): %r" % (bad,))
    return True


def state_sha256(state_dict):
    """Digest of a state_dict, for the frozen-artifact convention."""
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in state_dict.items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


class JointDSNNPE(nn.Module):
    """Encoder + flow. The encoder lives INSIDE the sbi estimator.

    Registering the backbone separately would double-count its parameters in
    `parameters()` and hand the optimiser two copies of the same tensors, so
    `encoder` is a property that reaches into the estimator rather than a
    child module.
    """

    def __init__(self, estimator, meta=None):
        super().__init__()
        self.estimator = estimator
        self.meta = dict(meta or {})
        self._encoder_frozen = False
        assert_no_batchnorm(self)

    def freeze_encoder(self):
        """Freeze psi AND pin the encoder to eval mode.

        [CORRECTION] Freezing the weights alone is not enough. `evaluate_npe`
        and the diagnostics call `model.train()` when they finish, which
        re-enables dropout inside the encoder, so a "frozen" A0 encoder was
        producing a STOCHASTIC z during flow training and a deterministic one
        at scoring time. The plan's A0 is "NPE on sim at frozen z", singular.
        Overriding `train()` keeps the encoder in eval for as long as it is
        frozen, whatever the rest of the model does.
        """
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._encoder_frozen = True
        self.encoder.eval()
        return self

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        self._encoder_frozen = False
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    # -- access --------------------------------------------------------------

    @property
    def encoder(self):
        """The embedding net, i.e. the DSN backbone h_psi."""
        return self.estimator._embedding_net

    def encoder_parameters(self):
        return self.encoder.parameters()

    def flow_parameters(self):
        """Everything that is not the encoder: omega."""
        enc = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in enc]

    # -- forward paths -------------------------------------------------------

    def encode(self, x):
        """z = h_psi(x)."""
        return self.encoder(x)

    def npe_loss(self, theta, x):
        """Per-row -log q_omega(theta | h_psi(x)). Shape (B,)."""
        return self.estimator.loss(theta, condition=x)

    def sample_posterior(self, n_draws, x):
        """(B, S, d) draws. NOT differentiable -- for diagnostics only.

        sbi returns (S, B, d); the transpose is done here once rather than at
        every call site, where it would eventually be forgotten.
        """
        s = self.estimator.sample((int(n_draws),), condition=x)
        return s.transpose(0, 1)

    def rsample_posterior(self, n_draws, x):
        """(B, S, d) REPARAMETERISED draws. Use this for the replicate term.

        [CORRECTION, found in Stage 3] `estimator.sample` calls the zuko
        distribution's `.sample()`, which detaches. Any loss built on it has
        no gradient path to psi at all -- the replicate term was silently
        disconnected, and the symptom was arm A5 scoring identically to A1
        rather than an error. The reparameterised `.rsample()` is what makes
        d(E[theta | z])/dpsi exist, which is the entire mechanism of S2.5e
        route 1 (improve agreement by moving z, not by flattening the flow).

        This reaches past sbi's public API into `_embedding_net` and `net`
        because sbi exposes no reparameterised sampler. Flagged rather than
        hidden: it is a coupling to sbi 0.27's internals and will need
        rechecking on an upgrade. Smoke test J14r is the check.
        """
        emb = self.estimator._embedding_net(x)
        dists = self.estimator.net(emb)
        if not hasattr(dists, "rsample"):
            raise RuntimeError(
                "the flow's distribution has no rsample(); the replicate term "
                "cannot be trained through this estimator.")
        s = dists.rsample((int(n_draws),))
        return s.transpose(0, 1)

    # -- checkpointing -------------------------------------------------------

    def checkpoint(self):
        """A dict carrying the encoder separately, so that
        `dsn_frozen.load_frozen_dsn` can still load the encoder ALONE."""
        enc_state = self.encoder.state_dict()
        return {
            "format": "joint_dsn_npe/1",
            "meta": self.meta,
            "encoder_state": enc_state,
            "encoder_sha256": state_sha256(enc_state),
            "model_state": self.state_dict(),
            "model_sha256": state_sha256(self.state_dict()),
        }

    def load_checkpoint(self, ckpt, strict=True):
        if ckpt.get("format") != "joint_dsn_npe/1":
            raise ValueError("unknown checkpoint format %r" % ckpt.get("format"))
        self.load_state_dict(ckpt["model_state"], strict=strict)
        got = state_sha256(self.state_dict())
        if got != ckpt["model_sha256"]:
            raise RuntimeError("state digest mismatch after load: %s vs %s"
                               % (got[:16], ckpt["model_sha256"][:16]))
        self.meta = dict(ckpt.get("meta", {}))
        return self


def build_joint_model(backbone, prior, theta_example, x_example,
                      hidden_features=64, num_transforms=5, num_bins=10,
                      z_score_theta="transform_to_unconstrained",
                      z_score_x="none", meta=None):
    """Build the estimator around an already-constructed backbone.

    `backbone` is passed in rather than built here so that the DSN's
    `build_backbone(BackboneConfig(...))` remains the single place the encoder
    architecture is defined (the module is imported, never reimplemented).

    `prior` is REQUIRED whenever z_score_theta transforms to the unconstrained
    space -- see the module docstring.
    """
    assert_no_batchnorm(backbone)
    kwargs = dict(model="zuko_nsf", embedding_net=backbone,
                  z_score_theta=z_score_theta, z_score_x=z_score_x,
                  hidden_features=hidden_features,
                  num_transforms=num_transforms, num_bins=num_bins)
    if z_score_theta == "transform_to_unconstrained":
        if prior is None:
            raise ValueError(
                "z_score_theta='transform_to_unconstrained' needs the prior; "
                "pass it, or choose a different z_score_theta deliberately.")
        kwargs["x_dist"] = prior          # sbi's internal name for theta's dist
    estimator = posterior_nn(**kwargs)(theta_example, x_example)

    meta = dict(meta or {})
    meta.setdefault("z_score_theta", z_score_theta)
    meta.setdefault("z_score_x", z_score_x)
    meta.setdefault("flow", {"hidden_features": hidden_features,
                             "num_transforms": num_transforms,
                             "num_bins": num_bins})
    meta.setdefault("backbone_repr", repr(getattr(backbone, "cfg", None)))
    return JointDSNNPE(estimator, meta=meta)


def meta_json(model):
    return json.dumps(model.meta, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
