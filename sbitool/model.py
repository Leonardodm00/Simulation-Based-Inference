"""Block 41: a member rebuilt from a bundle, with the pipeline's own
builders and no data (specs/SPEC.md sec. 2.7, "Rebuilding a member from a
bundle", and block 41).

CONTRACT (wave B): the signatures below are fixed.

The builders are the pipeline's, imported through sbitool._sources and never
copied: the backbone by run_joint_arms.make_backbone (or its
FixedStatsSummary), the flow by joint_model.build_joint_model on the box
prior BoxUniform on (0, 1)^p. The builders read example tensors for their
shapes only (the runner passed its first 64 training rows); here they are
drawn from a torch.Generator of a fixed seed, never from the global RNG, and
that they change nothing is shown by test T41a (the state's digest, the
densities, the draws and the embeddings equal the pipeline's rebuild).

This module never calls torch.load, pickle or numpy.load: a state comes from
the bundle's safetensors files (bundle.Bundle.state).

Pure ASCII, LF only.
"""

import os

from sbitool import _sources

N_EXAMPLE = 64            # rows of the example tensors: the runner's (ev_models.rebuild)
EXAMPLE_SEED = 0          # the fixed generator of the example tensors
BACKBONE_SEED = 0         # make_backbone's seed: it seeds only the initial weights,
                          # which the loaded state replaces
# make_backbone builds the DSN backbone when the folder it is given holds
# backbone.py, and its own fallback otherwise: for a "fallback" member it is
# given this package's folder, which holds no backbone.py
_NO_DSN_DIR = os.path.dirname(os.path.abspath(__file__))
_ENCODER_KEYS = ("depth_exponent", "width_multiplier", "block_family", "head_fusion", "dropout")


def _builders():
    """(torch, run_joint_arms, joint_model, BoxUniform), the sources imported
    through _sources.use("model")."""
    _sources.use("model")
    import torch
    import run_joint_arms
    import joint_model
    from sbi.utils import BoxUniform
    return torch, run_joint_arms, joint_model, BoxUniform


def _error(msg):
    from sbitool.bundle import BundleError
    return BundleError(msg)


def build(architecture, p, W):
    """An untrained member in eval mode: the backbone by the runner's
    builders from `architecture` (schema.ARCHITECTURE_KEYS), the flow by
    joint_model.build_joint_model on the box prior on (0, 1)^p, with example
    tensors of the shapes the runner used ((64, p) and (64, W)) drawn from a
    fixed generator."""
    torch, R, J, BoxUniform = _builders()
    a = architecture
    p, W = int(p), int(W)
    kind = a["backbone"]
    if kind == "fixed-stats":
        backbone = R.FixedStatsSummary()
    elif kind in ("dsn", "fallback"):
        if kind == "dsn":
            # the in-repo DSN tree (hpc/dsn), as the runner resolved it with
            # dsn_main_dir null; the encoder axes and head_pool_ops as the runner passed them
            dsn_dir, hpo = None, int(a["head_pool_ops"])
            enc = dict((k, a[k]) for k in _ENCODER_KEYS)
        else:
            # the fallback ignores the encoder axes and head_pool_ops (make_backbone's
            # docstring); they are not passed, so that it prints no warning
            dsn_dir, enc, hpo = _NO_DSN_DIR, None, 0
        backbone, got = R.make_backbone(W, int(a["embedding_size"]), dsn_dir, BACKBONE_SEED,
                                        encoder=enc, head_pool_ops=hpo)
        if got != kind:
            raise _error("the backbone built is %r, the bundle's is %r (is the DSN tree hpc/dsn "
                         "complete?)" % (got, kind))
    else:
        raise _error("architecture.backbone %r is none of dsn, fixed-stats, fallback" % (kind,))
    gen = torch.Generator().manual_seed(EXAMPLE_SEED)
    theta_ex = torch.rand((N_EXAMPLE, p), generator=gen, dtype=torch.float32)
    x_ex = torch.rand((N_EXAMPLE, W), generator=gen, dtype=torch.float32)
    prior = BoxUniform(low=torch.zeros(p), high=torch.ones(p))
    model = J.build_joint_model(backbone, prior, theta_ex, x_ex,
                                hidden_features=int(a["hidden_features"]),
                                num_transforms=int(a["num_transforms"]),
                                num_bins=int(a["num_bins"]),
                                z_score_theta=a["z_score_theta"], z_score_x=a["z_score_x"],
                                flow_hidden_layers=a["hidden_layers"])
    model.eval()
    return model


def load(bundle):
    """[model_0, ..., model_{M-1}] of a bundle.Bundle: each built, its state
    loaded strictly, then verified: joint_model.state_sha256 of the loaded
    state equals the manifest's `state_sha256`, and the backbone's repr the
    manifest's `backbone_repr`; bundle.BundleError otherwise."""
    _torch, _R, J, _B = _builders()
    man = bundle.manifest
    arch = man["architecture"]
    models = []
    for k, mem in enumerate(man["members"]):
        model = build(arch, man["p"], man["W"])
        got_repr = model.meta.get("backbone_repr")
        if got_repr != arch["backbone_repr"]:
            raise _error("member %d (%s): the backbone built is %r, the manifest's "
                         "backbone_repr is %r" % (k, mem["file"], got_repr,
                                                  arch["backbone_repr"]))
        state = bundle.state(k)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise _error("member %d (%s): its state does not fit the network built from the "
                         "manifest's architecture: %s" % (k, mem["file"], exc))
        got = J.state_sha256(model.state_dict())
        if got != mem["state_sha256"]:
            raise _error("member %d (%s): the loaded state's digest %s is not the manifest's "
                         "state_sha256 %s" % (k, mem["file"], got, mem["state_sha256"]))
        model.eval()
        models.append(model)
    return models
