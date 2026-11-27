"""Block 41: a member rebuilt from a bundle, with the pipeline's own
builders and no data (specs/SPEC.md sec. 2.7, "Rebuilding a member from a
bundle", and block 41).

CONTRACT (wave B): the signatures below are fixed.

Pure ASCII, LF only.
"""


def build(architecture, p, W):
    """An untrained member in eval mode: the backbone by the runner's
    builders from `architecture` (schema.ARCHITECTURE_KEYS), the flow by
    joint_model.build_joint_model on the box prior on (0, 1)^p, with example
    tensors of the shapes the runner used ((64, p) and (64, W)) drawn from a
    fixed generator."""
    raise NotImplementedError("block 41")


def load(bundle):
    """[model_0, ..., model_{M-1}] of a bundle.Bundle: each built, its state
    loaded strictly, then verified: joint_model.state_sha256 of the loaded
    state equals the manifest's `state_sha256`, and the backbone's repr the
    manifest's `backbone_repr`; bundle.BundleError otherwise."""
    raise NotImplementedError("block 41")
