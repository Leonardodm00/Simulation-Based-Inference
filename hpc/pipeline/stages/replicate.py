"""The replicate pair rule's runner flags (block 31; D-148 (3), D-146, D-147;
specs/SPEC.md block 31).

    runner_extra(cfg, base="")   the tuner's --runner-extra text: `base`
                                 followed by the flags of `replicate.*`

The pipeline passes the rule of the configuration's `replicate` block to
every runner it starts -- each trial (the tuner's `evaluate`), each budget
probe and each best-fit member (the tuner's `argv`) -- as run_joint_arms.py
flags appended through the tuner's --runner-extra:

    --rep-pairs <replicate.pairs> --rep-group-key <replicate.group_key>

Pure ASCII, LF only; stdlib only; Python 3.6 syntax.
"""

__all__ = ["runner_extra"]


def runner_extra(cfg, base=""):
    """`base` (a string of runner flags, possibly empty) followed by the
    replicate flags, whitespace-separated.

    Contract placeholder (2026-11-17): returns `base` unchanged until block
    31 lands, so that the runner, which does not know the flags before block
    31, is called as before; block 31 replaces this body."""
    return base
