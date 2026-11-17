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

    cfg : the configuration (config.Config or a run's view of it); reads
        `replicate.pairs` and `replicate.group_key` (sec. 3.1, v11)
    base : str
    Returns `base`'s items, then `--rep-pairs <replicate.pairs>
    --rep-group-key <replicate.group_key>`: one space between items, no
    leading or trailing space (the tuner splits --runner-extra on
    whitespace and appends the items to the runner's argv). A missing key
    raises KeyError (config.Config.get); a value that is not one word
    raises ValueError (the configuration's schema admits only the rule and
    key names, so this guards a hand-built configuration)."""
    vals = []
    for key in ("replicate.pairs", "replicate.group_key"):
        v = cfg.get(key)
        if not isinstance(v, str) or v.split() != [v]:
            raise ValueError("%s must be one word, got %r" % (key, v))
        vals.append(v)
    items = (base or "").split()
    items += ["--rep-pairs", vals[0], "--rep-group-key", vals[1]]
    return " ".join(items)
