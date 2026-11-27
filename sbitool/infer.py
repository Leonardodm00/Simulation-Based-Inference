"""Block 45: inference from a bundle (specs/SPEC.md sec. 2.7, "An
inference", and block 45 with its "Contract of wave C").

CONTRACT (wave C): the names, signatures, ARRAYS and JSON_KEYS below are
fixed by specs/SPEC.md; the bodies are block 45's. The numbers are the
evaluation's own (blocks 35 and 36), through sbitool._sources: this module
holds no estimator of its own.

Pure ASCII, LF only.
"""

import sys

from sbitool import _sources  # noqa: F401  (the evaluation's modules come through it)

POSTERIOR_SCHEMA = "sbitool_posterior/1"
EXIT_ARGS, EXIT_REFUSE = 2, 3

# posterior.npz: name -> (shape, dtype). n windows, n_post draws per
# window, p parameters, M members, d_z the embedding's size. Beside these:
# `rows` (n,) int64 = arange(n), and the ids given to run(), as given.
ARRAYS = {
    "draws": ("(n, n_post, p)", "float32"),          # block 35's coordinate, bounds not applied
    "draw_member": ("(n_post,)", "int64"),
    "logq_draws": ("(n, n_post)", "float32"),        # log q_E at each draw (nats)
    "logq_draws_own": ("(n, n_post)", "float32"),    # log q_m of the member that drew it (nats)
    "z": ("(M, n, d_z)", "float32"),
    "mhat": ("(n, p)", "float32"), "sigma": ("(n, p)", "float32"),
    "q05": ("(n, p)", "float32"), "q50": ("(n, p)", "float32"), "q95": ("(n, p)", "float32"),
    "contraction": ("(n, p)", "float32"),
    "gate_t": ("(M, n)", "float64"), "gate_r": ("(M, n)", "float64"),
    "gate_p_member": ("(M, n)", "float64"),
    "gate_T": ("(n,)", "float64"), "gate_p": ("(n,)", "float64"), "gate_flag": ("(n,)", "bool"),
    "mix_mean": ("(p,)", "float64"), "mix_sd": ("(p,)", "float64"),
    "mix_q05": ("(p,)", "float64"), "mix_q50": ("(p,)", "float64"), "mix_q95": ("(p,)", "float64"),
}
# posterior.json: exactly these keys
JSON_KEYS = ("schema", "bundle", "n", "n_post", "M", "p", "seed", "batch_rows", "parameters",
             "gate", "ids", "mixture_note", "software")
JSON_GATE_KEYS = ("alpha", "k_gate", "n_null", "size_mean", "size_lo", "size_hi", "level",
                  "n_flagged", "share_flagged")
NPZ_NAME, JSON_NAME = "posterior.npz", "posterior.json"


class InferRefused(ValueError):
    """An inference that is not made; the message says why."""


def run(bundle, windows, ids=None, n_post=None, seed=None, batch=None, log=print):
    """One inference (SPEC block 45, "Contract of wave C").

    bundle   a sbitool.bundle.Bundle
    windows  (n, W) float32, finite, n >= 1, W the manifest's
    ids      None, or {name: (n,) array}, written beside the arrays as given
    n_post, seed, batch
             None: the bundle's sample.n_post, sample.seed, sample.batch_rows.
             n_post a multiple of M; member k draws from seed + k.

    Returns {"arrays": {name: numpy array}, "json": dict}: ARRAYS, `rows`
    and the ids; JSON_KEYS. InferRefused for windows of another shape or
    dtype or with a non-finite value, n = 0, n_post not a positive multiple
    of M, ids of another length or under a name of ARRAYS or `rows`."""
    raise NotImplementedError("block 45: infer.run")


def write(result, out_dir):
    """Writes run()'s result as <out_dir>/posterior.npz (ev_sample.write_npz)
    and <out_dir>/posterior.json (sorted keys, ASCII), each through a
    temporary name; creates the folder. Returns (npz path, json path)."""
    raise NotImplementedError("block 45: infer.write")


def main(argv=None):
    """`python3 -m sbitool infer --bundle FILE (--windows windows.npz |
    --recording PATH --adapter NAME [--option KEY=VALUE ...]) --out DIR
    [--n-post N] [--seed S]`: 0 with `INFER OK: <n> window(s), <k> flagged
    by the gate`; EXIT_ARGS for the arguments (--option detection
    included); EXIT_REFUSE with one `REFUSED:` line, nothing written."""
    raise NotImplementedError("block 45: infer.main")


if __name__ == "__main__":
    sys.exit(main())
