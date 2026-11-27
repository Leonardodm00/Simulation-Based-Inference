"""Block 41: a model bundle as a file -- write, read, check (specs/SPEC.md
sec. 2.7, "The model bundle", and block 41; D-174, D-175).

A bundle is one zip: manifest.json, CARD.md, members/member_<k>.safetensors,
evaluation/sim.json, evaluation/gate.safetensors. Reading one never calls
torch.load, pickle or numpy.load.

CONTRACT (wave B): the names and signatures below are fixed.

Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE = 2, 3
EXTENSION = ".sbibundle"


class BundleError(ValueError):
    """A file is not a valid bundle; the message lists every problem."""


class Bundle:
    """An open bundle. Attributes: `path`; `manifest` (dict, validated);
    `card` (str, CARD.md's text); `sha256` (of the zip file). Methods:
    `state(k) -> {name: torch.Tensor}` member k's state, its file's sha256
    verified when read; `gate_sets() -> {"z_ref", "z_rank", "z_null":
    numpy float32 arrays (M, N, d_z)}`; `sim_metrics() -> dict`
    (evaluation/sim.json)."""

    def __init__(self, path):
        raise NotImplementedError("block 41")


def write(path, manifest, card_text, states, gate_sets, sim_json_bytes):
    """Writes a bundle to `path` (a temporary name, re-read and checked,
    then renamed). `manifest` complete except `members[k]["sha256"]` and
    `files`, which this function fills; `states` a list of M state dicts;
    `gate_sets` the three arrays; `sim_json_bytes` the metrics' file as
    bytes. Returns the final manifest. BundleError when the result does not
    pass check()."""
    raise NotImplementedError("block 41")


def read(path):
    """The Bundle at `path`; BundleError listing the problems when
    check(path) finds any."""
    raise NotImplementedError("block 41")


def check(path):
    """Every problem of the file as a bundle, as a list of strings ([] when
    valid): not a zip, members other than exactly sec. 2.7's, a manifest
    that does not validate, a digest that does not hold, a member whose
    name ends in .pt, .pkl, .npy or .npz."""
    raise NotImplementedError("block 41")


def main(argv=None):
    """`python3 -m sbitool bundle-check FILE [--rebuild]`: 0 with BUNDLE
    CHECK OK: ..., EXIT_ARGS, EXIT_REFUSE with one line per problem;
    --rebuild also builds and loads every member (model.load)."""
    raise NotImplementedError("block 41")


if __name__ == "__main__":
    sys.exit(main())
