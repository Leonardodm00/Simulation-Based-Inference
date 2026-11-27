"""Block 43: the recording adapters' registry (specs/SPEC.md sec. 2.7, "A
recording", and block 43; D-173, D-176).

CONTRACT (wave A): ADAPTERS and the signature are fixed.

Pure ASCII, LF only.
"""

ADAPTERS = ("ptrain", "spikeinterface")


class RecordingError(ValueError):
    """A recording cannot be read as asked: the message names what is
    missing or wrong (a format, a rate, a geometry)."""


def read(path, adapter, **options):
    """The Recording at `path` read by the named adapter (`ptrain.read` or
    `si.read` with `options`); RecordingError for an unknown adapter. The
    result has passed Recording.check."""
    raise NotImplementedError("block 43")
