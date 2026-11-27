"""Block 43: the files SpikeInterface reads as a Recording (specs/SPEC.md
block 43; D-173, D-176). spikeinterface and probeinterface are imported
inside the functions only.

CONTRACT (wave A): the signatures are fixed.

Pure ASCII, LF only.
"""


def from_sorting(sorting, n_samples, positions_um=None, grid=None, name=None, source=None):
    """A SpikeInterface sorting (spike trains) as a Recording: one unit per
    electrode required (the unit's id, or its channel property, an int);
    two units on one electrode are refused."""
    raise NotImplementedError("block 43")


def from_recording(recording, detection, name=None, source=None):
    """A SpikeInterface raw recording as a Recording: the peaks detected per
    channel by spikeinterface's detect_peaks with exactly the arguments of
    `detection` (schema.DETECTION_KEYS; None is refused: raw voltage needs
    a declared detection); the positions are the probe's [um]."""
    raise NotImplementedError("block 43")


def read(path, reader=None, detection=None, name=None, **reader_options):
    """The Recording of the file at `path`, read by the SpikeInterface
    extractor named `reader` (a key of its recording or sorting extractor
    lists; required). A sorting goes through from_sorting, a raw recording
    through from_recording."""
    raise NotImplementedError("block 43")
