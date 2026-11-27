"""Block 40: what a recording is for the tool, and the interfaces a
recording reader, a simulator and a scheduler offer (specs/SPEC.md sec. 2.7
and block 40).

CONTRACT (wave A): the dataclass's fields and the protocols are fixed by
specs/SPEC.md; `Recording.check` is the implementer's. numpy only.

Pure ASCII, LF only.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class Recording:
    """One recording as detected spikes per electrode (sec. 2.7, "A
    recording").

    name          a label for the outputs
    spikes        electrode id (int) -> (n_e,) int64, the sorted sample
                  indices of its spikes, each in [0, n_samples)
    fs_raw_hz     the raw sampling rate [Hz]; spike time = index / fs_raw_hz
    n_samples     the recording's length in raw samples
    positions_um  electrode id -> (x, y) [um], the same ids as `spikes`; or
                  None when the file holds no geometry
    grid          None, or (grid_width, index_base, pitch_um) of a regular
                  square grid on which an electrode's id is its linear index
    source        what was read and how (adapter, path, options)
    """
    name: str
    spikes: Dict[int, np.ndarray]
    fs_raw_hz: float
    n_samples: int
    positions_um: Optional[Dict[int, Tuple[float, float]]] = None
    grid: Optional[Tuple[int, int, float]] = None
    source: dict = field(default_factory=dict)

    @property
    def duration_s(self):
        """T_rec = n_samples / fs_raw_hz [s]."""
        return self.n_samples / self.fs_raw_hz

    def check(self):
        """Every problem of this recording as a list of strings ([] when
        valid): a non-positive or non-finite rate, n_samples < 1, a train
        that is not int64, not sorted or out of [0, n_samples), positions
        for other electrodes than the spikes', a grid that is not three
        valid values."""
        raise NotImplementedError("block 40")


class RecordingAdapter(Protocol):
    """Reads one recording from a path (block 43)."""
    def read(self, path, **options) -> Recording: ...


class Simulator(Protocol):
    """The simulator at a given parameter (wave 2 of step S2, D-172):
    theta (p,) in the unit box, seed an int; returns x (n_windows, W)
    float32, the windows the bundle's networks take."""
    def simulate(self, theta, seed): ...


class Scheduler(Protocol):
    """A cluster's scheduler (step S5): submit a job, ask its state."""
    def submit(self, job) -> str: ...
    def state(self, job_id) -> str: ...
