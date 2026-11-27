"""Block 40: what a recording is for the tool, and the interfaces a
recording reader, a simulator and a scheduler offer (specs/SPEC.md sec. 2.7
and block 40).

CONTRACT (wave A): the dataclass's fields and the protocols are fixed by
specs/SPEC.md. `Recording.check` was written with the contract (the
orchestrator's), so that block 43 can call it; block 40's T40b tests it.
numpy only.

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
                  (pitch_um None when the pitch is not known)
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
        probs = []
        fs = self.fs_raw_hz
        if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
            probs.append("fs_raw_hz must be a positive finite number, got %r" % (fs,))
        n = self.n_samples
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            probs.append("n_samples must be an integer >= 1, got %r" % (n,))
            n = None
        if not isinstance(self.spikes, dict) or not self.spikes:
            probs.append("spikes must be a non-empty dict electrode id -> sample indices")
            return probs
        for e in sorted(self.spikes, key=repr):
            if isinstance(e, bool) or not isinstance(e, (int, np.integer)):
                probs.append("spikes: electrode id %r is not an integer" % (e,))
                continue
            t = self.spikes[e]
            if not isinstance(t, np.ndarray) or t.ndim != 1 or t.dtype != np.int64:
                probs.append("spikes[%d] must be a 1-D int64 array" % e)
                continue
            if t.size and np.any(np.diff(t) <= 0):
                probs.append("spikes[%d] is not strictly increasing" % e)
            if t.size and (t[0] < 0 or (n is not None and t[-1] >= n)):
                probs.append("spikes[%d] has a sample index outside [0, n_samples)" % e)
        if self.positions_um is not None:
            if not isinstance(self.positions_um, dict) or set(self.positions_um) != set(self.spikes):
                probs.append("positions_um must hold exactly the electrodes of spikes")
            else:
                for e, xy in self.positions_um.items():
                    if (not isinstance(xy, tuple) or len(xy) != 2
                            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in xy)):
                        probs.append("positions_um[%r] must be a pair of finite numbers" % (e,))
        if self.grid is not None:
            g = self.grid
            ok = (isinstance(g, tuple) and len(g) == 3
                  and isinstance(g[0], (int, np.integer)) and not isinstance(g[0], bool) and g[0] >= 1
                  and g[1] in (0, 1) and not isinstance(g[1], bool)
                  and (g[2] is None or (isinstance(g[2], (int, float)) and np.isfinite(g[2])
                                        and g[2] > 0)))
            if not ok:
                probs.append("grid must be (grid_width >= 1, index_base 0 or 1, pitch_um > 0 or "
                             "None), got %r" % (g,))
        return probs


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
