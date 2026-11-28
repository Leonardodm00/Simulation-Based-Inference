"""Block 49: the session engine (specs/SPEC.md sec. 2.9 and block 49;
D-189, D-190).

CONTRACT (wave A of S6): the names, signatures and RESULT_KEYS below are
fixed by specs/SPEC.md; the bodies are block 49's. Everything the viewer
computes, with plain data in and out: no server, no browser. Nothing is
estimated here that another block estimates (the subregions, traces and
windows are block 44's, the inference block 45's); the statistics are
counts on the spike times.

Pure ASCII, LF only.
"""

RESULT_KEYS = ("n", "n_post", "seed", "batch_rows", "span", "excluded", "parameters", "coordinate",
               "gate", "windows", "files")
PARAMETER_KEYS = ("name", "unit", "bounds", "pooled", "windows_q50")
POOLED_KEYS = ("q05", "q50", "q95")
GATE_KEYS = ("alpha", "n_flagged", "share_flagged", "flag", "p")
WINDOW_KEYS = ("subregion", "window_idx", "t_start_s")
STAT_KEYS = ("n", "rate_hz", "isi_cv")
CV_MIN_SPIKES = 4        # the CV of the inter-spike intervals needs n >= 4 spikes


class SessionError(ValueError):
    """What a session refuses; the message says why."""


class Session:
    """A bundle and a recording opened together (SPEC block 49)."""

    def __init__(self, bundle_path, recording_path, adapter, options=None):
        """Reads the bundle (block 41) and the recording (block 44's
        from_file route: the bundle's detection for raw voltage).
        SessionError for any refusal of blocks 41, 43, 44."""
        raise NotImplementedError("block 49: Session")

    def describe(self):
        """{"bundle": {...}, "recording": {...}, "W": int, "window_s": float}
        as SPEC block 49 lists them; JSON-serialisable."""
        raise NotImplementedError("block 49: Session.describe")

    def spikes(self, t0=None, t1=None, electrodes=None):
        """{electrode id: [spike times, s, sorted]} with t0 <= t < t1."""
        raise NotImplementedError("block 49: Session.spikes")

    def statistics(self, t0=None, t1=None):
        """{electrode id: {"n", "rate_hz", "isi_cv"}} over [t0, t1): the
        spike count, n / (t1 - t0) [Hz], and std(isi, ddof 0) / mean(isi)
        for n >= CV_MIN_SPIKES, None otherwise."""
        raise NotImplementedError("block 49: Session.statistics")

    def set_excluded(self, ids):
        """The electrodes left out of the algorithm (D-189)."""
        raise NotImplementedError("block 49: Session.set_excluded")

    @property
    def excluded(self):
        raise NotImplementedError("block 49: Session.excluded")

    def subregions(self):
        """[{"index", "center", "members", "mean_mfr", "center_mfr"}] under
        the current exclusion (windows.subregions(..., exclude=...))."""
        raise NotImplementedError("block 49: Session.subregions")

    def traces(self):
        """{"fs_ifr_hz": float, "x": (n_sub, K) float32 [spikes per bin per
        electrode]}: block 44's traces of the current subregions."""
        raise NotImplementedError("block 49: Session.traces")

    def windows(self, t0=None, t1=None):
        """{"n", "per_subregion", "subregion", "window_idx", "t_start_s"}:
        the windows of the whole recording's traces with t0 <= t_start_s
        and t_start_s + window_s <= t1."""
        raise NotImplementedError("block 49: Session.windows")

    def infer(self, t0=None, t1=None, n_post=None, seed=None, out_dir=None, log=None):
        """Block 45's inference on the windows of the span; the summary,
        exactly RESULT_KEYS. SessionError when the span holds no window."""
        raise NotImplementedError("block 49: Session.infer")
