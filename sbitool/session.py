"""Block 49: the session engine (specs/SPEC.md sec. 2.9 and block 49;
D-189, D-190).

CONTRACT (wave A of S6): the names, signatures and RESULT_KEYS below are
fixed by specs/SPEC.md; the bodies are block 49's. Everything the viewer
computes, with plain data in and out: no server, no browser. Nothing is
estimated here that another block estimates (the subregions, traces and
windows are block 44's, the inference block 45's); the statistics are
counts on the spike times (numpy), the CV scipy.stats.variation (contract
amendment 1 of wave A).

Pure ASCII, LF only.
"""

import numpy as np

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


def _one_line(exc):
    return " ".join(str(exc).split())


def _refusals():
    """The refusals of blocks 41, 43, 44, 45 (and the sources'), which a
    session turns into SessionError."""
    from sbitool import _sources, bundle, infer, recordings, windows
    return (bundle.BundleError, recordings.RecordingError, windows.WindowsError,
            infer.InferRefused, _sources.SourceError)


def _jsonable(v):
    from sbitool import windows
    return windows._jsonable(v)


def _quiet(*_a, **_k):
    return None


class Session:
    """A bundle and a recording opened together (SPEC block 49)."""

    def __init__(self, bundle_path, recording_path, adapter, options=None):
        """Reads the bundle (block 41) and the recording (block 44's
        from_file route: the bundle's detection for raw voltage).
        SessionError for any refusal of blocks 41, 43, 44."""
        from sbitool import bundle as bundle_mod
        from sbitool import windows as WIN
        options = dict(options or {})
        if "detection" in options:
            raise SessionError("the option detection is refused: the detection is the bundle's "
                               "(its extraction.detection), never the user's")
        try:
            try:
                b = bundle_mod.read(bundle_path)
            except OSError as exc:
                raise bundle_mod.BundleError("the bundle %s cannot be read: %s"
                                             % (bundle_path, exc))
            extraction = b.manifest.get("extraction")
            if extraction is None:
                raise WIN.WindowsError("the bundle has no extraction (null): it accepts windows "
                                       "only, not a recording")
            try:
                rec, applied = WIN._read_recording(recording_path, adapter, options,
                                                   extraction.get("detection"))
            except (OSError, TypeError) as exc:
                raise SessionError("the recording %s cannot be read with the adapter %r: %s"
                                   % (recording_path, adapter, _one_line(exc)))
            WIN.grid_of(rec, extraction)     # the device must fit the bundle (block 44's Rule)
        except _refusals() as exc:
            raise SessionError(_one_line(exc)) from exc
        self._bundle = b
        self._extraction = extraction
        self._recording = rec
        self._detection_applied = applied
        self._fs = float(rec.fs_raw_hz)
        self._duration = float(rec.duration_s)
        self._ids = sorted(int(e) for e in rec.spikes)
        # spike times [s] = sample index / fs_raw_hz (sec. 2.7, "A recording")
        self._times = {e: np.asarray(rec.spikes[e], dtype=np.float64) / self._fs
                       for e in self._ids}
        self._W = int(b.manifest["W"])
        self._excluded = frozenset()
        self._cache = {}

    # ------------------------------------------------------------ helpers

    def _span(self, t0, t1):
        """(t0, t1) with the defaults 0 and duration_s; SessionError for a
        span outside [0, duration_s] or with t1 <= t0."""
        t0 = 0.0 if t0 is None else t0
        t1 = self._duration if t1 is None else t1
        for name, v in (("t0", t0), ("t1", t1)):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer,
                                                                     np.floating)):
                raise SessionError("%s must be a number [s], got %r" % (name, v))
        t0, t1 = float(t0), float(t1)
        if not (np.isfinite(t0) and np.isfinite(t1)):
            raise SessionError("the span [%r, %r) is not finite" % (t0, t1))
        if not (0.0 <= t0 < t1 <= self._duration):
            raise SessionError("the span [%r, %r) s is not inside [0, %r] s with t0 < t1"
                               % (t0, t1, self._duration))
        return t0, t1

    def _in_span(self, e, t0, t1):
        t = self._times[e]
        return t[(t >= t0) & (t < t1)]

    def _fs_ifr(self):
        return self._traces()[1]

    def _subs(self):
        from sbitool import windows as WIN
        if "subs" not in self._cache:
            try:
                self._cache["subs"] = WIN.subregions(self._recording, self._extraction,
                                                     exclude=sorted(self._excluded))
            except _refusals() as exc:
                raise SessionError(_one_line(exc)) from exc
        return self._cache["subs"]

    def _traces(self):
        from sbitool import windows as WIN
        if "traces" not in self._cache:
            subs = self._subs()
            try:
                self._cache["traces"] = WIN.traces(self._recording, subs, self._extraction)
            except _refusals() as exc:
                raise SessionError(_one_line(exc)) from exc
        return self._cache["traces"]

    def _cut(self):
        """Block 44's cut of the whole recording's traces: (windows, ids)."""
        from sbitool import windows as WIN
        if "cut" not in self._cache:
            x, fs_ifr = self._traces()
            try:
                self._cache["cut"] = WIN.cut(x, self._W, fs_ifr)
            except _refusals() as exc:
                raise SessionError(_one_line(exc)) from exc
        return self._cache["cut"]

    def _selected(self, t0, t1):
        """(t0, t1, mask) of the cut's windows whose span lies inside
        [t0, t1]: t0 <= t_start_s and t_start_s + window_s <= t1."""
        t0, t1 = self._span(t0, t1)
        _w, ids = self._cut()
        window_s = self._W / self._fs_ifr()
        ts = ids["t_start_s"]
        return t0, t1, (ts >= t0) & (ts + window_s <= t1)

    # ------------------------------------------------------------ the API

    def describe(self):
        """{"bundle": {...}, "recording": {...}, "W": int, "window_s": float}
        as SPEC block 49 lists them; JSON-serialisable."""
        from sbitool import card as CARD
        from sbitool.plotting import device
        man = self._bundle.manifest
        try:
            header = CARD.parse(self._bundle.card)[0]
        except CARD.CardError:
            header = None
        try:
            pos, unit = device.positions(self._recording)
        except device.FigureError as exc:
            raise SessionError(_one_line(exc)) from exc
        rec = self._recording
        electrodes = [{"id": e, "x": float(pos[e][0]), "y": float(pos[e][1]),
                       "n_spikes": int(self._times[e].size)} for e in self._ids]
        out = {"bundle": {"name": man["name"], "sha256": self._bundle.sha256,
                          "card_complete": bool(man["card_complete"]), "card": header,
                          "parameters": man["parameters"], "extraction": man["extraction"],
                          "sample": man["sample"], "gate": man["gate"]},
               "recording": {"name": rec.name, "duration_s": self._duration,
                             "fs_raw_hz": self._fs, "n_electrodes": len(self._ids),
                             "position_unit": unit, "electrodes": electrodes},
               "W": self._W, "window_s": self._W / self._fs_ifr()}
        return _jsonable(out)

    def spikes(self, t0=None, t1=None, electrodes=None):
        """{electrode id: [spike times, s, sorted]} with t0 <= t < t1."""
        t0, t1 = self._span(t0, t1)
        if electrodes is None:
            ids = self._ids
        else:
            if isinstance(electrodes, (str, bytes)) or not hasattr(electrodes, "__iter__"):
                raise SessionError("electrodes must be an iterable of electrode ids, got %r"
                                   % (electrodes,))
            ids = list(electrodes)
            bad = [e for e in ids if isinstance(e, (bool, np.bool_))
                   or not isinstance(e, (int, np.integer)) or int(e) not in self._times]
            if bad:
                raise SessionError("the recording %r has no electrode(s) %s"
                                   % (self._recording.name, bad[:10]))
            ids = sorted(set(int(e) for e in ids))
        return {e: self._in_span(e, t0, t1).tolist() for e in ids}

    def statistics(self, t0=None, t1=None):
        """{electrode id: {"n", "rate_hz", "isi_cv"}} over [t0, t1): the
        spike count, n / (t1 - t0) [Hz], and the coefficient of variation of
        the n - 1 inter-spike intervals, scipy.stats.variation (std with
        ddof 0 over the mean), for n >= CV_MIN_SPIKES, None otherwise. The
        intervals are positive (a train's sample indices strictly increase),
        so the mean is never 0."""
        from scipy.stats import variation
        t0, t1 = self._span(t0, t1)
        out = {}
        for e in self._ids:
            t = self._in_span(e, t0, t1)
            n = int(t.size)
            cv = float(variation(np.diff(t))) if n >= CV_MIN_SPIKES else None
            out[e] = {"n": n, "rate_hz": n / (t1 - t0), "isi_cv": cv}
        return out

    def set_excluded(self, ids):
        """The electrodes left out of the algorithm (D-189)."""
        if ids is None:
            ids = ()
        if isinstance(ids, (str, bytes)) or not hasattr(ids, "__iter__"):
            raise SessionError("the excluded electrodes must be an iterable of ids, got %r"
                               % (ids,))
        ids = list(ids)
        bad = [e for e in ids if isinstance(e, (bool, np.bool_))
               or not isinstance(e, (int, np.integer)) or int(e) not in self._times]
        if bad:
            raise SessionError("the recording %r has no electrode(s) %s to exclude"
                               % (self._recording.name, bad[:10]))
        new = frozenset(int(e) for e in ids)
        if new != self._excluded:
            self._excluded = new
            self._cache = {}

    @property
    def excluded(self):
        return sorted(self._excluded)

    def subregions(self):
        """[{"index", "center", "members", "mean_mfr", "center_mfr"}] under
        the current exclusion (windows.subregions(..., exclude=...))."""
        return [{"index": j, "center": int(s.center), "members": [int(m) for m in s.members],
                 "mean_mfr": float(s.mean_mfr), "center_mfr": float(s.center_mfr)}
                for j, s in enumerate(self._subs())]

    def traces(self):
        """{"fs_ifr_hz": float, "x": (n_sub, K) float32 [spikes per bin per
        electrode]}: block 44's traces of the current subregions."""
        x, fs_ifr = self._traces()
        return {"fs_ifr_hz": float(fs_ifr), "x": x}

    def windows(self, t0=None, t1=None):
        """{"n", "per_subregion", "subregion", "window_idx", "t_start_s"}:
        the windows of the whole recording's traces with t0 <= t_start_s
        and t_start_s + window_s <= t1. per_subregion: the number of those
        windows of each subregion, in subregion order; the three ids as
        lists, in the cut's order."""
        _t0, _t1, sel = self._selected(t0, t1)
        _w, ids = self._cut()
        n_sub = len(self._subs())
        sub = ids["subregion"][sel]
        return {"n": int(sel.sum()),
                "per_subregion": np.bincount(sub, minlength=n_sub).astype(int).tolist(),
                "subregion": sub.tolist(), "window_idx": ids["window_idx"][sel].tolist(),
                "t_start_s": ids["t_start_s"][sel].tolist()}

    def infer(self, t0=None, t1=None, n_post=None, seed=None, out_dir=None, log=None):
        """Block 45's inference on the windows of the span; the summary,
        exactly RESULT_KEYS. SessionError when the span holds no window."""
        from sbitool import infer as INF
        t0, t1, sel = self._selected(t0, t1)
        windows, ids = self._cut()
        if not sel.any():
            raise SessionError("no window: the span [%r, %r) s holds no whole window of %r s"
                               % (t0, t1, self._W / self._fs_ifr()))
        x = windows[sel]
        wids = {k: ids[k][sel] for k in WINDOW_KEYS}
        try:
            res = INF.run(self._bundle, x, ids=wids, n_post=n_post, seed=seed,
                          log=_quiet if log is None else log)
            files = None if out_dir is None else [str(p) for p in INF.write(res, out_dir)]
        except _refusals() as exc:
            raise SessionError(_one_line(exc)) from exc
        return self._summary(res, t0, t1, files)

    def _summary(self, res, t0, t1, files):
        A, js = res["arrays"], res["json"]
        par = self._bundle.manifest["parameters"]
        bounds = par["bounds"]
        p = len(par["names"])

        def phys(j, u):
            u = np.asarray(u, dtype=np.float64)
            if bounds is None:
                return u
            lo, hi = float(bounds[j][0]), float(bounds[j][1])
            return lo + u * (hi - lo)

        params = []
        for j in range(p):
            params.append({"name": par["names"][j], "unit": par["units"][j],
                           "bounds": None if bounds is None else list(bounds[j]),
                           "pooled": {k: float(phys(j, A["mix_" + k][j])) for k in POOLED_KEYS},
                           "windows_q50": phys(j, A["q50"][:, j]).tolist()})
        flag = A["gate_flag"]
        out = {"n": int(js["n"]), "n_post": int(js["n_post"]), "seed": int(js["seed"]),
               "batch_rows": int(js["batch_rows"]), "span": [t0, t1], "excluded": self.excluded,
               "parameters": params, "coordinate": "unit_box" if bounds is None else "physical",
               "gate": {"alpha": float(js["gate"]["alpha"]),
                        "n_flagged": int(js["gate"]["n_flagged"]),
                        "share_flagged": float(js["gate"]["share_flagged"]),
                        "flag": [bool(f) for f in flag], "p": A["gate_p"].tolist()},
               "windows": {k: A[k].tolist() for k in WINDOW_KEYS},
               "files": files}
        assert tuple(out) == RESULT_KEYS
        return out
