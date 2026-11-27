"""Block 43: the device's layout (specs/SPEC.md block 43).

CONTRACT (wave A): the signatures are fixed.

Pure ASCII, LF only.
"""

import numpy as np


class LayoutError(ValueError):
    """The positions are not a regular square grid within the tolerance."""


def grid(recording):
    """{electrode id: (row, col)} of a Recording that has a grid: the
    extractor's decoding of the linear index with its grid_width and
    index_base."""
    if recording.grid is None:
        raise LayoutError("the recording %r has no grid (grid is None)" % (recording.name,))
    width, base = int(recording.grid[0]), int(recording.grid[1])
    from sbitool.recordings import ptrain   # the extractor is imported in ptrain.py only
    cse = ptrain._extractor()
    ids = sorted(recording.spikes)
    try:
        cse.validate_grid(ids, width=width, base=base)
    except ValueError as exc:   # the extractor's GeometryError
        raise LayoutError("the recording %r: %s" % (recording.name, exc)) from exc
    out = {}
    for e in ids:
        c = cse.electrode_coords(e, width=width, base=base)
        out[int(e)] = (int(c.row), int(c.col))
    return out


def _fit(xy, col, row):
    """Least squares (x0, y0, pitch) of x = x0 + pitch col, y = y0 + pitch row."""
    n = xy.shape[0]
    a = np.zeros((2 * n, 3))
    a[:n, 0] = 1.0
    a[:n, 2] = col
    a[n:, 1] = 1.0
    a[n:, 2] = row
    b = np.concatenate([xy[:, 0], xy[:, 1]])
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    return float(sol[0]), float(sol[1]), float(sol[2])


def as_grid(positions_um, tol_um=1.0):
    """(grid_width, pitch_um, {electrode id: linear index}) for positions
    {electrode id: (x, y)} that lie on a regular square grid within tol_um
    (index_base 0, rows along y, the origin at the smallest x and y);
    LayoutError otherwise.
    # Custom: no library function turns positions into the extractor's grid
    # index. Checked: probeinterface.Probe (positions, no grid index)."""
    from scipy.spatial import cKDTree
    if isinstance(tol_um, bool) or not isinstance(tol_um, (int, float)) or not tol_um > 0 \
            or not np.isfinite(tol_um):
        raise LayoutError("tol_um must be a positive finite number, got %r" % (tol_um,))
    if not isinstance(positions_um, dict) or len(positions_um) < 2:
        raise LayoutError("a grid needs the positions of at least 2 electrodes")
    ids = sorted(positions_um)
    xy = np.array([positions_um[e] for e in ids], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
        raise LayoutError("every position must be a finite pair (x, y) [um]")
    # the pitch's first estimate: the median nearest-neighbour distance
    dist = cKDTree(xy).query(xy, k=2)[0][:, 1]
    pitch = float(np.median(dist))
    if not pitch > 2.0 * tol_um:
        raise LayoutError("the electrodes' spacing %.6g um is not larger than twice the "
                          "tolerance %.6g um: no grid can be resolved" % (pitch, tol_um))
    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    col = row = None
    for _ in range(5):   # assign the cells, refit origin and pitch, until the cells are stable
        c_new = np.rint((xy[:, 0] - x0) / pitch).astype(np.int64)
        r_new = np.rint((xy[:, 1] - y0) / pitch).astype(np.int64)
        c_new -= c_new.min()
        r_new -= r_new.min()
        if col is not None and np.array_equal(c_new, col) and np.array_equal(r_new, row):
            break
        col, row = c_new, r_new
        x0, y0, pitch = _fit(xy, col, row)
        if not pitch > 2.0 * tol_um:
            raise LayoutError("the fitted pitch %.6g um is not larger than twice the tolerance"
                              % pitch)
    res = np.maximum(np.abs(xy[:, 0] - (x0 + pitch * col)), np.abs(xy[:, 1] - (y0 + pitch * row)))
    worst = int(np.argmax(res))
    if res[worst] > tol_um:
        raise LayoutError("the positions are not a regular square grid within %.6g um: electrode "
                          "%r is %.6g um from its cell (pitch %.6g um)"
                          % (tol_um, ids[worst], float(res[worst]), pitch))
    width = int(max(col.max(), row.max()) + 1)
    lin = row * width + col
    if np.unique(lin).size != lin.size:
        raise LayoutError("two electrodes fall in one cell of the grid (pitch %.6g um)" % pitch)
    return width, pitch, {int(e): int(k) for e, k in zip(ids, lin)}
