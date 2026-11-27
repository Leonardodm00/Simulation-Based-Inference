"""Block 43: the device's layout (specs/SPEC.md block 43).

CONTRACT (wave A): the signatures are fixed.

Pure ASCII, LF only.
"""


class LayoutError(ValueError):
    """The positions are not a regular square grid within the tolerance."""


def grid(recording):
    """{electrode id: (row, col)} of a Recording that has a grid: the
    extractor's decoding of the linear index with its grid_width and
    index_base."""
    raise NotImplementedError("block 43")


def as_grid(positions_um, tol_um=1.0):
    """(grid_width, pitch_um, {electrode id: linear index}) for positions
    {electrode id: (x, y)} that lie on a regular square grid within tol_um
    (index_base 0, rows along y, the origin at the smallest x and y);
    LayoutError otherwise.
    # Custom: no library function turns positions into the extractor's grid
    # index. Checked: probeinterface.Probe (positions, no grid index)."""
    raise NotImplementedError("block 43")
