"""Tests for the pole-node-row collapse in ``Topo.write_esmf_mesh``.

When a regular lat-lon grid reaches a pole, its first/last node row is a fan
of nodes all at lat ±90 (distinct longitudes). ESMF treats these as separate
coincident nodes, defeating its pole-aware regridding. ``write_esmf_mesh``
collapses such a *full* edge row into a single shared node per pole.
"""

import numpy as np
import pytest
import xarray as xr

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_topo(**grid_kwargs):
    grid = Grid(**grid_kwargs)
    return grid, Topo(grid, min_depth=10.0, git=False)


def _nodes_two_rows(width, south_lats, north_lats):
    """Build a row-major (lon, lat) node table of two rows of ``width`` nodes."""
    lon = np.linspace(0.0, 360.0, width, endpoint=False)
    nc = np.empty((2 * width, 2))
    nc[:width, 0] = lon
    nc[width:, 0] = lon
    nc[:width, 1] = south_lats
    nc[width:, 1] = north_lats
    return nc


# A 4-cell connectivity (1-based) over a 2-row, 4-wide node table; the last cell
# wraps so that every one of the 8 nodes is referenced.
_CONN_1BASED = np.array(
    [[1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8]], dtype=np.int32
)


# --------------------------------------------------------------------------- #
# Unit tests of the staticmethod
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "south_lats, north_lats, expected, start_index",
    [
        ([-80] * 4, [90] * 4, [("north", 4)], 1),
        ([-90] * 4, [80] * 4, [("south", 4)], 1),
        ([-90] * 4, [90] * 4, [("south", 4), ("north", 4)], 1),
        ([-90] * 4, [90] * 4, [("south", 4), ("north", 4)], 0),  # 0-based conn
    ],
    ids=["north", "south", "both", "both_start0"],
)
def test_collapse_fires(south_lats, north_lats, expected, start_index):
    """Full pole rows collapse; node count, connectivity range, dtype, and
    resolved corner geometry are all preserved."""
    nc = _nodes_two_rows(4, south_lats, north_lats)
    conn = _CONN_1BASED - (1 - start_index)  # shift base to start_index
    new_nc, new_conn, collapsed = Topo._collapse_pole_node_rows(
        nc, conn, node_row_width=4, start_index=start_index
    )

    assert collapsed == expected
    # each collapsed fan of n nodes -> 1, i.e. removes (n-1) nodes
    removed = sum(n - 1 for _, n in expected)
    assert new_nc.shape[0] == nc.shape[0] - removed
    # exactly one shared node remains at each collapsed pole
    for name, _ in expected:
        pole = 90.0 if name == "north" else -90.0
        assert int(np.isclose(new_nc[:, 1], pole).sum()) == 1
    # connectivity stays in range, keeps dtype/base, and resolves to identical
    # corner coordinates (latitude exact; longitude only matters off the pole)
    assert new_conn.dtype == conn.dtype
    assert new_conn.min() >= start_index
    assert new_conn.max() <= start_index + new_nc.shape[0] - 1
    before = nc[conn - start_index]
    after = new_nc[new_conn - start_index]
    assert np.allclose(before[..., 1], after[..., 1])
    nonpole = ~np.isclose(np.abs(before[..., 1]), 90.0)
    assert np.allclose(before[..., 0][nonpole], after[..., 0][nonpole])


@pytest.mark.parametrize(
    "south_lats, north_lats",
    [
        ([-80] * 4, [90, 90, 90, 89.9]),  # partial edge row
        ([-80, 90, -80, -80], [80] * 4),  # stray interior pole node
        ([-40] * 4, [40] * 4),  # no pole at all
    ],
    ids=["partial_row", "stray_node", "no_pole"],
)
def test_collapse_is_noop(south_lats, north_lats):
    """Anything short of a *full* edge row on the pole leaves the mesh intact."""
    nc = _nodes_two_rows(4, south_lats, north_lats)
    conn = _CONN_1BASED.copy()
    new_nc, new_conn, collapsed = Topo._collapse_pole_node_rows(nc, conn, 4)
    assert collapsed == []
    assert new_nc.shape == nc.shape
    assert np.array_equal(new_conn, conn)


# --------------------------------------------------------------------------- #
# Integration tests through write_esmf_mesh
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kw, width, collapses",
    [
        (
            dict(nx=8, ny=6, lenx=360.0, leny=180.0, ystart=-90.0, cyclic_x=True),
            8,
            True,
        ),
        (
            dict(
                nx=8,
                ny=6,
                lenx=180.0,
                leny=180.0,
                xstart=0.0,
                ystart=-90.0,
                cyclic_x=False,
            ),
            9,
            True,
        ),
        (
            dict(nx=8, ny=6, lenx=360.0, leny=120.0, ystart=-60.0, cyclic_x=True),
            8,
            False,
        ),
    ],
    ids=["cyclic_pole", "noncyclic_pole", "midlatitude_noop"],
)
def test_write_esmf_mesh_pole_handling(tmp_path, kw, width, collapses):
    nx, ny = kw["nx"], kw["ny"]
    grid, topo = _make_topo(name="g", **kw)
    p = str(tmp_path / "mesh.nc")
    topo.write_esmf_mesh(p)
    m = xr.open_dataset(p)

    conn = m.elementConn.values.astype(int)
    assert conn.min() >= 1 and conn.max() <= m.sizes["nodeCount"]
    assert set(np.unique(m.numElementConn.values)) == {4}
    # geometry comes from the grid and must be untouched by any collapse
    assert np.allclose(m.centerCoords.values[:, 0], grid.tlon.data.flatten())
    assert np.allclose(m.elementArea.values, grid.tarea.data.flatten())

    if collapses:
        # one shared node per pole: nodeCount = width*(ny+1) - 2*(width-1)
        assert m.sizes["nodeCount"] == width * (ny + 1) - 2 * (width - 1)
        assert int(np.isclose(m.nodeCoords.values[:, 1], 90).sum()) == 1
        assert int(np.isclose(m.nodeCoords.values[:, 1], -90).sum()) == 1
        assert "history" in m.attrs
        # south apex repeated in the first cell, north apex in a top-row cell
        assert conn[0][0] == conn[0][1]
        assert conn[(ny - 1) * nx][2] == conn[(ny - 1) * nx][3]
    else:
        assert m.sizes["nodeCount"] == nx * (ny + 1)  # uncollapsed
        assert "history" not in m.attrs


def test_tripolar_branch_skips_collapse(tmp_path, monkeypatch):
    """A pole-reaching grid forced down the tripolar branch must NOT collapse
    (the collapse would otherwise fire), since tripolar grids fold the north
    seam with their own connectivity."""
    grid, topo = _make_topo(
        nx=8, ny=6, lenx=360.0, leny=180.0, ystart=-90.0, cyclic_x=True, name="tri"
    )
    monkeypatch.setattr(Grid, "is_tripolar", staticmethod(lambda supergrid: True))
    p = str(tmp_path / "mesh.nc")
    topo.write_esmf_mesh(p)
    m = xr.open_dataset(p)

    assert "history" not in m.attrs  # collapse skipped
    conn = m.elementConn.values.astype(int)
    assert conn.min() >= 1 and conn.max() <= m.sizes["nodeCount"]
