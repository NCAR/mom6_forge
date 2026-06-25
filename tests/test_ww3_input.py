"""Tests for Topo.write_ww3_input (the WW3 grid-preprocessor input writer)."""

import numpy as np
import pytest

WW3_FILE_SUFFIXES = ("_x.inp", "_y.inp", "_bottom.inp", "_mapsta.inp")


def test_write_ww3_input_creates_all_files(get_rect_topo, tmp_path):
    topo = get_rect_topo
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias)

    # The four per-grid arrays plus the ww3_grid.inp control file:
    for suffix in WW3_FILE_SUFFIXES:
        assert (tmp_path / f"{alias}{suffix}").exists()
    assert (tmp_path / "ww3_grid.inp").exists()


def test_write_ww3_input_array_contents(get_rect_topo, tmp_path):
    topo = get_rect_topo  # flat 1000 m depth, all-ocean, min_depth=0
    alias = topo._grid.name
    nx, ny = topo._grid.nx, topo._grid.ny

    topo.write_ww3_input(tmp_path, grid_alias=alias)

    xcoord = np.loadtxt(tmp_path / f"{alias}_x.inp")
    ycoord = np.loadtxt(tmp_path / f"{alias}_y.inp")
    bottom = np.loadtxt(tmp_path / f"{alias}_bottom.inp")
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    # Every file is (ny, nx), southernmost row first (IDLA=1).
    for arr in (xcoord, ycoord, bottom, mapsta):
        assert arr.shape == (ny, nx)

    # Coordinates round-trip the grid t-points.
    assert np.allclose(xcoord, topo._grid.tlon.data)
    assert np.allclose(ycoord, topo._grid.tlat.data)

    # Flat ocean: positive depth everywhere, all cells wet.
    assert np.allclose(bottom, 1000.0)
    assert (mapsta == 1).all()


def test_write_ww3_input_grid_control_file(get_rect_topo, tmp_path):
    topo = get_rect_topo
    alias = topo._grid.name
    nx, ny = topo._grid.nx, topo._grid.ny

    topo.write_ww3_input(tmp_path, grid_alias=alias)
    text = (tmp_path / "ww3_grid.inp").read_text()

    # Curvilinear grid, non-cyclic (rectangular regional grid) -> NONE closure.
    assert "'CURV'" in text
    assert "'NONE'" in text
    # Grid dimensions line.
    assert f"  {nx}  {ny}" in text
    # Bottom uses SBF=-1 so positive depths map to negative-down elevation.
    assert "-1." in text
    # References each generated data file.
    for suffix in WW3_FILE_SUFFIXES:
        assert f"{alias}{suffix}" in text


def test_write_ww3_input_masked_cells_are_land(get_rect_topo, tmp_path):
    """Land cells (per the mask) get depth 0 in the bottom file and 0 in mapsta,
    keeping the depth and status files mutually consistent."""
    topo = get_rect_topo
    alias = topo._grid.name

    # Introduce land by zeroing the depth of a couple of cells.
    topo.depth[0, 0] = 0.0
    topo.depth[1, 2] = 0.0

    topo.write_ww3_input(tmp_path, grid_alias=alias)
    bottom = np.loadtxt(tmp_path / f"{alias}_bottom.inp")
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    for j, i in [(0, 0), (1, 2)]:
        assert mapsta[j, i] == 0
        assert bottom[j, i] == 0.0

    # mapsta is exactly the land/sea mask, and depth is zero wherever land.
    assert np.array_equal(mapsta, topo.tmask.data)
    assert (bottom[mapsta == 0] == 0.0).all()
