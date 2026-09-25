"""Tests for Topo.write_ww3_input (the WW3 grid-preprocessor input writer)."""

import numpy as np
import pytest

from mom6_forge.topo import WW3_IC4_METHOD, WW3_STOKES_WAVENUMBERS

WW3_FILE_SUFFIXES = ("_x.inp", "_y.inp", "_bottom.inp", "_mapsta.inp")


def test_write_ww3_input_creates_all_files(get_rect_topo_without_vc, tmp_path):
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias)

    # The four per-grid arrays plus the ww3_grid.inp control file:
    for suffix in WW3_FILE_SUFFIXES:
        assert (tmp_path / f"{alias}{suffix}").exists()
    assert (tmp_path / "ww3_grid.inp").exists()


def test_write_ww3_input_array_contents(get_rect_topo_without_vc, tmp_path):
    topo = get_rect_topo_without_vc  # flat 1000 m depth, all-ocean, min_depth=0
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

    # Every edge is an active boundary by default; the interior is plain sea.
    assert (mapsta[1:-1, 1:-1] == 1).all()
    assert (mapsta > 0).all()


def test_write_ww3_input_grid_control_file(get_rect_topo_without_vc, tmp_path):
    topo = get_rect_topo_without_vc
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


def test_write_ww3_input_stokes_bands(get_rect_topo_without_vc, tmp_path):
    """The &OUTS namelist has to ask WW3 for the partitioned Stokes drift that
    MOM6's SURFACE_BANDS coupling expects, on the wavenumbers MOM6 will apply."""
    topo = get_rect_topo_without_vc

    topo.write_ww3_input(tmp_path, grid_alias=topo._grid.name)
    text = (tmp_path / "ww3_grid.inp").read_text()

    # Both caps size the Sw_pstokes field for 3 bands.
    assert len(WW3_STOKES_WAVENUMBERS) == 3
    assert f"USSP = 1, IUSSP = {len(WW3_STOKES_WAVENUMBERS)}" in text
    assert "STK_TAIL = T" in text
    assert "STK_WN = " + ", ".join(str(k) for k in WW3_STOKES_WAVENUMBERS) in text


def test_write_ww3_input_masked_cells_are_land(get_rect_topo_without_vc, tmp_path):
    """Land cells (per the mask) get depth 0 in the bottom file and 0 in mapsta,
    keeping the depth and status files mutually consistent."""
    topo = get_rect_topo_without_vc
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

    # mapsta land is exactly the mask's land, and depth is zero wherever land.
    assert np.array_equal(mapsta == 0, topo.tmask.data == 0)
    assert (bottom[mapsta == 0] == 0.0).all()


def test_write_ww3_input_default_edges_regional(get_rect_topo_without_vc, tmp_path):
    """No boundary_edges given: every edge of a regional grid is a boundary."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias)
    default = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")
    topo.write_ww3_input(
        tmp_path, grid_alias=alias, boundary_edges=["south", "north", "west", "east"]
    )
    assert np.array_equal(default, np.loadtxt(tmp_path / f"{alias}_mapsta.inp"))


def test_write_ww3_input_default_edges_cyclic(get_simple_global_grid, tmp_path):
    """No boundary_edges given on a cyclic-x grid: no boundary points, as
    with open_boundary=True before; a periodic band lists south/north itself."""
    from mom6_forge.topo import Topo

    topo = Topo(get_simple_global_grid, min_depth=0, git=False)
    topo.set_flat(1000)
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias)
    assert (np.loadtxt(tmp_path / f"{alias}_mapsta.inp") == 1).all()


def test_write_ww3_input_no_edges(get_rect_topo_without_vc, tmp_path):
    """An empty list is the explicit opt-out: a plain land/sea mask."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias, boundary_edges=[])
    assert (np.loadtxt(tmp_path / f"{alias}_mapsta.inp") == 1).all()


def test_write_ww3_input_open_boundary_deprecated(get_rect_topo_without_vc, tmp_path):
    """open_boundary still works, with a DeprecationWarning: True is the
    default edge set, False is no edges, and it cannot be combined with
    boundary_edges."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name
    mapsta_file = tmp_path / f"{alias}_mapsta.inp"

    with pytest.warns(DeprecationWarning):
        topo.write_ww3_input(tmp_path, grid_alias=alias, open_boundary=True)
    assert (np.loadtxt(mapsta_file) > 0).all() and (np.loadtxt(mapsta_file) == 2).any()
    with pytest.warns(DeprecationWarning):
        topo.write_ww3_input(tmp_path, grid_alias=alias, open_boundary=False)
    assert (np.loadtxt(mapsta_file) == 1).all()
    with pytest.raises(ValueError, match="not both"), pytest.warns(DeprecationWarning):
        topo.write_ww3_input(
            tmp_path, grid_alias=alias, boundary_edges=["south"], open_boundary=True
        )


def test_write_ww3_input_rejects_tripolar(
    get_simple_global_grid, tmp_path, monkeypatch
):
    """The closure written has no 'TRPL' case, so a tripolar grid is refused
    before anything is written."""
    from mom6_forge.topo import Topo

    topo = Topo(get_simple_global_grid, min_depth=0, git=False)
    topo.set_flat(1000)
    monkeypatch.setattr(
        type(topo._grid.supergrid), "is_tripolar", property(lambda self: True)
    )
    with pytest.raises(ValueError, match="tripolar"):
        topo.write_ww3_input(tmp_path, grid_alias=topo._grid.name)
    assert not (tmp_path / f"{topo._grid.name}_mapsta.inp").exists()


def test_write_ww3_input_all_boundary_edges(get_rect_topo_without_vc, tmp_path):
    """Requesting every edge flags the whole perimeter as active boundary
    points (2); the interior stays ordinary sea points (1)."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.write_ww3_input(
        tmp_path, grid_alias=alias, boundary_edges=["south", "north", "west", "east"]
    )
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    assert (mapsta[0, :] == 2).all()
    assert (mapsta[-1, :] == 2).all()
    assert (mapsta[:, 0] == 2).all()
    assert (mapsta[:, -1] == 2).all()
    assert (mapsta[1:-1, 1:-1] == 1).all()


def test_write_ww3_input_boundary_edges_subset(get_rect_topo_without_vc, tmp_path):
    """Only the requested edges are flagged. Row j=0 is the southernmost
    (IDLA=1) and column i=0 the westernmost, so a south+west request must
    leave the north and east edges as ordinary sea points."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias, boundary_edges=["south", "west"])
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    assert (mapsta[0, :] == 2).all()
    assert (mapsta[:, 0] == 2).all()
    assert (mapsta[-1, 1:] == 1).all()
    assert (mapsta[1:, -1] == 1).all()


def test_write_ww3_input_boundary_edges_accepts_a_bare_string(
    get_rect_topo_without_vc, tmp_path
):
    """A bare string is one edge, not an iterable of characters."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias, boundary_edges="NORTH")
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    assert (mapsta[-1, :] == 2).all()
    assert (mapsta[:-1, :] == 1).all()


def test_write_ww3_input_boundary_edge_land_cells_stay_land(
    get_rect_topo_without_vc, tmp_path
):
    """A land cell sitting on a flagged edge stays land (0), not a boundary
    point: only ocean cells can carry boundary data."""
    topo = get_rect_topo_without_vc
    alias = topo._grid.name

    topo.depth[0, 0] = 0.0  # southwest corner, on both flagged edges

    topo.write_ww3_input(tmp_path, grid_alias=alias, boundary_edges=["south", "west"])
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    assert mapsta[0, 0] == 0
    assert (mapsta[0, 1:] == 2).all()
    assert (mapsta[1:, 0] == 2).all()


def test_write_ww3_input_cyclic_grid_allows_north_south(
    get_simple_global_grid, tmp_path
):
    """A grid reentrant in x (e.g. a global band) still has physical
    north/south edges, which can be flagged as usual."""
    from mom6_forge.topo import Topo

    grid = get_simple_global_grid
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)
    alias = grid.name

    topo.write_ww3_input(tmp_path, grid_alias=alias, boundary_edges=["south", "north"])
    mapsta = np.loadtxt(tmp_path / f"{alias}_mapsta.inp")

    assert (mapsta[0, :] == 2).all()
    assert (mapsta[-1, :] == 2).all()
    assert (mapsta[1:-1, :] == 1).all()


def test_write_ww3_input_cyclic_grid_rejects_east_west(
    get_simple_global_grid, tmp_path
):
    """A reentrant edge has no physical boundary, so asking for boundary
    forcing there is an error rather than a silent no-op."""
    from mom6_forge.topo import Topo

    grid = get_simple_global_grid
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)

    with pytest.raises(ValueError, match="reentrant in x"):
        topo.write_ww3_input(
            tmp_path, grid_alias=grid.name, boundary_edges=["north", "east"]
        )


def test_write_ww3_input_unknown_boundary_edge_raises(
    get_rect_topo_without_vc, tmp_path
):
    """An unknown edge name is rejected before anything is written."""
    topo = get_rect_topo_without_vc
    out_dir = tmp_path / "ocnice"

    with pytest.raises(ValueError, match="Unknown WW3 boundary edge"):
        topo.write_ww3_input(
            out_dir, grid_alias=topo._grid.name, boundary_edges=["norht"]
        )

    assert not out_dir.exists()


def test_write_ww3_input_after_reconstruction_from_files(
    get_rect_topo_without_vc, tmp_path
):
    """visualCaseGen's WW3 input generator reconstructs the Grid/Topo from the saved ocean grid
    files (ocean_hgrid + ocean_topog) before writing the *.inp files. Exercise that exact
    sequence: save the grid/topo, reload via Grid.from_supergrid + Topo.from_topo_file, then
    write the WW3 inputs and confirm they are produced and consistent with the grid."""
    import xarray as xr
    from mom6_forge.grid import Grid
    from mom6_forge.topo import Topo

    topo = get_rect_topo_without_vc  # flat 1000 m depth, all ocean
    alias = topo._grid.name

    # Save the ocean grid files, as the mom6_forge notebook would.
    supergrid_file = tmp_path / "ocean_hgrid.nc"
    topo_file = tmp_path / "ocean_topog.nc"
    topo._grid.write_supergrid(supergrid_file.as_posix())
    topo.write_topo(topo_file.as_posix())

    # Reconstruct from the saved files (mirrors WW3InputGenerator): read min_depth from the
    # topog attribute so the WW3 land/sea mask matches the ocean mask. This also confirms
    # write_topo persists the min_depth attribute.
    with xr.open_dataset(topo_file) as ds_topo:
        min_depth = float(ds_topo.attrs["min_depth"])
    grid = Grid.from_supergrid(supergrid_file.as_posix())
    reloaded = Topo.from_topo_file(grid, topo_file.as_posix(), min_depth=min_depth)

    out_dir = tmp_path / "ocnice"
    out_dir.mkdir()
    reloaded.write_ww3_input(out_dir.as_posix(), grid_alias=alias)

    for suffix in WW3_FILE_SUFFIXES:
        assert (out_dir / f"{alias}{suffix}").exists()
    assert (out_dir / "ww3_grid.inp").exists()

    # Coordinates from the reconstructed grid still match the original t-points.
    xcoord = np.loadtxt(out_dir / f"{alias}_x.inp")
    ycoord = np.loadtxt(out_dir / f"{alias}_y.inp")
    assert np.allclose(xcoord, topo._grid.tlon.data)
    assert np.allclose(ycoord, topo._grid.tlat.data)


# --- WW3 time steps ---------------------------------------------------------


@pytest.mark.parametrize("min_dx", [2_000.0, 5_000.0, 10_952.0, 50_000.0])
@pytest.mark.parametrize("cpl_dt", [900.0, 1800.0, 3600.0])
def test_ww3_timesteps_from_spacing_properties(min_dx, cpl_dt):
    """The steps must respect propagation CFL on the smallest cell, and dtmax
    must land exactly on the coupling time."""
    from mom6_forge.topo import (
        WW3_CFL_SAFETY,
        WW3_F1,
        WW3_MAX_DT_RATIO,
        ww3_timesteps_from_spacing,
    )

    dt = ww3_timesteps_from_spacing(min_dx, cpl_dt)
    cg_max = 9.81 / (
        4.0 * np.pi * WW3_F1
    )  # deep-water group velocity, lowest frequency

    assert dt["dtcfl"] <= WW3_CFL_SAFETY * min_dx / cg_max * (1 + 1e-12)
    n_global = cpl_dt / dt["dtmax"]
    assert n_global == pytest.approx(round(n_global))
    n_sub = dt["dtmax"] / dt["dtcfl"]
    assert n_sub == pytest.approx(round(n_sub))
    assert 1 <= round(n_sub) <= WW3_MAX_DT_RATIO
    assert dt["dtcfli"] == dt["dtcfl"]
    assert dt["dtmin"] == pytest.approx(min(10.0, dt["dtcfl"] / 10.0))


def test_ww3_timesteps_from_spacing_coarse_grid_takes_one_step():
    """When the CFL limit exceeds the coupling interval, one global step and one
    propagation step cover it."""
    from mom6_forge.topo import ww3_timesteps_from_spacing

    dt = ww3_timesteps_from_spacing(100_000.0, 1800.0)
    assert dt["dtmax"] == dt["dtcfl"] == 1800.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(min_dx=0.0, cpl_dt=1800.0), "min_dx"),
        (dict(min_dx=5000.0, cpl_dt=-1.0), "cpl_dt"),
        (dict(min_dx=5000.0, cpl_dt=1800.0, max_ratio=0), "max_ratio"),
    ],
)
def test_ww3_timesteps_from_spacing_rejects_bad_input(kwargs, match):
    from mom6_forge.topo import ww3_timesteps_from_spacing

    with pytest.raises(ValueError, match=match):
        ww3_timesteps_from_spacing(**kwargs)


def test_write_ww3_input_time_steps_follow_the_grid(get_rect_topo_without_vc, tmp_path):
    """ww3_grid.inp carries the steps derived from this grid's smallest cell and
    the given coupling interval, not fixed values."""
    topo = get_rect_topo_without_vc
    grid = topo._grid
    smallest = min(float(np.min(grid.dxt)), float(np.min(grid.dyt)))
    assert topo.ww3_min_grid_spacing() == pytest.approx(smallest, rel=1e-3)

    dt = topo.ww3_timesteps(3600.0)
    topo.write_ww3_input(tmp_path, grid_alias=grid.name, cpl_dt=3600.0)
    text = (tmp_path / "ww3_grid.inp").read_text()
    assert (
        f"  {dt['dtmax']:.2f}  {dt['dtcfl']:.2f}  {dt['dtcfli']:.2f}  {dt['dtmin']:.2f}"
        in text
    )


def test_write_ww3_input_enables_langmuir_mixing(get_rect_topo_without_vc, tmp_path):
    """Without &LMPN, WW3 never accumulates the surface-layer Stokes drift and the
    coupler's Langmuir multiplier is 1 everywhere."""
    topo = get_rect_topo_without_vc
    topo.write_ww3_input(tmp_path, grid_alias=topo._grid.name)
    text = (tmp_path / "ww3_grid.inp").read_text()
    assert "&LMPN" in text
    assert "LMPENABLED = T" in text


def test_write_ww3_input_sets_ice_dissipation(get_rect_topo_without_vc, tmp_path):
    """Without &SIC4, WW3 uses IC4 method 1, whose first coefficient the CESM cap
    fills with the ice thickness: waves die within a cell of thin CICE ice, and
    DICE ice (no thickness) does not damp them at all. Write what CESM's own
    grids use instead, inside the namelist section."""
    topo = get_rect_topo_without_vc
    topo.write_ww3_input(tmp_path, grid_alias=topo._grid.name)
    namelists = (tmp_path / "ww3_grid.inp").read_text().split("END OF NAMELISTS")[0]

    # grid_inp.wgx3v7.260527 in CESM inputdata.
    assert WW3_IC4_METHOD == 10
    assert f"&SIC4\n  IC4METHOD = {WW3_IC4_METHOD}\n/" in namelists
    assert "&MISC\n  ICNUMERICS = T\n/" in namelists
