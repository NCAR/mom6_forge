import pytest
import tempfile
import socket
import numpy as np
import xarray as xr
import pytest
import pytest
import tempfile
import socket
import numpy as np
import xarray as xr
import pytest
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge._supergrid import SupergridBase, ProjectedSupergrid
from utils import on_cisl_machine
import os


def _rotated_supergrid_grid(
    rotation_deg, name="curv", nx=10, ny=10, d=0.1, center=10.0
):
    """Build a Grid whose supergrid is uniformly rotated, i.e. curvilinear."""
    theta = np.deg2rad(rotation_deg)
    nxp, nyp = 2 * nx + 1, 2 * ny + 1
    I, J = np.meshgrid((np.arange(nxp) - nx) * d, (np.arange(nyp) - ny) * d)
    x = center + I * np.cos(theta) - J * np.sin(theta)
    y = center + I * np.sin(theta) + J * np.cos(theta)
    sg = SupergridBase._init_from_xy(x, y)
    return Grid.from_supergrid_ds(sg.to_ds(), name=name)


def _sheared_supergrid_grid(
    shear, name="sheared", nx=10, ny=10, d=0.1, x0=280.0, y0=10.0
):
    """Build a Grid where each column's longitude drifts by ``shear`` deg per
    supergrid row, i.e. nearly- but not-quite rectangular."""
    nxp, nyp = 2 * nx + 1, 2 * ny + 1
    jj, ii = np.meshgrid(np.arange(nyp), np.arange(nxp), indexing="ij")
    x = x0 + ii * d + jj * shear
    y = y0 + jj * d
    sg = SupergridBase._init_from_xy(x, y)
    return Grid.from_supergrid_ds(sg.to_ds(), name=name)


def test_is_tripolar():
    """Check if Grid.is_tripolar() and .is_cyclic_x() methods work correctly for different MOM grids."""

    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")

    ds = xr.open_dataset(
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/gx1v6/ocean_hgrid_230424.nc"
    )
    assert not Grid.is_tripolar(ds)
    assert Grid.is_cyclic_x(ds)

    ds = xr.open_dataset(
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/tx0.66v1/ocean_hgrid_180829.nc"
    )
    assert Grid.is_tripolar(ds)
    assert Grid.is_cyclic_x(ds)

    ds = xr.open_dataset(
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/tx2_3v2/ocean_hgrid_221123.nc"
    )
    assert Grid.is_tripolar(ds)
    assert Grid.is_cyclic_x(ds)

    ds = xr.open_dataset(
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/tx0.25v1/ocean_hgrid.nc"
    )
    assert Grid.is_tripolar(ds)
    assert Grid.is_cyclic_x(ds)


def test_regional_grid():
    """Test the creation of a regional grid object from scratch."""

    # attempt to create a regional grid object from scratch
    grid = Grid(
        nx=100,  # Number of grid points in x direction
        ny=50,  # Number of grid points in y direction
        lenx=10.0,  # grid length in x direction, e.g., 360.0 (degrees)
        leny=5.0,  # grid length in y direction
        cyclic_x=False,  # non-reentrant, rectangular domain
    )

    # create a corresponding bathymetry object
    topo = Topo(grid, min_depth=10.0)

    # set the bathymetry to a flat bottom
    topo.set_flat(D=2000.0)

    # write the bathymetry to a netcdf file
    with tempfile.TemporaryDirectory() as tmpdirname:

        # write horizontal grid to netcdf file
        grid.write_supergrid(tmpdirname + "/ocean_hgrid_1.nc")

        # write topo to netcdf file
        topo.write_topo(tmpdirname + "/ocean_topog_1.nc")

        # write cice grid file
        topo.write_cice_grid(tmpdirname + "/cice_grid_1.nc")

        # write SCRIP grid file
        topo.write_scrip_grid(tmpdirname + "/SCRIP_grid_1.nc")

        # ESMF mesh file
        topo.write_esmf_mesh(tmpdirname + "/ESMF_mesh_1.nc")


def test_global_grid():
    """Test the creation of a global grid object from scratch."""

    # attempt to create a global grid object with lenx = 10.0 : should raise an error
    with pytest.raises(AssertionError):
        grid = Grid(
            nx=100,  # Number of grid points in x direction
            ny=50,  # Number of grid points in y direction
            lenx=10.0,  # grid length in x direction, e.g., 360.0 (degrees)
            leny=180.0,  # grid length in y direction
            cyclic_x=True,  # reentrant, global domain
        )

    # Noew attempt to create a global grid object with lenx = 360.0: should work
    grid = Grid(
        nx=100,  # Number of grid points in x direction
        ny=50,  # Number of grid points in y direction
        lenx=360.0,  # grid length in x direction, e.g., 360.0 (degrees)
        leny=180.0,  # grid length in y direction
        cyclic_x=True,  # reentrant, global domain
    )

    # create a corresponding bathymetry object
    topo = Topo(grid, min_depth=10.0)

    # set the bathymetry to a flat bottom
    topo.set_flat(D=2000.0)

    # try spoon bathymetry
    topo.set_spoon(1000.0, 100.0, expdecay=1e8)

    # try bowl bathymetry
    topo.set_bowl(100.0, 0.0, expdecay=1e8)

    # confirm that all edge points have tmask = 0
    assert (topo.tmask[0, :] == 0).all()
    assert (topo.tmask[-1, :] == 0).all()
    assert (topo.tmask[:, -1] == 0).all()
    assert (topo.tmask[:, -1] == 0).all()

    # confirm the middle point has tmask = 1
    assert topo.tmask[25, 50] == 1


def test_from_file():
    """Test the creation of a grid object from a supergrid file."""

    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")

    print("Running test_from_file")
    supergrid_path = (
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/tx2_3v2/ocean_hgrid_221123.nc"
    )

    topo_path = "/glade/p/cesmdata/inputdata/ocn/mom/tx2_3v2/ocean_topog_230413.nc"

    grid = Grid.from_supergrid(supergrid_path)
    topo = Topo.from_topo_file(grid, topo_path)

    # write the bathymetry to a netcdf file
    with tempfile.TemporaryDirectory() as tmpdirname:

        # write horizontal grid to netcdf file
        grid.write_supergrid(tmpdirname + "/ocean_hgrid_2.nc")

        ds_orig = xr.open_dataset(supergrid_path)
        ds_new = xr.open_dataset(tmpdirname + "/ocean_hgrid_2.nc")

        assert (ds_orig.x == ds_new.x).all()
        assert (ds_orig.y == ds_new.y).all()
        assert (ds_orig.dx == ds_new.dx).all()
        assert (ds_orig.dy == ds_new.dy).all()

        topo.write_topo(tmpdirname + "/ocean_topog_2.nc")

        ds_orig = xr.open_dataset(topo_path)
        ds_new = xr.open_dataset(tmpdirname + "/ocean_topog_2.nc")

        assert (ds_orig["geolon"].data == ds_new["x"].data).all()


def test_equatorial_refinement():
    """Test equatorial refinement of the grid and confirm grid metrics are accurately updated."""

    grid = Grid(
        nx=180,  # Number of grid points in x direction
        ny=80,  # Number of grid points in y direction
        lenx=360.0,  # grid length in x direction, e.g., 360.0 (degrees)
        leny=160,  # grid length in y direction
        cyclic_x=True,  # reentrant, spherical domain
        ystart=-80,  # start/end 10 degrees above/below poles to avoid singularity
    )

    # First, define a refinement function along longitutes:
    from scipy import interpolate

    f = 0.5
    r_y = [-80, -30, -10, 10, 30, 80]  # transition latitudes
    r_f = [1, 1, f, f, 1, 1]  # inverse refinement factors at transition latitudes
    interp_func = interpolate.interp1d(r_y, r_f, kind=3)
    r_f_mapped = interp_func(grid.supergrid.y[1:, 0])
    r_f_mapped = np.where(r_f_mapped < 1.0, r_f_mapped, 1.0)
    r_f_mapped = np.where(r_f_mapped > f, r_f_mapped, f)

    # now, apply the refinement function to the grid
    super_dy = grid.supergrid.y[1:, 0] - grid.supergrid.y[:-1, 0]
    super_dy_new = super_dy.mean() * r_f_mapped / r_f_mapped.mean()  # normalize
    super_y_new = grid.supergrid.y[:, 0].copy()
    super_y_new[1:] = grid.supergrid.y[0, 0] + super_dy_new.cumsum()
    xdat, ydat = np.meshgrid(grid.supergrid.x[0, :], super_y_new)

    # update the supergrid
    grid.update_supergrid(xdat, ydat)

    # check that the dyt grid metric is accurately updated after the refinement and supergrid update
    assert np.isclose(grid.dyt[0, 0], 2.0 * grid.dyt[40, 0], rtol=1e-06)


if __name__ == "__main__":
    test_is_tripolar()
    test_regional_grid()
    test_global_grid()
    test_from_file()
    test_equatorial_refinement()


def test_get_rectangular_segment_info(get_rect_grid):
    grid = get_rect_grid
    res = Grid.get_bounding_boxes(grid)
    assert "east" in res.keys()
    assert "west" in res.keys()
    assert "north" in res.keys()
    assert "south" in res.keys()
    assert "lat_min" in res["east"].keys()


def test_get_bounding_boxes_tight_for_seam_crossing_edge():
    """An edge that crosses a seam without needing the full circle should get a
    tight, contiguous range (not the whole globe); an edge/box that genuinely
    surrounds a pole (like the full "ic" domain here) still can't be tightened
    and should fall back to the honest full range."""
    sg = ProjectedSupergrid.from_crs(
        "EPSG:3995", -300_000, 300_000, -300_000, 300_000, resolution_m=100_000
    )
    boxes = Grid.get_bounding_boxes(sg.to_ds())

    north = boxes["north"]
    assert north["lon_max"] - north["lon_min"] < 180

    ic = boxes["ic"]
    assert ic["lon_min"] == -180.0
    assert ic["lon_max"] == 180.0


def _dateline_supergrid(lons):
    """A 2x2-degree band centred on the equator, spanning the given longitudes."""
    x, y = np.meshgrid(np.asarray(lons, dtype=float), np.arange(-5.0, 6.0, 1.0))
    return SupergridBase._init_from_xy(x, y)


def test_bounding_boxes_anchor_lon_min_to_a_real_longitude():
    """lon_min is the start of the arc, so it has to be a longitude. Where the
    box lands otherwise depends on the convention the grid was stored in, which
    is not something a caller can see."""
    cases = {
        "stored in [0, 360)": _dateline_supergrid(np.arange(350.0, 371.0)),
        "stored in [-180, 180]": _dateline_supergrid(
            ((np.arange(350.0, 371.0) + 180.0) % 360.0) - 180.0
        ),
        "away from any seam": _dateline_supergrid(np.arange(20.0, 41.0)),
        "polar cap": ProjectedSupergrid.from_crs(
            "EPSG:3995", -300_000, 300_000, -300_000, 300_000, resolution_m=100_000
        ),
    }
    for label, sg in cases.items():
        for edge, box in Grid.get_bounding_boxes(sg.to_ds()).items():
            assert -180.0 <= box["lon_min"] <= 180.0, f"{label}/{edge}"


def test_bounding_boxes_do_not_depend_on_the_stored_convention():
    """The same arc written in [0, 360) and in [-180, 180] is the same arc."""
    in_360 = Grid.get_bounding_boxes(
        _dateline_supergrid(np.arange(350.0, 371.0)).to_ds()
    )
    in_180 = Grid.get_bounding_boxes(
        _dateline_supergrid(((np.arange(350.0, 371.0) + 180.0) % 360.0) - 180.0).to_ds()
    )
    for edge in in_360:
        for key in ("lon_min", "lon_max", "crosses_antimeridian"):
            assert in_360[edge][key] == pytest.approx(
                in_180[edge][key]
            ), f"{edge}/{key}"


def test_bounding_boxes_flag_an_antimeridian_crossing():
    """A box that really does cross cannot be written with both ends in range,
    so it ends past 180 and says so rather than leaving a caller to notice."""
    crossing = Grid.get_bounding_boxes(
        _dateline_supergrid(np.arange(170.0, 191.0)).to_ds()
    )["ic"]
    assert crossing["crosses_antimeridian"] is True
    assert crossing["lon_max"] > 180.0
    assert crossing["lon_max"] - crossing["lon_min"] == pytest.approx(20.0)

    # The same width, written where it needs no crossing, is not flagged.
    plain = Grid.get_bounding_boxes(_dateline_supergrid(np.arange(20.0, 41.0)).to_ds())[
        "ic"
    ]
    assert plain["crosses_antimeridian"] is False
    assert plain["lon_max"] - plain["lon_min"] == pytest.approx(20.0)


def test_bounding_box_selects_the_intended_source_points():
    """The documented way to use a crossing box: split it at the antimeridian
    rather than comparing against a wrapped lon_max, which selects nothing."""
    box = Grid.get_bounding_boxes(_dateline_supergrid(np.arange(170.0, 191.0)).to_ds())[
        "ic"
    ]
    source_lon = np.arange(-180.0, 180.0, 1.0)

    if box["crosses_antimeridian"]:
        selected = (source_lon >= box["lon_min"]) | (
            source_lon <= box["lon_max"] - 360.0
        )
    else:
        selected = (source_lon >= box["lon_min"]) & (source_lon <= box["lon_max"])

    assert source_lon[selected].min() == -180.0
    assert set(np.round(source_lon[selected])) == set(
        np.round(((np.arange(170.0, 191.0) + 180.0) % 360.0) - 180.0)
    )


def test_slice_grid(get_rect_grid):
    grid = get_rect_grid
    sub = grid[1:, 1:]
    assert sub.tlon[0][0] == grid.tlon[0][1]


@pytest.fixture
def simple_2by2_grid():
    # Create a simple 2x2 grid for testing
    grid = Grid(
        lenx=2.0,
        leny=2.0,
        nx=2,
        ny=2,
        xstart=0.0,
        ystart=0.0,
        name="testgrid",
    )
    return grid


def test_grid_properties(simple_2by2_grid):
    grid = simple_2by2_grid
    assert grid.nx == 2
    assert grid.ny == 2
    assert grid._supergrid.lenx == 2.0
    assert grid._supergrid.leny == 2.0
    assert grid.name == "testgrid"


def test_grid_sanitize_name():
    with pytest.raises(AssertionError):
        g = Grid(lenx=2.0, leny=2.0, nx=2, ny=2, name="bad name!@#")


def test_grid_get_indices(simple_2by2_grid):
    grid = simple_2by2_grid
    # Should return a valid index for the center
    j, i = grid.get_indices(grid.tlat.values[0, 0], grid.tlon.values[0, 0])
    assert 0 <= j < grid.ny
    assert 0 <= i < grid.nx


def test_grid_is_rectangular(simple_2by2_grid):
    assert simple_2by2_grid.is_rectangular()


def test_grid_is_rectangular_false_for_curvilinear():
    # A uniformly rotated (30 deg) supergrid is curvilinear, not lat-lon.
    assert not _rotated_supergrid_grid(30.0).is_rectangular()


def test_grid_is_rectangular_uses_absolute_tolerance():
    # Each column's longitude drifts ~0.02 deg/row at lon~280. With a *relative*
    # tolerance this drift would be swamped by the large longitude magnitude and
    # the grid wrongly judged rectangular; an absolute tolerance catches it.
    grid = _sheared_supergrid_grid(shear=0.02)
    assert grid.is_rectangular(atol=1.0)  # loose absolute tol -> accepted
    assert not grid.is_rectangular(atol=1e-3)  # tight absolute tol -> rejected


def test_grid_slice(simple_2by2_grid):
    sub = simple_2by2_grid[0:1, 0:1]
    assert isinstance(sub, Grid)
    assert sub.nx == 1
    assert sub.ny == 1


def test_grid_supergrid_setter(simple_2by2_grid):
    sg = simple_2by2_grid.supergrid
    simple_2by2_grid.supergrid = sg  # Should not raise


def test_grid_to_netcdf_and_from_netcdf(tmp_path, simple_2by2_grid):
    path = tmp_path / "testgrid.nc"
    simple_2by2_grid.write_supergrid(str(path))
    assert os.path.exists(path)
    loaded = Grid.from_supergrid(path)
    assert loaded.nx == simple_2by2_grid.nx
    assert loaded.ny == simple_2by2_grid.ny
    assert loaded.name == simple_2by2_grid.name


def test_grid_rectilinear_cartesian():
    grid = Grid(lenx=10.0, leny=10.0, resolution=1.0, type="rectilinear_cartesian")
    assert isinstance(grid, Grid)
    assert grid.nx == 10
    assert grid.ny == 10


def test_grid_from_projection():
    grid = Grid.from_projection(
        "EPSG:3995", -500_000, 500_000, -500_000, 500_000, 50_000, name="arctic"
    )
    assert isinstance(grid, Grid)
    assert grid.nx == 20
    assert grid.ny == 20
    assert grid.name == "arctic"


def test_grid_from_center():
    grid = Grid.from_center(40.0, -70.0, 200_000, 200_000, 50_000, name="test")
    assert isinstance(grid, Grid)
    assert grid.nx == 4
    assert grid.ny == 4
    mid_lat = grid.tlat.values[grid.ny // 2, grid.nx // 2]
    mid_lon = grid.tlon.values[grid.ny // 2, grid.nx // 2]
    assert abs(mid_lat - 40.0) < 1.0
    assert abs(mid_lon - (-70.0)) < 1.0


# ---------------------------------------------------------------------------
# Grid.from_esmf_mesh tests
# ---------------------------------------------------------------------------


def test_grid_from_esmf_mesh_non_cyclic(tmp_path, get_rect_grid):
    mesh_path = str(tmp_path / "non_cyclic.nc")
    get_rect_grid.supergrid.to_esmf_mesh(mesh_path, mask="all_unmasked")
    grid2 = Grid.from_esmf_mesh(mesh_path)
    assert isinstance(grid2, Grid)
    assert grid2.nx == get_rect_grid.nx
    assert grid2.ny == get_rect_grid.ny
    assert not grid2.cyclic_x


def test_grid_from_esmf_mesh_cyclic(tmp_path, get_simple_global_grid):
    mesh_path = str(tmp_path / "cyclic.nc")
    get_simple_global_grid.supergrid.to_esmf_mesh(mesh_path, mask="all_unmasked")
    grid2 = Grid.from_esmf_mesh(mesh_path)
    assert isinstance(grid2, Grid)
    assert grid2.nx == get_simple_global_grid.nx
    assert grid2.ny == get_simple_global_grid.ny
    assert grid2.cyclic_x


def test_grid_from_esmf_mesh_coords_preserved(tmp_path, get_rect_grid):
    mesh_path = str(tmp_path / "coords.nc")
    get_rect_grid.supergrid.to_esmf_mesh(mesh_path, mask="all_unmasked")
    grid2 = Grid.from_esmf_mesh(mesh_path)
    np.testing.assert_allclose(grid2.tlon.values, get_rect_grid.tlon.values, atol=1e-6)
    np.testing.assert_allclose(grid2.tlat.values, get_rect_grid.tlat.values, atol=1e-6)


def test_encircles_globe_distinguishes_caps_from_bands():
    """Reaching a pole is not the same as wrapping it: a narrow polar domain's
    north edge touches 90 degrees while spanning a few degrees of longitude."""
    from mom6_forge._supergrid import ProjectedSupergrid
    from mom6_forge.grid import _encircles_globe

    cap = ProjectedSupergrid.from_crs("EPSG:3995", -1e6, 1e6, -1e6, 1e6, 100_000)
    assert _encircles_globe(cap.x)
    assert not _encircles_globe(cap.x[:, -1])  # one edge of that cap
    assert not _encircles_globe(np.linspace(0.0, 4.0, 41))
    assert not _encircles_globe(np.full(41, 4.0))
    assert not _encircles_globe(np.linspace(-170.0, 30.0, 201))


def test_bounding_boxes_stay_tight_for_a_narrow_domain_touching_the_pole():
    grid = Grid(
        nx=8,
        ny=20,
        lenx=4.0,
        leny=9.95,
        xstart=0.0,
        ystart=80.0,
        cyclic_x=False,
        name="narrow_polar",
    )
    boxes = Grid.get_bounding_boxes(grid)
    for edge in ("ic", "north", "south"):
        assert boxes[edge]["lon_min"] == pytest.approx(0.0)
        assert boxes[edge]["lon_max"] == pytest.approx(4.0)
    assert boxes["east"]["lon_min"] == pytest.approx(4.0)
    assert boxes["east"]["lon_max"] == pytest.approx(4.0)


def test_sliced_and_updated_grids_keep_their_metric_conventions():
    """__getitem__ and update_supergrid rebuild metrics, so they must reuse the
    radius and dx/dy method the grid was built with."""
    grid = Grid.from_projection(
        "EPSG:3995", -1e6, 1e6, -1e6, 1e6, 100_000, name="arctic"
    )
    assert grid.supergrid._dx_dy_calc_type == "haversine"

    sub = grid[0:6, 0:6]  # a corner, so the pole is not inside the slice
    assert np.abs(sub.supergrid.y).max() < 89.9
    assert sub.supergrid._dx_dy_calc_type == "haversine"
    assert sub.supergrid._R == grid.supergrid._R

    grid.update_supergrid(grid.supergrid.x.copy(), grid.supergrid.y.copy())
    assert grid.supergrid._dx_dy_calc_type == "haversine"


def test_encircles_globe_is_resolution_independent_for_a_2d_domain():
    """The 2D test counts a winding number, so a coarse cap is still a cap and a
    coarse near-global band is still a band. The 1D gap fallback cannot make
    that distinction, which is why 2D domains do not use it."""
    from mom6_forge._supergrid import ProjectedSupergrid
    from mom6_forge.grid import _encircles_globe

    for resolution_m in (1_000_000, 250_000, 100_000):
        cap = ProjectedSupergrid.from_crs(
            "EPSG:3995", -1e6, 1e6, -1e6, 1e6, resolution_m
        )
        assert _encircles_globe(cap.x), resolution_m

    for n in (36, 71, 351):  # 350-degree band, coarse to fine
        lon, _ = np.meshgrid(np.linspace(0.0, 350.0, n), np.linspace(-5.0, 5.0, 11))
        assert not _encircles_globe(lon), n
