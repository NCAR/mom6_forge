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
from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from utils import on_cisl_machine
import os


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
    res = Grid.get_bounding_boxes_of_rectangular_grid(grid)
    assert "east" in res.keys()
    assert "west" in res.keys()
    assert "north" in res.keys()
    assert "south" in res.keys()
    assert "lat_min" in res["east"].keys()


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
