import pytest
from mom6_forge._supergrid import *
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def non_cyclic_sg():
    return UniformSphericalSupergrid.from_extents(
        lon_min=0.0,
        len_x=90.0,
        lat_min=-30.0,
        len_y=60.0,
        nx=8,
        ny=6,
    )


@pytest.fixture
def cyclic_sg():
    return UniformSphericalSupergrid.from_extents(
        lon_min=0.0,
        len_x=360.0,
        lat_min=-30.0,
        len_y=60.0,
        nx=8,
        ny=6,
    )


@pytest.fixture
def non_cyclic_mesh(non_cyclic_sg, tmp_path):
    path = tmp_path / "non_cyclic.nc"
    non_cyclic_sg.to_esmf_mesh(str(path))
    return xr.open_dataset(path)


@pytest.fixture
def cyclic_mesh(cyclic_sg, tmp_path):
    path = tmp_path / "cyclic.nc"
    cyclic_sg.to_esmf_mesh(str(path))
    return xr.open_dataset(path)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ([0, 10], [0, 10]),
    ],
)
def test_even_spacing_hgrid(lat, lon):
    assert isinstance(
        RectilinearCartesianSupergrid(
            lon[0], lon[1] - lon[0], lat[0], lat[1] - lat[0], 0.05
        ),
        RectilinearCartesianSupergrid,
    )


def test_uniform_spherical_supergrid():
    nx, ny = 10, 10
    sg = UniformSphericalSupergrid.from_extents(
        lon_min=0.0, len_x=10.0, lat_min=40.0, len_y=10.0, nx=nx, ny=ny
    )
    assert isinstance(sg, UniformSphericalSupergrid)


def make_mom6_mask(ny, nx, frac_land=0.25):
    """MOM6-convention mask (1=ocean, 0=land). Land in bottom-left corner."""
    mask = np.ones((ny, nx), dtype=np.int32)
    land_ny = max(1, int(ny * frac_land))
    land_nx = max(1, int(nx * frac_land))
    mask[:land_ny, :land_nx] = 0
    return mask


# ---------------------------------------------------------------------------
# Non-cyclic tests
# ---------------------------------------------------------------------------


def test_non_cyclic_global_attrs(non_cyclic_mesh):
    assert non_cyclic_mesh.attrs["gridType"] == "unstructured mesh"
    assert non_cyclic_mesh.attrs["grid_topology"] == "non_cyclic"
    assert "date_created" in non_cyclic_mesh.attrs


def test_non_cyclic_node_count(non_cyclic_mesh):
    assert non_cyclic_mesh.dims["nodeCount"] == (8 + 1) * (6 + 1)


def test_non_cyclic_element_count(non_cyclic_mesh):
    assert non_cyclic_mesh.dims["elementCount"] == 8 * 6


def test_non_cyclic_num_element_conn_all_four(non_cyclic_mesh):
    assert (non_cyclic_mesh["numElementConn"].values == 4).all()


def test_non_cyclic_element_conn_in_bounds(non_cyclic_mesh):
    conn = non_cyclic_mesh["elementConn"].values
    nnodes = non_cyclic_mesh.dims["nodeCount"]
    i0 = non_cyclic_mesh["elementConn"].attrs["start_index"]
    assert conn.min() >= i0
    assert conn.max() <= nnodes + i0 - 1


def test_non_cyclic_node_ids_sequential(non_cyclic_mesh):
    ids = non_cyclic_mesh["nodeIds"].values
    assert ids[0] == 1
    assert np.all(np.diff(ids) == 1)


def test_non_cyclic_element_ids_sequential(non_cyclic_mesh):
    ids = non_cyclic_mesh["elementIds"].values
    assert ids[0] == 1
    assert np.all(np.diff(ids) == 1)


def test_non_cyclic_element_area_positive(non_cyclic_mesh):
    assert (non_cyclic_mesh["elementArea"].values > 0).all()


def test_non_cyclic_coord_units_preserved(non_cyclic_sg, non_cyclic_mesh):
    assert non_cyclic_mesh["nodeCoords"].attrs["units"] == non_cyclic_sg.axis_units


def test_non_cyclic_no_mask_when_not_provided(non_cyclic_mesh):
    assert "elementMask" not in non_cyclic_mesh


# ---------------------------------------------------------------------------
# Cyclic tests
# ---------------------------------------------------------------------------


def test_cyclic_global_attrs(cyclic_mesh):
    assert cyclic_mesh.attrs["grid_topology"] == "cyclic"


def test_cyclic_node_count_drops_wrap_column(cyclic_mesh):
    assert cyclic_mesh.dims["nodeCount"] == 8 * (6 + 1)


def test_cyclic_element_count(cyclic_mesh):
    assert cyclic_mesh.dims["elementCount"] == 8 * 6


def test_cyclic_connectivity_wraps_last_column(cyclic_mesh):
    """Last element in each row should wrap back to column-0 nodes."""
    nx = 8
    conn = cyclic_mesh["elementConn"].values
    i0 = cyclic_mesh["elementConn"].attrs["start_index"]
    last_elem = conn[nx - 1] - i0  # 0-based
    assert last_elem[1] % nx == 0  # lr wraps to col 0
    assert last_elem[2] % nx == 0  # ur wraps to col 0


def test_cyclic_element_conn_in_bounds(cyclic_mesh):
    conn = cyclic_mesh["elementConn"].values
    nnodes = cyclic_mesh.dims["nodeCount"]
    i0 = cyclic_mesh["elementConn"].attrs["start_index"]
    assert conn.min() >= i0
    assert conn.max() <= nnodes + i0 - 1


def test_cyclic_element_area_positive(cyclic_mesh):
    assert (cyclic_mesh["elementArea"].values > 0).all()


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sg_fixture,label",
    [
        ("non_cyclic_sg", "non_cyclic"),
        ("cyclic_sg", "cyclic"),
    ],
)
def test_roundtrip_corner_coords(sg_fixture, label, request, tmp_path):
    sg = request.getfixturevalue(sg_fixture)
    path = tmp_path / f"{label}.nc"
    sg.to_esmf_mesh(str(path))
    sg2 = SupergridBase.from_esmf_mesh(str(path))
    np.testing.assert_allclose(sg2.x[::2, ::2], sg.x[::2, ::2], atol=1e-10)
    np.testing.assert_allclose(sg2.y[::2, ::2], sg.y[::2, ::2], atol=1e-10)


@pytest.mark.parametrize(
    "sg_fixture,label",
    [
        ("non_cyclic_sg", "non_cyclic"),
        ("cyclic_sg", "cyclic"),
    ],
)
def test_roundtrip_center_coords(sg_fixture, label, request, tmp_path):
    sg = request.getfixturevalue(sg_fixture)
    path = tmp_path / f"{label}.nc"
    sg.to_esmf_mesh(str(path))
    sg2 = SupergridBase.from_esmf_mesh(str(path))
    np.testing.assert_allclose(sg2.x[1::2, 1::2], sg.x[1::2, 1::2], atol=1e-10)
    np.testing.assert_allclose(sg2.y[1::2, 1::2], sg.y[1::2, 1::2], atol=1e-10)


@pytest.mark.parametrize(
    "sg_fixture,label",
    [
        ("non_cyclic_sg", "non_cyclic"),
        ("cyclic_sg", "cyclic"),
    ],
)
def test_roundtrip_supergrid_shape(sg_fixture, label, request, tmp_path):
    sg = request.getfixturevalue(sg_fixture)
    path = tmp_path / f"{label}.nc"
    sg.to_esmf_mesh(str(path))
    sg2 = SupergridBase.from_esmf_mesh(str(path))
    assert sg2.x.shape == sg.x.shape
    assert sg2.y.shape == sg.y.shape


@pytest.mark.parametrize(
    "sg_fixture,label",
    [
        ("non_cyclic_sg", "non_cyclic"),
        ("cyclic_sg", "cyclic"),
    ],
)
def test_roundtrip_metrics(sg_fixture, label, request, tmp_path):
    sg = request.getfixturevalue(sg_fixture)
    path = tmp_path / f"{label}.nc"
    sg.to_esmf_mesh(str(path))
    sg2 = SupergridBase.from_esmf_mesh(str(path))
    np.testing.assert_allclose(sg2.dx, sg.dx, rtol=1e-6)
    np.testing.assert_allclose(sg2.dy, sg.dy, rtol=1e-6)
    np.testing.assert_allclose(sg2.area, sg.area, rtol=1e-6)


@pytest.mark.parametrize(
    "sg_fixture,label",
    [
        ("non_cyclic_sg", "non_cyclic"),
        ("cyclic_sg", "cyclic"),
    ],
)
def test_roundtrip_axis_units(sg_fixture, label, request, tmp_path):
    sg = request.getfixturevalue(sg_fixture)
    path = tmp_path / f"{label}.nc"
    sg.to_esmf_mesh(str(path))
    sg2 = SupergridBase.from_esmf_mesh(str(path))
    assert sg2.axis_units == sg.axis_units
