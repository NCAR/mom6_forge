import pytest
from mom6_forge._supergrid import *
from mom6_forge.grid import Grid
import numpy as np
import xarray as xr
from utils import on_cisl_machine

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
    non_cyclic_sg.to_esmf_mesh(str(path), mask="all_unmasked")
    return xr.open_dataset(path)


@pytest.fixture
def cyclic_mesh(cyclic_sg, tmp_path):
    path = tmp_path / "cyclic.nc"
    cyclic_sg.to_esmf_mesh(str(path), mask="all_unmasked")
    return xr.open_dataset(path)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ([0, 10], [0, 10]),
    ],
)
def test_even_spacing_hgrid_init_to_and_from(lat, lon):
    grid = RectilinearCartesianSupergrid.from_extents(
        lon[0], lon[1] - lon[0], lat[0], lat[1] - lat[0], 0.05
    )
    assert isinstance(
        grid,
        RectilinearCartesianSupergrid,
    )
    ds = grid.to_ds()
    grid2 = SupergridBase.from_ds(ds)
    assert grid == grid2


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ([0, 10], [0, 10]),
    ],
)
def test_to_and_from_no_grid_type(lat, lon):
    grid = RectilinearCartesianSupergrid.from_extents(
        lon[0], lon[1] - lon[0], lat[0], lat[1] - lat[0], 0.05
    )
    grid2 = SupergridBase._init_from_xy(grid.x, grid.y)
    ds = grid2.to_ds()
    grid2 = SupergridBase.from_ds(ds)
    assert grid == grid2


# --- ProjectedSupergrid tests ---


def test_projected_supergrid_from_crs():
    """from_crs returns a valid ProjectedSupergrid with correct array shapes."""
    resolution_m = 50_000
    x_min, x_max = -500_000, 500_000
    y_min, y_max = -500_000, 500_000
    sg = ProjectedSupergrid.from_crs(
        "EPSG:3995", x_min, x_max, y_min, y_max, resolution_m
    )
    assert isinstance(sg, ProjectedSupergrid)
    nx = int((x_max - x_min) / resolution_m)
    ny = int((y_max - y_min) / resolution_m)
    assert sg.x.shape == (2 * ny + 1, 2 * nx + 1)
    assert sg.y.shape == sg.x.shape
    assert sg.dx.shape == (2 * ny + 1, 2 * nx)
    assert sg.dy.shape == (2 * ny, 2 * nx + 1)
    assert sg.area.shape == (2 * ny, 2 * nx)
    assert np.all(sg.area > 0)


def test_projected_supergrid_from_center():
    """from_center returns a valid ProjectedSupergrid centred near the given location."""
    center_lat, center_lon = 40.0, -70.0
    width_m = height_m = 200_000
    resolution_m = 50_000
    sg = ProjectedSupergrid.from_center(
        center_lat, center_lon, width_m, height_m, resolution_m
    )
    assert isinstance(sg, ProjectedSupergrid)
    nx = int(width_m / resolution_m)
    ny = int(height_m / resolution_m)
    assert sg.x.shape == (2 * ny + 1, 2 * nx + 1)
    # Centre of the grid should be close to the requested geographic point
    centre_lat = sg.y[sg.y.shape[0] // 2, sg.y.shape[1] // 2]
    centre_lon = sg.x[sg.x.shape[0] // 2, sg.x.shape[1] // 2]
    assert abs(centre_lat - center_lat) < 1.0
    assert abs(centre_lon - center_lon) < 1.0


def test_projected_supergrid_from_center_rotated():
    """from_center with angle_deg produces a rotated grid distinct from the unrotated one."""
    center_lat, center_lon = 40.0, -70.0
    width_m = height_m = 200_000
    resolution_m = 50_000
    sg_0 = ProjectedSupergrid.from_center(
        center_lat, center_lon, width_m, height_m, resolution_m, angle_deg=0.0
    )
    sg_45 = ProjectedSupergrid.from_center(
        center_lat, center_lon, width_m, height_m, resolution_m, angle_deg=45.0
    )
    # Shapes must be identical
    assert sg_45.x.shape == sg_0.x.shape
    # Rotation should shift the corner coordinates
    assert not np.allclose(sg_0.x, sg_45.x)
    assert not np.allclose(sg_0.y, sg_45.y)
    # Centre node should still be near the requested point regardless of rotation
    cy = sg_45.y[sg_45.y.shape[0] // 2, sg_45.y.shape[1] // 2]
    cx = sg_45.x[sg_45.x.shape[0] // 2, sg_45.x.shape[1] // 2]
    assert abs(cy - center_lat) < 1.0
    assert abs(cx - center_lon) < 1.0
    assert ((sg_45.angle_dx + np.deg2rad(45.0)) < 0.1).all()


def test_uniform_spherical_supergrid():
    nx, ny = 10, 10
    sg = UniformSphericalSupergrid.from_extents(
        lon_min=0.0, len_x=10.0, lat_min=40.0, len_y=10.0, nx=nx, ny=ny
    )
    assert isinstance(sg, UniformSphericalSupergrid)


# ---------------------------------------------------------------------------
# Non-cyclic tests
# ---------------------------------------------------------------------------


def test_non_cyclic_global_attrs(non_cyclic_mesh):
    assert non_cyclic_mesh.attrs["gridType"] == "unstructured mesh"
    assert non_cyclic_mesh.attrs["grid_topology"] == "non_cyclic"
    assert "date_created" in non_cyclic_mesh.attrs


def test_non_cyclic_node_count(non_cyclic_mesh):
    assert non_cyclic_mesh.sizes["nodeCount"] == (8 + 1) * (6 + 1)


def test_non_cyclic_element_count(non_cyclic_mesh):
    assert non_cyclic_mesh.sizes["elementCount"] == 8 * 6


def test_non_cyclic_num_element_conn_all_four(non_cyclic_mesh):
    assert (non_cyclic_mesh["numElementConn"].values == 4).all()


def test_non_cyclic_element_conn_in_bounds(non_cyclic_mesh):
    conn = non_cyclic_mesh["elementConn"].values
    nnodes = non_cyclic_mesh.sizes["nodeCount"]
    i0 = non_cyclic_mesh["elementConn"].attrs["start_index"]
    assert conn.min() >= i0
    assert conn.max() <= nnodes + i0 - 1


def test_non_cyclic_element_area_positive(non_cyclic_mesh):
    assert (non_cyclic_mesh["elementArea"].values > 0).all()


def test_non_cyclic_coord_units_preserved(non_cyclic_sg, non_cyclic_mesh):
    assert non_cyclic_mesh["nodeCoords"].attrs["units"] == non_cyclic_sg.axis_units


def test_non_cyclic_all_unmasked(non_cyclic_mesh):
    assert (non_cyclic_mesh["elementMask"].values == 1).all()


# ---------------------------------------------------------------------------
# Cyclic tests
# ---------------------------------------------------------------------------


def test_cyclic_global_attrs(cyclic_mesh):
    assert cyclic_mesh.attrs["grid_topology"] == "cyclic"


def test_cyclic_node_count_drops_wrap_column(cyclic_mesh):
    assert cyclic_mesh.sizes["nodeCount"] == 8 * (6 + 1)


def test_cyclic_element_count(cyclic_mesh):
    assert cyclic_mesh.sizes["elementCount"] == 8 * 6


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
    nnodes = cyclic_mesh.sizes["nodeCount"]
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
def test_roundtrip(sg_fixture, label, request, tmp_path):
    sg = request.getfixturevalue(sg_fixture)
    path = tmp_path / f"{label}.nc"
    sg.to_esmf_mesh(str(path), mask="all_unmasked")
    sg2 = SupergridBase.reconstruct_from_esmf_mesh(str(path))

    # corner (q) and center (t) coords are stored verbatim — must be exact
    np.testing.assert_array_equal(sg2.x[::2, ::2], sg.x[::2, ::2])
    np.testing.assert_array_equal(sg2.y[::2, ::2], sg.y[::2, ::2])
    np.testing.assert_array_equal(sg2.x[1::2, 1::2], sg.x[1::2, 1::2])
    np.testing.assert_array_equal(sg2.y[1::2, 1::2], sg.y[1::2, 1::2])

    # v-points (even rows, odd cols) and u-points (odd rows, even cols) — exact
    np.testing.assert_array_equal(sg2.x[::2, 1::2], sg.x[::2, 1::2])
    np.testing.assert_array_equal(sg2.y[::2, 1::2], sg.y[::2, 1::2])
    np.testing.assert_array_equal(sg2.x[1::2, ::2], sg.x[1::2, ::2])
    np.testing.assert_array_equal(sg2.y[1::2, ::2], sg.y[1::2, ::2])

    # shape
    assert sg2.x.shape == sg.x.shape
    assert sg2.y.shape == sg.y.shape

    # metrics recomputed from coords — approximate
    np.testing.assert_allclose(sg2.dx, sg.dx, rtol=1e-6)
    np.testing.assert_allclose(sg2.dy, sg.dy, rtol=1e-6)
    np.testing.assert_allclose(sg2.area, sg.area, rtol=1e-6)

    # axis units preserved
    assert sg2.axis_units == sg.axis_units


# ---------------------------------------------------------------------------
# Tripolar tests (require CISL / GLADE access)
# ---------------------------------------------------------------------------

_TX2_3V3_HGRID = (
    "/glade/campaign/cesm/cesmdata/inputdata/ocn/mom/tx2_3v3/ocean_hgrid_250930.nc"
)
_TX2_3V3_ESMF_MESH = "/glade/campaign/cesm/cesmdata/inputdata/ocn/mom/tx2_3v3/ESMF_mesh_tx2_3v3_260305_cdf5.nc"


@pytest.fixture
def tripolar_sg():
    if not on_cisl_machine():
        pytest.skip("Requires CISL/GLADE access")
    return Grid.from_supergrid(_TX2_3V3_HGRID).supergrid


@pytest.fixture
def tripolar_mesh(tripolar_sg, tmp_path):
    path = tmp_path / "tripolar.nc"
    tripolar_sg.to_esmf_mesh(str(path), mask="all_unmasked")
    return xr.open_dataset(path)


def test_is_tripolar_from_file(tripolar_sg):
    assert tripolar_sg.is_tripolar is True


def test_tripolar_mesh_global_attrs(tripolar_mesh):
    assert tripolar_mesh.attrs["gridType"] == "unstructured mesh"
    assert tripolar_mesh.attrs["grid_topology"] == "tripolar"


def test_tripolar_node_count(tripolar_sg, tripolar_mesh):
    ny, nx = tripolar_sg.x[1::2, 1::2].shape
    expected_nnodes = nx * (ny + 1) - (nx // 2 - 1)
    assert tripolar_mesh.sizes["nodeCount"] == expected_nnodes


def test_tripolar_element_count(tripolar_sg, tripolar_mesh):
    ny, nx = tripolar_sg.x[1::2, 1::2].shape
    assert tripolar_mesh.sizes["elementCount"] == ny * nx


def test_tripolar_element_conn_in_bounds(tripolar_mesh):
    conn = tripolar_mesh["elementConn"].values
    nnodes = tripolar_mesh.sizes["nodeCount"]
    i0 = int(tripolar_mesh["elementConn"].attrs["start_index"])
    assert conn.min() >= i0
    assert conn.max() <= nnodes + i0 - 1


def test_tripolar_element_area_positive(tripolar_mesh):
    assert (tripolar_mesh["elementArea"].values > 0).all()


@pytest.fixture
def reference_tripolar_mesh():
    if not on_cisl_machine():
        pytest.skip("Requires CISL/GLADE access")
    return xr.open_dataset(_TX2_3V3_ESMF_MESH)


def test_tripolar_mesh_matches_reference(tripolar_mesh, reference_tripolar_mesh):
    """Mesh written from tx2_3v3 supergrid should match the reference ESMF mesh."""
    assert (
        tripolar_mesh.sizes["nodeCount"] == reference_tripolar_mesh.sizes["nodeCount"]
    )
    assert (
        tripolar_mesh.sizes["elementCount"]
        == reference_tripolar_mesh.sizes["elementCount"]
    )

    # Sort nodes by (lat, lon) for order-independent comparison
    our_nodes = tripolar_mesh["nodeCoords"].values
    ref_nodes = reference_tripolar_mesh["nodeCoords"].values
    our_idx = np.lexsort((our_nodes[:, 0], our_nodes[:, 1]))
    ref_idx = np.lexsort((ref_nodes[:, 0], ref_nodes[:, 1]))
    np.testing.assert_allclose(our_nodes[our_idx], ref_nodes[ref_idx], atol=1e-6)

    # Total area should match (per-cell values may differ due to different generation
    # code, but the global sum should agree to within floating-point tolerance)
    our_area_sum = tripolar_mesh["elementArea"].values.sum()
    ref_area_sum = reference_tripolar_mesh["elementArea"].values.sum()
    np.testing.assert_allclose(our_area_sum, ref_area_sum, rtol=1e-4)


def test_tripolar_roundtrip(tripolar_sg, tmp_path):
    """reconstruct_from_esmf_mesh should recover corner and center coords exactly for tx2_3v3."""
    path = tmp_path / "tripolar.nc"
    tripolar_sg.to_esmf_mesh(str(path), mask="all_unmasked")
    sg2 = SupergridBase.reconstruct_from_esmf_mesh(str(path))

    assert sg2.x.shape == tripolar_sg.x.shape
    assert sg2.y.shape == tripolar_sg.y.shape

    # corner (q) and center (t) coords stored verbatim — must be exact
    np.testing.assert_array_equal(sg2.x[::2, ::2], tripolar_sg.x[::2, ::2])
    np.testing.assert_array_equal(sg2.y[::2, ::2], tripolar_sg.y[::2, ::2])
    np.testing.assert_array_equal(sg2.x[1::2, 1::2], tripolar_sg.x[1::2, 1::2])
    np.testing.assert_array_equal(sg2.y[1::2, 1::2], tripolar_sg.y[1::2, 1::2])
