"""Structural validation of all MOM6/CESM runtime files that mom6_forge generates.

These tests verify every file type written by mom6_forge has the correct NetCDF
structure (variables, dimensions, value ranges) without requiring CESM or GLADE data.
"""

import numpy as np
import pytest
import xarray as xr

from mom6_forge.chl import gen_chl_empty_dataset, interpolate_and_fill_seawifs
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.vgrid import VGrid


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def domain(tmp_path_factory):
    """Small flat-bottom domain — no GEBCO or external data needed."""
    tmp = tmp_path_factory.mktemp("cesm_output")
    grid = Grid(
        resolution=0.5,
        xstart=278.0,
        lenx=4.0,
        ystart=7.0,
        leny=3.0,
        name="ci_test",
    )
    topo = Topo(grid=grid, min_depth=10.0, git=False)
    topo.set_flat(500.0)
    vgrid = VGrid.uniform(nk=10, depth=500.0)
    return grid, topo, vgrid, tmp


@pytest.fixture(scope="module")
def written_files(domain):
    """Write all six core MOM6 files once and return their paths."""
    grid, topo, vgrid, tmp = domain
    paths = {}
    paths["hgrid"] = str(tmp / "ocean_hgrid.nc")
    paths["topog"] = str(tmp / "ocean_topog.nc")
    paths["cice"] = str(tmp / "cice_grid.nc")
    paths["esmf"] = str(tmp / "ocean_mesh.nc")
    paths["scrip"] = str(tmp / "scrip_grid.nc")
    paths["vgrid"] = str(tmp / "vgrid.nc")
    paths["vgrid_z"] = str(tmp / "vgrid_z.nc")

    grid.write_supergrid(paths["hgrid"])
    topo.write_topo(paths["topog"])
    topo.write_cice_grid(paths["cice"])
    topo.write_esmf_mesh(paths["esmf"])
    topo.write_scrip_grid(paths["scrip"])
    vgrid.write(paths["vgrid"])
    vgrid.write_z_file(paths["vgrid_z"])
    return paths


# ---------------------------------------------------------------------------
# Core file tests
# ---------------------------------------------------------------------------


def test_supergrid_written(written_files):
    assert xr.open_dataset(written_files["hgrid"]) is not None


def test_supergrid_variables(written_files):
    ds = xr.open_dataset(written_files["hgrid"])
    for var in ("x", "y", "dx", "dy", "area", "angle_dx"):
        assert var in ds, f"ocean_hgrid.nc missing variable: {var}"


def test_supergrid_no_nan(written_files):
    ds = xr.open_dataset(written_files["hgrid"])
    assert not np.any(np.isnan(ds["x"].values)), "x has NaN"
    assert not np.any(np.isnan(ds["y"].values)), "y has NaN"


def test_supergrid_dimensions(written_files, domain):
    grid, _, _, _ = domain
    ds = xr.open_dataset(written_files["hgrid"])
    # supergrid is (2*ny+1) × (2*nx+1)
    assert ds.sizes["nyp"] == 2 * grid.ny + 1
    assert ds.sizes["nxp"] == 2 * grid.nx + 1


def test_topog_written(written_files):
    assert xr.open_dataset(written_files["topog"]) is not None


def test_topog_variables(written_files):
    ds = xr.open_dataset(written_files["topog"])
    for var in ("x", "y", "depth", "depth_raw", "mask"):
        assert var in ds, f"ocean_topog.nc missing variable: {var}"


def test_topog_depth_valid(written_files):
    ds = xr.open_dataset(written_files["topog"])
    depth = ds["depth"].values
    # All ocean points (where mask=1) should have depth > 0
    mask = ds["mask"].values
    assert np.all(depth[mask == 1] > 0), "ocean cells have depth ≤ 0"


def test_topog_mask_binary(written_files):
    ds = xr.open_dataset(written_files["topog"])
    mask = ds["mask"].values
    assert set(np.unique(mask)).issubset({0, 1}), "mask has values other than 0/1"


def test_cice_grid_written(written_files):
    assert xr.open_dataset(written_files["cice"]) is not None


def test_cice_grid_variables(written_files):
    ds = xr.open_dataset(written_files["cice"])
    for var in ("ulat", "ulon", "tlat", "tlon", "htn", "hte", "angle", "anglet", "kmt"):
        assert var in ds, f"cice_grid.nc missing variable: {var}"


def test_cice_grid_kmt_binary(written_files):
    ds = xr.open_dataset(written_files["cice"])
    kmt = ds["kmt"].values
    assert set(np.unique(kmt)).issubset({0.0, 1.0}), "kmt has values other than 0/1"


def test_esmf_mesh_written(written_files):
    assert xr.open_dataset(written_files["esmf"]) is not None


def test_esmf_mesh_variables(written_files):
    ds = xr.open_dataset(written_files["esmf"])
    for var in ("nodeCoords", "centerCoords", "elementConn", "numElementConn"):
        assert var in ds, f"ocean_mesh.nc missing variable: {var}"


def test_esmf_mesh_element_count(written_files, domain):
    grid, _, _, _ = domain
    ds = xr.open_dataset(written_files["esmf"])
    expected_cells = grid.nx * grid.ny
    assert ds.sizes["elementCount"] == expected_cells, (
        f"ESMF mesh elementCount {ds.dims['elementCount']} != grid cells {expected_cells}"
    )


def test_scrip_grid_written(written_files):
    assert xr.open_dataset(written_files["scrip"]) is not None


def test_scrip_grid_variables(written_files):
    ds = xr.open_dataset(written_files["scrip"])
    for var in (
        "grid_dims",
        "grid_center_lat",
        "grid_center_lon",
        "grid_corner_lat",
        "grid_corner_lon",
        "grid_imask",
        "grid_area",
    ):
        assert var in ds, f"scrip_grid.nc missing variable: {var}"


def test_scrip_grid_dims_match(written_files, domain):
    grid, _, _, _ = domain
    ds = xr.open_dataset(written_files["scrip"])
    assert ds.sizes["grid_size"] == grid.nx * grid.ny


def test_vgrid_written(written_files):
    assert xr.open_dataset(written_files["vgrid"]) is not None


def test_vgrid_variables(written_files):
    ds = xr.open_dataset(written_files["vgrid"])
    assert "dz" in ds, "vgrid.nc missing variable: dz"


def test_vgrid_dz_positive(written_files):
    ds = xr.open_dataset(written_files["vgrid"])
    assert np.all(ds["dz"].values > 0), "vgrid dz has non-positive layer thickness"


def test_vgrid_dz_sum(written_files, domain):
    _, _, vgrid, _ = domain
    ds = xr.open_dataset(written_files["vgrid"])
    assert abs(ds["dz"].values.sum() - vgrid.depth) < 1.0, (
        "vgrid dz sum differs from expected depth by more than 1 m"
    )


def test_vgrid_z_file_written(written_files):
    assert xr.open_dataset(written_files["vgrid_z"]) is not None


def test_vgrid_z_file_variables(written_files):
    ds = xr.open_dataset(written_files["vgrid_z"])
    for var in ("zi", "zl"):
        assert var in ds, f"vgrid_z.nc missing variable: {var}"


def test_vgrid_z_file_monotonic(written_files):
    ds = xr.open_dataset(written_files["vgrid_z"])
    zi = ds["zi"].values
    zl = ds["zl"].values
    assert np.all(np.diff(zi) > 0), "zi (interfaces) not monotonically increasing"
    assert np.all(np.diff(zl) > 0), "zl (centers) not monotonically increasing"


# ---------------------------------------------------------------------------
# Chlorophyll file test (uses synthetic SeaWiFS source — no GLADE needed)
# ---------------------------------------------------------------------------


def _make_synthetic_seawifs(path):
    """Write a minimal synthetic SeaWiFS-format source file."""
    lon = np.linspace(0, 360, 720, endpoint=False)
    lat = np.linspace(-89.75, 89.75, 360)
    chlor = np.full((12, len(lat), len(lon)), 0.3, dtype=np.float32)
    ds = xr.Dataset(
        {"chlor_a": (["time", "lat", "lon"], chlor)},
        coords={"time": np.arange(12, dtype=float), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(str(path))
    return path


def test_chlorophyll_file_written(tmp_path):
    grid = Grid(
        resolution=0.5,
        xstart=278.0,
        lenx=4.0,
        ystart=7.0,
        leny=3.0,
        name="ci_chl",
    )
    topo = Topo(grid=grid, min_depth=10.0, git=False)
    topo.set_flat(500.0)

    src_path = _make_synthetic_seawifs(tmp_path / "seawifs_src.nc")
    out_path = tmp_path / "seawifs-clim.nc"
    interpolate_and_fill_seawifs(grid, topo, src_path, output_path=out_path)

    assert out_path.exists()
    ds = xr.open_dataset(str(out_path))
    for var in ("CHL_A", "LON", "LAT"):
        assert var in ds, f"seawifs-clim.nc missing variable: {var}"
    assert ds.sizes["TIME"] == 12, "chlorophyll file should have 12 monthly timesteps"
