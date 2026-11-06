"""Smoke test for SourceBathy loader."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import xarray as xr
from mom6_forge._source_bathy import SourceBathy
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo


def test_simple_source_bathy_calls(get_rect_topo_without_vc, synthetic_bathy_file):
    src = SourceBathy(
        get_rect_topo_without_vc,
        synthetic_bathy_file,
        lon_name="lon",
        lat_name="lat",
        depth_name="elevation",
    )
    print(src, src.ds, src.lon, src.lat, src.depth)


def test_source_bathy_initialization(synthetic_bathy_file, get_rect_topo_without_vc):
    """Test SourceBathy initialization and coordinate names."""
    src = SourceBathy(
        get_rect_topo_without_vc,
        synthetic_bathy_file,
        lon_name="lon",
        lat_name="lat",
        depth_name="elevation",
    )

    assert src.path == Path(synthetic_bathy_file)
    assert src.lon_name == "lon"
    assert src.lat_name == "lat"
    assert src.depth_name == "depth"


def test_source_bathy_slice_to_domain(get_rect_topo_without_vc, synthetic_bathy_file):
    """Smoke test: load and slice depth to topo domain."""
    topo = get_rect_topo_without_vc

    src = SourceBathy(
        topo,
        synthetic_bathy_file,
        depth_name="elevation",
        is_input_positive_below_msl=False,
    )

    # Verify data was loaded
    assert src.lon is not None
    assert src.lat is not None

    # Verify shape makes sense
    assert len(src.lon) > 0, f"Expected lon data, got empty array"
    assert len(src.lat) > 0, f"Expected lat data, got empty array"
    assert src.depth.shape == (len(src.lat), len(src.lon))


def test_source_bathy_depth_conversion(get_rect_topo_without_vc, synthetic_bathy_file):
    """Test that depth is converted to positive-down depth."""
    topo = get_rect_topo_without_vc

    src = SourceBathy(
        topo,
        synthetic_bathy_file,
        depth_name="elevation",
        is_input_positive_below_msl=False,
    )

    # Get depth and verify sign conversion
    depth = src.depth

    # Verify no NaNs in the result
    assert not bool(np.isnan(depth).all()), "All depth values are NaN"

    # Verify positive depth values for ocean (depth is negative)
    non_nan_values = depth[~np.isnan(depth)]
    assert len(non_nan_values) > 0, "No valid depth values"
    assert np.any(non_nan_values > 0), "Expected positive depth values for ocean"

    assert depth.shape == src.depth.shape


def _global_source(path, lon):
    """All-ocean 0.5-degree source on the given longitudes, GEBCO sign."""
    lat = np.arange(-89.75, 90.0, 0.5)
    xr.Dataset(
        {"elevation": (["lat", "lon"], np.full((lat.size, lon.size), -1000.0))},
        coords={"lon": lon, "lat": lat},
    ).to_netcdf(path)
    return path


def _flat_topo(grid):
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)
    return topo


@pytest.mark.parametrize(
    "crs, x_min, x_max, y_min, y_max, res",
    [
        # Goes around the South Pole, but its qlon only spans -171 to 149
        # and its qlat stops short of -89.
        ("EPSG:3031", -360e3, 5640e3, -396e3, 3204e3, 300e3),
        ("EPSG:3995", -300e3, 300e3, -300e3, 300e3, 100e3),
    ],
)
def test_polar_cap_regrids_without_a_seam(
    crs, x_min, x_max, y_min, y_max, res, tmp_path
):
    """A cap around a pole crosses the antimeridian along its rim, so it needs
    every source longitude and a periodic regrid, or cells there stay at zero."""
    topo = _flat_topo(Grid.from_projection(crs, x_min, x_max, y_min, y_max, res))
    topo.set_src(
        _global_source(tmp_path / "global.nc", np.arange(-179.75, 180.0, 0.5)),
        "lon",
        "lat",
        "elevation",
    )
    topo.set_depth_from_xesmf()
    assert (topo.depth.values > 0).all()


def test_source_with_a_repeated_seam_column_regrids(tmp_path):
    """A source on -180..180 inclusive, like grid-registered ETOPO1, stores the
    seam column twice. ESMF cannot build a periodic grid from that, and on some
    builds aborts the whole process, so the repeat has to be gone before the
    regrid starts."""
    topo = _flat_topo(
        Grid.from_projection("EPSG:3995", -300e3, 300e3, -300e3, 300e3, 100e3)
    )
    src = topo.set_src(
        _global_source(tmp_path / "etopo.nc", np.arange(-180.0, 180.25, 0.5)),
        "lon",
        "lat",
        "elevation",
    )
    assert src.lon.size == 720
    assert src.is_periodic
    topo.set_depth_from_xesmf()
    assert (topo.depth.values > 0).all()


def test_grid_reaching_toward_a_pole_still_slices_the_source(tmp_path):
    """Reaching 89.5N is not going around the pole. A 30-degree-wide grid needs
    about 31 degrees of source longitude, not the whole globe."""
    topo = _flat_topo(
        Grid(resolution=0.1, xstart=0.0, lenx=30.0, ystart=80.0, leny=9.5)
    )
    src = topo.set_src(
        _global_source(tmp_path / "global.nc", np.arange(-179.75, 180.0, 0.5)),
        "lon",
        "lat",
        "elevation",
    )
    assert src.lon.max() - src.lon.min() < 32.0
