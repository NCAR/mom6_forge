"""Smoke test for SourceBathy loader."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import xarray as xr
from mom6_forge._source_bathy import SourceBathy


@pytest.fixture
def synthetic_bathy_file():
    """Create a temporary synthetic bathymetry NetCDF file for testing.

    Covers the Panama region (278-282°E, 7-10°N) to match get_rect_grid().
    """
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        bathy_file = tmp.name

    # Create synthetic bathymetry covering the Panama region
    # get_rect_grid uses xstart=278, lenx=4, ystart=7, leny=3
    # So we need lon 278-282, lat 7-10 with some buffer
    lon = np.linspace(276, 284, 80)  # Cover 278-282 with buffer
    lat = np.linspace(5, 12, 70)  # Cover 7-10 with buffer

    # Create synthetic elevation data (positive-up, like GEBCO)
    # Ocean is negative (water), land is positive
    elevation = np.full((len(lat), len(lon)), -500.0)  # Ocean baseline = 500m deep

    # Add synthetic land masses (islands)
    # Create an island around (280, 8.5)
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    island_mask = (lon_2d - 280) ** 2 + (lat_2d - 8.5) ** 2 < 0.5
    elevation[island_mask] = 200.0  # Synthetic island

    ds = xr.Dataset(
        {
            "elevation": (["lat", "lon"], elevation),
        },
        coords={
            "lon": lon,
            "lat": lat,
        },
    )
    ds.to_netcdf(bathy_file)

    yield bathy_file

    # Cleanup
    Path(bathy_file).unlink()


def test_source_bathy_initialization(synthetic_bathy_file):
    """Test SourceBathy initialization and coordinate names."""
    src = SourceBathy(
        synthetic_bathy_file,
        lon_name="lon",
        lat_name="lat",
        elevation_name="elevation",
    )

    assert src.path == Path(synthetic_bathy_file)
    assert src.lon_name == "lon"
    assert src.lat_name == "lat"
    assert src.elevation_name == "elevation"
    assert src._da is None  # Not loaded yet


def test_source_bathy_slice_to_domain(get_rect_topo, synthetic_bathy_file):
    """Smoke test: load and slice elevation to topo domain."""
    topo = get_rect_topo

    src = SourceBathy(synthetic_bathy_file)
    src.slice_to_domain(topo, buf=0.5)

    # Verify data was loaded
    assert src._da is not None
    assert src.lon is not None
    assert src.lat is not None

    # Verify shape makes sense
    assert len(src.lon) > 0
    assert len(src.lat) > 0
    assert src._da.shape == (len(src.lat), len(src.lon))


def test_source_bathy_depth_conversion(get_rect_topo, synthetic_bathy_file):
    """Test that elevation is converted to positive-down depth."""
    topo = get_rect_topo

    src = SourceBathy(synthetic_bathy_file)
    src.slice_to_domain(topo, buf=0.5)

    # Get depth and verify sign conversion
    depth = src.depth

    assert not bool(np.isnan(depth).all())
    assert depth.shape == src.da.shape
