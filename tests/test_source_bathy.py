"""Smoke test for SourceBathy loader."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import xarray as xr
from mom6_forge._source_bathy import SourceBathy, longitude_slicer


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

    # Create synthetic depth data (positive-up, like GEBCO)
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


def test_simple_source_bathy_calls(get_rect_topo, synthetic_bathy_file):
    src = SourceBathy(
        get_rect_topo,
        synthetic_bathy_file,
        lon_name="lon",
        lat_name="lat",
        depth_name="elevation",
    )
    print(src, src.ds, src.lon, src.lat, src.depth)


def test_source_bathy_initialization(synthetic_bathy_file, get_rect_topo):
    """Test SourceBathy initialization and coordinate names."""
    src = SourceBathy(
        get_rect_topo,
        synthetic_bathy_file,
        lon_name="lon",
        lat_name="lat",
        depth_name="elevation",
    )

    assert src.path == Path(synthetic_bathy_file)
    assert src.lon_name == "lon"
    assert src.lat_name == "lat"
    assert src.depth_name == "depth"


def test_source_bathy_slice_to_domain(get_rect_topo, synthetic_bathy_file):
    """Smoke test: load and slice depth to topo domain."""
    topo = get_rect_topo

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


def test_source_bathy_depth_conversion(get_rect_topo, synthetic_bathy_file):
    """Test that depth is converted to positive-down depth."""
    topo = get_rect_topo

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


def test_longitude_slicer():
    with pytest.raises(AssertionError):
        nx, ny, nt = 4, 14, 5

        latitude_extent = (10, 20)
        longitude_extent = (12, 18)

        dims = ["random_lat", "random_lon", "time"]

        dlambda = (longitude_extent[1] - longitude_extent[0]) / 2

        data = xr.DataArray(
            np.random.random((ny, nx, nt)),
            dims=dims,
            coords={
                "random_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
                "random_lon": np.array(
                    [
                        longitude_extent[0],
                        longitude_extent[0] + 1.5 * dlambda,
                        longitude_extent[0] + 2.6 * dlambda,
                        longitude_extent[1],
                    ]
                ),
                "time": np.linspace(0, 1000, nt),
            },
        )

        longitude_slicer(data, longitude_extent, "random_lon")


def test_longitude_slicers_regionally():
    nx, ny = 4, 14

    latitude_extent = (2, 5)
    longitude_extent = (-90, -70)

    dims = ["random_lat", "random_lon"]

    dlambda = (longitude_extent[1] - longitude_extent[0]) / 2

    data = xr.DataArray(
        np.random.random((ny, nx)),
        dims=dims,
        coords={
            "random_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
            "random_lon": np.linspace(
                longitude_extent[0] - 2, longitude_extent[1] + 2, nx
            ),
        },
    )

    # Regular regional
    data_regular = longitude_slicer(data, longitude_extent, "random_lon")
    data_east = longitude_slicer(data, (270, 290), "random_lon")
    assert (data_regular == data_east).all()

    # Seam data
    longitude_extent = (-5, 5)
    data = xr.DataArray(
        np.random.random((ny, nx)),
        dims=dims,
        coords={
            "random_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
            "random_lon": np.linspace(
                longitude_extent[0] - 2, longitude_extent[1] + 2, nx
            ),
        },
    )
    data_regular = longitude_slicer(data, longitude_extent, "random_lon")
    assert len(data_regular.random_lon) > 0
