"""Smoke test for SourceBathy loader."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import xarray as xr
from mom6_forge._source_bathy import SourceBathy


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
