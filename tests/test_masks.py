import regionmask
import numpy as np
from unittest.mock import MagicMock, patch
from mom6_forge.topo import Topo


def test_generate_mask_from_naturalearth():
    """Test that generate_mask_from_naturalearth correctly applies a land mask."""

    # Create a simple 4x4 grid
    lon = np.array([[0, 90, 180, 270]] * 4, dtype=float)
    lat = np.array(
        [
            [-60, -60, -60, -60],
            [-30, -30, -30, -30],
            [30, 30, 30, 30],
            [60, 60, 60, 60],
        ],
        dtype=float,
    )

    grid = MagicMock()
    grid.nx = 4
    grid.ny = 4
    grid.tlon.values = lon
    grid.tlat.values = lat

    topo = Topo.__new__(Topo)  # skip __init__
    topo._grid = grid
    topo._manual_mask = None
    topo._min_depth = 10.0
    topo._land_fillval = 0.0
    topo.tcm = MagicMock()

    # Capture what mask gets set

    mask = topo.generate_mask_from_naturalearth(resolution="110", version="v5_1_2")

    result = mask

    assert result.shape == (4, 4), "mask shape should match grid"
    assert set(np.unique(result)).issubset({0, 1}), "mask should be binary"

    # Spot check: lon=0, lat=0 is ocean (Atlantic)
    raw = regionmask.defined_regions.natural_earth_v5_1_2.land_110.mask(lon, lat)
    expected = raw.isnull().astype(int).values
    np.testing.assert_array_equal(result, expected)


def test_generate_mask_from_naturalearth_bad_version():
    """Test that a bad version raises a clear error."""
    import pytest

    topo = Topo.__new__(Topo)
    topo._grid = MagicMock()
    with pytest.raises(AssertionError, match="regionmask has no Natural Earth version"):
        topo.generate_mask_from_naturalearth(version="v999")


def test_generate_mask_from_naturalearth_bad_resolution():
    """Test that a bad resolution raises a clear error."""
    import pytest

    topo = Topo.__new__(Topo)
    topo._grid = MagicMock()
    with pytest.raises(AssertionError, match="has no resolution"):
        topo.generate_mask_from_naturalearth(resolution="999")
