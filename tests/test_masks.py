"""Test mask functionality: setting, applying, and depth masking behavior."""

import numpy as np
import pytest
import xarray as xr
from mom6_forge.edit_command import MaskEditCommand, ClearMaskCommand
import regionmask
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


def test_mask_setter_and_getter(get_rect_topo_without_vc):
    """Test setting and getting user_mask property."""
    topo = get_rect_topo_without_vc
    ny, nx = topo._grid.ny, topo._grid.nx

    # Create a simple binary mask (half ocean, half land)
    mask = np.ones((ny, nx), dtype=int)
    mask[:, : nx // 2] = 0  # Western half is land

    # Set mask
    topo.user_mask = mask

    # Get mask and verify
    retrieved_mask = topo.user_mask
    assert (retrieved_mask == mask).all()


def test_mask_applies_to_depth(get_rect_topo_without_vc):
    """Test that user_mask modifies masked_depth property correctly."""
    topo = get_rect_topo_without_vc
    ny, nx = topo._grid.ny, topo._grid.nx

    # Create binary mask: eastern half ocean (1), western half land (0)
    mask = np.zeros((ny, nx), dtype=int)
    mask[:, nx // 2 :] = 1

    topo.user_mask = mask

    # Get masked depth (with masking applied)
    masked_depth = topo.masked_depth

    # Verify ocean cells (mask=1) have depth >= min_depth+0.1 (enforced minimum)
    assert (masked_depth[:, nx // 2 :] >= topo.min_depth + 0.1 - 1e-10).all()

    # Verify land cells (mask=0) are set to _land_fillval
    assert (masked_depth[:, : nx // 2] == topo._land_fillval).all()


def test_mask_none_disables_masking(get_rect_topo_without_vc):
    """Test that setting user_mask=None disables user masking."""
    topo = get_rect_topo_without_vc
    ny, nx = topo._grid.ny, topo._grid.nx

    # Apply a mask
    mask = np.ones((ny, nx), dtype=int)
    mask[:, : nx // 2] = 0
    topo.user_mask = mask

    # Verify mask is applied
    initial_masked_depth = topo.depth.copy()

    # Clear mask
    topo.user_mask = None

    # After clearing, tmask should be derived from raw depth only
    # So depth should return to being derived only from raw depth (not both)
    assert (topo.tmask == topo._compute_tmask_from_raw_depth()).all()


def test_mask_shape_validation(get_rect_topo_without_vc):
    """Test that user_mask shape must match grid."""
    topo = get_rect_topo_without_vc

    # Try to set mask with wrong shape
    bad_mask = np.ones((10, 10), dtype=int)

    with pytest.raises(AssertionError):
        topo.user_mask = bad_mask


def test_mask_initialization_from_tmask(get_rect_topo_without_vc):
    """Test that MaskEditCommand initializes mask correctly."""
    topo = get_rect_topo_without_vc
    ny, nx = topo._grid.ny, topo._grid.nx

    # Initially no user mask
    assert topo._user_mask is None

    # Create a simple edit command (which should auto-init mask)
    indices = [(0, 0), (0, 1)]
    values = [1, 1]
    cmd = MaskEditCommand(topo, indices, values)
    cmd()

    # Verify mask was initialized
    assert topo._user_mask is not None
