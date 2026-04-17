"""Test mask functionality: setting, applying, and depth masking behavior."""

import numpy as np
import pytest
import xarray as xr
from mom6_forge.edit_command import MaskEditCommand, ClearMaskCommand


def test_mask_setter_and_getter(get_rect_topo):
    """Test setting and getting mask property."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Create a simple binary mask (half ocean, half land)
    mask = np.ones((ny, nx), dtype=int)
    mask[:, : nx // 2] = 0  # Western half is land

    # Set mask
    topo.mask = mask

    # Get mask and verify
    retrieved_mask = topo.mask
    assert (retrieved_mask == mask).all()


def test_mask_applies_to_depth(get_rect_topo):
    """Test that mask modifies depth property correctly."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Store raw depth before masking
    raw_depth = topo.depth.copy()

    # Create binary mask: eastern half ocean (1), western half land (0)
    mask = np.zeros((ny, nx), dtype=int)
    mask[:, nx // 2 :] = 1

    # Set land fill value
    topo.land_fillval = 0.0
    topo.mask = mask

    # Get masked depth
    masked_depth = topo.depth

    # Verify ocean cells (mask=1) keep original depth
    assert (masked_depth[:, nx // 2 :] > topo.min_depth).all()

    # Verify land cells (mask=0) are set to land_fillval
    assert (masked_depth[:, : nx // 2] == 0.0).all()


def test_mask_none_disables_masking(get_rect_topo):
    """Test that setting mask=None disables masking."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Apply a mask
    mask = np.ones((ny, nx), dtype=int)
    mask[:, : nx // 2] = 0
    topo.mask = mask

    # Verify mask is applied
    assert not (topo.depth == topo.depth_raw).all()

    # Clear mask
    topo.mask = None

    # Verify depth returns to raw
    assert (topo.depth == topo.depth_raw).all()


def test_land_fillval_property(get_rect_topo):
    """Test setting and validating land_fillval."""
    topo = get_rect_topo

    # Test setting land_fillval
    topo.land_fillval = 0.0
    assert topo.land_fillval == 0.0

    # Test validation: land_fillval must be <= min_depth
    with pytest.raises(AssertionError):
        topo.land_fillval = topo.min_depth + 1.0


def test_land_fillval_affects_masked_depth(get_rect_topo):
    """Test that land_fillval is used in masked depth."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Create all-land mask
    mask = np.zeros((ny, nx), dtype=int)

    # Set custom land fill value
    custom_fill = -100.0
    topo.land_fillval = custom_fill
    topo.mask = mask

    # Verify all depth values are land_fillval
    assert (topo.depth == custom_fill).all()


def test_depth_raw_property(get_rect_topo):
    """Test depth_raw property reads and writes correctly."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Get raw depth
    raw = topo.depth_raw
    assert raw.shape == (ny, nx)

    # Modify raw depth
    new_depth = np.full((ny, nx), 2000.0)
    topo.depth_raw = new_depth

    # Verify it was updated
    assert (topo.depth_raw == 2000.0).all()


def test_depth_raw_preserves_mask(get_rect_topo):
    """Test that setting depth_raw preserves existing mask."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Apply a mask
    mask = np.ones((ny, nx), dtype=int)
    mask[:, : nx // 2] = 0
    topo.mask = mask

    # Modify raw depth
    new_depth = np.full((ny, nx), 3000.0)
    topo.depth_raw = new_depth

    # Verify mask is still applied
    masked_depth = topo.depth
    assert (masked_depth[:, : nx // 2] == topo.land_fillval).all()
    assert (masked_depth[:, nx // 2 :] > topo.min_depth).all()


def test_mask_shape_validation(get_rect_topo):
    """Test that mask shape must match grid."""
    topo = get_rect_topo

    # Try to set mask with wrong shape
    bad_mask = np.ones((10, 10), dtype=int)

    with pytest.raises(AssertionError):
        topo.mask = bad_mask


def test_mask_initialization_from_depth_raw_mask(get_rect_topo):
    """Test that LandEditCommand initializes mask from depth_raw_mask."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Initially no manual mask
    assert topo._manual_mask is None

    # Create a simple edit command (which should auto-init mask)
    indices = [(0, 0), (0, 1)]
    values = [1, 1]
    cmd = MaskEditCommand(topo, indices, values)
    cmd()

    # Verify mask was initialized
    assert topo._manual_mask is not None
