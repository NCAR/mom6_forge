"""Test mask functionality: setting, applying, and depth masking behavior."""

import numpy as np
import pytest
import xarray as xr
from mom6_forge.edit_command import MaskEditCommand, ClearMaskCommand


def test_mask_setter_and_getter(get_rect_topo):
    """Test setting and getting user_mask property."""
    topo = get_rect_topo
    ny, nx = topo._grid.ny, topo._grid.nx

    # Create a simple binary mask (half ocean, half land)
    mask = np.ones((ny, nx), dtype=int)
    mask[:, : nx // 2] = 0  # Western half is land

    # Set mask
    topo.user_mask = mask

    # Get mask and verify
    retrieved_mask = topo.user_mask
    assert (retrieved_mask == mask).all()


def test_mask_applies_to_depth(get_rect_topo):
    """Test that user_mask modifies masked_depth property correctly."""
    topo = get_rect_topo
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


def test_mask_none_disables_masking(get_rect_topo):
    """Test that setting user_mask=None disables user masking."""
    topo = get_rect_topo
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


def test_mask_shape_validation(get_rect_topo):
    """Test that user_mask shape must match grid."""
    topo = get_rect_topo

    # Try to set mask with wrong shape
    bad_mask = np.ones((10, 10), dtype=int)

    with pytest.raises(AssertionError):
        topo.user_mask = bad_mask


def test_mask_initialization_from_tmask(get_rect_topo):
    """Test that MaskEditCommand initializes mask correctly."""
    topo = get_rect_topo
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
