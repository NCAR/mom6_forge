Mask and Depth Calculation
======================================

Overview
--------

The `Topo` class uses a separation-of-concerns architecture for handling bathymetry depth and ocean/land masking. This ensures clean version control, flexible masking capabilities, and intuitive API design.

Core Components
---------------

**Raw Depth Storage: ``_depth_raw``**

This is the internal storage for the actual bathymetry data as provided by the user or loaded from files. It represents the true water column depths without any masking applied. Access via the ``depth_raw`` property.

.. code-block:: python

    # Read raw depth
    raw_depth = topo.depth_raw  # xr.DataArray with shape (ny, nx)
    
    # Set raw depth (preserves any existing manual mask)
    topo.depth_raw = new_bathymetry

**Manual Ocean/Land Mask: ``_manual_mask``**

An optional binary mask indicating which cells are ocean (1) and which are land (0). When ``None``, no manual masking is applied. The mask is automatically enforced: ocean cells have min_depth enforcement, land cells receive the ``land_fillval``.

.. code-block:: python

    # Set a binary ocean/land mask
    topo.mask = ocean_mask  # xr.DataArray or np.ndarray with values 0 or 1
    
    # Disable manual masking
    topo.mask = None

**Depth Property: ``depth``**

The public interface that applies masking on-the-fly. When no manual mask is set, it returns ``_depth_raw``. When a manual mask is set, it calculates and returns masked depth values.

.. code-block:: python

    # Read masked or raw depth depending on manual mask state
    depth_array = topo.depth  # xr.DataArray with masking applied if mask is set

**Land Cell Fill Value: ``land_fillval``**

Configurable depth value assigned to land cells when a manual mask is applied. Default is ``0.0`` (dry). Must be ``<= min_depth`` to prevent land cells from being treated as ocean.

.. code-block:: python

    topo.land_fillval = 0.0  # Set all land cells to 0.0 m (default)
    topo.land_fillval = -100.0  # Or use negative values for special meaning

How Masking Works
------------------

When a manual mask is set, the `depth` property applies these rules:

1. **Ocean cells** (manual_mask == 1):
   
   - Values are preserved if they exceed ``min_depth``
   - Values below ``min_depth`` are bumped to ``min_depth + 0.1`` to ensure navigability
   
2. **Land cells** (manual_mask == 0):
   
   - Values are set to ``land_fillval`` (default: 0.0)
   - This is fully configurable via the ``land_fillval`` property

.. code-block:: python

    # Example: Set custom land fill value
    topo.land_fillval = -0.5  # Represent dry cells as negative
    
    # Now when depth is read, land cells will show -0.5
    depth = topo.depth  # Ocean cells have real depths, land cells are -0.5

Usage Examples
--------------

**Setup with Manual Masking**

.. code-block:: python

    from mom6_forge.topo import Topo
    
    # Create Topo object with initial bathymetry
    topo = Topo(depth=my_depth_array)
    
    # Apply ocean/land mask
    topo.mask = ocean_mask  # Binary mask, shape (ny, nx)
    
    # Configure land cell behavior
    topo.land_fillval = 0.0  # Land cells set to 0.0 m
    
    # Access masked depth values
    masked_depth = topo.depth
    print(f"Ocean cells averaged: {topo.depth.where(ocean_mask).mean()}")
    print(f"Land cells: {topo.depth.where(~ocean_mask.astype(bool)).unique()}")

**Disable Masking**

.. code-block:: python

    # Revert to unmasked raw depths
    topo.mask = None
    raw_depth = topo.depth  # Now returns _depth_raw unchanged

**Check Raw vs Masked**

.. code-block:: python

    # Compare raw and masked versions
    raw = topo.depth_raw
    masked = topo.depth
    
    if topo.mask is not None:
        diff = (masked - raw).sum()
        print(f"Masking applied - differences in {(masked != raw).sum()} cells")

Version Control and Edits
--------------------------

All depth modifications go through the ``send_entire_depth_change_to_tcm()`` method, which:

- Modifies only ``_depth_raw`` (the raw storage)
- Preserves the existing manual mask
- Records changes in the TopoCommandManager (TCM) for undo/redo support
- Allows full version control of bathymetry edits

.. code-block:: python

    # When you modify depth, the manual mask is preserved
    topo.depth = new_depth_array  # Only _depth_raw is updated
    
    # The manual mask remains intact
    assert topo.mask == original_mask

Internal Architecture
---------------------

.. code-block:: text

    User sets: topo.mask = ocean_mask
                ↓
    Stored in: self._manual_mask
                ↓
            When user calls: topo.depth
                ↓
            Property checks: if self._manual_mask is not None
                ↓
            If mask exists:
                Calculate: masked_depth = where(mask==1, enforce_min_depth(_depth_raw),
                                                           land_fillval)
                ↓
            If no mask:
                Return: self._depth_raw directly
                ↓
            Return: masked/raw depth array

Practical Example
-----------------

Setting up bathymetry with masking:

.. code-block:: python

    from mom6_forge.grid import Grid
    from mom6_forge.topo import Topo
    import numpy as np

    # Create grid and topo
    grid = Grid(nx=180, ny=90, lenx=360.0, leny=180.0)
    topo = Topo(grid, min_depth=100.0)
    
    # Set flat bathymetry
    topo.set_flat(5000.0)
    
    # Create a simple ocean/land mask (1 = ocean, 0 = land)
    mask = np.ones((grid.ny, grid.nx), dtype=int)
    mask[:, :40] = 0  # Western region is land
    
    # Apply the mask
    topo.mask = mask
    
    # Now when you read depth:
    # - Western cells (land) → 0.0 (or custom land_fillval)
    # - Eastern cells (ocean) → 5000.0 (or bumped to min_depth if shallower)
    depth = topo.depth
    
    # Configure land fill value
    topo.land_fillval = -100.0
    depth = topo.depth  # Now land cells show -100.0

API Reference
-------------

**Properties:**

- ``topo.depth`` — Get/set the depth array (applies masking when reading if mask is set)
- ``topo.mask`` — Get/set the binary ocean/land mask
- ``topo.land_fillval`` — Get/set the depth value for land cells (default: 0.0)
- ``topo.min_depth`` — Get/set the minimum ocean depth threshold

**Methods:**

- ``topo.generate_mask_from_landfrac_file(...)`` — Generate and apply ocean mask from land fraction data
- ``topo.send_entire_depth_change_to_tcm(depth)`` — Apply a full depth array update with version control

**Internal:**

- ``topo._depth_raw`` — Raw depth storage (use ``topo.depth_raw`` property or ``topo.depth`` instead)
- ``topo._manual_mask`` — Internal manual mask storage (use ``topo.mask`` property instead)
- ``topo._land_fillval`` — Internal land fill value storage (use ``topo.land_fillval`` property instead)

Benefits of This Design
-----------------------

✓ **Clean separation** — Raw data separate from masking logic

✓ **Full version control** — All edits tracked in TCM without bloating the history with mask applications

✓ **Flexible masking** — Mask can be applied, modified, or removed without affecting stored bathymetry

✓ **Efficient computation** — Masking applied on-read rather than stored, reducing file size

✓ **Intuitive API** — Users work with ``depth`` and ``mask`` properties; internal storage is hidden
