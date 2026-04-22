Mask and Depth Calculation
======================================

Overview
--------

The `Topo` class uses a two part architecture for handling bathymetry depth and ocean/land masking. This ensures clean version control and flexible masking capabilities.

Core Components
---------------

**Raw Depth Storage: ``_depth_raw``**

This is the internal storage for the actual bathymetry data as provided by the user or loaded from files. It represents the true water column depths without any masking applied. Access via the ``depth_raw`` property.

.. code-block:: python

    # Read raw depth
    raw_depth = topo._depth_raw  # xr.DataArray with shape (ny, nx)
    
    # Set raw depth (preserves any existing manual mask)
    topo._depth_raw[index] = new_vals

**Manual Ocean/Land Mask: ``user_mask``**

An optional binary mask indicating which cells are ocean (1) and which are land (0). When ``None``, no manual masking is applied. The mask is automatically enforced: ocean cells have min_depth enforcement, land cells receive depth of 0.

.. code-block:: python

    # Set a binary ocean/land mask
    topo.user_  mask = ocean_mask  # xr.DataArray or np.ndarray with values 0 or 1
    
    # Disable manual masking
    topo.user_mask = None

**Depth Property: ``depth``**

The public interface that applies masking on-the-fly. When no manual mask is set, it returns ``_depth_raw`` with minimum depth correction. When a manual mask is set, it calculates and returns masked depth values with mask and minimum depth correction

.. code-block:: python

    # Read masked or raw depth depending on manual mask state
    depth_array = topo.depth  # xr.DataArray with masking applied if mask is set

How Masking Works
------------------

When a manual mask is set, the `depth` property applies these rules:

1. **Ocean cells** (user_mask == 1):
   
   - Values are preserved if they exceed ``min_depth``
   - Values below ``min_depth`` are bumped to ``min_depth + 0.1`` to ensure navigability
   
2. **Land cells** (user_mask == 0):
   
   - Values are set to 0.0

.. caution::

    **Cannot directly index-edit the depth property!**
    
    Since ``depth`` is now a calculated property (not raw storage), direct index assignment will not persist:
    
    .. code-block:: python
    
        topo.depth[j, i] = 5000.0  # ❌ This will NOT work - depth is computed on-the-fly
    
    Instead, use one of these methods:
    
    1. **Replace entire depth array** (version controlled):
       
       .. code-block:: python
       
           topo.depth = new_depth_array  # This updates _depth_raw and preserves the manual mask
    
    2. **Edit specific indices** (recommended for targeted changes):
       
       .. code-block:: python
       
           # Single cell
           topo.edit_depth([(j, i)], [5000.0])
           
           # Multiple cells
           topo.edit_depth([(j1, i1), (j2, i2)], [3000.0, 4000.0])
           
           # Horizontal strip
           indices = [(2, i) for i in range(topo._grid.nx)]
           values = [0.0] * len(indices)
           topo.edit_depth(indices, values)
    
    Both methods register changes in the version control system automatically.

Usage Examples
--------------

**Setup with Manual Masking**

.. code-block:: python

    from mom6_forge.topo import Topo
    
    # Create Topo object with initial bathymetry
    topo = Topo.set_from_dataset(...)
    
    # Apply ocean/land mask
    topo.user_mask = ocean_mask  # Binary mask, shape (ny, nx)
    
    # Access masked depth values
    masked_depth = topo.depth
    print(f"Ocean cells averaged: {topo.depth.where(ocean_mask).mean()}")
    print(f"Land cells: {topo.depth.where(~ocean_mask.astype(bool)).unique()}}")

**Disable Masking**

.. code-block:: python

    # Revert to unmasked raw depths
    topo.user_mask = None
    raw_depth = topo.depth  # Now returns _depth_raw unchanged by a user mask (still applies minimum depth adjustment)

**Check Raw vs Masked**

.. code-block:: python

    # Compare raw and masked versions
    raw = topo._depth_raw
    masked = topo.depth
    
    if topo.user_mask is not None:
        diff = (masked - raw).sum()
        print(f"Masking applied - differences in {(masked != raw).sum()} cells")

Version Control and Edits
--------------------------

All full depth modifications go through the ``send_entire_depth_change_to_tcm()`` method, which:

- Modifies only ``_depth_raw`` (the raw storage)
- Preserves the existing manual mask
- Records changes in the TopoCommandManager (TCM) for undo/redo support
- Allows full version control of bathymetry edits

.. code-block:: python

    # When you modify depth, the manual mask is preserved
    topo.depth = new_depth_array  # Only _depth_raw is updated
    
    # The manual mask remains intact
    assert topo.mask == original_mask

You can also change the depth by index with the ``edit_depth`` function, which takes indexes and new values. This also preserves the manual mask and records changes in TCM.

Internal Architecture
---------------------

.. code-block:: text

    User sets: topo.user_mask = ocean_mask
                ↓
    Stored in: self._user_mask
                ↓
            When user calls: topo.depth
                ↓
            Property checks: if self._user_mask is not None
                ↓
            If mask exists:
                Calculate: masked_depth = where(mask==1, enforce_min_depth(_depth_raw),
                                                           0.0)
                ↓
            If no mask:
                Return: self._depth_raw directly with enforced minimum depth
                ↓
            Return: masked/raw depth array

API Reference
-------------

**Properties:**

- ``topo.depth`` — Get/set the depth array (applies masking when reading if mask is set)
- ``topo.tmask`` — Get the binary ocean/land mask
- ``topo.user_mask`` — Get/set the manual binary ocean/land mask
- ``topo.min_depth`` — Get/set the minimum ocean depth threshold

**Methods:**

- ``topo.apply_land_frac(...)`` — Generate and apply ocean mask from land fraction data
- ``topo.send_entire_depth_change_to_tcm(depth)`` — Apply a full depth array update with version control
- ``topo.edit_depth(index, new_values)`` — Edit depth values at specific indexes with version control

**Internal:**

- ``topo._depth_raw`` — Raw depth storage (use ``topo.depth_raw`` property or ``topo.depth`` instead)
- ``topo._user_mask`` — Internal manual mask storage (use ``topo.user_mask`` property instead)

Benefits of This Design
-----------------------

**Clean separation** — Raw data separate from masking logic

**Flexible masking** — Mask can be applied, modified, or removed without affecting stored bathymetry
