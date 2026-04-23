Mask and Depth Calculation
======================================

Overview
--------

The `Topo` class uses a two part architecture for handling bathymetry depth and ocean/land masking. This ensures clean version control and flexible masking capabilities.

Core Components
---------------

**Raw Depth Storage: ``_depth``**

This is the internal storage for the actual bathymetry data as provided by the user or loaded from files. It represents the true water column depths without any masking applied. Access via the ``depth`` property.

.. code-block:: python

    # Read raw depth
    raw_depth = topo._depth  # xr.DataArray with shape (ny, nx)
    
    # Set raw depth (preserves any existing manual mask)
    topo._depth[index] = new_vals

**Manual Ocean/Land Mask: ``user_mask``**

An optional binary mask indicating which cells are ocean (1) and which are land (0). When ``None``, no manual masking is applied. The mask is automatically enforced: ocean cells have min_depth enforcement, land cells receive depth of _land_fillval (typically 0).

.. code-block:: python

    # Set a binary ocean/land mask
    topo.user_mask = ocean_mask  # xr.DataArray or np.ndarray with values 0 or 1
    
    # Disable manual masking
    topo.user_mask = None

**Depth Property: ``masked_depth``**

The public interface that applies masking on-the-fly. When no manual mask is set, it returns ``depth`` with minimum depth correction. When a manual mask is set, it calculates and returns masked depth values with mask and minimum depth correction

.. code-block:: python

    # Read masked or raw depth depending on manual mask state
    depth_array = topo.masked_depth  # xr.DataArray with masking applied if mask is set

How Masking Works
------------------

When a manual mask is set, the `masked_depth` property applies these rules:

1. **Ocean cells** (user_mask == 1):
   
   - Values are preserved if they exceed ``min_depth``
   - Values below ``min_depth`` are bumped to ``min_depth + 0.1`` to ensure navigability
   
2. **Land cells** (user_mask == 0):
   
   - Values are set to _land_fillval (typically 0.0)

.. caution::

    **Cannot directly index-edit the masked_depth property!**
    
    Since ``masked_depth`` is computed on-the-fly, direct index assignment will not persist:
    
    .. code-block:: python
    
        topo.masked_depth[j, i] = 5000.0  # ❌ This will NOT work - masked_depth is computed on-the-fly
    
    Instead, use one of these methods:
    
    1. **Replace entire raw depth array** (version controlled):
       
       .. code-block:: python
       
           topo.depth = new_depth_array  # This updates _depth and preserves the manual mask
    
    2. **Edit specific indices** (recommended for targeted changes, version controlled):
       
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
    
    # Access masked depth values (with mask and min_depth enforcement applied)
    masked_depth = topo.masked_depth
    print(f"Ocean cells averaged: {masked_depth.where(ocean_mask.astype(bool)).mean()}")
    print(f"Land cells: {masked_depth.where(~ocean_mask.astype(bool)).unique()}")

**Disable Masking**

.. code-block:: python

    # Revert to no manual masking (depth-derived mask)
    topo.user_mask = None
    masked_depth = topo.masked_depth  # Now tmask is derived from raw depth (cells > min_depth are ocean)

**Check Raw vs Masked**

.. code-block:: python

    # Compare raw and masked versions
    raw = topo.depth
    masked = topo.masked_depth
    
    # If a user mask exists, land cells will be set to _land_fillval and shallow ocean cells bumped to min_depth+0.1
    if topo.user_mask is not None:
        diff_cells = (masked != raw).sum()
        print(f"Masking applied - differences in {diff_cells} cells")
    else:
        # Without user_mask, differences only occur where depth < min_depth (bumped to min_depth+0.1)
        diff_cells = (masked != raw).sum()
        print(f"Minimum depth enforcement applied to {diff_cells} cells")

Version Control and Edits
--------------------------

All depth modifications can be done through the ``depth`` property or ``edit_depth()`` method, which:

- Modifies only ``_depth`` (the raw storage)
- Preserves the existing manual mask
- Records changes in the TopoCommandManager (TCM) for undo/redo support
- Allows full version control of bathymetry edits

.. code-block:: python

    # When you modify depth, the manual mask is preserved
    topo.depth = new_depth_array  # Only _depth is updated
    
    # The manual mask remains intact
    assert topo.user_mask == original_mask
    
    # Use edit_depth for targeted, fine-grained changes
    topo.edit_depth([(2, 5), (3, 6)], [3000.0, 4000.0])

Internal Architecture
---------------------

.. code-block:: text

    User sets: topo.user_mask = ocean_mask
                ↓
    Stored in: self._user_mask
                ↓
            When user calls: topo.masked_depth
                ↓
            Property checks: if self._user_mask is not None
                ↓
            If mask exists:
                Calculate: masked_depth = where(mask==1, 
                                        max(_depth, min_depth+0.1),
                                        _land_fillval)
                ↓
            If no mask:
                Derive mask from depth: tmask = where(_depth > min_depth, 1, 0)
                Calculate: masked_depth = where(tmask==1,
                                        max(_depth, min_depth+0.1),
                                        _land_fillval)
                ↓
            Return: masked depth array

API Reference
-------------

**Properties:**

- ``topo.masked_depth`` — Get (read-only) the computed depth array with masking applied
- ``topo.depth`` — Get/set the raw underlying depth array (doesn't include masking)
- ``topo.tmask`` — Get the binary ocean/land mask (either user_mask if set, or computed from depth)
- ``topo.user_mask`` — Get/set the optional manual binary ocean/land mask (1=ocean, 0=land)
- ``topo.umask`` / ``topo.vmask`` / ``topo.qmask`` — Get ocean masks on staggered grids
- ``topo.min_depth`` — Get/set the minimum ocean depth threshold

**Methods:**

- ``topo.edit_depth(indices, values)`` — Edit depth at specific cell indices with version control
- ``topo.send_entire_depth_change_to_tcm(depth)`` — Apply a complete depth array replacement with version control
- ``topo.write_topo(file_path)`` — Write bathymetry file (includes both raw depth and masked depth)

**Internal:**

- ``topo._depth`` — Raw depth storage (use ``topo.depth`` property instead)
- ``topo._user_mask`` — Manual mask storage (use ``topo.user_mask`` property instead)

Benefits of This Design
-----------------------

**Clean separation** — Raw data separate from masking logic

**Flexible masking** — Mask can be applied, modified, or removed without affecting stored bathymetry
