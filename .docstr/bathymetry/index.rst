Bathymetry Workflow
======================================

The bathymetry workflow pipeline combines multiple components for creating and modifying MOM6 bathymetry files. This section documents the key concepts, operations, and best practices for working with bathymetry in mom6_forge.

.. toctree::
   :maxdepth: 2
   :caption: Bathymetry Topics:

   mask_depth_calculation

Overview of the Bathymetry Pipeline
------------------------------------

The mom6_forge bathymetry workflow consists of these key stages:

1. **Grid Creation** — Define the horizontal MOM6 grid
2. **Depth Initialization** — Load or generate initial bathymetry
3. **Masking** — Apply ocean/land masks (e.g., from land fraction data)
4. **Refinement** — Apply ridges, smooth features, or manual edits
5. **Validation** — Check consistency and apply min/max depth constraints
6. **Output** — Write final bathymetry to TOPO_FILE

Key Concepts
------------

**Separation of Concerns**

Raw depth data (``_depth``) is kept separate from masking logic (``tmask``). This enables:

- Efficient storage (only one copy of depth data)
- Flexible masking (masks can be applied/removed without re-computing)
- Clean version control (edits to depth vs mask are independent)

**Version Control Integration**

All bathymetry modifications are tracked through the TopoCommandManager (TCM):

- Every edit creates a record (index, old_value, new_value)
- Full undo/redo capability
- Version history saved to git
- Can restore any previous bathymetry state

**Minimum Depth Enforcement**

The ``min_depth`` parameter ensures all ocean cells meet a minimum depth for model stability:

- Ocean cells shallower than ``min_depth`` are boosted to ``min_depth + 0.1``
- Ensures model doesn't attempt to integrate in cells that are too shallow
- Configurable per-Topo instance

Common Workflows
----------------

**Workflow 1: Load existing bathymetry with mask**

.. code-block:: python

    # Load bathymetry from file
    topo = Topo.from_topo_file(grid, "input_topo.nc")
    
    # Apply an ocean/land mask
    topo.apply_land_frac("land_fraction.nc", landfrac_name="LANDFRAC", ...)
    
    # Write output
    topo.write_topo("output_topo.nc")

**Workflow 2: Create idealized bathymetry**

.. code-block:: python

    # Create flat or idealized bathymetry
    topo = Topo(grid, min_depth=100.0)
    topo.set_spoon(max_depth=5000.0, dedge=100.0)
    
    # Apply custom modifications
    topo.apply_ridge(height=1000.0, width=10.0, lon=-120.0, ilat=(10, 50))
    
    # Output
    topo.write_topo("idealized_topo.nc")

**Workflow 3: Load and edit interactively**

.. code-block:: python

    # Load existing bathymetry with version control
    topo = Topo.from_version_control("TopoLibrary/my_domain")
    
    # Make edits
    topo.apply_ridge(...)
    
    # Undo if needed
    topo.tcm.undo()
    
    # Save changes
    topo.save()

Documentation Index
-------------------

- **Mask and Depth Calculation** — Details on the separation-of-concerns design for depth and masking

Related Documentation
---------------------

- See the main :doc:`../quickstart` for basic grid and bathymetry setup
- Check :doc:`../examples` for Jupyter notebook examples
