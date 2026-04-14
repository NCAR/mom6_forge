Regional Bathymetry Workflows
=============================

``mom6_forge`` provides bathymetry pipelines for regional MOM6 configurations,
accessible through the :class:`~mom6_forge.topo.Topo` class. Two pipelines are
available, chosen automatically based on grid resolution.

.. contents:: Contents
   :local:
   :depth: 2


Background
----------

The standard pipeline regrids a source bathymetry (e.g. GEBCO) onto the model
grid using ``xesmf`` and derives the land/ocean mask from whether the regridded
depth is positive. This is fast and sufficient for fine regional grids (~1–3 km),
but has two limitations at coarser resolutions:

1. **Land contamination near coasts.** Standard bilinear or conservative regridding
   blends land elevations into ocean depth estimates for cells that straddle the
   coastline.

2. **Mask accuracy.** Deriving the mask from the sign of the regridded depth is
   sensitive to regridding artefacts. At coarser resolutions (≳ 0.05°), computing
   the ocean fraction directly from the high-resolution source data produces a
   more accurate mask.

A higher-accuracy pipeline using a Monte-Carlo ocean fraction mask and Cressman
distance-weighted interpolation is recommended for grids of ~0.05° (5 km) and
coarser. The top-level entry point :meth:`~mom6_forge.topo.Topo.set_from_dataset`
calls :meth:`~mom6_forge.topo.Topo.diagnose_resolution` automatically and routes
to the appropriate pipeline.


Pipeline Dispatch
-----------------

.. automethod:: mom6_forge.topo.Topo.set_from_dataset

``set_from_dataset`` is the recommended entry point for most workflows. It
inspects the model grid spacing relative to the source bathymetry pixel size and
dispatches to the appropriate pipeline::

    set_from_dataset()
        ├── diagnose_resolution() → True  (ratio ≥ 12×)
        │       └── high_res_regrid(mask_method="ocean_frac" default)
        │               ├── mask options: ocean_frac | cartopy | pre-computed DataArray
        │               ├── cressman_interp()
        │               └── tidy_dataset(mask=mask)
        │
        └── diagnose_resolution() → False (ratio < 12×)
                └── direct_xesmf_regrid(mask_method=None default)
                        ├── mask options: ocean_frac | cartopy | pre-computed DataArray
                        │                | None (derive from depth sign)
                        ├── config_dataset() + regrid_dataset_via_xesmf()
                        └── tidy_dataset(mask=mask)

The ``mask_method`` and ``mask`` keyword arguments are forwarded transparently to
whichever pipeline is selected.

.. code-block:: python

    # Fully automatic — pipeline chosen by diagnose_resolution
    topo.set_from_dataset(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
    )

    # Force ocean-fraction mask regardless of auto-dispatch
    topo.set_from_dataset(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="ocean_frac",
    )


Resolution Diagnosis
--------------------

.. automethod:: mom6_forge.topo.Topo.diagnose_resolution

Before choosing a pipeline manually, call ``diagnose_resolution`` to compare
the model grid spacing with the source bathymetry resolution:

.. code-block:: python

    topo.diagnose_resolution(
        "gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
    )

This prints the median, minimum, and maximum model cell spacing alongside the
dataset pixel size, computes the resolution ratio, and returns ``True`` if the
ratio is ≥ 12× (the point at which the stats-based mask and Cressman
interpolation provide meaningfully better results than plain ``xesmf``
regridding).


Mask Generation
---------------

Both pipelines accept an external mask in three ways:

1. **``mask=<DataArray>``** — pass a pre-computed boolean/integer ocean mask
   directly; no mask generation step is run.
2. **``mask_method="ocean_frac"``** — call
   :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac` internally.
3. **``mask_method="cartopy"``** — call
   :meth:`~mom6_forge.topo.Topo.generate_mask_cartopy` internally.

``direct_xesmf_regrid`` additionally supports **``mask=None``** (the default),
which skips all explicit mask generation and lets
:meth:`~mom6_forge.topo.Topo.tidy_dataset` derive the mask from the sign of the
regridded depth — the original behaviour for fine grids.

Ocean Fraction Mask (recommended for coarse grids)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mom6_forge.topo.Topo.generate_mask_ocean_frac

For each model grid cell, ``nx_sub × ny_sub`` interior sample points are
distributed using bilinear interpolation of the 4 supergrid corner coordinates.
Each point is snapped to the nearest source bathymetry pixel. The ocean fraction
``OCN_FRAC`` is the number of sample points with depth > ``mask_hmin`` divided
by the total. Cells with ``OCN_FRAC ≥ mask_threshold`` are marked as ocean.

This method also computes and stores per-cell depth statistics
(``D_mean``, ``D_min``, ``D_max``, ``D2_mean``) which are needed for topo drag
parameterisations (see :ref:`topo-drag`).

**When to use:** When the model grid is coarser than ~0.05° or when topo drag
statistics are required.

Cartopy Coastline Mask
~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mom6_forge.topo.Topo.generate_mask_cartopy

Rasterises Natural Earth coastline vectors onto the model grid using Cartopy.
Fast and independent of the bathymetry dataset. Use as a quick first-pass mask
or for grids where consistency with the bathymetry source is less critical.

**When to use:** Quick runs, coarse grids, or as a comparison against
``generate_mask_ocean_frac``.


Pipelines
---------

Pipeline A — direct_xesmf_regrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mom6_forge.topo.Topo.direct_xesmf_regrid

The standard pipeline. Regrids the source bathymetry with ``xesmf`` then runs
:meth:`~mom6_forge.topo.Topo.tidy_dataset` for lake removal, channel cleanup,
and minimum depth enforcement. Suitable for fine regional grids (~1–3 km).

Mask options:

- ``mask=None`` *(default)* — derive ocean mask from sign of regridded depth
- ``mask_method="ocean_frac"`` — use :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac`
- ``mask_method="cartopy"`` — use :meth:`~mom6_forge.topo.Topo.generate_mask_cartopy`
- ``mask=<DataArray>`` — supply a pre-computed ocean mask

.. code-block:: python

    # Default: derive mask from depth sign
    topo.direct_xesmf_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
    )

    # With ocean-fraction mask
    topo.direct_xesmf_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="ocean_frac",
        nx_sub=5,
        ny_sub=5,
    )


Pipeline B — high_res_regrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mom6_forge.topo.Topo.high_res_regrid

The high-accuracy pipeline for coarser grids. Internally runs:

1. Mask generation (``generate_mask_ocean_frac`` by default, or ``generate_mask_cartopy``,
   or a pre-computed ``mask`` DataArray)
2. :meth:`~mom6_forge.topo.Topo.cressman_interp` — ocean-aware distance-weighted
   depth interpolation
3. :meth:`~mom6_forge.topo.Topo.tidy_dataset` — lake removal, minimum depth

Mask options:

- ``mask_method="ocean_frac"`` *(default)* — use :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac`
- ``mask_method="cartopy"`` — use :meth:`~mom6_forge.topo.Topo.generate_mask_cartopy`
- ``mask=<DataArray>`` — supply a pre-computed ocean mask

Note that ``mask=None`` is not a valid option here because Cressman interpolation
requires an explicit ocean mask to exclude land source points.

.. code-block:: python

    # Default: ocean-fraction mask + Cressman
    topo.high_res_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
    )

    # Cartopy mask variant
    topo.high_res_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="cartopy",
    )

    # Pre-computed mask
    topo.high_res_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask=my_ocean_mask,
    )


Cressman Interpolation
----------------------

.. automethod:: mom6_forge.topo.Topo.cressman_interp

Assigns depth to each ocean cell by distance-weighted averaging of source
bathymetry points within a per-cell smoothing radius ``L``:

.. math::

    D = \frac{\sum_i w_i \, d_i}{\sum_i w_i}, \quad
    w_i = \left(\frac{L^2 - r_i^2}{L^2 + r_i^2}\right)^{c}

where :math:`r_i` is the great-circle distance from the model cell centre to
source point :math:`i`, :math:`L = \text{smooth\_scl} \times \sqrt{A_\text{cell}}`
is the per-cell smoothing radius, and :math:`c` is ``cressman_exp``. Source
points with depth ≤ 0 (land) are excluded from the weighted average.

Cells that receive no ocean source points within their radius are filled
iteratively from their nearest filled neighbours (vectorised numpy approach
modelled after the Fortran ``interp_smooth.f90`` in tx2_3).

The interpolation weights are saved to an ESMF-compatible netCDF file (``S``,
``row``, ``col`` variables) so that the ``xesmf`` regridder can apply them via
``reuse_weights=True``. Pass ``weights_path`` to control where this file is
written; if omitted a temporary file is used.

**Recommendation by resolution:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Grid spacing
     - Recommendation
   * - < 3 km
     - Use ``direct_xesmf_regrid`` — Cressman adds little benefit
   * - 3–10 km (0.05°)
     - Borderline; Cressman improves coastline accuracy
   * - > 10 km (0.1°+)
     - Cressman gives meaningfully better depths; use ``high_res_regrid``


.. _topo-drag:

Topo Drag Statistics
--------------------

MOM6 Lee-wave and bottom-drag parameterisations require a measure of subgrid
topographic roughness at each cell. This is quantified as the variance of ocean
depth within each model cell:

.. math::

    h_2 = \overline{D^2} - \bar{D}^2

where :math:`\overline{D^2}` is ``D2_mean`` and :math:`\bar{D}` is ``D_mean``
from the Monte-Carlo sub-sampling step in
:meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac`.

When :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac` has been called,
:meth:`~mom6_forge.topo.Topo.write_topo` automatically includes the ``h2``
variable (units: ``m²``) in the output file. To make the presence of ``h2``
mandatory, pass ``enforce_topo_drag=True``; this raises a ``RuntimeError`` if the
stats have not been computed yet.

.. automethod:: mom6_forge.topo.Topo.write_topo

.. note::

    ``h2`` requires
    :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac` to have been called
    first. Without it, ``write_topo`` writes the depth field only.

.. code-block:: python

    topo.generate_mask_ocean_frac(bathymetry_path="gebco_2023.nc", nx_sub=5, ny_sub=5)
    topo.write_topo("topog.nc")                     # h2 included automatically

    # Alternatively, raise if stats are missing:
    topo.write_topo("topog.nc", enforce_topo_drag=True)


MPI Regridding
--------------

.. automethod:: mom6_forge.topo.Topo.mpi_direct_xesmf_regrid

A parallelised variant of ``direct_xesmf_regrid`` for very large source
datasets. Equivalent to ``direct_xesmf_regrid`` but distributes the ``xesmf``
regridding across MPI ranks.


See Also
--------

- :doc:`Notebook 7 — Regional Bathymetry Workflow <../notebooks/7_regional_bathy_workflow>`
- :class:`~mom6_forge.topo.Topo`
- `GEBCO bathymetry <https://www.gebco.net>`_
