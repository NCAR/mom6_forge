Regional Bathymetry Workflows
=============================

``mom6_forge`` provides two bathymetry pipelines for regional MOM6 configurations,
both accessible through the :class:`~mom6_forge.topo.Topo` class. They differ in how
depth is assigned and are suited to different grid resolutions.

.. contents:: Contents
   :local:
   :depth: 2


Background
----------

The default :meth:`~mom6_forge.topo.Topo.direct_xesmf_regrid` pipeline regrids a
source bathymetry (e.g. GEBCO) onto the model grid using ``xesmf`` and derives the
land/ocean mask from whether the regridded depth is positive. This is fast and
sufficient for fine regional grids (~1–3 km), but has two limitations at coarser
resolutions:

1. **Land contamination near coasts.** Standard bilinear or conservative regridding
   blends land elevations into ocean depth estimates for cells that straddle the
   coastline.

2. **Mask accuracy.** Deriving the mask from the sign of the regridded depth is
   sensitive to regridding artefacts. At coarser resolutions (≳ 0.05°), computing
   the ocean fraction directly from the high-resolution source data produces a
   more accurate mask.

The :meth:`~mom6_forge.topo.Topo.high_res_regrid` pipeline addresses both issues
using a Monte-Carlo ocean fraction mask and Cressman distance-weighted interpolation.
It is recommended for grids of ~0.05° (5 km) and coarser.


Mask Generation
---------------

Both pipelines support an optional external mask. Two methods are provided.

Ocean Fraction Mask (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mom6_forge.topo.Topo.generate_mask_ocean_frac

For each model grid cell, ``nx_sub × ny_sub`` interior sample points are
distributed using bilinear interpolation of the 4 supergrid corner coordinates.
Each point is snapped to the nearest source bathymetry pixel. The ocean fraction
``OCN_FRAC`` is the number of sample points with depth > ``mask_hmin`` divided by
the total. Cells with ``OCN_FRAC ≥ mask_threshold`` are marked as ocean.

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
and minimum depth enforcement.

Accepts an optional ``mask`` argument. When provided, ``tidy_dataset`` uses it
directly instead of deriving the mask from the regridded depth. When omitted, the
behaviour is identical to the previous ``set_from_dataset`` method.

.. code-block:: python

    topo.direct_xesmf_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
    )

    # With an external mask:
    mask, ocn_frac = topo.generate_mask_ocean_frac("gebco_2023.nc", nx_sub=5, ny_sub=5)
    topo.direct_xesmf_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask=mask,
    )

Pipeline B — high_res_regrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: mom6_forge.topo.Topo.high_res_regrid

The high-accuracy pipeline for coarser grids. Internally runs:

1. :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac` to build the mask and
   compute depth statistics.
2. :meth:`~mom6_forge.topo.Topo.cressman_interp` to assign ocean-mask-aware
   smoothed depths.
3. :meth:`~mom6_forge.topo.Topo.tidy_dataset` for lake removal, channel cleanup,
   and minimum depth enforcement.

.. code-block:: python

    topo.high_res_regrid(
        bathymetry_path="gebco_2023.nc",
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        nx_sub=5,
        ny_sub=5,
        smooth_scl=2.0,
    )

Cressman Interpolation
-----------------------

.. automethod:: mom6_forge.topo.Topo.cressman_interp

Assigns depth to each ocean cell by distance-weighted averaging of source
bathymetry points within a smoothing radius ``L``:

.. math::

    D = \frac{\sum_i w_i \, d_i}{\sum_i w_i}, \quad
    w_i = \left(\frac{L^2 - r_i^2}{L^2 + r_i^2}\right)^{c}

where :math:`r_i` is the great-circle distance from the model cell centre to
source point :math:`i`, :math:`L = \text{smooth\_scl} \times \sqrt{A_\text{cell}}`
is the smoothing radius, and :math:`c` is ``cressman_exp``. Source points with
depth ≤ 0 (land) are excluded.

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
     - Use ``high_res_regrid`` — Cressman gives meaningfully better depths


.. _topo-drag:

Topo Drag Statistics
--------------------

.. automethod:: mom6_forge.topo.Topo.write_topo_drag

MOM6 Lee-wave and bottom-drag parameterisations require a measure of subgrid
topographic roughness at each cell. This is quantified as the variance of ocean
depth within each model cell:

.. math::

    h_2 = \overline{D^2} - \bar{D}^2

where :math:`\overline{D^2}` is ``D2_mean`` and :math:`\bar{D}` is ``D_mean``
from the Monte-Carlo sub-sampling step.

:meth:`~mom6_forge.topo.Topo.write_topo_drag` writes a netCDF file containing
``h2`` in units of ``meters^2``. This file is read by MOM6 at initialisation when
topo drag is enabled.

.. note::

    ``write_topo_drag`` requires
    :meth:`~mom6_forge.topo.Topo.generate_mask_ocean_frac` (or
    :meth:`~mom6_forge.topo.Topo.high_res_regrid`) to have been called first.
    It will raise a ``RuntimeError`` otherwise.

.. code-block:: python

    topo.high_res_regrid(bathymetry_path="gebco_2023.nc", ...)
    topo.write_topo_drag("topo_drag.nc")


See Also
--------

- :doc:`Notebook 7 — Regional Bathymetry Workflow <../notebooks/7_regional_bathy_workflow>`
- :class:`~mom6_forge.topo.Topo`
- `GEBCO bathymetry <https://www.gebco.net>`_
