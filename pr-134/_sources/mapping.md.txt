# Regridding & Mapping

`mom6_forge.mapping` provides the lower-level regridding machinery that powers
bathymetry ingestion (`Topo.set_from_dataset`, `Topo.set_depth_from_xesmf`,
`Topo.direct_cressman_interp`) and the generation of coupler mapping files
between component grids (e.g. runoff-to-ocean). Most users will not call it
directly — the `Topo` methods described in {doc}`quickstart` are the
recommended entry points for bathymetry — but it is useful to understand what
is happening underneath, and it is needed directly for generating coupler
mapping files outside of the bathymetry workflow.

## ESMF Meshes

Most functions in this module operate on ESMF mesh files (see {term}`ESMF
mesh`) rather than raw lon/lat arrays, since ESMF meshes are the format
expected by the NUOPC coupler and by `xESMF`/`ESMPy`. `grid_from_esmf_mesh`
converts a (possibly unstructured/flattened) ESMF mesh back into a 2D grid
dataset of longitude, latitude, and mask, suitable for use with `xESMF`.

## Generating a Mapping File

To generate a reusable ESMF mapping (weights) file between two component
grids:

```python
from mom6_forge.mapping import generate_ESMF_map_via_xesmf

generate_ESMF_map_via_xesmf(
    src_mesh_path="src_ESMF_mesh.nc",
    dst_mesh_path="ocean_ESMF_mesh.nc",
    mapping_file="src_to_ocean_bilinear.nc",
    method="bilinear",  # or 'conservative', 'nearest_s2d', 'nearest_d2s'
)
```

`generate_ESMF_map_via_esmpy` provides the same functionality using `ESMPy`
directly instead of `xESMF`, which can be preferable in MPI/HPC contexts.

For river-runoff-to-ocean mapping specifically, `gen_rof_maps` generates both
a nearest-neighbor and a smoothed nearest-neighbor mapping file in one call:

```python
from mom6_forge.mapping import gen_rof_maps

gen_rof_maps(
    rof_mesh_path="rof_ESMF_mesh.nc",
    ocn_mesh_path="ocean_ESMF_mesh.nc",
    output_dir="maps/",
    mapping_file_prefix="rx1_to_my_ocean",
    rmax=300.0,  # smoothing radius, km
    fold=1.0,
)
```

Note that ocean-runoff mapping still requires a {term}`SCRIP grid` file
(`Topo.write_scrip_grid`), even in configurations that otherwise use ESMF
meshes exclusively.

## Regridding a Dataset

`regrid_dataset_via_xesmf` regrids an arbitrary dataset onto a destination
grid using `xESMF` (bilinear by default), and can reuse a previously computed
weights file via `weights_path`/`reuse_weights=True` instead of recomputing
weights on every call.

## Cressman Interpolation

For coarse-to-fine bathymetry ingestion, where a simple bilinear regrid would
smear the coastline by letting nearby land elevations contaminate ocean
depths, `regrid_dataset_via_cressman` and the underlying
`compute_cressman_weights` implement a mask-aware Cressman distance-weighted
interpolation, mirroring the `interp_smooth.f90` program from the `tx2_3`
high-resolution topography workflow. For each destination ocean cell, source
ocean points within a smoothing radius `L = smooth_scl * sqrt(cell_area)` are
averaged with weight

```{math}
w = \left(\frac{L^2 - r^2}{L^2 + r^2}\right)^c
```

where `r` is the great-circle distance and `c` is `cressman_exp`. Only ocean
source points contribute, so estimates near the coast are not contaminated by
land elevations. Destination cells that receive no source coverage within `L`
are left for a fallback iterative-fill pass (see
`mom6_forge.utils.iterative_fill`).

This is exposed directly on `Topo` via `direct_cressman_interp()`, and is one
of the two depth methods `Topo.set_from_dataset` can choose automatically
(see {doc}`quickstart`). For a worked, from-scratch walkthrough of the math
and weight computation, see the `8_cressman_interpolation.ipynb` notebook.

## Subsampling Statistics

When the source dataset is much finer than the model grid, `regrid_with_subsampling`
and the `Topo.compute_stats`/`Topo.set_depth_from_stats` methods compute
per-cell statistics (mean depth, ocean fraction, etc.) over sub-sampled source
points rather than relying on a single interpolated value — this is the
"stats-based" mask/depth path referenced in {doc}`quickstart`.
