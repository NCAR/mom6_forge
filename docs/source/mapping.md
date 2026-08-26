# Regridding & Mapping

`mom6_forge.mapping` is the regridding machinery behind bathymetry ingestion
(`Topo.set_from_dataset` and friends — see {doc}`bathymetry_workflow`) and
coupler mapping-file generation. Most users won't call it directly for
bathymetry, but it's the direct entry point for generating mapping files
outside that workflow — most notably river-runoff-to-ocean mapping.

## ESMF Meshes

Most functions here operate on ESMF mesh files (see {term}`ESMF mesh`), the
format expected by the NUOPC coupler and by `xESMF`/`ESMPy`.
`grid_from_esmf_mesh` converts one back into a 2D lon/lat/mask grid dataset
for use with `xESMF`.

## Runoff Mapping

`gen_rof_maps` builds the mapping file(s) MOM6 needs to route river runoff
from a runoff (ROF) mesh onto the ocean grid:

```python
from mom6_forge.mapping import gen_rof_maps

gen_rof_maps(
    rof_mesh_path="rof_ESMF_mesh.nc",
    ocn_mesh_path="ocean_ESMF_mesh.nc",
    output_dir="maps/",
    mapping_file_prefix="rx1_to_my_ocean",
    rmax=300.0,  # smoothing radius, km
    fold=600.0,  # smoothing decay length, km
)
```

It always produces a nearest-neighbor map (`rx1_to_my_ocean_nn.nc`), which
routes each runoff cell straight into its nearest ocean cell — the problem
with this alone is that it dumps an entire river's freshwater into a single
coastal cell, which is rarely physical.

Passing `rmax`/`fold` additionally produces a **smoothed** nearest-neighbor
map (`rx1_to_my_ocean_nnsm.nc`) that spreads each injection across nearby
ocean cells instead of a single point:

* `rmax` (km) is the cutoff radius — only ocean cells within this distance of
  the injection point receive any of it.
* `fold` (km) is the decay length of the smoothing kernel,
  `weight = exp(-distance / fold)` — a smaller `fold` concentrates the
  injection near the coast, a larger one spreads it further before decaying.

Leave both `None` to skip smoothing and get only the nearest-neighbor map.
Rather than guessing values, `get_suggested_smoothing_params(ocn_mesh_path)`
derives a reasonable `rmax`/`fold` pair from the ocean mesh's own average
resolution (`rmax ≈ 5x` the average cell size, `fold = 2x rmax`), and
`gen_rof_maps` itself warns if a much larger `rmax` is passed than that,
since it can blow up compute time and memory.

## Generating a Mapping File

For any other component-to-component mapping, `generate_ESMF_map_via_xesmf`
builds a reusable weights file:

```python
from mom6_forge.mapping import generate_ESMF_map_via_xesmf

generate_ESMF_map_via_xesmf(
    src_mesh_path="src_ESMF_mesh.nc",
    dst_mesh_path="ocean_ESMF_mesh.nc",
    mapping_file="src_to_ocean_bilinear.nc",
    method="bilinear",  # or 'conservative', 'nearest_s2d', 'nearest_d2s'
)
```

`generate_ESMF_map_via_esmpy` does the same via `ESMPy` directly, which can
be preferable in MPI/HPC contexts.

## Regridding a Dataset

`regrid_dataset_via_xesmf` regrids a dataset onto a destination grid via
`xESMF` (bilinear by default), reusing a precomputed weights file via
`weights_path`/`reuse_weights=True` if given.

## Cressman Interpolation

A plain bilinear regrid smears the coastline when the source is much
finer-resolution than the model grid, letting land elevations bleed into
nearby ocean depths. `regrid_dataset_via_cressman` avoids this: for each
destination ocean cell, source ocean points within radius
`L = smooth_scl * sqrt(cell_area)` are averaged with weight

```{math}
w = \left(\frac{L^2 - r^2}{L^2 + r^2}\right)^c
```

(`r` = great-circle distance, `c` = `cressman_exp`), mirroring tx2_3's
`interp_smooth.f90`. Only ocean points contribute, so coastal depths aren't
land-contaminated; cells with no coverage within `L` fall back to
`mom6_forge.utils.iterative_fill`.

This is exposed on `Topo` via `direct_cressman_interp()`, one of the two
depth methods `Topo.set_from_dataset` can pick automatically (see
{doc}`bathymetry_workflow`). For a full walkthrough, see
`8_cressman_interpolation.ipynb`.

## Subsampling Statistics

When the source is much finer than the grid, `regrid_with_subsampling` and
`Topo.compute_stats`/`set_depth_from_stats` compute per-cell statistics (mean
depth, ocean fraction, etc.) over sub-sampled points instead of relying on a
single interpolated value — the "stats-based" path in
{doc}`bathymetry_workflow`.
