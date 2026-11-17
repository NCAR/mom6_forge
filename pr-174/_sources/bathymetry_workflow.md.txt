# Bathymetry Workflow

*Insights from Frank Bryan, Fred Castruccio, Gustavo Marques, Ashley Barnes, and others.*

`Topo.set_from_dataset` (see {doc}`quickstart`) is the flagship function of the Topo class. 
It takes a dataset and puts in on the Topo object. It is built out of several
smaller, independently-usable methods on `Topo`. This page documents each of
them — their arguments and the reasoning behind them — so they can be reached
for individually when the one-call workflow's defaults aren't the right fit.
For the regridding math itself (Cressman weights, xESMF), see {doc}`mapping`.

## Resolution Diagnosis

`Topo.diagnose_resolution(radius=6.378e6)` compares the model grid's median
cell spacing to the source dataset's pixel spacing and recommends a strategy:
below roughly a 12x ratio, a direct xESMF regrid is recommended; at or above
it, sub-sampling statistics and Cressman interpolation are recommended
instead. The 12x threshold mirrors the criterion tx2_3's `interp_smooth.f90`
uses — it's the point past which a single interpolated value per cell starts
throwing away enough real bathymetric detail that a statistical/mask-aware
approach is worth the extra cost.

## Masking Methods

`generate_mask_from_stats_ocean_frac(mask_threshold=0.5)` derives a land/ocean
mask from the per-cell statistics computed by `compute_stats` (see below):
a cell is ocean if at least `mask_threshold` of its sub-sampled points are
below the land cutoff depth. This mirrors tx2_3's `create_model_topo.f90` and
is the better choice when the source data is much finer than the grid,
because it's a majority vote over many real source points rather than a
single interpolated elevation value.

`generate_mask_from_naturalearth(resolution="10", version="v5_1_2")` derives
a mask from Natural Earth land polygons (via `regionmask`) instead of the
source bathymetry's own land/ocean values. Useful when substats isn't warranted.


## Depth Methods

`compute_stats(nx_sub, ny_sub, mask_hmin)` sub-samples each grid cell into
`nx_sub` × `ny_sub` points, snaps each to the nearest source pixel, and
returns per-cell `OCN_FRAC`, `D_mean`, `D_min`, `D_max`, and `D2_mean`.
`mask_hmin` is the land cutoff depth — typically 0 m for regional domains,
but -1 m for tx2_3, since what counts as "ocean" for masking purposes isn't
always exactly sea level. This method is a Python port of code Frank Bryan
originally wrote in Fortran for tx2_3; computing all of these statistics in
one sub-sampling pass avoids re-reading the (often very large) source dataset
once per depth/mask decision.

`set_depth_from_stats(statistic)` sets depth directly from one of the `D_*`
statistics above (`"mean"`, `"min"`, `"max"`, ...) — the cheapest depth
option once statistics are already computed.

`direct_cressman_interp(smooth_scl=2.0, cressman_exp=2.0, weights_path=None)`
sets depth via mask-aware Cressman distance-weighted interpolation (see
{doc}`mapping` for the weighting formula). Because only source ocean points
contribute to each destination cell, it avoids the coastal land-contamination
that a plain interpolation produces — the main weakness a resolution-ratio
based choice (see above) is meant to correct for.

## Cleanup

`fill_inland_lakes_and_channels()` fills one-cell-wide channels and inland
lakes left behind by either masking method. It takes no arguments and is
generally run as a last step, since masks derived from sub-sampling or land
polygons commonly leave small artifacts like these.

## Decision Flow

`set_from_dataset` exists so the resolution-ratio heuristic above doesn't
need to be applied by hand every time. Internally, it calls
`diagnose_resolution()` exactly once and reuses that single result to drive
two *separate* decisions — one for masking, one for depth — each of which
only falls back to the automatic pick if you didn't pass that argument
yourself.

### Allowed Values

```{list-table} mask_method
:header-rows: 1
:widths: 20 45 35

* - Value
  - What it does
  - Extra requirements
* - `'naturalearth'`
  - Natural Earth land polygons via `regionmask`
  - none
* - `'ocean_frac'`
  - Threshold the sub-sampled ocean fraction from `compute_stats`
  - `nx_sub`, `ny_sub`, `mask_hmin` (auto-derived from grid/source
    resolution if omitted; `mask_hmin` defaults to 0 m)
* - `'dataset'`
  - Derive the mask directly from the source dataset's own raw depth sign
  - none
* - `'manual'`
  - Use a `user_mask` you've already set on the `Topo` instance
  - `user_mask` must already be set — raises otherwise
* - `None` (default)
  - Auto-pick `'ocean_frac'` or `'naturalearth'` from `diagnose_resolution()`
  - —
```

```{list-table} depth_method
:header-rows: 1
:widths: 20 45 35

* - Value
  - What it does
  - Extra requirements
* - `'stats'`
  - Use the `D_mean` statistic from `compute_stats` (fixed to `"mean"` when
    chosen through `set_from_dataset`; call `set_depth_from_stats` directly
    for `min`/`max`)
  - `nx_sub`, `ny_sub`, `mask_hmin` (same as `'ocean_frac'` above)
* - `'cressman'`
  - Mask-aware Cressman distance-weighted interpolation
    (`direct_cressman_interp`)
  - none
* - `'xesmf'`
  - Direct xESMF regrid of the source dataset (`set_depth_from_xesmf`)
  - honors `regridding_method` (default `'bilinear'`)
* - `None` (default)
  - Auto-pick `'cressman'` or `'xesmf'` from `diagnose_resolution()`
  - —
```

### Automatic Selection

Only two of each argument's values are ever chosen automatically, and which
one depends on the same resolution-ratio check:

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Argument
  - Picked automatically when resolution ratio is below 12x (grid comparable
    to or finer than the source)
  - Picked automatically when resolution ratio is 12x or above (source much
    finer than the grid)
* - `mask_method`
  - `'naturalearth'`
  - `'ocean_frac'`
* - `depth_method`
  - `'xesmf'`
  - `'cressman'`
```

`'dataset'`/`'manual'` (mask) and `'stats'` (depth) are never picked
automatically — they only run if you ask for them explicitly, since none of
them is a clear default for either resolution regime.

Because masking and depth are decided independently, you can override either
one on its own without touching the other — the two don't have to move
together. Passing `mask_method="naturalearth"` on a high-ratio grid still
lets depth auto-pick Cressman; passing `depth_method="stats"` on a low-ratio
grid still lets the mask auto-pick Natural Earth.

```python
# Force the Natural Earth mask, but still let depth auto-pick Cressman
# on a grid that's much coarser than the source.
topo.set_from_dataset(
    bathymetry_path="GEBCO_2023.nc",
    longitude_coordinate_name="lon",
    latitude_coordinate_name="lat",
    vertical_coordinate_name="elevation",
    mask_method="naturalearth",
)
```

`mask_method="ocean_frac"` and `depth_method="stats"` both rely on the same
`compute_stats` call, so `nx_sub`/`ny_sub`/`mask_hmin` (passed through
`set_from_dataset`'s `**kwargs`) only need to be given once even if both are
in play. `diagnose_resolution`'s printout runs on every call regardless of
which methods end up chosen, so you always see the measured ratio and which
path it recommends, even when you're overriding it. `fill_channels=True` runs
`fill_inland_lakes_and_channels` afterward regardless of which mask/depth
combination was used.

## Channel Width Constraints

`ChannelWidth`/`ChannelWidthList` (see {doc}`quickstart`) exist alongside
the depth/mask methods above for a different reason: some straits and
channels are narrower than a single grid cell but still need to permit flow.
Rather than distorting the discretized bathymetry to force this, channel
width constraints record an effective width override that's applied at
runtime, independent of the depth and mask fields.
