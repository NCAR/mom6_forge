# Glossary

```{glossary}
supergrid
    A MOM6 supergrid contains the grid metrics and the areas at twice the
    nominal resolution of the actual computational grid. During runtime,
    MOM6 reads in the supergrid file, and then decomposes the supergrid into the four
    staggered grids, each containing different sets of prognostic variables, e.g.,
    tracers, velocities, etc.

tripolar grid
    A grid with three poles instead of one, typically with two of the three
    poles displaced into Northern Hemisphere land masses so that no
    singularity falls within the ocean domain. Commonly used for global MOM6
    configurations. `mom6_forge` can modify but not yet create tripolar grids
    from scratch.

displaced pole grid
    A grid in which a single pole is moved off of the true geographic pole
    and into a land mass, avoiding a singularity within the ocean domain
    without the added complexity of a tripolar grid.

ESMF mesh
    An unstructured-mesh NetCDF file format (produced by `Topo.write_esmf_mesh`)
    used by the NUOPC coupler in CESM to acquire grid and land/ocean mask
    information for a component. Most functions in {doc}`mapping` operate on
    ESMF mesh files rather than raw longitude/latitude arrays.

SCRIP grid
    An older NetCDF grid description format used by the SCRIP regridding
    library. Superseded by ESMF mesh files for most purposes in modern CESM,
    but still required to generate ocean-runoff mapping files (see
    {doc}`mapping`). Written by `Topo.write_scrip_grid`.

CICE grid
    The grid file format expected by the CICE sea ice model, describing the
    same horizontal grid as the MOM6 supergrid and topography files. Written
    by `Topo.write_cice_grid` when a configuration includes a CICE component.

Cressman interpolation
    A distance-weighted interpolation scheme in which source points within a
    smoothing radius contribute to a destination value with weight
    decreasing as a function of distance. Used by `mom6_forge` (see
    {doc}`mapping`) to fill bathymetry from a coarser source dataset without
    letting nearby land elevations contaminate coastal ocean depths.

TOPO_CONFIG
    A MOM6 runtime parameter (set in `MOM_input`) selecting an idealized
    bathymetry shape, such as `flat`, `bowl`, or `spoon`. The `Topo` class's
    `set_flat`, `set_bowl`, and `set_spoon` methods produce bathymetry
    matching these same idealized configurations.

channel width constraint
    An effective width override for a strait or channel that is too narrow to
    be resolved at the model's grid resolution but still needs to permit
    flow. Managed by `ChannelWidth`/`ChannelWidthList` and applied on top of
    the bathymetry at runtime, independently of the depth/mask fields
    themselves.
```
