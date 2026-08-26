# Quickstart Guide

`mom6_forge` library can be utilized via its Python API, i.e., directly within Python
scripts or within Jupyter notebooks. In this quickstart guide, we describe how
the tool can be utilized within a Jupyter Notebook, but the majority of
these instructions apply to Python scripts as well.

## Step 1: Import modules

The first step is to import the `Grid` and `Topo` classes of the
`mom6_forge` package. The `Grid` class represents
horizontal MOM6 grids, and is to be instantiated with the desired grid
configuration and resolution. After creating a grid instance, a `Topo` class
instance is to be created to generate an associated bathymetry.

```python
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
```

## Step 2: Create the horizontal grid

After having imported the modules, we can now create a horizontal grid.
An example Grid instantiation:

```python
grid = Grid(
    nx         = 180,         # Number of grid points in x direction
    ny         = 80,          # Number of grid points in y direction
    lenx       = 360.0,       # grid length in x direction, e.g., 360.0 (degrees)
    leny       = 160,         # grid length in y direction
    cyclic_x   = True,        # reentrant, spherical domain
    ystart     = -80.0        # start/end 10 degrees above/below poles to avoid singularity 
)
```

In the above example, the `Grid` object, named `grid`, is constructed by
specifying the required arguments `nx`, `ny`, `config`, `axis_units`, `lenx`,
and `leny`, in addition to the optional argument `ystart`. The full list of
`Grid` arguments and their descriptions may be printed by running
`Grid?` statement on a notebook cell:

```
Grid?

...

Parameters
----------
nx : int
    Number of grid points in x direction
ny : int
    Number of grid points in y direction
lenx : float
    grid length in x direction, e.g., 360.0 (degrees)
leny : float
    grid length in y direction, e.g., 160.0 (degrees)
srefine : int, optional
    refinement factor for the supergrid. 2 by default
xstart : float, optional
    starting x coordinate. 0.0 by default.
ystart : float, optional
    starting y coordinate. -0.5*leny by default.
cyclic_x : bool, optional
    flag to make the grid cyclic in x direction. False by default.
tripolar_n : bool, optional
    flag to make the grid tripolar. False by default.
displace_pole : bool, optional
    flag to make the grid displaced polar. False by default.
```

Note that tripolar and displaced pole grids cannot yet be created from scratch,
but existing tripolar and displaced pole grids can be modified via mom6_forge.

Instead of specifying `nx`/`ny` directly, they can be derived from a target
`resolution` (in the grid's axis units):

```python
grid = Grid(lenx=20.0, leny=10.0, resolution=0.25, xstart=270.0, ystart=10.0)
```

### *Alternative Grid Constructors*

Beyond the uniform-degree grid shown above, `Grid` provides several alternative
constructors for domains that aren't naturally expressed as `nx`/`ny`/`lenx`/`leny`:

* `Grid.from_center(center_lat, center_lon, width_m, height_m, resolution_m, angle_deg=0.0)`
  — builds a rectangular grid centred at a geographic point using an azimuthal
  equidistant projection, optionally rotated `angle_deg` degrees clockwise from
  north. Useful for aligning a regional domain with a coastline or estuary.
* `Grid.from_projection(crs, x_min, x_max, y_min, y_max, resolution_m)`
  — builds a uniform grid in a given `pyproj` CRS (e.g. a polar stereographic or
  Lambert conformal projection) and reprojects it to geographic coordinates,
  remaining accurate at high latitudes.
* `Grid.from_supergrid(path)` — loads a grid from an existing MOM6 supergrid file.
* `Grid.from_esmf_mesh(path)` — loads a grid from an existing ESMF mesh file.

Example:

```python
grid = Grid.from_center(
    center_lat=44.9,
    center_lon=-63.5,
    width_m=400_000,
    height_m=250_000,
    resolution_m=5_000,
    angle_deg=30.0,
)
```

These constructors are also available interactively through the `GridCreator`
widget — see {doc}`widgets`.

### *Avoiding singularity points*

To avoid singularity points within the ocean grid:
  * The grid poles (which may be different than the true poles) must be left out of the grid,
    by making sure that the extent of the grid in the y-direction do not cover the poles, 
    e.g., by setting `ystart` to -80.0 degrees
    and `leny` to 160.0 degrees.
  * Alternatively, one or two singularities (typically, in the northern hemisphere) may be 
    displaced into land masses if `displace_pole` or `tripolar_n` options are to be used.
    The other singularity (typically, in the southern hemisphere) would still need to be
    left out the geographic extent of the grid.

If a singularity (a pole) is present within the ocean grid, a land component (active or data) must be added
to the pose of hiding the singularity points of spherical ocean grids within the CESM framework.

### *Grid Metrics and Attributes*

When a `Grid` instance gets created, several grid metrics and attributes
on all staggerings are automatically computed and populated. These metrics and attributes
are accessible via the accessor operator (`.`). For example, to access "the array
of t-grid longitutes" of `grid`:

```python
grid.tlon
```

The full list of grid metrics and attributes:

* `tlon`: array of t-grid longitudes
* `tlat`: array of t-grid latitudes
* `ulon`: array of u-grid longitudes
* `ulat`: array of u-grid latitudes
* `vlon`: array of v-grid longitudes
* `vlat`: array of v-grid latitudes
* `qlon`: array of corner longitudes
* `qlat`: array of corner latitudes
* `dxt`: x-distance between U points, centered at t
* `dyt`: y-distance between V points, centered at t
* `dxCv`: x-distance between q points, centered at v
* `dyCu`: y-distance between q points, centered at u
* `dxCu`: x-distance between y points, centered at u
* `dyCv`: y-distance between t points, centered at v
* `angle`: angle T-grid makes with latitude line
* `tarea`: T-cell area

### *Supergrid*

In addition to above grid metrics and attributes, the `Grid` class incorporates an
underlying {term}`supergrid` instance associated the grid instance, which is again
accessible via the (`.`) operator:

```python
grid.supergrid
```

Any user changes to coordinates, e.g., increasing the equatorial resolution,
must be applied to the supergrid using the `update_supergrid` method. This is
because the supergrid is the underlying refined grid that is used to determine the
the four staggered grids (T,U,V,Q) that forms the actual computational grid.
Users can modify the supergrid by providing a new x and y coordinate arrays, e.g.,
as follows:

```python
grid.update_supergrid(xdat, ydat)
```

where `xdat` and `ydat` are user-defined 2-dimensional numpy arrays containing
the new x and y coordinates of the supergrid. Running the `update_supergrid`
method of a `Grid` instance automatically updates all other grid metrics listed
above.

## Step 3: Create the Vertical Grid

`mom6_forge` also provides a `VGrid` class to define the vertical layering of
the ocean model, independently of the horizontal `Grid`.

```python
from mom6_forge.vgrid import VGrid
```

A vertical grid is fundamentally an array of layer thicknesses (`dz`, in
meters). `VGrid` provides two constructors for generating common vertical
spacings, plus a way to load one from an existing file.

### *Uniform Vertical Grid*

```python
vgrid = VGrid.uniform(nk=75, depth=6000.0)
```

`nk` is the number of vertical levels and `depth` is the total depth of the
water column (meters). The bottom layer's thickness is adjusted slightly so
that the sum of all layers exactly equals `depth`.

### *Hyperbolic (Stretched) Vertical Grid*

To concentrate resolution near the surface, use a hyperbolic-tangent profile,
where `ratio` is the target ratio of the bottom layer's thickness to the top
layer's thickness:

```python
vgrid = VGrid.hyperbolic(nk=75, depth=6000.0, ratio=10.0)
```

### *Loading from an Existing File*

```python
vgrid = VGrid.from_file(
    "existing_vgrid.nc",
    variable_name="dz",
    variable_type="layer_thickness",  # or "cell_center" or "cell_interface"
)
```

### *Attributes*

* `vgrid.dz`: array of layer thicknesses (meters)
* `vgrid.nk`: number of vertical levels
* `vgrid.depth`: total water column depth (meters)
* `vgrid.zl`: array of layer-center depths (meters)
* `vgrid.zi`: array of layer-interface depths (meters), size `nk + 1`

### *Writing the Vertical Grid File*

```python
vgrid.write("my_vgrid.nc")
```

`vgrid.write_z_file("my_vgrid_z.nc")` writes an alternative file containing
the interface (`zi`) and center (`zl`) depths directly, rather than
thicknesses.

Like horizontal grids, vertical grids can also be created and edited
interactively via the `VGridCreator` widget — see {doc}`widgets`.

## Step 4: Create Bathymetry

After having generated the horizontal grid, we can now create an associated bathymetry
object as follows:

```python
topo = Topo(grid, min_depth=10.0)
```

The first argument (`grid`) of `Topo` constructor is the horizontal grid instance for which
the bathymetry is to be created, while the second argument (`min_depth`) is the minimum ocean depth.
Any column in the ocean grid with a depth shallower than `min_depth` is masked out of the ocean
domain. The minimum depth attribute of a bathymetry instance may be changed afterwards using the
assignment operator. For example:

```python
topo.min_depth = 5.0
```

### *Predefined Bathymetry Configurations*

The `Topo` class provides three predefined bathymetry configurations, which are also
available in MOM6 as idealized configurations. (See `TOPO_CONFIG` parameter in MOM_input)

  * `flat`: flat bottom set to MAXIMUM_DEPTH. Example:
  * `bowl`: an analytically specified bowl-shaped basin ranging between MAXIMUM_DEPTH and MINIMUM_DEPTH.
  * `spoon`: a similar shape to 'bowl', but with an vertical wall at the southern face.

Examples:

```python
# flat bottom
topo.set_flat(D=500.0)

# bowl
topo.set_bowl(500.0, 50.0, expdecay=1e7)

# spoon
topo.set_spoon(500.0, 50.0, expdecay=1e7)
```

The first and the second arguments of `set_bowl` and `set_spoon` methods are maximum depth
and minimum depth, respectively.

Check out the following notebook to see examples of above predefined bathymetry options: [1_spherical_grid.ipynb](https://github.com/NCAR/mom6_forge/blob/master/notebooks/1_spherical_grid.ipynb)

### *Custom Bathymetry*

In addition to the above predefined configurations, users may provide their own depth arrays. For
example:

```python
import numpy as np

# define a custom depth
i = grid.tlat.nx.data                # array of x-indices
j = grid.tlat.ny.data[:,np.newaxis]  # array of y-indices 
custom_depth = 400.0 + 80.0 * np.sin(i*np.pi/6.) * np.cos(j*np.pi/6.)

# update the bathymetry:
topo.depth = custom_depth
```

### *Adding ridges*

Simpler model bathymetry configurations typically include ridges to represent straits and
continents in an idealized manner. The `Topo` class provides `apply_ridge` method
to add ridges to the bathymetry. Example usage:

```python
topo.apply_ridge(height=200, width=8, lon=240, ilat=(10,80) )
```

Example notebook: [3_custom_bathy.ipynb](https://github.com/NCAR/mom6_forge/blob/master/notebooks/3_custom_bathy.ipynb)

### *Bathymetry from a Real Dataset*

For regional or realistic configurations, bathymetry is usually derived from
an observational dataset (e.g., GEBCO) rather than an idealized shape. The
`Topo.set_from_dataset` method is a high-level, opinionated workflow that sets
both the depth and the land/ocean mask from a source dataset in a single
call, automatically choosing a masking and interpolation strategy based on how
the source dataset's resolution compares to the model grid's:

```python
topo.set_from_dataset(
    bathymetry_path="GEBCO_2023.nc",
    longitude_coordinate_name="lon",
    latitude_coordinate_name="lat",
    vertical_coordinate_name="elevation",
    fill_channels=True,
)
```

By default, `set_from_dataset` diagnoses whether the source dataset is much
finer-resolution than the model grid and picks accordingly:

* If the grid is coarse relative to the source (each cell spans many source
  pixels — the model resolution is 12x or more coarser than the dataset's),
  it derives an ocean-fraction mask from sub-sampling statistics and fills
  depth via Cressman distance-weighted interpolation (mirroring the `tx2_3`
  high-resolution topography workflow).
* Otherwise (the grid and source are closer in resolution, or the grid is
  finer than the source), it uses a Natural Earth land mask and a direct
  xESMF regrid of depth.

See the {doc}`bathymetry_workflow` guide for the full decision flow (including
how to override each choice independently) and the {doc}`mapping` guide and
`8_cressman_interpolation.ipynb` notebook for the Cressman math.

Both the masking method (`mask_method`: `'naturalearth'`, `'ocean_frac'`,
`'dataset'`, or `'manual'`) and the depth method (`depth_method`: `'stats'`,
`'cressman'`, or `'xesmf'`) can be overridden explicitly instead of relying on
the automatic diagnosis. This is still an opinionated, multi-step workflow —
inspect the resulting mask and depth afterward, and adjust manually (e.g., via
`TopoEditor`) as needed. For the reasoning behind this workflow and how it
relates to NCAR's global bathymetry pipeline, see {doc}`bathymetry_workflow`.

### *Channel Width Constraints*

Some straits and channels are too narrow to be resolved at a given grid
resolution, but still need to permit flow in MOM6. `ChannelWidth` and
`ChannelWidthList` (in `mom6_forge.channel_width`) record effective
channel-width overrides that are applied on top of the bathymetry at runtime,
rather than by editing the depth/mask fields themselves:

```python
from mom6_forge.channel_width import ChannelWidth, ChannelWidthList

channel_widths = ChannelWidthList()
channel_widths.add(
    ChannelWidth(
        component="U_width",
        lon1=-6.50, lon2=-4.75,
        lat1=35.60, lat2=36.30,
        width=12000.0,
        place="Strait of Gibraltar",
    )
)
```

A `ChannelWidthList` can be passed directly to the `Topo` constructor
(`Topo(grid, min_depth=10.0, channel_widths=channel_widths)`) or loaded from
an existing ASCII file (`ChannelWidthList(filepath="channel_widths.txt")`).
Unlike depth/mask edits, channel width constraints are **not** tracked by the
`Topo` version-control history (see {doc}`widgets`) — they must be written out
separately:

```python
topo.channel_widths.write("channel_widths.txt")
```

## Step 5: Write Model Input Files

The final step of `mom6_forge` workflow is to write out the netcdf files containing grid
and bathymetry data. These files are to be read in by CESM and MOM6 during runtime.

### *Supergrid File*

The `write_supergrid` method of a `Grid` instance writes out the MOM6 supergrid file
in netcdf format. The `GRID_FILE` parameter in `MOM_input` file can then be set to
the path of the supergrid file written by the `Grid` instance.

```python
grid.write_supergrid("my_ocean_hgrid.nc")
```

The supergrid file is the only input file that is written by the `Grid` class. All other
input files require either topography (depth) or mask information. Hence, they are to be
written by the `Topo` class.

### *Topography (Bathymetry) File*

The `write_topo` method of the `Topo` class writes out the MOM6 bathymetry file in netcdf format.
`TOPO_FILE` parameter in `MOM_input` file can then be set to the path of the topography file
written by the `Topo` instance.

```python
topo.write_topo("my_ocean_topog.nc")
```

### *CICE grid file*

If the model is to be run with the CICE component, the `write_cice_grid` method of the 
`Topo` class writes out the CICE grid file in netcdf format. The relevant CICE namelist
parameters can then be updated to read in the CICE grid file written by the `Topo` instance.

```python
topo.write_cice_grid("my_cice_grid.nc")
```

### *ESMF Mesh file*

In addition to the MOM6 supergrid file, MOM6 topography file and CICE grid file, an
ESMF mesh file is required when running CESM. The ESMF mesh file is used
by the NUOPC coupler to acquire grid and mask information. The `write_esmf_mesh` method
of the `Topo` class writes out the ESMF mesh file in netcdf format.

```python
topo.write_esmf_mesh("my_esmf_mesh.nc")
```

### *WW3 (WaveWatch III) Input Files*

If the configuration includes a WaveWatch III wave component, the
`write_ww3_input` method of the `Topo` class writes the text-based WW3 grid
input files (`ww3_grid.inp`, and the `<grid_alias>_x.inp`, `<grid_alias>_y.inp`,
`<grid_alias>_mapsta.inp`, `<grid_alias>_bottom.inp` files) that WW3's
`mod_def` creator reads before runtime.

```python
topo.write_ww3_input("ww3_input/", grid_alias="my_grid")
```

### *SCRIP Grid File*

Modern CESM configurations use ESMF mesh files rather than SCRIP files for
most purposes, but a SCRIP file is still needed to generate custom
ocean-runoff mapping files (see {doc}`mapping`). The `write_scrip_grid` method
writes it out:

```python
topo.write_scrip_grid("my_ocean_scrip.nc")
```

### *Chlorophyll (Shortwave Penetration) Data*

MOM6's shortwave radiation penetration scheme can be driven by a chlorophyll
climatology. `mom6_forge.chl` provides helpers for preparing this input on a
model grid: `interpolate_and_fill_seawifs` interpolates and gap-fills a
SeaWiFS chlorophyll dataset onto the model's tracer grid, and
`gen_chl_empty_dataset` generates an empty placeholder climatology file with
the correct structure.

```python
from mom6_forge.chl import interpolate_and_fill_seawifs

interpolate_and_fill_seawifs(grid, topo, "processed_seawifs.nc", output_path="chl_a.nc")
```

## Step 6: Editing Grids and Bathymetry

Beyond creating standard grids and simple topographies, mom6_forge provides advanced tools
for interactively editing and creating complex model domains. These features are designed to
facilitate reproducible workflows for custom model configurations and model tuning. These
domain configurators can be used for tasks such as:

* Editing Bathymetry: Manually or programmatically modifying ocean depths.
* Creating New Grids: Defining entirely new horizontal grid structures.
* Creating Vertical Grids: Specifying the vertical layering of the ocean model.

Check out the notebook for examples of these advanced features: 
[7_demo_editors.ipynb](https://github.com/NCAR/mom6_forge/blob/master/notebooks/7_demo_editors.ipynb)

## Further steps

The remaining steps of configuring the model, which include specifying initial conditions,
forcings, and runtime parameters, are beyond the scope of the `mom6_forge` tool. Note that a 
complementary tool called `visualCaseGen`, which includes `mom6_forge` as a submodule, can be used
to generate a complete model configuration. `visualCaseGen` provides a graphical user interface
to set up the model grid, bathymetry, initial conditions, forcing, and runtime parameters for MOM6
and other CESM components. Hence, new users are encouraged to use `visualCaseGen` for a complete
model configuration. See: [visualCaseGen](https://github.com/ESMCI/visualCaseGen)
