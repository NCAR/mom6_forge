# UI  Widgets

This document explains the widget modules in `mom6_forge`.

`mom6_forge` comes with three UI modules/classes that wrap the three main
classes—`VGrid`, `Grid`, and `Topo`—to help with creating vertical grids
(`VGridCreator`), horizontal grids (`GridCreator`), and editing topography
(`TopoEditor`).

## Creators

The creators act as visual wrappers around the constructors of their respective
classes, providing sliders and visualizations. They automatically generate
folders called `VGridLibrary` and `GridLibrary` to store created grids.
The currently selected grid is directly accessible as an object inside each
creator.

### GridCreator

`GridCreator` is a split-panel widget: a control panel on the left and an
interactive cartopy map on the right.  It supports three creation modes,
selected via radio buttons before any grid is defined.

**Lat/Lon Corners**

Drag a rectangle on the map.  A uniform-degree `Grid` is created
from the bounding box.  After creation, five degree sliders appear
(`xstart`, `ystart`, `lenx`, `leny`, `resolution`) for live
adjustment.

**From Center**

Set the domain width (km), height (km), resolution (km), and a clockwise
rotation angle (degrees from north) in the input fields, then click the
domain centre on the map.  This calls `Grid.from_center()`, which builds a
rotated rectangular grid using an azimuthal equidistant projection centred at
the clicked point.  This mode is particularly useful for aligning a domain
with a coastline or estuary.

**From Projection**

Set a CRS (chosen from a preset dropdown or typed as any EPSG string) and
resolution (km), then click two corners on the map.  This calls
`Grid.from_projection()`.

When a preset CRS is selected the map axes switches to the matching native
cartopy projection so that clicks deliver coordinates in that projection's
metres directly — no additional transformation is needed, and domains near or
over the poles work correctly.  Supported presets:

```{list-table}
:header-rows: 1
:widths: 30 40 30

* - Preset
  - Cartopy projection
  - Default view
* - EPSG:4326 (Plate Carrée)
  - PlateCarree (no switch)
  - Global
* - EPSG:3995 (Arctic Polar Stereographic)
  - NorthPolarStereo
  - 45–90°N
* - EPSG:3031 (Antarctic Polar Stereographic)
  - SouthPolarStereo
  - 90–45°S
* - EPSG:5070 (CONUS Albers Equal Area)
  - AlbersEqualArea
  - CONUS
* - Any other EPSG string
  - PlateCarree (fallback)
  - CRS area-of-use (via pyproj)
```

**Editing projected grids**

After a projected grid (From Center or From Projection) is created or loaded,
the control panel shows the same parameter inputs pre-filled with the current
values alongside a **Recreate Grid** button.  Adjust the inputs and click
Recreate to rebuild the grid without needing to re-click the map.

**GridLibrary**

Grids are saved as MOM6 supergrid NetCDF files under `<repo_root>/GridLibrary/`
(naming convention: `grid_<name>.nc`).  The directory is created automatically
and a blanket `.gitignore` is written into it so saved grids are not
accidentally committed.

On **Load**, all creation parameters are restored from the file's metadata, so
the **Recreate** button works immediately after loading a projected grid without
any additional clicks.

### VGridCreator

`VGridCreator` is a control-panel-and-plot widget for building a `VGrid`
interactively: a live plot on the right shows each layer's center depth as a
horizontal line, updating as the controls on the left change.

A **Type** toggle switches between `Uniform` and `Hyperbolic` vertical grids
(see {doc}`quickstart`). **Levels** and **Depth (m)** sliders set `nk` and
`depth`; a **Top/Bottom Ratio** slider (enabled only in `Hyperbolic` mode)
sets the ratio of bottom to top layer thickness. If a `Topo` instance is
passed to the constructor (`VGridCreator(topo=topo)`), a warning is shown
whenever the chosen depth is shallower than that topo's `max_depth`, since the
vertical grid needs to extend at least to the deepest point of the bathymetry.

**VGridLibrary**

Like `GridCreator`, `VGridCreator` has a **Library** section for saving and
loading named vertical grids. Enter a name and message and click **Save
VGrid** to write `VgridLibrary/vgrid_<name>.nc`; the dropdown lists
previously saved vgrids (most recent first) with their name, depth, and level
count for **Load**. The `Reset` button reverts the current session's edits
back to the vgrid the widget was constructed with.

### GridSketcher

`GridSketcher` builds an orthogonal (conformal) regional `Grid` from a sketched
outline and four chosen corners, so the domain boundary can follow an irregular
coastline instead of straight lon/lat or projected edges.

The same solve is available without the widget as `Grid.from_outline`. Give it an
outline (at least 4 vertices, in degrees) and 4 corner indices (`[0, 1, 2, 3]` by
default for a 4-vertex outline), and either `resolution_km` (nominal cell size) or
`n_cells` (target `nx * ny`). `projection` is a kind name (`"lcc"`, `"merc"`,
`"tmerc"`, `"stere"`), a `MapProjection`, or `"auto"` (the default), which picks the
kind whose cells vary least in true size in a quick coarse solve of each
(`corner_diagnostics.size_ratios`): Mercator for a lon/lat box like this one, and
polar stereographic for an outline round a pole:

```python
from mom6_forge.grid import Grid
from mom6_forge.grid_sketcher import MapProjection

lon, lat = [235.0, 243.0, 243.0, 235.0], [31.0, 31.0, 39.0, 39.0]
grid = Grid.from_outline(lon, lat, resolution_km=10)
lcc = MapProjection("lcc", lon_0=-121, lat_0=35, lat_1=32, lat_2=38)
grid = Grid.from_outline(lon, lat, resolution_km=10, projection=lcc)
```

A longer outline needs its 4 corners, e.g.
`Grid.from_outline(sketch.outline.lon, sketch.outline.lat, sketch.outline.corners)`.

The grid carries a `.outline` attribute (the outline, corners, projection and
resolution it was built from), which `write_supergrid` stores in the file and
`Grid.from_supergrid` reads back. Hand it to `GridSketcher` to keep editing:

```python
%matplotlib ipympl
from mom6_forge.grid_sketcher import GridSketcher
sketch = GridSketcher(grid)   # reopens grid.outline exactly
sketch
```

`GridSketcher(resolution_km=5)` starts from a default box, `GridSketcher(blank=True)`
from an empty map, and `GridSketcher(grid, per_side=4)` traces the boundary of a
grid that has no outline (4 vertices per side).

The map, a globe facing the outline, sits on the left, with these mouse
instructions in a box under it, and the control panel on the right. On the map:

- **Drag** a vertex to move it (press anywhere on its dot); light grid lines follow
  the mouse and the grid is solved again on release.
- **Click an edge** to add a vertex there (keep dragging to place it).
- **Right-click** (or **Ctrl-click**) a vertex to delete it.
- **Double-click** a vertex to make or unmake it a corner (numbered 1-4).
- **Drag** the centre square to move the whole outline over the sphere, or its knob
  to rotate it; with **Box** on, **drag** on the map for a box along meridians and
  parallels.
- **Scroll** inside the framed map to zoom about the cursor, out to the whole
  hemisphere facing you; **drag** the empty map (or middle-drag anywhere) to pan:
  after a long pan the globe turns to face the middle of the view (near a pole,
  the pole), so pans reach any place on Earth. **Reset view** under the map goes
  back to the whole outline.
- Grid lines turn orange or red where ocean cells are badly shaped (orthogonality,
  size jump or aspect ratio); land cells are not judged. Rest the cursor on a grid
  cell and a box there says why, or that the cell is over land (no box appears off
  the grid).
- Each side is drawn thick where the coastline says it is open water and thin where
  it is land. An **X** marks an open stretch shorter than 4 cells (or under about
  3% of the side), which the warnings then list: move that side so it is all land
  or all sea there.

In the panel:

- **Resolution (km)** sets the nominal cell size and shows the `nx x ny` it gives.
- **Projection** picks the conformal projection the grid is solved in, one button
  each: Lambert conformal, Mercator, transverse Mercator or polar stereographic. The
  pressed button is the one in use, and the summary line names it. The green one is
  recommended (as `"auto"` above), since its cells vary least in size. A new sketch,
  and each newly drawn outline, is solved on the green one until you press another;
  your choice is kept while you edit, unless it can't take the outline (Lambert
  round a pole, say). Either way the projection is centred on the outline after each
  edit. The map stays as it is.
- **Shade cell size** toggles a log-scale shading of cell size.
- **Undo** / **Redo** step through outline edits. **Clear all** empties the map:
  click points (drag one to move it, right-click one to delete it, drag the map to
  pan), then click point 1 again to close the outline (4 points become the 4
  corners).
- **Build final grid** solves at full resolution and sets `sketch.grid`, a `Grid`
  with `.outline`, ready for `Topo` (and CrocoDash's `Case`). **Save** writes it to
  `GridLibrary/grid_<name>.nc`; reopen it with
  `GridSketcher(Grid.from_supergrid(path))`. Any later edit, resolution or projection
  change drops `sketch.grid` and greys out **Save** until the next **Build**.
- **Status**, under **Save**, logs each message with its time, newest on top;
  older ones are grey, and it scrolls back through the last 100.
- `sketch.open_boundaries` lists the sides with open water in the last preview
  (e.g. `["south", "west"]`), for CrocoDash's
  `case.configure_forcings(boundaries=...)`.
- **Details** shows the diagnostics table and a "smallest ocean cell ... MOM6 CFL"
  line. The preview has no bathymetry yet, so it counts only cells whose centre is
  off Natural Earth land (or all cells, if that data can't be loaded). **Edit
  coordinates** shows the vertices as text ("lon, lat" per line, `*` for a corner).
- Under them, **Errors** say why Build is greyed out: fewer than 4 corners, crossing
  edges, a failed solve, folded ocean cells or more than 2.4 million cells (over
  4 GB to build: use `Grid.from_outline` in a job with more memory). **Warnings**
  list what to check, such as abrupt size changes, non-orthogonal or stretched
  cells, nearly flat corners and small open boundaries along the coast; any warning
  turns Build amber. Each is one line, counts ocean cells only and says where the
  worst cell is (e.g. "near corner 3").

Once bathymetry exists, get the same numbers over the true wet cells:

```python
from mom6_forge import corner_diagnostics as diag

limits = diag.timestep_limits(grid, wet=topo.tmask)
print(diag.warnings(limits))
```

#### Acknowledgements

GridSketcher and `Grid.from_outline` build on earlier ocean-grid tools:

- [gridgen](https://github.com/sakov/gridgen-c) (Pavel Sakov) and
  [pygridgen](https://github.com/pygridgen/pygridgen) (Rob Hetland, Paul Hobson and
  contributors): orthogonal grids from a polygon with marked corners, mapped conformally
  onto a rectangle. They use the Schwarz-Christoffel CRDT method of Driscoll and Vavasis
  (1998); mom6_forge solves Laplace problems with finite elements instead.
- [pyroms](https://github.com/ESMG/pyroms) (Frederic Castruccio, Kate Hedstrom and
  contributors): its interactive boundary editor is the model for sketching an outline
  and marking its corners.
- Coastlines and land come from [Natural Earth](https://www.naturalearthdata.com)
  (public domain), through cartopy.

The code was written with AI assistance (Claude Code, Anthropic) under the author's
direction; the commits say so in their `Co-Authored-By` lines.

## Topo & TopoEditor Edits

The `Topo` & `TopoEditor` workflow is a bit more nuanced.

Grids are simple: they are created once and rarely modified. `Topo` objects,
however, are built on top of the horizontal grid and are heavily edited during
the development cycle (filling bays, deepening ridges, etc.).

Because of this, `Topo` has many editing functions. `TopoEditor` provides a
visual, point-and-click interface on top of these functions.

Current editing functions (`*` = available in `TopoEditor`):

1. `*` Edit depth at a specific point
2. `*` Edit the minimum depth
3. `*` Erase a basin at a selected point
4. `*` Erase every basin except the one containing the selected point
5. Generate and apply an ocean mask from a land-fraction dataset
6. Apply a ridge to the bathymetry

`TopoEditor(topo, open_boundaries=True)` also shows which domain edges stay open
to the sea under the current mask: open stretches in magenta (closed ones thin
grey), a red X on any tiny stretch that hugs the coast, and a summary line in
the panel. It updates after every edit; `editor.open_boundaries` lists the open
sides, ready for CrocoDash's `case.configure_forcings(boundaries=...)`.

You can also reapply an initializer (none supported in the TopoEditor):

1. Set flat bathy
2. Set spoon bathy
3. Set bowl bathy
4. Set from dataset
5. Set from previous topo object

## Undo & Redo

To support the iterative editing process, we provide **undo and redo**
functionality across sessions. This requires maintaining a structured history,
stored inside a directory associated with your `Topo` object.

This folder contains:

1. The grid underlying the topo
2. The original blank topo
3. A permanent command history

```{figure} images/TopoLibrarySample.png
:align: center
:width: 500px

Layout of a `Topo` folder.
```

```{figure} images/TopoTempCommandHistory.png
:align: center
:width: 500px

Structure of the command history JSON file.
```

## How It Works

1. You create your `Topo` object, which initializes the four files above.
2. You make changes using `Topo` or `TopoEditor`.
3. Each change is added to history.
4. Each change is then **committed via Git**—this powers undo/redo.

## Topo Git Functionality and How to Use It

We use Git to implement undo/redo functionality inside the topo directory.

You can view the history with:

`git log`

The command history is a JSON file mapping commit SHAs to change
metadata. You can cross-reference the Git log with this JSON file to inspect
details of each edit.

We also support simple version-control actions:

- `topo.tcm.create_branch("branchname")` — create a branch  
- `topo.tcm.checkout("branchname")` — switch branches  
- `topo.tcm.tag("tagname")` — create a tag and save a `tagname_topog.nc` file  

Tags cannot be checked out (this can cause unexpected states).

Undo and redo:

- `topo.tcm.undo()`
- `topo.tcm.redo()`

These appear in the Git log as:

- `UNDO-<sha>`
- `REDO-<sha>`

Be careful not to undo your initial set-function!

```{warning}
Do **not** run Git commands inside this folder except for harmless commands like `git log`.
We manage the folder internally. External Git commands (like `git checkout`)
may break state management.
```

```{figure} images/TopoSampleGitLog.png
:align: center
:width: 500px

Example `git log` for a topo editing session.
```

## Nuances (Initialization, Naming, etc.)

Folders are named after the hash of the grid's `tlon` variable.  
This means that **any topo using the same grid** will share the same folder.

You can initialize `Topo` in three ways:

1. `Topo()` — creates an empty topo with the provided minimum depth
2. `Topo.from_version_control(path)` — loads a folder, applies saved history, and returns the reconstructed topo
3. `Topo.from_topo_file(file)` — loads a topo file and applies it on top of any existing changes in the folder

All of the git versioning (undo, redo, reset, checkout, tag, create branch) is handled by the TopoCommandManager, which can be accessed as the class variable "tcm". The code for this manager resides in the command_manager class. 
We would use straight git functions, but the topo file itself is handled independently of the commits. (In the future, expect this to use icechunk)


See also the demonstration notebook:

[7_demo_editors.ipynb](https://github.com/NCAR/mom6_forge/blob/master/notebooks/7_demo_editors.ipynb)
