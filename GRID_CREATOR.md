# GridCreator — how it works

`GridCreator` is an ipywidgets/ipympl Jupyter widget for creating and saving MOM6
horizontal grids interactively.  It lives in `mom6_bathy/grid_creator.py`.

---

## Layout

```
┌────────────────────────┬──────────────────────────────┐
│  Left: control panel   │  Right: cartopy map canvas   │
│  ─ Grid Creator        │                              │
│  ─ Library             │                              │
└────────────────────────┴──────────────────────────────┘
```

The control panel has two sections stacked vertically:
- **Grid Creator** — changes depending on state (see below)
- **Library** — always visible; name/message fields, dropdown of saved grids,
  Save and Load buttons

---

## Creation modes

Selected via RadioButtons before a grid exists.  Each mode ends with a `Grid`
object being created and `_switch_to_grid_mode()` being called.

| Mode | Clicks | Grid constructor | Map projection |
|---|---|---|---|
| **Lat/Lon Corners** | 2 diagonal corners | `Grid(lenx, leny, xstart, ystart, resolution)` | PlateCarree (stays) |
| **From Center** | 1 centre point | `Grid.from_center(lat, lon, width_m, height_m, resolution_m, angle_deg)` | PlateCarree (stays) |
| **From Projection** | 2 corners | `Grid.from_projection(crs, x_min, x_max, y_min, y_max, resolution_m)` | Switches to native CRS projection |

### Lat/Lon Corners
Click two opposite corners.  Resolution defaults to `max(lenx, leny) / 20`
(~20 cells across the larger dimension).  After creation the control panel
shows degree sliders for live editing.

### From Center
Set width (km), height (km), resolution (km), and rotation angle (degrees CW
from north) in the inputs, then click the domain centre.  Useful for grids
that need to be aligned with a coastline feature.  Map stays in PlateCarree
since the domain is small enough to define by clicking a single geographic
point.

### From Projection
Set a CRS (preset dropdown or manual EPSG override) and resolution (km), then
click two corners.

**Map projection switching:** entering this mode calls `_crs_to_cartopy_proj`
which returns a native cartopy projection for known EPSG codes:

| EPSG | Cartopy projection | Default view |
|---|---|---|
| 4326 | PlateCarree (no switch) | global |
| 3995 | NorthPolarStereo | 45–90°N |
| 3031 | SouthPolarStereo | 90–45°S |
| 5070 | AlbersEqualArea | CONUS |
| other | PlateCarree | CRS area-of-use (via pyproj) |

When the map is in a native projection, `event.xdata/ydata` from matplotlib
clicks arrive in that projection's metres — these are passed directly to
`Grid.from_projection` with no further transformation.  When the map is
PlateCarree the click coords are lon/lat and are transformed to projected
metres via `pyproj.Transformer`.

The map is always restored to PlateCarree when leaving projection mode
(switching away, resetting, or loading a grid).

---

## Post-grid panel

After any grid is created or loaded the creator section rebuilds via
`_switch_to_grid_mode()`:

- **Lat/lon grid** → five FloatSliders (xstart, ystart, lenx, leny, resolution).
  Each slider change recreates the `Grid` object and replots.
- **Projected grid (center or projection)** → the same parameter inputs used
  during creation, pre-filled with the current values, plus a **Recreate Grid**
  button that rebuilds the grid from those inputs without new clicks.

**Reset** always returns to pre-grid click mode (PlateCarree, world view).

---

## Library

Grids are written as MOM6 supergrid NetCDF files under `<repo_root>/GridLibrary/`
with the naming convention `grid_<name>.nc`.

On **Save**: `Grid.write_supergrid()` is called, which writes the supergrid
dataset including construction metadata (`grid_type`, CRS WKT, extents, etc.)
as dataset attributes.

On **Load**: the `grid_type` attribute is read to determine mode, and the
construction parameters are restored into the relevant widgets so that
**Recreate** works immediately after loading.

| `grid_type` attr | `_grid_mode` | Restored widgets |
|---|---|---|
| `uniform_spherical` | `latlon` | sliders sync'd via `sync_sliders_to_grid` |
| `projected_center` | `center` | width, height, resolution, angle, `_center_latlon` |
| `projected_crs` | `projection` | CRS text, resolution, `_proj_extents` |

---

## Known limitations / gotchas

- **Observer stacking**: `construct_observances()` is called every time the
  panel rebuilds (`_switch_to_grid_mode`, `reset_grid`).  `on_click`/`observe`
  calls accumulate — handlers run multiple times but are idempotent so this
  doesn't cause visible bugs, just mild inefficiency.
- **Projected grids have no sliders**: there is no live-edit equivalent for
  projected grids.  Change the inputs and hit Recreate.
- **From Center rotation**: the only way to get a rotated rectangular domain.
  It uses an azimuthal equidistant projection centred at the clicked point
  internally — not a standard named EPSG code.
- **Map distortion for polar domains**: even with the native polar projection
  active, the grid preview is drawn with `transform=ccrs.PlateCarree()` (since
  `qlon`/`qlat` are always geographic), which is correct but looks compressed
  near the pole on a stereographic view.
