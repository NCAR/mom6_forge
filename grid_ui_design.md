# GridCreator UI Redesign — grid_ui

## Direction

Rewrite `GridCreator` (in `mom6_bathy/grid_creator.py`) to be a **standalone creation-only widget**:
- No initial `Grid` passed in
- No loading existing grids
- Just creation: click on a map, define a grid, save it

---

## What to remove from the current implementation

| Thing | Why |
|---|---|
| `grid` constructor parameter | Widget creates its own grid |
| `_initial_params` | Nothing to reset to |
| All sliders (`xstart`, `ystart`, `lenx`, `leny`, `resolution`) | Were for editing, not creating |
| `load_grid`, `_load_button` | Load is out of scope |
| `refresh_commit_dropdown`, `_commit_dropdown`, `_commit_details` | Tied to load workflow |
| `sync_sliders_to_grid` | Sliders are gone |
| `reset_grid` | No initial grid to reset to |
| `_on_slider_change` | Sliders are gone |

---

## Bugs in the current code (worth fixing regardless)

1. `sync_sliders_to_grid` removes observers but **never re-attaches them** — method ends mid-way
2. `reset_grid` calls `params["resolution"]` but `_initial_params` never stores `"resolution"` → `KeyError`
3. `_on_slider_change` hardcodes `Grid(...)` with no `type` → always makes a `uniform_spherical` grid, clobbering any projected grid on first slider touch

---

## Proposed new UX flow

### Controls panel
- **Mode selector** (radio or dropdown): `Lat-Lon corners` | `From center` | `From projection (CRS)`
- **Resolution input**: degrees for lat-lon mode, km for projected modes
- Mode-specific extras:
  - *From center*: angle (deg), width (km), height (km) — filled in after click
  - *From projection*: CRS picker (EPSG:3995, EPSG:3031, or custom string)
- **Status text**: tells user what to click next ("Click corner 1 of 2", "Click domain centre", etc.)
- **Name field** + **Save button**
- **Clear button**: restart click placement

### Map panel
- `ccrs.PlateCarree()` by default; should switch to native projection for polar CRS (e.g. `ccrs.NorthPolarStereo()` for EPSG:3995) to avoid distortion
- Click handler via `mpl_connect('button_press_event', ...)`
  - Lat-lon corners: collect 2 clicks → `Grid(lenx=..., leny=..., xstart=..., ystart=...)`
  - From center: collect 1 click → center lat/lon locked in, width/height/angle from widgets → `Grid.from_center(...)`
  - From projection: collect 2 clicks → transform (lon,lat) → projected (x,y) via `pyproj.Transformer` → `Grid.from_projection(crs, x_min, x_max, y_min, y_max, resolution_m)`
- Grid preview drawn once enough clicks are collected

### Key coordinate transform (for projection mode)
```python
from pyproj import Transformer
t = Transformer.from_crs("EPSG:4326", selected_crs, always_xy=True)
x, y = t.transform(clicked_lon, clicked_lat)
```

---

## Open questions (to answer before implementing)

1. **Save behaviour** — keep save-to-`GridLibrary` NetCDF, or just expose the `Grid` object?
2. **Which modes** — all three, or just the projected ones (branch focus)?
3. **After saving** — clear and start fresh, or keep grid displayed?
4. **Class name** — still `GridCreator`, or rename?

---

## Files involved

| File | Role |
|---|---|
| `mom6_bathy/grid_creator.py` | Widget to rewrite |
| `mom6_bathy/grid.py` | `Grid.from_center`, `Grid.from_projection` — constructors to call |
| `mom6_bathy/_supergrid.py` | `ProjectedSupergrid` — underlying grid math |
