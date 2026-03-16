import os
import numpy as np
import xarray as xr
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mom6_bathy.grid import Grid
from mom6_bathy._supergrid import ProjectedSupergrid
from pathlib import Path
from pyproj import CRS, Transformer

_CRS_PRESETS = [
    ("Plate Carree / Geographic (EPSG:4326)", "EPSG:4326"),
    ("Arctic Polar Stereographic (EPSG:3995)", "EPSG:3995"),
    ("Antarctic Polar Stereographic (EPSG:3031)", "EPSG:3031"),
    ("CONUS Albers Equal Area (EPSG:5070)", "EPSG:5070"),
]

# Preset EPSG → (cartopy_proj, default_extent_in_PlateCarree)
# extent is [lon_min, lon_max, lat_min, lat_max], or None for set_global()
_EPSG_TO_CARTOPY = {
    3995: (ccrs.NorthPolarStereo(), [-180, 180, 45, 90]),
    3031: (ccrs.SouthPolarStereo(), [-180, 180, -90, -45]),
    5070: (
        ccrs.AlbersEqualArea(
            central_longitude=-96, central_latitude=23, standard_parallels=(29.5, 45.5)
        ),
        [-130, -60, 20, 55],
    ),
}


class GridCreator(widgets.HBox):
    """Interactive Jupyter widget for creating and saving MOM6 horizontal grids.

    The widget is split into two panels:
      - Left: creator controls (mode selector / sliders / recreate inputs) + library section
      - Right: cartopy map (matplotlib canvas embedded via ipympl)

    Creation modes
    --------------
    Lat/Lon Corners   : click two diagonal corners on a PlateCarree map →
                        uniform-degree Grid via Grid(lenx, leny, ...)
    From Center       : set width/height/resolution/angle, click domain centre →
                        rotated projected grid via Grid.from_center(...)
    From Projection   : set a CRS + resolution, click two corners on the native
                        projection map → Grid.from_projection(...)

    After creation the creator section switches from click-mode to an edit panel:
      - Lat/Lon grids  : degree sliders (xstart, ystart, lenx, leny, resolution)
      - Projected grids: the same parameter inputs + a Recreate button

    Library
    -------
    Grids are saved as NetCDF supergrids under <repo_root>/GridLibrary/.
    The dropdown lists all grid_*.nc files there; Load restores the full
    creation parameters so Recreate still works after loading.

    Map projection
    --------------
    Entering "From Projection" mode switches the cartopy axes to the native
    projection for the selected CRS (preset EPSG codes only; unknown codes
    fall back to PlateCarree zoomed to the CRS area-of-use).  All other modes
    use PlateCarree.  Grid lines are always drawn in geographic coordinates
    (transform=PlateCarree) regardless of the active map projection.
    """

    def __init__(self, grid=None, repo_root=None):
        self.grid = grid
        self.repo_root = Path(repo_root if repo_root is not None else os.getcwd())
        self.grids_dir = Path(os.path.join(self.repo_root, "GridLibrary"))
        self.grids_dir.mkdir(exist_ok=True)
        (self.grids_dir / ".gitignore").write_text("*\n")

        self._initial_params = None
        if grid is not None:
            self._initial_params = {
                "lenx": grid.lenx,
                "leny": grid.leny,
                "nx": grid.nx,
                "ny": grid.ny,
                "xstart": grid.supergrid.x[0, 0],
                "ystart": grid.supergrid.y[0, 0],
                "name": grid.name,
            }

        # Click-capture state
        self._click_points = []  # accumulated (x, y) clicks in current map CRS
        self._click_cid = None  # mpl canvas connection id, or None when inactive

        # Redraw guard — prevents recursive xlim_changed → redraw loops
        self._in_redraw = False

        # Grid creation mode and associated stored parameters for Recreate
        self._grid_mode = "latlon"  # "latlon" | "center" | "projection"
        self._center_latlon = None  # (lat, lon) set after a From Center click
        self._proj_extents = None  # (x_min, x_max, y_min, y_max) in projected CRS
        self._current_map_proj = ccrs.PlateCarree()  # active cartopy projection

        self.construct_control_panel()
        self.construct_observances()

        # --- Plot ---
        plt.ioff()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        plt.ion()
        self.fig.canvas.layout.width = "100%"
        self.fig.canvas.layout.min_width = "0"
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = "top"

        self.ax.callbacks.connect("xlim_changed", self._on_extent_changed)

        super().__init__(
            [self._control_panel, self.fig.canvas],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )

        self.refresh_commit_dropdown()
        if self.grid is not None:
            if isinstance(self.grid.supergrid, ProjectedSupergrid):
                self._grid_mode = "projection"
            self.plot_grid()
        else:
            self.plot_world()
            self._start_click_mode()

    # ------------------------------------------------------------------
    # Control panel construction
    # ------------------------------------------------------------------

    def construct_control_panel(self):
        # --- Mode selector (pre-grid only) ---
        self._mode_selector = widgets.RadioButtons(
            options=["Lat/Lon Corners", "From Center", "From Projection"],
            value="Lat/Lon Corners",
            layout={"width": "90%"},
        )

        # --- Status text ---
        self._status_html = widgets.HTML(
            value="<p>Zoom/pan to your region, then activate point selection.</p>"
        )

        # --- Select button ---
        self._select_button = widgets.ToggleButton(
            value=False,
            description="Select Region",
            button_style="info",
            icon="crosshairs",
            layout={"width": "90%"},
        )

        # --- Lat/Lon mode panel ---
        self._latlon_grid_type = widgets.ToggleButtons(
            options=[
                ("Uniform Spherical", "uniform_spherical"),
                ("Rectilinear Cartesian", "rectilinear_cartesian"),
            ],
            value="uniform_spherical",
            style={"button_width": "auto"},
            layout={"width": "90%"},
        )
        self._latlon_panel = widgets.VBox(
            [
                widgets.HTML(
                    "<p><b>1st click:</b> one corner &nbsp;"
                    "<b>2nd click:</b> opposite corner</p>"
                ),
                self._latlon_grid_type,
            ]
        )

        # --- From Center mode panel ---
        _fw = {"width": "90%"}
        _ds = {"description_width": "initial"}
        self._center_width = widgets.FloatText(
            value=500, description="Width (km)", style=_ds, layout=_fw
        )
        self._center_height = widgets.FloatText(
            value=500, description="Height (km)", style=_ds, layout=_fw
        )
        self._center_resolution = widgets.FloatText(
            value=25, description="Res (km)", style=_ds, layout=_fw
        )
        self._center_angle = widgets.FloatText(
            value=0, description="Angle (deg)", style=_ds, layout=_fw
        )
        self._center_panel = widgets.VBox(
            [
                self._center_width,
                self._center_height,
                self._center_resolution,
                self._center_angle,
            ],
            layout={"display": "none"},
        )

        # --- From Projection mode panel ---
        self._proj_crs_dropdown = widgets.Dropdown(
            options=_CRS_PRESETS,
            description="CRS:",
            layout=_fw,
        )
        self._proj_crs_text = widgets.Text(
            value="EPSG:4326",
            description="Override:",
            placeholder="e.g. EPSG:32617",
            style=_ds,
            layout=_fw,
        )
        self._proj_resolution = widgets.FloatText(
            value=25, description="Res (km)", style=_ds, layout=_fw
        )
        self._proj_panel = widgets.VBox(
            [self._proj_crs_dropdown, self._proj_crs_text, self._proj_resolution],
            layout={"display": "none"},
        )

        # --- Post-grid action buttons ---
        self._recreate_button = widgets.Button(
            description="Recreate Grid",
            layout={"width": "48%"},
        )
        self._reset_button = widgets.Button(
            description="Reset",
            button_style="danger",
            layout={"width": "48%"},
        )

        # --- Library ---
        self._snapshot_name = widgets.Text(
            value="",
            placeholder="Enter grid name",
            description="Name:",
            layout={"width": "90%"},
        )
        self._commit_msg = widgets.Text(
            value="",
            placeholder="Enter grid message",
            description="Message:",
            layout={"width": "90%"},
        )
        self._commit_dropdown = widgets.Dropdown(
            options=[], description="Grids:", layout={"width": "90%"}
        )
        self._commit_details = widgets.HTML(
            value="", layout={"width": "90%", "min_height": "2em"}
        )
        self._save_button = widgets.Button(
            description="Save Grid", layout={"width": "44%"}
        )
        self._load_button = widgets.Button(
            description="Load Grid", layout={"width": "44%"}
        )

        creator_controls = self._build_creator_controls()

        library_section = widgets.VBox(
            [
                widgets.HTML("<h3>Library</h3>"),
                self._snapshot_name,
                self._commit_msg,
                self._commit_dropdown,
                self._commit_details,
                widgets.HBox([self._save_button, self._load_button]),
            ]
        )

        self._control_panel = widgets.VBox(
            [creator_controls, library_section],
            layout={"width": "45%", "height": "100%"},
        )

    def _build_creator_controls(self):
        """Return the top section of the left panel.

        Three possible states:
          1. No grid yet  → mode selector + mode-specific instruction panels + select button
          2. Lat/lon grid → degree sliders + reset button
          3. Projected grid (center or projection) → parameter inputs + recreate/reset buttons
        """
        layout = widgets.Layout(
            width="100%",
            min_width="200px",
            max_width="400px",
            align_items="stretch",
            overflow_y="auto",
        )

        if self.grid is None:
            return widgets.VBox(
                [
                    widgets.HTML("<h3>Grid Creator</h3>"),
                    self._mode_selector,
                    self._latlon_panel,
                    self._center_panel,
                    self._proj_panel,
                    self._status_html,
                    self._select_button,
                ],
                layout=layout,
            )

        if self._grid_mode == "latlon":
            # Build sliders from the current grid state
            initial_xstart = float(self.grid.supergrid.x[0, 0]) % 360
            slider_window = 30
            slider_min = max(initial_xstart - slider_window, -180.0)
            slider_max = min(initial_xstart + slider_window, 360.0)
            if slider_min >= slider_max:
                slider_min = max(-180.0, initial_xstart - 15)
                slider_max = min(360.0, initial_xstart + 15)

            self._xstart_slider = widgets.FloatSlider(
                value=initial_xstart,
                min=slider_min,
                max=slider_max,
                step=0.01,
                description="xstart",
            )
            self._lenx_slider = widgets.FloatSlider(
                value=self.grid.lenx, min=0.01, max=50.0, step=0.01, description="lenx"
            )
            initial_ystart = float(self.grid.supergrid.y[0, 0])
            self._ystart_slider = widgets.FloatSlider(
                value=initial_ystart,
                min=max(initial_ystart - 30, -90),
                max=min(initial_ystart + 30, 90),
                step=0.01,
                description="ystart",
            )
            self._leny_slider = widgets.FloatSlider(
                value=self.grid.leny, min=0.01, max=50.0, step=0.01, description="leny"
            )
            self._resolution_slider = widgets.FloatSlider(
                value=self.grid.lenx / self.grid.nx,
                min=0.01,
                max=1.0,
                step=0.01,
                description="Resolution",
            )

            return widgets.VBox(
                [
                    widgets.HTML("<h3>Grid Creator</h3>"),
                    self._resolution_slider,
                    self._xstart_slider,
                    self._lenx_slider,
                    self._ystart_slider,
                    self._leny_slider,
                    widgets.HBox(
                        [self._reset_button],
                        layout=widgets.Layout(justify_content="flex-end", width="100%"),
                    ),
                ],
                layout=layout,
            )

        # Projected grid (center or projection mode)
        if self._grid_mode == "center":
            center_info = ""
            if self._center_latlon is not None:
                lat, lon = self._center_latlon
                center_info = f"<p><b>Centre:</b> {lat:.3f}°N, {lon:.3f}°E</p>"
            header = widgets.HTML(f"<h3>Grid Creator</h3>{center_info}")
            mode_inputs = widgets.VBox(
                [
                    self._center_width,
                    self._center_height,
                    self._center_resolution,
                    self._center_angle,
                ]
            )
            self._recreate_button.disabled = self._center_latlon is None
        else:
            header = widgets.HTML("<h3>Grid Creator</h3>")
            mode_inputs = widgets.VBox(
                [
                    self._proj_crs_dropdown,
                    self._proj_crs_text,
                    self._proj_resolution,
                ]
            )
            self._recreate_button.disabled = self._proj_extents is None

        return widgets.VBox(
            [
                header,
                mode_inputs,
                widgets.HBox(
                    [self._recreate_button, self._reset_button],
                    layout=widgets.Layout(width="100%"),
                ),
            ],
            layout=layout,
        )

    def _switch_to_grid_mode(self):
        """Replace the creator controls panel after a grid is created or loaded."""
        creator_controls = self._build_creator_controls()
        library_section = self._control_panel.children[1]
        self._control_panel.children = [creator_controls, library_section]
        self.construct_observances()

    def _update_status_for_mode(self, mode):
        if mode == "Lat/Lon Corners":
            self._status_html.value = (
                "<p>Zoom/pan to your region, then activate point selection.</p>"
            )
        elif mode == "From Center":
            self._status_html.value = (
                "<p>Set dimensions, then click to place the domain centre.</p>"
            )
        else:
            self._status_html.value = (
                "<p>Set CRS and resolution, then click two corners.</p>"
            )

    def _crs_to_cartopy_proj(self, crs_str):
        """Return (cartopy_proj, extent) for a CRS string.

        extent is [lon_min, lon_max, lat_min, lat_max] in geographic coords,
        or None to call set_global().  Falls back to PlateCarree for unknown CRS.
        """
        try:
            epsg_code = int(crs_str.upper().replace("EPSG:", "").strip())
            if epsg_code in _EPSG_TO_CARTOPY:
                return _EPSG_TO_CARTOPY[epsg_code]
        except (ValueError, AttributeError):
            pass
        # Unknown CRS — stay on PlateCarree but zoom to the CRS area of use
        extent = None
        try:
            aou = CRS.from_user_input(crs_str).area_of_use
            if aou:
                extent = [aou.west, aou.east, aou.south, aou.north]
        except Exception:
            pass
        return ccrs.PlateCarree(), extent

    def _set_map_projection(self, proj, extent=None):
        """Recreate the map axes with a new cartopy projection."""
        self._in_redraw = True
        try:
            self.fig.clear()
            self.ax = self.fig.add_subplot(1, 1, 1, projection=proj)
            self.ax.callbacks.connect("xlim_changed", self._on_extent_changed)
            self._current_map_proj = proj
            self._draw_map_content()
            if extent:
                self.ax.set_extent(extent, crs=ccrs.PlateCarree())
            else:
                self.ax.set_global()
            self.fig.canvas.draw_idle()
        finally:
            self._in_redraw = False

    def construct_observances(self):
        # NOTE: on_click / observe calls accumulate across repeated invocations
        # (this method is called every time _switch_to_grid_mode rebuilds the panel).
        # Handlers are idempotent in practice, but it is a known minor inefficiency.
        self._save_button.on_click(self.save_grid)
        self._load_button.on_click(self.load_grid)
        self._reset_button.on_click(self.reset_grid)
        self._recreate_button.on_click(self._on_recreate_click)
        self._snapshot_name.observe(
            lambda change: self.refresh_commit_dropdown(), names="value"
        )
        self._commit_dropdown.observe(self.update_commit_details, names="value")

        if self.grid is None:
            self._mode_selector.observe(self._on_mode_change, names="value")
            self._proj_crs_dropdown.observe(
                self._on_proj_crs_preset_change, names="value"
            )
            return

        if self._grid_mode == "latlon":
            for slider in [
                self._resolution_slider,
                self._xstart_slider,
                self._lenx_slider,
                self._ystart_slider,
                self._leny_slider,
            ]:
                slider.observe(self._on_slider_change, names="value")

    # ------------------------------------------------------------------
    # Mode management (pre-grid)
    # ------------------------------------------------------------------

    def _on_mode_change(self, change):
        mode = change["new"]
        self._latlon_panel.layout.display = "" if mode == "Lat/Lon Corners" else "none"
        self._center_panel.layout.display = "" if mode == "From Center" else "none"
        self._proj_panel.layout.display = "" if mode == "From Projection" else "none"
        self._click_points = []
        if self._select_button.value:
            self._select_button.value = False
        self._update_status_for_mode(mode)
        if mode == "From Projection":
            proj, extent = self._crs_to_cartopy_proj(self._proj_crs_text.value)
            self._set_map_projection(proj, extent)
        elif not isinstance(self._current_map_proj, ccrs.PlateCarree):
            self._set_map_projection(ccrs.PlateCarree(), None)

    def _on_proj_crs_preset_change(self, change):
        if change["new"]:
            self._proj_crs_text.value = change["new"]
            if self._mode_selector.value == "From Projection":
                proj, extent = self._crs_to_cartopy_proj(change["new"])
                self._set_map_projection(proj, extent)

    # ------------------------------------------------------------------
    # Click-to-create
    # ------------------------------------------------------------------

    def _start_click_mode(self):
        self._click_points = []
        self._select_button.value = False
        self._select_button.observe(self._on_select_toggle, names="value")
        if self._click_cid is None:
            self._click_cid = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_map_click
            )

    def _stop_click_mode(self):
        if self._click_cid is not None:
            self.fig.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = None
        try:
            self._select_button.unobserve(self._on_select_toggle, names="value")
        except ValueError:
            pass

    def _on_select_toggle(self, change):
        mode = self._mode_selector.value
        if change["new"]:
            self._click_points = []
            if mode == "From Center":
                self._status_html.value = (
                    "<p><b>Click the domain centre on the map.</b></p>"
                )
            else:
                self._status_html.value = "<p><b>Click corner 1 of 2.</b></p>"
            self._select_button.description = "Cancel"
            self._select_button.button_style = "warning"
        else:
            self._click_points = []
            self._select_button.description = "Select Region"
            self._select_button.button_style = "info"
            self._update_status_for_mode(mode)

    def _on_map_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if not self._select_button.value:
            return

        mode = self._mode_selector.value
        # In PlateCarree: x=lon, y=lat (degrees).
        # In a native projection: x/y are in that CRS's units (usually metres).
        x, y = event.xdata, event.ydata

        # Marker: omit transform when in a native projection so cartopy doesn't
        # re-interpret the projected metres as geographic degrees.
        plot_kw = (
            {}
            if not isinstance(self._current_map_proj, ccrs.PlateCarree)
            else {"transform": ccrs.PlateCarree()}
        )
        self.ax.plot(x, y, "r+", markersize=10, **plot_kw)
        self.fig.canvas.draw_idle()

        if mode == "From Center":
            # Center mode always uses PlateCarree, so x/y are lon/lat here.
            self._select_button.value = False
            self._stop_click_mode()
            self._create_grid_from_center(x, y)
        else:
            self._click_points.append((x, y))
            if len(self._click_points) == 1:
                self._status_html.value = "<p><b>Now click corner 2 of 2.</b></p>"
            elif len(self._click_points) == 2:
                (x1, y1), (x2, y2) = self._click_points
                self._select_button.value = False
                self._stop_click_mode()
                if mode == "Lat/Lon Corners":
                    self._create_grid_from_clicks(x1, y1, x2, y2)
                else:
                    self._create_grid_from_projection(x1, y1, x2, y2)

    def _create_grid_from_clicks(self, x1, y1, x2, y2):
        xstart = min(x1, x2)
        ystart = min(y1, y2)
        lenx = abs(x2 - x1)
        leny = abs(y2 - y1)
        resolution = max(lenx, leny) / 20  # ~20 cells across the larger dimension

        self.grid = Grid(
            lenx=lenx,
            leny=leny,
            resolution=resolution,
            xstart=xstart,
            ystart=ystart,
            type=self._latlon_grid_type.value,
        )
        self._grid_mode = "latlon"
        self._initial_params = None
        self._switch_to_grid_mode()
        self.plot_grid()

    def _create_grid_from_center(self, lon, lat):
        width_m = self._center_width.value * 1000
        height_m = self._center_height.value * 1000
        resolution_m = self._center_resolution.value * 1000
        angle_deg = self._center_angle.value
        self._center_latlon = (lat, lon)
        try:
            self.grid = Grid.from_center(
                lat, lon, width_m, height_m, resolution_m, angle_deg
            )
        except Exception as e:
            print(f"Failed to create grid from centre: {e}")
            return
        self._grid_mode = "center"
        self._initial_params = None
        self._switch_to_grid_mode()
        self.plot_grid()

    def _create_grid_from_projection(self, x1, y1, x2, y2):
        crs_str = self._proj_crs_text.value.strip()
        resolution_m = self._proj_resolution.value * 1000
        if isinstance(self._current_map_proj, ccrs.PlateCarree):
            # Clicks are in lon/lat — transform to projected metres
            try:
                t = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
                px1, py1 = t.transform(x1, y1)
                px2, py2 = t.transform(x2, y2)
            except Exception as e:
                print(f"Failed to transform coordinates to {crs_str}: {e}")
                return
        else:
            # Clicks are already in the native projection's metres
            px1, py1, px2, py2 = x1, y1, x2, y2
        x_min, x_max = min(px1, px2), max(px1, px2)
        y_min, y_max = min(py1, py2), max(py1, py2)
        self._proj_extents = (x_min, x_max, y_min, y_max)
        try:
            self.grid = Grid.from_projection(
                crs_str, x_min, x_max, y_min, y_max, resolution_m
            )
        except Exception as e:
            print(f"Failed to create projected grid: {e}")
            return
        self._grid_mode = "projection"
        self._initial_params = None
        self._switch_to_grid_mode()
        self.plot_grid()

    def _on_recreate_click(self, _btn=None):
        if self._grid_mode == "center" and self._center_latlon is not None:
            lat, lon = self._center_latlon
            self._create_grid_from_center(lon, lat)
        elif self._grid_mode == "projection" and self._proj_extents is not None:
            crs_str = self._proj_crs_text.value.strip()
            resolution_m = self._proj_resolution.value * 1000
            x_min, x_max, y_min, y_max = self._proj_extents
            try:
                self.grid = Grid.from_projection(
                    crs_str, x_min, x_max, y_min, y_max, resolution_m
                )
            except Exception as e:
                print(f"Failed to recreate projected grid: {e}")
                return
            self.plot_grid()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _on_extent_changed(self, ax):
        """Redraw map content when the user zooms or pans, preserving the new extent."""
        if self._in_redraw:
            return
        self._in_redraw = True
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self._draw_map_content()
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.fig.canvas.draw_idle()
        finally:
            self._in_redraw = False

    def _draw_map_content(self):
        """Clear and redraw coastlines, features, grid lines, and labels."""
        self.ax.clear()
        self.ax.coastlines(resolution="10m", linewidth=0.8)
        self.ax.add_feature(cfeature.LAND, facecolor="0.9")
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        if self.grid is not None:
            n_jq, n_iq = self.grid.qlon.shape
            for i in range(n_iq):
                self.ax.plot(
                    self.grid.qlon[:, i],
                    self.grid.qlat[:, i],
                    color="k",
                    linewidth=0.1,
                    transform=ccrs.PlateCarree(),
                )
            for j in range(n_jq):
                self.ax.plot(
                    self.grid.qlon[j, :],
                    self.grid.qlat[j, :],
                    color="k",
                    linewidth=0.1,
                    transform=ccrs.PlateCarree(),
                )
            title = (
                "Use the sliders to adjust grid parameters."
                if self._grid_mode == "latlon"
                else "Grid created — adjust parameters in the control panel."
            )
            self.ax.set_title(title)
            gl = self.ax.gridlines(draw_labels=True, linewidth=0, color="none")
        else:
            self.ax.set_title(
                "Click two corners on the map to define your grid region."
            )
            gl = self.ax.gridlines(
                draw_labels=True, linewidth=0.3, color="gray", alpha=0.5
            )

        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}

    def plot_world(self):
        self._in_redraw = True
        try:
            self._draw_map_content()
            self.ax.set_global()
            self.fig.canvas.draw_idle()
        finally:
            self._in_redraw = False

    def plot_grid(self):
        self._in_redraw = True
        try:
            self._draw_map_content()
            lon_min, lon_max = float(self.grid.qlon.min()), float(self.grid.qlon.max())
            lat_min, lat_max = float(self.grid.qlat.min()), float(self.grid.qlat.max())
            self.ax.set_extent(
                [lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree()
            )
            self._draw_scale_bar(lon_min, lon_max, lat_min, lat_max)
            self.fig.canvas.draw_idle()
        finally:
            self._in_redraw = False

    def _nice_scale_length(self, length_m):
        import math

        if length_m == 0:
            return 0
        exp = math.floor(math.log10(length_m))
        base = length_m / (10**exp)
        if base < 1.5:
            nice = 1
        elif base < 3.5:
            nice = 2
        elif base < 7.5:
            nice = 5
        else:
            nice = 10
        return nice * (10**exp)

    def _draw_scale_bar(self, lon_min, lon_max, lat_min, lat_max):
        try:
            frac = 0.2
            bar_lat = lat_min + 0.05 * (lat_max - lat_min)
            bar_lon_start = lon_min + 0.05 * (lon_max - lon_min)
            bar_lon_end = bar_lon_start + frac * (lon_max - lon_min)

            R = 6371000
            lat_rad = np.deg2rad(bar_lat)
            dlon_rad = np.deg2rad(bar_lon_end - bar_lon_start)
            bar_length_m = abs(dlon_rad * np.cos(lat_rad) * R)

            nice_length_m = self._nice_scale_length(bar_length_m)
            nice_dlon_deg = np.rad2deg(nice_length_m / (np.cos(lat_rad) * R))
            bar_lon_end = bar_lon_start + nice_dlon_deg

            label = (
                f"{int(nice_length_m/1000)} km"
                if nice_length_m >= 1000
                else f"{int(nice_length_m)} m"
            )

            self.ax.plot(
                [bar_lon_start, bar_lon_end],
                [bar_lat, bar_lat],
                color="k",
                linewidth=3,
                transform=ccrs.PlateCarree(),
            )
            self.ax.text(
                (bar_lon_start + bar_lon_end) / 2,
                bar_lat + 0.01 * (lat_max - lat_min),
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                transform=ccrs.PlateCarree(),
            )
        except Exception as e:
            print(f"Failed to draw scale bar: {e}")

    # ------------------------------------------------------------------
    # Grid operations
    # ------------------------------------------------------------------

    def save_grid(self, _btn=None):
        name = self._snapshot_name.value.strip()
        msg = self._commit_msg.value.strip()
        if not name:
            print("Enter a grid name!")
            return
        if not msg:
            print("Enter a grid message!")
            return
        if self.grid is None:
            print("No grid to save — define a grid first.")
            return

        if name.lower().endswith(".nc"):
            name = name[:-3]
        self.grid.name = name

        nc_path = os.path.join(self.grids_dir, f"grid_{name}.nc")
        self.grid.write_supergrid(nc_path)
        print(f"Saved grid '{os.path.basename(nc_path)}' in '{self.grids_dir}'.")
        self.refresh_commit_dropdown()

    def load_grid(self, b=None):
        val = self._commit_dropdown.value
        if not val:
            return
        nc_path = os.path.join(self.grids_dir, val)
        try:
            self.grid = Grid.from_supergrid(nc_path)
            ds = xr.open_dataset(nc_path)
            grid_type = ds.attrs.get("grid_type", "uniform_spherical")

            self._center_latlon = None
            self._proj_extents = None

            if grid_type == "projected_center":
                self._grid_mode = "center"
                self._center_latlon = (ds.attrs["center_lat"], ds.attrs["center_lon"])
                self._center_width.value = ds.attrs["width_m"] / 1000
                self._center_height.value = ds.attrs["height_m"] / 1000
                self._center_resolution.value = ds.attrs["resolution_m"] / 1000
                self._center_angle.value = ds.attrs.get("angle_deg", 0.0)
            elif grid_type == "projected_crs":
                self._grid_mode = "projection"
                self._proj_extents = (
                    ds.attrs["x_min"],
                    ds.attrs["x_max"],
                    ds.attrs["y_min"],
                    ds.attrs["y_max"],
                )
                self._proj_resolution.value = ds.attrs["resolution_m"] / 1000
                epsg = CRS.from_wkt(ds.attrs["crs_wkt"]).to_epsg()
                self._proj_crs_text.value = (
                    f"EPSG:{epsg}" if epsg else ds.attrs["crs_wkt"]
                )
            else:
                self._grid_mode = "latlon"
                self._latlon_grid_type.value = (
                    "rectilinear_cartesian"
                    if grid_type == "rectilinear_cartesian"
                    else "uniform_spherical"
                )

            self._stop_click_mode()
            if not isinstance(self._current_map_proj, ccrs.PlateCarree):
                self._set_map_projection(ccrs.PlateCarree(), None)
            self._switch_to_grid_mode()
            if self._grid_mode == "latlon":
                self.sync_sliders_to_grid()
            self.plot_grid()
            print(f"Loaded grid from '{nc_path}'.")
        except Exception as e:
            print(f"Failed to load grid: {e}")
            import traceback

            traceback.print_exc()

    def sync_sliders_to_grid(self):
        if self.grid is None:
            return
        try:
            initial_xstart = float(self.grid.supergrid.x[0, 0]) % 360
            slider_window = 30
            slider_min = max(initial_xstart - slider_window, -180.0)
            slider_max = min(initial_xstart + slider_window, 360.0)
            if slider_min >= slider_max:
                slider_min = max(-180.0, initial_xstart - 15)
                slider_max = min(360.0, initial_xstart + 15)
                if slider_min >= slider_max:
                    slider_min = -180.0
                    slider_max = 360.0
            xstart_val = min(max(initial_xstart, slider_min), slider_max)

            initial_ystart = float(self.grid.supergrid.y[0, 0])
            y_min = max(initial_ystart - 30, -90)
            y_max = min(initial_ystart + 30, 90)
            if y_min >= y_max:
                y_min = max(-90, initial_ystart - 15)
                y_max = min(90, initial_ystart + 15)
                if y_min >= y_max:
                    y_min = -90
                    y_max = 90
            ystart_val = min(max(initial_ystart, y_min), y_max)

            res_min, res_max = 0.01, 1.0
            resolution_val = min(
                max(float(self.grid.lenx / self.grid.nx), res_min), res_max
            )
            lenx_val = min(max(float(self.grid.lenx), 0.01), 50.0)
            leny_val = min(max(float(self.grid.leny), 0.01), 50.0)

            for slider in [
                self._resolution_slider,
                self._xstart_slider,
                self._lenx_slider,
                self._ystart_slider,
                self._leny_slider,
            ]:
                slider.unobserve(self._on_slider_change, names="value")

            self._xstart_slider.min = slider_min
            self._xstart_slider.max = slider_max
            self._xstart_slider.value = xstart_val
            self._ystart_slider.min = y_min
            self._ystart_slider.max = y_max
            self._ystart_slider.value = ystart_val
            self._resolution_slider.value = resolution_val
            self._lenx_slider.value = lenx_val
            self._leny_slider.value = leny_val

            for slider in [
                self._resolution_slider,
                self._xstart_slider,
                self._lenx_slider,
                self._ystart_slider,
                self._leny_slider,
            ]:
                slider.observe(self._on_slider_change, names="value")

        except Exception as e:
            print(f"Error in sync_sliders_to_grid: {e}")

    def _on_slider_change(self, change):
        self.grid = Grid(
            lenx=self._lenx_slider.value,
            leny=self._leny_slider.value,
            resolution=self._resolution_slider.value,
            xstart=self._xstart_slider.value,
            ystart=self._ystart_slider.value,
            name=self.grid.name,
            type=self._latlon_grid_type.value,
        )
        self.plot_grid()

    def reset_grid(self, b=None):
        if self._initial_params is None:
            # No original grid — go back to click-to-create mode
            self.grid = None
            self._grid_mode = "latlon"
            self._center_latlon = None
            self._proj_extents = None
            self._control_panel.children = [
                self._build_creator_controls(),
                self._control_panel.children[1],
            ]
            self.construct_observances()
            if not isinstance(self._current_map_proj, ccrs.PlateCarree):
                self._set_map_projection(ccrs.PlateCarree(), None)
            else:
                self.plot_world()
            self._start_click_mode()
            return

        if not isinstance(self._current_map_proj, ccrs.PlateCarree):
            self._set_map_projection(ccrs.PlateCarree(), None)
        params = self._initial_params
        name = self._snapshot_name.value.strip() or params["name"]
        self.grid = Grid(
            lenx=params["lenx"],
            leny=params["leny"],
            resolution=params["lenx"] / params["nx"],
            xstart=params["xstart"],
            ystart=params["ystart"],
            name=name,
        )
        self._grid_mode = "latlon"
        self.sync_sliders_to_grid()
        self.plot_grid()
        grid_nc_name = f"grid_{name}.nc"
        option_values = [v for (l, v) in self._commit_dropdown.options]
        if grid_nc_name in option_values:
            self._commit_dropdown.value = grid_nc_name
        self.update_commit_details()

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    def refresh_commit_dropdown(self):
        grid_nc_files = [
            fname
            for fname in os.listdir(self.grids_dir)
            if fname.startswith("grid_") and fname.endswith(".nc")
        ]
        options = []
        current_grid_nc = None
        for fname in grid_nc_files:
            abs_path = os.path.join(self.grids_dir, fname)
            try:
                ds = xr.open_dataset(abs_path)
                name = ds.attrs.get("name", "")
                options.append((name, fname))
                if self.grid is not None and name == self.grid.name:
                    current_grid_nc = fname
            except Exception:
                continue

        options.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.grids_dir, x[1])),
            reverse=True,
        )

        self._commit_dropdown.options = options if options else []
        if options:
            option_values = [v for (l, v) in options]
            if current_grid_nc and current_grid_nc in option_values:
                self._commit_dropdown.value = current_grid_nc
            elif self._commit_dropdown.value not in option_values:
                self._commit_dropdown.value = options[0][1]
        else:
            self._commit_dropdown.value = None
        self.update_commit_details()

    def update_commit_details(self, change=None):
        val = self._commit_dropdown.value
        if not val:
            self._commit_details.value = ""
            return
        abs_path = os.path.join(self.grids_dir, val)
        try:
            grid = Grid.from_supergrid(abs_path)
            ds = xr.open_dataset(abs_path)
            name = ds.attrs.get("name", "")
            date = ds.attrs.get("Created", "")
            date_short = date.replace("T", " ")
            date_short = date_short.split(".")[0] if "." in date_short else date_short
            details = (
                f"<b>Name:</b> {name}<br>"
                f"<b>Created:</b> {date_short}<br>"
                f"<b>nx:</b> {grid.nx} <b>ny:</b> {grid.ny}"
            )
            self._commit_details.value = details
        except Exception as e:
            self._commit_details.value = f"<b>Error:</b> {e}"
