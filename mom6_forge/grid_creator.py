import os
import numpy as np
import xarray as xr
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import RectangleSelector
from mom6_forge.grid import Grid
from pathlib import Path
from pyproj import CRS, Transformer

# For projection grid creation, offer some convenient CRS presets in a dropdown
_CRS_PRESETS = [
    ("Equidistant Cylindrical / PlateCarree (EPSG:4087)", "EPSG:4087"),
    ("Arctic Polar Stereographic (EPSG:3995)", "EPSG:3995"),
    ("Antarctic Polar Stereographic (EPSG:3031)", "EPSG:3031"),
    ("CONUS Albers Equal Area (EPSG:5070)", "EPSG:5070"),
]

# Preset EPSG → (cartopy_proj, default_extent_in_PlateCarree)
# extent is [lon_min, lon_max, lat_min, lat_max], or None for set_global()
_EPSG_TO_CARTOPY = {
    4087: (ccrs.PlateCarree(), None),
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

    The widget has two modes depending on how it is constructed:

    Create Mode  (``GridCreator()``)
    ---------------------------------
    Select a creation method, then press **Select Region** and interact with the map:

    Lat/Lon Corners   : drag a rectangle on the PlateCarree map →
                        uniform-degree Grid via ``Grid(lenx, leny, ...)``
    From Center       : set width/height/resolution/angle, click once to place the
                        domain centre → ``Grid.from_center(...)``
    From Projection   : set a CRS + resolution, drag a rectangle on the native
                        projection map → ``Grid.from_projection(...)``

    Edit Mode  (``GridCreator(grid=some_latlon_grid)``)
    ----------------------------------------------------
    Accepts a lat/lon grid only — init args are backed out from the supergrid
    properties and exposed as live sliders.  Passing a *projected* grid raises
    ``ValueError`` because the creation parameters are not stored in the supergrid
    file.  Projected grids created *within this session* can be edited via their
    Recreate panel (the session holds the init args in memory).

    Library
    -------
    Grids are saved as NetCDF supergrids under ``<working_dir>/GridLibrary/``.
    The dropdown lists all ``grid_*.nc`` files there.  Loading a projected grid
    from the library is not supported (parameters cannot be recovered from the file).

    Map projection
    --------------
    Entering "From Projection" mode switches the cartopy axes to the native
    projection for the selected CRS (preset EPSG codes only; unknown codes fall
    back to PlateCarree).  All other modes use PlateCarree.
    """

    def __init__(self, grid=None, working_dir=None):
        self.grid = grid
        self.working_dir = Path(working_dir if working_dir is not None else os.getcwd())
        self.grids_dir = Path(os.path.join(self.working_dir, "GridLibrary"))
        self.grids_dir.mkdir(exist_ok=True)
        (self.grids_dir / ".gitignore").write_text("*\n")

        # Click-capture state
        self._click_cid = None  # mpl canvas connection id for center-click, or None
        self._rect_selector = None  # matplotlib RectangleSelector, or None

        # Redraw guard — prevents recursive xlim_changed → redraw loops
        self._in_redraw = False

        # Grid creation mode and associated stored parameters for Recreate
        self._edit_mode = "latlon"  # "latlon" - default | "center" | "projection"
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

        self.ax.callbacks.connect(
            "xlim_changed", self._on_extent_changed
        )  # This connects to the ipywidget zoom feature

        super().__init__(
            [self._control_panel, self.fig.canvas],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )

        self.refresh_library_dropdown()
        if self.grid is not None:
            grid_type = self.grid.supergrid.grid_type
            if grid_type in ("projected_crs", "projected_center"):
                raise ValueError(
                    "GridCreator cannot accept a projected grid directly — "
                    "init args (CRS, extents, center) cannot be recovered from the "
                    "supergrid file. Create the grid interactively via the creator "
                    "instead, or load it from the GridLibrary after creating it in "
                    "this session."
                )
            self.load_grid(grid=self.grid)
        else:
            self.plot_world()
            self._start_click_mode()  # Interactivity

    # ------------------------------------------------------------------
    # Control panel construction
    # ------------------------------------------------------------------

    def construct_control_panel(self):
        # --- Mode selector (create mode) ---
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
                    "<p>Drag a rectangle on the map to define the grid extent.</p>"
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
            value="EPSG:4087",
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
        self._move_center_button = widgets.ToggleButton(
            value=False,
            description="Move Center",
            button_style="info",
            icon="crosshairs",
            layout={"width": "90%"},
        )
        self._move_center_cid = None

        # --- Lat/lon edit sliders ---
        _fw = {"width": "90%"}  # already defined above but repeated for clarity
        self._resolution_slider = widgets.FloatSlider(
            value=0.5, min=0.01, max=1.0, step=0.01, description="Resolution"
        )
        self._xstart_slider = widgets.FloatSlider(
            value=0, min=-180, max=360, step=0.01, description="xstart"
        )
        self._lenx_slider = widgets.FloatSlider(
            value=10, min=0.01, max=50.0, step=0.01, description="lenx"
        )
        self._ystart_slider = widgets.FloatSlider(
            value=0, min=-90, max=90, step=0.01, description="ystart"
        )
        self._leny_slider = widgets.FloatSlider(
            value=10, min=0.01, max=50.0, step=0.01, description="leny"
        )

        # --- Library ---
        self._grid_name = widgets.Text(
            value="",
            placeholder="Enter grid name",
            description="Name:",
            layout={"width": "90%"},
        )
        self._library_dropdown = widgets.Dropdown(
            options=[], description="Grids:", layout={"width": "90%"}
        )
        self._grid_details = widgets.HTML(
            value="", layout={"width": "90%", "min_height": "2em"}
        )
        self._save_button = widgets.Button(
            description="Save Grid", layout={"width": "44%"}
        )
        self._load_button = widgets.Button(
            description="Load Grid", layout={"width": "44%"}
        )

        creator_controls = self._build_controls()

        library_section = widgets.VBox(
            [
                widgets.HTML("<h3>Library</h3>"),
                self._grid_name,
                self._library_dropdown,
                self._grid_details,
                widgets.HBox([self._save_button, self._load_button]),
            ]
        )

        self._control_panel = widgets.VBox(
            [creator_controls, library_section],
            layout={"width": "45%", "height": "100%"},
        )

    def _build_controls(self):
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
                    widgets.HTML("<h3>Grid Creator &mdash; Create Mode</h3>"),
                    self._mode_selector,
                    self._latlon_panel,
                    self._center_panel,
                    self._proj_panel,
                    self._status_html,
                    self._select_button,
                ],
                layout=layout,
            )

        if self._edit_mode == "latlon":
            return widgets.VBox(
                [
                    widgets.HTML("<h3>Grid Creator &mdash; Edit Mode</h3>"),
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

        # Projected grid (center or projection mode) — edit mode, init args held in session
        if self._edit_mode == "center":
            center_info = ""
            if self._center_latlon is not None:
                lat, lon = self._center_latlon
                center_info = f"<p><b>Centre:</b> {lat:.3f}°N, {lon:.3f}°E</p>"
            header = widgets.HTML(
                f"<h3>Grid Creator &mdash; Edit Mode</h3>{center_info}"
            )
            mode_inputs = widgets.VBox(
                [
                    self._center_width,
                    self._center_height,
                    self._center_resolution,
                    self._center_angle,
                    self._move_center_button,
                ]
            )
            self._recreate_button.disabled = self._center_latlon is None
        else:  # projection mode
            header = widgets.HTML("<h3>Grid Creator &mdash; Edit Mode</h3>")
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

    def _update_slider_ranges(self):
        # Capture all values before any slider changes fire _on_slider_change
        initial_xstart = float(self.grid.supergrid.x[0, 0]) % 360
        initial_ystart = float(self.grid.supergrid.y[0, 0])
        lenx = float(self.grid.lenx)
        leny = float(self.grid.leny)
        resolution = lenx / self.grid.nx

        slider_window = 30
        xmin = max(initial_xstart - slider_window, -180.0)
        xmax = min(initial_xstart + slider_window, 360.0)
        if xmin >= xmax:
            xmin = max(-180.0, initial_xstart - 15)
            xmax = min(360.0, initial_xstart + 15)

        ymin = max(initial_ystart - slider_window, -90.0)
        ymax = min(initial_ystart + slider_window, 90.0)
        if ymin >= ymax:
            ymin = max(-90.0, initial_ystart - 15)
            ymax = min(90.0, initial_ystart + 15)

        sliders = [
            self._resolution_slider,
            self._xstart_slider,
            self._lenx_slider,
            self._ystart_slider,
            self._leny_slider,
        ]
        for slider in sliders:
            slider.unobserve(self._on_slider_change, names="value")

        self._xstart_slider.min = -180
        self._xstart_slider.max = 360
        self._xstart_slider.value = initial_xstart
        self._xstart_slider.min = xmin
        self._xstart_slider.max = xmax

        self._lenx_slider.value = lenx

        self._ystart_slider.min = -90
        self._ystart_slider.max = 90
        self._ystart_slider.value = initial_ystart
        self._ystart_slider.min = ymin
        self._ystart_slider.max = ymax

        self._leny_slider.value = leny
        self._resolution_slider.value = resolution

        for slider in sliders:
            slider.observe(self._on_slider_change, names="value")

    def _switch_to_edit_mode(self):
        """Replace the creator controls panel after a grid is created or loaded."""
        if self._edit_mode == "latlon":
            self._update_slider_ranges()
        creator_controls = self._build_controls()
        library_section = self._control_panel.children[1]
        self._control_panel.children = [creator_controls, library_section]

    def _update_status_for_mode(self, mode):
        if mode == "Lat/Lon Corners":
            self._status_html.value = (
                "<p>Zoom/pan to your region, then press <b>Select Region</b> "
                "and drag a rectangle on the map.</p>"
            )
        elif mode == "From Center":
            self._status_html.value = (
                "<p>Set dimensions, then press <b>Select Region</b> "
                "and click to place the domain centre.</p>"
            )
        else:
            self._status_html.value = (
                "<p>Set CRS and resolution, then press <b>Select Region</b> "
                "and drag a rectangle on the map.</p>"
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
        # RectangleSelector is bound to a specific Axes — recreate it on the new axes
        if self.grid is None:
            was_active = (
                self._select_button.value and self._mode_selector.value != "From Center"
            )
            self._setup_rect_selector()
            if was_active and self._rect_selector is not None:
                self._rect_selector.set_active(True)

    def _on_grid_name_change(self, change):
        self.refresh_library_dropdown()

    def _unregister_observances(self):
        for btn, handler in [
            (self._save_button, self.save_grid),
            (self._load_button, self.load_grid),
            (self._reset_button, self.reset_grid),
            (self._recreate_button, self._on_recreate_click),
        ]:
            try:
                btn.on_click(handler, remove=True)
            except ValueError:
                pass
        for widget, handler in [
            (self._grid_name, self._on_grid_name_change),
            (self._library_dropdown, self.update_grid_details),
            (self._mode_selector, self._on_mode_change),
            (self._proj_crs_dropdown, self._on_proj_crs_preset_change),
            (self._move_center_button, self._on_move_center_toggle),
            (self._resolution_slider, self._on_slider_change),
            (self._xstart_slider, self._on_slider_change),
            (self._lenx_slider, self._on_slider_change),
            (self._ystart_slider, self._on_slider_change),
            (self._leny_slider, self._on_slider_change),
        ]:
            try:
                widget.unobserve(handler, names="value")
            except ValueError:
                pass

    def construct_observances(self):
        self._unregister_observances()
        self._save_button.on_click(self.save_grid)
        self._load_button.on_click(self.load_grid)
        self._reset_button.on_click(self.reset_grid)
        self._recreate_button.on_click(self._on_recreate_click)
        self._grid_name.observe(self._on_grid_name_change, names="value")
        self._library_dropdown.observe(self.update_grid_details, names="value")
        for slider in [
            self._resolution_slider,
            self._xstart_slider,
            self._lenx_slider,
            self._ystart_slider,
            self._leny_slider,
        ]:
            slider.observe(self._on_slider_change, names="value")
        self._move_center_button.observe(self._on_move_center_toggle, names="value")

        if self.grid is None:
            self._mode_selector.observe(self._on_mode_change, names="value")
            self._proj_crs_dropdown.observe(
                self._on_proj_crs_preset_change, names="value"
            )

    # ------------------------------------------------------------------
    # Mode management (pre-grid)
    # ------------------------------------------------------------------

    def _on_mode_change(self, change):
        mode = change["new"]
        self._latlon_panel.layout.display = "" if mode == "Lat/Lon Corners" else "none"
        self._center_panel.layout.display = "" if mode == "From Center" else "none"
        self._proj_panel.layout.display = "" if mode == "From Projection" else "none"
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
    # Rectangle-select-to-create  (single-click for From Center)
    # ------------------------------------------------------------------

    def _setup_rect_selector(self):
        """Create a new RectangleSelector on the current axes (starts inactive)."""
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
        self._rect_selector = RectangleSelector(
            self.ax,
            self._on_rect_select,
            useblit=False,
            button=[1],
            interactive=False,
            props=dict(
                edgecolor="royalblue", facecolor="lightblue", alpha=0.4, fill=True
            ),
        )
        self._rect_selector.set_active(False)

    def _start_click_mode(self):
        self._select_button.value = False
        self._select_button.observe(self._on_select_toggle, names="value")
        # Single-click handler (From Center mode only)
        if self._click_cid is None:
            self._click_cid = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_map_click
            )
        # Rectangle selector (Lat/Lon and Projection modes)
        self._setup_rect_selector()

    def _stop_click_mode(self):
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
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
            if mode == "From Center":
                self._status_html.value = (
                    "<p><b>Click the domain centre on the map.</b></p>"
                )
            else:
                self._status_html.value = "<p><b>Drag a rectangle on the map to define the grid extent.</b></p>"
                if self._rect_selector is not None:
                    self._rect_selector.set_active(True)
            self._select_button.description = "Cancel"
            self._select_button.button_style = "warning"
        else:
            if self._rect_selector is not None:
                try:
                    self._rect_selector.set_active(False)
                except Exception:
                    pass
            self._select_button.description = "Select Region"
            self._select_button.button_style = "info"
            self._update_status_for_mode(mode)

    def _on_rect_select(self, eclick, erelease):
        """Called by RectangleSelector when the user finishes drawing a rectangle."""
        if not self._select_button.value:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        # Deactivate selector and reset the toggle button (triggers _on_select_toggle OFF)
        if self._rect_selector is not None:
            self._rect_selector.set_active(False)
        self._select_button.value = False
        self._stop_click_mode()
        mode = self._mode_selector.value
        if mode == "Lat/Lon Corners":
            self._create_grid_from_clicks(x1, y1, x2, y2)
        else:  # From Projection
            self._create_grid_from_projection(x1, y1, x2, y2)

    def _on_map_click(self, event):
        """Click handler — only used for From Center mode."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        if not self._select_button.value:
            return
        if self._mode_selector.value != "From Center":
            return
        x, y = event.xdata, event.ydata
        # Center mode always uses PlateCarree, so x/y are lon/lat
        self.ax.plot(x, y, "r+", markersize=10, transform=ccrs.PlateCarree())
        self.fig.canvas.draw_idle()
        self._select_button.value = False
        self._stop_click_mode()
        self._create_grid_from_center(x, y)

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
        self._edit_mode = "latlon"
        self._switch_to_edit_mode()
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
        self._edit_mode = "center"
        self._switch_to_edit_mode()
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
        self._edit_mode = "projection"
        self._switch_to_edit_mode()
        self.plot_grid()

    def _on_recreate_click(self, _btn=None):
        if self._edit_mode == "center" and self._center_latlon is not None:
            lat, lon = self._center_latlon
            self._create_grid_from_center(lon, lat)
        elif self._edit_mode == "projection" and self._proj_extents is not None:
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

    def _on_move_center_toggle(self, change):
        if change["new"]:
            self._move_center_button.description = "Cancel"
            self._move_center_button.button_style = "warning"
            self._move_center_cid = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_center_edit_click
            )
        else:
            self._move_center_button.description = "Move Center"
            self._move_center_button.button_style = "info"
            if self._move_center_cid is not None:
                self.fig.canvas.mpl_disconnect(self._move_center_cid)
                self._move_center_cid = None

    def _on_center_edit_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if not self._move_center_button.value:
            return
        lon, lat = event.xdata, event.ydata
        self.ax.plot(lon, lat, "r+", markersize=10, transform=ccrs.PlateCarree())
        self.fig.canvas.draw_idle()
        self._move_center_button.value = False  # triggers _on_move_center_toggle OFF
        self._create_grid_from_center(lon, lat)

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
                if self._edit_mode == "latlon"
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
        name = self._grid_name.value.strip()
        if not name:
            print("Enter a grid name!")
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
        self.refresh_library_dropdown()

    def load_grid(self, b=None, grid=None):
        loading_from_library = grid is None
        if loading_from_library:
            val = self._library_dropdown.value
            if not val:
                return
            nc_path = os.path.join(self.grids_dir, val)
            candidate = Grid.from_supergrid(nc_path)
        else:
            candidate = grid

        grid_type = candidate.supergrid.grid_type
        if grid_type in ("projected_crs", "projected_center"):
            # Can only edit a projected grid if this session created it (init args in memory)
            if loading_from_library or (
                self._center_latlon is None and self._proj_extents is None
            ):
                msg = (
                    "Cannot load a projected grid from the library for editing — "
                    "the creation parameters (CRS, extents, center) are not stored "
                    "in the supergrid file. Load the grid programmatically and "
                    "visualise it, or recreate it interactively."
                )
                self._grid_details.value = f"<span style='color:red'>{msg}</span>"
                print(msg)
                return

        self.grid = candidate
        try:
            if grid_type in ("projected_crs", "projected_center"):
                # init args already in memory from this session — just switch to edit
                # _center_latlon / _proj_extents / widget values remain as set
                self._edit_mode = (
                    "center" if self._center_latlon is not None else "projection"
                )
            else:
                self._edit_mode = "latlon"
                self._latlon_grid_type.value = (
                    "rectilinear_cartesian"
                    if grid_type == "rectilinear_cartesian"
                    else "uniform_spherical"
                )

            self._stop_click_mode()
            if not isinstance(self._current_map_proj, ccrs.PlateCarree):
                self._set_map_projection(ccrs.PlateCarree(), None)
            self._switch_to_edit_mode()
            if self._edit_mode == "latlon":
                self.sync_sliders_to_grid()
            self.plot_grid()
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
        from mom6_forge.grid import Grid

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
        # go back to click-to-create mode
        self.grid = None
        self._edit_mode = "latlon"
        self._center_latlon = None
        self._proj_extents = None
        self._proj_crs_dropdown.value = _CRS_PRESETS[0][1]
        self._control_panel.children = [
            self._build_controls(),
            self._control_panel.children[1],
        ]
        self.construct_observances()
        if not isinstance(self._current_map_proj, ccrs.PlateCarree):
            self._set_map_projection(ccrs.PlateCarree(), None)
        else:
            self.plot_world()
        self._start_click_mode()

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    def refresh_library_dropdown(self):
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

        self._library_dropdown.options = options if options else []
        if options:
            option_values = [v for (l, v) in options]
            if current_grid_nc and current_grid_nc in option_values:
                self._library_dropdown.value = current_grid_nc
            elif self._library_dropdown.value not in option_values:
                self._library_dropdown.value = options[0][1]
        else:
            self._library_dropdown.value = None
        self.update_grid_details()

    def update_grid_details(self, change=None):
        val = self._library_dropdown.value
        if not val:
            self._grid_details.value = ""
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
            self._grid_details.value = details
        except Exception as e:
            self._grid_details.value = f"<b>Error:</b> {e}"
