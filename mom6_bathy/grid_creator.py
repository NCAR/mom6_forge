import os
import numpy as np
import xarray as xr
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mom6_bathy.grid import Grid
from pathlib import Path


class GridCreator(widgets.HBox):

    def __init__(self, grid=None, repo_root=None):
        self.grid = grid
        self.repo_root = Path(repo_root if repo_root is not None else os.getcwd())
        self.grids_dir = Path(os.path.join(self.repo_root, "GridLibrary"))
        self.grids_dir.mkdir(exist_ok=True)

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

        self._click_points = []
        self._click_cid = None
        self._in_redraw = False

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
            self.plot_grid()
        else:
            self.plot_world()
            self._start_click_mode()

    # ------------------------------------------------------------------
    # Control panel construction
    # ------------------------------------------------------------------

    def construct_control_panel(self):
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
        self._reset_button = widgets.Button(
            description="Reset", layout={"width": "100%"}, button_style="danger"
        )
        self._select_button = widgets.ToggleButton(
            value=False,
            description="Select Region",
            button_style="info",
            icon="crosshairs",
            layout={"width": "90%"},
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
        """Return the top section of the control panel.

        If no grid exists, shows click-to-create instructions.
        If a grid exists, builds and returns sliders.

        # TODO: add a grid type selector here (uniform spherical /
        #   projected center / projected CRS) to dispatch to the right
        #   constructor in _create_grid_from_clicks and _on_slider_change.
        """
        if self.grid is None:
            return widgets.VBox(
                [
                    widgets.HTML("<h3>Grid Creator</h3>"),
                    widgets.HTML(
                        "<p>Zoom/pan to your region, then activate point selection to define the grid.</p>"
                        "<p><b>1st click:</b> one corner &nbsp;"
                        "<b>2nd click:</b> opposite corner</p>"
                    ),
                    self._select_button,
                ],
                layout=widgets.Layout(
                    width="100%",
                    min_width="200px",
                    max_width="400px",
                    align_items="stretch",
                ),
            )

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
            layout=widgets.Layout(
                width="100%",
                min_width="200px",
                max_width="400px",
                align_items="stretch",
                overflow_y="auto",
            ),
        )

    def _switch_to_slider_mode(self):
        """Replace the click-instructions panel with sliders after a grid is created."""
        creator_controls = self._build_creator_controls()
        library_section = self._control_panel.children[1]
        self._control_panel.children = [creator_controls, library_section]
        self.construct_observances()

    def construct_observances(self):
        self._save_button.on_click(self.save_grid)
        self._load_button.on_click(self.load_grid)
        self._reset_button.on_click(self.reset_grid)
        self._snapshot_name.observe(
            lambda change: self.refresh_commit_dropdown(), names="value"
        )
        self._commit_dropdown.observe(self.update_commit_details, names="value")

        if self.grid is None:
            return

        for slider in [
            self._resolution_slider,
            self._xstart_slider,
            self._lenx_slider,
            self._ystart_slider,
            self._leny_slider,
        ]:
            slider.observe(self._on_slider_change, names="value")

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
        if change["new"]:
            self._click_points = []
            self._select_button.description = "Cancel Selection"
            self._select_button.button_style = "warning"
        else:
            self._click_points = []
            self._select_button.description = "Select Region"
            self._select_button.button_style = "info"

    def _on_map_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if not self._select_button.value:
            return

        self._click_points.append((event.xdata, event.ydata))
        self.ax.plot(
            event.xdata, event.ydata, "r+", markersize=10,
            transform=ccrs.PlateCarree(),
        )
        self.fig.canvas.draw_idle()

        if len(self._click_points) == 2:
            (x1, y1), (x2, y2) = self._click_points
            self._select_button.value = False
            self._stop_click_mode()
            self._create_grid_from_clicks(x1, y1, x2, y2)

    def _create_grid_from_clicks(self, x1, y1, x2, y2):
        xstart = min(x1, x2)
        ystart = min(y1, y2)
        lenx = abs(x2 - x1)
        leny = abs(y2 - y1)
        resolution = max(lenx, leny) / 20  # ~20 cells across the larger dimension

        # TODO: dispatch to Grid.from_center / Grid.from_projection based on
        #   grid type selector when projected supergrid support is added.
        self.grid = Grid(
            lenx=lenx,
            leny=leny,
            resolution=resolution,
            xstart=xstart,
            ystart=ystart,
        )
        self._initial_params = None  # no original params to reset to
        self._switch_to_slider_mode()
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
            self.ax.set_title("Use the sliders to adjust grid parameters.")
            gl = self.ax.gridlines(draw_labels=True, linewidth=0, color="none")
        else:
            self.ax.set_title("Click two corners on the map to define your grid region.")
            gl = self.ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

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
            self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
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

            label = f"{int(nice_length_m/1000)} km" if nice_length_m >= 1000 else f"{int(nice_length_m)} m"

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
            self._stop_click_mode()
            self._switch_to_slider_mode()
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
            resolution_val = min(max(float(self.grid.lenx / self.grid.nx), res_min), res_max)
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
        # TODO: dispatch to Grid.from_center / Grid.from_projection based on
        #   grid type when projected supergrid support is added.
        self.grid = Grid(
            lenx=self._lenx_slider.value,
            leny=self._leny_slider.value,
            resolution=self._resolution_slider.value,
            xstart=self._xstart_slider.value,
            ystart=self._ystart_slider.value,
            name=self.grid.name,
        )
        self.plot_grid()

    def reset_grid(self, b=None):
        if self._initial_params is None:
            # No original grid — go back to click-to-create mode
            self.grid = None
            self._control_panel.children = [
                self._build_creator_controls(),
                self._control_panel.children[1],
            ]
            self.construct_observances()
            self.plot_world()
            self._start_click_mode()
            return

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
