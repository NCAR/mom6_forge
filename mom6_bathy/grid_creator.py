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

    def __init__(self, grid, repo_root=None):
        self.grid = grid
        self.repo_root = Path(repo_root if repo_root is not None else os.getcwd())
        self.grids_dir = Path(os.path.join(self.repo_root, "GridLibrary"))
        self.grids_dir.mkdir(exist_ok=True)
        self._initial_params = {
            "lenx": grid.lenx,
            "leny": grid.leny,
            "nx": grid.nx,
            "ny": grid.ny,
            "xstart": grid.supergrid.x[0, 0],
            "ystart": grid.supergrid.y[0, 0],
            "name": grid.name,
        }

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

        super().__init__(
            [self._control_panel, self.fig.canvas],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )

        self.refresh_commit_dropdown()
        self.plot_grid()

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

        # Use initial values for slider ranges
        initial_xstart = float(self.grid.supergrid.x[0, 0]) % 360

        # Handle longitude wrap-around and negative values for slider
        slider_window = 30
        slider_min = initial_xstart - slider_window
        slider_max = initial_xstart + slider_window

        # Clamp to [-180, 360] (or [-180, 180] if you prefer)
        if slider_min < -180:
            slider_min = -180.0
        if slider_max > 360:
            slider_max = 360.0

        # If min >= max (e.g., initial_xstart near -180 or 360), set a default window
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

        controls = widgets.VBox(
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
            [
                controls,
                library_section,
            ],
            layout={"width": "45%", "height": "100%"},
        )

    def construct_observances(self):
        self._save_button.on_click(self.save_grid)
        self._load_button.on_click(self.load_grid)
        self._reset_button.on_click(self.reset_grid)
        self._snapshot_name.observe(
            lambda change: self.refresh_commit_dropdown(), names="value"
        )
        self._commit_dropdown.observe(self.update_commit_details, names="value")

        for slider in [
            self._resolution_slider,
            self._xstart_slider,
            self._lenx_slider,
            self._ystart_slider,
            self._leny_slider,
        ]:
            slider.observe(self._on_slider_change, names="value")

    def plot_grid(self):
        self.ax.clear()
        self.ax.coastlines(resolution="10m", linewidth=1)
        self.ax.add_feature(cfeature.LAND, facecolor="0.9")
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.5)

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

        lon_min, lon_max = float(self.grid.qlon.min()), float(self.grid.qlon.max())
        lat_min, lat_max = float(self.grid.qlat.min()), float(self.grid.qlat.max())
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        gl = self.ax.gridlines(draw_labels=True, linewidth=0, color="none")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}

        self._draw_scale_bar(lon_min, lon_max, lat_min, lat_max)

        self.fig.canvas.draw_idle()

    def _nice_scale_length(self, length_m):
        """
        Given a length in meters, return a 'nice' rounded value (1, 2, 5, 10, 20, 50, 100, etc.)
        in meters or kilometers.
        """
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
        """Draw a fixed-length scale bar with dynamic label using geometric calculation."""
        try:
            frac = 0.2
            bar_lat = lat_min + 0.05 * (lat_max - lat_min)
            bar_lon_start = lon_min + 0.05 * (lon_max - lon_min)
            bar_lon_end = bar_lon_start + frac * (lon_max - lon_min)

            # Convert to radians
            R = 6371000  # meters
            lat_rad = np.deg2rad(bar_lat)
            dlon_rad = np.deg2rad(bar_lon_end - bar_lon_start)
            bar_length_m = abs(dlon_rad * np.cos(lat_rad) * R)

            # Use the nice rounding function
            nice_length_m = self._nice_scale_length(bar_length_m)

            # Now, recalculate bar_lon_end so the bar matches the nice length
            nice_dlon_deg = np.rad2deg(nice_length_m / (np.cos(lat_rad) * R))
            bar_lon_end = bar_lon_start + nice_dlon_deg

            if nice_length_m >= 1000:
                label = f"{int(nice_length_m/1000)} km"
            else:
                label = f"{int(nice_length_m)} m"

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

    def save_grid(self, _btn=None):
        name = self._snapshot_name.value.strip()
        msg = self._commit_msg.value.strip()
        if not name:
            print("Enter a grid name!")
            return
        if not msg:
            print("Enter a grid message!")
            return

        if name.lower().endswith(".nc"):
            name = name[:-3]
        self.grid.name = name

        nc_path = os.path.join(self.grids_dir, f"grid_{name}.nc")
        self.grid.write_supergrid(nc_path)
        print(f"Saved grid '{os.path.basename(nc_path)}' in '{self.grids_dir}'.")
        self.refresh_commit_dropdown()
        return

    def load_grid(self, b=None):
        val = self._commit_dropdown.value
        if not val:
            return
        nc_path = os.path.join(self.grids_dir, val)
        try:
            self.grid = self.grid.from_supergrid(nc_path)
            self.sync_sliders_to_grid()
            self.construct_observances()
            self.plot_grid()
            print(f"Loaded grid from '{nc_path}'.")
        except Exception as e:
            print(f"Failed to load grid: {e}")
            import traceback

            traceback.print_exc()

    def sync_sliders_to_grid(self):
        try:
            # --- xstart ---
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

            # --- ystart ---
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

            # --- resolution ---
            res_min, res_max = 0.01, 1.0
            resolution_val = min(
                max(float(self.grid.lenx / self.grid.nx), res_min), res_max
            )

            # --- lenx ---
            lenx_min, lenx_max = 0.01, 50.0
            lenx_val = min(max(float(self.grid.lenx), lenx_min), lenx_max)

            # --- leny ---
            leny_min, leny_max = 0.01, 50.0
            leny_val = min(max(float(self.grid.leny), leny_min), leny_max)

            # Remove old observers
            for slider in [
                self._resolution_slider,
                self._xstart_slider,
                self._lenx_slider,
                self._ystart_slider,
                self._leny_slider,
            ]:
                slider.unobserve(self._on_slider_change, names="value")

        except Exception as e:
            print(f"Error in sync_sliders_to_grid: {e}")

    def _on_slider_change(self, change):
        from mom6_bathy.grid import Grid

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
        params = self._initial_params
        name = self._snapshot_name.value.strip() or params["name"]
        self.grid = Grid(
            lenx=params["lenx"],
            leny=params["leny"],
            resolution=params["resolution"],
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

    def refresh_commit_dropdown(self):
        # List all .nc files in the Grids directory (no subfolders)
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
                # Only show the name in the dropdown
                label = f"{name}"
                options.append((label, fname))
                if name == self.grid.name:
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
            # Format date: replace 'T' with ' ', trim to seconds
            date_short = date.replace("T", " ")
            date_short = date_short.split(".")[0] if "." in date_short else date_short
            nx = grid.nx
            ny = grid.ny
            details = (
                f"<b>Name:</b> {name}<br>"
                f"<b>Created:</b> {date_short}<br>"
                f"<b>nx:</b> {nx} <b>ny:</b> {ny}"
            )
            self._commit_details.value = details
        except Exception as e:
            self._commit_details.value = f"<b>Error:</b> {e}"
