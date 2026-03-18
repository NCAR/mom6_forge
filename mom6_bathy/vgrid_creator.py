import os
import xarray as xr
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import threading
from mom6_bathy.vgrid import VGrid


class VGridCreator(widgets.HBox):
    """
    Interactive creator for vertical grids (VGrid).
    Allows creation, editing, saving, and loading of vertical grid profiles.
    """

    def __init__(self, vgrid=None, repo_root=None, topo=None):
        self.repo_root = repo_root if repo_root is not None else os.getcwd()
        self.vgrids_dir = os.path.join(self.repo_root, "VGridLibrary")
        os.makedirs(self.vgrids_dir, exist_ok=True)

        self.topo = topo

        # Try to infer min_depth from topo if provided
        self.min_depth = 1.0
        if topo is not None and hasattr(topo, "min_depth"):
            self.min_depth = float(topo.min_depth)

        if vgrid is None:
            vgrid = VGrid.uniform(nk=10, depth=100.0)
        self.vgrid = vgrid
        self._initial_dz = np.copy(self.vgrid.dz)

        # Infer ratio and grid type from dz
        ratio_value, grid_type = self.infer_ratio_and_type(self.vgrid.dz)

        # --- Disable observers during setup ---
        self._observers_attached = False
        self.construct_control_panel(ratio_value=ratio_value, grid_type=grid_type)

        # Set widget values to match vgrid BEFORE attaching observers
        self._nk_slider.value = self.vgrid.nk
        self._depth_slider.value = float(self.vgrid.depth)
        self._ratio_slider.value = ratio_value
        self._type_toggle.value = grid_type

        self.plot_vgrid()
        super().__init__(
            [self._control_panel, self.fig.canvas],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )
        self.construct_observances()
        self._observers_attached = True
        self.refresh_commit_dropdown()

    @staticmethod
    def infer_ratio_and_type(dz, tol=1e-2):
        dz0 = dz[0]
        dzbot = dz[-1]
        ratio = dzbot / dz0 if dz0 != 0 else 1.0
        if np.isclose(ratio, 1.0, atol=tol):
            return 1.0, "Uniform"
        else:
            return ratio, "Hyperbolic"

    def construct_control_panel(self, ratio_value=1.0, grid_type="Uniform"):
        label_style = {"description_width": "120px"}

        self._nk_slider = widgets.IntSlider(
            value=self.vgrid.nk,
            min=2,
            max=100,
            step=1,
            description="Levels",
            layout={"width": "98%"},
            style=label_style,
        )
        self._depth_slider = widgets.FloatSlider(
            value=max(self.vgrid.depth, self.min_depth),
            min=1.0,
            max=10000.0,
            step=1.0,
            description="Depth (m)",
            layout={"width": "98%"},
            style=label_style,
        )
        self._warning_label = widgets.HTML(
            value="",
            layout={"width": "98%", "color": "red", "display": "none"},  # Start hidden
        )
        self._ratio_slider = widgets.FloatSlider(
            value=ratio_value,
            min=0.1,
            max=20.0,
            step=0.01,
            description="Top/Bottom Ratio:",
            layout={"width": "98%"},
            style=label_style,
        )
        self.ratio_help = widgets.HTML(
            value="<span style='font-size: 90%; color: #888;'>Ratio of bottom layer thickness to top layer thickness</span>",
            layout={"width": "98%", "display": "none"},
        )

        # Timer handle for hiding help
        self._ratio_help_timer = None

        def show_help(change=None):
            self.ratio_help.layout.display = "block"
            if self._ratio_help_timer is not None:
                self._ratio_help_timer.cancel()
            self._ratio_help_timer = threading.Timer(3.0, hide_help)
            self._ratio_help_timer.start()

        def hide_help():
            self.ratio_help.layout.display = "none"

        self._ratio_slider.observe(show_help, names="value")

        self._type_toggle = widgets.ToggleButtons(
            options=["Uniform", "Hyperbolic"],
            value=grid_type,
            description="Type",
            layout={"width": "98%"},
            style=label_style,
        )
        self._snapshot_name = widgets.Text(
            value="",
            placeholder="Enter vgrid name",
            description="Name:",
            layout={"width": "98%"},
            style=label_style,
        )
        self._commit_msg = widgets.Text(
            value="",
            placeholder="Enter vgrid message",
            description="Message:",
            layout={"width": "98%"},
            style=label_style,
        )
        self._commit_dropdown = widgets.Dropdown(
            options=[],
            description="VGrids:",
            layout={"width": "98%"},
            style=label_style,
        )
        self._commit_details = widgets.HTML(
            value="", layout={"width": "98%", "min_height": "2em"}
        )
        self._save_button = widgets.Button(
            description="Save VGrid", layout={"width": "49%"}
        )
        self._load_button = widgets.Button(
            description="Load VGrid", layout={"width": "49%"}
        )
        self._reset_button = widgets.Button(
            description="Reset", layout={"width": "98%"}, button_style="danger"
        )

        controls = widgets.VBox(
            [
                widgets.HTML("<h3>Vertical Grid Creator</h3>"),
                self._type_toggle,
                self._nk_slider,
                self._depth_slider,
                self._warning_label,
                self._ratio_slider,
                self.ratio_help,
                self._reset_button,
            ],
            layout=widgets.Layout(
                width="100%", align_items="stretch", overflow_y="visible"
            ),
        )

        library_section = widgets.VBox(
            [
                widgets.HTML("<h3>Library</h3>"),
                self._snapshot_name,
                self._commit_msg,
                self._commit_dropdown,
                self._commit_details,
                widgets.HBox(
                    [self._save_button, self._load_button], layout={"width": "100%"}
                ),
            ],
            layout={"width": "100%"},
        )

        self._control_panel = widgets.VBox(
            [
                controls,
                library_section,
            ],
            layout={"width": "35%", "height": "100%"},
        )

    def construct_observances(self):
        if getattr(self, "_observers_attached", False):
            return
        self._save_button.on_click(self.save_vgrid)
        self._load_button.on_click(self.load_vgrid)
        self._reset_button.on_click(self.reset_vgrid)
        self._snapshot_name.observe(
            lambda change: self.refresh_commit_dropdown(), names="value"
        )
        self._commit_dropdown.observe(self.update_commit_details, names="value")
        self._type_toggle.observe(self._on_param_change, names="value")
        self._nk_slider.observe(self._on_param_change, names="value")
        self._depth_slider.observe(self._on_param_change, names="value")
        self._ratio_slider.observe(self._on_param_change, names="value")

    def plot_vgrid(self):
        if not hasattr(self, "fig"):
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(6, 5))
            plt.ion()
        else:
            self.ax.clear()
        for depth in self.vgrid.zl:
            self.ax.axhline(y=depth, color="steelblue")
        self.ax.set_ylim(max(self.vgrid.zl) + 10, min(self.vgrid.zl) - 10)
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_title("Use the sliders to adjust vertical grid parameters.")
        self.fig.canvas.draw_idle()

    def _on_param_change(self, change):

        nk = self._nk_slider.value
        depth = self._depth_slider.value
        ratio = self._ratio_slider.value
        grid_type = self._type_toggle.value
        name = self.vgrid.name

        # Warn if depth < topo max depth
        topo_max = getattr(self, "min_depth", 1.0)
        if (
            hasattr(self, "topo")
            and self.topo is not None
            and hasattr(self.topo, "max_depth")
        ):
            topo_max = float(self.topo.max_depth)

        if depth < topo_max - 0.5:
            self._warning_label.value = f"<span style='color:red'>Warning: Depth is less than topo max depth ({topo_max:.2f} m)!</span>"
            self._warning_label.layout.display = "block"
        else:
            self._warning_label.value = ""
            self._warning_label.layout.display = "none"

        if grid_type == "Uniform":
            self.vgrid = VGrid.uniform(
                nk=nk,
                depth=depth,
                name=name,
            )
            self._ratio_slider.disabled = True
        else:
            self.vgrid = VGrid.hyperbolic(
                nk=nk,
                depth=depth,
                ratio=ratio,
                name=name,
            )
            self._ratio_slider.disabled = False
        self.plot_vgrid()

    def save_vgrid(self, _btn=None):
        name = self._snapshot_name.value.strip()
        msg = self._commit_msg.value.strip()
        if not name:
            print("Enter a vgrid name!")
            return
        if not msg:
            print("Enter a vgrid message!")
            return

        self.vgrid.name = name

        nc_path = os.path.join(self.vgrids_dir, f"vgrid_{name}.nc")
        self.vgrid.write(nc_path, message=msg)
        print(f"Saved vgrid '{os.path.basename(nc_path)}' in '{self.vgrids_dir}'.")
        self.refresh_commit_dropdown()
        return

    def load_vgrid(self, b=None):
        val = self._commit_dropdown.value
        if not val:
            print("No vgrid selected.")
            return
        nc_path = os.path.join(self.vgrids_dir, val)
        try:
            self.vgrid = VGrid.from_file(nc_path, name=None)
            # Infer ratio and grid type from dz
            ratio_value, grid_type = self.infer_ratio_and_type(self.vgrid.dz)
            # Temporarily remove observers to avoid recursion
            self._nk_slider.unobserve(self._on_param_change, names="value")
            self._depth_slider.unobserve(self._on_param_change, names="value")
            self._type_toggle.unobserve(self._on_param_change, names="value")
            self._ratio_slider.unobserve(self._on_param_change, names="value")

            self._nk_slider.value = self.vgrid.nk
            self._depth_slider.value = float(self.vgrid.depth)
            self._type_toggle.value = grid_type
            self._ratio_slider.value = ratio_value

            self._nk_slider.observe(self._on_param_change, names="value")
            self._depth_slider.observe(self._on_param_change, names="value")
            self._type_toggle.observe(self._on_param_change, names="value")
            self._ratio_slider.observe(self._on_param_change, names="value")

            self.plot_vgrid()
            print(f"Loaded vgrid from '{nc_path}'.")
        except Exception as e:
            print(f"Failed to load vgrid: {e}")

    def reset_vgrid(self, b=None):
        self.vgrid = VGrid(self._initial_dz.copy())
        ratio_value, grid_type = self.infer_ratio_and_type(self.vgrid.dz)
        self._nk_slider.value = self.vgrid.nk
        self._depth_slider.value = float(self.vgrid.depth)
        self._type_toggle.value = grid_type
        self._ratio_slider.value = ratio_value
        self.plot_vgrid()

    def refresh_commit_dropdown(self):
        # List all .nc files in the VGrids directory (no subfolders)
        vgrid_nc_files = [
            fname
            for fname in os.listdir(self.vgrids_dir)
            if fname.startswith("vgrid_") and fname.endswith(".nc")
        ]
        options = []
        for fname in vgrid_nc_files:
            abs_path = os.path.join(self.vgrids_dir, fname)
            try:
                ds = xr.open_dataset(abs_path)
                name = ds.attrs.get("title", fname)
                # Only show the name in the dropdown
                label = f"{fname[len('vgrid_'):-3]}"  # Remove prefix and .nc
                options.append((label, fname))
            except Exception:
                continue

        options.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.vgrids_dir, x[1])),
            reverse=True,
        )
        self._commit_dropdown.options = options if options else []
        if options:
            option_values = [v for (l, v) in options]
            if self._commit_dropdown.value not in option_values:
                self._commit_dropdown.value = options[0][1]
        else:
            self._commit_dropdown.value = None
        self.update_commit_details()

    def update_commit_details(self, change=None):
        val = self._commit_dropdown.value
        if not val:
            self._commit_details.value = ""
            return
        abs_path = os.path.join(self.vgrids_dir, val)
        try:
            ds = xr.open_dataset(abs_path)
            name = ds.attrs.get("title", "")
            date = ds.attrs.get("date_created", ds.attrs.get("history", ""))
            # Format date: replace 'T' with ' ', trim to seconds
            date_short = date.replace("T", " ")
            date_short = date_short.split(".")[0] if "." in date_short else date_short
            depth = ds.attrs.get("maximum_depth", "")
            nk = len(ds["dz"]) if "dz" in ds else ""
            details = (
                f"<b>Name:</b> {name}<br>"
                f"<b>Date:</b> {date_short}<br>"
                f"<b>Depth:</b> {depth} m<br>"
                f"<b>nk:</b> {nk}"
            )
            self._commit_details.value = details
        except Exception:
            self._commit_details.value = ""
