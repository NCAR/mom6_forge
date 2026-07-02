# All cell indices in this file are in (j, i) order to match (y, x) ordering

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from mom6_forge.edit_command import *
from mom6_forge.git_utils import *
from matplotlib.widgets import RectangleSelector


class TopoEditor(widgets.HBox):
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, topo, build_ui=True):
        self.topo = topo
        self.ny = self.topo.masked_depth.data.shape[0]
        self.nx = self.topo.masked_depth.data.shape[1]
        self._selected_cell = None
        self.build_ui = build_ui

        # --- Command Manager ---
        if self.has_version_control:
            self.current_branch = self.topo.tcm.get_current_branch()
            self._original_depth = np.array(self.topo.masked_depth.data)
            self._original_min_depth = self.topo.min_depth
        self._selected_cells = []

        # --- Build UI controls, plot, and observers ---
        self.construct_control_panel()
        self.construct_interactive_plot()
        self.construct_observances()
        self.update_undo_redo_buttons()
        self.trigger_refresh()

        # --- Initialize the widget layout ---
        super().__init__([self._control_panel, self._interactive_plot])

    def construct_control_panel(self):
        """
        Construct the control panel widgets for the topography editor.

        This includes controls for display mode, cell editing, undo/redo,
        snapshots, and git/domain management. The controls are grouped
        into logical sections for clarity.
        """
        # --- Display and global settings ---
        self._min_depth_specifier = widgets.BoundedFloatText(
            value=self.topo.min_depth,
            min=-1000.0,
            max=float(np.nanmax(self.topo.masked_depth.data)),
            step=10.0,
            description="Min depth (m):",
            disabled=False,
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )
        self._display_mode_toggle = widgets.ToggleButtons(
            options=["depth", "mask", "basinmask"],
            description="Field:",
            disabled=False,
            tooltips=["Display depth values", "Display mask values", "Display Basins"],
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "40px", "button_width": "85px"},
        )

        # --- Cell editing widgets ---
        self._selected_cell_label = widgets.Label(
            "Selected cell: None (double click to select a cell)."
        )
        self._rect_or_single_select_button = widgets.ToggleButtons(
            options=["Single Cell", "Rectangular Area"],
            description="Selection Mode:",
            disabled=False,
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )
        self._clear_selection_button = widgets.Button(
            description="Clear Selection",
            disabled=True,
            button_style="warning",
            layout={"width": "80%"},
        )
        self._depth_specifier = widgets.FloatText(
            value=None,
            step=10.0,
            description="Depth (m):",
            disabled=True,
            placeholder="Select a cell first.",
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )
        self._set_to_mean_button = widgets.Button(
            description="Mean", disabled=True, layout={"width": "30%"}
        )
        self._set_to_max_button = widgets.Button(
            description="Max", disabled=True, layout={"width": "30%"}
        )
        self._set_to_min_button = widgets.Button(
            description="Min", disabled=True, layout={"width": "30%"}
        )

        self._mask_specifier = widgets.ToggleButtons(
            value=None,
            options=["Land", "Ocean"],
            description="Mask:",
            disabled=True,
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )
        self._clear_user_mask_button = widgets.Button(
            description="Clear Manual Mask",
            disabled=True,
            button_style="warning",
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )

        # --- Basin editing widgets ---
        self._basin_specifier_toggle = widgets.Button(
            description="Erase Disconnected Basins",
            disabled=True,
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "100px"},
        )
        self._basin_specifier_delete_selected = widgets.Button(
            description="Erase Selected Basin",
            disabled=True,
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "100px"},
        )
        self._basin_specifier = widgets.Label(
            value="Basin Label Number: None",
            layout={"width": "80%"},
            style={"description_width": "auto"},
        )

        # --- Undo/Redo/Reset ---
        self._undo_button = widgets.Button(
            description="Undo", disabled=True, layout={"width": "44%"}
        )
        self._redo_button = widgets.Button(
            description="Redo", disabled=True, layout={"width": "44%"}
        )
        self._reset_button = widgets.Button(
            description="Reset", layout={"width": "44%"}, button_style="danger"
        )

        if self.has_version_control:
            # --- Snapshot controls ---
            self._tag_name = widgets.Text(
                value="",
                placeholder="Enter tag name",
                description="Name:",
                layout={"width": "90%"},
            )
            self._save_button = widgets.Button(
                description="Save Tag", layout={"width": "44%"}
            )

            self._git_branch_name = widgets.Text(
                value="",
                placeholder="New branch name",
                description="Branch:",
                layout={"width": "90%"},
            )
            self._git_create_branch_button = widgets.Button(
                description="Create Branch", layout={"width": "44%"}
            )
            self._git_branch_dropdown = widgets.Dropdown(
                options=self.topo.tcm.list_branches(),
                description="Checkout:",
                layout={"width": "90%"},
            )
            self._git_checkout_button = widgets.Button(
                description="Checkout", layout={"width": "44%"}
            )
        # --- Group controls into logical sections ---
        self.display_section = widgets.VBox(
            [
                widgets.HTML("<h3>Display</h3>"),
                self._display_mode_toggle,
            ]
        )
        self.global_settings_section = widgets.VBox(
            [
                widgets.HTML("<h3>Global Settings</h3>"),
                self._min_depth_specifier,
            ]
        )
        cell_editing_section_children = [
            widgets.HTML("<h3>Cell Editing</h3>"),
            self._rect_or_single_select_button,
            self._clear_selection_button,
            self._selected_cell_label,
            self._depth_specifier,
            self._mask_specifier,
            self._clear_user_mask_button,
        ]

        # Only add stats section if statistics are available
        has_stats = self.topo.stats is not None
        if has_stats:
            cell_editing_section_children.extend(
                [
                    widgets.HTML(
                        "<p style='margin: 5px 0; font-size: 12px;'>Set to statistic:</p>"
                    ),
                    widgets.HBox(
                        [
                            self._set_to_mean_button,
                            self._set_to_max_button,
                            self._set_to_min_button,
                        ],
                        layout={"justify_content": "space-between"},
                    ),
                ]
            )

        self.cell_editing_section = widgets.VBox(cell_editing_section_children)
        self.basin_section = widgets.VBox(
            [
                widgets.HTML("<h3>Basin Selector</h3>"),
                self._basin_specifier,
                self._basin_specifier_toggle,
                self._basin_specifier_delete_selected,
            ]
        )
        self.history_section = widgets.VBox(
            [
                widgets.HTML("<h3>Edit History</h3>"),
                widgets.HBox(
                    [self._undo_button, self._redo_button, self._reset_button]
                ),
            ]
        )
        if self.has_version_control:
            self.git_section = widgets.VBox(
                [
                    # Domain controls
                    widgets.HTML("<hr>"),
                    # Snapshot controls
                    self._tag_name,
                    widgets.HBox([self._save_button]),
                    widgets.HTML("<hr>"),
                    # Git controls
                    self._git_branch_name,
                    widgets.HBox([self._git_create_branch_button]),
                    self._git_branch_dropdown,
                    self._git_checkout_button,
                ]
            )

        # --- Layout: always-visible controls and advanced accordions ---
        main_panel = [
            self.display_section,
            self.global_settings_section,
            self.cell_editing_section,
            self.basin_section,
        ]
        if self.has_version_control:
            main_panel.append(self.history_section)
        self.main_controls = widgets.VBox(main_panel)

        # --- Combine everything into the control panel ---
        cp = [widgets.HTML("<h2>Topo Editor</h2>"), self.main_controls]
        if self.has_version_control:
            git_accordion = widgets.Accordion(children=[self.git_section])
            git_accordion.set_title(0, "Git Version Control")
            git_accordion.selected_index = None  # collapsed by default
            cp.append(git_accordion)
            # Set the current branch in the dropdown if available
            current_branch = self.topo.tcm.get_current_branch()
            if current_branch in self._git_branch_dropdown.options:
                self._git_branch_dropdown.value
        self._control_panel = widgets.VBox(
            cp,
            layout={"width": "30%", "height": "100%", "overflow_y": "auto"},
        )

    def construct_interactive_plot(self):
        """
        Construct the interactive matplotlib plot for the topography editor.

        This sets up the main map display, colorbar, and coordinate formatting.
        The plot is embedded in a widget for use in the Jupyter interface.
        """
        # Close any existing figure to avoid memory leaks
        if hasattr(self, "fig") and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        plt.ioff()  # Turn off interactive mode for setup

        # Create the figure and axis
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self.ax.set_aspect("auto")

        # Custom coordinate formatter for mouse hover
        def format_coord(x, y):
            j, i = self.topo._grid.get_indices(y, x)
            return f"x={x:.2f}, y={y:.2f}, i={i}, j={j} depth={self.topo.masked_depth.data[j, i]:.2f}"

        self.ax.format_coord = format_coord

        # Set up colormap and plot the depth field
        self.cmap = plt.get_cmap("viridis")
        self.cmap.set_under("w")
        self.im = self.ax.pcolormesh(
            self.topo._grid.qlon.data,
            self.topo._grid.qlat.data,
            self.topo.masked_depth.data,
            vmin=self.topo.min_depth,
            cmap=self.cmap,
            transform=ccrs.PlateCarree(),
        )

        # Axis labels and title
        self.ax.set_title("Double click on a cell to change its depth.")
        self.ax.set_xlabel(
            f'x ({self.topo._grid.qlon.attrs.get("units", "degrees_east")})'
        )
        self.ax.set_ylabel(
            f'y ({self.topo._grid.qlat.attrs.get("units", "degrees_north")})'
        )

        # Add colorbar for depth
        self.cbar = self.fig.colorbar(
            self.im, ax=self.ax, orientation="vertical", pad=0.02
        )
        self.cbar.set_label(f"Depth ({self.topo.masked_depth.units})")
        self.cbar.set_ticks(MaxNLocator(integer=True))

        # Enable toolbar and layout
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = "top"
        self.fig.tight_layout()
        plt.ion()  # Restore interactive mode

        # Wrap the figure in a widget for display

        if self.build_ui:
            self._interactive_plot = widgets.HBox(
                children=(self.fig.canvas,), layout={"border_left": "1px solid grey"}
            )
        else:
            self._interactive_plot = widgets.VBox([])
        self._rect_selector = RectangleSelector(
            self.ax,
            self._on_rect_select,
            # useblit=False: the editor does plenty of full canvas redraws of
            # its own (draw_idle on every selection/edit/mode change) that
            # aren't coordinated with the selector's blit background cache,
            # which goes stale and makes the rectangle fill vanish.
            useblit=False,
            button=[1],
            interactive=True,
            props=dict(edgecolor="red", facecolor="red", alpha=0.2, fill=True),
        )
        self._rect_selector.set_active(False)  # off by default

    def construct_observances(self):
        """Attach event observers and callbacks to all interactive widgets and plot elements."""
        # Display mode toggle
        self._display_mode_toggle.observe(
            self.refresh_display_mode, names="value", type="change"
        )

        # Double click event for cell selection on the plot
        self.fig.canvas.mpl_connect("button_press_event", self.on_double_click)
        self._rect_or_single_select_button.observe(
            self._on_rect_or_single_select_toggle, names="value"
        )
        self._clear_selection_button.on_click(self._on_clear_selection)

        # Min depth change observer
        self._min_depth_specifier.observe(
            self.on_min_depth_change, names="value", type="change"
        )

        # Basin erase buttons
        self._basin_specifier_toggle.on_click(self.erase_disconnected_basin)
        self._basin_specifier_delete_selected.on_click(self.erase_selected_basin)

        # Depth change observer for selected cell
        self._depth_specifier.observe(
            self.on_depth_change, names="value", type="change"
        )

        # Mask change observer for selected cell
        self._mask_specifier.observe(self.on_mask_change, names="value", type="change")
        self._clear_user_mask_button.on_click(self.clear_user_mask)
        # Statistic buttons
        self._set_to_mean_button.on_click(self.set_depth_to_mean)
        self._set_to_max_button.on_click(self.set_depth_to_max)
        self._set_to_min_button.on_click(self.set_depth_to_min)

        if self.has_version_control:
            # Undo/Redo/Reset buttons
            self._undo_button.on_click(self.undo_last_edit)
            self._redo_button.on_click(self.redo_last_edit)
            self._reset_button.on_click(self.reset)

            # Snapshot controls
            self._save_button.on_click(self.on_tag)

            # Git/domain controls
            self._git_create_branch_button.on_click(self.on_git_create_branch)
            self._git_checkout_button.on_click(self.on_git_checkout)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_version_control(self):
        """Check if the topo's has version control."""
        return self.topo.has_version_control

    @property
    def active_cells(self):
        if hasattr(self, "_selected_cells") and self._selected_cells:
            return list(self._selected_cells)
        elif self._selected_cell is not None:
            j, i, *_ = self._selected_cell
            return [(j, i)]
        return []

    # ------------------------------------------------------------------
    # Display / refresh
    # ------------------------------------------------------------------

    def refresh_display_mode(self, change):
        """Refresh the display mode of the topography plot based on the selected mode."""
        mode = change["new"]
        self._depth_specifier.layout.display = "flex" if mode == "depth" else "none"
        self._mask_specifier.layout.display = "flex" if mode == "mask" else "none"
        self._clear_user_mask_button.layout.display = (
            "flex" if mode == "mask" else "none"
        )
        self.basin_section.layout.display = "flex" if mode == "basinmask" else "none"
        if mode == "depth":
            self.im.set_clim(
                vmin=self.topo.min_depth,
                vmax=float(np.nanmax(self.topo.masked_depth.data)),
            )
            self.im.set_array(self.topo.masked_depth.data)
            self.im.set_clim(
                vmin=self.topo.min_depth,
                vmax=float(np.nanmax(self.topo.masked_depth.data)),
            )  # For some reason, this needs to be set twice to get the correct minimum bound
            self.cbar.set_label(f"Depth ({self.topo.masked_depth.units})")
        elif mode == "mask":
            self.im.set_array(self.topo.tmask.data)
            self.im.set_clim((0, 1))
            self.cbar.set_label("Land Mask")
        elif mode == "basinmask":
            self.im.set_array(self.topo.basintmask.data)
            self.im.set_clim((0, self.topo.basintmask.data.max()))
            self.cbar.set_label("Basin Mask")
        else:
            raise ValueError(f"Unknown display mode: {mode}")
        self.fig.canvas.draw_idle()

    def trigger_refresh(self):
        """Trigger a refresh of the interactive plot and min depth specifier."""
        self.refresh_display_mode({"new": self._display_mode_toggle.value})
        self._min_depth_specifier.value = self.topo.min_depth
        self.update_undo_redo_buttons()

    # ------------------------------------------------------------------
    # Cell selection
    # ------------------------------------------------------------------

    def _select_cell(self, i, j):
        """Select a cell in the topography grid and update the UI accordingly."""
        # Remove old patch if it exists
        if (
            self._selected_cell is not None
            and len(self._selected_cell) > 2
            and self._selected_cell[2] is not None
            and hasattr(self, "ax")
        ):
            try:
                self._selected_cell[2].remove()
            except Exception:
                pass

        polygon = None
        if hasattr(self, "ax"):
            try:
                qlon = self.topo._grid.qlon.data
                qlat = self.topo._grid.qlat.data
                if (j + 1 < qlon.shape[0]) and (i + 1 < qlon.shape[1]):
                    vertices = np.array(
                        [
                            [qlon[j, i], qlat[j, i]],
                            [qlon[j, i + 1], qlat[j, i + 1]],
                            [qlon[j + 1, i + 1], qlat[j + 1, i + 1]],
                            [qlon[j + 1, i], qlat[j + 1, i]],
                        ]
                    )
                    polygon = patches.Polygon(
                        vertices,
                        edgecolor="r",
                        facecolor="none",
                        alpha=0.8,
                        linewidth=2,
                        label="Selected cell",
                        transform=ccrs.PlateCarree(),
                    )
                    self.ax.add_patch(polygon)
                    self.fig.canvas.draw_idle()
            except Exception as e:
                print(f"Failed to draw polygon patch: {e}")

        self._selected_cell = (j, i, polygon)

        # UI updates
        if hasattr(self, "_selected_cell_label"):
            self._selected_cell_label.value = f"Selected cell: {i}, {j}"
        if hasattr(self, "_depth_specifier"):
            self._depth_specifier.disabled = False
            self._depth_specifier.value = self.topo.depth.data[j, i]
        if hasattr(self, "_mask_specifier"):
            self._mask_specifier.disabled = False
            self._mask_specifier.value = (
                "Ocean" if self.topo.tmask.data[j, i] == 1 else "Land"
            )
        if hasattr(self, "_clear_user_mask_button"):
            if self.topo._user_mask is not None:
                self._clear_user_mask_button.disabled = False
            else:
                self._clear_user_mask_button.disabled = True

        # Enable statistic buttons and show values if statistics are available
        has_stats = self.topo.stats is not None
        for btn, stat_name, label in [
            (self._set_to_mean_button, "D_mean", "Mean"),
            (self._set_to_max_button, "D_max", "Max"),
            (self._set_to_min_button, "D_min", "Min"),
        ]:
            btn.disabled = not has_stats
            if has_stats:
                val = self._get_statistic_value(stat_name)
                btn.description = (
                    f"{label}: {val:.1f}m"
                    if val is not None and np.isfinite(val)
                    else label
                )
            else:
                btn.description = label

        if hasattr(self, "_basin_specifier"):
            label = self.topo.basintmask.data[j, i]
            self._basin_specifier.value = f"Basin Label Number: {str(label)}"
            if hasattr(self, "_basin_specifier_toggle") and hasattr(
                self, "_basin_specifier_delete_selected"
            ):
                if label != 0:
                    self._basin_specifier_toggle.disabled = False
                    self._basin_specifier_delete_selected.disabled = False
                else:
                    self._basin_specifier_toggle.disabled = True
                    self._basin_specifier_delete_selected.disabled = True

    def on_double_click(self, event):
        """Handle double-click events on the plot to select a cell."""
        if (
            self._rect_or_single_select_button.value == "Rectangular Area"
        ):  # rectangle mode active, ignore double clicks
            return
        if event.dblclick and event.xdata is not None and event.ydata is not None:
            # Convert lon/lat to grid indices
            j, i = self.topo._grid.get_indices(event.ydata, event.xdata)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self._select_cell(i, j)
        self._clear_selection_button.disabled = False

    def _on_rect_select(self, eclick, erelease):
        self._clear_selection_button.disabled = False
        lon_min, lon_max = sorted([eclick.xdata, erelease.xdata])
        lat_min, lat_max = sorted([eclick.ydata, erelease.ydata])
        lon_min = (lon_min + 360) % 360
        lon_max = (lon_max + 360) % 360
        tlon = (self.topo._grid.tlon.data + 360) % 360  # normalize to [0, 360]
        tlat = self.topo._grid.tlat.data

        if lon_min <= lon_max:
            lon_mask = (tlon >= lon_min) & (tlon <= lon_max)
        else:  # selection crosses the 0/360 boundary after normalization
            lon_mask = (tlon >= lon_min) | (tlon <= lon_max)

        mask = lon_mask & (tlat >= lat_min) & (tlat <= lat_max)
        self._selected_cells = list(zip(*np.where(mask)))

        n = len(self._selected_cells)
        self._selected_cell_label.value = (
            f"Selected cells: n = {n}"
            if n > 0
            else f"Selected cell: None (draw box to select cells)."
        )
        self._depth_specifier.disabled = False
        self._mask_specifier.disabled = False

        # Enable statistic buttons if statistics are available (no single
        # value to preview across multiple cells, so just show the label)
        has_stats = self.topo.stats is not None
        for btn in (
            self._set_to_mean_button,
            self._set_to_max_button,
            self._set_to_min_button,
        ):
            btn.disabled = not (has_stats and n > 0)

    def _on_rect_or_single_select_toggle(self, change):
        if change["new"] == "Rectangular Area":  # rectangle mode ON
            # Deactivate matplotlib toolbar zoom mode if it's active
            toolbar = getattr(self.fig.canvas, "toolbar", None)
            if (
                toolbar is not None
                and hasattr(toolbar, "mode")
                and "zoom" in str(toolbar.mode).lower()
            ):
                toolbar.zoom()
            # Clear the single-cell selection entirely
            if self._selected_cell is not None and self._selected_cell[2] is not None:
                try:
                    self._selected_cell[2].remove()
                except Exception:
                    pass
            self._selected_cell = None
            self._reset_statistic_buttons()
            self._rect_selector.set_active(True)
            self.fig.canvas.draw_idle()  # force redraw so the selector patch appears
            self._clear_selection_button.disabled = True
        else:  # rectangle mode OFF (Single Cell)
            self._rect_selector.set_active(False)
            self._rect_selector.clear()  # removes the drawn box
            self.fig.canvas.draw_idle()  # force redraw
            self._selected_cells = []
            self._selected_cell_label.value = (
                "Selected cell: None (double click to select a cell)."
            )
            self._depth_specifier.disabled = True
            self._mask_specifier.disabled = True
            self._clear_selection_button.disabled = True
            self._reset_statistic_buttons()

    # ------------------------------------------------------------------
    # Cell deselection
    # ------------------------------------------------------------------

    def _on_clear_selection(self, button_instance):
        """Clear the current selection of cells."""
        if self._rect_or_single_select_button.value != "Rectangular Area":
            self._selected_cell = None
            self._selected_cell_label.value = "Selected cell: None "
            # Remove any existing polygon patches from the plot
            if hasattr(self, "ax"):
                for patch in list(self.ax.patches):
                    patch.remove()
        else:
            self._selected_cells = []
            self._rect_selector.clear()  # removes the drawn box
        self._depth_specifier.disabled = True
        self._mask_specifier.disabled = True
        self._clear_selection_button.disabled = True
        self._reset_statistic_buttons()
        self.fig.canvas.draw_idle()

    def _reset_statistic_buttons(self):
        """Disable the statistic buttons and clear their value labels."""
        for btn, label in [
            (self._set_to_mean_button, "Mean"),
            (self._set_to_max_button, "Max"),
            (self._set_to_min_button, "Min"),
        ]:
            btn.disabled = True
            btn.description = label

    # ------------------------------------------------------------------
    # Setters (depth / mask edits)
    # ------------------------------------------------------------------

    def on_depth_change(self, change):
        """Handle changes to the depth specifier for the selected cell."""
        if not self.active_cells:
            return
        cells = self.active_cells
        old_values = [self.topo.depth.data[j, i] for j, i in cells]
        new_val = change["new"]
        if all(v == new_val for v in old_values):
            return
        cmd = DepthEditCommand(
            self.topo, cells, [new_val] * len(cells), old_values=old_values
        )
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

    def on_mask_change(self, change):
        if not self.active_cells:
            return
        cells = self.active_cells
        mask_map = {"Land": 0, "Ocean": 1}
        new_val = mask_map[change["new"]]
        old_values = [self.topo.tmask.data[j, i] for j, i in cells]

        if all(v == new_val for v in old_values):
            return

        cmd = MaskEditCommand(
            self.topo,
            [(j, i) for j, i in cells],
            [new_val] * len(cells),
            old_values=old_values,
        )
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

    def on_min_depth_change(self, change):
        """Handle changes to the minimum depth specifier."""
        old_val = self.topo.min_depth
        new_val = change["new"]
        if old_val != new_val:
            cmd = MinDepthEditCommand(
                self.topo, attr="min_depth", new_value=new_val, old_value=old_val
            )
            self.apply_edit(cmd)
            self.update_undo_redo_buttons()

    def clear_user_mask(self, b):
        """Clear the manual mask if it exists."""
        if self.topo._user_mask is not None:
            self.topo.clear_user_mask()
            self.update_undo_redo_buttons()
            self.trigger_refresh()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _get_statistic_value(self, stat_name):
        """Get a statistic value for the selected cell."""
        if self._selected_cell is None or self.topo.stats is None:
            return None

        j, i, _ = self._selected_cell
        ds = self.topo.stats

        if ds is None or stat_name not in ds.data_vars:
            return None

        return float(ds[stat_name].data[j, i])

    def _apply_statistic(self, stat_name):
        if self.topo.stats is None:
            return
        ds = self.topo.stats
        if stat_name not in ds.data_vars:
            return
        cells = self.active_cells
        if not cells:
            return
        valid = [
            (j, i, float(ds[stat_name].data[j, i]))
            for j, i in cells
            if np.isfinite(ds[stat_name].data[j, i])
        ]
        if not valid:
            return
        valid_cells = [(j, i) for j, i, _ in valid]
        new_values = [v for _, _, v in valid]
        old_values = [self.topo.depth.data[j, i] for j, i in valid_cells]
        cmd = DepthEditCommand(
            self.topo, valid_cells, new_values, old_values=old_values
        )
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

    def set_depth_to_mean(self, b):
        self._apply_statistic("D_mean")

    def set_depth_to_max(self, b):
        self._apply_statistic("D_max")

    def set_depth_to_min(self, b):
        self._apply_statistic("D_min")

    # ------------------------------------------------------------------
    # Basin editing
    # ------------------------------------------------------------------

    def erase_disconnected_basin(self, b):
        """Erase all disconnected basins in the topography."""
        if self._selected_cell is None:
            return
        j, i, _ = self._selected_cell
        self.topo.erase_disconnected_basin(i, j)
        self.update_undo_redo_buttons()

    def erase_selected_basin(self, b):
        """Erase the basin associated with the currently selected cell."""
        if self._selected_cell is None:
            return
        j, i, _ = self._selected_cell
        self.topo.erase_selected_basin(i, j)
        self.update_undo_redo_buttons()

    # ------------------------------------------------------------------
    # Undo / redo / history
    # ------------------------------------------------------------------

    def apply_edit(self, cmd):
        """Apply an edit command, update the UI, and autosave the working state."""
        self.topo.apply_edit(cmd)
        self.trigger_refresh()

    def undo_last_edit(self, b=None):
        """Undo the last edit command and update the UI."""
        assert self.topo.tcm.undo()
        self.trigger_refresh()

    def redo_last_edit(self, b=None):
        """Redo the last undone edit command and update the UI."""
        assert self.topo.tcm.redo()
        self.trigger_refresh()

    def reset(self, change):
        """Reset the topo to its original state and update the UI."""
        self.topo.tcm.reset()
        self.trigger_refresh()

    def update_undo_redo_buttons(self):
        """Enable or disable the undo/redo buttons based on command history."""
        if not self.has_version_control:
            return
        if hasattr(self, "_undo_button"):
            self._undo_button.disabled = not self.topo.tcm.undo(check_only=True)
        if hasattr(self, "_redo_button"):
            self._redo_button.disabled = not self.topo.tcm.redo(check_only=True)

    # ------------------------------------------------------------------
    # Git / version control
    # ------------------------------------------------------------------

    def on_tag(self, _btn=None):
        """Save the current state as a snapshot and commit it to the repository."""
        name = self._tag_name.value.strip()
        if not name:
            print("Enter a snapshot name!")
            return

        self.topo.tcm.tag(name)  # TODO: Save a tag!
        print(f"Saved tag '{name}'.")
        return

    def on_git_create_branch(self, b):
        """Create a new git branch"""
        name = self._git_branch_name.value.strip()
        if not name:
            print("Please enter a branch name.")
            return
        try:
            branch = self.topo.tcm.create_branch(name)
            self._git_branch_dropdown.options = self.topo.tcm.list_branches()
            self._git_branch_dropdown.value = self.topo.tcm.get_current_branch()
        except Exception as e:
            print(f"Error creating branch: {str(e)}")

    def on_git_checkout(self, b):
        """Checkout the specified git branch."""
        target = self._git_branch_dropdown.value
        if not target:
            print("Please select a branch to checkout.")
            return
        try:
            self.topo.tcm.checkout(target)
            print(f"Checked out to branch '{target}'.")

            # Update branch dropdowns
            self._git_branch_dropdown.options = self.topo.tcm.list_branches()
            self._git_branch_dropdown.value = self.topo.tcm.get_current_branch()

            self.trigger_refresh()
        except Exception as e:
            print(f"Error checking out branch '{target}' with error {e}.")
