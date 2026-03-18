import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import ipywidgets as widgets
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
from mom6_bathy.edit_command import *
from mom6_bathy.git_utils import *


class TopoEditor(widgets.HBox):
    def __init__(self, topo, build_ui=True):
        self.topo = topo
        self.ny = self.topo.depth.data.shape[0]
        self.nx = self.topo.depth.data.shape[1]

        # --- Command Manager ---
        self.current_branch = self.topo.tcm.get_current_branch()
        self._selected_cell = None
        self._original_depth = np.array(self.topo.depth.data)
        self._original_min_depth = self.topo.min_depth

        # --- Build UI controls, plot, and observers ---
        self.construct_control_panel()
        self.construct_interactive_plot()
        self.construct_observances()
        self.update_undo_redo_buttons()

        # --- Initialize the widget layout ---
        super().__init__([self._control_panel, self._interactive_plot])

    def apply_edit(self, cmd):
        """Apply an edit command, update the UI, and autosave the working state."""
        self.topo.tcm.execute(cmd)
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
        if hasattr(self, "_undo_button"):
            self._undo_button.disabled = not self.topo.tcm.undo(check_only=True)
        if hasattr(self, "_redo_button"):
            self._redo_button.disabled = not self.topo.tcm.redo(check_only=True)

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

        # Create the figure and axis with Cartopy projection
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set map extent based on grid longitude/latitude
        lon_min = float(self.topo._grid.qlon.data.min())
        lon_max = float(self.topo._grid.qlon.data.max())
        lat_min = float(self.topo._grid.qlat.data.min())
        lat_max = float(self.topo._grid.qlat.data.max())
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Custom coordinate formatter for mouse hover
        def format_coord(x, y):
            j, i = self.topo._grid.get_indices(y, x)
            return f"x={x:.2f}, y={y:.2f}, i={i}, j={j} depth={self.topo.depth.data[j, i]:.2f}"

        self.ax.format_coord = format_coord

        # Set up colormap and plot the depth field
        self.cmap = plt.get_cmap("viridis")
        self.cmap.set_under("w")
        self.im = self.ax.pcolormesh(
            self.topo._grid.qlon.data,
            self.topo._grid.qlat.data,
            self.topo.depth.data,
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
        self.cbar.set_label(f"Depth ({self.topo.depth.units})")
        self.cbar.set_ticks(MaxNLocator(integer=True))

        # Enable toolbar and layout
        self.fig.canvas.toolbar_visible = True
        self.fig.canvas.toolbar_position = "top"
        self.fig.tight_layout()
        plt.ion()  # Restore interactive mode

        # Wrap the figure in a widget for display
        self._interactive_plot = widgets.HBox(
            children=(self.fig.canvas,), layout={"border_left": "1px solid grey"}
        )

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
            max=float(np.nanmax(self.topo.depth.data)),
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
        self._depth_specifier = widgets.FloatText(
            value=None,
            step=10.0,
            description="Depth (m):",
            disabled=True,
            placeholder="Select a cell first.",
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
        display_section = widgets.VBox(
            [
                widgets.HTML("<h3>Display</h3>"),
                self._display_mode_toggle,
            ]
        )
        global_settings_section = widgets.VBox(
            [
                widgets.HTML("<h3>Global Settings</h3>"),
                self._min_depth_specifier,
            ]
        )
        cell_editing_section = widgets.VBox(
            [
                widgets.HTML("<h3>Cell Editing</h3>"),
                self._selected_cell_label,
                self._depth_specifier,
            ]
        )
        basin_section = widgets.VBox(
            [
                widgets.HTML("<h3>Basin Selector</h3>"),
                self._basin_specifier,
                self._basin_specifier_toggle,
                self._basin_specifier_delete_selected,
            ]
        )
        history_section = widgets.VBox(
            [
                widgets.HTML("<h3>Edit History</h3>"),
                widgets.HBox(
                    [self._undo_button, self._redo_button, self._reset_button]
                ),
            ]
        )
        git_section = widgets.VBox(
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
        main_controls = widgets.VBox(
            [
                display_section,
                global_settings_section,
                cell_editing_section,
                basin_section,
                history_section,
            ]
        )
        git_accordion = widgets.Accordion(children=[git_section])
        git_accordion.set_title(0, "Git Version Control")
        git_accordion.selected_index = None  # collapsed by default

        # --- Combine everything into the control panel ---
        self._control_panel = widgets.VBox(
            [
                widgets.HTML("<h2>Topo Editor</h2>"),
                main_controls,
                git_accordion,
            ],
            layout={"width": "30%", "height": "100%", "overflow_y": "auto"},
        )

        # Set the current branch in the dropdown if available
        current_branch = self.topo.tcm.get_current_branch()
        if current_branch in self._git_branch_dropdown.options:
            self._git_branch_dropdown.value

    def refresh_display_mode(self, change):
        """Refresh the display mode of the topography plot based on the selected mode."""
        mode = change["new"]
        if mode == "depth":
            self.im.set_clim(
                vmin=self.topo.min_depth, vmax=float(np.nanmax(self.topo.depth.data))
            )
            self.im.set_array(self.topo.depth.data)
            self.im.set_clim(
                vmin=self.topo.min_depth, vmax=float(np.nanmax(self.topo.depth.data))
            )  # For some reason, this needs to be set twice to get the correct minimum bound
            self.cbar.set_label(f"Depth ({self.topo.depth.units})")
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

        self._selected_cell = (i, j, polygon)

        # UI updates
        if hasattr(self, "_selected_cell_label"):
            self._selected_cell_label.value = f"Selected cell: {i}, {j}"
        if hasattr(self, "_depth_specifier"):
            self._depth_specifier.disabled = False
            self._depth_specifier.value = self.topo.depth.data[j, i]
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

    def construct_observances(self):
        """Attach event observers and callbacks to all interactive widgets and plot elements."""
        # Display mode toggle
        self._display_mode_toggle.observe(
            self.refresh_display_mode, names="value", type="change"
        )

        # Double click event for cell selection on the plot
        self.fig.canvas.mpl_connect("button_press_event", self.on_double_click)

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

        # Undo/Redo/Reset buttons
        self._undo_button.on_click(self.undo_last_edit)
        self._redo_button.on_click(self.redo_last_edit)
        self._reset_button.on_click(self.reset)

        # Snapshot controls
        self._save_button.on_click(self.on_tag)

        # Git/domain controls
        self._git_create_branch_button.on_click(self.on_git_create_branch)
        self._git_checkout_button.on_click(self.on_git_checkout)
        self._display_mode_toggle.observe(
            self.refresh_display_mode, names="value", type="change"
        )

    # --- UI Callback Methods ---

    def on_tag(self, _btn=None):
        """Save the current state as a snapshot and commit it to the repository."""
        name = self._tag_name.value.strip()
        if not name:
            print("Enter a snapshot name!")
            return

        self.topo.tcm.tag(name)  # TODO: Save a tag!
        print(f"Saved tag '{name}'.")
        return

    def on_double_click(self, event):
        """Handle double-click events on the plot to select a cell."""
        if event.dblclick and event.xdata is not None and event.ydata is not None:
            # Convert lon/lat to grid indices
            j, i = self.topo._grid.get_indices(event.ydata, event.xdata)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self._select_cell(i, j)

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

    def erase_disconnected_basin(self, b):
        """Erase all disconnected basins in the topography."""
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        self.topo.erase_disconnected_basin(i, j)
        self.update_undo_redo_buttons()

    def erase_selected_basin(self, b):
        """Erase the basin associated with the currently selected cell."""
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        self.topo.erase_selected_basin(i, j)
        self.update_undo_redo_buttons()

    def on_depth_change(self, change):
        """Handle changes to the depth specifier for the selected cell."""
        if self._selected_cell is None:
            return
        i, j, _ = self._selected_cell
        old_val = self.topo.depth.data[j, i]
        new_val = change["new"]
        if old_val == new_val:
            return
        cmd = DepthEditCommand(self.topo, [(j, i)], [new_val], old_values=[old_val])
        self.apply_edit(cmd)
        self.update_undo_redo_buttons()

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
