import time
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.path import shapely_to_path
from contourpy import contour_generator
from dataclasses import dataclass, field, replace
from functools import lru_cache
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.path import Path as MplPath
from matplotlib.ticker import AutoLocator, FixedLocator, FuncFormatter, NullFormatter
from pathlib import Path
from pyproj import Transformer
from scipy.spatial import cKDTree
import mom6_forge._conformal as cf
from mom6_forge import corner_diagnostics as diag
from mom6_forge._conformal import MapProjection
from mom6_forge.grid import Grid

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# CrocoDash tutorial box: lon_min, lon_max, lat_min, lat_max
_DEFAULT_BBOX = (235, 243, 31, 39)
# CSS pixels within which a press grabs a vertex or an edge, and a click may stray
# before it pans (Retina: twice the dpi)
_VERTEX_PX, _EDGE_PX, _CLICK_PX = 10, 8, 5
# Drag frames at most every 1/30 s (or 1.2x the last sketch); a skipped one 60 ms later
_FRAME_S, _CATCHUP_MS = 1 / 30, 60
# Pan/zoom: a redraw per 45 ms, full detail 250 ms after the last step, 1.25x per click
_COALESCE_MS, _LITE_MS, _ZOOM_STEP = 45, 250, 1.25
_HOVER_MS = 250  # the tooltip shows once the cursor rests this long
# Build peaks near 0.3 GB + 1.6 kB per cell (measured): about 4 GB at 2.4 million cells
_MAX_CELLS = 2_400_000
_EDGE_STEPS = 8  # points per outline edge drawn on the globe
_WORLD = (-180.0, 180.0, -90.0, 90.0)  # the land window of views off or round the globe
# The projection buttons: labelled by the first word, the whole name in the tooltip
_PROJECTIONS = [("Lambert conformal", "lcc"), ("Mercator", "merc")]
_PROJECTIONS += [("Transverse Mercator", "tmerc"), ("Polar stereographic", "stere")]
# The mouse instructions, in a box under the map
_HELP = (
    '<div style="border:1px solid #b0b0b0;border-radius:4px;background:#f4f6f9;'
    'padding:4px 8px;line-height:1.45;font-size:12px;overflow-wrap:anywhere">'
    "<b>Drag</b> a vertex to move it · <b>click an edge</b> to add a vertex<br>"
    "<b>Right-click</b> (or Ctrl-click) a vertex to delete it<br>"
    "<b>Double-click</b> a vertex to make or unmake a corner (4 needed)<br>"
    "<b>Clear all</b> to draw anew: <b>click</b> points, then point 1 to close<br>"
    "<b>Scroll</b> to zoom · <b>drag the map</b> (or middle-drag) to pan</div>"
)


# ------------------------------------------------------------------
# Outline
# ------------------------------------------------------------------


@dataclass
class Outline:
    """An editable polygon: vertex lon/lat, up to 4 corners, and undo/redo.

    ``is_open`` is True while the polygon is being drawn; undo and redo bring it back.

    Parameters
    ----------
    lon, lat : list of float
        Vertex coordinates in degrees, not closed.
    corners : list of int, optional
        Up to 4 corner vertex indices, in the order they were picked.
    """

    lon: list
    lat: list
    corners: list = field(default_factory=list)

    def __post_init__(self):
        self._restore(self.to_dict())
        self._undo, self._redo, self._dragging = [], [], False

    @classmethod
    def from_bbox(cls, lon_min, lon_max, lat_min, lat_max):
        lon = [lon_min, lon_max, lon_max, lon_min]
        return cls(lon, [lat_min, lat_min, lat_max, lat_max], [0, 1, 2, 3])

    @classmethod
    def from_grid(cls, grid, per_side=1):
        # Counter-clockwise from the SW corner; the grid corners become the corners
        qlon, qlat = np.asarray(grid.qlon, float), np.asarray(grid.qlat, float)
        ny, nx = qlon.shape[0] - 1, qlon.shape[1] - 1
        t = np.linspace(0.0, 1.0, per_side, endpoint=False)
        (i, i2), (j, j2) = (np.round([t * n, n - t * n]).astype(int) for n in (nx, ny))
        jj, ii = np.r_[0 * i, j, 0 * i + ny, j2], np.r_[i, 0 * j + nx, i2, 0 * j]
        corners = [0, per_side, 2 * per_side, 3 * per_side]
        return cls(qlon[jj, ii].tolist(), qlat[jj, ii].tolist(), corners)

    def to_dict(self):
        return dict(lon=list(self.lon), lat=list(self.lat), corners=list(self.corners))

    @property
    def n(self):
        return len(self.lon)

    def move(self, i, lon, lat):
        self._push()
        self.lon[i], self.lat[i] = lon, lat

    def insert(self, edge, lon, lat):
        self._push()
        self.lon.insert(edge + 1, lon)
        self.lat.insert(edge + 1, lat)
        self.corners = [c + (c > edge) for c in self.corners]
        return edge + 1

    def delete(self, i):
        self._push()
        self.corners = [c - (c > i) for c in self.corners if c != i]
        del self.lon[i], self.lat[i]

    def clear(self):
        self._push()
        self._restore(dict(lon=[], lat=[], corners=[], is_open=True))

    def toggle_corner(self, i):
        if i not in self.corners and len(self.corners) >= 4:
            return "Four corners already: remove one first."
        self._push()
        if i in self.corners:
            self.corners.remove(i)
            return f"Vertex {i + 1} is no longer a corner."
        self.corners.append(i)
        return f"Vertex {i + 1} is now a corner."

    def begin_drag(self):
        self._push()
        self._dragging = True

    def end_drag(self, moved=True):
        self._dragging = False
        if not moved and self._undo:
            self._undo.pop()

    def undo(self):
        return self._step(self._undo, self._redo)

    def redo(self):
        return self._step(self._redo, self._undo)

    def _step(self, source, target):
        if not source:
            return False
        target.append(self._snapshot())
        self._restore(source.pop())
        return True

    def _snapshot(self):
        return self.to_dict() | dict(is_open=self.is_open)

    def _restore(self, state):
        self.lon, self.lat = list(state["lon"]), list(state["lat"])
        self.corners = list(state["corners"])
        self.is_open = state.get("is_open", False)

    def _push(self):
        # A drag's moves share the one snapshot begin_drag took
        if not self._dragging:
            self._undo = (self._undo + [self._snapshot()])[-100:]
            self._redo.clear()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _div(html, style=""):
    return f'<div style="overflow-wrap:anywhere;line-height:1.4;{style}">{html}</div>'


def _messages_html(errors, warnings):
    """A bold Errors (dark red) and Warnings header, each over its lines; "" if none."""
    html, heads = "", ["<b style='color:#b00020'>Errors</b>", "<b>Warnings</b>"]
    for head, lines in zip(heads, (errors, warnings)):
        if lines:
            html += f"<div style='margin-top:8px'>{head}</div>"
            html += "".join(f"<div style='margin-top:3px'>{m}</div>" for m in lines)
    return html and _div(html)


def _cbar_formatter(decades, minor=False):
    """Plain log-colourbar labels; minor ones only at 2s and 5s over >1 decade."""
    sig = min(6, 2 + max(0, int(np.ceil(-np.log10(max(decades, 1e-6))))))

    def label(x, _pos=None):
        if x <= 0:
            return "0"
        mantissa = round(x / 10 ** np.floor(np.log10(x)))
        if minor and decades > 1 and mantissa not in (2, 5):
            return ""
        return f"{float(f'{x:.{sig}g}'):g}"

    return FuncFormatter(label)


def _nearest(px, py, vx, vy, r_vertex, r_edge):
    """(vertex, edge) at (px, py): a vertex within its radius wins, else an edge."""
    x, y = np.asarray(vx, float), np.asarray(vy, float)
    dx, dy = np.roll(x, -1) - x, np.roll(y, -1) - y
    t = ((px - x) * dx + (py - y) * dy) / np.maximum(dx * dx + dy * dy, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    dv, de = np.hypot(x - px, y - py), np.hypot(x + t * dx - px, y + t * dy - py)
    if (dv <= r_vertex).any():
        return int(np.argmin(np.where(dv <= r_vertex, dv, np.inf))), None
    return None, (int(np.argmin(de)) if de.min() <= r_edge else None)


class _Globe:
    """The map: the globe as seen from far above (lon_0, lat_0), in metres."""

    def __init__(self, lon_0, lat_0):
        self.centre, self.crs = (lon_0, lat_0), ccrs.Orthographic(lon_0, lat_0)
        self.radius, lonlat = self.crs.x_limits[1], "EPSG:4326"
        self._fwd = Transformer.from_crs(lonlat, self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, lonlat, always_xy=True)

    def to_xy(self, lon, lat):
        """``(x, y)`` of ``(lon, lat)`` degrees: NaN behind the globe."""
        x, y = (np.asarray(v, float) for v in self._fwd.transform(lon, lat))
        return np.where(np.isfinite(x + y), [x, y], np.nan)

    def to_lonlat(self, x, y):
        """``(lon, lat)`` of ``(x, y)``: a point off the globe gives its nearest on the rim."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        f = np.minimum(1.0, self.radius / np.maximum(np.hypot(x, y), 1.0))
        return self._inv.transform(x * f, y * f)


@lru_cache(maxsize=32)
def _land_paths(centre, window, scale):
    """Land fills and coastlines in a lon/lat window, on the globe facing `centre`."""
    globe, paths, rings = _Globe(*centre), [], []
    for poly in cf.land_polygons(window, scale):
        path = shapely_to_path(poly)
        xy = globe.to_xy(*path.vertices.T).T
        if np.isnan(xy).any():  # across the rim: cartopy cuts it there
            path = shapely_to_path(globe.crs.project_geometry(poly, ccrs.PlateCarree()))
            xy = path.vertices
        paths.append(MplPath(xy, path.codes))
        for ring in (poly.exterior, *poly.interiors):
            ll = np.asarray(ring.coords)[:, :2]
            # No coast along the dateline or a pole, where Natural Earth cuts land
            cut = ((np.diff(ll, axis=0) == 0) & (np.abs(ll[1:]) >= [180, 89.99])).any(1)
            ll = np.insert(ll, np.flatnonzero(cut) + 1, np.nan, axis=0)
            rings.append(globe.to_xy(*ll.T).T)
    return tuple(paths), tuple(rings)


def _contour_segments(fields, count):
    """About `count` xi and eta level curves each, clear of the raster's jagged edge."""
    segments = []
    for key, n in (("xi", fields["nx"]), ("eta", fields["ny"])):
        levels = np.arange(0, n + 1, max(1, round(n / max(count - 1, 1)))) / n
        if len(levels) > 1:
            half = float(np.median(np.diff(levels))) / 2
            levels = levels[(levels > half) & (levels < 1 - half)]
        x, y, z = fields["X"], fields["Y"], fields[key]
        gen = contour_generator(x, y, z, line_type="Separate")
        segments += [line for level in levels for line in gen.lines(level)]
    return segments


def _summary_html(q):
    rows = [("cell size, km (min/med/max)", "size_min size_median size_max")]
    rows += [("size ratio (max/min)", "size_ratio"), ("aspect max", "aspect_max")]
    rows += [("orthogonality, deg (med/p99/max)", "ortho_median ortho_p99 ortho_max")]
    rows += [("neighbour size ratio", "max_neighbour_ratio")]
    cells = [("grid", f"{q['nx']} x {q['ny']}  ({q['n_cells']} cells)")]
    cells += [(k, " / ".join(f"{q[v]:.2f}" for v in keys.split())) for k, keys in rows]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in cells)
    return f"<table>{body}</table>"


class GridSketcher(widgets.HBox):
    """Sketch a 4-corner outline on a map and build its conformal MOM6 grid.

    Drag a vertex to move it, press an edge to add one, right-click (or Ctrl-click) to
    delete one and double-click to make or unmake a corner; scroll zooms, dragging the
    empty map pans. Each edit is solved at preview resolution and drawn with its
    quality, cell size and open boundaries. "Build final grid" sets `grid`, a
    `mom6_forge.grid.Grid` whose ``outline`` lets ``GridSketcher(grid)`` reopen it;
    "Save" writes it to GridLibrary.

    Parameters
    ----------
    outline : Outline or mom6_forge.grid.Grid, optional
        A `Grid` with an ``outline`` reopens it; other grids have their edge traced.
    resolution_km : float, optional
        Target cell size in km. Default: the grid's own, or about 10,000 cells.
    projection : str or MapProjection, optional
        "lcc", "merc", "tmerc", "stere" or a `MapProjection`, re-centred on the outline
        after each edit. Default (None or "auto"): the recommended kind, whose cells
        vary least in size (the green button), until a button picks another.
    working_dir : str or pathlib.Path, optional
        "Save" writes to its ``GridLibrary/``. Default: the current directory.
    name : str, optional
        Grid name. Default: the grid's own name, else "sketch".
    per_side : int, optional
        Vertices per side when tracing a grid.
    figsize : (float, float), optional
        Map size in inches. Default (4.5, 5.0) fits beside the panel.
    blank : bool, optional
        Start with no outline, ready to click one in.

    Notes
    -----
    Sketching an outline and marking its corners follows the interactive boundary
    editors of pyroms (https://github.com/ESMG/pyroms) and pygridgen, which feed
    gridgen; see mom6_forge._conformal for the grid itself.
    """

    def __init__(
        self,
        outline=None,
        *,
        resolution_km=None,
        projection=None,
        working_dir=None,
        name=None,
        per_side=1,
        figsize=None,
        blank=False,
    ):
        if blank:
            outline = Outline([], [])
        elif outline is not None and not isinstance(outline, Outline):
            grid, stored = outline, outline.outline
            if stored:
                outline = Outline(stored["lon"], stored["lat"], stored["corners"])
                resolution_km = resolution_km or stored.get("resolution_km")
                if projection is None and stored.get("projection"):
                    projection = MapProjection(**stored["projection"])
            else:
                outline = Outline.from_grid(grid, per_side)
                km = cf.nominal_resolution_km(grid)
                resolution_km = resolution_km or float(f"{km:.2g}")
            name = name or grid.name
        self.outline = outline or Outline.from_bbox(*_DEFAULT_BBOX)
        if resolution_km is None and self.outline.n:
            area = cf.outline_area_km2(self.outline.lon, self.outline.lat)
            resolution_km = float(f"{np.sqrt(area / 10_000):.2g}")
        self.resolution_km = float(resolution_km or 5.0)
        self._ratios, self._auto = (None, {}), projection in (None, "auto")
        self.projection = cf.as_projection(projection, *self._extent_lonlat())
        if self._auto:
            self._fit_projection()
        self.working_dir = Path(working_dir or Path.cwd())
        self.preview = self.quality = self.grid = None
        self.timings = {"frame": None, "preview": None, "full": None}
        self.frame_times, self._timers = [], {}
        self.ax = self.shade_mesh = self._cbar = self._ring = None
        self._runs = self._metrics = self._wet = self._tree = self._hover_xy = None
        self._drag = self._pending = self._pan = self._click = None
        self._drag_changed = self._lite_on = False
        self._last_frame = self._last_redraw = self._closed_at = 0.0
        self._drawing = self.outline.n == 0
        self._build_controls(name or "sketch")
        self._build_figure(figsize or (4.5, 5.0))
        self._update_outline_artists()
        self._recompute_preview()
        children = [self.control_panel]
        if self._webagg:
            # The mouse instructions sit in a box under the map, as wide as the map
            self.help_html.layout.width = self.fig.canvas.layout.width
            reset = widgets.HBox([self.reset_view_button])
            column = [self.fig.canvas, self.help_html, reset]
            children.insert(0, widgets.VBox(column, layout={"flex": "0 0 auto"}))
        layout = dict(width="100%", align_items="flex-start", flex_flow="row nowrap")
        super().__init__(children, layout=layout)

    def close(self):
        """Close the sketch's figure (each GridSketcher opens its own) and the widget."""
        if getattr(self, "fig", None) is not None:
            plt.close(self.fig)
        super().close()

    @property
    def _drawing(self):
        """Whether the outline is still being clicked in (kept in its undo history)."""
        return self.outline.is_open

    @_drawing.setter
    def _drawing(self, value):
        self.outline.is_open = bool(value)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _button(self, label, callback):
        button = widgets.Button(description=label, layout={"width": "auto"})
        button.on_click(lambda _b: self._safe(callback))
        return button

    def _build_controls(self, name):
        W, row = widgets, {"flex_flow": "row wrap"}
        style = {"description_width": "initial"}
        self.help_html, self.status, self.summary = W.HTML(_HELP), W.HTML(), W.HTML()
        self.obc_html, self.details_html = W.HTML(), W.HTML()
        self.cells_actual, self.messages = W.HTML(), W.HTML()
        res = dict(value=self.resolution_km, min=0.1, max=1e3, step=0.5, style=style)
        self.resolution_box = W.BoundedFloatText(description="Resolution (km)", **res)
        self.resolution_box.layout.width = "160px"
        self.projection_buttons = {}
        for label, kind in _PROJECTIONS:
            b = W.ToggleButton(description=label.split()[0], layout={"width": "auto"})
            b.observe(lambda c, k=kind: self._safe(self._on_projection, k, c), "value")
            self.projection_buttons[kind] = b
        self.shade_cells = W.Checkbox(value=True, description="Shade cell size")
        self.undo_button = self._button("Undo", lambda: self._edit("undo"))
        self.redo_button = self._button("Redo", lambda: self._edit("redo"))
        self.clear_button = self._button("Clear all", lambda: self._edit("clear"))
        self.build_button = self._button("Build final grid", self._on_build)
        self.name_box = W.Text(value=name, description="Name", style=style)
        self.name_box.layout.width = "60%"
        self.save_button = self._button("Save", self._on_save)
        self.vertex_text = W.Textarea(description="Vertices", rows=10, style=style)
        self.apply_button = self._button("Apply coordinates", self._on_apply_vertices)
        self.reset_view_button = self._button("Reset view", self._on_reset_view)
        boxes = [self.resolution_box, self.shade_cells]
        fns = [self._on_resolution, self._on_shade]
        for box, fn in zip(boxes, fns):
            box.observe(lambda change, fn=fn: self._safe(fn, change), "value")
        details = W.Accordion([self.details_html], titles=("Details",))
        coords = W.VBox([self.vertex_text, self.apply_button])
        coords = W.Accordion([coords], titles=("Edit coordinates",))
        panel = [self.status]
        panel += [W.HBox([self.resolution_box, self.cells_actual], layout=row)]
        projections = [W.HTML("Projection"), *self.projection_buttons.values()]
        panel += [W.HBox(projections, layout=row), self.shade_cells, self.obc_html]
        edits = [self.undo_button, self.redo_button, self.clear_button]
        panel += [W.HBox(edits, layout=row)]
        panel += [self.build_button]
        panel += [W.HBox([self.name_box, self.save_button], layout=row)]
        panel += [self.summary, details, coords, self.messages]
        # Wide enough that the panel is no taller than the map; never a sideways scroll
        w = "400px"
        layout = dict(width=w, min_width=w, max_width=w, flex=f"0 0 {w}")
        layout.update(overflow="hidden auto")
        self.control_panel = W.VBox(panel, layout=layout)

    def _build_figure(self, figsize):
        with plt.ioff():
            self.fig = plt.figure(figsize=figsize, dpi=100)
        canvas = self.fig.canvas
        self._webagg = isinstance(canvas, widgets.DOMWidget)
        if self._webagg:
            # A 1 px frame marks where scrolling zooms the map; +6 px = frame + margins
            canvas.layout.width = f"{round(figsize[0] * 100) + 6}px"
            canvas.layout.flex, canvas.layout.border = "0 0 auto", "1px solid #b0b0b0"
            canvas.toolbar_visible = canvas.header_visible = False
            # Its drag handle can't widen the fixed-width frame, so no resizing
            canvas.resizable = False
            # No readout line under the map: the box at the cursor tells about the cell
            canvas.footer_visible, canvas.capture_scroll = False, True
            # Matplotlib's shortcut keys (p pan, o zoom, q close, ...) fight the editing
            canvas.mpl_disconnect(canvas.manager.key_press_handler_id)
        self.cbar_ax = self.fig.add_axes([0.15, 0.0, 0.7, 0.01], visible=False)
        self._new_axes()
        handlers = dict(button_press=self._on_press, motion_notify=self._on_motion)
        handlers.update(button_release=self._on_release, resize=self._relayout)
        handlers.update(scroll=self._on_scroll, axes_leave=self._cancel_hover)
        # Leaving the canvas straight from the map sends figure_leave but no axes_leave
        handlers.update(figure_leave=self._cancel_hover)
        for name, fn in handlers.items():
            canvas.mpl_connect(f"{name}_event", lambda e, fn=fn: self._safe(fn, e))

    def _safe(self, fn, *args):
        try:
            fn(*args)
        except Exception as exc:
            self._set_status(f"<b>Error:</b> {exc}")

    def _after(self, name, ms, fn=None):
        """Run `fn` once after `ms` ms, replacing any pending run of `name`."""
        if name in self._timers:
            self._timers.pop(name).stop()
        if fn is not None:
            self._timers[name] = timer = self.fig.canvas.new_timer(interval=ms)
            timer.single_shot = True
            timer.add_callback(lambda: (self._timers.pop(name, None), self._safe(fn)))
            timer.start()

    # ------------------------------------------------------------------
    # Map and axes
    # ------------------------------------------------------------------

    def _extent_lonlat(self):
        o = self.outline if self.outline.n >= 3 else Outline.from_bbox(*_DEFAULT_BBOX)
        return o.lon, o.lat

    def _new_axes(self, centre=None, half=None):
        """The map: a globe facing `centre` (default: the outline's middle, or the pole
        it winds round), fitted to the outline or `half` a view wide and high."""
        if self.ax is not None:
            self.fig.delaxes(self.ax)
        lon, lat = self._extent_lonlat()
        if centre is None:
            lat = np.asarray(lat, float)
            try:
                lon = cf.unwrap_lon(lon)
                mid = 0.5 * (lon.max() + lon.min())
                lat_0 = 0.5 * (lat.max() + lat.min())
            except ValueError:  # round a pole: the mean meridian down, as in the solve
                mid = np.degrees(np.angle(np.exp(1j * np.radians(lon)).mean()))
                lat_0 = 90.0 * np.sign(lat.mean())
            centre = (float(mid), float(lat_0))
        self.globe = _Globe(*centre)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=self.globe.crs)
        self.ax.format_coord = self._format_coord
        # A wide or short map sits on the colourbar, not in the middle of its box
        self.ax.set_anchor("S")
        self.shade_mesh = None
        self._relayout()
        if half is None:
            # The outline, edges as the grid follows them, plus 10% each way
            xy = self._edges_xy() if self.outline.n >= 3 else self.globe.to_xy(lon, lat)
            for v, set_lim in zip(xy, (self.ax.set_xlim, self.ax.set_ylim)):
                pad = 0.1 * (np.nanmax(v) - np.nanmin(v))
                set_lim(np.nanmin(v) - pad, np.nanmax(v) + pad)
            self._home = (self.ax.get_xlim(), self.ax.get_ylim())
        else:
            self.ax.set_xlim(-half[0], half[0])
            self.ax.set_ylim(-half[1], half[1])
        self._init_artists()
        if half is None:
            self._set_land(*self._extent_lonlat(), 6.0)
        else:  # the land in view
            self._set_land(*(self._view_lonlat() or (None, None)), 2.0)

    def _relayout(self, resize=None):
        # 0.48 in of labels under 0.12 in of colourbar, 0.16 in clear of the corner
        # markers on the map's edge; at most 60% of the figure
        f = min(1 / max(self.fig.get_size_inches()[1], 1e-3), 0.6 / 0.76)
        self.ax.set_position([0.02, 0.76 * f, 0.96, 0.98 - 0.76 * f])
        self.cbar_ax.set_position([0.15, 0.48 * f, 0.7, 0.12 * f])
        if resize is not None:
            self._redraw()

    def _set_land(self, lon, lat, factor):
        """Land round the points, or (lon None) on the whole globe."""
        window = _WORLD if lon is None else cf.land_window(lon, lat, factor)
        lon0, lon1, lat0, lat1 = window
        # Whole degrees, rounded outward, so nearby views share a cache entry
        lat0, lat1 = max(np.floor(lat0), -90.0), min(np.ceil(lat1), 90.0)
        self._land_window = (float(np.floor(lon0)), float(np.ceil(lon1)), lat0, lat1)
        scale = "110m" if self._land_window == _WORLD else "50m"
        paths, rings = _land_paths(self.globe.centre, self._land_window, scale)
        self.land_collection.set_paths(paths)
        self.coast_collection.set_segments(rings)

    def _view_lonlat(self):
        """Lon/lat round the view's edge; None once it passes the globe's rim or is
        over 4,000 km wide, when the whole globe's (coarser) land is drawn."""
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        s, one = np.linspace(0.0, 1.0, 9), np.ones(9)
        x = np.r_[x0 + (x1 - x0) * s, x1 * one, x1 - (x1 - x0) * s, x0 * one]
        y = np.r_[y0 * one, y0 + (y1 - y0) * s, y1 * one, y1 - (y1 - y0) * s]
        if np.hypot(x, y).max() >= self.globe.radius or x1 - x0 > 4e6:
            return None
        return tuple(np.asarray(v, float) for v in self.globe.to_lonlat(x, y))

    def _ensure_land_window(self):
        view = self._view_lonlat()
        if view is None:
            if self._land_window != _WORLD:
                self._set_land(None, None, 0.0)
            return
        lon, lat = view
        lon0, lon1, lat0, lat1 = self._land_window
        inside = self._land_window != _WORLD and lat.min() >= lat0 and lat.max() <= lat1
        if lon1 - lon0 < 360.0:
            inside &= bool(np.all((lon - lon0) % 360.0 <= lon1 - lon0))
        if not inside:
            self._set_land(lon, lat, 2.0)

    def _init_artists(self):
        ax, add = self.ax, self.ax.add_collection
        self.land_collection = add(PathCollection([], fc="#ece5d8", lw=0, zorder=0))
        coast = LineCollection([], lw=0.6, colors="#555555", zorder=0.6)
        self.coast_collection = add(coast)
        (self.outline_line,) = ax.plot([], [], "-", color="#222222", lw=1.2, zorder=3)
        (self.vertex_scatter,) = ax.plot([], [], "wo", ms=5, mec="#333333", zorder=4)
        (self.corner_scatter,) = ax.plot([], [], "ko", ms=12, zorder=5)
        ring = dict(ms=17, mfc="none", mec="#d62728", mew=2, zorder=5.5)
        (self.corner_ring,) = ax.plot([], [], "o", **ring)
        text = dict(color="w", ha="center", va="center", size=9, weight="bold")
        # Clipped to the map like their discs, never spilling onto the colourbar
        text.update(clip_on=True)
        self.corner_labels = [ax.text(0, 0, "", zorder=6, **text) for _ in range(4)]
        self.grid_lines = add(LineCollection([], lw=0.4, zorder=2))
        self.flag_lines = add(LineCollection([], zorder=2.1))
        self.obc_lines = add(LineCollection([], zorder=3.5))
        tiny = dict(ms=9.5, mfc="#d7191c", mec="w", zorder=6.5)
        (self.obc_tiny_markers,) = ax.plot([], [], "X", **tiny)
        box = dict(boxstyle="round,pad=0.3", fc="#ffffe0", ec="#999999", alpha=0.95)
        kw = dict(textcoords="offset points", size=9, bbox=box, zorder=10)
        self.hover_annotation = ax.annotate("", (0, 0), (12, 12), visible=False, **kw)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self, coalesce=False, full=False):
        """Ask the browser for a frame; pan/zoom: one per 45 ms, then a late one."""
        now, gap = time.monotonic(), _COALESCE_MS / 1000
        if coalesce == "late" or (coalesce and now - self._last_redraw >= gap):
            self._last_redraw = now
        elif coalesce:
            if "redraw" not in self._timers:
                self._after("redraw", _COALESCE_MS, lambda: self._redraw("late"))
            return
        if full:
            # A whole frame, not a diff, repaints any frame the browser dropped
            self.fig.canvas._force_full = True
        self.fig.canvas.draw_idle()

    def _vertex_pixels(self):
        x, y = self.globe.to_xy(self.outline.lon, self.outline.lat)
        pts = self.ax.transData.transform(np.column_stack([x, y]))
        return pts[:, 0], pts[:, 1]

    def _edges_xy(self):
        """Map (x, y) round the outline, _EDGE_STEPS points per edge from its first
        vertex: straight in the solve plane, as the grid's edge is."""
        o, t = self.outline, np.arange(_EDGE_STEPS) / _EDGE_STEPS
        xy = np.asarray(self.projection.to_xy(o.lon, o.lat), float)
        xy = (xy[..., None] + t * (np.roll(xy, -1, 1) - xy)[..., None]).reshape(2, -1)
        return self.globe.to_xy(*self.projection.to_lonlat(*xy))

    def _pick(self, px, py):
        """(vertex, edge) under pixel (px, py); a whole marker grabs its vertex."""
        if not self.outline.n:
            return None, None
        vx, vy, dpi = *self._vertex_pixels(), self.fig.dpi
        # Radius: the biggest marker on the vertex plus 1 pt of rim, or _VERTEX_PX
        r = np.full(len(vx), _VERTEX_PX * dpi / 100)
        for m in (self.vertex_scatter, self.corner_scatter, self.corner_ring):
            mx, my = self.ax.transData.transform(np.column_stack(m.get_data())).T
            on = np.hypot(vx[:, None] - mx, vy[:, None] - my).min(1, initial=9e9) < 1
            size = ((m.get_markersize() + m.get_markeredgewidth()) / 2 + 1) * dpi / 72
            r[on] = np.maximum(r[on], size)
        # Along the drawn edges: only each edge's first point is a vertex
        ex, ey = self.ax.transData.transform(self._edges_xy().T).T
        rr = np.full(len(ex), -1.0)
        rr[::_EDGE_STEPS] = r
        i, edge = _nearest(px, py, ex, ey, rr, _EDGE_PX * dpi / 100)
        return tuple(None if k is None else k // _EDGE_STEPS for k in (i, edge))

    def _update_outline_artists(self, sync_text=True):
        o = self.outline
        # Below 3 vertices draw a new outline; undo can bring a closed one back
        if not self._drawing and o.n < 3:
            self._drawing, o.corners = True, []
        self._drawing = self._drawing and not (o.n >= 3 and o.corners)
        ex, ey = self._edges_xy()
        x, y = ex[::_EDGE_STEPS], ey[::_EDGE_STEPS]
        # Closed, or open after the last vertex while being drawn
        ring = np.r_[: len(ex), 0][: -_EDGE_STEPS if self._drawing else None]
        self.outline_line.set_data(ex[ring], ey[ring])
        self.vertex_scatter.set_data(x, y)
        cx = cy = np.empty(0)
        if o.corners:
            # Numbered and judged in the solve plane, which winds as the map does
            px, py, pc = cf.orient_ccw(*self.projection.to_xy(o.lon, o.lat), o.corners)
            order = cf.order_corners(px, py, pc)
            ox, oy, _ = cf.orient_ccw(x, y, o.corners)
            cx, cy = ox[order], oy[order]
        four = len(cx) == 4
        flat = cf.interior_angles(px, py)[order] > 170 if four else []
        self.corner_ring.set_data(cx[flat], cy[flat])
        labels = [str(k + 1) if four else "" for k in range(4)]
        if self._drawing and o.n:
            # Point 1, where a click closes the outline, looks like corner 1
            cx, cy, labels[0] = x[:1], y[:1], "1"
        self.corner_scatter.set_data(cx, cy)
        for k, label in enumerate(self.corner_labels):
            label.set_text(labels[k])
            label.set_position((cx[k], cy[k]) if labels[k] else (0, 0))
        self.undo_button.disabled = not o._undo
        self.redo_button.disabled = not o._redo
        if sync_text:
            star = ["*" * (k in o.corners) for k in range(o.n)]
            rows = [f"{a:.4f}, {b:.4f}{s}" for a, b, s in zip(o.lon, o.lat, star)]
            self.vertex_text.value = "\n".join(rows)

    def _draw_grid_lines(self, cg, lite=False):
        """Interior grid lines as whole grey polylines, warn/bad edges on top.

        While the view moves (`lite`), every 4th line in flat grey and no flags.
        """
        q = np.stack([cg.x[::2, ::2], cg.y[::2, ::2]], axis=-1)
        # Thin the lines once cells are under 2 pixels on screen
        box, (x0, x1), (y0, y1) = self.ax.bbox, self.ax.get_xlim(), self.ax.get_ylim()
        scale = 0.5 * (box.width / abs(x1 - x0) + box.height / abs(y1 - y0))
        px = 1e3 * float(np.median(self._metrics["size"])) * scale
        k = 1 if px >= 2 or px <= 0 else int(np.ceil(2.0 / px))
        k *= 4 if lite else 1
        rows, cols = np.arange(k, cg.ny, k), np.arange(k, cg.nx, k)
        self.grid_lines.set_segments([q[j] for j in rows] + [q[:, i] for i in cols])
        if lite:
            self.flag_lines.set_visible(False)
            self.grid_lines.set(color="#999999", linewidth=0.4, antialiased=False)
            return
        grey = to_rgba(diag.STATUS_COLORS[0], 0.35)
        self.grid_lines.set(color=grey, linewidth=0.4, antialiased=True)
        # An edge takes the worse status of its two cells; -1 pads the boundary
        pad = np.pad(self._metrics["status"], 1, constant_values=-1)
        horiz = np.maximum(pad[:-1, 1:-1], pad[1:, 1:-1])[rows]
        vert = np.maximum(pad[1:-1, :-1], pad[1:-1, 1:])[:, cols]
        (jh, ih), (jv, iv) = np.nonzero(horiz > 0), np.nonzero(vert > 0)
        h = np.stack([q[rows[jh], ih], q[rows[jh], ih + 1]], axis=1)
        v = np.stack([q[jv, cols[iv]], q[jv + 1, cols[iv]]], axis=1)
        worst = np.concatenate([horiz[jh, ih], vert[jv, iv]])
        self.flag_lines.set_segments(np.concatenate([h, v]))
        colors = [diag.STATUS_COLORS[s] for s in worst]
        self.flag_lines.set(color=colors, lw=0.8 + 0.1 * (worst == 2), visible=True)

    def _draw_shading(self, cg):
        if self.shade_mesh is not None:
            self.shade_mesh.remove()
            self.shade_mesh = None
        self.cbar_ax.set_visible(bool(self.shade_cells.value and cg is not None))
        if not self.cbar_ax.get_visible():
            return
        size = np.clip(self._metrics["size"], 1e-9, None)
        norm = LogNorm(size.min(), max(size.max(), size.min() * 1.0001))
        qx, qy, kw = cg.x[::2, ::2], cg.y[::2, ::2], dict(cmap="Blues", alpha=0.75)
        # No shade on cells with a corner behind the globe
        off = np.isnan(qx[1:, 1:] + qx[1:, :-1] + qx[:-1, 1:] + qx[:-1, :-1])
        size, qx, qy = np.ma.masked_where(off, size), *np.nan_to_num([qx, qy])
        self.shade_mesh = self.ax.pcolormesh(qx, qy, size, norm=norm, zorder=1.5, **kw)
        if self._cbar is None:
            kw = dict(orientation="horizontal", label="cell size (km)")
            self._cbar = self.fig.colorbar(self.shade_mesh, cax=self.cbar_ax, **kw)
            # Minor ticks as long as major ones put every label on one line
            self.cbar_ax.tick_params(which="both", length=3)
        else:
            self._cbar.update_normal(self.shade_mesh)
        # update_normal resets the ticks, so set them after every update: both ends,
        # plus up to 4 of the values the log ticks label (plain ones when that leaves
        # fewer than 3), none within 12% of the bar of an end so no labels collide
        axis, lo, hi = self.cbar_ax.xaxis, norm.vmin, norm.vmax
        decades = float(np.log10(hi / lo))
        fmt, minor = _cbar_formatter(decades), _cbar_formatter(decades, minor=True)
        major = [t for t in axis.get_majorticklocs() if lo <= t <= hi]
        ticks = sorted([*major, *filter(minor, axis.get_minorticklocs())])
        ticks = [t for t in ticks if lo <= t <= hi]
        if len(ticks) < 3:
            ticks = [t for t in AutoLocator().tick_values(lo, hi) if lo <= t <= hi]
        room = 0.12 * decades
        inner = [t for t in ticks if min(np.log10(t / lo), np.log10(hi / t)) > room]
        if len(inner) > 4:  # too many: the powers of ten, else every other one
            inner = [t for t in inner if t in major][:4] or inner[::2][:4]
        axis.set_major_locator(FixedLocator([lo, *inner, hi]))
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())

    def _on_shade(self, _change):
        self._draw_shading(self._shown)
        self._redraw()

    def _sketch(self):
        """Sketch the dragged outline's grid as coarse xi/eta curves, no inversion."""
        segments, o, t0 = [], self.outline, time.perf_counter()
        if self._outline_problem() is None:
            try:
                kw = dict(projection=self.projection, raster_points=2_000)
                fields = cf.solve_fields(o.lon, o.lat, o.corners, self._n_cells(), **kw)
                lines = _contour_segments(fields, 15)
                to_lonlat = self.projection.to_lonlat
                segments = [self.globe.to_xy(*to_lonlat(*s.T)).T for s in lines]
            except Exception:
                pass
        self.grid_lines.set_segments(segments)
        self.grid_lines.set(color="#999999", antialiased=False)
        self.timings["frame"] = time.perf_counter() - t0
        self.frame_times = self.frame_times[-199:] + [self.timings["frame"]]

    # ------------------------------------------------------------------
    # Preview and diagnostics
    # ------------------------------------------------------------------

    def _n_cells(self):
        o = self.outline
        return cf.n_cells_for_resolution(o.lon, o.lat, self.resolution_km)

    def _set_status(self, html):
        self.status.value = _div(html)

    def _outline_problem(self):
        n, c = self.outline.n, len(self.outline.corners)
        if self._drawing:
            points = f"{n} point{'s' * (n != 1)}"
            return f"{points}: click to add one, then click point 1 to close."
        if c != 4:
            first = "click an edge to add a 4th vertex, then " * (n < 4)
            return f"Only {c} of 4 corners: {first}double-click a vertex to add one."
        try:
            x, y = self.projection.to_xy(self.outline.lon, self.outline.lat)
            if not cf.is_simple(x, y):
                return "Outline edges cross: drag a vertex to uncross them."
        except Exception:
            return "A vertex is somewhere the map projection can't represent."
        cells = self._n_cells()
        if cells > _MAX_CELLS:  # the preview solves at full size too
            cells = f"{cells / 1e6:.3g} million cells need over 4 GB to build"
            return f"{cells}: use Grid.from_outline in a job with more memory."
        return None

    def _size_ratios(self):
        """`diag.size_ratios` of the outline, solved again only once it changes."""
        o = self.outline
        key = (tuple(o.lon), tuple(o.lat), tuple(o.corners))
        if self._ratios[0] != key:
            ratios = diag.size_ratios(*key) if len(o.corners) == 4 else {}
            self._ratios = (key, ratios)
        return self._ratios[1]

    def _fit_projection(self):
        """Centre the solve plane on the outline: the recommended kind until a button
        picks one, and whenever the picked one can't take the outline."""
        ratios, kind = self._size_ratios(), self.projection.kind
        if self._auto or kind not in ratios:
            kind = min(ratios, key=ratios.get, default=kind)
        try:
            self.projection = cf.projection_for(kind, *self._extent_lonlat())
        except ValueError:
            pass  # e.g. a ring being drawn round a pole: kept until it closes

    def _show_projections(self):
        """The button of the projection in use pressed; the recommended one green."""
        ratios = self._size_ratios()
        best = min(ratios, key=ratios.get, default=None)
        for label, kind in _PROJECTIONS:
            b, green = self.projection_buttons[kind], kind == best
            b.value, b.button_style = kind == self.projection.kind, "success" * green
            b.tooltip = label + " (recommended: its cells vary least in size)" * green

    def _recompute_preview(self):
        self._show_projections()
        problem, o = self._outline_problem(), self.outline
        kw = dict(projection=self.projection, raster_points=cf.PREVIEW_RASTER_POINTS)
        if problem is None:
            try:
                t0 = time.perf_counter()
                cg = cf.solve(o.lon, o.lat, o.corners, self._n_cells(), **kw)
                self.timings["preview"] = time.perf_counter() - t0
                return self._show_solved_grid(cg)
            except Exception as exc:
                problem = f"Preview failed: {str(exc).rstrip('.')}."
        self.preview = self.quality = self._runs = self._tree = self._ring = None
        self._shown = None
        self.grid_lines.set_segments([])
        self.flag_lines.set_segments([])
        self.summary.value = self.details_html.value = self.cells_actual.value = ""
        self._cancel_hover()
        self._draw_shading(None)
        self._update_obc()
        self._show_messages([] if self._drawing else [problem], [])
        if self._drawing:
            self._set_status(f"<i>{problem}</i>")

    def _show_solved_grid(self, cg):
        """Run the diagnostics once for a solved grid; redraw every layer from them."""
        wet = diag.ocean_mask(cg)
        # Land cells are neither flagged nor warned about; no coastline: all count
        metrics = diag.cell_metrics(cg, wet=wet)
        q = diag.grid_quality(cg, metrics=metrics)
        ts = diag.timestep_limits(cg, wet=wet, ww3=False)
        runs = diag.open_boundary_runs(cg, ocean=wet)
        self.preview, self.quality, self._runs = cg, q, runs
        self._metrics, self._wet = metrics, wet
        size = f"cell size {q['size_min']:.2f}–{q['size_max']:.2f} km"
        name = {k: label for label, k in _PROJECTIONS}[cg.projection.kind]
        head = f"{name}: <b>{cg.nx} × {cg.ny}</b> cells &middot; nominal "
        head += f"{cf.nominal_resolution_km(cg):.2f} km &middot; {size}"
        self.summary.value = _div(head)
        coast = "ocean cells, from coastline"
        if wet is None:
            coast = "all cells: no coastline data"
        dx = f"smallest ocean cell {ts['min_wet_dx'] / 1000.0:.2f} km ({coast})"
        ts_line = f"<div style='margin-top:4px'>{dx} &middot; {ts['mom6_hint']}</div>"
        self.details_html.value = _div(_summary_html(q) + ts_line, "font-size:11px")
        self.cells_actual.value = f"&nbsp;≈ {cg.nx} × {cg.ny} cells"
        # Drawn on the globe: the grid with its nodes in map metres
        x, y = self.globe.to_xy(cg.lon, cg.lat)
        self._shown = replace(cg, x=x, y=y)
        self._draw_grid_lines(self._shown)
        self._draw_shading(self._shown)
        centres = np.column_stack([x[1::2, 1::2].ravel(), y[1::2, 1::2].ravel()])
        self._tree = cKDTree(np.nan_to_num(centres, nan=1e30))
        q = [a[::2, ::2] for a in (x, y)]
        ring = [np.r_[a[0], a[1:, -1], a[-1, -2::-1], a[-2:0:-1, 0]] for a in q]
        self._ring = MplPath(np.column_stack(ring))
        self._cancel_hover()
        self._update_obc()
        self._show_messages(*diag.quality_messages(cg, wet, metrics, runs))

    @property
    def open_boundaries(self):
        """Sides with open water in the last preview, in S/E/N/W order."""
        return [s for s in diag.OBC_SIDES if self._runs and self._runs[s]["runs"]]

    def _update_obc(self):
        cg, runs, tiny = self._shown, self._runs, []
        segs, opens = [np.empty((0, 2, 2))], [np.empty(0, bool)]
        if cg is not None:
            q = np.stack([cg.x[::2, ::2], cg.y[::2, ::2]], axis=-1)
            # Sides: south = row 0, east = column -1, north = row -1, west = column 0
            for side, p in zip(diag.OBC_SIDES, [q[0], q[:, -1], q[-1], q[:, 0]]):
                segs.append(np.stack([p[:-1], p[1:]], axis=1))
                opens.append(np.zeros(len(p) - 1, bool))
                for r in runs[side]["runs"]:
                    opens[-1][r["start"] : r["stop"]] = True
                tiny += [r for r in runs[side]["runs"] if r["tiny"]]
        xy = [self.globe.to_xy(r["lon"], r["lat"]) for r in tiny]
        self.obc_tiny_markers.set_data(*np.reshape(xy, (-1, 2)).T)
        is_open = np.concatenate(opens)
        self.obc_lines.set_segments(np.concatenate(segs))
        colors = np.where(is_open[:, None], to_rgba("#CC79A7"), to_rgba("#4d4d4d"))
        self.obc_lines.set(color=colors, linewidth=np.where(is_open, 3.2, 0.9))
        html = ""
        if cg is not None:
            note = "Final open cells depend on the bathymetry."
            if not runs["ocean_known"]:
                note = "Coastline unavailable: shading assumes all-ocean."
            names = ", ".join(self.open_boundaries) or "none"
            html = f"Open boundaries (from coastline): <b>{names}</b> "
            html += f"(for configure_forcings(boundaries=...))<br><i>{note}</i>"
        self.obc_html.value = _div(html)

    def _show_messages(self, errors, warnings):
        """Errors under the panel grey out Build; warnings turn it amber."""
        self.messages.value = _messages_html(errors, warnings)
        off = self.preview is None or bool(errors)
        self.build_button.disabled = self.save_button.disabled = off
        self.build_button.button_style = "warning" if warnings else "primary"
        self.build_button.tooltip = "See the warnings below" if warnings else ""
        self._set_status("Fix the errors below to build." if errors else "")

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _refresh(self):
        self._fit_projection()
        self._update_outline_artists()
        self._recompute_preview()
        self._redraw(full=True)

    def _on_press(self, event):
        if self._drag is not None or self._pan is not None:
            # A release off the map never arrives, so this press ends the gesture
            return self._on_release(event)
        xy = self._map_xy(event)
        if xy is None or event.button not in (1, 2, 3):
            return
        self._cancel_hover()
        if event.button == 2:
            return self._begin_pan(event)
        o, delete = self.outline, event.button == 3 or "ctrl" in event.modifiers
        i, edge = self._pick(event.x, event.y)
        if self._drawing and i == 0 and o.n >= 3 and not delete:
            return self._close_outline()  # a click on point 1 closes the outline
        if event.dblclick and (self._drawing or time.monotonic() < self._closed_at + 1):
            return  # ignored while drawing, and at the end of one that closed it
        if delete or event.dblclick:
            if i is not None:
                message = o.delete(i) if delete else o.toggle_corner(i)
                self._refresh()
                if message:
                    self._set_status(message)
            return
        self._drag_changed = False
        if i is not None:
            o.begin_drag()
            self._drag = i
        elif edge is not None and not self._drawing:
            # begin_drag first, so the insert and the drag after it are one undo step
            o.begin_drag()
            lon, lat = self.globe.to_lonlat(*xy)
            self._drag = o.insert(edge, float(lon), float(lat))
            self._drag_changed = True
        else:
            # While drawing, a click on the globe adds a point
            on_globe = self._drawing and np.hypot(*xy) < self.globe.radius
            self._begin_pan(event, xy if on_globe else None)

    def _close_outline(self):
        o, self._closed_at = self.outline, time.monotonic()
        o._push()
        # A new outline: on the recommended projection again
        self._drawing, self._auto = False, True
        if o.n == 4:
            o.corners = [0, 1, 2, 3]
        self._refresh()

    def _on_motion(self, event):
        if self._pan is not None:
            return self._pan_to(event)
        if self._drag is None:
            return self._on_hover(event)
        if event.inaxes is not self.ax:
            return
        # Draw the latest position now, or once the frame gap has passed
        self._pending, self._drag_changed = (event.xdata, event.ydata), True
        gap = max(_FRAME_S, 1.2 * (self.timings["frame"] or 0.0))
        if time.monotonic() - self._last_frame >= gap:
            self._drag_frame()
        elif "catchup" not in self._timers:
            self._after("catchup", _CATCHUP_MS, self._drag_frame)

    def _drag_frame(self):
        if self._drag is None or self._pending is None:
            return
        self._last_frame = time.monotonic()
        self._move_dragged(*self._pending)
        self._pending = None
        self._redraw()

    def _move_dragged(self, x, y, sketch=True):
        lon, lat = self.globe.to_lonlat(x, y)
        self.outline.move(self._drag, float(lon), float(lat))
        for artist in (self.shade_mesh, self.cbar_ax, self.flag_lines):
            if artist is not None:
                artist.set_visible(False)
        self._update_outline_artists(sync_text=False)
        if sketch:
            self._sketch()

    def _on_release(self, event):
        if self._pan is not None:
            (*_, xlim, ylim), click = self._pan, self._click
            self._pan = self._click = None
            (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
            if click is not None:
                # It never became a pan: add a point where it was pressed
                lon, lat = self.globe.to_lonlat(*click)
                self.outline.insert(self.outline.n - 1, float(lon), float(lat))
                self._refresh()
            elif self._drawing and (xlim, ylim) != ((x0, x1), (y0, y1)):
                # While drawing, the globe turns to face the view's middle, so pans
                # can reach the far side of the Earth
                mid = map(float, self.globe.to_lonlat(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
                self._on_reset_view(tuple(mid), (0.5 * (x1 - x0), 0.5 * (y1 - y0)))
            return
        if self._drag is None:
            return
        self._after("catchup", None)
        xy = (event.xdata, event.ydata) if event.inaxes is self.ax else self._pending
        if self._drag_changed and xy is not None:
            self._move_dragged(*xy, sketch=False)
        self.outline.end_drag(moved=self._drag_changed)
        self._drag = self._pending = None
        if self._drag_changed:
            self._refresh()

    def _edit(self, method):
        if getattr(self.outline, method)() is not False:
            self._refresh()

    def _on_apply_vertices(self):
        rows = [r.strip() for r in self.vertex_text.value.strip().splitlines()]
        try:
            xy = np.array([[float(v) for v in r.rstrip("*").split(",")] for r in rows])
        except ValueError:
            xy = np.empty(0)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 4:
            return self._set_status("<b>Need 4 or more lines of: lon, lat</b>")
        corners = [k for k, r in enumerate(rows) if r.endswith("*")][:4]
        self.outline = Outline(xy[:, 0].tolist(), xy[:, 1].tolist(), corners)
        self._drawing = False
        self._new_axes()  # facing the new outline
        self._refresh()

    # ------------------------------------------------------------------
    # Pan, zoom and hover
    # ------------------------------------------------------------------

    def _map_xy(self, event):
        """Map (x, y) under the mouse anywhere in the map's frame, also off the globe
        (where matplotlib gives no xdata); None outside the frame."""
        if event.inaxes is self.ax:
            return event.xdata, event.ydata
        if event.x is not None and self.ax.bbox.contains(event.x, event.y):
            return tuple(self.ax.transData.inverted().transform((event.x, event.y)))
        return None

    def _begin_pan(self, event, click=None):
        # A `click` (map x, y) adds a point on release, unless it strays into a pan
        self._pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
        self._click = click

    def _pan_to(self, event):
        # Keep the pressed map point under the cursor
        x, y, (x0, x1), (y0, y1) = self._pan
        if self._click is not None:
            if np.hypot(event.x - x, event.y - y) <= _CLICK_PX * self.fig.dpi / 100:
                return
            self._click = None
        dx = (event.x - x) * (x1 - x0) / self.ax.bbox.width
        dy = (event.y - y) * (y1 - y0) / self.ax.bbox.height
        self._move_view((x0 - dx, x1 - dx), (y0 - dy, y1 - dy))

    def _on_scroll(self, event):
        xy = self._map_xy(event)
        if xy is None:
            return
        (x0, x1), (y0, y1), (x, y) = self.ax.get_xlim(), self.ax.get_ylim(), xy
        # Zoom about the cursor, from 1/50 of the fitted width out to the whole disc
        fit = np.ptp(self._home[0]) / (x1 - x0)
        globe = 2.1 * self.globe.radius / min(x1 - x0, y1 - y0)
        s = 1 / _ZOOM_STEP if event.step > 0 else _ZOOM_STEP
        s = float(np.clip(s, 0.02 * fit, max(globe, 1.0)))
        xlim = (x - (x - x0) * s, x + (x1 - x) * s)
        ylim = (y - (y - y0) * s, y + (y1 - y) * s)
        if s >= globe:  # all the way out: the disc in the middle
            xlim, ylim = (tuple(np.subtract(v, np.mean(v))) for v in (xlim, ylim))
        self._move_view(xlim, ylim)

    def _move_view(self, xlim, ylim):
        """Pan or zoom: cheap frames while the view moves, full detail once it stops."""
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self._ensure_land_window()
        self._cancel_hover()
        if not self._lite_on:
            self._set_lite(True)
        self._after("lite", _LITE_MS, lambda: self._set_lite(False))
        self._redraw(coalesce=True)

    def _set_lite(self, on):
        """While `on`: no shading, flags or antialiasing; a quarter of the lines."""
        if not on and self._drag is not None:
            # A drag begun mid-gesture draws its own sketch: restore after it ends
            return self._after("lite", _LITE_MS, lambda: self._set_lite(False))
        self._lite_on = on
        for artist in (self.grid_lines, self.land_collection, self.coast_collection):
            artist.set_antialiased(not on)
        if self.shade_mesh is not None:
            self.shade_mesh.set_visible(not on)
            self.cbar_ax.set_visible(not on)
        if self._shown is not None:
            self._draw_grid_lines(self._shown, lite=on)
        if not on:
            self._redraw(full=True)

    def _on_reset_view(self, centre=None, half=None):
        # A new globe, facing the outline wherever it has moved, fitted to it
        self._new_axes(centre, half)
        self._update_outline_artists()
        if self.preview is not None:
            self._show_solved_grid(self.preview)
        self._redraw(full=True)

    def _on_hover(self, event):
        # The box shows only over a grid cell, not over empty map or land outside it
        if event.inaxes is not self.ax or not self._in_grid(event.xdata, event.ydata):
            return self._cancel_hover()
        self._hover_xy = (event.xdata, event.ydata)
        self._after("hover", _HOVER_MS, self._show_tooltip)

    def _show_tooltip(self):
        if self._hover_xy is None or self._tree is None:
            return
        x, y = self._hover_xy
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        # Offset the box toward the middle of the view, so it stays on the map
        right, up = int(x < 0.5 * (x0 + x1)), int(y < 0.5 * (y0 + y1))
        ann = self.hover_annotation
        ann.xy, ann.xyann = (x, y), (24 * right - 12, 24 * up - 12)
        text = self._format_coord(x, y).replace(" · ", "\n")
        ann.set(text=text, ha=["right", "left"][right], va=["top", "bottom"][up])
        ann.set_visible(True)
        self._redraw()

    def _cancel_hover(self, _event=None):
        self._after("hover", None)
        self._hover_xy = None
        if self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self._redraw()

    def _in_grid(self, x, y):
        """Whether the map point (x, y) is inside the last preview's grid."""
        return self._ring is not None and self._ring.contains_point((x, y))

    def _hover_cell(self, x, y):
        """(j, i) of the cell under (x, y); on the edge between two cells, the worse."""
        status = self._metrics["status"]
        dist, idx = self._tree.query([x, y], k=min(2, status.size))
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        cells = [divmod(int(a), status.shape[1]) for a in idx]
        if len(cells) == 2 and 0 < dist[0] and dist[1] <= 1.15 * dist[0]:
            (j0, i0), (j1, i1) = cells
            if abs(j0 - j1) + abs(i0 - i1) == 1:
                return cells[1] if status[j1, i1] >= status[j0, i0] else cells[0]
        return cells[0]

    def _format_coord(self, x, y):
        if not self._in_grid(x, y):
            return ""
        (j, i), m, wet = self._hover_cell(x, y), self._metrics, self._wet
        (ny, nx), th = m["status"].shape, m["thresholds"]
        at = [j == 0, i == nx - 1, j == ny - 1, i == 0]
        sides, note = "/".join(s for s, on in zip(diag.OBC_SIDES, at) if on), ""
        if sides and wet is not None:
            state = "open boundary (ocean)" if wet[j, i] else "closed (land)"
            note = f" · {sides} side: {state}"
        status = int(m["status"][j, i])
        if wet is not None and not wet[j, i]:
            return f"Cell (i={i}, j={j}): over land{note}"
        if status == 0:
            return f"Cell (i={i}, j={j}): OK{note}"
        ratio, ortho, aspect = (float(m[k][j, i]) for k in ("ratio", "ortho", "aspect"))
        rw, ow, aw = th["ratio_warn"], th["ortho_warn"], th["aspect_warn"]
        reasons = [
            f"size jump {ratio:.2f}x vs neighbour (warn > {rw:.1f}x)" * (ratio > rw),
            f"angle off 90 by {ortho:.1f} deg (warn > {ow:.0f})" * (ortho > ow),
            f"aspect {aspect:.2f} (warn > {aw:.1f})" * (aspect > aw),
            "folded (inside out)" * bool(m["folded"][j, i]),
        ]
        reasons = " · ".join(filter(None, reasons))
        return f"Cell (i={i}, j={j}) {['WARN', 'BAD'][status - 1]}: {reasons}{note}"

    # ------------------------------------------------------------------
    # Settings, build and save
    # ------------------------------------------------------------------

    def _on_projection(self, kind, change):
        if change["new"] == (kind == self.projection.kind):
            return  # the buttons already show it
        try:
            # A click on the pressed button does nothing; an unusable kind (round a
            # pole, Lambert on the equator) raises and keeps the one in use
            if change["new"]:
                self.projection = cf.projection_for(kind, *self._extent_lonlat())
                self._auto = False  # kept while the outline is edited
                # The map stays: only the plane the grid is solved in changes
                self._refresh()
        finally:
            self._show_projections()

    def _on_resolution(self, change):
        self.resolution_km = float(change["new"])
        self._recompute_preview()
        self._redraw()

    def _on_build(self):
        o, problem = self.outline, self._outline_problem()
        if problem is not None:
            return self._set_status(f"<i>{problem}</i>")
        self._set_status("Building final grid...")
        self.build_button.disabled = True
        name, res = self.name_box.value.strip() or "sketch", self.resolution_km
        try:
            t0, n = time.perf_counter(), self._n_cells()
            cg = cf.solve(o.lon, o.lat, o.corners, n, projection=self.projection)
            grid = Grid._from_conformal(cg, o.lon, o.lat, o.corners, res, name)
            self.timings["full"] = time.perf_counter() - t0
            self._show_solved_grid(cg)
        except Exception as exc:
            self.build_button.disabled = False
            self._set_status(f"<b>Build failed:</b> {exc}")
        else:
            self.grid, seconds = grid, self.timings["full"]
            done = f"Built {cg.nx} × {cg.ny} grid in {seconds:.1f} s: sketch.grid is"
            self._set_status(f"{done} ready for Topo and CrocoDash's Case")
        self._redraw()

    def _on_save(self):
        if self.grid is None:
            return self._set_status("<b>Build a grid before saving.</b>")
        name = self.name_box.value.strip() or "sketch"
        path = self.working_dir / "GridLibrary" / f"grid_{name}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.grid.name = name
        self.grid.write_supergrid(str(path))
        self._set_status(f"Saved {path}")
