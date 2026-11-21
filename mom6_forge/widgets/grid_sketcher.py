"""GridSketcher: sketch an outline on a map, pick its 4 corners and build its conformal grid.

GridSketcher's logic comes from NCAR/mom6_forge PR #149 by Dan Amrhein: the outline with
its corners and undo/redo history, the checks on it, the projection choice, the preview
solve with its diagnostics, Build, Save, and reopening a saved grid. This module keeps
that logic; its front end, ``grid_sketcher.js``, draws it with OpenLayers in the projection
the grid is solved in (the ``proj4`` trait), so an edge drawn straight on screen is
straight in the solver's plane. Everything on the wire is lon/lat degrees.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import shapely
import traitlets

import mom6_forge._conformal as cf
from mom6_forge import corner_diagnostics as diag
from mom6_forge.grid import Grid
from mom6_forge.widgets._base import ActionWidget, as_bytes, from_buffer

# The CrocoDash tutorial box, the outline when none is given
_DEFAULT = dict(lon=[235, 243, 243, 235], lat=[31, 31, 39, 39], corners=[0, 1, 2, 3])
# Build peaks near 0.3 GB + 1.6 kB per cell (measured in PR #149): about 4 GB here
_MAX_CELLS = 2_400_000
_MAX_LINES = 60  # preview grid lines drawn each way
_NAMES = {
    "lcc": "Lambert conformal",
    "merc": "Mercator",
    "tmerc": "Transverse Mercator",
    "stere": "Polar stereographic",
}


class Outline:
    """An editable polygon: vertex lon/lat (degrees, not closed), up to 4 corner vertex
    indices in the order they were picked, and an undo/redo history of whole states."""

    def __init__(self, lon=(), lat=(), corners=()):
        self._restore(dict(lon=lon, lat=lat, corners=corners))
        self._undo, self._redo = [], []

    @classmethod
    def from_grid(cls, grid, per_side=1):
        """A grid's edge, counter-clockwise from its south-west corner, `per_side`
        vertices a side; the grid's corners are the corners."""
        qlon, qlat = np.asarray(grid.qlon, float), np.asarray(grid.qlat, float)
        ny, nx = qlon.shape[0] - 1, qlon.shape[1] - 1
        t = np.linspace(0.0, 1.0, per_side, endpoint=False)
        (i, i2), (j, j2) = (np.round([t * n, n - t * n]).astype(int) for n in (nx, ny))
        jj, ii = np.r_[0 * i, j, 0 * i + ny, j2], np.r_[i, 0 * j + nx, i2, 0 * j]
        corners = [0, per_side, 2 * per_side, 3 * per_side]
        return cls(qlon[jj, ii], qlat[jj, ii], corners)

    @property
    def n(self):
        return len(self.lon)

    def to_dict(self):
        return dict(lon=list(self.lon), lat=list(self.lat), corners=list(self.corners))

    def set(self, lon, lat, corners):
        """Replace the outline, as one undo step."""
        self._undo = self._undo[-99:] + [self.to_dict()]
        self._redo = []
        self._restore(dict(lon=lon, lat=lat, corners=corners))

    def undo(self):
        return self._step(self._undo, self._redo)

    def redo(self):
        return self._step(self._redo, self._undo)

    def _step(self, source, target):
        if not source:
            return False
        target.append(self.to_dict())
        self._restore(source.pop())
        return True

    def _restore(self, state):
        self.lon = [float(v) for v in state["lon"]]
        self.lat = [float(v) for v in state["lat"]]
        self.corners = [int(c) for c in state["corners"]]


class GridSketcher(ActionWidget):
    """Sketch an outline on a map, pick its 4 corners and build its conformal grid.

    Every edit is solved at preview resolution and shown with its quality checks. Build
    sets ``grid``, a `Grid` whose ``outline`` lets ``GridSketcher(grid=...)`` reopen it
    (an edit, a projection or a resolution change drops it until the next Build); Save
    writes it to ``GridLibrary/grid_<name>.nc``.

    Parameters
    ----------
    grid : mom6_forge.grid.Grid, optional
        Reopen this grid: its ``outline`` (kept by grids this widget or
        `Grid.from_outline` built), or else its edge traced with `per_side` vertices a side.
    outline : dict, optional
        Start from ``lon``, ``lat`` (degrees) and ``corners`` (vertex indices), as in
        `Grid.outline`. Default: a box off California.
    resolution_km : float, optional
        Target cell size, km. Default: the outline's own, or 10 km.
    projection : str or MapProjection, optional
        "lcc", "merc", "tmerc" or "stere", re-centred on the outline after each edit, or
        a `MapProjection`. Default: the kind whose cells vary least in size, which the
        widget recommends, until a projection button picks another.
    working_dir : str or pathlib.Path, optional
        Save writes to its ``GridLibrary/``. Default: the current directory.
    name : str, optional
        Grid name. Default: the grid's own name, or "sketch".
    per_side : int, optional
        Vertices a side when tracing a grid that has no outline.
    """

    _js_file = "grid_sketcher.js"

    vertices = traitlets.Bytes(b"").tag(sync=True)
    """float64 (n, 2) lon/lat of the outline, not closed."""
    corners = traitlets.List(traitlets.Int()).tag(sync=True)
    """Corner vertices. With 4, corners[k] is labelled k + 1: the grid's own order, from
    its south-west corner counter-clockwise. Fewer are in the order picked."""
    flat_corners = traitlets.List(traitlets.Int()).tag(sync=True)
    """Corner vertices whose interior angle is over 170 degrees."""
    kind = traitlets.Unicode("").tag(sync=True)
    proj4 = traitlets.Unicode("").tag(sync=True)
    """The projection the grid is solved in, and the map drawn in."""
    recommended = traitlets.Unicode("").tag(sync=True)
    """The kind whose cells vary least in size ("" when the outline can't be solved)."""
    resolution_km = traitlets.Float(10.0).tag(sync=True)
    name = traitlets.Unicode("sketch").tag(sync=True)
    nodes = traitlets.Bytes(b"").tag(sync=True)
    """float64 (rows, cols, 2) lon/lat of every k-th preview grid node, k chosen so at
    most ``_MAX_LINES`` lines show each way; ``node_shape`` is [rows, cols]."""
    node_shape = traitlets.List(traitlets.Int(), [0, 0]).tag(sync=True)
    flags = traitlets.Bytes(b"").tag(sync=True)
    """uint8 (rows - 1, cols - 1): worst `corner_diagnostics.cell_metrics` status (0 ok,
    1 warn, 2 bad) of the cells in each block between drawn lines."""
    land = traitlets.Unicode("").tag(sync=True)
    """GeoJSON FeatureCollection (lon/lat): land polygons and graticule lines."""
    cells = traitlets.Unicode("").tag(sync=True)
    summary = traitlets.Unicode("").tag(sync=True)
    details = traitlets.List().tag(sync=True)
    """[label, value] rows: the quality table, the time-step hint and open boundaries."""
    preview_errors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    """What stops Build: an outline problem, a failed preview or folded cells."""
    preview_warnings = traitlets.List(traitlets.Unicode()).tag(sync=True)
    can_build = traitlets.Bool(False).tag(sync=True)
    can_save = traitlets.Bool(False).tag(sync=True)
    can_undo = traitlets.Bool(False).tag(sync=True)
    can_redo = traitlets.Bool(False).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    """What the last Build, Save or edit did, or what to do next."""

    def __init__(
        self,
        grid=None,
        *,
        outline=None,
        resolution_km=None,
        projection=None,
        working_dir=None,
        name=None,
        per_side=1,
    ):
        super().__init__()
        start = outline or getattr(grid, "outline", None)
        if start:
            self.outline = Outline(start["lon"], start["lat"], start["corners"])
            resolution_km = resolution_km or start.get("resolution_km")
            if projection is None and start.get("projection"):
                projection = cf.MapProjection(**start["projection"])
        elif grid is not None:
            self.outline = Outline.from_grid(grid, per_side)
            km = cf.nominal_resolution_km(grid)
            resolution_km = resolution_km or float(f"{km:.2g}")
        else:
            self.outline = Outline(**_DEFAULT)
        o = self.outline
        self._auto = projection in (None, "auto")
        self.projection = cf.as_projection(projection, o.lon, o.lat)
        self.working_dir = Path(working_dir or Path.cwd())
        self.grid = self.preview = self.quality = self._view = None
        self._ratios = (None, {})
        with self.hold_sync():
            self.name = (name or getattr(grid, "name", None) or "sketch").removeprefix(
                "grid_"
            )
            self.resolution_km = float(resolution_km or 10.0)
            self._refresh(refit=self._auto)
            self._send_land(cf.land_window(o.lon, o.lat, 6.0))

    def _handle(self, widget, msg, buffers):
        with self.hold_sync():  # one update per action: the page never draws half a state
            super()._handle(widget, msg, buffers)

    # ------------------------------------------------------------------ actions

    def _op_outline(self, msg, buffers):
        """A vertex drag or an edge insert ended: the whole outline, as edited."""
        o = self.outline
        lon, lat, corners = _edited(o, *_lonlat(buffers))
        if (lon, lat, corners) != (o.lon, o.lat, o.corners):
            self._edit(lon, lat, corners)

    def _op_draw(self, msg, buffers):
        """A new outline, clicked in or a Box: with 4 vertices, all are corners."""
        lon, lat = _lonlat(buffers)
        self._auto = True  # a new outline gets the recommended projection again
        self._edit(lon, lat, [0, 1, 2, 3] if len(lon) == 4 else [])

    def _op_corner(self, msg, buffers):
        o, i = self.outline, int(msg["i"])
        corners = [c for c in o.corners if c != i]
        if len(corners) == len(o.corners):
            if len(corners) >= 4:
                raise ValueError("There are 4 corners already: unmake one first.")
            corners.append(i)
        self._edit(o.lon, o.lat, corners)

    def _op_delete(self, msg, buffers):
        o, i = self.outline, int(msg["i"])
        if o.n <= 3:
            raise ValueError("An outline needs 3 vertices: press Clear to draw anew.")
        keep = [k for k in range(o.n) if k != i]
        corners = [c - (c > i) for c in o.corners if c != i]
        self._edit([o.lon[k] for k in keep], [o.lat[k] for k in keep], corners)

    def _op_clear(self, msg, buffers):
        self._edit([], [], [])

    def _op_undo(self, msg, buffers):
        if self.outline.undo():
            self._refresh()

    def _op_redo(self, msg, buffers):
        if self.outline.redo():
            self._refresh()

    def _op_projection(self, msg, buffers):
        # An unusable kind (Lambert on the equator, any but Polar round a pole) raises
        # and keeps the one in use.
        self.projection = cf.projection_for(str(msg["kind"]), *self._where())
        self._auto = False  # kept while the outline is edited
        self._refresh(refit=False)

    def _op_resolution(self, msg, buffers):
        km = float(msg.get("km") or "nan")
        if not km > 0:
            raise ValueError(f"The resolution must be more than 0 km (got {km:g}).")
        self.resolution_km = km
        self._refresh(refit=False)

    def _op_build(self, msg, buffers):
        if not self.can_build:
            raise ValueError("Fix the errors listed in the panel first.")
        self.name = _check_name(msg.get("name", self.name))
        o, t0 = self.outline, time.perf_counter()
        cg = cf.solve(o.lon, o.lat, o.corners, self._n_cells(), self.projection)
        km = self.resolution_km
        self.grid = Grid._from_conformal(cg, o.lon, o.lat, o.corners, km, self.name)
        self._show(cg)
        self.can_save = True
        seconds = time.perf_counter() - t0
        self.status = (
            f"Built a {cg.nx} × {cg.ny} grid in {seconds:.1f} s. It is in .grid, "
            "ready for Topo and CrocoDash's Case."
        )

    def _op_save(self, msg, buffers):
        if self.grid is None:
            raise ValueError("Build the grid before saving it.")
        self.name = self.grid.name = _check_name(msg.get("name", self.name))
        path = self.working_dir / "GridLibrary" / f"grid_{self.name}.nc"
        if path.exists() and not msg.get("overwrite"):
            raise FileExistsError(
                f"{path.name} is already in {path.parent}: tick Overwrite to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.grid.write_supergrid_file(str(path))
        self.status = f"Saved {path}."

    def _op_view(self, msg, buffers):
        """The map moved: points round its edge, to send the land around it."""
        lon, lat = (np.asarray(v) for v in _lonlat(buffers))
        ok = np.isfinite(lon) & np.isfinite(lat)
        if ok.sum() < 3:  # off the projection's world
            return
        self._view = lon[ok].tolist(), lat[ok].tolist()
        if not _within(cf.land_window(*self._view, 1.0), self._land_window):
            self._send_land(cf.land_window(*self._view, 2.0))

    # ------------------------------------------------------------------ state

    def _edit(self, lon, lat, corners):
        self.outline.set(lon, lat, corners)
        self._refresh()

    def _where(self):
        """Lon/lat to centre a projection on: the outline, or the view while it's empty."""
        o = self.outline
        return (o.lon, o.lat) if o.n >= 3 else self._view or (o.lon, o.lat)

    def _n_cells(self):
        return cf.n_cells_for_resolution(
            self.outline.lon, self.outline.lat, self.resolution_km
        )

    def _refresh(self, refit=True):
        """After any change: centre the projection, drop a built grid, send the outline
        and a new preview."""
        if refit:
            self._fit_projection()
        self.status = ""
        if self.grid is not None:
            self.grid, self.can_save = None, False
            self.status = "The sketch changed: press Build again."
        o = self.outline
        if o.n == 0:
            self.status = "Click points on the map, then click the first one to close."
        ratios = self._size_ratios()
        self.kind, self.proj4 = self.projection.kind, self.projection.proj4
        self.recommended = min(ratios, key=ratios.get, default="")
        self.vertices = as_bytes(np.column_stack([o.lon, o.lat]), "<f8")
        self.corners, self.flat_corners = self._corner_labels()
        self.can_undo, self.can_redo = bool(o._undo), bool(o._redo)
        problem = self._outline_problem()
        if problem is None:
            try:
                kw = dict(raster_points=cf.PREVIEW_RASTER_POINTS)
                cg = cf.solve(
                    o.lon, o.lat, o.corners, self._n_cells(), self.projection, **kw
                )
                return self._show(cg)
            except Exception as exc:
                problem = f"Preview failed: {str(exc).rstrip('.')}."
        self._show(None, [problem] if problem else [])

    def _fit_projection(self):
        """Centre the solve plane on the outline: the recommended kind until a button
        picks one, and whenever the picked one can't take the outline."""
        ratios, kind = self._size_ratios(), self.projection.kind
        if self._auto or kind not in ratios:
            kind = min(ratios, key=ratios.get, default=kind)
        if self.outline.n >= 3:
            try:
                self.projection = cf.projection_for(
                    kind, self.outline.lon, self.outline.lat
                )
            except ValueError:
                pass  # e.g. Lambert for an outline that now straddles the equator

    def _size_ratios(self):
        """`diag.size_ratios` of the outline, solved again only once it changes."""
        o = self.outline
        key = (tuple(o.lon), tuple(o.lat), tuple(o.corners))
        if self._ratios[0] != key:
            self._ratios = (key, diag.size_ratios(*key) if len(o.corners) == 4 else {})
        return self._ratios[1]

    def _corner_labels(self):
        """Corners in label order and the nearly flat ones, judged in the solve plane on
        the outline turned counter-clockwise, as the solver numbers them."""
        o = self.outline
        if len(o.corners) != 4:
            return list(o.corners), []
        x, y = (np.asarray(v, float) for v in self.projection.to_xy(o.lon, o.lat))
        if not np.isfinite([x, y]).all():
            return list(o.corners), []
        flip = cf._signed_area(x, y) < 0
        if flip:
            x, y = x[::-1], y[::-1]
        ring = lambda i: o.n - 1 - i if flip else i  # its own inverse
        order = cf.order_corners(x, y, [ring(c) for c in o.corners])
        angles = cf.interior_angles(x, y)
        return [ring(c) for c in order], [ring(c) for c in order if angles[c] > 170]

    def _outline_problem(self):
        """Why the outline can't be previewed, or None ("" while one is drawn anew)."""
        o, c = self.outline, len(self.outline.corners)
        if o.n == 0:
            return ""
        if c != 4:
            first = "drag an edge to add a 4th vertex, then " * (o.n < 4)
            return f"Only {c} of 4 corners: {first}double-click a vertex to add one."
        x, y = self.projection.to_xy(o.lon, o.lat)
        if not np.isfinite([x, y]).all():
            return "A vertex is somewhere the map projection can't represent."
        if not cf.is_simple(x, y):
            return "Outline edges cross: drag a vertex to uncross them."
        if (n := self._n_cells()) > _MAX_CELLS:
            return (
                f"{n / 1e6:.3g} million cells need over 4 GB to build: use "
                "Grid.from_outline in a job with more memory, or a coarser resolution."
            )
        return None

    def _show(self, cg, errors=()):
        """Send a solved grid (or None) with its diagnostics: lines, flags, messages."""
        self.preview, warnings = cg, []
        if cg is None:
            o = self.outline
            self.nodes, self.flags, self.node_shape = b"", b"", [0, 0]
            self.summary, self.details, self.quality = "", [], None
            self.cells = f"≈ {self._n_cells():,} cells" if o.n >= 3 else ""
        else:
            wet = diag.ocean_mask(cg)  # land cells are neither flagged nor warned about
            metrics = diag.cell_metrics(cg, wet=wet)
            q = self.quality = diag.grid_quality(cg, metrics=metrics)
            ts = diag.timestep_limits(cg, wet=wet, ww3=False)
            runs = diag.open_boundary_runs(cg, ocean=wet)
            errors, warnings = diag.quality_messages(cg, wet, metrics, runs)
            # Every k-th node, and the worst status of the cells between those lines
            k = max(1, -(-max(cg.nx, cg.ny) // _MAX_LINES))
            jj, ii = (np.unique(np.r_[0:n:k, n]) for n in (cg.ny, cg.nx))
            at = np.ix_(jj, ii)
            self.nodes = as_bytes(np.stack([cg.qlon[at], cg.qlat[at]], -1), "<f8")
            status = np.maximum.reduceat(metrics["status"], jj[:-1], axis=0)
            self.flags = as_bytes(np.maximum.reduceat(status, ii[:-1], axis=1), "u1")
            self.node_shape = [len(jj), len(ii)]
            self.cells = f"≈ {cg.nx} × {cg.ny} cells"
            self.summary = (
                f"{_NAMES[cg.projection.kind]}: {cg.nx} × {cg.ny} cells, nominal "
                f"{cf.nominal_resolution_km(cg):.2f} km, cell size "
                f"{q['size_min']:.2f} to {q['size_max']:.2f} km."
            )
            fmt = lambda *keys: " / ".join(f"{q[key]:.2f}" for key in keys)
            coast = "of the ocean cells" if wet is not None else "no coastline data"
            sides = [s for s in diag.OBC_SIDES if runs[s]["runs"]]
            self.details = [
                ["Cells", f"{cg.nx} × {cg.ny} ({cg.nx * cg.ny:,})"],
                [
                    "Cell size, km (min / median / max)",
                    fmt("size_min", "size_median", "size_max"),
                ],
                ["Cell size ratio (max / min)", fmt("size_ratio")],
                ["Largest aspect ratio", fmt("aspect_max")],
                [
                    "Orthogonality, deg (median / p99 / max)",
                    fmt("ortho_median", "ortho_p99", "ortho_max"),
                ],
                ["Largest neighbour size ratio", fmt("max_neighbour_ratio")],
                ["Smallest cell", f"{ts['min_wet_dx'] / 1000:.2f} km ({coast})"],
                ["Time step", ts["mom6_hint"]],
                ["Open boundaries (from the coastline)", ", ".join(sides) or "none"],
            ]
        self.preview_errors, self.preview_warnings = list(errors), list(warnings)
        self.can_build = cg is not None and not errors

    def _send_land(self, window):
        """Natural Earth land and a graticule in a lon/lat window, as GeoJSON."""
        lon0, lon1, lat0, lat1 = window
        # Short of the poles, which Mercator can't draw
        window = self._land_window = (lon0, lon1, max(lat0, -89.99), min(lat1, 89.99))
        scale = "110m" if lat1 - lat0 > 60 else "50m"
        # 1-degree steps, so long straight lon/lat edges curve in the view
        land = [shapely.segmentize(p, 1.0) for p in cf.land_polygons(window, scale)]
        land = shapely.transform(shapely.MultiPolygon(land), lambda xy: xy.round(3))
        geometries = dict(
            land=json.loads(shapely.to_geojson(land)),
            graticule=dict(type="MultiLineString", coordinates=_graticule(window)),
        )
        features = [
            dict(type="Feature", properties=dict(kind=kind), geometry=geometry)
            for kind, geometry in geometries.items()
        ]
        self.land = json.dumps(dict(type="FeatureCollection", features=features))


def _lonlat(buffers):
    """Lon and lat lists from a float64 buffer of (lon, lat) pairs sent by the page."""
    lonlat = from_buffer(buffers[0], "<f8").reshape(-1, 2)
    return lonlat[:, 0].tolist(), lonlat[:, 1].tolist()


def _edited(o, lon, lat):
    """``(lon, lat, corners)`` of outline `o` as the page edited it: the vertices before
    and after the changed run keep their exact values, changed ones keep near their old
    longitudes, and each corner stays on its vertex (or goes, if that was deleted)."""
    old, new = np.column_stack([o.lon, o.lat]), np.column_stack([lon, lat])
    n0, n, m = len(old), len(new), min(len(old), len(new))

    def same(a, b):  # how many leading points match (lon modulo 360)
        d = np.abs(a - b)
        d[:, 0] = np.abs((a[:, 0] - b[:, 0] + 180.0) % 360.0 - 180.0)
        return int(np.argmin(np.r_[(d < 1e-7).all(axis=1), False]))

    head = same(old[:m], new[:m])
    tail = same(old[::-1][: m - head], new[::-1][: m - head])
    mid = new[head : n - tail]
    ref = o.lon[min(head, n0 - 1)] if n0 else 0.0
    mid_lon = ref + (mid[:, 0] - ref + 180.0) % 360.0 - 180.0
    lon = o.lon[:head] + mid_lon.tolist() + o.lon[n0 - tail :]
    lat = o.lat[:head] + mid[:, 1].tolist() + o.lat[n0 - tail :]
    shift = n - n0
    corners = [
        c + shift * (c >= n0 - tail)
        for c in o.corners
        if shift == 0 or not head <= c < n0 - tail
    ]
    return lon, lat, corners


def _within(inner, outer):
    """Whether lon/lat window `inner` lies inside window `outer`."""
    if not (outer[2] <= inner[2] and inner[3] <= outer[3]):
        return False
    width = outer[1] - outer[0]
    return width >= 360 or (inner[0] - outer[0]) % 360 + inner[1] - inner[0] <= width


def _graticule(window):
    """Meridians and parallels across a lon/lat window, about 6 to 12 of each."""
    lon0, lon1, lat0, lat1 = window
    step = lambda span: next((s for s in (1, 2, 5, 10, 15, 30) if span <= 12 * s), 30)
    a, b = step(lon1 - lon0), step(lat1 - lat0)
    lons, lats = np.linspace(lon0, lon1, 120), np.linspace(lat0, lat1, 60)
    meridians = np.arange(np.ceil(lon0 / a) * a, lon1, a)
    parallels = np.arange(np.ceil(lat0 / b) * b, lat1, b)
    lines = [np.column_stack([np.full(60, x), lats]) for x in meridians]
    lines += [np.column_stack([lons, np.full(120, y)]) for y in parallels]
    return [line.round(3).tolist() for line in lines]


def _check_name(name):
    name = str(name or "").strip()
    if not re.fullmatch(r"[\w.-]+", name):
        raise ValueError("Name the grid with letters, digits, '_', '-' or '.'.")
    return name
