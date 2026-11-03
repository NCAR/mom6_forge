"""Conformal orthogonal grids from an outline and 4 corners: two finite-element Laplace
solves for the logical coordinates xi and eta, whose Dirichlet energies set nx/ny for
square cells, then linear interpolation back to a uniform (xi, eta) supergrid, on the
mesh's own triangles.

The idea, a polygon with 4 marked corners mapped conformally onto a rectangle, follows
the orthogonal grid generators gridgen (Pavel Sakov, https://github.com/sakov/gridgen-c)
and pygridgen (Rob Hetland, Paul Hobson and contributors,
https://github.com/pygridgen/pygridgen). They compute the map with the
Schwarz-Christoffel CRDT method [1]_ (background in [2]_); this module solves Laplace
problems on a mesh that follows the outline instead.

References
----------
.. [1] T. A. Driscoll and S. A. Vavasis, "Numerical conformal mapping using cross-ratios
   and Delaunay triangulation", SIAM J. Sci. Comput. 19(6), 1783-1803, 1998.
.. [2] T. A. Driscoll and L. N. Trefethen, "Schwarz-Christoffel Mapping", Cambridge
   University Press, 2002.
"""

from dataclasses import dataclass
from functools import cached_property
import numpy as np
import scipy.sparse as sp
import shapely
from matplotlib.path import Path as MplPath
from pyproj import Geod, Proj
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LinearRing

KINDS = ("lcc", "merc", "tmerc", "stere")
BUILD_RASTER_POINTS = 250_000


@dataclass
class MapProjection:
    """A conformal map projection on the WGS84 ellipsoid.

    Attributes
    ----------
    kind : {"lcc", "merc", "tmerc", "stere"}
        Lambert conformal conic, Mercator, transverse Mercator or stereographic.
    lon_0, lat_0 : float
        Central longitude and latitude, degrees.
    lat_1, lat_2 : float, optional
        Standard parallels of "lcc"; `lat_1` is the true-scale latitude of "stere"
        and (optionally) "merc".
    proj4 : str
        The proj4 string of this projection (read-only).
    """

    kind: str
    lon_0: float
    lat_0: float
    lat_1: float | None = None
    lat_2: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown projection kind {str(self.kind)[:40]!r}; use one of {', '.join(KINDS)}"
            )

    @property
    def proj4(self):
        tail = "+datum=WGS84 +units=m"
        if self.kind == "lcc":
            return f"+proj=lcc +lat_1={self.lat_1} +lat_2={self.lat_2} +lat_0={self.lat_0} +lon_0={self.lon_0} {tail}"
        if self.kind == "merc":
            return f"+proj=merc +lon_0={self.lon_0} +lat_ts={self.lat_1 or 0.0} {tail}"
        if self.kind == "tmerc":
            return f"+proj=tmerc +lon_0={self.lon_0} +lat_0={self.lat_0} +k=1 {tail}"
        return f"+proj=stere +lat_0={self.lat_0} +lon_0={self.lon_0} +lat_ts={self.lat_1} {tail}"

    def cartopy(self):
        """The matching cartopy.crs.Projection, for plotting."""
        import cartopy.crs as ccrs

        if self.kind == "lcc":
            p = (self.lat_1, self.lat_2)
            return ccrs.LambertConformal(self.lon_0, self.lat_0, standard_parallels=p)
        if self.kind == "merc":
            return ccrs.Mercator(self.lon_0, latitude_true_scale=self.lat_1 or 0.0)
        if self.kind == "tmerc":
            return ccrs.TransverseMercator(self.lon_0, self.lat_0, scale_factor=1.0)
        return ccrs.Stereographic(
            self.lat_0, self.lon_0, true_scale_latitude=self.lat_1
        )

    @cached_property
    def _proj(self):
        # 70x quicker to make than a Transformer from EPSG:4326, with the same results
        return Proj(self.proj4)

    def to_xy(self, lon, lat):
        """``(x, y)`` in metres, projecting ``(lon, lat)`` degrees; same shape as the input."""
        return self._proj(lon, lat)

    def to_lonlat(self, x, y):
        """``(lon, lat)`` in degrees, unprojecting ``(x, y)`` metres; same shape as the input."""
        return self._proj(x, y, inverse=True)


def _lon_steps(lon):
    """Longitude step from each vertex to the next (ring closed), each in [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return (np.diff(np.append(lon, lon[0])) + 180.0) % 360.0 - 180.0


def _pole_contained(lon, lat):
    """90.0 or -90.0 if the ring winds round that pole (by its mean latitude), else None."""
    if abs(_lon_steps(lon).sum()) < 180.0:
        return None
    return 90.0 if np.mean(lat) > 0 else -90.0


def unwrap_lon(lon):
    """Longitudes shifted by multiples of 360 so no step across the dateline exceeds 180.

    Raises ValueError for an outline that winds round a pole.
    """
    lon = np.asarray(lon, dtype=float)
    steps = _lon_steps(lon)
    if abs(steps.sum()) > 180.0:
        raise ValueError(
            "outlines that contain a pole are not supported: keep the pole outside the outline"
        )
    return lon[0] + np.concatenate([[0.0], np.cumsum(steps[:-1])])


def projection_for(kind, lon, lat):
    """A projection of the given kind centred on an outline.

    "auto" picks polar stereographic above 65 degrees mean latitude, Mercator across or
    within 15 degrees of the equator, and otherwise Lambert conformal conic with standard
    parallels at 1/6 and 5/6 of the latitude span. Round a pole only "auto" and "stere"
    work, centred on the pole. (`Grid.from_outline` chooses its "auto" kind
    differently, by `size_ratios`.)

    Parameters
    ----------
    kind : {"auto", "lcc", "merc", "tmerc", "stere"}
        Projection kind.
    lon, lat : array_like
        Outline vertices (not closed), degrees.

    Returns
    -------
    MapProjection
        The centred projection.

    Raises
    ------
    ValueError
        For an unknown kind, a kind other than "auto" or "stere" round a pole, or "lcc"
        for an outline centred on the equator (its cone would be flat).
    """
    lat = np.asarray(lat, dtype=float)
    pole = _pole_contained(lon, lat)
    if pole is not None:
        if kind not in ("auto", "stere"):
            hemisphere = "North" if pole > 0 else "South"
            raise ValueError(
                f"this outline contains the {hemisphere} Pole: use Polar stereographic"
            )
        # A circular mean needs no unwrapping, so it works round a pole.
        rad = np.radians(np.asarray(lon, dtype=float))
        lon_0 = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
        return MapProjection("stere", float(lon_0), pole, lat_1=np.sign(pole) * 70.0)
    lon_0 = float((np.mean(unwrap_lon(lon)) + 180.0) % 360.0 - 180.0)
    mean_lat = float(np.mean(lat))
    equatorial = (lat.min() < 0 < lat.max()) or abs(mean_lat) < 15
    if kind == "auto":
        kind = "stere" if abs(mean_lat) > 65 else "merc" if equatorial else "lcc"
    if kind == "stere":
        pole = 90.0 if mean_lat >= 0 else -90.0
        return MapProjection("stere", lon_0, pole, lat_1=np.sign(pole) * 70.0)
    if kind == "merc":
        return MapProjection("merc", lon_0, 0.0)
    if kind == "tmerc":
        return MapProjection("tmerc", lon_0, mean_lat)
    if kind == "lcc":
        lat_1, lat_2 = lat.min() + np.array([1, 5]) * (lat.max() - lat.min()) / 6
        if abs(lat_1 + lat_2) < 1e-6:
            raise ValueError(
                "Lambert conformal needs an outline off the equator: use Mercator"
            )
        return MapProjection("lcc", lon_0, mean_lat, float(lat_1), float(lat_2))
    raise ValueError(
        f"unknown projection kind {str(kind)[:40]!r}; use auto or one of {', '.join(KINDS)}"
    )


def as_projection(projection, lon, lat):
    """Resolve a ``projection=`` argument to a `MapProjection`.

    Parameters
    ----------
    projection : None, str or MapProjection
        None or "auto" for `projection_for`'s choice, a kind name to centre that kind
        on the outline, or a `MapProjection`, returned as it is.
    lon, lat : array_like
        Outline vertices (not closed), degrees.

    Returns
    -------
    MapProjection
        The resolved projection.

    Raises
    ------
    ValueError
        For any other `projection`, or one other than stereographic round a pole.
    """
    if isinstance(projection, MapProjection):
        if projection.kind != "stere" and _pole_contained(lon, lat) is not None:
            projection_for(projection.kind, lon, lat)  # raises: use Polar stereographic
        return projection
    if projection is None or projection in ("auto",) + KINDS:
        return projection_for(projection or "auto", lon, lat)
    raise ValueError(
        f"projection must be None, 'auto', one of {', '.join(KINDS)} or a MapProjection, "
        f"not {str(projection)[:40]!r}"
    )


def _signed_area(x, y):
    """Shoelace area of a polygon, positive if it winds counter-clockwise."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_simple(x, y):
    """Whether the polygon ``(x, y)``, not closed, has no self-intersecting edges."""
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    return bool(LinearRing(xy).is_simple)


def orient_ccw(x, y, corners):
    """``(x, y, corners)`` reoriented counter-clockwise, reversing and remapping if needed."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    corners = list(corners)
    if _signed_area(x, y) > 0:
        return x, y, corners
    n = len(x)
    return x[::-1].copy(), y[::-1].copy(), [(n - 1 - c) % n for c in corners]


def interior_angles(x, y):
    """Interior angle at each vertex of a counter-clockwise polygon, degrees (above 180 at a reflex vertex)."""
    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    into = pts - np.roll(pts, 1, axis=0)
    out = np.roll(pts, -1, axis=0) - pts
    cross = into[:, 0] * out[:, 1] - into[:, 1] * out[:, 0]
    dot = into[:, 0] * out[:, 0] + into[:, 1] * out[:, 1]
    return 180.0 - np.degrees(np.arctan2(cross, dot))


def order_corners(x, y, corners):
    """Corner indices reordered: south-west (smallest x + y) first, then counter-clockwise."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    start = min(corners, key=lambda i: x[i] + y[i])
    return sorted(corners, key=lambda c: (c - start) % len(x))


@dataclass
class ConformalGrid:
    """A solved orthogonal supergrid, shape (2 ny + 1, 2 nx + 1), axis 0 south to north.

    Attributes
    ----------
    lon, lat, x, y : numpy.ndarray
        Supergrid coordinates: degrees, and projection metres.
    nx, ny : int
        Cell counts.
    projection : MapProjection
        The plane of the solve.
    aspect : float
        Conformal modulus of the outline (ny / nx for square cells).
    h : float
        Mesh spacing of the solve (finer round singular vertices), metres.
    corners : list of int
        Corner indices into the oriented (CCW, unwrapped) outline, south-west first.
    corner_angles : list of float, optional
        Outline interior angle at each corner, degrees.
    qlon, qlat : numpy.ndarray
        Cell-corner coordinates, shape (ny + 1, nx + 1) (read-only).
    """

    lon: np.ndarray
    lat: np.ndarray
    x: np.ndarray
    y: np.ndarray
    nx: int
    ny: int
    projection: MapProjection
    aspect: float
    h: float
    corners: list
    corner_angles: list | None = None

    @property
    def qlon(self):
        return self.lon[::2, ::2]

    @property
    def qlat(self):
        return self.lat[::2, ::2]


def outline_area_km2(lon, lat):
    """Geodesic area enclosed by the outline on the WGS84 ellipsoid, square kilometres."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return abs(Geod(ellps="WGS84").polygon_area_perimeter(lon, lat)[0]) / 1e6


def n_cells_for_resolution(lon, lat, resolution_km):
    """Cell count for square cells of a given mean size over an outline.

    Parameters
    ----------
    lon, lat : array_like
        Outline vertices (not closed), degrees.
    resolution_km : float
        Mean cell edge length, km.

    Returns
    -------
    int
        Outline area over `resolution_km` squared, at least 1.
    """
    n = outline_area_km2(lon, lat) / max(float(resolution_km), 1e-9) ** 2
    return max(1, int(round(n)))


def nominal_resolution_km(cg):
    """Edge length of a square cell with the grid's mean cell area, km."""
    ring = [np.s_[0, :], np.s_[1:, -1], np.s_[-1, -2::-1], np.s_[-2:0:-1, 0]]
    qlon, qlat = np.asarray(cg.qlon, dtype=float), np.asarray(cg.qlat, dtype=float)
    area_km2 = outline_area_km2(
        np.concatenate([qlon[s] for s in ring]), np.concatenate([qlat[s] for s in ring])
    )
    return float(np.sqrt(area_km2 / max(int(cg.nx) * int(cg.ny), 1)))


def _validate_and_orient(lon, lat, corners):
    """Check the inputs; return the outline unwrapped and CCW with its corners remapped."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    if lon.shape != lat.shape or lon.ndim != 1 or len(lon) < 4:
        raise ValueError("polygon needs at least 4 (lon, lat) vertices")
    corners = list(corners)
    if len(corners) != 4 or len(set(corners)) != 4:
        raise ValueError("need exactly 4 distinct corner indices")
    if any(c < 0 or c >= len(lon) for c in corners):
        raise ValueError(f"corner indices must be in [0, {len(lon)})")
    if len({(lon[c], lat[c]) for c in corners}) < 4:
        raise ValueError("two corners are at the same point: pick 4 different places")
    pole = _pole_contained(lon, lat)
    if pole is None:
        lon = unwrap_lon(lon)
        if not is_simple(lon, lat):
            raise ValueError("polygon is not simple (edges self-intersect)")
        return orient_ccw(lon, lat, corners)
    # Round a pole, judge simplicity and winding on a nominal polar plane.
    x, y = MapProjection("stere", 0.0, pole, np.sign(pole) * 70.0).to_xy(lon, lat)
    if not is_simple(x, y):
        raise ValueError("polygon is not simple (edges self-intersect)")
    _, _, corners = orient_ccw(x, y, corners)
    if _signed_area(x, y) <= 0:
        lon, lat = lon[::-1].copy(), lat[::-1].copy()
    return lon, lat, corners


def _sides(P, corners, h):
    """Each side (0..3 = S, E, N, W: corner k to k + 1) as ``(points, arclength)``: its
    vertices and points between them about h apart, both corners included."""
    n, sides = len(P), []
    for k in range(4):
        a = corners[k]
        V = P[np.arange(a, a + (corners[(k + 1) % 4] - a) % n + 1) % n]
        V = V[np.r_[True, np.any(np.diff(V, axis=0) != 0, axis=1)]]
        seg = np.hypot(*np.diff(V, axis=0).T)
        m = np.ceil(seg / h).astype(int).clip(1)
        e = np.repeat(np.arange(len(seg)), m)
        t = (np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)) / m[e]
        pts = V[e] + t[:, None] * (V[e + 1] - V[e])
        s = np.append(np.r_[0.0, np.cumsum(seg)][e] + t * seg[e], seg.sum())
        sides.append((np.vstack([pts, V[-1:]]), s))
    return sides


def _mesh(P, sides, h):
    """Mesh nodes (corners, side points, then lattice points inside, h apart), triangles
    and each side's node ids."""
    inner = [pts[1:-1] for pts, _ in sides]
    ends = np.cumsum([4] + [len(p) for p in inner])
    ids = [np.r_[k, np.arange(ends[k], ends[k + 1]), (k + 1) % 4] for k in range(4)]
    lo, ring, path = P.min(axis=0) + h / 2, LinearRing(P), MplPath(P)
    X, Y = np.meshgrid(*(np.arange(a, b, h) for a, b in zip(lo, P.max(axis=0))))
    xy, d = np.column_stack([X.ravel(), Y.ravel()]), np.zeros(X.size)
    ins = path.contains_points(xy)
    d[ins] = shapely.distance(shapely.points(xy[ins]), ring) / h
    d = d.reshape(X.shape)
    use = d > 0.5
    num = np.cumsum(use).reshape(X.shape) + ends[-1] - 1
    deep = d > 1.5  # squares keep h clear of the outline
    full = deep[:-1, :-1] & deep[:-1, 1:] & deep[1:, :-1] & deep[1:, 1:]
    q = [num[:-1, :-1][full], num[:-1, 1:][full], num[1:, 1:][full], num[1:, :-1][full]]
    squares = np.r_[np.column_stack(q[:3]), np.column_stack([q[0], q[2], q[3]])]
    nodes = np.vstack([[pts[0] for pts, _ in sides], *inner, xy[use.ravel()]])
    # No other point is in a square's circumcircle, so the squares' edges are Delaunay
    # edges and no triangle crosses one: drop those outside the outline or in a square.
    near = np.r_[np.arange(ends[-1]), num[use & (d < 3.0)]]
    tri = near[Delaunay(nodes[near]).simplices]
    c = nodes[tri].mean(axis=1)
    i, j = (np.floor((c - lo) / h).astype(int) + 1).T
    pad = np.pad(full, 1)
    in_square = pad[j.clip(0, pad.shape[0] - 1), i.clip(0, pad.shape[1] - 1)]
    d1, d2 = nodes[tri[:, 1]] - nodes[tri[:, 0]], nodes[tri[:, 2]] - nodes[tri[:, 0]]
    flat = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) < 1e-9 * h * h
    keep = ~flat & ~in_square & path.contains_points(c)
    return nodes, np.vstack([tri[keep], squares]), ids


def _stiffness(nodes, tri):
    """P1 finite-element stiffness matrix of the Laplacian on the triangles."""
    e = nodes[np.roll(tri, -1, axis=1)] - nodes[np.roll(tri, 1, axis=1)]
    area2 = np.abs(e[:, 0, 0] * e[:, 1, 1] - e[:, 0, 1] * e[:, 1, 0])
    k = np.einsum("tid,tjd->tij", e, e) / (2.0 * area2)[:, None, None]
    rows, cols = np.repeat(tri, 3, axis=1), np.tile(tri, 3)
    n = len(nodes)
    K = sp.csr_matrix((k.ravel(), (rows.ravel(), cols.ravel())), shape=(n, n))
    K.eliminate_zeros()  # the hypotenuses of right triangles
    return K


def _harmonic(K, low, high):
    """Laplace's equation on the mesh: 0 on nodes `low`, 1 on `high`, zero flux elsewhere.

    Returns the nodal values (NaN at nodes in no triangle) and the Dirichlet energy.
    """
    u, fixed = np.zeros(K.shape[0]), np.zeros(K.shape[0], dtype=bool)
    fixed[low] = fixed[high] = True
    u[high] = 1.0
    free = ~fixed & (K.diagonal() > 0)
    A, B = K[free][:, free], K[free][:, fixed]
    u[free] = spsolve(A.tocsc(), -(B @ u[fixed]))
    energy = float(u @ (K @ u))
    u[K.diagonal() == 0] = np.nan
    return u, energy


def _solve_mesh(lon, lat, corners, n_cells, projection, raster_points):
    """The mesh, the two Laplace fields on it and the cell counts they give."""
    lon, lat, corners = _validate_and_orient(lon, lat, corners)
    projection = as_projection(projection, lon, lat)
    x, y = (np.asarray(a, dtype=float) for a in projection.to_xy(lon, lat))
    P, area = np.column_stack([x, y]), abs(_signed_area(x, y))
    # A sliver (nearly collinear vertices) would need billions of points along its edges
    if not np.hypot(*(P - np.roll(P, 1, axis=0)).T).sum() < 200 * np.sqrt(area):
        raise ValueError("the outline has almost no area in this projection")
    h = float(np.sqrt(area / raster_points))
    corners = order_corners(x, y, corners)
    corner_angles = [float(a) for a in interior_angles(x, y)[corners]]
    sides = _sides(P, corners, h)
    nodes, tri, ids = _mesh(P, sides, h)
    K = _stiffness(nodes, tri)
    xi, e_xi = _harmonic(K, ids[3], ids[1])
    eta, e_eta = _harmonic(K, ids[0], ids[2])
    aspect = float(np.sqrt(e_xi / e_eta))
    # Floor both so a tiny, zero or NaN n_cells never gives an empty grid.
    nx = max(1, int(round(np.sqrt(max(n_cells, 1) / aspect))))
    ny = max(1, int(round(nx * aspect)))
    f = dict(nodes=nodes, tri=tri, sides=sides, ids=ids, xi=xi, eta=eta)
    f.update(nx=nx, ny=ny, aspect=aspect, projection=projection, h=h, lon=lon, lat=lat)
    return f | dict(P=P, corners=corners, corner_angles=corner_angles)


def _sample(xy, tri, values, lo, step, shape):
    """Nodal `values` interpolated linearly at the lattice points ``lo + step * (i, j)``
    (array shape `shape`, index j first) in the triangles of nodes `xy` holding them;
    NaN at points in no triangle."""
    g = ((xy - lo) / step)[tri]  # triangle vertices in lattice units
    a = np.ceil(g.min(axis=1) - 1e-9).astype(int).clip(0)
    top = np.minimum(np.floor(g.max(axis=1) + 1e-9), np.array(shape[::-1]) - 1)
    n = (top.astype(int) - a + 1).clip(0)
    count = n[:, 0] * n[:, 1]
    out = np.full(tuple(shape) + values.shape[1:], np.nan)
    # About a million candidate points at a time
    cuts = np.searchsorted(np.cumsum(count), np.arange(1 << 20, count.sum(), 1 << 20))
    for t in np.split(np.arange(len(tri)), cuts):
        c = count[t]
        t = np.repeat(t, c)
        k = np.arange(len(t)) - np.repeat(np.cumsum(c) - c, c)
        p = a[t] + np.column_stack([k % n[t, 0], k // n[t, 0]])
        e1, e2, d = g[t, 1] - g[t, 0], g[t, 2] - g[t, 0], p - g[t, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            w1 = (d[:, 0] * e2[:, 1] - d[:, 1] * e2[:, 0]) / det
            w2 = (e1[:, 0] * d[:, 1] - e1[:, 1] * d[:, 0]) / det
        hit = (w1 >= -1e-9) & (w2 >= -1e-9) & (w1 + w2 <= 1 + 1e-9) & (det != 0)
        w = np.column_stack([1 - w1 - w2, w1, w2])[hit]
        v = np.einsum("pk,pk...->p...", w, values[tri[t[hit]]])
        out[p[hit, 1], p[hit, 0]] = v
    return out


def solve(
    lon,
    lat,
    corners,
    n_cells=10_000,
    projection=None,
    raster_points=BUILD_RASTER_POINTS,
):
    """Solve the orthogonal supergrid of an outline and its 4 chosen corners.

    Parameters
    ----------
    lon, lat : array_like
        Outline vertices (not closed), degrees; at least 4.
    corners : sequence of int
        The 4 corner vertex indices, in any order.
    n_cells : int, optional
        Target cell count nx * ny, split between nx and ny for square cells.
    projection : None, str or MapProjection, optional
        Anything `as_projection` accepts; None: `projection_for`'s "auto" choice.
    raster_points : int, optional
        Approximate mesh size.

    Returns
    -------
    ConformalGrid
        The solved grid.

    Raises
    ------
    ValueError
        For fewer than 4 vertices, not exactly 4 distinct valid corners, a
        self-intersecting outline or an unusable projection.
    """
    f = _solve_mesh(lon, lat, corners, n_cells, projection, raster_points)
    nx, ny, xi, eta, nodes = f["nx"], f["ny"], f["xi"], f["eta"], f["nodes"]
    # Interpolate (x, y) at uniform (xi, eta) in the mesh triangles' images, then put
    # each outer row and column on its side, where xi (S, N) or eta (E, W) takes the
    # uniform values.
    along = [np.linspace(0, 1, 2 * n + 1) for n in (nx, ny)]
    uv, step = np.column_stack([xi, eta]), 0.5 / np.array([nx, ny])
    XY = _sample(uv, f["tri"], nodes, 0.0, step, (2 * ny + 1, 2 * nx + 1))
    bad = np.isnan(XY[..., 0])
    if bad.any():  # points in no triangle's image: the nearest node's
        ok = np.isfinite(xi)
        near = cKDTree(uv[ok]).query(np.column_stack(np.nonzero(bad)[::-1]) * step)[1]
        XY[bad] = nodes[ok][near]
    edges = [np.s_[0, :], np.s_[:, -1], np.s_[-1, :], np.s_[:, 0]]
    for k, ((side, s), ids) in enumerate(zip(f["sides"], f["ids"])):
        u, a = (xi, eta)[k % 2][ids], along[k % 2]
        if k >= 2:  # N and W run backwards
            u, a = 1.0 - u, 1.0 - a
        m = np.isfinite(u)
        at = np.interp(a, np.maximum.accumulate(np.clip(u[m], 0.0, 1.0)), s[m])
        XY[edges[k]] = np.column_stack([np.interp(at, s, side[:, d]) for d in (0, 1)])
    # The grid corners on the outline's, whatever the field's overshoot round them
    XY[[0, 0, -1, -1], [0, -1, -1, 0]] = f["P"][f["corners"]]
    x, y = XY[..., 0], XY[..., 1]
    lon_s, lat_s = f["projection"].to_lonlat(x, y)
    # Keep nodes just off the poles, where longitude is undefined.
    lat_s = np.clip(lat_s, -90.0 + 1e-6, 90.0 - 1e-6)
    keys = ("nx", "ny", "projection", "aspect", "h", "corners", "corner_angles")
    return ConformalGrid(np.asarray(lon_s), lat_s, x, y, *map(f.get, keys))


def _cell_geometry(cg):
    """Corner nodes (qx, qy), cell size (km) and cell aspect of `cg`, memoised on it."""
    if getattr(cg, "_geometry", None) is None:
        qx, qy = cg.x[::2, ::2], cg.y[::2, ::2]
        factors = Proj(cg.projection.proj4).get_factors(cg.qlon, cg.qlat)
        k = np.asarray(factors.meridional_scale)
        k_i, k_j = 0.5 * (k[:, 1:] + k[:, :-1]), 0.5 * (k[1:, :] + k[:-1, :])
        dx = np.hypot(np.diff(qx, axis=1), np.diff(qy, axis=1)) / k_i
        dy = np.hypot(np.diff(qx, axis=0), np.diff(qy, axis=0)) / k_j
        dx = 0.5 * (dx[:-1, :] + dx[1:, :])
        dy = 0.5 * (dy[:, :-1] + dy[:, 1:])
        cg._geometry = (qx, qy, np.sqrt(dx * dy) / 1000.0, np.maximum(dx / dy, dy / dx))
    return cg._geometry


def size_ratios(lon, lat, corners):
    """How much the true cell size varies in each projection kind an outline can use.

    A quick coarse solve (256 cells) in each kind: the smallest ratio marks the
    recommended kind, which ``projection="auto"`` picks in
    `mom6_forge.grid.Grid.from_outline`. The ratio is of percentiles, so the few cells
    at a sharp corner, which shrink as the grid is refined, don't decide it.

    Parameters
    ----------
    lon, lat : array_like
        Outline vertices (not closed), degrees.
    corners : sequence of int
        The 4 corner vertex indices.

    Returns
    -------
    dict
        {kind: 99th over 1st percentile of cell size} for each kind whose solve works;
        empty for an outline that can't be solved.
    """
    ratios = {}
    for kind in KINDS:
        try:
            cg = solve(lon, lat, corners, 256, kind, raster_points=1_000)
        except Exception:  # unusable here, e.g. Lambert on the equator
            continue
        low, high = np.percentile(_cell_geometry(cg)[2], [1, 99])
        ratios[kind] = float(high / low)
    return ratios
