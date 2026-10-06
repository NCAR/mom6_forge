"""Quality, time-step and open-boundary diagnostics for a conformal grid.

Lengths are measured in the projection plane and corrected to true lengths with the
local scale factor (one factor serves every direction, since the map is conformal).
"""

import numpy as np
from pyproj import Geod, Proj
from scipy import ndimage
from shapely import contains_xy
from mom6_forge import _conformal as cf

OBC_SIDES = ("south", "east", "north", "west")
STATUS_COLORS = {0: "#707070", 1: "#e08214", 2: "#d7191c"}
_SIDES = dict(zip(OBC_SIDES, (np.s_[0, :], np.s_[:, -1], np.s_[-1, :], np.s_[:, 0])))
_GEOD = Geod(ellps="WGS84")


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


def _node_orthogonality(qx, qy):
    """Deviation from 90 degrees between grid lines at each node, one-sided at the edges."""
    ax, ay, bx, by = (np.gradient(q, axis=k) for k in (1, 0) for q in (qx, qy))
    cosang = (ax * bx + ay * by) / (np.hypot(ax, ay) * np.hypot(bx, by))
    return np.abs(90.0 - np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def _neighbour_ratio(size_km, wet=None):
    """Largest size ratio of each cell to its (up to 4) neighbours, ocean ones only if
    `wet` is given (NaN for a cell with none)."""
    size_km = np.where(True if wet is None else wet, size_km, np.nan)
    sides = np.full((4,) + size_km.shape, np.nan)
    a, b = size_km[:, :-1], size_km[:, 1:]
    sides[0, :, 1:] = sides[1, :, :-1] = np.maximum(a, b) / np.minimum(a, b)
    a, b = size_km[:-1, :], size_km[1:, :]
    sides[2, 1:, :] = sides[3, :-1, :] = np.maximum(a, b) / np.minimum(a, b)
    return np.fmax.reduce(sides, axis=0)


def _ortho_boundary_rings(cg, qx, qy):
    """Node rings next to the outline that orthogonality checks leave out."""
    # Cells beside sharp outline vertices are skewed in the exact map too: skip ~2 h.
    dx = np.median(np.hypot(np.diff(qx, axis=1), np.diff(qy, axis=1)))
    rings = int(np.ceil(0.75 + 2.0 * cg.h / dx))
    return max(1, min(rings, (qx.shape[0] - 1) // 2, (qx.shape[1] - 1) // 2))


def cell_metrics(
    cg,
    wet=None,
    ortho_warn=3,
    ortho_bad=10,
    ratio_warn=1.3,
    ratio_bad=2.0,
    aspect_warn=2.0,
    aspect_bad=4.0,
):
    """Per-cell quality checks and the status they give.

    Parameters
    ----------
    cg : mom6_forge._conformal.ConformalGrid
        A solved grid.
    wet : numpy.ndarray of bool, optional
        (ny, nx) ocean mask: land cells then have status 0.
    ortho_warn, ortho_bad : float, optional
        Deviation from 90 degrees at a corner node above which a cell is warn / bad.
    ratio_warn, ratio_bad : float, optional
        Size ratio to a neighbouring cell above which a cell is warn / bad.
    aspect_warn, aspect_bad : float, optional
        Cell aspect ratio above which a cell is warn / bad.

    Returns
    -------
    dict
        (ny, nx) arrays ``ortho`` (worst node, degrees; nodes near the outline are
        exempt), ``ratio`` (to ocean neighbours only, given `wet`), ``aspect``,
        ``size`` (km), ``folded`` (cells turned inside out, always bad) and
        ``status`` (0 ok, 1 warn, 2 bad: the worst check), plus ``thresholds``, the
        six values used.
    """
    qx, qy, size_km, aspect = _cell_geometry(cg)
    node = _node_orthogonality(qx, qy)
    n = _ortho_boundary_rings(cg, qx, qy)
    node[:n, :] = node[-n:, :] = node[:, :n] = node[:, -n:] = 0.0
    ortho = np.maximum(node[:-1, :-1], node[:-1, 1:])
    ortho = np.maximum(ortho, np.maximum(node[1:, :-1], node[1:, 1:]))
    ratio = _neighbour_ratio(size_km, wet)
    # Twice the signed area (diagonal cross product): the wrong sign means folded
    ax, ay = qx[1:, 1:] - qx[:-1, :-1], qy[1:, 1:] - qy[:-1, :-1]
    area = ax * (qy[1:, :-1] - qy[:-1, 1:]) - ay * (qx[1:, :-1] - qx[:-1, 1:])
    folded = ~(area * np.sign(np.nansum(area)) > 0)
    bad = folded | (ortho > ortho_bad) | (ratio > ratio_bad) | (aspect > aspect_bad)
    warn = (ortho > ortho_warn) | (ratio > ratio_warn) | (aspect > aspect_warn)
    th = dict(ortho_warn=ortho_warn, ortho_bad=ortho_bad, ratio_warn=ratio_warn)
    th.update(ratio_bad=ratio_bad, aspect_warn=aspect_warn, aspect_bad=aspect_bad)
    ocean = True if wet is None else np.asarray(wet, dtype=bool)
    status = np.where(bad, 2, np.where(warn, 1, 0)) * ocean
    m = dict(ortho=ortho, ratio=ratio, aspect=aspect, size=size_km, folded=folded)
    return m | dict(status=status, thresholds=th)


def grid_quality(cg, wet=None, metrics=None):
    """Cell size, shape, orthogonality and corner-angle summary of a solved grid.

    Parameters
    ----------
    cg : mom6_forge._conformal.ConformalGrid
        A solved grid.
    wet : numpy.ndarray of bool, optional
        (ny, nx) ocean mask: orthogonality is then judged at wet nodes only, against
        5 degrees instead of 3.
    metrics : dict, optional
        ``cell_metrics(cg)``, if already computed.

    Returns
    -------
    dict
        ``nx, ny, n_cells``; cell size in km (``size_min, size_median, size_max,
        size_ratio``); ``aspect_max``; node orthogonality in degrees (``ortho_median,
        ortho_p99, ortho_max``, over the nodes `cell_metrics` judges);
        ``max_neighbour_ratio``; ``ortho_warn_threshold``; and ``corner_angles``
        ({corner 1-4: interior angle in degrees}).
    """
    qx, qy, size_km, aspect = _cell_geometry(cg)
    ratio = _neighbour_ratio(size_km) if metrics is None else metrics["ratio"]
    n = _ortho_boundary_rings(cg, qx, qy)
    ortho = _node_orthogonality(qx, qy)[n:-n, n:-n]
    threshold = 3.0
    if wet is not None:
        node_wet = np.zeros((cg.ny + 1, cg.nx + 1), dtype=bool)
        for j, i in ((0, 0), (0, 1), (1, 0), (1, 1)):
            node_wet[j : j + cg.ny, i : i + cg.nx] |= np.asarray(wet, dtype=bool)
        if node_wet[n:-n, n:-n].any():
            ortho = ortho[node_wet[n:-n, n:-n]]
        threshold = 5.0
    return {
        "nx": cg.nx,
        "ny": cg.ny,
        "n_cells": cg.nx * cg.ny,
        "size_min": float(size_km.min()),
        "size_median": float(np.median(size_km)),
        "size_max": float(size_km.max()),
        "size_ratio": float(size_km.max() / size_km.min()),
        "aspect_max": float(aspect.max()),
        "ortho_median": float(np.median(ortho)),
        "ortho_p99": float(np.percentile(ortho, 99)),
        "ortho_max": float(ortho.max()),
        "max_neighbour_ratio": float(np.fmax.reduce(ratio, axis=None)),
        "ortho_warn_threshold": threshold,
        "corner_angles": dict(enumerate(map(float, cg.corner_angles or []), start=1)),
    }


def size_ratios(lon, lat, corners):
    """How much the true cell size varies in each projection kind an outline can use.

    A quick coarse solve (256 cells) in each kind: the smallest ratio marks the
    recommended kind, which ``projection="auto"`` picks in `GridSketcher` and
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
    for kind in cf.KINDS:
        try:
            cg = cf.solve(lon, lat, corners, 256, kind, raster_points=1_000)
        except Exception:  # unusable here, e.g. Lambert on the equator
            continue
        low, high = np.percentile(_cell_geometry(cg)[2], [1, 99])
        ratios[kind] = float(high / low)
    return ratios


def _ocean(lon, lat, scale):
    """True off Natural Earth land at each point, or None if the land data can't load."""
    try:
        lon = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
        lat = np.asarray(lat, dtype=float)
        on_land = contains_xy(cf._land_geometry(scale), lon.ravel(), lat.ravel())
        return ~on_land.reshape(lon.shape)
    except Exception:
        return None


def _cell_centers(grid):
    """(ny, nx) cell-centre lon/lat of a `Grid` (tlon/tlat) or a ConformalGrid supergrid."""
    if getattr(grid, "tlon", None) is not None:
        return np.asarray(grid.tlon, dtype=float), np.asarray(grid.tlat, dtype=float)
    lon, lat = np.asarray(grid.lon, dtype=float), np.asarray(grid.lat, dtype=float)
    return lon[1::2, 1::2], lat[1::2, 1::2]


def ocean_mask(cg, scale="50m"):
    """Coastline-based ocean mask of a solved grid's cells.

    Parameters
    ----------
    cg : mom6_forge._conformal.ConformalGrid
        A solved grid.
    scale : {"110m", "50m", "10m"}, optional
        Natural Earth scale (the land the widget draws).

    Returns
    -------
    numpy.ndarray of bool or None
        (ny, nx), True where the cell centre is off land; None if the land data
        can't be loaded. Never raises.
    """
    return _ocean(*_cell_centers(cg), scale)


def timestep_limits(
    grid,
    wet=None,
    coupling_dt=3600.0,
    ice_speed=1.0,
    ocean_speed=1.0,
    mom6_dt=900.0,
    ww3=True,
):
    """MOM6 and CICE (and WW3, for reference) time-step limits set by the smallest cell.

    Parameters
    ----------
    grid : mom6_forge._conformal.ConformalGrid or mom6_forge.grid.Grid
        A solved grid (a `Grid` uses its true ``tarea``).
    wet : array_like of bool, optional
        (ny, nx) ocean mask; the CFL numbers then use the smallest wet cell.
    coupling_dt : float, optional
        MOM6-CICE coupling interval, seconds.
    ice_speed, ocean_speed : float, optional
        Nominal ice and ocean speeds, m/s.
    mom6_dt : float, optional
        MOM6 baroclinic time step, seconds.
    ww3 : bool, optional
        Also compute the WW3 numbers (imports `mom6_forge.topo`, and so xesmf).

    Returns
    -------
    dict
        ``min_dx, min_wet_dx`` (metres), ``metric`` ("sqrt(area)"), the echoed
        ``coupling_dt, ice_speed, ocean_speed, mom6_dt``, ``cice_cfl``,
        ``cice_max_coupling_dt`` (largest coupling interval with CICE CFL <= 0.5),
        ``mom6_cfl``, ``mom6_hint`` (one line) and ``ww3`` (a dict, or None).
    """
    tarea = getattr(grid, "tarea", None)
    if tarea is not None:
        size_m = np.sqrt(np.asarray(tarea, dtype=float))
    else:
        size_m = _cell_geometry(grid)[2] * 1000.0
    min_dx = min_wet_dx = float(size_m.min())
    if wet is not None and np.asarray(wet, dtype=bool).any():
        min_wet_dx = float(size_m[np.asarray(wet, dtype=bool)].min())
    mom6_cfl = ocean_speed * mom6_dt / min_wet_dx
    comfort = "comfortable" if mom6_cfl <= 0.5 else "keep DT here or add sub-cycling"
    ww3_ref = None
    if ww3:
        from mom6_forge.topo import ww3_timesteps_from_spacing

        ww3_ref = ww3_timesteps_from_spacing(min_wet_dx, coupling_dt)
    return {
        "min_dx": min_dx,
        "min_wet_dx": min_wet_dx,
        "metric": "sqrt(area)",
        "coupling_dt": coupling_dt,
        "ice_speed": ice_speed,
        "ocean_speed": ocean_speed,
        "mom6_dt": mom6_dt,
        "cice_cfl": ice_speed * coupling_dt / min_wet_dx,
        "cice_max_coupling_dt": 0.5 * min_wet_dx / ice_speed,
        "mom6_cfl": mom6_cfl,
        "mom6_hint": f"MOM6 CFL {mom6_cfl:.2f} at DT={mom6_dt:.0f} s ({ocean_speed:.1f} m/s): {comfort}",
        "ww3": ww3_ref,
    }


def open_boundary_runs(
    grid, ocean=None, min_cells=4, min_fraction=0.03, many_transitions=4, scale="50m"
):
    """Ocean/land split of each side's boundary cells and the open-water runs along it.

    Sides follow `mom6_forge.grid.Grid`: south is cell row 0, north row ny - 1, west
    column 0 and east column nx - 1.

    Parameters
    ----------
    grid : mom6_forge._conformal.ConformalGrid or mom6_forge.grid.Grid
        A solved grid.
    ocean : numpy.ndarray of bool, optional
        (ny, nx) ocean mask (a real wet mask). By default it is estimated from the
        coastline; if that fails, every side is all ocean and ``ocean_known`` is False.
    min_cells : int, optional
        A run shorter than this is "tiny", unless it spans the whole side.
    min_fraction : float, optional
        A run shorter than this fraction of its side's length is also "tiny".
    many_transitions : int, optional
        A side with at least this many land/ocean changes is "alternating".
    scale : {"110m", "50m", "10m"}, optional
        Natural Earth scale of the coastline estimate.

    Returns
    -------
    dict
        ``ocean_known`` plus, for each name in `OBC_SIDES`, a dict with ``ocean``
        (bool along the side), ``length_km``, ``n_transitions``, ``alternating`` and
        ``runs``: a list of ``{start, stop, length_cells, length_km, tiny, lon, lat}``
        (cell indices along the side, stop exclusive; lon/lat of the middle cell).
    """
    clon, clat = _cell_centers(grid)
    if ocean is None:
        ocean = _ocean(clon, clat, scale)
    result = {"ocean_known": ocean is not None}
    ocean = np.ones(clon.shape, bool) if ocean is None else np.array(ocean, bool)
    for side, s in _SIDES.items():
        side_ocean, lon, lat = ocean[s], clon[s], clat[s]
        # Each cell's width along the side: half of each neighbouring gap, or an end gap.
        width_m = np.zeros(len(lon))
        if len(lon) > 1:
            gap_m = np.asarray(_GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2])
            width_m[0], width_m[-1] = gap_m[0], gap_m[-1]
            width_m[1:-1] = 0.5 * (gap_m[:-1] + gap_m[1:])
        width_km = width_m / 1000.0
        length_km = float(width_km.sum())
        edges = np.flatnonzero(np.diff(np.concatenate(([0], side_ocean, [0]))))
        runs = []
        for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist()):
            run_km = float(width_km[start:stop].sum())
            mid = (start + stop - 1) // 2
            short = stop - start < min_cells and stop - start < len(side_ocean)
            tiny = short or (length_km > 0 and run_km < min_fraction * length_km)
            run = dict(
                start=start, stop=stop, length_cells=stop - start, length_km=run_km
            )
            runs.append(run | dict(tiny=tiny, lon=float(lon[mid]), lat=float(lat[mid])))
        n_transitions = int(np.sum(side_ocean[1:] != side_ocean[:-1]))
        result[side] = {
            "ocean": side_ocean,
            "length_km": length_km,
            "n_transitions": n_transitions,
            "alternating": n_transitions >= many_transitions,
            "runs": runs,
        }
    return result


def warnings(q):
    """Plain-language warnings for a `grid_quality` and/or `timestep_limits` dict.

    The summary-dict counterpart of `quality_messages`, which also says where.

    Parameters
    ----------
    q : dict
        Either dict, or both merged; checks whose keys are missing are skipped.

    Returns
    -------
    list of str
        One sentence per problem.
    """
    msgs = []
    if "size_ratio" in q and q["size_ratio"] > 50:
        msgs.append(
            f"Cell sizes vary by {q['size_ratio']:.0f}x "
            f"({q['size_min']:.2f}-{q['size_max']:.2f} km)."
        )
    for corner, angle in q.get("corner_angles", {}).items():
        if angle > 170:
            msgs.append(
                f"Corner {corner} is nearly flat: its corner cell degenerates; "
                "consider masking it."
            )
    if "ortho_p99" in q and q["ortho_p99"] > q.get("ortho_warn_threshold", 3.0):
        msgs.append(
            f"Orthogonality: 99th-percentile deviation from square is {q['ortho_p99']:.1f} deg."
        )
    if "max_neighbour_ratio" in q and q["max_neighbour_ratio"] > 1.3:
        msgs.append(
            f"Abrupt cell-size change: neighbouring cells differ by up to {q['max_neighbour_ratio']:.2f}x."
        )
    if q.get("cice_cfl", 0.0) > 0.5:
        msgs.append(
            f"Smallest wet cell {q['min_wet_dx'] / 1000.0:.1f} km: CICE CFL "
            f"{q['cice_cfl']:.2f} at {q['coupling_dt']:.0f} s coupling "
            f"(ice {q['ice_speed']:.1f} m/s); mask or enlarge the tiny cells, or couple "
            f"at {q['cice_max_coupling_dt']:.0f} s or less."
        )
    return msgs


def _place(j, i, ny, nx):
    """Near which corner (numbered as on the map) or side cell (j, i) lies, if any."""
    t = 0.2 * min(nx, ny)  # cells
    near = [j < t, nx - 1 - i < t, ny - 1 - j < t, i < t]
    corners = [k + 1 for k in range(4) if near[k - 1] and near[k]]
    sides = [side for side, on in zip(OBC_SIDES, near) if on]
    if corners:
        return f"near corner {corners[0]}"
    return f"near the {sides[0]} side" if sides else "in the interior"


def _and(items):
    *head, last = map(str, items)
    return f"{', '.join(head)} and {last}" if head else last


def _lonlat(lon, lat):
    lon, lat = (float(lon) + 180.0) % 360.0 - 180.0, float(lat)
    return f"{abs(lat):.2f}{'NS'[lat < 0]} {abs(lon):.2f}{'EW'[lon < 0]}"


def _areas(text, value, flagged, fmt, lon, lat, top=3):
    """One sentence on the flagged cells of a kind: how many, in how many areas
    (8-connected groups), and the worst cell of each of the `top` worst areas."""
    labels, n_areas = ndimage.label(flagged, np.ones((3, 3)))
    value = np.where(flagged, value, -np.inf)
    pos = ndimage.maximum_position(value, labels, np.arange(1, n_areas + 1))
    pos = sorted(pos, key=lambda p: -value[p])[:top]
    spots = [f"{fmt.format(value[p])}at {_lonlat(lon[p], lat[p])}" for p in pos]
    n, where = int(flagged.sum()), _place(*pos[0], *flagged.shape)
    head = f"{text}: {n:,} cell{'s' * (n > 1)}"
    head += f" in {n_areas:,} areas" if n_areas > 1 else ""
    lead = ("worst " if n > 1 else "") if fmt else ("one " if n_areas > 1 else "")
    first = lead + spots[0].replace("at ", f"{where} (", 1) + ")"
    parts = [head, first] if fmt or n_areas > 1 else [f"{head} {first}"]
    parts += [f"also {_and(spots[1:])}"] if len(spots) > 1 else []
    more = n_areas - len(pos)
    parts += [f"and {more:,} more area{'s' * (more > 1)}"] if more > 0 else []
    return ", ".join(parts) + "."


def quality_messages(cg, wet=None, metrics=None, runs=None):
    """Errors and warnings about a solved grid's ocean cells, one sentence per kind.

    Parameters
    ----------
    cg : mom6_forge._conformal.ConformalGrid
        A solved grid.
    wet : numpy.ndarray of bool, optional
        (ny, nx) ocean mask: land cells are left out. Default: every cell counts.
    metrics : dict, optional
        ``cell_metrics(cg)``, if already computed.
    runs : dict, optional
        ``open_boundary_runs`` of the grid, to add its open-boundary warnings.

    Returns
    -------
    errors, warnings : list of str
        Errors are folded cells, which make the grid unusable. Warnings are cell-size
        jumps, non-orthogonal or stretched cells (each with how many cells in how many
        areas, and where the worst cells of the 3 worst areas are), crowding, nearly
        flat corners and open-boundary problems.
    """
    m = cell_metrics(cg, wet) if metrics is None else metrics
    ny, nx = m["size"].shape
    ocean = np.ones((ny, nx), bool) if wet is None else np.asarray(wet, dtype=bool)
    th, errors, warns = m["thresholds"], [], []
    lon, lat = _cell_centers(cg)
    checks = [(errors, "Folded (inside-out) cells", "folded", "")]
    checks += [(warns, "Abrupt cell-size change", "ratio", "{:.2f}x ")]
    checks += [(warns, "Non-orthogonal cells", "ortho", "{:.1f} deg ")]
    checks += [(warns, "Stretched cells", "aspect", "aspect {:.1f} ")]
    for out, text, key, fmt in checks:
        flagged = ocean & (m[key] if key == "folded" else m[key] > th[f"{key}_warn"])
        if flagged.any():
            out.append(_areas(text, m[key], flagged, fmt, lon, lat))
    size = m["size"][ocean]
    if size.size and size.max() > 50 * size.min():
        lo, hi = size.min(), size.max()
        warns.append(f"Cell sizes vary by {hi / lo:.0f}x ({lo:.2f}-{hi:.2f} km).")
    ends = [(0, 0), (0, nx - 1), (ny - 1, nx - 1), (ny - 1, 0)]
    angles = enumerate(cg.corner_angles or [])
    flat = [k + 1 for k, a in angles if a > 170 and ocean[ends[k]]]
    if flat:
        s = "s" * (len(flat) > 1)
        what = f"Nearly flat corner{s} {_and(flat)}"
        warns.append(f"{what}: consider masking {'their' if s else 'its'} cell{s}.")
    if runs is not None:
        n = sum(r["tiny"] for d in OBC_SIDES for r in runs[d]["runs"])
        if n:  # as the TopoEditor words it
            warns.append(
                f"{n} small open boundar{'y' if n == 1 else 'ies'} along coastline"
            )
        alt = [d for d in OBC_SIDES if runs[d]["alternating"]]
        if alt:
            changes = _and(runs[d]["n_transitions"] for d in alt) + " land/sea changes"
            verb = "sides run" if len(alt) > 1 else "side runs"
            warns.append(f"The {_and(alt)} {verb} along the coast ({changes}).")
    return errors, warns
