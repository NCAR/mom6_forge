import time
import warnings
import cartopy.crs as ccrs
import numpy as np
import pytest
import shapely
from pyproj import Geod
from shapely.geometry import LinearRing
import mom6_forge._conformal as cf
from mom6_forge import corner_diagnostics as diag
from utils import on_cisl_machine

FAST = dict(n_cells=64, raster_points=4000)


def _box(lon0, lon1, lat0, lat1):
    return [lon0, lon1, lon1, lon0], [lat0, lat0, lat1, lat1]


def _pole_square(lat):
    return [0.0, 90.0, 180.0, 270.0], [lat] * 4


@pytest.fixture(scope="module")
def get_rectangle():
    """A true 1000 x 600 km rectangle in a fixed LCC plane (a lon/lat box is not one)."""
    projection = cf.MapProjection("lcc", -100.0, 35.0, 33.0, 37.0)
    x, y = [-5e5, 5e5, 5e5, -5e5], [-3e5, -3e5, 3e5, 3e5]
    lon, lat = (np.asarray(a) for a in projection.to_lonlat(x, y))
    cg = cf.solve(lon, lat, [0, 1, 2, 3], 375, projection, raster_points=20_000)
    return lon, lat, projection, cg


def _assert_valid(cg):
    """Finite nodes and positive geodesic cell areas with no wild outlier (no folds)."""
    assert np.isfinite(cg.qlon).all() and np.isfinite(cg.qlat).all()
    q = np.stack([cg.qlon, cg.qlat])
    rings = [q[:, :-1, :-1], q[:, :-1, 1:], q[:, 1:, 1:], q[:, 1:, :-1]]
    corners = np.stack(rings, axis=-1).reshape(2, -1, 4)
    geod = Geod(ellps="WGS84")
    areas = np.abs([geod.polygon_area_perimeter(*c)[0] for c in zip(*corners)])
    assert (areas > 0).all() and areas.max() / areas.min() < 500


# --- MapProjection ---


def test_map_projection_rejects_unknown_kind():
    with pytest.raises(ValueError):
        cf.MapProjection("aea", -100.0, 35.0)


@pytest.mark.parametrize(
    "projection",
    [
        cf.MapProjection("lcc", -100.0, 35.0, 25.0, 45.0),
        cf.MapProjection("merc", -100.0, 0.0, 0.0),
        cf.MapProjection("tmerc", -100.0, 35.0),
        cf.MapProjection("stere", -100.0, 90.0, 70.0),
    ],
)
def test_pyproj_and_cartopy_agree(projection):
    lon, lat = np.array([-105.0, -100, -95, -100]), np.array([33.0, 33, 33, 40])
    x, y = projection.to_xy(lon, lat)
    pts = projection.cartopy().transform_points(ccrs.PlateCarree(), lon, lat)
    assert np.allclose(x, pts[:, 0], atol=1.0) and np.allclose(y, pts[:, 1], atol=1.0)
    assert np.allclose(projection.to_lonlat(x, y), (lon, lat))


# --- projection_for / as_projection ---


@pytest.mark.parametrize(
    "lon, lat, kind, lat_0",
    [
        ([-170.0, -150, -150, -170], [80.0, 80, 85, 85], "stere", 90.0),
        ([-170.0, -150, -150, -170], [-80.0, -80, -85, -85], "stere", -90.0),
        (*_box(150, 210, 70, 80), "stere", 90.0),  # Beaufort-Chukchi, over the dateline
        (*_box(20, 60, 70, 80), "stere", 90.0),  # Barents Sea
        (*_box(-10, 10, -5, 5), "merc", 0.0),
        (*_pole_square(75.0), "stere", 90.0),
        (*_pole_square(-75.0), "stere", -90.0),
    ],
)
def test_auto_projection_choice(lon, lat, kind, lat_0):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        projection = cf.projection_for("auto", lon, lat)
    assert (projection.kind, projection.lat_0) == (kind, lat_0)
    if kind == "stere":
        assert projection.lat_1 == np.sign(lat_0) * 70.0


def test_auto_projection_midlatitude_and_dateline():
    lon, lat = _box(-127, -116, 30, 39.5)
    projection = cf.projection_for("auto", lon, lat)
    assert projection.kind == "lcc" and 30 < projection.lat_1 < projection.lat_2 < 39.5
    projection = cf.projection_for("auto", [179.0, -179, -179, 179], [30.0, 30, 31, 31])
    assert projection.kind == "lcc" and abs(projection.lon_0) > 170
    with pytest.raises(ValueError, match="off the equator"):
        cf.projection_for("lcc", *_box(-10, 10, -5, 5))


@pytest.mark.parametrize("kind", ["lcc", "merc", "tmerc"])
@pytest.mark.parametrize("lat, pole", [(75.0, "North"), (-75.0, "South")])
def test_pole_outline_rejects_non_stereographic(kind, lat, pole):
    with pytest.raises(ValueError, match=pole):
        cf.projection_for(kind, *_pole_square(lat))
    # Nor is it solved on one made for another outline: its ring would be a sliver
    given = cf.projection_for(kind, *_box(-10, 10, 40, 50))
    with pytest.raises(ValueError, match=pole):
        cf.solve(*_pole_square(lat), [0, 1, 2, 3], 100, given, raster_points=1_000)


def test_as_projection_accepts_only_auto_kinds_and_map_projections():
    lon, lat = _box(-105, -95, 33, 37)
    assert cf.as_projection(None, lon, lat) == cf.as_projection("auto", lon, lat)
    assert cf.as_projection("tmerc", lon, lat).kind == "tmerc"
    given = cf.MapProjection("stere", -100.0, 90.0, 70.0)
    assert cf.as_projection(given, lon, lat) is given
    for bad in (3857, "EPSG:3857", "+proj=merc", "bogus"):
        with pytest.raises(ValueError):
            cf.as_projection(bad, lon, lat)


# --- polygon helpers ---


def test_polygon_helpers_on_a_square():
    x, y = np.array([0.0, 1, 1, 0]), np.array([0.0, 0, 1, 1])
    assert np.allclose(cf.interior_angles(x, y), 90.0)
    assert cf.is_simple(x, y) and not cf.is_simple([0.0, 1, 0, 1], [0.0, 1, 1, 0])
    ox, oy, corners = cf.orient_ccw(x, y, [0, 1, 2, 3])
    assert (ox == x).all() and (oy == y).all() and corners == [0, 1, 2, 3]
    ox, oy, corners = cf.orient_ccw(x[::-1], y[::-1], [0, 1, 2, 3])
    assert (ox == x).all() and (oy == y).all() and corners == [3, 2, 1, 0]


def test_order_corners_starts_south_west_then_ccw():
    x, y = np.array([0.0, 1, 2, 2, 1, 0]), np.array([0.0, 0, 0, 1, 1, 1])
    assert cf.order_corners(x, y, [3, 0, 5, 2]) == [0, 2, 3, 5]


def test_unwrap_lon():
    signed = cf.unwrap_lon([170.0, -170.0, -170.0, 170.0])
    assert np.allclose(signed, [170.0, 190.0, 190.0, 170.0])
    unwrapped_360 = cf.unwrap_lon([170.0, 190.0, 190.0, 170.0])
    assert np.allclose(np.diff(unwrapped_360), np.diff(signed))
    with pytest.raises(ValueError):
        cf.unwrap_lon([0.0, 90.0, 180.0, -90.0])


# --- solve ---


def test_rectangle_gives_uniform_square_cells(get_rectangle):
    lon, lat, _, cg = get_rectangle
    assert (cg.nx, cg.ny) == (25, 15) and cg.aspect == pytest.approx(0.6, rel=1e-9)
    qx, qy = cg.x[::2, ::2], cg.y[::2, ::2]
    assert np.allclose(np.diff(qx, axis=1), 4e4) and np.allclose(np.diff(qx, axis=0), 0)
    assert np.allclose(np.diff(qy, axis=0), 4e4) and np.allclose(np.diff(qy, axis=1), 0)
    px, py = cg.projection.to_xy(lon, lat)
    for k, (j, i) in enumerate([(0, 0), (0, -1), (-1, -1), (-1, 0)]):
        assert (cg.x[j, i], cg.y[j, i]) == pytest.approx((px[k], py[k]), abs=1e-3)


def test_clockwise_outline_gives_the_same_grid(get_rectangle):
    lon, lat, projection, cg = get_rectangle
    kw = dict(n_cells=375, projection=projection, raster_points=20_000)
    cw = cf.solve(lon[::-1], lat[::-1], [0, 1, 2, 3], **kw)
    assert np.allclose(cw.lon, cg.lon) and np.allclose(cw.lat, cg.lat)


def test_solve_fields_is_the_first_half_of_solve(get_rectangle):
    lon, lat, projection, cg = get_rectangle
    f = cf.solve_fields(lon, lat, [0, 1, 2, 3], 375, projection, raster_points=20_000)
    keys = "X Y xi eta inside nx ny aspect projection h lon lat P corners corner_angles"
    assert set(f) == set(keys.split())
    assert (f["nx"], f["ny"], f["aspect"]) == (cg.nx, cg.ny, cg.aspect)
    assert f["corners"] == cg.corners
    assert f["xi"].shape == f["inside"].shape == f["X"].shape


@pytest.mark.parametrize(
    "lon, lat, corners",
    [
        ([0.0, 1, 2], [0.0, 1, 0], [0, 1, 2]),
        (*_box(0, 1, 0, 1), [0, 1, 2, 2]),
        (*_box(0, 1, 0, 1), [0, 1, 2]),
        (*_box(0, 1, 0, 1), [0, 1, 2, 9]),
        ([0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0, 1, 2, 3]),
        ([0.0, 1, 1, 1, 0], [0.0, 0, 0, 1, 1], [0, 1, 2, 4]),  # a side of length 0
        ([0.0, 1, 2, 1], [0.0, 0, 0, 1e-9], [0, 1, 2, 3]),  # almost no area
    ],
)
def test_bad_outlines_raise(lon, lat, corners):
    with pytest.raises(ValueError):
        cf.solve(lon, lat, corners, n_cells=100, raster_points=5_000)


def test_dateline_signed_and_0_360_longitudes_agree():
    lon = np.array([170.0, 179.0, -179.0, -170.0, -170.0, 170.0])
    lat = [30.0, 30.0, 30.0, 30.0, 31.0, 31.0]
    a = cf.solve(lon, lat, [0, 2, 3, 5], n_cells=200, raster_points=20_000)
    b = cf.solve(lon % 360.0, lat, [0, 2, 3, 5], n_cells=200, raster_points=20_000)
    assert (a.nx, a.ny) == (b.nx, b.ny) and a.nx > 0


@pytest.mark.parametrize("lat", [75.0, -75.0])
def test_pole_outline_solves_on_polar_stereographic(lat):
    cg = cf.solve(*_pole_square(lat), [0, 1, 2, 3], n_cells=200, raster_points=20_000)
    assert cg.projection.kind == "stere" and cg.projection.lat_0 == np.sign(lat) * 90
    assert np.isfinite(cg.lon).all() and (np.abs(cg.lat) > 74.0).all()
    assert (np.abs(cg.lat) < 90.0).all()
    assert diag.grid_quality(cg)["ortho_p99"] < 5.0


def test_california_cell_count_reflex_corners_and_outline_fit(get_california):
    lon, lat, corners = get_california.values()
    cg = cf.solve(lon, lat, corners, 12_000, raster_points=cf.PREVIEW_RASTER_POINTS)
    assert (cg.nx, cg.ny) == (pytest.approx(63, abs=3), pytest.approx(193, abs=3))
    assert 175 < cg.corner_angles[1] < 215 and 175 < cg.corner_angles[2] < 215
    q = diag.grid_quality(cg)
    assert any("nearly flat" in m for m in diag.warnings(q))
    assert q["ortho_max"] < 5.0  # the mesh resolves the fields round those corners
    # The outer rows and columns lie on the outline, to the millimetre.
    ring = LinearRing(np.column_stack(cg.projection.to_xy(lon, lat)))
    edges = [np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]]
    outer = [np.concatenate([a[s] for s in edges]) for a in (cg.x, cg.y)]
    assert shapely.distance(shapely.points(*outer), ring).max() < 1e-3


def _cross(x, y):
    """Twice the signed area of each quad (from its diagonals)."""
    ax, ay = x[1:, 1:] - x[:-1, :-1], y[1:, 1:] - y[:-1, :-1]
    return ax * (y[1:, :-1] - y[:-1, 1:]) - ay * (x[1:, :-1] - x[:-1, 1:])


def test_a_reflex_grid_corner_folds_no_cell():
    """Corner 3 on a 231-degree vertex of a ring round the pole; the grid corners sit
    on their vertices, sharp corners included."""
    lon = np.linspace(0, 360, 48, endpoint=False)
    lat = 80 + 9 * np.cos(np.radians(lon))
    kw = dict(raster_points=cf.PREVIEW_RASTER_POINTS)
    cg = cf.solve(lon, lat, [0, 12, 24, 36], 40_000, **kw)
    assert (_cross(cg.x, cg.y) > 0).all() and (_cross(cg.x, cg.y)[::2, ::2] > 0).all()
    projection = cf.MapProjection("lcc", -100.0, 35.0, 33.0, 37.0)
    t = np.radians(10.0)
    x, y = 1e6 * np.array(
        [[0, 1, 1 + np.cos(t), np.cos(t)], [0, 0, np.sin(t), np.sin(t)]]
    )
    cg = cf.solve(*projection.to_lonlat(x, y), [0, 1, 2, 3], 2_000, projection, **kw)
    ends = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
    assert np.allclose(
        [(cg.x[e], cg.y[e]) for e in ends], np.transpose([x, y]), atol=1e-3
    )


def test_many_reflex_vertices_keep_the_preview_mesh_small():
    """160 2 km land teeth on a 1000 km side refine the mesh round 5 of them only."""
    projection = cf.MapProjection("lcc", -100.0, 35.0, 33.0, 37.0)
    teeth = np.arange(-480.0, 480.0, 6.0)[:, None] + [0, 0, 2, 2]
    x = np.r_[-500, teeth.ravel(), 500, 500, -500] * 1e3
    y = np.r_[-300, np.tile([-300, -298, -298, -300], len(teeth)), -300, 300, 300] * 1e3
    n = len(x)
    lon, lat = projection.to_lonlat(x, y)
    rp = cf.PREVIEW_RASTER_POINTS
    f = cf._solve_mesh(lon, lat, [0, n - 3, n - 2, n - 1], 10_000, projection, rp)
    assert len(f["nodes"]) < 1.6 * rp


def test_straight_sides_have_smooth_edge_cells():
    """A trapezoid 10% wider at the bottom: along-edge neighbour sizes within 2%."""
    projection = cf.MapProjection("lcc", -100.0, 35.0, 33.0, 37.0)
    x, y = [-5.5e5, 5.5e5, 5e5, -5e5], [-5e5, -5e5, 5e5, 5e5]
    lon, lat = projection.to_lonlat(x, y)
    kw = dict(projection=projection, raster_points=cf.PREVIEW_RASTER_POINTS)
    size = diag.cell_metrics(cf.solve(lon, lat, [0, 1, 2, 3], 2_500, **kw))["size"]
    for row in (size[0, 2:-2], size[-1, 2:-2], size[2:-2, 0], size[2:-2, -1]):
        assert np.maximum(row[1:] / row[:-1], row[:-1] / row[1:]).max() < 1.02


@pytest.mark.parametrize(
    "lon, lat",
    [
        _box(170, -170, 50, 55),
        _box(170, 180, -45, -35),
        _box(-10, 10, -5, 5),
        _box(-20, 20, 80, 89),
        _box(-100.2, -99.8, 39.85, 40.15),
        _box(-140, -104, 20, 56),
        _box(-150, -30, 34, 38),
    ],
    ids=["aleutians", "south", "equator", "near_pole", "30km", "4000km", "15_to_1"],
)
def test_edge_case_outlines_give_valid_grids(lon, lat):
    _assert_valid(cf.solve(lon, lat, [0, 1, 2, 3], **FAST))


def test_duplicate_vertex_and_near_flat_corner_still_solve():
    lon, lat = [-120, -120, -110, -110, -120], [30, 30, 30, 40, 40]
    _assert_valid(cf.solve(lon, lat, [0, 2, 3, 4], **FAST))
    cg = cf.solve([-120, -110, -100, -110], [30, 30.05, 30, 40], [0, 1, 2, 3], **FAST)
    assert cg.corner_angles[1] == pytest.approx(175.0, abs=2.0)
    _assert_valid(cg)


def test_tiny_cell_counts_floor_to_one_cell():
    lon, lat = _box(-100.2, -99.8, 39.85, 40.15)
    assert cf.n_cells_for_resolution(lon, lat, 10_000.0) == 1
    cg = cf.solve(lon, lat, [0, 1, 2, 3], n_cells=1, raster_points=4000)
    assert (cg.nx, cg.ny) == (1, 1)
    _assert_valid(
        cf.solve(*_box(-120, -110, 30, 40), [0, 1, 2, 3], 4, raster_points=4000)
    )


def test_resolution_helpers_agree():
    lon, lat = _box(-125, -120, 35, 40)
    n = cf.n_cells_for_resolution(lon, lat, 20.0)
    assert n == round(cf.outline_area_km2(lon, lat) / 400.0)
    cg = cf.solve(lon, lat, [0, 1, 2, 3], n_cells=n, raster_points=20_000)
    assert cf.nominal_resolution_km(cg) == pytest.approx(20.0, rel=0.05)


# --- land ---


def test_land_window_across_the_dateline_stays_unwrapped():
    lon, lat = _box(-127, -116, 30, 39.5)
    lon_min, lon_max, lat_min, lat_max = cf.land_window(lon, lat, factor=6.0)
    assert (lon_max - lon_min, lat_max - lat_min) == pytest.approx((66.0, 57.0))
    lon_min, lon_max, _, _ = cf.land_window(*_box(170, -165, 52, 63))
    assert lon_max > 180.0


@pytest.mark.parametrize("lat", [75.0, -75.0])
def test_land_window_round_a_pole_spans_all_longitudes(lat):
    lon_min, lon_max, lat_min, lat_max = cf.land_window(*_pole_square(lat))
    assert (lon_min, lon_max) == (-180.0, 180.0)
    assert (lat_min, lat_max) == ((0.0, 90.0) if lat > 0 else (-90.0, 0.0))


def test_land_polygons_cover_both_sides_of_the_dateline():
    polygons = cf.land_polygons((170.0, 200.0, 50.0, 66.0))
    lons = np.concatenate([np.asarray(p.exterior.coords)[:, 0] for p in polygons])
    assert lons.max() > 150.0 and lons.min() < -150.0
    assert len(cf.land_polygons((-500.0, 500.0, 60.0, 70.0))) > 0


# --- benchmark ---


@pytest.mark.benchmark
def test_full_resolution_california_is_fast_and_mostly_square(get_california):
    """Run with: pytest tests/test_conformal.py -m benchmark -v -s"""
    if not on_cisl_machine():
        pytest.skip("This benchmark is only run on the derecho and casper machines")
    t0 = time.perf_counter()
    cg = cf.solve(*get_california.values(), 12_000)
    elapsed = time.perf_counter() - t0
    print(f"\nfull-resolution solve (California, {cg.nx}x{cg.ny}): {elapsed:.2f} s")
    q = diag.grid_quality(cg)
    assert elapsed < 60.0 and cg.nx * cg.ny == pytest.approx(12_000, rel=0.05)
    assert q["ortho_median"] < 1.0 and q["size_ratio"] < 500 and q["aspect_max"] < 20
