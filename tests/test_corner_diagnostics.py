import copy
import time
from types import SimpleNamespace
import numpy as np
import pytest
import mom6_forge._conformal as cf
from mom6_forge import corner_diagnostics as diag
from mom6_forge.grid import Grid
from mom6_forge.topo import ww3_timesteps_from_spacing
from utils import on_cisl_machine

BOX = ([235.0, 243.0, 243.0, 235.0], [31.0, 31.0, 39.0, 39.0], [0, 1, 2, 3])


def _box_grid(raster_points, resolution_km=8.0):
    """The GridSketcher default box: plain straight edges that must not be flagged."""
    n_cells = cf.n_cells_for_resolution(BOX[0], BOX[1], resolution_km)
    return cf.solve(*BOX, n_cells=n_cells, raster_points=raster_points)


@pytest.fixture(scope="module")
def get_california_grid(get_california):
    lon, lat, corners = get_california.values()
    return cf.solve(lon, lat, corners, 2_000, raster_points=cf.PREVIEW_RASTER_POINTS)


@pytest.fixture(scope="module")
def get_trapezoid():
    """A trapezoid with a different lon/lat range on every side, so swapped sides show."""
    lon, lat = [-125.0, -117.0, -118.0, -124.0], [32.0, 33.0, 40.0, 39.0]
    return cf.solve(lon, lat, [0, 1, 2, 3], 400, raster_points=cf.PREVIEW_RASTER_POINTS)


def _fake_grid(cell_m, ny=4, nx=5):
    return SimpleNamespace(tarea=np.full((ny, nx), cell_m**2))


def _no_land(*args, **kwargs):
    raise RuntimeError("no offline Natural Earth cache and no network")


# --- cell_metrics ---


@pytest.mark.parametrize("raster_points", [cf.PREVIEW_RASTER_POINTS, 1_000_000])
def test_plain_box_edges_are_clean_at_any_raster(raster_points):
    status = diag.cell_metrics(_box_grid(raster_points))["status"]
    assert (status == 0).all(), f"{int((status > 0).sum())} spuriously flagged cells"


def test_plain_box_edges_stay_square_on_a_fine_grid():
    """3 km cells on a ~4 km preview raster: the exempt band grows with h/dx."""
    m = diag.cell_metrics(_box_grid(cf.PREVIEW_RASTER_POINTS, 3.0))
    assert (m["ortho"] <= m["thresholds"]["ortho_warn"]).all()


def test_california_flags_the_reflex_corners_only(get_california_grid):
    cg = get_california_grid
    assert cg.corner_angles[0] < 170 < cg.corner_angles[1]
    assert cg.corner_angles[3] < 170 < cg.corner_angles[2]
    s, k = diag.cell_metrics(cg)["status"], 6
    assert (s[:k, -k:] > 0).any() and (s[-k:, -k:] > 0).any()
    assert not (s[:k, :k] > 0).any() and not (s[-k:, :k] > 0).any()


def test_cell_metrics_explains_every_flag_and_honours_thresholds(get_california_grid):
    cg = get_california_grid
    m = diag.cell_metrics(cg)
    th = m["thresholds"]
    names = [f"{k}_{w}" for k in ("ortho", "ratio", "aspect") for w in ("warn", "bad")]
    assert list(th) == names
    flagged = m["status"] > 0
    assert flagged.any() and set(np.unique(m["status"])) <= {0, 1, 2}
    over = [m[k][flagged] > th[f"{k}_warn"] for k in ("ortho", "ratio", "aspect")]
    assert np.all(over[0] | over[1] | over[2] | m["folded"][flagged])
    kw = dict(ratio_warn=1e9, aspect_warn=1e9)
    lenient = diag.cell_metrics(cg, ortho_warn=89, ortho_bad=89.5, **kw)["status"]
    strict = diag.cell_metrics(cg, ortho_warn=0.01, ortho_bad=0.02, **kw)["status"]
    assert (strict > 0).sum() >= (lenient > 0).sum()
    assert set(diag.STATUS_COLORS) == {0, 1, 2}


# --- grid_quality ---


def test_grid_quality_uses_the_nodes_cell_metrics_judges(get_california_grid):
    for cg in (get_california_grid, _box_grid(cf.PREVIEW_RASTER_POINTS)):
        m = diag.cell_metrics(cg)
        q = diag.grid_quality(cg)
        assert q["ortho_max"] == pytest.approx(m["ortho"].max())
        assert (q["size_min"], q["size_max"]) == (m["size"].min(), m["size"].max())
        assert diag.grid_quality(cg, metrics=m) == q


def test_grid_quality_wet_mask_relaxes_orthogonality(get_california_grid):
    cg = get_california_grid
    wet = np.zeros((cg.ny, cg.nx), dtype=bool)
    wet[0, 0] = True
    q_all, q_wet = diag.grid_quality(cg), diag.grid_quality(cg, wet=wet)
    assert (q_all["ortho_warn_threshold"], q_wet["ortho_warn_threshold"]) == (3.0, 5.0)
    assert q_wet["size_min"] == q_all["size_min"] and q_wet["nx"] == q_all["nx"]


# --- size_ratios ---


def test_size_ratios_recommend_the_plane_where_the_outline_is_a_rectangle():
    # A lon/lat box is a rectangle on Mercator, a square on the polar plane on Polar
    ratios = diag.size_ratios(*BOX)
    assert min(ratios, key=ratios.get) == "merc" and list(ratios) == list(cf.KINDS)
    polar = cf.MapProjection("stere", 0.0, 90.0, 70.0)
    lon, lat = polar.to_lonlat([-1e6, 1e6, 1e6, -1e6], [-2.3e6, -2.3e6, -3e5, -3e5])
    ratios = diag.size_ratios(lon, lat, [0, 1, 2, 3])
    assert min(ratios, key=ratios.get) == "stere" and len(ratios) == 4
    # Only the kinds that work: Polar round a pole, no Lambert on the equator
    pole, equator = ([0, 90, 180, 270], [78] * 4), ([-10, 10, 10, -10], [-5, -5, 5, 5])
    assert list(diag.size_ratios(*pole, [0, 1, 2, 3])) == ["stere"]
    assert list(diag.size_ratios(*equator, [0, 1, 2, 3])) == ["merc", "tmerc", "stere"]
    assert diag.size_ratios([0, 1, 1, 0], [0, 0, 1, 1], [0, 1]) == {}


@pytest.mark.benchmark
def test_size_ratios_take_under_a_tenth_of_a_second():
    """GridSketcher solves them again after each edit of the outline."""
    if not on_cisl_machine():
        pytest.skip("This benchmark is only run on the derecho and casper machines")
    diag.size_ratios(*BOX)
    t0 = time.perf_counter()
    diag.size_ratios(*BOX)
    assert time.perf_counter() - t0 < 0.1


# --- warnings ---


def test_warnings_orthogonality_threshold():
    q = dict(size_ratio=1.0, size_min=1.0, size_max=1.0, corner_angles={})
    q.update(ortho_p99=4.0, max_neighbour_ratio=1.0)
    assert any(w.startswith("Orthogonality") for w in diag.warnings(q))
    assert not diag.warnings(q | dict(ortho_warn_threshold=5.0))
    assert not diag.warnings(q | dict(ortho_p99=0.1, ortho_warn_threshold=3.0))


def test_warnings_on_a_timestep_dict_alone():
    ts = diag.timestep_limits(_fake_grid(700.0), coupling_dt=3600.0)
    assert ts["cice_cfl"] > 0.5
    assert any(f"{ts['cice_max_coupling_dt']:.0f}" in w for w in diag.warnings(ts))
    ts = diag.timestep_limits(_fake_grid(10_000.0))
    assert not any("CICE CFL" in w for w in diag.warnings(ts))


def test_messages_are_sentences_about_ocean_cells_only(get_california_grid):
    cg = get_california_grid
    m = diag.cell_metrics(cg)
    errors, warns = diag.quality_messages(cg, metrics=m)
    assert not errors and all(w[0].isupper() and w.endswith(".") for w in warns)
    assert any(w.startswith("Abrupt cell-size change ") for w in warns)
    assert any(" near " in w and w.endswith(" cells).") for w in warns)
    at = [(0, 0), (0, 19), (49, 19), (49, 0), (25, 19), (25, 10)]
    places = ["near corner 1", "near corner 2", "near corner 3", "near corner 4"]
    places += ["near the east side", "in the interior"]
    assert [diag._place(j, i, 50, 20) for j, i in at] == places
    # Every flagged cell and every corner cell on land: nothing to say
    wet = m["status"] == 0
    wet[[0, 0, -1, -1], [0, -1, -1, 0]] = False
    land = diag.cell_metrics(cg, wet=wet)
    assert (land["status"] == 0).all() and m["status"].any()
    assert diag.quality_messages(cg, wet=wet, metrics=land) == ([], [])


def test_size_jumps_to_land_neighbours_do_not_count():
    size, wet = np.array([[1.0, 1.1, 3.0]]), np.array([[True, True, False]])
    assert np.allclose(diag._neighbour_ratio(size), [[1.1, 3 / 1.1, 3 / 1.1]])
    assert np.allclose(diag._neighbour_ratio(size, wet)[0, :2], 1.1)


def test_a_folded_cell_is_an_error(get_trapezoid):
    cg = copy.copy(get_trapezoid)
    cg.x, cg._geometry = cg.x.copy(), None
    # Node (5, 5) pushed two cells past node (5, 7) turns its cells inside out
    cg.x[10, 10] = 2 * cg.x[10, 14] - cg.x[10, 10]
    m = diag.cell_metrics(cg)
    assert m["folded"].any() and (m["status"][m["folded"]] == 2).all()
    errors, _ = diag.quality_messages(cg, metrics=m)
    assert len(errors) == 1 and errors[0].startswith("Folded (inside-out) cells in the")


# --- timestep_limits ---


def test_timestep_limits_uniform_grid_is_exact():
    ts = diag.timestep_limits(_fake_grid(3000.0))
    assert ts["metric"] == "sqrt(area)" and ts["min_dx"] == ts["min_wet_dx"] == 3000.0
    assert ts["cice_cfl"] == pytest.approx(3600.0 / 3000.0)
    assert ts["cice_max_coupling_dt"] == pytest.approx(1500.0)
    assert ts["mom6_cfl"] == pytest.approx(900.0 / 3000.0)
    assert "MOM6 CFL" in ts["mom6_hint"]
    assert ts["ww3"] == ww3_timesteps_from_spacing(3000.0, 3600.0)
    lite = diag.timestep_limits(_fake_grid(3000.0), ww3=False)
    assert lite == ts | dict(ww3=None)


def test_timestep_limits_matches_the_bering_calibration():
    """A coupled Bering Sea build's smallest wet cell is sqrt(tarea) = 3173.7 m."""
    for ice_speed, cfl in ((0.5, 0.567), (1.0, 1.134)):
        ts = diag.timestep_limits(_fake_grid(3173.7), ice_speed=ice_speed)
        assert ts["cice_cfl"] == pytest.approx(cfl, abs=0.01)


def test_timestep_limits_wet_mask_skips_a_land_locked_sliver():
    grid = _fake_grid(3000.0, ny=3, nx=3)
    grid.tarea[1, 1] = 300.0**2
    wet = np.ones((3, 3), dtype=bool)
    wet[1, 1] = False
    q_all, q_wet = diag.timestep_limits(grid), diag.timestep_limits(grid, wet=wet)
    assert q_all["min_wet_dx"] == q_wet["min_dx"] == pytest.approx(300.0)
    assert q_wet["min_wet_dx"] == pytest.approx(3000.0)


def test_ocean_only_minimum_on_a_real_coast(get_california):
    """The coarse California sketch's smallest cell is a land corner: the ocean mask skips it."""
    lon, lat, corners = get_california.values()
    n_cells = cf.n_cells_for_resolution(lon, lat, 20.0)
    cg = cf.solve(lon, lat, corners, n_cells, raster_points=cf.PREVIEW_RASTER_POINTS)
    q_all = diag.timestep_limits(cg, ww3=False)
    q_wet = diag.timestep_limits(cg, wet=diag.ocean_mask(cg), ww3=False)
    assert np.isfinite(q_all["min_dx"]) and q_all["min_dx"] > 0
    assert q_wet["min_wet_dx"] > q_all["min_dx"]
    assert q_wet["cice_cfl"] < q_all["cice_cfl"]


# --- ocean_mask ---


def test_ocean_mask_known_points_in_either_longitude_convention():
    """Kansas, the mid-Pacific, Paris and Hawaii, as signed and as 0-360 longitudes."""
    lat = np.zeros((3, 9))
    lat[1, 1::2] = [39.83, 0.0, 48.86, 19.6]
    for centres in ([-98.58, -140.0, 2.35, -155.5], [261.42, 220.0, 2.35, 204.5]):
        lon = np.zeros((3, 9))
        lon[1, 1::2] = centres
        wet = diag.ocean_mask(SimpleNamespace(lon=lon, lat=lat))
        assert wet.tolist() == [[False, True, False, False]]


def test_ocean_mask_is_none_without_land_data(monkeypatch):
    monkeypatch.setattr(cf, "_land_geometry", _no_land)
    cg = SimpleNamespace(lon=np.zeros((3, 3)), lat=np.zeros((3, 3)))
    assert diag.ocean_mask(cg) is None


# --- open_boundary_runs ---


def test_side_names_match_a_built_grid(get_trapezoid):
    cg = get_trapezoid
    grid = Grid.from_lonlat_arrays(cg.lon, cg.lat, name="obc")
    ocean = np.ones((cg.ny, cg.nx), dtype=bool)
    runs = diag.open_boundary_runs(grid, ocean=ocean)
    assert runs["south"]["runs"][0]["lat"] < runs["north"]["runs"][0]["lat"]
    assert runs["west"]["runs"][0]["lon"] < runs["east"]["runs"][0]["lon"]
    assert grid.tlat.values[0].mean() < grid.tlat.values[-1].mean()
    ocean[0, :] = False
    ocean[0, 3:5] = True
    south = diag.open_boundary_runs(grid, ocean=ocean)["south"]["runs"]
    assert south == diag.open_boundary_runs(cg, ocean=ocean)["south"]["runs"]
    assert [r["length_cells"] for r in south] == [2]


def test_tiny_run_next_to_the_coast(get_trapezoid):
    cg = get_trapezoid
    ocean = np.zeros((cg.ny, cg.nx), dtype=bool)
    ocean[0, 3:5] = True
    ocean[1:-1, -1] = True
    runs = diag.open_boundary_runs(cg, ocean=ocean, min_cells=4, min_fraction=0.03)
    south = runs["south"]["runs"]
    assert [(r["length_cells"], r["tiny"]) for r in south] == [(2, True)]
    assert [r["tiny"] for r in runs["east"]["runs"]] == [False]
    assert runs["north"]["runs"] == runs["west"]["runs"] == []


@pytest.mark.parametrize("small", [False, True])
def test_all_sea_sides_are_one_run_not_tiny(get_trapezoid, small):
    """A fully open side is never tiny, even when it is shorter than min_cells."""
    grid = get_trapezoid
    if small:
        lon, lat = np.meshgrid(np.linspace(-70.0, -69.0, 3), np.linspace(40.0, 41.0, 3))
        grid = SimpleNamespace(ny=3, nx=3, tlon=lon, tlat=lat)
    runs = diag.open_boundary_runs(grid, ocean=np.ones((grid.ny, grid.nx), dtype=bool))
    for side, n in zip(diag.OBC_SIDES, (grid.nx, grid.ny, grid.nx, grid.ny)):
        r = runs[side]
        assert [(run["length_cells"], run["tiny"]) for run in r["runs"]] == [(n, False)]
        assert (r["n_transitions"], r["alternating"]) == (0, False)


def test_alternating_side(get_trapezoid):
    ocean = np.zeros((get_trapezoid.ny, get_trapezoid.nx), dtype=bool)
    ocean[0, 0::2] = True
    south = diag.open_boundary_runs(get_trapezoid, ocean=ocean)["south"]
    assert south["n_transitions"] >= 4 and south["alternating"] is True


def test_coastline_estimate_and_its_fallback(
    get_california_grid, get_trapezoid, monkeypatch
):
    runs = diag.open_boundary_runs(get_california_grid)
    assert runs["ocean_known"] is True
    assert all(runs[side]["length_km"] > 0 for side in diag.OBC_SIDES)
    monkeypatch.setattr(cf, "_land_geometry", _no_land)
    runs = diag.open_boundary_runs(get_trapezoid)
    assert runs["ocean_known"] is False
    assert [r["length_cells"] for r in runs["south"]["runs"]] == [get_trapezoid.nx]
