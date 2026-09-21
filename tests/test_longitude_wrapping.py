"""Every supergrid builder must produce continuous, bounded longitude at both
360-degree seams (Prime Meridian and antimeridian). GRID_MATRIX covers all
4 builders x 3 seam positions; poles are exempt (longitude legitimately
converges there)."""

import numpy as np
import pytest

from mom6_forge._supergrid import (
    ProjectedSupergrid,
    RectilinearCartesianSupergrid,
    SupergridBase,
    UniformSphericalSupergrid,
    _max_adjacent_diff,
    haversine,
)

EXTENT_LEN, EXTENT_LAT_MIN, EXTENT_LEN_Y = 20.0, -5.0, 10.0  # degrees


def _uniform_spherical(lon_min):
    return UniformSphericalSupergrid.from_extents(
        lon_min, EXTENT_LEN, EXTENT_LAT_MIN, EXTENT_LEN_Y, nx=10, ny=5
    )


def _rectilinear_cartesian(lon_min):
    return RectilinearCartesianSupergrid.from_extents(
        lon_min, EXTENT_LEN, EXTENT_LAT_MIN, EXTENT_LEN_Y, resolution=2.0
    )


# EPSG:3995 (Arctic polar stereographic): +x -> lon=90, +y -> lon=180, -y -> lon=0.
_CRS_OFFSETS = {
    "regular": dict(x=(3_200_000, 3_800_000), y=(-300_000, 300_000)),
    "pm_seam": dict(x=(-300_000, 300_000), y=(-3_800_000, -3_200_000)),
    "dateline_seam": dict(x=(-300_000, 300_000), y=(3_200_000, 3_800_000)),
}


def _projected_from_crs(seam):
    off = _CRS_OFFSETS[seam]
    return ProjectedSupergrid.from_crs(
        "EPSG:3995", *off["x"], *off["y"], resolution_m=100_000
    )


def _projected_from_center(center_lon):
    return ProjectedSupergrid.from_center(
        center_lat=40.0,
        center_lon=center_lon,
        width_m=400_000,
        height_m=400_000,
        resolution_m=100_000,
    )


# regular = away from either seam, pm_seam = straddles 0/360, dateline_seam = straddles +/-180.
GRID_MATRIX = [
    ("uniform_spherical", "regular", lambda: _uniform_spherical(15.0)),
    ("uniform_spherical", "pm_seam", lambda: _uniform_spherical(350.0)),
    ("uniform_spherical", "dateline_seam", lambda: _uniform_spherical(170.0)),
    ("rectilinear_cartesian", "regular", lambda: _rectilinear_cartesian(15.0)),
    ("rectilinear_cartesian", "pm_seam", lambda: _rectilinear_cartesian(350.0)),
    ("rectilinear_cartesian", "dateline_seam", lambda: _rectilinear_cartesian(170.0)),
    ("projected_from_crs", "regular", lambda: _projected_from_crs("regular")),
    ("projected_from_crs", "pm_seam", lambda: _projected_from_crs("pm_seam")),
    (
        "projected_from_crs",
        "dateline_seam",
        lambda: _projected_from_crs("dateline_seam"),
    ),
    ("projected_from_center", "regular", lambda: _projected_from_center(-70.0)),
    ("projected_from_center", "pm_seam", lambda: _projected_from_center(0.5)),
    ("projected_from_center", "dateline_seam", lambda: _projected_from_center(180.0)),
]


@pytest.mark.parametrize(
    ("grid_type", "seam", "builder"),
    GRID_MATRIX,
    ids=[f"{grid_type}-{seam}" for grid_type, seam, _ in GRID_MATRIX],
)
def test_longitude_is_continuous_and_bounded(grid_type, seam, builder):
    x = builder().x
    print(f"{grid_type}/{seam}: x at seam row = {np.round(x[x.shape[0] // 2], 2)}")

    max_jump = _max_adjacent_diff(x)
    assert max_jump < 180.0, (
        f"{grid_type}/{seam}: adjacent-node jump of {max_jump:.2f} degrees "
        "indicates an un-wrapped seam discontinuity"
    )

    span = x.max() - x.min()
    assert (
        span < 360.0 + 1e-6
    ), f"{grid_type}/{seam}: longitude span of {span:.2f} degrees looks unbounded"


def test_projected_supergrid_dx_dy_sane_when_pole_inside_domain():
    """ProjectedSupergrid.from_crs/from_center must produce physically sane dx/dy
    even when the domain's pole (the origin, for EPSG:3995) sits inside it -- the
    row/column of supergrid nodes straddling the pole necessarily crosses the
    antimeridian branch cut internally, which corrupted dx by orders of magnitude
    under the old smallangle-based calculation."""
    resolution_m = 100_000.0
    sg = ProjectedSupergrid.from_crs(
        "EPSG:3995", -300_000, 300_000, -300_000, 300_000, resolution_m=resolution_m
    )

    # No node-to-node metric should be wildly larger than the requested resolution:
    # the old (broken) calculation produced dx values off by ~2-3 orders of magnitude
    # at the seam crossing the pole.
    assert np.all(sg.dx > 0)
    assert np.all(sg.dy > 0)
    assert sg.dx.max() < 5 * resolution_m
    assert sg.dy.max() < 5 * resolution_m
    assert np.all(sg.area > 0)

    # Cross-check dx directly against haversine ground truth for the row of nodes
    # passing right through the pole (the worst case for the old bug).
    mid_row = sg.x.shape[0] // 2
    expected_dx = haversine(
        sg.y[mid_row, :-1],
        sg.x[mid_row, :-1],
        sg.y[mid_row, 1:],
        sg.x[mid_row, 1:],
        R=6.371e6,
    )
    np.testing.assert_allclose(sg.dx[mid_row], expected_dx)


def test_global_cyclic_grid_still_spans_exactly_360():
    """A legitimate global grid (span == 360) must build and keep its cyclic
    0->360 span, not get rejected or collapsed to a smaller wrapped range."""
    x = UniformSphericalSupergrid.from_extents(
        lon_min=0.0, len_x=360.0, lat_min=-5.0, len_y=10.0, nx=180, ny=5
    ).x
    assert np.isclose(x.max() - x.min(), 360.0)
    assert _max_adjacent_diff(x) < 180.0


def test_expand_keeps_polar_metrics_after_dataset_round_trip():
    """A pole-inside-domain grid reloaded from a dataset must still expand with
    usable metrics. from_ds cannot infer how dx/dy were computed, so without the
    recorded dx_dy_calc_type it falls back to smallangle, which is invalid at a
    pole and yields negative dx/dy."""
    sg = ProjectedSupergrid.from_crs("EPSG:3995", -1e6, 1e6, -1e6, 1e6, 100_000)
    assert sg.y.max() == 90.0, "this test needs the pole inside the domain"

    reloaded = SupergridBase.from_ds(sg.to_ds())
    assert reloaded._dx_dy_calc_type == sg._dx_dy_calc_type
    assert reloaded._R == sg._R

    expanded = reloaded.expand(1)
    assert (expanded.dx > 0).all()
    assert (expanded.dy > 0).all()
    # same ballpark as the 100 km requested resolution, not a wrapped blow-up
    assert expanded.dx.max() < 2 * 100_000


def test_smallangle_is_upgraded_at_the_pole():
    """Even a dataset written before dx_dy_calc_type was recorded (so smallangle
    is assumed) must not produce negative metrics on a pole-touching grid."""
    sg = ProjectedSupergrid.from_crs("EPSG:3995", -1e6, 1e6, -1e6, 1e6, 100_000)
    forced = ProjectedSupergrid._init_from_xy(
        sg.x, sg.y, "projected_crs", dx_dy_calc_type="smallangle"
    )
    assert (forced.dx > 0).all()
    assert (forced.dy > 0).all()
    assert forced._dx_dy_calc_type == "haversine"


def test_seam_repair_left_alone_when_the_domain_encircles_a_pole():
    """A domain encircling a pole spans all 360 degrees, so re-centering cannot
    remove its seam -- it only moves it, while pushing x out of range and
    scrambling angle_dx across the new seam. Checking the repaired result is
    what rules that out; no latitude test could, since such a box can keep every
    node well short of the pole."""
    sg = ProjectedSupergrid.from_crs(
        "EPSG:3995", -975_000, 1_025_000, -975_000, 1_025_000, 100_000
    )
    assert np.abs(sg.y).max() < 89.9, "this test needs the nodes to miss the pole"
    assert sg.x.max() - sg.x.min() > 180.0, "and the domain to encircle it"

    assert sg.x.min() >= -180.0 and sg.x.max() <= 180.0
    expected = -((sg.x + 180.0) % 360.0 - 180.0)
    err = np.abs(((sg.angle_dx - expected + 180.0) % 360.0) - 180.0)[1:-1, 1:-1]
    # One seam's worth of error is unavoidable here; two is the repair moving it.
    assert err.max() < 120.0


def test_smallangle_kept_for_a_rectilinear_grid_touching_the_pole():
    """smallangle is exactly MOM6's along-parallel convention for a lat/lon
    grid, so merely reaching a pole must not switch the method: that would
    silently change dx on every global grid."""
    sg = UniformSphericalSupergrid.from_extents(0.0, 360.0, 60.0, 30.0, 180, 30)
    assert np.abs(sg.y).max() == 90.0
    expected, _ = SupergridBase._calc_dx_dy(sg.x, sg.y, type="smallangle")
    assert sg._dx_dy_calc_type == "smallangle"
    np.testing.assert_array_equal(sg.dx, expected)


@pytest.mark.parametrize(
    "extent",
    [
        (-1e6, 1e6, -1e6, 1e6),  # pole on a node
        (-975_000, 1_025_000, -975_000, 1_025_000),  # pole between nodes
    ],
)
def test_smallangle_upgraded_for_a_curvilinear_grid_around_a_pole(extent):
    """A curvilinear grid encircling a pole breaks smallangle whether or not any
    node gets close enough to the pole for a latitude test to notice."""
    cap = ProjectedSupergrid.from_crs("EPSG:3995", *extent, 100_000)
    built = ProjectedSupergrid._init_from_xy(
        cap.x, cap.y, "projected_crs", dx_dy_calc_type="smallangle"
    )
    assert built._dx_dy_calc_type == "haversine"
    assert (built.dx > 0).all() and (built.dy > 0).all()


def test_seam_repair_is_independent_of_storage_convention():
    """The same grid written in [0, 360) and in [-180, 180] must come out with
    the same metrics. A latitude guard in front of the seam repair used to skip
    the repair for a pole-adjacent grid, leaving the seam in place in one
    convention only and silently switching that copy to haversine."""
    base = UniformSphericalSupergrid.from_extents(170.0, 20.0, 79.95, 10.0, 20, 20)
    x_0_360, y = base.x.copy(), base.y
    x_pm_180 = ((x_0_360 + 180.0) % 360.0) - 180.0
    assert np.abs(y).max() > 89.9, "this test needs a pole-adjacent grid"

    a = SupergridBase._init_from_xy(x_0_360, y, "uniform_spherical")
    b = SupergridBase._init_from_xy(x_pm_180, y, "uniform_spherical")
    assert a._dx_dy_calc_type == b._dx_dy_calc_type == "smallangle"
    assert _max_adjacent_diff(b.x) < 180.0
    np.testing.assert_allclose(a.dx, b.dx)
    np.testing.assert_allclose(a.dy, b.dy)


def test_reconstruct_rejects_a_polar_mesh_with_averaged_midpoints(tmp_path):
    """reconstruct_from_esmf_mesh recovers u/v points by averaging corner
    longitudes and latitudes, which lands halfway around the world instead of at
    the pole when an edge crosses one. The result is large positive dx/dy, so
    the negative-metric fallback cannot see it -- on an Arctic cap it turns
    50 km spacing into 2000 km. Better to refuse than to hand back numbers that
    look plausible."""
    sg = ProjectedSupergrid.from_crs("EPSG:3995", -1e6, 1e6, -1e6, 1e6, 100_000)
    path = tmp_path / "polar_mesh.nc"
    sg.to_esmf_mesh(str(path), mask="all_unmasked")

    with pytest.raises(ValueError, match="edge midpoints are not on their edges"):
        SupergridBase.reconstruct_from_esmf_mesh(str(path))


def test_reconstruct_accepts_a_pole_adjacent_lat_lon_mesh(tmp_path):
    """The midpoint check must not fire on a rectilinear mesh that merely
    reaches high latitude, where averaging corners is correct."""
    sg = UniformSphericalSupergrid.from_extents(0.0, 20.0, 79.5, 10.0, 20, 20)
    path = tmp_path / "polar_adjacent.nc"
    sg.to_esmf_mesh(str(path), mask="all_unmasked")

    rebuilt = SupergridBase.reconstruct_from_esmf_mesh(str(path))
    assert (rebuilt.dx > 0).all() and (rebuilt.dy > 0).all()


@pytest.mark.parametrize(
    "hgrid",
    [
        # A displaced-pole grid: its cells curve hard near the displaced pole,
        # so its midpoints sit well off the corner-to-corner chord by geometry
        # rather than by error (1.12 in x, 1.56 in y).
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/gx1v6/ocean_hgrid_230424.nc",
        "/glade/p/cesmdata/cseg/inputdata/ocn/mom/tx0.66v1/ocean_hgrid_180829.nc",
    ],
)
def test_midpoint_check_accepts_production_curvilinear_grids(hgrid):
    """The midpoint check must clear real grids. Calibrating it on lat/lon and
    tripolar geometry alone set the tolerance below what a displaced-pole grid
    reaches natively, which would have rejected a lossless reconstruction."""
    import xarray as xr
    from utils import on_cisl_machine

    if not on_cisl_machine():
        pytest.skip("Requires CISL/GLADE access")

    ds = xr.open_dataset(hgrid)
    SupergridBase._check_edge_midpoints(ds.x.values, ds.y.values)
