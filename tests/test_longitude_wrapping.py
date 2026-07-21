"""Exhaustive check that every supergrid builder produces continuous, bounded
longitude, regardless of where the domain sits relative to the two seams that
360-degree periodicity creates: the Prime Meridian (0/360 wrap) and the
antimeridian (+/-180 wrap).

Each of the 4 grid-building entry points (UniformSphericalSupergrid,
RectilinearCartesianSupergrid, ProjectedSupergrid.from_crs,
ProjectedSupergrid.from_center) is exercised in 3 configurations:

- regular:        domain far from both seams (sanity check / no-op case)
- pm_seam:        domain straddling the Prime Meridian
- dateline_seam:  domain straddling the antimeridian

giving a 4x3 = 12-grid matrix, all checked by one parametrized test against
the two invariants SupergridBase.__init__ enforces:

1. No discontinuous jump between adjacent supergrid nodes (would corrupt
   dx/dy/area/angle_dx, which are computed by differencing adjacent x values).
2. No unbounded longitude span, and no value sitting at an implausible
   absolute offset (either is a sign x was never wrapped/normalized at all --
   the latter guards against e.g. an upstream unit bug that leaves a locally
   well-behaved but globally nonsensical value like ~500,000 degrees).

Domains that reach within 0.1 degrees of a pole are exempt from both checks:
every longitude legitimately converges there, so a jump is an unavoidable
geometric singularity, not a wrap bug (see test_pole_adjacent_domain_is_exempt).

For the extent-based grids (Uniform/Rectilinear), the Prime Meridian case
uses the >360 overflow representation (lon_min=350) since that is the exact
numeric form that broke prior to the fix; the antimeridian case uses a
plain 170-190 range, which was never discontinuous for these linspace-built
grids but should still land inside the invariants.

For the projected grids, the antimeridian case is the one that produces a
genuine raw discontinuity (pyproj's inverse transform has its branch cut at
+/-180, not at 0/360), so it is the interesting case for rule 1. The
Prime-Meridian case for these two mainly exercises rule 2's re-centering step
at a boundary-adjacent value.
"""

import numpy as np
import pytest

from mom6_forge._supergrid import (
    ProjectedSupergrid,
    RectilinearCartesianSupergrid,
    UniformSphericalSupergrid,
)

EXTENT_LEN = 20.0  # degrees
EXTENT_LAT_MIN = -5.0
EXTENT_LEN_Y = 10.0


def _uniform_spherical(lon_min):
    return UniformSphericalSupergrid.from_extents(
        lon_min, EXTENT_LEN, EXTENT_LAT_MIN, EXTENT_LEN_Y, nx=10, ny=5
    )


def _rectilinear_cartesian(lon_min):
    return RectilinearCartesianSupergrid.from_extents(
        lon_min, EXTENT_LEN, EXTENT_LAT_MIN, EXTENT_LEN_Y, resolution=2.0
    )


# EPSG:3995 (Arctic polar stereographic) directions, found by sweeping angle
# around the pole: +x -> lon=90, +y -> lon=180 (antimeridian), -y -> lon=0 (PM).
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


def _max_adjacent_jump(x):
    max_jump = 0.0
    if x.shape[1] > 1:
        max_jump = max(max_jump, np.abs(np.diff(x, axis=1)).max())
    if x.shape[0] > 1:
        max_jump = max(max_jump, np.abs(np.diff(x, axis=0)).max())
    return max_jump


@pytest.mark.parametrize(
    ("grid_type", "seam", "builder"),
    GRID_MATRIX,
    ids=[f"{grid_type}-{seam}" for grid_type, seam, _ in GRID_MATRIX],
)
def test_longitude_is_continuous_and_bounded(grid_type, seam, builder):
    grid = builder()
    x = grid.x

    max_jump = _max_adjacent_jump(x)
    assert max_jump < 180.0, (
        f"{grid_type}/{seam}: adjacent-node longitude jump of {max_jump:.2f} "
        "degrees indicates an un-wrapped seam discontinuity"
    )

    span = x.max() - x.min()
    assert span < 360.0 + 1e-6, (
        f"{grid_type}/{seam}: longitude span of {span:.2f} degrees looks "
        "unbounded/unwrapped"
    )


def test_pole_adjacent_domain_is_exempt_from_continuity_check():
    """A projected domain centered on the pole (e.g. a full Arctic cap) legitimately
    contains every longitude -- there is no wrap that makes this continuous, since
    the pole is the one point where all meridians meet. This must build without
    raising, unlike a genuine un-wrapped dateline bug at lower latitudes."""
    grid = ProjectedSupergrid.from_crs(
        "EPSG:3995", -500_000, 500_000, -500_000, 500_000, resolution_m=50_000
    )
    assert np.abs(grid.y).max() >= 89.9
    assert (
        _max_adjacent_jump(grid.x) > 180.0
    )  # confirms this is the case we're exempting


def test_global_cyclic_grid_still_spans_exactly_360():
    """Regression guard: a legitimate global grid (span == 360, the one case the
    invariant checks must not reject) must still build and keep its cyclic
    0->360 span, not get collapsed to a smaller wrapped range."""
    grid = UniformSphericalSupergrid.from_extents(
        lon_min=0.0, len_x=360.0, lat_min=-5.0, len_y=10.0, nx=180, ny=5
    )
    assert np.isclose(grid.x.max() - grid.x.min(), 360.0)
    assert _max_adjacent_jump(grid.x) < 180.0


@pytest.mark.parametrize(
    ("x", "expect_error", "match"),
    [
        pytest.param(
            np.array([[178.0, 179.0, -179.0, -178.0]]),
            True,
            "jump",
            id="raw-discontinuity",
        ),
        pytest.param(
            np.array([[0.0, 200.0, 400.0, 600.0]]),
            True,
            "span",
            id="unbounded-span",
        ),
        pytest.param(
            np.array([[100_000.0, 100_001.0, 100_002.0]]),
            True,
            "exceed",
            id="huge-absolute-offset-small-local-span",
        ),
        pytest.param(
            np.array([[178.0, 179.0, 180.0, 181.0]]),
            False,
            None,
            id="continuous-but-unconventional-range",
        ),
        pytest.param(
            np.array([[0.0, 90.0, 180.0, 270.0, 360.0]]),
            False,
            None,
            id="exact-360-span-boundary",
        ),
    ],
)
def test_init_validates_longitude_directly(x, expect_error, match):
    """Unit-test the SupergridBase.__init__ guard itself (independent of any
    builder), including the from_ds load path which bypasses _init_from_xy's
    wrap and so relies on this check as the only safety net."""
    y = np.zeros_like(x)
    dx = np.diff(x, axis=1) if x.shape[1] > 1 else np.zeros((x.shape[0], 0))
    dy = np.zeros((0, x.shape[1]))
    area = np.zeros((max(x.shape[0] - 1, 0), max(x.shape[1] - 1, 0)))
    angle_dx = np.zeros_like(x)

    from mom6_forge._supergrid import SupergridBase

    if expect_error:
        with pytest.raises(ValueError, match=match):
            SupergridBase(x, y, dx, dy, area, angle_dx, "degrees", grid_type="test")
    else:
        grid = SupergridBase(x, y, dx, dy, area, angle_dx, "degrees", grid_type="test")
        assert grid.x is x


def test_init_validation_exempts_pole_adjacent_latitude():
    """The same raw discontinuity that raises at low latitude must be waved
    through when y indicates the domain sits at a pole -- directly exercising
    the __init__ guard's exemption, independent of any projection machinery."""
    from mom6_forge._supergrid import SupergridBase

    x = np.array([[178.0, 179.0, -179.0, -178.0]])
    dx = np.diff(x, axis=1)
    dy = np.zeros((0, x.shape[1]))
    area = np.zeros((0, x.shape[1] - 1))
    angle_dx = np.zeros_like(x)

    y_equator = np.zeros_like(x)
    with pytest.raises(ValueError, match="jump"):
        SupergridBase(x, y_equator, dx, dy, area, angle_dx, "degrees", grid_type="test")

    y_pole = np.full_like(x, 89.95)
    grid = SupergridBase(x, y_pole, dx, dy, area, angle_dx, "degrees", grid_type="test")
    assert grid.x is x
