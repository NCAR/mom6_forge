"""Every supergrid builder must produce continuous, bounded longitude,
regardless of where the domain sits relative to the two 360-degree seams:
the Prime Meridian (0/360 wrap) and the antimeridian (+/-180 wrap).

GRID_MATRIX exercises all 4 builders x 3 seam positions (12 grids) against
the two invariants SupergridBase.__init__ enforces: no discontinuous jump
between adjacent nodes, and no unbounded/implausible longitude range.
Domains within 0.1 degrees of a pole are exempt from both -- see
test_init_validation_exempts_pole_adjacent_latitude -- since every longitude
converges there; a jump is a geometric singularity, not a wrap bug.
"""

import numpy as np
import pytest

from mom6_forge._supergrid import (
    ProjectedSupergrid,
    RectilinearCartesianSupergrid,
    SupergridBase,
    UniformSphericalSupergrid,
    _max_adjacent_diff,
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


# lon_min/center_lon per seam: regular = away from either seam, pm_seam =
# straddles 0/360 (extent grids use the >360 overflow form), dateline_seam =
# straddles +/-180.
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


def test_global_cyclic_grid_still_spans_exactly_360():
    """A legitimate global grid (span == 360) must build and keep its cyclic
    0->360 span, not get rejected or collapsed to a smaller wrapped range."""
    x = UniformSphericalSupergrid.from_extents(
        lon_min=0.0, len_x=360.0, lat_min=-5.0, len_y=10.0, nx=180, ny=5
    ).x
    assert np.isclose(x.max() - x.min(), 360.0)
    assert _max_adjacent_diff(x) < 180.0


# dx/dy/area/angle_dx are never inspected by the __init__ guard, so a single
# zeros array of x's shape stands in for all four below.
@pytest.mark.parametrize(
    ("x", "match"),
    [
        pytest.param([[178.0, 179.0, -179.0, -178.0]], "jump", id="raw-discontinuity"),
        pytest.param([[0.0, 200.0, 400.0, 600.0]], "span", id="unbounded-span"),
        pytest.param(
            [[100_000.0, 100_001.0, 100_002.0]],
            "outside",
            id="huge-absolute-offset-small-local-span",
        ),
        pytest.param(
            [[178.0, 179.0, 180.0, 181.0]], None, id="continuous-but-unconventional"
        ),
        pytest.param(
            [[0.0, 90.0, 180.0, 270.0, 360.0]], None, id="exact-360-span-boundary"
        ),
    ],
)
def test_init_validates_longitude_directly(x, match):
    """Unit-test the __init__ guard itself, independent of any builder -- this
    is also the only safety net for the from_ds load path, which bypasses
    _init_from_xy's wrap entirely."""
    x = np.array(x)
    zeros = np.zeros_like(x)
    if match:
        with pytest.raises(ValueError, match=match):
            SupergridBase(x, zeros, zeros, zeros, zeros, zeros, "degrees", "test")
    else:
        grid = SupergridBase(x, zeros, zeros, zeros, zeros, zeros, "degrees", "test")
        assert grid.x is x


def test_init_validation_exempts_pole_adjacent_latitude():
    """The same raw discontinuity that raises at low latitude must be waved
    through when y indicates the domain sits at a pole."""
    x = np.array([[178.0, 179.0, -179.0, -178.0]])
    zeros = np.zeros_like(x)

    with pytest.raises(ValueError, match="jump"):
        SupergridBase(x, zeros, zeros, zeros, zeros, zeros, "degrees", "test")

    y_pole = np.full_like(x, 89.95)
    grid = SupergridBase(x, y_pole, zeros, zeros, zeros, zeros, "degrees", "test")
    assert grid.x is x
