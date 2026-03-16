import pytest
from mom6_bathy._supergrid import *
import numpy as np


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ([0, 10], [0, 10]),
    ],
)
def test_even_spacing_hgrid(lat, lon):
    assert isinstance(
        RectilinearCartesianSupergrid(
            lon[0], lon[1] - lon[0], lat[0], lat[1] - lat[0], 0.05
        ),
        RectilinearCartesianSupergrid,
    )


# --- ProjectedSupergrid tests ---


def test_projected_supergrid_from_crs():
    """from_crs returns a valid ProjectedSupergrid with correct array shapes."""
    resolution_m = 50_000
    x_min, x_max = -500_000, 500_000
    y_min, y_max = -500_000, 500_000
    sg = ProjectedSupergrid.from_crs(
        "EPSG:3995", x_min, x_max, y_min, y_max, resolution_m
    )
    assert isinstance(sg, ProjectedSupergrid)
    nx = int((x_max - x_min) / resolution_m)
    ny = int((y_max - y_min) / resolution_m)
    assert sg.x.shape == (2 * ny + 1, 2 * nx + 1)
    assert sg.y.shape == sg.x.shape
    assert sg.dx.shape == (2 * ny + 1, 2 * nx)
    assert sg.dy.shape == (2 * ny, 2 * nx + 1)
    assert sg.area.shape == (2 * ny, 2 * nx)
    assert np.all(sg.area > 0)


def test_projected_supergrid_from_center():
    """from_center returns a valid ProjectedSupergrid centred near the given location."""
    center_lat, center_lon = 40.0, -70.0
    width_m = height_m = 200_000
    resolution_m = 50_000
    sg = ProjectedSupergrid.from_center(
        center_lat, center_lon, width_m, height_m, resolution_m
    )
    assert isinstance(sg, ProjectedSupergrid)
    nx = int(width_m / resolution_m)
    ny = int(height_m / resolution_m)
    assert sg.x.shape == (2 * ny + 1, 2 * nx + 1)
    # Centre of the grid should be close to the requested geographic point
    centre_lat = sg.y[sg.y.shape[0] // 2, sg.y.shape[1] // 2]
    centre_lon = sg.x[sg.x.shape[0] // 2, sg.x.shape[1] // 2]
    assert abs(centre_lat - center_lat) < 1.0
    assert abs(centre_lon - center_lon) < 1.0


def test_projected_supergrid_from_center_rotated():
    """from_center with angle_deg produces a rotated grid distinct from the unrotated one."""
    pytest.importorskip("pyproj")
    center_lat, center_lon = 40.0, -70.0
    width_m = height_m = 200_000
    resolution_m = 50_000
    sg_0 = ProjectedSupergrid.from_center(
        center_lat, center_lon, width_m, height_m, resolution_m, angle_deg=0.0
    )
    sg_45 = ProjectedSupergrid.from_center(
        center_lat, center_lon, width_m, height_m, resolution_m, angle_deg=45.0
    )
    # Shapes must be identical
    assert sg_45.x.shape == sg_0.x.shape
    # Rotation should shift the corner coordinates
    assert not np.allclose(sg_0.x, sg_45.x)
    assert not np.allclose(sg_0.y, sg_45.y)
    # Centre node should still be near the requested point regardless of rotation
    cy = sg_45.y[sg_45.y.shape[0] // 2, sg_45.y.shape[1] // 2]
    cx = sg_45.x[sg_45.x.shape[0] // 2, sg_45.x.shape[1] // 2]
    assert abs(cy - center_lat) < 1.0
    assert abs(cx - center_lon) < 1.0


def test_projected_supergrid_from_latlon():
    """_from_latlon builds a valid ProjectedSupergrid from synthetic lat/lon arrays."""
    ny, nx = 4, 6  # logical grid cells → supergrid shape (2*ny+1, 2*nx+1)
    lon = np.linspace(-10, 10, 2 * nx + 1)
    lat = np.linspace(30, 40, 2 * ny + 1)
    lon2d, lat2d = np.meshgrid(lon, lat)
    sg = ProjectedSupergrid._from_latlon(lon2d, lat2d, {"grid_type": "projected_crs"})
    assert isinstance(sg, ProjectedSupergrid)
    assert sg.x.shape == (2 * ny + 1, 2 * nx + 1)
    assert sg.area.shape == (2 * ny, 2 * nx)
    assert np.all(sg.area > 0)
    assert sg.axis_units == "degrees"
