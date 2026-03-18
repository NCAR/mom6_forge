import pytest
from mom6_bathy.utils import (
    get_avg_resolution,
    get_avg_resolution_km,
    longitude_slicer,
    quadrilateral_area,
    latlon_to_cartesian,
    quadrilateral_areas,
)
from utils import on_cisl_machine
import xarray as xr
import numpy as np


def test_avg_resolution():
    """Test the average resolution calculation for a grid."""

    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")

    t232_avg_res = get_avg_resolution(
        "/glade/campaign/cesm/cesmdata/inputdata/share/meshes/tx2_3v2_230415_ESMFmesh.nc"
    )
    assert (
        0.49 < t232_avg_res < 0.50
    ), "Average resolution for tx2_3v2 should be around 0.5 degrees"

    t232_avg_res_km = get_avg_resolution_km(
        "/glade/campaign/cesm/cesmdata/inputdata/share/meshes/tx2_3v2_230415_ESMFmesh.nc"
    )
    assert (
        40.0 < t232_avg_res_km < 41.0
    ), "Average resolution for tx2_3v2 should be around 40 km"


def test_longitude_slicer():
    with pytest.raises(AssertionError):
        nx, ny, nt = 4, 14, 5

        latitude_extent = (10, 20)
        longitude_extent = (12, 18)

        dims = ["random_lat", "random_lon", "time"]

        dlambda = (longitude_extent[1] - longitude_extent[0]) / 2

        data = xr.DataArray(
            np.random.random((ny, nx, nt)),
            dims=dims,
            coords={
                "random_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
                "random_lon": np.array(
                    [
                        longitude_extent[0],
                        longitude_extent[0] + 1.5 * dlambda,
                        longitude_extent[0] + 2.6 * dlambda,
                        longitude_extent[1],
                    ]
                ),
                "time": np.linspace(0, 1000, nt),
            },
        )

        longitude_slicer(data, longitude_extent, "random_lon")


@pytest.mark.parametrize(
    ("v1", "v2", "v3", "v4", "true_area"),
    [
        (
            np.dstack(latlon_to_cartesian(0, 0)),
            np.dstack(latlon_to_cartesian(0, 90)),
            np.dstack(latlon_to_cartesian(90, 0)),
            np.dstack(latlon_to_cartesian(0, -90)),
            np.pi,
        ),
        (
            np.dstack(latlon_to_cartesian(0, 0)),
            np.dstack(latlon_to_cartesian(90, 0)),
            np.dstack(latlon_to_cartesian(0, 90)),
            np.dstack(latlon_to_cartesian(-90, 0)),
            np.pi,
        ),
    ],
)
def test_quadrilateral_area(v1, v2, v3, v4, true_area):
    assert np.isclose(quadrilateral_area(v1, v2, v3, v4), true_area)


# create a lat-lon mesh that covers 1/4 of the North Hemisphere
lon1, lat1 = np.meshgrid(np.linspace(0, 90, 5), np.linspace(0, 90, 5))
area1 = 1 / 8 * (4 * np.pi)

# create a lat-lon mesh that covers 1/4 of the whole globe
lon2, lat2 = np.meshgrid(np.linspace(-45, 45, 5), np.linspace(-90, 90, 5))
area2 = 1 / 4 * (4 * np.pi)


@pytest.mark.parametrize(
    ("lat", "lon", "true_area"),
    [
        (lat1, lon1, area1),
        (lat2, lon2, area2),
    ],
)
def test_quadrilateral_areas(lat, lon, true_area):
    assert np.isclose(np.sum(quadrilateral_areas(lat, lon)), true_area)


@pytest.mark.parametrize(
    ("lat", "lon", "true_xyz"),
    [
        (0, 0, (1, 0, 0)),
        (90, 0, (0, 0, 1)),
        (0, 90, (0, 1, 0)),
        (-90, 0, (0, 0, -1)),
    ],
)
def test_latlon_to_cartesian(lat, lon, true_xyz):
    assert np.isclose(latlon_to_cartesian(lat, lon), true_xyz).all()


def test_quadrilateral_area_exception():
    v1 = np.dstack(latlon_to_cartesian(0, 0, R=2))
    v2 = np.dstack(latlon_to_cartesian(90, 0, R=2))
    v3 = np.dstack(latlon_to_cartesian(0, 90, R=2))
    v4 = np.dstack(latlon_to_cartesian(-90, 0, R=2.1))
    with pytest.raises(ValueError) as excinfo:
        quadrilateral_area(v1, v2, v3, v4)

    assert str(excinfo.value) == "vectors provided must have the same length"
