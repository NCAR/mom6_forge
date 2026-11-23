import pytest
from mom6_forge.utils import (
    get_avg_resolution,
    get_avg_resolution_km,
    longitude_slicer,
    iterative_fill,
)
from mom6_forge._supergrid import (
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


def test_longitude_slicers_regionally():
    nx, ny = 4, 14

    latitude_extent = (2, 5)
    longitude_extent = (-90, -70)

    dims = ["random_lat", "random_lon"]

    dlambda = (longitude_extent[1] - longitude_extent[0]) / 2

    data = xr.DataArray(
        np.random.random((ny, nx)),
        dims=dims,
        coords={
            "random_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
            "random_lon": np.linspace(
                longitude_extent[0] - 2, longitude_extent[1] + 2, nx
            ),
        },
    )

    # Regular regional
    data_regular = longitude_slicer(data, longitude_extent, "random_lon")
    data_east = longitude_slicer(data, (270, 290), "random_lon")
    assert (data_regular == data_east).all()

    # Seam data
    longitude_extent = (-5, 5)
    data = xr.DataArray(
        np.random.random((ny, nx)),
        dims=dims,
        coords={
            "random_lat": np.linspace(latitude_extent[0], latitude_extent[1], ny),
            "random_lon": np.linspace(
                longitude_extent[0] - 2, longitude_extent[1] + 2, nx
            ),
        },
    )
    data_regular = longitude_slicer(data, longitude_extent, "random_lon")
    assert len(data_regular.random_lon) > 0


def test_quadrilateral_area_exception():
    v1 = np.dstack(latlon_to_cartesian(0, 0, R=2))
    v2 = np.dstack(latlon_to_cartesian(90, 0, R=2))
    v3 = np.dstack(latlon_to_cartesian(0, 90, R=2))
    v4 = np.dstack(latlon_to_cartesian(-90, 0, R=2.1))
    with pytest.raises(ValueError) as excinfo:
        quadrilateral_area(v1, v2, v3, v4)

    assert str(excinfo.value) == "vectors provided must have the same length"


def test_iterative_fill():
    """
    Simple 3x3 grid with one unfilled ocean cell surrounded by known values.
    The fill should average the neighbours and converge in 1 iteration.
    """
    # known depth everywhere except centre cell
    depth = np.array(
        [
            [100.0, 200.0, 100.0],
            [200.0, np.nan, 200.0],
            [100.0, 200.0, 100.0],
        ]
    )
    unfilled = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    mask = np.ones((3, 3), dtype=bool)

    result = iterative_fill(depth.copy(), unfilled, mask=xr.DataArray(mask))

    # centre cell should be filled with mean of 4 direct neighbours = 200
    assert result[1, 1] == pytest.approx(200.0), f"Expected 200, got {result[1,1]}"

    # all other cells unchanged
    depth[1, 1] = 200.0
    assert np.allclose(result, depth), "Non-unfilled cells were modified"

    # --- land cell should never be filled ---
    depth2 = np.array(
        [
            [100.0, 200.0, 100.0],
            [200.0, 0.0, 0.0],  # two unfilled: one ocean, one land
            [100.0, 200.0, 100.0],
        ]
    )
    unfilled2 = np.array(
        [
            [False, False, False],
            [False, True, True],
            [False, False, False],
        ]
    )
    mask2 = np.array(
        [
            [1, 1, 1],
            [1, 1, 0],  # right cell is land
            [1, 1, 1],
        ],
        dtype=bool,
    )

    result2 = iterative_fill(depth2.copy(), unfilled2, mask=xr.DataArray(mask2))
    assert result2[1, 1] != 0, "Ocean unfilled cell was not filled"
    assert result2[1, 2] == 0, "Land cell should not be filled"


@pytest.mark.parametrize("start", [-180.0, 0.0])
@pytest.mark.parametrize("nlon", [4320, 4319, 721, 720])
def test_longitude_slicer_returns_whole_axis_for_global_request(nlon, start):
    """An at-least-global extent (e.g. a polar-cap bounding box plus a buffer)
    must come back as the whole longitude axis, not a sliver at the seam and not
    one point short on an odd-length axis. Cell-centred data on 0..360 doesn't
    hold 0, the midpoint of a -180..180 request, and must work too."""
    data = xr.Dataset(
        {"v": ("lon", np.arange(nlon, dtype=float))},
        coords={"lon": start + (np.arange(nlon) + 0.5) * 360.0 / nlon},
    )
    for extent in ([-180.0, 180.0], [-181.0, 181.0], [-200.0, 200.0]):
        sliced = longitude_slicer(data, extent, "lon")
        assert sliced.lon.size == nlon, f"{extent} on nlon={nlon}"


@pytest.mark.parametrize(
    "start, extent",
    [(-180.0, (174.5, 185.5)), (-180.0, (-185.5, -174.5)), (0.0, (-5.5, 5.5))],
)
def test_longitude_slicer_box_centred_on_data_seam(start, extent):
    """Global cell-centred data stops half a cell short of its seam (GEBCO on
    -180..180). A box centred on the seam must still slice: increasing
    longitudes, the right data under each, and cells covering the box."""
    d = 0.25
    lon = start + (np.arange(1440) + 0.5) * d
    data = xr.Dataset({"v": ("lon", lon)}, coords={"lon": lon})

    sliced = longitude_slicer(data, extent, "lon")

    out = sliced.lon.values
    assert np.all(np.diff(out) > 0)
    np.testing.assert_array_equal(np.mod(sliced.v.values - out, 360.0), 0.0)
    assert out[0] - d / 2 <= extent[0] and out[-1] + d / 2 >= extent[1]
