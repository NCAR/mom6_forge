import pytest
from mom6_forge._supergrid import *
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


def test_uniform_spherical_supergrid():
    nx, ny = 10, 10
    sg = UniformSphericalSupergrid.from_extents(
        lon_min=0.0, len_x=10.0, lat_min=40.0, len_y=10.0, nx=nx, ny=ny
    )
    assert isinstance(sg, UniformSphericalSupergrid)
