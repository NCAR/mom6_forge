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
