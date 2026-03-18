from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from mom6_bathy.chl import interpolate_and_fill_seawifs
import pytest
import os
from utils import on_cisl_machine


def test_chl(tmp_path, get_rect_grid):
    """Test the creation of chl files."""
    if not on_cisl_machine():
        pytest.skip("This test is only for the derecho and casper machines")
    # attempt to create a regional grid object from scratch
    grid = get_rect_grid
    grid.name = "pan2"
    # create a corresponding bathymetry object
    topo = Topo(
        grid=grid,
        min_depth=9.5,  # in meters
    )
    topo.set_spoon(1000, 10)

    interpolate_and_fill_seawifs(
        grid,
        topo,
        processed_seawifs_path="/glade/campaign/cesm/cesmdata/cseg/inputdata/ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc",
        output_path=tmp_path / "seawifs-clim-1997-2010-pan-xesmf.nc",
    )

    assert os.path.exists(tmp_path / "seawifs-clim-1997-2010-pan-xesmf.nc")
