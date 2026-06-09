from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.chl import interpolate_and_fill_seawifs
import numpy as np
import pytest
import os
import xarray as xr
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


def test_chl_synthetic_source(tmp_path, get_rect_grid, get_rect_topo_without_vc):
    """Test interpolate_and_fill_seawifs with synthetic source data (no GLADE needed)."""
    grid = get_rect_grid
    grid.name = "ci_chl"
    topo = get_rect_topo_without_vc

    # Build a minimal synthetic SeaWiFS-format source file
    lon = np.linspace(0, 360, 720, endpoint=False)
    lat = np.linspace(-89.75, 89.75, 360)
    src = xr.Dataset(
        {"chlor_a": (["time", "lat", "lon"], np.full((12, len(lat), len(lon)), 0.3, dtype=np.float32))},
        coords={"time": np.arange(12, dtype=float), "lat": lat, "lon": lon},
    )
    src_path = tmp_path / "seawifs_src.nc"
    src.to_netcdf(str(src_path))

    out_path = tmp_path / "seawifs-clim.nc"
    interpolate_and_fill_seawifs(grid, topo, src_path, output_path=out_path)

    assert out_path.exists()
    ds = xr.open_dataset(str(out_path))
    for var in ("CHL_A", "LON", "LAT"):
        assert var in ds, f"seawifs-clim.nc missing variable: {var}"
    assert ds.sizes["TIME"] == 12
