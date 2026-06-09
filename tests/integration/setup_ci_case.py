"""
CrocoDash case setup for CESM integration CI.
Verifies all mom6_forge output files by running the full configure/process forcings flow.
Runs inside the crocontainer after this mom6_forge commit is pip-installed.

Env vars: CESMROOT (required), DIN_LOC_ROOT, CASEROOT, INPUTDIR
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

CESMROOT = os.environ.get("CESMROOT") or sys.exit("CESMROOT not set")
CASEROOT = Path(os.environ.get("CASEROOT", "/workspace/case"))
INPUTDIR = Path(os.environ.get("INPUTDIR", "/workspace/inputdir"))
DIN_LOC_ROOT = Path(os.environ.get("DIN_LOC_ROOT", "/root/cesm/inputdata"))

os.environ.setdefault("CIME_MACHINE", "ubuntu-latest")

from CrocoDash.case import Case
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.vgrid import VGrid

# Flat 3x3 degree box (Panama region, same as tutorial)
grid = Grid(
    resolution=0.1, xstart=278.0, lenx=3.0, ystart=7.0, leny=3.0, name="ci_flat"
)
topo = Topo(grid=grid, min_depth=9.5, git=False)
topo.set_flat(500.0)
vgrid = VGrid.hyperbolic(nk=20, depth=500.0, ratio=20.0)

# T62 ESMF mesh bundled with NYF inputdata; used as ROF mesh stand-in for gen_rof_maps
t62_mesh = DIN_LOC_ROOT / "share/meshes/T62_040121_ESMFmesh.nc"

# Case() writes hgrid, topog, cice_grid, esmf_mesh, scrip_grid, vgrid to inputdir
# NYF atmosphere + GLOFAS runoff so process_forcings exercises both chl and runoff paths
case = Case(
    cesmroot=CESMROOT,
    caseroot=str(CASEROOT),
    inputdir=str(INPUTDIR),
    ocn_grid=grid,
    ocn_vgrid=vgrid,
    ocn_topo=topo,
    project="PROJ123",
    machine="ubuntu-latest",
    compset="1850_DATM%NYF_SLND_SICE_MOM6_DROF%GLOFAS_SGLC_SWAV",
    atm_grid_name="T62",
    rof_grid_name="GLOFAS",
    override=True,
)

# Synthetic GLORYS-format OBC/IC data so process_forcings() has something to regrid
bounds = Grid.get_bounding_boxes_of_rectangular_grid(grid)["ic"]
raw_data = INPUTDIR / "extract_forcings" / "raw_data"
raw_data.mkdir(parents=True, exist_ok=True)


def _synthetic_glorys(path):
    lat = np.linspace(bounds["lat_min"], bounds["lat_max"], 20)
    lon = np.linspace(bounds["lon_min"], bounds["lon_max"], 20)
    depth = np.array([0, 500, 1000, 2000, 3000, 4000, 5000], dtype=np.float64)
    xr.Dataset(
        {
            "so": (
                ("time", "depth", "latitude", "longitude"),
                np.full((32, 7, 20, 20), 35.0),
            ),
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                np.full((32, 7, 20, 20), 20.0),
            ),
            "uo": (
                ("time", "depth", "latitude", "longitude"),
                np.zeros((32, 7, 20, 20)),
            ),
            "vo": (
                ("time", "depth", "latitude", "longitude"),
                np.zeros((32, 7, 20, 20)),
            ),
            "zos": (("time", "latitude", "longitude"), np.zeros((32, 20, 20))),
        },
        coords={
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
            "time": np.arange(32),
        },
    ).to_netcdf(str(path))


for fname in [
    "ic_unprocessed.nc",
    "east_unprocessed.20200101_20200201.nc",
    "west_unprocessed.20200101_20200201.nc",
    "north_unprocessed.20200101_20200201.nc",
    "south_unprocessed.20200101_20200201.nc",
]:
    _synthetic_glorys(raw_data / fname)

# Chlorophyll source — use real SeaWiFS if available, otherwise synthetic
seawifs_src = (
    DIN_LOC_ROOT / "ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc"
)
if not seawifs_src.exists():
    seawifs_src = INPUTDIR / "ocnice" / "seawifs_src.nc"
    seawifs_src.parent.mkdir(parents=True, exist_ok=True)
    lon = np.linspace(0, 360, 720, endpoint=False)
    lat = np.linspace(-89.75, 89.75, 360)
    xr.Dataset(
        {
            "chlor_a": (
                ["time", "lat", "lon"],
                np.full((12, len(lat), len(lon)), 0.3, dtype=np.float32),
            )
        },
        coords={"time": np.arange(12, dtype=float), "lat": lat, "lon": lon},
    ).to_netcdf(str(seawifs_src))

# configure_forcings activates ChlConfigurator (via chl_processed_filepath) and
# RunoffConfigurator (via rof_esmf_mesh_filepath; DROF%GLOFAS compset makes it required).
# process_forcings then handles OBC/IC, chlorophyll, and runoff mapping in one call.
case.configure_forcings(
    date_range=["2020-01-01 00:00:00", "2020-02-01 00:00:00"],
    function_name="get_glorys_data_from_cds_api",
    chl_processed_filepath=seawifs_src,
    rof_esmf_mesh_filepath=t62_mesh,
)
case.process_forcings()
