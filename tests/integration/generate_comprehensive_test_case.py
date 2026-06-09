"""
Generate a mom6_forge test case and bundle it for CESM execution.
Runs inside the crocontainer after this mom6_forge commit is pip-installed.

Env vars: CESMROOT (required), DIN_LOC_ROOT, CASEROOT, INPUTDIR, BUNDLE_DIR
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

CESMROOT = os.environ.get("CESMROOT") or sys.exit("CESMROOT not set")
CASEROOT = Path(os.environ.get("CASEROOT", "/workspace/case"))
INPUTDIR = Path(os.environ.get("INPUTDIR", "/workspace/inputdir"))
BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "/workspace/bundle"))
DIN_LOC_ROOT = Path(os.environ.get("DIN_LOC_ROOT", "/root/cesm/inputdata"))

os.environ.setdefault("CIME_MACHINE", "ubuntu-latest")

from CrocoDash.case import Case
from CrocoDash.grid import Grid
from CrocoDash.shareable.bundle import BundleCrocoDashCase
from CrocoDash.topo import Topo
from CrocoDash.vgrid import VGrid
from mom6_forge.chl import interpolate_and_fill_seawifs
from mom6_forge.mapping import gen_rof_maps

# --- Grid / topo / vgrid (flat bottom, no GEBCO needed) ---
grid = Grid(resolution=0.1, xstart=278.0, lenx=3.0, ystart=7.0, leny=3.0, name="ci_flat")
topo = Topo(grid=grid, min_depth=9.5, git=False)
topo.set_flat(500.0)
vgrid = VGrid.hyperbolic(nk=20, depth=500.0, ratio=20.0)

# --- CESM case (writes hgrid, topog, cice_grid, esmf_mesh, scrip_grid, vgrid) ---
# rof_grid_name="GLOFAS" is required for CR_JRA_GLOFAS (DROF component, not SROF)
case = Case(
    cesmroot=CESMROOT,
    caseroot=str(CASEROOT),
    inputdir=str(INPUTDIR),
    ocn_grid=grid,
    ocn_vgrid=vgrid,
    ocn_topo=topo,
    project="PROJ123",
    machine="ubuntu-latest",
    compset="CR_JRA_GLOFAS",
    rof_grid_name="GLOFAS",
    override=True,
)

# --- Synthetic OBC / IC forcing data ---
bounds = Grid.get_bounding_boxes_of_rectangular_grid(grid)["ic"]
raw_data = INPUTDIR / "extract_forcings" / "raw_data"
raw_data.mkdir(parents=True, exist_ok=True)

def _synthetic_glorys(path):
    lat = np.linspace(bounds["lat_min"], bounds["lat_max"], 20)
    lon = np.linspace(bounds["lon_min"], bounds["lon_max"], 20)
    depth = np.array([0, 500, 1000, 2000, 3000, 4000, 5000], dtype=np.float64)
    ds = xr.Dataset(
        {
            "so":     (("time","depth","latitude","longitude"), np.full((32,7,20,20), 35.0)),
            "thetao": (("time","depth","latitude","longitude"), np.full((32,7,20,20), 20.0)),
            "uo":     (("time","depth","latitude","longitude"), np.zeros((32,7,20,20))),
            "vo":     (("time","depth","latitude","longitude"), np.zeros((32,7,20,20))),
            "zos":    (("time","latitude","longitude"),         np.zeros((32,20,20))),
        },
        coords={"depth": depth, "latitude": lat, "longitude": lon, "time": np.arange(32)},
    )
    ds.to_netcdf(str(path))

for fname in [
    "ic_unprocessed.nc",
    "east_unprocessed.20200101_20200201.nc",
    "west_unprocessed.20200101_20200201.nc",
    "north_unprocessed.20200101_20200201.nc",
    "south_unprocessed.20200101_20200201.nc",
]:
    _synthetic_glorys(raw_data / fname)

# ForcingConfigRegistry auto-detects DROF from CR_JRA_GLOFAS and activates
# RunoffConfigurator; process_forcings() will call process_runoff() accordingly
case.configure_forcings(
    date_range=["2020-01-01 00:00:00", "2020-02-01 00:00:00"],
    function_name="get_glorys_data_from_cds_api",
)
case.process_forcings()

# --- Chlorophyll (real SeaWiFS if available, else synthetic) ---
seawifs_src = DIN_LOC_ROOT / "ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc"
if not seawifs_src.exists():
    seawifs_src = INPUTDIR / "ocnice" / "seawifs_src.nc"
    lon = np.linspace(0, 360, 720, endpoint=False)
    lat = np.linspace(-89.75, 89.75, 360)
    xr.Dataset(
        {"chlor_a": (["time","lat","lon"], np.full((12,len(lat),len(lon)), 0.3, dtype=np.float32))},
        coords={"time": np.arange(12, dtype=float), "lat": lat, "lon": lon},
    ).to_netcdf(str(seawifs_src))

interpolate_and_fill_seawifs(
    grid, topo, seawifs_src,
    output_path=INPUTDIR / "ocnice" / f"seawifs-clim-1997-2010-{grid.name}.nc",
)

# --- Runoff mapping via mom6_forge.mapping (explicit code-path coverage) ---
# process_forcings() above also runs gen_rof_maps internally (writes to INPUTDIR/mapping/).
# This call exercises the same function from a different entry point, writing to ocnice/.
glofas_mesh = DIN_LOC_ROOT / "ocn/mom/croc/rof/glofas/dis24/GLOFAS_esmf_mesh_v4.nc"
gen_rof_maps(
    rof_mesh_path=glofas_mesh,
    ocn_mesh_path=case.esmf_mesh_path,
    output_dir=INPUTDIR / "ocnice",
    mapping_file_prefix=f"rof_to_{grid.name}",
)

# --- Bundle ---
bundler = BundleCrocoDashCase(str(CASEROOT))
bundler.identify_non_standard_CrocoDashCase_information(
    cesmroot=CESMROOT, machine="ubuntu-latest", project_number="PROJ123"
)
bundler.bundle(str(BUNDLE_DIR))
