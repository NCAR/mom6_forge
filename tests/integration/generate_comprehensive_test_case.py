"""
Generate a comprehensive mom6_forge test case and bundle it for CESM execution.

Runs inside the crocontainer (ghcr.io/crocodile-cesm/crocontainer:latest-amd64).
This specific version of mom6_forge has been pip-installed into the CrocoDash conda env
before this script is called, so all imports come from the freshly-tested commit.

Usage:
    conda run -n CrocoDash python generate_comprehensive_test_case.py \
        --caseroot /workspace/case \
        --inputdir /workspace/inputdir \
        --bundle-dir /workspace/bundle \
        [--din-loc-root /root/cesm/inputdata]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data helpers (inline so this script is self-contained)
# ---------------------------------------------------------------------------


def _make_synthetic_glorys(lat_min, lat_max, lon_min, lon_max, output_path):
    """Glorys-formatted synthetic ocean state dataset used for OBC/IC data."""
    lat = np.linspace(lat_min, lat_max, 20)
    lon = np.linspace(lon_min, lon_max, 20)
    depth = np.array([0, 500, 1000, 2000, 3000, 4000, 5000], dtype=np.float64)
    time = np.arange(32)

    ds = xr.Dataset(
        {
            "so": (
                ("time", "depth", "latitude", "longitude"),
                np.full((32, len(depth), 20, 20), 35.0, dtype=np.float64),
            ),
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                np.full((32, len(depth), 20, 20), 20.0, dtype=np.float64),
            ),
            "uo": (
                ("time", "depth", "latitude", "longitude"),
                np.zeros((32, len(depth), 20, 20), dtype=np.float64),
            ),
            "vo": (
                ("time", "depth", "latitude", "longitude"),
                np.zeros((32, len(depth), 20, 20), dtype=np.float64),
            ),
            "zos": (
                ("time", "latitude", "longitude"),
                np.zeros((32, 20, 20), dtype=np.float64),
            ),
        },
        coords={
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
            "time": time,
        },
    )
    ds.to_netcdf(str(output_path))
    log.info("Wrote synthetic Glorys dataset to %s", output_path)
    return ds


def _make_synthetic_seawifs(output_path):
    """SeaWiFS-formatted synthetic source file for chlorophyll interpolation."""
    lon = np.linspace(0, 360, 720, endpoint=False)
    lat = np.linspace(-89.75, 89.75, 360)
    chlor = np.full((12, len(lat), len(lon)), 0.3, dtype=np.float32)
    ds = xr.Dataset(
        {"chlor_a": (["time", "lat", "lon"], chlor)},
        coords={"time": np.arange(12, dtype=float), "lat": lat, "lon": lon},
    )
    ds.to_netcdf(str(output_path))
    log.info("Wrote synthetic SeaWiFS dataset to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--caseroot",
        default=os.environ.get("CASEROOT", "/workspace/case"),
        help="CESM caseroot directory",
    )
    parser.add_argument(
        "--inputdir",
        default=os.environ.get("INPUTDIR", "/workspace/inputdir"),
        help="CrocoDash input directory",
    )
    parser.add_argument(
        "--bundle-dir",
        default=os.environ.get("BUNDLE_DIR", "/workspace/bundle"),
        help="Directory to write the case bundle",
    )
    parser.add_argument(
        "--din-loc-root",
        default=os.environ.get("DIN_LOC_ROOT", "/root/cesm/inputdata"),
        help="CESM inputdata root (DIN_LOC_ROOT)",
    )
    args = parser.parse_args()

    cesmroot = os.environ.get("CESMROOT")
    if not cesmroot:
        sys.exit("CESMROOT environment variable is not set")

    caseroot = Path(args.caseroot)
    inputdir = Path(args.inputdir)
    bundle_dir = Path(args.bundle_dir)
    din_loc_root = Path(args.din_loc_root)

    # ------------------------------------------------------------------
    # Step 1: Build mom6_forge objects (flat bathymetry — no GEBCO)
    # ------------------------------------------------------------------
    log.info("=== Step 1: Creating grid / topo / vgrid ===")

    # Delayed import: at runtime inside the container, the freshly pip-installed
    # mom6_forge is on sys.path ahead of the baked-in version.
    from CrocoDash.grid import Grid
    from CrocoDash.topo import Topo
    from CrocoDash.vgrid import VGrid

    grid = Grid(
        resolution=0.1,
        xstart=278.0,
        lenx=3.0,
        ystart=7.0,
        leny=3.0,
        name="ci_flat",
    )
    topo = Topo(grid=grid, min_depth=9.5, git=False)
    topo.set_flat(500.0)
    vgrid = VGrid.hyperbolic(nk=20, depth=500.0, ratio=20.0)

    log.info(
        "Grid: %d × %d cells at 0.1° resolution (flat 500 m bathymetry, 20 levels)",
        grid.nx,
        grid.ny,
    )

    # ------------------------------------------------------------------
    # Step 2: Create CESM case — writes all 6 core mom6_forge files
    # ------------------------------------------------------------------
    log.info("=== Step 2: Creating CESM case (writes MOM6 grid files) ===")

    from CrocoDash.case import Case

    os.environ.setdefault("CIME_MACHINE", "ubuntu-latest")

    case = Case(
        cesmroot=cesmroot,
        caseroot=str(caseroot),
        inputdir=str(inputdir),
        ocn_grid=grid,
        ocn_vgrid=vgrid,
        ocn_topo=topo,
        project="PROJ123",
        machine="ubuntu-latest",
        compset="CR_JRA",
        override=True,
    )
    log.info("Case created at %s", caseroot)
    log.info(
        "Grid files written to %s/ocnice/", inputdir
    )

    # ------------------------------------------------------------------
    # Step 3: Set up synthetic OBC / IC forcing data
    # ------------------------------------------------------------------
    log.info("=== Step 3: Writing synthetic OBC / IC forcing data ===")

    from CrocoDash.grid import Grid as _Grid

    bounds = _Grid.get_bounding_boxes_of_rectangular_grid(grid)
    ic_bounds = bounds["ic"]

    raw_data = inputdir / "extract_forcings" / "raw_data"
    raw_data.mkdir(parents=True, exist_ok=True)

    for fname in [
        "ic_unprocessed.nc",
        "east_unprocessed.20200101_20200201.nc",
        "west_unprocessed.20200101_20200201.nc",
        "north_unprocessed.20200101_20200201.nc",
        "south_unprocessed.20200101_20200201.nc",
    ]:
        _make_synthetic_glorys(
            ic_bounds["lat_min"],
            ic_bounds["lat_max"],
            ic_bounds["lon_min"],
            ic_bounds["lon_max"],
            raw_data / fname,
        )

    # ------------------------------------------------------------------
    # Step 4: Configure and process forcings
    # ------------------------------------------------------------------
    log.info("=== Step 4: Configuring forcings ===")

    case.configure_forcings(
        date_range=["2020-01-01 00:00:00", "2020-02-01 00:00:00"],
        function_name="get_glorys_data_from_cds_api",
    )

    log.info("=== Step 5: Processing forcings (regridding synthetic data) ===")
    case.process_forcings()

    # ------------------------------------------------------------------
    # Step 6: Generate chlorophyll file
    # ------------------------------------------------------------------
    log.info("=== Step 6: Generating chlorophyll file ===")

    from mom6_forge.chl import interpolate_and_fill_seawifs

    seawifs_candidates = [
        din_loc_root
        / "ocn/mom/croc/chl/data/SeaWIFS.L3m.MC.CHL.chlor_a.0.25deg.nc",
    ]
    seawifs_src = next((p for p in seawifs_candidates if p.exists()), None)

    if seawifs_src is None:
        log.warning(
            "Real SeaWiFS source not found under DIN_LOC_ROOT=%s; "
            "using synthetic data instead",
            din_loc_root,
        )
        seawifs_src = inputdir / "ocnice" / "seawifs_synthetic_src.nc"
        _make_synthetic_seawifs(seawifs_src)

    chl_out = inputdir / "ocnice" / f"seawifs-clim-1997-2010-{grid.name}.nc"
    interpolate_and_fill_seawifs(grid, topo, seawifs_src, output_path=chl_out)
    log.info("Chlorophyll file: %s", chl_out)

    # ------------------------------------------------------------------
    # Step 7: Generate runoff mapping files
    # ------------------------------------------------------------------
    log.info("=== Step 7: Generating runoff mapping files ===")

    from mom6_forge.mapping import gen_rof_maps

    esmf_mesh = case.esmf_mesh_path
    rof_mesh_candidates = list(
        din_loc_root.glob("rof/mizuroute/**/mosart_*.nc")
    ) + list(din_loc_root.glob("rof/**/*esmf*.nc"))

    if rof_mesh_candidates:
        rof_mesh = rof_mesh_candidates[0]
        log.info("Using ROF mesh: %s", rof_mesh)
        gen_rof_maps(
            rof_mesh_path=rof_mesh,
            ocn_mesh_path=esmf_mesh,
            output_dir=inputdir / "ocnice",
            mapping_file_prefix=f"rof_to_{grid.name}",
        )
    else:
        log.warning(
            "No ROF ESMF mesh found under DIN_LOC_ROOT=%s; "
            "skipping runoff mapping (CR_JRA uses SROF so this is optional)",
            din_loc_root,
        )

    # ------------------------------------------------------------------
    # Step 8: Bundle the case
    # ------------------------------------------------------------------
    log.info("=== Step 8: Bundling case to %s ===", bundle_dir)

    from CrocoDash.shareable.bundle import BundleCrocoDashCase

    bundler = BundleCrocoDashCase(str(caseroot))
    bundler.identify_non_standard_CrocoDashCase_information(
        cesmroot=cesmroot,
        machine="ubuntu-latest",
        project_number="PROJ123",
    )
    bundler.bundle(str(bundle_dir))
    log.info("Bundle written to %s", bundle_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=== Done. Files written by this mom6_forge commit: ===")
    for f in sorted((inputdir / "ocnice").iterdir()):
        log.info("  %s  (%d bytes)", f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
