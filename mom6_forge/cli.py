"""
Command line interface for mom6_forge.

Each subcommand builds one MOM6 input file from command line arguments:

    mom6_forge grid  --lenx 4 --leny 3 --resolution 0.05 --xstart 278 --ystart 7 \
                     --name panama -o ocean_hgrid.nc
    mom6_forge topo  --grid ocean_hgrid.nc --min-depth 10 --flat 1000 -o ocean_topog.nc
    mom6_forge vgrid --nk 50 --depth 1000 -o ocean_vgrid.nc
"""

import argparse
import sys
from pathlib import Path


def _check_output(args):
    if Path(args.output).exists() and not args.overwrite:
        print(
            f"Refusing to overwrite {args.output} (pass --overwrite).",
            file=sys.stderr,
        )
        sys.exit(1)


def _grid(args):
    from mom6_forge.grid import Grid

    _check_output(args)
    grid = Grid(
        lenx=args.lenx,
        leny=args.leny,
        nx=args.nx,
        ny=args.ny,
        resolution=args.resolution,
        xstart=args.xstart,
        ystart=args.ystart,
        cyclic_x=args.cyclic_x,
        name=args.name,
        type=args.type,
    )
    grid.write_supergrid(args.output)
    print(f"Grid written to: {args.output}")


def _topo(args):
    from mom6_forge.grid import Grid
    from mom6_forge.topo import Topo

    _check_output(args)
    grid = Grid.from_supergrid(args.grid)
    topo = Topo(grid, args.min_depth)
    if args.flat is not None:
        topo.set_flat(args.flat)
    else:
        missing = [
            f"--{flag}"
            for flag in ("lon-name", "lat-name", "elevation-name")
            if getattr(args, flag.replace("-", "_")) is None
        ]
        if missing:
            args.subparser.error(f"--dataset also needs {', '.join(missing)}")
        topo.set_from_dataset(
            bathymetry_path=args.dataset,
            longitude_coordinate_name=args.lon_name,
            latitude_coordinate_name=args.lat_name,
            vertical_coordinate_name=args.elevation_name,
            fill_channels=args.fill_channels,
            is_input_positive_below_msl=args.positive_down,
            mask_method=args.mask_method,
            depth_method=args.depth_method,
            regridding_method=args.regridding_method,
        )
    topo.write_topo(args.output)
    print(f"Bathymetry written to: {args.output}")


def _vgrid(args):
    from mom6_forge.vgrid import VGrid

    _check_output(args)
    if args.type == "uniform":
        if args.ratio is not None:
            args.subparser.error("--ratio only applies to --type hyperbolic")
        vgrid = VGrid.uniform(nk=args.nk, depth=args.depth, name=args.name)
    else:
        if args.ratio is None:
            args.subparser.error("--type hyperbolic needs --ratio")
        vgrid = VGrid.hyperbolic(
            nk=args.nk, depth=args.depth, ratio=args.ratio, name=args.name
        )
    vgrid.write(args.output)
    print(f"Vertical grid written to: {args.output}")


def _add_output_args(parser):
    parser.add_argument(
        "-o", "--output", required=True, help="Path of the netCDF file to write."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mom6_forge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- grid ---
    grid_parser = subparsers.add_parser(
        "grid", help="Write a MOM6 horizontal grid (supergrid) file."
    )
    grid_parser.add_argument(
        "--lenx", type=float, required=True, help="Grid length in x (degrees)."
    )
    grid_parser.add_argument(
        "--leny", type=float, required=True, help="Grid length in y (degrees)."
    )
    grid_parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Grid spacing (degrees). Give either this or --nx and --ny.",
    )
    grid_parser.add_argument("--nx", type=int, default=None, help="Cells in x.")
    grid_parser.add_argument("--ny", type=int, default=None, help="Cells in y.")
    grid_parser.add_argument(
        "--xstart", type=float, default=0.0, help="Starting x coordinate."
    )
    grid_parser.add_argument(
        "--ystart",
        type=float,
        default=None,
        help="Starting y coordinate (default: -leny/2, centering on the Equator).",
    )
    grid_parser.add_argument(
        "--cyclic-x",
        action="store_true",
        default=False,
        dest="cyclic_x",
        help="Make the grid cyclic in x (needs --lenx 360).",
    )
    grid_parser.add_argument(
        "--type",
        choices=["uniform_spherical", "rectilinear_cartesian"],
        default="uniform_spherical",
        help="Grid type.",
    )
    grid_parser.add_argument("--name", default=None, help="Grid name.")
    _add_output_args(grid_parser)
    grid_parser.set_defaults(func=_grid)

    # --- topo ---
    topo_parser = subparsers.add_parser(
        "topo", help="Write a MOM6 bathymetry (TOPO_FILE) for an existing grid."
    )
    topo_parser.add_argument(
        "--grid", required=True, help="Path to the ocean_hgrid (supergrid) file."
    )
    topo_parser.add_argument(
        "--min-depth",
        type=float,
        required=True,
        dest="min_depth",
        help="Minimum ocean depth (m); shallower cells become land.",
    )
    source = topo_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--flat", type=float, default=None, help="Set a uniform depth (m)."
    )
    source.add_argument(
        "--dataset",
        default=None,
        help="Regrid depth from this bathymetry netCDF file (e.g. GEBCO).",
    )
    ds_args = topo_parser.add_argument_group("--dataset options")
    ds_args.add_argument(
        "--lon-name", dest="lon_name", help="Longitude coordinate name (e.g. lon)."
    )
    ds_args.add_argument(
        "--lat-name", dest="lat_name", help="Latitude coordinate name (e.g. lat)."
    )
    ds_args.add_argument(
        "--elevation-name",
        dest="elevation_name",
        help="Elevation/depth variable name (e.g. elevation).",
    )
    ds_args.add_argument(
        "--positive-down",
        action="store_true",
        default=False,
        dest="positive_down",
        help="The dataset's variable is positive below sea level (depth, not elevation).",
    )
    ds_args.add_argument(
        "--fill-channels",
        action="store_true",
        default=False,
        dest="fill_channels",
        help="Fill narrow channels in the final depth.",
    )
    ds_args.add_argument(
        "--mask-method",
        choices=["naturalearth", "ocean_frac", "dataset"],
        default=None,
        dest="mask_method",
        help="Land/ocean masking method (default: chosen from resolution diagnostics).",
    )
    ds_args.add_argument(
        "--depth-method",
        choices=["stats", "xesmf"],
        default=None,
        dest="depth_method",
        help="Depth method (default: chosen from resolution diagnostics).",
    )
    ds_args.add_argument(
        "--regridding-method",
        default="bilinear",
        dest="regridding_method",
        help="xESMF regridding method when the xesmf depth method is used.",
    )
    _add_output_args(topo_parser)
    topo_parser.set_defaults(func=_topo, subparser=topo_parser)

    # --- vgrid ---
    vgrid_parser = subparsers.add_parser(
        "vgrid", help="Write a MOM6 vertical grid (layer thickness) file."
    )
    vgrid_parser.add_argument(
        "--nk", type=int, required=True, help="Number of vertical layers."
    )
    vgrid_parser.add_argument(
        "--depth", type=float, required=True, help="Total depth (m)."
    )
    vgrid_parser.add_argument(
        "--type",
        choices=["uniform", "hyperbolic"],
        default="uniform",
        help="Layer spacing.",
    )
    vgrid_parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Target ratio of top to bottom layer thicknesses (--type hyperbolic only).",
    )
    vgrid_parser.add_argument("--name", default=None, help="Vertical grid name.")
    _add_output_args(vgrid_parser)
    vgrid_parser.set_defaults(func=_vgrid, subparser=vgrid_parser)

    args = parser.parse_args(argv)
    args.func(args)
