# Command line

`mom6_forge` installs a `mom6_forge` command with one subcommand per MOM6 input
file. Each writes a single netCDF file and refuses to overwrite an existing one
unless you pass `--overwrite`. Run `mom6_forge <subcommand> --help` for every option.

## Horizontal grid

```bash
mom6_forge grid --lenx 4 --leny 3 --resolution 0.05 --xstart 278 --ystart 7 \
    --name panama -o ocean_hgrid_panama.nc
```

Give either `--resolution` or `--nx` and `--ny`. `--type` selects
`uniform_spherical` (default) or `rectilinear_cartesian`, and `--cyclic-x` makes a
360° grid periodic in x.

## Bathymetry

`topo` needs an existing horizontal grid file. Set a flat bottom:

```bash
mom6_forge topo --grid ocean_hgrid_panama.nc --min-depth 10 --flat 1000 \
    -o ocean_topog_panama.nc
```

or regrid from a bathymetry dataset such as GEBCO (see
[the bathymetry workflow](bathymetry_workflow.md) for what the mask and depth
methods do):

```bash
mom6_forge topo --grid ocean_hgrid_panama.nc --min-depth 10 \
    --dataset GEBCO_2024.nc --lon-name lon --lat-name lat --elevation-name elevation \
    -o ocean_topog_panama.nc
```

## Vertical grid

```bash
mom6_forge vgrid --nk 50 --depth 1000 -o ocean_vgrid_panama.nc
mom6_forge vgrid --type hyperbolic --nk 50 --depth 1000 --ratio 0.1 -o ocean_vgrid_panama.nc
```

`--ratio` is the target ratio of top to bottom layer thickness, and only applies
to `--type hyperbolic`.
