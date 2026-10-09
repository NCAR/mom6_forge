# Command line

`mom6_forge` installs a `mom6_forge` command with one subcommand per class:
`grid`, `topo`, and `vgrid`. The command line mirrors the Python API. A command's
options are the class constructor's arguments, and after them you chain the
methods to run, in order, each with its own options:

```bash
mom6_forge topo --grid ocean_hgrid.nc --min-depth 10 \
    set_flat --D 1000 \
    write_topo --file-path ocean_topog.nc
```

is the same as

```python
topo = Topo(Grid.from_supergrid("ocean_hgrid.nc"), min_depth=10)
topo.set_flat(D=1000)
topo.write_topo(file_path="ocean_topog.nc")
```

Option names are the Python argument names with `_` written as `-`. Values are
read as Python literals where possible (`500`, `True`, `[1, 2]`) and as strings
otherwise. Run `mom6_forge <command> --help` to list a command's options and
methods, and `mom6_forge <command> [options] <method> --help` for a method's options.

## Horizontal grid

Options are the arguments of `Grid()`; chain `write_supergrid` to save it.

```bash
mom6_forge grid --lenx 4 --leny 3 --resolution 0.05 --xstart 278 --ystart 7 --name panama \
    write_supergrid --path ocean_hgrid_panama.nc
```

## Bathymetry

`--grid` (a supergrid file) and `--min-depth` start the bathymetry. Set the depth
with `set_flat`, `set_from_dataset`, `set_bowl`, or `set_spoon`, then write it with
`write_topo` (and, if needed, `write_scrip_grid` or `write_esmf_mesh` in the same
chain).

```bash
mom6_forge topo --grid ocean_hgrid_panama.nc --min-depth 10 \
    set_from_dataset --bathymetry-path GEBCO_2024.nc \
        --longitude-coordinate-name lon --latitude-coordinate-name lat \
        --vertical-coordinate-name elevation \
    write_topo --file-path ocean_topog_panama.nc
```

See [the bathymetry workflow](bathymetry_workflow.md) for what `set_from_dataset`'s
mask and depth methods do.

## Vertical grid

Start the chain with the constructor to use, `uniform` or `hyperbolic`, then `write`:

```bash
mom6_forge vgrid uniform --nk 50 --depth 1000 write --filename ocean_vgrid_panama.nc
mom6_forge vgrid hyperbolic --nk 50 --depth 1000 --ratio 0.1 write --filename ocean_vgrid_panama.nc
```

## Exposing more methods

The commands are declared in `COMMANDS` in `mom6_forge/cli.py`. To expose another
method, add its name to that command's `methods` list; its options and help text
are generated from its signature and docstring.
