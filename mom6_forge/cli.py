"""
Command line interface for mom6_forge.

Most of mom6_forge is a multi-step process: build an object, call methods on
it, then write it out (``Topo(grid, min_depth)``, ``set_flat(D)``,
``write_topo(path)``). Each top-level command wraps one class that way: its
options are the constructor's arguments, and a short list of exposed methods
can be chained after it, each with options taken from its own signature:

    mom6_forge grid --lenx 4 --leny 3 --resolution 0.05 --xstart 278 --ystart 7 \\
        write_supergrid --path ocean_hgrid.nc
    mom6_forge topo --grid ocean_hgrid.nc --min-depth 10 \\
        set_flat --D 1000 write_topo --file-path ocean_topog.nc
    mom6_forge vgrid hyperbolic --nk 50 --depth 1000 --ratio 0.1 \\
        write --filename ocean_vgrid.nc

To expose another method, add its name to that command's ``methods`` in
COMMANDS; its options and help are generated from its signature and docstring.
"""

import ast
import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional

import click

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.vgrid import VGrid


def _load_topo(grid, min_depth):
    """Start a bathymetry on the horizontal grid in a supergrid (ocean_hgrid) file."""
    return Topo(Grid.from_supergrid(grid), min_depth)


@dataclass
class Command:
    """A class exposed as a command: how to construct it, and which methods to chain."""

    cls: type
    help: str
    # Builds the object from the command's own options. None means the chain
    # starts with a classmethod that returns a new instance (e.g. VGrid.uniform).
    init: Optional[Callable] = None
    methods: list = field(default_factory=list)


COMMANDS = {
    "grid": Command(
        Grid,
        init=Grid,
        methods=["write_supergrid"],
        help="Build a MOM6 horizontal grid (supergrid).",
    ),
    "topo": Command(
        Topo,
        init=_load_topo,
        methods=[
            "set_flat",
            "set_from_dataset",
            "set_bowl",
            "set_spoon",
            "write_topo",
            "write_scrip_grid",
            "write_esmf_mesh",
        ],
        help="Build a MOM6 bathymetry on an existing horizontal grid.",
    ),
    "vgrid": Command(
        VGrid,
        methods=["uniform", "hyperbolic", "write"],
        help="Build a MOM6 vertical grid.",
    ),
}


class _Literal(click.ParamType):
    """A value read as a Python literal where it parses as one, else a string."""

    name = "value"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def _help(func):
    """First paragraph of a docstring."""
    doc = inspect.getdoc(func) or ""
    return doc.split("\n\n")[0].replace("\n", " ")


def _options(func):
    """One click option per parameter of ``func``; required if it has no default."""
    options = []
    for p in inspect.signature(func).parameters.values():
        if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        flag = "--" + p.name.replace("_", "-")
        if isinstance(p.default, bool):
            options.append(
                click.Option(
                    [f"{flag}/--no-{flag[2:]}", p.name],
                    default=p.default,
                    show_default=True,
                )
            )
        else:
            if p.default is p.empty:
                # No default= at all: passing one, even None, satisfies required.
                option = click.Option([flag, p.name], type=_Literal(), required=True)
            else:
                option = click.Option(
                    [flag, p.name],
                    type=_Literal(),
                    default=p.default,
                    show_default=True,
                )
            options.append(option)
    return options


def _is_constructor(cls, name):
    return isinstance(inspect.getattr_static(cls, name), classmethod)


def _method_command(command, name):
    """A chainable subcommand that runs ``name`` on the object built so far."""
    func = getattr(command.cls, name)

    def callback(**kwargs):
        # The group's state dict is shared by every command in the chain. (Click
        # creates all chained contexts up front, so ctx.obj itself can't be
        # reassigned mid-chain.)
        state = click.get_current_context().obj
        if state["obj"] is None and command.init:
            state["obj"] = command.init(**state.pop("init_kwargs"))
        if state["obj"] is None and not _is_constructor(command.cls, name):
            starters = [m for m in command.methods if _is_constructor(command.cls, m)]
            raise click.UsageError(
                f"Start the chain with one of: {', '.join(starters)}"
            )
        target = state["obj"] if state["obj"] is not None else command.cls
        result = getattr(target, name)(**kwargs)
        if isinstance(result, command.cls):
            state["obj"] = result

    return click.Command(
        name, params=_options(func), callback=callback, help=_help(func)
    )


def _class_group(name, command):
    """A chained command group: options build the object, subcommands act on it."""

    def callback(**kwargs):
        # Build the object lazily, in the first chained command, so that
        # `<command> ... <method> --help` works without building it.
        click.get_current_context().obj = {"obj": None, "init_kwargs": kwargs}

    group = click.Group(
        name,
        params=_options(command.init) if command.init else [],
        callback=callback,
        chain=True,
        help=command.help,
    )
    for method in command.methods:
        group.add_command(_method_command(command, method))
    return group


cli = click.Group(
    "mom6_forge", help="Build MOM6 grid, bathymetry, and vertical grid files."
)
for _name, _command in COMMANDS.items():
    cli.add_command(_class_group(_name, _command))
