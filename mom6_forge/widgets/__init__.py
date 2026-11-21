"""mom6_forge's web widgets.

``GridSketcher`` sketches an outline and builds its conformal horizontal grid,
``TopoEditor`` edits bathymetry and ``VGridCreator`` builds vertical grids. Each is a thin
front end over ordinary mom6_forge calls; see ``_base.py`` for how they are built.

Imports are lazy, so ``from mom6_forge.widgets import VGridCreator`` does not load the topo
stack (xesmf and friends). That keeps VGridCreator usable in a browser-only Python.
"""

__all__ = ["GridSketcher", "TopoEditor", "VGridCreator"]


def __getattr__(name):
    if name == "GridSketcher":
        from mom6_forge.widgets.grid_sketcher import GridSketcher

        return GridSketcher
    if name == "TopoEditor":
        from mom6_forge.widgets.topo_editor import TopoEditor

        return TopoEditor
    if name == "VGridCreator":
        from mom6_forge.widgets.vgrid_creator import VGridCreator

        return VGridCreator
    raise AttributeError(f"module 'mom6_forge.widgets' has no attribute {name!r}")
