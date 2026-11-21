"""TopoEditor: edit a Topo's bathymetry in a notebook, drawn in grid index space.

The front end (``topo_editor.js``) draws cell (i, j) as one pixel of an nx x ny image,
j up, so picking is exact on rotated, curvilinear and polar grids. Every edit is one
``topo.apply_edits`` call, so one brush stroke or one button press is one history step.

Arrays are row-major (j, i), j = 0 first. After each action Python sends one message:

* ``{"kind": "patch", "n": n}`` with buffers ``[cells int32 (j, i) pairs, depth float32,
  mask uint8]`` holding the new values of the n changed cells, or
* ``{"kind": "reload"}`` when more than a quarter of the cells changed; the ``depth`` and
  ``mask`` traits were updated first, so the page redraws from them.
"""

import numpy as np
import traitlets

from mom6_forge.widgets._base import ActionWidget, as_bytes, from_buffer


class TopoEditor(ActionWidget):
    """Edit a :class:`~mom6_forge.topo.Topo` in place: paint, select, erase basins.

    Parameters
    ----------
    topo : mom6_forge.topo.Topo
        The bathymetry to edit. It is changed in place and stays available as
        ``editor.topo``. Undo, redo, reset, tags and branches need ``Topo(..., git=True)``.
    open_boundaries : bool, optional
        Start with the open-boundary overlay on: the grid edges where ocean meets the
        domain boundary, drawn just outside the grid. The page's "Show open boundaries"
        checkbox turns it on and off.
    """

    _js_file = "topo_editor.js"

    # The grid and the full arrays. ``depth`` is the masked depth (m, land at the topo's
    # land_fillval), ``mask`` the tmask (1 = ocean), ``lon``/``lat`` the cell centres.
    nx = traitlets.Int().tag(sync=True)
    ny = traitlets.Int().tag(sync=True)
    depth = traitlets.Bytes().tag(sync=True)
    mask = traitlets.Bytes().tag(sync=True)
    lon = traitlets.Bytes().tag(sync=True)
    lat = traitlets.Bytes().tag(sync=True)
    basins = traitlets.Bytes(b"").tag(sync=True)
    """int32 basin labels (``topo.basintmask``), kept current only in the basinmask view."""

    view = traitlets.Enum(["depth", "mask", "basinmask"], "depth").tag(sync=True)

    # Small state, refreshed after every action.
    min_depth = traitlets.Float().tag(sync=True)
    max_depth = traitlets.Float().tag(sync=True)
    has_history = traitlets.Bool().tag(sync=True)
    has_stats = traitlets.Bool().tag(sync=True)
    has_user_mask = traitlets.Bool().tag(sync=True)
    can_undo = traitlets.Bool().tag(sync=True)
    can_redo = traitlets.Bool().tag(sync=True)
    branches = traitlets.List(traitlets.Unicode()).tag(sync=True)
    current_branch = traitlets.Unicode("").tag(sync=True)
    tags = traitlets.List(traitlets.Unicode()).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    """What the last action did, shown under the map. Errors go to ``errors`` instead."""

    show_open_boundaries = traitlets.Bool(False).tag(sync=True)
    open_boundary_runs = traitlets.Dict().tag(sync=True)
    """Open-water runs along each side, ``{side: [{"start", "stop", "tiny"}, ...]}`` with
    cell indices along the side (stop exclusive), while the overlay is on; else ``{}``."""

    def __init__(self, topo, open_boundaries=False):
        grid = topo._grid
        super().__init__(
            nx=int(grid.nx),
            ny=int(grid.ny),
            lon=as_bytes(grid.tlon, "<f4"),
            lat=as_bytes(grid.tlat, "<f4"),
            has_history=topo.has_version_control,
            show_open_boundaries=bool(open_boundaries),
        )
        self.topo = topo
        self._send_full()

    def _handle(self, widget, msg, buffers):
        self.status = ""  # the status line describes the last action only
        super()._handle(widget, msg, buffers)

    @property
    def open_boundaries(self):
        """Sides with open water in the current mask, in S/E/N/W order, as
        ``configure_forcings(boundaries=...)`` takes them."""
        return [side for side, side_runs in _open_runs(self.topo).items() if side_runs]

    # ------------------------------------------------------------------ actions

    def _op_sync(self, msg, buffers):
        """A view was just drawn from the traits; patches may have made them stale."""
        self._send_full()

    def _op_view(self, msg, buffers):
        with self.hold_sync():  # one update, so the page never draws basins it lacks
            self.view = msg["view"]
            self._update_basins()

    def _op_open_boundaries(self, msg, buffers):
        self.show_open_boundaries = bool(msg["show"])
        self._refresh()

    def _op_edit(self, msg, buffers):
        """One ``topo.apply_edits`` entry. Cells come as a brush stroke's (j, i) buffer
        or as a selected box ``rect = [j0, i0, j1, i1]`` (inclusive)."""
        edit = dict(msg["edit"])
        rect = edit.pop("rect", None)
        if buffers:
            edit["cells"] = from_buffer(buffers[0], "<i4").reshape(-1, 2)
        elif rect is not None:
            j0, i0, j1, i1 = (int(v) for v in rect)
            jj, ii = np.mgrid[j0 : j1 + 1, i0 : i1 + 1]
            edit["cells"] = np.column_stack([jj.ravel(), ii.ravel()])
        self._publish(self.topo.apply_edits([edit]), edit["op"].replace("_", " "))

    def _op_undo(self, msg, buffers):
        self._publish(self.topo.undo(), "undo")

    def _op_redo(self, msg, buffers):
        self._publish(self.topo.redo(), "redo")

    def _op_reset(self, msg, buffers):
        self._publish(self.topo.reset(), "reset")

    def _op_tag(self, msg, buffers):
        name = _name(msg)
        self.topo.tcm.tag(name)
        self._refresh()
        self.status = f"Tagged the current bathymetry as '{name}'."

    def _op_create_branch(self, msg, buffers):
        name = _name(msg)
        self.topo.tcm.create_branch(name)
        self._refresh()
        self.status = f"Created branch '{name}'. Check it out to edit on it."

    def _op_checkout(self, msg, buffers):
        name = _name(msg)
        self.topo.tcm.checkout(name)
        self._send_full()
        self.status = f"Checked out branch '{name}'."

    # ------------------------------------------------------------------ state

    def _publish(self, changed, label):
        """Show the result of an edit: a patch of the changed cells, or full arrays."""
        changed = np.asarray(changed, dtype="<i4").reshape(-1, 2)
        n = len(changed)
        if n > self.nx * self.ny // 4:
            self._send_full()
        else:
            j, i = changed.T
            buffers = [
                as_bytes(changed, "<i4"),
                as_bytes(self.topo.masked_depth.data[j, i], "<f4"),
                as_bytes(self.topo.tmask.data[j, i], "u1"),
            ]
            with self.hold_sync():
                self._update_basins()
                self._refresh()
            self.send({"kind": "patch", "n": n}, buffers=buffers if n else None)
        cells = "no cells" if n == 0 else "1 cell" if n == 1 else f"{n:,} cells"
        self.status = f"{label.capitalize()}: {cells} changed."

    def _send_full(self):
        with self.hold_sync():
            self.depth = as_bytes(self.topo.masked_depth, "<f4")
            self.mask = as_bytes(self.topo.tmask, "u1")
            self._update_basins()
            self._refresh()
        self.send({"kind": "reload"})

    def _update_basins(self):
        if self.view == "basinmask":
            self.basins = as_bytes(self.topo.basintmask, "<i4")

    def _refresh(self):
        topo = self.topo
        self.min_depth = float(topo.min_depth)
        self.max_depth = float(np.nan_to_num(topo.max_depth))
        self.has_stats = topo.stats is not None
        self.has_user_mask = topo.user_mask is not None
        # Every action that can change the mask ends here, so the overlay stays current.
        self.open_boundary_runs = _open_runs(topo) if self.show_open_boundaries else {}
        if topo.has_version_control:
            self.can_undo = topo.can_undo
            self.can_redo = topo.can_redo
            self.branches = topo.tcm.list_branches()
            self.current_branch = topo.tcm.get_current_branch()
            self.tags = topo.tcm.get_tag_names()


def _open_runs(topo):
    """The ``open_boundary_runs`` trait: each side's open-water runs in ``topo``'s mask."""
    # Imported here: the editor itself runs without the grid-sketching stack (shapely).
    from mom6_forge import corner_diagnostics as diag

    runs = diag.open_boundary_runs(topo._grid, ocean=topo.tmask > 0)
    keep = ("start", "stop", "tiny")
    return {
        s: [{k: r[k] for k in keep} for r in runs[s]["runs"]] for s in diag.OBC_SIDES
    }


def _name(msg):
    name = str(msg.get("name") or "").strip()
    if not name:
        raise ValueError("Enter a name first.")
    return name
