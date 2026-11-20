"""VGridCreator: build a vertical grid (``VGrid``) in a notebook and keep a library of them.

The front end is ``vgrid_creator.js``. Every rule lives here; the JavaScript only draws the
traits below and sends one action per click or edit (see ``_base.py``).
"""

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import traitlets
import xarray as xr

from mom6_forge.vgrid import VGrid
from mom6_forge.widgets._base import ActionWidget, as_bytes


class VGridCreator(ActionWidget):
    """Build a uniform or hyperbolic vertical grid, or load one from ``VGridLibrary/``.

    Parameters
    ----------
    vgrid : VGrid, optional
        Grid to start from, and to return to on Reset. Default: uniform, 10 levels, 100 m.
    repo_root : str or Path, optional
        Folder that holds ``VGridLibrary/``. Default: the current directory. The library
        folder is created on the first save.
    topo : optional
        Anything with a ``max_depth`` (such as a ``Topo``). The widget warns when the grid
        is shallower than that.

    The grid being built is ``widget.vgrid``.
    """

    _js_file = "vgrid_creator.js"

    dz = traitlets.Bytes(b"").tag(sync=True)
    """Layer thicknesses of ``vgrid`` (float64, top layer first)."""
    name = traitlets.Unicode("").tag(sync=True)
    kind = traitlets.Unicode("uniform").tag(sync=True)
    """Control setting: ``"uniform"`` or ``"hyperbolic"``."""
    nk = traitlets.Int(10).tag(sync=True)
    depth = traitlets.Float(100.0).tag(sync=True)
    ratio = traitlets.Float(10.0).tag(sync=True)
    """Bottom layer thickness / top layer thickness, as in ``VGrid.hyperbolic``."""
    source = traitlets.Unicode("").tag(sync=True)
    """Where ``dz`` came from when it was not built from the controls, else ``""``."""
    warning = traitlets.Unicode("").tag(sync=True)
    library = traitlets.List(traitlets.Dict()).tag(sync=True)
    """Saved grids, newest first: file, name, nk, depth, date, message."""

    def __init__(self, vgrid=None, repo_root=None, topo=None):
        super().__init__()
        self.topo = topo
        root = repo_root if repo_root is not None else os.getcwd()
        self.library_dir = Path(root) / "VGridLibrary"
        if vgrid is None:
            vgrid, source = VGrid.uniform(nk=10, depth=100.0), ""
        else:
            source = "Passed in to VGridCreator"
        self._initial = (np.array(vgrid.dz, dtype=float), vgrid.name, source)
        self._adopt(vgrid, source)
        self.library = self._read_library()

    # ---- actions from the front end

    def _op_params(self, msg, _buffers):
        kind = msg.get("kind")
        nk, depth, ratio = (
            float("nan" if msg.get(key) is None else msg[key])
            for key in ("nk", "depth", "ratio")
        )
        if kind not in ("uniform", "hyperbolic"):
            raise ValueError(f"Unknown grid type {kind!r}; use uniform or hyperbolic.")
        if not (nk.is_integer() and nk >= 2):
            raise ValueError(f"Levels must be a whole number, 2 or more (got {nk:g}).")
        if not depth > 0:
            raise ValueError(f"Depth must be more than 0 m (got {depth:g}).")
        if kind == "hyperbolic" and not ratio > 0:
            raise ValueError(
                f"The bottom/top ratio must be more than 0 (got {ratio:g})."
            )
        if kind == "uniform":
            vgrid = VGrid.uniform(int(nk), depth, name=self.vgrid.name)
        else:
            vgrid = VGrid.hyperbolic(int(nk), depth, ratio, name=self.vgrid.name)
            self.ratio = ratio
        self.kind, self.nk, self.depth = kind, int(nk), depth
        self._show(vgrid, source="")

    def _op_reset(self, _msg, _buffers):
        dz, name, source = self._initial
        self._adopt(VGrid(dz.copy(), name), source)

    def _op_save(self, msg, _buffers):
        name = str(msg.get("name", "")).strip()
        message = str(msg.get("message", "")).strip()
        if not re.fullmatch(r"[\w.-]+", name):
            raise ValueError("Name the grid with letters, digits, '_', '-' or '.'.")
        if not message:
            raise ValueError("Add a message that says what the grid is for.")
        path = self.library_dir / f"vgrid_{name}.nc"
        if path.exists() and not msg.get("overwrite"):
            self.send({"confirm_overwrite": name})
            raise FileExistsError(f"{path.name} is already in {self.library_dir}.")
        vgrid = VGrid(self.vgrid.dz, name)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        vgrid.write_vgrid_file(str(path), message=message)
        self.vgrid, self.name = vgrid, name
        self.library = self._read_library()

    def _op_load(self, msg, _buffers):
        path = self.library_dir / Path(str(msg.get("file", ""))).name
        self._adopt(VGrid.from_vgrid_file(str(path)), f"Loaded from {path.name}")

    # ---- state

    def _adopt(self, vgrid, source):
        """Show a grid that was not built from the controls; set the controls near it."""
        dz = np.asarray(vgrid.dz, dtype=float)
        ratio = dz[-1] / dz[0]
        self.kind = "uniform" if np.isclose(ratio, 1.0, atol=1e-2) else "hyperbolic"
        if self.kind == "hyperbolic":
            self.ratio = round(float(ratio), 3)
        self.nk, self.depth = len(dz), round(float(dz.sum()), 6)
        self._show(vgrid, source)

    def _show(self, vgrid, source):
        self.vgrid = vgrid
        self.dz = as_bytes(vgrid.dz, np.float64)
        self.name = vgrid.name or ""
        self.source = source
        deepest = getattr(self.topo, "max_depth", None)
        if deepest is not None and vgrid.depth < float(deepest) - 1e-6:
            self.warning = (
                f"The grid is {vgrid.depth:.1f} m deep, but the topography reaches "
                f"{float(deepest):.1f} m. Deepen the grid to cover it."
            )
        else:
            self.warning = ""

    def _read_library(self):
        """Saved grids, newest first, read from file metadata (files are closed after)."""
        files = sorted(
            self.library_dir.glob("vgrid_*.nc"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        entries = []
        for path in files:
            saved = datetime.fromtimestamp(path.stat().st_mtime)
            entry = {"file": path.name, "name": path.stem[len("vgrid_") :]}
            entry.update(date=saved.strftime("%Y-%m-%d %H:%M"), nk=None, depth=None)
            try:
                with xr.open_dataset(path) as ds:
                    entry["name"] = str(ds.attrs.get("name", entry["name"]))
                    entry["nk"] = int(ds["dz"].size)
                    entry["depth"] = float(ds["dz"].sum())
                    entry["message"] = str(ds.attrs.get("message", ""))
            except Exception as exc:  # listed with the reason, not hidden
                entry["message"] = f"Could not read this file: {exc}"
            entries.append(entry)
        return entries
