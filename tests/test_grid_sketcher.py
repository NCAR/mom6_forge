"""GridSketcher's Python side: every action sent as the page would (``widget._act``).

The page's own behaviour is tested in ``test_widgets_js.py``. Behaviour kept from the
PR #149 widget's tests: the recommended projection, corner numbering and flat corners,
the errors that block Build, undo/redo, Build/Save/reopen and the pole outline.
"""

import json

import numpy as np
import pytest

from mom6_forge.grid import Grid
from mom6_forge.widgets import GridSketcher

BOX = dict(lon=[-10, 10, 10, -10], lat=[-5, -5, 5, 5], corners=[0, 1, 2, 3])


def _pairs(lon, lat):
    """An outline as the page sends it: one float64 buffer of (lon, lat) pairs."""
    return [np.column_stack([lon, lat]).astype("<f8").tobytes()]


def test_the_default_box_previews_on_the_recommended_projection():
    s = GridSketcher()
    # A lon/lat box is a rectangle on Mercator, whose cells vary least
    assert s.kind == s.recommended == "merc" and "+proj=merc" in s.proj4
    assert s.outline.lon == [235, 243, 243, 235] and s.corners == [0, 1, 2, 3]
    cg = s.preview
    assert 5_000 < cg.nx * cg.ny < 8_000 and s.cells == f"≈ {cg.nx} × {cg.ny} cells"
    rows, cols = s.node_shape
    assert len(s.nodes) == rows * cols * 16 and len(s.flags) == (rows - 1) * (cols - 1)
    assert max(rows, cols) <= 61 and s.can_build and not s.preview_errors
    assert "MOM6 CFL" in dict(s.details)["Time step"]
    kinds = [f["properties"]["kind"] for f in json.loads(s.land)["features"]]
    assert kinds == ["land", "graticule"]
    # Land follows the view: the page reports a view far away
    view = _pairs([0, 10, 10, 0], [40, 40, 50, 50])
    s._act("view", buffers=view)
    lon0, lon1, lat0, lat1 = s._land_window
    assert lon0 <= 0 and 10 <= lon1 and lat0 <= 40 and 50 <= lat1


def test_corners_are_numbered_as_the_grid_and_a_flat_one_is_flagged():
    s = GridSketcher(
        outline=dict(lon=[-10, -10, 10, 10], lat=[-5, 5, 5, -5], corners=[0, 1, 2, 3]),
        resolution_km=110,
    )
    o, q = s.outline, s.preview
    first, third = s.corners[0], s.corners[2]
    assert np.allclose([o.lon[first], o.lat[first]], [q.qlon[0, 0], q.qlat[0, 0]])
    assert np.allclose([o.lon[third], o.lat[third]], [q.qlon[-1, -1], q.qlat[-1, -1]])
    assert s.flat_corners == []
    flat = dict(
        lon=[-10, 0, 10, 10, -10], lat=[-5, -4.9, -5, 5, 5], corners=[0, 1, 3, 4]
    )
    s = GridSketcher(outline=flat, resolution_km=110)
    assert s.flat_corners == [1] and any("flat" in w for w in s.preview_warnings)


def test_missing_corners_crossing_edges_and_too_many_cells_block_build():
    five = dict(lon=[-10, 0, 10, 10, -10], lat=[-5, -6, -5, 5, 5], corners=[0, 2, 3, 4])
    s = GridSketcher(outline=five, resolution_km=110)
    s._act("corner", i=1)  # a 5th: refused, nothing changes
    assert "4 corners already" in s.errors[0] and s.can_build
    s._act("corner", i=4)
    assert s.preview_errors == [
        "Only 3 of 4 corners: double-click a vertex to add one."
    ]
    assert not s.can_build and s.preview is None
    s._act("corner", i=4)
    s._act("outline", buffers=_pairs([-10, 0, 10, -10, 10], [-5, -6, -5, 5, 5]))
    assert "cross" in s.preview_errors[0] and not s.can_build
    s._act("undo")
    s._act("resolution", km=0.1)
    assert "million cells" in s.preview_errors[0] and not s.can_build
    s._act("resolution", km=110)
    assert s.can_build


def test_an_edge_insert_keeps_the_corners_on_their_vertices_and_undoes_in_one_step():
    s = GridSketcher(outline=BOX, resolution_km=110)
    # The page sends the whole outline; the round trip through its projection is inexact
    lon, lat = [-10 + 1e-10, 0, 10, 10, -10], [-5, -6, -5, 5, 5]
    s._act("outline", buffers=_pairs(lon, lat))
    assert s.outline.lon == [-10, 0, 10, 10, -10] and s.outline.corners == [0, 2, 3, 4]
    assert s.preview is not None and s.can_undo and not s.can_redo
    s._act("undo")
    assert s.outline.to_dict() == BOX and s.can_redo
    s._act("redo")
    assert s.outline.corners == [0, 2, 3, 4]
    s._act("delete", i=2)  # a corner vertex: its corner goes with it
    assert s.outline.lon == [-10, 0, 10, -10] and s.outline.corners == [0, 2, 3]


def test_box_and_drawn_outlines_replace_the_old_one():
    s = GridSketcher(outline=BOX, resolution_km=110)
    s._act("clear")
    assert s.outline.n == 0 and not s.can_build and not s.preview_errors
    assert "click the first one" in s.status
    s._act("draw", buffers=_pairs([0, 20, 20, 0], [10, 10, 20, 20]))  # a Box
    assert s.outline.corners == [0, 1, 2, 3] and s.can_build
    s._act("draw", buffers=_pairs([0, 20, 25, 20, 0], [10, 10, 15, 20, 20]))
    assert s.outline.corners == [] and "Only 0 of 4" in s.preview_errors[0]
    s._act("undo")
    assert s.outline.lon == [0, 20, 20, 0]


def test_build_save_and_reopen_keep_the_outline(tmp_path):
    s = GridSketcher(outline=BOX, resolution_km=110, working_dir=tmp_path)
    s._act("save", name="box")
    assert "Build the grid" in s.errors[0]
    s._act("build", name="box")
    assert s.grid is not None and s.can_save and "Built" in s.status
    assert (s.grid.nx, s.grid.ny) == (s.preview.nx, s.preview.ny)
    s._act("resolution", km=100)  # any change drops the built grid
    assert s.grid is None and not s.can_save
    s._act("build", name="box")
    s._act("save", name="box")
    path = tmp_path / "GridLibrary" / "grid_box.nc"
    assert path.exists() and s.status == f"Saved {path}."
    s._act("save", name="box")
    assert "Overwrite" in s.errors[0]
    s._act("save", name="box", overwrite=True)
    assert not s.errors
    g = Grid.from_supergrid_file(str(path))
    assert g.outline["lon"] == BOX["lon"] and g.outline["resolution_km"] == 100
    r = GridSketcher(grid=g)
    assert r.outline.to_dict() == BOX and r.resolution_km == 100
    assert r.name == "box" and r.kind == s.kind


def test_a_ring_round_the_pole_is_polar_stereographic_and_builds():
    ring = dict(lon=[0, 90, 180, 270], lat=[75] * 4, corners=[0, 1, 2, 3])
    s = GridSketcher(outline=ring, resolution_km=150)
    assert s.kind == s.recommended == "stere" and "+lat_0=90" in s.proj4
    s._act("projection", kind="merc")  # only Polar can take a pole
    assert "Polar stereographic" in s.errors[0] and s.kind == "stere"
    s._act("build", name="arctic")
    assert s.grid is not None and not s.errors
