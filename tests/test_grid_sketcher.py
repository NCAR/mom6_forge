import random
import re
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import LocationEvent
from matplotlib.colors import to_hex
from matplotlib.ticker import FuncFormatter

import mom6_forge._conformal as cf
from mom6_forge import corner_diagnostics as diag
from mom6_forge import grid_sketcher as gs
from mom6_forge.grid import Grid
from mom6_forge.grid_sketcher import GridSketcher, Outline

PINK, RED = "#cc79a7", diag.STATUS_COLORS[2]


def _event(
    s, px=None, py=None, button=1, dblclick=False, modifiers=(), inside=True, step=0
):
    """A mouse event at display pixel (px, py), as matplotlib would build it."""
    xy = s.ax.transData.inverted().transform((px, py)) if inside else [None] * 2
    kw = dict(button=button, dblclick=dblclick, modifiers=modifiers, step=step)
    ax = s.ax if inside else None
    return types.SimpleNamespace(x=px, y=py, xdata=xy[0], ydata=xy[1], inaxes=ax, **kw)


def _new(*args, **kw):
    """A sketch drawn once, so its pixel transforms are final as in a browser."""
    s = GridSketcher(*args, **kw)
    s.fig.draw_without_rendering()
    return s


def _px(s, i):
    vx, vy = s._vertex_pixels()
    return vx[i], vy[i]


def _px_of(s, lon, lat):
    return tuple(s.ax.transData.transform(s.globe.to_xy(lon, lat)))


def _click(s, kind):
    """Click a projection button, as a browser does: flip its value."""
    s.projection_buttons[kind].value = not s.projection_buttons[kind].value


def _pressed(s):
    """The projection kinds whose buttons are down."""
    return [k for k, b in s.projection_buttons.items() if b.value]


def _green(s):
    """The projection kinds whose buttons are green (recommended)."""
    return [k for k, b in s.projection_buttons.items() if b.button_style == "success"]


def _drag(s, start, *points, **kw):
    s._on_press(_event(s, *start, **kw))
    for p in points:
        s._on_motion(_event(s, *p))
    s._on_release(_event(s, *(points[-1] if points else start)))


def _fire(s, name):
    """Run a pending timer's callback now (Agg timers never fire on their own)."""
    func, args, kwargs = s._timers[name].callbacks[0]
    func(*args, **kwargs)


def _middle(s):
    """The middle of the map, inside the test box and away from its edges."""
    return s.ax.bbox.x0 + s.ax.bbox.width / 2, s.ax.bbox.y0 + s.ax.bbox.height / 2


def _boom(*args, **kwargs):
    raise ValueError("boom")


def _count(monkeypatch, module, name):
    calls, real = [], getattr(module, name)
    monkeypatch.setattr(module, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def get_sketch():
    return _new(Outline.from_bbox(-10, 10, -5, 5), resolution_km=110)


# --- construction ---


def test_the_default_box_previews_about_ten_thousand_cells_with_a_mom6_line():
    s = _new()
    assert s.outline.lon == [235, 243, 243, 235] and s.outline.corners == [0, 1, 2, 3]
    assert 8_000 < s.preview.nx * s.preview.ny < 12_000
    assert f"{s.preview.nx} × {s.preview.ny}</b> cells" in s.summary.value
    assert "cells" not in s.status.value
    assert f"{s.preview.nx} × {s.preview.ny}" in s.cells_actual.value
    assert "nominal" in s.summary.value and "orthogonality" in s.details_html.value
    assert "smallest ocean cell" in s.details_html.value
    assert "MOM6 CFL" in s.details_html.value
    assert "CICE" not in s.details_html.value + s.summary.value
    assert not s.build_button.disabled and s.grid is None
    assert _pressed(s) == [s.projection.kind] == ["merc"]


def test_the_panel_stretches_its_rows_and_never_scrolls_sideways(get_sketch):
    def walk(w):
        return [w] + [d for c in getattr(w, "children", ()) for d in walk(c)]

    # "100%" plus the 2 px widget margins is wider than the panel
    panel = get_sketch.control_panel
    assert "100%" not in [w.layout.width for w in walk(panel)]
    assert panel.layout.overflow == "hidden auto"


def test_an_outline_without_four_corners_explains_and_greys_out_build():
    s = _new(Outline([0, 1, 2, 3], [0, 0, 1, 1], [0, 1]))
    assert s.preview is None and "Only 2 of 4 corners" in s.messages.value
    assert "Warnings" not in s.messages.value and "Fix the errors" in s.status.value
    assert s.build_button.disabled and s.save_button.disabled
    assert len(s.grid_lines.get_segments()) == 0 and not s.cbar_ax.get_visible()


def test_a_grid_outline_reopens_exactly_and_a_plain_grid_is_traced():
    g = Grid(resolution=0.5, xstart=235.0, lenx=8.0, ystart=31.0, leny=8.0, name="box")
    s = _new(g, per_side=4)
    assert s.outline.n == 16 and s.outline.corners == [0, 4, 8, 12]
    assert min(s.outline.lon) >= 235 and s.name_box.value == "box"
    assert s.resolution_km == pytest.approx(cf.nominal_resolution_km(g), rel=0.05)
    merc = dict(kind="merc", lon_0=0.0, lat_0=0.0, lat_1=None, lat_2=None)
    lon, lat = [-10, 10, 10, -10], [-5, -5, 5, 5]
    g.outline = dict(lon=lon, lat=lat, corners=[1, 2, 3, 0], projection=merc)
    g.outline.update(resolution_km=150.0, open_boundaries=["south"])
    s = _new(g)
    assert s.outline.lon == lon and s.outline.corners == [1, 2, 3, 0]
    assert s.resolution_km == 150
    assert _pressed(s) == [s.projection.kind] == ["merc"]


# --- editing ---


@pytest.mark.parametrize("dpi", [100, 200])
def test_a_press_anywhere_on_a_drawn_marker_grabs_its_vertex(get_sketch, dpi):
    s, css, pt = get_sketch, dpi / 100, dpi / 72
    s.fig.set_dpi(dpi)
    s.fig.draw_without_rendering()
    # Corner 1's disc is 13 pt across (with its edge); 1 pt past its rim still grabs it
    x0, y0 = _px(s, 0)
    assert s._pick(x0 + 7.4 * pt, y0) == (0, None)
    assert s._pick(x0 + 12 * css, y0) == (None, 0)
    # A nearly flat corner's red ring (19 pt across) grabs it too
    flat = Outline([-10, 0, 10, 10, -10], [-5, -4.9, -5, 5, 5], [0, 1, 3, 4])
    f = _new(flat, resolution_km=110)
    f.fig.set_dpi(dpi)
    f.fig.draw_without_rendering()
    x1, y1 = _px(f, 1)
    assert f._pick(x1 + 10.3 * pt, y1) == (1, None)


def test_a_drag_sketches_paced_frames_then_previews_once(get_sketch, monkeypatch):
    s = get_sketch
    solves = _count(monkeypatch, cf, "solve")
    draws = _count(monkeypatch, s.fig.canvas, "draw_idle")
    monkeypatch.setattr(gs, "_FRAME_S", 5.0)
    _drag(s, _px(s, 2))
    assert not solves and s.undo_button.disabled and not draws
    x0, y0 = _px(s, 0)
    s._on_press(_event(s, x0, y0))
    for k in range(1, 4):
        s._on_motion(_event(s, x0 + 5 * k, y0 + 3 * k))
    # The first motion draws at once; the catch-up timer draws the latest of the rest
    assert len(s.frame_times) == len(draws) == 1 and "catchup" in s._timers
    s.frame_times = list(range(250))
    _fire(s, "catchup")
    assert len(s.frame_times) == 200 and len(draws) == 2 and s.timings["frame"] > 0
    assert np.allclose(np.subtract(_px(s, 0), (x0, y0)), (15, 9), atol=0.5)
    assert not solves and to_hex(s.grid_lines.get_color()[0]) == "#999999"
    assert not s.shade_mesh.get_visible() and not s.flag_lines.get_visible()
    s._on_release(_event(s, x0 + 15, y0 + 9))
    # The preview, and a quick coarse one per projection for the recommendation
    assert len(solves) == 1 + len(cf.KINDS) and s.shade_mesh.get_visible()
    assert np.allclose(np.subtract(_px(s, 0), (x0, y0)), (15, 9), atol=0.5)
    assert s.outline.undo() and not s.outline.undo()


def test_pressing_an_edge_inserts_a_vertex_and_its_drag_is_one_undo_step(get_sketch):
    s = get_sketch
    (ax, ay), (bx, by) = _px(s, 0), _px(s, 1)
    _drag(s, ((ax + bx) / 2, ay), ((ax + bx) / 2, ay - 20))
    assert s.outline.n == 5 and s.outline.corners == [0, 2, 3, 4]
    assert s.outline.lat[1] < -5 and s.preview is not None
    assert s.outline.undo() and s.outline.n == 4 and not s.outline.undo()


@pytest.mark.parametrize("button, modifiers", [(3, ()), (1, ("ctrl",))])
def test_right_or_ctrl_click_deletes_a_vertex(get_sketch, button, modifiers):
    s = get_sketch
    s._on_press(_event(s, *_px(s, 0), button=button, modifiers=modifiers))
    assert s.outline.n == 3 and s.outline.corners == [0, 1, 2]
    assert s.preview is None and "3 of 4 corners" in s.messages.value
    assert "add a 4th vertex" in s.messages.value


def test_a_browser_double_click_toggles_a_corner_exactly_once(get_sketch):
    s = get_sketch
    p = _px(s, 1)
    # The browser sends press/release twice, then one press flagged dblclick
    _drag(s, p)
    _drag(s, p)
    s._on_press(_event(s, *p, dblclick=True))
    assert s.outline.corners == [0, 2, 3] and "no longer a corner" in s.status.value
    _drag(s, p)
    _drag(s, p, dblclick=True)
    assert s.outline.corners == [0, 2, 3, 1] and s.preview is not None
    s.outline.insert(0, 0.0, -5.0)
    s._update_outline_artists()
    s._on_press(_event(s, *_px(s, 1), dblclick=True))
    assert "Four corners already" in s.status.value and s.outline.n == 5


def test_corner_labels_follow_the_grid_and_mark_a_nearly_flat_corner():
    cw = Outline([-10, -10, 10, 10], [-5, 5, 5, -5], [0, 1, 2, 3])
    s = _new(cw, resolution_km=110)
    label, q = {t.get_text(): t.get_position() for t in s.corner_labels}, s._shown
    assert np.allclose(label["1"], (q.x[0, 0], q.y[0, 0]))
    assert np.allclose(label["3"], (q.x[-1, -1], q.y[-1, -1]))
    assert len(s.corner_ring.get_xdata()) == 0
    flat = Outline([-10, 0, 10, 10, -10], [-5, -4.9, -5, 5, 5], [0, 1, 3, 4])
    assert len(_new(flat, resolution_km=110).corner_ring.get_xdata()) == 1


def test_undo_and_redo_buttons_step_through_the_history(get_sketch):
    s = get_sketch
    assert s.undo_button.disabled and s.redo_button.disabled
    _drag(s, _px(s, 2), np.add(_px(s, 2), 10))
    moved = s.outline.lon[2]
    s.undo_button.click()
    assert s.outline.lon[2] == 10 and not s.redo_button.disabled
    s.redo_button.click()
    assert s.outline.lon[2] == moved and s.redo_button.disabled


def test_a_press_ends_a_drag_whose_release_was_lost(get_sketch):
    s = get_sketch
    x0, y0 = _px(s, 2)
    s._on_press(_event(s, x0, y0))
    s._on_motion(_event(s, x0 + 10, y0 + 10))
    # The mouse went up off the map, so no release came
    s._on_press(_event(s, x0 + 12, y0 + 12))
    assert s.shade_mesh.get_visible() and np.allclose(_px(s, 2), (x0 + 12, y0 + 12))
    s.undo_button.click()
    assert (s.outline.lon[2], s.outline.lat[2]) == (10, 5) and s.undo_button.disabled


def test_edit_coordinates_replaces_the_outline_and_rejects_bad_text(get_sketch):
    s = get_sketch
    s.vertex_text.value = "-8, -4*\n8, -4*\n8, 4*\n0, 5\n-8, 4*"
    s.apply_button.click()
    assert s.outline.n == 5 and s.outline.corners == [0, 1, 2, 4]
    assert s.preview is not None and "0.0000, 5.0000\n" in s.vertex_text.value
    s.vertex_text.value = "1, 2\nnot a number"
    s.apply_button.click()
    assert "Need 4" in s.status.value and s.outline.n == 5


def test_a_self_crossing_drag_clears_the_sketch_and_says_why(get_sketch):
    s = get_sketch
    x0, y0 = _px(s, 0)
    s._on_press(_event(s, x0, y0))
    s._on_motion(_event(s, *_px_of(s, -5, 8)))
    assert len(s.grid_lines.get_segments()) == 0
    s._on_release(_event(s, *_px_of(s, -5, 8)))
    assert s.preview is None and "cross" in s.messages.value


# --- drawing ---


def test_a_blank_sketch_draws_closes_on_point_1_clears_and_undoes():
    s = _new(blank=True, resolution_km=100)
    rows = [getattr(w, "children", ()) for w in s.control_panel.children]
    edits = [b.description for r in rows if s.undo_button in r for b in r]
    assert edits == ["Undo", "Redo", "Clear all"]
    assert s.build_button.disabled and s.messages.value == ""
    points = [(236, 32), (242, 32), (242, 38), (236, 38), (236, 32)]
    for lon, lat in points[:4]:
        _drag(s, _px_of(s, lon, lat))
    assert s.outline.n == 4 and s.preview is None and "click point 1" in s.status.value
    _drag(s, _px_of(s, *points[4]))
    assert s.outline.corners == [0, 1, 2, 3] and not s.build_button.disabled
    assert _pressed(s) == _green(s) == ["merc"]  # the recommended kind, once closed
    s.clear_button.click()
    assert s.outline.n == 0 and s.build_button.disabled and s.messages.value == ""
    s.undo_button.click()
    assert s.outline.n == 4 and s.preview is not None
    s.redo_button.click()
    assert s.outline.n == 0 and "0 points" in s.status.value


def test_point_1_is_marked_and_a_double_click_on_it_closes_once():
    s = _new(blank=True, resolution_km=100)
    points = [(236, 32), (242, 32), (242, 38), (236, 38)]
    _drag(s, _px_of(s, *points[0]))
    assert s.corner_labels[0].get_text() == "1" and "1 point:" in s.status.value
    _drag(s, _px_of(s, *points[1]))
    _drag(s, _px_of(s, *points[0]))  # too soon to close: adds none
    assert s.outline.n == 2 and s._drawing
    for lon, lat in points[2:]:
        _drag(s, _px_of(s, lon, lat))
    # The browser's double-click: press/release twice, then one press flagged dblclick
    p = _px(s, 0)
    _drag(s, p)
    _drag(s, p)
    s._on_press(_event(s, *p, dblclick=True))
    assert s.outline.corners == [0, 1, 2, 3] and s.preview is not None
    s.undo_button.click()
    assert s._drawing and s.outline.n == 4 and s.outline.corners == []
    s.redo_button.click()
    assert not s._drawing and s.outline.corners == [0, 1, 2, 3]


def test_clicking_the_first_point_closes_and_three_points_ask_for_a_fourth():
    s = _new(blank=True, resolution_km=100)
    for lon, lat in [(236, 32), (242, 32), (239, 38), (236, 32)]:
        _drag(s, _px_of(s, lon, lat))
    assert s.outline.n == 3 and not s._drawing and s.build_button.disabled
    assert "Only 0 of 4 corners: click an edge to add a 4th vertex" in s.messages.value


def test_while_drawing_a_click_adds_a_point_and_a_drag_pans_or_moves_one(monkeypatch):
    s = _new(blank=True, resolution_km=100)
    solves = _count(monkeypatch, cf, "solve")
    # A click that strays under 5 px adds its point where it was pressed
    p = _px_of(s, 236, 32)
    _drag(s, p, np.add(p, (3, -3)))
    assert s.outline.n == 1 and np.allclose(_px(s, 0), p)
    view = s.ax.get_xlim(), s.ax.get_ylim()
    _drag(s, _middle(s), np.add(_middle(s), (-30, 10)))
    assert s.outline.n == 1 and view != (s.ax.get_xlim(), s.ax.get_ylim())
    q = _px(s, 0)
    _drag(s, q, np.add(q, (20, 0)))
    assert s.outline.n == 1 and np.allclose(_px(s, 0), np.add(q, (20, 0)))
    assert not solves and s.outline.undo() and np.allclose(_px(s, 0), q)


def test_while_drawing_a_pan_turns_the_globe_to_face_the_middle_of_the_view():
    s = _new(blank=True, resolution_km=100)
    end = np.add(_middle(s), (-150, 40))
    s._on_press(_event(s, *_middle(s)))
    s._on_motion(_event(s, *end))
    middle, size = _view_centre(s), (np.ptp(s.ax.get_xlim()), np.ptp(s.ax.get_ylim()))
    s._on_release(_event(s, *end))
    assert np.allclose(s.globe.centre, middle) and np.allclose(_view_centre(s), middle)
    assert np.allclose((np.ptp(s.ax.get_xlim()), np.ptp(s.ax.get_ylim())), size)
    assert s.outline.n == 0 and len(s.land_collection.get_paths())


@pytest.mark.parametrize("button, modifiers", [(3, ()), (1, ("ctrl",))])
def test_right_or_ctrl_click_deletes_a_point_while_drawing(button, modifiers):
    s = _new(blank=True, resolution_km=100)
    for lon, lat in [(236, 32), (242, 32), (242, 38)]:
        _drag(s, _px_of(s, lon, lat))
    # Point 1 too: the next point becomes point 1, and undo brings it back
    s._on_press(_event(s, *_px(s, 0), button=button, modifiers=modifiers))
    assert s._drawing and np.mod(s.outline.lon, 360) == pytest.approx([242, 242])
    assert "2 points" in s.status.value
    assert np.allclose(s.corner_labels[0].get_position(), s.globe.to_xy(242, 32))
    s.undo_button.click()
    assert s._drawing and np.mod(s.outline.lon, 360) == pytest.approx([236, 242, 242])


# --- pan and zoom ---


def test_scroll_zooms_about_the_cursor_within_limits(get_sketch, monkeypatch):
    s = get_sketch
    draws = _count(monkeypatch, s.fig.canvas, "draw_idle")
    monkeypatch.setattr(gs, "_COALESCE_MS", 60_000)
    (x0, x1), (y0, y1) = s.ax.get_xlim(), s.ax.get_ylim()
    p = _px_of(s, 5, 2)
    under = s.ax.transData.inverted().transform(p)
    for _ in range(3):
        s._on_scroll(_event(s, *p, step=1))
    assert np.allclose(s.ax.transData.inverted().transform(p), under)
    assert np.isclose(np.ptp(s.ax.get_xlim()), (x1 - x0) / 1.25**3)
    # A burst asks for one frame now and one when the coalescing ends
    assert len(draws) == 1 and "redraw" in s._timers
    views = []
    for step in [-1] * 40 + [1] * 60:
        s._on_scroll(_event(s, *p, step=step))
        width = np.ptp(s.ax.get_xlim()) / (x1 - x0)
        views.append(min(np.ptp(s.ax.get_xlim()), np.ptp(s.ax.get_ylim())))
        assert width >= 0.02 - 1e-9
    # Out to the whole globe (its narrower side 2.1 radii), in to 1/50 of the fit
    assert np.isclose(max(views), 2.1 * s.globe.radius) and np.isclose(width, 0.02)
    s._on_scroll(_event(s, inside=False, step=-1))
    s.reset_view_button.click()
    assert np.allclose([s.ax.get_xlim(), s.ax.get_ylim()], [(x0, x1), (y0, y1)])


def test_dragging_empty_map_pans_and_a_middle_drag_pans_over_a_vertex(get_sketch):
    s = get_sketch
    (x0, x1), (y0, y1), before = s.ax.get_xlim(), s.ax.get_ylim(), s.outline.to_dict()
    kx, ky = (x1 - x0) / s.ax.bbox.width, (y1 - y0) / s.ax.bbox.height
    _drag(s, _middle(s), np.add(_middle(s), (-30, 10)))
    assert np.allclose(s.ax.get_xlim(), np.add((x0, x1), 30 * kx))
    assert np.allclose(s.ax.get_ylim(), np.subtract((y0, y1), 10 * ky))
    _drag(s, _px(s, 0), np.add(_px(s, 0), (-15, 0)), button=2)
    assert np.allclose(s.ax.get_xlim(), np.add((x0, x1), 45 * kx))
    assert s.outline.to_dict() == before and s.undo_button.disabled


def test_off_the_globe_a_press_pans_and_a_scroll_zooms(get_sketch):
    s = get_sketch
    for _ in range(40):  # all the way out, off-centre: then the whole disc, centred
        s._on_scroll(_event(s, *_px_of(s, 5, 2), step=-1))
    (x0, x1), (y0, y1), r = s.ax.get_xlim(), s.ax.get_ylim(), s.globe.radius
    assert x0 < -r and r < x1 and y0 < -r and r < y1
    # Matplotlib gives no inaxes in space round the disc
    corner = (s.ax.bbox.x0 + 3, s.ax.bbox.y0 + 3)
    _drag(s, corner, np.add(corner, (40, 0)), inside=False)
    assert s.ax.get_xlim()[0] < x0
    s._on_scroll(_event(s, *corner, inside=False, step=1))
    assert np.ptp(s.ax.get_xlim()) < x1 - x0


def test_a_moving_view_draws_lite_frames_until_detail_returns(get_sketch):
    s = get_sketch
    full = len(s.grid_lines.get_segments())
    s._on_scroll(_event(s, *_middle(s), step=1))
    assert s._lite_on and not s.shade_mesh.get_visible() and not s.cbar_ax.get_visible()
    assert not s.flag_lines.get_visible() and not any(
        s.land_collection.get_antialiased()
    )
    assert 0 < len(s.grid_lines.get_segments()) < full
    # A drag begun before the restore delays it until the drag ends
    s._on_press(_event(s, *_px(s, 0)))
    _fire(s, "lite")
    assert s._lite_on
    s._on_release(_event(s, *_px(s, 0)))
    _fire(s, "lite")
    assert not s._lite_on and s.shade_mesh.get_visible() and s.cbar_ax.get_visible()
    assert all(s.land_collection.get_antialiased()) and s.flag_lines.get_visible()
    assert len(s.grid_lines.get_segments()) >= full


def test_panning_far_away_brings_land_into_the_new_view(get_sketch):
    s = get_sketch
    # The Balearic Sea to the middle: north of the land first loaded, to 30 N
    _drag(s, _px_of(s, 5, 40), _middle(s), button=2)
    assert s._land_window[3] > 40
    v = np.concatenate([p.vertices for p in s.land_collection.get_paths()])
    (x0, x1), (y0, y1) = s.ax.get_xlim(), s.ax.get_ylim()
    assert np.any((v[:, 0] > x0) & (v[:, 0] < x1) & (v[:, 1] > y0) & (v[:, 1] < y1))


# --- projection and resolution ---


def _view_centre(s):
    return s.globe.to_lonlat(np.mean(s.ax.get_xlim()), np.mean(s.ax.get_ylim()))


def test_a_projection_change_keeps_the_zoomed_view(get_sketch):
    s = get_sketch
    s._on_scroll(_event(s, *_px_of(s, 5, 2), step=1))
    _drag(s, _middle(s), np.add(_middle(s), (-60, 20)))
    centre = _view_centre(s)
    _click(s, "tmerc")
    assert s.projection.kind == "tmerc" and s.preview is not None
    assert np.allclose(_view_centre(s), centre, atol=0.5)


def test_a_dateline_outline_fits_its_own_box_and_keeps_the_view():
    s = _new(Outline.from_bbox(170, -170, 50, 60), resolution_km=150)
    x, _ = s.globe.to_xy(s.outline.lon, s.outline.lat)
    assert np.ptp(s.ax.get_xlim()) < 1.5 * np.ptp(x) and s.preview is not None
    _click(s, "tmerc")
    assert abs(abs(_view_centre(s)[0]) - 180) < 2 and s.preview is not None
    _click(s, "merc")
    assert s.projection.kind == "merc" and s.preview is not None


def test_the_buttons_show_the_recommendation_and_switch_the_projection(monkeypatch):
    assert [kind for _, kind in gs._PROJECTIONS] == list(cf.KINDS)
    s = _new()  # a lon/lat box: a rectangle on Mercator, whose cells vary least
    assert _pressed(s) == _green(s) == [s.projection.kind] == ["merc"]
    assert s.projection_buttons["lcc"].tooltip == "Lambert conformal"
    tip = "Mercator (recommended: its cells vary least in size)"
    assert s.projection_buttons["merc"].tooltip == tip
    assert ">Mercator: <b>" in s.summary.value  # the one in use, named
    calls = _count(monkeypatch, diag, "size_ratios")
    _click(s, "lcc")
    assert _pressed(s) == [s.projection.kind] == ["lcc"] and s.preview is not None
    assert ">Lambert conformal: <b>" in s.summary.value
    projection = s.projection
    _click(s, "lcc")  # the pressed button: nothing changes
    assert s.projection is projection and _pressed(s) == ["lcc"]
    # Solved again only for a new outline, not for a new projection or resolution
    s.resolution_box.value = 20
    assert not calls and _green(s) == ["merc"]
    _drag(s, _px(s, 2), np.add(_px(s, 2), 10))
    assert len(calls) == 1 and _pressed(s) == [s.projection.kind] == ["lcc"]


def test_the_projection_follows_the_outline_and_a_new_one_gets_the_green_kind():
    s = _new()
    _click(s, "lcc")
    # New coordinates keep the picked kind, centred on them
    s.vertex_text.value = "20, 70*\n60, 70*\n60, 80*\n20, 80*"
    s.apply_button.click()
    assert s.projection.kind == "lcc" and s.projection.lon_0 == pytest.approx(40)
    assert 71 < s.projection.lat_1 < s.projection.lat_2 < 79 and s.preview is not None
    # A ring drawn round the North Pole: only Polar can take it
    s.clear_button.click()
    for lon in (0, 90, 180, 270):
        _drag(s, _px_of(s, lon, 78))
    _drag(s, _px(s, 0))
    assert _pressed(s) == _green(s) == ["stere"] and s.preview is not None
    assert s.projection.lat_0 == 90 and "Errors" not in s.messages.value


def test_an_unusable_projection_is_refused_and_the_buttons_kept():
    s = _new(Outline.from_bbox(-10, 10, -5, 5), resolution_km=110, projection="auto")
    assert _pressed(s) == [s.projection.kind] == ["merc"]
    # Lambert's cone is flat for a box centred on the equator
    _click(s, "lcc")
    assert _pressed(s) == [s.projection.kind] == ["merc"]
    assert "off the equator" in s.status.value


def test_a_pole_outline_uses_stere_and_a_bad_projection_changes_nothing():
    s = _new(Outline([0, 90, 180, 270], [75] * 4, [0, 1, 2, 3]))
    assert s.projection.kind == "stere" and s.preview is not None
    xlim = s.ax.get_xlim()
    _click(s, "merc")
    assert _pressed(s) == [s.projection.kind] == ["stere"]
    assert "<b>" in s.status.value
    assert s.ax.get_xlim() == xlim and s.preview is not None


# --- diagnostics on the map ---


def test_flagged_edges_and_the_hover_on_a_shared_edge(get_sketch, monkeypatch):
    real = diag.cell_metrics

    def fake(cg, **kw):
        m = real(cg, **kw)
        m["status"] = 0 * m["status"]
        m["status"][4, 5], m["status"][4, 6] = 2, 1
        return m

    monkeypatch.setattr(diag, "cell_metrics", fake)
    s = get_sketch
    s.resolution_box.value = 100
    cg, colors = s._shown, [to_hex(c) for c in s.flag_lines.get_colors()]
    assert len(colors) == 7 and colors.count(RED) == 4
    lengths = {len(seg) for seg in s.grid_lines.get_segments()}
    assert lengths == {cg.nx + 1, cg.ny + 1}
    assert "(i=5, j=4) BAD" in s.ax.format_coord(cg.x[9, 10], cg.y[9, 10])
    assert "(i=4, j=4): OK" in s.ax.format_coord(cg.x[9, 9], cg.y[9, 9])


def test_only_ocean_cells_are_flagged_and_warned_about(get_california, monkeypatch):
    s = _new(Outline(**get_california), resolution_km=12)
    cg, wet = s._shown, s._wet

    def hover(j, i):
        return s.ax.format_coord(cg.x[2 * j + 1, 2 * i + 1], cg.y[2 * j + 1, 2 * i + 1])

    land = np.argwhere((diag.cell_metrics(s.preview)["status"] > 0) & ~wet)
    assert len(land) and not s._metrics["status"][~wet].any()
    assert "over land" in hover(*land[0]) and "warn >" not in hover(*land[0])
    assert "warn >" in hover(*np.argwhere(s._metrics["status"] > 0)[0])
    lines = re.findall(r"margin-top:3px'>(.*?)</div>", s.messages.value)
    # Sentences, but for the TopoEditor's "N small open boundaries along coastline"
    sentence = [m[0].isupper() and m.endswith(".") or "coastline" in m for m in lines]
    assert lines and all(sentence)
    assert "Warnings" in s.messages.value and "Errors" not in s.messages.value
    # With no coastline every cell counts, so land cells are flagged too
    monkeypatch.setattr(diag, "ocean_mask", lambda cg: None)
    everywhere = _new(Outline(**get_california), resolution_km=12)
    assert len(everywhere.flag_lines.get_segments()) > len(s.flag_lines.get_segments())


def test_the_messages_block_shows_only_headers_with_lines():
    assert gs._messages_html([], []) == ""
    warn = gs._messages_html([], ["Abrupt cell-size change near corner 3 (4 cells)."])
    assert "<b>Warnings</b>" in warn and "Errors" not in warn and "<li>" not in warn
    both = gs._messages_html(["Folded (inside-out) cells near corner 1."], ["W."])
    assert both.index("color:#b00020'>Errors</b>") < both.index("<b>Warnings</b>")


def test_resolution_sets_the_cells_and_the_colourbar_reads(get_sketch, monkeypatch):
    wide, minor = gs._cbar_formatter(2.0), gs._cbar_formatter(2.0, minor=True)
    assert [wide(v) for v in (0.5, 20, 1000)] == ["0.5", "20", "1000"]
    assert [minor(v) for v in (2, 3, 50, 70)] == ["2", "", "50", ""]
    narrow = gs._cbar_formatter(0.03)
    assert len({narrow(v) for v in (19.0, 19.3, 19.6)}) == 3
    s = get_sketch
    n = s.preview.nx * s.preview.ny
    s.resolution_box.value = 55
    assert 3 < s.preview.nx * s.preview.ny / n < 5
    assert f"{s.preview.nx} × {s.preview.ny}" in s.cells_actual.value
    assert isinstance(s.cbar_ax.xaxis.get_major_formatter(), FuncFormatter)
    real, flat = diag.cell_metrics, np.full((s.preview.ny, s.preview.nx), 50.0)
    monkeypatch.setattr(
        diag, "cell_metrics", lambda cg, **kw: real(cg, **kw) | dict(size=flat)
    )
    s._recompute_preview()
    assert s.cbar_ax.get_visible() and s.shade_mesh.get_array().min() == 50.0
    # A narrow range with 2 log ticks in it gets plain ones
    flat = np.linspace(7.05, 9.06, flat.size).reshape(flat.shape)
    s._recompute_preview()
    s.fig.draw_without_rendering()
    axis, labels = s.cbar_ax.xaxis, []
    for minor in (False, True):
        ticks = zip(axis.get_ticklocs(minor=minor), axis.get_ticklabels(minor=minor))
        labels += [t.get_text() for v, t in ticks if 7.05 <= v <= 9.06 and t.get_text()]
    assert len(labels) >= 3
    assert labels[0] == "7.05" and labels[-1] == "9.06"  # both ends labelled
    s.shade_cells.value = False
    assert s.shade_mesh is None and not s.cbar_ax.get_visible()


def test_open_boundaries_and_a_tiny_run_that_turns_build_amber(get_sketch, monkeypatch):
    real = diag.open_boundary_runs

    def fake(cg, ocean=None):
        runs = real(cg, ocean=ocean)
        tiny = dict(start=2, stop=3, length_cells=1, length_km=9.0, tiny=True)
        runs["south"]["runs"] = [dict(tiny, lon=0.0, lat=-5.0)]
        runs["north"]["runs"] = []
        return runs

    monkeypatch.setattr(diag, "open_boundary_runs", fake)
    s = get_sketch
    s.resolution_box.value = 100
    assert s.open_boundaries == ["south", "east", "west"]
    line = "<div style='margin-top:3px'>2 small open boundaries along coastline</div>"
    assert line in s.messages.value  # one line for both sides, under Warnings
    x_tiny, _ = s.globe.to_xy(0.0, -5.0)
    assert np.isclose(s.obc_tiny_markers.get_xdata(), x_tiny).sum() == 1
    assert s.build_button.button_style == "warning" and not s.build_button.disabled
    q, segs = s._shown, s.obc_lines.get_segments()
    south = [(q.x[0, 4], q.y[0, 4]), (q.x[0, 6], q.y[0, 6])]
    assert np.allclose(segs[2], south) and to_hex(s.obc_lines.get_colors()[2]) == PINK
    north = slice(q.nx + q.ny, 2 * q.nx + q.ny)
    assert PINK not in [to_hex(c) for c in s.obc_lines.get_colors()[north]]


def test_the_tooltip_appears_after_a_rest_and_hides_on_leave_or_press(get_sketch):
    s = get_sketch
    cg = s._shown
    cell = s.ax.transData.transform((cg.x[1, 1], cg.y[1, 1]))
    s._on_hover(_event(s, *cell))
    assert not s.hover_annotation.get_visible()
    _fire(s, "hover")
    text = s.hover_annotation.get_text()
    assert s.hover_annotation.get_visible() and text.startswith("Cell (i=0, j=0): OK")
    assert "\nsouth/west side" in text
    s._on_hover(_event(s, inside=False))
    assert not s.hover_annotation.get_visible()
    # Leaving the canvas straight from the map sends no axes_leave; it still disarms
    s._on_hover(_event(s, *cell))
    LocationEvent("figure_leave_event", s.fig.canvas, *cell)._process()
    assert "hover" not in s._timers
    for gesture in (s._on_press, lambda e: s._on_scroll(e) or s._on_release(e)):
        s._on_hover(_event(s, *cell))
        _fire(s, "hover")
        assert s.hover_annotation.get_visible()
        gesture(_event(s, *_px(s, 0), step=1))
        assert not s.hover_annotation.get_visible() and "hover" not in s._timers
        s._on_release(_event(s, *_px(s, 0)))


def test_the_hover_box_shows_only_over_grid_cells(get_sketch):
    s = get_sketch
    # Open sea west of the grid, then the coast north of it
    for lon, lat in [(-10.5, 0.0), (0.0, 5.5)]:
        s._on_hover(_event(s, *_px_of(s, lon, lat)))
        assert "hover" not in s._timers and not s.hover_annotation.get_visible()
        assert s.ax.format_coord(*s.globe.to_xy(lon, lat)) == ""
    s._on_hover(_event(s, *_px_of(s, 9.5, 4.5)))
    _fire(s, "hover")
    assert s.hover_annotation.get_text().startswith("Cell (i=")


# --- build and save ---


def test_build_then_save_writes_one_reopenable_file(get_sketch, tmp_path, monkeypatch):
    real = cf.solve
    fast = dict(raster_points=40_000)
    monkeypatch.setattr(cf, "solve", lambda *a, **k: real(*a, **{**k, **fast}))
    s = get_sketch
    s.working_dir, s.name_box.value = tmp_path, "box"
    s.save_button.click()
    assert "Build a grid" in s.status.value
    s.build_button.click()
    g = s.grid
    assert (g.nx, g.ny) == (s.preview.nx, s.preview.ny) and "Built" in s.status.value
    assert set(g.outline) == {"lon", "lat", "corners", "projection", "resolution_km"}
    assert g.outline["lon"] == [-10, 10, 10, -10] and g.outline["resolution_km"] == 110
    s.save_button.click()
    files = list((tmp_path / "GridLibrary").iterdir())
    assert [f.name for f in files] == ["grid_box.nc"]
    r = _new(Grid.from_supergrid(str(files[0])))
    assert r.outline.to_dict() == s.outline.to_dict() and r.resolution_km == 110


def test_a_failed_build_or_preview_is_reported(get_sketch, monkeypatch):
    s = get_sketch
    monkeypatch.setattr(cf, "solve", _boom)
    s.build_button.click()
    assert "Build failed" in s.status.value and not s.build_button.disabled
    assert s.grid is None and s.preview is not None
    s.resolution_box.value = 100
    assert "Preview failed: boom." in s.messages.value and s.build_button.disabled
    assert "Fix the errors below to build." in s.status.value


def test_a_grid_too_big_to_build_here_is_an_error(get_sketch):
    s = get_sketch
    s.resolution_box.value = 0.1  # the 100 m floor: hundreds of millions of cells
    too_big = f"{s._n_cells() / 1e6:.3g} million cells need over 4 GB to build"
    assert f"{too_big}: use Grid.from_outline in a job" in s.messages.value
    assert s.preview is None and s.build_button.disabled
    s.resolution_box.value = 110
    assert s.preview is not None and not s.build_button.disabled


# --- robustness ---


def test_a_callback_error_goes_to_the_status_line(get_sketch, monkeypatch):
    s = get_sketch
    monkeypatch.setattr(s, "_new_axes", _boom)
    s.reset_view_button.click()
    assert "Error:</b> boom" in s.status.value


def test_random_gestures_keep_the_preview_and_buttons_consistent(get_sketch):
    s, rng = get_sketch, random.Random(1)
    for _ in range(25):
        if s.outline.n < 3:
            s.undo_button.click()
            continue
        i = rng.randrange(s.outline.n)
        p, q = _px(s, i), _px(s, (i + 1) % s.outline.n)
        step = rng.choice(["drag", "dbl", "del", "undo", "redo", "view", "edge"])
        if step == "drag":
            _drag(s, p, np.add(p, (rng.uniform(-15, 15), rng.uniform(-15, 15))))
        elif step in ("dbl", "del"):
            button = 3 if step == "del" else 1
            s._on_press(_event(s, *p, dblclick=step == "dbl", button=button))
        elif step in ("undo", "redo"):
            getattr(s, f"{step}_button").click()
        elif step == "view":
            s._on_scroll(_event(s, *p, step=rng.choice([1, -1])))
            _drag(s, p, np.add(p, (rng.uniform(-30, 30), 10)), button=2)
            _fire(s, "lite")
        else:
            _drag(s, np.add(p, q) / 2, np.add(p, q) / 2 + 5)
        # Build is greyed out only while drawing or with errors listed to say why
        assert s.build_button.disabled == (s._drawing or "Errors" in s.messages.value)
        assert s.cbar_ax.get_visible() == (s.preview is not None)
        assert (len(s.grid_lines.get_segments()) > 0) == (s.preview is not None)
        assert _pressed(s) == [s.projection.kind]
    s.close()
    assert s.fig.number not in plt.get_fignums() and s.comm is None
