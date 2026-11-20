"""TopoEditor's Python half: state it sends, and what each front-end action does."""

import numpy as np
import pytest

from mom6_forge.widgets._base import as_bytes
from mom6_forge.widgets.topo_editor import TopoEditor


def _editor(topo, monkeypatch):
    """An editor whose custom messages (patches, reloads) are kept in ``editor.sent``."""
    editor = TopoEditor(topo)
    editor.sent = []
    monkeypatch.setattr(
        editor,
        "send",
        lambda content, buffers=None: editor.sent.append((content, buffers)),
    )
    return editor


def _decode(raw, dtype, editor):
    return np.frombuffer(raw, dtype=dtype).reshape(editor.ny, editor.nx)


def _patch(editor):
    """(cells, depth, mask) from the single patch the last action sent."""
    ((content, buffers),) = editor.sent
    assert content["kind"] == "patch"
    cells = np.frombuffer(buffers[0], "<i4").reshape(-1, 2)
    return cells, np.frombuffer(buffers[1], "<f4"), np.frombuffer(buffers[2], "u1")


@pytest.mark.parametrize(
    "fixture, history",
    [("get_rect_topo_with_vc", True), ("get_rect_topo_without_vc", False)],
)
def test_construction_sends_the_topo(request, fixture, history):
    topo = request.getfixturevalue(fixture)
    editor = TopoEditor(topo)

    assert (editor.ny, editor.nx) == topo.depth.shape
    np.testing.assert_array_equal(
        _decode(editor.depth, "<f4", editor), topo.masked_depth.data.astype("<f4")
    )
    np.testing.assert_array_equal(_decode(editor.mask, "u1", editor), topo.tmask.data)
    assert editor.has_history is history


def test_brush_stroke_changes_exactly_its_cells(get_rect_topo_without_vc, monkeypatch):
    topo = get_rect_topo_without_vc
    editor = _editor(topo, monkeypatch)
    cells = np.array([[0, 0], [2, 3], [5, 7]])

    editor._act(
        "edit", buffers=[as_bytes(cells, "<i4")], edit={"op": "depth", "value": 250.0}
    )

    assert not editor.errors, editor.last_traceback
    expected = np.full(topo.depth.shape, 1000.0)
    expected[cells[:, 0], cells[:, 1]] = 250.0
    np.testing.assert_array_equal(topo.depth.data, expected)
    sent_cells, depth, mask = _patch(editor)
    assert {tuple(c) for c in sent_cells} == {tuple(c) for c in cells}
    np.testing.assert_array_equal(depth, 250.0)
    np.testing.assert_array_equal(mask, 1)


@pytest.mark.parametrize("op, value", [("land", 0), ("ocean", 1)])
def test_mask_edit_on_a_selected_box(get_rect_topo_without_vc, monkeypatch, op, value):
    topo = get_rect_topo_without_vc
    if op == "ocean":
        topo.depth.data[2:5, 3:8] = -10.0  # start the box as land
    editor = _editor(topo, monkeypatch)

    editor._act("edit", edit={"op": op, "rect": [2, 3, 4, 7]})  # [j0, i0, j1, i1]

    assert not editor.errors, editor.last_traceback
    box = np.zeros(topo.depth.shape, dtype=bool)
    box[2:5, 3:8] = True
    np.testing.assert_array_equal(topo.tmask.data[box], value)
    np.testing.assert_array_equal(topo.tmask.data[~box], 1)
    sent_cells, _, mask = _patch(editor)
    assert len(sent_cells) == box.sum()
    np.testing.assert_array_equal(mask, value)


def test_large_change_resends_the_full_arrays(get_rect_topo_without_vc, monkeypatch):
    """Past a quarter of the grid the page reloads from the traits, so they must be current."""
    topo = get_rect_topo_without_vc
    editor = _editor(topo, monkeypatch)

    editor._act(
        "edit", edit={"op": "depth", "value": 300.0, "rect": [0, 0, editor.ny - 1, 20]}
    )

    assert not editor.errors, editor.last_traceback
    assert [content for content, _ in editor.sent] == [{"kind": "reload"}]
    np.testing.assert_array_equal(
        _decode(editor.depth, "<f4", editor), topo.masked_depth.data.astype("<f4")
    )


def test_erase_disconnected_basins_in_the_basinmask_view(
    get_rect_topo_without_vc, monkeypatch
):
    topo = get_rect_topo_without_vc
    topo.depth.data[:, 20] = -10.0  # a land column splits the domain in two basins
    editor = _editor(topo, monkeypatch)

    editor._act("view", view="basinmask")
    assert len(np.unique(_decode(editor.basins, "<i4", editor))) == 3  # land + 2 basins

    editor._act("edit", edit={"op": "erase_disconnected_basin", "cell": [5, 30]})

    assert not editor.errors, editor.last_traceback
    assert not topo.tmask.data[:, :20].any()  # the western basin is gone
    assert topo.tmask.data[:, 21:].all()
    np.testing.assert_array_equal(
        _decode(editor.basins, "<i4", editor), topo.basintmask.data
    )


def test_undo_redo_reset_round_trip(get_rect_topo_with_vc, monkeypatch):
    topo = get_rect_topo_with_vc
    editor = _editor(topo, monkeypatch)
    cells = np.array([[1, 1], [1, 2]])
    stroke = as_bytes(cells, "<i4")

    editor._act("edit", buffers=[stroke], edit={"op": "depth", "value": 250.0})
    assert editor.can_undo and not editor.can_redo
    editor._act("undo")
    assert (topo.depth.data[1, 1:3] == 1000.0).all() and editor.can_redo
    editor._act("redo")
    assert (topo.depth.data[1, 1:3] == 250.0).all()
    editor.sent.clear()
    editor._act("reset")

    assert not editor.errors, editor.last_traceback
    np.testing.assert_array_equal(topo.depth.data, 1000.0)
    sent_cells, depth, _ = _patch(editor)  # the page gets the restored cells
    assert {tuple(c) for c in sent_cells} == {tuple(c) for c in cells}
    np.testing.assert_array_equal(depth, 1000.0)


def test_undo_without_history_reports_in_the_widget(get_rect_topo_without_vc):
    editor = TopoEditor(get_rect_topo_without_vc)

    editor._act("undo")

    assert editor.errors and editor.errors[0].startswith("RuntimeError")
