from unittest.mock import MagicMock

# --- active_cells ---


def test_active_cells_empty(get_editor):
    """No selection returns empty list."""
    assert get_editor.active_cells == []


def test_active_cells_single_cell(get_editor):
    """Single cell selection returns correct (j, i) tuple."""
    get_editor._select_cell(3, 2)  # note (i, j) order
    assert get_editor.active_cells == [(2, 3)]


def test_active_cells_rect(get_editor):
    """Rectangle selection returns all selected (j, i) tuples."""
    get_editor._selected_cells = [(0, 0), (0, 1), (1, 0)]
    assert get_editor.active_cells == [(0, 0), (0, 1), (1, 0)]


def test_active_cells_rect_takes_priority(get_editor):
    """Rectangle selection takes priority over single cell."""
    get_editor._select_cell(0, 0)
    get_editor._selected_cells = [(1, 1), (2, 2)]
    assert get_editor.active_cells == [(1, 1), (2, 2)]


# --- mask ---


def test_mask_single_cell(get_editor):
    get_editor._select_cell(0, 0)
    get_editor.on_mask_change({"new": "Land"})
    assert get_editor.topo.tmask.data[0, 0] == 0


def test_mask_no_op_if_same_value(get_editor):
    """Should not apply edit if mask value unchanged."""
    get_editor._select_cell(0, 0)
    get_editor.on_mask_change({"new": "Ocean"})
    history_len = len(get_editor.topo.tcm.history_dict)
    get_editor.on_mask_change({"new": "Ocean"})
    assert len(get_editor.topo.tcm.history_dict) == history_len  # no new command


def test_mask_multi_cell_rect(get_editor):
    """Rectangle selection applies mask to all selected cells."""
    get_editor._selected_cells = [(0, 0), (0, 1), (1, 0)]
    get_editor.on_mask_change({"new": "Land"})
    assert get_editor.topo.tmask.data[0, 0] == 0
    assert get_editor.topo.tmask.data[0, 1] == 0
    assert get_editor.topo.tmask.data[1, 0] == 0


def test_mask_no_selection(get_editor):
    """No selection should not raise or apply any edit."""
    get_editor.on_mask_change({"new": "Land"})  # should just return


# --- depth ---


def test_depth_single_cell(get_editor):
    get_editor._select_cell(0, 0)
    get_editor.on_depth_change({"new": 500.0})
    assert get_editor.topo.depth.data[0, 0] == 500.0


def test_depth_no_op_if_same_value(get_editor):
    get_editor._select_cell(0, 0)
    get_editor.on_depth_change({"new": 1000.0})  # already flat 1000
    history_len = len(get_editor.topo.tcm.history_dict)
    get_editor.on_depth_change({"new": 1000.0})
    assert len(get_editor.topo.tcm.history_dict) == history_len


def test_depth_multi_cell_rect(get_editor):
    """Rectangle selection applies depth to all selected cells."""
    get_editor._selected_cells = [(0, 0), (0, 1), (1, 0)]
    get_editor.on_depth_change({"new": 200.0})
    assert get_editor.topo.depth.data[0, 0] == 200.0
    assert get_editor.topo.depth.data[0, 1] == 200.0
    assert get_editor.topo.depth.data[1, 0] == 200.0


# --- undo/redo ---


def test_undo_depth_change(get_editor):
    get_editor._select_cell(0, 0)
    get_editor.on_depth_change({"new": 500.0})
    assert get_editor.topo.depth.data[0, 0] == 500.0
    get_editor.undo_last_edit()
    assert get_editor.topo.depth.data[0, 0] == 1000.0


def test_redo_depth_change(get_editor):
    get_editor._select_cell(0, 0)
    get_editor.on_depth_change({"new": 500.0})
    get_editor.undo_last_edit()
    get_editor.redo_last_edit()
    assert get_editor.topo.depth.data[0, 0] == 500.0


def test_undo_mask_change(get_editor):
    get_editor._select_cell(0, 0)
    original = get_editor.topo.tmask.data[0, 0]
    get_editor.on_mask_change({"new": "Land"})
    get_editor.undo_last_edit()
    assert get_editor.topo.tmask.data[0, 0] == original


# --- rect select toggle ---


def test_rect_toggle_gates_double_click(get_editor):
    """Double click should be ignored when rect mode is active."""
    get_editor._rect_or_single_select_button.value = "Rectangular Area"
    mock_event = MagicMock()
    mock_event.dblclick = True
    mock_event.xdata = 279.0
    mock_event.ydata = 8.0
    get_editor.on_double_click(mock_event)
    assert get_editor._selected_cell is None


def test_rect_toggle_off_clears_selection(get_editor):
    """Turning off rect mode clears selected cells."""
    get_editor._selected_cells = [(0, 0), (1, 1)]
    get_editor._rect_or_single_select_button.value = "Rectangular Area"
    get_editor._rect_or_single_select_button.value = "Single Cell"
    assert get_editor._selected_cells == []


# --- rect select logic ---


def test_rect_select_finds_cells_in_bounds(get_editor):
    """Cells within the rectangle bounds are selected."""
    mock_eclick = MagicMock()
    mock_erelease = MagicMock()
    mock_eclick.xdata = 278.0
    mock_eclick.ydata = 7.0
    mock_erelease.xdata = 279.0
    mock_erelease.ydata = 8.0
    get_editor._on_rect_select(mock_eclick, mock_erelease)
    assert len(get_editor._selected_cells) > 0


def test_rect_select_empty_outside_bounds(get_editor):
    """No cells selected when rectangle is outside the grid."""
    mock_eclick = MagicMock()
    mock_erelease = MagicMock()
    mock_eclick.xdata = 0.0
    mock_eclick.ydata = 0.0
    mock_erelease.xdata = 1.0
    mock_erelease.ydata = 1.0
    get_editor._on_rect_select(mock_eclick, mock_erelease)
    assert len(get_editor._selected_cells) == 0
