from unittest.mock import MagicMock

import numpy as np
import pytest

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.topo_editor import TopoEditor

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
    mock_event.xdata = 279.0 - get_editor._central_longitude
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
    # The axes are centred on the domain, so event.xdata is native to that
    # shifted frame -- offset the true longitudes the same way a real drag
    # inside the domain would arrive.
    central_lon = get_editor._central_longitude
    mock_eclick = MagicMock()
    mock_erelease = MagicMock()
    mock_eclick.xdata = 278.0 - central_lon
    mock_eclick.ydata = 7.0
    mock_erelease.xdata = 279.0 - central_lon
    mock_erelease.ydata = 8.0
    get_editor._on_rect_select(mock_eclick, mock_erelease)
    assert len(get_editor._selected_cells) > 0


def test_rect_select_empty_outside_bounds(get_editor):
    """No cells selected when rectangle is outside the grid."""
    # 120-121E is far outside the 278-282E domain, so this exercises the
    # longitude mask and not just the latitude one.
    central_lon = get_editor._central_longitude
    mock_eclick = MagicMock()
    mock_erelease = MagicMock()
    mock_eclick.xdata = 120.0 - central_lon
    mock_eclick.ydata = 0.0
    mock_erelease.xdata = 121.0 - central_lon
    mock_erelease.ydata = 1.0
    get_editor._on_rect_select(mock_eclick, mock_erelease)
    assert len(get_editor._selected_cells) == 0


# --- antimeridian-crossing domains ---


def _editor_for(xstart, lenx, ystart=-10.0, leny=20.0):
    """A TopoEditor over a flat domain spanning [xstart, xstart + lenx]."""
    grid = Grid(
        resolution=0.5,
        xstart=xstart,
        lenx=lenx,
        ystart=ystart,
        leny=leny,
        name="dateline",
    )
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)
    return grid, TopoEditor(topo, build_ui=False)


@pytest.mark.parametrize(
    "xstart, lenx",
    [
        (170.0, 20.0),  # crosses +180
        (-190.0, 20.0),  # crosses -180
        (278.0, 4.0),  # ordinary out-of-[-180, 180] domain
        (-82.0, 4.0),  # ordinary in-range domain
    ],
)
def test_axes_zoom_to_domain_not_globe(xstart, lenx):
    """set_extent must zoom to the domain, including one running past +/-180.

    A plain PlateCarree() axes has its own seam at +/-180 and silently resets
    to the whole globe when asked for an extent that crosses it.
    """
    _, editor = _editor_for(xstart, lenx)
    x_min, x_max = editor.ax.get_xlim()
    assert x_max - x_min == pytest.approx(lenx)


def test_click_recovers_true_longitude_across_dateline():
    """event.xdata is native to the shifted frame; handlers must undo the shift."""
    grid, editor = _editor_for(170.0, 20.0)
    true_lon, true_lat = 185.0, 4.0
    expected = grid.get_indices(true_lat, true_lon)

    event = MagicMock()
    event.dblclick = True
    event.xdata = true_lon - editor._central_longitude
    event.ydata = true_lat
    editor._rect_or_single_select_button.value = "Single Cell"
    editor.on_double_click(event)

    assert editor._selected_cell[:2] == expected


def test_rect_select_spans_the_antimeridian():
    """A drag straddling +/-180 selects the cells it actually covers."""
    grid, editor = _editor_for(170.0, 20.0)
    central_lon = editor._central_longitude

    eclick, erelease = MagicMock(), MagicMock()
    eclick.xdata, eclick.ydata = 178.0 - central_lon, -2.0
    erelease.xdata, erelease.ydata = 182.0 - central_lon, 2.0
    editor._on_rect_select(eclick, erelease)

    tlon = (grid.tlon.data + 360) % 360
    tlat = grid.tlat.data
    expected = (
        (tlon >= 178.0) & (tlon <= 182.0) & (tlat >= -2.0) & (tlat <= 2.0)
    ).sum()

    assert expected > 0
    assert len(editor._selected_cells) == expected


def test_hover_readout_reports_true_longitude():
    """format_coord converts back out of the shifted frame."""
    _, editor = _editor_for(170.0, 20.0)
    readout = editor.ax.format_coord(185.0 - editor._central_longitude, 0.0)
    assert "x=185.00" in readout


# --- canvas refresh after basin edits ---


def _editor_with_two_basins(tmp_path):
    """An editor in basinmask mode over a domain split into two basins."""
    grid = Grid(
        resolution=0.5,
        xstart=0.0,
        lenx=20.0,
        ystart=0.0,
        leny=10.0,
        name="basins",
    )
    topo = Topo(grid, min_depth=0, version_control_dir=tmp_path, git=True)
    topo.set_flat(1000)
    topo.depth.data[:, 20] = -10.0  # land barrier splits the domain in two

    editor = TopoEditor(topo, build_ui=False)
    editor._display_mode_toggle.value = "basinmask"
    editor._select_cell(30, 5)  # a cell east of the barrier
    return topo, editor


@pytest.mark.parametrize(
    "handler", ["erase_selected_basin", "erase_disconnected_basin"]
)
def test_erase_basin_redraws_canvas(tmp_path, handler):
    """Erasing a basin must redraw the canvas, not just update the buttons.

    Topo.erase_* applies its edit through the topo, bypassing
    TopoEditor.apply_edit and therefore its refresh, so the mask used to
    change underneath a stale plot.
    """
    topo, editor = _editor_with_two_basins(tmp_path)
    assert len(set(topo.basintmask.data.ravel().tolist())) > 2  # land + 2 basins

    before = np.array(editor.im.get_array(), dtype=float)
    getattr(editor, handler)(None)
    after = np.array(editor.im.get_array(), dtype=float)

    assert not np.array_equal(before, after)
    assert np.array_equal(after, np.asarray(topo.basintmask.data, dtype=float))
