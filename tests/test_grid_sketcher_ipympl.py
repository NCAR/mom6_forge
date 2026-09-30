import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parent / "_ipympl_worker.py"


@pytest.fixture(scope="module")
def get_results():
    """The worker's results; it runs apart because the backend choice is per process."""
    path = os.pathsep.join([str(WORKER.parents[1]), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, PYTHONPATH=path)
    run = [sys.executable, str(WORKER)]
    proc = subprocess.run(run, capture_output=True, text=True, timeout=600, env=env)
    if "No module named 'ipympl'" in proc.stderr:
        pytest.skip("ipympl is not installed")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --- display ---


def test_the_png_fills_the_canvas_at_device_pixel_ratio_1_and_2(get_results):
    assert get_results["png"] == [[100, 450, 500], [200, 900, 1000]]


def test_the_framed_map_sits_beside_the_panel_and_captures_scroll(get_results):
    """The mouse instructions sit right under the map, as wide as it, with no footer."""
    assert get_results["layout"] == dict(
        flow="row nowrap",
        canvas=["456px", "0 0 auto", "1px solid #b0b0b0", True],
        bars=[False, False, False, False, ""],
        panel="400px",
        help=["456px", 1],
        edits=["Undo", "Redo", "Clear all", "Draw Box", "Reset view"],
        tips_without_description=[],
    )


# --- gestures ---


def test_a_browser_double_click_toggles_a_corner_once(get_results):
    assert get_results["double_click"] == [0, 2, 3]


def test_right_click_and_ctrl_click_delete_a_vertex(get_results):
    assert get_results["delete"] == [4, 4]


def test_a_press_on_a_corner_disc_rim_grabs_it_at_pixel_ratio_1_and_2(get_results):
    assert get_results["grab"] == [[0, 4], [0, 4]]


def test_while_drawing_clicks_add_a_drag_pans_right_click_deletes(get_results):
    assert get_results["drawing"] == [5, True, 5, 4, [0, 1, 2, 3]]


def test_a_drag_moves_by_the_dragged_pixels_and_sketches_at_once(get_results):
    r = get_results["drag"]
    assert abs(r["dx"] - 15) < 1 and abs(r["dy"] - 10) < 1
    assert r["sketched"] == 1 and r["motion"] == [1, 0] and r["answer"] == [0, 1]
    assert r["release"][0] >= 1 and r["release"][1] == 0 and r["mode"] == "diff"


def test_leaving_mid_pan_ends_it_so_box_then_drag_draws_a_box(get_results):
    r = get_results["gestures"]
    assert r["still"] and r["box"] == [4, False]  # 4 corners, no intermediate points


def test_the_handles_move_and_turn_and_the_knob_keeps_its_turn(get_results):
    r = get_results["gestures"]
    assert r["moved"] and r["turned"] and abs(r["knob"]) < 20


# --- frames: only ever in reply to the browser's draw request ---


def test_no_frame_is_ever_sent_unasked(get_results):
    assert get_results["unasked"] == 0


def test_the_tooltip_asks_for_one_frame_and_an_edge_press_hides_it(get_results):
    r = get_results["tooltip"]
    assert r["left"] and r["visible"] and r["rest"] == [1, 0] and r["answer"] == [0, 1]
    assert "Cell (i=0, j=0)" in r["footer"]
    assert r["hidden"] and r["edge_press"] == [1, 0]


def test_scroll_and_pan_coalesce_their_draw_requests(get_results):
    r = get_results["view"]
    assert 1 <= r["scroll"][0] <= 2 and r["scroll"][1] == 0 and r["lite"]
    assert r["after"][0] <= 2 and r["after"][1] == 0 and not r["lite_after"]
    # The frame that restores full detail is a whole one: it repaints any lost diff
    assert r["end_mode"] == "full" and r["end"] == [0, 1]
    assert r["pan"][0] <= 2 and r["pan"][1] == 0 and not r["moved"]
