"""Topo's JSON-able edits (apply_edits) and its git-backed undo, redo and reset."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from mom6_forge.topo import Topo


def _cells(array):
    return set(map(tuple, np.asarray(array).tolist()))


def test_undo_redo_round_trip(get_rect_topo_with_vc):
    topo = get_rect_topo_with_vc
    edit = {"op": "depth", "cells": [[1, 2]], "value": 500.0}

    assert _cells(topo.apply_edits([edit])) == {(1, 2)}
    assert topo.can_undo and not topo.can_redo

    assert _cells(topo.undo()) == {(1, 2)}
    assert topo.depth[1, 2] == 1000.0 and topo.can_redo

    assert _cells(topo.redo()) == {(1, 2)}
    assert topo.depth[1, 2] == 500.0 and not topo.can_redo


def test_reset_returns_to_baseline_and_survives_reload(get_rect_grid, tmp_path):
    topo = Topo(get_rect_grid, min_depth=10.0, version_control_dir=tmp_path, git=True)
    topo.set_flat(1000.0)
    baseline = topo.masked_depth.values.copy()
    topo.apply_edits(
        [
            {"op": "depth", "cells": [[1, 1], [2, 3]], "value": 5.0},  # now land
            {"op": "land", "cells": [[4, 4]]},  # starts the manual mask
            {"op": "ocean", "cells": [[1, 1]]},
        ]
    )

    assert _cells(topo.reset()) == {(1, 1), (2, 3), (4, 4)}
    np.testing.assert_array_equal(topo.masked_depth.values, baseline)

    reloaded = Topo.from_version_control(topo.domain_dir)
    assert reloaded.min_depth == 10.0
    np.testing.assert_array_equal(reloaded.masked_depth.values, baseline)
    assert reloaded.can_redo  # loading adds no commits, so the redo stack is kept


# A wall of land at i = 5 splits the 30 x 40 grid into two basins; (2, 2) is 50 m deep.
WALL = [[j, 5] for j in range(30)]
LEFT = {(j, i) for j in range(30) for i in range(5)}
RIGHT = {(j, i) for j in range(30) for i in range(6, 40)}


@pytest.mark.parametrize(
    "setup, edit, expected",
    [
        (
            [],
            {"op": "depth", "cells": [[1, 1], [20, 30]], "value": 500.0},
            {(1, 1), (20, 30)},
        ),
        ([], {"op": "land", "cells": [[1, 1], [2, 7], [3, 5]]}, {(1, 1), (2, 7)}),
        ([], {"op": "ocean", "cells": [[3, 5]]}, {(3, 5)}),
        ([{"op": "land", "cells": [[1, 1]]}], {"op": "clear_mask"}, {(1, 1)}),
        ([], {"op": "min_depth", "value": 100.0}, {(2, 2)}),
        (
            [],
            {
                "op": "fill_from_stats",
                "cells": [[1, 1], [2, 2], [4, 4]],
                "statistic": "max",
            },
            {(1, 1)},
        ),
        ([], {"op": "erase_selected_basin", "cell": [20, 2]}, LEFT),
        ([], {"op": "erase_disconnected_basin", "cell": [20, 2]}, RIGHT),
    ],
    ids=lambda x: x["op"] if isinstance(x, dict) else None,
)
def test_apply_edits_op(get_rect_topo_with_vc, setup, edit, expected):
    topo = get_rect_topo_with_vc
    topo.edit_depth([tuple(c) for c in WALL] + [(2, 2)], [0.0] * 30 + [50.0])
    d_max = np.full(topo.depth.shape, np.nan)
    d_max[1, 1], d_max[2, 2] = 700.0, 50.0  # (2, 2) unchanged, (4, 4) has no data
    topo.src = SimpleNamespace(stats=xr.Dataset({"D_max": (("ny", "nx"), d_max)}))
    topo.apply_edits(setup)
    before = topo.masked_depth.values.copy(), topo.tmask.values.copy()
    n_applied = len(topo.tcm._history()["applied"])

    assert _cells(topo.apply_edits([edit])) == expected
    assert len(topo.tcm._history()["applied"]) == n_applied + 1  # one history step

    assert _cells(topo.undo()) == expected
    np.testing.assert_array_equal(topo.masked_depth.values, before[0])
    np.testing.assert_array_equal(topo.tmask.values, before[1])


@pytest.mark.parametrize(
    "edit, message",
    [
        ({"op": "flood"}, "Unknown edit op"),
        ({"op": "depth", "cells": [[0, 0]]}, "missing"),
        ({"op": "land", "cells": [[0, 40]]}, "off the"),
        ({"op": "fill_from_stats", "cells": [[0, 0]], "statistic": "mean"}, "stats"),
    ],
)
def test_apply_edits_rejects_bad_edit_before_applying_any(
    get_rect_topo_without_vc, edit, message
):
    topo = get_rect_topo_without_vc
    good = {"op": "depth", "cells": [[0, 0]], "value": 5.0}
    with pytest.raises(ValueError, match=message):
        topo.apply_edits([good, edit])
    assert topo.depth[0, 0] == 1000.0


def test_history_checks_do_not_walk_git_log(get_rect_topo_with_vc, monkeypatch):
    topo = get_rect_topo_with_vc
    assert topo.can_undo  # builds the cache

    def walk(*args, **kwargs):
        raise AssertionError("walked the git log")

    monkeypatch.setattr(topo.tcm.repo, "iter_commits", walk)
    topo.apply_edits([{"op": "depth", "cells": [[0, 0]], "value": 5.0}])
    topo.undo()
    topo.redo()
    assert topo.can_undo and not topo.can_redo


def test_history_needs_git(get_rect_topo_without_vc):
    topo = get_rect_topo_without_vc
    assert not topo.can_undo and not topo.can_redo
    for step in (topo.undo, topo.redo, topo.reset):
        with pytest.raises(RuntimeError, match="create it with git=True"):
            step()


def test_topo_imports_without_xesmf_regionmask_or_git():
    code = """
import sys
sys.modules["xesmf"] = sys.modules["regionmask"] = sys.modules["git"] = None
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
grid = Grid(lenx=4.0, leny=3.0, resolution=1.0, xstart=278.0, ystart=7.0, name="t")
topo = Topo(grid, min_depth=10.0, git=False)
topo.set_flat(1000.0)
assert topo.apply_edits([{"op": "depth", "cells": [[0, 1]], "value": 5.0}]).tolist() == [[0, 1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True)
