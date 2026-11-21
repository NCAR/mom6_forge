from types import SimpleNamespace

import numpy as np
import pytest

from mom6_forge.vgrid import VGrid
from mom6_forge.widgets._base import from_buffer
from mom6_forge.widgets.vgrid_creator import VGridCreator


def test_params_build_the_same_grid_as_vgrid_hyperbolic(tmp_path):
    widget = VGridCreator(repo_root=tmp_path)
    # Beyond the old sliders' 100-level / 10000 m limits, which must not clamp.
    widget._act("params", kind="hyperbolic", nk=150, depth=12000.0, ratio=20.0)

    expected = VGrid.hyperbolic(nk=150, depth=12000.0, ratio=20.0).dz
    assert widget.errors == []
    np.testing.assert_array_equal(widget.vgrid.dz, expected)
    np.testing.assert_array_equal(from_buffer(widget.dz, np.float64), expected)


@pytest.mark.parametrize(
    "bad", [{"nk": 1}, {"nk": 2.5}, {"nk": None}, {"depth": 0.0}, {"ratio": -1.0}]
)
def test_invalid_params_show_an_error_and_keep_the_grid(tmp_path, bad):
    widget = VGridCreator(repo_root=tmp_path)
    before = widget.vgrid
    fields = {"kind": "hyperbolic", "nk": 20, "depth": 500.0, "ratio": 5.0, **bad}
    widget._act("params", **fields)

    assert widget.errors
    assert widget.vgrid is before


def test_save_then_load_round_trips_and_lists_the_grid(tmp_path):
    widget = VGridCreator(repo_root=tmp_path)
    assert not (tmp_path / "VGridLibrary").exists()  # created on save, not before

    widget._act("params", kind="hyperbolic", nk=30, depth=3000.0, ratio=8.0)
    saved = widget.vgrid.dz.copy()
    widget._act("save", name="deep", message="for the gulf")
    widget._act("params", kind="uniform", nk=5, depth=50.0, ratio=8.0)
    [entry] = widget.library
    assert (entry["name"], entry["nk"], entry["message"]) == (
        "deep",
        30,
        "for the gulf",
    )

    widget._act("load", file=entry["file"])
    assert widget.errors == []
    np.testing.assert_allclose(widget.vgrid.dz, saved)
    assert widget.vgrid.name == "deep"
    assert "vgrid_deep.nc" in widget.source  # the UI says it came from a file


def test_save_refuses_to_overwrite_until_confirmed(tmp_path, monkeypatch):
    widget = VGridCreator(repo_root=tmp_path)
    sent = []
    monkeypatch.setattr(
        widget, "send", lambda content, buffers=None: sent.append(content)
    )
    widget._act("save", name="g", message="first")
    widget._act("params", kind="uniform", nk=20, depth=100.0, ratio=1.0)

    widget._act("save", name="g", message="second")
    assert widget.errors and sent == [{"confirm_overwrite": "g"}]
    assert widget.library[0]["nk"] == 10

    widget._act("save", name="g", message="second", overwrite=True)
    assert widget.errors == []
    assert (widget.library[0]["nk"], widget.library[0]["message"]) == (20, "second")


def test_reset_restores_the_starting_grid_after_a_load(tmp_path):
    # Neither uniform nor hyperbolic, so the controls cannot rebuild it.
    start = VGrid(np.array([1.0, 2.0, 4.0, 8.0]), name="custom")
    widget = VGridCreator(vgrid=start, repo_root=tmp_path)
    widget._act("params", kind="uniform", nk=10, depth=100.0, ratio=1.0)
    widget._act("save", name="other", message="m")
    widget._act("load", file="vgrid_other.nc")

    widget._act("reset")
    assert widget.errors == []
    np.testing.assert_array_equal(widget.vgrid.dz, start.dz)
    assert widget.vgrid.name == "custom"


def test_warns_when_the_grid_is_shallower_than_the_topo(tmp_path):
    widget = VGridCreator(repo_root=tmp_path, topo=SimpleNamespace(max_depth=500.0))
    assert "500.0 m" in widget.warning  # default grid is 100 m deep

    widget._act("params", kind="uniform", nk=10, depth=600.0, ratio=1.0)
    assert widget.warning == ""
