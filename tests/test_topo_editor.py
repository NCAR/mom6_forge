import os
import numpy as np
import pytest
import git
import json

from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo
from mom6_bathy.edit_command import DepthEditCommand
from mom6_bathy.topo_editor import TopoEditor


@pytest.fixture
def minimal_grid_and_topo():
    """Test that a minimal 5x5 grid can be set up for the Panama region"""
    grid = Grid(
        resolution=0.1, xstart=278.0, lenx=0.5, ystart=7.0, leny=0.5, name="testpanama"
    )
    topo = Topo(grid=grid, min_depth=10.0)
    topo.set_spoon(2000.0, 200)
    return topo


def test_grid_topo(minimal_grid_and_topo):
    topo = minimal_grid_and_topo
    assert topo._grid.nx == 5
    assert topo._grid.ny == 5
    assert topo.depth.shape == (5, 5)
