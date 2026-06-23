"""
Benchmark tests demonstrating git efficiency improvements for large-domain depth/mask edits.

These tests require GLADE access for the tx2_3v3 global grid (~1M cells).
Run with: pytest tests/test_git_efficiency.py -m benchmark -v -s
"""

import json
import os
import time

import numpy as np
import pytest

from mom6_forge.edit_command import MaskEditCommand, SIZE_THRESHOLD
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo

TX2_3V3_HGRID = (
    "/glade/campaign/cesm/cesmdata/inputdata/ocn/mom/tx2_3v3/ocean_hgrid_250930.nc"
)


@pytest.fixture(scope="module")
def tx2_3v3_grid():
    return Grid.from_supergrid(TX2_3V3_HGRID)


@pytest.mark.benchmark
def test_large_domain_depth_commit_is_fast(tx2_3v3_grid, tmp_path):
    """set_flat() on a ~1M-cell global grid should commit quickly and leave a small JSON."""
    topo = Topo(
        tx2_3v3_grid,
        min_depth=10.0,
        version_control_dir=str(tmp_path / "TopoLibrary"),
        git=True,
    )

    t0 = time.perf_counter()
    topo.set_flat(1000.0)
    elapsed = time.perf_counter() - t0

    history_file = topo.tcm.history_file_path
    history_size = os.path.getsize(history_file)
    head_entry = json.loads(json.loads(history_file.read_text())["head"])

    assert "nc_path" in head_entry, "Expected nc_path key for large-domain depth edit"
    assert (
        "affected_indices" not in head_entry
    ), "Should not store inline indices for large edit"
    assert (
        history_size < 2_000
    ), f"command_history.json too large: {history_size} bytes (expected < 2 KB)"

    nc_path = head_entry["nc_path"]
    assert os.path.exists(nc_path), f"Large-edit .nc file not found: {nc_path}"

    print(
        f"\ntx2_3v3 set_flat: {elapsed:.2f}s | history JSON: {history_size} bytes | nc: {os.path.getsize(nc_path) // 1024} KB"
    )


@pytest.mark.benchmark
def test_small_depth_edit_stays_inline(tx2_3v3_grid, tmp_path):
    """Edits below SIZE_THRESHOLD should still use inline JSON — no .nc file."""
    topo = Topo(
        tx2_3v3_grid,
        min_depth=10.0,
        version_control_dir=str(tmp_path / "TopoLibrary"),
        git=True,
    )
    topo.set_flat(1000.0)

    small_indices = [(j, i) for j in range(5) for i in range(5)]  # 25 cells
    topo.edit_depth(small_indices, [500.0] * len(small_indices))

    head_entry = json.loads(json.loads(topo.tcm.history_file_path.read_text())["head"])
    assert "affected_indices" in head_entry, "Small edit should use inline JSON"
    assert len(head_entry["affected_indices"]) == 25


@pytest.mark.benchmark
def test_large_mask_edit_uses_nc(tx2_3v3_grid, tmp_path):
    """Large mask edits (above SIZE_THRESHOLD) should also use .nc storage."""
    topo = Topo(
        tx2_3v3_grid,
        min_depth=10.0,
        version_control_dir=str(tmp_path / "TopoLibrary"),
        git=True,
    )
    topo.set_flat(1000.0)

    large_indices = list(np.ndindex(topo._depth.shape))[: SIZE_THRESHOLD + 1]
    cmd = MaskEditCommand(topo, large_indices, [0] * len(large_indices))
    topo.apply_edit(cmd)

    head_entry = json.loads(json.loads(topo.tcm.history_file_path.read_text())["head"])
    assert "nc_path" in head_entry, "Expected nc_path key for large mask edit"
    assert "affected_indices" not in head_entry


@pytest.mark.benchmark
def test_undo_redo_on_large_domain(tx2_3v3_grid, tmp_path):
    """Undo/redo round-trip on a large domain should restore depth correctly."""
    topo = Topo(
        tx2_3v3_grid,
        min_depth=10.0,
        version_control_dir=str(tmp_path / "TopoLibrary"),
        git=True,
    )
    topo.set_flat(1000.0)
    topo.set_flat(500.0)

    assert float(topo._depth.mean()) == pytest.approx(500.0)
    topo.tcm.undo()
    assert float(topo._depth.mean()) == pytest.approx(1000.0)
    topo.tcm.redo()
    assert float(topo._depth.mean()) == pytest.approx(500.0)
