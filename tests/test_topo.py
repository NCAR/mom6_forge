import numpy as np
import xarray as xr
import pytest
from mom6_forge.topo import *
from mom6_forge.channel_width import ChannelWidth, ChannelWidthList


def test_topo_from_version_control(get_rect_topo_with_vc):
    topo = get_rect_topo_with_vc  # this topo has a version control directory
    topo_from_version_control = Topo.from_version_control(topo.domain_dir)
    assert topo_from_version_control.min_depth == topo.min_depth
    assert topo_from_version_control.depth.equals(topo.depth)


def test_topo_from_topo_file(get_rect_topo_with_vc, tmp_path):
    topo = get_rect_topo_with_vc
    j, i = 1, 1
    new_val = 12123
    old_val = topo.depth[j, i]
    command = DepthEditCommand(topo, [(j, i)], [new_val], old_values=[old_val])
    command()  # execute command skip_version_control so that the topo version control doesn't control it (this way if I did from version control, it wouldn't pick up this change)
    assert not Topo.from_version_control(topo.domain_dir).depth.equals(
        topo.depth
    )  # Assert command was quiet and not registered in version control
    topo_file_path = (
        tmp_path / "bleh.nc"
    )  # Would have this crazy depth because of the command in cell (1,1)
    topo.write_topo(topo_file_path)
    topo_from_file = Topo.from_topo_file(
        topo._grid,
        topo_file_path,
        topo.min_depth,
        version_control_dir=topo.domain_dir.parent,
    )
    assert topo_from_file.min_depth == topo.min_depth
    assert topo_from_file.depth.equals(topo.depth)
    assert topo_from_file.depth[j, i] == 12123


def test_send_entire_depth_change_to_tcm(get_rect_topo_with_vc):
    topo = get_rect_topo_with_vc
    old_depth = topo.depth.copy()
    new_depth = old_depth + 5.0
    topo.send_entire_depth_change_to_tcm(new_depth)
    assert (topo.depth == new_depth).all()
    topo.tcm.undo()
    assert (topo.depth == old_depth).all()
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    topo.send_entire_depth_change_to_tcm(new_depth, skip_version_control=True)
    assert prev_hist == sum(
        1 for _ in topo.tcm.repo.iter_commits()
    )  # Assert no new commit


def test_erase_selected_basin(get_rect_topo_without_vc):
    topo = get_rect_topo_without_vc
    # Make a land barrier in the middle
    topo.depth[2, :] = 0  # horizontal land strip
    topo.depth[:, 2] = 0  # vertical land strip
    j, i = 1, 1
    old_depth = topo.depth.copy()

    topo.erase_selected_basin(j, i)
    # Since we have a land barrier, only bottom left should be erased to zero
    assert (topo.masked_depth[:2, :2] == 0).all()
    # Other basins are untouched
    assert topo.masked_depth[:2, 3:].equals(old_depth[:2, 3:])
    assert topo.masked_depth[3:, :2].equals(old_depth[3:, :2])
    assert topo.masked_depth[3:, 3:].equals(old_depth[3:, 3:])


def test_erase_disconnected_basin(get_rect_topo_without_vc):
    topo = get_rect_topo_without_vc
    # Make a land barrier in the middle
    topo.depth[2, :] = 0  # horizontal land strip
    topo.depth[:, 2] = 0  # vertical land strip
    j, i = 1, 1
    old_depth = topo.depth.copy()

    topo.erase_disconnected_basin(j, i)
    # Since we have a land barrier, only bottom left should be erased to zero
    assert topo.masked_depth[:2, :2].equals(old_depth[:2, :2])

    # Other basins are erased
    assert (topo.masked_depth[:2, 3:] == 0).all()
    assert (topo.masked_depth[3:, :2] == 0).all()
    assert (topo.masked_depth[3:, 3:] == 0).all()


def test_keep_largest_basin(get_rect_topo_without_vc):
    topo = get_rect_topo_without_vc
    # Carve the domain into four basins of unequal size
    topo.depth[2, :] = 0  # horizontal land strip
    topo.depth[:, 2] = 0  # vertical land strip
    old_depth = topo.depth.copy()
    assert len(set(topo.basintmask.data.ravel().tolist())) == 5  # land + 4 basins

    topo.keep_largest_basin()

    # The upper right quadrant is the largest of the four, so it is the survivor
    assert topo.masked_depth[3:, 3:].equals(old_depth[3:, 3:])
    assert (topo.masked_depth[:2, :2] == 0).all()
    assert (topo.masked_depth[:2, 3:] == 0).all()
    assert (topo.masked_depth[3:, :2] == 0).all()
    assert len(set(topo.basintmask.data.ravel().tolist())) == 2  # land + 1 basin


def test_keep_largest_basin_uses_area_not_cell_count():
    """A basin of fewer but larger cells still wins over one of many small cells."""
    # 1 deg grid spanning the equator to 80N, so cell area shrinks sharply with j
    grid = Grid(
        resolution=1.0, xstart=0.0, lenx=10.0, ystart=0.0, leny=80.0, name="area_test"
    )
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)

    mask = np.zeros(topo.tmask.shape, dtype=int)
    mask[0, 0:3] = 1  # equatorial basin: 3 large cells
    mask[-1, 0:7] = 1  # high-latitude basin: 7 small cells
    tarea = grid.tarea.data
    assert tarea[0, 0:3].sum() > tarea[-1, 0:7].sum()
    topo.user_mask = mask

    topo.keep_largest_basin()

    assert (topo.tmask.data[0, 0:3] == 1).all()
    assert (topo.tmask.data[-1, 0:7] == 0).all()


def test_keep_largest_basin_is_noop_on_single_basin(get_rect_topo_with_vc):
    topo = get_rect_topo_with_vc  # flat 1000 m, one basin covering the domain
    old_depth = topo.depth.copy()
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())

    topo.keep_largest_basin()

    assert topo.depth.equals(old_depth)
    assert prev_hist == sum(1 for _ in topo.tcm.repo.iter_commits())


def test_keep_largest_basin_is_undoable(get_rect_topo_with_vc):
    topo = get_rect_topo_with_vc
    topo.depth[:, 2] = 0  # vertical land strip splits the domain in two
    old_depth = topo.masked_depth.copy()

    topo.keep_largest_basin()
    assert (topo.masked_depth[:, :2] == 0).all()

    topo.tcm.undo()
    assert topo.masked_depth.equals(old_depth)


def _topo_with_cyclic_seam_basin(grid):
    """Topo on a cyclic grid holding two basins.

    The first straddles the i=0 / i=nx-1 seam, split 2/2 across the wrap point;
    the second is 3 contiguous interior cells, so it is the larger of the two
    unless the seam halves are recognized as a single basin.
    """
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)
    nx = grid.nx

    mask = np.zeros(topo.tmask.shape, dtype=int)
    mask[0, :2] = 1  # seam basin, eastern half
    mask[0, -2:] = 1  # seam basin, western half
    mask[-1, nx // 2 : nx // 2 + 3] = 1  # interior basin
    topo.user_mask = mask
    return topo


def test_basintmask_merges_across_cyclic_seam(get_simple_global_grid):
    """A basin straddling the i=0 / i=nx-1 seam is labeled once, not twice."""
    topo = _topo_with_cyclic_seam_basin(get_simple_global_grid)
    nx = topo._grid.nx

    basins = topo.basintmask.data

    assert basins[0, 0] == basins[0, -1]  # the two halves share a label
    assert basins[0, 0] != basins[-1, nx // 2]  # the interior basin is still its own
    assert sorted(set(basins.ravel().tolist())) == [0, 1, 2]  # labels stay contiguous
    # Merged, the seam basin is the larger of the two, so it holds the highest label
    assert basins[0, 0] == basins.max()


def test_basintmask_gives_largest_basin_the_highest_label(get_rect_topo_without_vc):
    topo = get_rect_topo_without_vc
    topo.depth[2, :] = 0  # horizontal land strip
    topo.depth[:, 2] = 0  # vertical land strip

    basins = topo.basintmask.data
    tarea = topo._grid.tarea.data
    quadrants = [np.s_[:2, :2], np.s_[:2, 3:], np.s_[3:, :2], np.s_[3:, 3:]]

    assert sorted(set(basins.ravel().tolist())) == [0, 1, 2, 3, 4]  # land + 4 basins
    largest = max(quadrants, key=lambda q: tarea[q].sum())
    assert (basins[largest] == basins.max()).all()


def test_erase_disconnected_basin_keeps_whole_cyclic_basin(get_simple_global_grid):
    """Selecting one half of a seam-straddling basin must not erase the other."""
    topo = _topo_with_cyclic_seam_basin(get_simple_global_grid)
    nx = topo._grid.nx

    topo.erase_disconnected_basin(0, 0)  # select the eastern half of the seam basin

    assert (topo.tmask.data[0, :2] == 1).all()
    assert (topo.tmask.data[0, -2:] == 1).all()
    assert (topo.tmask.data[-1, nx // 2 : nx // 2 + 3] == 0).all()


def test_keep_largest_basin_merges_across_cyclic_seam(get_simple_global_grid):
    """The 4-cell seam basin outweighs the 3-cell interior one once merged."""
    topo = _topo_with_cyclic_seam_basin(get_simple_global_grid)
    nx = topo._grid.nx

    topo.keep_largest_basin()

    assert (topo.tmask.data[0, :2] == 1).all()
    assert (topo.tmask.data[0, -2:] == 1).all()
    assert (topo.tmask.data[-1, nx // 2 : nx // 2 + 3] == 0).all()


def test_erase_basin_on_land_cell_is_noop(get_rect_topo_without_vc):
    """Land has no basin, so neither erase method may treat it as one."""
    topo = get_rect_topo_without_vc
    topo.depth[:, 2] = 0  # vertical land strip
    old_depth = topo.masked_depth.copy()

    topo.erase_selected_basin(2, 1)  # (i, j) on the land strip
    topo.erase_disconnected_basin(2, 1)

    assert topo.masked_depth.equals(old_depth)
    assert topo.user_mask is None  # no edit was applied at all


def test_erase_disconnected_basin_edits_ocean_cells_only(
    get_rect_topo_without_vc, monkeypatch
):
    """The edit covers the erased basin, not every land cell in the domain."""
    topo = get_rect_topo_without_vc
    topo.depth[:, 2] = 0  # land strip splits the domain in two

    captured = []
    monkeypatch.setattr(
        Topo, "apply_edit", lambda self, cmd, **kwargs: captured.append(cmd)
    )
    topo.erase_disconnected_basin(30, 1)  # keep the eastern basin

    (cmd,) = captured
    tmask = topo.tmask.data
    assert all(tmask[j, i] == 1 for j, i in cmd.affected_indices)
    # The western basin is the 2 columns west of the strip, and nothing else
    assert len(cmd.affected_indices) == topo._grid.ny * 2


def test_topo_no_git(get_rect_topo_without_vc):
    topo = get_rect_topo_without_vc
    assert topo.tcm is None
    # Make an edit
    j, i = 1, 1
    new_val = 12123
    old_val = topo.depth[j, i]
    command = DepthEditCommand(
        topo, [(j, i)], [new_val], old_values=[old_val]
    )  # This command should still work even without version control, but it just won't be registered in version control
    topo.apply_edit(command)
    assert topo.depth[j, i] == new_val


# ---------------------------------------------------------------------------
# Topo.from_esmf_mesh tests
# ---------------------------------------------------------------------------


def test_topo_from_esmf_mesh_roundtrip(get_rect_topo_with_vc, tmp_path):
    topo = get_rect_topo_with_vc
    # Stamp some land cells into the mask before writing
    land_mask = topo.tmask.values.copy()
    land_mask[:2, :3] = 0
    topo._user_mask = xr.DataArray(land_mask, dims=["ny", "nx"])
    mesh_path = str(tmp_path / "test.nc")
    topo.write_esmf_mesh(mesh_path)
    topo2 = Topo.from_esmf_mesh(mesh_path, git=False)
    assert topo2.tmask.shape == topo.tmask.shape
    assert topo2._grid.nx == topo._grid.nx
    assert topo2._grid.ny == topo._grid.ny
    np.testing.assert_array_equal(topo2.tmask.values, land_mask)


def test_topo_from_esmf_mesh_accepts_dataset(get_rect_topo_with_vc, tmp_path):
    topo = get_rect_topo_with_vc
    mesh_path = str(tmp_path / "test.nc")
    topo.write_esmf_mesh(mesh_path)
    ds = xr.open_dataset(mesh_path)
    topo2 = Topo.from_esmf_mesh(ds, git=False)
    assert topo2.tmask.shape == topo.tmask.shape


def test_topo_from_esmf_mesh_raises_without_mask(get_rect_topo_with_vc, tmp_path):
    topo = get_rect_topo_with_vc
    mesh_path = str(tmp_path / "with_mask.nc")
    no_mask_path = str(tmp_path / "no_mask.nc")
    topo._grid.supergrid.to_esmf_mesh(mesh_path, mask="all_unmasked")
    # Simulate a mesh file with no elementMask, e.g. from an external tool
    with xr.open_dataset(mesh_path) as ds:
        ds.drop_vars("elementMask").load().to_netcdf(no_mask_path)
    with pytest.raises(ValueError, match="elementMask"):
        Topo.from_esmf_mesh(no_mask_path, git=False)


def test_topo_channel_widths_none(get_rect_grid):
    """channel_widths=None creates an empty ChannelWidthList."""
    topo = Topo(get_rect_grid, min_depth=0, git=False)
    assert isinstance(topo.channel_widths, ChannelWidthList)
    assert len(topo.channel_widths.get_all()) == 0


def test_topo_channel_widths_object(get_rect_grid):
    """Passing a ChannelWidthList object attaches it directly."""
    cwl = ChannelWidthList()
    cwl.add(
        ChannelWidth(
            component="U_width",
            lon1=-6.5,
            lon2=-4.75,
            lat1=35.6,
            lat2=36.3,
            width=12000.0,
            place="St. of Gibralter",
        )
    )
    topo = Topo(get_rect_grid, min_depth=0, git=False, channel_widths=cwl)
    assert topo.channel_widths is cwl
    assert len(topo.channel_widths.get_all()) == 1


def test_topo_channel_widths_filepath(get_rect_grid, tmp_path):
    """Passing a filepath loads ChannelWidthList from disk."""
    cwl = ChannelWidthList()
    cwl.add(
        ChannelWidth(
            component="U_width",
            lon1=-6.5,
            lon2=-4.75,
            lat1=35.6,
            lat2=36.3,
            width=12000.0,
            place="St. of Gibralter",
        )
    )
    filepath = tmp_path / "channels.txt"
    cwl.write(filepath)

    topo = Topo(get_rect_grid, min_depth=0, git=False, channel_widths=filepath)
    loaded = topo.channel_widths.get_all()
    assert len(loaded) == 1
    assert loaded[0].component == "U_width"
    assert loaded[0].place == "St. of Gibralter"
