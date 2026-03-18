from mom6_bathy.topo import *


def test_topo_from_version_control(get_rect_topo):
    topo = get_rect_topo  # this topo has a version control directory
    topo_from_version_control = Topo.from_version_control(topo.domain_dir)
    assert topo_from_version_control.min_depth == topo.min_depth
    assert topo_from_version_control.depth.equals(topo.depth)


def test_topo_from_topo_file(get_rect_topo, tmp_path):
    topo = get_rect_topo
    j, i = 1, 1
    new_val = 12123
    old_val = topo.depth[j, i]
    command = DepthEditCommand(topo, [(j, i)], [new_val], old_values=[old_val])
    command()  # execute command quietly so that the topo version control doesn't control it (this way if I did from version control, it wouldn't pick up this change)
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


def test_send_entire_depth_change_to_tcm(get_rect_topo):
    topo = get_rect_topo
    old_depth = topo.depth.copy()
    new_depth = old_depth + 5.0
    topo.send_entire_depth_change_to_tcm(new_depth)
    assert (topo.depth == new_depth).all()
    topo.tcm.undo()
    assert (topo.depth == old_depth).all()
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    topo.send_entire_depth_change_to_tcm(new_depth, quietly=True)
    assert prev_hist == sum(
        1 for _ in topo.tcm.repo.iter_commits()
    )  # Assert no new commit


def test_erase_selected_basin(get_rect_topo):
    topo = get_rect_topo
    # Make a land barrier in the middle
    topo.depth[2, :] = 0  # horizontal land strip
    topo.depth[:, 2] = 0  # vertical land strip
    j, i = 1, 1
    old_depth = topo.depth.copy()

    topo.erase_selected_basin(j, i)
    # Since we have a land barrier, only bottom left should be erased to zero
    assert (topo.depth[:2, :2] == 0).all()
    # Other basins are untouched
    assert topo.depth[:2, 3:].equals(old_depth[:2, 3:])
    assert topo.depth[3:, :2].equals(old_depth[3:, :2])
    assert topo.depth[3:, 3:].equals(old_depth[3:, 3:])


def test_erase_disconnected_basin(get_rect_topo):
    topo = get_rect_topo
    # Make a land barrier in the middle
    topo.depth[2, :] = 0  # horizontal land strip
    topo.depth[:, 2] = 0  # vertical land strip
    j, i = 1, 1
    old_depth = topo.depth.copy()

    topo.erase_disconnected_basin(j, i)
    # Since we have a land barrier, only bottom left should be erased to zero
    assert topo.depth[:2, :2].equals(old_depth[:2, :2])

    # Other basins are erased
    assert (topo.depth[:2, 3:] == 0).all()
    assert (topo.depth[3:, :2] == 0).all()
    assert (topo.depth[3:, 3:] == 0).all()
