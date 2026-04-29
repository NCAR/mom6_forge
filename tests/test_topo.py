import pytest

from mom6_forge.topo import *


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


# =============================================================================
# Mask property tests (umask / vmask / qmask / supergridmask / basintmask)
# =============================================================================


def test_masks_pure_ocean(get_rect_topo):
    """An all-ocean topo (depth > min_depth everywhere) should yield all-ocean masks,
    with the exception of the 4 Q corners which are always set to land."""
    import numpy as np

    topo = get_rect_topo  # flat 1000 m, min_depth=0
    assert (topo.tmask.data == 1).all()
    assert (topo.umask.data == 1).all()
    assert (topo.vmask.data == 1).all()

    q = topo.qmask.data
    # Corners always land
    assert q[0, 0] == 0 and q[0, -1] == 0 and q[-1, 0] == 0 and q[-1, -1] == 0
    # Interior should all be ocean
    assert (q[1:-1, 1:-1] == 1).all()

    # basintmask: one connected basin (label 1) everywhere
    b = topo.basintmask.data
    assert b.max() == 1
    assert (b == 1).all()

    # supergridmask: same shape as supergrid nodes; corners are always land,
    # interior should be all ocean.
    sg = topo.supergridmask
    assert sg.shape == topo._grid._supergrid.x.shape
    assert sg.data[0, 0] == 0 and sg.data[0, -1] == 0
    assert sg.data[-1, 0] == 0 and sg.data[-1, -1] == 0
    assert (sg.data[1:-1, 1:-1] == 1).all()


def test_masks_with_land_strip(get_rect_topo):
    """Insert a land strip and verify masks propagate correctly."""
    topo = get_rect_topo
    # Make row 2 fully land (depth == 0 is land since min_depth=0)
    topo.depth[2, :] = 0

    tmask = topo.tmask.data
    assert (tmask[2, :] == 0).all()
    assert (tmask[0, :] == 1).all()

    # umask/vmask must reflect the land strip
    vmask = topo.vmask.data
    # Any v-point whose top OR bottom T cell is land becomes land.
    # v has shape (ny+1, nx); rows 2 and 3 are adjacent to the land strip row 2.
    assert (vmask[2, :] == 0).all()
    assert (vmask[3, :] == 0).all()

    # qmask interior land strip check
    qmask = topo.qmask.data
    # qmask row 2 & 3 should be zero (adjacent to land strip)
    assert (qmask[2, :] == 0).all()


def test_point_is_ocean(get_rect_topo):
    """point_is_ocean looks up matching supergrid nodes and returns their ocean-mask values."""
    topo = get_rect_topo
    sg_x = topo._grid._supergrid.x
    sg_y = topo._grid._supergrid.y

    # Pick two existing supergrid nodes.
    pt1 = (float(sg_x[0, 0]), float(sg_y[0, 0]))
    pt2 = (float(sg_x[5, 5]), float(sg_y[5, 5]))
    result = topo.point_is_ocean([pt1[0], pt2[0]], [pt1[1], pt2[1]])
    assert len(result) == 2
    # For an all-ocean grid, interior supergrid nodes should be ocean.
    assert result[1] == 1


# =============================================================================
# Shape-setter tests: set_spoon / set_bowl
# =============================================================================


def test_set_spoon_produces_valid_bathymetry(get_rect_topo):
    topo = get_rect_topo
    topo.set_spoon(max_depth=4000.0, dedge=100.0)
    # depth should be finite and have some variation
    import numpy as np

    d = topo.depth.data
    assert np.all(np.isfinite(d))
    assert d.min() < d.max()  # not flat


def test_set_bowl_produces_valid_bathymetry(get_rect_topo):
    topo = get_rect_topo
    topo.set_bowl(max_depth=4000.0, dedge=100.0)
    import numpy as np

    d = topo.depth.data
    assert np.all(np.isfinite(d))
    assert d.min() < d.max()


# =============================================================================
# apply_ridge test
# =============================================================================


def test_apply_ridge_shallows_ocean(get_rect_topo):
    """apply_ridge should decrease depth in the affected latitude band."""
    import numpy as np

    topo = get_rect_topo
    before = topo.depth.data.copy()

    # Center the ridge at the middle of the grid in longitude.
    center_lon = float(topo._grid.tlon[0, topo._grid.nx // 2])
    width = float(topo._grid.tlon[0, -1] - topo._grid.tlon[0, 0]) / 4
    topo.apply_ridge(height=500.0, width=width, lon=center_lon, ilat=(0, 3))

    after = topo.depth.data
    # Rows 0..2 should be (mostly) shallower in the ridge band
    assert (after[0:3] <= before[0:3]).all()
    # Outside the ilat band is untouched
    assert np.allclose(after[3:], before[3:])


# =============================================================================
# Writer tests: gen_topo_ds / write_topo / write_cice_grid / write_scrip_grid
# / write_esmf_mesh
# =============================================================================


def test_gen_topo_ds_contents(get_rect_topo):
    """gen_topo_ds should produce x, y, mask, depth variables with matching shapes."""
    topo = get_rect_topo
    ds = topo.gen_topo_ds(title="my_test_topo")
    assert ds.attrs["title"] == "my_test_topo"
    assert ds.attrs["min_depth"] == topo.min_depth
    assert ds.attrs["max_depth"] == topo.max_depth
    for var in ("x", "y", "mask", "depth"):
        assert var in ds
        assert ds[var].dims == ("ny", "nx")
    assert ds.sizes["ny"] == topo._grid.ny
    assert ds.sizes["nx"] == topo._grid.nx


def test_gen_topo_ds_default_title(get_rect_topo):
    """When no title is supplied, a default is used."""
    topo = get_rect_topo
    ds = topo.gen_topo_ds()
    assert ds.attrs["title"] == "MOM6 topography file"


def test_write_topo_roundtrip(get_rect_topo, tmp_path):
    import xarray as xr

    topo = get_rect_topo
    out = tmp_path / "topo.nc"
    topo.write_topo(out, title="roundtrip")
    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        assert ds.attrs["title"] == "roundtrip"
        assert "depth" in ds and "mask" in ds
        assert ds.sizes["ny"] == topo._grid.ny
        assert ds.sizes["nx"] == topo._grid.nx
    finally:
        ds.close()


def test_write_cice_grid_writes_expected_vars(get_rect_topo, tmp_path):
    import xarray as xr

    topo = get_rect_topo
    out = tmp_path / "cice_grid.nc"
    topo.write_cice_grid(out)
    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        for var in (
            "ulat",
            "ulon",
            "tlat",
            "tlon",
            "htn",
            "hte",
            "angle",
            "anglet",
            "kmt",
        ):
            assert var in ds, f"{var} missing from CICE grid file"
        assert ds.attrs["title"] == "CICE grid file"
    finally:
        ds.close()


def test_write_scrip_grid_writes_expected_vars(get_rect_topo, tmp_path):
    import xarray as xr

    topo = get_rect_topo
    out = tmp_path / "scrip.nc"
    topo.write_scrip_grid(out, title="scrip_test")
    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        assert ds.attrs["Conventions"] == "SCRIP"
        assert ds.attrs["title"] == "scrip_test"
        for var in (
            "grid_dims",
            "grid_center_lat",
            "grid_center_lon",
            "grid_imask",
            "grid_corner_lat",
            "grid_corner_lon",
            "grid_area",
        ):
            assert var in ds
        assert ds.sizes["grid_size"] == topo._grid.nx * topo._grid.ny
        assert ds.sizes["grid_corners"] == 4
    finally:
        ds.close()


def test_write_esmf_mesh_writes_expected_vars(get_rect_topo, tmp_path):
    """Non-cyclic rectangular grid path through write_esmf_mesh."""
    import xarray as xr

    topo = get_rect_topo
    out = tmp_path / "esmf_mesh.nc"
    topo.write_esmf_mesh(out, title="esmf_test")
    assert out.exists()
    ds = xr.open_dataset(out)
    try:
        assert ds.attrs["title"] == "esmf_test"
        assert ds.attrs["gridType"] == "unstructured mesh"
        for var in (
            "centerCoords",
            "numElementConn",
            "elementArea",
            "elementMask",
        ):
            assert var in ds, f"{var} missing from ESMF mesh file"
        assert ds.sizes["elementCount"] == topo._grid.nx * topo._grid.ny
    finally:
        ds.close()


# =============================================================================
# set_depth_via_topog_file: error paths & subregion path
# =============================================================================


def test_set_depth_via_topog_file_missing_file_raises(get_rect_topo, tmp_path):
    topo = get_rect_topo
    missing = tmp_path / "no_such_file.nc"
    with pytest.raises(AssertionError):
        topo.set_depth_via_topog_file(str(missing))


def test_set_depth_via_topog_file_wrong_varname_raises(get_rect_topo, tmp_path):
    import numpy as np
    import xarray as xr

    topo = get_rect_topo
    path = tmp_path / "bad_topo.nc"
    xr.Dataset(
        {"not_depth": (("ny", "nx"), np.zeros((topo._grid.ny, topo._grid.nx)))}
    ).to_netcdf(path)
    with pytest.raises(AssertionError):
        topo.set_depth_via_topog_file(str(path), varname="depth")


def test_set_depth_via_topog_file_smaller_raises(get_rect_topo, tmp_path):
    import numpy as np
    import xarray as xr

    topo = get_rect_topo
    path = tmp_path / "small_topo.nc"
    xr.Dataset({"depth": (("ny", "nx"), np.zeros((2, 2)))}).to_netcdf(path)
    with pytest.raises(ValueError):
        topo.set_depth_via_topog_file(str(path))


# =============================================================================
# send_entire_depth_change_to_tcm quiet path
# =============================================================================


def test_send_entire_depth_change_quietly_bypasses_tcm(get_rect_topo):
    """quietly=True should directly apply the change without a new TCM commit."""
    topo = get_rect_topo
    prev_commits = sum(1 for _ in topo.tcm.repo.iter_commits())

    new_depth = topo.depth.copy() + 42.0
    topo.send_entire_depth_change_to_tcm(new_depth, quietly=True)

    assert (topo.depth == new_depth).all()
    new_commits = sum(1 for _ in topo.tcm.repo.iter_commits())
    assert new_commits == prev_commits  # no new commit


# =============================================================================
# save() sanity check (delegates to tcm.save)
# =============================================================================


def test_save_delegates_to_tcm(get_rect_topo, monkeypatch):
    topo = get_rect_topo
    called = {"n": 0}
    monkeypatch.setattr(
        topo.tcm, "save", lambda: called.__setitem__("n", called["n"] + 1)
    )
    topo.save()
    assert called["n"] == 1
