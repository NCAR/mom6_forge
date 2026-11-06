"""The file I/O names and keywords renamed in 0.4.0 still work for one release,
with a DeprecationWarning, and pass everything on to the new ones."""

from types import SimpleNamespace

import pytest

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.vgrid import VGrid


@pytest.mark.parametrize(
    "cls, old, new",
    [
        (Grid, "from_supergrid", "from_supergrid_file"),
        (Grid, "from_esmf_mesh", "from_esmf_mesh_file"),
        (Grid, "write_supergrid", "write_supergrid_file"),
        (Topo, "from_esmf_mesh", "from_esmf_mesh_file"),
        (Topo, "write_topo", "write_topo_file"),
        (Topo, "write_cice_grid", "write_cice_grid_file"),
        (Topo, "write_scrip_grid", "write_scrip_grid_file"),
        (Topo, "write_esmf_mesh", "write_esmf_mesh_file"),
        (VGrid, "from_file", "from_vgrid_file"),
        (VGrid, "write", "write_vgrid_file"),
    ],
)
def test_old_method_names_call_the_new_ones(monkeypatch, cls, old, new):
    calls = []

    def new_method(*args):
        calls.append(args)
        return "result"

    with pytest.warns(DeprecationWarning, match=rf"{cls.__name__}\.{old}\(\)"):
        if isinstance(vars(cls)[old], classmethod):
            monkeypatch.setattr(cls, new, new_method)
            result = getattr(cls, old)("some.nc")
        else:
            self = SimpleNamespace(**{new: new_method})
            result = getattr(cls, old)(self, "some.nc")
    assert result == "result"
    assert calls[0][0] == "some.nc"


def test_old_keyword_names_still_work(get_rect_topo_without_vc, tmp_path):
    topo = get_rect_topo_without_vc
    grid = topo._grid
    grid.write_supergrid_file(tmp_path / "hgrid.nc")
    topo.write_topo_file(tmp_path / "topog.nc")

    with pytest.warns(DeprecationWarning, match="path="):
        sub = Grid.subgrid_from_supergrid(
            path=tmp_path / "hgrid.nc", llc=(7.5, 278.5), urc=(9.5, 281.5), name="sub"
        )
    assert sub.name == "sub"

    with pytest.warns(DeprecationWarning, match="topo_file_path="):
        read = Topo.from_topo_file(
            grid, topo_file_path=tmp_path / "topog.nc", version_control_dir=tmp_path
        )
    assert read.depth.equals(topo.depth)

    with pytest.warns(DeprecationWarning, match="topog_file_path="):
        read.set_depth_via_topog_file(topog_file_path=tmp_path / "topog.nc")
    assert read.depth.equals(topo.depth)

    with pytest.warns(DeprecationWarning, match="filename="):
        VGrid.uniform(nk=5, depth=100.0).write_z_file(filename=tmp_path / "z.nc")
    assert (tmp_path / "z.nc").exists()
