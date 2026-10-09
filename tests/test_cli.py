"""
Smoke tests for the mom6_forge command line interface (mom6_forge/cli.py).

These only check that each subcommand runs and writes its file; correctness
of the grids themselves is covered by the Grid/Topo/VGrid tests.
"""

import pytest

from mom6_forge.cli import main


@pytest.fixture
def hgrid_path(tmp_path):
    path = tmp_path / "ocean_hgrid.nc"
    main(["grid", "--lenx", "4", "--leny", "3", "--resolution", "0.5"]
         + ["--xstart", "278", "--ystart", "7", "-o", str(path)])  # fmt: skip
    return path


def test_grid(hgrid_path):
    assert hgrid_path.exists()


def test_topo(hgrid_path, tmp_path):
    path = tmp_path / "ocean_topog.nc"
    main(["topo", "--grid", str(hgrid_path), "--min-depth", "10"]
         + ["--flat", "500", "-o", str(path)])  # fmt: skip
    assert path.exists()


@pytest.mark.parametrize(
    "extra", [["--type", "uniform"], ["--type", "hyperbolic", "--ratio", "0.2"]]
)
def test_vgrid(tmp_path, extra):
    path = tmp_path / "ocean_vgrid.nc"
    main(["vgrid", "--nk", "10", "--depth", "500", "-o", str(path)] + extra)
    assert path.exists()


def test_refuses_overwrite(tmp_path):
    path = tmp_path / "ocean_vgrid.nc"
    path.write_text("keep me")
    with pytest.raises(SystemExit):
        main(["vgrid", "--nk", "5", "--depth", "100", "-o", str(path)])
    assert path.read_text() == "keep me"
