"""
Smoke tests for the mom6_forge command line interface (mom6_forge/cli.py).

These only check that each command runs and writes its file; correctness of
the grids themselves is covered by the Grid/Topo/VGrid tests.
"""

import pytest
from click.testing import CliRunner

from mom6_forge.cli import cli


def _run(args):
    result = CliRunner().invoke(cli, args.split())
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def hgrid_path(tmp_path):
    path = tmp_path / "ocean_hgrid.nc"
    _run(
        "grid --lenx 4 --leny 3 --resolution 0.5 --xstart 278 --ystart 7 "
        f"write_supergrid --path {path}"
    )
    return path


def test_grid(hgrid_path):
    assert hgrid_path.exists()


def test_topo(hgrid_path, tmp_path):
    path = tmp_path / "ocean_topog.nc"
    _run(
        f"topo --grid {hgrid_path} --min-depth 10 "
        f"set_flat --D 500 write_topo --file-path {path}"
    )
    assert path.exists()


@pytest.mark.parametrize(
    "start",
    ["uniform --nk 10 --depth 500", "hyperbolic --nk 10 --depth 500 --ratio 0.2"],
)
def test_vgrid(tmp_path, start):
    path = tmp_path / "ocean_vgrid.nc"
    _run(f"vgrid {start} write --filename {path}")
    assert path.exists()


def test_chain_must_start_with_constructor(tmp_path):
    result = CliRunner().invoke(cli, ["vgrid", "write", "--filename", "v.nc"])
    assert result.exit_code != 0
    assert "Start the chain with one of" in result.output
