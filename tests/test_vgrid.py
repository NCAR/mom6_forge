import pytest
import socket
import os
import numpy as np
import xarray as xr

from mom6_bathy.vgrid import (
    VGrid,
    _cell_center_to_layer_thickness,
    _cell_interface_to_layer_thickness,
)


def _create_vgrid_dataset(
    layer_thickness,
    cell_center,
    cell_interface,
    var_names=("layer_thickness", "cell_center", "cell_interface"),
):
    """Create a dataset from the given vgrid information"""
    ds = xr.Dataset(
        {
            var_names[0]: (("z_test",), layer_thickness),
            var_names[1]: (("z_test",), cell_center),
            var_names[2]: (("z_i_test",), cell_interface),
        }
    )

    return ds


def test_default_init(get_realistic_vgrid_elements, get_faulty_vgrid_elements):
    """Test default init behavior for a VGrid object."""
    # Initialize realistic vgrid object
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    vgrid = VGrid(dz=layer_thickness)
    assert np.allclose(vgrid.dz, layer_thickness)
    assert np.allclose(vgrid.zl, cell_center)
    assert np.allclose(vgrid.zi, cell_interface)

    # Test all negative layer_thickness
    with pytest.raises(AssertionError):
        vgrid = VGrid(dz=(-1 * layer_thickness))

    # Test faulty data, should raise assertion error
    layer_thickness, cell_center, cell_interface = get_faulty_vgrid_elements
    with pytest.raises(AssertionError):
        vgrid = VGrid(dz=layer_thickness)


def test_cell_center_to_layer_thickness(
    get_realistic_vgrid_elements, get_faulty_vgrid_elements
):
    """Test conversion between cell center depths and layer thickness."""
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    dz = _cell_center_to_layer_thickness(cell_center)
    assert np.allclose(dz, layer_thickness)

    # Negative Depth Example, should automatically correct
    dz = _cell_center_to_layer_thickness((-1 * cell_center))
    assert np.allclose(dz, layer_thickness)

    # Assertion Error
    layer_thickness, cell_center, cell_interface = get_faulty_vgrid_elements
    with pytest.raises(AssertionError):
        dz = _cell_center_to_layer_thickness(cell_center)


def test_cell_interface_to_layer_thickness(
    get_realistic_vgrid_elements, get_faulty_vgrid_elements
):
    """Test cell interface depth to layer thickness conversion."""
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    dz = _cell_interface_to_layer_thickness(cell_interface)
    assert np.allclose(dz, layer_thickness)

    # Negative Depth Example, should automatically correct
    dz = _cell_interface_to_layer_thickness((-1 * cell_interface))
    assert np.allclose(dz, layer_thickness)

    # Assertion Error
    layer_thickness, cell_center, cell_interface = get_faulty_vgrid_elements
    with pytest.raises(AssertionError):
        dz = _cell_interface_to_layer_thickness(cell_interface)


def test_from_file_layer_thickness(get_realistic_vgrid_elements, tmp_path):
    """Test basic from_file functionality, loading from layer_thickness with different variable names."""

    # Create temp directory with pytest features
    tmp_dir = tmp_path / "vgrid_testing"
    tmp_dir.mkdir(exist_ok=True)

    # Default Variable Names/layer thickness
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    ds = _create_vgrid_dataset(
        layer_thickness, cell_center, cell_interface, var_names=("dz", "z", "zi")
    )

    with open(tmp_dir / "temp_vgrid.nc", "w") as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()

        vgrid = VGrid.from_file(filename=tmpfile.name)

        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.zl, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)

    # Different variable name for layer_thickness
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface)

    with open(tmp_dir / "temp_vgrid.nc", "w") as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()

        vgrid = VGrid.from_file(
            filename=tmpfile.name,
            variable_name="layer_thickness",
            variable_type="layer_thickness",
        )

        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.zl, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)


def test_from_file_cell_center(get_realistic_vgrid_elements, tmp_path):
    """Test loading from file with cell center depth information."""

    # Create temp directory with pytest features
    tmp_dir = tmp_path / "vgrid_testing"
    tmp_dir.mkdir(exist_ok=True)

    # Conventional variable name test
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    ds = _create_vgrid_dataset(
        layer_thickness, cell_center, cell_interface, var_names=("dz", "z", "zi")
    )

    with open(tmp_dir / "temp_vgrid.nc", "w") as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()

        vgrid = VGrid.from_file(
            filename=tmpfile.name, variable_name="z", variable_type="cell_center"
        )

        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.zl, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)

    # Negative depth from file
    ds = _create_vgrid_dataset(
        layer_thickness,
        (-1 * cell_center),
        (-1 * cell_interface),
        var_names=("dz", "z", "zi"),
    )

    with open(tmp_dir / "temp_vgrid.nc", "w") as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()

        vgrid = VGrid.from_file(
            filename=tmpfile.name, variable_name="z", variable_type="cell_center"
        )

        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.zl, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)


def test_from_file_cell_interface(get_realistic_vgrid_elements, tmp_path):
    """Test loading from file with cell interface depth data."""

    # Create temp directory with pytest features
    tmp_dir = tmp_path / "vgrid_testing"
    tmp_dir.mkdir(exist_ok=True)

    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    ds = _create_vgrid_dataset(
        layer_thickness, cell_center, cell_interface, var_names=("dz", "z", "zi")
    )

    with open(tmp_dir / "temp_vgrid.nc", "w") as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()

        vgrid = VGrid.from_file(
            filename=tmpfile.name, variable_name="zi", variable_type="cell_interface"
        )

        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.zl, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)

    # Example
    ds = _create_vgrid_dataset(
        layer_thickness,
        (-1 * cell_center),
        (-1 * cell_interface),
        var_names=("dz", "z", "zi"),
    )

    with open(tmp_dir / "temp_vgrid.nc", "w") as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()

        vgrid = VGrid.from_file(
            filename=tmpfile.name, variable_name="zi", variable_type="cell_interface"
        )

        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.zl, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)


@pytest.mark.parametrize(
    ("nlayers", "total_depth"),
    [
        (23, 2000),
        (50, 1000),
        (50, 3000),
    ],
)
def test_hyperbolictan_thickness_profile_equispaced(nlayers, total_depth):

    assert np.isclose(
        VGrid.hyperbolic(nlayers, total_depth, 1).dz,
        np.ones(nlayers) * total_depth / nlayers,
    ).all()


@pytest.mark.parametrize(
    ("nlayers", "ratio", "total_depth"),
    [
        (20, 1 / 3, 1000),
        (20, 2, 1000),
        (20, 10, 1000),
        (20, 2, 3000),
        (50, 1 / 3, 1000),
        (50, 2, 1000),
        (50, 10, 1000),
        (50, 2, 3000),
    ],
)
def test_hyperbolictan_thickness_profile_symmetric(nlayers, ratio, total_depth):
    assert np.isclose(
        VGrid.hyperbolic(nlayers, total_depth, ratio).dz,
        np.flip(VGrid.hyperbolic(nlayers, total_depth, 1 / ratio).dz),
    ).all()


if __name__ == "__main__":
    test_default_init()
    test_cell_center_to_layer_thickness()
    test_cell_interface_to_layer_thickness()
    test_from_file_layer_thickness()
    test_from_file_cell_center()
    test_from_file_cell_interface()
