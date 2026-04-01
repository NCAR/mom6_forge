import os
from datetime import datetime
import numpy as np
import xarray as xr

from typing import Literal


class VGrid:
    """
    Vertical grid class.

    Attributes
    ----------
    dz: np.ndarray
        Array of vertical grid spacings
    nk: int
        Number of vertical levels
    depth: float
        Total depth of the vertical grid
    z: np.ndarray
        Array of layer center depths
    """

    def __init__(self, dz: np.ndarray, name=None):
        """Create a vertical grid.

        Parameters
        ----------
        dz: np.ndarray
            Array of vertical grid spacings (meters)
        """

        assert np.all(dz > 0), "Layer thickness cannot be zero or negative."

        self.dz = dz
        self.name = name

    @property
    def nk(self):
        """Number of vertical levels."""
        return len(self.dz)

    @property
    def depth(self):
        """Total depth of the water column in meters."""
        return np.sum(self.dz)

    @property
    def zl(self):
        """Array of vertical grid cell center depths (meters)."""
        return np.cumsum(self.dz) - 0.5 * self.dz

    @property
    def zi(self):
        """Array of vertical grid cell interface depths (meters) - size nk+1.
        Assumes there is a surface interface at 0 meters."""
        return np.insert(np.cumsum(self.dz), 0, 0)

    @classmethod
    def uniform(cls, nk: int, depth: float, name: str = None):
        """Create a uniform vertical grid instance.

        Parameters
        ----------
        nk: int
            Number of vertical levels
        depth: float
            Total depth of the water column (meters)
        name: str, optional
            Name of the vertical grid
        """

        assert nk > 1, "Number of layers must be greater than 1"
        assert depth > 0, "Depth must be greater than 0"

        # create a uniform vertical grid
        dz = np.ones(nk) * depth / nk

        # update the bottom layer thickness to ensure the total depth is correct
        dz[-1] = depth - np.sum(dz[:-1])

        return cls(dz, name)

    @classmethod
    def hyperbolic(cls, nk: int, depth: float, ratio: float, name: str = None):
        """Create a hyperbolic vertical grid instance. (Adapted from regional-mom6)

        Parameters
        ----------
        nk: int
            Number of vertical levels
        ratio: float
            Target ratio of top to bottom layer thicknesses
        depth: float
            Total depth of the water column (meters)
        name: str, optional
            Name of the vertical grid
        """

        assert nk > 1, "Number of layers must be greater than 1"
        assert ratio > 0, "Ratio must be greater than 0"
        assert depth > 0, "Depth must be greater than 0"

        dz0 = 2 * depth / (nk * (1 + ratio))
        dzbot = ratio * dz0
        dz = dz0 + 0.5 * (dzbot - dz0) * (
            1 + np.tanh(2 * np.pi * (np.arange(nk) / (nk - 1) - 1 / 2))
        )

        # update the bottom layer thickness to ensure the total depth is correct
        dz[-1] = depth - np.sum(dz[:-1])

        return cls(dz, name=name)

    @classmethod
    def from_file(
        cls,
        filename: str,
        variable_name: str = "dz",
        variable_type: Literal[
            "layer_thickness", "cell_center", "cell_interface"
        ] = "layer_thickness",
        name: str = None,
    ):
        """Create a vertical grid from an existing vertical grid file.

        Parameters
        ----------
        filename: str
            Name of the NetCDF file containing the vertical grid
        coordinate_name: str, default: "dz"
            Name of the variable to access within the given NetCDF file to get
            a vertical coordinate.
        coordinate_type: {"layer_thickness", "cell_center", "cell_interface"}, default: "layer_thickness"
            Type of vertical coordinate information given by the file and variable.

            - layer_thickness: specifies the thickness of each layer in the vertical grid. Size of n for n layers.
            - cell_center: gives the depth at the center of each vertical cell. Size of n for n layers.
            - cell_interface: gives the depth at the interface between each vertical cell. Size of n+1 for n layers.
        """

        assert filename.endswith(".nc"), f"File {filename} is not a NetCDF file"
        assert os.path.exists(filename), f"File {filename} does not exist"

        valid_coordinate_types = {"layer_thickness", "cell_center", "cell_interface"}
        assert (
            variable_type in valid_coordinate_types
        ), f"Coordinate type {variable_type} is not a valid option from {valid_coordinate_types}"

        ds = xr.open_dataset(filename)
        assert (
            variable_name in ds
        ), f"File {filename} does not contain a '{variable_name}' variable"

        if variable_type == "layer_thickness":
            dz = ds[variable_name].values
        if variable_type == "cell_center":
            dz = _cell_center_to_layer_thickness(ds[variable_name].values)
        if variable_type == "cell_interface":
            dz = _cell_interface_to_layer_thickness(ds[variable_name].values)

        assert np.all(dz > 0), "Layer thickness cannot be zero."

        if name is None:
            name = ds.attrs.get("name", None)

        return cls(dz, name)

    def write(self, filename: str, message: str = None, author: str = None):
        """Write the vertical grid (in thickness) to a NetCDF file.

        Parameters
        ----------
        filename: str
            Name of the NetCDF file to write
        """
        dz0 = float(self.dz[0])
        dzbot = float(self.dz[-1])
        ratio = dzbot / dz0 if dz0 != 0 else 1.0

        ds = xr.Dataset(
            data_vars={
                "dz": (
                    "z",
                    self.dz,
                    {
                        "units": "meters",
                        "long_name": "nominal thickness of layer",
                        "valid_min": np.min(self.dz),
                        "valid_max": np.max(self.dz),
                    },
                )
            },
        )

        ds.attrs["title"] = f"Vertical grid for MOM6 simulation"
        ds.attrs["maximum_depth"] = self.depth
        if self.name is not None:
            ds.attrs["name"] = self.name
        ds.attrs["top_bottom_ratio"] = float(ratio)
        ds.attrs["history"] = f"Created on {datetime.now()}"
        if message:
            ds.attrs["message"] = message
        if author:
            ds.attrs["author"] = author
        ds.to_netcdf(
            filename,
            format="NETCDF3_64BIT",
        )
        return ds

    def write_z_file(self, filename: str):
        """Write the vertical grid to a NetCDF file with interface and midpoints depths

        Parameters
        ----------
        filename: str
            Name of the NetCDF file to write
        """
        ds = xr.Dataset(
            data_vars={
                "zi": (
                    "zi",
                    self.zi,
                    {
                        "units": "meters",
                        "long_name": "Cell Interface Depths",
                        "valid_min": np.min(self.zi),
                        "valid_max": np.max(self.zi),
                    },
                ),
                "zl": (
                    "zl",
                    self.zl,
                    {
                        "units": "meters",
                        "long_name": "Cell Center Depths",
                        "valid_min": np.min(self.zl),
                        "valid_max": np.max(self.zl),
                    },
                ),
            },
        )

        ds.attrs["title"] = f"Vertical grid for MOM6 simulation"
        ds.attrs["maximum_depth"] = self.depth
        ds.attrs["history"] = f"Created on {datetime.now()}"
        ds.to_netcdf(
            filename,
            format="NETCDF3_64BIT",
        )
        return ds


def _cell_center_to_layer_thickness(cell_centers: np.ndarray):
    """Convert depth of cell centers to layer thickness.

    Parameters
    ----------
    cell_centers: np.ndarray
        Depth of cell centers in meters. Needs to be strictly monotonic.
        We will forcibly correct values to be positive and increasing with index.
    """
    # Check uniform sign of values
    check_sign = np.all(cell_centers > 0) or np.all(cell_centers < 0)
    assert check_sign, "Cell center depths must be all positive or all negative."

    # Convert to all positive values, sort, and check monotonicity
    # must be strictly monotonic, meaning no repeating values.
    cell_centers = np.abs(cell_centers)

    monotonic = np.all(np.diff(cell_centers) > 0)
    assert monotonic, "Cell center depths must be strictly monotonic."

    # put in increasing order
    cell_centers = np.sort(cell_centers)

    # Convert from cell centers to layer thickness.
    temp_data = np.diff(cell_centers, prepend=0)
    for i in range(0, temp_data.size - 1):
        temp_data[i + 1] = temp_data[i + 1] - temp_data[i]

    layer_thickness = temp_data * 2

    return layer_thickness


def _cell_interface_to_layer_thickness(cell_interfaces: np.ndarray):
    """Convert cell interfaces to layer thickness.

    Note: number of cell interfaces = number of layers + 1

    Parameters
    ----------
    cell_interfaces: np.ndarray
        Depth of cell interfaces in meters. Needs to be strictly monotonic and surface interface must be at 0 meters. We will forcibly correct values to be positive and increasing with index.
    """
    # Check uniform sign of values
    check_sign = np.all(cell_interfaces >= 0) or np.all(cell_interfaces <= 0)
    assert (
        check_sign
    ), "Cell interface depths must be all positive or all negative (one value, the surface interface, can be 0)."

    # Convert to all positive values and check monotonicity
    # must be strictly monotonic, meaning no repeating values.
    cell_interfaces = np.abs(cell_interfaces)

    monotonic = np.all(np.diff(cell_interfaces) > 0)
    assert monotonic, "Cell interface depths must be strictly monotonic."

    # put in increasing order
    cell_interfaces = np.sort(cell_interfaces)

    # Convert cell interface depths to layer thickness
    layer_thickness = np.diff(cell_interfaces)

    return layer_thickness
