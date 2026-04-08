"""
This module defines MOM6-style supergrid classes and associated utilities. It sits underneath the mom6_forge.grid class and fills the roll of calculating the grid geometry: angle_dx, area, dx, dy, x, and y.

Classes defined here:
- SupergridBase: Base class defining the MOM6-style supergrid interface.
- UniformSphericalSupergrid: MOM6-style supergrid with constant-degree spacing (lon/lat grid).
- RectilinearCartesianSupergrid: MOM6-style supergrid with (as close to) uniform Cartesian spacing (still a lat/lon grid).

The code for these classes does not originally come from mom6_forge, but was adapted: UniformSphericalSupergrid by Mathew Harrison in MIDAS (https://github.com/mjharriso/MIDAS) and RectilinearCartesianSupergrid by Ashley Barnes in regional_mom6 (https://github.com/COSIMA/regional-mom6).
"""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional
from mom6_forge.utils import normalize_deg
from mom6_forge.supergrid_metric_helpers import (
    quadrilateral_areas,
    mom6_angle_calculation_method,
)


class SupergridBase:
    """Base class defining the MOM6-style supergrid interface."""

    R = 6.371e6  # mean radius of the Earth (IUGG), in metres

    @property
    def is_cyclic_x(self):
        return np.allclose(
            normalize_deg(self.x[:, 0]),
            normalize_deg(self.x[:, -1]),
            rtol=1e-5,
        )

    @property
    def lenx(self):
        return self.x.max() - self.x.min()

    @property
    def leny(self):
        return self.y.max() - self.y.min()

    def __init__(self, x, y, dx, dy, area, angle_dx, axis_units):
        """
        Initialize a generic supergrid.

        Parameters
        ----------
        x, y : 2D arrays
            Grid point longitudes and latitudes (or x/y positions).
        dx, dy : 2D arrays
            Cell widths in x and y directions.
        area : 2D array
            Grid cell areas.
        angle : 2D array
            Local grid angle relative to east.
        axis_units : str
            Units of x and y (e.g. "degrees" or "meters").
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.area = area
        self.angle_dx = angle_dx
        self.axis_units = axis_units

    @staticmethod
    def _calc_dx_dy(x, y):
        """Compute supergrid dx and dy from coordinate arrays.

        Parameters
        ----------
        x, y : 2D arrays
            Supergrid longitude and latitude in degrees, shape (2*ny+1, 2*nx+1).

        Returns
        -------
        dx : 2D array, shape (2*ny+1, 2*nx)
            Arc lengths between horizontally adjacent nodes, in metres.
        dy : 2D array, shape (2*ny, 2*nx+1)
            Arc lengths between vertically adjacent nodes, in metres.
        """
        dx = (
            SupergridBase.R
            * np.cos(np.deg2rad(y[:, :-1]))
            * np.deg2rad(np.diff(x, axis=1))
        )
        dy = SupergridBase.R * np.deg2rad(np.diff(y, axis=0))
        return dx, dy

    def summary(self):
        """Print a short summary of the grid geometry (shape and dx/dy ranges)."""
        print(
            f"{self.__class__.__name__}: shape={self.x.shape}, "
            f"dx=({self.dx.min()}–{self.dx.max()}), "
            f"dy=({self.dy.min()}–{self.dy.max()})"
        )

    def to_ds(self, name=None, author: Optional[str] = None) -> xr.Dataset:
        """
        Export the supergrid to an xarray.Dataset compatible with MOM6.

        Parameters
        ----------
        author : str, optional
            If provided, stored as metadata in the output dataset.
        """
        ds = xr.Dataset()

        # ---- Metadata ----
        ds.attrs["type"] = "MOM6 supergrid"
        if name is not None:
            ds.attrs["name"] = name
        ds.attrs["Created"] = datetime.now().isoformat()
        if author:
            ds.attrs["Author"] = author

        # ---- Data variables ----
        ds["y"] = xr.DataArray(
            self.y, dims=["nyp", "nxp"], attrs={"units": self.axis_units}
        )
        ds["x"] = xr.DataArray(
            self.x, dims=["nyp", "nxp"], attrs={"units": self.axis_units}
        )
        ds["dy"] = xr.DataArray(self.dy, dims=["ny", "nxp"], attrs={"units": "meters"})
        ds["dx"] = xr.DataArray(self.dx, dims=["nyp", "nx"], attrs={"units": "meters"})
        ds["area"] = xr.DataArray(self.area, dims=["ny", "nx"], attrs={"units": "m2"})
        ds["angle_dx"] = xr.DataArray(
            self.angle_dx, dims=["nyp", "nxp"], attrs={"units": "radians"}
        )

        return ds

    def calculate_supergrid_rotation_angles_using_expanded_supergrid_method(
        self,
    ) -> xr.Dataset:
        """
        Calculate the ``angle_dx`` (in degrees) from the true ``x`` direction (parallel to latitude)
        counter-clockwise and return as a dataarray.

        Parameters
        ----------
        supergrid: xr.Dataset
            The supergrid dataset

        Returns
        -------
        xr.DataArray
            The t-point angles
        """
        # Get expanded (pseudo) grid
        expanded_supergrid = self._create_expanded_supergrid()

        point = xr.Dataset(
            {
                "x": (["nyp", "nxp"], self.x),
                "y": (["nyp", "nxp"], self.y),
            }
        )
        return mom6_angle_calculation_method(
            expanded_supergrid.x.max() - expanded_supergrid.x.min(),
            expanded_supergrid.isel(nyp=slice(2, None), nxp=slice(0, -2)),
            expanded_supergrid.isel(nyp=slice(2, None), nxp=slice(2, None)),
            expanded_supergrid.isel(nyp=slice(0, -2), nxp=slice(0, -2)),
            expanded_supergrid.isel(nyp=slice(0, -2), nxp=slice(2, None)),
            point,
        ).values

    def _create_expanded_supergrid(self, expansion_width=1) -> xr.Dataset:
        """
        Adds an additional boundary to the supergrid to allow for the calculation of the ``angle_dx`` for the boundary points using :func:`~mom6_angle_calculation_method`.
        """
        if expansion_width != 1:
            raise NotImplementedError("Only expansion_width = 1 is supported")

        ny, nx = self.x.shape
        pseudo_supergrid_x = np.full((ny + 2, nx + 2), np.nan)
        pseudo_supergrid_y = np.full((ny + 2, nx + 2), np.nan)

        ## Fill Boundaries
        pseudo_supergrid_x[1:-1, 1:-1] = self.x
        pseudo_supergrid_x[0, 1:-1] = self.x[0, :] - (
            self.x[1, :] - self.x[0, :]
        )  # Bottom Fill
        pseudo_supergrid_x[-1, 1:-1] = self.x[-1, :] + (
            self.x[-1, :] - self.x[-2, :]
        )  # Top Fill
        pseudo_supergrid_x[1:-1, 0] = self.x[:, 0] - (
            self.x[:, 1] - self.x[:, 0]
        )  # Left Fill
        pseudo_supergrid_x[1:-1, -1] = self.x[:, -1] + (
            self.x[:, -1] - self.x[:, -2]
        )  # Right Fill

        pseudo_supergrid_y[1:-1, 1:-1] = self.y
        pseudo_supergrid_y[0, 1:-1] = self.y[0, :] - (
            self.y[1, :] - self.y[0, :]
        )  # Bottom Fill
        pseudo_supergrid_y[-1, 1:-1] = self.y[-1, :] + (
            self.y[-1, :] - self.y[-2, :]
        )  # Top Fill
        pseudo_supergrid_y[1:-1, 0] = self.y[:, 0] - (
            self.y[:, 1] - self.y[:, 0]
        )  # Left Fill
        pseudo_supergrid_y[1:-1, -1] = self.y[:, -1] + (
            self.y[:, -1] - self.y[:, -2]
        )  # Right Fill

        ## Fill Corners
        pseudo_supergrid_x[0, 0] = self.x[0, 0] - (
            self.x[1, 1] - self.x[0, 0]
        )  # Bottom Left
        pseudo_supergrid_x[-1, 0] = self.x[-1, 0] - (
            self.x[-2, 1] - self.x[-1, 0]
        )  # Top Left
        pseudo_supergrid_x[0, -1] = self.x[0, -1] - (
            self.x[1, -2] - self.x[0, -1]
        )  # Bottom Right
        pseudo_supergrid_x[-1, -1] = self.x[-1, -1] - (
            self.x[-2, -2] - self.x[-1, -1]
        )  # Top Right

        pseudo_supergrid_y[0, 0] = self.y[0, 0] - (
            self.y[1, 1] - self.y[0, 0]
        )  # Bottom Left
        pseudo_supergrid_y[-1, 0] = self.y[-1, 0] - (
            self.y[-2, 1] - self.y[-1, 0]
        )  # Top Left
        pseudo_supergrid_y[0, -1] = self.y[0, -1] - (
            self.y[1, -2] - self.y[0, -1]
        )  # Bottom Right
        pseudo_supergrid_y[-1, -1] = self.y[-1, -1] - (
            self.y[-2, -2] - self.y[-1, -1]
        )  # Top Right

        pseudo_supergrid = xr.Dataset(
            {
                "x": (["nyp", "nxp"], pseudo_supergrid_x),
                "y": (["nyp", "nxp"], pseudo_supergrid_y),
            }
        )
        return pseudo_supergrid


class UniformSphericalSupergrid(SupergridBase):
    """MOM6-style supergrid with constant-degree spacing (lon/lat grid)."""

    @classmethod
    def from_extents(cls, lon_min, len_x, lat_min, len_y, nx, ny):
        """Create a grid from domain extents (lon/lat degrees)."""
        x, y = cls._calc_xy_from_extents(lon_min, len_x, lat_min, len_y, nx, ny)
        dx, dy, area, angle_dx, axis_units = cls._calc_geometry(x, y)
        return cls(x, y, dx, dy, area, angle_dx, axis_units)

    @classmethod
    def from_xy(cls, x, y):
        """Create a grid directly from coordinate arrays."""
        dx, dy, area, angle_dx, axis_units = cls._calc_geometry(x, y)
        return cls(x, y, dx, dy, area, angle_dx, axis_units)

    @classmethod
    def _calc_xy_from_extents(cls, lon_min, len_x, lat_min, len_y, nx, ny):
        """Compute full grid geometry for equal-degree spacing."""
        # This builds all geometric quantities (x, y, dx, dy, area, angle)
        # for a supergrid defined in equal-degree (lon/lat) coordinates.

        # ---------------------------------------------------------------------
        # Determine grid resolution and index arrays
        # ---------------------------------------------------------------------
        nx_total = nx * 2  # number of longitudinal cells
        ny_total = ny * 2  # number of latitudinal cells

        jind = np.arange(ny_total)  # latitude cell indices
        iind = np.arange(nx_total)  # longitude cell indices
        jindp = np.arange(ny_total + 1)  # latitude point indices (cell edges)
        iindp = np.arange(nx_total + 1)  # longitude point indices (cell edges)

        # ---------------------------------------------------------------------
        # Compute grid coordinates in degrees
        # ---------------------------------------------------------------------
        grid_y = lat_min + jindp * len_y / ny_total  # latitude edges
        grid_x = lon_min + iindp * len_x / nx_total  # longitude edges

        # Form full 2D coordinate arrays for all cell corners
        x = np.tile(grid_x, (ny_total + 1, 1))
        y = np.tile(grid_y.reshape((ny_total + 1, 1)), (1, nx_total + 1))

        return x, y

    @classmethod
    def _calc_geometry(cls, x, y):
        """Compute full grid geometry for equal-degree spacing."""

        # Update cell counts (used later for shape-dependent arrays)
        nx = x.shape[1] - 1
        ny = x.shape[0] - 1

        # ---------------------------------------------------------------------
        # Compute metric distances on a sphere
        # ---------------------------------------------------------------------
        dx, dy = cls._calc_dx_dy(x, y)

        # ---------------------------------------------------------------------
        # Compute cell areas (approximate rectangular areas)
        # ---------------------------------------------------------------------
        area = dx[:-1, :] * dy[:, :-1]

        # ---------------------------------------------------------------------
        # Grid Angle is zero for uniform grids!ß
        # ---------------------------------------------------------------------
        angle_dx = np.zeros((ny + 1, nx + 1))

        # ---------------------------------------------------------------------
        # Record axis units and return all quantities
        # ---------------------------------------------------------------------
        axis_units = "degrees"

        return dx, dy, area, angle_dx, axis_units


class RectilinearCartesianSupergrid(SupergridBase):
    """MOM6-style supergrid with uniform Cartesian spacing (x/y in meters). Originally by Ashley Barnes in regional_mom6"""

    def __init__(self, lon_min, len_x, lat_min, len_y, resolution):
        x, y, dx, dy, area, angle, axis_units = self._build_grid(
            lon_min, len_x, lat_min, len_y, resolution
        )
        super().__init__(x, y, dx, dy, area, angle, axis_units)

    def _build_grid(self, lon_min, len_x, lat_min, len_y, resolution):
        """Compute full grid geometry for even physical spacing."""
        lon_max = lon_min + len_x
        lat_max = lat_min + len_y

        nx = int(len_x / (resolution / 2))
        if nx % 2 != 1:
            nx += 1

        lons = np.linspace(lon_min, lon_max, nx)  # longitudes in degrees

        # Latitudes evenly spaced by dx * cos(central_latitude)
        central_latitude = np.mean([lat_min, lat_max])  # degrees
        latitudinal_resolution = resolution * np.cos(np.deg2rad(central_latitude))

        ny = int(len_y / (latitudinal_resolution / 2)) + 1

        if ny % 2 != 1:
            ny += 1
        lats = np.linspace(lat_min, lat_max, ny)  # latitudes in degrees

        assert np.all(
            np.diff(lons) > 0
        ), "longitudes array lons must be monotonically increasing"
        assert np.all(
            np.diff(lats) > 0
        ), "latitudes array lats must be monotonically increasing"

        R = SupergridBase.R

        # ensure that longitudes are uniformly spaced
        dlons = lons[1] - lons[0]
        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
        ), "provided array of longitudes must be uniformly spaced"

        lon, lat = np.meshgrid(lons, lats)

        # Calculate dx & dy in meters, accounting for spherical geometry
        dx, dy = SupergridBase._calc_dx_dy(lon, lat)

        area = quadrilateral_areas(lat, lon, R)

        angle_dx = np.zeros_like(lon)

        axis_units = "degrees"
        return lon, lat, dx, dy, area, angle_dx, axis_units
