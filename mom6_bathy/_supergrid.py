"""
This module defines MOM6-style supergrid classes and associated utilities. It sits underneath the mom6_bathy.grid class and fills the roll of calculating the grid geometry: angle_dx, area, dx, dy, x, and y.

Classes defined here:
- SupergridBase: Base class defining the MOM6-style supergrid interface.
- UniformSphericalSupergrid: MOM6-style supergrid with constant-degree spacing (lon/lat grid).
- RectilinearCartesianSupergrid: MOM6-style supergrid with (as close to) uniform Cartesian spacing (still a lat/lon grid).
- ProjectedSupergrid: MOM6-style supergrid built from a pyproj map projection. Use this
  for polar domains (e.g., EPSG:3995/3031) or rotated regional grids (e.g., estuary-aligned).

The code for these classes does not originally come from mom6_bathy, but was adapted: UniformSphericalSupergrid by Mathew Harrison in MIDAS (https://github.com/mjharriso/MIDAS) and RectilinearCartesianSupergrid by Ashley Barnes in regional_mom6 (https://github.com/COSIMA/regional-mom6).
"""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional
from mom6_bathy.utils import quadrilateral_areas, mdist, normalize_deg
from pyproj import CRS, Transformer


class SupergridBase:
    """Base class defining the MOM6-style supergrid interface."""

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

    def __init__(self, x, y, dx, dy, area, angle_dx, axis_units, grid_params):
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
        grid_params : dict
            Construction parameters written as dataset attributes on save.
            Should include at minimum a "grid_type" key.
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.area = area
        self.angle_dx = angle_dx
        self.axis_units = axis_units
        self._grid_params = grid_params

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
        ds.attrs.update(self._grid_params)

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

    @classmethod
    def from_ds(cls, ds: xr.Dataset) -> "SupergridBase":
        """Load a supergrid from a Dataset written by to_ds, returning a SupergridBase instance.

        Does not dispatch to subclasses — use supergrid_class_from_ds to identify
        the originating class if subclass-specific reconstruction is needed.
        """
        return cls(
            ds.x.data,
            ds.y.data,
            ds.dx.data,
            ds.dy.data,
            ds.area.data,
            ds.angle_dx.data,
            ds.x.attrs.get("units", "degrees"),
            {},
        )


class UniformSphericalSupergrid(SupergridBase):
    """MOM6-style supergrid with constant-degree spacing (lon/lat grid)."""

    @classmethod
    def from_extents(cls, lon_min, len_x, lat_min, len_y, nx, ny):
        """Create a grid from domain extents (lon/lat degrees)."""
        x, y = cls._calc_xy_from_extents(lon_min, len_x, lat_min, len_y, nx, ny)
        dx, dy, area, angle_dx, axis_units = cls._calc_geometry(x, y)
        return cls(
            x,
            y,
            dx,
            dy,
            area,
            angle_dx,
            axis_units,
            dict(
                grid_type="uniform_spherical",
                lon_min=lon_min,
                len_x=len_x,
                lat_min=lat_min,
                len_y=len_y,
                nx=nx,
                ny=ny,
            ),
        )

    @classmethod
    def from_xy(cls, x, y):
        """Create a grid directly from coordinate arrays."""
        dx, dy, area, angle_dx, axis_units = cls._calc_geometry(x, y)
        return cls(x, y, dx, dy, area, angle_dx, axis_units, {})

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
        # Compute metric distances on a sphere (approximate)
        # ---------------------------------------------------------------------
        radius = 6.378e6  # Earth radius in meters
        metric = np.deg2rad(radius)  # degrees → meters scaling factor

        # Compute midpoints in each direction
        ymid_j = 0.5 * (y + np.roll(y, shift=-1, axis=0))
        ymid_i = 0.5 * (y + np.roll(y, shift=-1, axis=1))

        # Differences in latitude (dy) and longitude (dx) between adjacent cells
        dy_j = np.roll(y, shift=-1, axis=0) - y
        dy_i = np.roll(y, shift=-1, axis=1) - y
        dx_i = mdist(np.roll(x, shift=-1, axis=1), x)
        dx_j = mdist(np.roll(x, shift=-1, axis=0), x)

        # Compute true distances accounting for spherical geometry
        dx = (
            metric
            * metric
            * (dy_i * dy_i + dx_i * dx_i * np.cos(np.deg2rad(ymid_i)) ** 2)
        )
        dx = np.sqrt(dx)

        dy = (
            metric
            * metric
            * (dy_j * dy_j + dx_j * dx_j * np.cos(np.deg2rad(ymid_j)) ** 2)
        )
        dy = np.sqrt(dy)

        # Trim grid edges for consistency
        dx = dx[:, :-1]
        dy = dy[:-1, :]

        # ---------------------------------------------------------------------
        # Compute cell areas (approximate rectangular areas)
        # ---------------------------------------------------------------------
        area = dx[:-1, :] * dy[:, :-1]

        # ---------------------------------------------------------------------
        # Compute local grid angle relative to east
        # ---------------------------------------------------------------------
        angle_dx = np.zeros((ny + 1, nx + 1))

        # Interior points
        angle_dx[:, 1:-1] = np.arctan2(
            y[:, 2:] - y[:, :-2],
            (x[:, 2:] - x[:, :-2]) * np.cos(np.deg2rad(y[:, 1:-1])),
        )
        # Western boundary
        angle_dx[:, 0] = np.arctan2(
            y[:, 1] - y[:, 0],
            (x[:, 1] - x[:, 0]) * np.cos(np.deg2rad(y[:, 0])),
        )
        # Eastern boundary
        angle_dx[:, -1] = np.arctan2(
            y[:, -1] - y[:, -2],
            (x[:, -1] - x[:, -2]) * np.cos(np.deg2rad(y[:, -1])),
        )

        # Convert angle from degrees to radians
        angle_dx = np.deg2rad(angle_dx)

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
        super().__init__(
            x,
            y,
            dx,
            dy,
            area,
            angle,
            axis_units,
            dict(
                grid_type="rectilinear_cartesian",
                lon_min=lon_min,
                len_x=len_x,
                lat_min=lat_min,
                len_y=len_y,
                resolution=resolution,
            ),
        )

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

        R = 6.371e6  # mean radius of the Earth; https://en.wikipedia.org/wiki/Earth_radius in m

        # compute longitude spacing and ensure that longitudes are uniformly spaced
        dlons = lons[1] - lons[0]

        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
        ), "provided array of longitudes must be uniformly spaced"

        # Note: division by 2 because we're on the supergrid
        dx = np.broadcast_to(
            R * np.cos(np.deg2rad(lats)) * np.deg2rad(dlons) / 2,
            (lons.shape[0] - 1, lats.shape[0]),
        ).T

        # dy = R * np.deg2rad(dlats) / 2
        # Note: division by 2 because we're on the supergrid
        dy = np.broadcast_to(
            R * np.deg2rad(np.diff(lats)) / 2, (lons.shape[0], lats.shape[0] - 1)
        ).T

        lon, lat = np.meshgrid(lons, lats)

        area = quadrilateral_areas(lat, lon, R)

        angle_dx = np.zeros_like(lon)

        axis_units = "degrees"
        return lon, lat, dx, dy, area, angle_dx, axis_units


class ProjectedSupergrid(SupergridBase):
    """MOM6-style supergrid built from a map projection.

    Constructs a uniform grid in a given pyproj CRS and reprojects node
    coordinates to geographic degrees for the MOM6 supergrid file. Grid metrics
    (dx, dy, area, angle_dx) are computed from exact great-circle geometry rather
    than the approximate cos(lat) scaling used by UniformSphericalSupergrid and
    RectilinearCartesianSupergrid.

    Use this instead of RectilinearCartesianSupergrid when:
    - The domain is near a pole (e.g., "EPSG:3995" Arctic / "EPSG:3031" Antarctic).
    - The grid needs to align with a non-lat/lon feature like an estuary mouth
      (use from_center with angle_deg).
    """

    @classmethod
    def from_crs(cls, crs, x_min, x_max, y_min, y_max, resolution_m):
        """Create a grid from projected coordinate extents.

        Parameters
        ----------
        crs : pyproj.CRS, int, or str
            Coordinate reference system. Accepts a pyproj.CRS object, an EPSG
            code (int or "EPSG:XXXX"), or a PROJ string.
            Examples:
                "EPSG:3995"  — Arctic Polar Stereographic
                "EPSG:3031"  — Antarctic Polar Stereographic
                "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96"  — Lambert conformal
        x_min, x_max : float
            Projected x-coordinate extent in metres.
        y_min, y_max : float
            Projected y-coordinate extent in metres.
        resolution_m : float
            Grid resolution in metres, uniform in both projected x and y.
        """

        if not isinstance(crs, CRS):
            crs = CRS.from_user_input(crs)

        nx = int((x_max - x_min) / resolution_m)
        ny = int((y_max - y_min) / resolution_m)

        x_sg = np.linspace(x_min, x_max, 2 * nx + 1)
        y_sg = np.linspace(y_min, y_max, 2 * ny + 1)
        xx, yy = np.meshgrid(x_sg, y_sg)

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xx, yy)

        return cls._from_latlon(
            lon,
            lat,
            dict(
                grid_type="projected_crs",
                crs_wkt=crs.to_wkt(),
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                resolution_m=resolution_m,
            ),
        )

    @classmethod
    def from_center(
        cls, center_lat, center_lon, width_m, height_m, resolution_m, angle_deg=0.0
    ):
        """Create a rotated rectangular grid centered at a geographic point.

        Uses an azimuthal equidistant projection centred at (center_lat, center_lon)
        and rotates the domain by angle_deg clockwise from north. This is the right
        tool when one grid boundary needs to align with a feature like an estuary
        mouth: rotate until the southern (or northern) edge of the domain lies
        perpendicular to the channel axis.

        Parameters
        ----------
        center_lat, center_lon : float
            Geographic centre of the domain in degrees.
        width_m, height_m : float
            Domain width (x-direction) and height (y-direction) in metres.
        resolution_m : float
            Grid resolution in metres.
        angle_deg : float, optional
            Clockwise rotation from north in degrees. Default 0 (north-up).
            Example: angle_deg=45 rotates so that the x-axis points NE,
            useful for a NE-SW estuary mouth.
        """

        proj_str = (
            f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m"
        )
        crs = CRS.from_proj4(proj_str)

        nx = int(width_m / resolution_m)
        ny = int(height_m / resolution_m)

        xi = np.linspace(-width_m / 2, width_m / 2, 2 * nx + 1)
        yi = np.linspace(-height_m / 2, height_m / 2, 2 * ny + 1)
        xx, yy = np.meshgrid(xi, yi)

        # Rotate clockwise by angle_deg (standard compass bearing convention)
        theta = np.deg2rad(angle_deg)
        xx_rot = xx * np.cos(theta) + yy * np.sin(theta)
        yy_rot = -xx * np.sin(theta) + yy * np.cos(theta)

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xx_rot, yy_rot)

        return cls._from_latlon(
            lon,
            lat,
            dict(
                grid_type="projected_center",
                center_lat=center_lat,
                center_lon=center_lon,
                width_m=width_m,
                height_m=height_m,
                resolution_m=resolution_m,
                angle_deg=angle_deg,
            ),
        )

    @classmethod
    def from_ds(cls, ds: xr.Dataset) -> "ProjectedSupergrid":
        """Reconstruct a ProjectedSupergrid from a Dataset written by to_ds.

        Re-runs the original factory (from_center or from_crs) using the
        construction parameters stored as dataset attributes, giving an exact
        reconstruction of the projected grid.
        """
        grid_type = ds.attrs.get("grid_type")
        if grid_type == "projected_center":
            return cls.from_center(
                center_lat=ds.attrs["center_lat"],
                center_lon=ds.attrs["center_lon"],
                width_m=ds.attrs["width_m"],
                height_m=ds.attrs["height_m"],
                resolution_m=ds.attrs["resolution_m"],
                angle_deg=ds.attrs.get("angle_deg", 0.0),
            )
        if grid_type == "projected_crs":
            return cls.from_crs(
                crs=ds.attrs["crs_wkt"],
                x_min=ds.attrs["x_min"],
                x_max=ds.attrs["x_max"],
                y_min=ds.attrs["y_min"],
                y_max=ds.attrs["y_max"],
                resolution_m=ds.attrs["resolution_m"],
            )
        raise ValueError(
            f"Cannot reconstruct ProjectedSupergrid: unrecognised grid_type {grid_type!r}. "
            "Use SupergridBase.from_ds to load raw arrays instead."
        )

    @classmethod
    def _from_latlon(cls, lon, lat, grid_params):
        """Build supergrid metrics from reprojected lat/lon node arrays. Should not really be called directly by users (unless experienced); use from_crs or from_center instead.

        Parameters
        ----------
        lon, lat : np.ndarray, shape (2*ny+1, 2*nx+1)
            Geographic coordinates of all supergrid nodes in degrees.
        grid_params : dict
            Construction parameters to store on the instance.
        """
        R = 6.378e6

        # Clamp to valid geographic range (floating-point overshoot from projection)
        lat = np.clip(lat, -90.0, 90.0)

        # dx: great-circle distance between horizontally adjacent nodes
        # shape: (2*ny+1, 2*nx)
        dx = _haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:], R)

        # dy: great-circle distance between vertically adjacent nodes
        # shape: (2*ny, 2*nx+1)
        dy = _haversine(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :], R)

        # area: exact spherical quadrilateral areas of supergrid sub-cells
        # shape: (2*ny, 2*nx)
        area = quadrilateral_areas(lat, lon, R)

        # angle_dx: angle of grid i-direction relative to east (radians)
        # shape: (2*ny+1, 2*nx+1)
        # _dlon_signed handles grids that cross the antimeridian.
        angle_dx = np.zeros_like(lon)
        angle_dx[:, 1:-1] = np.arctan2(
            lat[:, 2:] - lat[:, :-2],
            _dlon_signed(lon[:, :-2], lon[:, 2:]) * np.cos(np.deg2rad(lat[:, 1:-1])),
        )
        angle_dx[:, 0] = np.arctan2(
            lat[:, 1] - lat[:, 0],
            _dlon_signed(lon[:, 0], lon[:, 1]) * np.cos(np.deg2rad(lat[:, 0])),
        )
        angle_dx[:, -1] = np.arctan2(
            lat[:, -1] - lat[:, -2],
            _dlon_signed(lon[:, -2], lon[:, -1]) * np.cos(np.deg2rad(lat[:, -1])),
        )

        return cls(lon, lat, dx, dy, area, angle_dx, "degrees", grid_params)


def _haversine(lat1, lon1, lat2, lon2, R=6.378e6):
    """Great-circle distance (metres) between arrays of points given in degrees."""
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2) ** 2
    )
    return (
        2
        * R
        * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    )


def _dlon_signed(lon_a, lon_b):
    """Signed longitude difference lon_b - lon_a mapped to (-180, 180]."""
    return ((lon_b - lon_a + 180.0) % 360.0) - 180.0


_GRID_TYPE_TO_CLASS = {
    "uniform_spherical": UniformSphericalSupergrid,
    "rectilinear_cartesian": RectilinearCartesianSupergrid,
    "projected_center": ProjectedSupergrid,
    "projected_crs": ProjectedSupergrid,
}


def supergrid_class_from_ds(ds: xr.Dataset):
    """Return the supergrid class that produced a dataset, without constructing an instance.

    Parameters
    ----------
    ds : xr.Dataset
        A supergrid dataset written by SupergridBase.to_ds (or Grid.write_supergrid).

    Returns
    -------
    type
        One of UniformSphericalSupergrid, RectilinearCartesianSupergrid,
        ProjectedSupergrid, or SupergridBase (fallback for datasets without a
        grid_type attribute).
    """
    return _GRID_TYPE_TO_CLASS.get(ds.attrs.get("grid_type"), SupergridBase)
