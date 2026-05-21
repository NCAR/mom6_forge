"""
This module defines MOM6-style supergrid classes and associated utilities. It sits underneath the mom6_forge.grid class and fills the roll of calculating the grid geometry: angle_dx, area, dx, dy, x, and y.

Classes defined here:
- SupergridBase: Base class defining the MOM6-style supergrid interface.
- UniformSphericalSupergrid: MOM6-style supergrid with constant-degree spacing (lon/lat grid).
- RectilinearCartesianSupergrid: MOM6-style supergrid with (as close to) uniform Cartesian spacing (still a lat/lon grid).
- ProjectedSupergrid: MOM6-style supergrid built from a pyproj map projection. Use this
  for polar domains (e.g., EPSG:3995/3031) or rotated regional grids (e.g., estuary-aligned).

The code for these classes does not originally come from mom6_forge, but was adapted: UniformSphericalSupergrid by Mathew Harrison in MIDAS (https://github.com/mjharriso/MIDAS) and RectilinearCartesianSupergrid by Ashley Barnes in regional_mom6 (https://github.com/COSIMA/regional-mom6).
"""

import numpy as np
import xarray as xr
from datetime import datetime
from typing import Optional

_DEFAULT_RADIUS = 6.371e6  # mean radius of the Earth (IUGG), in metres
from pyproj import CRS, Transformer
from mom6_forge.utils import normalize_deg


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

    def __init__(self, x, y, dx, dy, area, angle_dx, axis_units, grid_type, radius):
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
        grid_type : str
            the type of grid being created
        """
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.area = area
        self.angle_dx = angle_dx
        self.axis_units = axis_units
        self.grid_type = grid_type
        self.radius = radius

    def __eq__(self, other):
        if not isinstance(other, SupergridBase):
            return NotImplemented
        return (self.x == other.x).all() and (self.y == other.y).all()

    @staticmethod
    def _calc_dx_dy(x, y, R=_DEFAULT_RADIUS, type="smallangle"):
        """Compute supergrid dx and dy from coordinate arrays.

        Parameters
        ----------
        x, y : 2D arrays
            Supergrid longitude and latitude in degrees, shape (2*ny+1, 2*nx+1).
        R : float, optional
            Sphere radius in metres. Defaults to Earth's IUGG mean radius.
        type: str, optional
            The method to use to calculate dx and dy (smallangle or haversine)


        Returns
        -------
        dx : 2D array, shape (2*ny+1, 2*nx)
            Arc lengths between horizontally adjacent nodes, in metres.
        dy : 2D array, shape (2*ny, 2*nx+1)
            Arc lengths between vertically adjacent nodes, in metres.
        """
        if type == "smallangle":
            # Use small angle approximation for dx, which is faster to compute and not much different
            dx = R * np.cos(np.deg2rad(y[:, :-1])) * np.deg2rad(np.diff(x, axis=1))
            dy = R * np.deg2rad(np.diff(y, axis=0))
        elif type == "haversine":
            dx = haversine(y[:, :-1], x[:, :-1], y[:, 1:], x[:, 1:], R)
            dy = haversine(y[:-1, :], x[:-1, :], y[1:, :], x[1:, :], R)
        else:
            raise ValueError(f"Unrecognized dx/dy calc type: {type}")
        return dx, dy

    @staticmethod
    def _calc_area(x, y, R=_DEFAULT_RADIUS):
        """Compute supergrid cell areas from coordinate arrays.

        Parameters
        ----------
        x, y : 2D arrays
            Supergrid longitude and latitude in degrees, shape (2*ny+1, 2*nx+1).
        R : float, optional
            Sphere radius in metres. Defaults to Earth's IUGG mean radius.

        Returns
        -------
        area : 2D array, shape (2*ny, 2*nx)
            Cell areas in square metres.
        """
        return quadrilateral_areas(y, x, R)

    @classmethod
    def _init_from_xy(
        cls,
        x,
        y,
        grid_type=None,
        R=_DEFAULT_RADIUS,
        angles_are_zero=False,
        dx_dy_calc_type="smallangle",
    ):
        """Build supergrid metrics from y/x arrays. Should not really be called directly by users (unless experienced); use from_* method instead

        Parameters
        ----------
        x, y : np.ndarray, shape (2*ny+1, 2*nx+1)
            Geographic coordinates of all supergrid nodes in degrees.
        grid_type : str
            The type of grid being created
        radius : float, optional
            Sphere radius in metres. Defaults to Earth's IUGG mean radius.
        angles_are_zero: bool, optional
            Angle calculation is an approximation for regional grids.
            If angles are known to be zero, we can skip angle calculation (which may be slighly off from zero)
            and set angles to zero
        dx_dy_calc_type: str, optional
            The method to use to calculate dx and dy (smallangle or haversine)
        """
        # Clamp to valid geographic range (floating-point overshoot from projection in some cases)
        y = np.clip(y, -90.0, 90.0)

        # dx, dy, area: use base class consistent calculation methods
        dx, dy = SupergridBase._calc_dx_dy(x, y, R=R, type=dx_dy_calc_type)
        area = SupergridBase._calc_area(x, y, R=R)

        if angles_are_zero:
            angle_dx = np.zeros_like(x)
        else:
            angle_dx = SupergridBase.calc_supergrid_rotation_angles_using_expanded_supergrid_method(
                x, y
            )
        return cls(
            x, y, dx, dy, area, angle_dx, "degrees", grid_type=grid_type, radius=R
        )

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
        if self.grid_type is not None:
            ds.attrs["grid_type"] = self.grid_type
        ds.attrs["radius"] = self.radius
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

    @classmethod
    def from_ds(cls, ds: xr.Dataset) -> "SupergridBase":
        """Load a supergrid from a Dataset written by to_ds, returning a SupergridBase instance.

        Does not dispatch to subclasses
        """
        return cls(
            ds.x.data,
            ds.y.data,
            ds.dx.data,
            ds.dy.data,
            ds.area.data,
            ds.angle_dx.data,
            ds.x.attrs.get("units", "degrees"),
            grid_type=ds.attrs.get("grid_type"),
            radius=ds.attrs["radius"],
        )

    @staticmethod
    def calc_supergrid_rotation_angles_using_expanded_supergrid_method(
        x,
        y,
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
        expanded_supergrid = SupergridBase._create_expanded_supergrid(x, y)

        point = xr.Dataset(
            {
                "x": (["nyp", "nxp"], x),
                "y": (["nyp", "nxp"], y),
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

    @staticmethod
    def _create_expanded_supergrid(x, y, expansion_width=1) -> xr.Dataset:
        """
        Adds an additional boundary to the supergrid to allow for the calculation of the ``angle_dx`` for the boundary points using :func:`~mom6_angle_calculation_method`.
        """
        if expansion_width != 1:
            raise NotImplementedError("Only expansion_width = 1 is supported")

        ny, nx = x.shape
        pseudo_supergrid_x = np.full((ny + 2, nx + 2), np.nan)
        pseudo_supergrid_y = np.full((ny + 2, nx + 2), np.nan)

        ## Fill Boundaries
        pseudo_supergrid_x[1:-1, 1:-1] = x
        pseudo_supergrid_x[0, 1:-1] = x[0, :] - (x[1, :] - x[0, :])  # Bottom Fill
        pseudo_supergrid_x[-1, 1:-1] = x[-1, :] + (x[-1, :] - x[-2, :])  # Top Fill
        pseudo_supergrid_x[1:-1, 0] = x[:, 0] - (x[:, 1] - x[:, 0])  # Left Fill
        pseudo_supergrid_x[1:-1, -1] = x[:, -1] + (x[:, -1] - x[:, -2])  # Right Fill

        pseudo_supergrid_y[1:-1, 1:-1] = y
        pseudo_supergrid_y[0, 1:-1] = y[0, :] - (y[1, :] - y[0, :])  # Bottom Fill
        pseudo_supergrid_y[-1, 1:-1] = y[-1, :] + (y[-1, :] - y[-2, :])  # Top Fill
        pseudo_supergrid_y[1:-1, 0] = y[:, 0] - (y[:, 1] - y[:, 0])  # Left Fill
        pseudo_supergrid_y[1:-1, -1] = y[:, -1] + (y[:, -1] - y[:, -2])  # Right Fill

        ## Fill Corners
        pseudo_supergrid_x[0, 0] = x[0, 0] - (x[1, 1] - x[0, 0])  # Bottom Left
        pseudo_supergrid_x[-1, 0] = x[-1, 0] - (x[-2, 1] - x[-1, 0])  # Top Left
        pseudo_supergrid_x[0, -1] = x[0, -1] - (x[1, -2] - x[0, -1])  # Bottom Right
        pseudo_supergrid_x[-1, -1] = x[-1, -1] - (x[-2, -2] - x[-1, -1])  # Top Right

        pseudo_supergrid_y[0, 0] = y[0, 0] - (y[1, 1] - y[0, 0])  # Bottom Left
        pseudo_supergrid_y[-1, 0] = y[-1, 0] - (y[-2, 1] - y[-1, 0])  # Top Left
        pseudo_supergrid_y[0, -1] = y[0, -1] - (y[1, -2] - y[0, -1])  # Bottom Right
        pseudo_supergrid_y[-1, -1] = y[-1, -1] - (y[-2, -2] - y[-1, -1])  # Top Right

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
    def from_extents(
        cls, lon_min, len_x, lat_min, len_y, nx, ny, radius=_DEFAULT_RADIUS
    ):
        """Create a grid from domain extents (lon/lat degrees)."""
        x, y = cls._calc_xy_from_extents(lon_min, len_x, lat_min, len_y, nx, ny)
        return cls._init_from_xy(
            x, y, "uniform_spherical", radius, angles_are_zero=True
        )

    @classmethod
    def _calc_xy_from_extents(cls, lon_min, len_x, lat_min, len_y, nx, ny):
        """Compute full grid geometry for equal-degree spacing."""

        # ---------------------------------------------------------------------
        # Determine grid resolution and index arrays
        # ---------------------------------------------------------------------
        nx_total = nx * 2  # number of longitudinal cells
        ny_total = ny * 2  # number of latitudinal cells

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


class RectilinearCartesianSupergrid(SupergridBase):
    """MOM6-style supergrid with uniform Cartesian spacing (x/y in meters). Originally by Ashley Barnes in regional_mom6"""

    @classmethod
    def from_extents(
        cls, lon_min, len_x, lat_min, len_y, resolution, radius=_DEFAULT_RADIUS
    ):
        x, y = cls._build_grid(lon_min, len_x, lat_min, len_y, resolution)
        return cls._init_from_xy(
            x, y, "rectilinear_cartesian", radius, angles_are_zero=True
        )

    @classmethod
    def _build_grid(self, lon_min, len_x, lat_min, len_y, resolution):
        """Compute x,y for even physical spacing."""
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

        # ensure that longitudes are uniformly spaced
        dlons = lons[1] - lons[0]
        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
        ), "provided array of longitudes must be uniformly spaced"

        lon, lat = np.meshgrid(lons, lats)

        return lon, lat


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
    def from_crs(
        cls, crs, x_min, x_max, y_min, y_max, resolution_m, radius=_DEFAULT_RADIUS
    ):
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
        radius : float, optional
            Sphere radius in metres. Defaults to Earth's IUGG mean radius.
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

        return cls._init_from_xy(lon, lat, "projected_crs", radius)

    @classmethod
    def from_center(
        cls,
        center_lat,
        center_lon,
        width_m,
        height_m,
        resolution_m,
        angle_deg=0.0,
        radius=_DEFAULT_RADIUS,
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
        radius : float, optional
            Sphere radius in metres. Defaults to Earth's IUGG mean radius.
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

        return cls._init_from_xy(lon, lat, "projected_crs", radius)


def angle_between(v1, v2, v3):
    """Return the angle ``v2``-``v1``-``v3`` (in radians), where
    ``v1``, ``v2``, ``v3`` are 3-vectors. That is, the angle that
    is formed between vectors ``v2 - v1`` and vector ``v3 - v1``.

    Example:

        >>> from regional_mom6.utils import angle_between
        >>> v1 = (0, 0, 1)
        >>> v2 = (1, 0, 0)
        >>> v3 = (0, 1, 0)
        >>> angle_between(v1, v2, v3)
        1.5707963267948966
        >>> from numpy import rad2deg
        >>> rad2deg(angle_between(v1, v2, v3))
        90.0
    """

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    norm_v1xv2 = np.sqrt(vecdot(v1xv2, v1xv2))
    norm_v1xv3 = np.sqrt(vecdot(v1xv3, v1xv3))

    cosangle = vecdot(v1xv2, v1xv3) / (norm_v1xv2 * norm_v1xv3)

    return np.arccos(cosangle)


def vecdot(v1, v2):
    """Return the dot product of vectors ``v1`` and ``v2``.
    ``v1`` and ``v2`` can be either numpy vectors or numpy.ndarrays
    in which case the last dimension is considered the dimension
    over which the dot product is taken.
    """
    return np.sum(v1 * v2, axis=-1)


def latlon_to_cartesian(lat, lon, R=1):
    """Convert latitude and longitude (in degrees) to Cartesian coordinates on
    a sphere of radius ``R``. By default ``R = 1``.

    Arguments:
        lat (float): Latitude (in degrees).
        lon (float): Longitude (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        tuple: Tuple with the Cartesian coordinates ``x, y, z``

    Examples:

        Find the Cartesian coordinates that correspond to point with
        ``(lat, lon) = (0, 0)`` on a sphere with unit radius.

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> latlon_to_cartesian(0, 0)
        (1.0, 0.0, 0.0)

        Now let's do the same on a sphere with Earth's radius

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> R = 6371e3
        >>> latlon_to_cartesian(0, 0, R)
        (6371000.0, 0.0, 0.0)
    """

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """Return the area of spherical quadrilaterals on a sphere of radius ``R``.
    By default, ``R = 1``. The quadrilaterals are formed by constant latitude and
    longitude lines on the ``lat``-``lon`` grid provided.

    Arguments:
        lat (numpy.array): Array of latitude points (in degrees).
        lon (numpy.array): Array of longitude points (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        numpy.array: Array with the areas of the quadrilaterals defined by the ``lat``-``lon`` grid
        provided. If the provided ``lat``, ``lon`` arrays are of dimension *m* :math:`\\times` *n*
        then returned areas array is of dimension (*m-1*) :math:`\\times` (*n-1*).

    Example:

        Let's construct a lat-lon grid on the sphere with 60 degree spacing.
        Then we compute the areas of each grid cell and confirm that the
        sum of the areas gives us the total area of the sphere.

        >>> from regional_mom6.utils import quadrilateral_areas
        >>> import numpy as np
        >>> λ = np.linspace(0, 360, 7)
        >>> φ = np.linspace(-90, 90, 4)
        >>> lon, lat = np.meshgrid(λ, φ)
        >>> lon
        array([[  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.]])
        >>> lat
        array([[-90., -90., -90., -90., -90., -90., -90.],
               [-30., -30., -30., -30., -30., -30., -30.],
               [ 30.,  30.,  30.,  30.,  30.,  30.,  30.],
               [ 90.,  90.,  90.,  90.,  90.,  90.,  90.]])
        >>> R = 6371e3
        >>> areas = quadrilateral_areas(lat, lon, R)
        >>> areas
        array([[1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13],
               [4.56284230e+13, 4.56284230e+13, 4.56284230e+13, 4.56284230e+13,
                4.56284230e+13, 4.56284230e+13],
               [1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13]])
        >>> np.isclose(areas.sum(), 4 * np.pi * R**2, atol=np.finfo(areas.dtype).eps)
        True
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )


def quadrilateral_area(v1, v2, v3, v4):
    """Return the area of a spherical quadrilateral on the unit sphere that
    has vertices on the 3-vectors ``v1``, ``v2``, ``v3``, ``v4``
    (counter-clockwise orientation is implied). The area is computed via
    the excess of the sum of the spherical angles of the quadrilateral from 2π.

    Example:

        Calculate the area that corresponds to half the Northern hemisphere
        of a sphere of radius *R*. This should be 1/4 of the sphere's total area,
        that is π *R*:sup:`2`.

        >>> from regional_mom6.utils import quadrilateral_area, latlon_to_cartesian
        >>> R = 434.3
        >>> v1 = latlon_to_cartesian(0, 0, R)
        >>> v2 = latlon_to_cartesian(0, 90, R)
        >>> v3 = latlon_to_cartesian(90, 0, R)
        >>> v4 = latlon_to_cartesian(0, -90, R)
        >>> quadrilateral_area(v1, v2, v3, v4)
        592556.1793298927
        >>> from numpy import pi
        >>> quadrilateral_area(v1, v2, v3, v4) == pi * R**2
        True
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    v4 = np.array(v4)

    if not (
        np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v3, v3)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v4, v4)))
    ):
        raise ValueError("vectors provided must have the same length")

    R = np.sqrt(vecdot(v1, v1))

    a1 = angle_between(v1, v2, v4)
    a2 = angle_between(v2, v3, v1)
    a3 = angle_between(v3, v4, v2)
    a4 = angle_between(v4, v1, v3)

    return (a1 + a2 + a3 + a4 - 2 * np.pi) * R**2


def mom6_angle_calculation_method(
    len_lon,
    top_left: xr.DataArray,
    top_right: xr.DataArray,
    bottom_left: xr.DataArray,
    bottom_right: xr.DataArray,
    point: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the angle of the grid point's local x-direction compared to East-West direction
    using the MOM6 method adapted from: https://github.com/mom-ocean/MOM6/blob/05d8cc395c1c3c04dd04885bf8dd6df50a86b862/src/initialization/MOM_shared_initialization.F90#L572-L587

    Note: this is exactly the same as the angle of the grid point's local y-direction compared to North-South direction.

    This method can handle vectorized computations.

    Parameters
    ----------
    len_lon: float
        The extent of the longitude of the regional domain (in degrees).
    top_left, top_right, bottom_left, bottom_right: xr.DataArray
        The four points around the point to calculate the angle from the ``supergrid``;
        requires both an ``x``` and ``y`` component (both in degrees).
    point: xr.DataArray
        The point to calculate the angle from the ``supergrid``

    Returns
    -------
    xr.DataArray
        The angle of the grid point's local ``x``-direction compared to East-West direction.
    """

    # Compute lonB for all points
    lonB = np.zeros((2, 2, len(point.nyp), len(point.nxp)))

    # Vectorized computation of lonB
    lonB[0][0] = modulo_around_point(bottom_left.x, point.x, len_lon)  # Bottom Left
    lonB[1][0] = modulo_around_point(top_left.x, point.x, len_lon)  # Top Left
    lonB[1][1] = modulo_around_point(top_right.x, point.x, len_lon)  # Top Right
    lonB[0][1] = modulo_around_point(bottom_right.x, point.x, len_lon)  # Bottom Right

    cos_meanlat = np.cos(
        np.deg2rad((bottom_left.y + bottom_right.y + top_right.y + top_left.y) / 4)
    )

    # Quadrilateral diagonals

    # top-left--bottom-right diagonal components
    TL_BR_diagonal_x = cos_meanlat * (lonB[1, 0] - lonB[0, 1])
    TL_BR_diagonal_y = top_left.y - bottom_right.y

    # top-right--bottom-left diagonal components
    TR_BL_diagonal_x = cos_meanlat * (lonB[1, 1] - lonB[0, 0])
    TR_BL_diagonal_y = top_right.y - bottom_left.y

    # Sum of diagonals components
    sum_of_diagonals_x = TR_BL_diagonal_x + TL_BR_diagonal_x
    sum_of_diagonals_y = TR_BL_diagonal_y + TL_BR_diagonal_y

    # Angle of sum-of-diagonals vector with the North-South direction
    # Note: the minus sign changes convention from clockwise to counter-clockwise
    angle = -np.arctan2(sum_of_diagonals_x, sum_of_diagonals_y)  # = - atan(x/y)

    # Convert to degrees and assign to angles_arr
    angles_arr = np.rad2deg(angle)

    # Assign angles_arr to supergrid
    t_angles = xr.DataArray(
        angles_arr,
        dims=["nyp", "nxp"],
        coords={
            "nyp": point.nyp.values,
            "nxp": point.nxp.values,
        },
    )
    return t_angles


def modulo_around_point(x, x0, L):
    """
    Returns the modulo-:math:`L` value of :math:`x` within the interval :math:`[x_0 - L/2, x_0 + L/2]`.
    If :math:`L ≤ 0`, then method returns :math:`x`.

    (Adapted from MOM6 code; https://github.com/mom-ocean/MOM6/blob/776be843e904d85c7035ffa00233b962a03bfbb4/src/initialization/MOM_shared_initialization.F90#L592-L606)

    Parameters
    ----------
    x: xr.DataArray
       Value(s) to which to apply modulo arithmetic
    x0: xr.DataArray
        Center(s) of modulo range
    L: float
       Modulo range width

    Returns
    -------
    float
        ``x`` shifted by an integer multiple of ``L`` to be closer to ``x0``, i.e., within the interval ``[x0 - L/2, x0 + L/2]``
    """
    if L <= 0:
        return x
    else:
        # Find that boundary point x0 + L/2
        edge_indexes = np.where((x == x0 + L / 2))

        # Modulo calculation
        calc = ((x - (x0 - L / 2)) % L) + (x0 - L / 2)

        # Find that boundary point x0 + L/2 does not flip to x0 - L/2
        calc[edge_indexes] = x[edge_indexes]

        return calc


def haversine(lat1, lon1, lat2, lon2, R):
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
