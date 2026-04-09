import numpy as np
import xarray as xr


def mdist(x1, x2):
    """
    Returns positive distance modulo 360.

    >>> x1=0.0;x2=730.
    >>> d=mdist(x1,x2)
    >>> print(d)
    10.0
    """

    a = np.mod(x1 - x2 + 720.0, 360.0)
    b = np.mod(x2 - x1 + 720.0, 360.0)

    d = np.minimum(a, b)

    return d


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
