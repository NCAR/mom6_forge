import xarray as xr
import numpy as np


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


def calculate_supergrid_rotation_angles_using_expanded_supergrid_method(
    supergrid: xr.Dataset,
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
    expanded_supergrid = create_expanded_supergrid(supergrid)

    return mom6_angle_calculation_method(
        expanded_supergrid.x.max() - expanded_supergrid.x.min(),
        expanded_supergrid.isel(nyp=slice(2, None), nxp=slice(0, -2)),
        expanded_supergrid.isel(nyp=slice(2, None), nxp=slice(2, None)),
        expanded_supergrid.isel(nyp=slice(0, -2), nxp=slice(0, -2)),
        expanded_supergrid.isel(nyp=slice(0, -2), nxp=slice(2, None)),
        supergrid,
    )


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


def create_expanded_supergrid(supergrid: xr.Dataset, expansion_width=1) -> xr.Dataset:
    """
    Adds an additional boundary to the supergrid to allow for the calculation of the ``angle_dx`` for the boundary points using :func:`~mom6_angle_calculation_method`.
    """
    if expansion_width != 1:
        raise NotImplementedError("Only expansion_width = 1 is supported")

    pseudo_supergrid_x = np.full(
        (len(supergrid.nyp) + 2, len(supergrid.nxp) + 2), np.nan
    )
    pseudo_supergrid_y = np.full(
        (len(supergrid.nyp) + 2, len(supergrid.nxp) + 2), np.nan
    )

    ## Fill Boundaries
    pseudo_supergrid_x[1:-1, 1:-1] = supergrid.x.values
    pseudo_supergrid_x[0, 1:-1] = supergrid.x.values[0, :] - (
        supergrid.x.values[1, :] - supergrid.x.values[0, :]
    )  # Bottom Fill
    pseudo_supergrid_x[-1, 1:-1] = supergrid.x.values[-1, :] + (
        supergrid.x.values[-1, :] - supergrid.x.values[-2, :]
    )  # Top Fill
    pseudo_supergrid_x[1:-1, 0] = supergrid.x.values[:, 0] - (
        supergrid.x.values[:, 1] - supergrid.x.values[:, 0]
    )  # Left Fill
    pseudo_supergrid_x[1:-1, -1] = supergrid.x.values[:, -1] + (
        supergrid.x.values[:, -1] - supergrid.x.values[:, -2]
    )  # Right Fill

    pseudo_supergrid_y[1:-1, 1:-1] = supergrid.y.values
    pseudo_supergrid_y[0, 1:-1] = supergrid.y.values[0, :] - (
        supergrid.y.values[1, :] - supergrid.y.values[0, :]
    )  # Bottom Fill
    pseudo_supergrid_y[-1, 1:-1] = supergrid.y.values[-1, :] + (
        supergrid.y.values[-1, :] - supergrid.y.values[-2, :]
    )  # Top Fill
    pseudo_supergrid_y[1:-1, 0] = supergrid.y.values[:, 0] - (
        supergrid.y.values[:, 1] - supergrid.y.values[:, 0]
    )  # Left Fill
    pseudo_supergrid_y[1:-1, -1] = supergrid.y.values[:, -1] + (
        supergrid.y.values[:, -1] - supergrid.y.values[:, -2]
    )  # Right Fill

    ## Fill Corners
    pseudo_supergrid_x[0, 0] = supergrid.x.values[0, 0] - (
        supergrid.x.values[1, 1] - supergrid.x.values[0, 0]
    )  # Bottom Left
    pseudo_supergrid_x[-1, 0] = supergrid.x.values[-1, 0] - (
        supergrid.x.values[-2, 1] - supergrid.x.values[-1, 0]
    )  # Top Left
    pseudo_supergrid_x[0, -1] = supergrid.x.values[0, -1] - (
        supergrid.x.values[1, -2] - supergrid.x.values[0, -1]
    )  # Bottom Right
    pseudo_supergrid_x[-1, -1] = supergrid.x.values[-1, -1] - (
        supergrid.x.values[-2, -2] - supergrid.x.values[-1, -1]
    )  # Top Right

    pseudo_supergrid_y[0, 0] = supergrid.y.values[0, 0] - (
        supergrid.y.values[1, 1] - supergrid.y.values[0, 0]
    )  # Bottom Left
    pseudo_supergrid_y[-1, 0] = supergrid.y.values[-1, 0] - (
        supergrid.y.values[-2, 1] - supergrid.y.values[-1, 0]
    )  # Top Left
    pseudo_supergrid_y[0, -1] = supergrid.y.values[0, -1] - (
        supergrid.y.values[1, -2] - supergrid.y.values[0, -1]
    )  # Bottom Right
    pseudo_supergrid_y[-1, -1] = supergrid.y.values[-1, -1] - (
        supergrid.y.values[-2, -2] - supergrid.y.values[-1, -1]
    )  # Top Right

    pseudo_supergrid = xr.Dataset(
        {
            "x": (["nyp", "nxp"], pseudo_supergrid_x),
            "y": (["nyp", "nxp"], pseudo_supergrid_y),
        }
    )
    return pseudo_supergrid


def calculate_t_point_rotation_angles(supergrid: xr.Dataset) -> xr.DataArray:
    """
    Calculate the ``angle_dx`` in degrees from the true ``x`` direction (parallel to latitude) counter-clockwise
    and return as a dataarray. (Mimics MOM6 angle calculation function :func:`~mom6_angle_calculation_method`)

    Parameters
    ----------
    supergrid: xr.Dataset
        The supergrid dataset

    Returns
    -------
    xr.DataArray
        The t-point angles
    """
    # t-points: cell centers at every other supergrid point starting at index 1
    t_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), supergrid.x.values[1::2, 1::2]),
            "y": (("nyp", "nxp"), supergrid.y.values[1::2, 1::2]),
        }
    )
    # q-points: cell corners at every other supergrid point starting at index 0
    q_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), supergrid.x.values[0::2, 0::2]),
            "y": (("nyp", "nxp"), supergrid.y.values[0::2, 0::2]),
        }
    )

    return mom6_angle_calculation_method(
        supergrid.x.max() - supergrid.x.min(),
        q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
        q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
        q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
        q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
        t_points,
    )
