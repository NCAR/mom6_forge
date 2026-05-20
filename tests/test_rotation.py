"""Initially setup in regional_mom6"""

from mom6_forge._supergrid import (
    SupergridBase,
    mom6_angle_calculation_method,
    modulo_around_point,
)
from mom6_forge.grid import Grid
import math
import pytest
import xarray as xr
import numpy as np

tol_angle = 5e-1  # tolerance for angles (in degrees) from seperate calculations, this is slightly higher than previously
tol_angle_unit_test = 0  # tolerance for angles (in degrees) from unit test generation


def test_expanded_supergrid_generation(get_curvilinear_supergrid):
    supergrid = get_curvilinear_supergrid
    sg = SupergridBase._init_from_xy(supergrid.x.values, supergrid.y.values)
    expanded_supergrid = sg._create_expanded_supergrid(sg.x, sg.y)

    # Check Size
    assert len(expanded_supergrid.nxp) == (len(supergrid.nxp) + 2)
    assert len(expanded_supergrid.nyp) == (len(supergrid.nyp) + 2)

    # Check pseudo_supergrid keeps the same values
    assert (expanded_supergrid.x.values[1:-1, 1:-1] == supergrid.x.values).all()
    assert (expanded_supergrid.y.values[1:-1, 1:-1] == supergrid.y.values).all()

    # Check extra boundary has realistic values
    assert (
        np.abs(
            expanded_supergrid.x.values[0, 1:-1]
            - (
                supergrid.x.values[0, :]
                - (supergrid.x.values[1, :] - supergrid.x.values[0, :])
            )
        )
        < tol_angle
    ).all()
    assert (
        np.abs(
            expanded_supergrid.x.values[1:-1, 0]
            - (
                supergrid.x.values[:, 0]
                - (supergrid.x.values[:, 1] - supergrid.x.values[:, 0])
            )
        )
        < tol_angle
    ).all()
    assert (
        np.abs(
            expanded_supergrid.x.values[-1, 1:-1]
            - (
                supergrid.x.values[-1, :]
                - (supergrid.x.values[-2, :] - supergrid.x.values[-1, :])
            )
        )
        < tol_angle
    ).all()
    assert (
        np.abs(
            expanded_supergrid.x.values[1:-1, -1]
            - (
                supergrid.x.values[:, -1]
                - (supergrid.x.values[:, -2] - supergrid.x.values[:, -1])
            )
        )
        < tol_angle
    ).all()

    # Check corners for the same...
    assert (
        abs(
            expanded_supergrid.x.values[0, 0]
            - (
                supergrid.x.values[0, 0]
                - (supergrid.x.values[1, 1] - supergrid.x.values[0, 0])
            )
        )
        < tol_angle
    )
    assert (
        abs(
            expanded_supergrid.x.values[-1, 0]
            - (
                supergrid.x.values[-1, 0]
                - (supergrid.x.values[-2, 1] - supergrid.x.values[-1, 0])
            )
        )
        < tol_angle
    )
    assert (
        abs(
            expanded_supergrid.x.values[0, -1]
            - (
                supergrid.x.values[0, -1]
                - (supergrid.x.values[1, -2] - supergrid.x.values[0, -1])
            )
        )
        < tol_angle
    )
    assert (
        abs(
            expanded_supergrid.x.values[-1, -1]
            - (
                supergrid.x.values[-1, -1]
                - (supergrid.x.values[-2, -2] - supergrid.x.values[-1, -1])
            )
        )
        < tol_angle
    )

    # Same for y
    assert (
        np.abs(
            expanded_supergrid.y.values[0, 1:-1]
            - (
                supergrid.y.values[0, :]
                - (supergrid.y.values[1, :] - supergrid.y.values[0, :])
            )
        )
        < tol_angle
    ).all()
    assert (
        np.abs(
            expanded_supergrid.y.values[1:-1, 0]
            - (
                supergrid.y.values[:, 0]
                - (supergrid.y.values[:, 1] - supergrid.y.values[:, 0])
            )
        )
        < tol_angle
    ).all()
    assert (
        np.abs(
            expanded_supergrid.y.values[-1, 1:-1]
            - (
                supergrid.y.values[-1, :]
                - (supergrid.y.values[-2, :] - supergrid.y.values[-1, :])
            )
        )
        < tol_angle
    ).all()
    assert (
        np.abs(
            expanded_supergrid.y.values[1:-1, -1]
            - (
                supergrid.y.values[:, -1]
                - (supergrid.y.values[:, -2] - supergrid.y.values[:, -1])
            )
        )
        < tol_angle
    ).all()

    assert (
        abs(
            expanded_supergrid.y.values[0, 0]
            - (
                supergrid.y.values[0, 0]
                - (supergrid.y.values[1, 1] - supergrid.y.values[0, 0])
            )
        )
        < tol_angle
    )
    assert (
        abs(
            expanded_supergrid.y.values[-1, 0]
            - (
                supergrid.y.values[-1, 0]
                - (supergrid.y.values[-2, 1] - supergrid.y.values[-1, 0])
            )
        )
        < tol_angle
    )
    assert (
        abs(
            expanded_supergrid.y.values[0, -1]
            - (
                supergrid.y.values[0, -1]
                - (supergrid.y.values[1, -2] - supergrid.y.values[0, -1])
            )
        )
        < tol_angle
    )
    assert (
        abs(
            expanded_supergrid.y.values[-1, -1]
            - (
                supergrid.y.values[-1, -1]
                - (supergrid.y.values[-2, -2] - supergrid.y.values[-1, -1])
            )
        )
        < tol_angle
    )

    return


@pytest.mark.parametrize(("angle"), [0, 12.5, 65, -20])
def test_mom6_angle_calculation_method_simple_square_grids(angle):
    """
    Create a square of length 2 (square_size). Rotate it by an `angle` and then compute
    the angle using mom6_angle_calculation_method to ensure it gets
    the angle right.
    """

    # Rotation matrix
    θ = np.deg2rad(angle)  # radians
    R = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])

    # Define four point points on a square with side of
    # length 2 (square_size) and centered at (0, 0)
    square_size = 2.0
    top_left = np.array([-square_size / 2, +square_size / 2])
    top_right = np.array([+square_size / 2, +square_size / 2])
    bottom_left = np.array([-square_size / 2, -square_size / 2])
    bottom_right = np.array([+square_size / 2, -square_size / 2])

    # Apply the rotation
    top_left = R @ top_left
    top_right = R @ top_right
    bottom_left = R @ bottom_left
    bottom_right = R @ bottom_right

    # translate the 4 rotated square points so that
    # the center of the square is at (center_x, center_y)
    center_x, center_y = 0, 0

    top_left[0] += center_x
    top_left[1] += center_y
    top_right[0] += center_x
    top_right[1] += center_y
    bottom_left[0] += center_x
    bottom_left[1] += center_y
    bottom_right[0] += center_x
    bottom_right[1] += center_y

    # create that dataset with the points
    top_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[top_left[0]]]),
            "y": (("nyp", "nxp"), [[top_left[1]]]),
        }
    )
    top_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[top_right[0]]]),
            "y": (("nyp", "nxp"), [[top_right[1]]]),
        }
    )
    bottom_left = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[bottom_left[0]]]),
            "y": (("nyp", "nxp"), [[bottom_left[1]]]),
        }
    )
    bottom_right = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[bottom_right[0]]]),
            "y": (("nyp", "nxp"), [[bottom_right[1]]]),
        }
    )
    point = xr.Dataset(
        {
            "x": (("nyp", "nxp"), [[center_x]]),
            "y": (("nyp", "nxp"), [[center_y]]),
        }
    )

    # Calculate len_lon
    top_left_bottom_right_diag = abs(top_left.x.item() - bottom_right.x.item())
    top_right_bottom_left_diag = abs(top_right.x.item() - bottom_left.x.item())
    len_lon = max(top_left_bottom_right_diag, top_right_bottom_left_diag)
    computed_angle = mom6_angle_calculation_method(
        len_lon, top_left, top_right, bottom_left, bottom_right, point
    )
    assert math.isclose(computed_angle[0, 0].values, angle)


def test_mom6_angle_calculation_method(get_curvilinear_supergrid):
    # Rotated grid
    supergrid = get_curvilinear_supergrid

    # t-points: cell centers at [1::2, 1::2]; q-points: cell corners at [0::2, 0::2]
    t_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), supergrid.x.values[1::2, 1::2]),
            "y": (("nyp", "nxp"), supergrid.y.values[1::2, 1::2]),
        }
    )
    q_points = xr.Dataset(
        {
            "x": (("nyp", "nxp"), supergrid.x.values[0::2, 0::2]),
            "y": (("nyp", "nxp"), supergrid.y.values[0::2, 0::2]),
        }
    )

    t_nyp_indices = list(range(1, len(supergrid.nyp), 2))
    t_nxp_indices = list(range(1, len(supergrid.nxp), 2))
    assert (
        np.abs(
            mom6_angle_calculation_method(
                supergrid.x.max() - supergrid.x.min(),
                q_points.isel(nyp=slice(1, None), nxp=slice(0, -1)),
                q_points.isel(nyp=slice(1, None), nxp=slice(1, None)),
                q_points.isel(nyp=slice(0, -1), nxp=slice(0, -1)),
                q_points.isel(nyp=slice(0, -1), nxp=slice(1, None)),
                t_points,
            )
            - supergrid["angle_dx"].isel(nyp=t_nyp_indices, nxp=t_nxp_indices).values
        )
        < tol_angle
    ).all()

    return


def test_initialize_grid_rotation_angle(get_curvilinear_supergrid):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    supergrid = get_curvilinear_supergrid
    sg = SupergridBase._init_from_xy(supergrid.x.values, supergrid.y.values)
    sg.angle_dx = sg.calc_supergrid_rotation_angles_using_expanded_supergrid_method(
        sg.x, sg.y
    )
    angle = xr.DataArray(sg.angle_dx[1::2, 1::2], dims=["nyp", "nxp"])

    t_nyp_indices = list(range(1, len(supergrid.nyp), 2))
    t_nxp_indices = list(range(1, len(supergrid.nxp), 2))
    assert (
        np.abs(
            angle.values
            - supergrid["angle_dx"].isel(nyp=t_nyp_indices, nxp=t_nxp_indices).values
        )
        < tol_angle
    ).all()  # Angle is correct
    assert (
        angle.values.shape == supergrid.x.values[1::2, 1::2].shape
    )  # Shape is correct
    return


def test_calculate_grid_rotation_angle_using_expanded_supergrid(
    get_curvilinear_supergrid,
):
    """
    Generate a curvilinear grid and test the grid rotation angle at t_points based on what we pass to generate_curvilinear_grid
    """
    supergrid = get_curvilinear_supergrid
    sg = SupergridBase._init_from_xy(supergrid.x.values, supergrid.y.values)
    sg.angle_dx = sg.calc_supergrid_rotation_angles_using_expanded_supergrid_method(
        sg.x, sg.y
    )
    angle = xr.DataArray(sg.angle_dx, dims=["nyp", "nxp"])

    assert (np.abs(angle.values - supergrid.angle_dx) < tol_angle).all()
    assert angle.values.shape == supergrid.x.shape
    return


def test_modulo_around_point():
    """
    Test the modulo_around_point function
    """

    # Edge Cases if x is on the boundary of the domain
    x = xr.DataArray([0.5])
    x0 = xr.DataArray([0])
    L = 1

    assert modulo_around_point(x, x0, L) == x
    x = xr.DataArray([-0.5])
    x0 = xr.DataArray([0])
    L = 1
    assert modulo_around_point(x, x0, L) == x

    # Inside Case
    x = xr.DataArray([-0.2])
    x0 = xr.DataArray([0])
    L = 1
    assert modulo_around_point(x, x0, L) == x

    # Outside Case
    x = xr.DataArray([-0.6])
    x0 = xr.DataArray([0])
    L = 1
    assert modulo_around_point(x, x0, L) == x + L

    # Multiple Values Case
    x = xr.DataArray([[0.5, 0.6], [0.5, 0.6]])
    x0 = xr.DataArray([[0, 0.1], [0, 0.1]])
    L = 1
    assert np.all(modulo_around_point(x, x0, L) == x)
