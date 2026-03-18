import numpy as np
import pytest
from mom6_bathy.grid import Grid
from mom6_bathy.topo import Topo


@pytest.fixture
def get_realistic_vgrid_elements():
    """Generate all three relevant vertical data pieces based on a 65 layer vertical grid for the Caribbean."""
    layer_thickness = np.array(
        [
            2.5,
            2.5,
            2.5,
            2.5,
            2.77,
            3.38,
            4.01,
            4.65,
            5.29,
            5.95,
            6.61,
            7.28,
            7.97,
            8.66,
            9.37,
            10.08,
            10.81,
            11.54,
            12.29,
            13.06,
            13.85,
            14.69,
            15.59,
            16.56,
            17.61,
            18.76,
            20.02,
            21.42,
            23,
            24.77,
            26.79,
            29.1,
            31.76,
            34.87,
            38.5,
            42.79,
            47.9,
            54.01,
            61.37,
            70.25,
            80.95,
            93.75,
            108.8,
            126.04,
            145.04,
            164.81,
            184.05,
            201.34,
            215.66,
            226.64,
            234.5,
            239.84,
            243.31,
            245.52,
            246.88,
            247.72,
            248.23,
            248.54,
            248.73,
            248.84,
            248.64,
            248.68,
            248.71,
            248.72,
            248.73,
        ]
    )

    cell_center = np.array(
        [
            1.25,
            3.75,
            6.25,
            8.75,
            11.385,
            14.46,
            18.155,
            22.485,
            27.455,
            33.075,
            39.355,
            46.3,
            53.925,
            62.24,
            71.255,
            80.98,
            91.425,
            102.6,
            114.515,
            127.19,
            140.645,
            154.915,
            170.055,
            186.13,
            203.215,
            221.4,
            240.79,
            261.51,
            283.72,
            307.605,
            333.385,
            361.33,
            391.76,
            425.075,
            461.76,
            502.405,
            547.75,
            598.705,
            656.395,
            722.205,
            797.805,
            885.155,
            986.43,
            1103.85,
            1239.39,
            1394.315,
            1568.745,
            1761.44,
            1969.94,
            2191.09,
            2421.66,
            2658.83,
            2900.405,
            3144.82,
            3391.02,
            3638.32,
            3886.295,
            4134.68,
            4383.315,
            4632.1,
            4880.84,
            5129.5,
            5378.195,
            5626.91,
            5875.635,
        ]
    )

    cell_interface = np.array(
        [
            0,
            2.5,
            5,
            7.5,
            10,
            12.77,
            16.15,
            20.16,
            24.81,
            30.1,
            36.05,
            42.66,
            49.94,
            57.91,
            66.57,
            75.94,
            86.02,
            96.83,
            108.37,
            120.66,
            133.72,
            147.57,
            162.26,
            177.85,
            194.41,
            212.02,
            230.78,
            250.8,
            272.22,
            295.22,
            319.99,
            346.78,
            375.88,
            407.64,
            442.51,
            481.01,
            523.8,
            571.7,
            625.71,
            687.08,
            757.33,
            838.28,
            932.03,
            1040.83,
            1166.87,
            1311.91,
            1476.72,
            1660.77,
            1862.11,
            2077.77,
            2304.41,
            2538.91,
            2778.75,
            3022.06,
            3267.58,
            3514.46,
            3762.18,
            4010.41,
            4258.95,
            4507.68,
            4756.52,
            5005.16,
            5253.84,
            5502.55,
            5751.27,
            6000,
        ]
    )

    return (layer_thickness, cell_center, cell_interface)


@pytest.fixture
def get_faulty_vgrid_elements():
    """Generate purposefully faulty vgrid elements. They follow appropriate mathematical rules but violate real world."""

    layer_thickness = np.array([1, 5, 6, 3, 5, 1, 0, -3, 5, 6])

    cell_center = np.array([0.5, 3.5, 8.5, 12.5, 16.5, 19.5, 20, 18.5, 19.5, 25])

    cell_interface = np.array([0, 1, 6, 11, 14, 19, 20, 20, 17, 22, 28])

    return (layer_thickness, cell_center, cell_interface)


@pytest.fixture
def get_rect_grid():
    grid = Grid(
        resolution=0.1,
        xstart=278.0,
        lenx=4.0,
        ystart=7.0,
        leny=3.0,
        name="panama1",
    )
    return grid


@pytest.fixture
def get_rect_topo(get_rect_grid, tmp_path):
    topo = Topo(get_rect_grid, min_depth=0, version_control_dir=tmp_path)
    topo.set_flat(1000)
    return topo
