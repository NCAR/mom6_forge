import numpy as np
import pytest
from pathlib import Path
import xarray as xr
from mom6_forge.grid import Grid
from mom6_forge.topo import Topo
from mom6_forge.topo_editor import TopoEditor
from mom6_forge._source_bathy import SourceBathy


@pytest.fixture
def get_editor(get_rect_topo_with_vc):
    return TopoEditor(get_rect_topo_with_vc, build_ui=False)


@pytest.fixture
def get_curvilinear_supergrid():
    """
    Synthetic supergrid uniformly rotated by 30 degrees from East.
    Every point has angle_dx = 30 degrees by construction, giving a known
    ground truth to test against.
    """
    rotation_angle = 30.0  # degrees
    nx, ny = 10, 10  # model cells
    dx = dy = 0.1  # half-cell spacing in degrees
    center_x, center_y = 10.0, 10.0  # well away from lat=0 to avoid cos issues

    θ = np.deg2rad(rotation_angle)
    nxp, nyp = 2 * nx + 1, 2 * ny + 1

    # Build rotated grid via meshgrid
    i_offsets = (np.arange(nxp) - nx) * dx
    j_offsets = (np.arange(nyp) - ny) * dy
    I, J = np.meshgrid(i_offsets, j_offsets)

    x = center_x + I * np.cos(θ) - J * np.sin(θ)
    y = center_y + I * np.sin(θ) + J * np.cos(θ)
    angle_dx = np.full((nyp, nxp), rotation_angle)

    return xr.Dataset(
        {
            "x": (["nyp", "nxp"], x),
            "y": (["nyp", "nxp"], y),
            "angle_dx": (["nyp", "nxp"], angle_dx),
        }
    )


# ---------------------------------------------------------------------------
# Vertical grid (vgrid)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def get_realistic_vgrid_elements():
    """65-layer Caribbean vertical grid: (layer_thickness, cell_center, cell_interface).

    Use for any test that needs physically realistic z* coordinates — layer
    thicknesses grow from 2.5 m near the surface to ~249 m at depth, reaching
    6000 m total.  The values are derived from an actual MOM6 Caribbean
    configuration, so they exercise real-world layer-count and depth-range
    edge cases.  Pure data; safe to share across the entire session.
    """
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


@pytest.fixture(scope="session")
def get_faulty_vgrid_elements():
    """10-layer vgrid with deliberate physical violations (zero and negative thicknesses).

    Use for tests that validate VGrid rejection / error-handling logic.  The
    arrays are arithmetically self-consistent so they pass low-level shape
    checks, but contain a zero-thickness layer and a negative-thickness layer
    that should trigger domain-validity errors.  Pure data; safe to share
    across the entire session.
    """
    layer_thickness = np.array([1, 5, 6, 3, 5, 1, 0, -3, 5, 6])
    cell_center = np.array([0.5, 3.5, 8.5, 12.5, 16.5, 19.5, 20, 18.5, 19.5, 25])
    cell_interface = np.array([0, 1, 6, 11, 14, 19, 20, 20, 17, 22, 28])
    return (layer_thickness, cell_center, cell_interface)


# ---------------------------------------------------------------------------
# Panama / rectangular region  (0.1°, 278-282°E, 7-10°N)
# ---------------------------------------------------------------------------


@pytest.fixture
def get_rect_grid():
    """0.1° rectilinear Grid over the Panama region (278-282°E, 7-10°N), 40×30 cells.

    The general-purpose horizontal grid fixture.  Use it when a test needs a
    realistic mid-resolution regional domain — grid slicing, segment metadata,
    supergrid properties, or as the base for get_rect_topo.  The Panama region
    is free of polar singularities and has well-behaved Mercator geometry.
    """
    grid = Grid(
        resolution=0.1,
        xstart=278.0,
        lenx=4.0,
        ystart=7.0,
        leny=3.0,
        name="panama1",
    )
    return grid


@pytest.fixture(scope="module")
def synthetic_bathy_file(tmp_path_factory):
    """80×70 NetCDF bathymetry covering the Panama region (276-284°E, 5-12°N).

    Pairs with get_rect_grid / get_rect_topo.  Use for tests that exercise
    SourceBathy ingestion, stat computation, mask generation from ocean
    fraction, or set_from_dataset on a realistically-sized source grid.
    Contains a synthetic circular island near (280°E, 8.5°N) so that
    land/ocean masking logic sees at least one non-trivial land cell.
    Elevation is positive-upward (GEBCO convention): ocean = -500 m, island
    = +200 m.  Written once per test module.
    """
    bathy_file = str(tmp_path_factory.mktemp("bathy") / "synthetic_bathy.nc")

    lon = np.linspace(276, 284, 80)
    lat = np.linspace(5, 12, 70)
    elevation = np.full((len(lat), len(lon)), -500.0)

    lon_2d, lat_2d = np.meshgrid(lon, lat)
    island_mask = (lon_2d - 280) ** 2 + (lat_2d - 8.5) ** 2 < 0.5
    elevation[island_mask] = 200.0

    ds = xr.Dataset(
        {"elevation": (["lat", "lon"], elevation)},
        coords={"lon": lon, "lat": lat},
    )
    ds.to_netcdf(bathy_file)
    yield bathy_file
    Path(bathy_file).unlink()


@pytest.fixture
def get_rect_topo_with_vc(get_rect_grid, tmp_path):
    topo = Topo(get_rect_grid, min_depth=0, version_control_dir=tmp_path, git=True)
    topo.set_flat(1000)
    return topo


@pytest.fixture
def get_rect_topo_without_vc(get_rect_grid):
    topo = Topo(get_rect_grid, min_depth=0, git=False)
    topo.set_flat(1000)
    return topo


# ---------------------------------------------------------------------------
# Tiny equatorial Pacific  (1°, 170-173°E, -3-0°N)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_topo():
    """3×3 cell, 1° Topo over open equatorial Pacific (170-173°E, -3-0°N), no git.

    Preferred fixture for parametrized set_from_dataset tests.  The 9-cell
    domain is ~180× smaller than get_rect_topo, which matters when
    set_from_dataset runs once per (depth_method × mask_method) combination.
    All ocean — no land cells — so masking logic stays out of the way unless
    the test explicitly sets a mask.
    """
    grid = Grid(
        resolution=1.0, xstart=170.0, lenx=3.0, ystart=-3.0, leny=3.0, name="tiny"
    )
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)
    return topo


@pytest.fixture(scope="module")
def tiny_bathy_file(tmp_path_factory):
    """17×17 NetCDF bathymetry at 0.25° spacing covering 169.5-173.5°E, -3.5-0.5°N.

    Pairs with tiny_topo.  Use alongside tiny_topo for fast parametrized
    set_from_dataset tests.  Entirely ocean at -1000 m (positive-upward,
    GEBCO convention) with a 0.5° buffer around the tiny_topo domain so the
    regridder always has source points near every destination cell.  Written
    once per test module.
    """
    path = str(tmp_path_factory.mktemp("bathy") / "tiny_bathy.nc")

    lon = np.linspace(169.5, 173.5, 17)
    lat = np.linspace(-3.5, 0.5, 17)
    elevation = np.full((17, 17), -1000.0)

    ds = xr.Dataset(
        {"elevation": (["lat", "lon"], elevation)},
        coords={"lon": lon, "lat": lat},
    )
    ds.to_netcdf(path)
    yield path
    Path(path).unlink()


# ---------------------------------------------------------------------------
# Supergrid (rotation tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def get_curvilinear_supergrid():
    """10×10 synthetic supergrid uniformly rotated 30° from East.

    Use for rotation-angle unit tests.  Every point has angle_dx = 30° by
    construction, providing an exact analytical ground truth to compare
    against computed rotation angles.  The grid is centred at (10°E, 10°N)
    to stay well away from the equator (where cos(lat) → 1 trivially) and
    from the poles.  Returns an xr.Dataset with x, y, and angle_dx on the
    supergrid (nyp × nxp) stencil.
    """
    rotation_angle = 30.0
    nx, ny = 10, 10
    dx = dy = 0.1
    center_x, center_y = 10.0, 10.0

    θ = np.deg2rad(rotation_angle)
    nxp, nyp = 2 * nx + 1, 2 * ny + 1

    i_offsets = (np.arange(nxp) - nx) * dx
    j_offsets = (np.arange(nyp) - ny) * dy
    I, J = np.meshgrid(i_offsets, j_offsets)

    x = center_x + I * np.cos(θ) - J * np.sin(θ)
    y = center_y + I * np.sin(θ) + J * np.cos(θ)
    angle_dx = np.full((nyp, nxp), rotation_angle)

    return xr.Dataset(
        {
            "x": (["nyp", "nxp"], x),
            "y": (["nyp", "nxp"], y),
            "angle_dx": (["nyp", "nxp"], angle_dx),
        }
    )


# ---------------------------------------------------------------------------
# Mapping / seam grids  (all module-scoped, read-only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def get_simple_grid():
    """2×2 cell, 1° Grid at (1-3°E, 1-3°N) — minimal domain for mapping tests.

    Use when a test only needs a small destination grid to check regridding
    logic, subgrid-point generation, or weight lookups.  The tiny footprint
    keeps test runtime low for operations (like Cressman weighting) that scale
    with destination-cell count.
    """
    grid = Grid(
        resolution=1,
        xstart=1,
        lenx=2,
        ystart=1,
        leny=2,
        name="simple",
    )
    return grid


@pytest.fixture(scope="module")
def get_simple_global_grid():
    """Cyclic 360°×10° band (1° resolution, equatorial strip -5–5°N).

    Use when a test needs a domain that wraps in longitude (cyclic_x=True)
    but does not need global lat coverage.  The reduced latitude extent
    (10° instead of 180°) makes this ~18× cheaper to construct than a full
    global grid while still exercising all cyclic-wraparound code paths.
    """
    grid = Grid(
        resolution=1,
        xstart=0,
        lenx=360,
        ystart=-5,
        leny=10,
        name="global",
        cyclic_x=True,
    )
    return grid


@pytest.fixture(scope="module")
def get_PM_seam_grid():
    """2×2 grid straddling the Prime Meridian seam (359-361°E, -1-1°N).

    Use to verify that longitude arithmetic handles the 0°/360° wraparound
    correctly — e.g. subgrid-point generation or weight lookups near the
    seam.  Pair with get_dateline_seam_grid to distinguish PM-seam bugs from
    dateline-seam bugs.
    """
    grid = Grid(
        resolution=1,
        xstart=359,
        lenx=2,
        ystart=-1,
        leny=2,
        name="pm_seam",
    )
    return grid


@pytest.fixture(scope="module")
def get_dateline_seam_grid():
    """2×2 grid straddling the dateline / antimeridian seam (-1-1°E, -1-1°N).

    Use to verify that longitude arithmetic handles the ±180° wraparound
    correctly.  Pair with get_PM_seam_grid to distinguish dateline-seam bugs
    from Prime-Meridian-seam bugs.
    """
    grid = Grid(
        resolution=1,
        xstart=-1,
        lenx=2,
        ystart=-1,
        leny=2,
        name="dateline_seam",
    )
    return grid
