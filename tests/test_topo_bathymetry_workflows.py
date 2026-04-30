import numpy as np
import pytest
from mom6_forge.topo import *
from mom6_forge._source_bathy import SourceBathy


@pytest.fixture
def small_grid():
    """Small rectangular grid over a Gulf of Mexico sub-region for fast tests."""
    return Grid(
        resolution=0.5,
        xstart=260.0,
        lenx=10.0,
        ystart=18.0,
        leny=8.0,
        name="gulf_test",
    )


@pytest.fixture
def small_topo(small_grid, tmp_path):
    """Flat 1000 m topo on the small grid with version control."""
    topo = Topo(small_grid, min_depth=5.0, version_control_dir=tmp_path)
    topo.set_flat(1000)
    return topo


@pytest.fixture
def synthetic_gebco(tmp_path):
    """
    Write a minimal synthetic GEBCO-style netCDF for testing.
    Covers the small_grid domain with 0.05-degree resolution.
    Depths are negative (elevation convention): ocean cells < 0, land >= 0.
    A strip of land is included to test masking behaviour.
    """
    lons = np.arange(259.5, 271.0, 0.05)
    lats = np.arange(17.5, 27.0, 0.05)
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Ocean everywhere, with a land strip at lon 265-266
    elevation = np.where(
        (lon2d >= 265.0) & (lon2d <= 266.0),
        100.0,  # land
        -1000.0,  # ocean
    ).astype("float32")

    ds = xr.Dataset(
        {"elevation": (["lat", "lon"], elevation)},
        coords={"lon": lons, "lat": lats},
    )
    ds.elevation.attrs["units"] = "m"
    path = tmp_path / "synthetic_gebco.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def src_bathy(small_topo, synthetic_gebco):
    """SourceBathy sliced to the small_topo domain."""
    return SourceBathy(synthetic_gebco).slice_to_domain(small_topo)


def test_generate_mask_ocean_frac_returns_binary_mask(small_topo, src_bathy):
    """Mask values must be 0 (land) or 1 (ocean) only."""
    small_topo._src = src_bathy  # Set the source bathy for the topo
    mask = small_topo.generate_mask_from_stats_oceanfrac(nx_sub=3, ny_sub=3)
    assert set(np.unique(mask.values)).issubset({0, 1})


def test_compute_topo_stats(small_topo, src_bathy):
    """Test _compute_topo_stats: per-cell depth statistics via xesmf nearest neighbor regridding.

    This test validates the refactored _compute_topo_stats method which:
    - Generates sub-points within each grid cell
    - Uses xesmf nearest_s2d regridding to snap sub-points to nearest source data
    - Computes per-cell statistics (OCN_FRAC, D_mean, D_min, D_max, D2_mean)
    """
    topo = small_topo

    # Load source bathymetry and slice to topo domain
    src = src_bathy
    src.slice_to_domain(topo, buf=0.5)
    topo._src = src

    # Test with different sub-sampling densities
    for nx_sub, ny_sub in [(2, 2), (3, 3)]:
        # Call _compute_topo_stats
        stats = topo._compute_stats(nx_sub=nx_sub, ny_sub=ny_sub, mask_hmin=0.0)

        # Verify output is a Dataset with expected variables
        assert isinstance(stats, xr.Dataset)
        required_vars = ["OCN_FRAC", "D_mean", "D_min", "D_max", "D2_mean"]
        for var in required_vars:
            assert var in stats.data_vars, f"Missing {var} in output"

        # Verify shapes match topo grid
        expected_shape = (topo.depth.shape[0], topo.depth.shape[1])
        assert stats["OCN_FRAC"].shape == expected_shape
        assert stats["D_mean"].shape == expected_shape
        assert stats["D_min"].shape == expected_shape
        assert stats["D_max"].shape == expected_shape
        assert stats["D2_mean"].shape == expected_shape

        # Verify OCN_FRAC is between 0 and 1
        assert (stats["OCN_FRAC"] >= 0).all()
        assert (stats["OCN_FRAC"] <= 1).all()

        # Verify D_min <= D_mean <= D_max
        ocean_cells = stats["OCN_FRAC"].values > 0
        assert (
            stats["D_min"].values[ocean_cells] <= stats["D_mean"].values[ocean_cells]
        ).all()
        assert (
            stats["D_mean"].values[ocean_cells] <= stats["D_max"].values[ocean_cells]
        ).all()

        # Verify caching: second call should return cached result
        stats2 = topo._compute_stats(nx_sub=nx_sub, ny_sub=ny_sub, mask_hmin=0.0)
        # Should be the exact same object (cached)
        assert stats2 is stats
