import numpy as np
import pytest
from mom6_forge.topo import *
from mom6_forge._source_bathy import SourceBathy
from mom6_forge.grid import Grid


def test_generate_mask_ocean_frac_returns_binary_mask(
    get_rect_topo, synthetic_bathy_file
):
    """Mask values must be 0 (land) or 1 (ocean) only."""
    get_rect_topo._src = SourceBathy(
        get_rect_topo, synthetic_bathy_file, depth_name="elevation"
    )
    get_rect_topo._compute_stats(
        nx_sub=2, ny_sub=2, mask_hmin=0.0
    )  # Compute stats to populate cache
    mask = get_rect_topo.generate_mask_from_stats_ocean_frac()
    assert set(np.unique(mask.values)).issubset({0, 1})


def test_compute_topo_stats(get_rect_topo, synthetic_bathy_file):
    """Test _compute_topo_stats: per-cell depth statistics via xesmf nearest neighbor regridding.

    This test validates the refactored _compute_topo_stats method which:
    - Generates sub-points within each grid cell
    - Uses xesmf nearest_s2d regridding to snap sub-points to nearest source data
    - Computes per-cell statistics (OCN_FRAC, D_mean, D_min, D_max, D2_mean)
    """
    topo = get_rect_topo

    # Load source bathymetry and slice to topo domain
    src = SourceBathy(topo, synthetic_bathy_file, depth_name="elevation")
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


def test_set_depth_from_stats(get_rect_topo, synthetic_bathy_file):
    """Test set_depth_from_stats sets topo depth to the chosen statistic from _compute_stats."""
    topo = get_rect_topo

    # Load source bathymetry and slice to topo domain
    src = SourceBathy(
        topo,
        synthetic_bathy_file,
        depth_name="elevation",
        is_input_positive_below_msl=False,
    )
    topo.src = src
    topo._compute_stats(nx_sub=2, ny_sub=2, mask_hmin=0.0)

    topo.set_depth_from_stats("mean")

    mask = ~np.isnan(topo.depth.values)
    assert np.isclose(
        topo.depth.values[mask], topo.src.stats["D_mean"].values[mask]
    ).all()


def test_diagnose_resolution_below_threshold(get_rect_topo, synthetic_bathy_file):
    """When model and source have similar resolution, diagnose_resolution returns False."""
    # get_rect_grid is 0.1 deg; synthetic_bathy_file is also ~0.1 deg → ratio ~1x, below 12x
    get_rect_topo.src = SourceBathy(
        get_rect_topo, synthetic_bathy_file, depth_name="elevation"
    )
    result = get_rect_topo.diagnose_resolution()
    assert result is False


def test_diagnose_resolution_above_threshold(synthetic_bathy_file, tmp_path):
    """When model cells are much coarser than the source, diagnose_resolution returns True."""
    # 2-degree model over the Panama region → ~222 km cells vs ~11 km source → ratio ~20x
    coarse_grid = Grid(
        resolution=2.0,
        xstart=278.0,
        lenx=4.0,
        ystart=7.0,
        leny=4.0,
        name="coarse_test",
    )
    coarse_topo = Topo(coarse_grid, min_depth=0, version_control_dir=tmp_path)
    coarse_topo.set_flat(1000)
    coarse_topo.src = SourceBathy(
        coarse_topo, synthetic_bathy_file, depth_name="elevation"
    )
    result = coarse_topo.diagnose_resolution()
    assert result is True


def test_set_from_dataset_stats_path(get_rect_topo, synthetic_bathy_file):
    """set_from_dataset with explicit mask_method='ocean_frac' and depth_method='stats' sets depth from stats."""
    get_rect_topo.set_from_dataset(
        bathymetry_path=synthetic_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="ocean_frac",
        depth_method="stats",
        nx_sub=2,
        ny_sub=2,
        mask_hmin=0.0,
    )
    # Depth should be set (not all NaN) and user_mask should be populated
    assert get_rect_topo.user_mask is not None
    assert not np.all(np.isnan(get_rect_topo.depth.values))
