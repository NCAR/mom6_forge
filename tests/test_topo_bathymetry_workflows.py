import pytest
import xarray as xr
import numpy as np
from mom6_forge.topo import *
from mom6_forge._source_bathy import SourceBathy


def test_generate_mask_ocean_frac_raises_without_src(get_rect_topo):
    """generate_mask_from_stats_ocean_frac must raise if src has not been set."""
    with pytest.raises(AssertionError, match="Source bathymetry"):
        get_rect_topo.generate_mask_from_stats_ocean_frac()


def test_generate_mask_ocean_frac_raises_without_stats(
    get_rect_topo, synthetic_bathy_file
):
    """generate_mask_from_stats_ocean_frac must raise if compute_stats has not been called."""
    get_rect_topo._src = SourceBathy(
        get_rect_topo, synthetic_bathy_file, depth_name="elevation"
    )
    with pytest.raises(AssertionError, match="compute_stats"):
        get_rect_topo.generate_mask_from_stats_ocean_frac()


def test_generate_mask_ocean_frac_returns_binary_mask(
    get_rect_topo, synthetic_bathy_file
):
    """Mask values must be 0 (land) or 1 (ocean) only."""
    get_rect_topo._src = SourceBathy(
        get_rect_topo, synthetic_bathy_file, depth_name="elevation"
    )
    get_rect_topo.compute_stats(
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
        stats = topo.compute_stats(nx_sub=nx_sub, ny_sub=ny_sub, mask_hmin=0.0)

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
        stats2 = topo.compute_stats(nx_sub=nx_sub, ny_sub=ny_sub, mask_hmin=0.0)
        # Should be the exact same object (cached)
        assert stats2 is stats


def test_direct_cressman_interp(get_rect_topo, synthetic_bathy_file, tmp_path):
    """Smoke test: direct_cressman_interp runs end-to-end and updates topo depth."""
    topo = get_rect_topo  # flat 1000 m depth, all ocean

    # synthetic_bathy_file stores elevation positive-upward (ocean = -500 m);
    # is_input_positive_below_msl=False flips sign so ocean cells have depth +500 m,
    # which is what Cressman requires (source ocean = depth > 0).
    src = SourceBathy(
        topo,
        synthetic_bathy_file,
        depth_name="elevation",
        is_input_positive_below_msl=False,
    )
    topo._src = src

    old_depth = topo.depth.copy()
    weights_path = tmp_path / "weights.nc"

    topo.direct_cressman_interp(weights_path=weights_path)

    # weights file was written
    assert weights_path.exists()

    # depth was modified from the initial flat 1000 m
    assert not topo.depth.equals(old_depth)

    # output depth is a DataArray with the correct shape
    assert isinstance(topo.depth, xr.DataArray)
    assert topo.depth.shape == old_depth.shape

    # Depth values are dominated by the ~500 m source ocean; the mean should
    # be close to 500 m (the synthetic ocean floor depth).
    assert np.nanmean(topo.depth.values) == pytest.approx(500.0, abs=100.0)


def test_set_depth_from_stats(get_rect_topo, synthetic_bathy_file):
    """Test set_depth_from_stats sets topo depth to the chosen statistic from compute_stats."""
    topo = get_rect_topo

    # Load source bathymetry and slice to topo domain
    src = SourceBathy(
        topo,
        synthetic_bathy_file,
        depth_name="elevation",
        is_input_positive_below_msl=False,
    )
    topo.src = src
    topo.compute_stats(nx_sub=2, ny_sub=2, mask_hmin=0.0)

    topo.set_depth_from_stats("mean")

    mask = ~np.isnan(topo.depth.values)
    assert np.isclose(
        topo.depth.values[mask], topo.src.stats["D_mean"].values[mask]
    ).all()
