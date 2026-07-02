import pytest
import xarray as xr
import numpy as np
from mom6_forge.topo import *
from mom6_forge._source_bathy import SourceBathy
from mom6_forge.grid import Grid


def test_generate_mask_ocean_frac_raises_without_src(get_rect_topo_without_vc):
    """generate_mask_from_stats_ocean_frac must raise if src has not been set."""
    with pytest.raises(AssertionError, match="Source bathymetry"):
        get_rect_topo_without_vc.generate_mask_from_stats_ocean_frac()


def test_generate_mask_ocean_frac_raises_without_stats(
    get_rect_topo_without_vc, synthetic_bathy_file
):
    """generate_mask_from_stats_ocean_frac must raise if compute_stats has not been called."""
    get_rect_topo_without_vc._src = SourceBathy(
        get_rect_topo_without_vc, synthetic_bathy_file, depth_name="elevation"
    )
    with pytest.raises(AssertionError, match="compute_stats"):
        get_rect_topo_without_vc.generate_mask_from_stats_ocean_frac()


def test_generate_mask_ocean_frac_returns_binary_mask(
    get_rect_topo_without_vc, synthetic_bathy_file
):
    """Mask values must be 0 (land) or 1 (ocean) only."""
    get_rect_topo_without_vc._src = SourceBathy(
        get_rect_topo_without_vc, synthetic_bathy_file, depth_name="elevation"
    )
    get_rect_topo_without_vc.compute_stats(
        nx_sub=2, ny_sub=2, mask_hmin=0.0
    )  # Compute stats to populate cache
    mask = get_rect_topo_without_vc.generate_mask_from_stats_ocean_frac()
    assert set(np.unique(mask.values)).issubset({0, 1})


def test_compute_topo_stats(get_rect_topo_without_vc, synthetic_bathy_file):
    """Test _compute_topo_stats: per-cell depth statistics via xesmf nearest neighbor regridding.

    This test validates the refactored _compute_topo_stats method which:
    - Generates sub-points within each grid cell
    - Uses xesmf nearest_s2d regridding to snap sub-points to nearest source data
    - Computes per-cell statistics (OCN_FRAC, D_mean, D_min, D_max, D2_mean)
    """
    topo = get_rect_topo_without_vc

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


def test_direct_cressman_interp(
    get_rect_topo_without_vc, synthetic_bathy_file, tmp_path
):
    """Smoke test: direct_cressman_interp runs end-to-end and updates topo depth."""
    topo = get_rect_topo_without_vc  # flat 1000 m depth, all ocean

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


def test_set_depth_from_stats(get_rect_topo_without_vc, synthetic_bathy_file):
    """Test set_depth_from_stats sets topo depth to the chosen statistic from compute_stats."""
    topo = get_rect_topo_without_vc

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


def test_diagnose_resolution_below_threshold(
    get_rect_topo_without_vc, synthetic_bathy_file
):
    """When model and source have similar resolution, diagnose_resolution returns False."""
    # get_rect_grid is 0.1 deg; synthetic_bathy_file is also ~0.1 deg → ratio ~1x, below 12x
    get_rect_topo_without_vc.src = SourceBathy(
        get_rect_topo_without_vc, synthetic_bathy_file, depth_name="elevation"
    )
    result = get_rect_topo_without_vc.diagnose_resolution()
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


def test_set_from_dataset_stats_path(get_rect_topo_without_vc, synthetic_bathy_file):
    """set_from_dataset with explicit mask_method='ocean_frac' and depth_method='stats' sets depth from stats."""
    get_rect_topo_without_vc.set_from_dataset(
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
    assert get_rect_topo_without_vc.user_mask is not None
    assert not np.all(np.isnan(get_rect_topo_without_vc.depth.values))


def test_set_from_dataset_auto_fine_resolution(
    get_rect_topo_without_vc, synthetic_bathy_file, tmp_path
):
    """Auto path (None/None) on a fine grid: diagnose_resolution returns False → naturalearth mask + xesmf depth."""
    get_rect_topo_without_vc.set_from_dataset(
        bathymetry_path=synthetic_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        output_dir=tmp_path,
        regridding_method="bilinear",
    )
    assert get_rect_topo_without_vc.user_mask is not None
    assert not np.all(np.isnan(get_rect_topo_without_vc.depth.values))


def test_set_from_dataset_auto_coarse_resolution(synthetic_bathy_file, tmp_path):
    """Auto path (None/None) on a coarse grid: diagnose_resolution returns True → ocean_frac mask + cressman depth."""
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
    coarse_topo.set_from_dataset(
        bathymetry_path=synthetic_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        output_dir=tmp_path,
    )
    assert coarse_topo.user_mask is not None
    assert not np.all(np.isnan(coarse_topo.depth.values))


@pytest.mark.parametrize(
    "depth_method,extra_kwargs",
    [
        ("stats", {}),
        ("cressman", {}),
        ("xesmf", {"regridding_method": "bilinear"}),
    ],
    ids=["stats", "cressman", "xesmf"],
)
def test_set_from_dataset_each_depth_method(
    tiny_topo, tiny_bathy_file, tmp_path, depth_method, extra_kwargs
):
    """Each supported depth_method runs end-to-end with mask_method='ocean_frac'."""
    tiny_topo.set_from_dataset(
        bathymetry_path=tiny_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="ocean_frac",
        depth_method=depth_method,
        output_dir=tmp_path,
        nx_sub=2,
        ny_sub=2,
        mask_hmin=0.0,
        **extra_kwargs,
    )
    assert tiny_topo.user_mask is not None
    assert not np.all(np.isnan(tiny_topo.depth.values))


@pytest.mark.parametrize(
    "mask_method",
    ["naturalearth", "ocean_frac", "dataset"],
)
def test_set_from_dataset_each_mask_method(tiny_topo, tiny_bathy_file, mask_method):
    """Each supported mask_method runs end-to-end with depth_method='stats'."""
    tiny_topo.set_from_dataset(
        bathymetry_path=tiny_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method=mask_method,
        depth_method="stats",
        nx_sub=2,
        ny_sub=2,
        mask_hmin=0.0,
    )
    assert not np.all(np.isnan(tiny_topo.depth.values))


def test_set_from_dataset_mask_method_manual(tiny_topo, tiny_bathy_file):
    """set_from_dataset with mask_method='manual' uses the user_mask set before the call."""
    tiny_topo.set_src(tiny_bathy_file, "lon", "lat", "elevation")
    tiny_topo.compute_stats(nx_sub=2, ny_sub=2, mask_hmin=0.0)
    tiny_topo.user_mask = tiny_topo.generate_mask_from_stats_ocean_frac()

    tiny_topo.set_from_dataset(
        bathymetry_path=tiny_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="manual",
        depth_method="stats",
        nx_sub=2,
        ny_sub=2,
        mask_hmin=0.0,
    )
    assert tiny_topo.user_mask is not None
    assert not np.all(np.isnan(tiny_topo.depth.values))


def test_set_from_dataset_fill_channels_end_to_end(tiny_topo, tiny_bathy_file):
    """set_from_dataset with fill_channels=True invokes channel-filling at the end of the workflow."""
    tiny_topo.set_from_dataset(
        bathymetry_path=tiny_bathy_file,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask_method="ocean_frac",
        depth_method="stats",
        fill_channels=True,
        nx_sub=2,
        ny_sub=2,
        mask_hmin=0.0,
    )
    assert tiny_topo.user_mask is not None
    assert not np.all(np.isnan(tiny_topo.depth.values))


def test_fill_inland_lakes_and_channels_removes_lake():
    """fill_inland_lakes_and_channels converts a fully-enclosed inland lake cell to land."""
    grid = Grid(
        resolution=1.0,
        xstart=1.0,
        lenx=7.0,
        ystart=1.0,
        leny=7.0,
        name="lake_test",
    )
    topo = Topo(grid, min_depth=0, git=False)
    topo.set_flat(1000)

    # Build a mask: all ocean, with a ring of land at rows 2-4, cols 2-4,
    # leaving a single ocean cell at (row=3, col=3) enclosed inside the ring.
    mask = np.ones((grid.ny, grid.nx), dtype=int)
    mask[2, 2:5] = 0  # top edge of ring
    mask[4, 2:5] = 0  # bottom edge of ring
    mask[3, 2] = 0  # left edge of ring
    mask[3, 4] = 0  # right edge of ring
    # mask[3, 3] stays 1: the isolated lake

    topo.user_mask = mask
    assert topo.tmask.values[3, 3] == 1  # lake cell is ocean before fill

    topo.fill_inland_lakes_and_channels()

    assert topo.tmask.values[3, 3] == 0  # lake cell is land after fill


@pytest.mark.parametrize(
    "mask_method,depth_method,extra_kwargs,match",
    [
        (
            "invalid_mask",
            "stats",
            {"nx_sub": 2, "ny_sub": 2, "mask_hmin": 0.0},
            "naturalearth.*ocean_frac.*dataset.*manual",
        ),
        (
            "ocean_frac",
            "invalid_depth",
            {"nx_sub": 2, "ny_sub": 2, "mask_hmin": 0.0},
            "stats.*cressman.*xesmf",
        ),
    ],
    ids=["bad_mask_method", "bad_depth_method"],
)
def test_set_from_dataset_invalid_method_raises(
    tiny_topo, tiny_bathy_file, mask_method, depth_method, extra_kwargs, match
):
    """set_from_dataset raises ValueError whose message names the accepted values."""
    with pytest.raises(ValueError, match=match):
        tiny_topo.set_from_dataset(
            bathymetry_path=tiny_bathy_file,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            mask_method=mask_method,
            depth_method=depth_method,
            **extra_kwargs,
        )
