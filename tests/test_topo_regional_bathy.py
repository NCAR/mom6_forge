"""
Tests for regional bathymetry pipeline methods on Topo.

These tests cover the new mask generation methods, Cressman interpolation,
the two named pipeline methods, and topo drag output. They are written ahead
of implementation and will fail until the corresponding methods are added to
mom6_forge/topo.py.

New methods under test:
    Topo.generate_mask_ocean_frac()
    Topo.generate_mask_cartopy()
    Topo.cressman_interp()
    Topo.direct_xesmf_regrid()     (renamed from set_from_dataset)
    Topo.high_res_regrid()
    Topo.write_topo_drag()
"""

import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from mom6_forge.grid import Grid
from mom6_forge.topo import Topo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        100.0,    # land
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


# ---------------------------------------------------------------------------
# generate_mask_ocean_frac
# ---------------------------------------------------------------------------


class TestGenerateMaskOceanFrac:

    def test_returns_binary_mask(self, small_topo, synthetic_gebco):
        """Mask values must be 0 (land) or 1 (ocean) only."""
        mask, ocn_frac = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        unique = np.unique(mask.values)
        assert set(unique).issubset({0, 1}), f"Unexpected mask values: {unique}"

    def test_ocn_frac_bounds(self, small_topo, synthetic_gebco):
        """OCN_FRAC values must be in [0, 1]."""
        _, ocn_frac = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        assert float(ocn_frac.min()) >= 0.0
        assert float(ocn_frac.max()) <= 1.0

    def test_mask_shape_matches_grid(self, small_topo, synthetic_gebco):
        """Mask must have the same spatial shape as the model grid."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        assert mask.shape == (small_topo._grid.ny, small_topo._grid.nx)

    def test_land_strip_is_masked(self, small_topo, synthetic_gebco):
        """Cells over the synthetic land strip should have mask=0."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=5,
            ny_sub=5,
        )
        # Find cells whose centre longitude falls in the land strip
        land_cols = np.where(
            (small_topo._grid.tlon.values >= 265.0)
            & (small_topo._grid.tlon.values <= 266.0)
        )
        assert (mask.values[land_cols] == 0).all()

    def test_threshold_controls_mask(self, small_topo, synthetic_gebco):
        """Higher threshold should produce equal or fewer ocean cells."""
        mask_loose, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
            mask_threshold=0.2,
        )
        mask_strict, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
            mask_threshold=0.8,
        )
        assert mask_loose.values.sum() >= mask_strict.values.sum()

    def test_stats_available_after_call(self, small_topo, synthetic_gebco):
        """D_mean, D_min, D_max, D2_mean should be stored on the Topo object."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        assert hasattr(small_topo, "d_mean")
        assert hasattr(small_topo, "d_min")
        assert hasattr(small_topo, "d_max")
        assert hasattr(small_topo, "d2_mean")

    def test_stats_shapes_match_grid(self, small_topo, synthetic_gebco):
        """Depth statistics must have the same shape as the model grid."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        shape = (small_topo._grid.ny, small_topo._grid.nx)
        assert small_topo.d_mean.shape == shape
        assert small_topo.d2_mean.shape == shape


# ---------------------------------------------------------------------------
# generate_mask_cartopy
# ---------------------------------------------------------------------------


class TestGenerateMaskCartopy:

    def test_returns_binary_mask(self, small_topo):
        """Mask values must be 0 or 1."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        unique = np.unique(mask.values)
        assert set(unique).issubset({0, 1}), f"Unexpected mask values: {unique}"

    def test_mask_shape_matches_grid(self, small_topo):
        """Mask must have the same spatial shape as the model grid."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        assert mask.shape == (small_topo._grid.ny, small_topo._grid.nx)

    def test_open_ocean_is_unmasked(self, small_topo):
        """Deep Gulf of Mexico cells should be ocean (mask=1)."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        # Find a cell near the Gulf centre (~25N, ~270E)
        j = np.argmin(np.abs(small_topo._grid.tlat[:, 0].values - 25.0))
        i = np.argmin(np.abs(small_topo._grid.tlon[0, :].values - 270.0))
        assert mask.values[j, i] == 1

    def test_resolution_options(self, small_topo):
        """All supported Cartopy Natural Earth resolutions should work."""
        for res in ("10m", "50m", "110m"):
            mask = small_topo.generate_mask_cartopy(resolution=res)
            assert mask is not None


# ---------------------------------------------------------------------------
# cressman_interp
# ---------------------------------------------------------------------------


class TestCressmanInterp:

    def test_ocean_cells_get_nonzero_depth(self, small_topo, synthetic_gebco):
        """All cells with mask=1 should have a positive depth after Cressman."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        small_topo.cressman_interp(
            bathymetry_path=synthetic_gebco,
            mask=mask,
            hmin=5.0,
        )
        ocean_depths = small_topo.depth.values[mask.values == 1]
        assert (ocean_depths > 0).all()

    def test_land_cells_have_zero_depth(self, small_topo, synthetic_gebco):
        """Cells with mask=0 must have depth=0 after Cressman."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        small_topo.cressman_interp(
            bathymetry_path=synthetic_gebco,
            mask=mask,
            hmin=5.0,
        )
        land_depths = small_topo.depth.values[mask.values == 0]
        assert (land_depths == 0).all()

    def test_min_depth_enforced(self, small_topo, synthetic_gebco):
        """No ocean cell should be shallower than hmin."""
        hmin = 10.0
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        small_topo.cressman_interp(
            bathymetry_path=synthetic_gebco,
            mask=mask,
            hmin=hmin,
        )
        ocean_depths = small_topo.depth.values[mask.values == 1]
        assert (ocean_depths >= hmin).all()

    def test_smooth_scl_affects_result(self, small_topo, synthetic_gebco):
        """Different smooth_scl values should produce different depth fields."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        small_topo.cressman_interp(
            bathymetry_path=synthetic_gebco, mask=mask, hmin=5.0, smooth_scl=1.0
        )
        depth_tight = small_topo.depth.values.copy()

        small_topo.cressman_interp(
            bathymetry_path=synthetic_gebco, mask=mask, hmin=5.0, smooth_scl=4.0
        )
        depth_wide = small_topo.depth.values.copy()

        assert not np.allclose(depth_tight, depth_wide)

    def test_output_shape_matches_grid(self, small_topo, synthetic_gebco):
        """Depth field after Cressman must have the correct grid shape."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        small_topo.cressman_interp(
            bathymetry_path=synthetic_gebco, mask=mask, hmin=5.0
        )
        assert small_topo.depth.shape == (small_topo._grid.ny, small_topo._grid.nx)


# ---------------------------------------------------------------------------
# direct_xesmf_regrid  (renamed from set_from_dataset)
# ---------------------------------------------------------------------------


class TestDirectXesmfRegrid:

    def test_runs_without_external_mask(self, small_topo, synthetic_gebco, tmp_path):
        """Backwards-compatible: should work exactly as set_from_dataset did."""
        small_topo.direct_xesmf_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            output_dir=tmp_path,
        )
        assert not np.all(small_topo.depth.values == 0)

    def test_accepts_external_mask(self, small_topo, synthetic_gebco, tmp_path):
        """When an external mask is supplied, tidy_dataset should use it."""
        mask, _ = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        small_topo.direct_xesmf_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            mask=mask,
            output_dir=tmp_path,
        )
        # Cells masked as land should have depth = 0
        land_depths = small_topo.depth.values[mask.values == 0]
        assert (land_depths == 0).all()

    def test_external_mask_and_no_mask_differ(
        self, small_grid, synthetic_gebco, tmp_path
    ):
        """Using an external mask should produce a different result than no mask."""
        topo_no_mask = Topo(small_grid, min_depth=5.0, version_control_dir=tmp_path / "a")
        topo_no_mask.set_flat(1000)
        topo_no_mask.direct_xesmf_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            output_dir=tmp_path / "a",
        )

        topo_mask = Topo(small_grid, min_depth=5.0, version_control_dir=tmp_path / "b")
        topo_mask.set_flat(1000)
        mask, _ = topo_mask.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=5, ny_sub=5
        )
        topo_mask.direct_xesmf_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            mask=mask,
            output_dir=tmp_path / "b",
        )

        assert not topo_no_mask.depth.equals(topo_mask.depth)


# ---------------------------------------------------------------------------
# high_res_regrid
# ---------------------------------------------------------------------------


class TestHighResRegrid:

    def test_produces_valid_depth_field(self, small_topo, synthetic_gebco, tmp_path):
        """Pipeline should produce a depth field with no NaNs."""
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            nx_sub=3,
            ny_sub=3,
            output_dir=tmp_path,
        )
        assert not np.any(np.isnan(small_topo.depth.values))

    def test_min_depth_enforced(self, small_topo, synthetic_gebco, tmp_path):
        """No ocean cell should be shallower than topo.min_depth."""
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            nx_sub=3,
            ny_sub=3,
            output_dir=tmp_path,
        )
        ocean_mask = small_topo.tmask.values
        ocean_depths = small_topo.depth.values[ocean_mask == 1]
        assert (ocean_depths >= small_topo.min_depth).all()

    def test_stats_available_after_call(self, small_topo, synthetic_gebco, tmp_path):
        """Depth statistics must be populated after high_res_regrid."""
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            vertical_coordinate_name="elevation",
            nx_sub=3,
            ny_sub=3,
            output_dir=tmp_path,
        )
        assert hasattr(small_topo, "d_mean")
        assert hasattr(small_topo, "d2_mean")


# ---------------------------------------------------------------------------
# write_topo_drag
# ---------------------------------------------------------------------------


class TestWriteTopoDrag:

    def test_h2_written_to_file(self, small_topo, synthetic_gebco, tmp_path):
        """write_topo_drag should produce a netCDF with an h2 variable."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out_path = tmp_path / "topo_drag.nc"
        small_topo.write_topo_drag(out_path)
        ds = xr.open_dataset(out_path)
        assert "h2" in ds

    def test_h2_non_negative(self, small_topo, synthetic_gebco, tmp_path):
        """h2 = D2_mean - D_mean^2 is a variance and must be >= 0 everywhere."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out_path = tmp_path / "topo_drag.nc"
        small_topo.write_topo_drag(out_path)
        ds = xr.open_dataset(out_path)
        assert float(ds.h2.min()) >= 0.0

    def test_h2_shape_matches_grid(self, small_topo, synthetic_gebco, tmp_path):
        """h2 must have the same spatial shape as the model grid."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out_path = tmp_path / "topo_drag.nc"
        small_topo.write_topo_drag(out_path)
        ds = xr.open_dataset(out_path)
        assert ds.h2.shape == (small_topo._grid.ny, small_topo._grid.nx)

    def test_h2_units_attribute(self, small_topo, synthetic_gebco, tmp_path):
        """h2 should carry a units attribute of 'meters^2'."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out_path = tmp_path / "topo_drag.nc"
        small_topo.write_topo_drag(out_path)
        ds = xr.open_dataset(out_path)
        assert ds.h2.attrs.get("units") == "meters^2"

    def test_raises_without_stats(self, small_topo, tmp_path):
        """write_topo_drag must raise if generate_mask_ocean_frac was not called."""
        out_path = tmp_path / "topo_drag.nc"
        with pytest.raises(RuntimeError, match="generate_mask_ocean_frac"):
            small_topo.write_topo_drag(out_path)

    def test_h2_formula(self, small_topo, synthetic_gebco, tmp_path):
        """Verify h2 = D2_mean - D_mean^2 directly against stored stats."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out_path = tmp_path / "topo_drag.nc"
        small_topo.write_topo_drag(out_path)
        ds = xr.open_dataset(out_path)
        expected_h2 = small_topo.d2_mean.values - small_topo.d_mean.values ** 2
        np.testing.assert_allclose(ds.h2.values, expected_h2, rtol=1e-5)
