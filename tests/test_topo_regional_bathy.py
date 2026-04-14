"""
Tests for regional bathymetry pipeline methods on Topo.

Implemented and tested here:
    Topo.diagnose_resolution()
    Topo.generate_mask_ocean_frac()
    Topo.generate_mask_cartopy()
    Topo._compute_topo_stats()          (cache behaviour)
    Topo.tidy_dataset()                 (external mask param)
    Topo.set_from_dataset()             (dispatch to high_res vs direct)
    Topo.direct_xesmf_regrid()          (mask options, bad mask_method)
    Topo.high_res_regrid()              (end-to-end with ocean_frac / cartopy / pre-computed mask)
    Topo.mpi_direct_xesmf_regrid()      (renamed from mpi_set_from_dataset)
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


# ---------------------------------------------------------------------------
# generate_mask_ocean_frac
# ---------------------------------------------------------------------------


class TestGenerateMaskOceanFrac:

    def test_returns_binary_mask(self, small_topo, synthetic_gebco):
        """Mask values must be 0 (land) or 1 (ocean) only."""
        mask = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        assert set(np.unique(mask.values)).issubset({0, 1})

    def test_mask_shape_matches_grid(self, small_topo, synthetic_gebco):
        """Mask must have the same spatial shape as the model grid."""
        mask = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=3,
            ny_sub=3,
        )
        assert mask.shape == (small_topo._grid.ny, small_topo._grid.nx)

    def test_land_strip_is_masked(self, small_topo, synthetic_gebco):
        """Cells over the synthetic land strip should have mask=0."""
        mask = small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco,
            nx_sub=5,
            ny_sub=5,
        )
        land_cols = np.where(
            (small_topo._grid.tlon.values >= 265.0)
            & (small_topo._grid.tlon.values <= 266.0)
        )
        assert (mask.values[land_cols] == 0).all()

    def test_stats_stored_after_call(self, small_topo, synthetic_gebco):
        """_topo_stats Dataset with OCN_FRAC, D_mean, D_min, D_max, D2_mean
        must be available on the instance after the call."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        assert small_topo._topo_stats is not None
        for var in ("OCN_FRAC", "D_mean", "D_min", "D_max", "D2_mean"):
            assert var in small_topo._topo_stats

    def test_stats_cached(self, small_topo, synthetic_gebco, monkeypatch):
        """Calling generate_mask_ocean_frac twice must not re-run the computation."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        # Poison xr.open_dataset so a second file-read would raise
        monkeypatch.setattr(
            xr,
            "open_dataset",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("_compute_topo_stats ran again")
            ),
        )
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )

    def test_ocn_frac_in_bounds(self, small_topo, synthetic_gebco):
        """OCN_FRAC must be in [0, 1] for every cell."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        ocn_frac = small_topo._topo_stats["OCN_FRAC"].values
        assert float(ocn_frac.min()) >= 0.0
        assert float(ocn_frac.max()) <= 1.0


# ---------------------------------------------------------------------------
# generate_mask_cartopy
# ---------------------------------------------------------------------------


class TestGenerateMaskCartopy:

    def test_returns_binary_mask(self, small_topo):
        """Mask values must be 0 or 1."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        assert set(np.unique(mask.values)).issubset({0, 1})

    def test_mask_shape_matches_grid(self, small_topo):
        """Mask must have the same spatial shape as the model grid."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        assert mask.shape == (small_topo._grid.ny, small_topo._grid.nx)

    def test_open_ocean_is_unmasked(self, small_topo):
        """Deep Gulf of Mexico cells should be ocean (mask=1)."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        j = np.argmin(np.abs(small_topo._grid.tlat[:, 0].values - 25.0))
        i = np.argmin(np.abs(small_topo._grid.tlon[0, :].values - 270.0))
        assert mask.values[j, i] == 1


# ---------------------------------------------------------------------------
# h2 written via write_topo (enforce_topo_drag)
# ---------------------------------------------------------------------------


class TestTopoDragInWriteTopo:

    def test_h2_in_output_when_stats_computed(
        self, small_topo, synthetic_gebco, tmp_path
    ):
        """write_topo should include h2 when _topo_stats are available."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out = tmp_path / "topog.nc"
        small_topo.write_topo(out)
        assert "h2" in xr.open_dataset(out)

    def test_h2_non_negative(self, small_topo, synthetic_gebco, tmp_path):
        """h2 = D2_mean - D_mean^2 is a variance and must be >= 0."""
        small_topo.generate_mask_ocean_frac(
            bathymetry_path=synthetic_gebco, nx_sub=3, ny_sub=3
        )
        out = tmp_path / "topog.nc"
        small_topo.write_topo(out)
        assert float(xr.open_dataset(out).h2.min()) >= 0.0

    def test_enforce_topo_drag_raises_without_stats(self, small_topo, tmp_path):
        """enforce_topo_drag=True must raise if stats have not been computed."""
        with pytest.raises(RuntimeError, match="generate_mask_ocean_frac"):
            small_topo.write_topo(tmp_path / "topog.nc", enforce_topo_drag=True)


# ---------------------------------------------------------------------------
# tidy_dataset — external mask parameter
# ---------------------------------------------------------------------------


class TestTidyDatasetExternalMask:

    def test_external_mask_overrides_depth_sign(self, small_topo):
        """Cells marked as land in an external mask should end up depth=0
        even when the raw bathymetry depth is positive at those cells."""
        ny, nx = small_topo._grid.ny, small_topo._grid.nx
        bathy_ds = xr.Dataset(
            {"depth": (["ny", "nx"], np.full((ny, nx), 500.0))},
            coords={
                "lon": (["ny", "nx"], small_topo._grid.tlon.values),
                "lat": (["ny", "nx"], small_topo._grid.tlat.values),
            },
        )
        # Mark first column as land
        mask = xr.DataArray(np.ones((ny, nx), dtype=int), dims=["ny", "nx"])
        mask[:, 0] = 0

        small_topo.tidy_dataset(
            positive_down=True,
            vertical_coordinate_name="depth",
            bathymetry=bathy_ds,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            mask=mask,
        )
        assert (small_topo.depth.values[:, 0] == 0).all()
        assert (small_topo.depth.values[:, 1:] > 0).all()

    def test_no_mask_derives_from_depth_sign(self, small_topo):
        """Without an external mask, tidy_dataset must derive ocean from depth > 0."""
        ny, nx = small_topo._grid.ny, small_topo._grid.nx
        depth_arr = np.full((ny, nx), 500.0)
        depth_arr[:, 0] = -1.0  # first column is land by depth sign
        bathy_ds = xr.Dataset(
            {"depth": (["ny", "nx"], depth_arr)},
            coords={
                "lon": (["ny", "nx"], small_topo._grid.tlon.values),
                "lat": (["ny", "nx"], small_topo._grid.tlat.values),
            },
        )
        small_topo.tidy_dataset(
            positive_down=True,
            vertical_coordinate_name="depth",
            bathymetry=bathy_ds,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
        )
        assert (small_topo.depth.values[:, 0] == 0).all()
        assert (small_topo.depth.values[:, 1:] > 0).all()


# ---------------------------------------------------------------------------
# set_from_dataset — dispatch logic
# ---------------------------------------------------------------------------


class TestSetFromDatasetDispatch:

    def test_dispatches_to_high_res_when_recommended(
        self, small_topo, synthetic_gebco, monkeypatch
    ):
        """When diagnose_resolution returns True, high_res_regrid must be called."""
        monkeypatch.setattr(small_topo, "diagnose_resolution", lambda *a, **kw: True)
        called = []
        monkeypatch.setattr(
            small_topo, "high_res_regrid", lambda **kw: called.append(kw)
        )
        small_topo.set_from_dataset(bathymetry_path=synthetic_gebco)
        assert called, "high_res_regrid was not called"

    def test_dispatches_to_direct_xesmf_when_not_recommended(
        self, small_topo, synthetic_gebco, monkeypatch
    ):
        """When diagnose_resolution returns False, direct_xesmf_regrid must be called."""
        monkeypatch.setattr(small_topo, "diagnose_resolution", lambda *a, **kw: False)
        called = []
        monkeypatch.setattr(
            small_topo, "direct_xesmf_regrid", lambda **kw: called.append(kw)
        )
        small_topo.set_from_dataset(bathymetry_path=synthetic_gebco)
        assert called, "direct_xesmf_regrid was not called"

    def test_default_mask_method_ocean_frac_for_high_res(
        self, small_topo, synthetic_gebco, monkeypatch
    ):
        """When dispatching to high_res_regrid with no mask_method set, 'ocean_frac' is used."""
        monkeypatch.setattr(small_topo, "diagnose_resolution", lambda *a, **kw: True)
        captured = {}
        monkeypatch.setattr(
            small_topo, "high_res_regrid", lambda **kw: captured.update(kw)
        )
        small_topo.set_from_dataset(bathymetry_path=synthetic_gebco)
        assert captured.get("mask_method") == "ocean_frac"

    def test_no_default_mask_for_direct_xesmf(
        self, small_topo, synthetic_gebco, monkeypatch
    ):
        """When dispatching to direct_xesmf_regrid with no mask_method, None is forwarded
        so tidy_dataset falls back to deriving the mask from depth sign."""
        monkeypatch.setattr(small_topo, "diagnose_resolution", lambda *a, **kw: False)
        captured = {}
        monkeypatch.setattr(
            small_topo, "direct_xesmf_regrid", lambda **kw: captured.update(kw)
        )
        small_topo.set_from_dataset(bathymetry_path=synthetic_gebco)
        assert captured.get("mask_method") is None

    def test_explicit_mask_method_forwarded_to_direct(
        self, small_topo, synthetic_gebco, monkeypatch
    ):
        """An explicit mask_method is forwarded unchanged to direct_xesmf_regrid."""
        monkeypatch.setattr(small_topo, "diagnose_resolution", lambda *a, **kw: False)
        captured = {}
        monkeypatch.setattr(
            small_topo, "direct_xesmf_regrid", lambda **kw: captured.update(kw)
        )
        small_topo.set_from_dataset(
            bathymetry_path=synthetic_gebco, mask_method="cartopy"
        )
        assert captured.get("mask_method") == "cartopy"


# ---------------------------------------------------------------------------
# direct_xesmf_regrid
# ---------------------------------------------------------------------------


class TestDirectXesmfRegrid:

    def test_bad_mask_method_raises(self, small_topo, synthetic_gebco):
        """Unknown mask_method must raise ValueError."""
        with pytest.raises(ValueError, match="mask_method"):
            small_topo.direct_xesmf_regrid(
                bathymetry_path=synthetic_gebco,
                mask_method="invalid",
            )

    def test_precomputed_mask_bypasses_generation(
        self, small_topo, synthetic_gebco, monkeypatch
    ):
        """When mask= is provided, generate_mask_ocean_frac must not be called."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        called = []
        monkeypatch.setattr(
            small_topo, "generate_mask_ocean_frac", lambda **kw: called.append(kw)
        )
        monkeypatch.setattr(
            small_topo, "generate_mask_cartopy", lambda **kw: called.append(kw)
        )
        # We don't need direct_xesmf_regrid to complete — just check mask generation is skipped
        monkeypatch.setattr(
            small_topo,
            "config_dataset",
            lambda **kw: (_ for _ in ()).throw(StopIteration),
        )
        with pytest.raises(StopIteration):
            small_topo.direct_xesmf_regrid(
                bathymetry_path=synthetic_gebco,
                mask=mask,
                mask_method="ocean_frac",  # should be ignored because mask= is set
            )
        assert not called, "mask generation should not run when mask= is provided"


# ---------------------------------------------------------------------------
# high_res_regrid
# ---------------------------------------------------------------------------


class TestHighResRegrid:

    def test_bad_mask_method_raises(self, small_topo, synthetic_gebco):
        """Unknown mask_method must raise ValueError."""
        with pytest.raises(ValueError, match="mask_method"):
            small_topo.high_res_regrid(
                bathymetry_path=synthetic_gebco,
                mask_method="invalid",
            )

    def test_ocean_frac_mask_produces_valid_depth(
        self, small_topo, synthetic_gebco, tmp_path
    ):
        """End-to-end with ocean_frac mask: ocean cells must have depth >= min_depth."""
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            mask_method="ocean_frac",
            nx_sub=3,
            ny_sub=3,
            weights_path=tmp_path / "cressman_weights.nc",
        )
        ocean = small_topo.depth.values > 0
        assert ocean.any(), "No ocean cells after high_res_regrid"
        assert (small_topo.depth.values[ocean] >= small_topo.min_depth).all()

    def test_cartopy_mask_produces_valid_depth(
        self, small_topo, synthetic_gebco, tmp_path
    ):
        """End-to-end with cartopy mask: ocean cells must have depth >= min_depth."""
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            mask_method="cartopy",
            cartopy_resolution="50m",
            weights_path=tmp_path / "cressman_weights.nc",
        )
        ocean = small_topo.depth.values > 0
        assert ocean.any(), "No ocean cells after high_res_regrid with cartopy mask"
        assert (small_topo.depth.values[ocean] >= small_topo.min_depth).all()

    def test_precomputed_mask_accepted(self, small_topo, synthetic_gebco, tmp_path):
        """Passing a pre-computed mask directly should skip mask generation."""
        mask = small_topo.generate_mask_cartopy(resolution="50m")
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            mask=mask,
            weights_path=tmp_path / "cressman_weights.nc",
        )
        ocean = small_topo.depth.values > 0
        assert ocean.any(), "No ocean cells with pre-computed mask"

    def test_weights_file_written(self, small_topo, synthetic_gebco, tmp_path):
        """Cressman weights netCDF must be written to weights_path."""
        wp = tmp_path / "cressman_weights.nc"
        small_topo.high_res_regrid(
            bathymetry_path=synthetic_gebco,
            mask_method="ocean_frac",
            nx_sub=3,
            ny_sub=3,
            weights_path=wp,
        )
        assert wp.exists()
        ds = xr.open_dataset(wp)
        for var in ("S", "row", "col"):
            assert var in ds


# ---------------------------------------------------------------------------
# mpi_direct_xesmf_regrid rename
# ---------------------------------------------------------------------------


class TestMpiRename:

    def test_mpi_direct_xesmf_regrid_exists(self, small_topo):
        """mpi_direct_xesmf_regrid must be callable (renamed from mpi_set_from_dataset)."""
        assert callable(getattr(small_topo, "mpi_direct_xesmf_regrid", None))

    def test_mpi_set_from_dataset_removed(self, small_topo):
        """Old name mpi_set_from_dataset must no longer exist."""
        assert not hasattr(small_topo, "mpi_set_from_dataset")
