from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from mom6_bathy import mapping

TX2_3_MESH = Path("/Users/altuntas/work/meshes/tx2_3v2_230415_ESMFmesh.nc")


# ---------------------------------------------------------------------------
# flatten_to_mesh
# ---------------------------------------------------------------------------

def test_flatten_to_mesh_c_order():
    """flatten_to_mesh must use row-major (C) ordering."""
    field_2d = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    result = mapping.flatten_to_mesh(field_2d)
    np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])


def test_flatten_to_mesh_dataarray():
    """flatten_to_mesh accepts xr.DataArray and returns a numpy array."""
    da = xr.DataArray(
        np.array([[10, 20], [30, 40]]),
        dims=("nlat", "nlon"),
    )
    result = mapping.flatten_to_mesh(da)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [10, 20, 30, 40])


def test_flatten_to_mesh_roundtrip():
    """Reshape to (ny, nx) then flatten_to_mesh must recover the original 1D array."""
    original_1d = np.arange(24)
    field_2d = original_1d.reshape((4, 6), order="C")
    recovered = mapping.flatten_to_mesh(field_2d)
    np.testing.assert_array_equal(recovered, original_1d)


# ---------------------------------------------------------------------------
# grid_from_esmf_mesh / flatten_to_mesh roundtrip  (integration, real mesh)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not TX2_3_MESH.exists(),
    reason=f"Mesh file not found: {TX2_3_MESH}",
)
def test_grid_from_esmf_mesh_flatten_to_mesh_mask_roundtrip():
    """grid_from_esmf_mesh followed by flatten_to_mesh must recover the
    original elementMask exactly."""
    mesh = xr.open_dataset(TX2_3_MESH)

    original_mask_1d = mesh["elementMask"].values  # shape (n_elements,)

    # 1D -> 2D
    grid_2d = mapping.grid_from_esmf_mesh(mesh)
    mask_2d = grid_2d["mask"]  # xr.DataArray, shape (ny, nx)

    # 2D -> 1D using the standardized helper
    recovered_mask_1d = mapping.flatten_to_mesh(mask_2d)

    assert recovered_mask_1d.shape == original_mask_1d.shape, (
        f"Shape mismatch: got {recovered_mask_1d.shape}, "
        f"expected {original_mask_1d.shape}"
    )
    np.testing.assert_array_equal(
        recovered_mask_1d,
        original_mask_1d,
        err_msg="Roundtrip 1D->2D->1D changed elementMask values",
    )


def test_generate_esmf_map_via_xesmf_coastline_masking_only_coastal_nonzero(monkeypatch):
    """When coastline_masking=True, nonzero destination rows in generated
    weights should only occur on coastal destination cells."""

    src_grid = xr.Dataset(
        data_vars={"mask": (("nlat", "nlon"), np.ones((2, 2), dtype=int))},
        coords={
            "lon": (("nlat", "nlon"), np.array([[0.0, 1.0], [2.0, 3.0]])),
            "lat": (("nlat", "nlon"), np.array([[10.0, 10.0], [11.0, 11.0]])),
        },
    )
    dst_grid = xr.Dataset(
        data_vars={"mask": (("nlat", "nlon"), np.ones((2, 3), dtype=int))},
        coords={
            "lon": (("nlat", "nlon"), np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
            "lat": (("nlat", "nlon"), np.array([[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]])),
        },
    )
    coastline_mask = xr.DataArray(
        np.array([[0, 1, 0], [0, 1, 0]], dtype=int),
        dims=("nlat", "nlon"),
    )

    def _fake_grid_from_esmf_mesh(mesh):
        return src_grid if "src" in str(mesh) else dst_grid

    def _fake_extract_coastline_mask(grid):
        return coastline_mask

    class _FakeRegridder:
        def __init__(self, ds_in, ds_out, **kwargs):
            # Build synthetic weights with nonzero rows only on active dst cells.
            active_rows = np.flatnonzero(mapping.flatten_to_mesh(ds_out["mask"].data != 0))
            src_cols = np.arange(active_rows.size) % ds_in["mask"].size

            class _FakeSparse:
                def __init__(self, rows, cols):
                    self.data = np.ones(rows.size, dtype=float)
                    self.coords = np.vstack([rows, cols])
                    self.shape = (ds_out["mask"].size, ds_in["mask"].size)

            class _FakeWeights:
                def __init__(self, rows, cols):
                    self.data = _FakeSparse(rows, cols)

            self.weights = _FakeWeights(active_rows, src_cols)

    captured = {}

    def _fake_write_mapping_file(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mapping, "grid_from_esmf_mesh", _fake_grid_from_esmf_mesh)
    monkeypatch.setattr(mapping, "extract_coastline_mask", _fake_extract_coastline_mask)
    monkeypatch.setattr(mapping, "is_mesh_cyclic_x", lambda *_: False)
    monkeypatch.setattr(mapping.xe, "Regridder", _FakeRegridder)
    monkeypatch.setattr(mapping, "write_mapping_file", _fake_write_mapping_file)

    mapping.generate_ESMF_map_via_xesmf(
        src_mesh_path="src_mesh.nc",
        dst_mesh_path="dst_mesh.nc",
        mapping_file="out.nc",
        method="nearest_d2s",
        map_overlap=False,
        coastline_masking=True,
    )

    assert "weights" in captured
    w = captured["weights"].data

    coastal_flat = np.flatnonzero(mapping.flatten_to_mesh(coastline_mask.data == 1))
    nonzero_rows = np.unique(w.coords[0])

    # Ensure all nonzero destination rows are coastal cells.
    assert set(nonzero_rows).issubset(set(coastal_flat))

    # And every non-coastal destination row has zero entries.
    row_nnz = np.bincount(w.coords[0], minlength=dst_grid["mask"].size)
    for row in set(range(dst_grid["mask"].size)) - set(coastal_flat):
        assert row_nnz[row] == 0


@pytest.mark.skipif(
    not TX2_3_MESH.exists(),
    reason=f"Mesh file not found: {TX2_3_MESH}",
)
def test_generate_esmf_map_via_xesmf_coastline_masking_tx2_3v2(tmp_path):
    """Integration test on tx2_3v2 mesh: with coastline_masking=True,
    nonzero destination rows must be coastal cells."""

    mapping_file = tmp_path / "tx2_3v2_coast_nn.nc"

    mapping.generate_ESMF_map_via_xesmf(
        src_mesh_path=TX2_3_MESH,
        dst_mesh_path=TX2_3_MESH,
        mapping_file=mapping_file,
        method="nearest_d2s",
        area_normalization=False,
        map_overlap=False,
        coastline_masking=True,
    )

    ds_map = xr.open_dataset(mapping_file)
    dst_mesh = xr.open_dataset(TX2_3_MESH)
    try:
        dst_grid = mapping.grid_from_esmf_mesh(dst_mesh)

        # In ESMF map files, `row` indexes destination cells (1-based).
        mapped_dst_rows = ds_map["row"].data.astype(np.int64) - 1

        coastal_mask_2d = mapping.extract_coastline_mask(dst_grid)
        coastal_mask_1d = mapping.flatten_to_mesh(coastal_mask_2d == 1).astype(bool)
        coastal_rows = np.flatnonzero(coastal_mask_1d)

        # All mapped destination rows must be coastal.
        assert set(np.unique(mapped_dst_rows)).issubset(set(coastal_rows))

        # Guard against malformed row indexing before bincount.
        assert np.all(mapped_dst_rows >= 0)

        # All non-coastal destination rows must have zero entries in the weights.
        row_counts = np.bincount(mapped_dst_rows, minlength=dst_grid["mask"].size)
        for row in set(range(dst_grid["mask"].size)) - set(coastal_rows):
            assert row_counts[row] == 0
    finally:
        ds_map.close()
        dst_mesh.close()


def test_generate_esmf_map_via_xesmf_nonfast_path(monkeypatch):
    """Non-fast path should keep the original write_mapping_file(weights=...) flow."""

    grid = xr.Dataset(
        data_vars={"mask": (("nlat", "nlon"), np.ones((2, 2), dtype=int))},
        coords={
            "lon": (("nlat", "nlon"), np.array([[0.0, 1.0], [2.0, 3.0]])),
            "lat": (("nlat", "nlon"), np.array([[0.0, 0.0], [1.0, 1.0]])),
        },
    )

    def _fake_grid_from_esmf_mesh(mesh):
        return grid

    class _FakeRegridder:
        def __init__(self, ds_in, ds_out, **kwargs):
            self.weights = "sentinel-weights"

    captured = {}

    def _fake_write_mapping_file(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mapping, "grid_from_esmf_mesh", _fake_grid_from_esmf_mesh)
    monkeypatch.setattr(mapping, "is_mesh_cyclic_x", lambda *_: False)
    monkeypatch.setattr(mapping.xe, "Regridder", _FakeRegridder)
    monkeypatch.setattr(mapping, "write_mapping_file", _fake_write_mapping_file)

    mapping.generate_ESMF_map_via_xesmf(
        src_mesh_path="src_mesh.nc",
        dst_mesh_path="dst_mesh.nc",
        mapping_file="out.nc",
        method="nearest_s2d",
        map_overlap=False,
        coastline_masking=True,
    )

    assert captured.get("weights") == "sentinel-weights"
    assert "weights_coo" not in captured


