import xarray as xr
import numpy as np
import netCDF4
import scipy.sparse as sp
import pytest
from pathlib import Path
from mom6_forge.mapping import (
    compute_cressman_weights,
    dst_to_source,
    source_to_dst,
    regrid_dataset_via_cressman,
    _make_subgrid_points,
    regrid_with_subsampling,
    _get_mesh_bbox,
    crop_esmf_mesh_to_bbox,
    write_mapping_file,
    gen_rof_maps,
    get_nn_map_filepath,
)
from mom6_forge._supergrid import haversine
from mom6_forge.grid import Grid
from mom6_forge.utils import get_mesh_dimensions


def make_synthetic_grids():
    """Tiny 8x8 source, 2x2 destination — all ocean, known depths."""
    src_lon = np.linspace(0.5, 7.5, 8)
    src_lat = np.linspace(0.5, 7.5, 8)
    src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)

    # constant depth so remapped value is exactly known
    src_depth = np.full((8, 8), 1000.0)

    src_ds = xr.Dataset(
        {"depth": (["lat", "lon"], src_depth)},
        coords={"lon": src_lon, "lat": src_lat},
    )

    # 2x2 destination centred in the source domain
    dst_lon = np.array([[2.0, 6.0], [2.0, 6.0]])
    dst_lat = np.array([[2.0, 2.0], [6.0, 6.0]])
    dst_area = np.full((2, 2), 1.2e11)
    dst_mask = np.ones((2, 2), dtype=bool)

    dst_ds = xr.Dataset(
        {
            "lon": (["y", "x"], dst_lon),
            "lat": (["y", "x"], dst_lat),
            "area": (["y", "x"], dst_area),
            "mask": (["y", "x"], dst_mask),
        }
    )

    return src_ds, dst_ds


def test_compute_cressman_weights_correctness():
    src_ds, dst_ds = make_synthetic_grids()
    ds_w = compute_cressman_weights(src_ds, dst_ds, smooth_scl=2.0)

    # --- shape metadata is present and consistent ---
    assert ds_w.sizes["n_a"] == 64, "n_a should be 8*8=64"
    assert ds_w.sizes["n_b"] == 4, "n_b should be 2*2=4"
    assert int(ds_w["nj_a"].values) == 8
    assert int(ds_w["ni_a"].values) == 8
    assert int(ds_w["nj_b"].values) == 2
    assert int(ds_w["ni_b"].values) == 2

    # --- weight sums to 1 for every filled destination cell ---

    row = ds_w["row"].values - 1
    col = ds_w["col"].values - 1
    data = ds_w["S"].values
    S = sp.coo_matrix(
        (data, (row, col)), shape=(ds_w.sizes["n_b"], ds_w.sizes["n_a"])
    ).tocsr()

    weight_sums = np.asarray(S.sum(axis=1)).ravel()
    filled = ~ds_w["unfilled"].values
    assert np.allclose(
        weight_sums[filled], 1.0, atol=1e-6
    ), f"Weight sums not ~1: {weight_sums}"

    # --- no negative weights ---
    assert (ds_w["S"].values >= 0).all(), "Negative weights found"

    # --- constant field reproduces exactly ---
    out = S @ np.ones(ds_w.sizes["n_a"])
    assert np.allclose(
        out[filled], 1.0, atol=1e-6
    ), f"Constant field not reproduced: {out}"

    # --- remapped depth of constant 1000m field is 1000m ---
    depth_out = S @ np.ones(ds_w.sizes["n_a"]) * 1000.0
    assert np.allclose(
        depth_out[filled], 1000.0, atol=1e-3
    ), f"Depth not reproduced: {depth_out}"

    # --- no unfilled cells (all ocean, generous radius) ---
    assert not ds_w["unfilled"].values.any(), "Unexpected unfilled cells"

    # pick dst cell (0,0) — centred at lon=2, lat=2
    dst_flat = 0
    row_vec = S.getrow(dst_flat)
    src_indices = row_vec.nonzero()[1]
    weights = np.asarray(row_vec[0, src_indices].todense()).ravel()

    # compute great-circle distances from dst centre to each contributing source pixel
    dst_lon = ds_w["xc_b"].values[dst_flat]
    dst_lat = ds_w["yc_b"].values[dst_flat]
    src_lons = ds_w["xc_a"].values[src_indices]
    src_lats = ds_w["yc_a"].values[src_indices]

    distances = haversine(src_lats, src_lons, dst_lat, dst_lon, R=6.371e6)

    # sort by distance and check weights are non-increasing
    order = np.argsort(distances)
    sorted_weights = weights[order]
    sorted_distances = distances[order]

    assert np.all(
        np.diff(sorted_weights) <= 1e-10
    ), f"Weights not monotonically decreasing with distance:\n  distances={sorted_distances}\n  weights={sorted_weights}"


def test_regrid_dataset_via_cressman_smoke(tmp_path):
    src_ds, dst_ds = make_synthetic_grids()

    weights_path = tmp_path / "weights.nc"
    output_path = tmp_path / "regridded.nc"

    depth_dst, unfilled = regrid_dataset_via_cressman(
        src_ds,
        dst_ds,
        weights_path=weights_path,
        output_path=output_path,
        write_to_file=True,
    )

    # --- returned arrays have right shape ---
    assert depth_dst.depth.shape == (2, 2), f"Wrong shape: {depth_dst.depth.shape}"
    assert unfilled.shape == (2, 2), f"Wrong shape: {unfilled.shape}"

    # --- weights file was written ---
    assert weights_path.exists(), "Weights file not written"

    # --- no unfilled cells ---
    assert not unfilled.any(), "Unexpected unfilled cells"

    # --- output file written if requested ---
    assert output_path.exists(), "Output file not written"


def test_smoke_weight_lookups():
    """
    Smoke test for source_to_dst and dst_to_source using a tiny synthetic grid
    where the correct answer is known analytically.
    """

    # --- tiny 4x4 source grid, 2x2 destination grid ---
    src_lon = np.array([0.0, 1.0, 2.0, 3.0])
    src_lat = np.array([0.0, 1.0, 2.0, 3.0])
    src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)

    # all ocean, depth = 1000m everywhere so weights are easy to reason about
    src_depth = np.full((4, 4), 1000.0)

    src_ds = xr.Dataset(
        {"depth": (["lat", "lon"], src_depth)},
        coords={"lon": src_lon, "lat": src_lat},
    )

    # 2x2 destination grid centred on the source grid
    dst_lon = np.array([[1.0, 2.0], [1.0, 2.0]])
    dst_lat = np.array([[1.0, 1.0], [2.0, 2.0]])
    dst_area = np.full((2, 2), 1.2e11)  # ~approx area for 1° cell in m²
    dst_mask = np.ones((2, 2), dtype=bool)

    dst_ds = xr.Dataset(
        {
            "lon": (["y", "x"], dst_lon),
            "lat": (["y", "x"], dst_lat),
            "area": (["y", "x"], dst_area),
            "mask": (["y", "x"], dst_mask),
        }
    )

    # --- compute weights ---
    print("Building synthetic weight dataset...")
    ds_w = compute_cressman_weights(
        src_ds, dst_ds, smooth_scl=0.5
    )  # use a small smoothing scale to get more localized weights and a more interesting test case
    print(
        f"  n_s={ds_w.sizes['n_s']}, n_a={ds_w.sizes['n_a']}, n_b={ds_w.sizes['n_b']}"
    )
    print(f"  src_shape=({ds_w['nj_a'].values}, {ds_w['ni_a'].values})")
    print(f"  dst_shape=({ds_w['nj_b'].values}, {ds_w['ni_b'].values})")

    # --- test 1: dst_to_source ---
    print("\n--- dst_to_source(ds_w, (0, 0)) ---")
    src_indices, weights = dst_to_source(ds_w, (0, 0))
    assert len(src_indices) > 0, "dst cell (0,0) should have source pixels"
    assert np.isclose(
        weights.sum(), 1.0, atol=1e-6
    ), f"weights should sum to 1, got {weights.sum()}"
    print(f"  PASS: {len(src_indices)} source pixels, weight sum={weights.sum():.6f}")

    # --- test 2: source_to_dst ---
    print("\n--- source_to_dst(ds_w, (1, 1)) ---")
    # source pixel (1,1) is at lon=1, lat=1 — right on top of dst cell (0,0)
    # so it should have a high weight toward that cell
    dst_indices, weights = source_to_dst(ds_w, (1, 1))
    assert len(dst_indices) > 0, "source pixel (1,1) should feed at least one dst cell"
    assert all(w > 0 for w in weights), "all weights should be positive"
    print(f"  PASS: feeds {len(dst_indices)} dst cells")

    # --- test 3: round-trip consistency ---
    print("\n--- Round-trip consistency ---")
    # every source pixel that dst (0,0) draws from should list (0,0) as a destination
    src_indices_fwd, _ = dst_to_source(ds_w, (0, 0))
    for src_flat in src_indices_fwd:
        src_2d = np.unravel_index(src_flat, (4, 4))
        dst_indices_back, _ = source_to_dst(ds_w, src_2d)
        assert (
            0 in dst_indices_back
        ), f"src {src_2d} feeds dst (0,0) fwd but not bwd — inconsistency!"
    print(f"  PASS: all {len(src_indices_fwd)} source pixels point back to dst (0,0)")

    # --- test 4: constant field reproduces exactly ---
    print("\n--- Constant field reproduction ---")
    import scipy.sparse as sp

    row = ds_w["row"].values - 1
    col = ds_w["col"].values - 1
    data = ds_w["S"].values
    S = sp.coo_matrix(
        (data, (row, col)), shape=(ds_w.sizes["n_b"], ds_w.sizes["n_a"])
    ).tocsr()
    out = S @ np.ones(ds_w.sizes["n_a"])
    assert np.allclose(
        out[dst_mask.ravel()], 1.0, atol=1e-6
    ), f"constant field not reproduced: {out}"
    print(f"  PASS: constant field → {out} (all ~1.0 for ocean cells)")


def test_make_subgrid_points(get_simple_grid):
    # Test with a simple 2x2 grid and 2 sub-points per cell
    nx_sub = ny_sub = 2
    grid = get_simple_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )

    expected_sub_lon = np.array(
        [
            [[[4 / 3, 5 / 3], [4 / 3, 5 / 3]], [[7 / 3, 8 / 3], [7 / 3, 8 / 3]]],
            [[[4 / 3, 5 / 3], [4 / 3, 5 / 3]], [[7 / 3, 8 / 3], [7 / 3, 8 / 3]]],
        ]
    )
    expected_sub_lat = np.array(
        [
            [[[4 / 3, 4 / 3], [5 / 3, 5 / 3]], [[4 / 3, 4 / 3], [5 / 3, 5 / 3]]],
            [[[7 / 3, 7 / 3], [8 / 3, 8 / 3]], [[7 / 3, 7 / 3], [8 / 3, 8 / 3]]],
        ]
    )

    assert np.allclose(
        sub_lon, expected_sub_lon
    ), "Sub-grid longitudes do not match expected values."
    assert np.allclose(
        sub_lat, expected_sub_lat
    ), "Sub-grid latitudes do not match expected values."


def test_smoke_seams_and_global_make_subgrid_points(
    get_dateline_seam_grid, get_PM_seam_grid, get_simple_global_grid
):
    # Test with a simple 2x2 grid and 2 sub-points per cell
    nx_sub = ny_sub = 2
    grid = get_dateline_seam_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    grid = get_PM_seam_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    grid = get_simple_global_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )


def test_regrid_with_subsampling(get_simple_grid):
    # Test with a simple 2x2 grid and 2 sub-points per cell with data that lands exactly on the sub points (subtracted by 0.1 to show snapping to sub points)
    nx_sub = ny_sub = 2
    grid = get_simple_grid
    lon = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    lat = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    input_ds = xr.Dataset(
        {
            "data": (
                ["lon", "lat"],
                [
                    np.arange(1, 5, 1),
                    np.arange(1, 5, 1),
                    np.arange(1, 5, 1),
                    np.arange(1, 5, 1),
                ],
            )
        },
        coords={
            "lon": (["lon"], [x - 0.1 for x in lon]),
            "lat": (["lat"], [x - 0.1 for x in lat]),
        },
    )
    ds, _ = regrid_with_subsampling(
        input_ds, grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    assert ds["data"].shape == (2, 2, 2, 2), "Output shape is incorrect."
    expected_data = np.array(
        [[[[1, 1], [2, 2]], [[1, 1], [2, 2]]], [[[3, 3], [4, 4]], [[3, 3], [4, 4]]]]
    )
    assert np.allclose(
        ds["data"].values, expected_data
    ), "Regridded data does not match expected values."


# ---------------------------------------------------------------------------
# Runoff-mapping scaling fixes: _get_mesh_bbox, crop_esmf_mesh_to_bbox, and
# write_mapping_file's shape lookup.
# ---------------------------------------------------------------------------


def _write_esmf_mesh(grid, path):
    grid.supergrid.to_esmf_mesh(str(path), mask="all_unmasked")
    return path


def _physical_mapping(ds):
    """Key each nonzero mapping entry by (dst_lon, dst_lat, src_lon, src_lat)
    instead of raw row/col index, so a cropped and an uncropped mapping (which
    have different n_a/array sizes by design) can still be compared."""
    xc_a, yc_a = ds["xc_a"].values, ds["yc_a"].values
    xc_b, yc_b = ds["xc_b"].values, ds["yc_b"].values
    row, col, S = ds["row"].values - 1, ds["col"].values - 1, ds["S"].values
    return {
        (
            round(float(xc_b[r]), 6),
            round(float(yc_b[r]), 6),
            round(float(xc_a[c]), 6),
            round(float(yc_a[c]), 6),
        ): float(s)
        for r, c, s in zip(row, col, S)
    }


@pytest.fixture
def crop_rof_grid():
    """10x10 deg, 1 deg resolution, straddling the equator (lat -5..5) so
    _get_mesh_bbox's latitude handling is exercised on negative values."""
    return Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=-5.0, leny=10.0, name="crop_rof"
    )


@pytest.fixture
def crop_ocn_grid():
    """3x3 deg destination domain fully inside crop_rof_grid's extent."""
    return Grid(
        resolution=1.0, xstart=13.0, lenx=3.0, ystart=-2.0, leny=3.0, name="crop_ocn"
    )


def test_get_mesh_bbox_preserves_negative_latitude(tmp_path, crop_rof_grid):
    mesh_path = _write_esmf_mesh(crop_rof_grid, tmp_path / "rof.nc")
    lon_min, lon_max, lat_min, lat_max = _get_mesh_bbox(mesh_path)

    assert lon_min == pytest.approx(10.0)
    assert lon_max == pytest.approx(20.0)
    # Regression guard: latitude must NOT be wrapped through normalize_deg's
    # mod-360 longitude logic (that would turn -5.0 into ~355.0).
    assert lat_min == pytest.approx(-5.0)
    assert lat_max == pytest.approx(5.0)


def test_crop_esmf_mesh_to_bbox_shrinks_to_expected_window(
    tmp_path, crop_rof_grid, crop_ocn_grid
):
    rof_path = _write_esmf_mesh(crop_rof_grid, tmp_path / "rof.nc")
    ocn_path = _write_esmf_mesh(crop_ocn_grid, tmp_path / "ocn.nc")

    lon_min, lon_max, lat_min, lat_max = _get_mesh_bbox(ocn_path)
    cropped_path = crop_esmf_mesh_to_bbox(
        rof_path,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        tmp_path / "rof_cropped.nc",
        buffer_deg=0.5,
    )

    cropped = xr.open_dataset(cropped_path)
    nx_c, ny_c = get_mesh_dimensions(cropped)
    # lon centers 12.5..16.5 (buffer-widened [12.5, 16.5]), lat centers
    # -2.5..1.5 (buffer-widened [-2.5, 1.5]) -> 5x5 window.
    assert (nx_c, ny_c) == (5, 5)

    center_lon = cropped["centerCoords"].data[:, 0].reshape(ny_c, nx_c)
    center_lat = cropped["centerCoords"].data[:, 1].reshape(ny_c, nx_c)
    np.testing.assert_allclose(center_lon[0, :], [12.5, 13.5, 14.5, 15.5, 16.5])
    np.testing.assert_allclose(center_lat[:, 0], [-2.5, -1.5, -0.5, 0.5, 1.5])


def test_crop_esmf_mesh_to_bbox_rejects_nonmonotonic_mesh(tmp_path, crop_rof_grid):
    rof_path = _write_esmf_mesh(crop_rof_grid, tmp_path / "rof.nc")
    ds = xr.open_dataset(rof_path)
    lon = ds["centerCoords"].data[:, 0].copy()
    lon[5] = lon[1]  # break strict monotonicity along the x-axis of row 0
    ds["centerCoords"].data[:, 0] = lon
    ds.to_netcdf(rof_path)

    with pytest.raises(AssertionError, match="monotonic"):
        crop_esmf_mesh_to_bbox(rof_path, 13.0, 16.0, -2.0, 1.0, tmp_path / "out.nc")


def test_write_mapping_file_uses_shape_lookup_not_full_reconstruction(
    tmp_path, monkeypatch
):
    """write_mapping_file only ever needs each mesh's (nx, ny) shape - it must not
    reconstruct full mesh geometry (Topo.from_esmf_mesh) just to get that shape."""
    src_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=3.0, ystart=0.0, leny=2.0, name="src_shape"
    )
    dst_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=2.0, ystart=0.0, leny=2.0, name="dst_shape"
    )
    src_path = _write_esmf_mesh(src_grid, tmp_path / "src.nc")
    dst_path = _write_esmf_mesh(dst_grid, tmp_path / "dst.nc")

    def _boom(*args, **kwargs):
        raise AssertionError("write_mapping_file must not call Topo.from_esmf_mesh")

    monkeypatch.setattr("mom6_forge.topo.Topo.from_esmf_mesh", _boom)

    weights_coo = sp.coo_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(4, 6))
    out_path = tmp_path / "out.nc"
    write_mapping_file(
        src_mesh=str(src_path),
        dst_mesh=str(dst_path),
        filename=out_path,
        weights_coo=weights_coo,
    )

    ds = xr.open_dataset(out_path)
    assert list(ds["src_grid_dims"].values) == [3, 2]
    assert list(ds["dst_grid_dims"].values) == [2, 2]


def test_write_mapping_file_produces_valid_classic_format(tmp_path):
    """write_mapping_file writes via an intermediate NETCDF4 file converted with
    `nccopy -6` (much faster than writing NETCDF3_64BIT directly - see NOTES.md
    Step 9), but the output must still be a genuine 64-bit-offset classic file,
    since that's what CESM's mapping-file readers require. Also guards the
    int64-vs-classic-format bug found while implementing this: `nj_a`/`ni_a`/
    `nj_b`/`ni_b` are coordinate variables (their DataArray name matches their
    sole dimension), so an encoding loop scoped to `ds.data_vars` silently skips
    them and leaves them as int64 - a dtype classic format can't represent at
    all, which makes `nccopy` fail outright. Every integer variable, coordinate
    or not, must come out as int32."""
    src_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=3.0, ystart=0.0, leny=2.0, name="src_classic"
    )
    dst_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=2.0, ystart=0.0, leny=2.0, name="dst_classic"
    )
    src_path = _write_esmf_mesh(src_grid, tmp_path / "src.nc")
    dst_path = _write_esmf_mesh(dst_grid, tmp_path / "dst.nc")

    weights_coo = sp.coo_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(4, 6))
    out_path = tmp_path / "out.nc"
    write_mapping_file(
        src_mesh=str(src_path),
        dst_mesh=str(dst_path),
        filename=out_path,
        weights_coo=weights_coo,
    )

    nc = netCDF4.Dataset(out_path)
    try:
        assert nc.data_model == "NETCDF3_64BIT_OFFSET"
        for name, var in nc.variables.items():
            if np.issubdtype(var.dtype, np.integer):
                assert var.dtype == np.int32, f"{name} is {var.dtype}, expected int32"
    finally:
        nc.close()


def test_gen_rof_maps_crop_matches_uncropped_physical_mapping(tmp_path):
    """Cropping is purely an internal speed-up for computing weights - it must
    not change what mesh the written mapping file claims to describe. The
    runtime component that actually consumes this file (e.g. DROF) declares
    its own domain as the FULL, uncropped ROF mesh, and CMEPS's remap requires
    the mapping file's `n_a` to match that domain's element count exactly.
    A cropped-but-mismatched `n_a` is a real, confirmed bug (found by
    inspecting how CrocoDash wires up ROF_DOMAIN_MESH - it never gets
    re-pointed at a cropped mesh), not just a cosmetic difference - so this
    checks both that the mapping is physically the same AND that `n_a` did
    not shrink.

    Uses positive-latitude grids (not the equator-straddling crop_rof_grid
    fixture): generate_ESMF_map_via_xesmf's own map_overlap masking still
    runs latitude through normalize_deg - a separate, pre-existing bug not
    touched by this change (see _get_mesh_bbox's docstring) - and this test
    should check the crop, not that unrelated bug.
    """
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=25.0, leny=10.0, name="rof"
    )
    ocn_grid = Grid(
        resolution=1.0, xstart=13.0, lenx=3.0, ystart=28.0, leny=3.0, name="ocn"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof.nc")
    ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / "ocn.nc")

    cropped_dir, uncropped_dir = tmp_path / "cropped", tmp_path / "uncropped"
    gen_rof_maps(rof_path, ocn_path, cropped_dir, "test_map", crop_source_mesh=True)
    gen_rof_maps(rof_path, ocn_path, uncropped_dir, "test_map", crop_source_mesh=False)

    ds_crop = xr.open_dataset(get_nn_map_filepath("test_map", cropped_dir))
    ds_uncrop = xr.open_dataset(get_nn_map_filepath("test_map", uncropped_dir))

    # The regression guard for the actual bug: cropping must not shrink n_a.
    # rof_grid is 10x10 = 100 elements - if this is 900 instead, the crop
    # window leaked into the output and the file is DROF-domain-inconsistent.
    assert ds_crop.sizes["n_a"] == ds_uncrop.sizes["n_a"] == 100

    map_crop = _physical_mapping(ds_crop)
    map_uncrop = _physical_mapping(ds_uncrop)
    assert map_crop, "expected at least one nonzero mapping entry"
    assert map_crop.keys() == map_uncrop.keys()
    for key, weight in map_crop.items():
        assert weight == pytest.approx(map_uncrop[key], abs=1e-9)


def test_regrid_with_subsampling_time_dim(get_simple_grid):
    nx_sub = ny_sub = 2
    grid = get_simple_grid
    lon = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    lat = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    spatial_data = np.array(
        [
            np.arange(1, 5, 1),
            np.arange(1, 5, 1),
            np.arange(1, 5, 1),
            np.arange(1, 5, 1),
        ],
        dtype=float,
    )
    nt = 2
    input_ds = xr.Dataset(
        {"data": (["time", "lon", "lat"], np.stack([spatial_data] * nt))},
        coords={
            "lon": (["lon"], [x - 0.1 for x in lon]),
            "lat": (["lat"], [x - 0.1 for x in lat]),
        },
    )
    ds, _ = regrid_with_subsampling(
        input_ds, grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    assert ds["data"].shape == (
        nt,
        2,
        2,
        2,
        2,
    ), "Output shape with time dim is incorrect."
    expected_spatial = np.array(
        [[[[1, 1], [2, 2]], [[1, 1], [2, 2]]], [[[3, 3], [4, 4]], [[3, 3], [4, 4]]]]
    )
    for t in range(nt):
        assert np.allclose(
            ds["data"].values[t], expected_spatial
        ), f"Regridded data at t={t} does not match expected values."
