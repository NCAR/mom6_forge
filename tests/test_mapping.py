import xarray as xr
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from mom6_forge.mapping import (
    compute_cressman_weights,
    dst_to_source,
    source_to_dst,
    regrid_dataset_via_cressman,
    _make_subgrid_points,
    regrid_with_subsampling,
)
from mom6_forge._supergrid import haversine


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
