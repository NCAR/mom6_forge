import xarray as xr
import numpy as np
import scipy.sparse as sp
import pytest
from pathlib import Path
from mom6_forge import mapping
from mom6_forge.mapping import (
    compute_cressman_weights,
    dst_to_source,
    source_to_dst,
    regrid_dataset_via_cressman,
    _make_subgrid_points,
    regrid_with_subsampling,
    write_mapping_file,
    _lon_window,
    gen_rof_maps,
    get_nn_map_filepath,
)
from mom6_forge._supergrid import haversine
from mom6_forge.grid import Grid


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


# ---------------------------------------------------------------------------
# write_mapping_file mesh-shape lookup
# ---------------------------------------------------------------------------


def _write_esmf_mesh(grid, path):
    grid.supergrid.to_esmf_mesh(str(path), mask="all_unmasked")
    return path


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


# ---------------------------------------------------------------------------
# map_overlap masking: 0/360 seam and equator-straddling destinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lon, expected",
    [
        # a regional window, nowhere near the seam
        (np.arange(10.0, 21.0), (10.0, 10.0)),
        # the same window ending exactly on lon 360, where normalize_deg sends
        # the last node to 0 and min()/max() reported 0.00..359.00
        (np.arange(350.0, 361.0), (350.0, 10.0)),
        # written the other way round, as negative longitudes
        (np.arange(-10.0, 1.0), (350.0, 10.0)),
        # a single point has no width
        (np.array([42.0]), (42.0, 0.0)),
        # fully periodic: every gap is the grid spacing, so the window is the
        # whole globe bar one spacing - what min()/max() also gave for it
        (np.arange(0.0, 360.0), (1.0, 359.0)),
    ],
)
def test_lon_window(lon, expected):
    start, width = _lon_window(lon)
    assert (start, width) == pytest.approx(expected)


def test_map_overlap_masking_is_longitude_rotation_invariant(tmp_path):
    """map_overlap must mask the same way whether or not the domain touches lon 360.

    The source mesh is uniform in longitude, so two destination windows of equal
    width must select equally many source cells no matter where they sit. Before
    the fix the seam window's bbox came back as ~0..359, masked nothing, and
    mapped nearly the whole source mesh - 261 entries against the interior
    window's 90.
    """
    rof_grid = Grid(
        resolution=1.0, xstart=330.0, lenx=30.0, ystart=-5.0, leny=10.0, name="rof_rot"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof_rot.nc")

    counts = {}
    # "seam" ends on exactly lon 360; "interior" is the same width, 10 deg west.
    for tag, xstart in (("seam", 350.0), ("interior", 340.0)):
        ocn_grid = Grid(
            resolution=1.0, xstart=xstart, lenx=10.0, ystart=-3.0, leny=6.0, name=tag
        )
        ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / f"ocn_{tag}.nc")
        out_dir = tmp_path / f"out_{tag}"
        gen_rof_maps(rof_path, ocn_path, out_dir, f"map_{tag}", rmax=50, fold=100)
        with xr.open_dataset(get_nn_map_filepath(f"map_{tag}", out_dir)) as ds:
            counts[tag] = int(ds["S"].size)

    assert counts["interior"] > 0, "interior destination should map some source cells"
    assert counts["seam"] == counts["interior"], (
        f"equal-width domains must map equally many source cells regardless of "
        f"where they sit in longitude, got {counts} - the domain touching lon 360 "
        f"is getting a near-global window again, so map_overlap masks nothing"
    )


def test_map_overlap_masking_is_latitude_translation_invariant(tmp_path):
    """map_overlap must mask by true latitude, not by a mod-360 wrap of it.

    The old code ran both the destination bbox and the source latitudes through
    normalize_deg. That is self-consistent for a destination lying entirely in one
    hemisphere - which is why it went unspotted - but not for one straddling the
    equator: a -3..3 destination normalizes into two clusters, 357..360 and 0..3,
    so min()/max() collapse to ~0 and ~359 and the latitude test masks nothing at
    all.

    The source mesh is uniform in latitude, so three destination windows of equal
    height must select equally many source cells wherever they sit. Before the fix
    the equator-straddling window mapped 92 entries against the other two's 24.
    """
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=-12.0, leny=24.0, name="rof_lat"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof_lat.nc")

    counts = {}
    for tag, ystart in (("straddle", -3.0), ("north", 3.0), ("south", -9.0)):
        ocn_grid = Grid(
            resolution=1.0, xstart=13.0, lenx=4.0, ystart=ystart, leny=6.0, name=tag
        )
        ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / f"ocn_{tag}.nc")
        out_dir = tmp_path / f"out_{tag}"
        gen_rof_maps(rof_path, ocn_path, out_dir, f"map_{tag}", rmax=50, fold=100)
        with xr.open_dataset(get_nn_map_filepath(f"map_{tag}", out_dir)) as ds:
            counts[tag] = int(ds["S"].size)

    assert counts["north"] > 0, "northern destination should map some source cells"
    assert counts["straddle"] == counts["north"] == counts["south"], (
        f"equal-height domains must map equally many source cells regardless of "
        f"where they sit in latitude, got {counts} - the equator-straddling domain "
        f"is getting a ~0..359 latitude bbox again, so map_overlap masks nothing "
        f"in latitude"
    )


# ---------------------------------------------------------------------------
# ESMF_FactorRead float32 PET-split padding
# ---------------------------------------------------------------------------


def test_pad_weights_for_esmf_is_noop_below_threshold():
    """Below 2**24 the float32 PET split in ESMF_FactorRead is provably exact,
    so nothing is added - a small map must come out byte-for-byte as built."""
    S = xr.DataArray(np.array([0.5, 0.25, 0.25]), dims=["n_s"])
    row = xr.DataArray(np.array([7, 7, 9], dtype="i4"), dims=["n_s"])
    col = xr.DataArray(np.array([3, 4, 5], dtype="i4"), dims=["n_s"])

    S2, row2, col2 = mapping._pad_weights_for_esmf(S, row, col)

    assert S2.sizes["n_s"] == 3
    assert np.array_equal(S2.values, S.values)
    assert np.array_equal(row2.values, row.values)
    assert np.array_equal(col2.values, col.values)


def test_pad_weights_for_esmf_aligns_and_stays_inert(monkeypatch):
    """Above the threshold n_s is rounded up to a multiple of 1024 with
    zero-weight entries that reuse the first real (row, col) pair.

    Reusing an existing pair matters: a pair the map does not otherwise use
    would pull an extra source cell into the route handle, and 0 * NaN would
    then poison its destination cell. ESMF sums duplicate index pairs.
    """
    monkeypatch.setattr(mapping, "_ESMF_FACTORREAD_EXACT_MAX", 4)
    n = 1500
    S = xr.DataArray(np.linspace(0.1, 0.9, n), dims=["n_s"])
    row = xr.DataArray(np.arange(1, n + 1, dtype="i4"), dims=["n_s"])
    col = xr.DataArray(np.arange(101, 101 + n, dtype="i4"), dims=["n_s"])

    S2, row2, col2 = mapping._pad_weights_for_esmf(S, row, col)

    assert S2.sizes["n_s"] == 2048
    assert S2.sizes["n_s"] % mapping._N_S_ALIGNMENT == 0
    # the real weights are untouched, and the pad contributes nothing
    assert np.array_equal(S2.values[:n], S.values)
    assert (S2.values[n:] == 0).all()
    assert S2.values.sum() == pytest.approx(S.values.sum())
    # the pad introduces no source or destination cell the map didn't already use
    assert (row2.values[n:] == row.values[0]).all()
    assert (col2.values[n:] == col.values[0]).all()
    assert set(np.unique(row2.values)) == set(np.unique(row.values))
    assert set(np.unique(col2.values)) == set(np.unique(col.values))
    assert row2.dtype == row.dtype and col2.dtype == col.dtype


def test_pad_weights_for_esmf_rejects_unfixable_size(monkeypatch):
    """Past 2**33 a 1024 alignment no longer lands on an exact float32 value, so
    padding cannot make the split safe - fail loudly rather than pretend."""
    monkeypatch.setattr(mapping, "_ESMF_FACTORREAD_EXACT_MAX", 4)

    class _Fake:
        sizes = {"n_s": 2**33 + 7}

    with pytest.raises(ValueError, match=r"2\*\*33"):
        mapping._pad_weights_for_esmf(_Fake(), None, None)
