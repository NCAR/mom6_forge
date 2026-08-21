import xarray as xr
import numpy as np
import netCDF4
import scipy.sparse as sp
import pytest
from pathlib import Path
from mom6_forge.mapping import (
    MINIMAL_MAP_VARIABLES,
    compute_cressman_weights,
    dst_to_source,
    source_to_dst,
    regrid_dataset_via_cressman,
    _make_subgrid_points,
    regrid_with_subsampling,
    _get_mesh_bbox,
    write_mapping_file,
    gen_rof_maps,
    get_full_map_filepath,
    get_nn_map_filepath,
    get_smoothed_map_filepath,
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
# Runoff-mapping scaling fixes: _get_mesh_bbox and write_mapping_file's shape
# lookup / classic-format write path.
# ---------------------------------------------------------------------------


def _write_esmf_mesh(grid, path):
    grid.supergrid.to_esmf_mesh(str(path), mask="all_unmasked")
    return path


@pytest.fixture
def crop_rof_grid():
    """10x10 deg, 1 deg resolution, straddling the equator (lat -5..5) so
    _get_mesh_bbox's latitude handling is exercised on negative values."""
    return Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=-5.0, leny=10.0, name="crop_rof"
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


def test_map_overlap_masking_is_hemisphere_symmetric(tmp_path):
    """map_overlap must mask source cells by true latitude, not a mod-360 wrap.

    generate_ESMF_map_via_xesmf() used to run source latitude through
    normalize_deg(), turning every negative latitude into a large positive one
    (-5 -> 355) so it compared as "north of lat_max" and got masked out. A
    northern-hemisphere destination never noticed, which is why this went
    unspotted; a southern one lost every source point and produced an empty
    mapping file.

    Source mesh here is symmetric about the equator, and the two destination
    domains are exact mirror images, so the two mappings must contain the same
    number of entries. Before the fix the southern count was 0.
    """
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=-10.0, leny=20.0, name="rof_sym"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof_sym.nc")

    counts = {}
    for tag, ystart in (("north", 2.0), ("south", -8.0)):
        ocn_grid = Grid(
            resolution=1.0, xstart=13.0, lenx=4.0, ystart=ystart, leny=6.0, name=tag
        )
        ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / f"ocn_{tag}.nc")
        out_dir = tmp_path / f"out_{tag}"
        gen_rof_maps(rof_path, ocn_path, out_dir, f"map_{tag}", rmax=50, fold=100)
        with xr.open_dataset(get_nn_map_filepath(f"map_{tag}", out_dir)) as ds:
            counts[tag] = int(ds["S"].size)

    assert counts["north"] > 0, "northern destination should map some source cells"
    assert counts["south"] > 0, (
        "southern destination mapped no source cells — source latitude is being "
        "wrapped through normalize_deg again"
    )
    assert counts["south"] == counts["north"], (
        f"mirror-image domains must map equally many source cells, got "
        f"north={counts['north']} south={counts['south']}"
    )


def test_get_mesh_bbox_handles_0_360_seam(tmp_path):
    """A mesh whose eastern edge is exactly lon 360 must not report a global bbox.

    normalize_deg is mod(x + 360, 360), so nodes sitting on exactly 360 come back
    as 0. Taking min()/max() of that then reports 0.00..359.x - nearly the whole
    globe - for a mesh spanning ten degrees. Downstream that turned
    generate_ESMF_map_via_xesmf's map_overlap masking into a no-op in longitude,
    so the regrid ran against a global latitude band; a 576K-cell domain went from
    106 s to not finishing in over an hour.

    The interval is allowed to wrap (lon_min > lon_max) - that is how a
    seam-crossing domain is expressed - so what this asserts is the *width*.
    """
    grid = Grid(
        resolution=1.0, xstart=350.0, lenx=10.0, ystart=-3.0, leny=6.0, name="seam"
    )
    mesh_path = _write_esmf_mesh(grid, tmp_path / "seam.nc")
    lon_min, lon_max, lat_min, lat_max = _get_mesh_bbox(mesh_path)

    width = lon_max - lon_min if lon_max >= lon_min else 360.0 - lon_min + lon_max
    assert width == pytest.approx(10.0), (
        f"bbox spans {width:.2f} deg for a 10 deg mesh - longitudes touching the "
        f"0/360 seam are being collapsed by normalize_deg again "
        f"(got lon_min={lon_min}, lon_max={lon_max})"
    )
    assert lon_min == pytest.approx(350.0)
    assert lat_min == pytest.approx(-3.0)
    assert lat_max == pytest.approx(3.0)


def test_map_overlap_masking_is_rotation_invariant(tmp_path):
    """map_overlap must mask the same way whether or not the domain touches lon 360.

    Companion to test_map_overlap_masking_is_hemisphere_symmetric: that one covers
    latitude, this one longitude. The source mesh is uniform in longitude, so two
    destination windows of equal width must select equally many source cells no
    matter where they sit. Before the fix the seam window's bbox came back as
    ~0..359, masked nothing, and mapped the entire source mesh - three times as
    many cells as the interior window.
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
        f"where they sit in longitude, got seam={counts['seam']} "
        f"interior={counts['interior']} - the domain touching lon 360 is getting a "
        f"near-global bbox again, so map_overlap masks nothing"
    )


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


def test_gen_rof_maps_end_to_end(tmp_path):
    """Smoke test for gen_rof_maps() end-to-end (nn map + smoothing), on
    positive-latitude grids to avoid the separate, pre-existing
    normalize_deg(src_lat) bug in generate_ESMF_map_via_xesmf's map_overlap
    masking (see _get_mesh_bbox's docstring) - unrelated to what this test
    covers."""
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=25.0, leny=10.0, name="rof"
    )
    ocn_grid = Grid(
        resolution=1.0, xstart=13.0, lenx=3.0, ystart=28.0, leny=3.0, name="ocn"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof.nc")
    ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / "ocn.nc")

    out_dir = tmp_path / "out"
    gen_rof_maps(rof_path, ocn_path, out_dir, "test_map", rmax=50, fold=100)

    nn_path = get_nn_map_filepath("test_map", out_dir)
    sm_path = get_smoothed_map_filepath("test_map", out_dir, 50, 100)
    ds_nn = xr.open_dataset(nn_path)
    ds_sm = xr.open_dataset(sm_path)
    assert ds_nn["S"].size > 0, "expected at least one nonzero nn mapping entry"
    assert ds_sm["S"].size > 0, "expected at least one nonzero smoothed mapping entry"
    # gen_rof_maps writes minimal files by default, so the per-source-cell
    # geometry (and its n_a dimension) lives in the _full companions instead.
    for ds in (ds_nn, ds_sm):
        assert "n_a" not in ds.sizes
    with xr.open_dataset(get_full_map_filepath(nn_path)) as ds_nn_full:
        with xr.open_dataset(get_full_map_filepath(sm_path)) as ds_sm_full:
            # rof_grid is 10x10
            assert ds_nn_full.sizes["n_a"] == ds_sm_full.sizes["n_a"] == 100


def test_gen_rof_maps_minimal_and_full_agree(tmp_path):
    """With save_full_map=True the output pair must be self-consistent: the file
    CESM reads carries exactly the variables ESMF's ESMF_FactorRead() needs, the
    _full companion carries the whole CESM/SCRIP variable set, and the weights in
    the two are identical. Also pins the requested-vs-legacy variable sets, so
    dropping a variable ESMF does read can't pass silently."""
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=25.0, leny=10.0, name="rof_min"
    )
    ocn_grid = Grid(
        resolution=1.0, xstart=13.0, lenx=3.0, ystart=28.0, leny=3.0, name="ocn_min"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof.nc")
    ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / "ocn.nc")

    out_dir = tmp_path / "out"
    gen_rof_maps(rof_path, ocn_path, out_dir, "m", rmax=50, fold=100)

    for path in (
        get_nn_map_filepath("m", out_dir),
        get_smoothed_map_filepath("m", out_dir, 50, 100),
    ):
        full_path = get_full_map_filepath(path)
        assert full_path.exists(), f"missing record-keeping companion {full_path}"
        with xr.open_dataset(path) as ds, xr.open_dataset(full_path) as ds_full:
            assert set(ds.variables) == set(MINIMAL_MAP_VARIABLES)
            # "row", "col" and "S" over dimension "n_s" are what ESMF_FactorRead
            # requires; the grid-dims arrays are two ints each and record the
            # shape that row/col index into.
            assert {"S", "row", "col"}.issubset(ds.variables)
            assert ds["S"].dims == ("n_s",)
            assert set(ds_full.variables) > set(MINIMAL_MAP_VARIABLES)
            for var in MINIMAL_MAP_VARIABLES:
                assert np.array_equal(ds[var].values, ds_full[var].values), var
            # minimal must be strictly smaller - that's the entire point
            assert path.stat().st_size < full_path.stat().st_size


def test_gen_rof_maps_full_companion_is_compressed_netcdf4(tmp_path):
    """The CESM-facing file must stay NETCDF3_64BIT; the record-keeping companion
    must not be held to it. Classic format cannot address a variable over 4 GiB -
    S alone is 11.8 GiB for CrocIndoPacific_112 at rmax=100 - and nothing reads
    the companion from CESM, so it goes out as compressed NETCDF4 instead."""
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=25.0, leny=10.0, name="rof_fmt"
    )
    ocn_grid = Grid(
        resolution=1.0, xstart=13.0, lenx=3.0, ystart=28.0, leny=3.0, name="ocn_fmt"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof.nc")
    ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / "ocn.nc")

    out_dir = tmp_path / "out"
    gen_rof_maps(rof_path, ocn_path, out_dir, "m", rmax=50, fold=100)

    for path in (
        get_nn_map_filepath("m", out_dir),
        get_smoothed_map_filepath("m", out_dir, 50, 100),
    ):
        nc = netCDF4.Dataset(path)
        try:
            assert nc.data_model == "NETCDF3_64BIT_OFFSET", path.name
        finally:
            nc.close()

        full_path = get_full_map_filepath(path)
        nc = netCDF4.Dataset(full_path)
        try:
            assert nc.data_model == "NETCDF4", full_path.name
            # integers stay int32 on this path too - nothing here needs 64-bit
            for name, var in nc.variables.items():
                if np.issubdtype(var.dtype, np.integer):
                    assert var.dtype == np.int32, f"{name} is {var.dtype}"
        finally:
            nc.close()


def test_write_mapping_file_deflates_large_vars_only(tmp_path):
    """Deflate is applied to the arrays big enough for it to pay off, and skipped
    on the handful of tiny index/grid-dims arrays where chunking is pure
    overhead."""
    src_grid = Grid(
        resolution=0.5, xstart=0.0, lenx=40.0, ystart=0.0, leny=40.0, name="src_defl"
    )
    dst_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=2.0, ystart=0.0, leny=2.0, name="dst_defl"
    )
    src_path = _write_esmf_mesh(src_grid, tmp_path / "src.nc")
    dst_path = _write_esmf_mesh(dst_grid, tmp_path / "dst.nc")
    weights_coo = sp.coo_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(4, 6400))

    write_mapping_file(
        src_mesh=str(src_path),
        dst_mesh=str(dst_path),
        filename=tmp_path / "out.nc",
        weights_coo=weights_coo,
        minimal=True,
        full_map_filename=tmp_path / "out_full.nc",
    )

    nc = netCDF4.Dataset(tmp_path / "out_full.nc")
    try:
        # src side is 80x80 = 6400 elements, over the deflate threshold
        assert nc.variables["xc_a"].filters()["zlib"] is True
        assert nc.variables["mask_a"].filters()["zlib"] is True
        # dst side is 2x2 and the grid-dims arrays are length 2 - far below it
        assert nc.variables["dst_grid_dims"].filters()["zlib"] is False
        assert nc.variables["xc_b"].filters()["zlib"] is False
    finally:
        nc.close()


def test_gen_rof_maps_single_precision(tmp_path):
    """single_precision=True must halve the float variables' on-disk width without
    changing the integer indices, in both output files, and must survive the
    nccopy -6 conversion (a float32 S is still a valid ESMF weight file:
    ESMF_FactorRead reads it into a real(R8) buffer and netCDF converts on
    read)."""
    rof_grid = Grid(
        resolution=1.0, xstart=10.0, lenx=10.0, ystart=25.0, leny=10.0, name="rof_sp"
    )
    ocn_grid = Grid(
        resolution=1.0, xstart=13.0, lenx=3.0, ystart=28.0, leny=3.0, name="ocn_sp"
    )
    rof_path = _write_esmf_mesh(rof_grid, tmp_path / "rof.nc")
    ocn_path = _write_esmf_mesh(ocn_grid, tmp_path / "ocn.nc")

    dtypes = {}
    for tag, single in (("f8", False), ("f4", True)):
        out_dir = tmp_path / tag
        gen_rof_maps(
            rof_path,
            ocn_path,
            out_dir,
            "m",
            rmax=50,
            fold=100,
            single_precision=single,
        )
        # read the _full companion: it is the one that carries every float
        # variable, so it exercises the downcast on more than just S
        nc = netCDF4.Dataset(get_full_map_filepath(get_nn_map_filepath("m", out_dir)))
        try:
            dtypes[tag] = {n: v.dtype for n, v in nc.variables.items()}
        finally:
            nc.close()
        # and the CESM-facing file, which must survive the nccopy -6 conversion
        nc = netCDF4.Dataset(get_nn_map_filepath("m", out_dir))
        try:
            assert nc.data_model == "NETCDF3_64BIT_OFFSET"
            expected = np.float32 if single else np.float64
            assert nc.variables["S"].dtype == expected
        finally:
            nc.close()

    assert dtypes["f8"].keys() == dtypes["f4"].keys()
    floats = [n for n, d in dtypes["f8"].items() if np.issubdtype(d, np.floating)]
    assert floats, "expected some float variables to downcast"
    for name, dtype in dtypes["f4"].items():
        if name in floats:
            assert dtype == np.float32, f"{name} is {dtype}, expected float32"
        else:
            assert dtype == dtypes["f8"][name] == np.int32, name


def test_write_mapping_file_rejects_full_without_minimal(tmp_path):
    """A full_map_filename alongside minimal=False would be a byte-for-byte
    duplicate of filename, so it's an error rather than a silent waste of disk."""
    src_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=3.0, ystart=0.0, leny=2.0, name="src_dup"
    )
    dst_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=2.0, ystart=0.0, leny=2.0, name="dst_dup"
    )
    src_path = _write_esmf_mesh(src_grid, tmp_path / "src.nc")
    dst_path = _write_esmf_mesh(dst_grid, tmp_path / "dst.nc")
    weights_coo = sp.coo_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(4, 6))

    with pytest.raises(ValueError, match="only meaningful with minimal=True"):
        write_mapping_file(
            src_mesh=str(src_path),
            dst_mesh=str(dst_path),
            filename=tmp_path / "out.nc",
            weights_coo=weights_coo,
            full_map_filename=tmp_path / "out_full.nc",
        )


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
