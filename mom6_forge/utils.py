"""Auxiliary functions for grid property computations."""

from pathlib import Path
import xarray as xr
import numpy as np
import scipy


def longitude_slicer(data, longitude_extent, longitude_coords):
    """Slice a dataset in longitude, handling periodicity and domain seams.

    Correctly clips datasets whose longitude coordinate may use any
    convention (e.g. ``[0, 360]`` or ``[-180, 180]``) and where the
    requested ``longitude_extent`` may straddle the wrap-around seam.

    The algorithm proceeds in five steps:

    1. Determine the integer multiple of 360° needed to shift the midpoint
       of ``longitude_extent`` into the range covered by ``data``.
    2. Roll the dataset so that its centre aligns with the midpoint of the
       target extent.
    3. Rebuild a monotonically increasing longitude coordinate that removes
       the seam introduced by the roll.
    4. Index out the required number of longitude points symmetrically
       around the new centre.
    5. Re-centre the coordinate values to match the target domain.

    Parameters
    ----------
    data : xarray.Dataset
        Global (or at least periodic) dataset to slice.  The longitude
        coordinate must be uniformly spaced.
    longitude_extent : array-like of float
        Target longitude bounds ``(west, east)`` in degrees, in increasing
        order.
    longitude_coords : str or list of str
        Name(s) of the longitude coordinate(s) in ``data`` along which to
        slice.

    Returns
    -------
    xarray.Dataset
        Dataset sliced to ``longitude_extent``, with the longitude
        coordinate re-centred to match the target domain.

    Raises
    ------
    AssertionError
        If any named longitude coordinate is not uniformly spaced.
    """

    if isinstance(longitude_coords, str):
        longitude_coords = [longitude_coords]

    for lon in longitude_coords:

        central_longitude = np.mean(longitude_extent)  ## Midpoint of target domain

        ## Find a corresponding value for the intended domain midpoint in our data.
        ## It's assumed that data has equally-spaced longitude values.

        lons = data[lon].data
        dlons = lons[1] - lons[0]
        lon_span = lons[-1] - lons[0] + dlons

        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1), atol=1e-4, rtol=0
        ), "provided longitude coordinate must be uniformly spaced"

        is_longitude_extent_in_data = (
            False  # This boolean checks if the 360 + i adjustment isn't found
        )
        for i in range(-1, 2, 1):
            if data[lon][0] <= central_longitude + 360 * i <= data[lon][-1]:
                is_longitude_extent_in_data = True

                ## Shifted version of target midpoint; e.g., could be -90 vs 270
                ## integer i keeps track of what how many multiples of 360 we need to shift entire
                ## grid by to match central_longitude
                _central_longitude = central_longitude + 360 * i

                ## Midpoint of the data
                central_data = data[lon][data[lon].shape[0] // 2].values

                ## Number of indices between the data midpoint and the target midpoint.
                ## Sign indicates direction needed to shift.
                shift = int(
                    -(data[lon].shape[0] * (_central_longitude - central_data))
                    // lon_span
                )

                ## Shift data so that the midpoint of the target domain is the middle of
                ## the data for easy slicing.
                new_data = data.roll({lon: 1 * shift}, roll_coords=True)

                ## Create a new longitude coordinate.
                ## We'll modify this to remove any seams (i.e., jumps like -270 -> 90)
                new_lon = new_data[lon].values.copy()

                ## Take the 'seam' of the data, and either backfill or forward fill based on
                ## whether the data was shifted F or west
                if shift > 0:
                    new_seam_index = shift

                    new_lon[0:new_seam_index] -= 360

                if shift < 0:
                    new_seam_index = data[lon].shape[0] + shift

                    new_lon[new_seam_index:] += 360

                ## new_lon is used to re-centre the midpoint to match that of target domain
                new_lon -= i * 360

                new_data = new_data.assign_coords({lon: new_lon})

                ## Choose the number of lon points to take from the middle, including a buffer.
                ## Use this to index the new global dataset
                num_lonpoints = abs(
                    int(
                        int(
                            data[lon].shape[0]
                            * (central_longitude - longitude_extent[0])
                        )
                        // lon_span
                    )
                )

        if not is_longitude_extent_in_data:
            raise ValueError(
                "The longitude of the data doesn't seem to include the longitude of the grid."
            )

        data = new_data.isel(
            {
                lon: slice(
                    data[lon].shape[0] // 2 - num_lonpoints,
                    data[lon].shape[0] // 2 + num_lonpoints,
                )
            }
        )

    return data


def normalize_deg(coord):
    """Normalize a coordinate in degrees to the range [0, 360).

    Parameters
    ----------
    coord : float or np.ndarray
        Coordinate(s) in degrees.

    Returns
    -------
    normalized_coord : float or np.ndarray
        Normalized coordinate(s) in the range [0, 360).
    """
    return np.mod(coord + 360.0, 360.0)


def get_mesh_dimensions(mesh):
    """Given an ESMF mesh where the grid metrics are stored in 1D (flattened) arrays,
    compute the dimensions of the 2D grid and return them as nx, ny.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.

    Returns
    -------
    nx : int
        Number of points in the x-direction.
    ny : int
        Number of points in the y-direction.
    """

    if not isinstance(mesh, xr.Dataset):
        assert (
            isinstance(mesh, (Path, str)) and Path(mesh).exists()
        ), "mesh must be a path to an existing file"
        mesh = xr.open_dataset(mesh)

    centerCoords = mesh["centerCoords"].values
    econn = mesh["elementConn"].values
    econn0 = econn[0]

    nx = None
    for i in range(2, len(econn)):
        # if the number of shared nodes is 2, we are at the start of a new row
        if len(np.intersect1d(econn[i], econn0)) == 2:
            if i < len(econn) - 1 and len(np.intersect1d(econn[i + 1], econn0)) == 2:
                nx = i + 1  # domain is cyclic
            else:
                nx = i  # domain is Not cyclic
            break

    if nx is None:
        raise ValueError(
            "Could not determine the number of points in the x-direction (nx)."
        )

    ny = len(centerCoords) // nx

    # Check that nx is indeed nx and not ny, and if not, swap them
    assert (
        "units" in mesh["centerCoords"].attrs
    ), "centerCoords must have 'units' attribute"
    assert (
        "degrees" in mesh["centerCoords"].attrs["units"]
    ), "get_mesh_dimensions() expects centerCoords in degrees"
    coords = centerCoords[
        :, :2
    ]  # Use only the first two columns for x and y coordinates
    x0, y0 = centerCoords[0]  # First coordinate
    x0 = normalize_deg(x0)  # Normalize to [0, 360)
    if np.abs(np.mod(coords[nx // 2, 0] - x0, 360)) < np.abs(coords[nx // 2, 1] - y0):
        nx, ny = ny, nx

    assert nx * ny == len(
        centerCoords
    ), f"Mesh dimensions do not match the number of coordinates: {nx} * {ny} != {len(centerCoords)}"
    return nx, ny


def get_avg_resolution(mesh):
    """Calculate the average resolution of the mesh.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.

    Returns
    -------
    avg_resolution : float
        Average resolution of the mesh in degrees.
    """
    if not isinstance(mesh, xr.Dataset):
        assert (
            isinstance(mesh, (Path, str)) and Path(mesh).exists()
        ), "mesh must be a path to an existing file"
        mesh = xr.open_dataset(mesh)

    assert (
        "units" in mesh["centerCoords"].attrs
    ), "centerCoords must have 'units' attribute"
    assert (
        "degrees" in mesh["centerCoords"].attrs["units"]
    ), "get_mesh_dimensions() expects centerCoords in degrees"

    centerCoords = mesh["centerCoords"].values
    nx, ny = get_mesh_dimensions(mesh)

    coords = centerCoords.reshape(ny, nx, 2)
    dy = np.diff(coords[:, :, 1], axis=0)  # y-direction
    dx = np.diff(coords[:, :, 0], axis=1)  # x-direction
    avg_resolution = np.mean(np.concatenate([dx.ravel(), dy.ravel()]))

    return avg_resolution


def get_avg_resolution_km(mesh):
    """Calculate the average resolution of the mesh in kilometers.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.

    Returns
    -------
    avg_resolution_km : float
        Average resolution of the mesh in kilometers.
    """

    if not isinstance(mesh, xr.Dataset):
        assert (
            isinstance(mesh, (Path, str)) and Path(mesh).exists()
        ), "mesh must be a path to an existing file"
    mesh = xr.open_dataset(mesh)

    assert (
        "units" in mesh["centerCoords"].attrs
    ), "centerCoords must have 'units' attribute"
    assert (
        "degrees" in mesh["centerCoords"].attrs["units"]
    ), "get_mesh_dimensions() expects centerCoords in degrees"

    centerCoords = mesh["centerCoords"].values
    nx, ny = get_mesh_dimensions(mesh)

    earth_radius_km = 6371.0

    # Compute distances between all neighboring points (both x and y directions)
    coords = centerCoords.reshape(ny, nx, 2)
    # Compute dx in km (longitude difference * cos(latitude) * earth_radius)
    # and dy in km (latitude difference * earth_radius)
    # Use the latitude at the midpoint for dx
    lons = coords[:, :, 0]
    lats = coords[:, :, 1]

    # dx: shape (ny, nx-1)
    dx_deg = np.diff(lons, axis=1)
    lat_mid_dx = 0.5 * (lats[:, :-1] + lats[:, 1:])
    dx_km = np.deg2rad(dx_deg) * earth_radius_km * np.cos(np.deg2rad(lat_mid_dx))

    # dy: shape (ny-1, nx)
    dy_deg = np.diff(lats, axis=0)
    dy_km = np.deg2rad(dy_deg) * earth_radius_km

    avg_resolution_km = np.mean(np.concatenate([dx_km.ravel(), dy_km.ravel()]))
    return avg_resolution_km


def is_mesh_cyclic_x(mesh):
    """Check if the mesh is cyclic in the x-direction.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.

    Returns
    -------
    bool
        True if the mesh is cyclic in the x-direction, False otherwise.
    """
    if not isinstance(mesh, xr.Dataset):
        assert (
            isinstance(mesh, (Path, str)) and Path(mesh).exists()
        ), "mesh must be a path to an existing file"
        mesh = xr.open_dataset(mesh)

    nx, _ = get_mesh_dimensions(mesh)
    econn = mesh["elementConn"].values
    if len(np.intersect1d(econn[nx - 1], econn[0])) == 2:
        return True
    return False


def _spherical_angle(a, b):
    """Compute angle between unit vectors a and b on a sphere.
    This function is used for area computation.

    Parameters
    ----------
    a : np.ndarray, shape (..., 3)
        First unit vector (x, y, z).
    b : np.ndarray, shape (..., 3)
        Second unit vector (x, y, z).

    Returns
    -------
    angle : np.ndarray
        Angle in radians.
    """

    cross = np.cross(a, b)
    sin_angle = np.linalg.norm(cross, axis=-1)
    cos_angle = np.sum(a * b, axis=-1)
    return np.arctan2(sin_angle, cos_angle)


def _tri_area(u, v, w):
    """Vectorized spherical triangle area using L'Huilier's theorem.

    Parameters:
    -----------
    u, v, w: np.ndarray of shape (..., 3)
        Arrays of unit vectors representing triangle vertices.

    Returns:
    --------
    np.ndarray of shape (...,): Areas of the spherical triangles.
    """
    a = _spherical_angle(v, w)
    b = _spherical_angle(u, w)
    c = _spherical_angle(u, v)
    s = 0.5 * (a + b + c)

    t = (
        np.tan(0.5 * s)
        * np.tan(0.5 * (s - a))
        * np.tan(0.5 * (s - b))
        * np.tan(0.5 * (s - c))
    )

    area = np.abs(4.0 * np.arctan(np.sqrt(np.abs(t))))
    return area


def _great_circle_area(polygon_unitvecs):
    """
    Compute area of multiple spherical polygons using triangulation from vertex 0.

    Parameters:
    -----------
    polygon_unitvecs: np.ndarray of shape (n_poly, n_verts, 3)

    Returns:
    --------
    np.ndarray of shape (n_poly,): Areas of the spherical polygons
    """
    n_poly, n_verts, _ = polygon_unitvecs.shape
    if n_verts < 3:
        return np.zeros(n_poly)

    pnt0 = polygon_unitvecs[:, 0:1, :]  # shape (n_poly, 1, 3)
    pnt1 = polygon_unitvecs[:, 1:-1, :]  # shape (n_poly, n_verts-2, 3)
    pnt2 = polygon_unitvecs[:, 2:, :]  # shape (n_poly, n_verts-2, 3)

    u = np.broadcast_to(pnt0, pnt1.shape)  # broadcast pnt0
    areas = _tri_area(u, pnt1, pnt2)  # shape (n_poly, n_verts-2)
    return np.sum(areas, axis=1)  # sum over triangles per polygon


def cell_area_rad(xv_coords, yv_coords):
    """Compute the area of cell(s) using spherical polygons.

    Parameters
    ----------
    xv_coords : np.ndarray
        X-coordinate(s) of cell vertices in degrees.
    yv_coords : np.ndarray
        Y-coordinate(s) of cell vertices in degrees.

    Returns
    -------
    area : np.ndarray
        Area of the cell(s) in square radians.
    """

    # Convert coordinates to radians
    xv_rad = np.deg2rad(xv_coords)
    yv_rad = np.deg2rad(yv_coords)

    # Construct unit vectors
    x = np.cos(yv_rad) * np.cos(xv_rad)
    y = np.cos(yv_rad) * np.sin(xv_rad)
    z = np.sin(yv_rad)

    area = _great_circle_area(np.stack([x, y, z], axis=-1))
    return area


def iterative_fill(depth, unfilled, mask=None, max_iter=100):
    """
    Iteratively fill unfilled ocean cells from their neighbours. May not work for periodic grids.

    Parameters
    ----------
    depth : np.ndarray, shape (ny, nx)
        Depth field to fill in-place.
    unfilled : np.ndarray of bool, shape (ny, nx)
        True for cells that need filling.
    mask : array-like or xr.DataArray, optional
        Ocean mask (1 = ocean, 0 = land). If provided, land cells are never filled.
    max_iter : int
        Maximum number of neighbour-averaging passes. Default 100.

    Returns
    -------
    depth : np.ndarray
        Depth array with unfilled ocean cells replaced by neighbour averages.
    """

    # --- Iterative neighbour fill for cells with no source coverage ---
    if unfilled.any():
        n_miss = int(unfilled.sum())
        print(f"Filling {n_miss} cells by iterative neighbour averaging…")
        if mask is not None:
            mask_2d = mask.values.astype(bool)
            unfilled_2d = unfilled & mask_2d
        else:
            unfilled_2d = unfilled

        for _ in range(max_iter):
            if not unfilled_2d.any():
                break
            filled_f = (~unfilled_2d).astype(float)
            d_pad = np.pad(depth, 1, mode="edge")  # pad 1 cell with edge values
            f_pad = np.pad(
                filled_f, 1, mode="constant", constant_values=0
            )  # pad 1 mask cell with land (unfilled)
            d_nbr = (
                d_pad[:-2, 1:-1] + d_pad[2:, 1:-1] + d_pad[1:-1, :-2] + d_pad[1:-1, 2:]
            )
            f_nbr = (
                f_pad[:-2, 1:-1] + f_pad[2:, 1:-1] + f_pad[1:-1, :-2] + f_pad[1:-1, 2:]
            )
            can_fill = unfilled_2d & (
                f_nbr > 0
            )  # only fill if there is at least one filled neighbor, which is done by adding all the neighboring cells
            depth = np.where(
                can_fill, d_nbr / np.maximum(f_nbr, 1), depth
            )  # Takes the average of the neighboring cells to fill, only where can_fill is True
            unfilled_2d = unfilled_2d & ~can_fill

    return depth


def fill_missing_data(idata, mask, maxiter=0, stabilizer=1.0e-14, tripole=False):
    """
    Returns data with masked values "objectively interpolated" except where values exist or is over land. Does not work for periodic grids.

    Arguments:
    data - numpy array with nan values where there is missing data or land.
    mask - np.array of 0 or 1, 0 for land, 1 for ocean.

    Returns an array.
    """

    nj, ni = idata.shape
    fdata = np.nan_to_num(idata, nan=0.0)
    missing_j, missing_i = np.where(np.isnan(idata) & (mask > 0))

    n_missing = missing_i.size

    # ind contains column of matrix/row of vector corresponding to point [j,i]
    ind = np.zeros(fdata.shape, dtype=int) - int(1e6)
    ind[missing_j, missing_i] = np.arange(n_missing)
    A = scipy.sparse.lil_matrix((n_missing, n_missing))
    b = np.zeros((n_missing))
    ld = np.zeros((n_missing))
    A[range(n_missing), range(n_missing)] = 0.0
    for n in range(n_missing):
        j, i = missing_j[n], missing_i[n]
        im1 = max(i - 1, 0)
        ip1 = min(i + 1, ni - 1)
        jm1 = max(j - 1, 0)
        jp1 = min(j + 1, nj - 1)
        if j > 0 and mask[jm1, i] > 0:
            ld[n] -= 1.0
            ij = ind[jm1, i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[jm1, i]
        if i > 0 and mask[j, im1] > 0:
            ld[n] -= 1.0
            ij = ind[j, im1]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, im1]
        if i < ni - 1 and mask[j, ip1] > 0:
            ld[n] -= 1.0
            ij = ind[j, ip1]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, ip1]
        if j < nj - 1 and mask[jp1, i] > 0:
            ld[n] -= 1.0
            ij = ind[jp1, i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[jp1, i]
        if j == nj - 1 and mask[j, ni - 1 - i] > 0 and tripole:  # Tri-polar fold
            ld[n] -= 1.0
            ij = ind[j, ni - 1 - i]
            if ij >= 0:
                A[n, ij] = 1.0
            else:
                b[n] -= fdata[j, ni - 1 - i]
    # Set leading diagonal
    b[ld >= 0] = 0.0
    A[range(n_missing), range(n_missing)] = ld - stabilizer

    A = scipy.sparse.csr_matrix(A)
    new_data = np.ma.array(fdata, mask=(mask == 0))
    if maxiter is None:
        x, info = scipy.sparse.linalg.bicg(A, b)
    elif maxiter == 0:
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        x, info = scipy.sparse.linalg.bicg(A, b, maxiter=maxiter)
    new_data[missing_j, missing_i] = x
    return new_data


def compute_subsampling_factor(src_nj, src_ni, dst_nj, dst_ni):
    """
    Compute the sub-sampling factors needed so that the super-sampled
    destination grid has at least as many points as the source grid.

    Parameters
    ----------
    src_nj, src_ni : int
        Source grid dimensions.
    dst_nj, dst_ni : int
        Destination grid dimensions.

    Returns
    -------
    ny_sub, nx_sub : int
    """
    nx_sub = 1
    while nx_sub * dst_ni < src_ni:
        nx_sub += 1

    ny_sub = 1
    while ny_sub * dst_nj < src_nj:
        ny_sub += 1

    return ny_sub, nx_sub
