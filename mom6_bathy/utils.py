"""Auxiliary functions for grid property computations."""

from pathlib import Path
import xarray as xr
import numpy as np
import scipy


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


def longitude_slicer(data, longitude_extent, longitude_coords):
    """
    Slices longitudes while handling periodicity and the 'seams', that is the
    longitude values where the data wraps around in a global domain (for example,
    longitudes are defined, usually, within domain [0, 360] or [-180, 180]).

    The algorithm works in five steps:

    - Determine whether we need to add or subtract 360 to get the middle of the
      ``longitude_extent`` to lie within ``data``'s longitude range (hereby ``old_lon``).

    - Shift the dataset so that its midpoint matches the midpoint of
      ``longitude_extent`` (up to a multiple of 360). Now, the modified ``old_lon``
      does not increase monotonically from West to East since the 'seam'
      has moved.

    - Fix ``old_lon`` to make it monotonically increasing again. This uses
      the information we have about the way the dataset was shifted/rolled.

    - Slice the ``data`` index-wise. We know that ``|longitude_extent[1] - longitude_extent[0]| / 360``
      multiplied by the number of discrete longitude points in the global input data gives
      the number of longitude points in our slice, and we've already set the midpoint
      to be the middle of the target domain.

    - Add back the correct multiple of 360 so the whole domain matches the target.

    Arguments:
        data (xarray.Dataset): The global data you want to slice in longitude.
        longitude_extent (Tuple[float, float]): The target longitudes (in degrees)
            we want to slice to. Must be in increasing order.
        longitude_coords (Union[str, list[str]): The name or list of names of the
            longitude coordinates(s) in ``data``.

    Returns:
        xarray.Dataset: The sliced ``data``.
    """

    if isinstance(longitude_coords, str):
        longitude_coords = [longitude_coords]

    for lon in longitude_coords:
        central_longitude = np.mean(longitude_extent)  ## Midpoint of target domain

        ## Find a corresponding value for the intended domain midpoint in our data.
        ## It's assumed that data has equally-spaced longitude values.

        lons = data[lon].data
        dlons = lons[1] - lons[0]

        assert np.allclose(
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
        ), "provided longitude coordinate must be uniformly spaced"

        for i in range(-1, 2, 1):
            if data[lon][0] <= central_longitude + 360 * i <= data[lon][-1]:

                ## Shifted version of target midpoint; e.g., could be -90 vs 270
                ## integer i keeps track of what how many multiples of 360 we need to shift entire
                ## grid by to match central_longitude
                _central_longitude = central_longitude + 360 * i

                ## Midpoint of the data
                central_data = data[lon][data[lon].shape[0] // 2].values

                ## Number of indices between the data midpoint and the target midpoint.
                ## Sign indicates direction needed to shift.
                shift = int(
                    -(data[lon].shape[0] * (_central_longitude - central_data)) // 360
                )

                ## Shift data so that the midpoint of the target domain is the middle of
                ## the data for easy slicing.
                new_data = data.roll({lon: 1 * shift}, roll_coords=True)

                ## Create a new longitude coordinate.
                ## We'll modify this to remove any seams (i.e., jumps like -270 -> 90)
                new_lon = new_data[lon].values

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
                num_lonpoints = (
                    int(data[lon].shape[0] * (central_longitude - longitude_extent[0]))
                    // 360
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


def mdist(x1, x2):
    """
    Returns positive distance modulo 360.

    >>> x1=0.0;x2=730.
    >>> d=mdist(x1,x2)
    >>> print(d)
    10.0
    """

    a = np.mod(x1 - x2 + 720.0, 360.0)
    b = np.mod(x2 - x1 + 720.0, 360.0)

    d = np.minimum(a, b)

    return d


def angle_between(v1, v2, v3):
    """Return the angle ``v2``-``v1``-``v3`` (in radians), where
    ``v1``, ``v2``, ``v3`` are 3-vectors. That is, the angle that
    is formed between vectors ``v2 - v1`` and vector ``v3 - v1``.

    Example:

        >>> from regional_mom6.utils import angle_between
        >>> v1 = (0, 0, 1)
        >>> v2 = (1, 0, 0)
        >>> v3 = (0, 1, 0)
        >>> angle_between(v1, v2, v3)
        1.5707963267948966
        >>> from numpy import rad2deg
        >>> rad2deg(angle_between(v1, v2, v3))
        90.0
    """

    v1xv2 = np.cross(v1, v2)
    v1xv3 = np.cross(v1, v3)

    norm_v1xv2 = np.sqrt(vecdot(v1xv2, v1xv2))
    norm_v1xv3 = np.sqrt(vecdot(v1xv3, v1xv3))

    cosangle = vecdot(v1xv2, v1xv3) / (norm_v1xv2 * norm_v1xv3)

    return np.arccos(cosangle)


def vecdot(v1, v2):
    """Return the dot product of vectors ``v1`` and ``v2``.
    ``v1`` and ``v2`` can be either numpy vectors or numpy.ndarrays
    in which case the last dimension is considered the dimension
    over which the dot product is taken.
    """
    return np.sum(v1 * v2, axis=-1)


def latlon_to_cartesian(lat, lon, R=1):
    """Convert latitude and longitude (in degrees) to Cartesian coordinates on
    a sphere of radius ``R``. By default ``R = 1``.

    Arguments:
        lat (float): Latitude (in degrees).
        lon (float): Longitude (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        tuple: Tuple with the Cartesian coordinates ``x, y, z``

    Examples:

        Find the Cartesian coordinates that correspond to point with
        ``(lat, lon) = (0, 0)`` on a sphere with unit radius.

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> latlon_to_cartesian(0, 0)
        (1.0, 0.0, 0.0)

        Now let's do the same on a sphere with Earth's radius

        >>> from regional_mom6.utils import latlon_to_cartesian
        >>> R = 6371e3
        >>> latlon_to_cartesian(0, 0, R)
        (6371000.0, 0.0, 0.0)
    """

    x = R * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = R * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = R * np.sin(np.deg2rad(lat))

    return x, y, z


def quadrilateral_areas(lat, lon, R=1):
    """Return the area of spherical quadrilaterals on a sphere of radius ``R``.
    By default, ``R = 1``. The quadrilaterals are formed by constant latitude and
    longitude lines on the ``lat``-``lon`` grid provided.

    Arguments:
        lat (numpy.array): Array of latitude points (in degrees).
        lon (numpy.array): Array of longitude points (in degrees).
        R (float): The radius of the sphere; default: 1.

    Returns:
        numpy.array: Array with the areas of the quadrilaterals defined by the ``lat``-``lon`` grid
        provided. If the provided ``lat``, ``lon`` arrays are of dimension *m* :math:`\\times` *n*
        then returned areas array is of dimension (*m-1*) :math:`\\times` (*n-1*).

    Example:

        Let's construct a lat-lon grid on the sphere with 60 degree spacing.
        Then we compute the areas of each grid cell and confirm that the
        sum of the areas gives us the total area of the sphere.

        >>> from regional_mom6.utils import quadrilateral_areas
        >>> import numpy as np
        >>> λ = np.linspace(0, 360, 7)
        >>> φ = np.linspace(-90, 90, 4)
        >>> lon, lat = np.meshgrid(λ, φ)
        >>> lon
        array([[  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.],
               [  0.,  60., 120., 180., 240., 300., 360.]])
        >>> lat
        array([[-90., -90., -90., -90., -90., -90., -90.],
               [-30., -30., -30., -30., -30., -30., -30.],
               [ 30.,  30.,  30.,  30.,  30.,  30.,  30.],
               [ 90.,  90.,  90.,  90.,  90.,  90.,  90.]])
        >>> R = 6371e3
        >>> areas = quadrilateral_areas(lat, lon, R)
        >>> areas
        array([[1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13],
               [4.56284230e+13, 4.56284230e+13, 4.56284230e+13, 4.56284230e+13,
                4.56284230e+13, 4.56284230e+13],
               [1.96911611e+13, 1.96911611e+13, 1.96911611e+13, 1.96911611e+13,
                1.96911611e+13, 1.96911611e+13]])
        >>> np.isclose(areas.sum(), 4 * np.pi * R**2, atol=np.finfo(areas.dtype).eps)
        True
    """

    coords = np.dstack(latlon_to_cartesian(lat, lon, R))

    return quadrilateral_area(
        coords[:-1, :-1, :], coords[:-1, 1:, :], coords[1:, 1:, :], coords[1:, :-1, :]
    )


def quadrilateral_area(v1, v2, v3, v4):
    """Return the area of a spherical quadrilateral on the unit sphere that
    has vertices on the 3-vectors ``v1``, ``v2``, ``v3``, ``v4``
    (counter-clockwise orientation is implied). The area is computed via
    the excess of the sum of the spherical angles of the quadrilateral from 2π.

    Example:

        Calculate the area that corresponds to half the Northern hemisphere
        of a sphere of radius *R*. This should be 1/4 of the sphere's total area,
        that is π *R*:sup:`2`.

        >>> from regional_mom6.utils import quadrilateral_area, latlon_to_cartesian
        >>> R = 434.3
        >>> v1 = latlon_to_cartesian(0, 0, R)
        >>> v2 = latlon_to_cartesian(0, 90, R)
        >>> v3 = latlon_to_cartesian(90, 0, R)
        >>> v4 = latlon_to_cartesian(0, -90, R)
        >>> quadrilateral_area(v1, v2, v3, v4)
        592556.1793298927
        >>> from numpy import pi
        >>> quadrilateral_area(v1, v2, v3, v4) == pi * R**2
        True
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    v4 = np.array(v4)

    if not (
        np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v2, v2)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v3, v3)))
        & np.all(np.isclose(vecdot(v1, v1), vecdot(v4, v4)))
    ):
        raise ValueError("vectors provided must have the same length")

    R = np.sqrt(vecdot(v1, v1))

    a1 = angle_between(v1, v2, v4)
    a2 = angle_between(v2, v3, v1)
    a3 = angle_between(v3, v4, v2)
    a4 = angle_between(v4, v1, v3)

    return (a1 + a2 + a3 + a4 - 2 * np.pi) * R**2
