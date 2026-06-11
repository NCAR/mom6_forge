#!/usr/bin/env python3

import os
import argparse
import xarray as xr
import numpy as np
import datetime
from time import time
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import xesmf as xe

MPI = None
rank = lambda: MPI.COMM_WORLD.Get_rank() if MPI else 0

from mom6_forge.utils import (
    get_mesh_dimensions,
    cell_area_rad,
    normalize_deg,
    is_mesh_cyclic_x,
    get_avg_resolution_km,
)


def grid_from_esmf_mesh(mesh: xr.Dataset | str | Path) -> "Grid":
    """Given an ESMF mesh where the grid metrics are stored in 1D (flattened) arrays,
    compute the dimensions of the 2D grid and return a 2D horizontal grid dataset
    containing the longitude, latitude, and mask of the grid points which are all that
    is needed to create a regridder.

    Parameters
    ----------
    mesh : xr.Dataset or str or Path
        The ESMF mesh dataset or the path to the mesh file.

    Returns
    -------
    ds : xr.Dataset
        A 2D horizontal grid dataset containing the longitude, latitude, and mask of the grid points.
    """

    if not isinstance(mesh, xr.Dataset):
        assert (
            isinstance(mesh, (Path, str)) and Path(mesh).exists()
        ), "mesh must be a path to an existing file"
        mesh = xr.open_dataset(mesh)

    nx, ny = get_mesh_dimensions(mesh)

    lon = mesh["centerCoords"][:, 0].values.reshape((ny, nx))
    lat = mesh["centerCoords"][:, 1].values.reshape((ny, nx))
    mask = mesh["elementMask"].values.reshape((ny, nx))

    ds = xr.Dataset(
        data_vars={
            "mask": (("nlat", "nlon"), mask),
        },
        coords={
            "lon": (
                ("nlat", "nlon"),
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                ("nlat", "nlon"),
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "nlat": np.arange(ny),
            "nlon": np.arange(nx),
        },
    )

    return ds


def extract_coastline_mask(horiz_grid):
    """Given a 2D horizontal grid dataset, extract the coastline mask.

    Parameters
    ----------
    horiz_grid : xr.Dataset
        A 2D horizontal grid dataset containing the longitude, latitude, and mask of the grid points.
        This dataset may be obtained from the grid_from_esmf_mesh function by passing an ESMF mesh.

    Returns
    -------
    da_coastline_mask : xr.DataArray
        A 2D DataArray containing the coastline mask.
    """

    mask = horiz_grid["mask"]

    # Apply padding to facilitate the diff operation
    mask_padded = np.pad(mask, pad_width=1, mode="wrap")

    coastline_mask = np.where(
        (
            (np.diff(mask_padded[:-1, 1:-1], axis=0) == 1)
            | (np.diff(mask_padded[1:, 1:-1], axis=0) == -1)
            | (np.diff(mask_padded[1:-1, :-1], axis=1) == 1)
            | (np.diff(mask_padded[1:-1, 1:], axis=1) == -1)
        ),
        1,
        0,
    )

    da_coastline_mask = mask.copy(data=coastline_mask)
    return da_coastline_mask


def sum_weights(regridder, stride=1):
    nx_in, ny_in = regridder.grid_in.get_shape()
    nx_out, ny_out = regridder.grid_out.get_shape()

    s = sum(
        [
            (regridder.weights[:, i].data.reshape((ny_out, nx_out)).todense())
            for i in range(0, ny_in * nx_in, stride)
        ]
    )
    da = xr.DataArray(
        s, dims=["nlat", "nlon"], coords={"nlat": range(ny_out), "nlon": range(nx_out)}
    )
    return da


def get_suggested_smoothing_params(ocn_mesh_path):
    """Given an ocean mesh path, suggest RMAX and FOLD values for smoothed runoff to ocean mapping generation.

    Parameters
    ----------
    rof_mesh_path : str or Path
        Path to the runoff ESMF mesh file.
    ocn_mesh_path : str or Path
        Path to the ocean ESMF mesh file.

    Returns
    -------
    suggested_rmax : int
        Suggested RMAX value in kilometers.
    suggested_fold : int
        Suggested FOLD value in kilometers.
    """

    average_resolution_km = get_avg_resolution_km(ocn_mesh_path)
    suggested_rmax = int(5 * average_resolution_km)

    # most significant digit rounding
    magnitude = 10 ** (len(str(suggested_rmax)) - 1)
    suggested_rmax = (suggested_rmax // magnitude) * magnitude

    suggested_fold = suggested_rmax * 2

    return suggested_rmax, suggested_fold


def _construct_vertex_coords(mesh):
    """Construct the vertex coordinates for a given mesh that's participating in a
    mapping generation. The mesh could be src or dst mesh.

    Parameters
    ----------
    mesh : xr.Dataset
        The ESMF mesh dataset.
    Returns
    -------
    xv_data : np.ndarray
        The vertex coordinates in the x-direction.
    yv_data : np.ndarray
        The vertex coordinates in the y-direction.
    """

    xv_data = np.full((len(mesh.centerCoords.data), 4), np.nan)
    yv_data = np.full((len(mesh.centerCoords.data), 4), np.nan)

    element_conn = mesh.elementConn.data - 1  # one-based to zero-based indexing
    node_coords = mesh.nodeCoords.data

    for e, nodes in enumerate(element_conn):
        if np.isnan(nodes[3]):
            valid_nodes = nodes[:3][~np.isnan(nodes[:3])].astype(int)
            xv_data[e, :3] = node_coords[valid_nodes, 0]
            xv_data[e, 3] = xv_data[e, 2]
            yv_data[e, :3] = node_coords[valid_nodes, 1]
            yv_data[e, 3] = yv_data[e, 2]
        else:
            valid_nodes = nodes.astype(int)
            xv_data[e, :] = node_coords[valid_nodes, 0]
            yv_data[e, :] = node_coords[valid_nodes, 1]

    return xv_data, yv_data


def write_mapping_file(
    src_mesh,
    dst_mesh,
    filename,
    weights=None,
    weights_esmpy=None,
    weights_coo=None,
    area_normalization=False,
):
    """Based on a given source mesh, destination mesh, and weights, write out an ESMF map file.

    Parameters
    ----------
    src_mesh : str, xr.Dataset
        ESMF mesh object or path to the source ESMF mesh file.
    dst_mesh : str, xr.Dataset
        ESMF mesh object or path to the destination ESMF mesh file.
    filename : str
        Path to the output ESMF map file.
    weights : xr.DataArray
        xESMF-generated weights to be used for the regridding.
    weights_esmpy:
        esmpy-generated weights to be used for the regridding.
    weights_coo : scipy.sparse.coo_matrix
        COO format sparse matrix containing the weights.
    regrid_method : str, optional
        Regridding method, by default 'bilinear'.
    area_normalization : bool, optional
        Whether to multiply the weights by (area_source / area_destination), by default False.

    Returns
    -------
    None
    """

    if rank() != 0:
        return  # TODO: parallelize this function

    if isinstance(src_mesh, (str, Path)):
        src_mesh = xr.open_dataset(src_mesh)
    if isinstance(dst_mesh, (str, Path)):
        dst_mesh = xr.open_dataset(dst_mesh)

    if weights is weights_esmpy is weights_coo is None:
        raise ValueError(
            "At least one of weights, weights_esmpy, or weights_coo must be provided."
        )
    if weights is not None:
        assert weights_esmpy is None, "Cannot provide both weights and weights_esmpy"
        assert weights_coo is None, "Cannot provide both weights and weights_coo"
        assert isinstance(weights, xr.DataArray), "weights must be an xarray DataArray"
    if weights_esmpy is not None:
        assert weights_coo is None, "Cannot provide both weights and weights_coo"
        assert isinstance(
            weights_esmpy, (xr.Dataset, str)
        ), "weights_esmpy must be an xarray Dataset or a path to a file"
        if isinstance(weights_esmpy, str):
            weights_esmpy = xr.open_dataset(weights_esmpy)
        if not isinstance(weights_esmpy, coo_matrix):
            assert {"S", "row", "col"}.issubset(
                weights_esmpy
            ), "weights_esmpy must contain 'S', 'row', and 'col'"
    if weights_coo is not None:
        assert isinstance(
            weights_coo, coo_matrix
        ), "weights_coo must be a scipy sparse COO matrix"

    # From 1D ESMF mesh to 2D grid
    src_grid = grid_from_esmf_mesh(src_mesh)
    dst_grid = grid_from_esmf_mesh(dst_mesh)

    # 1/3: Source Domain Fields
    # -------------------

    xc_a = xr.DataArray(
        src_mesh.centerCoords.data[:, 0],
        dims=["n_a"],
        attrs={
            "long_name": "longitude of grid cell center (input)",
            "units": "degrees east",
        },
    )

    yc_a = xr.DataArray(
        src_mesh.centerCoords.data[:, 1],
        dims=["n_a"],
        attrs={
            "long_name": "latitude of grid cell center (input)",
            "units": "degrees north",
        },
    )

    xv_a_data, yv_a_data = _construct_vertex_coords(src_mesh)

    xv_a = xr.DataArray(
        xv_a_data,
        dims=["n_a", "nv_a"],
        attrs={
            "long_name": "longitude of grid cell verticies (input)",
            "units": "degrees east",
        },
    )

    yv_a = xr.DataArray(
        yv_a_data,
        dims=["n_a", "nv_a"],
        attrs={
            "long_name": "latitude of grid cell verticies (input)",
            "units": "degrees north",
        },
    )

    mask_a = xr.DataArray(
        src_mesh.elementMask.data,
        dims=["n_a"],
        attrs={
            "long_name": "domain mask (input)",
        },
    )

    area_a = xr.DataArray(
        cell_area_rad(xv_a, yv_a),
        dims=["n_a"],
        attrs={"long_name": "area of cell (input)", "units": "radians^2"},
    )

    frac_a = xr.DataArray(
        src_mesh.elementMask.data.astype(np.float64),
        dims=["n_a"],
        attrs={
            "long_name": "fraction of domain intersection (input)",
            #'units': ''
        },
    )

    src_grid_dims = xr.DataArray(
        np.array(src_grid.mask.shape[::-1]).astype(np.int32),
        dims=["src_grid_rank"],
        # attrs={
        #    'long_name': 'dimensions of the source grid',
        # }
    )

    nj_a = xr.DataArray(
        [i + 1 for i in range(src_grid.mask.shape[0])],
        dims=["nj_a"],
    )

    ni_a = xr.DataArray(
        [i + 1 for i in range(src_grid.mask.shape[1])],
        dims=["ni_a"],
    )

    # 2/3: Destination Domain Fields
    # -------------------

    xc_b = xr.DataArray(
        dst_mesh.centerCoords.data[:, 0],
        dims=["n_b"],
        attrs={
            "long_name": "longitude of grid cell center (output)",
            "units": "degrees east",
        },
    )

    yc_b = xr.DataArray(
        dst_mesh.centerCoords.data[:, 1],
        dims=["n_b"],
        attrs={
            "long_name": "latitude of grid cell center (output)",
            "units": "degrees north",
        },
    )

    xv_b_data, yv_b_data = _construct_vertex_coords(dst_mesh)

    xv_b = xr.DataArray(
        xv_b_data,
        dims=["n_b", "nv_b"],
        attrs={
            "long_name": "longitude of grid cell verticies (output)",
            "units": "degrees east",
        },
    )

    yv_b = xr.DataArray(
        yv_b_data,
        dims=["n_b", "nv_b"],
        attrs={
            "long_name": "latitude of grid cell verticies (output)",
            "units": "degrees north",
        },
    )

    mask_b = xr.DataArray(
        dst_mesh.elementMask.data,
        dims=["n_b"],
        attrs={
            "long_name": "domain mask (output)",
        },
    )

    area_b = xr.DataArray(
        cell_area_rad(xv_b, yv_b),
        dims=["n_b"],
        attrs={"long_name": "area of cell (output)", "units": "radians^2"},
    )

    frac_b = xr.DataArray(
        dst_mesh.elementMask.data.astype(np.float64),
        dims=["n_b"],
        attrs={
            "long_name": "fraction of domain intersection (output)",
            #'units': ''
        },
    )

    dst_grid_dims = xr.DataArray(
        np.array(dst_grid.mask.shape[::-1]).astype(np.int32),
        dims=["dst_grid_rank"],
        # dst_grid_dimsattrs={
        #    'long_name': 'dimensions of the destination grid',
        # }
    )

    nj_b = xr.DataArray(
        [i + 1 for i in range(dst_grid.mask.shape[0])],
        dims=["nj_b"],
    )

    ni_b = xr.DataArray(
        [i + 1 for i in range(dst_grid.mask.shape[1])],
        dims=["ni_b"],
    )

    # 3/3: Weights
    # -------------------

    if weights is not None:  # xESMF weights
        w = weights.data.copy()
        col_data = w.coords[1, :] + 1
        row_data = w.coords[0, :] + 1
    elif weights_esmpy is not None:  # esmpy weights
        w = weights_esmpy.S.copy()
        col_data = weights_esmpy.col.data
        row_data = weights_esmpy.row.data
    elif weights_coo is not None:  # scipy sparse COO matrix
        w = weights_coo.data
        col_data = weights_coo.col + 1
        row_data = weights_coo.row + 1

    if area_normalization:
        w.data *= area_a.data[col_data - 1] / area_b.data[row_data - 1]

    S = xr.DataArray(
        w.data,
        dims=["n_s"],
        attrs={
            "long_name": "sparse matrix for mapping S:a->b",
        },
    )

    col = xr.DataArray(
        col_data,
        dims=["n_s"],
        attrs={
            "long_name": "column corresponding to matrix elements",
        },
    )

    row = xr.DataArray(
        row_data,
        dims=["n_s"],
        attrs={
            "long_name": "row corresponding to matrix elements",
        },
    )

    # Drop NaN values from S, col, and row
    non_nan_indices = np.where(~np.isnan(S))[0]
    S_new = S[non_nan_indices]
    row_new = row[non_nan_indices]
    col_new = col[non_nan_indices]

    # Create the dataset and write to a netCDF file
    # ------------------------------------------------

    ds = xr.Dataset(
        {
            "xc_a": xc_a,
            "yc_a": yc_a,
            "xv_a": xv_a,
            "yv_a": yv_a,
            "mask_a": mask_a,
            "area_a": area_a,
            "frac_a": frac_a,
            "src_grid_dims": src_grid_dims,
            "nj_a": nj_a,
            "ni_a": ni_a,
            "xc_b": xc_b,
            "yc_b": yc_b,
            "xv_b": xv_b,
            "yv_b": yv_b,
            "mask_b": mask_b,
            "area_b": area_b,
            "frac_b": frac_b,
            "dst_grid_dims": dst_grid_dims,
            "nj_b": nj_b,
            "ni_b": ni_b,
            "S": S_new,
            "col": col_new,
            "row": row_new,
        }
    )

    ds.attrs["title"] = "CESM mapping file generated by mom6_forge"
    # ds.attrs['normalization'] = 'conservative' # todo
    # ds.attrs['map_method'] = 'nearest neighbor smoothing'
    ds.attrs["history"] = "Generated by mom6_forge"
    ds.attrs["conventions"] = "NCAR-CCSM"
    ds.attrs["date_created"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if src_mesh_path := src_mesh.encoding.get("source"):
        ds.attrs["domain_a"] = src_mesh_path
    if dst_mesh_path := dst_mesh.encoding.get("source"):
        ds.attrs["domain_b"] = dst_mesh_path

    ds.to_netcdf(
        filename,
        format="NETCDF3_64BIT",
        encoding={var: {"_FillValue": None} for var in ds.data_vars},
    )


def generate_ESMF_map_via_xesmf(
    src_mesh_path,
    dst_mesh_path,
    mapping_file,
    method,
    area_normalization=False,
    map_overlap=True,
):
    """Generate an ESMF mapping file using xesmf.

    Parameters
    ----------
    src_mesh_path : str or Path
        Path to the source ESMF mesh file.
    dst_mesh_path : str or Path
        Path to the destination ESMF mesh file.
    mapping_file : str or Path
        Path to the output ESMF mapping file to be created.
    method : str
        Regridding method to use. Options are 'nearest_d2s', 'nearest_s2d', 'bilinear', 'conservative'.
    area_normalization : bool
        Whether to apply area normalization to the weights.
    map_overlap : bool
        If True, only map the overlapping area between the source and destination meshes, i.e.,
        zero out the mask in the source mesh that falls outside the rectangle defined by the destination mesh.
    """

    src_grid = grid_from_esmf_mesh(src_mesh_path)
    dst_grid = grid_from_esmf_mesh(dst_mesh_path)

    # from dst mesh, find the lower left corner and upper right corner
    # then, in the src mesh, zero out the mask falling outside of this rectangle
    if map_overlap:
        assert isinstance(
            dst_mesh_path, (str, Path)
        ), "dst_mesh_path must be a path to an existing file"
        assert Path(
            dst_mesh_path
        ).exists(), "dst_mesh_path must point to an existing ESMF mesh file"
        dst_mesh = xr.open_dataset(dst_mesh_path)
        assert (
            "units" in dst_mesh["centerCoords"].attrs
        ), "centerCoords must have 'units' attribute"
        assert (
            "degrees" in dst_mesh["centerCoords"].attrs["units"]
        ), "get_mesh_dimensions() expects centerCoords in degrees"
        dst_mesh_lon = normalize_deg(dst_mesh["nodeCoords"].data[:, 0])
        dst_mesh_lat = normalize_deg(dst_mesh["nodeCoords"].data[:, 1])
        lon_min, lon_max = dst_mesh_lon.min(), dst_mesh_lon.max()
        lat_min, lat_max = dst_mesh_lat.min(), dst_mesh_lat.max()

        # normalize source grid coordinates
        src_lon = normalize_deg(src_grid["lon"].data)
        src_lat = normalize_deg(src_grid["lat"].data)

        src_grid["mask"].data = np.where(
            (src_lon < lon_min)
            | (src_lon > lon_max)
            | (src_lat < lat_min)
            | (src_lat > lat_max),
            0,
            src_grid["mask"].data,
        )

    dst_is_cyclic_x = is_mesh_cyclic_x(dst_mesh_path)

    import xesmf as xe

    regridder = xe.Regridder(
        src_grid,
        dst_grid,
        method=method,
        periodic=dst_is_cyclic_x,
    )

    write_mapping_file(
        src_mesh=src_mesh_path,
        dst_mesh=dst_mesh_path,
        weights=regridder.weights,
        filename=mapping_file,
        area_normalization=area_normalization,
    )


def generate_ESMF_map_via_esmpy(
    src_mesh_path, dst_mesh_path, mapping_file, method, area_normalization
):
    """Generate an ESMF mapping file using esmpy.

    Parameters
    ----------
    src_mesh_path : str or Path
        Path to the source ESMF mesh file.
    dst_mesh_path : str or Path
        Path to the destination ESMF mesh file.
    mapping_file : str or Path
        Path to the output ESMF mapping file to be created.
    method : str
        Regridding method to use. Options are 'nearest_d2s', 'nearest_s2d', 'bilinear', 'conservative'.
    area_normalization : bool
        Whether to apply area normalization to the weights.
    """

    import esmpy

    assert isinstance(
        src_mesh_path, (str, Path)
    ), "src_mesh_path must be a path to an existing file"
    assert isinstance(
        dst_mesh_path, (str, Path)
    ), "dst_mesh_path must be a path to an existing file"

    match method:
        case "nearest_d2s":
            method = esmpy.RegridMethod.NEAREST_DTOS
        case "nearest_s2d":
            method = esmpy.RegridMethod.NEAREST_STOD
        case "bilinear":
            raise NotImplementedError("Bilinear regridding is not yet tested.")
        case "conservative":
            raise NotImplementedError("Conservative regridding is not yet tested.")
        case _:
            raise ValueError(f"Invalid regridding method: {method}")

    # Create src and dst meshes and fields
    src_mesh = esmpy.Mesh(
        filename=src_mesh_path,
        filetype=esmpy.FileFormat.ESMFMESH,
        mask_flag=esmpy.api.constants.MeshLoc.ELEMENT,
        varname="elementMask",
    )

    src_field = esmpy.Field(src_mesh, meshloc=esmpy.MeshLoc.ELEMENT)

    dst_mesh = esmpy.Mesh(
        filename=dst_mesh_path,
        filetype=esmpy.FileFormat.ESMFMESH,
        mask_flag=esmpy.api.constants.MeshLoc.ELEMENT,
        varname="elementMask",
    )

    dst_field = esmpy.Field(dst_mesh, meshloc=esmpy.MeshLoc.ELEMENT)

    # Run the regridder
    if rank() == 0 and os.path.exists(mapping_file):
        os.remove(mapping_file)

    if MPI:
        MPI.COMM_WORLD.Barrier()

    # Create regridder and save the weights to the mapping file temporarily
    # This file will later be extended to include the full ESMF map fields
    esmpy.Regrid(
        src_field,
        dst_field,
        filename=mapping_file,
        src_mask_values=np.array([0.0]),
        dst_mask_values=np.array([0.0]),
        regrid_method=method,
        unmapped_action=esmpy.UnmappedAction.ERROR,
    )

    # Now, read in the weights from the mapping file
    weights_ds = xr.open_dataset(mapping_file)

    write_mapping_file(
        src_mesh=src_mesh_path,
        dst_mesh=dst_mesh_path,
        filename=mapping_file,
        weights_esmpy=weights_ds,
        area_normalization=area_normalization,
    )


def compute_smoothing_weights(mesh_ds, rmax, fold=1.0, xv_data=None, yv_data=None):
    """Compute smoothing weights for a given mesh dataset, using a radius in kilometers.

    Parameters
    ----------
    mesh_ds : xr.Dataset
        The ESMF mesh dataset.
    rmax : float
        Maximum distance for smoothing weights, in kilometers.
    fold : float
        Fold factor (km) determining the strength of smoothing based on distance.
    xv_data : np.ndarray, optional
        The x-coordinates of the vertices. If not provided, they will be computed from the mesh dataset.
    yv_data : np.ndarray, optional
        The y-coordinates of the vertices. If not provided, they will be computed from the mesh dataset.

    Returns
    -------
    weights : scipy.sparse.coo_matrix
        The computed smoothing weights in coordinate (COO) format.
    """

    if not isinstance(mesh_ds, xr.Dataset):
        raise ValueError("mesh_ds must be an xarray Dataset.")

    # Extract coordinates and mask
    coords = mesh_ds["centerCoords"].values
    if xv_data is None or yv_data is None:
        xv_data, yv_data = _construct_vertex_coords(mesh_ds)
    areas = cell_area_rad(xv_data, yv_data)
    mask = mesh_ds["elementMask"].values
    mask_bool = mask != 0

    # Convert lon/lat (degrees) to radians
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])

    # Earth's radius in kilometers
    R_earth = 6371.0

    # Convert lon/lat to 3D Cartesian coordinates for great-circle distance calculation
    x = R_earth * np.cos(lat) * np.cos(lon)
    y = R_earth * np.cos(lat) * np.sin(lon)
    z = R_earth * np.sin(lat)
    xyz = np.stack([x, y, z], axis=1)

    # Build KDTree in 3D Cartesian space
    tree = cKDTree(xyz)
    indices = tree.query_ball_tree(tree, rmax)

    row_indices, col_indices, data = [], [], []

    for i, neighbors in enumerate(indices):
        if not mask_bool[i]:
            continue
        neighbors = np.array(neighbors)
        neighbors = neighbors[mask_bool[neighbors]]
        row_indices.extend([i] * len(neighbors))
        col_indices.extend(neighbors)
        # Compute great-circle distances in km
        d_xyz = xyz[i] - xyz[neighbors]
        # Chord length
        chord = np.linalg.norm(d_xyz, axis=1)
        # Convert chord length to arc length (distance on sphere)
        arc = 2 * R_earth * np.arcsin(np.clip(chord / (2 * R_earth), 0, 1))
        weights_data = np.exp(-arc / fold)
        # Normalize by area
        weights_data /= np.sum(weights_data * (areas[neighbors] / areas[i]))
        data.extend(weights_data)

    weights = coo_matrix(
        (data, (row_indices, col_indices)), shape=(len(coords), len(coords))
    )
    return weights


def get_nn_map_filepath(mapping_file_prefix, output_dir):
    """Get the standardized runoff to ocean nearest neighbor mapping file path
    based on the prefix and output directory.

    Parameters
    ----------
    mapping_file_prefix : str
        Prefix for the mapping files to be created.
    output_dir : str or Path
        Directory where the mapping files will be saved.

    Returns
    -------
    nn_map_filepath : Path
        Path to the nearest neighbor mapping file.
    """
    output_dir = Path(output_dir)
    nn_map_filepath = output_dir / f"{mapping_file_prefix}_nn.nc"
    return nn_map_filepath


def get_smoothed_map_filepath(mapping_file_prefix, output_dir, rmax, fold):
    """Get the standardized runoff to ocean smoothed nearest neighbor mapping file name
    based on the prefix, output directory, rmax, and fold.

    Parameters
    ----------
    mapping_file_prefix : str
        Prefix for the mapping files to be created.
    output_dir : str or Path
        Directory where the mapping files will be saved.
    rmax : float
        Maximum distance for smoothing weights, in kilometers.
    fold : float
        Fold factor (km) determining the strength of smoothing based on distance.

    Returns
    -------
    nnsm_map_filepath : Path
        Path to the smoothed nearest neighbor mapping file.
    """
    output_dir = Path(output_dir)
    nnsm_map_filepath = (
        output_dir / f"{mapping_file_prefix}_r{int(rmax)}_f{int(fold)}_nnsm.nc"
    )
    return nnsm_map_filepath


def gen_rof_maps(
    rof_mesh_path, ocn_mesh_path, output_dir, mapping_file_prefix, rmax=None, fold=None
):
    """Generate (1) nearest neighbor and (2) smoothed nearest neighbor mapping files
    from a runoff mesh to an ocean mesh.

    Parameters
    ----------
    rof_mesh_path : str or Path
        Path to the ROF ESMF mesh file.
    ocn_mesh_path : str or Path
        Path to the OCN ESMF mesh file.
    output_dir : str or Path
        Directory where the mapping files will be saved.
    mapping_file_prefix : str
        Prefix for the mapping files to be created. Example: "rx1_to_t232" will create files
        "rx1_to_t232_nn.nc" and "rx1_to_t232_nnsm.nc".
    rmax : float, optional
        Maximum distance for smoothing weights, in kilometers. If None, no smoothing is applied.
    fold : float, optional
        Fold factor (km) determining the strength of smoothing based on distance. If None, no smoothing is applied.
    """

    print(f"Generating nearest neighbor mapping file...")
    print(f"  This may take a while depending on the mesh sizes.")
    t0 = time()

    if rmax is None or fold is None:
        assert (
            rmax == fold == None
        ), "If rmax is not provided, fold must also be None, and vice versa."

    assert (
        isinstance(ocn_mesh_path, (str, Path)) and Path(ocn_mesh_path).exists()
    ), "ocn_mesh_path must be a path to an existing file"
    assert (
        isinstance(rof_mesh_path, (str, Path)) and Path(rof_mesh_path).exists()
    ), "rof_mesh_path must be a path to an existing file"
    assert isinstance(output_dir, (str, Path)), "output_dir must be a string or Path"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nn_map_filepath = get_nn_map_filepath(mapping_file_prefix, output_dir)

    generate_ESMF_map_via_xesmf(
        src_mesh_path=rof_mesh_path,
        dst_mesh_path=ocn_mesh_path,
        mapping_file=nn_map_filepath,
        method="nearest_d2s",
        area_normalization=True,
    )

    print(f"  Generated nearest mapping file in {(t1:=time()) - t0:.2f} seconds at:")
    print(f"  {nn_map_filepath}")

    # Compute and apply smoothing weights
    if rmax is not None:

        print(f"Generating smoothed nearest neighbor mapping file...")
        print(
            f"  This may take a while depending on the mesh sizes and smoothing parameters:"
        )
        print(f"  rmax = {rmax} km, fold = {fold} km")

        nn_map = xr.open_dataset(nn_map_filepath)
        avg_dst_res_km = get_avg_resolution_km(ocn_mesh_path)
        if rmax > avg_dst_res_km * 10:
            print(
                f"Warning: rmax ({rmax} km) is significantly larger than the average resolution of the destination mesh ({avg_dst_res_km} km)."
            )
            print(
                "This may lead to excessive memory usage, long computation times, or crashes."
            )

        ocn_mesh = xr.open_dataset(ocn_mesh_path)

        sw = compute_smoothing_weights(
            mesh_ds=ocn_mesh,
            rmax=rmax,
            fold=fold,
        )

        # mesh dimensions
        rof_nx = len(nn_map.ni_a)
        rof_ny = len(nn_map.nj_a)
        ocn_nx = len(nn_map.ni_b)
        ocn_ny = len(nn_map.nj_b)

        # Create a COO matrix of the mapping weights
        S_coo = coo_matrix(
            (nn_map.S.data, (nn_map.row.data - 1, nn_map.col.data - 1)),
            shape=(ocn_nx * ocn_ny, rof_nx * rof_ny),
        )

        # Apply smoothing by multiplying (transpose of) S_coo and smoothing weights (and taking transpose again):
        S_smooth = S_coo.transpose().dot(sw).transpose()
        assert isinstance(
            S_smooth, csc_matrix
        ), "S_smooth should be a csc_matrix after multiplication"
        S_smooth = S_smooth.tocoo()  # Convert to COO format again

        nnsm_map_filepath = get_smoothed_map_filepath(
            mapping_file_prefix, output_dir, rmax, fold
        )

        write_mapping_file(
            src_mesh=rof_mesh_path,
            dst_mesh=ocn_mesh_path,
            filename=nnsm_map_filepath,
            weights_coo=S_smooth,
            area_normalization=False,  # No area normalization for smoothing
        )

        print(f"  Generated smoothed mapping file in {time() - t1:.2f} seconds at:")
        print(f"  {nnsm_map_filepath}")

    print("Done generating runoff to ocean mapping file(s).")


def regrid_dataset_via_xesmf(
    input_dataset,
    output_dataset,
    regridding_method=None,
    write_to_file=False,
    output_path=Path("regridded_dataset.nc"),
    weights_path=None,
    reuse_weights=False,
    locstream_out=False,
    periodic=False,
):
    """
    Regrids the dataset given ``input_dataset`` which contains the original dataset and ``output_dataset`` which is a template for the regridded product.
    Uses xESMF for regridding.

    Args:
        input_dataset (Xarray Dataset): original dataset with proper  metadata and structure for ESMF regridding.
        output_dataset (Xarray Dataset): Template for the regridded dataset regridding_method: (Optional[str]) The type of regridding method to use. Defaults to bilinear
        write_to_file (Optional[bool]): Files saved to ``output_path`` Defaults to ``False``. Must be set to true if using manual regridding methods with ESMF_regrid.
        weights_path (Optional[str]): Path to pre-computed regridding weights file.
        output_path (Optional[str]): Path to save the regridded dataset if ``write_to_file`` is True. Defaults to "regridded_dataset.nc".
        reuse_weights (Optional[bool]): Whether to reuse the weights from ``weights_path`` if provided. Defaults to False. If False, weights will be recomputed even if ``weights_path`` is provided. If True, weights will be reused from ``weights_path`` if provided, and an error will be raised if ``weights_path`` is not provided.
        locstream_out (Optional[bool]): Whether the output grid is a location stream. Defaults to False.
        periodic (Optional[bool]): Whether the grid is periodic in the longitude direction. Defaults to False.
    Returns:
        regridded_dataset (Xarray.Dataset):
    """

    if regridding_method is None:
        regridding_method = "bilinear"

    input_dataset = input_dataset.load()

    print(
        "Begin regridding dataset...\n\n"
        + f"Original dataset size: {input_dataset.nbytes/1e6:.2f} Mb\n"
        + f"Regridded size: {output_dataset.nbytes/1e6:.2f} Mb\n"
    )

    if not reuse_weights:
        print(
            f"Generating regridding weights using xESMF with method '{regridding_method}'..."
        )
        regridder = xe.Regridder(
            input_dataset,
            output_dataset,
            method=regridding_method,
            locstream_out=locstream_out,
            periodic=periodic,
        )
    else:
        assert (
            weights_path is not None and Path(weights_path).exists()
        ), "weights_path must be provided and exist if reuse_weights is True"
        print(f"Using pre-computed regridding weights from {weights_path}...")
        regridder = xe.Regridder(
            input_dataset,
            output_dataset,
            method=regridding_method,  # This argument is required but ignored when reuse_weights=True
            locstream_out=locstream_out,
            periodic=periodic,
            weights=weights_path,
            reuse_weights=reuse_weights,
        )

    dataset = regridder(input_dataset)

    if write_to_file:
        dataset.to_netcdf(
            output_path,
            mode="w",
            engine="netcdf4",
        )

    return dataset


def compute_cressman_weights(
    src_ds: xr.Dataset,
    dst_ds: xr.Dataset,
    smooth_scl: float = 2.0,
    cressman_exp: float = 2.0,
    radius: float = 6_371_000.0,
) -> xr.Dataset:
    """Compute Cressman distance-weighted interpolation weights from a regular
    source grid to a model destination grid. Mirrors ``interp_smooth.f90`` from
    the tx2_3 topography workflow.

    For each ocean destination cell ``i``:

    * The smoothing radius is ``L = smooth_scl * sqrt(dst_area[i])`` (metres).
    * All source ocean points within ``L`` are gathered.
    * Weights are ``w = ((L²-r²)/(L²+r²))^cressman_exp`` where ``r`` is the
      great-circle arc distance from the model cell centre to the source point.
    * Row ``i`` of the returned matrix holds the normalised weights
      (sum = 1) for destination cell ``i``.

    Parameters
    ----------
    src_ds : xr.Dataset
        Must have 1D coordinates ``lon`` and ``lat``, and a 2D variable ``depth``
        of shape (ny_src, nx_src), positive-down. Land cells should be <= 0.
    dst_ds : xr.Dataset
        Must have 2D variables or coordinates ``lon``, ``lat``, ``area``, and
        ``mask`` of shape (ny_dst, nx_dst).
    smooth_scl : float
        Smoothing scale multiplier: ``L = smooth_scl * sqrt(cell_area)``.
        Default ``2.0`` (matches tx2_3 default).
    cressman_exp : float
        Exponent for the Cressman weight function. Default ``2.0``.
    radius : float
        Earth radius in metres. Default is 6,371,000 m.

    Returns
    -------
    xr.Dataset
        ESMF-compatible weights dataset with the following variables:

        - ``S``: non-zero Cressman weights, shape (n_s,)
        - ``row``: destination cell index for each weight, 1-based, shape (n_s,)
        - ``col``: source cell index for each weight, 1-based, shape (n_s,)
        - ``unfilled``: bool mask, True for ocean destination cells that had no
          source ocean point within ``L`` and need a fallback fill, shape (n_b,)
        - source/destination grid metadata variables (``xc_a``, ``yc_a``, ``xc_b``,
          ``yc_b``, ``mask_a``, ``mask_b``, ``area_b``, etc.)
    """
    for var in ("lon", "lat"):
        assert var in src_ds.coords, f"src_ds missing coordinate: '{var}'"
        assert src_ds[var].ndim == 1, f"src_ds['{var}'] must be 1D (regular grid)"

    assert "depth" in src_ds, "src_ds missing variable: 'depth'"
    assert src_ds["depth"].ndim == 2, "src_ds['depth'] must be 2D (ny, nx)"

    for var in ("lon", "lat", "area", "mask"):
        assert var in dst_ds or var in dst_ds.coords, f"dst_ds missing: '{var}'"
        arr = dst_ds[var]
        assert arr.ndim == 2, f"dst_ds['{var}'] must be 2D (ny, nx)"

    g = dict(
        src_lon=src_ds["lon"].values,
        src_lat=src_ds["lat"].values,
        src_depth=src_ds["depth"].values,
        dst_lon=dst_ds["lon"].values,
        dst_lat=dst_ds["lat"].values,
        dst_area=dst_ds["area"].values,
        dst_mask=dst_ds["mask"].values.astype(bool),
    )

    R = radius

    src_ocean_mask = g["src_depth"] > 0.0

    # --- Source grid: 2-D coordinates and Cartesian points ---
    src_lon_2d, src_lat_2d = np.meshgrid(g["src_lon"], g["src_lat"])  # (ny_src, nx_src)
    n_src = src_lon_2d.size
    lon_src_rad = np.deg2rad(src_lon_2d.ravel())
    lat_src_rad = np.deg2rad(src_lat_2d.ravel())
    xyz_src = np.stack(
        [
            R * np.cos(lat_src_rad) * np.cos(lon_src_rad),
            R * np.cos(lat_src_rad) * np.sin(lon_src_rad),
            R * np.sin(lat_src_rad),
        ],
        axis=1,
    )

    # KDTree only over ocean source points
    ocean_src_idx = np.where(src_ocean_mask.ravel())[0]
    tree = cKDTree(xyz_src[ocean_src_idx])

    # --- Destination grid: Cartesian points ---
    dst_lon_flat = g["dst_lon"].ravel()
    dst_lat_flat = g["dst_lat"].ravel()
    dst_area_flat = g["dst_area"].ravel()
    dst_mask_flat = g["dst_mask"].ravel()

    lon_dst_rad = np.deg2rad(dst_lon_flat)
    lat_dst_rad = np.deg2rad(dst_lat_flat)
    xyz_dst = np.stack(
        [
            R * np.cos(lat_dst_rad) * np.cos(lon_dst_rad),
            R * np.cos(lat_dst_rad) * np.sin(lon_dst_rad),
            R * np.sin(lat_dst_rad),
        ],
        axis=1,
    )

    rows, cols, data = [], [], []
    unfilled = np.zeros(len(dst_lon_flat), dtype=bool)

    for i in np.where(dst_mask_flat)[0]:
        L = smooth_scl * np.sqrt(dst_area_flat[i])  # metres
        L2 = L**2

        neighbors = tree.query_ball_point(xyz_dst[i], L)
        if not neighbors:
            unfilled[i] = True
            continue

        nbr = np.asarray(neighbors)
        d_xyz = xyz_src[ocean_src_idx[nbr]] - xyz_dst[i]
        chord = np.linalg.norm(d_xyz, axis=1)
        arc = 2.0 * R * np.arcsin(np.clip(chord / (2.0 * R), 0.0, 1.0))
        d2 = arc**2

        in_radius = d2 <= L2
        if not in_radius.any():
            unfilled[i] = True
            continue

        w = ((L2 - d2[in_radius]) / (L2 + d2[in_radius])) ** cressman_exp
        w_sum = w.sum()
        if w_sum == 0.0:
            unfilled[i] = True
            continue

        src_cols = ocean_src_idx[nbr[in_radius]]
        rows.extend([i] * int(in_radius.sum()))
        cols.extend(src_cols.tolist())
        data.extend((w / w_sum).tolist())

    S = csr_matrix((data, (rows, cols)), shape=(len(dst_lon_flat), n_src))

    S_coo = S.tocoo()
    ny_src, nx_src = src_lon_2d.shape
    ny_dst, nx_dst = g["dst_lon"].shape
    # Trying to copy the weights writing function above without the esmf files.
    ds = xr.Dataset(
        {
            # --- sparse weight triplets ---
            "S": xr.DataArray(
                S_coo.data.astype(np.float64),
                dims=["n_s"],
                attrs={"long_name": "Cressman interpolation weight"},
            ),
            "row": xr.DataArray(
                (S_coo.row + 1).astype(np.int32),
                dims=["n_s"],
                attrs={"long_name": "destination cell index (1-based)"},
            ),
            "col": xr.DataArray(
                (S_coo.col + 1).astype(np.int32),
                dims=["n_s"],
                attrs={"long_name": "source cell index (1-based)"},
            ),
            # --- source grid ---
            "xc_a": xr.DataArray(
                src_lon_2d.ravel(),
                dims=["n_a"],
                attrs={
                    "long_name": "source grid center longitude",
                    "units": "degrees_east",
                },
            ),
            "yc_a": xr.DataArray(
                src_lat_2d.ravel(),
                dims=["n_a"],
                attrs={
                    "long_name": "source grid center latitude",
                    "units": "degrees_north",
                },
            ),
            "mask_a": xr.DataArray(
                src_ocean_mask.ravel().astype(np.int32),
                dims=["n_a"],
                attrs={"long_name": "source grid mask", "units": "unitless"},
            ),
            "area_a": xr.DataArray(
                np.zeros(n_src),
                dims=["n_a"],
                attrs={
                    "long_name": "[PLACEHOLDER NOT FILLED] source grid area",
                    "units": "radians^2",
                },
            ),
            "frac_a": xr.DataArray(
                src_ocean_mask.ravel().astype(np.float64),
                dims=["n_a"],
                attrs={
                    "long_name": "[PLACEHOLDER NOT FILLED] source grid fraction",
                    "units": "unitless",
                },
            ),
            "nj_a": xr.DataArray(
                ny_src, attrs={"long_name": "source grid j dimension"}
            ),
            "ni_a": xr.DataArray(
                nx_src, attrs={"long_name": "source grid i dimension"}
            ),
            "src_grid_dims": xr.DataArray(
                np.array([ny_src, nx_src], dtype=np.int32),
                dims=["src_grid_rank"],
                attrs={"long_name": "source grid dimensions"},
            ),
            # --- destination grid ---
            "xc_b": xr.DataArray(
                g["dst_lon"].ravel(),
                dims=["n_b"],
                attrs={
                    "long_name": "destination grid center longitude",
                    "units": "degrees_east",
                },
            ),
            "yc_b": xr.DataArray(
                g["dst_lat"].ravel(),
                dims=["n_b"],
                attrs={
                    "long_name": "destination grid center latitude",
                    "units": "degrees_north",
                },
            ),
            "mask_b": xr.DataArray(
                dst_mask_flat.astype(np.int32),
                dims=["n_b"],
                attrs={"long_name": "destination grid mask", "units": "unitless"},
            ),
            "area_b": xr.DataArray(
                dst_area_flat,
                dims=["n_b"],
                attrs={"long_name": "destination grid area", "units": "m^2"},
            ),
            "frac_b": xr.DataArray(
                dst_mask_flat.astype(np.float64),
                dims=["n_b"],
                attrs={
                    "long_name": "[PLACEHOLDER NOT FILLED] destination grid fraction",
                    "units": "unitless",
                },
            ),
            "nj_b": xr.DataArray(
                ny_dst, attrs={"long_name": "destination grid j dimension"}
            ),
            "ni_b": xr.DataArray(
                nx_dst, attrs={"long_name": "destination grid i dimension"}
            ),
            "dst_grid_dims": xr.DataArray(
                np.array([ny_dst, nx_dst], dtype=np.int32),
                dims=["dst_grid_rank"],
                attrs={"long_name": "destination grid dimensions"},
            ),
            # --- unfilled flag (non-standard but useful) ---
            "unfilled": xr.DataArray(
                unfilled,
                dims=["n_b"],
                attrs={
                    "long_name": "True for ocean destination cells with no source points within smoothing radius"
                },
            ),
        }
    )

    ds.attrs["title"] = "Cressman interpolation weights generated by mom6_forge"
    ds.attrs["history"] = "Generated by mom6_forge"
    ds.attrs["date_created"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ds


def regrid_dataset_via_cressman(
    src_ds: xr.Dataset,
    dst_ds: xr.Dataset,
    weights_path: Path,
    smooth_scl: float = 2.0,
    cressman_exp: float = 2.0,
    write_to_file: bool = False,
    output_path: Path = Path("cressman_regridded.nc"),
) -> tuple[xr.Dataset, np.ndarray]:
    """Compute mask-aware Cressman weights, save to an ESMF weights file, and regrid
    source depths to the destination model grid using ``xe.Regridder``.

    Parameters
    ----------
    src_ds : xr.Dataset
        Must have 1D coordinates ``lon`` and ``lat``, and a 2D variable ``depth``
        of shape (ny_src, nx_src), positive-down. Land cells should be <= 0.
    dst_ds : xr.Dataset
        Must have 2D variables or coordinates ``lon``, ``lat``, ``area``, and
        ``mask`` of shape (ny_dst, nx_dst).
    weights_path : str or Path
        Path where the ESMF-compatible weights netCDF will be written.
        If the file already exists it is **overwritten**.
    smooth_scl : float
        Cressman smoothing scale multiplier. Default ``2.0``.
    cressman_exp : float
        Cressman weight function exponent. Default ``2.0``.
    write_to_file : bool
        If True, save the regridded dataset to ``output_path``. Default ``False``.
    output_path : Path
        Path for the regridded output netCDF. Default ``cressman_regridded.nc``.

    Returns
    -------
    depth_dst : xr.Dataset, shape (ny_dst, nx_dst)
        Cressman-interpolated depth field. Unfilled ocean cells are 0.
    unfilled : np.ndarray of bool, shape (ny_dst, nx_dst)
        True for ocean cells that had no source ocean point within radius L
        and therefore need a fallback (e.g. iterative neighbour fill).
    """
    # --- Compute per-cell Cressman weights ---
    print("Computing Cressman weights...")
    weights_ds = compute_cressman_weights(
        src_ds,
        dst_ds,
        smooth_scl=smooth_scl,
        cressman_exp=cressman_exp,
    )

    # --- Save weights to ESMF-compatible netCDF ---
    weights_ds.to_netcdf(weights_path)
    print(f"Cressman weights written to {weights_path}")

    # --- Regrid using xesmf Regridder from the pre-computed weights ---
    # The reason to do this is because it handles seams well
    depth_dst = regrid_dataset_via_xesmf(
        input_dataset=src_ds,
        output_dataset=dst_ds,
        regridding_method="bilinear",  # THis doesn't matter if you provide the weights directly, but we need to specify something to initialize the regridder
        weights_path=weights_path,
        write_to_file=write_to_file,  # We'll write the final regridded dataset at the end of this function
        output_path=output_path,
        reuse_weights=True,
    )

    unfilled = weights_ds["unfilled"].values.reshape(dst_ds["lon"].shape)

    return depth_dst, unfilled


def source_to_dst(ds_weights, src_idx_2d):
    """
    Given a 2D source index and a weights file, find which destination cells it contributes to.

    Parameters
    ----------
    ds_weights  : xr.Dataset   SCRIP-format weight dataset
    src_idx_2d  : tuple (j, i) 2D index into source grid
    """

    src_shape = (int(ds_weights["nj_a"].values), int(ds_weights["ni_a"].values))
    dst_shape = (int(ds_weights["nj_b"].values), int(ds_weights["ni_b"].values))
    n_src = int(ds_weights.sizes["n_a"])
    n_dst = int(ds_weights.sizes["n_b"])

    row = ds_weights["row"].values - 1
    col = ds_weights["col"].values - 1
    data = ds_weights["S"].values
    S = coo_matrix((data, (row, col)), shape=(n_dst, n_src)).tocsc()

    src_flat = np.ravel_multi_index(src_idx_2d, src_shape)
    col_vec = S.getcol(src_flat)
    dst_indices = col_vec.nonzero()[0]
    weights = np.asarray(col_vec[dst_indices].todense()).ravel()

    print(
        f"Source {src_idx_2d} (flat={src_flat}) → {len(dst_indices)} destination cells:"
    )
    for dst, w in zip(dst_indices, weights):
        print(f"  dst {np.unravel_index(dst, dst_shape)} (flat={dst})  weight={w:.4f}")

    return dst_indices, weights


def dst_to_source(ds_weights, dst_idx_2d):
    """
    Given a 2D destination index and a weights file, find which source pixels contribute to it.

    Parameters
    ----------
    ds_weights  : xr.Dataset   SCRIP-format weight dataset
    dst_idx_2d  : tuple (j, i) 2D index into destination grid
    """

    src_shape = (int(ds_weights["nj_a"].values), int(ds_weights["ni_a"].values))
    dst_shape = (int(ds_weights["nj_b"].values), int(ds_weights["ni_b"].values))
    n_src = int(ds_weights.sizes["n_a"])
    n_dst = int(ds_weights.sizes["n_b"])

    row = ds_weights["row"].values - 1
    col = ds_weights["col"].values - 1
    data = ds_weights["S"].values
    S = coo_matrix((data, (row, col)), shape=(n_dst, n_src)).tocsr()

    dst_flat = np.ravel_multi_index(dst_idx_2d, dst_shape)
    row_vec = S.getrow(dst_flat)
    src_indices = row_vec.nonzero()[1]
    weights = np.asarray(row_vec[0, src_indices].todense()).ravel()

    print(f"Dest {dst_idx_2d} (flat={dst_flat}) ← {len(src_indices)} source pixels:")
    for src, w in zip(src_indices, weights):
        print(f"  src {np.unravel_index(src, src_shape)} (flat={src})  weight={w:.4f}")

    return src_indices, weights


def _make_subgrid_points(qlon, qlat, nx_sub, ny_sub):
    """
    Given corner coordinates of a low-res grid (qlon/qlat on the q-point staggering),
    return sub-sampled lon/lat arrays of shape (ny, nx, ny_sub, nx_sub). (Originally created by Frank Bryan in Fortran for NCAR/tx2_3, reimplemented in Python)

    Parameters
    ----------
    qlon, qlat : np.ndarray  shape (ny+1, nx+1)
        Corner (q-point) coordinates of the low-res grid.
    nx_sub, ny_sub : int
        Number of sub-points per cell in each direction.

    Returns
    -------
    sub_lon, sub_lat : np.ndarray  shape (ny, nx, ny_sub, nx_sub)
    """
    assert isinstance(qlon, np.ndarray), "qlon must be a numpy array"
    assert isinstance(qlat, np.ndarray), "qlat must be a numpy array"

    SW_lon = qlon[:-1, :-1]
    SW_lat = qlat[:-1, :-1]
    SE_lon = qlon[:-1, 1:]
    SE_lat = qlat[:-1, 1:]
    NE_lon = qlon[1:, 1:]
    NE_lat = qlat[1:, 1:]
    NW_lon = qlon[1:, :-1]
    NW_lat = qlat[1:, :-1]

    # Fix antimeridian-straddling cells
    def _fix_period(lon, ref):
        lon = np.where(lon - ref > 270, lon - 360, lon)
        lon = np.where(lon - ref < -270, lon + 360, lon)
        return lon

    SW_lon = _fix_period(SW_lon, NE_lon)
    SE_lon = _fix_period(SE_lon, NE_lon)
    NW_lon = _fix_period(NW_lon, NE_lon)

    ifrac = (np.arange(1, nx_sub + 1) / (nx_sub + 1)).astype(float)
    jfrac = (np.arange(1, ny_sub + 1) / (ny_sub + 1)).astype(float)

    i_ = ifrac[np.newaxis, np.newaxis, np.newaxis, :]  # (1,1,1,nx_sub)
    j_ = jfrac[np.newaxis, np.newaxis, :, np.newaxis]  # (1,1,ny_sub,1)

    # Broadcast all corners to (ny, nx, 1, 1),
    SW_lon = SW_lon[:, :, np.newaxis, np.newaxis]
    SE_lon = SE_lon[:, :, np.newaxis, np.newaxis]
    NE_lon = NE_lon[:, :, np.newaxis, np.newaxis]
    NW_lon = NW_lon[:, :, np.newaxis, np.newaxis]
    SW_lat = SW_lat[:, :, np.newaxis, np.newaxis]
    SE_lat = SE_lat[:, :, np.newaxis, np.newaxis]
    NE_lat = NE_lat[:, :, np.newaxis, np.newaxis]
    NW_lat = NW_lat[:, :, np.newaxis, np.newaxis]

    sub_lon = (
        (1 - i_) * (1 - j_) * SW_lon
        + i_ * (1 - j_) * SE_lon
        + i_ * j_ * NE_lon
        + (1 - i_) * j_ * NW_lon
    )
    sub_lat = (
        (1 - i_) * (1 - j_) * SW_lat
        + i_ * (1 - j_) * SE_lat
        + i_ * j_ * NE_lat
        + (1 - i_) * j_ * NW_lat
    )

    return sub_lon, sub_lat


def regrid_with_subsampling(
    input_dataset,
    qlon,
    qlat,
    nx_sub,
    ny_sub,
    regridding_method="nearest_s2d",
    regridder=None,
):
    """
    Regrids input_dataset to sub_sampled_grid to
    properly analyze high-res source data into each coarse cell.  (Originally created by Frank Bryan in Fortran for NCAR/tx2_3, reimplemented in Python)

    Parameters
    ----------
    input_dataset : xr.Dataset (not curvilinear)
    qlon, qlat : np.ndarray  shape (ny+1, nx+1)
        Corner coordinates of the destination grid.
    nx_sub, ny_sub : int
        Number of sub-points per cell (typically from compute_subsampling_factor).
    Returns
    -------
    regridded_dataset : xr.Dataset
            Regridded dataset with dimensions (..., ny, nx, ny_sub, nx_sub), where the sub-sampling points are kept as separate dimensions. (User should perform stats calc)
    """
    assert len(input_dataset.lon.dims) == 1 and input_dataset.lat.dims == (
        "lat",
    ), "input_dataset must have 1D 'lon' and 'lat' coordinates"
    ny, nx = qlon.shape[0] - 1, qlon.shape[1] - 1

    # Build the (ny, nx, ny_sub, nx_sub) sub-point grid
    sub_lon, sub_lat = _make_subgrid_points(qlon, qlat, nx_sub, ny_sub)

    # Flatten to (ny, nx*ny_sub*nx_sub) so xesmf sees a 2D destination
    flat_lon = sub_lon.reshape(ny, nx * ny_sub * nx_sub)
    flat_lat = sub_lat.reshape(ny, nx * ny_sub * nx_sub)

    flat_output = xr.Dataset(
        {
            "lon": xr.DataArray(flat_lon, dims=["ny", "nx"]),
            "lat": xr.DataArray(flat_lat, dims=["ny", "nx"]),
        }
    )

    if regridder is None:
        regridder = xe.Regridder(
            input_dataset,
            flat_output,
            method=regridding_method,
            locstream_out=False,
            periodic=False,
        )

    regridded_flat = regridder(input_dataset)

    # Reshape to 4D, keeping sub-points as their own dimension
    data_vars = {}
    for var in regridded_flat.data_vars:
        data = regridded_flat[var].values  # (..., ny, nx*ny_sub*nx_sub)
        reshaped = data.reshape(*data.shape[:-2], ny, nx, ny_sub, nx_sub)

        original_dims = regridded_flat[var].dims
        new_dims = (*original_dims[:-2], "ny", "nx", "ny_sub", "nx_sub")

        data_vars[var] = xr.DataArray(
            reshaped,
            dims=new_dims,
            attrs=regridded_flat[var].attrs,
        )

    coords = {
        k: v
        for k, v in regridded_flat.coords.items()
        if "ny" not in v.dims and "nx" not in v.dims
    }
    coords["ny_sub"] = np.arange(ny_sub)
    coords["nx_sub"] = np.arange(nx_sub)

    return xr.Dataset(data_vars, coords=coords, attrs=input_dataset.attrs), regridder


def main(args):

    if args.parallel:
        global MPI
        from mpi4py import MPI

    if args.override and Path(args.mapping_file).exists() and rank() == 0:
        os.remove(args.mapping_file)

    if not isinstance(args.src_mesh, (str, Path)) or not Path(args.src_mesh).exists():
        raise ValueError("src_mesh must be a path to an existing ESMF mesh file.")
    if not isinstance(args.dst_mesh, (str, Path)) or not Path(args.dst_mesh).exists():
        raise ValueError("dst_mesh must be a path to an existing ESMF mesh file.")
    if (
        not isinstance(args.mapping_file, (str, Path))
        or Path(args.mapping_file).exists()
    ):
        raise ValueError("mapping_file must be a path to a new ESMF mapping file.")

    if args.xesmf:
        generate_ESMF_map_via_xesmf(
            src_mesh_path=args.src_mesh,
            dst_mesh_path=args.dst_mesh,
            mapping_file=args.mapping_file,
            method=args.method,
            area_normalization=args.area_normalization,
        )

    else:
        generate_ESMF_map_via_esmpy(
            src_mesh_path=args.src_mesh,
            dst_mesh_path=args.dst_mesh,
            mapping_file=args.mapping_file,
            method=args.method,
            area_normalization=args.area_normalization,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map 1D ESMF mesh to a 2D horizontal grid"
    )
    parser.add_argument(
        "--src_mesh", type=str, help="Path to the source ESMF mesh file or dataset"
    )
    parser.add_argument(
        "--dst_mesh", type=str, help="Path to the destination ESMF mesh file or dataset"
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        help="Path to the output ESMF mapping file to be created",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["nearest_d2s", "nearest_s2d", "bilinear", "conservative"],
        help="Regridding method to use",
        default="nearest_d2s",
    )
    parser.add_argument(
        "--area_normalization",
        action="store_true",
        help="Whether to apply area normalization to the weights",
        default=False,
    )
    parser.add_argument(
        "--xesmf",
        action="store_true",
        help="Use xESMF for regridding instead of esmpy",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--override",
        action="store_true",
        help="Override the existing mapping file if it exists",
        default=False,
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run in parallel using MPI. Must also be run with mpirun.",
        default=False,
    )

    args = parser.parse_args()
    main(args)
