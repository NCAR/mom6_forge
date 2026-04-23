import matplotlib.pyplot as plt
from mom6_forge.utils import get_mesh_dimensions

import numpy as np
import xarray as xr
from pathlib import Path


def plot_esmf_mesh(
    mesh_ds,
    field=None,
    cells_to_mark=None,
    figsize=None,
    xlim=None,
    ylim=None,
    index_axes=False,
    **kwargs
):
    """
    Plot the mesh using matplotlib.

    Parameters
    ----------
    mesh_ds : xarray.Dataset
        The ESMF mesh dataset containing the mesh information.
    field : np.ndarray, optional
        The field to plot on the mesh. If None, the mesh mask will be plotted.
    cells_to_mark : dict of {int: str}
        A dictionary where keys are (flattened) cell indices and values are colors to mark the cells.
    figsize : tuple, optional
        The size of the figure. Default is (10, 8).
    xlim : tuple, optional
        The x-axis limits. Default is None.
    ylim : tuple, optional
        The y-axis limits. Default is None.
    index_axes : bool, optional
        If True, the mesh will be indexed from 0 to nx-1 and 0 to ny-1. If False,
        geographical coordinates will be used as axes. Default is False.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted mesh.
    """

    assert isinstance(
        mesh_ds, (xr.Dataset, str, Path)
    ), "mesh_ds must be an xarray Dataset or a path to a mesh file"

    if figsize is None:
        figsize = (10, 8)
    assert (
        isinstance(figsize, (tuple, list)) and len(figsize) == 2
    ), "figsize must be a tuple or list of length 2"

    fig, ax = plt.subplots(figsize=figsize)

    nx, ny = get_mesh_dimensions(mesh_ds)

    if index_axes:
        lon_2d, lat_2d = np.meshgrid(np.arange(nx), np.arange(ny))
    else:
        lon_2d = mesh_ds["centerCoords"][:, 0].values.reshape((ny, nx))
        lat_2d = mesh_ds["centerCoords"][:, 1].values.reshape((ny, nx))

    mask_2d = mesh_ds["elementMask"].values.reshape((ny, nx))

    if field is not None:
        assert isinstance(
            field, (np.ndarray, xr.DataArray)
        ), "Field must be a numpy array or xarray DataArray"
        assert (
            field.shape == (ny, nx) or len(field) == nx * ny
        ), "Field must be 2D or flattened 1D array that matches the mesh dimensions"
        field_2d = field.reshape((ny, nx)) if len(field) == nx * ny else field
        # apply mask:
        field_2d = np.ma.masked_where(mask_2d == 0, field_2d)
    else:
        field_2d = mask_2d

    ax.pcolormesh(lon_2d, lat_2d, field_2d, **kwargs)

    # Highlight cells
    if cells_to_mark:

        # determine the size of the cells based on the first cell to mark:
        j0, i0 = divmod(list(cells_to_mark.keys())[0], nx)
        dx = (
            lon_2d[j0, i0] - lon_2d[j0, i0 - 1]
            if i0 > 0
            else lon_2d[j0, i0 + 1] - lon_2d[j0, i0]
        )
        dy = (
            lat_2d[j0, i0] - lat_2d[j0 - 1, i0]
            if j0 > 0
            else lat_2d[j0 + 1, i0] - lat_2d[j0, i0]
        )

        # Mark the cells
        for cell_index, color in cells_to_mark.items():
            j, i = divmod(cell_index, nx)
            if 0 <= i < nx and 0 <= j < ny:
                ax.add_patch(
                    plt.Rectangle(
                        (lon_2d[j, i] - dx / 2, lat_2d[j, i] - dy / 2),
                        dx,
                        dy,
                        color=color,
                        alpha=0.2,
                    )
                )

    # Set xlim and ylim
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # add colorbar
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Field Value")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("ESMF Mesh Plot")

    return ax
