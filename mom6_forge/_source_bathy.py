"""Source bathymetry loader for mom6_forge.

``SourceBathy`` is a lightweight data container for a regional slice of a
source bathymetry dataset.  Users who call pipeline
methods directly
should construct a ``SourceBathy`` explicitly::

    from mom6_forge._source_bathy import SourceBathy
    src = SourceBathy("gebco_2023.nc").slice_to_domain(topo)
"""

import numpy as np
import xarray as xr
from pathlib import Path
from mom6_forge.utils import longitude_slicer


class SourceBathy:
    """Regional slice of a source bathymetry dataset (e.g. GEBCO).

    Holds the loaded, domain-clipped, ESMF prepped elevation DataArray together with its
    coordinate-name metadata.  Per-cell depth statistics are computed and
    cached here so repeated calls with the same source file skip the
    expensive sub-sampling step.

    Parameters
    ----------
    path : str or Path
    lon_name : str   — longitude coordinate name. Default ``"lon"``.
    lat_name : str   — latitude coordinate name. Default ``"lat"``.
    depth_name : str — depth variable. Default ``"depth"``.
    """

    def __init__(
        self,
        topo,
        path,
        lon_name="lon",
        lat_name="lat",
        depth_name="depth",
        is_input_positive_below_msl=True,
        buf=0.5,
    ):
        self.path = Path(path)
        self._ds = xr.open_dataset(self.path, chunks="auto")
        self._rename_dims_and_format_ds(
            lon_name=lon_name, lat_name=lat_name, depth_name=depth_name
        )  # ensure consistent coordinate names for slicing
        self._slice_to_domain(topo, buf=buf)
        self._ensure_depth_is_positive_below_msl(is_input_positive_below_msl)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _rename_dims_and_format_ds(self, lon_name, lat_name, depth_name):
        """Rename dimensions in the source dataset to match the provided names. This helps prep the dataset for ESMF regridding, which expects specific coordinate names."""

        self._ds = self._ds.rename(
            {
                lon_name: "lon",
                lat_name: "lat",
                depth_name: "depth",
            }
        )
        self.lon_name = "lon"
        self.lat_name = "lat"
        self.depth_name = "depth"
        self._ds.depth.attrs["missing_value"] = (
            -1e20
        )  # missing value expected by FRE tools
        self._ds.depth.attrs["_FillValue"] = -1e20
        self._ds.depth.attrs["units"] = "meters"
        self._ds.depth.attrs["standard_name"] = "height_above_reference_ellipsoid"
        self._ds.depth.attrs["long_name"] = "Elevation relative to sea level"
        self._ds.depth.attrs["coordinates"] = "lon lat"
        if "units" not in self._ds[self.lon_name].attrs:
            self._ds[self.lon_name].attrs["units"] = "degrees_east"
        if "units" not in self._ds[self.lat_name].attrs:
            self._ds[self.lat_name].attrs["units"] = "degrees_north"

    def _slice_to_domain(self, topo, buf=0.5):
        """Load and clip elevation to the topo grid extent plus ``buf`` degrees.

        Handles the global-longitude seam automatically.  Mutates ``self``
        in place and returns ``self`` for chaining.

        Parameters
        ----------
        topo : Topo — only ``topo._grid.qlon`` / ``topo._grid.qlat`` are used.
        buf : float — degree buffer around the Q-grid bounding box. Default 0.5.
        """
        self.topo = topo
        lon_extent = (float(topo._grid.qlon.min()), float(topo._grid.qlon.max()))
        lat_extent = (float(topo._grid.qlat.min()), float(topo._grid.qlat.max()))
        print(
            f"Slicing source bathymetry to domain: {lon_extent} x {lat_extent} with buffer {buf}"
        )

        self._ds = self._ds.sel(
            {self.lat_name: slice(lat_extent[0] - buf, lat_extent[1] + buf)}
        )

        dlon = float(self._ds[self.lon_name][1] - self._ds[self.lon_name][0])
        self._ds = longitude_slicer(
            self._ds,
            np.array(lon_extent) + np.array([-buf, buf]),
            self.lon_name,
        )

        return self._ds

    def _ensure_depth_is_positive_below_msl(self, depth_positive):
        """Ensure depth is positive-down. Mutates self in place."""
        if not depth_positive:
            self._ds[self.depth_name] = -self._ds[self.depth_name]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lon(self):
        """1-D longitude array."""
        return self.ds[self.lon_name].values

    @property
    def lat(self):
        """1-D latitude array."""
        return self.ds[self.lat_name].values

    @property
    def depth(self):
        """2-D depth array, depth positive (ocean > 0)"""
        return self.ds[self.depth_name].values

    @property
    def ds(self):
        """Raw dataset with source coordinate names (positive-down)."""
        return self._ds

    def __repr__(self):
        shape = self._da.shape if self._da is not None else "not loaded"
        return (
            f"SourceBathy({self.path.name!r}, lon={self.lon_name!r}, "
            f"lat={self.lat_name!r}, elevation={self.elevation_name!r}, shape={shape})"
        )
