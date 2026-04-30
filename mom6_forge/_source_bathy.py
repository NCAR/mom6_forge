"""Source bathymetry loader for mom6_forge.

``SourceBathy`` is a lightweight data container for a regional slice of a
source bathymetry dataset.  ``Topo._get_src()`` creates and caches one
automatically when ``set_from_dataset`` is called.  Users who call pipeline
methods directly (e.g. ``high_res_regrid``, ``generate_mask_ocean_frac``)
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

    Holds the loaded, domain-clipped elevation DataArray together with its
    coordinate-name metadata.  Per-cell depth statistics are computed and
    cached here so repeated calls with the same source file skip the
    expensive sub-sampling step.

    Parameters
    ----------
    path : str or Path
    lon_name : str   — longitude coordinate name. Default ``"lon"``.
    lat_name : str   — latitude coordinate name. Default ``"lat"``.
    elevation_name : str — elevation variable (positive-up). Default ``"elevation"``.
    """

    def __init__(
        self,
        path,
        lon_name="lon",
        lat_name="lat",
        elevation_name="elevation",
        positive_down=False,
    ):
        self.path = Path(path)
        self.lon_name = lon_name
        self.lat_name = lat_name
        self.elevation_name = elevation_name
        self.positive_down = positive_down  # depth should be positive down (ocean > 0) if True, otherwise positive up (ocean < 0)
        self._da = None  # set by slice_to_domain
        self._ds = None  # set by slice_to_domain
        self._topo_stats = None  # set by compute_topo_stats

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def slice_to_domain(self, topo, buf=0.5):
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

        ds_src = xr.open_dataset(self.path, chunks="auto")

        ds = ds_src.sel(
            {self.lat_name: slice(lat_extent[0] - buf, lat_extent[1] + buf)}
        )

        dlon = float(ds[self.lon_name][1] - ds[self.lon_name][0])
        total_lon = float(ds[self.lon_name][-1] - ds[self.lon_name][0] + dlon)
        if np.isclose(total_lon, 360):
            ds = longitude_slicer(
                ds,
                np.array(lon_extent) + np.array([-buf, buf]),
                self.lon_name,
            )
        else:
            ds = ds.sel(
                {self.lon_name: slice(lon_extent[0] - buf, lon_extent[1] + buf)}
            )
        self._ds = ds
        self._da = ds[self.elevation_name].load()
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lon(self):
        """1-D longitude array."""
        return self._da[self.lon_name].values

    @property
    def lat(self):
        """1-D latitude array."""
        return self._da[self.lat_name].values

    @property
    def depth(self):
        """2-D depth array, positive-down (ocean > 0), shape (ny_src, nx_src)."""
        if self.positive_down:
            return self._da.values.astype(float)
        else:
            return -self._da.values.astype(float)

    @property
    def da(self):
        """Raw elevation DataArray with source coordinate names (positive-up)."""
        if not self.positive_down:
            return -self._da
        else:
            return self._da

    @property
    def ds(self):
        """Raw dataset with source coordinate names (positive-up)."""
        if not self.positive_down:
            ds = self._ds.copy()
            ds[self.elevation_name] = -ds[self.elevation_name]
            return ds
        else:
            return self._ds

    def __repr__(self):
        shape = self._da.shape if self._da is not None else "not loaded"
        return (
            f"SourceBathy({self.path.name!r}, lon={self.lon_name!r}, "
            f"lat={self.lat_name!r}, elevation={self.elevation_name!r}, shape={shape})"
        )
