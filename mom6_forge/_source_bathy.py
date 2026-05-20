"""Source bathymetry loader for mom6_forge.

``SourceBathy`` is a lightweight data container for a regional slice of a
source bathymetry dataset.  Users who want to call pipeline methods directly
should construct a ``SourceBathy`` explicitly::

    from mom6_forge._source_bathy import SourceBathy
    src = SourceBathy(
        topo,
        "gebco_2023.nc",
        lon_name="lon",
        lat_name="lat",
        depth_name="elevation",
    )
"""

import numpy as np
import xarray as xr
from pathlib import Path


class SourceBathy:
    """Regional slice of a source bathymetry dataset (e.g. GEBCO).

    Holds the loaded, domain-clipped, ESMF-prepped bathymetry ``DataArray``
    together with its coordinate-name metadata.

    Parameters
    ----------
    topo : Topo
        Target grid object.  Only ``topo._grid.qlon`` and
        ``topo._grid.qlat`` are used to determine the clipping extent.
    path : str or Path
        Path to the source bathymetry NetCDF file.
    lon_name : str, optional
        Longitude coordinate name in the source file.  Default ``"lon"``.
    lat_name : str, optional
        Latitude coordinate name in the source file.  Default ``"lat"``.
    depth_name : str, optional
        Depth variable name in the source file.  Default ``"depth"``.
    is_input_positive_below_msl : bool, optional
        Whether depth values in the source file are positive below mean sea
        level.  If ``False`` the sign is flipped on load.  Default ``True``.
    buf : float, optional
        Degree buffer added around the Q-grid bounding box when clipping the
        source dataset.  Default ``0.5``.
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
        """Rename source-file dimensions and set required depth attributes.

        Renames ``lon_name``, ``lat_name``, and ``depth_name`` to the
        canonical names ``"lon"``, ``"lat"``, and ``"depth"`` so that
        subsequent methods can rely on consistent coordinate names.  Also
        sets the ``missing_value``, ``_FillValue``, ``units``, and
        ``coordinates`` attributes expected by FRE tools (some by ESMF as well), and assigns
        ``units`` to the longitude/latitude coordinates if absent.

        Sets ``self.lon_name``, ``self.lat_name``, and ``self.depth_name``
        to the renamed canonical strings.  Mutates ``self._ds`` in place.

        Parameters
        ----------
        lon_name : str
            Original longitude coordinate name in ``self._ds``.
        lat_name : str
            Original latitude coordinate name in ``self._ds``.
        depth_name : str
            Original depth variable name in ``self._ds``.
        """
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
        self._ds.depth.attrs["coordinates"] = "lon lat"
        if "units" not in self._ds[self.lon_name].attrs:
            self._ds[self.lon_name].attrs["units"] = "degrees"
        if "units" not in self._ds[self.lat_name].attrs:
            self._ds[self.lat_name].attrs["units"] = "degrees_north"

    def _slice_to_domain(self, topo, buf=0.5):
        """Clip ``self._ds`` to the topo grid extent plus ``buf`` degrees.

        Handles the global-longitude seam and regional domains automatically.
        Stores ``topo`` as ``self.topo`` and mutates ``self._ds`` in place.

        Parameters
        ----------
        topo : Topo
            Target grid object; only ``topo._grid.qlon`` and
            ``topo._grid.qlat`` are used.
        buf : float, optional
            Degree buffer added around the Q-grid bounding box.
            Default ``0.5``.

        Returns
        -------
        xarray.Dataset
            The clipped dataset (also stored as ``self._ds``).
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

        self._ds = longitude_slicer(
            self._ds,
            np.array(lon_extent) + np.array([-buf, buf]),
            self.lon_name,
        )

        return self._ds

    def _ensure_depth_is_positive_below_msl(self, is_input_positive_below_msl):
        """Flip depth sign if the source convention is positive *above* MSL.

        Mutates ``self._ds`` in place.  After this call the ``"depth"``
        variable is always positive below mean sea level (ocean > 0).

        Parameters
        ----------
        is_input_positive_below_msl : bool
            If ``True`` the source data are already positive below MSL and no
            change is made.  If ``False`` the sign of the depth variable is
            inverted.
        """
        if not is_input_positive_below_msl:
            self._ds[self.depth_name] = -self._ds[self.depth_name]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lon(self):
        """1-D longitude array of the clipped source dataset.

        Returns
        -------
        numpy.ndarray
            Longitude values in degrees.
        """
        return self.ds[self.lon_name].values

    @property
    def lat(self):
        """1-D latitude array of the clipped source dataset.

        Returns
        -------
        numpy.ndarray
            Latitude values in degrees.
        """
        return self.ds[self.lat_name].values

    @property
    def depth(self):
        """2-D depth array, positive below mean sea level (ocean > 0).

        Returns
        -------
        numpy.ndarray
            Depth values in metres, shape ``(lat, lon)``.
        """
        return self.ds[self.depth_name].values

    @property
    def ds(self):
        """Clipped source dataset with canonical coordinate names.

        Depth is positive below mean sea level.

        Returns
        -------
        xarray.Dataset
        """
        return self._ds

    def __repr__(self):
        return (
            f"SourceBathy({self.path.name!r}, lon={self.lon_name!r}, "
            f"lat={self.lat_name!r}, depth={self.depth_name!r}, shape={self.depth.shape})"
        )


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
            np.diff(lons), dlons * np.ones(np.size(lons) - 1)
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
                num_lonpoints = int(
                    int(data[lon].shape[0] * (central_longitude - longitude_extent[0]))
                    // lon_span
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
