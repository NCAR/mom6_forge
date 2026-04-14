import os
import numpy as np
import xarray as xr
import shapely
import cartopy.io.shapereader as shpreader
from datetime import datetime
from scipy import interpolate
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import unary_union
from mom6_forge.utils import cell_area_rad, longitude_slicer
from mom6_forge.grid import Grid
from mom6_forge.git_utils import get_domain_dir, get_repo
from pathlib import Path
from mom6_forge.edit_command import *
from mom6_forge.command_manager import TopoCommandManager, CommandType
from mom6_forge.mapping import regrid_dataset_via_xesmf, cressman_regrid
from mom6_forge._source_bathy import SourceBathy


class Topo:
    """
    Bathymetry Generator for MOM6 grids (mom6_forge.grid.Grid).
    """

    def __init__(self, grid, min_depth, version_control_dir="TopoLibrary"):
        """
        MOM6 Simpler Models bathymetry constructor.

        Parameters
        ----------
        grid: mom6_forge.grid.Grid
            horizontal grid instance for which the bathymetry is to be created.
        min_depth: float
            Minimum water column depth. Columns with shallow depths are to be masked out.
        """

        self._grid = grid
        self._depth = xr.DataArray(
            np.full((grid.ny, grid.nx), np.nan, dtype=float),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )  # Initialize depth with NaNs
        self._min_depth = min_depth
        self._src = None  # cached SourceBathy; set by _get_src()

        if version_control_dir is None:
            raise ValueError(
                "version_control_dir cannot be None. Version control is required for Topo objects. Old Topo Files can be added through from_topo_file() or from_topo_version_control() classmethods."
            )

        self.version_control = True

        # Create a folder to store bathymetry objects in
        self.topos_root = Path(version_control_dir).mkdir(exist_ok=True)

        # Create the subfolder for this specific bathymetry
        self.domain_dir = Path(get_domain_dir(grid, base_dir=version_control_dir))
        self.domain_dir.mkdir(exist_ok=True)  # This folder should not already exist.

        # Save the grid info there (there can only be 1 grid per bathymetry)
        self.grid_file_path = self.domain_dir / "grid.nc"
        grid.write_supergrid(self.grid_file_path)

        initial_command = MinDepthEditCommand(
            self, attr="min_depth", new_value=min_depth
        )

        # Initialize the git repo
        self.repo = get_repo(self.domain_dir)

        # Set up TCM (requires that self.domain_dir exists)
        self.tcm = TopoCommandManager(self, command_registry=COMMAND_REGISTRY)
        self.tcm.execute(initial_command, cmd_type=CommandType.COMMAND)

    @classmethod
    def from_version_control(cls, folder_path: str | Path):
        """
        Create a bathymetry object from an existing version-controlled bathymetry folder.

        Parameters
        ----------
        folder_path: str | Path
            Path to an existing bathymetry folder created by mom6_forge with version control enabled.
        """

        folder_path = Path(folder_path)
        assert folder_path.exists(), f"Cannot find bathymetry folder at {folder_path}."

        grid_file_path = folder_path / "grid.nc"
        assert grid_file_path.exists(), f"Cannot find grid file at {grid_file_path}."

        grid = Grid.from_supergrid(grid_file_path)

        # Create the topo object
        topo = Topo(
            grid, 0.0, version_control_dir=folder_path.parent
        )  # Because we hash the grid, the correct domain will be selected

        # Reapply any changes
        topo.tcm.reapply_changes()
        topo.tcm.undo()  # Undo the initialization min_depth set to 0.0. (How it works is the changes are ordered from the previous state to the new state, so undoing the initial set to 0.0 leaves the correct min_depth)

        return topo

    @classmethod
    def from_topo_file(
        cls,
        grid,
        topo_file_path,
        min_depth=0.0,
        varname="depth",
        version_control_dir="TopoLibrary",
    ):
        """
        Create a bathymetry object from an existing topog file.

        Parameters
        ----------
        grid: mom6_forge.grid.Grid
            horizontal grid instance for which the bathymetry is to be created.
        topo_file_path: str
            Path to an existing MOM6 topog file.
        min_depth: float, optional
            Minimum water column depth (m). Columns with shallower depths are to be masked out.
        varname : str, optional
            Name of the variable representing ocean depth in the dataset. Default is "depth".
        """

        topo = cls(grid, min_depth, version_control_dir=version_control_dir)
        topo.tcm.reapply_changes()
        topo.set_depth_via_topog_file(topo_file_path, varname)
        return topo

    @property
    def depth(self):
        """
        MOM6 grid depth array (m). Positive below MSL.
        """
        return self._depth

    @depth.setter
    def depth(self, depth):
        """
        Apply a custom bathymetry via a user-defined depth array.

        Parameters
        ----------
        depth: np.array
            2-D Array of ocean depth (m).
        """

        if np.isscalar(depth):
            self.set_flat(depth)
            return

        assert depth.shape == (
            self._grid.ny,
            self._grid.nx,
        ), "Incompatible depth array shape"

        if isinstance(depth, xr.DataArray):
            depth = depth.data
        else:
            assert isinstance(
                depth, np.ndarray
            ), "depth must be a numpy array or xarray DataArray"

        self._depth = xr.DataArray(
            depth,
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )

    @property
    def min_depth(self):
        """
        Minimum water column depth. Columns with shallow depths are to be masked out.
        """
        return self._min_depth

    @property
    def max_depth(self):
        """
        Maximum water column depth.
        """
        return self.depth.max().item()

    @min_depth.setter
    def min_depth(self, new_min_depth):
        self._min_depth = new_min_depth

    @property
    def tmask(self):
        """
        Ocean domain mask at T grid. 1 if ocean, 0 if land.
        """
        tmask_da = xr.DataArray(
            np.where(self._depth > self._min_depth, 1, 0),
            dims=["ny", "nx"],
            attrs={"name": "T mask"},
        )
        return tmask_da

    @property
    def umask(self):
        """
        Ocean domain mask on U grid. 1 if ocean, 0 if land.
        """
        tmask = self.tmask

        # Create empty mask DataArray for umask
        umask = xr.DataArray(
            np.ones(self._grid.ulat.shape, dtype=int),
            dims=["yh", "xq"],
            attrs={"name": "U mask"},
        )

        # Fill umask with mask values
        umask[:, :-1] &= tmask.values  # h-point translates to the left u-point
        umask[:, 1:] &= tmask.values  # h-point translates to the right u-point

        return umask

    @property
    def vmask(self):
        """
        Ocean domain mask on V grid. 1 if ocean, 0 if land.
        """
        tmask = self.tmask

        # Create empty mask DataArray for umask
        vmask = xr.DataArray(
            np.ones(self._grid.vlat.shape, dtype=int),
            dims=["yq", "xh"],
            attrs={"name": "V mask"},
        )

        # Fill vmask with mask values
        vmask[:-1, :] &= tmask.values  # h-point translates to the bottom v-point
        vmask[1:, :] &= tmask.values  # h-point translates to the top v-point

        return vmask

    @property
    def qmask(self):
        """
        Ocean domain mask on Q grid. 1 if ocean, 0 if land.
        """
        tmask = self.tmask

        # Create empty mask DataArray for umask
        qmask = xr.DataArray(
            np.ones(self._grid.qlat.shape, dtype=int),
            dims=["yq", "xq"],
            attrs={"name": "Q mask"},
        )

        # Fill qmask with mask values
        qmask[:-1, :-1] &= tmask.values  # top-left of h goes to top-left q
        qmask[:-1, 1:] &= tmask.values  # top-right
        qmask[1:, :-1] &= tmask.values  # bottom-left
        qmask[1:, 1:] &= tmask.values  # bottom-right

        # Corners of the qmask are always land -> regional cases
        qmask[0, 0] = 0
        qmask[0, -1] = 0
        qmask[-1, 0] = 0
        qmask[-1, -1] = 0

        return qmask

    @property
    def basintmask(self):
        """
        Ocean domain mask at T grid. Seperate number for each connected water cell, 0 if land.
        """
        res, num_features = label(self.tmask)

        return xr.DataArray(res)

    @property
    def supergridmask(self):
        """
        Ocean domain mask on supergrid. 1 if ocean, 0 if land.
        """

        supergridmask = xr.DataArray(
            np.zeros(self._grid._supergrid.x.shape, dtype=int),
            dims=["nyp", "nxp"],
            attrs={"name": "supergrid mask"},
        )
        supergridmask[::2, ::2] = self.qmask.values
        supergridmask[::2, 1::2] = self.vmask.values
        supergridmask[1::2, ::2] = self.umask.values
        supergridmask[1::2, 1::2] = self.tmask.values
        return supergridmask

    def point_is_ocean(self, lons, lats):
        """
        Given a list of coordinates, return a list of booleans indicating if the coordinates are in the ocean (True) or land (False)
        """
        assert len(lons) == len(
            lats
        ), "Lons & Lats must be the same length, they describe a set of points"

        is_ocean = []
        for i in range(len(lons)):
            match = np.where(
                (self._grid._supergrid.x == lons[i])
                & (self._grid._supergrid.y == lats[i])
            )
            is_ocean.append(self.supergridmask[match[0], match[1]].item())
        return is_ocean

    def send_entire_depth_change_to_tcm(self, depth, quietly=False):
        """
        This function takes an entire depth change and adds it through the TopoCommandManager (TCM) or directly if quietly is enabled.
        """
        # 1. Generate all affected indices (row-major order)
        all_indices = list(np.ndindex(self.depth.shape))  # list of (j, i) tuples

        # 2. Flatten the new values to match the indices
        new_values = depth.values.ravel().tolist()

        # 3. Flatten old values if depth exists
        old_values = (
            self.depth.values.ravel().tolist() if self.depth is not None else None
        )

        # 4. Build command
        depth_edit_command = DepthEditCommand(
            self, all_indices, new_values, old_values=old_values
        )

        if not quietly:
            self.tcm.execute(depth_edit_command, cmd_type=CommandType.COMMAND)
        else:
            depth_edit_command()

    def set_flat(self, D):
        """
        Create a flat bottom bathymetry with a given depth D.

        Parameters
        ----------
        D: float
            Bathymetric depth of the flat bottom to be generated.
        """

        depth = xr.DataArray(
            np.full((self._grid.ny, self._grid.nx), D),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )

        # Save to object
        self.send_entire_depth_change_to_tcm(depth)

    def set_depth_via_topog_file(self, topog_file_path, varname="depth", quietly=False):
        """
        Apply a bathymetry read from an existing topog file

        Parameters
        ----------
        topog_file_path: str
            absolute path to an existing MOM6 topog file
        varname : str
            Name of the variable representing ocean depth in the dataset.
        """

        assert os.path.exists(
            topog_file_path
        ), f"Cannot find topog file at {topog_file_path}."

        ds_topo = xr.open_dataset(topog_file_path)
        assert (
            varname in ds_topo
        ), f"Cannot find the '{varname}' field in topog file {topog_file_path}"
        depth = ds_topo[varname]

        if depth.shape[0] < self._grid.ny or depth.shape[1] < self._grid.nx:
            raise ValueError(
                f"Topography data in {topog_file_path} is smaller than the grid size "
                f"({depth.shape[0]}x{depth.shape[1]} < {self._grid.ny}x{self._grid.nx}). "
            )
        elif depth.shape[0] > self._grid.ny or depth.shape[1] > self._grid.nx:
            assert (
                "geolat" in ds_topo and "geolon" in ds_topo
            ), f"Topog file {topog_file_path} does not contain geolat and geolon fields, "
            "which are required to determine if the grid is a subgrid of the topog file, "
            "since the topography data is larger than the grid (in index space). "

            # Determine if the grid is a subgrid of the topog file
            geolat = ds_topo["geolat"]
            geolon = ds_topo["geolon"]

            # find the closest cell in the topog file to the (sub)grid's origin (southwest corner)
            topog_kdtree = cKDTree(
                np.column_stack((geolat.data.flatten(), geolon.data.flatten()))
            )
            _, indices = topog_kdtree.query(
                [self._grid.tlat[0, 0].item(), self._grid.tlon[0, 0].item()]
            )
            cj, ci = np.unravel_index(indices, geolon.shape)

            assert 0 <= cj <= geolat.shape[0] - self._grid.ny, (
                f"Topography data in {topog_file_path} appears to only contain a subregion "
                f"of the grid, and does not contain enough rows to accommodate the grid size "
                f"({self._grid.ny}). "
            )
            assert 0 <= ci <= geolon.shape[1] - self._grid.nx, (
                f"Topography data in {topog_file_path} appears to only contain a subregion "
                f"of the grid, and does not contain enough columns to accommodate the grid size "
                f"({self._grid.nx}). "
            )

            # Compare the coords of grid with the coords of the subregion of the topog
            # data where it may overlap with the grid
            grid_overlaps_topo = np.all(
                np.isclose(
                    geolat[cj : cj + self._grid.ny, ci : ci + self._grid.nx],
                    self._grid.tlat.data,
                    rtol=1e-5,
                )
            ) and np.all(
                np.isclose(
                    geolon[cj : cj + self._grid.ny, ci : ci + self._grid.nx],
                    self._grid.tlon.data,
                    rtol=1e-5,
                )
            )
            if not grid_overlaps_topo:
                raise ValueError(
                    f"The topography data in {topog_file_path} is larger than the grid "
                    f"data which does not appear to be a subgrid of the topography data. "
                    f"Topography data shape: {depth.shape}, grid shape: "
                    f"({self._grid.ny}, {self._grid.nx}). "
                )

            # If the grid is a subgrid of the topog data, extract the subregion
            depth = depth[cj : cj + self._grid.ny, ci : ci + self._grid.nx]

        else:
            pass  # the depth array is the right size

        # Set all NaNs to land
        depth = depth.fillna(0)

        # Save to object (Build TCM Object)
        self.send_entire_depth_change_to_tcm(depth, quietly=quietly)

    def set_spoon(self, max_depth, dedge, rad_earth=6.378e6, expdecay=400000.0):
        """
        Create a spoon-shaped bathymetry. Same effect as setting the TOPO_CONFIG
        parameter to "spoon".

        Parameters
        ----------
        max_depth : float
            Maximum depth of model in the units of D.
        dedge : float
            The depth [Z ~> m], at the basin edge
        rad_earth : float, optional
            Radius of earth
        expdecay : float, optional
            A decay scale of associated with the sloping boundaries [m]
        """

        west_lon = self._grid.tlon[0, 0]
        south_lat = self._grid.tlat[0, 0]
        nx = self._grid.nx
        ny = self._grid.ny
        leny = self._grid.supergrid.leny

        D0 = (max_depth - dedge) / (
            (1.0 - np.exp(-0.5 * leny * rad_earth * np.pi / (180.0 * expdecay)))
            * (1.0 - np.exp(-0.5 * leny * rad_earth * np.pi / (180.0 * expdecay)))
        )

        new_values = dedge + D0 * (
            np.sin(
                np.pi * (self._grid.tlon[:, :] - west_lon) / self._grid.supergrid.lenx
            )
            * (
                1.0
                - np.exp(
                    (self._grid.tlat[:, :] - (south_lat + leny))
                    * rad_earth
                    * np.pi
                    / (180.0 * expdecay)
                )
            )
        )

        # Save to object (Build TCM Object)
        self.send_entire_depth_change_to_tcm(new_values)

    def set_bowl(self, max_depth, dedge, rad_earth=6.378e6, expdecay=400000.0):
        """
        Create a bowl-shaped bathymetry. Same effect as setting the TOPO_CONFIG parameter to "bowl".

        Parameters
        ----------
        max_depth : float
            Maximum depth of model in the units of D.
        dedge : float
            The depth [Z ~> m], at the basin edge
        rad_earth : float, optional
            Radius of earth
        expdecay : float, optional
            A decay scale of associated with the sloping boundaries [m]
        """

        west_lon = self._grid.tlon[0, 0]
        south_lat = self._grid.tlat[0, 0]
        len_lon = self._grid.supergrid.lenx
        len_lat = self._grid.supergrid.leny

        D0 = (max_depth - dedge) / (
            (1.0 - np.exp(-0.5 * len_lat * rad_earth * np.pi / (180.0 * expdecay)))
            * (1.0 - np.exp(-0.5 * len_lat * rad_earth * np.pi / (180.0 * expdecay)))
        )

        new_values = dedge + D0 * (
            np.sin(np.pi * (self._grid.tlon[:, :] - west_lon) / len_lon)
            * (
                (
                    1.0
                    - np.exp(
                        -(self._grid.tlat[:, :] - south_lat)
                        * rad_earth
                        * np.pi
                        / (180.0 * expdecay)
                    )
                )
                * (
                    1.0
                    - np.exp(
                        (self._grid.tlat[:, :] - (south_lat + len_lat))
                        * rad_earth
                        * np.pi
                        / (180.0 * expdecay)
                    )
                )
            )
        )

        # Save to object (Build TCM Object)
        self.send_entire_depth_change_to_tcm(new_values)

    def _get_src(
        self,
        bathymetry_path,
        longitude_coordinate_name,
        latitude_coordinate_name,
        vertical_coordinate_name,
    ):
        """Return a cached :class:`SourceBathy`, creating and slicing a new one
        only when the path or coordinate names differ from the current cache."""
        path = Path(bathymetry_path)
        if (
            self._src is None
            or self._src.path != path
            or self._src.lon_name != longitude_coordinate_name
            or self._src.lat_name != latitude_coordinate_name
            or self._src.elevation_name != vertical_coordinate_name
        ):
            self._src = SourceBathy(
                path,
                longitude_coordinate_name,
                latitude_coordinate_name,
                vertical_coordinate_name,
            ).slice_to_domain(self)
        return self._src

    def diagnose_resolution(self, src):
        """
        Print resolution diagnostics comparing the model grid to a source bathymetry
        dataset, and recommend whether Cressman interpolation / stats-based masking
        is worthwhile.

        The recommendation threshold is a resolution ratio of 12x (model dx /
        dataset dx), equivalent to ~0.05° (~5 km) for GEBCO 15-arcsecond source
        data. This matches the criterion used by the tx2_3 global workflow
        (interp_smooth.f90) and is the scale at which ocean-aware Cressman
        interpolation meaningfully improves coastal depth estimates over standard
        xesmf conservative regridding.

        Parameters
        ----------
        src : SourceBathy
            Source bathymetry object.  The DataArray need not be loaded —
            coordinate arrays are read from the file if ``src`` has not yet
            been sliced to the domain.

        Returns
        -------
        bool
            True if Cressman / stats-based masking is recommended (ratio >= 12x),
            False otherwise.
        """
        CRESSMAN_THRESHOLD = 12.0

        # --- Model T-cell spacing in meters ---
        # sqrt(tarea) gives the geometric mean cell spacing (equiv. to sqrt(dxt * dyt))
        cell_dx_m = np.sqrt(self._grid.tarea.values)

        median_dx_m = float(np.median(cell_dx_m))
        min_dx_m = float(np.min(cell_dx_m))
        max_dx_m = float(np.max(cell_dx_m))

        # --- Source dataset spacing ---
        if src._da is not None:
            lon = src.lon
            lat = src.lat
        else:
            ds = xr.open_dataset(src.path)
            lon = ds[src.lon_name].values
            lat = ds[src.lat_name].values
            ds.close()

        dlon_deg = float(abs(lon[1] - lon[0]))
        dlat_deg = float(abs(lat[1] - lat[0]))

        mid_lat_deg = float(self._grid.tlat.mean())
        R = 6371000.0
        dataset_dx_m = dlon_deg * (np.pi / 180) * R * np.cos(mid_lat_deg * np.pi / 180)
        dataset_dy_m = dlat_deg * (np.pi / 180) * R

        ratio_median = median_dx_m / dataset_dx_m
        ratio_max = max_dx_m / dataset_dx_m

        # --- Print ---
        sep = "=" * 58
        print(sep)
        print("  Resolution Diagnostics")
        print(sep)
        print(f"\n  Source dataset ({src.path.name}):")
        print(f"    dlon = {dlon_deg * 3600:.1f} arcsec  ({dlon_deg:.6f}°)")
        print(f"    dlat = {dlat_deg * 3600:.1f} arcsec  ({dlat_deg:.6f}°)")
        print(
            f"    dx   ~ {dataset_dx_m:.0f} m  (at domain mid-lat {mid_lat_deg:.1f}°)"
        )
        print(f"    dy   ~ {dataset_dy_m:.0f} m")
        print(f"\n  Model grid (T-cell spacing):")
        print(f"    median = {median_dx_m / 1000:.2f} km")
        print(f"    min    = {min_dx_m / 1000:.2f} km")
        print(f"    max    = {max_dx_m / 1000:.2f} km")
        print(f"\n  Resolution ratio (model dx / dataset dx):")
        print(f"    median = {ratio_median:.1f}x")
        print(f"    max    = {ratio_max:.1f}x")
        print(f"\n  Cressman / stats-mask threshold: {CRESSMAN_THRESHOLD:.0f}x")
        if ratio_median >= CRESSMAN_THRESHOLD:
            print(f"  → RECOMMENDED: high_res_regrid()  (Cressman + stats mask)")
            print(
                f"    Each model cell spans ~{ratio_median:.0f} dataset pixels per side."
            )
            print(f"    Ocean-aware Cressman interpolation will meaningfully reduce")
            print(f"    land contamination of coastal depth estimates.")
        else:
            print(f"  → RECOMMENDED: direct_xesmf_regrid()  (bilinear / conservative)")
            print(
                f"    Ratio {ratio_median:.1f}x is below the threshold where Cressman"
            )
            print(f"    provides significant benefit over xesmf regridding.")
        print(sep)
        return ratio_median >= CRESSMAN_THRESHOLD

    def _compute_topo_stats(self, src, nx_sub, ny_sub, mask_hmin):
        """Compute per-cell depth statistics by Monte-Carlo sub-sampling.

        Results are cached on ``src._topo_stats`` so a second call with the
        same source file returns immediately without recomputation.

        Parameters
        ----------
        src : SourceBathy
        nx_sub, ny_sub : int
        mask_hmin : float

        Returns
        -------
        xr.Dataset  —  ``OCN_FRAC``, ``D_mean``, ``D_min``, ``D_max``, ``D2_mean``.
        """
        if src._topo_stats is not None:
            return src._topo_stats

        dlon = float(src.lon[1] - src.lon[0])
        dlat = float(src.lat[1] - src.lat[0])

        SW_lon = self._grid.qlon.values[:-1, :-1]
        SE_lon = self._grid.qlon.values[:-1, 1:]
        NE_lon = self._grid.qlon.values[1:, 1:]
        NW_lon = self._grid.qlon.values[1:, :-1]
        SW_lat = self._grid.qlat.values[:-1, :-1]
        SE_lat = self._grid.qlat.values[:-1, 1:]
        NE_lat = self._grid.qlat.values[1:, 1:]
        NW_lat = self._grid.qlat.values[1:, :-1]

        ifrac = (np.arange(1, nx_sub + 1) / (nx_sub + 1)).astype(float)
        jfrac = (np.arange(1, ny_sub + 1) / (ny_sub + 1)).astype(float)

        i_ = ifrac[np.newaxis, np.newaxis, np.newaxis, :]
        j_ = jfrac[np.newaxis, np.newaxis, :, np.newaxis]
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

        ii = np.round((sub_lon - src.lon[0]) / dlon).astype(int)
        jj = np.round((sub_lat - src.lat[0]) / dlat).astype(int)
        ii = np.clip(ii, 0, len(src.lon) - 1)
        jj = np.clip(jj, 0, len(src.lat) - 1)

        depth_sub = src.depth[jj, ii]  # positive-down

        is_ocean = depth_sub > mask_hmin
        ocn_frac = is_ocean.sum(axis=(-2, -1)) / (nx_sub * ny_sub)

        depth_ocean = np.where(is_ocean, depth_sub, np.nan)
        with np.errstate(all="ignore"):
            D_mean = np.nanmean(depth_ocean, axis=(-2, -1))
            D_min = np.nanmin(depth_ocean, axis=(-2, -1))
            D_max = np.nanmax(depth_ocean, axis=(-2, -1))
            D2_mean = np.nanmean(depth_ocean**2, axis=(-2, -1))

        dims = ["ny", "nx"]
        src._topo_stats = xr.Dataset(
            {
                "OCN_FRAC": xr.DataArray(
                    ocn_frac,
                    dims=dims,
                    attrs={
                        "long_name": "ocean fraction from sub-sampling",
                        "units": "1",
                    },
                ),
                "D_mean": xr.DataArray(
                    D_mean,
                    dims=dims,
                    attrs={"long_name": "mean ocean depth in cell", "units": "m"},
                ),
                "D_min": xr.DataArray(
                    D_min,
                    dims=dims,
                    attrs={"long_name": "minimum ocean depth in cell", "units": "m"},
                ),
                "D_max": xr.DataArray(
                    D_max,
                    dims=dims,
                    attrs={"long_name": "maximum ocean depth in cell", "units": "m"},
                ),
                "D2_mean": xr.DataArray(
                    D2_mean,
                    dims=dims,
                    attrs={
                        "long_name": "mean squared ocean depth in cell",
                        "units": "m2",
                    },
                ),
            }
        )
        return src._topo_stats

    def generate_mask_ocean_frac(
        self,
        src,
        nx_sub=5,
        ny_sub=5,
        mask_threshold=0.5,
        mask_hmin=0.0,
    ):
        """
        Generate an ocean mask by Monte-Carlo sub-sampling of the source
        bathymetry. Mirrors the algorithm in tx2_3's create_model_topo.f90.

        For each T-cell, distributes nx_sub x ny_sub interior points via
        bilinear interpolation of the Q-point corners and snaps each to the
        nearest source pixel. A cell is ocean if its ocean sub-point fraction
        (OCN_FRAC) meets or exceeds mask_threshold.

        Per-cell depth statistics (D_mean, D_min, D_max, D2_mean) are cached
        on the source bathymetry object for downstream use by ``write_topo()``.

        Parameters
        ----------
        src : SourceBathy
            Loaded (sliced) source bathymetry object.
        nx_sub, ny_sub : int
            Sub-sampling resolution per cell. Default 5x5.
        mask_threshold : float
            Minimum OCN_FRAC for a cell to be classified as ocean. Default 0.5.
        mask_hmin : float
            Minimum depth (m) for a sub-point to count as ocean. Default 0.0.

        Returns
        -------
        xr.DataArray
            Binary ocean mask on the T-grid (1 = ocean, 0 = land),
            dims ``["ny", "nx"]``.
        """
        self._src = src
        stats = self._compute_topo_stats(src, nx_sub, ny_sub, mask_hmin)

        ocean_mask = (stats["OCN_FRAC"].values >= mask_threshold).astype(int)

        return xr.DataArray(
            ocean_mask,
            dims=["ny", "nx"],
            attrs={
                "long_name": "ocean mask from sub-sampling",
                "mask_threshold": mask_threshold,
                "mask_hmin": mask_hmin,
                "nx_sub": nx_sub,
                "ny_sub": ny_sub,
            },
        )

    def generate_mask_cartopy(self, resolution="10m"):
        """
        Generate an ocean mask by rasterising Natural Earth land polygons
        onto the model T-grid using Cartopy.

        Faster than generate_mask_ocean_frac but coarser — does not account
        for sub-cell ocean fraction, only whether the T-cell centre falls on
        land or ocean. Useful as a quick first-pass mask or for comparison.

        Parameters
        ----------
        resolution : str
            Natural Earth resolution: ``'10m'``, ``'50m'``, or ``'110m'``.
            Default ``'10m'`` (finest, ~1:10M scale).

        Returns
        -------
        xr.DataArray
            Binary ocean mask on the T-grid (1 = ocean, 0 = land),
            dims ``["ny", "nx"]``.
        """
        # --- Load and clip land polygons to domain ---
        lon_min = float(self._grid.tlon.min())
        lon_max = float(self._grid.tlon.max())
        lat_min = float(self._grid.tlat.min())
        lat_max = float(self._grid.tlat.max())
        domain_box = box(lon_min - 1, lat_min - 1, lon_max + 1, lat_max + 1)

        land_shp = shpreader.natural_earth(
            resolution=resolution, category="physical", name="land"
        )
        reader = shpreader.Reader(land_shp)
        clipped = [g for g in reader.geometries() if g.intersects(domain_box)]
        land_union = unary_union(clipped)

        # --- Normalize longitudes to -180→180 to match Natural Earth ---
        tlon = self._grid.tlon.values.copy()
        tlon = np.where(tlon > 180, tlon - 360, tlon)
        tlat = self._grid.tlat.values

        # --- Vectorised point-in-polygon (Shapely 2.0) ---
        points = shapely.points(tlon.ravel(), tlat.ravel())
        is_ocean = ~shapely.contains(land_union, points)
        is_ocean = is_ocean.reshape(self._grid.ny, self._grid.nx).astype(int)

        return xr.DataArray(
            is_ocean,
            dims=["ny", "nx"],
            attrs={
                "long_name": "Cartopy ocean mask at T-points",
                "resolution": resolution,
            },
        )

    def cressman_interp(
        self,
        src,
        mask,
        smooth_scl=2.0,
        cressman_exp=2.0,
        hmin=None,
        weights_path=None,
    ):
        """
        Assign ocean depths using Cressman distance-weighted interpolation.
        Mirrors ``interp_smooth.f90`` from the tx2_3 topography workflow.

        For each ocean T-cell a smoothing radius ``L = smooth_scl * sqrt(cell_area)``
        is computed. Source ocean points within ``L`` are averaged with weights

        .. math::

            w = \\left(\\frac{L^2 - r^2}{L^2 + r^2}\\right)^{c}

        where ``r`` is the great-circle arc distance and ``c = cressman_exp``.
        Only source points with positive depth (ocean) contribute, so depth
        estimates are never contaminated by land elevations.

        Weights are computed in :func:`~mom6_forge.mapping.compute_cressman_weights`,
        saved to an ESMF-compatible netCDF via
        :func:`~mom6_forge.mapping.write_cressman_weights`, and applied through
        ``xe.Regridder`` — all orchestrated by
        :func:`~mom6_forge.mapping.cressman_regrid`. Cells that receive no source
        coverage are filled by iterative neighbour averaging (up to 100 passes).

        Parameters
        ----------
        src : SourceBathy
            Loaded (sliced) source bathymetry object.
        mask : xr.DataArray
            Binary ocean mask on the T-grid (1 = ocean, 0 = land). Obtain from
            :meth:`generate_mask_ocean_frac` or :meth:`generate_mask_cartopy`.
        smooth_scl : float
            Smoothing scale multiplier for the Cressman radius. Default ``2.0``.
        cressman_exp : float
            Exponent for the Cressman weight function. Default ``2.0``.
        hmin : float or None
            Minimum ocean depth (m). Defaults to ``self.min_depth``.
        weights_path : str or Path or None
            Where to save the ESMF weights netCDF. If ``None``, a file named
            ``cressman_weights.nc`` is written next to the bathymetry file.
        """
        if weights_path is None:
            weights_path = src.path.parent / "cressman_weights.nc"

        # --- Regrid via mapping module (weights → file → xe.Regridder) ---
        depth_dst, unfilled = cressman_regrid(
            src.lon,
            src.lat,
            src.depth,
            self._grid.tlon.values,
            self._grid.tlat.values,
            self._grid.tarea.values,
            mask.values.astype(bool),
            weights_path=weights_path,
            smooth_scl=smooth_scl,
            cressman_exp=cressman_exp,
        )

        # --- Iterative neighbour fill for cells with no source coverage ---
        if unfilled.any():
            n_miss = int(unfilled.sum())
            print(f"Filling {n_miss} cells by iterative neighbour averaging…")
            mask_2d = mask.values.astype(bool)
            unfilled_2d = unfilled & mask_2d

            for _ in range(100):
                if not unfilled_2d.any():
                    break
                filled_f = (~unfilled_2d).astype(float)
                d_pad = np.pad(depth_dst, 1, mode="edge")
                f_pad = np.pad(filled_f, 1, mode="constant", constant_values=0)
                d_nbr = (
                    d_pad[:-2, 1:-1]
                    + d_pad[2:, 1:-1]
                    + d_pad[1:-1, :-2]
                    + d_pad[1:-1, 2:]
                )
                f_nbr = (
                    f_pad[:-2, 1:-1]
                    + f_pad[2:, 1:-1]
                    + f_pad[1:-1, :-2]
                    + f_pad[1:-1, 2:]
                )
                can_fill = unfilled_2d & (f_nbr > 0)
                depth_dst = np.where(can_fill, d_nbr / np.maximum(f_nbr, 1), depth_dst)
                unfilled_2d = unfilled_2d & ~can_fill

        # --- Enforce mask and minimum depth ---
        mask_2d = mask.values.astype(bool)
        depth_dst = np.where(mask_2d, depth_dst, 0.0)
        _hmin = self._min_depth if hmin is None else hmin
        depth_dst = np.where(mask_2d & (depth_dst < _hmin), _hmin, depth_dst)

        self.send_entire_depth_change_to_tcm(
            xr.DataArray(
                depth_dst.astype(float), dims=["ny", "nx"], attrs={"units": "m"}
            )
        )

    def direct_xesmf_regrid(
        self,
        src,
        regridding_method="bilinear",
        fill_channels=False,
        positive_down=False,
        mask=None,
        mask_method=None,
        nx_sub=5,
        ny_sub=5,
        mask_threshold=0.5,
        cartopy_resolution="50m",
    ):
        """
        Regrid source bathymetry onto the model grid using ``xesmf`` and run
        lake-removal and channel cleanup via :meth:`tidy_dataset`.

        This is equivalent to :meth:`set_from_dataset` but exposes mask options
        so that an externally computed ocean mask can be passed to
        :meth:`tidy_dataset` instead of having it derived from the sign of the
        regridded depth field.

        Parameters
        ----------
        src : SourceBathy
            Loaded (sliced) source bathymetry object.
        regridding_method : str
            xesmf regridding method. Default ``"bilinear"``.
        fill_channels : bool
            Fill diagonal one-cell channels. Default ``False``.
        positive_down : bool
            Set ``True`` if the source elevation is already positive-downward.
            Default ``False``.
        mask : xr.DataArray or None
            Pre-computed binary ocean mask (1=ocean, 0=land) to pass directly to
            :meth:`tidy_dataset`. Overrides ``mask_method``.
        mask_method : {``"ocean_frac"``, ``"cartopy"``} or None
            Auto-generate a mask before regridding.  ``None`` (default) falls
            back to the original :meth:`tidy_dataset` behaviour of deriving the
            mask from the sign of the regridded depth.
        nx_sub, ny_sub : int
            Sub-sampling resolution for ``mask_method="ocean_frac"``.
        mask_threshold : float
            OCN_FRAC threshold for ``mask_method="ocean_frac"``. Default 0.5.
        cartopy_resolution : str
            Natural Earth resolution for ``mask_method="cartopy"``. Default ``"50m"``.
        """
        # --- Optional mask generation ---
        if mask is None and mask_method is not None:
            if mask_method == "ocean_frac":
                mask = self.generate_mask_ocean_frac(
                    src,
                    nx_sub=nx_sub,
                    ny_sub=ny_sub,
                    mask_threshold=mask_threshold,
                )
            elif mask_method == "cartopy":
                mask = self.generate_mask_cartopy(resolution=cartopy_resolution)
            else:
                raise ValueError(
                    f"Unknown mask_method '{mask_method}'. Use 'ocean_frac' or 'cartopy'."
                )

        # --- xesmf regrid ---
        bathymetry_output, empty_bathy = self.config_dataset(
            src,
            fill_channels=fill_channels,
            positive_down=positive_down,
            write_to_file=False,
        )
        regridded = regrid_dataset_via_xesmf(
            input_dataset=bathymetry_output,
            output_dataset=empty_bathy,
            regridding_method=regridding_method,
            write_to_file=False,
        )

        # --- Tidy (lake fill, channel cleanup, min depth) ---
        self.tidy_dataset(
            fill_channels=fill_channels,
            positive_down=positive_down,
            vertical_coordinate_name="depth",
            bathymetry=regridded,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            mask=mask,
        )

    def high_res_regrid(
        self,
        src,
        mask=None,
        mask_method="ocean_frac",
        nx_sub=5,
        ny_sub=5,
        mask_threshold=0.5,
        cartopy_resolution="50m",
        smooth_scl=2.0,
        cressman_exp=2.0,
        hmin=None,
        weights_path=None,
        fill_channels=False,
    ):
        """
        High-accuracy bathymetry pipeline for coarser grids (≳ 0.05°).

        Runs: **mask generation** → :meth:`cressman_interp` →
        :meth:`tidy_dataset` (lake fill, channel cleanup, minimum depth).

        Both the Cressman interpolation and tidy cleanup use the same mask, so
        land contamination of coastal depth estimates is minimised at every step.

        Parameters
        ----------
        src : SourceBathy
            Loaded (sliced) source bathymetry object.
        mask : xr.DataArray or None
            Pre-computed binary ocean mask (1=ocean, 0=land). When provided,
            ``mask_method`` is ignored.
        mask_method : {``"ocean_frac"``, ``"cartopy"``}
            Mask generation method when ``mask`` is not provided.
            Default ``"ocean_frac"`` (Monte-Carlo sub-sampling).
        nx_sub, ny_sub : int
            Sub-sampling resolution for ``mask_method="ocean_frac"``. Default 5.
        mask_threshold : float
            OCN_FRAC threshold for ``mask_method="ocean_frac"``. Default 0.5.
        cartopy_resolution : str
            Natural Earth resolution for ``mask_method="cartopy"``. Default ``"50m"``.
        smooth_scl : float
            Cressman smoothing scale multiplier. Default ``2.0``.
        cressman_exp : float
            Cressman weight exponent. Default ``2.0``.
        hmin : float or None
            Minimum ocean depth (m). Defaults to ``self.min_depth``.
        weights_path : str or Path or None
            Where to save the Cressman ESMF weights netCDF. Passed through to
            :meth:`cressman_interp`.
        fill_channels : bool
            Fill diagonal one-cell channels in :meth:`tidy_dataset`. Default ``False``.
        """
        # --- Mask ---
        if mask is None:
            if mask_method == "ocean_frac":
                mask = self.generate_mask_ocean_frac(
                    src,
                    nx_sub=nx_sub,
                    ny_sub=ny_sub,
                    mask_threshold=mask_threshold,
                )
            elif mask_method == "cartopy":
                mask = self.generate_mask_cartopy(resolution=cartopy_resolution)
            else:
                raise ValueError(
                    f"Unknown mask_method '{mask_method}'. Use 'ocean_frac' or 'cartopy'."
                )

        # --- Cressman interpolation ---
        self.cressman_interp(
            src,
            mask=mask,
            smooth_scl=smooth_scl,
            cressman_exp=cressman_exp,
            hmin=hmin,
            weights_path=weights_path,
        )

        # --- Tidy (lake fill, channel cleanup) ---
        # Build a dataset from the Cressman-interpolated depth for tidy_dataset
        bathy_ds = xr.Dataset(
            {"depth": (["ny", "nx"], self.depth.values)},
            coords={
                "lon": (["ny", "nx"], self._grid.tlon.values),
                "lat": (["ny", "nx"], self._grid.tlat.values),
            },
        )
        self.tidy_dataset(
            fill_channels=fill_channels,
            positive_down=True,  # cressman_interp produces positive-down depths
            vertical_coordinate_name="depth",
            bathymetry=bathy_ds,
            longitude_coordinate_name="lon",
            latitude_coordinate_name="lat",
            mask=mask,
        )

    def set_from_dataset(
        self,
        bathymetry_path,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        vertical_coordinate_name="elevation",
        mask=None,
        mask_method=None,
        nx_sub=5,
        ny_sub=5,
        mask_threshold=0.5,
        cartopy_resolution="50m",
        smooth_scl=2.0,
        cressman_exp=2.0,
        hmin=None,
        weights_path=None,
        fill_channels=False,
        positive_down=False,
        regridding_method="bilinear",
    ):
        """
        Auto-selecting bathymetry pipeline.

        Calls :meth:`diagnose_resolution` to compare the model grid spacing
        with the source bathymetry resolution.  Based on the result it dispatches
        to one of two pipelines:

        * **ratio ≥ 12×** → :meth:`high_res_regrid`
          (Monte-Carlo mask + Cressman interpolation + tidy cleanup).
          Recommended for grids of ~0.05° and coarser.
        * **ratio < 12×** → :meth:`direct_xesmf_regrid`
          (xesmf bilinear/conservative regrid + tidy cleanup).
          Sufficient for fine grids (~1–3 km).

        All parameters are forwarded to the selected pipeline.

        Parameters
        ----------
        bathymetry_path : str or Path
        longitude_coordinate_name, latitude_coordinate_name, vertical_coordinate_name : str
        mask : xr.DataArray or None
            Pre-computed binary ocean mask to pass directly to the chosen
            pipeline, skipping mask generation. Overrides ``mask_method``.
        mask_method : {``"ocean_frac"``, ``"cartopy"``} or None
            Mask generation method.  When ``None`` and the high-res pipeline
            is selected, defaults to ``"ocean_frac"``.  When ``None`` and the
            direct pipeline is selected, :meth:`tidy_dataset` derives the mask
            from the sign of the regridded depth (original behaviour).
        nx_sub, ny_sub : int
            Sub-sampling resolution for ``mask_method="ocean_frac"``.
        mask_threshold : float
            OCN_FRAC threshold for ``mask_method="ocean_frac"``. Default 0.5.
        cartopy_resolution : str
            Natural Earth resolution for ``mask_method="cartopy"``. Default ``"50m"``.
        smooth_scl : float
            Cressman smoothing scale (high-res pipeline only). Default ``2.0``.
        cressman_exp : float
            Cressman weight exponent (high-res pipeline only). Default ``2.0``.
        hmin : float or None
            Minimum ocean depth (m). Defaults to ``self.min_depth``.
        weights_path : str or Path or None
            Where to save Cressman weights (high-res pipeline only).
        fill_channels : bool
            Fill diagonal one-cell channels. Default ``False``.
        positive_down : bool
            Set ``True`` if the source elevation is already positive-downward
            (direct pipeline only). Default ``False``.
        regridding_method : str
            xesmf regridding method (direct pipeline only). Default ``"bilinear"``.
        """
        src = self._get_src(
            bathymetry_path,
            longitude_coordinate_name,
            latitude_coordinate_name,
            vertical_coordinate_name,
        )
        use_cressman = self.diagnose_resolution(src)

        if use_cressman:
            self.high_res_regrid(
                src,
                mask=mask,
                mask_method=mask_method or "ocean_frac",
                nx_sub=nx_sub,
                ny_sub=ny_sub,
                mask_threshold=mask_threshold,
                cartopy_resolution=cartopy_resolution,
                smooth_scl=smooth_scl,
                cressman_exp=cressman_exp,
                hmin=hmin,
                weights_path=weights_path,
                fill_channels=fill_channels,
            )
        else:
            self.direct_xesmf_regrid(
                src,
                regridding_method=regridding_method,
                fill_channels=fill_channels,
                positive_down=positive_down,
                mask=mask,
                mask_method=mask_method,
                nx_sub=nx_sub,
                ny_sub=ny_sub,
                mask_threshold=mask_threshold,
                cartopy_resolution=cartopy_resolution,
            )

    def mpi_direct_xesmf_regrid(
        self,
        src,
        *,
        fill_channels=False,
        positive_down=False,
        output_dir=Path(""),
        write_to_file=True,
        verbose=True,
    ):
        if verbose:
            print(f"""
            *MANUAL REGRIDDING INSTRUCTIONS*

            Calling `[object_name].mpi_direct_xesmf_regrid` sets up the files necessary for regridding
            the bathymetry using mpirun and ESMF_Regrid. See below for the step-by-step instructions:

            1. There should be two files: `bathymetry_original.nc` and `bathymetry_unfinished.nc` located at
            {output_dir}.

            2. Open a terminal and change to this directory (e.g. `cd {output_dir}`).

            3. Request appropriate computational resources (see example script below), and run the command:

            `mpirun -np NUMBER_OF_CPUS ESMF_Regrid -s bathymetry_original.nc -d bathymetry_unfinished.nc -m bilinear --src_var depth --dst_var depth --netcdf4 --src_regional --dst_regional`

            4. Run Topo_object.tidy_bathymetry(args) to finish processing the bathymetry.

            Example PBS script using NCAR's Casper Machine: https://gist.github.com/AidanJanney/911290acaef62107f8e2d4ccef9d09be

            For additional details see: https://xesmf.readthedocs.io/en/latest/large_problems_on_HPC.html
            """)

        self.bathymetry_output, self.empty_bathy = self.config_dataset(
            src,
            fill_channels=fill_channels,
            positive_down=positive_down,
            output_dir=output_dir,
            write_to_file=write_to_file,
        )

        print(
            "Configuration complete. Ready for regridding with MPI. See documentation for more details."
        )

    def config_dataset(
        self,
        src,
        fill_channels=False,
        positive_down=False,
        output_dir=Path(""),
        write_to_file=True,
    ):
        """
        Sets up necessary objects/files for regridding bathymetry. Can be flexibly used with
        mapping.regrid_bathy_dataset() or user can manually regrid with ESMF_regrid.

        If manual regridding is necessary, write_to_file must be set to True.

        Arguments:
            src (SourceBathy): Loaded (sliced) source bathymetry object.
            output_dir: str | Path
                The str or Path the write to file should write to. Defaults to the directory the script is running in.
            write_to_file (Optional[bool]): Files saved to ``output_dir``. Defaults to ``True``. Must be set to true if using manual regridding methods with ESMF_regrid.

        Returns:
            (``bathymetry_output``,``empty_bathy``) (tuple of Datasets): where ``bathymetry_output`` is the original bathymetry data with proper metadata and attributes and ``empty_bathy`` is a template for the regridder.
        """
        # Use the cached, already-sliced DataArray — output format is unchanged.
        bathymetry = src.da.astype("float")
        bathymetry.attrs["missing_value"] = -1e20  # missing value expected by FRE tools
        bathymetry_output = xr.Dataset({"depth": bathymetry})

        bathymetry_output = bathymetry_output.rename(
            {src.lon_name: "lon", src.lat_name: "lat"}
        )

        bathymetry_output.depth.attrs["_FillValue"] = -1e20
        bathymetry_output.depth.attrs["units"] = "meters"
        bathymetry_output.depth.attrs["standard_name"] = (
            "height_above_reference_ellipsoid"
        )
        bathymetry_output.depth.attrs["long_name"] = "Elevation relative to sea level"
        bathymetry_output.depth.attrs["coordinates"] = "lon lat"
        if write_to_file:
            bathymetry_output.to_netcdf(
                output_dir / "bathymetry_original.nc",
                mode="w",
                engine="netcdf4",
            )

        empty_bathy = xr.Dataset(
            {
                "lon": self._grid.tlon,
                "lat": self._grid.tlat,
            }
        )

        empty_bathy = empty_bathy.set_coords(("lon", "lat"))
        empty_bathy["depth"] = xr.zeros_like(empty_bathy["lon"])
        empty_bathy.lon.attrs["units"] = "degrees_east"
        empty_bathy.lon.attrs["_FillValue"] = 1e20
        empty_bathy.lat.attrs["units"] = "degrees_north"
        empty_bathy.lat.attrs["_FillValue"] = 1e20
        empty_bathy.depth.attrs["units"] = "meters"
        empty_bathy.depth.attrs["coordinates"] = "lon lat"
        if write_to_file:
            empty_bathy.to_netcdf(
                output_dir / "bathymetry_unfinished.nc",
                mode="w",
                engine="netcdf4",
            )
            empty_bathy.close()
        return bathymetry_output, empty_bathy

    def tidy_dataset(
        self,
        fill_channels=False,
        positive_down=False,
        vertical_coordinate_name="depth",
        bathymetry=None,
        output_dir=Path(""),
        write_to_file=True,
        longitude_coordinate_name="lon",
        latitude_coordinate_name="lat",
        mask=None,
    ):
        """
        An auxiliary method for bathymetry used to fix up the metadata and remove inland
        lakes after regridding the bathymetry. Having :func:`~tidy_dataset` as a separate
        method from :func:`~setup_bathymetry` allows for the regridding to be done separately,
        since regridding can be really expensive for large domains.

        If the bathymetry is already regridded and what is left to be done is fixing the metadata
        or fill in some channels, then :func:`~tidy_dataset` directly can read the existing
        ``bathymetry_unfinished.nc`` file that should be in the input directory.

        Arguments:
            fill_channels (Optional[bool]): Whether to fill in diagonal channels.
                This removes more narrow inlets, but can also connect extra islands to land.
                Default: ``False``.
            positive_down (Optional[bool]): If ``False`` (default), assume that
                bathymetry vertical coordinate is positive down, as is the case in GEBCO for example.
            bathymetry (Optional[xr.Dataset]): The bathymetry dataset to tidy up. If not provided,
                it will read the bathymetry from the file ``bathymetry_unfinished.nc`` in the input directory
                that was created by :func:`~config/regrid_dataset`.
        """
        ## reopen bathymetry to modify
        print(
            "Tidy bathymetry: Reading in regridded bathymetry to fix up metadata...",
            end="",
        )
        if read_bathy_from_file := bathymetry is None:
            bathymetry = xr.open_dataset(
                output_dir / "bathymetry_unfinished.nc", engine="netcdf4"
            )

        ## Ensure correct encoding
        bathymetry = xr.Dataset(
            {"depth": (["ny", "nx"], bathymetry[vertical_coordinate_name].values)},
            coords={
                "lon": (["ny", "nx"], bathymetry[longitude_coordinate_name].values),
                "lat": (["ny", "nx"], bathymetry[latitude_coordinate_name].values),
            },
        )
        bathymetry.attrs["depth"] = "meters"
        bathymetry.attrs["standard_name"] = "bathymetric depth at T-cell centers"
        bathymetry.attrs["coordinates"] = "zi"

        bathymetry.expand_dims("tiles", 0)

        if not positive_down:
            ## Ensure that coordinate is positive down!
            bathymetry["depth"] *= -1

        ## Make a land mask — use external mask if provided, otherwise derive from depth sign
        if mask is not None:
            ocean_mask = mask.astype(int)
        else:
            ocean_mask = xr.where(bathymetry.depth <= 0, 0, 1)
        land_mask = np.abs(ocean_mask - 1)

        ## REMOVE INLAND LAKES
        print("done. Filling in inland lakes and channels... ", end="")

        changed = True  ## keeps track of whether solution has converged or not

        forward = True  ## only useful for iterating through diagonal channel removal. Means iteration goes SW -> NE

        while changed == True:
            ## First fill in all lakes.
            ## scipy.ndimage.binary_fill_holes fills holes made of 0's within a field of 1's
            land_mask[:, :] = binary_fill_holes(land_mask.data)
            ## Get the ocean mask instead of land- easier to remove channels this way
            ocean_mask = np.abs(land_mask - 1)

            ## Now fill in all one-cell-wide channels
            newmask = xr.where(
                ocean_mask * (land_mask.shift(nx=1) + land_mask.shift(nx=-1)) == 2, 1, 0
            )
            newmask += xr.where(
                ocean_mask * (land_mask.shift(ny=1) + land_mask.shift(ny=-1)) == 2, 1, 0
            )

            if fill_channels == True:
                ## fill in all one-cell-wide horizontal channels
                newmask = xr.where(
                    ocean_mask * (land_mask.shift(nx=1) + land_mask.shift(nx=-1)) == 2,
                    1,
                    0,
                )
                newmask += xr.where(
                    ocean_mask * (land_mask.shift(ny=1) + land_mask.shift(ny=-1)) == 2,
                    1,
                    0,
                )
                ## Diagonal channels
                if forward == True:
                    ## horizontal channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=1))
                        * (
                            land_mask.shift({"nx": 1, "ny": 1})
                            + land_mask.shift({"ny": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up right & below
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=1))
                        * (
                            land_mask.shift({"nx": 1, "ny": -1})
                            + land_mask.shift({"ny": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down right & above
                    ## Vertical channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=1))
                        * (
                            land_mask.shift({"nx": 1, "ny": 1})
                            + land_mask.shift({"nx": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up right & left
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=1))
                        * (
                            land_mask.shift({"nx": -1, "ny": 1})
                            + land_mask.shift({"nx": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up left & right

                    forward = False

                if forward == False:
                    ## Horizontal channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=-1))
                        * (
                            land_mask.shift({"nx": -1, "ny": 1})
                            + land_mask.shift({"ny": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## up left & below
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(nx=-1))
                        * (
                            land_mask.shift({"nx": -1, "ny": -1})
                            + land_mask.shift({"ny": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down left & above
                    ## Vertical channels
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=-1))
                        * (
                            land_mask.shift({"nx": 1, "ny": -1})
                            + land_mask.shift({"nx": -1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down right & left
                    newmask += xr.where(
                        (ocean_mask * ocean_mask.shift(ny=-1))
                        * (
                            land_mask.shift({"nx": -1, "ny": -1})
                            + land_mask.shift({"nx": 1})
                        )
                        == 2,
                        1,
                        0,
                    )  ## down left & right

                    forward = True

            newmask = xr.where(newmask > 0, 1, 0)
            changed = np.max(newmask) == 1
            land_mask += newmask

        ocean_mask = np.abs(land_mask - 1)

        bathymetry["depth"] *= ocean_mask

        ## Now, any points in the bathymetry that are shallower than minimum depth are set to minimum depth.
        ## This preserves the true land/ocean mask.
        bathymetry["depth"] = bathymetry["depth"].where(bathymetry["depth"] > 0, np.nan)
        bathymetry["depth"] = bathymetry["depth"].where(
            ~(bathymetry.depth <= self.min_depth), self.min_depth + 0.1
        )
        bathymetry = bathymetry.fillna(
            0
        )  # After min_depth filtering, move the land values to zero
        bathymetry.depth.attrs["units"] = "meters"
        new_values = bathymetry.depth

        # Save to object (Build TCM Object)
        self.send_entire_depth_change_to_tcm(new_values)

    def erase_selected_basin(self, i, j):
        label = self.basintmask.data[j, i]
        affected = np.where(self.basintmask.data == label)
        indices = list(zip(affected[0], affected[1]))
        if not indices:
            return
        old_values = [self.depth.data[jj, ii] for jj, ii in indices]
        new_values = [0] * len(indices)
        cmd = DepthEditCommand(self, indices, new_values, old_values=old_values)
        self.tcm.execute(cmd)

    def erase_disconnected_basin(self, i, j):
        label = self.basintmask.data[j, i]
        affected = np.where(self.basintmask.data != label)
        indices = list(zip(affected[0], affected[1]))
        if not indices:
            return
        old_values = [self.depth.data[jj, ii] for jj, ii in indices]
        new_values = [0] * len(indices)
        cmd = DepthEditCommand(self, indices, new_values, old_values=old_values)
        self.tcm.execute(cmd)

    def apply_ridge(self, height, width, lon, ilat):
        """
        Apply a ridge to the bathymetry.

        Parameters
        ----------
        height : float
            Height of the ridge to be added.
        width : float
            Width of the ridge to be added.
        lon : float
            Longitude where the ridge is to be centered.
        ilat : pair of integers
            Initial and final latitude indices for the ridge.
        """

        ridge_lon = [
            self._grid.tlon[0, 0].data,
            lon - width / 2.0,
            lon,
            lon + width / 2.0,
            self._grid.tlon[0, -1].data,
        ]
        ridge_height = [0.0, 0.0, -height, 0.0, 0.0]
        interp_func = interpolate.interp1d(ridge_lon, ridge_height, kind=2)
        ridge_height_mapped = interp_func(self._grid.tlon[0, :])
        ridge_height_mapped = np.where(
            ridge_height_mapped <= 0.0, ridge_height_mapped, 0.0
        )
        affected_indices = []
        old_vals = []
        new_vals = []
        for j in range(ilat[0], ilat[1]):
            affected_indices.extend([(j, i) for i in range(self._grid.nx)])
            old_vals.extend(self._depth[j, :].values)
            new_vals.extend((self._depth[j, :] + ridge_height_mapped).values)
        depth_edit_command = DepthEditCommand(
            self, affected_indices, new_vals, old_values=old_vals
        )
        self.tcm.execute(depth_edit_command)

    def apply_land_frac(
        self,
        landfrac_filepath,
        landfrac_name,
        xcoord_name,
        ycoord_name,
        depth_fillval=0.0,
        cutoff_frac=0.5,
        method="bilinear",
    ):
        """
        Given a dataset containing land fraction, generate and apply ocean mask.

        Parameters
        ----------
        landfrac_filepath : str
            Path the netcdf file containing the land fraction field.
        landfrac_name : str
            The field name corresponding to the land fraction  (e.g., "landfrac").
        xcoord_name : str
            The name of the x coordinate of the landfrac dataset (e.g., "lon").
        ycoord_name : str
            The name of the y coordinate of the landfrac dataset (e.g., "lat").
        depth_fillval : float
            The depth value for dry cells.
        cutoff_frac : float
            Cells with landfrac > cutoff_frac are deemed land cells.
        method : str
            Mapping method for determining the ocean mask (lnd -> ocn)
        """

        import xesmf as xe

        assert isinstance(landfrac_filepath, str), "landfrac_filepath must be a string"
        assert landfrac_filepath.endswith(
            ".nc"
        ), "landfrac_filepath must point to a netcdf file"
        ds = xr.open_dataset(landfrac_filepath)

        assert isinstance(landfrac_name, str), "landfrac_name must be a string"
        assert (
            landfrac_name in ds
        ), f"Couldn't find {landfrac_name} in {landfrac_filepath}"
        assert isinstance(xcoord_name, str), "xcoord_name must be a string"
        assert (
            landfrac_name in ds
        ), f"Couldn't find {xcoord_name} in {landfrac_filepath}"
        assert isinstance(ycoord_name, str), "ycoord_name must be a string"
        assert (
            landfrac_name in ds
        ), f"Couldn't find {ycoord_name} in {landfrac_filepath}"
        assert isinstance(
            depth_fillval, float
        ), f"depth_fillval={depth_fillval} must be a float"
        assert (
            depth_fillval < self._min_depth
        ), f"depth_fillval (the depth of dry cells) must be smaller than the minimum depth {self._min_depth}"
        assert isinstance(
            cutoff_frac, float
        ), f"cutoff_frac={cutoff_frac} must be a float"
        assert (
            0.0 <= cutoff_frac <= 1.0
        ), f"cutoff_frac={cutoff_frac} must be 0<= and <=1"

        valid_methods = [
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
            "nearest_s2d",
            "nearest_d2s",
        ]
        assert (
            method in valid_methods
        ), f"{method} is not a valid mapping method. Choose from: {valid_methods}"

        ds_mapped = xr.Dataset(
            data_vars={}, coords={"lat": self._grid.tlat, "lon": self._grid.tlon}
        )

        regridder = xe.Regridder(ds, ds_mapped, method, periodic=self._grid.is_cyclic_x)
        mask_mapped = regridder(ds.landfrac)

        # Build TCM Object - This is not the entire depth change, just the cells to be filled
        mask = mask_mapped > cutoff_frac  # boolean mask
        ny, nx = self._depth.shape

        affected_indices = []
        old_vals = []
        new_vals = []

        for j in range(ny):
            for i in range(nx):

                if mask[j, i]:
                    affected_indices.append((j, i))
                    old_val = self._depth[j, i]
                    new_val = depth_fillval

                    old_vals.append(old_val)
                    new_vals.append(new_val)

        depth_edit_command = DepthEditCommand(
            self, affected_indices, new_vals, old_values=old_vals
        )
        self.tcm.execute(depth_edit_command)

    def gen_topo_ds(self, title=None):
        """
        Write the TOPO_FILE (bathymetry file) in xarray Dataset.

        Parameters
        ----------
        title: str, optional
            File title.
        """
        ds = xr.Dataset()

        # global attrs:
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title
        else:
            ds.attrs["title"] = "MOM6 topography file"
        ds.attrs["min_depth"] = self.min_depth
        ds.attrs["max_depth"] = self.max_depth

        ds["y"] = xr.DataArray(
            self._grid.tlat,
            dims=["ny", "nx"],
            attrs={
                "long_name": "array of t-grid latitudes",
                "units": self._grid.tlat.units,
            },
        )

        ds["x"] = xr.DataArray(
            self._grid.tlon,
            dims=["ny", "nx"],
            attrs={
                "long_name": "array of t-grid longitutes",
                "units": self._grid.tlon.units,
            },
        )

        ds["mask"] = xr.DataArray(
            self.tmask.astype(np.int32),
            dims=["ny", "nx"],
            attrs={
                "long_name": "landsea mask at t points: 1 ocean, 0 land",
                "units": "nondim",
            },
        )

        ds["depth"] = xr.DataArray(
            self._depth.data,
            dims=["ny", "nx"],
            attrs={"long_name": "t-grid cell depth", "units": "m"},
        )

        topo_stats = self._src._topo_stats if self._src is not None else None
        if topo_stats is not None:
            h2 = topo_stats["D2_mean"] - topo_stats["D_mean"] ** 2
            ds["h2"] = xr.DataArray(
                h2.values,
                dims=["ny", "nx"],
                attrs={
                    "long_name": "subgrid topographic height variance at T-points",
                    "units": "m2",
                    "comment": "h2 = D2_mean - D_mean^2; input to MOM6 Lee wave / topo drag parameterisation",
                },
            )

        return ds

    def save(self):
        """
        Save the TOPO_FILE (bathymetry file) in netcdf format to version control
        """

        self.tcm.save()

    def write_topo(self, file_path, title=None, enforce_topo_drag=False):
        """
        Write the TOPO_FILE (bathymetry file) in netcdf format. The written file is
        to be read in by MOM6 during runtime.

        Parameters
        ----------
        file_path: str
            Path to TOPO_FILE to be written.
        title: str, optional
            File title.
        enforce_topo_drag: bool, optional
            If ``True``, raise an error if topo drag stats (``h2``) have not been
            computed. Call ``generate_mask_ocean_frac()`` first to compute them.
            Default ``False``.
        """
        has_topo_stats = self._src is not None and self._src._topo_stats is not None
        if enforce_topo_drag and not has_topo_stats:
            raise RuntimeError(
                "enforce_topo_drag=True but topo stats have not been computed. "
                "Call generate_mask_ocean_frac() first to compute them."
            )

        ds = self.gen_topo_ds(title=title)
        ds.to_netcdf(file_path, format="NETCDF3_64BIT")

    def write_cice_grid(self, file_path):
        """
        Write the CICE grid file in netcdf format. The written file is
        to be read in by CICE during runtime.

        Parameters
        ----------
        file_path: str
            Path to CICE grid file to be written.
        """

        assert (
            "degrees" in self._grid.tlat.units and "degrees" in self._grid.tlon.units
        ), "Unsupported coord"

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["title"] = "CICE grid file"

        ny = self._grid.ny
        nx = self._grid.nx

        ds["ulat"] = xr.DataArray(
            np.deg2rad(self._grid.qlat[1:, 1:].data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "U grid center latitude",
                "units": "radians",
                "bounds": "latu_bounds",
            },
        )

        ds["ulon"] = xr.DataArray(
            np.deg2rad(self._grid.qlon[1:, 1:].data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "U grid center longitude",
                "units": "radians",
                "bounds": "lonu_bounds",
            },
        )

        ds["tlat"] = xr.DataArray(
            np.deg2rad(self._grid.tlat.data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "T grid center latitude",
                "units": "degrees_north",
                "bounds": "latt_bounds",
            },
        )

        ds["tlon"] = xr.DataArray(
            np.deg2rad(self._grid.tlon.data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "T grid center longitude",
                "units": "degrees_east",
                "bounds": "lont_bounds",
            },
        )

        ds["htn"] = xr.DataArray(
            self._grid.dxCv.data * 100.0,
            dims=["nj", "ni"],
            attrs={
                "long_name": "T cell width on North side",
                "units": "cm",
                "coordinates": "TLON TLAT",
            },
        )

        ds["hte"] = xr.DataArray(
            self._grid.dyCu.data * 100,
            dims=["nj", "ni"],
            attrs={
                "long_name": "T cell width on East side",
                "units": "cm",
                "coordinates": "TLON TLAT",
            },
        )

        ds["angle"] = xr.DataArray(
            np.deg2rad(
                self._grid.angle_q.data[
                    1:, 1:
                ]  # Slice the q-grid from MOM6 (which is u-grid in CICE/POP) to CICE/POP convention, the top right of the t points
            ),
            dims=["nj", "ni"],
            attrs={
                "long_name": "angle grid makes with latitude line on U grid",
                "units": "radians",
                "coordinates": "ULON ULAT",
            },
        )

        ds["anglet"] = xr.DataArray(
            np.deg2rad(self._grid.angle.data),
            dims=["nj", "ni"],
            attrs={
                "long_name": "angle grid makes with latitude line on T grid",
                "units": "radians",
                "coordinates": "TLON TLAT",
            },
        )

        ds["kmt"] = xr.DataArray(
            self.tmask.astype(np.float32),
            dims=["nj", "ni"],
            attrs={
                "long_name": "mask of T grid cells",
                "units": "unitless",
                "coordinates": "TLON TLAT",
            },
        )

        ds.to_netcdf(
            file_path,
            format="NETCDF3_64BIT",
        )

    def write_scrip_grid(self, file_path, title=None):
        """
        Write the SCRIP grid file. In latest CESM versions, SCRIP grid files are
        no longer required and are replaced by ESMF mesh files. However, SCRIP
        files are still needed to generate custom ocean-runoff mapping files.

        Parameters
        ----------
        file_path: str
            Path to SCRIP file to be written.
        title: str, optional
            File title.
        """

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["Conventions"] = "SCRIP"
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title

        ds["grid_dims"] = xr.DataArray(
            np.array([self._grid.nx, self._grid.ny]).astype(np.int32),
            dims=["grid_rank"],
        )
        ds["grid_center_lat"] = xr.DataArray(
            self._grid.tlat.data.flatten(),
            dims=["grid_size"],
            attrs={"units": self._grid.supergrid.axis_units},
        )
        ds["grid_center_lon"] = xr.DataArray(
            self._grid.tlon.data.flatten(),
            dims=["grid_size"],
            attrs={"units": self._grid.supergrid.axis_units},
        )
        ds["grid_imask"] = xr.DataArray(
            self.tmask.data.astype(np.int32).flatten(),
            dims=["grid_size"],
            attrs={"units": "unitless"},
        )

        ds["grid_corner_lat"] = xr.DataArray(
            np.zeros((ds.sizes["grid_size"], 4)),
            dims=["grid_size", "grid_corners"],
            attrs={"units": self._grid.supergrid.axis_units},
        )
        ds["grid_corner_lon"] = xr.DataArray(
            np.zeros((ds.sizes["grid_size"], 4)),
            dims=["grid_size", "grid_corners"],
            attrs={"units": self._grid.supergrid.axis_units},
        )

        i_range = range(self._grid.nx)
        j_range = range(self._grid.ny)
        j, i = np.meshgrid(j_range, i_range, indexing="ij")
        k = j * self._grid.nx + i

        ds["grid_corner_lat"].data[k] = np.stack(
            (
                self._grid.qlat.data[j, i],
                self._grid.qlat.data[j, i + 1],
                self._grid.qlat.data[j + 1, i + 1],
                self._grid.qlat.data[j + 1, i],
            ),
            axis=-1,
        )

        ds["grid_corner_lon"].data[k] = np.stack(
            (
                self._grid.qlon.data[j, i],
                self._grid.qlon.data[j, i + 1],
                self._grid.qlon.data[j + 1, i + 1],
                self._grid.qlon.data[j + 1, i],
            ),
            axis=-1,
        )

        ds["grid_area"] = xr.DataArray(
            cell_area_rad(ds.grid_corner_lon.data, ds.grid_corner_lat.data),
            dims=["grid_size"],
            attrs={"units": "radians^2"},
        )

        ds.to_netcdf(
            file_path,
            format="NETCDF3_64BIT",
        )

    def write_esmf_mesh(self, file_path, title=None):
        """
        Write the ESMF mesh file

        Parameters
        ----------
        file_path: str
            Path to ESMF mesh file to be written.
        title: str, optional
            File title.
        """

        ds = xr.Dataset()

        # global attrs:
        ds.attrs["gridType"] = "unstructured mesh"
        ds.attrs["date_created"] = datetime.now().isoformat()
        if title:
            ds.attrs["title"] = title

        tlon_flat = self._grid.tlon.data.flatten()
        tlat_flat = self._grid.tlat.data.flatten()
        ncells = len(tlon_flat)  # i.e., elementCount in ESMF mesh nomenclature

        coord_units = self._grid.supergrid.axis_units

        ds["centerCoords"] = xr.DataArray(
            [[tlon_flat[i], tlat_flat[i]] for i in range(ncells)],
            dims=["elementCount", "coordDim"],
            attrs={"units": coord_units},
        )

        ds["numElementConn"] = xr.DataArray(
            np.full(ncells, 4).astype(np.int8),
            dims=["elementCount"],
            attrs={"long_name": "Node indices that define the element connectivity"},
        )

        ds["elementArea"] = xr.DataArray(
            self._grid.tarea.data.flatten(),
            dims=["elementCount"],
            attrs={"units": self._grid.tarea.units},
        )

        ds["elementMask"] = xr.DataArray(
            self.tmask.data.astype(np.int32).flatten(), dims=["elementCount"]
        )

        i0 = 1  # start index for node id's

        if self._grid.is_tripolar(self._grid._supergrid):
            nx, ny = self._grid.nx, self._grid.ny
            qlon_flat = self._grid.qlon.data[:, :-1].flatten()[: -(nx // 2 - 1)]
            qlat_flat = self._grid.qlat.data[:, :-1].flatten()[: -(nx // 2 - 1)]
            nnodes = len(qlon_flat)
            assert nnodes + (nx // 2 - 1) == nx * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            def get_element_conn(i):
                is_final_column = (i + 1) % nx == 0
                on_top_row = i // nx == ny - 1
                on_second_half_of_stitch = on_top_row and (i % nx) >= nx // 2

                # lower left corner
                ll = i0 + i % nx + (i // nx) * (nx)

                # lower right corner
                lr = ll + 1
                if is_final_column:
                    lr -= nx

                # upper right corner
                ur = lr + nx
                if on_second_half_of_stitch and not is_final_column:
                    ur -= 2 * (i % nx + 1 - nx // 2)

                # upper left corner
                ul = ll + nx
                if on_second_half_of_stitch:
                    ul = ur + 1

                return [ll, lr, ur, ul]

        elif self._grid.is_cyclic_x == True:

            nx, ny = self._grid.nx, self._grid.ny
            qlon_flat = self._grid.qlon.data[:, :-1].flatten()
            qlat_flat = self._grid.qlat.data[:, :-1].flatten()
            nnodes = len(qlon_flat)
            assert nnodes == nx * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            get_element_conn = lambda i: [
                i0 + i % nx + (i // nx) * (nx),
                i0 + i % nx + (i // nx) * (nx) + 1 - (((i + 1) % nx) == 0) * nx,
                i0 + i % nx + (i // nx + 1) * (nx) + 1 - (((i + 1) % nx) == 0) * nx,
                i0 + i % nx + (i // nx + 1) * (nx),
            ]

        else:  # non-cyclic grid
            nx, ny = self._grid.nx, self._grid.ny
            qlon_flat = self._grid.qlon.data.flatten()
            qlat_flat = self._grid.qlat.data.flatten()
            nnodes = len(qlon_flat)
            assert nnodes == (nx + 1) * (ny + 1)

            # Below returns element connectivity of i-th element
            # (assuming 0 based node and element indexing)
            get_element_conn = lambda i: [
                i0 + i % nx + (i // nx) * (nx + 1),
                i0 + i % nx + (i // nx) * (nx + 1) + 1,
                i0 + i % nx + (i // nx + 1) * (nx + 1) + 1,
                i0 + i % nx + (i // nx + 1) * (nx + 1),
            ]

        ds["nodeCoords"] = xr.DataArray(
            np.column_stack((qlon_flat, qlat_flat)),
            dims=["nodeCount", "coordDim"],
            attrs={"units": coord_units},
        )

        ds["elementConn"] = xr.DataArray(
            np.array([get_element_conn(i) for i in range(ncells)]).astype(np.int32),
            dims=["elementCount", "maxNodePElement"],
            attrs={
                "long_name": "Node indices that define the element connectivity",
                "start_index": np.int32(i0),
            },
        )

        self.mesh_path = file_path
        ds.to_netcdf(self.mesh_path, format="NETCDF3_64BIT")
