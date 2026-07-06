import os
import copy
import numpy as np
import xarray as xr
import xesmf as xe
from datetime import datetime
from typing import Optional
from scipy import interpolate
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import cKDTree
from mom6_forge.utils import cell_area_rad, iterative_fill, compute_subsampling_factor
from mom6_forge.grid import Grid
from mom6_forge.git_utils import get_domain_dir, get_repo
from pathlib import Path
from mom6_forge.edit_command import *
from mom6_forge.command_manager import TopoCommandManager, CommandType
from mom6_forge.mapping import (
    regrid_dataset_via_xesmf,
    regrid_with_subsampling,
    regrid_dataset_via_cressman,
)
from mom6_forge._source_bathy import SourceBathy
from mom6_forge.channel_width import ChannelWidthList
from mom6_forge._supergrid import haversine, _DEFAULT_RADIUS
import regionmask
from mom6_forge._supergrid import _DEFAULT_RADIUS, SupergridBase

VALID_MASK_METHODS = ("naturalearth", "ocean_frac", "dataset", "manual")
VALID_DEPTH_METHODS = ("stats", "cressman", "xesmf")


class Topo:
    """
    Bathymetry Generator for MOM6 grids (mom6_forge.grid.Grid).
    """

    def __init__(
        self,
        grid,
        min_depth,
        channel_widths=None,
        version_control_dir="TopoLibrary",
        git=False,
    ):
        """
        MOM6 Simpler Models bathymetry constructor.

        Parameters
        ----------
        grid: mom6_forge.grid.Grid
            horizontal grid instance for which the bathymetry is to be created.
        min_depth: float
            Minimum water column depth. Columns with shallow depths are to be masked out.
        channel_widths: str | Path | ChannelWidthList, optional
            Channel width constraints. Can be a filepath to load from, a ChannelWidthList object, or None.
        version_control_dir: str, optional
            Directory in which to store version-controlled bathymetry data. Defaults to
            "TopoLibrary". Ignored if git is False (version control is no longer used)
        git: bool, optional
            If True (default), initialize a git repository for version control and
            undo/redo support. If False, skip all git and filesystem side-effects;
            note that undo(), redo(), branch, and tag operations will be
            unavailable. TopoEditor also requires git=True.
        """

        self._grid = grid
        self._depth = xr.DataArray(
            np.full((grid.ny, grid.nx), np.nan, dtype=float),
            dims=["ny", "nx"],
            attrs={"units": "m"},
        )  # Initialize raw depth with NaNs
        self._user_mask: Optional[xr.DataArray] = (
            None  # Binary ocean/land mask (None = no mask applied)
        )
        self._min_depth = min_depth
        self._src = None  # SourceBathy object; set by set_src()
        self.land_fillval = 0.0  # Depth value for land cells
        initial_command = MinDepthEditCommand(
            self, attr="min_depth", new_value=min_depth
        )

        # Initialize channel widths
        if channel_widths is None:
            self.channel_widths = ChannelWidthList()
        elif isinstance(channel_widths, ChannelWidthList):
            self.channel_widths = channel_widths
        else:
            # Assume it's a filepath
            self.channel_widths = ChannelWidthList(filepath=channel_widths)

        if git:

            # Create a folder to store bathymetry objects in
            self.topos_root = Path(version_control_dir).mkdir(exist_ok=True)
            (Path(version_control_dir) / ".gitignore").write_text("*")

            # Create the subfolder for this specific bathymetry
            self.domain_dir = Path(get_domain_dir(grid, base_dir=version_control_dir))
            self.domain_dir.mkdir(
                exist_ok=True
            )  # This folder should not already exist.

            # Save the grid info there (there can only be 1 grid per bathymetry)
            self.grid_file_path = self.domain_dir / "grid.nc"
            grid.write_supergrid(self.grid_file_path)

            # Initialize the git repo
            self.repo = get_repo(self.domain_dir)

            # Set up TCM (requires that self.domain_dir exists)
            self.tcm = TopoCommandManager(self, command_registry=COMMAND_REGISTRY)
            self.apply_edit(initial_command)

        else:
            self.tcm = None
            # Apply the initial min_depth command directly without git recording
            initial_command()

    def __getitem__(self, slices):
        """
        Get a subgrid copy based on the provided slices.

        Parameters
        ----------
        slices:
            A tuple of two slices, e.g., [A:B:C, D:E:F]
            The first slice A:B:C corresponds to the j-axis (y-axis) and the second
            slice D:E:F corresponds to the i-axis (x-axis). Examples:
            topo[0:10, 0:20] or topo[:, 0:20] or topo[0:10, :].

        Returns
        -------
        topo: Topo
            A new Topo object with the same grid but with the depth and grid subsetted according to the provided slices.
        """

        new_grid = self._grid[slices]
        new_topo = Topo(
            new_grid,
            self._min_depth,
            git=self.has_version_control,
            channel_widths=copy.deepcopy(self.channel_widths),
        )  # Create new topo with the same version control setting
        if self._depth is not None:
            new_topo._depth = self._depth[slices]
        return new_topo

    @classmethod
    def from_version_control(cls, folder_path: str | Path, channel_widths=None):
        """
        Create a bathymetry object from an existing version-controlled bathymetry folder.

        Parameters
        ----------
        folder_path: str | Path
            Path to an existing bathymetry folder created by mom6_forge with version control enabled.
        channel_widths: str | Path | ChannelWidthList, optional
            Channel width constraints. Can be a filepath to load from, a ChannelWidthList object, or None.
        """

        folder_path = Path(folder_path)
        assert folder_path.exists(), f"Cannot find bathymetry folder at {folder_path}."

        grid_file_path = folder_path / "grid.nc"
        assert grid_file_path.exists(), f"Cannot find grid file at {grid_file_path}."

        grid = Grid.from_supergrid(grid_file_path)

        # Create the topo object
        topo = Topo(
            grid,
            0.0,
            version_control_dir=folder_path.parent,
            channel_widths=channel_widths,
            git=True,
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
        git=False,
        channel_widths=None,
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
        git: bool, optional
            Passed through to Topo.__init__. See Topo docstring for details.
        version_control_dir: str, optional
            Directory for version control. Default is "TopoLibrary".
        channel_widths: str | Path | ChannelWidthList, optional
            Channel width constraints. Can be a filepath to load from, a ChannelWidthList object, or None.
        """

        topo = cls(
            grid,
            min_depth,
            version_control_dir=version_control_dir,
            channel_widths=channel_widths,
            git=git,
        )
        if topo.tcm is not None:
            topo.tcm.reapply_changes()
        topo.set_depth_via_topog_file(topo_file_path, varname)
        return topo

    @classmethod
    def from_esmf_mesh(
        cls,
        mesh_path,
        radius=_DEFAULT_RADIUS,
        min_depth=0.0,
        version_control_dir="TopoLibrary",
        git=False,
    ):
        """
        Construct a topo object from the ESMF mesh + Mask, there is no depth associated with this

        Parameters
        ----------
        mesh_path: str or xr.Dataset
            Path to ESMF mesh file to read, or an already-opened Dataset.
        radius: float, optional
            Radius of the Earth (m). Default is _DEFAULT_RADIUS.
        min_depth: float, optional
            Minimum water column depth (m). Columns with shallower depths are to be masked out. Default is 0.0.
        git: bool, optional
            Passed through to Topo.__init__. See Topo docstring for details. Default is False.
        """

        supergrid, mask = SupergridBase.reconstruct_from_esmf_mesh(
            mesh_path, radius, return_mask=True
        )
        grid = Grid.from_supergrid_ds(supergrid.to_ds())
        topo = cls(grid, min_depth, version_control_dir=version_control_dir, git=git)
        topo.user_mask = mask
        return topo

    @property
    def masked_depth(self):
        """
        Computed depth array with masking applied (m). Positive below MSL.

        This is a computed property that applies land/ocean masking on-the-fly. When a user mask is set,
        returns masked depth: ocean cells enforced to at least min_depth+0.1, land cells set to _land_fillval.
        When mask is None, derives mask from raw depth (cells > min_depth are ocean).

        Note: Index-based assignment like `topo.masked_depth[j, i] = value` will not persist because
        this property is computed on-the-fly. To modify depth, use `topo.depth = new_array` or `topo.edit_depth()`.
        """

        # Enforce minimum depth criteria for ocean cells: if depth ≤ min_depth and mask is ocean, set to min_depth + 0.1 to ensure they are still treated as ocean. If mask is land and depth > min_depth, set to _land_fillval to ensure they are treated as land.
        masked_depth = xr.where(
            self.tmask,
            xr.where(
                self._depth > self._min_depth,
                self._depth,
                self._min_depth + 0.1,
            ),
            self._land_fillval,  # Land cells to _land_fillval
        )
        masked_depth = masked_depth.fillna(0)
        return masked_depth

    @property
    def src(self):
        """
        SourceBathy object representing the source bathymetry dataset sliced to the topo grid extent. This is set by set_src() when a new source bathymetry is specified, and can be accessed for any source dataset.
        """
        return self._src

    @src.setter
    def src(self, new_src):
        self._src = new_src

    @property
    def depth(self):
        return self._depth

    @property
    def has_version_control(self):
        return self.tcm is not None

    @depth.setter
    def depth(self, depth):
        """
        Set raw depth array. Preserves existing mask.

        Parameters
        ----------
        depth: np.array or xr.DataArray
            2-D Array of ocean depth (m).
        """
        if isinstance(depth, xr.DataArray):
            depth = depth.data
        else:
            assert isinstance(
                depth, np.ndarray
            ), "depth_raw must be a numpy array or xarray DataArray"

        assert depth.shape == (
            self._grid.ny,
            self._grid.nx,
        ), "Incompatible depth array shape"

        self.send_entire_depth_change_to_tcm(depth)

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
    def land_fillval(self):
        """
        Depth value assigned to land cells when mask is applied. Default is 0.0 (dry).
        """
        return self._land_fillval

    @land_fillval.setter
    def land_fillval(self, new_land_fillval):
        assert isinstance(
            new_land_fillval, (int, float)
        ), "land_fillval must be a number"
        assert (
            new_land_fillval <= self._min_depth
        ), "land_fillval should be less than or equal to min_depth to ensure land cells are not mistakenly treated as ocean"
        self._land_fillval = float(new_land_fillval)

    @property
    def user_mask(self):
        """
        Optional binary ocean/land mask. 1 if ocean, 0 if land.
        When set, the depth property applies masking: ocean cells enforced greater than min_depth+0.1, land cells set to _land_fillval.
        """
        return self._user_mask

    @user_mask.setter
    def user_mask(self, new_mask):
        """
        Set the binary ocean/land mask.

        Parameters
        ----------
        new_mask: xr.DataArray or np.ndarray or None
            Binary mask (1=ocean, 0=land) with shape (ny, nx), or None to disable masking.
        """
        if new_mask is None:
            self.clear_user_mask()  # Clear the manual mask if None is passed
            return

        if isinstance(new_mask, xr.DataArray):
            new_mask = new_mask.data
        else:
            assert isinstance(
                new_mask, np.ndarray
            ), "mask must be a numpy array, xarray DataArray, or None"

        assert new_mask.shape == (
            self._grid.ny,
            self._grid.nx,
        ), "Incompatible mask shape"

        all_indices = list(np.ndindex(new_mask.shape))
        new_values = new_mask.ravel().tolist()
        old_values = (
            self.tmask.values.ravel().tolist()
        )  # effective mask (manual or raw depth mask)

        cmd = MaskEditCommand(
            self, all_indices, new_values, old_values=old_values, message="Set mask"
        )
        self.apply_edit(cmd)

    @property
    def tmask(self):
        """
        Ocean domain mask at T grid. 1 if ocean, 0 if land.

        This is a derived mask computed from the depth property. The computation depends on whether
        a user mask has been set:

        - If _user_mask is None: Returns a mask derived from raw depth (_depth).
          Cells with depth > min_depth are marked as ocean (1), otherwise land (0).
        - If _user_mask is set: Returns a mask set by the user. The depth property will apply this mask to the raw depth, ensuring that ocean cells are at least min_depth+0.1 and land cells are set to 0.

        This design means tmask is always consistent with the current depth state, whether
        driven by raw bathymetry or by a manually applied mask.
        """

        if self._user_mask is None:
            return self._compute_tmask_from_raw_depth()
        else:
            return self.user_mask

    def _compute_tmask_from_raw_depth(self):
        """
        If no user_mask is calculated, tmask is derived directly from raw depth: ocean (1) where depth > min_depth, else land (0).
        This is a separate helper to keep the logic clear and maintainable, it is only used if the user mask is not set (which may be the case for many users who just want to set depth and have the mask be automatically derived from it).
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

    def apply_edit(self, cmd, skip_version_control=False, cmd_type=CommandType.COMMAND):
        """Apply an edit command aware of version control. If version control is enabled, the command is executed through the TopoCommandManager (TCM) to ensure it is recorded in the version history and can be undone/redone.
        If version control is disabled, the command is executed directly without recording.

        Parameters
        -----------
        cmd: Command
            The command to be applied.
        skip_version_control: bool
            If True, the command will be executed without recording in version control even if version control is enabled. This can be useful for programmatic changes that should not be part of the user-facing version history.
        cmd_type: CommandType
            The type of the command, used for version control categorization. Ignored if skip_version_control is True or if version control is disabled.
        """
        if skip_version_control or not self.has_version_control:
            cmd()
        else:
            self.tcm.execute(cmd, cmd_type=cmd_type)

    def set_src(
        self,
        bathymetry_path,
        longitude_coordinate_name,
        latitude_coordinate_name,
        vertical_coordinate_name,
        is_input_positive_below_msl=False,
        buf=0.5,
    ):
        """Set a :class:`SourceBathy` into a class object called src"""
        self.src = SourceBathy(
            self,
            Path(bathymetry_path),
            longitude_coordinate_name,
            latitude_coordinate_name,
            vertical_coordinate_name,
            is_input_positive_below_msl=is_input_positive_below_msl,
            buf=buf,
        )
        return self.src

    @property
    def stats(self):
        return self.src.stats if self.src is not None else None

    def clear_user_mask(self):
        cmd = ClearMaskCommand(
            self, message="Clear manual mask"
        )  # Resets back to reading from depth
        self.apply_edit(cmd)

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

    def send_entire_depth_change_to_tcm(self, depth, skip_version_control=False):
        """
        This function takes an entire depth change and adds it through the TopoCommandManager (TCM) or directly if skip_version_control is enabled.
        Modifies _depth, preserves mask.
        """
        # 1. Generate all affected indices (row-major order)
        all_indices = list(np.ndindex(self._depth.shape))  # list of (j, i) tuples

        # 2. Flatten the new values to match the indices
        if type(depth) == xr.DataArray:
            depth = depth.data
        new_values = depth.ravel().tolist()

        # 3. Flatten old values from raw depth
        old_values = (
            self._depth.values.ravel().tolist() if self._depth is not None else None
        )

        # 4. Build command
        depth_edit_command = DepthEditCommand(
            self, all_indices, new_values, old_values=old_values
        )

        self.apply_edit(depth_edit_command, skip_version_control=skip_version_control)

    def edit_depth(self, indices, values):
        """
        Edit depth at specific indices with version control.

        Parameters
        ----------
        indices : list of tuples
            List of (j, i) indices to modify.
        values : list or array-like
            New depth values for each index.
        """
        # Convert single index to list of indices
        if isinstance(indices, tuple):
            indices = [indices]

        # Get old values
        old_values = [self._depth.values[j, i] for j, i in indices]

        # Create and execute command
        cmd = DepthEditCommand(self, indices, values, old_values=old_values)
        self.apply_edit(cmd)

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

    def set_depth_via_topog_file(
        self, topog_file_path, varname="depth", skip_version_control=False
    ):
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
        self.send_entire_depth_change_to_tcm(
            depth, skip_version_control=skip_version_control
        )

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

    def compute_stats(self, nx_sub, ny_sub, mask_hmin):
        """Compute per-cell depth statistics by uniform sub-sampling.

        Results are stored on ``stats`` so a second call with the
        same source file returns immediately without recomputation.
        (Originally created by Frank Bryan in Fortran for NCAR/tx2_3, reimplemented in Python)

        Parameters
        ----------
        src : SourceBathy (part of class)
        nx_sub, ny_sub : int
        mask_hmin : float

        Returns
        -------
        xr.Dataset  —  ``OCN_FRAC``, ``D_mean``, ``D_min``, ``D_max``, ``D2_mean``.
        """
        assert (
            self.src is not None
        ), "Source bathymetry must be loaded to compute topo stats"
        if (
            self.stats is not None
            and self.stats.attrs.get("nx_sub") == nx_sub
            and self.stats.attrs.get("ny_sub") == ny_sub
            and self.stats.attrs.get("mask_hmin") == mask_hmin
        ):
            return self.stats

        # Compute subsampling factor and generate sub-point grid
        ds, _ = regrid_with_subsampling(
            input_dataset=self.src.ds,
            qlon=self._grid.qlon.values,
            qlat=self._grid.qlat.values,
            nx_sub=nx_sub,
            ny_sub=ny_sub,
            regridding_method="nearest_s2d",
        )

        depth_sub = ds[self.src.depth_name].values  # (ny, nx, ny_sub, nx_sub)

        is_ocean = depth_sub > mask_hmin
        ocn_frac = is_ocean.sum(axis=(-2, -1)) / (nx_sub * ny_sub)

        depth_ocean = np.where(is_ocean, depth_sub, np.nan)
        with np.errstate(all="ignore"):
            D_mean = np.nanmean(depth_ocean, axis=(-2, -1))
            D_min = np.nanmin(depth_ocean, axis=(-2, -1))
            D_max = np.nanmax(depth_ocean, axis=(-2, -1))
            D2_mean = np.nanmean(depth_ocean**2, axis=(-2, -1))

        dims = ["ny", "nx"]
        self.src.stats = xr.Dataset(
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
            },
            attrs={
                "nx_sub": nx_sub,
                "ny_sub": ny_sub,
                "mask_hmin": mask_hmin,
            },
        )
        return self.src.stats

    def set_depth_from_stats(self, statistic):
        """
        Set the topo depth to a statistic computed by compute_stats.

        Parameters
        ----------
        statistic : str
            Which depth statistic to use. Must be one of the "D_*" keys
            in self.src.stats (e.g. "mean", "min", "max").
        """

        assert (
            self.src is not None and self.src.stats is not None
        ), "Source bathymetry must be provided and must have topo stats computed, please call compute_stats first if you have not already"
        approved_list = []
        for key in self.src.stats.data_vars:
            if key.startswith("D_"):
                approved_list.append(key[2:])
        assert (
            statistic in approved_list
        ), f"Invalid statistic {statistic}, must be one of {approved_list}"

        self.send_entire_depth_change_to_tcm(self.src.stats[f"D_{statistic}"])

    def diagnose_resolution(self, radius=_DEFAULT_RADIUS):
        """
        Print resolution diagnostics comparing the model grid to a source bathymetry
        dataset, and recommend whether Cressman interpolation / stats-based masking
        is worthwhile.

        The recommendation threshold is a resolution ratio of 12x (model dx /
        dataset dx), equivalent to ~0.05° (~5 km) for GEBCO 15-arcsecond source
        data. This matches the criterion used by the tx2_3 global workflow
        (interp_smooth.f90) and might be the scale at which ocean-aware Cressman
        interpolation meaningfully improves coastal depth estimates over standard
        xesmf conservative regridding.

        Parameters
        ----------
        src : SourceBathy (provided internally by the class)
            Source bathymetry object.
        radius: float, optional
            Radius of the Earth in meters, used to convert source dataset lat/lon spacing to approximate physical spacing in meters at the domain's mid-latitude. Default is 6.371e6 m.

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
        src = self.src

        dlon_deg = float(abs(src.lon[1] - src.lon[0]))
        dlat_deg = float(abs(src.lat[1] - src.lat[0]))
        R = radius
        dataset_dx_m = haversine(src.lat[0], src.lon[0], src.lat[0], src.lon[1], R)
        dataset_dy_m = haversine(src.lat[0], src.lon[0], src.lat[1], src.lon[0], R)

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
        print(f"    dx   ~ {dataset_dx_m:.0f} m)")
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
            print(f"    likely provides benefit over xesmf regridding.")
        print(sep)
        return bool(ratio_median >= CRESSMAN_THRESHOLD)

    def generate_mask_from_stats_ocean_frac(
        self,
        mask_threshold=0.5,
    ):
        """
        Generate an ocean mask by uniform sub-sampling of the source bathymetry.

        Mirrors the algorithm in tx2_3's create_model_topo.f90. For each T-cell,
        distributes nx_sub x ny_sub interior points via bilinear interpolation of
        the Q-point corners and snaps each to the nearest source pixel. A cell is
        ocean if its ocean sub-point fraction (OCN_FRAC) meets or exceeds
        ``mask_threshold``.

        Parameters
        ----------
        mask_threshold : float, optional
            Minimum OCN_FRAC for a cell to be classified as ocean. Default 0.5.

        Returns
        -------
        xr.DataArray
            Binary ocean mask on the T-grid (1 = ocean, 0 = land),
            dims ``["ny", "nx"]``.

        Notes
        -----
        ``compute_stats`` must be called before this method. Per-cell depth
        statistics (D_mean, D_min, D_max, D2_mean) are stored on the source
        bathymetry object for use by this and other downstream methods.
        """

        assert (
            self.src is not None
        ), "Source bathymetry must be set before generating mask."
        assert (
            self.src.stats is not None
        ), f"Per-cell statistics must be computed before generating mask. Call compute_stats() first. src={self.src}"

        ocean_mask = (self.src.stats["OCN_FRAC"].values >= mask_threshold).astype(int)

        return xr.DataArray(
            ocean_mask,
            dims=["ny", "nx"],
            attrs={
                "long_name": "ocean mask from sub-sampling",
                "mask_threshold": mask_threshold,
            },
        )

    def set_from_dataset(
        self,
        bathymetry_path,
        longitude_coordinate_name,
        latitude_coordinate_name,
        vertical_coordinate_name,
        fill_channels=False,
        is_input_positive_below_msl=False,
        output_dir=Path(""),
        write_to_file=False,
        mask_method=None,
        depth_method=None,
        regridding_method="bilinear",
        **kwargs,
    ):
        """
        This is a high-level workflow function that runs multiple steps in sequence to set the bathymetry from a source dataset, with optional masking and depth method choices. It is designed to be opinionated and make recommendations based on resolution diagnostics, but users can override the defaults by specifying mask_method and depth_method.
        The workflow is as follows:
        1. Set the source dataset (this will not modify the depth or mask yet,
            it just loads the dataset and prepares it for regridding)
        2. Diagnose whether to use stats-based masking and Cressman interpolation based on resolution comparison between the source dataset and the model grid
        3. Apply a mask based on the user's choice or the resolution diagnostics recommendation (options are 'naturalearth', 'ocean_frac', 'dataset', 'manual', or None)
        4. Set the depth based on the user's choice or the resolution diagnostics recommendation (options are 'stats', 'xesmf', or None)
        5. Call ``fill_inland_lakes_and_channels()`` to finish processing the bathymetry (fill channels, apply mask, etc.)

        Parameters
        ----------
        bathymetry_path (str): Path to the netCDF file with the bathymetry
        longitude_coordinate_name (str): The name of the longitude coordinate in the bathymetry dataset at ``bathymetry_path``. For example, for GEBCO bathymetry: ``'lon'``.
        latitude_coordinate_name (str): The name of the latitude coordinate in the bathymetry dataset at ``bathymetry_path``. For example, for GEBCO bathymetry: ``'lat'``.
        vertical_coordinate_name (str): The name of the vertical coordinate (elevation) in the bathymetry dataset at ``bathymetry_path``. For example, for GEBCO bathymetry: ``'elevation'``.
        fill_channels (bool, optional): Whether to fill narrow channels in the final depth. Default is False.
        is_input_positive_below_msl (bool, optional): Whether the vertical coordinate in the source dataset is positive down (e.g. depth) rather than positive up (e.g. elevation). Default is False (positive up).
        output_dir (str or Path, optional): Directory to save intermediate files if needed (e.g. for Cressman interpolation). Default is current directory.
        write_to_file (bool, optional): Whether to write intermediate files to disk (e.g. for Cressman interpolation). Default is False.
        mask_method (str or None, optional): Method to use for masking land vs ocean. Options are 'naturalearth' (use natural earth land polygons), 'ocean_frac' (use ocean fraction from sub-sampling stats), 'dataset' (derive mask directly from raw depth from dataset), 'manual' (use a user-provided mask set via the user_mask property), or None to automatically choose based on resolution diagnostics. Default is None.
        depth_method (str or None, optional): Method to use for setting depth. Options are 'stats' (use a statistic from the sub-sampling stats), 'xesmf' (do a direct xesmf regrid of the source dataset depth), or None to automatically choose based on resolution diagnostics. Default is None.
        regridding_method (str, optional): The xESMF regridding method to use when depth_method is 'xesmf' (or when xesmf is chosen automatically). Default is 'bilinear'.
        **kwargs: Additional keyword arguments needed for specific mask or depth methods. For example, if mask_method is 'ocean_frac', then kwargs should include 'nx_sub', 'ny_sub', and 'mask_hmin' for the sub-sampling stats calculation.

        """
        # Set source
        self.set_src(
            bathymetry_path,
            longitude_coordinate_name,
            latitude_coordinate_name,
            vertical_coordinate_name,
            is_input_positive_below_msl,
        )

        # Diagnose wether to use stats-based masking and Cressman interpolation based on resolution comparison between the source dataset and the model grid
        use_stats_depth = self.diagnose_resolution()

        # Check necessary attributes for different conditions.
        # Note ocean_frac, stats-based masking, and stats-based depth all rely on the same sub-sampling stats, so we check for those together.
        if (
            (mask_method == "ocean_frac")
            or (mask_method is None and use_stats_depth)
            or (depth_method == "stats")
            or (depth_method is None and use_stats_depth)
        ):
            if kwargs.get("mask_hmin") is None:
                print(
                    "Masking depth threshold (mask_hmin) not provided in kwargs, defaulting to 0 m"
                )
                mask_hmin = 0.0
            else:
                mask_hmin = kwargs["mask_hmin"]

            if kwargs.get("nx_sub") is None or kwargs.get("ny_sub") is None:
                print(
                    "Sub-sampling factors (nx_sub, ny_sub) not provided in kwargs, computing based on source dataset and model grid resolution"
                )
                src_nj, src_ni = self.src.depth.shape
                ocn_ni, ocn_nj = self._grid.nx, self._grid.ny
                ny_sub, nx_sub = compute_subsampling_factor(
                    src_nj, src_ni, ocn_nj, ocn_ni
                )
            else:
                ny_sub = kwargs["ny_sub"]
                nx_sub = kwargs["nx_sub"]

        # Apply a mask if specified
        if mask_method is not None:
            if mask_method == "naturalearth":
                self.user_mask = self.generate_mask_from_naturalearth()
            elif mask_method == "ocean_frac":
                self.compute_stats(
                    nx_sub=nx_sub,
                    ny_sub=ny_sub,
                    mask_hmin=mask_hmin,
                )
                self.user_mask = self.generate_mask_from_stats_ocean_frac()
            elif mask_method == "dataset":
                self.clear_user_mask()  # ensure no user mask is set so that the mask is derived from the raw depth, which is directly from the dataset
            elif mask_method == "manual":
                assert (
                    self.user_mask is not None
                ), "Mask method set to 'manual' but no user mask has been set. Please set the user mask before calling set_from_dataset with mask_method='manual'"
            else:
                raise ValueError(
                    f"Invalid mask option {mask_method!r}, must be one of {VALID_MASK_METHODS}"
                )
        else:
            if use_stats_depth:
                print(
                    "Resolution diagnostics recommend using stats-based masking, which we will set because no mask option was specified"
                )
                self.compute_stats(
                    nx_sub=nx_sub,
                    ny_sub=ny_sub,
                    mask_hmin=mask_hmin,
                )
                self.user_mask = self.generate_mask_from_stats_ocean_frac()
            else:
                print(
                    "Resolution diagnostics recommend not using stats-based masking, so we'll use the natural earth mask"
                )
                self.user_mask = self.generate_mask_from_naturalearth()

        if depth_method is not None:
            if depth_method == "stats":
                if not use_stats_depth:
                    print(
                        "Resolution diagnostics recommend not using stats-based masking, but stats-based depth was requested"
                    )
                self.compute_stats(
                    nx_sub=nx_sub,
                    ny_sub=ny_sub,
                    mask_hmin=mask_hmin,
                )
                self.set_depth_from_stats(statistic="mean")
            elif depth_method == "cressman":
                self.direct_cressman_interp(
                    weights_path=Path(output_dir) / "cressman_weights.nc"
                )
            elif depth_method == "xesmf":
                if use_stats_depth:
                    print(
                        "Resolution diagnostics recommend using stats-based masking, but xesmf-based depth was requested, so we will do a direct xesmf regrid anyway because it was explicitly requested, but be aware that this may lead to significant land contamination of coastal depth estimates"
                    )
                self.set_depth_from_xesmf(
                    output_dir=output_dir,
                    write_to_file=write_to_file,
                    regridding_method=regridding_method,
                )
            else:
                raise ValueError(
                    f"Invalid depth option {depth_method!r}, must be one of {VALID_DEPTH_METHODS}"
                )
        else:
            if not use_stats_depth:
                print(
                    "Resolution diagnostics recommend not using stats-based depth method, so we'll do a direct xesmf regrid for depth because no depth option was specified"
                )
                self.set_depth_from_xesmf(
                    output_dir=output_dir,
                    write_to_file=write_to_file,
                    regridding_method=regridding_method,
                )
            else:
                print(
                    "Resolution diagnostics recommend using stats-based method, which we will set for depth as cressman method because no depth option was specified"
                )
                self.direct_cressman_interp(
                    weights_path=Path(output_dir) / "cressman_weights.nc"
                )

        # Tidy the dataset (fill channels, is_input_positive_below_msl, etc...)
        if fill_channels:
            self.fill_inland_lakes_and_channels()

        print(
            "Warning! This was an opionated workflow function that ran multiple steps in sequence. Please edit the mask manually if need be (Some depth methods, like cressman, are mask-aware and may need to be rerun)! "
        )

    def set_depth_from_xesmf(
        self,
        output_dir=Path(""),
        write_to_file=False,
        regridding_method="bilinear",
    ):
        """
        Regrid the source bathymetry onto the model grid using xESMF.

        This code was originally written by Ashley Barnes in regional_mom6
        (https://github.com/COSIMA/regional-mom6) and adapted for this package.

        Requires that ``set_src`` has been called first (or that the source dataset
        has been set via ``set_from_dataset``).

        Arguments:
            output_dir (str or Path, optional): Directory where intermediate files are
                written when ``write_to_file=True``. Default: current directory.
            write_to_file (Optional[bool]): Whether to write the source and unfinished
                destination grids to netCDF before regridding. Default: ``False``.
            regridding_method (Optional[str]): The xESMF regridding method to use.
                Default: ``'bilinear'``.

        """
        print(
            """**NOTE**
            If bathymetry setup fails (e.g. kernel crashes), restart the kernel and edit this cell.
            Call ``[topo_object_name].mpi_set_depth_from_xesmf()`` instead. Follow the given instructions for using mpi
            and ESMF_Regrid outside of a python environment. This breaks up the process, so be sure to call
            ``[topo_object_name].fill_inland_lakes_and_channels()`` after regridding with mpi."""
        )
        output_dir = Path(output_dir)
        self.src_bathymetry_dataset = self.src.ds
        self.destination_bathymetry = self._grid.get_esmf_ready_tracer_ds()
        self.destination_bathymetry["depth"] = xr.zeros_like(
            self.destination_bathymetry.tarea
        )
        self.destination_bathymetry.depth.attrs["units"] = "meters"
        self.destination_bathymetry.depth.attrs["coordinates"] = "lon lat"
        if write_to_file:
            self.src_bathymetry_dataset.to_netcdf(output_dir / "bathymetry_original.nc")
            self.destination_bathymetry.to_netcdf(
                output_dir / "bathymetry_unfinished.nc"
            )

        self.depth = regrid_dataset_via_xesmf(
            input_dataset=self.src_bathymetry_dataset,
            output_dataset=self.destination_bathymetry,
            regridding_method=regridding_method,
            write_to_file=write_to_file,
            output_path=output_dir / "bathymetry_unfinished.nc",
        )["depth"]
        if write_to_file:
            self.write_topo(
                output_dir / "bathymetry_unfinished.nc"
            )  # This is called unfinished because the regridding is not fully complete until the one-cell channels are filled

    def mpi_set_depth_from_xesmf(
        self,
        *,
        bathymetry_path,
        longitude_coordinate_name,
        latitude_coordinate_name,
        vertical_coordinate_name,
        is_input_positive_below_msl=False,
        output_dir=Path(""),
        verbose=True,
    ):
        """
        Prepare input files for MPI-parallel bathymetry regridding with ESMF_Regrid.

        Writes ``bathymetry_original.nc`` (source) and ``bathymetry_unfinished.nc``
        (destination grid shell) to ``output_dir``. The user then runs ``ESMF_Regrid``
        externally with MPI, and finally calls ``fill_inland_lakes_and_channels()`` to
        complete post-processing. Use this instead of ``set_depth_from_xesmf`` when the
        domain is too large to regrid within a single Python process.

        Arguments:
            bathymetry_path (str): Path to the netCDF file with the source bathymetry.
            longitude_coordinate_name (str): Name of the longitude coordinate in the
                source dataset (e.g. ``'lon'`` for GEBCO).
            latitude_coordinate_name (str): Name of the latitude coordinate in the
                source dataset (e.g. ``'lat'`` for GEBCO).
            vertical_coordinate_name (str): Name of the vertical/elevation coordinate in
                the source dataset (e.g. ``'elevation'`` for GEBCO).
            is_input_positive_below_msl (bool, optional): Set ``True`` if the source
                vertical coordinate is positive downwards (depth convention). Default: ``False``.
            output_dir (str or Path, optional): Directory where the two netCDF files are
                written. Default: current directory.
            verbose (bool, optional): Print step-by-step MPI regridding instructions.
                Default: ``True``.

        """
        if verbose:
            print(f"""
            *MANUAL REGRIDDING INSTRUCTIONS*

            Calling `[object_name].mpi_set_depth_from_xesmf` sets up the files necessary for regridding
            the bathymetry using mpirun and ESMF_Regrid. See below for the step-by-step instructions:

            1. There should be two files: `bathymetry_original.nc` and `bathymetry_unfinished.nc` located at
            {output_dir}.

            2. Open a terminal and change to this directory (e.g. `cd {output_dir}`).

            3. Request appropriate computational resources (see example script below), and run the command:

            `mpirun -np NUMBER_OF_CPUS ESMF_Regrid -s bathymetry_original.nc -d bathymetry_unfinished.nc -m bilinear --src_var depth --dst_var depth --netcdf4 --src_regional --dst_regional`

            4. Run Topo_object.fill_inland_lakes_and_channels() to finish processing the bathymetry.

            Example PBS script using NCAR's Casper Machine: https://gist.github.com/AidanJanney/911290acaef62107f8e2d4ccef9d09be

            For additional details see: https://xesmf.readthedocs.io/en/latest/large_problems_on_HPC.html
            """)

        output_dir = Path(output_dir)
        self.set_src(
            bathymetry_path=bathymetry_path,
            longitude_coordinate_name=longitude_coordinate_name,
            latitude_coordinate_name=latitude_coordinate_name,
            vertical_coordinate_name=vertical_coordinate_name,
            is_input_positive_below_msl=is_input_positive_below_msl,
        )
        self.src_bathymetry_dataset = self.src.ds
        self.destination_bathymetry = self._grid.get_esmf_ready_tracer_ds()
        self.destination_bathymetry["depth"] = xr.zeros_like(
            self.destination_bathymetry.tarea
        )
        self.destination_bathymetry.depth.attrs["units"] = "meters"
        self.destination_bathymetry.depth.attrs["coordinates"] = "lon lat"
        self.src_bathymetry_dataset.to_netcdf(output_dir / "bathymetry_original.nc")
        self.destination_bathymetry.to_netcdf(output_dir / "bathymetry_unfinished.nc")

        print(
            "Configuration complete. Ready for regridding with MPI. See documentation for more details."
        )

    def direct_cressman_interp(
        self,
        smooth_scl=2.0,
        cressman_exp=2.0,
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

        Weights are computed by :func:`~mom6_forge.mapping.compute_cressman_weights`,
        saved to an ESMF-compatible netCDF, and applied through ``xe.Regridder`` —
        all orchestrated by :func:`~mom6_forge.mapping.regrid_dataset_via_cressman`.
        Cells that receive no source coverage are filled by iterative neighbour
        averaging (up to 100 passes).

        Parameters
        ----------
        smooth_scl : float
            Smoothing scale multiplier for the Cressman radius. Default ``2.0``.
        cressman_exp : float
            Exponent for the Cressman weight function. Default ``2.0``.
        weights_path : str or Path or None
            Where to save the ESMF weights netCDF. If ``None``, a file named
            ``cressman_weights.nc`` is written next to the bathymetry file.
        """
        if weights_path is None:
            weights_path = self.src.path.parent / "cressman_weights.nc"

        # --- Regrid via mapping module (weights → file → cressman Regridder) ---

        dst_ds = self._grid.get_esmf_ready_tracer_ds()
        dst_ds["area"] = self._grid.tarea
        dst_ds["mask"] = self.tmask

        depth_dst, unfilled = regrid_dataset_via_cressman(
            self.src.ds,
            dst_ds,
            smooth_scl=smooth_scl,
            cressman_exp=cressman_exp,
            weights_path=weights_path,
        )

        depth_arr = iterative_fill(depth_dst["depth"].values, unfilled, self.tmask)

        self.send_entire_depth_change_to_tcm(
            xr.DataArray(
                depth_arr.astype(float), dims=["ny", "nx"], attrs={"units": "m"}
            )
        )

    def fill_inland_lakes_and_channels(self):
        """
        Fill in one-cell-wide channels and inland lakes. This removes more narrow inlets, but can also connect extra islands to land.
        """
        changed = True  ## keeps track of whether solution has converged or not

        forward = True  ## only useful for iterating through diagonal channel removal. Means iteration goes SW -> NE
        ocean_mask = self.tmask
        land_mask = np.abs(ocean_mask - 1)

        while changed == True:
            ## First fill in all lakes.
            ## scipy.ndimage.binary_fill_holes fills holes made of 0's within a field of 1's
            land_mask[:, :] = binary_fill_holes(land_mask.data)
            ## Get the ocean mask instead of land- easier to remove channels this way
            ocean_mask = np.abs(land_mask - 1)

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

        # Reset the mask through Mask Edit
        self.user_mask = ocean_mask

    def erase_selected_basin(self, i, j):
        label = self.basintmask.data[j, i]
        affected = np.where(self.basintmask.data == label)
        indices = list(zip(affected[0], affected[1]))
        if not indices:
            return
        old_values = [self.tmask.data[jj, ii] for jj, ii in indices]
        new_values = [0] * len(indices)
        cmd = MaskEditCommand(self, indices, new_values, old_values=old_values)
        self.apply_edit(cmd)

    def erase_disconnected_basin(self, i, j):
        label = self.basintmask.data[j, i]
        affected = np.where(self.basintmask.data != label)
        indices = list(zip(affected[0], affected[1]))
        if not indices:
            return
        old_values = [self.tmask.data[jj, ii] for jj, ii in indices]
        new_values = [0] * len(indices)
        cmd = MaskEditCommand(self, indices, new_values, old_values=old_values)
        self.apply_edit(cmd)

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
            old_vals.extend(self.depth[j, :].values)
            new_vals.extend((self.depth[j, :] + ridge_height_mapped).values)
        depth_edit_command = DepthEditCommand(
            self, affected_indices, new_vals, old_values=old_vals
        )
        self.apply_edit(depth_edit_command)

    def generate_mask_from_landfrac_file(
        self,
        landfrac_filepath,
        landfrac_name,
        xcoord_name,
        ycoord_name,
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
        cutoff_frac : float
            Cells with landfrac > cutoff_frac are deemed land cells.
        method : str
            Mapping method for determining the ocean mask (lnd -> ocn)
        """

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
        assert xcoord_name in ds, f"Couldn't find {xcoord_name} in {landfrac_filepath}"
        assert isinstance(ycoord_name, str), "ycoord_name must be a string"
        assert ycoord_name in ds, f"Couldn't find {ycoord_name} in {landfrac_filepath}"
        assert isinstance(
            cutoff_frac, float
        ), f"cutoff_frac={cutoff_frac} must be a float"
        assert (
            0.0 <= cutoff_frac <= 1.0
        ), f"cutoff_frac={cutoff_frac} must be between 0 and 1"

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

        regridder = xe.Regridder(
            ds, ds_mapped, method, periodic=self._grid.supergrid.is_cyclic_x
        )
        mask_mapped = regridder(ds.landfrac)

        # Convert land fraction to binary mask (1=ocean, 0=land)
        binary_mask = (mask_mapped <= cutoff_frac).astype(int)

        # Use MaskEditCommand for version-controlled mask application
        # Generate all affected indices
        all_indices = list(np.ndindex(binary_mask.shape))  # list of (j, i) tuples
        new_values = binary_mask.values.ravel().tolist()

        # Get old values (current mask or 0 if no mask set)
        old_mask = self.tmask
        old_values = (
            old_mask.values.ravel().tolist()
            if isinstance(old_mask, xr.DataArray)
            else old_mask.ravel().tolist()
        )

        # Build and execute command
        mask_edit_command = MaskEditCommand(
            self,
            all_indices,
            new_values,
            old_values=old_values,
            message="Apply Land Fraction Mask",
        )

        self.apply_edit(mask_edit_command)

        # legacy code set the depth of land cells to depth_fillval, but now that is handled in the depth property.

    def generate_mask_from_naturalearth(
        self,
        resolution="10",
        version="v5_1_2",
    ):
        """
        Generate and apply a land mask using regionmask's Natural Earth dataset. (Recommended by Fred Castruccio)

        Parameters
        ----------
        resolution : str
            Resolution of the Natural Earth land dataset. One of "10", "50", "110" (minutes).
        version : str
            Version of the Natural Earth dataset in regionmask. e.g. "v5_1_2".
        """

        region_attr = f"natural_earth_{version}"
        land_attr = f"land_{resolution}"

        assert hasattr(
            regionmask.defined_regions, region_attr
        ), f"regionmask has no Natural Earth version '{version}'. Available: {dir(regionmask.defined_regions)}"

        ne = getattr(regionmask.defined_regions, region_attr)

        assert hasattr(
            ne, land_attr
        ), f"Natural Earth '{version}' has no resolution '{resolution}'. Available: {dir(ne)}"

        land = getattr(ne, land_attr)

        raw_mask = land.mask(self._grid.tlon.values, self._grid.tlat.values)

        # NaN = ocean (not masked by any land region), non-NaN = land
        binary_mask = raw_mask.isnull().astype(int)  # 1 = ocean, 0 = land

        return binary_mask.values

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

        # Write raw depth to file
        ds["depth_raw"] = xr.DataArray(
            self.depth.data,
            dims=["ny", "nx"],
            attrs={
                "long_name": "t-grid cell depth (raw, before masking)",
                "units": "m",
            },
        )

        # Write masked depth (from property) to file
        ds["depth"] = xr.DataArray(
            self.masked_depth.data,
            dims=["ny", "nx"],
            attrs={"long_name": "t-grid cell masked depth", "units": "m"},
        )

        return ds

    def write_topo(self, file_path, title=None):
        """
        Write the TOPO_FILE (bathymetry file) in netcdf format. The written file is
        to be read in by MOM6 during runtime.

        Parameters
        ----------
        file_path: str
            Path to TOPO_FILE to be written.
        title: str, optional
            File title.

        Note
        ----
        If channel_widths is not empty, remember to also write those constraints using
        channel_widths.write(channel_file_path).
        """

        if self.channel_widths.get_all():
            print(
                "Note: Channel widths are defined. Remember to write them with "
                "channel_widths.write(filepath)"
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

    def write_ww3_input(self, file_dir, grid_alias):
        """
        Write the text-based WW3 input files ww3_grid.inp, [grid_alias]_x.inp, [grid_alias]_y.inp,
        [grid_alias]_mapsta.inp, [grid_alias]_bottom.inp, which are to be read by the WW3
        mod_def creator before runtime to generate the WW3 grid files.

        Parameters
        ----------
        file_dir: str
            Directory to write the WW3 input files to.
        grid_alias: str
            The alias for the grid, which will be used in the file names of the WW3 input files.
        """

        assert (
            "degrees" in self._grid.tlat.units and "degrees" in self._grid.tlon.units
        ), "Unsupported coord"

        file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        nx = self._grid.nx
        ny = self._grid.ny

        def _write_rows(filename, fmt_cell, sep):
            """Write the nx*ny grid values to a WW3 text input file, southernmost
            row (j=0) first to match IDLA=1 in ww3_grid.inp."""
            with open(file_dir / filename, "w") as f:
                for j in range(ny):
                    f.write(sep.join(fmt_cell(j, i) for i in range(nx)) + "\n")

        tlon = self._grid.tlon.data  # (ny, nx), degrees
        tlat = self._grid.tlat.data  # (ny, nx), degrees
        # Define ocean cells from the land/sea mask so the depth and status files
        # stay consistent even if the mask has been edited.
        tmask = self.tmask.data  # (ny, nx), 1=ocean, 0=land
        depth_m = self.masked_depth.data

        x_file = f"{grid_alias}_x.inp"
        y_file = f"{grid_alias}_y.inp"
        bottom_file = f"{grid_alias}_bottom.inp"
        mapsta_file = f"{grid_alias}_mapsta.inp"

        # --- x/y coordinate files (longitudes/latitudes in degrees) ---
        _write_rows(x_file, lambda j, i: f"{tlon[j, i]:15.8f}", sep="")
        _write_rows(y_file, lambda j, i: f"{tlat[j, i]:15.8f}", sep="")

        # --- bottom depth file (positive depth in meters for ocean) ---
        # WW3 stores seabed as elevation ZB (negative-down); DW = WLV - ZB.
        # The preprocessor computes ZBIN = SBF * file_values, then flags a
        # cell as sea when ZBIN <= ZLIM. We write positive depths in meters
        # and set SBF=-1.0 in ww3_grid.inp so ZBIN comes out as the correct
        # negative-down elevation. Matches the convention in WW3 regtest
        # ww3_tp2.5 (regtests/ww3_tp2.5/input/depth.361x361.IDLA1.dat).
        _write_rows(bottom_file, lambda j, i: f"{depth_m[j, i]:.8f}", sep=" ")

        # --- map status file (1=ocean, 0=land) ---
        # TODO: WW3 also supports mapsta codes 2 (active boundary), 3 (excluded),
        # and negative values (ice). Extend when nested/boundary-forced runs are needed.
        _write_rows(mapsta_file, lambda j, i: str(int(tmask[j, i])), sep=" ")

        # --- Write ww3_grid.inp ---
        # Use IDLA=1 (bottom-to-top) and IDFM=1 (free format) to match the
        # row ordering used above (j=0 is the southernmost row).
        with open(file_dir / "ww3_grid.inp", "w") as f:
            f.write(
                "$ -------------------------------------------------------------------- $\n"
                "$ WAVEWATCH III Grid preprocessor input file                           $\n"
                "$ -------------------------------------------------------------------- $\n"
                "$\n"
                "$ Grid name (C*30, in quotes)\n"
                "$\n"
            )
            grid_name = grid_alias.ljust(30)[:30]
            f.write(f"  '{grid_name}'\n")
            # TODO: frequency/direction counts, model flags, and timesteps below
            # are copied from the ww3a reference grid. Parameterize when this
            # method is used for grids with different resolution or physics.
            nk = 25  # number of frequencies (wavenumbers)
            nth = 24  # number of directions
            f.write(
                "$\n"
                "$ Frequency increment factor and first frequency (Hz) ---------------- $\n"
                "$ number of frequencies (wavenumbers) and directions, relative offset\n"
                "$ of first direction in terms of the directional increment [-0.5,0.5].\n"
                "$\n"
                f"  1.1  0.04118  {nk}  {nth}  0.0\n"
                "$\n"
                "$ Set model flags ---------------------------------------------------- $\n"
                "$  - FLDRY         Dry run (input/output only, no calculation).\n"
                "$  - FLCX, FLCY    Activate X and Y component of propagation.\n"
                "$  - FLCTH, FLCK   Activate direction and wavenumber shifts.\n"
                "$  - FLSOU         Activate source terms.\n"
                "  F  T  T  T  F  T \n"
                "$\n"
                "$ Set time steps ----------------------------------------------------- $\n"
                "$ - Time step information (this information is always read)\n"
                "$     maximum global time step, maximum CFL time step for x-y and\n"
                "$     k-theta, minimum source term time step (all in seconds).\n"
                "$\n"
                "  600.00  300.00  300.00   30.00\n"
                "$\n"
                "$ Start of namelist input section ------------------------------------ $\n"
                "$\n"
                "&OUTS\n"
                f"  E3D = 1, I1E3D = 1, I2E3D = {nk}\n"
                "/\n"
                "\n"
                "END OF NAMELISTS\n"
                "$\n"
                "$ Define grid -------------------------------------------------------- $\n"
                "$\n"
            )
            closure = "SMPL" if self._grid.supergrid.is_cyclic_x else "NONE"
            f.write(f"  'CURV'  T  '{closure}'\n")
            f.write(f"  {nx}  {ny}\n")
            f.write(f"  21 1.0 0.0 1 1 '(....)' 'NAME' '{x_file}'\n")
            f.write(f"  22 1.0 0.0 1 1 '(....)' 'NAME' '{y_file}'\n")
            f.write(
                f"  -0.1 {self._min_depth:.2f} 23 -1. 1 1 '(....)' 'NAME' '{bottom_file}'\n"
            )
            f.write(f"  24 1 1 '(....)' 'NAME' '{mapsta_file}'\n")
            f.write(
                "$\n"
                "$  Close list by defining line with 0 points (mandatory)\n"
                "$\n"
                "    0.  0.  0.  0.  0  \n"
                "$ -------------------------------------------------------------------- $\n"
                "$ End of input file                                                    $\n"
                "$ -------------------------------------------------------------------- $\n"
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

        self._grid.supergrid.to_esmf_mesh(file_path, mask=self.tmask.data, title=title)
        self.mesh_path = file_path
