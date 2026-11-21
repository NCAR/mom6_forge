import hashlib
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from pathlib import Path

SIZE_THRESHOLD = 10_000  # cells; above this, serialize to NetCDF instead of inline JSON
LARGE_EDITS_DIR = "large_edits_optimization"


class EditCommand(ABC):
    @abstractmethod
    def __call__(self):
        """Execute the command. Derived classes should implement this method to perform the command's action."""
        pass

    @abstractmethod
    def serialize(self) -> dict:
        """Serialize the command to a dictionary format suitable for JSON encoding.

        Returns a dictionary with the command type and necessary data.

        Notes: Derived classes should override this method to include their specific attributes.
        Output should be compatible with corresponding deserialize method."""
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict):
        """Deserialize the command from a dictionary format.

        Parameters: Dictionary containing serialized command data.
        Returns an instance of the command class.

        Notes: Derived classes should override this method to reconstruct their specific attributes.
        The input dictionary should match the output of the corresponding serialize method.
        """
        pass

    @classmethod
    @abstractmethod
    def reverse_deserialize(cls, data: dict):
        """Deserialize a command for undoing purposes.

        This method creates a command instance that can revert the changes made by the original command.

        Parameters: Dictionary containing serialized command data.
        Returns an instance of the command class configured for undoing the original action.
        """
        raise NotImplementedError(
            "reverse_deserialize is not implemented for this command type. It probably doesn't make sense!"
        )


# Registry for type <-> class mapping
COMMAND_REGISTRY = {}


def register_command(cls):
    COMMAND_REGISTRY[cls.__name__] = cls
    return cls


def to_native(val):
    # Converts numpy scalars to Python native types, otherwise returns as-is.
    return val.item() if isinstance(val, (np.generic,)) else val


def to_native_tuple(t):
    # Converts a tuple of numpy ints to native ints
    return tuple(int(x) for x in t)


class _CellEditCommand(EditCommand):
    """An edit that sets new values on a list of (j, i) cells. Subclasses say where."""

    default_message = "Cell Edit"
    nc_dtype = np.float64  # dtype of the values in a large-edit NetCDF file

    def __init__(
        self, topo, affected_indices, new_values, old_values=None, message=None
    ):
        self._topo = topo
        # Convert indices and values to native types for consistency and serialization
        self.affected_indices = [to_native_tuple(idx) for idx in affected_indices]
        self.new_values = [to_native(v) for v in new_values]
        self.old_values = (
            [to_native(v) for v in old_values] if old_values is not None else None
        )
        self.message = message if message is not None else self.default_message

    @abstractmethod
    def _get_value(self, j, i):
        pass

    @abstractmethod
    def _set_value(self, j, i, value):
        pass

    def __call__(self):
        if self.old_values is None:
            self.old_values = [
                to_native(self._get_value(j, i)) for j, i in self.affected_indices
            ]
        for (j, i), value in zip(self.affected_indices, self.new_values):
            self._set_value(j, i, value)

    def _get_large_edits_dir(self):
        if not self._topo.has_version_control:
            return None
        d = self._topo.domain_dir / LARGE_EDITS_DIR
        d.mkdir(exist_ok=True)
        return d

    def serialize(self):
        """Inline JSON for small edits; a NetCDF file under the domain dir for large ones."""
        data = {"type": self.__class__.__name__}
        large_edits_dir = (
            self._get_large_edits_dir()
            if len(self.affected_indices) > SIZE_THRESHOLD
            else None
        )
        if large_edits_dir is None:
            data["affected_indices"] = list(self.affected_indices)
            data["new_values"] = list(self.new_values)
            data["old_values"] = (
                list(self.old_values) if self.old_values is not None else None
            )
            return data

        idx_arr = np.array(self.affected_indices, dtype=np.int32)
        nv_arr = np.array(self.new_values, dtype=self.nc_dtype)
        ov_arr = np.array(
            self.old_values if self.old_values is not None else [],
            dtype=self.nc_dtype,
        )
        h = hashlib.sha256(
            idx_arr.tobytes() + nv_arr.tobytes() + ov_arr.tobytes()
        ).hexdigest()[:16]
        nc_path = large_edits_dir / f"edit_{h}.nc"

        if not nc_path.exists():
            xr.Dataset(
                {
                    "indices": xr.DataArray(idx_arr, dims=["n", "coord"]),
                    "new_values": xr.DataArray(nv_arr, dims=["n"]),
                    "old_values": xr.DataArray(ov_arr, dims=["n"]),
                }
            ).to_netcdf(nc_path)

        data["nc_filename"] = nc_path.name
        return data

    @classmethod
    def deserialize(cls, data):
        return cls._factory(data, "new_values", "old_values")

    @classmethod
    def reverse_deserialize(cls, data):
        return cls._factory(data, "old_values", "new_values")  # swapped

    @classmethod
    def _factory(cls, data, new_key, old_key):
        """topo -> command, taking the new and old values from the given keys."""

        def factory(topo):
            if "nc_filename" not in data:
                return cls(
                    topo,
                    affected_indices=[tuple(idx) for idx in data["affected_indices"]],
                    new_values=data[new_key],
                    old_values=data[old_key],
                )
            nc_path = topo.domain_dir / LARGE_EDITS_DIR / data["nc_filename"]
            with xr.open_dataset(nc_path) as ds:
                return cls(
                    topo,
                    affected_indices=[
                        tuple(row) for row in ds["indices"].values.tolist()
                    ],
                    new_values=ds[new_key].values.tolist(),
                    old_values=ds[old_key].values.tolist(),
                )

        return factory


@register_command
class DepthEditCommand(_CellEditCommand):
    """Set the raw depth of one or more cells."""

    default_message = "Depth Edit"

    def _get_value(self, j, i):
        return self._topo._depth.data[j, i]

    def _set_value(self, j, i, value):
        self._topo._depth.data[j, i] = value


@register_command
class MaskEditCommand(_CellEditCommand):
    """Set the binary ocean/land manual mask (1 ocean, 0 land) of one or more cells."""

    default_message = "Mask Edit"
    nc_dtype = np.int8
    creates_mask = (
        None  # set on the first run: True if this edit started the manual mask
    )

    def _get_value(self, j, i):
        return self._topo.tmask.data[j, i]

    def _set_value(self, j, i, value):
        assert value in [0, 1], f"Mask value must be 0 (land) or 1 (ocean), got {value}"
        self._topo._user_mask.data[j, i] = value

    def __call__(self):
        """If there is no manual mask yet, start one from the depth-derived mask."""
        if self.creates_mask is None:
            self.creates_mask = self._topo._user_mask is None
        if self._topo._user_mask is None:
            print(
                "The manual mask is not initialized. Initializing it now from the depth raw mask"
            )
            self._topo._user_mask = self._topo._compute_tmask_from_raw_depth()
        super().__call__()

    def serialize(self):
        return {**super().serialize(), "creates_mask": self.creates_mask}

    @classmethod
    def reverse_deserialize(cls, data):
        """Undoing the edit that started the manual mask clears the mask again."""
        if data.get("creates_mask"):
            return lambda topo: ClearMaskCommand(topo)
        return super().reverse_deserialize(data)


@register_command
class MinDepthEditCommand(EditCommand):
    """Define any edit that affects a single scalar attribute of an object"""

    def __init__(self, topo, attr, new_value, old_value=None, message="Min Depth Edit"):
        self._topo = topo
        self.attr = attr
        self.new_value = to_native(new_value)
        self.old_value = to_native(old_value) if old_value is not None else None
        self.message = message

    def __call__(self):
        if self.old_value is None:
            self.old_value = to_native(getattr(self._topo, self.attr))
        setattr(self._topo, self.attr, self.new_value)

    def serialize(self):
        return {
            "type": self.__class__.__name__,
            "attr": self.attr,
            "new_value": to_native(self.new_value),
            "old_value": to_native(self.old_value),
        }

    @classmethod
    def deserialize(cls, data):
        return lambda topo: cls(
            topo,
            attr=data["attr"],
            new_value=data["new_value"],
            old_value=data["old_value"],
        )

    @classmethod
    def reverse_deserialize(cls, data):
        return lambda topo: cls(
            topo,
            attr=data["attr"],
            new_value=data["old_value"],
            old_value=data["new_value"],
        )


@register_command
class ClearMaskCommand(EditCommand):
    """Clears the manual mask, reverting to depth_raw_mask derived mask."""

    def __init__(self, topo, message="Clear Manual Mask"):
        self._topo = topo
        self.old_mask = topo._user_mask
        self.message = message

    def __call__(self):
        self._topo._user_mask = None

    def serialize(self):
        return {
            "type": self.__class__.__name__,
            "old_mask": (
                self.old_mask.values.tolist() if self.old_mask is not None else None
            ),
        }

    @classmethod
    def deserialize(cls, data):
        def factory(topo):
            old_mask = data["old_mask"]
            if old_mask is not None:
                old_mask = xr.DataArray(
                    np.array(old_mask),
                    dims=["ny", "nx"],
                    attrs={"name": "binary ocean/land mask"},
                )
            return cls(topo, old_mask=old_mask)

        return factory

    @classmethod
    def reverse_deserialize(cls, data):
        """Undo a clear by restoring the old mask via MaskEditCommand."""

        def factory(topo):
            old_mask = data["old_mask"]
            if old_mask is None:
                return cls(topo)  # was already None, just clear again
            mask_array = np.array(old_mask)
            all_indices = list(np.ndindex(mask_array.shape))
            new_values = mask_array.ravel().tolist()
            old_values = None  # We don't need old values for undoing a clear, since the command will just restore the old mask as-is
            return MaskEditCommand(
                topo,
                all_indices,
                new_values,
                old_values=old_values,
                message="Restore mask (undo clear)",
            )

        return factory
