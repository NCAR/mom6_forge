import numpy as np
from abc import ABC, abstractmethod


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


@register_command
class DepthEditCommand(EditCommand):
    """Define any edit that affects one or more elements of an array"""

    def __init__(
        self, topo, affected_indices, new_values, old_values=None, message="Depth Edit"
    ):
        self._topo = topo
        # Convert indices and values to native types for consistency and serialization
        self.affected_indices = [to_native_tuple(idx) for idx in affected_indices]
        self.new_values = [to_native(v) for v in new_values]
        self.old_values = (
            [to_native(v) for v in old_values] if old_values is not None else None
        )
        self.message = message

    def _get_value(self, j, i):
        return self._topo.depth.data[j, i]

    def _set_value(self, j, i, value):
        self._topo.depth.data[j, i] = value

    def __call__(self):
        if self.old_values is None:
            self.old_values = [
                to_native(self._get_value(j, i)) for j, i in self.affected_indices
            ]
        for idx, (j, i) in enumerate(self.affected_indices):
            self._set_value(j, i, self.new_values[idx])

    def serialize(self):
        return {
            "type": self.__class__.__name__,
            "affected_indices": [to_native_tuple(idx) for idx in self.affected_indices],
            "new_values": [to_native(v) for v in self.new_values],
            "old_values": (
                [to_native(v) for v in self.old_values]
                if self.old_values is not None
                else None
            ),
        }

    @classmethod
    def deserialize(cls, data):
        return lambda topo: cls(
            topo,
            affected_indices=[tuple(idx) for idx in data["affected_indices"]],
            new_values=data["new_values"],
            old_values=data["old_values"],
        )

    @classmethod
    def reverse_deserialize(cls, data):
        return lambda topo: cls(
            topo,
            affected_indices=[tuple(idx) for idx in data["affected_indices"]],
            new_values=data["old_values"],
            old_values=data["new_values"],
        )


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
