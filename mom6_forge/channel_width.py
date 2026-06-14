"""
A barebones set of classes to manage channel width constraints for MOM6 grids. It is effectively a list that can be applied on top of the bathymetry, and is not git-backed.
The channels represent a separate concern and can be independently shared, or managed outside the topography versioning system.
It is a basically a dict of info for each channel and then a list wrapped with write/load functions.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class ChannelWidth:
    """Single channel width constraint for MOM6 grid"""

    component: str  # 'U_width' or 'V_width'
    lon1: float
    lon2: float
    lat1: float
    lat2: float
    width: float  # meters
    place: str  # comment/location name

    def __post_init__(self):
        if self.component.lower() == "u_width":
            self.component = "U_width"
        elif self.component.lower() == "v_width":
            self.component = "V_width"
        else:
            raise ValueError(
                f"component must be 'U_width' or 'V_width', got '{self.component}'"
            )
        try:
            self.lon1 = float(self.lon1)
            self.lon2 = float(self.lon2)
            self.lat1 = float(self.lat1)
            self.lat2 = float(self.lat2)
            self.width = float(self.width)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"lon1, lon2, lat1, lat2, and width must be numeric: {e}"
            ) from e

    @classmethod
    def from_line(cls, line: str) -> "ChannelWidth":
        """Parse a single ASCII channel width line into a ChannelWidth object.

        Expected format::

            U_width,   -6.50,   -4.75,   35.60,   36.30,     12000.0 ! St. of Gibralter
        """
        parts = line.split("!")
        comment = parts[1].strip() if len(parts) > 1 else ""
        vals = parts[0].split(",")
        if len(vals) < 6:
            raise ValueError(
                f"Malformed channel width line (expected 6 comma-separated fields before '!'): {line!r}"
            )
        return cls(
            component=vals[0].strip(),
            lon1=vals[1].strip(),
            lon2=vals[2].strip(),
            lat1=vals[3].strip(),
            lat2=vals[4].strip(),
            width=vals[5].strip(),
            place=comment,
        )


class ChannelWidthList:
    """
    Manages list of channel width constraints.

    Note: Channel widths are NOT git-backed. These are additive configuration constraints
    applied on top of the bathymetry, not edits to the bathymetry itself. They represent
    a separate concern and can be independently shared, or managed outside the
    topography versioning system.
    """

    FMT_OUT = "{0:s}, {1:8.2f}, {2:8.2f}, {3:8.2f}, {4:8.2f}, {5:10.1f} ! {6:s}\n"

    def __init__(self, filepath: Optional[str | Path] = None):
        self.channels: List[ChannelWidth] = []
        if filepath is not None:
            self.load(filepath)

    def add(self, channel: ChannelWidth):
        """Add a channel width constraint"""
        self.channels.append(channel)

    def get_all(self) -> List[ChannelWidth]:
        """Get all channels"""
        return self.channels

    def write(self, filepath: str | Path):
        """Persist to ASCII file"""
        with open(filepath, "w") as f:
            for ch in self.channels:
                line = self.FMT_OUT.format(
                    ch.component, ch.lon1, ch.lon2, ch.lat1, ch.lat2, ch.width, ch.place
                )
                f.write(line)

    def load(self, filepath: str | Path):
        """Load from ASCII file, skipping comment/header lines (any line that
        does not begin with U_width or V_width), matching MOM6 behaviour."""
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not (
                        line.lower().startswith("u_width")
                        or line.lower().startswith("v_width")
                    ):
                        continue
                    self.channels.append(ChannelWidth.from_line(line))
        else:
            raise FileNotFoundError(f"Channel width file not found at {filepath}")
