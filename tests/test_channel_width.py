import pytest
from mom6_forge.channel_width import ChannelWidth, ChannelWidthList


def test_channel_width_validation():
    """Test that ChannelWidth validates component is U_width or V_width"""
    # Valid cases
    ch_u = ChannelWidth(
        component="U_width",
        lon1=-6.5,
        lon2=-4.75,
        lat1=35.6,
        lat2=36.3,
        width=12000.0,
        place="St. of Gibralter",
    )
    assert ch_u.component == "U_width"

    ch_v = ChannelWidth(
        component="V_width",
        lon1=28.75,
        lon2=29.5,
        lat1=41.1,
        lat2=41.3,
        width=5000.0,
        place="Bosphorus",
    )
    assert ch_v.component == "V_width"

    # Invalid component
    with pytest.raises(ValueError, match="component must be 'U_width' or 'V_width'"):
        ChannelWidth(
            component="X_width",
            lon1=0.0,
            lon2=1.0,
            lat1=0.0,
            lat2=1.0,
            width=1000.0,
            place="Invalid",
        )

    # Non-numeric coordinate fields
    with pytest.raises(ValueError, match="must be numeric"):
        ChannelWidth(
            component="U_width",
            lon1="bad",
            lon2=1.0,
            lat1=0.0,
            lat2=1.0,
            width=1000.0,
            place="Invalid",
        )


def test_channel_width_from_line():
    """ChannelWidth.from_line parses a valid ASCII line correctly."""
    line = "U_width,   -6.50,   -4.75,   35.60,   36.30,     12000.0 ! St. of Gibralter"
    ch = ChannelWidth.from_line(line)
    assert ch.component == "U_width"
    assert ch.lon1 == -6.50
    assert ch.lon2 == -4.75
    assert ch.lat1 == 35.60
    assert ch.lat2 == 36.30
    assert ch.width == 12000.0
    assert ch.place == "St. of Gibralter"


def test_channel_width_from_line_malformed():
    """ChannelWidth.from_line raises ValueError on a malformed line."""
    with pytest.raises(ValueError, match="Malformed channel width line"):
        ChannelWidth.from_line("U_width, -6.50, -4.75 ! missing fields")


def test_channel_width_list_write_load(tmp_path):
    """Test write and load roundtrip for ChannelWidthList"""
    # Create a list with some channels
    channels = ChannelWidthList()
    channels.add(
        ChannelWidth(
            component="U_width",
            lon1=-6.5,
            lon2=-4.75,
            lat1=35.6,
            lat2=36.3,
            width=12000.0,
            place="St. of Gibralter",
        )
    )
    channels.add(
        ChannelWidth(
            component="V_width",
            lon1=28.75,
            lon2=29.5,
            lat1=41.1,
            lat2=41.3,
            width=5000.0,
            place="Bosphorus",
        )
    )

    # Write to file
    output_file = tmp_path / "channels.txt"
    channels.write(output_file)
    assert output_file.exists()

    # Load from file
    loaded_channels = ChannelWidthList(filepath=output_file)
    assert len(loaded_channels.get_all()) == 2

    # Verify content
    all_channels = loaded_channels.get_all()
    assert all_channels[0].component == "U_width"
    assert all_channels[0].width == 12000.0
    assert all_channels[0].place == "St. of Gibralter"
    assert all_channels[1].component == "V_width"
    assert all_channels[1].width == 5000.0
    assert all_channels[1].place == "Bosphorus"


def test_channel_width_list_load_malformed_line(tmp_path):
    """load() raises ValueError on a line with fewer than 6 comma-separated fields."""
    bad_file = tmp_path / "bad_channels.txt"
    bad_file.write_text("U_width, -6.50, -4.75 ! missing fields\n")
    with pytest.raises(ValueError, match="Malformed channel width line"):
        ChannelWidthList(filepath=bad_file)
