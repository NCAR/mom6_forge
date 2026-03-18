from mom6_bathy.edit_command import *
import pytest


@pytest.fixture
def gen_MinDepthCommand(get_rect_topo):
    topo = get_rect_topo
    command = MinDepthEditCommand(topo, "min_depth", 10.0, 0.0)
    return command


def test_MinDepthCommand_init_and_execute(gen_MinDepthCommand):
    command = gen_MinDepthCommand  # Init is tested in fixture
    assert command._topo.min_depth == 0.0
    command()
    assert command._topo.min_depth == 10.0


def test_serialize_deserialize_MinDepthCommand(gen_MinDepthCommand):
    command = gen_MinDepthCommand
    serialized = command.serialize()
    deserialized_command = MinDepthEditCommand.deserialize(serialized)(command._topo)
    rdc = MinDepthEditCommand.reverse_deserialize(serialized)(command._topo)
    assert deserialized_command.attr == command.attr
    assert deserialized_command.new_value == command.new_value
    assert deserialized_command.old_value == command.old_value
    assert rdc.attr == command.attr
    assert rdc.old_value == command.new_value
    assert rdc.new_value == command.old_value


@pytest.fixture
def gen_DepthEditCommand(get_rect_topo):
    topo = get_rect_topo
    j, i = 1, 2
    new_val = 10
    old_val = topo.depth[j, i]
    command = DepthEditCommand(topo, [(j, i)], [new_val], old_values=[old_val])
    return command


def test_DepthEditCommand_init_and_execute(gen_DepthEditCommand):
    command = gen_DepthEditCommand  # Init is tested in fixture
    command()
    assert command._topo.depth[1, 2] == 10.0


def test_serialize_deserialize_DepthEditCommand(gen_DepthEditCommand):
    command = gen_DepthEditCommand
    serialized = command.serialize()
    deserialized_command = DepthEditCommand.deserialize(serialized)(command._topo)
    rdc = DepthEditCommand.reverse_deserialize(serialized)(command._topo)
    assert deserialized_command.affected_indices == command.affected_indices
    assert deserialized_command.new_values == command.new_values
    assert deserialized_command.old_values == command.old_values
    assert rdc.affected_indices == command.affected_indices
    assert rdc.new_values == command.old_values
    assert rdc.old_values == command.new_values
