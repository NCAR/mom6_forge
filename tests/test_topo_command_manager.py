from mom6_bathy.command_manager import *
from mom6_bathy.edit_command import *
from test_edit_commands import gen_MinDepthCommand
import pytest
import xarray as xr
import json


def test_TopoCommandManager_init(get_rect_topo):
    topo = get_rect_topo  # TCM is generated in the topo object
    assert topo.tcm is not None
    assert isinstance(topo.tcm, TopoCommandManager)


def test_TopoCommandManager_execute(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    assert topo.min_depth == 0.0
    command = gen_MinDepthCommand
    command.message = "BLEEP"
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    topo.tcm.execute(gen_MinDepthCommand)
    assert topo.min_depth == 10.0  # Assert Action taken

    assert (
        sum(1 for _ in topo.tcm.repo.iter_commits()) == prev_hist + 1
    )  # 2 from init + 1 from execute, assert command is in history
    # Check the head commit has a message that includes BLEEP, which would confirm that the command executed is the one we passed
    assert "BLEEP" in topo.tcm.repo.head.commit.message
    with pytest.raises(ValueError):
        topo.tcm.execute("NotACommand")


def test_TopoCommandManager_undo(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    command = gen_MinDepthCommand

    topo.tcm.execute(gen_MinDepthCommand)
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    assert topo.min_depth == 10.0  # Assert Action taken
    sha_of_command = topo.tcm.repo.head.commit.hexsha
    topo.tcm.undo()
    assert topo.min_depth == 0.0  # Assert undo worked
    assert (
        sum(1 for _ in topo.tcm.repo.iter_commits()) == prev_hist + 1
    )  # Assert history has executed and undo commits
    assert (
        topo.tcm.repo.head.commit.message.startswith("UNDO")
        and sha_of_command in topo.tcm.repo.head.commit.message
    )  # Assert head commit is an undo commit and the reference is to the correct sha
    assert (
        len(topo.tcm._history_state()) == 4
    )  # Initial + execute + the undone order, there should not be an additional history command for the undo in the history state, after all it is just the opposite of the executed command
    topo.tcm.undo()  # Undo the initial two commands
    assert topo.tcm.undo(check_only=True)  # At least one command to undo
    topo.tcm.undo()
    assert not topo.tcm.undo(check_only=True)  # No more commands to undo


def test_TopoCommandManager_redo(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    command = gen_MinDepthCommand
    topo.tcm.execute(gen_MinDepthCommand)
    assert topo.min_depth == 10.0  # Assert Action taken
    sha_of_command = topo.tcm.repo.head.commit.hexsha
    topo.tcm.undo()
    assert topo.min_depth == 0.0  # Assert undo worked
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    topo.tcm.redo()
    assert topo.min_depth == 10.0  # Assert redo worked
    assert (
        sum(1 for _ in topo.tcm.repo.iter_commits()) == prev_hist + 1
    )  # Assert history has executed and redo commits
    assert (
        topo.tcm.repo.head.commit.message.startswith("REDO")
        and sha_of_command in topo.tcm.repo.head.commit.message
    )  # Assert head commit is an redo commit and the reference is to the correct sha
    assert (
        len(topo.tcm._history_state()) == 4
    )  # Initial + execute + the undone order, there should not be an additional history command for the undo or redo in the history state, after all it is just the opposite and reapplication of the executed command
    topo.tcm.undo()  # Undo the three commands
    topo.tcm.undo()
    topo.tcm.undo()
    assert not topo.tcm.undo(check_only=True)  # No more commands to undo
    assert topo.tcm.redo(check_only=True)  # At least one command to redo

    # Redo all three commands
    topo.tcm.redo()
    topo.tcm.redo()
    topo.tcm.redo()
    assert not topo.tcm.redo(check_only=True)  # No more commands to redo
    topo.tcm.undo()
    assert topo.tcm.redo(check_only=True)  # One command to redo
    topo.tcm.execute(
        gen_MinDepthCommand
    )  # New command after undo should clear redo stack
    assert not topo.tcm.redo(check_only=True)  # No more commands to redo


def test_TopoCommandManager_reapply_changes(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    topo.tcm.execute(gen_MinDepthCommand)
    assert topo.min_depth == 10.0  # Assert Action taken
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    store_depth = topo._depth.copy()
    topo._depth = xr.zeros_like(
        topo._depth
    )  # Corrupt the depth to ensure reapply_changes works
    topo.tcm.reapply_changes()
    assert (topo.depth == store_depth).all()  # Assert reapply worked
    assert (
        sum(1 for _ in topo.tcm.repo.iter_commits()) == prev_hist
    )  # Assert history only has the executed commits (reapply is quiet)


def test_TopoCommandManager_reset(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    topo.tcm.execute(gen_MinDepthCommand)
    assert topo.min_depth == 10.0  # Assert Action taken
    topo.tcm.reset()
    assert (np.isnan(topo.depth)).all()  # Assert reset worked
    assert topo.min_depth == 0.0  # Assert min depth reset


def test_tcm_checkout(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    current_branch = topo.tcm.repo.active_branch.name
    topo.tcm.create_branch("test_branch")
    topo.tcm.checkout("test_branch")
    topo.tcm.execute(gen_MinDepthCommand)
    assert topo.min_depth == 10.0  # Assert Action taken
    topo.tcm.checkout(current_branch)
    assert topo.min_depth == 0.0  # Assert back to main branch state


def test_tcm_parse_commit_message(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo

    # Build some history
    topo.tcm.execute(gen_MinDepthCommand)
    execute_sha = topo.tcm.repo.head.commit.hexsha
    topo.tcm.undo()
    topo.tcm.redo()
    topo.tcm.execute(gen_MinDepthCommand)

    # Parse each commit message and verify
    for index, commit in enumerate(topo.tcm.repo.iter_commits()):
        commit_sha = commit.hexsha
        cmd_type, affected_sha, cmd_data = topo.tcm.parse_commit_message(commit_sha)
        if index == 0:
            # Most recent commit should be an execute
            assert cmd_type == CommandType.COMMAND
            assert cmd_data["type"] == "MinDepthEditCommand"
            assert affected_sha == None  # Execute commands have no affected sha
        if index == 1:
            # Second most recent should be an redo
            assert cmd_type == CommandType.REDO
            assert affected_sha == execute_sha
            assert cmd_data == None
        if index == 2:
            # Third most recent should be an undo
            assert cmd_type == CommandType.UNDO
            assert affected_sha == execute_sha
            assert cmd_data == None
        if index == 3:
            # Fourth most recent should be the original execute
            assert cmd_type == CommandType.COMMAND
            assert cmd_data["type"] == "MinDepthEditCommand"
            assert affected_sha == None
            break


def test_tcm_history_init(get_rect_topo):
    topo = get_rect_topo
    assert topo.tcm.history_file_path.exists()


def test_tcm_add_to_history(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    topo.tcm.execute(gen_MinDepthCommand)
    history = topo.tcm.history_dict
    assert "head" in history  # head should be in history
    current_sha = topo.tcm.repo.head.commit.hexsha
    topo.tcm.add_to_history("testsha", "testmsg")
    topo.tcm.load_history()
    history = topo.tcm.history_dict
    assert history["testsha"] == "testmsg"
    assert current_sha in history  # current head sha should be in history


def test_tcm_commit(get_rect_topo, gen_MinDepthCommand):
    topo = get_rect_topo
    topo.tcm.commit(gen_MinDepthCommand, CommandType.COMMAND)
    history = topo.tcm.history_dict
    assert "head" in history  # head should be in history
    assert history["head"] == json.dumps(gen_MinDepthCommand.serialize())
    prev_hist = sum(1 for _ in topo.tcm.repo.iter_commits())
    old_history = history.copy()
    topo.tcm.commit(gen_MinDepthCommand, CommandType.UNDO)
    assert sum(1 for _ in topo.tcm.repo.iter_commits()) == prev_hist + 1
    topo.tcm.commit(gen_MinDepthCommand, CommandType.REDO)
    topo.tcm.load_history()
    assert (
        len(topo.tcm.history_dict.keys()) == len(old_history.keys()) + 1
        and "touch" in topo.tcm.history_dict
        and "touch" not in old_history.keys()
    )  # History should not change on undo/redo commits
