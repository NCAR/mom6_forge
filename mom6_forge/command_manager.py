import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from mom6_forge.git_utils import get_repo
from pathlib import Path
from mom6_forge.edit_command import EditCommand


class CommandType(Enum):
    """Enumeration for command types in the history."""

    INITIAL = "INITIAL"
    COMMAND = "COMMAND"
    UNDO = "UNDO"
    REDO = "REDO"


class CommandManager(ABC):
    def __init__(self, directory, command_registry):
        self.directory = Path(directory)
        self.repo = get_repo(directory)
        self.command_registry = command_registry

        self.history_file_path = self.directory / f"command_history.json"
        if not self.history_file_path.exists():
            self.history_dict = {"Description": "Command history"}
            self.write_history()
        else:
            with self.history_file_path.open("r") as f:
                self.history_dict = json.load(f)

        # Edit history (see _history_state), kept in step with each commit and
        # rebuilt from git only when HEAD moves under it (checkout, another process).
        self._cache = None
        self._cache_head = None

    def write_history(self):
        """Write the current history dictionary to the history file."""
        with self.history_file_path.open("w") as f:
            json.dump(self.history_dict, f, indent=2)

    def load_history(self):
        """Load the history dictionary from the history file."""
        with self.history_file_path.open("r") as f:
            history_dict = json.load(f)
        self.history_dict = history_dict

    @abstractmethod
    def execute(self, cmd, message=None):
        """Execute a command, push it onto the undo stack, and clear the redo stack."""
        pass

    def commit(self, command, cmd_type: CommandType, command_message=None):
        """Commit a command to git and step the cached history to match."""
        state = self._history()  # bring the cache up to date before HEAD moves

        command_data = None
        if cmd_type == CommandType.COMMAND:
            command_data = command.serialize()  # must be JSON-serializable
            self.add_to_history("head", json.dumps(command_data))
        else:
            # UNDO/REDO commits store no data; just bump history to let the commit happen
            self.touch_history()
        # git add it
        rel_path = os.path.relpath(self.history_file_path, self.repo.working_tree_dir)

        self.repo.git.add(rel_path)

        self.repo.git.commit(
            "--no-gpg-sign",
            "-m",
            f"{cmd_type.value}-{command_message if command_message is not None else ''}",
        )
        sha = self.repo.head.commit.hexsha
        self._step(state, cmd_type, sha, command_message, command_data)
        self._cache_head = sha

    def parse_commit_message(self, sha):
        """
        Parse a commit message to extract command type and data.
        Expected format: "<CommandType>"
        """
        commit = self.repo.commit(sha)

        # Access the commit message
        commit_msg = commit.message
        # Split at the first colon
        cmd_raw, sep, affected_sha = commit_msg.partition("-")
        cmd_type = CommandType(cmd_raw.strip())
        if cmd_type in (CommandType.UNDO, CommandType.REDO):
            affected_sha = affected_sha.strip()
        else:
            affected_sha = None

        # write  acheck that if the shaw it's head instead we use key head
        if sha == self.repo.head.commit.hexsha:
            sha = "head"

        if cmd_type == CommandType.COMMAND:
            cmd_data = json.loads(self.history_dict[sha])
        else:
            cmd_data = None
        return cmd_type, affected_sha, cmd_data

    def touch_history(self):
        """make a change to the history for commit purposes"""
        if "head" in self.history_dict:
            self.history_dict[self.repo.head.commit.hexsha] = self.history_dict["head"]
            del self.history_dict["head"]

        if "touch" in self.history_dict:
            self.history_dict["touch"] = not self.history_dict["touch"]
        else:
            self.history_dict["touch"] = True
        self.write_history()

    def add_to_history(self, sha, command_data: str):
        """
        Ensure the command unoffical history file exists (true history is generated from commit messages), append a line to it.
        """

        # Load existing history
        self.load_history()

        # Move previous head entry to sha entry of current head
        if "head" in self.history_dict:
            self.history_dict[self.repo.head.commit.hexsha] = self.history_dict["head"]
            del self.history_dict["head"]

        # 2. Add/overwrite the SHA entry
        self.history_dict[sha] = command_data

        # 3. Write back to file
        self.write_history()

        # Return path to history file
        return self.history_file_path

    @abstractmethod
    def undo(self):
        """Undo the last command."""
        pass

    @abstractmethod
    def redo(self):
        """Redo the last undone command."""
        pass

    def _history_state(self):
        """
        Rebuild the edit history by walking the git log.

        Returns {"cmd_data": {sha: data}, "applied": [sha, ...], "undone": [sha, ...]}:
        the applied commands oldest first, and the redo stack with the next redo last.
        """
        self.load_history()
        state = {"cmd_data": {}, "applied": [], "undone": []}
        if not self.repo.head.is_valid():
            return state  # no commits yet
        for commit in self.repo.iter_commits(reverse=True):
            try:
                cmd_type, target, cmd_data = self.parse_commit_message(commit.hexsha)
            except ValueError:
                continue  # not a history commit
            self._step(state, cmd_type, commit.hexsha, target, cmd_data)
        return state

    @staticmethod
    def _step(state, cmd_type, sha, target, cmd_data):
        """Advance a history state by one commit. An UNDO or REDO that matches nothing is ignored."""
        if cmd_type == CommandType.COMMAND:
            state["cmd_data"][sha] = cmd_data
            state["applied"].append(sha)
            state["undone"].clear()
        elif cmd_type == CommandType.UNDO and target in state["applied"]:
            state["applied"].remove(target)
            state["undone"].append(target)
        elif cmd_type == CommandType.REDO and target in state["undone"]:
            state["undone"].remove(target)
            state["applied"].append(target)

    def _history(self):
        """The edit history, cached; the git log is walked again only if HEAD has moved."""
        head = self.repo.head.commit.hexsha if self.repo.head.is_valid() else None
        if self._cache is None or head != self._cache_head:
            self._cache = self._history_state()
            self._cache_head = head
        return self._cache

    def get_current_branch(self):
        return self.repo.active_branch.name

    def list_branches(self):
        return [head.name for head in self.repo.heads]

    def create_branch(self, branch):
        self.repo.create_head(branch)

    def get_tag_names(self):
        return [tag.name for tag in self.repo.tags]


class TopoCommandManager(CommandManager):
    """
    The unique part here is we need to handle the topo object seperately!
    """

    def __init__(self, topo, command_registry):
        super().__init__(topo.domain_dir, command_registry)
        self._topo = topo
        self._original_topo_path = self.directory / "original_topog.nc"
        if not self._original_topo_path.exists():  # I.E. first time init
            self._topo.write_topo_file(self._original_topo_path)
        self.load_history()

    def save(self, file_name="topog.nc"):
        """Save the current topo state, and make history permanent as a git tag with the given name."""
        # Write Topo
        self._topo.write_topo_file(self.directory / file_name)

    def tag(self, tag_name):
        """Tag the current state of the topo."""
        self.repo.git.tag(tag_name)
        self.save(file_name=f"{tag_name}_topog.nc")

    def checkout(self, branch_name):
        """Switch to a branch, loading the topo state from that branch."""
        self.repo.git.checkout(branch_name)

        # Replay New History
        self.reapply_changes()

    def reapply_changes(self):
        """Rebuild the topo from the original file by replaying the applied commands."""
        self._topo.set_depth_via_topog_file(
            self._original_topo_path, skip_version_control=True
        )
        self._topo._user_mask = None  # the original has no manual mask
        state = self._history()
        for sha in state["applied"]:
            cmd_data = state["cmd_data"][sha]
            self.command_registry[cmd_data["type"]].deserialize(cmd_data)(self._topo)()

    def execute(self, cmd, cmd_type: CommandType = CommandType.COMMAND, message=None):
        """
        Execute a command object. If it's a user edit, push to history and clear redo.
        For system commands (undo, redo, save, load, reset), the command object handles everything.
        """
        if isinstance(cmd, EditCommand):
            cmd()
            self.commit(
                cmd,
                cmd_type=cmd_type,
                command_message=message if message is not None else cmd.message,
            )
        else:
            raise ValueError(
                "Unsupported command type for execute. {}".format(
                    cmd.__class__.__name__
                )
            )

    def undo(self, check_only=False):
        """Undo the latest applied command with an UNDO commit. Returns whether there was one."""
        state = self._history()
        if check_only or not state["applied"]:
            return bool(state["applied"])
        sha = state["applied"][-1]
        cmd_data = state["cmd_data"][sha]
        command_class = self.command_registry[cmd_data["type"]]
        cmd = command_class.reverse_deserialize(cmd_data)(self._topo)
        self.execute(cmd, cmd_type=CommandType.UNDO, message=sha)
        return True

    def redo(self, check_only=False):
        """Redo the latest undone command with a REDO commit. Returns whether there was one."""
        state = self._history()
        if check_only or not state["undone"]:
            return bool(state["undone"])
        sha = state["undone"][-1]
        cmd_data = state["cmd_data"][sha]
        command_class = self.command_registry[cmd_data["type"]]
        cmd = command_class.deserialize(cmd_data)(self._topo)
        self.execute(cmd, cmd_type=CommandType.REDO, message=sha)
        return True

    def reset(self):
        """
        Undo back to the latest command that set every cell's depth or mask.

        That command (set_flat, set_from_dataset, topo.user_mask = ..., ...) is the
        baseline the later edits were made on. The first command, made when the Topo
        was created, is never undone. Each step is an UNDO commit, so a reset is kept
        on reload and can be redone.
        """
        while True:
            state = self._history()
            applied = state["applied"]
            if len(applied) <= 1 or self._sets_every_cell(
                state["cmd_data"][applied[-1]]
            ):
                return
            self.undo()

    def _sets_every_cell(self, cmd_data):
        """True for a depth or mask edit that covers the whole grid (old histories included)."""
        if cmd_data["type"] not in ("DepthEditCommand", "MaskEditCommand"):
            return False
        cmd = self.command_registry[cmd_data["type"]].deserialize(cmd_data)(self._topo)
        return len(cmd.affected_indices) == self._topo.depth.size
