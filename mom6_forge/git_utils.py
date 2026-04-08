import os
import git
import hashlib
from pathlib import Path


def get_domain_dir(grid, base_dir="TopoLibrary"):
    """
    Returns a unique directory path for a given grid object.
    Works for both rectilinear and curvilinear grids.
    """

    # Flatten and convert to bytes (deterministic)
    data_bytes = grid.tlon.values.tobytes() + grid.tlat.values.tobytes()

    # Generate SHA256
    sha = hashlib.sha256(data_bytes).hexdigest()
    return Path(base_dir) / sha


def get_repo(path):
    """
    Ensure a git repository exists at the given path. If not, initialize one.
    Returns the repo object.
    """
    if not os.path.exists(os.path.join(path, ".git")):
        repo = git.Repo.init(path)
        # Path to .gitignore
        gitignore_path = os.path.join(path, ".gitignore")
        lines = ["*.nc"]
        with open(gitignore_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    else:
        repo = git.Repo(path)
    return repo
