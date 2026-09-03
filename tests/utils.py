"""Functions that are used in tests."""

import hashlib
import os
import shutil
import socket
import urllib.request
from pathlib import Path


def on_cisl_machine():
    """Return True if the current machine is a CISL machine, False otherwise."""
    fqdn = socket.getfqdn()
    return "hpc.ucar.edu" in fqdn


# ---------------------------------------------------------------------------
# CESM input data
# ---------------------------------------------------------------------------

CESM_INPUTDATA_URL = "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata"
GLADE_INPUTDATA = Path("/glade/campaign/cesm/cesmdata/inputdata")

# Files the test suite fetches on demand, keyed by path relative to the CESM
# input data root. Checksums guard against truncated or corrupted downloads;
# regenerate with `shasum -a 256` if a file is ever revised.
INPUTDATA_FILES = {
    "share/meshes/gx1v7_151008_ESMFmesh.nc": (
        "b1b892dfa5da00447c35b58a5bc3a35913e6eb1330b48ae46c2e8bcb88a3c641"
    ),
    "share/meshes/rx1_nomask_181022_ESMFmesh.nc": (
        "e67e140e6df410d3ca2b2a574d82959433a1bfabad464e71240370f83f0c41d4"
    ),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_inputdata(relpath):
    """Return a local path to a CESM input data file, downloading if needed.

    Looks in $CESMDATAROOT, then GLADE when on a CISL machine, then a local
    cache ($MOM6_FORGE_TEST_DATA, else ~/.cache/mom6_forge/inputdata),
    downloading on a miss. Raises rather than skipping if the file cannot be
    obtained, so missing coverage is never silent.
    """
    expected = INPUTDATA_FILES[relpath]

    root = os.environ.get("CESMDATAROOT")
    if root and (Path(root) / relpath).exists():
        return Path(root) / relpath
    if on_cisl_machine() and (GLADE_INPUTDATA / relpath).exists():
        return GLADE_INPUTDATA / relpath

    cache = Path(
        os.environ.get("MOM6_FORGE_TEST_DATA")
        or Path.home() / ".cache" / "mom6_forge" / "inputdata"
    )
    dest = cache / relpath
    if dest.exists():
        if _sha256(dest) == expected:
            return dest
        dest.unlink()  # corrupt or stale; re-fetch

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CESM_INPUTDATA_URL}/{relpath}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:  # network, DNS, HTTP error, ...
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Set CESMDATAROOT to an existing input "
            f"data tree, or MOM6_FORGE_TEST_DATA to a writable cache directory."
        ) from exc

    actual = _sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {url}\n  expected {expected}\n  got      {actual}"
        )
    tmp.rename(dest)
    return dest
