from pathlib import Path
import pytest
import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings("ignore")
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

from utils import on_cisl_machine


def test_notebooks():
    """Run all notebooks in the ../notebooks directory."""

    notebook_files = [
        "1_spherical_grid.ipynb",
        "2_equatorial_res.ipynb",
        "3_custom_bathy.ipynb",
    ]

    if on_cisl_machine():
        notebook_files.extend(
            [
                "4_ingest_landmask.ipynb",
                "5_modify_existing.ipynb",
            ]
        )

    notebooks_path = Path(__file__).parent / "../notebooks"

    for notebook_file in notebook_files:
        if not notebook_file.endswith(".ipynb"):
            continue
        with open(Path(notebooks_path) / notebook_file) as f:
            print(f"Running notebook: {notebook_file}")

            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
            ep.preprocess(nb, {"metadata": {"path": notebooks_path}})


if __name__ == "__main__":
    test_notebooks()
