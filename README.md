# mom6_forge

`mom6_forge` (formerly `mom6_bathy`) is a Python tool for generating MOM6 horizontal grids, vertical grids, bathymetry files, mapping, and other input files for use within the context of idealized and regional modeling.

**Documentation:** https://ncar.github.io/mom6_forge/

## Installation

### Git Clone

```bash
git clone https://github.com/NCAR/mom6_forge.git
cd mom6_forge
conda env create -f environment.yml
conda activate mom6_forge
```

### Conda Forge

```bash
conda install -c conda-forge mom6_forge
```

### PyPI (pip)

`mom6_forge` depends on [ESMPy](https://earthsystemmodeling.org/esmpy/), which must be installed before `mom6_forge` because it is not available on PyPI. Install it via conda first, then install `mom6_forge` with pip:

```bash
conda install esmpy
pip install mom6_forge
```

## Quick Start

See the tutorial notebooks in [`docs/source/notebooks/`](docs/source/notebooks/) for guided examples:

1. [Spherical Grid](docs/source/notebooks/1_spherical_grid.ipynb) — Create a basic spherical grid
2. [Equatorial Refinement](docs/source/notebooks/2_equatorial_res.ipynb) — Add enhanced equatorial resolution
3. [Custom Bathymetry](docs/source/notebooks/3_custom_bathy.ipynb) — Generate bathymetry from topography data
4. [Ingest Land Mask](docs/source/notebooks/4_ingest_landmask.ipynb) — Apply an external land mask
5. [Modify Existing](docs/source/notebooks/5_modify_existing.ipynb) — Modify an existing grid/bathymetry
6. [Coarsen Existing](docs/source/notebooks/6_coarsen_existing.ipynb) — Create a coarser version of an existing MOM6 grid
7. [Interactive Editors](docs/source/notebooks/7_demo_editors.ipynb) — The `TopoEditor` and `VGridCreator` notebook widgets
8. [Cressman Interpolation](docs/source/notebooks/8_cressman_interpolation.ipynb) — Deep dive into the Cressman distance-weighted interpolation used for bathymetry

## Requirements

- Python >=3.11.10, <3.15
- See [`environment.yml`](environment.yml) for the full dependency list
