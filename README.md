# mom6_forge

`mom6_forge` (formerly `mom6_bathy`) is a Python tool for generating MOM6 horizontal grids, vertical grids, bathymetry files, mapping, and other input files for use within the context of idealized and regional modeling.

**Documentation:** https://ncar.github.io/mom6_forge/

## Installation

```bash
conda env create -f environment.yml
conda activate mom6_forge
```

## Quick Start

See the tutorial notebooks in [`notebooks/`](notebooks/) for guided examples:

1. [Spherical Grid](notebooks/1_spherical_grid.ipynb) — Create a basic spherical grid
2. [Equatorial Refinement](notebooks/2_equatorial_res.ipynb) — Add enhanced equatorial resolution
3. [Custom Bathymetry](notebooks/3_custom_bathy.ipynb) — Generate bathymetry from topography data
4. [Ingest Land Mask](notebooks/4_ingest_landmask.ipynb) — Apply an external land mask
5. [Modify Existing](notebooks/5_modify_existing.ipynb) — Modify an existing grid/bathymetry
6. [Demo Editors](notebooks/6_demo_editors.ipynb) — Interactive bathymetry editing tools

## Requirements

- Python >=3.11.10, <3.12
- See [`environment.yml`](environment.yml) for the full dependency list