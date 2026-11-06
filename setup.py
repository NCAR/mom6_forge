import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mom6_forge",  # Replace with your own username
    version="0.4.0",
    author="Alper Altuntas",
    author_email="altuntas@ucar.edu",
    description="MOM6 simple grid and bathymetry generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NCAR/mom6_forge",
    packages=["mom6_forge"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11,<3.15",
    # What mom6_forge needs to run, with no upper bounds. The versions NCAR's
    # environment is tested with (and its JupyterHub needs) are in environment.yml.
    install_requires=[
        "numpy>=1.26",
        "xarray>=2023.12",
        "scipy>=1.11",
        "matplotlib>=3.9",
        "netcdf4>=1.6",
        "dask>=2026.3.0",
        "cartopy>=0.23",
        "cf_xarray>=0.11.1",
        "xesmf>=0.8.10",
        "regionmask>=0.13.0",
        "gitpython>=3.1",
        "ipywidgets>=8.1.1",
        "ipympl>=0.9.4",
    ],
    extras_require={
        "notebook": ["jupyterlab>=4.0"],
        "docs": [
            "sphinx>=8.1",
            "sphinx_rtd_theme>=3.0",
            "nbsphinx>=0.9.8",
            "myst-parser>=4.0",
        ],
        "dev": ["black>=24.1", "pytest>=8.0", "pytest-cov>=7.0"],
    },
)
