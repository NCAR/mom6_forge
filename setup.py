import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mom6_bathy",  # Replace with your own username
    version="0.0.1",
    author="Alper Altuntas",
    author_email="altuntas@ucar.edu",
    description="MOM6 simple grid and bathymetry generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NCAR/mom6-bathy",
    packages=["mom6_bathy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "setuptools>=69.0",
        "numpy",
        "xarray>=2023.12",
        "matplotlib>=3.9",
        "scipy>=1.11",
        "netcdf4>=1.6",
        "jupyterlab>=4.0",
        "ipympl>=0.9.4",
        "ipywidgets>=8.1.1",
        "sphinx>=8.1",
        "sphinx_rtd_theme>=3.0",
        "black>=24.1",
        "pytest>=8.0",
        "pytest-cov>=7.0",
        "gitpython>=3.1",
        "cartopy>=0.23",
    ],
)
