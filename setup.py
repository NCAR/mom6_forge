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
    python_requires=">=3.11.10,<3.12",
    install_requires=[
        "setuptools>=69.0,<69.1",
        "numpy>=1.26,<1.27",
        "xarray>=2023.12,<2024",
        "matplotlib>=3.9,<3.10",
        "scipy>=1.11,<1.12",
        "netcdf4>=1.6,<1.7",
        "jupyterlab>=4.0,<4.1",
        "ipympl>=0.9.4,<0.10",
        "ipywidgets>=8.1.1,<8.2",
        "sphinx>=8.1,<8.2",
        "sphinx_rtd_theme>=3.0,<3.1",
        "black>=24.1,<24.2",
        "pytest>=8.0",
        "pytest-cov>=7.0",
        "gitpython>=3.1,<4.0",
        "cartopy>=0.23,<0.24",
    ],
)
