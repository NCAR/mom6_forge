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
    python_requires=">=3.11,<3.15",
    install_requires=[
        "setuptools>=69.0,<82.1",
        "numpy>=1.26,<2.5.0",
        "xarray>=2023.12,<2026.3.0",
        "matplotlib>=3.9,<3.11.0",
        "scipy>=1.11,<1.18.0",
        "netcdf4>=1.6,<1.8.0",
        "jupyterlab>=4.0,<4.6.0",
        "ipympl>=0.9.4,<0.11.0",
        "ipywidgets>=8.1.1,<8.2.0",
        "sphinx>=8.1,<9.2.0",
        "sphinx_rtd_theme>=3.0,<3.2.0",
        "black>=24.1,<26.4.0",
        "pytest>=8.0,<9.1.0",
        "pytest-cov>=7.0,<7.2.0",
        "gitpython>=3.1,<3.2.0",
        "cartopy>=0.23,<0.3.0",
    ],
)
