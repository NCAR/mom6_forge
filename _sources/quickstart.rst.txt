Quickstart Guide
======================================

`mom6_tools` library can be utilized via its Python API, i.e., directly within Python
scripts or within Jupyter notebooks. In this quickstart guide, we describe how
the tool can be utilized within a Jupyter Notebook, but the majority of
these instructions apply to Python scripts as well.

Step 1: Import modules
----------------------------------------------

The first step is to import the ``Grid`` and ``Topo`` classes of the 
`mom6_bathy` package. The ``Grid`` class represents
horizontal MOM6 grids, and is to be instantiated with the desired grid
configuration and resolution. After creating a grid instance, a ``Topo`` class
instance is to be created to generate an associated bathymetry.

.. code-block:: python

    from mom6_bathy.grid import Grid
    from mom6_bathy.topo import Topo

Step 2: Create the horizontal grid 
-------------------------------------------

After having imported the modules, we can now create a horizontal grid.
An example Grid instantiation:

.. code-block:: python

  grid = Grid(
      nx         = 180,         # Number of grid points in x direction
      ny         = 80,          # Number of grid points in y direction
      lenx       = 360.0,       # grid length in x direction, e.g., 360.0 (degrees)
      leny       = 160,         # grid length in y direction
      cyclic_x   = True,        # reentrant, spherical domain
      ystart     = -80.0        # start/end 10 degrees above/below poles to avoid singularity 
  )

In the above example, the ``Grid`` object, named ``grid``, is constructed by
specifying the required arguments ``nx``, ``ny``, ``config``, ``axis_units``,  ``lenx``,
and ``leny``, in addition to the optional argument ``ystart``. The full list of 
``Grid`` arguments and their descriptions may be printed by running
``Grid?`` statement on a notebook cell:


.. code-block::

    Grid?

    ...

    Parameters
    ----------
    nx : int
        Number of grid points in x direction
    ny : int
        Number of grid points in y direction
    lenx : float
        grid length in x direction, e.g., 360.0 (degrees)
    leny : float
        grid length in y direction, e.g., 160.0 (degrees)
    srefine : int, optional
        refinement factor for the supergrid. 2 by default
    xstart : float, optional
        starting x coordinate. 0.0 by default.
    ystart : float, optional
        starting y coordinate. -0.5*leny by default.
    cyclic_x : bool, optional
        flag to make the grid cyclic in x direction. False by default.
    tripolar_n : bool, optional
        flag to make the grid tripolar. False by default.
    displace_pole : bool, optional
        flag to make the grid displaced polar. False by default.

Note that tripolar and displaced pole grids cannot yet be created from scratch,
but existing tripolar and displaced pole grids can be modified via mom6_bathy.

*Avoiding singularity points*
*****************************

To avoid singularity points within the ocean grid:
  * The grid poles (which may be different than the true poles) must be left out of the grid,
    by making sure that the extent of the grid in the y-direction do not cover the poles, 
    e.g., by setting ``ystart`` to -80.0 degrees
    and ``leny`` to 160.0 degrees.
  * Alternatively, one or two singularities (typically, in the northern hemisphere) may be 
    displaced into land masses if ``displace_pole`` or ``tripolar_n`` options are to be used.
    The other singularity (typically, in the southern hemisphere) would still need to be
    left out the geographic extent of the grid.

If a singularity (a pole) is present within the ocean grid, a land component (active or data) must be added
to the pose of hiding the singularity points of spherical ocean grids within the CESM framework.

*Grid Metrics and Attributes*
*****************************

When a ``Grid`` instance gets created, several grid metrics and attributes
on all staggerings are automatically computed and populated. These metrics and attributes
are accessible via the accessor operator (``.``). For example, to access "the array
of t-grid longitutes" of ``grid``:
    
.. code-block:: python

    grid.tlon

The full list of grid metrics and attributes:

* ``tlon``: array of t-grid longitudes
* ``tlat``: array of t-grid latitudes
* ``ulon``: array of u-grid longitudes
* ``ulat``: array of u-grid latitudes
* ``vlon``: array of v-grid longitudes
* ``vlat``: array of v-grid latitudes
* ``qlon``: array of corner longitudes
* ``qlat``: array of corner latitudes
* ``dxt``: x-distance between U points, centered at t
* ``dyt``: y-distance between V points, centered at t
* ``dxCv``: x-distance between q points, centered at v
* ``dyCu``: y-distance between q points, centered at u
* ``dxCu``: x-distance between y points, centered at u
* ``dyCv``: y-distance between t points, centered at v
* ``angle``: angle T-grid makes with latitude line
* ``tarea``: T-cell area


*Supergrid*
*****************************
In addition to above grid metrics and attributes, the ``Grid`` class incorporates an
underlying :term:`supergrid` instance associated the grid instance, which is again
accessible via the (``.``) operator:

.. code-block:: python

    grid.supergrid

Any user changes to coordinates, e.g., increasing the equatorial resolution,
must be applied to the supergrid using the ``update_supergrid`` method. This is
because the supergrid is the underlying refined grid that is used to determine the
the four staggered grids (T,U,V,Q) that forms the actual computational grid.
Users can modify the supergrid by providing a new x and y coordinate arrays, e.g.,
as follows:

.. code-block:: python

  grid.update_supergrid(xdat, ydat)

where ``xdat`` and ``ydat`` are user-defined 2-dimensional numpy arrays containing
the new x and y coordinates of the supergrid. Running the ``update_supergrid``
method of a ``Grid`` instance automatically updates all other grid metrics listed
above.

Note: the supergrid implementation in `mom6_bathy` relies on `MIDAS <https://github.com/mjharriso/MIDAS>`_,
a python library developed by M. Harrison (GFDL).

Step 3: Create Bathymetry
----------------------------------------------

After having generated the horizontal grid, we can now create an associated bathymetry
using the ``Topo`` class of the `mom6_bathy` tool. We instantiate a bathymetry 
object as follows:

.. code-block:: python

    topo = Topo(grid, min_depth=10.0)

The first argument (``grid``) of ``Topo`` constructor is the horizontal grid instance for which
the bathymetry is to be created, while the second argument (``min_depth``) is the minimum ocean depth.
Any column in the ocean grid with a depth shallower than ``min_depth``  is masked out of the ocean
domain. The minimum depth attribute of a bathymetry instance may be changed afterwards using the
assignment operator. For example:

.. code-block:: python

    topo.min_depth = 5.0

*Predefined Bathymetry Configurations*
**************************************
The ``Topo`` class provides three predefined bathymetry configurations, which are also
available in MOM6 as idealized configurations. (See `TOPO_CONFIG` parameter in MOM_input)

  * `flat`: flat bottom set to MAXIMUM_DEPTH. Example:
  * `bowl`: an analytically specified bowl-shaped basin ranging between MAXIMUM_DEPTH and MINIMUM_DEPTH.
  * `spoon`: a similar shape to 'bowl', but with an vertical wall at the southern face.

Examples:

.. code-block:: python

    # flat bottom
    topo.set_flat(D=500.0)

    # bowl
    topo.set_bowl(500.0, 50.0, expdecay=1e7)

    # spoon
    topo.set_spoon(500.0, 50.0, expdecay=1e7)
    
The first and the second arguments of ``set_bowl`` and ``set_spoon`` methods are maximum depth
and minimum depth, respectively.

Check out the following notebook to see examples of above predefined bathymetry options: `1_spherical_grid.ipynb 
<https://github.com/NCAR/mom6_bathy/blob/master/notebooks/1_spherical_grid.ipynb>`_

*Custom Bathymetry*
*************************************
In addition to the above predefined configurations, users may provide their own depth arrays. For
example:
  
.. code-block:: python

    import numpy as np

    # define a custom depth
    i = grid.tlat.nx.data                # array of x-indices
    j = grid.tlat.ny.data[:,np.newaxis]  # array of y-indices 
    custom_depth = 400.0 + 80.0 * np.sin(i*np.pi/6.) * np.cos(j*np.pi/6.)

    # update the bathymetry:
    topo.depth = custom_depth


*Adding ridges*
*************************************
Simpler model bathymetry configurations typically include ridges to represent straits and
continents in an idealized manner. The ``Topo`` class provides ``apply_ridge`` method
to add ridges to the bathymetry. Example usage:

.. code-block:: python

  topo.apply_ridge(height=200, width=8, lon=240, ilat=(10,80) )

Example notebook: `3_custom_bathy.ipynb 
<https://github.com/NCAR/mom6_bathy/blob/master/notebooks/3_custom_bathy.ipynb>`_


Step 4: Write Model Input Files
----------------------------------------------

The final step of `mom6_bathy` workflow is to write out the netcdf files containing grid
and bathymetry data. These files are to be read in by CESM and MOM6 during runtime.

*Supergrid File*
****************

The ``write_supergrid`` method of a ``Grid`` instance writes out the MOM6 supergrid file
in netcdf format. The ``GRID_FILE`` parameter in ``MOM_input`` file can then be set to
the path of the supergrid file written by the ``Grid`` instance.

.. code-block:: python

  grid.write_supergrid("my_ocean_hgrid.nc")

The supergrid file is the only input file that is written by the `Grid` class. All other
input files require either topography (depth) or mask information. Hence, they are to be
written by the `Topo` class.

*Topography (Bathymetry) File*
******************************

The ``write_topo`` method of the ``Topo`` class writes out the MOM6 bathymetry file in netcdf format.
``TOPO_FILE`` parameter in ``MOM_input`` file can then be set to the path of the topography file
written by the ``Topo`` instance.

.. code-block:: python

  topo.write_topo("my_ocean_topog.nc")

*CICE grid file*
******************************

If the model is to be run with the CICE component, the ``write_cice_grid`` method of the 
``Topo`` class writes out the CICE grid file in netcdf format. The relevant CICE namelist
parameters can then be updated to read in the CICE grid file written by the ``Topo`` instance.

.. code-block:: python

  topo.write_cice_grid("my_cice_grid.nc")

*ESMF Mesh file*
*******************************

In addition to the MOM6 supergrid file, MOM6 topography file and CICE grid file, an
ESMF mesh file is required when running CESM. The ESMF mesh file is used
by the NUOPC coupler to acquire grid and mask information. The ``write_esmf_mesh`` method
of the ``Topo`` class writes out the ESMF mesh file in netcdf format.

.. code-block:: python

  topo.write_esmf_mesh("my_esmf_mesh.nc")

Step 5: Editing Grids and Bathymetry
----------------------------------------------
Beyond creating standard grids and simple topographies, mom6_bathy provides advanced tools
for interactively editing and creating complex model domains. These features are designed to
facilitate reproducible workflows for custom model configurations and model tuning. These
domain configurators can be used for tasks such as:

* Editing Bathymetry: Manually or programmatically modifying ocean depths.
* Creating New Grids: Defining entirely new horizontal grid structures.
* Creating Vertical Grids: Specifying the vertical layering of the ocean model.

Check out the notebook for examples of these advanced features: 
`6_demo_editors.ipynb <https://github.com/NCAR/mom6_bathy/blob/master/notebooks/6_demo_editors.ipynb>`_

Further steps
----------------------------------------------

The remaining steps of configuring the model, which include specifying initial conditions,
forcings, and runtime parameters, are beyond the scope of the `mom6_bathy` tool. Note that a 
complementary tool called `visualCaseGen`, which includes `mom6_bathy` as a submodule, can be used
to generate a complete model configuration. `visualCaseGen` provides a graphical user interface
to set up the model grid, bathymetry, initial conditions, forcing, and runtime parameters for MOM6
and other CESM components. Hence, new users are encouraged to use `visualCaseGen` for a complete
model configuration. See: `visualCaseGen <https://github.com/ESMCI/visualCaseGen>`_