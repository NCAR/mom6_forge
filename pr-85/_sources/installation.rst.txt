Installation
======================================

Prerequisite
-------------

* conda: Make sure that you have conda installed (either via Miniconda or Anaconda). See the following link for installing miniconda on your local machine:

https://docs.conda.io/en/latest/miniconda.html

Instructions
-------------

First, clone the `mom6_forge` GitHub repository as follows:

.. code-block:: bash

    git clone --recursive https://github.com/NCAR/mom6_forge.git

Then, `cd` into your newly checked out `mom6_forge` clone and run the
following command to install `mom6_forge` and all dependencies.

.. code-block:: bash

    cd mom6_forge
    conda env create -f environment.yml

The above command creates a new conda environment called `mom6_forge`. You can
activate this environment by running:

.. code-block:: bash

    conda activate mom6_forge

To confirm that the installation was successful, execute the following command:

.. code-block:: bash

    python -c "import mom6_forge"

If no error message is displayed, then the installation is successful. Note that
you will need to activate the `mom6_forge` environment before every use.
