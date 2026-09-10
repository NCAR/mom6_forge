Glossary
======================================

.. glossary::

    supergrid
        A MOM6 supergrid contains the grid metrics and the areas at twice the 
        nominal resolution of the actual computational grid. During runtime, 
        MOM6 reads in the supergrid file, and then decomposes the supergrid into the four
        staggered grids, each containing different sets of prognostic variables, e.g.,
        tracers, velocities, etc.
