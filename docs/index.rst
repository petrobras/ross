.. ross documentation master file, created by
   sphinx-quickstart on Mon Feb  4 18:19:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. raw:: html

   <div class="container-fluid">
     <div class="row">

.. |ross-logo| image:: ross-logo.svg

|ross-logo| ROSS: Rotordynamic Open-Source Software
===================================================

.. raw:: html

     </div>
   </div>

   <div class="container-fluid">
     <div class="row">
       <div class="col-md-9">

ROSS is a library written in Python for rotordynamic analysis. The source code is
available at `github <https://github.com/ross-rotordynamics/ross>`_.

For a fast introduction to the library you can check out the :doc:`tutorials <tutorials>`
and :doc:`examples <examples>`. These are also available on `binder <https://mybinder.org/v2/gh/ross-rotordynamics/ross/0.4?filepath=%2Fdocs%2Fexamples>`_.

Additional Packages
"""""""""""""""""""

ROSS has a core library in which you can create elements (shaft, disks, bearings etc.)
and assemble these elements in a rotor object where you can run analysis. Additional
packages are also available:

**FluidFlow** is a fluid dynamics package where you can create bearings from
geometrical and fluid parameters and check the bearing pressure field,
equilibrium position, evaluate the effect of wear on bearing coefficients and more.

**ROSS-Stochastic** enables you to add statistical analysis to your project by
You can create rotor models with uncertainties to any parameter of any element
and evaluate how these uncertainties impact your model, running the same analyzes
presented on ROSS, and checking for percentiles and confidence intervals.

**ROSS-Machine Learning** is a Deep Learning package from ROSS, based in Keras and Scikit-Learn
where you can create your own surrogate models using data from both experimental or computational models.
This package returns a .h5 file that can be used in ROSS, Stochastic ROSS or any other
package written in python.

.. raw:: html

     </div>
   </div>
   <div class="col-md-3">
     <div class="panel panel-default">
       <div class="panel-heading">
         <h3 class="panel-title">Contents</h3>
       </div>
       <div class="panel-body">

.. toctree::
   :maxdepth: 1

   installation
   tutorials
   examples
   download
   api
   contributing
   citing
   release_notes

.. raw:: html

       </div>
     </div>
   </div>
