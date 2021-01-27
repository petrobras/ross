.. ross documentation master file, created by
   sphinx-quickstart on Mon Feb  4 18:19:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |ross-logo| image:: ross-logo.svg

|ross-logo| ROSS: Rotordynamic Open-Source Software
===================================================

ROSS is a library written in Python for rotordynamic analysis. The source is
available at `github <https://github.com/ross-rotordynamics/ross>`_.

ROSS prsent other packages that will allow you to improve your rotodynamic analysis:

FluidFlow
---------

FluidFlow is a fluid dynamics package from ROSS.

Create your bearings from geometrical and fluid parameters.

Check for bearing pressure field, equilibrium position, create cylindrical or elliptical
geometries, evaluate the effect of wear on bearing coefficients and more.

Stochastic ROSS
---------------

Stochastic ROSS package brings statistical analysis to your project.

Create rotor models with uncertainties to any parameter of any element and evaluate how these
uncertainties impact your modeling, running the same analyzes presented on ROSS, and checking
for percentiles and confidence intervals.

ROSS - Machine Learning
-----------------------

`ROSS-ML <https://github.com/ross-rotordynamics/ross-ml>`_ is a Deep Learning package from ROSS, based in Keras and Scikit-Learn

Create your own surrogate models using data from both experimental or computational models.

This package returns a .h5 file that can be used in ROSS, Stochastic ROSS or any other
package writen in python.

Contents
--------

You can check the tutorial and examples on `binder <https://mybinder.org/v2/gh/ross-rotordynamics/ross/0.3?filepath=%2Fdocs%2Fexamples>`_.

Please, cite our `paper <https://joss.theoj.org/papers/10.21105/joss.02120>`_ published in JOSS - Journal of Open Source Software.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   tutorials
   examples
   download
   api
   contributing
   release_notes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
