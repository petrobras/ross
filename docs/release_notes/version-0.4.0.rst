Version 0.4.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Class MagneticBearingElement was created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This class creates a magnetic bearing element. 
The element can be created by:

* defining directly the stifness and damping coefficients with the method **\ **init**\ **\ ;
* defining the magnetic parameters g0, i0, ag, nw, alpha and the proportional and derivative gains of a PID controller  kp_pid, kd_pid with the method **param_to_coef**.

The reference for the equations is:
**Book**\ : Magnetic Bearings. Theory, Design, and Application to Rotating Machinery
**Authors**\ : Gerhard Schweitzer and Eric H. Maslen
**Page**\ : 84-95 (magnetic parameters) and 354 (PID gains)

Stochastic Ross module
^^^^^^^^^^^^^^^^^^^^^^

The files follow the hierarchy of ROSS: different files to build the elements and a main file to build the rotors.

The main difference between these new files and the element files from ROSS is that the stochastic elements files build a list of elements instead of a single one.

The analysis will also follow the ROSS structure, but it's not added yet.

6dof assembly
^^^^^^^^^^^^^

Generalization of the assembly and evaluation codes of ROSS for a generic "N" DoF sized formulation. This is intended to enable the 6 DoF implementation to be executed appropriately, and is made in a generalized fashion by default.

Add scale_factor argument to DiskElement class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Argument to scale the disk patch. Used to increase (scale_factor > 1.0) or reduce (scale_factor < 1.0) the size of disk patches when plotting the rotor.


Add units to the DiskElement, BearingElement and PointMass classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arguments for these classes can be passed as pint.Quantity now.


Replacing plotting libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have replaced Matplotlib and Bokeh plotting libraries with Plotly.

Add methods to plot the deflected shape

Move "sparse" argument to run_modal()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The eigenvalue / eigenvector solver is chosen when running the run_modal() method, and is no longer a parameter from the rotor model

Set default values to log_dec color scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The color scale to represent the log_dec values ranges between 0 and 1, independently of the log_dec values calculated.
It can be manually modified by insert 2 simple kwargs: coloraxis_cmin and coloraxis_cmax. It will change
respectively the minimum and maximum values for the color scale.
Also adds exponentformat = "none" to X and Y axis for better visualization of values

Example with default options: 

.. code-block:: python

   >>> speed_range = np.linspace(0, 3000, 51)
   >>> campbell = rotor.run_campbell(speed_range)
   >>> campbell.plot(harmonics=[0.5, 1])


.. image:: https://user-images.githubusercontent.com/45969994/87157591-def88a00-c294-11ea-9f26-8807122feb08.png
   :target: https://user-images.githubusercontent.com/45969994/87157591-def88a00-c294-11ea-9f26-8807122feb08.png
   :alt: image


Example changing the max value for the scale: 

.. code-block:: python

   >>> speed_range = np.linspace(0, 3000, 51)
   >>> campbell = rotor.run_campbell(speed_range)
   >>> campbell.plot(harmonics=[0.5, 1], coloraxis_cmax=2.0)


.. image:: https://user-images.githubusercontent.com/45969994/87157716-06e7ed80-c295-11ea-9eff-89e5a2c6164c.png
   :target: https://user-images.githubusercontent.com/45969994/87157716-06e7ed80-c295-11ea-9eff-89e5a2c6164c.png
   :alt: image


Replace interp1d by line_shape from Plotly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It removes the need of creating sub arrays from interpolation to smooth some results in StaticResults.


Add probe argument to ForcedResponseResults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Replace ``dof`` by ``probe``\ argument probe is a list of tuples.
Each tuple refers to a probe, which must include the node and the
orientation.

The orientation units is set by ``probe_units`` argument.
in ``plot()`` method, ``data.showlegend`` is turn off to
avoid repetitive legend marks on graphs

Change .use_material() to .load_material()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have change most methods to use the word 'save' and 'load'
instead of 'get', 'use' etc.


Move "sparse" argument to run_modal()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The eigenvalue / eigenvector solver is chosen when running the run_modal() method, and is no longer a parameter from the rotor model


Bug fixes
~~~~~~~~~

Fix issue with check_units
^^^^^^^^^^^^^^^^^^^^^^^^^^

Related to Issue #511 

Add a condition to verify if the argument value is None. It allows the user to pass "None" args without getting an error with the code trying to assign a unit to a None value.


Fix missing parameters in ST_ShaftElement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* The parameters ``axial_force`` and ``torque`` were missing from the ``ST_ShaftElement.__init__()``.


Fix missing y_pos and y_pos_sup in df_seals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related to Issue #571.

Seals can be displayed in different colors than bearings

.. image:: https://user-images.githubusercontent.com/45969994/83309725-6e177a00-a1e0-11ea-95a3-8a69ebce9eda.png
   :target: https://user-images.githubusercontent.com/45969994/83309725-6e177a00-a1e0-11ea-95a3-8a69ebce9eda.png
   :alt: image


Fix ValueError when inputting an integer to disk properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related to Issue #593.

