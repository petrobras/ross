Version 1.6.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Fix plot torsional and axial modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corrections related to the plotting of mode shapes and includes new methods to handle the plotting of torsional and axial modes in the 6-DoF model.

The ``plot_with_mode_shape`` method was not functioning due to changes in the ``frequency_units`` parameter in the ``ModalResults`` class from previous PRs. To resolve this, corrections were made to manage ``speed_units`` separately from ``frequency_units``.

Additionally, new methods were added to the Shape class, including ``_plot_torsional`` and ``_plot_axial``\ , which are now integrated into the ``plot_mode_2d`` and ``plot_mode_3d`` methods of ``ModalResults``. For 3D plots, an optional animation feature was added, which can be activated by passing ``animation=True`` as an argument.


Added CouplingElement
^^^^^^^^^^^^^^^^^^^^^

Added a CouplingElement class that creates a coupling element from input data 
of inertia and mass from the left station and right station, and also translational and rotational
stiffness and damping values. The matrices will be defined considering the
same local coordinate vector of the `ShaftElement`.

API changes
~~~~~~~~~~~

Change the X Axis (Rotor Speed) in Campbell Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, it's possible change the x axis (rotor speed) with variable ``speed_units`` in plot().

.. code-block:: python

   import ross as rs
   import numpy as np

   Q_ = rs.Q_
   rotor = rs.rotor_example()
   speed = np.linspace(0, 400, 101)
   camp = rotor.run_campbell(speed)
   fig = camp.plot(
           harmonics=[1, 2],
           damping_parameter="damping_ratio",
           frequency_range=Q_((2000, 10000), "RPM"),
           damping_range=(-0.1, 100),
           frequency_units="Hz",
           speed_units="RPM",
           )

This PR resolves issue #1087.


Bug fixes
~~~~~~~~~

