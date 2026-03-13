Version 1.0.0
--------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Add loop to calculate frequency-dependent coefficients in from_fluid_flow()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users will be able to input as many frequency values as needed and ``from_fluid_flow()`` will run a loop calculating a set of coefficients for each given frequency.
Now, ``omega`` argument must be a list of frequencies.
It also changes the main function used to calculate the coefficients. ``calculate_stiffness_and_damping_coefficients()`` is now used to do it numerically.


Extract fluid flow
^^^^^^^^^^^^^^^^^^

The .from_fluid_flow method has been extracted from the BearingElement to a class
called BearingFluidFlow.
This is more in line with how we have all the Bearing classes, and
makes it easier for the user to find it.

Add minor and major axes to response plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now select the major and minor axis when plotting responses:

.. code-block:: python

    # plot response for major or minor axis:
    >>> probe_node = 3
    >>> probe_angle = "major"   # for major axis
    >>> # probe_angle = "minor" # for minor axis
    >>> probe_tag = "my_probe"  # optional
    >>> fig = response.plot(probe=[(probe_node, probe_angle, probe_tag)])

 Calculate velocity and acceleration responses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method ``run_freq_response()`` now returns, not only the displacement results,
but also the velocity and acceleration. The data is then input to
``FrequencyResponseResults`` and ``ForcedResponseResults`` classes.

The user is able to choose the response by changing the amplitude_units argument.
 * Inputing '[length]/[force]', it displays the displacement;
 * Inputing '[speed]/[force]', it displays the velocity;
 * Inputing '[acceleration]/[force]', it displays the acceleration.

Add units to stochastic ross module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Units use is now possible in the stochastic module.

Crack, rubbing and misalignment simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is now possible to run simulations for some defects such as cracks, rubbing
and misalignment.

API changes
~~~~~~~~~~~

Change methods to `run_`
^^^^^^^^^^^^^^^^^^^^^^^^

The plot_level1 and plot_ucs methods were refactored so that they behave as other run methods.
Now we use the run* prefix to call them and they will return Results objects.

.. code-block:: python

    ucs = rotor.run_ucs(...)
    ucs.plot(...)

Bug fixes
~~~~~~~~~

Fix GlobalIndex with non integer values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PointMass`` and ``BearingElements`` with ``n_link`` argument were getting non integer values.
Matrices would not be built, once these values are used as reference to array indexes
(which must be integers).

Fix DoF error when using "n_link"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Add ``link_nodes`` attribute to Rotor object that holds the nodes created using ``n_link``
* Correct DoFs with ``link_nodes`` in ``ForcedResponseResults`` and ``TimeResponseResults``
