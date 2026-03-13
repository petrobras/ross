Version 1.1.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Removed python 3.6 support
^^^^^^^^^^^^^^^^^^^^^^^^^

Since we are now using pint 0.18 which requires python 3.7+, we can no
longer support python 3.6.

Add damping parameter and filtering to campbell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three additional arguments have been added to the Campbell plot:

* damping_parameter: choose between log_dec or damping_ratio;
* frequency_range: filter out modes that are out of this frequency range;
* damping_range: filter out modes that are out of this damping range;

Improve bearing save method
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save and load methods for bearings have been modified in order to have
additional info in the save file, such as attributes used in bearing
initialization.


API changes
~~~~~~~~~~~

Change pkpk parameter used in plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The peak to peak parameter (pkpk) has been changed from prefix to constant,
which simplifies it use. To use peak to peak use ‘<unit> pkpk’ (e.g. ‘m pkpk’)

Remove coefficient class
^^^^^^^^^^^^^^^^^^^^^^^^

Remove the auxiliary _Coefficient class in favor of a more simple data
processing within the BearingElement class.
The _Coefficient class was used to process the coefficient data,
creating the interpolation functions and also providing plots (e.g.
``bearing.kxx.plot()``\ ). This was making the code more complicated since we
basically had to reproduce the methods of an iterable object in this
class. The discoverability of the plot function was also improved,
since it is more natural for the user to search for a plot function
within the bearing element and not as an attribute of each coefficient
(\ ``bearing.plot('kxx')`` instead of ``bearing.kxx.plot()`` is now used).
Another improvement regarding the plot is that the user can now pass
multiple coefficients to plot with.
Before the user would have to do something like:

.. code-block::

   fig = bearing.kxx.plot()
   fig = bearing.kyy.plot(fig=fig)
   fig = bearing.kxy.plot(fig=fig)
   fig = bearing.kyx.plot(fig=fig)

Now we can do:

.. code-block::

    bearing.plot(coefficients=['kxx', 'kyy', 'kxy', 'kyx'])


Bug fixes
~~~~~~~~~

Correct an error in the damping coefficient of the Magnetic Bearings Element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A small error in the magnetic bearing calculation has been corrected. See #816.

Fix error in static analysis for bearings with high stiffness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This adds an auxiliary rotor with 0 stiffness in the bearing location
to fix calculations, otherwise, a rotor with a bearing with
high stiffness would have 0 reaction force at the bearing location.


Fix error in static analysis for bearings and disks added to the same node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For rotors with a bearing and a disk added to the same node the static analysis would fail.
See #845 for more details.


Add check units to run modal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run modal was not working with units. Now we can use `rotor.run_modal(speed=Q_(1000, "RPM")`.
