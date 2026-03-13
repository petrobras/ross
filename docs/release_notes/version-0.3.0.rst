Version 0.3.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Remove ShaftElement and replace with the ShaftTaperedElement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we only have one class for the shaft element which is capable of handling
cylindrical and tapered elements.


Allow user choose material in "Rotor.from_section()"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As requested in Issue #401, the user will be able to define a material when using ``.from_section()`` classmethod.
In addition, more modifications have been done regarding the new ShaftElement API.
Now, the user has the option to instantiate tapered sections.

CoAxialRotor class
^^^^^^^^^^^^^^^^^^^^^^

A class is now available for modeling coaxial rotors.

Bug fixes
~~~~~~~~~

Fix links not working for more than one support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Related to Issue #382.

We were having problems defining the global indexes for the elements using this method internally in an element class.
It makes more sense to use .dof_global_index() when build the rotor model.
The routine to calculate the elements' global indexes have been moved to rotor_assembly (\ ``.__init__()`` method).
Now, each element has an attribute that keeps those index, instead of calling a function to this job.


Fix bearing position bug
^^^^^^^^^^^^^^^^^^^^^^^^

Fix bug mentioned in Issue #276

Sorting the bearing_seal_elements make it to cycle correctly while assuming ``y_pos`` values to them.


