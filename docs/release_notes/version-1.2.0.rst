Version 1.2.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Added Shape and Orbit classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have added a ``Shape`` and an ``Orbit`` class to the ``results`` module.
These classes are now responsible for calculating the mode shapes in the 
``ModalResults`` class and also the deflected shapes in the 
``ForcedResponseResults`` class. 
This simplifies the code since it avoids the repetition of code between these 
two classes. It also selects a single method for calculation of the major/minor
axes and precession (forward or backward). Before, in the ``ModalResults``\ , we 
were doing this using the method described by Friswell with the eigenvalues of 
the H matrix, and in the ``ForcedResponseResults`` we were using a calculation 
based on forward and backward vectors. The current code in the ``Orbit`` uses the 
method described by Friswell.

Bug fixes
~~~~~~~~~

Fix ucs map
^^^^^^^^^^^

Fix calculation of bearing stiffness for plotting 

Since we have removed the Coefficient class we no longer calculate the
bearing coefficient with bearing.kxx.interpolated. Now we use
bearing.kxx_interpolated.

Fix number of modes

It was only possible to plot 4 modes. Now we can change the num_modes
which is the number of calculated modes, and we will have that value
divided by 4 being plotted.
Value is divided by 4 because for each pair of eigenvalues calculated
we have one wn, and we show only the forward mode in the plots,
therefore we have num_modes / 2 / 2.


Fix stiffness expression for tapered element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the stiffness expression described on appendix A of the Genta and
Gugliotta paper, when substituting chi = 12\ *E*\ I/(phi\ *G*\ A*L**2) in the
constant that multiplies the second matrix, we had canceled the A here
with the Aj from the equation, but they are actually not the same area.
The area Aj in the equation refers to the area in the element's left
side, while the area in the chi calculation refers to the area in the
middle of the element. Therefore, we were having an error in this
formula by a factor of Aj / A.
