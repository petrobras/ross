Version 1.5.0
-------------

The following enhancements and bug fixes were implemented for this release:

Enhancements
~~~~~~~~~~~~

Dev sync ucs map
^^^^^^^^^^^^^^^^

For the UCS, the eigenvalues for the synchronous case are now computed directly, eliminating the need to iterate the rotor speed.

Improve numerical integrator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Newmark method has been added as an option for the numerical integrator used in the 
.run_time_response() method. 
Also, the following modifications have been made:

* Moved the ``integrate_system()`` function from *utils.py* to be a method of the rotor in *rotor_assembly.py*.
* Added a new ``ignore`` argument for constructing the rotor's $C$ and $K$ matrices. This argument is a list of elements to be disregarded when constructing these matrices.
* Provided additional information to the ``add_to_RHS()`` function. It is now possible to access, at each time step, the ``step`` (required), and optionally, ``time_step``\ , ``disp_resp``\ , ``velc_resp``\ , and ``accl_resp``. For example:

.. code-block:: python

   # If you only need to access the step number:
   def other_forces(step, **current_state):
       ...

   # If you need to access the time step value:
   def other_forces(step, time_step=None, **curr_state):
       ...

   # If you need to access some response at the current step:
   def other_forces(step, velc_resp=None, disp_resp=None, **extra_info):
       ...

   # Run time integration:
   response = rotor.run_time_response(speed, F, t, method="newmark", add_to_RHS=other_forces)

* Enabled the possibility to pass any matrices ($M, C, K, G, K_{sdt}$) as arguments to the function, allowing them to replace the rotor matrices constructed in the code. For example:

.. code-block:: python

   # Create new matrices (be careful with the size of the arrays):
   M = np.array([...])
   C = np.array([...])
   K = np.array([...])
   G = np.array([...])
   Ksdt = np.array([...])

   # Run time integration:
   response = rotor.run_time_response(speed, F, t, method="newmark", M=M, C=C, K=K, G=G, Ksdt=Ksdt)


* Created unit tests (_test_transient\ *newmark.py*\ ) to cover the ``run_time_response(method="newmark")``.

Improve Campbell diagram for 6 dof model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improved Shape class by adding a new function to classify mode shapes into "Lateral," "Axial," or "Torsional." With the new attribute ``mode_type``\ , it is now possible to identify the mode shape in the Campbell diagram for the 6 dof model, leading to the creation of two new labels: “Axial” and “Torsional.”

Additionally, an issue was observed in the eigen solver results for certain 6 dof model cases. The ``sigma`` argument in ``las.eigs()`` function was previously set to 0, which resulted in poor accuracy. By adjusting ``sigma`` to a value slightly greater than 0, the accuracy of the results has improved. However, it is not yet clear whether the value now assumed for sigma presents good results for other cases. For more details about this parameter, refer to the solver documentation  `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs>`_.

The values in the hover info of critical speeds were fixed, and I suggested changing the x-axis of the Campbell diagram to “Rotor Speed”.

Some comparison of results are shown below:

Rotor model from tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: https://github.com/petrobras/ross/assets/82293939/4dd8d452-ae53-4b61-b90c-f6cf11d35c33
   :target: https://github.com/petrobras/ross/assets/82293939/4dd8d452-ae53-4b61-b90c-f6cf11d35c33
   :alt: 6_dof_without


.. image:: https://github.com/petrobras/ross/assets/82293939/0a52f555-27d4-455e-a6b8-930738bc056a
   :target: https://github.com/petrobras/ross/assets/82293939/0a52f555-27d4-455e-a6b8-930738bc056a
   :alt: tutorial_3_zoom


LMEst 2 disk rotor
^^^^^^^^^^^^^^^^^^


.. image:: https://github.com/petrobras/ross/assets/82293939/c941c4a6-5dca-4321-9ff5-1a419cb92bc7
   :target: https://github.com/petrobras/ross/assets/82293939/c941c4a6-5dca-4321-9ff5-1a419cb92bc7
   :alt: 4_dof


.. image:: https://github.com/petrobras/ross/assets/82293939/e6cc0f6e-45fa-4d27-a479-d437ef3b29bb
   :target: https://github.com/petrobras/ross/assets/82293939/e6cc0f6e-45fa-4d27-a479-d437ef3b29bb
   :alt: 6_dof_with


.. image:: https://github.com/petrobras/ross/assets/82293939/27c00bd2-ccc6-4386-84e7-55488228a3a6
   :target: https://github.com/petrobras/ross/assets/82293939/27c00bd2-ccc6-4386-84e7-55488228a3a6
   :alt: 6_dof_without


Gp68
^^^^


.. image:: https://github.com/petrobras/ross/assets/82293939/0abea146-0214-4caa-98e1-bb4b62f2f531
   :target: https://github.com/petrobras/ross/assets/82293939/0abea146-0214-4caa-98e1-bb4b62f2f531
   :alt: 4_dof


.. image:: https://github.com/petrobras/ross/assets/82293939/f253df06-504e-47da-9226-dbe62402d556
   :target: https://github.com/petrobras/ross/assets/82293939/f253df06-504e-47da-9226-dbe62402d556
   :alt: 6_dof_with


.. image:: https://github.com/petrobras/ross/assets/82293939/9afe52c7-f739-41e2-b3bb-194636e645b6
   :target: https://github.com/petrobras/ross/assets/82293939/9afe52c7-f739-41e2-b3bb-194636e645b6
   :alt: 6_dof_without




API changes
~~~~~~~~~~~

Change defects to faults
^^^^^^^^^^^^^^^^^^^^^^^^

We have renamed the 'defects' module to 'faults'.


Bug fixes
~~~~~~~~~

Fix copying a bearing and setting node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed a bug when the user tried to copy a bearing and then set it to a different node.

A common requirement is to run an expensive bearing calculation and then copy this bearing and just set a different node where it will be located in the rotor:

.. code-block:: python

   import ross as rs
   from copy import copy

   bearing_de = rs.SpecialBearing(n=0, ...)
   bearing_nde = copy(bearing_de)
   bearing_nde.n = 20

The above code would give an error, since in the rotor assembly we have code which relies on bearings having n_l and n_d attributes (although this only makes sense for shaft elements).

This code removes the attribute from bearing initialization and sets it at rotor initialization. This way it is possible to modify the attribute before creating the rotor.


Fix y label for frequency response plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed the y label for the frequency response plots, showing 'magnitude' instead of displacement, velocity or acceleration.


Fix use of Kst matrix
^^^^^^^^^^^^^^^^^^^^^

Corrected the stiffness matrix Kst resulting from the transient motion. This matrix should be multiplied by the acceleration, rather than the rotor speed as was previously done.


Fix disk K matrix
^^^^^^^^^^^^^^^^^

The K matrix for the disk elements should be zero. In the 6 dof model, there was confusion between the K matrix and the Kdt matrix of disk elements. This PR resolves this misunderstanding. Additionally, it proposes the incorporation of the Kdt matrix into the rotor Ksdt matrix, which represents the stiffness matrix related to transient motion of the shaft and discs (formerly Kst).


Fix 6 dof matrices
^^^^^^^^^^^^^^^^^^
* the signs of ``K`` and ``M`` matrices of the shaft elements;
* the consideration of shear effect on ``M`` matrix of the shaft elements;


Fix run_ucs for 6 dof model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Closes issues #1034 and #772

Fixed the ``run_ucs()`` method to not show the axial and torsional modes in the 6 dof model results. A new function has been added to ``utils.py`` to convert a 6 dof rotor model to a 4 dof model to support this update.

Comparisons of old UCS maps are presented below with those obtained after modification:

.. code-block:: python

   rotor.run_ucs(stiffness_range=(6, 11), num=20, num_modes=16)


.. image:: https://github.com/petrobras/ross/assets/82293939/4732d034-184c-4929-86eb-4ec27bc8c501
   :target: https://github.com/petrobras/ross/assets/82293939/4732d034-184c-4929-86eb-4ec27bc8c501
   :alt: ucs2_4dof


.. image:: https://github.com/petrobras/ross/assets/82293939/07bf3460-ce94-4171-9c6b-e81efa776856
   :target: https://github.com/petrobras/ross/assets/82293939/07bf3460-ce94-4171-9c6b-e81efa776856
   :alt: ucs2_6dof


.. image:: https://github.com/petrobras/ross/assets/82293939/7af35589-2460-4ff2-a192-df5e1db33149
   :target: https://github.com/petrobras/ross/assets/82293939/7af35589-2460-4ff2-a192-df5e1db33149
   :alt: ucs2_6dof_mod


.. code-block:: python

   rotor.run_ucs(stiffness_range=(6, 11), num=20, num_modes=24)


.. image:: https://github.com/petrobras/ross/assets/82293939/64c80098-cfb5-4425-b9c0-08a561e8a674
   :target: https://github.com/petrobras/ross/assets/82293939/64c80098-cfb5-4425-b9c0-08a561e8a674
   :alt: ucs_4dof


.. image:: https://github.com/petrobras/ross/assets/82293939/02eb60da-9442-4fbe-99a8-8906f8b423d1
   :target: https://github.com/petrobras/ross/assets/82293939/02eb60da-9442-4fbe-99a8-8906f8b423d1
   :alt: ucs_6dof


.. image:: https://github.com/petrobras/ross/assets/82293939/417c2fb9-2c84-4e0a-9743-a1f51020aa48
   :target: https://github.com/petrobras/ross/assets/82293939/417c2fb9-2c84-4e0a-9743-a1f51020aa48
   :alt: ucs_6dof_mod



