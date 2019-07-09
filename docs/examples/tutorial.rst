
Tutorial
========

`Download This
Notebook <https://ross-rotordynamics.github.io/ross-website/_downloads/51a7c8bd4026689b99005339576b2193/tutorial.ipynb>`__

| This is a basic tutorial on how to use ross (rotordynamics open-source
  software), a Python library for rotordynamic analysis. The majority of
  this code follows object-oriented coding, which is represented in this
  `UML
  DIAGRAM <https://user-images.githubusercontent.com/32821252/50386686-131c5200-06d3-11e9-9806-f5746295be81.png>`__.
| In the following topics we are going to discuss the most relevant
  classes for a quick start to use ROSS.

.. code:: ipython3

    import ross as rs
    from bokeh.io import output_notebook
    import numpy as np
    import matplotlib.pyplot as plt
    
    output_notebook()



.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="1001">Loading BokehJS ...</span>
        </div>




Materials
---------

There is a class called Material to hold material’s properties, where:

.. code-block:: text

   name : str
       Material name.
   E : float
       Young's modulus (N/m**2).
   G_s : float
       Shear modulus (N/m**2).
   rho : float
       Density (N/m**3).

Note that, to instatiate a Material class, you only need to give 2 out
of the following parameters: ‘E’, ‘G_s’ ,‘Poisson’.

.. code:: ipython3

    Steel = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)

Saving a Material
~~~~~~~~~~~~~~~~~

To save an already instantiated Material object, you need to use the
following method.

.. code:: ipython3

    Steel.save_material()

Loading a Material
~~~~~~~~~~~~~~~~~~

To load a material, first of all, use the available_materials() method
to check if your material is available in the database, then you should
use the Material.use_material(‘name’) method with the name of the
material as a parameter.

.. code:: ipython3

    rs.Material.available_materials()




.. code-block:: text

    ['AISI4140', 'Steel']



.. code:: ipython3

    steel = rs.Material.use_material('Steel')

Element
-------

Element is an abstract class (not directly used in the program), this
class is mainly used to organize the code and make it more intuitive.

-  All the classes which derives from Element ends with Element in their
   respective names.
-  Every element is placed in a node, which is the junction of two
   elements.

ShaftElement
------------

There are two methods that you could use to model this element:

-  Euler–Bernoulli beam Theory
-  Timoshenko beam Theory (used as default)

| This Element represents the rotor’s shaft, all the other elements are
  correlated with this one.
| This class can be instantiated as the code that follows. Where (as per
  the documentation):

.. code-block:: text

   L : float
       Element length.
   i_d : float
       Inner diameter of the element.
   o_d : float
       Outer diameter of the element.
   material : ross.material
       Shaft material.
   n : int, optional
       Element number (coincident with it's first node).
       If not given, it will be set when the rotor is assembled
       according to the element's position in the list supplied to
       the rotor constructor.

.. code:: ipython3

    i_d = 0
    o_d = 0.05
    n = 6
    l_list = [0.25 for _ in range(n)]
    
    shaft_elements = [rs.ShaftElement(L=l,
                                      i_d=i_d,
                                      o_d=o_d,
                                      material=steel,
                                      shear_effects=True,
                                      rotary_inertia=True,
                                      gyroscopic=True
                                      ) for l in l_list]

DiskElement
-----------

This class represents a Disk element. We can see an example of
instantiation of this class in the following lines of code.

Where:

This class can be instantiated as the code that follows.

.. code-block:: text

   n: int
       Node in which the disk will be inserted.
   m : float
       Mass of the disk element.
   Id : float
       Diametral moment of inertia.
   Ip : float
       Polar moment of inertia

All the values are following the S.I. convention for the units.

.. code:: ipython3

    Disk = rs.DiskElement(n=0, m=32.58972765, Id=0.17808928, Ip=0.32956362)
    print(Disk)


.. code-block:: text

    DiskElement(Id=0.17809, Ip=0.32956, m=32.59, color='#bc625b', n=0)


From geometry DiskElement instantiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides the instatiation previously explained, there is a way to
instantiate a DiskElement with only geometrical parameters (for
cylindrical disks) and the material which this disk is made of, as we
can see in the following code.

.. code-block:: text

   n: int
       Node in which the disk will be inserted.
   material : lavirot.Material
        Shaft material.
   width: float
       The disk width.
   i_d: float
       Inner diameter.
   o_d: float
       Outer diameter.

.. code:: ipython3

    disk0 = rs.DiskElement.from_geometry(n=2,
                                         material=steel,
                                         width=0.07,
                                         i_d=0.05,
                                         o_d=0.28)
    disk1 = rs.DiskElement.from_geometry(n=4,
                                         material=steel,
                                         width=0.07,
                                         i_d=0.05,
                                         o_d=0.28)
    disks = [disk0,disk1]

BearingElement
--------------

| As it says on the name, this class is a bearing.
| The following code demonstrate how to properly instantiate it.

.. code-block:: text

   n: int
       Node which the bearing will be located in
   kxx: float, array
       Direct stiffness in the x direction.
   cxx: float, array
       Direct damping in the x direction.
   kyy: float, array, optional
       Direct stiffness in the y direction.
       (defaults to kxx)
   cyy: float, array, optional
       Direct damping in the y direction.
       (defaults to cxx)
   kxy: float, array, optional
       Cross coupled stiffness in the x direction.
       (defaults to 0)
   cxy: float, array, optional
       Cross coupled damping in the x direction.
       (defaults to 0)
   kyx: float, array, optional
       Cross coupled stiffness in the y direction.
       (defaults to 0)
   cyx: float, array, optional
       Cross coupled damping in the y direction.
       (defaults to 0)
   w: array, optional
       Array with the speeds (rad/s).

P.S.: Note that the coefficients could be an array with different
coefficients for different rotation speeds, in that case you only have
to give a parameter ‘w’ which is a array with the same size as the
coefficients array.

.. code:: ipython3

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = rs.BearingElement(n=0, kxx=stfx, kyy=stfy, cxx=1e3, w=np.linspace(0,200,101))
    bearing1 = rs.BearingElement(n=6, kxx=stfx, kyy=stfy, cxx=1e3, w=np.linspace(0,200,101))
    bearings = [bearing0, bearing1]

Instantiating bearings from excel archives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There’s a class method to instantiate a bearing from excel tables, as we
can see in the following code.

**There will be a class method to instantiate a bearing from excel
tables. - work in progress**

Rotor
-----

This class takes as argument lists with all elements program and
assembles the mass, damping and stiffness global matrices for the
system. It also outputs all the results classes obtained by the
simulation.

To use this class, you only have to give all the already instantiated
elements in a list format, as it follows.

.. code:: ipython3

    rotor1 = rs.Rotor(shaft_elements,
                      disks,
                      bearings 
                      )

From section instantiation of a Rotor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| In this form of instantiation, the number of shaft elements used in
  FEM are not fixed, instead, the program does a convergence analysis,
  testing the number of elements to a point where the relative error
  between iterations reaches a value that can be neglected.
| To use this method, you should divide the rotor in a way where the
  number of shaft elements is minimal and place every element (except
  for the shaft elements) in the minimal nodes

.. code:: ipython3

    i_d = 0
    o_d = 0.05
    
    i_ds_data = [0,0,0]
    o_ds_data = [0.05, 0.05, 0.05]
    leng_data = [0.5, 0.5, 0.5]
    
    stfx = 1e6
    stfy = 0.8e6
    bearing0 = rs.BearingElement(n=0, kxx=stfx, kyy=stfy, cxx=1e3, w=np.linspace(0,200,101))
    bearing1 = rs.BearingElement(n=3, kxx=stfx, kyy=stfy, cxx=1e3, w=np.linspace(0,200,101))
    bearings = [bearing0, bearing1]
    
    disk0 = rs.DiskElement.from_geometry(n=1,
                                         material=steel,
                                         width=0.07,
                                         i_d=0.05,
                                         o_d=0.28
                                        )
    disk1 = rs.DiskElement.from_geometry(n=2,
                                         material=steel,
                                         width=0.07,
                                         i_d=0.05,
                                         o_d=0.28
                                        )
    disks = [disk0,disk1]
    
    rotor2 = rs.Rotor.from_section(brg_seal_data=bearings,
                                   disk_data=disks,
                                   i_ds_data=i_ds_data,
                                   leng_data=leng_data,
                                   o_ds_data=o_ds_data, 
                                  )


Visualizing the Rotor
~~~~~~~~~~~~~~~~~~~~~

It is interesting to plot the rotor to check if the geometry checks with
what you wanted to instantiate, you can plot it with the following code.

Note: For almost every plot functions, there are two options for plots,
one with bokeh library and one with matplotlib.

.. code:: ipython3

    rotor1.plot_rotor()




.. code-block:: text

    <matplotlib.axes._subplots.AxesSubplot at 0x...>




.. image:: tutorial_files/tutorial_36_1.png


Running the simulation
~~~~~~~~~~~~~~~~~~~~~~

After you verify that everything is fine with the rotor, you should run
the simulation and obtain results. To do that you only need to use the
one of the ``run_()`` methods available, as shown in like the code
bellow.

.. code:: ipython3

    rotor1.run_modal()
    rotor2.run_modal()

Obtaining results
-----------------

These are the following analysis you can do with the program: - Static
analysis - Campbell Diagram - Frequency response - Forced response -
Mode Shapes

Static analysis
~~~~~~~~~~~~~~~

This method gives a free body diagram and a amplificated plot of the
rotor response to gravity effects.

.. code:: ipython3

    static = rotor1.run_static()

Campbell Diagram
~~~~~~~~~~~~~~~~

In this example we can see the campbell diagram from 0 to 4000 RPM.

.. code:: ipython3

    campbell = rotor1.run_campbell(np.linspace(0,200,101))
    campbell.plot()




.. code-block:: text

    (<Figure size 432x288 with 2 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x...>)




.. image:: tutorial_files/tutorial_42_1.png


Frenquency Response
~~~~~~~~~~~~~~~~~~~

We can put the frequency response of selecting the input and output
node.

.. code:: ipython3

    rotor1.run_freq_response().plot(inp=0,out=0)
    plt.rcParams["figure.figsize"] = (15,10)



.. image:: tutorial_files/tutorial_44_0.png


Mode Shapes
~~~~~~~~~~~

You can also generate the plot for each mode shape.

.. code:: ipython3

    modes = rotor1.run_mode_shapes()
    modes.plot(0)




.. code-block:: text

    (<Figure size 1080x...xes>,
     <matplotlib.axes._subplots.Axes3DSubplot at 0x...>)




.. image:: tutorial_files/tutorial_46_1.png

