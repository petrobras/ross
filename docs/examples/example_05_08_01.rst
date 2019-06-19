
Example 1 - Number of DOF influence in Natural Frequency
========================================================

In this example, we use the rotor seen in Example 5.8.1 from ‘Dynamics
of Rotating Machinery’ by MI Friswell, JET Penny, SD Garvey & AW Lees,
published by Cambridge University Press, 2010. Which is a symmetric
rotor with a single disk in the center. The shaft is hollow with an
outside diameter of :math:`80 mm`, an inside diameter of :math:`30 mm`,
and a length of :math:`1.2 m` and it is modeled using Euler-Bernoulli
elements, with no internal shaft damping. The bearings are rigid and
short and the disk has a diameter of :math:`400 mm` and a thickness of
:math:`80 mm`. The disk and shaft elements are made of steel.

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    import ross as rs
    from bokeh.io import output_notebook
    output_notebook()
    from bokeh.io.showing import show



.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="1001">Loading BokehJS ...</span>
        </div>




.. code:: ipython3

    steel = rs.materials.steel

.. code:: ipython3

    number_of_elements = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40, 60]

.. code:: ipython3

    def create_rotor(n_el):
        """Create example rotor with given number of elements."""
        shaft = [
            rs.ShaftElement(1.2/(n_el), i_d=0.03, o_d=0.08, material=steel)
            for i in range(n_el)
        ]
    
        disks = [
            rs.DiskElement.from_geometry(n=(n_el / 2), material=steel,
                                         width=0.08, i_d=0.08, o_d=0.4)
        ]
    
        bearings = [
            rs.BearingElement(0, kxx=1e15, cxx=0),
            rs.BearingElement(n_el, kxx=1e15, cxx=0)
        ]
        
        return rs.Rotor(shaft, disks, bearings, sparse=False)

.. code:: ipython3

    def analysis(speed):
        """Perform convergence analysis for a given speed."""
        # create reference rotor with 80 elements
        rotor_80 = create_rotor(80)
        rotor_80.w = speed
        rotor_80.run_modal()
    
        n_eigen = 8 
        errors = np.zeros([len(number_of_elements), n_eigen])
    
        for i, n_el in enumerate(number_of_elements):
            rotor = create_rotor(n_el)
            rotor.w = speed
            rotor.run_modal()
            errors[i, :] = abs(
                100 * (rotor.wn[:n_eigen] - rotor_80.wn[:n_eigen]) 
                / rotor_80.wn[:n_eigen])
            
        fig, ax = plt.subplots()
        ax.set_xlabel('Number of degrees of freedom')
        ax.set_ylabel('Natural Frequency Error(%)')
        for i in range(8):
            ax.semilogy(number_of_elements, errors[:, i])

.. code:: ipython3

    analysis(speed=0)



.. image:: example_05_08_01_files/example_05_08_01_6_0.png


.. code:: ipython3

    analysis(speed=5000*np.pi/30)



.. image:: example_05_08_01_files/example_05_08_01_7_0.png


.. code:: ipython3

    rotor_10 = create_rotor(10)
    rotor_10.w = 4000*np.pi/30
    rotor_10.run_modal()

-  Campbell Diagram

.. code:: ipython3

    speed_range = np.linspace(0,4000*np.pi/30,100)
    campbell = rotor_10.run_campbell(speed_range)
    campbell.plot()




.. code-block:: python

    (<Figure size 432x288 with 2 Axes>,
     <matplotlib.axes._subplots.AxesSubplot at 0x7efd78b76780>)




.. image:: example_05_08_01_files/example_05_08_01_10_1.png

