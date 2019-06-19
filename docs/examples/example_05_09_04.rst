
Example 5 - Cross-coupled bearings.
===================================

In this example, we use the rotor seen in Example 5.9.4 from ‘Dynamics
of Rotating Machinery’ by MI Friswell, JET Penny, SD Garvey & AW Lees,
published by Cambridge University Press, 2010.

This system is the same as that of Example 3, except that some coupling
is introduced in the bearings between the x and y directions. The
bearings have direct stiffnesses of :math:`1 MN/m` and cross-coupling
stiffnesses of :math:`0.5 MN/m`.

.. code:: ipython3

    from bokeh.io import output_notebook
    import ross as rs
    import numpy as np
    output_notebook()



.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="1001">Loading BokehJS ...</span>
        </div>




.. code:: ipython3

    #Classic Instantiation of the rotor
    shaft_elements = []
    bearing_seal_elements = []
    disk_elements = []
    Steel = rs.steel
    for i in range(6):
        shaft_elements.append(rs.ShaftElement(L=0.25, material=Steel, n=i, i_d=0, o_d=0.05))
        
    disk_elements.append(rs.DiskElement.from_geometry(n=2,
                                                      material=Steel, 
                                                      width=0.07,
                                                      i_d=0.05, 
                                                      o_d=0.28
                                                     )
                        )
    
    disk_elements.append(rs.DiskElement.from_geometry(n=4,
                                                      material=Steel, 
                                                      width=0.07,
                                                      i_d=0.05, 
                                                      o_d=0.35
                                                     )
                        )
    
    bearing_seal_elements.append(rs.BearingElement(n=0, kxx=1e6, kyy=1e6, kxy=.5e6, cxx=0, cyy=0))
    bearing_seal_elements.append(rs.BearingElement(n=6, kxx=1e6, kyy=1e6, kxy=.5e6, cxx=0, cyy=0))
    
    rotor594c = rs.Rotor(shaft_elements=shaft_elements,
                         bearing_seal_elements=bearing_seal_elements,
                         disk_elements=disk_elements,n_eigen = 12)
    
    rotor594c.plot_rotor()




.. code-block:: python

    <matplotlib.axes._subplots.AxesSubplot at 0x7fc98fed70f0>




.. image:: example_05_09_04_files/example_05_09_04_2_1.png


.. code:: ipython3

    #From_section class method instantiation.
    bearing_seal_elements = []
    disk_elements = []
    shaft_length_data = 3*[0.5]
    i_d = 3*[0]
    o_d = 3*[0.05]
    
    disk_elements.append(rs.DiskElement.from_geometry(n=1,
                                                      material=Steel, 
                                                      width=0.07,
                                                      i_d=0.05, 
                                                      o_d=0.28
                                                     )
                        )
    
    disk_elements.append(rs.DiskElement.from_geometry(n=2,
                                                      material=Steel, 
                                                      width=0.07,
                                                      i_d=0.05, 
                                                      o_d=0.35
                                                     )
                        )
    bearing_seal_elements.append(rs.BearingElement(n=0, kxx=1e6, kyy=1e6, cxx=0, cyy=0))
    bearing_seal_elements.append(rs.BearingElement(n=3, kxx=1e6, kyy=1e6, cxx=0, cyy=0))
    
    rotor594fs = rs.Rotor.from_section(brg_seal_data=bearing_seal_elements,
                                       disk_data=disk_elements,leng_data=shaft_length_data,
                                       i_ds_data=i_d,o_ds_data=o_d
                                      )
    rotor594fs.plot_rotor()





.. code-block:: python

    <matplotlib.axes._subplots.AxesSubplot at 0x7fc98eeb7ba8>




.. image:: example_05_09_04_files/example_05_09_04_3_1.png


.. code:: ipython3

    #Obtaining results for w=0 (wn is in rad/s)
    
    
    print('Normal Instantiation =', rotor594c.wn)
    print('\n')
    print('From Section Instantiation =', rotor594fs.wn)


.. code-block:: python

    Normal Instantiation = [ 86.65808619  86.65814251 274.31285373 274.31285411 716.78471745
     716.78790695]
    
    
    From Section Instantiation = [ 86.65926451  86.65926451 274.37573752 274.37573752 718.87267817
     718.87267818]


.. code:: ipython3

    #Obtaining results for w=4000RPM (wn is in rad/s)
    
    rotor594c.w=4000*np.pi/30
    
    print('Normal Instantiation =', rotor594c.wn)


.. code-block:: python

    Normal Instantiation = [ 84.19790045  89.03269396 246.44121183 301.08623205 594.81950178
     834.0420378 ]

