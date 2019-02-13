import numpy as np
import matplotlib as plt
from ross.bearing_seal_element import BearingElement 
from ross.disk_element import DiskElement            
from ross.shaft_element import ShaftElement          
from ross.materials import Material                  
from ross.rotor_assembly import Rotor                

steel = Material(name='steel', E=211e9,G_s=81.2e9, rho=7810)

class Convergence(object):
    
    r"""A rotor class 
    
    This class will verify the eigenvalues calculation, done by
    "rotor_assembly.py" and check its convergence to minimize the numerical 
    errors.
    Regions are not elements. Each region will have n elements with 
    equal lenght
    
    Parameters
    ----------
    leng_data : list
        List with the lengths of rotor regions.
    o_d_data : list
        List with the outer diameters of rotor regions.
    i_d_data : list 
        List with the inner diameters of rotor regions.
    disk_data : list, optional
        List holding lists of disks datas.
        Example : disk_data = [[n, material, width, i_d, o_d], [n, ...]]
        ***See 'disk_element.py' docstring for more information***  
    brg_seal_data : dictionary, optional
        list holding lists of bearings and seals datas.
        Example : brg_seal_data = [[n, kxx, cxx, kyy=None, kxy=0, kyx=0, 
                                    cyy=None, cxy=0, cyx=0, w=None], [n, ...]]
        ***See 'bearing_seal_element.py' docstring for more information***
    w : float, optional
        Rotor speed.
    nel_r : int
        Initial number or elements per shaft region.
        Default is 1
    eigval : int
        Indicates which eingenvalue convergence to check.
        default is 7 (7th eigenvalue).
    err_max : float, optional
        maximum allowed for eigenvalues calculation.
        default is 0.01 (or 1%).
              
    Returns
    -------
    el_num : array    
        Number or elements in each trial
    eigv_arr : array
        The nth eigenvalue calculated for each trial
    error_arr : array
        Relative error between the two least analysis with different number of
        elements
        
    Example
    -------
    >>> converg = Convergence(leng_data=[0.5,0.5,0.5],
                              o_ds_data=[0.05,0.05,0.05],
                              i_ds_data=[0,0,0],
                              disk_data=[[1, steel, 0.07, 0, 0.28], 
                                         [2, steel, 0.07, 0, 0.35]], 
                              brg_seal_data=[[0, 1e6, 0, 1e6, 0,0,0,0,0,None],
                                             [3, 1e6, 0, 1e6,0,0,0,0,0,None]],
                              w=0, nel_r=2, eigval=1, err_max=0.01))
    
    >>> converg.rotor_regions(2)
    [[ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material=steel, n=None),
      ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material=steel, n=None),
      ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material=steel, n=None),
      ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material=steel, n=None),
      ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material=steel, n=None),
      ShaftElement(L=0.25, i_d=0.0, o_d=0.05, material=steel, n=None)],
    [<disk_element.DiskElement at 0x17132ec8fd0>,
     <disk_element.DiskElement at 0x17132ec86a0>],
     [BearingElement, BearingElement]]
    
    >>> converg.conv_analysis()
    (array([6, 9]),
     array([85.76232175, 85.76225176]),
     array([0.00000000e+00, 8.16102012e-07]))
    
    """
    
    def __init__(self, 
                 leng_data=list, 
                 o_ds_data=list,
                 i_ds_data=list,
                 disk_data = None,
                 #pos_disk=None,              
                 brg_seal_data = None,
                 #pos_bearing=None,             
                 w=0,
                 nel_r=1,
                 eigval=7,
                 err_max=0.01
                 ):
        
        #each item added to these three lists below means a new region created
        self.leng_data = leng_data
        self.o_ds_data = o_ds_data
        self.i_ds_data = i_ds_data
        
        r"""
        
        Instructions to positioning elements:
            
        it's advised to position these elements in the beggining of each region,
        i.e., each disk, bearing or seal, should mark the beggining of a new 
        region to the rotor, so each disk, bearing and seal nodes matches the 
        shaft elements
        
        """
        self.disk_data = disk_data
        #self.pos_disk = pos_disk
        self.brg_seal_data = brg_seal_data
        #self.pos_bearing = pos_bearing
        self.eigval = eigval
        self.err_max = err_max
        self.w = w
        
        self.regions = []
        self.nel_r = nel_r
        
        #check if all regions lists have the same lenght
        if len(self.leng_data) != (len(self.o_ds_data) or len(self.o_ds_data)):
            raise ValueError('The matrices lenght do not match')
    
    def rotor_regions(self, nel_r=1):
        
        regions = self.regions
        regions = []
        shaft_elm = []
        disk_elm = []
        brng_elm = []
        #nel_r = initial number of elements per regions
       
        #loop through rotor regions    
        for i in range(len(self.leng_data)): 
            
            le = self.leng_data[i]/nel_r
            o_ds = self.o_ds_data[i]
            i_ds = self.i_ds_data[i]
            
            for j in range(nel_r): #loop to generate n elements per region
                 shaft_elm.append(ShaftElement(le, i_ds, o_ds, material=steel,
                                               shear_effects=True,
                                               rotary_inertia=True,
                                               gyroscopic=True))
        
        regions.extend([shaft_elm])

        for i in range(len(self.leng_data)):   
            for j in range(len(self.disk_data)):
                if self.disk_data != None and len(self.disk_data[j]) == 5 and i == self.disk_data[j][0]:
                    disk_elm.append(DiskElement.from_geometry(n=i*nel_r, 
                                                              material=self.disk_data[j][1], 
                                                              width=self.disk_data[j][2], 
                                                              i_d=self.disk_data[j][3], 
                                                              o_d=self.disk_data[j][4]))
                
        for i in range(len(self.leng_data)):   
            for j in range(len(self.disk_data)):
                if self.disk_data != None and len(self.disk_data[j]) == 4 and i == self.disk_data[j][0]:
                    disk_elm.append(DiskElement(n=i*nel_r, 
                                                m=self.disk_data[j][1], 
                                                Id=self.disk_data[j][2], 
                                                Ip=self.disk_data[j][3]))
            
        for i in range(len(self.leng_data)+1):   
            for j in range(len(self.brg_seal_data)):
                if self.brg_seal_data != None and i == self.brg_seal_data[j][0]:            
                    brng_elm.append(BearingElement(n=i*nel_r, 
                                                   kxx=self.brg_seal_data[j][1], 
                                                   cxx=self.brg_seal_data[j][2],
                                                   kyy=self.brg_seal_data[j][3], 
                                                   kxy=self.brg_seal_data[j][4],
                                                   kyx=self.brg_seal_data[j][5],
                                                   cyy=self.brg_seal_data[j][6],
                                                   cxy=self.brg_seal_data[j][7],
                                                   cyx=self.brg_seal_data[j][8],
                                                   w=self.brg_seal_data[j][9]))
                
        regions.append(disk_elm)
        regions.append(brng_elm)
        self.regions = regions
        
        return regions
        
    def conv_analysis(self):
            
        regions = self.regions
        w = self.w
        nel_r = self.nel_r
        
        el_num = np.array([nel_r*len(self.leng_data)])
        eigv_arr = np.array([])
        error_arr = np.array([0])
        
        rotor0 = Rotor(regions[0],regions[1],regions[2], w=w, n_eigen=16)
        eigv_arr = np.append(eigv_arr, rotor0.wn[self.eigval])

        error = 1  #this value is up to start the loop while
        nel_r = 2
        
        while error > self.err_max:
                
            regions = self.rotor_regions(nel_r)
            rotor = Rotor(regions[0], regions[1], regions[2], w=w, n_eigen=16)
             
            eigv_arr = np.append(eigv_arr, rotor.wn[self.eigval])
            el_num = np.append(el_num, nel_r*len(self.leng_data))
            
            error = min(eigv_arr[-1], eigv_arr[-2]) / max(eigv_arr[-1], eigv_arr[-2])
            error = 1 - error
            error_arr = np.append(error_arr, error)
                        
            nel_r *= 2
        
        return el_num, eigv_arr, error_arr
        
        # eigenvalue graph plot    
        ax = np.linspace(0, el_num[-1], len(el_num))
        ay = eigv_arr
        
        plt.figure()
        plt.plot(ax, ay)
        plt.ylabel('Relative error')
        plt.xlabel('Number of elements')
        plt.show()
        
        # relative error graph plot
        az = error_arr
        plt.figure()
        plt.plot(ax, az)
        plt.ylabel('Relative error')
        plt.xlabel('Number of elements')
        plt.show()
        
        
