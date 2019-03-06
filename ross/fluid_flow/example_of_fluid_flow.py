"""Exemple of fluid flow
These are the data related to the fluid flow problem
"""

from ross.fluid_flow import fluid_flow as flow
import numpy as np
import matplotlib.pyplot as plt

"""Grid
Describes the discretization of the problem
Parameters
----------
nz: int
    Number of points along the Z direction (direction of flow).
ntheta: int
    Number of points along the direction theta. NOTE: ntheta must be odd!
nradius: int
    Number of points along the direction r.
n_interv_z: int
    Number of intervals on Z.
n_interv_theta: int
  Number of intervals on theta.
n_interv_radius: int
    Number of intervals on r.
lb: float
    Length in the Z direction (m).
ltheta: float
    Length in the theta direction (rad).
dz: float
  Range size in the Z direction.
dtheta: float
    Range size in the theta direction.
ntotal: int
    Number of nodes in the grid.
"""

nz = 150
"""Examples
   --------
>>>nz = 100
>>>nz = 30
"""

ntheta = 37
"""Examples
   --------
>>>ntheta = 17
>>>ntheta = 48
"""

nradius = 11
"""Examples
   --------
>>>nradius = 30
"""

n_interv_z = nz-1
n_interv_theta = ntheta-1
n_interv_radius = nradius-1
lb = 1.
ltheta = 2.*np.pi
dz = lb/n_interv_z
dtheta = ltheta/n_interv_theta
ntotal = nz*ntheta

"""Operation conditions
Describes the operation conditions.
Parameters
----------
omega: float
    Rotation of the rotor (rad/s).
p_in: float
    Input Pressure (Pa).
o_out: float
    Output Pressure (Pa).
"""

omega = -100.*2*np.pi/60
"""Examples
>>>omega = 5*np.pi/3
"""

p_in = 392266.
"""Examples
>>>p_in = 192244
"""

p_out = 100000.
"""Examples
>>>p_out = 392266.
"""

"""Geometric data of the problem
Describes the geometric data of the problem.
Parameters
----------
radius_valley: float
    Smallest rotor radius (m).
radius_crest: float
    Larger rotor radius (m).
radius_stator: float
    Stator Radius (m).
lwave: float
    Rotor step (m) (sine wave length).
xe and ye: float
    Eccentricity (m) (distance between rotor and stator centers).
"""

radius_valley = 0.034
"""Examples
>>>radius_valley = 0.036
"""

radius_crest = 0.039
"""Examples
>>>radius_crest = 0.037
"""

radius_stator = 0.04

lwave = 0.18
"""Examples
>>>lwave = 0.059995
"""

xe = 0.
"""Examples
>>>xe = 0.01
>>>xe = 0.0015
>>>xe = 0.002
"""

ye = 0.
"""Examples
>>>ye = 0.01
>>>ye = 0.0015
>>>ye = 0.002
"""

"""Fluid characteristics
Describes the fluid characteristics.
Parameters
----------
visc: float
    Viscosity (Pa.s).
rho: float
    Fluid density(Kg/m^3).
Examples
--------
>>>WATER = (vics=0.001, rho=1000.)
>>>Purolub 46 = (vics=0.042, rho=868.)
>>>Purolub 150 = (vics=0.433, rho=885.)
>>>ISO VG 32 (40ºC) = (vics=0.032, rho=857.)
>>>ISO VG 32 (100°C) = (vics=0.0054, rho=857.)
>>>ISO VG 46 (40°C) = (vics=0.046, rho=861.)
ISO VG 46 (100°C) = (vics=0.0068, rho=861.)
"""

visc = 0.001
rho = 1000.

if __name__ == "__main__":
    my_pressure_matrix = flow.PressureMatrix(
        nz, ntheta, nradius, n_interv_z, n_interv_theta, n_interv_radius, lb,
        ltheta, dz, dtheta, ntotal, omega, p_in, p_out, radius_valley,
        radius_crest, radius_stator, lwave, xe, ye,  visc, rho
        )

    P = my_pressure_matrix.calculate_pressure_matrix()
    my_pressure_matrix.plot_pressure_z()
    my_pressure_matrix.plot_shape()
    my_pressure_matrix.plot_pressure_theta()
    plt.show()
