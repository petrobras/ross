# Todo: check PEP 8

from ross.fluid_flow import fluid_flow as flow
import numpy as np
import matplotlib.pyplot as plt

# These are the data related to the fluid flow problem

# GRID:
# Number of points along the Z direction (direction of flow):
#nz = 150
#nz = 100
nz = 30

# Number of points along the direction THETA:
# NOTE: ntheta must be odd!
#ntheta = 37
#ntheta = 17
ntheta = 48

# Number of points along the direction r:
# nradius = 11
nradius = 30

# Number of intervals on Z:
n_interv_z = nz-1

# Number of intervals on THETA:
n_interv_theta = ntheta-1

# Number of intervals on r:
n_interv_radius = nradius-1

# Length in the Z direction (m):
lb = 1.

# Length in the THETA direction (rad):
ltheta = 2.*np.pi

# Range size in the Z direction:
dz = lb/n_interv_z

# Range size in the THETA direction:
dtheta = ltheta/n_interv_theta

# Number of nodes in the grid:
ntotal = nz*ntheta

###########################################################################
# OPERATION CONDITIONS

# Rotation of the rotor (rad/s):
omega = -100.*2*np.pi/60
#omega = 5*np.pi/3

# Input Pressure (Pa):
p_in = 392266.
#p_in=192244

# Output Pressure (Pa):
p_out = 100000.
# p_out = 392266.

###########################################################################
# GEOMETRIC DATA OF THE PROBLEM

# Smallest rotor radius (m):
radius_valley = 0.034
#radius_valley = 0.036

# Larger rotor radius (m):
radius_crest = 0.039
#radius_crest = 0.037

# Stator Radius (m):
radius_stator = 0.04

# Rotor step (m) (sine wave length):
lwave = 0.18
#lwave = 0.059995

# Eccentricity (m) (distance between rotor and stator centers):
# xe = 0.
# xe = 0.01
# xe = 0.0015
xe = 0.002

# ye = 0.
# ye = 0.01
# ye = 0.0015
ye = 0.002

###########################################################################
# FLUID CHARACTERISTICS:

# Viscosity (Pa.s):
visc=0.001 # Water
#visc=0.042 # Purolub 46
#visc=0.433 # Purolub 150

# Fluid density(Kg/m^3):
rho = 1000. # agua
#rho = 868. # Purolub 46
# rho=885. # Purolub 150


if __name__ == "__main__":
    my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, n_interv_z, n_interv_theta, n_interv_radius, lb, ltheta, dz,
                                             dtheta, ntotal, omega, p_in, p_out, radius_valley, radius_crest, radius_stator, lwave, xe,
                                             ye, visc, rho)
    P = my_pressure_matrix.calculate_pressure_matrix()
    # print(P)
    my_pressure_matrix.plot_pressure_z()
    my_pressure_matrix.plot_shape()
    my_pressure_matrix.plot_pressure_theta()
    plt.show()
