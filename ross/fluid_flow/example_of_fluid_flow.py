"""Example of fluid flow
These are the data related to the fluid flow problem.
You may run this code to quickly see how the fluid flow routine works.
"""

from ross.fluid_flow import fluid_flow as flow
import numpy as np
import matplotlib.pyplot as plt

nz = 150
ntheta = 37
nradius = 11
lb = 1.
omega = -100.*2*np.pi/60
p_in = 392266.
p_out = 100000.
radius_valley = 0.034
radius_crest = 0.039
radius_stator = 0.04
lwave = 0.18
xe = 0.
ye = 0.
visc = 0.001
rho = 1000.

if __name__ == "__main__":
    my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, lb, omega, p_in, p_out, radius_valley,
                                             radius_crest, radius_stator, lwave, xe, ye,  visc, rho,
                                             plot_eccentricity=True)
    P = my_pressure_matrix.calculate_pressure_matrix()
    my_pressure_matrix.plot_pressure_z(show_immediately=False)
    my_pressure_matrix.plot_shape(show_immediately=False)
    my_pressure_matrix.plot_pressure_theta(z=int(nz/2), show_immediately=False)
    plt.show()
