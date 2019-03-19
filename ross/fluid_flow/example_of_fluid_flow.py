"""Example of fluid flow
These are the data related to the fluid flow problem.
You may run this code to quickly see how the fluid flow routine works.
"""

from ross.fluid_flow import fluid_flow as flow
import numpy as np
import matplotlib.pyplot as plt

nz = 20
ntheta = 100
nradius = 11
length = 0.01
omega = -100.*2*np.pi/60
p_in = 1.
p_out = 1.
radius_rotor = 0.08
radius_stator = 0.1
xi = 0.007
yi = -0.007
visc = 0.015
rho = 860.


def calculate_pressure(epi, C, R, phi, mi, om, pa):
    """ Calculates pressure according to Tribology Series 33, page 119.
    """
    return ((6*mi*om*(R/C)*(R/C))/(1 - epi*epi)**(3/2))\
        * (phi - epi*np.sin(phi) - (2*phi - 4*epi*np.sin(phi) + epi*epi*phi + epi*epi*np.sin(phi)*np.cos(phi))
            / (2 + epi*epi)) + pa


if __name__ == "__main__":
    my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in, p_out, radius_rotor,
                                             radius_stator, xi, yi,  visc, rho,
                                             plot_eccentricity=True)
    P = my_pressure_matrix.calculate_pressure_matrix()
    my_pressure_matrix.plot_pressure_z(show_immediately=False)
    my_pressure_matrix.plot_shape(show_immediately=False)
    my_pressure_matrix.plot_pressure_theta_cylindrical(z=int(nz/2), show_immediately=False)
    my_pressure_matrix.plot_pressure_theta(z=int(nz/2), show_immediately=False)
    plt.show()
    plt.close('all')



