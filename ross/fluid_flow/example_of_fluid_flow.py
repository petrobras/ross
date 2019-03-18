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
lb = 0.1
omega = -100.*2*np.pi/60
p_in = 392266.
p_out = 100000.
radius_valley = 0.034
radius_crest = 0.039
radius_stator = 0.04
lwave = 0.18
xe = 0.003
ye = 0.003
visc = 0.015
rho = 860.


def calculate_pressure(epi, C, R, phi, mi, om, pa):
    """ Calculates pressure according to Tribology Series 33, page 119.
    """
    return ((6*mi*om*(R/C)*(R/C))/(1 - epi*epi)**(3/2))\
        * (phi - epi*np.sin(phi) - (2*phi - 4*epi*np.sin(phi) + epi*epi*phi + epi*epi*np.sin(phi)*np.cos(phi))
            / (2 + epi*epi)) + pa


def plot_pressure_graph(pressure_matrix):
    """ Given an instantiated PressureMatrix, plot its pressure along theta in z = lb/2 and compare with
        Tribology Series's results.
    """
    C = pressure_matrix.radius_stator - pressure_matrix.radius_valley
    epi = pressure_matrix.distance_between_centers / C
    pressure_book = []
    theta_list = []
    plt.figure(5)
    for theta in range(0, pressure_matrix.ntheta):
        phi = theta*pressure_matrix.dtheta
        pressure_book.append(calculate_pressure(epi, C, pressure_matrix.radius_valley, phi, pressure_matrix.visc,
                                                pressure_matrix.omega,
                                                pressure_matrix.p_mat[int(pressure_matrix.nz/2)][0]))
        theta_list.append(theta*pressure_matrix.dtheta)
        plt.plot()
    plt.plot(theta_list, pressure_book, 'r')
    plt.plot(theta_list, pressure_matrix.p_mat[int(pressure_matrix.nz/2)], 'b')
    plt.show()


if __name__ == "__main__":
    my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, lb, omega, p_in, p_out, radius_valley,
                                             radius_crest, radius_stator, lwave, xe, ye,  visc, rho,
                                             plot_eccentricity=True)
    P = my_pressure_matrix.calculate_pressure_matrix()
    my_pressure_matrix.plot_pressure_z(show_immediately=False)
    my_pressure_matrix.plot_shape(show_immediately=False)
    my_pressure_matrix.plot_pressure_theta(z=int(nz/2), show_immediately=False)
    plt.show()
    plot_pressure_graph(my_pressure_matrix)



