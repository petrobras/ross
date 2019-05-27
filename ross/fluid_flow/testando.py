from ross.fluid_flow import fluid_flow as flow
import numpy as np
nz = 8
ntheta = 64
nradius = 11
length = 0.01
omega = 100. * 2 * np.pi / 60
p_in = 0.
p_out = 0.
radius_rotor = 0.08
radius_stator = 0.1
visc = 0.015
rho = 860.
beta = np.pi
eccentricity = 0.01
my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, length,
                                         omega, p_in, p_out, radius_rotor,
                                         radius_stator, visc, rho, beta, eccentricity = eccentricity)
my_pressure_matrix.calculate_pressure_matrix_analytical()
my_pressure_matrix.calculate_pressure_matrix_numerical()
my_pressure_matrix.plot_eccentricity()
my_pressure_matrix.plot_pressure_z()
my_pressure_matrix.plot_shape()
my_pressure_matrix.plot_pressure_theta(z=int(nz / 2))
my_pressure_matrix.matplot_pressure_theta_cylindrical(z=int(nz / 2), show_immediately=True)