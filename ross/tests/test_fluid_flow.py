import math
import numpy as np

from ross.fluid_flow import fluid_flow as flow
from bokeh.plotting import figure
import matplotlib.pyplot as plt


def fluid_flow_short_eccentricity():
    nz = 30
    ntheta = 20
    nradius = 11
    length = 0.03
    omega = 157.1
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.0499
    radius_stator = 0.05
    eccentricity = (radius_stator - radius_rotor)*0.2663
    visc = 0.1
    rho = 860.
    return flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                               p_out, radius_rotor, radius_stator,
                               visc, rho, eccentricity=eccentricity)


def fluid_flow_short_load():
    nz = 30
    ntheta = 20
    nradius = 11
    length = 0.03
    omega = 157.1
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.0499
    radius_stator = 0.05
    load = 525
    visc = 0.1
    rho = 860.
    return flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                               p_out, radius_rotor, radius_stator,
                               visc, rho, load=load)


def test_sommerfeld_number():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the sommerfeld number and eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_load()
    assert math.isclose(bearing.eccentricity_ratio, 0.2663, rel_tol=0.001)


def test_get_rotor_load():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the load over the rotor, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_eccentricity()
    assert math.isclose(bearing.load, 525, rel_tol=0.1)


def test_stiffness_matrix():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the stiffness matrix, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_eccentricity()
    kxx, kxy, kyx, kyy = bearing.get_analytical_stiffness_matrix()
    assert math.isclose(kxx/10**6, 12.81, rel_tol=0.01)
    assert math.isclose(kxy/10**6, 16.39, rel_tol=0.01)
    assert math.isclose(kyx/10**6, -25.06, rel_tol=0.01)
    assert math.isclose(kyy/10**6, 8.815, rel_tol=0.01)


def test_damping_matrix():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the damping matrix, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_load()
    cxx, cxy, cyx, cyy = bearing.get_analytical_damping_matrix()
    assert math.isclose(cxx/10**3, 232.9, rel_tol=0.01)
    assert math.isclose(cxy/10**3, -81.92, rel_tol=0.01)
    assert math.isclose(cyx/10**3, -81.92, rel_tol=0.01)
    assert math.isclose(cyy/10**3, 294.9, rel_tol=0.01)


def fluid_flow_short_numerical():
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
    eccentricity = 0.001
    return flow.PressureMatrix(nz, ntheta, nradius, length,
                               omega, p_in, p_out, radius_rotor,
                               radius_stator, visc, rho, beta=beta, eccentricity=eccentricity)


def fluid_flow_long_numerical():
    nz = 8
    ntheta = 132
    nradius = 11
    omega = 100. * 2 * np.pi / 60
    p_in = 0.
    p_out = 0.
    radius_rotor = 1
    h = 0.000194564
    radius_stator = radius_rotor + h
    length = 8 * radius_stator
    visc = 0.015
    rho = 860.
    beta = np.pi
    eccentricity = 0.0001
    return flow.PressureMatrix(nz, ntheta, nradius, length,
                               omega, p_in, p_out, radius_rotor,
                               radius_stator, visc, rho, beta=beta, eccentricity=eccentricity)


def test_numerical_abs_error():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_analytical()
    bearing.calculate_pressure_matrix_numerical()
    error = np.linalg.norm(bearing.p_mat_analytical[:][int(bearing.nz / 2)] -
                           bearing.p_mat_numerical[:][int(bearing.nz / 2)], ord=np.inf)
    assert math.isclose(error, 0, abs_tol=0.001)


def test_numerical_abs_error2():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_analytical(method=1)
    bearing.calculate_pressure_matrix_numerical()
    error = np.linalg.norm(bearing.p_mat_analytical[:][int(bearing.nz / 2)] -
                           bearing.p_mat_numerical[:][int(bearing.nz / 2)], ord=np.inf)
    assert math.isclose(error, 0, abs_tol=0.001)


def test_long_bearing():
    bearing = fluid_flow_long_numerical()
    bearing.calculate_pressure_matrix_analytical()
    bearing.calculate_pressure_matrix_numerical()
    error = (max(bearing.p_mat_analytical[int(bearing.nz / 2)]) -
             max(bearing.p_mat_numerical[int(bearing.nz / 2)])) / max(bearing.p_mat_numerical[int(bearing.nz / 2)])
    assert math.isclose(error, 0, abs_tol=0.02)


def test_bokeh_plots():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_numerical()
    figure_type = type(figure())
    assert isinstance(bearing.plot_shape(), figure_type)
    assert isinstance(bearing.plot_eccentricity(), figure_type)
    assert isinstance(bearing.plot_pressure_theta(), figure_type)
    assert isinstance(bearing.plot_pressure_z(), figure_type)


def test_matplotlib_plots():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_analytical()
    ax_type = type(plt.gca())
    assert isinstance(bearing.matplot_eccentricity(), ax_type)
    assert isinstance(bearing.matplot_pressure_theta(), ax_type)
    assert isinstance(bearing.matplot_pressure_z(), ax_type)
    assert isinstance(bearing.matplot_shape(), ax_type)
