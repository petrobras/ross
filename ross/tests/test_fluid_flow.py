# fmt: off
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from bokeh.plotting import figure
from numpy.testing import assert_allclose

from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_damping_matrix, calculate_oil_film_force,
    calculate_stiffness_matrix, find_equilibrium_position)
from ross.fluid_flow.fluid_flow_geometry import move_rotor_center
from ross.fluid_flow.fluid_flow_graphics import (matplot_eccentricity,
                                                 matplot_pressure_theta,
                                                 matplot_pressure_z,
                                                 matplot_shape,
                                                 plot_eccentricity,
                                                 plot_pressure_theta,
                                                 plot_pressure_z, plot_shape)

# fmt: on


@pytest.fixture
def fluid_flow_short_eccentricity():
    nz = 8
    ntheta = 32
    nradius = 8
    omega = 100.0 * 2 * np.pi / 60
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 0.1999996
    radius_stator = 0.1999996 + 0.000194564
    length = (1 / 10) * (2 * radius_stator)
    eccentricity = 0.0001
    visc = 0.015
    rho = 860.0
    return flow.FluidFlow(
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        eccentricity=eccentricity,
        immediately_calculate_pressure_matrix_numerically=False,
    )


def fluid_flow_short_friswell(set_load=True):
    nz = 8
    ntheta = 32
    nradius = 8
    length = 0.03
    omega = 157.1
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 0.0499
    radius_stator = 0.05
    load = 525
    visc = 0.1
    rho = 860.0
    eccentricity = (radius_stator - radius_rotor) * 0.2663
    if set_load:
        return flow.FluidFlow(
            nz,
            ntheta,
            nradius,
            length,
            omega,
            p_in,
            p_out,
            radius_rotor,
            radius_stator,
            visc,
            rho,
            load=load,
            immediately_calculate_pressure_matrix_numerically=False,
        )
    else:
        return flow.FluidFlow(
            nz,
            ntheta,
            nradius,
            length,
            omega,
            p_in,
            p_out,
            radius_rotor,
            radius_stator,
            visc,
            rho,
            eccentricity=eccentricity,
            immediately_calculate_pressure_matrix_numerically=False,
        )


def test_sommerfeld_number():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the sommerfeld number and eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_friswell()
    assert math.isclose(bearing.eccentricity_ratio, 0.2663, rel_tol=0.001)


def test_get_rotor_load():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the load over the rotor, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_friswell(set_load=False)
    assert math.isclose(bearing.load, 525, rel_tol=0.1)


def test_stiffness_matrix():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the stiffness matrix, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_friswell()
    kxx, kxy, kyx, kyy = calculate_stiffness_matrix(bearing, force_type="short")
    assert math.isclose(kxx / 10 ** 6, 12.81, rel_tol=0.01)
    assert math.isclose(kxy / 10 ** 6, 16.39, rel_tol=0.01)
    assert math.isclose(kyx / 10 ** 6, -25.06, rel_tol=0.01)
    assert math.isclose(kyy / 10 ** 6, 8.815, rel_tol=0.01)


def test_stiffness_matrix_numerical(fluid_flow_short_eccentricity):
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the stiffness matrix based on Friswell's book formulas, given the
    eccentricity ratio.
    Taken from chapter 5, page 179 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_eccentricity
    bearing.calculate_pressure_matrix_numerical()
    kxx, kxy, kyx, kyy = calculate_stiffness_matrix(bearing, force_type="numerical")
    k_xx, k_xy, k_yx, k_yy = calculate_stiffness_matrix(bearing, force_type="short")
    assert_allclose(kxx, k_xx, rtol=0.33)
    assert_allclose(kxy, k_xy, rtol=0.21)
    assert_allclose(kyx, k_yx, rtol=0.17)
    assert_allclose(kyy, k_yy, rtol=0.19)


def test_damping_matrix():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the damping matrix, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow_short_friswell()
    cxx, cxy, cyx, cyy = calculate_damping_matrix(bearing, force_type="short")
    assert math.isclose(cxx / 10 ** 3, 232.9, rel_tol=0.01)
    assert math.isclose(cxy / 10 ** 3, -81.92, rel_tol=0.01)
    assert math.isclose(cyx / 10 ** 3, -81.92, rel_tol=0.01)
    assert math.isclose(cyy / 10 ** 3, 294.9, rel_tol=0.01)


def fluid_flow_short_numerical():
    nz = 8
    ntheta = 32
    nradius = 8
    length = 0.01
    omega = 100.0 * 2 * np.pi / 60
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 0.08
    radius_stator = 0.1
    visc = 0.015
    rho = 860.0
    attitude_angle = None
    eccentricity = 0.001
    return flow.FluidFlow(
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        attitude_angle=attitude_angle,
        eccentricity=eccentricity,
        immediately_calculate_pressure_matrix_numerically=False,
    )


def fluid_flow_long_numerical():
    nz = 8
    ntheta = 32
    nradius = 8
    omega = 100.0 * 2 * np.pi / 60
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 1
    h = 0.000194564
    radius_stator = radius_rotor + h
    length = 8 * 2 * radius_stator
    visc = 0.015
    rho = 860.0
    eccentricity = 0.0001
    return flow.FluidFlow(
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        eccentricity=eccentricity,
        immediately_calculate_pressure_matrix_numerically=False,
    )


def test_numerical_abs_error():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_analytical()
    bearing.calculate_pressure_matrix_numerical()
    error = np.linalg.norm(
        bearing.p_mat_analytical[:][int(bearing.nz / 2)]
        - bearing.p_mat_numerical[:][int(bearing.nz / 2)],
        ord=np.inf,
    )
    assert math.isclose(error, 0, abs_tol=0.001)


def test_numerical_abs_error2():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_analytical(method=1)
    bearing.calculate_pressure_matrix_numerical()
    error = np.linalg.norm(
        bearing.p_mat_analytical[:][int(bearing.nz / 2)]
        - bearing.p_mat_numerical[:][int(bearing.nz / 2)],
        ord=np.inf,
    )
    assert math.isclose(error, 0, abs_tol=0.001)


def test_long_bearing():
    bearing = fluid_flow_long_numerical()
    bearing.calculate_pressure_matrix_analytical()
    bearing.calculate_pressure_matrix_numerical()
    error = (
        max(bearing.p_mat_analytical[int(bearing.nz / 2)])
        - max(bearing.p_mat_numerical[int(bearing.nz / 2)])
    ) / max(bearing.p_mat_numerical[int(bearing.nz / 2)])
    assert math.isclose(error, 0, abs_tol=0.02)


def test_oil_film_force_short():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_numerical()
    n, t, force_x, force_y = calculate_oil_film_force(bearing)
    (
        n_numerical,
        t_numerical,
        force_x_numerical,
        force_y_numerical,
    ) = calculate_oil_film_force(bearing, force_type="numerical")
    assert_allclose(n, n_numerical, rtol=0.5)
    assert_allclose(t, t_numerical, rtol=0.25)
    assert_allclose(force_x_numerical, 0, atol=1e-07)
    assert_allclose(force_y_numerical, bearing.load, atol=1e-05)


def test_oil_film_force_long():
    bearing = fluid_flow_long_numerical()
    bearing.calculate_pressure_matrix_numerical()
    n, t, force_x, force_y = calculate_oil_film_force(bearing)
    (
        n_numerical,
        t_numerical,
        force_x_numerical,
        force_y_numerical,
    ) = calculate_oil_film_force(bearing, force_type="numerical")
    assert_allclose(n, n_numerical, rtol=0.07)
    assert_allclose(t, t_numerical, rtol=0.22)


def test_bokeh_plots():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_numerical()
    figure_type = type(figure())
    assert isinstance(plot_shape(bearing), figure_type)
    assert isinstance(plot_eccentricity(bearing), figure_type)
    assert isinstance(plot_pressure_theta(bearing), figure_type)
    assert isinstance(plot_pressure_z(bearing), figure_type)


def test_matplotlib_plots():
    bearing = fluid_flow_short_numerical()
    bearing.calculate_pressure_matrix_analytical()
    ax_type = type(plt.gca())
    assert isinstance(matplot_pressure_theta(bearing), ax_type)
    assert isinstance(matplot_pressure_z(bearing), ax_type)
    assert isinstance(matplot_shape(bearing), ax_type)

    # Create new axes to avoid using gca(), which can get a axes with polar projection
    fig, ax = plt.subplots()
    ax_type = type(ax)
    assert isinstance(matplot_eccentricity(bearing, ax=ax), ax_type)


def test_find_equilibrium_position():
    bearing = flow.fluid_flow_example2()
    find_equilibrium_position(
        bearing,
        print_along=False,
        tolerance=0.1,
        increment_factor=0.01,
        max_iterations=2,
        increment_reduction_limit=1e-03,
    )
    assert_allclose(
        bearing.eccentricity,
        (bearing.radius_stator - bearing.radius_rotor) * 0.2663,
        atol=0.001,
    )


def test_move_rotor_center():
    bearing = fluid_flow_short_friswell()
    eccentricity = bearing.eccentricity
    attitude_angle = bearing.attitude_angle
    move_rotor_center(bearing, 0.001, 0)
    move_rotor_center(bearing, -0.001, 0)
    assert_allclose(bearing.eccentricity, eccentricity)
    assert_allclose(bearing.attitude_angle, attitude_angle)
    move_rotor_center(bearing, 0, 0.001)
    move_rotor_center(bearing, 0, -0.001)
    assert_allclose(bearing.eccentricity, eccentricity)
    assert_allclose(bearing.attitude_angle, attitude_angle)
    move_rotor_center(bearing, 0, 0.001)
    assert bearing.eccentricity != eccentricity
    assert bearing.attitude_angle != attitude_angle
