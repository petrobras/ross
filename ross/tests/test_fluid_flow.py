from ross.fluid_flow import fluid_flow as flow
import math


def fluid_flow0(eccentricity_ratio):
    nz = 30
    ntheta = 20
    nradius = 11
    length = 0.03
    omega = 157.1
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.0499
    radius_stator = 0.05
    eccentricity = (radius_stator - radius_rotor)*eccentricity_ratio
    visc = 0.1
    rho = 860.
    return flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                               p_out, radius_rotor, radius_stator,
                               eccentricity, visc, rho)


def test_sommerfeld_number():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the sommerfeld number and eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow0(0)
    e = bearing.calculate_eccentricity_ratio(525)
    assert math.isclose(e, 0.2663, rel_tol=0.001)


def test_get_rotor_load():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the load over the rotor, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow0(0.2663)
    f = bearing.get_rotor_load()
    assert math.isclose(f, 525, rel_tol=0.1)


def test_stiffness_matrix():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the stiffness matrix, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    bearing = fluid_flow0(0.2663)
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
    bearing = fluid_flow0(0.2663)
    cxx, cxy, cyx, cyy = bearing.get_analytical_damping_matrix()
    assert math.isclose(cxx/10**3, 232.9, rel_tol=0.01)
    assert math.isclose(cxy/10**3, -81.92, rel_tol=0.01)
    assert math.isclose(cyx/10**3, -81.92, rel_tol=0.01)
    assert math.isclose(cyy/10**3, 294.9, rel_tol=0.01)
