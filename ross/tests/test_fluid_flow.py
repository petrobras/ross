from ross.fluid_flow import fluid_flow as flow
import math


def test_sommerfeld_number():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the sommerfeld number and eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
    nz = 30
    ntheta = 20
    nradius = 11
    length = 0.03
    omega = 157.1
    p_in = 0.
    p_out = 0.
    radius_rotor = 0.0499
    radius_stator = 0.05
    eccentricity = 0
    visc = 0.1
    rho = 860.
    bearing = flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                                  p_out, radius_rotor, radius_stator,
                                  eccentricity, visc, rho)
    e = bearing.calculate_eccentricity_ratio(525)
    assert math.isclose(e, 0.2663, rel_tol=0.001)


def test_get_rotor_load():
    """
    This function instantiate a bearing using the fluid flow class and test if it matches the
    expected results for the load over the rotor, given the eccentricity ratio.
    Taken from example 5.5.1, page 181 (Dynamics of rotating machine, FRISSWELL)
    """
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
    bearing = flow.PressureMatrix(nz, ntheta, nradius, length, omega, p_in,
                                  p_out, radius_rotor, radius_stator,
                                  eccentricity, visc, rho)
    f = bearing.get_rotor_load()
    assert math.isclose(f, 525, rel_tol=0.1)
