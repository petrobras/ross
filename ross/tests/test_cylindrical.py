import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.fluid_flow.cylindrical import THDCylindrical
from ross.units import Q_


@pytest.fixture
def cylindrical():

    bearing = THDCylindrical(
        axial_length=0.263144,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_in_circunferencial_direction=11,
        elements_in_axial_direction=3,
        n_y=None,
        number_of_segments=2,
        segment_arc_length=176,
        reservoir_temperature=50,
        viscosity_at_reservoir_temperature=0.02,
        speed=Q_(900, "RPM"),
        load_x_direction=0,
        load_y_direction=-112814.91,
        k_t=0.15327,
        Cp=1915.24,
        rho=854.952,
        groove_factor=[0.52, 0.48],
        T_muI=50,
        T_muF=80,
        mu_I=0.02,
        mu_F=0.01,
        sommerfeld_type=2,
    )
    bearing.run([0.1, -0.1])

    return bearing


@pytest.fixture
def cylindrical_units():

    speed = Q_(900, "RPM")
    L = Q_(10.3600055944, "in")

    bearing = THDCylindrical(
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_in_circunferencial_direction=11,
        elements_in_axial_direction=3,
        n_y=None,
        number_of_segments=2,
        segment_arc_length=176,
        reservoir_temperature=50,
        viscosity_at_reservoir_temperature=0.02,
        speed=speed,
        load_x_direction=0,
        load_y_direction=-112814.91,
        k_t=0.15327,
        Cp=1915.24,
        rho=854.952,
        groove_factor=[0.52, 0.48],
        T_muI=50,
        T_muF=80,
        mu_I=0.02,
        mu_F=0.01,
        sommerfeld_type=2, 
    )

    return bearing


def test_cylindrical_parameters(cylindrical):
    assert cylindrical.L == 0.263144
    assert cylindrical.R == 0.2
    assert cylindrical.speed == 94.24777960769379
    assert cylindrical.rho == 854.952
    assert cylindrical.T_reserv == 50


def test_cylindrical_parameters_units(cylindrical_units):
    assert math.isclose(cylindrical_units.L, 0.263144, rel_tol=0.0001)
    assert cylindrical_units.R == 0.2
    assert cylindrical_units.speed == 94.24777960769379
    assert cylindrical_units.rho == 854.952
    assert cylindrical_units.T_reserv == 50


def test_cylindrical_equilibrium_pos(cylindrical):
    assert math.isclose(cylindrical.equilibrium_pos[0], 0.57086823, rel_tol=0.01)
    assert math.isclose(cylindrical.equilibrium_pos[1], -0.70347935, rel_tol=0.01)


def test_cylindrical_coefficients(cylindrical):
    coefs = cylindrical.coefficients(method="perturbation")
    kxx = coefs[0][0]
    kxy = coefs[0][1]
    kyx = coefs[0][2]
    kyy = coefs[0][3]
    cxx = coefs[1][0]
    cxy = coefs[1][1]
    cyx = coefs[1][2]
    cyy = coefs[1][3]

    assert math.isclose(kxx, 1096226726.7794268, rel_tol=0.0001)
    assert math.isclose(kxy, 427850632.6256644, rel_tol=0.0001)
    assert math.isclose(kyx, -1319278493.1171515, rel_tol=0.0001)
    assert math.isclose(kyy, 979041960.0738384, rel_tol=0.0001)
    assert math.isclose(cxx, 6542293924.5259, rel_tol=0.0001)
    assert math.isclose(cxy, -5549524826.626654, rel_tol=0.0001)
    assert math.isclose(cyx, -17071752.093898583, rel_tol=0.0001)
    assert math.isclose(cyy, 6226232.102500191, rel_tol=0.0001)
