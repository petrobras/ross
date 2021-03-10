import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.fluid_flow.cylindrical import THDCylindrical
from ross.units import Q_


@pytest.fixture
def cylindrical():

    bearing = THDCylindrical(
        L=0.263144,
        R=0.2,
        c_r=1.95e-4,
        n_theta=41,
        n_z=5,
        n_y=None,
        n_gap=1,
        betha_s=176,
        mu_ref=0.02,
        speed=94.24777960769379,
        Wx=0,
        Wy=-112814.91,
        k_t=0.15327,
        Cp=1915.24,
        rho=854.952,
        T_reserv=50,
        fat_mixt=0.52,
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
        L=L,
        R=0.2,
        c_r=1.95e-4,
        n_theta=41,
        n_z=5,
        n_y=None,
        n_gap=1,
        betha_s=176,
        mu_ref=0.02,
        speed=speed,
        Wx=0,
        Wy=-112814.91,
        k_t=0.15327,
        Cp=1915.24,
        rho=854.952,
        T_reserv=50,
        fat_mixt=0.52,
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
    assert math.isclose(cylindrical.equilibrium_pos[0], 0.58656872, rel_tol=0.01)
    assert math.isclose(cylindrical.equilibrium_pos[1], -0.67207557, rel_tol=0.01)


def test_cylindrical_coefficients(cylindrical):
    coefs = cylindrical.coefficients()
    kxx = coefs[0][0]
    kxy = coefs[0][1]
    kyx = coefs[0][2]
    kyy = coefs[0][3]
    cxx = coefs[1][0]
    cxy = coefs[1][1]
    cyx = coefs[1][2]
    cyy = coefs[1][3]

    assert math.isclose(kxx, 977553273.755312, rel_tol=0.0001)
    assert math.isclose(kxy, 413651036.60706633, rel_tol=0.0001)
    assert math.isclose(kyx, -1357492195.2509518, rel_tol=0.0001)
    assert math.isclose(kyy, 950696959.9668411, rel_tol=0.0001)
    assert math.isclose(cxx, -6788417498.235009, rel_tol=0.0001)
    assert math.isclose(cxy, 5412613308.752171, rel_tol=0.0001)
    assert math.isclose(cyx, 16755460.575634956, rel_tol=0.0001)
    assert math.isclose(cyy, -7127858.166934674, rel_tol=0.0001)
