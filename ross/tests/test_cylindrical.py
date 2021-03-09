import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.fluid_flow.cylindrical import THDCylindrical
from ross.units import Q_

if __name__ == "__main__":

    x0 = [0.1, -0.1]
    L = float(0.263144)  # [metros]
    R = float(0.2)  # [metros]
    Cr = float(1.945e-4)  # [metros]
    nTheta = int(41)
    nZ = int(10)
    nY = None

    mu = float(0.02)  # [Ns/m²]
    speed = Q_(900, "RPM")  # [RPM]
    Wx = float(0)  # [N]
    Wy = float(-112814.91)  # [N]
    k = float(0.15327)  # Thermal conductivity [J/s.m.°C]
    Cp = float(1915.5)  # Specific heat [J/kg°C]
    rho = float(854.952)  # Specific mass [kg/m³]
    Treserv = float(50)  # Temperature of oil tank [ºC]
    mix = float(0.52)  # Mixing factor. Used because the oil supply flow is not known.
    nGap = int(1)  #    Number of volumes in recess zone
    nPad = int(2)  #    Number of pads
    betha_s = 176
    T_1 = float(50)
    T_2 = float(80)
    mu_ref1 = float(0.02)
    mu_ref2 = float(0.01)

    mancal = THDCylindrical(
        L,
        R,
        Cr,
        nTheta,
        nZ,
        nY,
        nGap,
        betha_s,
        mu,
        speed,
        Wx,
        Wy,
        k,
        Cp,
        rho,
        Treserv,
        mix,
        T_1,
        T_2,
        mu_ref1,
        mu_ref2,
    )
    mancal.run(x0, print_progress=True)


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
    bearing.run([0.1, -0.1])

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


def test_cylindrical_equilibrium_pos_units(cylindrical_units):
    assert math.isclose(cylindrical_units.equilibrium_pos[0], 0.58656872, rel_tol=0.01)
    assert math.isclose(cylindrical_units.equilibrium_pos[1], -0.67207557, rel_tol=0.01)
