import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.fluid_flow.cylindrical import THDCylindrical
from ross.units import Q_


@pytest.fixture
def cylindrical():

    speed = Q_([900], "RPM")
    L = Q_(10.3600055944, "in")

    bearing = THDCylindrical(
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=4,
        elements_axial=3,
        n_y=None,
        n_pad=2,
        pad_arc_length=176,
        reference_temperature=50,
        reference_viscosity=0.02,
        speed=speed,
        load_x_direction=0,
        load_y_direction=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        node=3,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        show_coef=False,
        print_result=True,
        print_progress=False,
        print_time=False,
    )

    return bearing


def test_cylindrical_parameters(cylindrical):
    assert math.isclose(cylindrical.axial_length, 0.263144, rel_tol=0.0001)
    assert cylindrical.journal_radius == 0.2
    assert cylindrical.speed == 94.24777960769379
    assert cylindrical.rho == 873.99629
    assert cylindrical.reference_temperature == 50


def test_cylindrical_equilibrium_pos(cylindrical):
    assert math.isclose(
        cylindrical.equilibrium_pos[0], 0.6063220830888834, rel_tol=0.01
    )
    assert math.isclose(
        cylindrical.equilibrium_pos[1], -0.8080463672910241, rel_tol=0.01
    )


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

    assert math.isclose(kxx, 1154092585.2914207, rel_tol=0.0001)
    assert math.isclose(kxy, -78688969.0436872, rel_tol=0.0001)
    assert math.isclose(kyx, -1399698008.045516, rel_tol=0.0001)
    assert math.isclose(kyy, 1333930910.0158768, rel_tol=0.0001)
    assert math.isclose(cxx, 5853947611.807541, rel_tol=0.0001)
    assert math.isclose(cxy, -6125302956.996735, rel_tol=0.0001)
    assert math.isclose(cyx, -16878755.32663667, rel_tol=0.0001)
    assert math.isclose(cyy, 4366663.510970946, rel_tol=0.0001)
