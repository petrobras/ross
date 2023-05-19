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
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        reference_temperature=50,
        speed=speed,
        load_x_direction=0,
        load_y_direction=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        node=3,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        injection_pressure=0,
        oil_flow=37.86,
        show_coef=False,
        print_result=False,
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
    assert math.isclose(cylindrical.equilibrium_pos[0], 0.6873316, rel_tol=0.01)
    assert math.isclose(cylindrical.equilibrium_pos[1], -0.79393636, rel_tol=0.01)


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

    assert math.isclose(kxx, 1080948741.8512235, rel_tol=0.0001)
    assert math.isclose(kxy, 339258572.34310913, rel_tol=0.0001)
    assert math.isclose(kyx, -1359170639.0567815, rel_tol=0.0001)
    assert math.isclose(kyy, 1108970752.2456105, rel_tol=0.0001)
    assert math.isclose(cxx, 9815899.503793057, rel_tol=0.0001)
    assert math.isclose(cxy, -9963602.922357056, rel_tol=0.0001)
    assert math.isclose(cyx, -11312462.69772395, rel_tol=0.0001)
    assert math.isclose(cyy, 27194995.506247465, rel_tol=0.0001)
