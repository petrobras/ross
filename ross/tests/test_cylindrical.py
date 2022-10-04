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
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        reference_temperature=50,
        reference_viscosity=0.02,
        speed=Q_([900], "RPM"),
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
        oil_flow=18.93,
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    )
    return bearing


@pytest.fixture
def cylindrical_units():

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
        operating_type="flooded",
        injection_pressure=0,
        oil_flow=18.93,
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    )

    return bearing


def test_cylindrical_parameters(cylindrical):
    assert cylindrical.axial_length == 0.263144
    assert cylindrical.journal_radius == 0.2
    assert cylindrical.speed == 94.24777960769379
    assert cylindrical.rho == 873.99629
    assert cylindrical.reference_temperature == 50


def test_cylindrical_parameters_units(cylindrical_units):
    assert math.isclose(cylindrical_units.axial_length, 0.263144, rel_tol=0.0001)
    assert cylindrical_units.journal_radius == 0.2
    assert cylindrical_units.speed == 94.24777960769379
    assert cylindrical_units.rho == 873.99629
    assert cylindrical_units.reference_temperature == 50


def test_cylindrical_equilibrium_pos(cylindrical):
    assert math.isclose(cylindrical.equilibrium_pos[0], 0.59937565, rel_tol=0.01)
    assert math.isclose(cylindrical.equilibrium_pos[1], -0.72341011, rel_tol=0.01)


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

    assert math.isclose(kxx, 1003481981.4164313, rel_tol=0.0001)
    assert math.isclose(kxy, 368311014.54999083, rel_tol=0.0001)
    assert math.isclose(kyx, -1311924669.6961021, rel_tol=0.0001)
    assert math.isclose(kyy, 981749354.0559089, rel_tol=0.0001)
    assert math.isclose(cxx, 6393354010.436003, rel_tol=0.0001)
    assert math.isclose(cxy, -5646094641.365241, rel_tol=0.0001)
    assert math.isclose(cyx, -6596242.650627648, rel_tol=0.0001)
    assert math.isclose(cyy, 17777768.4589808, rel_tol=0.0001)
