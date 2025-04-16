import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.bearings.cylindrical import THDCylindrical
from ross.units import Q_


@pytest.fixture
def cylindrical():
    frequency = Q_([900], "RPM")
    L = Q_(10.3600055944, "in")

    bearing = THDCylindrical(
        n=3,
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        preload=0,
        geometry="circular",
        reference_temperature=50,
        frequency=frequency,
        fxs_load=0,
        fys_load=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        oil_supply_pressure=0,
        oil_flow_v=37.86,
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    )

    return bearing


def test_cylindrical_parameters(cylindrical):
    assert math.isclose(cylindrical.axial_length, 0.263144, rel_tol=0.0001)
    assert cylindrical.journal_radius == 0.2
    assert cylindrical.frequency == 94.24777960769379
    assert cylindrical.rho == 873.99629
    assert cylindrical.reference_temperature == 50


def test_cylindrical_equilibrium_pos(cylindrical):
    assert math.isclose(cylindrical.equilibrium_pos[0], 0.68733194, rel_tol=0.01)
    assert math.isclose(cylindrical.equilibrium_pos[1], -0.79394211, rel_tol=0.01)


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

    assert math.isclose(kxx, 2497674531.1749372, rel_tol=0.0001)
    assert math.isclose(kxy, 783937669.6587772, rel_tol=0.0001)
    assert math.isclose(kyx, -3140562821.5290236, rel_tol=0.0001)
    assert math.isclose(kyy, 2562440911.734241, rel_tol=0.0001)
    assert math.isclose(cxx, 36950674.61976142, rel_tol=0.0001)
    assert math.isclose(cxy, -37265296.2322692, rel_tol=0.0001)
    assert math.isclose(cyx, -42642543.712838694, rel_tol=0.0001)
    assert math.isclose(cyy, 100992315.0043159, rel_tol=0.0001)
