import pytest
from numpy.testing import assert_allclose

from ross.bearings.tilting_pad import TiltingPad
from ross.units import Q_


@pytest.fixture
def tilting_pad():
    frequency = Q_([3000], "RPM")
    pivot_angle = Q_([18, 90, 162, 234, 306], "deg")
    pad_arc = Q_([60, 60, 60, 60, 60], "deg")
    pad_axial_length = Q_([50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3], "m")
    oil_supply_temperature = Q_(40, "degC")
    attitude_angle = Q_(267.5, "deg")

    bearing = TiltingPad(
        n=1,
        frequency=frequency,
        equilibrium_type="match_eccentricity",
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=pivot_angle,
        pad_arc=pad_arc,
        pad_axial_length=pad_axial_length,
        pre_load=[0.5, 0.5, 0.5, 0.5, 0.5],
        offset=[0.5, 0.5, 0.5, 0.5, 0.5],
        lubricant="ISOVG32",
        oil_supply_temperature=oil_supply_temperature,
        nx=30,
        nz=30,
        print_result=False,
        print_progress=False,
        print_time=False,
        eccentricity=0.483,
        attitude_angle=attitude_angle,
    )

    return bearing


def test_tilting_pad_parameters(tilting_pad):
    assert_allclose(tilting_pad.journal_radius, 0.0508)
    assert_allclose(tilting_pad.radial_clearance, 74.9e-6)
    assert_allclose(tilting_pad.frequency, 314.1592653589793)
    assert_allclose(tilting_pad.reference_temperature, 40)
    assert_allclose(tilting_pad.n_pad, 5)


def test_tilting_pad_equilibrium_pos(tilting_pad):
    assert_allclose(tilting_pad.eccentricity, 0.483, rtol=0.01)
    assert_allclose(tilting_pad.attitude_angle, 4.667, rtol=0.01)

    expected_angles = [0.001135, -0.000726, 0.000095, 0.000372, 0.001039]
    for i, expected_angle in enumerate(expected_angles):
        assert_allclose(tilting_pad.psi_pad[i], expected_angle, rtol=0.1)


def test_tilting_pad_coefficients(tilting_pad):
    assert_allclose(tilting_pad.kxx, 166388569.24521738, rtol=0.001)
    assert_allclose(tilting_pad.kxy, 24935707.314652264, rtol=0.001)
    assert_allclose(tilting_pad.kyx, 24935707.314652324, rtol=0.001)
    assert_allclose(tilting_pad.kyy, 242414350.50060275, rtol=0.001)

    assert_allclose(tilting_pad.cxx, 464269.3452154938, rtol=0.001)
    assert_allclose(tilting_pad.cxy, 12498.24122981852, rtol=0.001)
    assert_allclose(tilting_pad.cyx, 12498.241229842806, rtol=0.001)
    assert_allclose(tilting_pad.cyy, 568361.6809310445, rtol=0.001)


def test_tilting_pad_forces(tilting_pad):
    assert_allclose(tilting_pad.force_x_dim, 186.968026, rtol=0.01)
    assert_allclose(tilting_pad.force_y_dim, 4835.829052, rtol=0.01)
