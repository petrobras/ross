import pytest
from numpy.testing import assert_allclose
import numpy as np

from ross.bearings.tilting_pad import TiltingPad
from ross.units import Q_


@pytest.fixture
def tilting_pad():
    frequency = Q_([3000], "RPM")
    pivot_angle = Q_([18, 90, 162, 234, 306], "deg")
    pad_arc = Q_([60, 60, 60, 60, 60], "deg")
    pad_axial_length = Q_([50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3], "m")
    oil_supply_temperature = Q_(40, "degC")
    attitude_angle = Q_(287.5, "deg")

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
        eccentricity=0.35,
        attitude_angle=attitude_angle,
        load=[8.8405e02, -2.6704e03],
    )

    return bearing


def test_tilting_pad_parameters(tilting_pad):
    assert_allclose(tilting_pad.journal_radius, 0.0508)
    assert_allclose(tilting_pad.radial_clearance, 74.9e-6)
    assert_allclose(tilting_pad.frequency, 314.1592653589793)
    assert_allclose(tilting_pad.reference_temperature, 40)
    assert_allclose(tilting_pad.n_pad, 5)


def test_tilting_pad_equilibrium_pos(tilting_pad):
    assert_allclose(tilting_pad.eccentricity, 0.35, rtol=0.01)
    assert_allclose(tilting_pad.attitude_angle, 5.017821599483698, rtol=0.01)

    expected_angles = [0.00107421, 0.00072079, 0.00029369, 0.00034969, 0.00081604]
    for i, expected_angle in enumerate(expected_angles):
        assert_allclose(tilting_pad.psi_pad[i], expected_angle, rtol=0.1)


def test_tilting_pad_coefficients(tilting_pad):
    # Stiffness coefficients
    assert_allclose(tilting_pad.kxx,  1.06151681e+08, rtol=0.001)
    assert_allclose(tilting_pad.kxy, -1.59240211e+07, rtol=0.001)
    assert_allclose(tilting_pad.kyx, -1.59240211e+07, rtol=0.001)
    assert_allclose(tilting_pad.kyy,  1.32123081e+08, rtol=0.001)

    # Damping coefficients
    assert_allclose(tilting_pad.cxx,  355116.7381765,  rtol=0.001)
    assert_allclose(tilting_pad.cxy, -43069.39477648,  rtol=0.001)
    assert_allclose(tilting_pad.cyx, -43069.39477648,  rtol=0.001)
    assert_allclose(tilting_pad.cyy,  439616.8694803,  rtol=0.001)


def test_tilting_pad_forces(tilting_pad):
    # force_x_dim and force_y_dim are arrays (one value per pad)
    expected_force_x = np.array(
        [-9.22582848e02, 3.02996821e-01, 5.51054460e02, 1.04734115e03, -1.57342924e03]
    )
    expected_force_y = np.array(
        [-300.86140003, -420.36660054, -178.8695415, 1442.6020094, 2161.92734728]
    )
    assert_allclose(tilting_pad.force_x_dim, expected_force_x, rtol=0.01)
    assert_allclose(tilting_pad.force_y_dim, expected_force_y, rtol=0.01)
