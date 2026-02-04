import pytest
from numpy.testing import assert_allclose
import numpy as np

from ross.bearings.tilting_pad import TiltingPad
from ross.units import Q_


@pytest.fixture
def tilting_pad_match():
    """
    Tilting pad bearing with match_eccentricity equilibrium type.

    Parameters
    ----------
    frequency : pint.Quantity
        Rotational speed (3000 RPM)
    journal_diameter : float
        Journal diameter (101.6 mm)
    radial_clearance : float
        Radial clearance (74.9 μm)
    pivot_angle : pint.Quantity
        Pivot angles for 5 pads [18°, 90°, 162°, 234°, 306°]
    pad_arc : pint.Quantity
        Arc length for each pad (60° each)
    lubricant : str
        Lubricant type (ISOVG32)
    oil_supply_temperature : pint.Quantity
        Supply temperature (40°C)

    Returns
    -------
    bearing : TiltingPad
        Configured tilting pad bearing object

    """
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


@pytest.fixture
def tilting_pad_determine():
    """
    Tilting pad bearing with determine_eccentricity equilibrium type.

    Uses thermo-hydrodynamic model to determine equilibrium position
    from initial guesses of eccentricity and attitude angle.

    Parameters
    ----------
    frequency : pint.Quantity
        Rotational speed (3000 RPM)
    journal_diameter : float
        Journal diameter (101.6 mm)
    radial_clearance : float
        Radial clearance (74.9 μm)
    model_type : str
        Analysis model type ('thermo_hydro_dynamic')
    eccentricity : float
        Initial guess for eccentricity (0.30)
    attitude_angle : pint.Quantity
        Initial guess for attitude angle (267.5°)

    Returns
    -------
    bearing : TiltingPad
        Configured tilting pad bearing object

    """
    frequency = Q_([3000], "RPM")
    pivot_angle = Q_([18, 90, 162, 234, 306], "deg")
    pad_arc = Q_([60, 60, 60, 60, 60], "deg")
    pad_axial_length = Q_([50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3], "m")
    oil_supply_temperature = Q_(40, "degC")

    bearing = TiltingPad(
        n=1,
        frequency=frequency,
        equilibrium_type="determine_eccentricity",
        model_type="thermo_hydro_dynamic",
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
        attitude_angle=Q_(287.5, "deg"),
        load=[8.8405e02, -2.6704e03],
        initial_pads_angles=[1.0742e-03, 7.2080e-04, 2.9369e-04, 3.4969e-04, 8.1604e-04],
        equilibrium_options={"xtol": 1e-2, "ftol": 1e-2, "maxiter": 1000},
    )

    return bearing


def test_tilting_pad_parameters(tilting_pad_match):
    """Test basic geometric and operational parameters."""
    assert_allclose(tilting_pad_match.journal_radius, 0.0508)
    assert_allclose(tilting_pad_match.radial_clearance, 74.9e-6)
    assert_allclose(tilting_pad_match.frequency, 314.1592653589793)
    assert_allclose(tilting_pad_match.reference_temperature, 40)
    assert_allclose(tilting_pad_match.n_pad, 5)


def test_tilting_pad_parameters_determine(tilting_pad_determine):
    """Test basic geometric and operational parameters for determine case."""
    assert_allclose(tilting_pad_determine.journal_radius, 0.0508)
    assert_allclose(tilting_pad_determine.radial_clearance, 74.9e-6)
    assert_allclose(tilting_pad_determine.frequency, 314.1592653589793)
    assert_allclose(tilting_pad_determine.reference_temperature, 40)
    assert_allclose(tilting_pad_determine.n_pad, 5)


def test_tilting_pad_equilibrium_pos(tilting_pad_match):
    """
    Test equilibrium position for match_eccentricity case.

    Validates eccentricity, attitude angle, and pad tilting angles (psi_pad)
    against expected values with 1% tolerance for position and 10% for angles.

    """
    assert_allclose(tilting_pad_match.eccentricity, 0.35, rtol=0.01)
    assert_allclose(tilting_pad_match.attitude_angle, 5.017821599483698, rtol=0.01)

    expected_angles = [0.00107426, 0.00072933, 0.00029538, 0.00034972, 0.00081735]
    for i, expected_angle in enumerate(expected_angles):
        assert_allclose(tilting_pad_match.psi_pad[i], expected_angle, rtol=0.1)


def test_tilting_pad_equilibrium_pos_determine(tilting_pad_determine):
    """
    Test equilibrium position for determine_eccentricity case.

    Validates converged eccentricity, attitude angle, and pad tilting angles.
    Expected values obtained from numerical convergence.

    """
    assert_allclose(tilting_pad_determine.eccentricity, 0.35, rtol=0.01)
    assert_allclose(tilting_pad_determine.attitude_angle, 5.017821599483698, rtol=0.01)

    expected_angles = [0.0010742, 0.0007208, 0.00029369, 0.00034969, 0.00081604]
    for i, expected_angle in enumerate(expected_angles):
        assert_allclose(tilting_pad_determine.psi_pad[i], expected_angle, rtol=0.1)


def test_tilting_pad_coefficients(tilting_pad_match):
    """
    Test dynamic coefficients for match_eccentricity case.

    Validates stiffness coefficients (kxx, kxy, kyx, kyy) in N/m and
    damping coefficients (cxx, cxy, cyx, cyy) in N·s/m.

    """
    # Stiffness coefficients
    assert_allclose(tilting_pad_match.kxx, 8.9622e07, rtol=0.001)
    assert_allclose(tilting_pad_match.kxy, -3.3557e07, rtol=0.001)
    assert_allclose(tilting_pad_match.kyx, -3.3557e07, rtol=0.001)
    assert_allclose(tilting_pad_match.kyy, 1.1616e08, rtol=0.001)

    # Damping coefficients
    assert_allclose(tilting_pad_match.cxx, 283383.66451631, rtol=0.001)
    assert_allclose(tilting_pad_match.cxy, -27675.95645314, rtol=0.001)
    assert_allclose(tilting_pad_match.cyx, -27675.95645314, rtol=0.001)
    assert_allclose(tilting_pad_match.cyy, 331146.04573972, rtol=0.001)


def test_tilting_pad_coefficients_determine(tilting_pad_determine):
    """
    Test dynamic coefficients for determine_eccentricity case.

    Validates stiffness and damping coefficients obtained from
    thermo-hydrodynamic analysis.

    """
    # Stiffness coefficients
    assert_allclose(tilting_pad_determine.kxx, 9.0236e07, rtol=0.01)
    assert_allclose(tilting_pad_determine.kxy, -3.1795e07, rtol=0.01)
    assert_allclose(tilting_pad_determine.kyx, -3.1795e07, rtol=0.01)
    assert_allclose(tilting_pad_determine.kyy, 1.1730e08, rtol=0.01)

    # Damping coefficients
    assert_allclose(tilting_pad_determine.cxx, 287474.99912376, rtol=0.01)
    assert_allclose(tilting_pad_determine.cxy, -28051.03595784, rtol=0.01)
    assert_allclose(tilting_pad_determine.cyx, -28051.03595784, rtol=0.01)
    assert_allclose(tilting_pad_determine.cyy, 338837.0059512, rtol=0.01)


def test_tilting_pad_forces(tilting_pad_match):
    """
    Test dimensional forces for match_eccentricity case.

    Validates hydrodynamic forces in x and y directions for each pad.
    Forces are in Newtons.

    """
    expected_force_x = np.array(
        [-9.22636632e02, 3.11178501e-01, 5.52613265e02, 1.04737975e03, -1.57703246e03]
    )
    expected_force_y = np.array(
        [-300.87898774, -426.66181956, -179.37448498, 1442.65526681, 2166.8722571]
    )
    assert_allclose(tilting_pad_match.force_x_dim, expected_force_x, rtol=0.01)
    assert_allclose(tilting_pad_match.force_y_dim, expected_force_y, rtol=0.01)


def test_tilting_pad_forces_determine(tilting_pad_determine):
    """
    Test dimensional forces for determine_eccentricity case.

    Validates hydrodynamic forces obtained from thermo-hydrodynamic
    equilibrium solution. Forces are in Newtons.

    """
    expected_force_x = np.array(
        [-9.22442351e02, 3.02970480e-01, 5.51003780e02, 1.04686245e03, -1.57352854e03]
    )
    expected_force_y = np.array(
        [-300.81556983, -420.32523295, -178.85308935, 1441.94264606, 2162.0637731]
    )
    assert_allclose(tilting_pad_determine.force_x_dim, expected_force_x, rtol=0.01)
    assert_allclose(tilting_pad_determine.force_y_dim, expected_force_y, rtol=0.01)